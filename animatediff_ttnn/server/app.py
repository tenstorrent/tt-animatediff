# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""ASGI app serving tt-animatediff, for tt-model-manager's ``tt-dit-server`` kind.

A diffusion transformer has no tokens, no KV cache and no continuous batching, so vLLM has
nothing to do for it. ``tt-dit-server`` therefore installs a small HTTP stack instead of an
engine and launches this module with uvicorn:

    python -m uvicorn --host 0.0.0.0 --port 8000 --lifespan on \\
        animatediff_ttnn.server.app:app

TWO CONTRACTS THIS FILE HAS TO HONOUR
-------------------------------------
**1. Readiness is the lifespan.** tt-model-manager decides the server is up when uvicorn
logs ``Application startup complete``, which it prints *after* ASGI lifespan startup
returns. So the device open and the model warm belong in the lifespan and nowhere else: do
them lazily on the first request and the supervisor calls the server ready while it is
still loading, then times out the first request it sends.

**2. Importing this module must not touch hardware.** ``verify_lines`` imports the ASGI
attribute at image-build time, on a machine with no card, purely to prove the code
allowlist actually shipped the server. Every ttnn/tt-metal import is therefore inside a
function, never at module scope. That is also what lets the tests below run on CPU.

ENVIRONMENT, set by the launcher
--------------------------------
``MESH_DEVICE``            the SKU, e.g. ``P300x2`` — informational here
``ANIMATEDIFF_MESH_SHAPE`` the resolved shape, e.g. ``1x4`` (the manifest points
                           ``runtime.mesh_shape_env`` at this name; the kind's default is
                           FLUX.2's, which means nothing to this model)
``HF_MODEL``               the weights repo id, reported by ``/v1/models``

ONE REQUEST AT A TIME
---------------------
There is no continuous batching to hide behind: the pipeline owns the device, and two
concurrent denoise loops would interleave on it. Requests are serialised on a lock and the
blocking work runs in a worker thread so the event loop can still answer ``/health`` while
a generation is in flight. This is a deliberate cap, not an oversight — a queue that
admits work the device cannot do concurrently is a queue that reports success and returns
corrupted frames.
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import time
from contextlib import asynccontextmanager
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

#: The env var carrying the resolved mesh shape. Named for THIS model: the kind's default
#: is ``FLUX2_MESH_SHAPE``, which is the default only because FLUX.2 is what it was built
#: for. A manifest serving this app sets ``runtime.mesh_shape_env`` to the name below.
MESH_SHAPE_ENV = "ANIMATEDIFF_MESH_SHAPE"

#: Frames per second baked into the returned GIF.
DEFAULT_FPS = 8


class VideoGenerationRequest(BaseModel):
    """OpenAI-shaped video request.

    Field names follow ``/v1/videos/generations`` as tt-inference-server's tt-media-server
    already serves it, so the same model can sit behind either surface without a client
    change. Everything beyond ``prompt`` is optional with the pipeline's own defaults.
    """

    prompt: str = Field(min_length=1)
    negative_prompt: str = ""
    model: Optional[str] = None
    num_frames: int = Field(default=16, ge=1, le=64)
    num_inference_steps: int = Field(default=25, ge=1, le=100)
    guidance_scale: float = Field(default=7.5, ge=0.0, le=20.0)
    seed: int = 42
    temporal_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    height: int = Field(default=512, ge=64, le=1024)
    width: int = Field(default=512, ge=64, le=1024)
    #: Only b64_json is offered. A URL response would need the server to host files it has
    #: no store for, and a bundle that returns links to a path the consumer cannot read is
    #: worse than one that returns bytes.
    response_format: str = "b64_json"


class VideoData(BaseModel):
    b64_json: str


class VideoGenerationResponse(BaseModel):
    created: int
    data: List[VideoData]


def mesh_shape_from_env(env: Optional[dict] = None) -> tuple:
    """Parse ``ANIMATEDIFF_MESH_SHAPE`` (``"RxC"``) into ``(rows, cols)``.

    Defaults to ``(1, 1)`` when unset, which is what a bare local run wants. A malformed
    value RAISES rather than falling back: silently converting against the wrong mesh is
    the failure this variable exists to prevent, and it would show up as bad frames rather
    than as an error.
    """
    raw = (env if env is not None else os.environ).get(MESH_SHAPE_ENV, "").strip()
    if not raw:
        return (1, 1)
    try:
        rows, cols = (int(p) for p in raw.lower().split("x", 1))
    except ValueError as exc:
        raise ValueError(
            f"{MESH_SHAPE_ENV}={raw!r} is not a mesh shape like '1x4'"
        ) from exc
    if rows < 1 or cols < 1:
        raise ValueError(f"{MESH_SHAPE_ENV}={raw!r} must be positive")
    return (rows, cols)


def _open_device_and_models(shape: tuple) -> tuple:
    """Claim the mesh and load the TTNN model set. Blocking, hardware, lifespan-only."""
    from animatediff_ttnn.generation_helpers import load_sd14_ttnn
    from animatediff_ttnn.ttnn_pipeline import setup_blackhole

    rows, cols = shape
    # setup_blackhole(device_ids=None) opens every available chip as a 1xN mesh, which is
    # what a (1, N) shape means. An explicit id list is passed only for a single chip so a
    # 1x1 run cannot accidentally claim a neighbour someone else is using.
    device = setup_blackhole(device_ids=[0] if rows * cols == 1 else None)
    return device, load_sd14_ttnn(device)


def _frames_to_gif_b64(frames: List[Any], fps: int = DEFAULT_FPS) -> str:
    """PIL frames -> base64 GIF, in memory. Never touches disk: the container has no
    volume to write to and a served artifact should not depend on one."""
    buf = io.BytesIO()
    frames[0].save(
        buf, format="GIF", save_all=True, append_images=frames[1:],
        duration=1000 // max(fps, 1), loop=0,
    )
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _generate(state: dict, req: VideoGenerationRequest) -> List[Any]:
    """Run the denoise loop. Blocking; called in a worker thread under the device lock."""
    from animatediff_ttnn.generation_helpers import encode_prompt
    from animatediff_ttnn.temporal_attention import generate_frames_temporal

    ttnn_model, ttnn_vae, config, torch_time_proj = state["models"]
    return generate_frames_temporal(
        device=state["device"],
        ttnn_model=ttnn_model,
        ttnn_vae=ttnn_vae,
        config=config,
        torch_time_proj=torch_time_proj,
        text_embeddings=encode_prompt(req.prompt, req.negative_prompt),
        num_frames=req.num_frames,
        num_steps=req.num_inference_steps,
        guidance_scale=req.guidance_scale,
        seed=req.seed,
        temporal_alpha=req.temporal_alpha,
        height=req.height,
        width=req.width,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Claim the device and warm the pipeline BEFORE the server reports ready.

    Run in a worker thread so a slow warm cannot wedge the event loop, and awaited to
    completion so ``Application startup complete`` means what tt-model-manager reads it to
    mean. A failure here is deliberately fatal: a server that starts without a device
    would answer /health cheerfully and fail every generation.
    """
    shape = mesh_shape_from_env()
    device, models = await asyncio.to_thread(_open_device_and_models, shape)
    app.state.engine = {
        "device": device, "models": models, "mesh_shape": shape,
        "hf_model": os.environ.get("HF_MODEL", "unknown"),
        "mesh_device": os.environ.get("MESH_DEVICE", "unknown"),
    }
    app.state.device_lock = asyncio.Lock()
    try:
        yield
    finally:
        # Let ttnn close the mesh itself. Bypassing its teardown is what triggers an abort
        # in MetalContext::destroy_all_instances.
        try:
            import ttnn

            ttnn.close_mesh_device(device)
        except Exception:  # noqa: BLE001 - shutdown must not mask the original error
            pass


app = FastAPI(title="tt-animatediff", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    """Liveness. Cheap and lock-free on purpose, so it still answers during a generation."""
    engine = getattr(app.state, "engine", None)
    return {
        "status": "ok" if engine else "starting",
        "model": (engine or {}).get("hf_model"),
        "mesh_device": (engine or {}).get("mesh_device"),
        "mesh_shape": "x".join(str(n) for n in (engine or {}).get("mesh_shape", ())),
    }


@app.get("/v1/models")
async def models() -> dict:
    engine = getattr(app.state, "engine", None)
    name = (engine or {}).get("hf_model", "unknown")
    return {"object": "list", "data": [{"id": name, "object": "model",
                                        "owned_by": "tenstorrent"}]}


@app.post("/v1/videos/generations", response_model=VideoGenerationResponse)
async def videos_generations(req: VideoGenerationRequest) -> VideoGenerationResponse:
    engine = getattr(app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="model is still starting")
    if req.response_format != "b64_json":
        raise HTTPException(
            status_code=400,
            detail=f"response_format {req.response_format!r} is not supported; this "
                   "server returns b64_json only",
        )
    # One denoise loop at a time: the pipeline owns the device.
    async with app.state.device_lock:
        frames = await asyncio.to_thread(_generate, engine, req)
    return VideoGenerationResponse(
        created=int(time.time()),
        data=[VideoData(b64_json=_frames_to_gif_b64(frames))],
    )
