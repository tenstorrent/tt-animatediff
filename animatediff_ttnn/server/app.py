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
``ANIMATEDIFF_LORAS``      optional JSON object ``{"name": "SOURCE[:FILE][@SCALE]", ...}``; each
                           entry becomes a resident LoRA variant a request can select with
                           ``"lora": "name"``. A value may also be a list of specs, which
                           are stacked. Unset means no variants. See ``lora_catalog_from_env``.

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
import re
import shutil
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

#: The env var carrying the resolved mesh shape. Named for THIS model: the kind's default
#: is ``FLUX2_MESH_SHAPE``, which is the default only because FLUX.2 is what it was built
#: for. A manifest serving this app sets ``runtime.mesh_shape_env`` to the name below.
MESH_SHAPE_ENV = "ANIMATEDIFF_MESH_SHAPE"

#: The env var carrying the LoRA catalog: the only way a LoRA reaches this server.
LORAS_ENV = "ANIMATEDIFF_LORAS"

#: What an LCM variant uses when the request leaves steps / guidance_scale out.
LCM_DEFAULT_STEPS = 4
LCM_DEFAULT_GUIDANCE = 1.5

#: Frames per second baked into the returned GIF.
DEFAULT_FPS = 8


#: The only image size this server can produce. The TTNN UNet is compiled at a fixed
#: 64x64 latent, and 64 * 8 = 512. Named rather than repeated so the request model, the
#: error message and any future widening all move together.
SERVABLE_SIZE = 512


class PreviewStore:
    """Progress and the latest preview frame of the generation in flight.

    Written from the worker thread that runs the denoise loop and read from the event loop by
    ``GET /v1/videos/preview``. It holds only plain values behind its own small lock, never
    the device lock: the endpoint must answer while a generation holds the device, the same
    rule ``/tt-liveness`` follows. A reader that waited on the device would show a frozen
    preview for exactly as long as the work it is meant to show.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._dir = tempfile.mkdtemp(prefix="tt-animatediff-preview-")
        #: Where ``make_step_callback`` writes the rolling GIF (atomically, so a read never
        #: sees half a file).
        self.path = os.path.join(self._dir, "preview.gif")
        self._state = {"step": 0, "total": 0, "gif_b64": None, "running": False}

    def begin(self, total: int) -> None:
        """Reset for a new request: a stale frame from the last one must not show."""
        with self._lock:
            self._state = {"step": 0, "total": total, "gif_b64": None, "running": True}

    def step(self, step: int, total: int) -> None:
        with self._lock:
            self._state.update(step=step, total=total)

    def publish(self, line: str) -> None:
        """``emit`` for ``make_step_callback``: read the GIF it just wrote and keep it.

        Reading here, in the worker thread, keeps the endpoint a pure memory read. Any failure
        leaves the previous frame in place: a lost preview must not fail the generation.
        """
        from animatediff_ttnn.preview import parse_preview_line

        parsed = parse_preview_line(line)
        if parsed is None:
            return
        step, total, path = parsed
        try:
            with open(path, "rb") as fh:
                gif = base64.b64encode(fh.read()).decode("ascii")
        except OSError:
            return
        with self._lock:
            self._state.update(step=step, total=total, gif_b64=gif)

    def end(self) -> None:
        with self._lock:
            self._state["running"] = False

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._state)

    def close(self) -> None:
        shutil.rmtree(self._dir, ignore_errors=True)


def _step_callback(preview: "PreviewStore", req: "VideoGenerationRequest", steps: int):
    """The ``on_step`` hook for one request: always count steps, render GIFs only if asked."""
    render = None
    if req.preview:
        from animatediff_ttnn.preview import make_step_callback

        render = make_step_callback(preview.path, steps, every=req.preview_every, emit=preview.publish)

    def on_step(step_idx: int, total: int, frame_latents) -> None:
        # 1-based, to match the PREVIEW line and what a progress bar shows.
        preview.step(step_idx + 1, total)
        if render is not None:
            render(step_idx, total, frame_latents)

    return on_step


class PromptKeyframe(BaseModel):
    """One keyframe of a prompt schedule ("prompt travel"): frame index and the prompt there."""

    frame: int = Field(ge=0)
    prompt: str = Field(min_length=1)


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
    #: Name of a LoRA variant from the operator's catalog (``ANIMATEDIFF_LORAS``). Omit for
    #: the plain model. A name, never a repo id or a path: the operator decides which files
    #: this server will load, so a caller cannot make it fetch or open one.
    lora: Optional[str] = Field(default=None, min_length=1, max_length=64)
    #: Render a rolling preview GIF of the in-flight latents, readable from
    #: ``GET /v1/videos/preview`` while this request runs. Off by default: the step counter
    #: there is always kept (it is free), but each GIF is a CPU render and export between
    #: denoise steps, which is time the device spends waiting.
    preview: bool = False
    #: Render a preview every Nth step (the last step always renders). Omit for the library
    #: default: every step on a run of 10 steps or fewer, every other step beyond that. Each
    #: render added about 0.4 s on one Blackhole chip, so a larger value trades preview
    #: smoothness for generation time.
    preview_every: Optional[int] = Field(default=None, ge=1, le=100)
    #: Keyframe prompts interpolated across the frames ("prompt travel"). ``prompt`` stays
    #: required and is what the schedule falls back to for a frame between none; at least
    #: two keyframes are needed for anything to travel.
    prompt_schedule: Optional[List[PromptKeyframe]] = Field(default=None, min_length=1, max_length=16)
    # 0.35, because that is what the rest of the project calibrated: generate_animation(),
    # hf/pipeline.py AND generate_frames_temporal -- the function this server calls -- all
    # use it. This said 0.5, which matched none of them, so an HTTP client omitting the
    # field got quietly different motion from every other entry point into the same code.
    # test_the_server_request_default_matches_the_librarys_calibrated_one asserts it
    # against the library rather than against a literal, so it cannot drift again.
    temporal_alpha: float = Field(default=0.35, ge=0.0, le=1.0)
    # 512 exactly, and not as a conservative default: the TTNN UNet is compiled ONCE at a
    # fixed 64x64 latent (generation_helpers.py: `UNet2D(device, parameters, 2, 64, 64)`),
    # so 512x512 is the only thing this server can produce. These were bounded 64..1024,
    # which is pydantic-valid and kernel-invalid -- the request passed validation, took the
    # device lock, and then raised a TT_FATAL deep in the denoise loop as an uncaught 500,
    # having held the one device to do it. Refuse at the edge instead, with a message that
    # says why. Widening this means compiling the UNet per requested shape, which is a
    # ~2-3 minute cost per shape and a different design.
    height: int = Field(default=SERVABLE_SIZE, ge=SERVABLE_SIZE, le=SERVABLE_SIZE,
                        description=f"only {SERVABLE_SIZE} is supported: the UNet is "
                                    "compiled at a fixed 64x64 latent")
    width: int = Field(default=SERVABLE_SIZE, ge=SERVABLE_SIZE, le=SERVABLE_SIZE,
                       description=f"only {SERVABLE_SIZE} is supported: the UNet is "
                                   "compiled at a fixed 64x64 latent")
    #: Only b64_json is offered. A URL response would need the server to host files it has
    #: no store for, and a bundle that returns links to a path the consumer cannot read is
    #: worse than one that returns bytes.
    response_format: str = "b64_json"

    @model_validator(mode="after")
    def _keyframes_fall_inside_the_clip(self):
        # A keyframe past the last frame would be silently ignored by the interpolator;
        # refuse it where the caller can see why.
        for kf in self.prompt_schedule or ():
            if kf.frame >= self.num_frames:
                raise ValueError(
                    f"prompt_schedule frame {kf.frame} is outside the clip (num_frames={self.num_frames})"
                )
        return self


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
    # Only 1x1, and refused here rather than absorbed. _open_device_and_models passes
    # `device_ids=None` for any other shape, and setup_blackhole(None) opens EVERY chip on
    # the box as a 1xN mesh -- so on a shared machine a stray 1x4 takes four chips out of
    # circulation, and then dies in the lifespan anyway, because this model cannot shard:
    # a real serve at P300x2 failed with `Can't convert a tensor distributed on
    # MeshShape([1, 4]) mesh to row-major logical tensor`. The manifest declares P150 for
    # the same reason. Widen this only alongside the mesh frame-sharding work.
    if rows * cols != 1:
        raise ValueError(
            f"{MESH_SHAPE_ENV}={raw!r} requests {rows * cols} chips, but this model cannot "
            f"shard -- only 1x1 is supported until mesh frame-sharding ships "
            f"(docs/superpowers/plans/2026-06-15-mesh-frame-sharding.md). A larger shape "
            f"would claim every chip on the box and still fail in the lifespan."
        )
    return (rows, cols)


_LORA_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def lora_catalog_from_env(env: Optional[dict] = None) -> dict:
    """Parse ``ANIMATEDIFF_LORAS`` into ``{name: [LoraSpec, ...]}``.

    Unset or empty is an empty catalog. Anything malformed RAISES, for the same reason
    ``mesh_shape_from_env`` does: a server that quietly dropped a LoRA would come up healthy
    and then answer a request for it with the plain model's pictures. Raising happens
    before the device is opened, so a typo costs nothing.

    The value is a JSON object. Each value is one spec string or a list of them (stacked, in
    order), in the CLI form ``SOURCE[:FILE][@SCALE]``. ``base`` is refused as a name only
    because it reads like the plain model; omit ``lora`` to get that.
    """
    import json

    from animatediff_ttnn.lora import parse_lora_spec

    raw = (env if env is not None else os.environ).get(LORAS_ENV, "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"{LORAS_ENV} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f'{LORAS_ENV} must be a JSON object like {{"name": "repo:file.safetensors@0.8"}}')
    catalog = {}
    for name, value in data.items():
        if not _LORA_NAME.match(name) or name == "base":
            raise ValueError(f"{LORAS_ENV}: {name!r} is not a usable LoRA name (letters, digits, . _ -; not 'base')")
        texts = [value] if isinstance(value, str) else value
        if not isinstance(texts, list) or not texts or not all(isinstance(t, str) for t in texts):
            raise ValueError(f"{LORAS_ENV}: {name!r} must be a spec string or a non-empty list of them")
        catalog[name] = [parse_lora_spec(t) for t in texts]
    return catalog


def _load_lora_variants(device, models: tuple, catalog: dict) -> dict:
    """Build one resident model set per catalog entry, next to the plain ``models``.

    Blocking, hardware, lifespan-only. Each variant costs a UNet's weights on the device
    (the VAE is shared), and no more compile time than the plain model, because the kernels
    are keyed on shapes. A bad file raises here, at startup, which is where a readiness
    probe wants the failure: ``merge_loras`` rejects wrong-base and unsupported files before
    it loads anything.
    """
    from animatediff_ttnn.generation_helpers import load_sd14_variant

    vae = models[1]
    return {name: load_sd14_variant(device, specs, vae) for name, specs in catalog.items()}


def _open_device_and_models(shape: tuple) -> tuple:
    """Claim the mesh and load the TTNN model set. Blocking, hardware, lifespan-only."""
    from animatediff_ttnn.generation_helpers import load_sd14_ttnn
    from animatediff_ttnn.ttnn_pipeline import setup_blackhole

    from animatediff_ttnn.generation_helpers import encode_prompt

    rows, cols = shape
    # setup_blackhole(device_ids=None) opens every available chip as a 1xN mesh, which is
    # what a (1, N) shape means. An explicit id list is passed only for a single chip so a
    # 1x1 run cannot accidentally claim a neighbour someone else is using.
    device = setup_blackhole(device_ids=[0] if rows * cols == 1 else None)
    # Everything after the open is wrapped: a failure here used to leave the chip claimed
    # by a container that never started, and on a restart loop that is one claimed device
    # per attempt while TTNN refuses to reopen any of them. Same leak session.py was fixed
    # for; the server has its own open path, so it needs its own rollback.
    try:
        models = load_sd14_ttnn(device)

    # Warm the TEXT encoder too, and not as an optimisation. load_sd14_ttnn brings up the
    # UNet and the VAE; CLIP's tokenizer and text encoder were left to be downloaded
    # lazily by the first encode_prompt() call -- which happens inside _generate, under
    # the device lock, AFTER /health has already answered 200. So the readiness contract
    # this module is built on ("the device open and the model warm happen in the lifespan
    # and nowhere else") was false for one of the three models, and in a container with
    # only unet/vae cached and no egress, the server reported ready and then failed every
    # request. Downloading here means a missing weight fails startup, which is what a
    # readiness probe is for.
    #
    # NOT a full denoise pass: that would also pay TTNN kernel compilation here, which is
    # minutes on a cold cache and would delay readiness for every boot. The first request
    # is therefore still slower than the rest -- slow, but correct, and measured in
    # docs/measurements/serving-benchmark.json rather than hidden.
        encode_prompt("warmup", "")
    except Exception:
        _close_device(device)
        raise
    return device, models


def _close_device(device) -> None:
    """Release the mesh, swallowing any error.

    Swallowed deliberately: on the failure path this must not mask the exception the
    operator actually needs to read, and a close that fails is not a reason to hold the
    chip. Mirrors animatediff_ttnn.session._close_device.
    """
    try:
        import ttnn
        ttnn.close_mesh_device(device)
    except Exception:
        pass


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

    variant = state.get("variants", {}).get(req.lora) if req.lora else None
    # A variant brings its own text encoder: its UNet and encoder were merged together.
    models, clip = (variant.models, variant.clip) if variant else (state["models"], None)
    ttnn_model, ttnn_vae, config, torch_time_proj = models
    use_lcm = bool(variant and variant.lcm)
    steps, guidance = req.num_inference_steps, req.guidance_scale
    if use_lcm:
        # An LCM variant is a 2-8 step, low-guidance model; the request defaults (25, 7.5)
        # would waste time and oversaturate it. Only fields the caller did not send change.
        if "num_inference_steps" not in req.model_fields_set:
            steps = LCM_DEFAULT_STEPS
        if "guidance_scale" not in req.model_fields_set:
            guidance = LCM_DEFAULT_GUIDANCE

    embeddings = encode_prompt(req.prompt, req.negative_prompt, clip=clip)
    per_frame = None
    if req.prompt_schedule:
        per_frame = encode_prompt(
            req.prompt, req.negative_prompt, clip=clip,
            prompt_schedule=[(kf.frame, kf.prompt) for kf in req.prompt_schedule],
            num_frames=req.num_frames,
        )
    preview = state.get("preview")
    on_step = _step_callback(preview, req, steps) if preview is not None else None
    if preview is not None:
        preview.begin(steps)
    try:
        return generate_frames_temporal(
            device=state["device"],
            ttnn_model=ttnn_model,
            ttnn_vae=ttnn_vae,
            config=config,
            torch_time_proj=torch_time_proj,
            text_embeddings=embeddings,
            text_embeddings_per_frame=per_frame,
            num_frames=req.num_frames,
            num_steps=steps,
            guidance_scale=guidance,
            seed=req.seed,
            temporal_alpha=req.temporal_alpha,
            use_lcm=use_lcm,
            on_step=on_step,
            height=req.height,
            width=req.width,
        )
    finally:
        if preview is not None:
            preview.end()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Claim the device and warm the pipeline BEFORE the server reports ready.

    Run in a worker thread so a slow warm cannot wedge the event loop, and awaited to
    completion so ``Application startup complete`` means what tt-model-manager reads it to
    mean. A failure here is deliberately fatal: a server that starts without a device
    would answer /health cheerfully and fail every generation.
    """
    shape = mesh_shape_from_env()
    # Parsed before the device opens: a malformed catalog should cost nothing.
    catalog = lora_catalog_from_env()
    device, models = await asyncio.to_thread(_open_device_and_models, shape)
    try:
        variants = await asyncio.to_thread(_load_lora_variants, device, models, catalog) if catalog else {}
    except BaseException:
        # Same rollback _open_device_and_models does: a failed start must not keep a chip.
        _close_device(device)
        raise
    preview = PreviewStore()
    app.state.engine = {
        "device": device, "models": models, "variants": variants, "mesh_shape": shape,
        "preview": preview,
        "hf_model": os.environ.get("HF_MODEL", "unknown"),
        "mesh_device": os.environ.get("MESH_DEVICE", "unknown"),
    }
    app.state.device_lock = asyncio.Lock()
    try:
        yield
    finally:
        preview.close()
        # Let ttnn close the mesh itself. Bypassing its teardown is what triggers an abort
        # in MetalContext::destroy_all_instances.
        try:
            import ttnn

            ttnn.close_mesh_device(device)
        except Exception:  # noqa: BLE001 - shutdown must not mask the original error
            pass


app = FastAPI(title="tt-animatediff", lifespan=lifespan)


def _readiness() -> dict:
    """What both probes report about the model, in one place so they cannot disagree."""
    engine = getattr(app.state, "engine", None)
    return {
        "model_ready": engine is not None,
        "model": (engine or {}).get("hf_model"),
        "mesh_device": (engine or {}).get("mesh_device"),
        "mesh_shape": "x".join(str(n) for n in (engine or {}).get("mesh_shape", ())),
    }


@app.get("/tt-liveness")
async def tt_liveness() -> dict:
    """LIVENESS probe -- "is this process able to respond at all", restart on failure.

    Matches tt-inference-server's tt-media-server
    (``open_ai_api/tt_maintenance_api.py``): 200 with ``{"status": "alive", ...}`` plus the
    model-readiness payload, and a non-200 ONLY on unrecoverable failure. A model that is
    still warming is alive, not broken -- that is exactly the distinction this endpoint
    exists to draw, and reporting it as failure would have an orchestrator restart a server
    that is loading correctly.

    Deliberately lock-free and engine-free, so it keeps answering while a generation holds
    the device. A liveness probe that blocks behind the work it is monitoring is how a
    healthy-but-busy server gets killed.
    """
    try:
        return {"status": "alive", **_readiness()}
    except Exception as exc:  # noqa: BLE001 - the one case that IS unrecoverable
        raise HTTPException(status_code=500, detail=f"Liveness check failed: {exc}")


@app.get("/health")
async def health() -> dict:
    """READINESS probe -- "should traffic be routed here", gate on it.

    Despite the names, `/health` is readiness and `/tt-liveness` is liveness. That is
    tt-media-server's convention, kept for compatibility with the Python server, and the
    body is vLLM's: an empty dict when ready, 503 when not.

    This returned 200 with ``{"status": "starting"}`` until the semantics were checked
    against tt-media-server. A readiness probe answering 200 while the pipeline is still
    loading tells a load balancer to send work to a server that cannot do it -- the precise
    failure the lifespan design exists to prevent, reintroduced one layer up. The
    diagnostic payload now lives on /tt-liveness, which is where tt-media-server puts it.
    """
    if not _readiness()["model_ready"]:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {}


@app.get("/v1/videos/preview")
async def videos_preview() -> dict:
    """Progress and latest preview frame of the generation in flight.

    ``{"step": n, "total": N, "gif_b64": "<base64 GIF>" | null, "running": bool}``. ``step``
    is 1-based. ``gif_b64`` is null until a request sent ``"preview": true`` has produced a
    frame, and is reset at the start of each request. Never blocks on the device lock, so it
    answers while a generation holds the device; it is a memory read behind a tiny lock.
    Always 200, including before the model is ready (all zeros, not running): a poller
    should not need to tell "starting" from "idle".
    """
    preview = (getattr(app.state, "engine", None) or {}).get("preview")
    if preview is None:
        return {"step": 0, "total": 0, "gif_b64": None, "running": False}
    return preview.snapshot()


@app.get("/v1/models")
async def models() -> dict:
    engine = getattr(app.state, "engine", None)
    name = (engine or {}).get("hf_model", "unknown")
    return {"object": "list", "data": [{"id": name, "object": "model",
                                        "owned_by": "tenstorrent"}],
            # The names a request may pass as "lora"; empty when the operator set none.
            "loras": sorted((engine or {}).get("variants", {}))}


@app.post("/v1/videos/generations", response_model=VideoGenerationResponse)
async def videos_generations(req: VideoGenerationRequest) -> VideoGenerationResponse:
    engine = getattr(app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="model is still starting")
    if req.lora is not None and req.lora not in engine.get("variants", {}):
        # 400, and before the device lock: an unknown name must not queue behind a
        # generation just to be refused.
        available = sorted(engine.get("variants", {}))
        raise HTTPException(
            status_code=400,
            detail=f"unknown lora {req.lora!r}; this server offers "
                   f"{available if available else 'none (ANIMATEDIFF_LORAS is not set)'}",
        )
    if req.response_format != "b64_json":
        raise HTTPException(
            status_code=400,
            detail=f"response_format {req.response_format!r} is not supported; this "
                   "server returns b64_json only",
        )
    # One denoise loop at a time: the pipeline owns the device.
    async with app.state.device_lock:
        frames = await asyncio.to_thread(_generate, engine, req)
    # Off the loop as well, and NOT because it is slow in absolute terms: PIL's GIF
    # palette quantization plus base64 of a multi-MB payload is seconds of pure CPU, and
    # run inline here it blocks every coroutine -- including /health and /tt-liveness,
    # whose whole purpose is to answer while this endpoint is busy. An orchestrator with a
    # 1-2 s liveness timeout would restart a server that is merely encoding. Outside the
    # device lock, because the device is free once the frames are in host memory.
    b64 = await asyncio.to_thread(_frames_to_gif_b64, frames)
    return VideoGenerationResponse(
        created=int(time.time()),
        data=[VideoData(b64_json=b64)],
    )
