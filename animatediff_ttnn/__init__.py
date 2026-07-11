# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""AnimateDiff for Tenstorrent hardware.

High-level API (use this from external callers, e.g. an InvokeAI node)::

    from animatediff_ttnn import generate_animation, export_gif, export_mp4

    frames = generate_animation(
        prompt="swirling nebula, teal and gold, cinematic",
        num_frames=8,
        num_steps=25,
        seed=42,
    )
    export_mp4(frames, "output.mp4")

The mode parameter selects the compute backend:
  "auto"      — Blackhole if TTNN is importable, CPU otherwise
  "blackhole" — require real Blackhole hardware
  "sim"       — ttsim virtual device (pass sim_so= for a non-default path)
  "cpu"       — Phase 1: AnimateDiffPipeline on CPU (any machine, no TTNN)

Low-level phase-specific APIs for scripts and tests::

    # Phase 1 — CPU
    from animatediff_ttnn.pipeline import create_animatediff_pipeline, generate

    # Phase 2.5 — Blackhole temporal attention
    from animatediff_ttnn.temporal_attention import generate_frames_temporal

    # Phase 3 — MotionAdapter on Blackhole
    from animatediff_ttnn.temporal_attention import generate_frames_motion
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, List, Optional

try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("animatediff-ttnn")
    except PackageNotFoundError:
        # Editable / source checkout: fall back to repo-root VERSION file
        __version__ = (Path(__file__).parent.parent / "VERSION").read_text().strip()
except Exception:
    __version__ = "unknown"

# Phase 1 pipeline helpers — re-exported for backwards compatibility.
from .pipeline import create_animatediff_pipeline, generate, export_gif

# Per-mode CPU pipeline cache: (use_lightning, lightning_steps) → pipe
_cpu_pipe_cache: dict = {}
_cpu_cache_lock = threading.Lock()


def generate_animation(
    prompt: str,
    negative_prompt: str = "",
    num_frames: int = 8,
    num_steps: int = 25,
    guidance_scale: float = 7.5,
    seed: int = 42,
    temporal_alpha: float = 0.35,
    height: int = 512,
    width: int = 512,
    mode: str = "auto",
    sim_so: Optional[str] = None,
    use_lightning: bool = False,
    lightning_steps: int = 4,
    chain_from: Optional[str] = None,
    chain_save: Optional[str] = None,
    chain_alpha: float = 0.6,
    on_step: Optional[Callable] = None,
) -> List:
    """Generate an AnimateDiff animation and return frames as PIL Images.

    This is the single entry point for external callers (InvokeAI nodes,
    scripts, test harnesses). It manages device and model lifetime internally
    — the TTNN device and compiled UNet are initialized on the first call and
    cached for all subsequent calls in the same process.

    Args:
        prompt: Text description of the animation.
        negative_prompt: Features to suppress in the output.
        num_frames: Number of frames to generate (8 or 16 recommended).
        num_steps: Denoising steps. 25 is a good default; Lightning mode
                   accepts any count but was trained at 4/8.
        guidance_scale: CFG scale. 7.5 is standard; use 1.0 with Lightning.
        seed: Random seed. Shared base noise + per-frame perturbation, so
              the same seed gives the same animation across identical calls.
        temporal_alpha: Cross-frame attention blend weight (Blackhole/sim only).
                        0.0 → shared-noise coherence only (Phase 2 behaviour);
                        1.0 → full attention; 0.35 is the calibrated default.
        height: Output frame height in pixels. 512 recommended.
        width: Output frame width in pixels. 512 recommended.
        mode: Compute backend — "auto", "blackhole", "sim", or "cpu".
              "auto" tries Blackhole/sim first and falls back to CPU if TTNN
              is not importable (e.g. on a Mac or a machine without tt-metal).
        sim_so: Path to libttsim_bh.so (required when mode="sim").
                Defaults to ~/sim/libttsim_bh.so if not given.
        use_lightning: Use EulerDiscreteScheduler instead of PNDM. On CPU
                       mode this requires the distilled Lightning checkpoint
                       and a matching lightning_steps value (2/4/8 only). On
                       Blackhole/sim the base SD 1.4 UNet is used with Euler
                       so any step count is valid.
        lightning_steps: CPU-only Lightning checkpoint step count (2, 4, or 8).
                         Ignored in Blackhole/sim mode.
        chain_from: Path to a .pt latent file saved by a previous run's
                    chain_save. Its latents are blended into this run's base
                    noise at chain_alpha weight for visual continuity across
                    sequential prompts. Blackhole/sim only.
        chain_save: Path to write this run's final latents as a .pt file so
                    the next run can use them via chain_from. Blackhole/sim only.
        chain_alpha: Blend weight for chain_from latents (0 = ignore, 1 = replace
                     base noise entirely). Default 0.6.
        on_step: Optional callback called after each complete denoising step
                 with signature (step_idx: int, num_steps: int,
                 frame_latents: list[torch.Tensor]). Blackhole/sim only.
                 Use animatediff_ttnn.temporal_attention._latent_preview() to
                 convert frame_latents to preview PIL images cheaply (no VAE).

    Returns:
        List of PIL.Image objects, one per frame, in generation order.

    Raises:
        FileNotFoundError: sim_so path does not exist (mode="sim").
        RuntimeError: Blackhole device failed to open.
        ValueError: num_frames is not a multiple of num_chips (mesh sharding
                    constraint; only raised on multi-chip Blackhole setups).
    """
    resolved = _resolve_mode(mode)

    if resolved == "cpu":
        return _generate_cpu(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            num_steps=lightning_steps if use_lightning else num_steps,
            guidance_scale=1.0 if use_lightning else guidance_scale,
            seed=seed,
            use_lightning=use_lightning,
            lightning_steps=lightning_steps,
        )

    # Blackhole or sim — Phase 2.5: temporal cross-frame attention on TTNN.
    from animatediff_ttnn.session import ensure_blackhole
    from animatediff_ttnn.generation_helpers import encode_prompt
    from animatediff_ttnn.temporal_attention import generate_frames_temporal

    device, (ttnn_model, ttnn_vae, config, torch_time_proj) = ensure_blackhole(
        mode=resolved, sim_so=sim_so
    )
    text_embeddings = encode_prompt(prompt, negative_prompt)

    return generate_frames_temporal(
        device=device,
        ttnn_model=ttnn_model,
        ttnn_vae=ttnn_vae,
        config=config,
        torch_time_proj=torch_time_proj,
        text_embeddings=text_embeddings,
        num_frames=num_frames,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        temporal_alpha=temporal_alpha,
        use_lightning=use_lightning,
        chain_from=chain_from,
        chain_save=chain_save,
        chain_alpha=chain_alpha,
        on_step=on_step,
        height=height,
        width=width,
    )


def export_mp4(
    frames: List,
    output_path: str,
    fps: int = 8,
) -> None:
    """Save a list of PIL Images as an MP4 video file via ffmpeg.

    Requires ffmpeg on PATH (standard on Ubuntu; brew install ffmpeg on macOS).
    Writes a libx264-encoded MP4 with yuv420p pixel format for broad playback
    compatibility (including InvokeAI's video preview and browser players).

    Args:
        frames: List of PIL.Image objects (all must be the same size).
        output_path: Destination file path; should end in .mp4.
        fps: Frames per second for playback. 8 is the AnimateDiff default.

    Raises:
        RuntimeError: ffmpeg is not on PATH or encoding fails.
        ValueError: frames is empty.
    """
    if not frames:
        raise ValueError("frames list is empty")

    import io
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it with:\n"
            "  Ubuntu: sudo apt install ffmpeg\n"
            "  macOS:  brew install ffmpeg"
        )

    # Feed frames to ffmpeg via stdin as raw PNG data. This avoids writing
    # temporary files and works regardless of output path permissions.
    w, h = frames[0].size
    cmd = [
        "ffmpeg", "-y",
        "-f", "image2pipe",
        "-framerate", str(fps),
        "-i", "pipe:0",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in frames:
        buf = io.BytesIO()
        frame.save(buf, format="PNG")
        proc.stdin.write(buf.getvalue())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg exited with code {proc.returncode}:\n"
            + proc.stderr.read().decode(errors="replace")
        )


# ── private helpers ────────────────────────────────────────────────────────────

def _resolve_mode(mode: str) -> str:
    """Map "auto" to "blackhole" or "cpu" based on TTNN availability."""
    if mode != "auto":
        return mode
    return "blackhole" if _ttnn_available() else "cpu"


def _ttnn_available() -> bool:
    try:
        import ttnn  # noqa: F401
        return True
    except ImportError:
        return False


def _generate_cpu(
    prompt: str,
    negative_prompt: str,
    num_frames: int,
    num_steps: int,
    guidance_scale: float,
    seed: int,
    use_lightning: bool,
    lightning_steps: int,
) -> List:
    from .pipeline import create_animatediff_pipeline, create_lightning_pipeline, generate as _gen

    key = (use_lightning, lightning_steps)
    with _cpu_cache_lock:
        if key not in _cpu_pipe_cache:
            _cpu_pipe_cache[key] = (
                create_lightning_pipeline(step=lightning_steps)
                if use_lightning
                else create_animatediff_pipeline()
            )
    pipe = _cpu_pipe_cache[key]
    return _gen(
        pipe,
        prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
        seed=seed,
    )


__all__ = [
    # High-level API (prefer these)
    "generate_animation",
    "export_gif",
    "export_mp4",
    # Phase 1 pipeline helpers (for callers that want explicit pipeline control)
    "create_animatediff_pipeline",
    "generate",
]
