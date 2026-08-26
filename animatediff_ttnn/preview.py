# SPDX-License-Identifier: Apache-2.0
"""Per-step latent previews for the CLI runner.

``generate_frames_temporal`` has accepted an ``on_step`` hook, and
``temporal_attention._latent_preview`` has rendered a fast CPU-side preview of
the in-flight latents, since the Gradio UI landed.  But only ``app.py`` ever
wired the two together — the CLI runner (``examples/generate.py``), which is the
entry point ``tt-local-generator`` execs, ran blind.  So a generation could only
be *watched* inside the Hugging Face Space.

This module is the shared, importable half of closing that gap: it builds the
``on_step`` callback and defines the one line format the runner prints, so a
consumer draining the runner's stdout can show the image forming.  The runner
itself stays thin, and everything here is unit-testable without torch, PIL, or
hardware.

**The preview is free.** ``_latent_preview`` maps 3 of the 4 latent channels to
RGB with a tanh soft-clip and a bilinear upsample — pure CPU tensor ops on the
small latent grid, no VAE decode and no device work.  It adds no latency to the
hardware pipeline, which is what makes per-step previews affordable at all.

**A preview must never break a generation.** Every failure here is swallowed:
losing a preview frame costs a UI update, whereas raising would cost the user
their whole run.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

#: Marker prefixed to every preview announcement on stdout.  Consumers match on
#: this rather than parsing free-form log text.
PREVIEW_PREFIX = "PREVIEW:"

#: ``PREVIEW: <step>/<total> <path>``.  The path group runs to the end of the
#: line, so a path containing spaces survives the round trip; it is non-greedy
#: precisely so the trailing ``\s*$`` can strip trailing whitespace rather than
#: swallowing it into the path.
_PREVIEW_RE = re.compile(
    r"^\s*" + re.escape(PREVIEW_PREFIX) + r"\s+(\d+)/(\d+)\s+(.+?)\s*$"
)

#: Default preview cadence, mirroring ``app.py``: every step on a short run,
#: every other step on a longer one.  Frequent enough to feel live, sparse
#: enough not to flood a consumer on a 25-step run.
_SHORT_RUN_STEPS = 10


def default_every(num_steps: int) -> int:
    """Preview cadence for a run of ``num_steps`` steps."""
    return 1 if num_steps <= _SHORT_RUN_STEPS else 2


# Indirection so tests can substitute these without importing torch/PIL, and so
# the heavy imports stay lazy (this module is imported by argument parsing).
def _latent_preview_fn():
    from animatediff_ttnn.temporal_attention import _latent_preview

    return _latent_preview


def _export_gif_fn():
    from animatediff_ttnn.pipeline import export_gif

    return export_gif


def emit_line(line: str) -> None:
    """Default ``emit``: print the announcement and FLUSH.

    A bare ``print`` is not enough.  Python block-buffers stdout whenever it is
    not a TTY, which is exactly the case this feature exists for — the consumer
    runs us under ``subprocess.Popen(stdout=PIPE)``.  Measured with the plain
    default: three lines printed a second apart all arrived together at process
    exit, so a "live" preview would have shown nothing until the generation had
    already finished.  Flushing per line is what makes the streaming contract
    real without requiring the caller to remember ``python -u``.
    """
    print(line, flush=True)


def parse_preview_line(line: str) -> "tuple[int, int, str] | None":
    """Parse a preview announcement into ``(step, total, path)``.

    ``step`` is 1-based, matching what the line reports.  Returns ``None`` for
    any line that is not a preview announcement, so a consumer can feed it every
    line of the runner's output.
    """
    if not line:
        return None
    m = _PREVIEW_RE.match(line)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), m.group(3)


def make_step_callback(
    preview_path: "str | None",
    num_steps: int,
    *,
    height: int = 256,
    width: int = 256,
    every: "int | None" = None,
    emit=emit_line,
):
    """Build the ``on_step`` callback for ``generate_frames_temporal``.

    Args:
        preview_path: where to write the rolling preview GIF.  ``None``/empty
            returns ``None`` — previews are opt-in, and without them the runner
            behaves exactly as it did before.
        num_steps: total denoising steps, used to pick the default cadence.
        height, width: size the preview is upsampled to.  Deliberately HALF the
            512x512 output: the source is a 64x64 latent grid, so a 512 upsample
            carries no more information than a 256 one — it just costs more CPU,
            a bigger GIF to rewrite each step, and (for a GUI consumer) a
            progress thumbnail that dwarfs the finished result.
        every: emit on every Nth step; defaults to :func:`default_every`.  The
            FINAL step always emits regardless — it is the closest thing to the
            real result, so skipping it would leave the last thing on screen
            noticeably rougher than what was actually produced.
        emit: where the announcement line goes.  Defaults to
            :func:`emit_line`, which prints AND flushes — see its docstring for
            why an unflushed ``print`` silently breaks streaming.

    Returns:
        ``on_step(step_idx, total_steps, frame_latents)``, or ``None``.
    """
    if not preview_path:
        return None

    cadence = every if every and every > 0 else default_every(num_steps)
    target = Path(preview_path)

    def on_step(step_idx: int, total_steps: int, frame_latents) -> None:
        is_final = step_idx == total_steps - 1
        if step_idx % cadence != 0 and not is_final:
            return
        try:
            frames = _latent_preview_fn()(frame_latents, height, width)
            _write_atomic(frames, target)
        except Exception:
            return  # a lost preview frame is never worth failing a run over
        try:
            emit(f"{PREVIEW_PREFIX} {step_idx + 1}/{total_steps} {target}")
        except Exception:
            pass

    return on_step


def _write_atomic(frames, target: Path) -> None:
    """Write the preview GIF so a reader never sees a half-written file.

    The consumer is a GUI polling this exact path while we rewrite it each
    step, so writing in place would sooner or later hand it a truncated GIF.
    Render to a temp file in the same directory (same filesystem, so the
    replace is atomic) and move it into position.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(target.parent), prefix=target.stem + ".", suffix=".gif"
    )
    os.close(fd)
    try:
        _export_gif_fn()(frames, tmp)
        os.replace(tmp, target)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def as_diffusers_callback(on_step, total_steps: int = 0):
    """Adapt an ``on_step`` callback to diffusers' ``callback_on_step_end``.

    The TTNN path calls ``on_step(step_idx, total, frame_latents)`` directly with
    a list of per-frame latents.  The CPU path runs a diffusers
    ``AnimateDiffPipeline``, which calls
    ``callback(pipe, step, timestep, callback_kwargs)`` and carries all frames in
    ONE ``(B, C, F, H, W)`` tensor.  Splitting on the frame axis lets a single
    preview callback serve both, so ``--preview-path`` behaves identically in
    either mode rather than being silently ignored on CPU.

    Returns ``None`` when ``on_step`` is ``None``, so the caller can pass the
    result straight through to the pipeline.
    """
    if on_step is None:
        return None

    def _callback(_pipe, step: int, _timestep, callback_kwargs: dict) -> dict:
        try:
            latents = (callback_kwargs or {}).get("latents")
            if latents is not None:
                # (B, C, F, H, W) -> one (1, C, H, W) per frame.
                frames = [latents[:1, :, i] for i in range(latents.shape[2])]
                on_step(step, total_steps, frames)
        except Exception:
            pass  # a preview must never take down the pipeline it is watching
        return {}

    return _callback
