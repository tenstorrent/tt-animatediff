# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Hugging Face Space for tt-animatediff — capped CPU reference demo.

Blackhole hardware is unreachable from HF infrastructure, so this Space runs the
**CPU** path with ByteDance's distilled 4-step Lightning checkpoint and hard
caps that keep a generation finishable on a free-tier 2-vCPU box. It exists to
prove the pipeline loads and runs anywhere; the pre-rendered gallery below is
what the hardware path actually produces.

The pipeline is loaded exactly the way a user would load it, through
from_pretrained against the Hub repo, so this file is also a working usage
example.
"""

import os
import tempfile
import uuid
from pathlib import Path

import gradio as gr
from diffusers import DiffusionPipeline

MODEL_REPO = os.environ.get("TT_MODEL_REPO", "episod/tt-animatediff")

# Hard caps for a free-tier 2-vCPU CPU box. Raising any of these makes a
# generation outlast the request timeout rather than merely being slow.
MAX_FRAMES = 4
#: Distilled Lightning checkpoints available on the CPU path are 2, 4, and 8
#: steps. Only 2 and 4 are offered here — 8 is needlessly slow on free-tier
#: CPU. This is NOT a "max steps" cap the way MAX_FRAMES is a max: it is the
#: exhaustive set of choices, because each step count is a distinct
#: distilled checkpoint, not an arbitrary denoising-loop length.
STEP_CHOICES = (2, 4)
DEFAULT_STEPS = 4

# There is no resolution control here: animatediff_ttnn's CPU backend
# (_generate_cpu()) accepts no height/width parameters at all and is
# hardwired to 512x512. An earlier version of this file passed
# height=width=384 to the pipeline, which _generate_cpu() silently ignored —
# the Space actually ran at 512x512 the whole time. Fixed by not passing
# height/width at all rather than by resizing.

GALLERY_DIR = Path(__file__).parent / "gallery"

BANNER = """
### CPU reference demo — not representative of Blackhole performance

This Space runs on a free-tier **CPU** with a distilled Lightning checkpoint
(2 or 4 steps, selectable below), capped to 4 frames at 512×512. Tenstorrent
hardware is not reachable from Hugging Face infrastructure.

**A 4-frame generation takes several minutes on free-tier CPU.** That wait is
expected — the queue below is doing its job, not stuck.

On a Blackhole P300C the same model runs **~12.5 s/frame** at 25 steps and
512×512 — see the [measured numbers](https://github.com/tenstorrent/tt-animatediff#modes-reference)
and the pre-rendered hardware output below.
"""

_pipe = None


def _pipeline():
    """Load the custom pipeline once, on first generation."""
    global _pipe
    if _pipe is None:
        _pipe = DiffusionPipeline.from_pretrained(
            MODEL_REPO, custom_pipeline=MODEL_REPO, trust_remote_code=True
        )
    return _pipe


def generate(prompt, negative_prompt, num_frames, lightning_steps, seed):
    if not prompt or not prompt.strip():
        raise gr.Error("Enter a prompt first.")

    frames = _pipeline()(
        prompt=prompt.strip(),
        negative_prompt=negative_prompt.strip(),
        num_frames=min(int(num_frames), MAX_FRAMES),
        guidance_scale=1.0,          # Lightning is distilled for CFG 1.0
        seed=int(seed),
        # No height/width: the CPU backend ignores them and is fixed at
        # 512x512 (see the module comment above STEP_CHOICES).
        mode="cpu",
        use_lightning=True,
        lightning_steps=int(lightning_steps),
    ).frames

    # One file per request, not a fixed path. Gradio serves the returned path
    # after this function returns, so a shared filename lets the next job
    # overwrite the bytes a previous visitor is still being served — they would
    # see someone else's animation. concurrency_limit=1 serializes generation
    # but does nothing about that serve-after-return window.
    out = Path(tempfile.gettempdir()) / f"tt-animatediff-{uuid.uuid4().hex}.gif"
    frames[0].save(
        out, save_all=True, append_images=frames[1:], duration=125, loop=0
    )
    return str(out)


def _gallery_items():
    if not GALLERY_DIR.is_dir():
        return []
    return sorted(str(p) for p in GALLERY_DIR.glob("*.gif"))


with gr.Blocks(title="tt-animatediff") as demo:
    gr.Markdown("# tt-animatediff — AnimateDiff on Tenstorrent Blackhole")
    gr.Markdown(BANNER)

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                value="a swirling nebula, teal and gold, cinematic",
                lines=2,
            )
            negative_prompt = gr.Textbox(
                label="Negative prompt",
                value="blurry, low quality, distorted, text",
                lines=1,
            )
            with gr.Row():
                frames_slider = gr.Slider(
                    2, MAX_FRAMES, value=MAX_FRAMES, step=1, label="Frames (max 4)"
                )
                # Radio, not a slider: these are two distinct distilled
                # Lightning checkpoints (2-step, 4-step), not arbitrary points
                # on a range. The value is forwarded as lightning_steps.
                steps_radio = gr.Radio(
                    choices=list(STEP_CHOICES),
                    value=DEFAULT_STEPS,
                    label="Lightning steps (distilled checkpoint)",
                )
            seed_num = gr.Number(value=42, label="Seed", precision=0)
            run = gr.Button("Generate", variant="primary")
        with gr.Column(scale=1):
            output = gr.Image(label="Result", type="filepath")

    run.click(
        generate,
        inputs=[prompt, negative_prompt, frames_slider, steps_radio, seed_num],
        outputs=output,
        concurrency_limit=1,   # one job at a time on 2 vCPUs
    )

    gr.Markdown("## Pre-rendered Blackhole output")
    gr.Markdown(
        "Generated on a Blackhole P300C at 512×512 — what the hardware path "
        "actually produces, without waiting for the CPU demo above."
    )
    gr.Gallery(value=_gallery_items(), columns=3, height="auto", label=None)

demo.queue(max_size=8)

if __name__ == "__main__":
    demo.launch()
