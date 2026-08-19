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
from pathlib import Path

import gradio as gr
from diffusers import DiffusionPipeline

MODEL_REPO = os.environ.get("TT_MODEL_REPO", "episod/tt-animatediff")

# Hard caps for a free-tier 2-vCPU CPU box. Raising any of these makes a
# generation outlast the request timeout rather than merely being slow.
MAX_FRAMES = 4
MAX_STEPS = 4
RESOLUTION = 384

GALLERY_DIR = Path(__file__).parent / "gallery"

BANNER = """
### CPU reference demo — not representative of Blackhole performance

This Space runs on a free-tier **CPU** with a distilled **4-step** checkpoint,
capped to 4 frames at 384×384. Tenstorrent hardware is not reachable from
Hugging Face infrastructure.

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


def generate(prompt, negative_prompt, num_frames, num_steps, seed):
    if not prompt or not prompt.strip():
        raise gr.Error("Enter a prompt first.")

    frames = _pipeline()(
        prompt=prompt.strip(),
        negative_prompt=negative_prompt.strip(),
        num_frames=min(int(num_frames), MAX_FRAMES),
        num_steps=min(int(num_steps), MAX_STEPS),
        guidance_scale=1.0,          # Lightning is distilled for CFG 1.0
        seed=int(seed),
        height=RESOLUTION,
        width=RESOLUTION,
        mode="cpu",
        use_lightning=True,
        lightning_steps=4,
    ).frames

    out = "/tmp/tt-animatediff-space.gif"
    frames[0].save(
        out, save_all=True, append_images=frames[1:], duration=125, loop=0
    )
    return out


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
                steps_slider = gr.Slider(
                    1, MAX_STEPS, value=MAX_STEPS, step=1, label="Steps (max 4)"
                )
            seed_num = gr.Number(value=42, label="Seed", precision=0)
            run = gr.Button("Generate", variant="primary")
        with gr.Column(scale=1):
            output = gr.Image(label="Result", type="filepath")

    run.click(
        generate,
        inputs=[prompt, negative_prompt, frames_slider, steps_slider, seed_num],
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
