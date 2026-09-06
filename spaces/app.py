# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Hugging Face Space for tt-animatediff — capped CPU reference demo.

Blackhole hardware is unreachable from HF infrastructure, so this Space runs the
**CPU** path with ByteDance's distilled Lightning checkpoints (2 or 4 steps,
selectable) and hard caps that keep a generation finishable on a free-tier
2-vCPU box. It exists to prove the pipeline loads and runs anywhere; the
pre-rendered gallery below is what the hardware path actually produces.

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

# The banner is HTML rather than Markdown so the CPU warning reads as a callout rather
# than as more prose. Colours are Tenstorrent's accent over a translucent tint and the
# text colour is inherited, so it renders correctly in both Gradio themes without
# knowing which one is active.
#
# Every number below is measured, and the two sides are comparable: same resolution, same
# step schedule, normalised per frame. CPU figures come from this Space's own run logs;
# Blackhole figures from docs/measurements/serving-benchmark.json in the repo (8 frames,
# 512x512, one P300C chip, warm, median of 3). They are NOT the same checkpoint -- the CPU
# path needs a distilled Lightning adapter to make 2-4 steps viable, Blackhole runs the
# standard one -- so this compares time for a given frames x steps job, not identical
# output. Said plainly in the footnote rather than left for someone to discover.
BANNER = """
<div style="border-left:4px solid #1B8EB1;background:rgba(27,142,177,0.10);
            border-radius:6px;padding:14px 18px;margin:4px 0 10px">
  <div style="font-weight:600;font-size:1.05em;margin-bottom:6px">
    This demo runs on a CPU &mdash; it is not Tenstorrent hardware
  </div>
  <p style="margin:6px 0">
    Hugging Face infrastructure cannot reach a Blackhole card, so this Space runs the
    <strong>CPU reference path</strong> on a free-tier 2&nbsp;vCPU box, with a distilled
    Lightning checkpoint (2 or 4 steps), capped to 4 frames at 512&times;512.
    <strong>Nothing you time here reflects hardware performance.</strong>
  </p>

  <table style="border-collapse:collapse;margin:12px 0 6px;font-size:0.94em">
    <thead>
      <tr style="text-align:left;border-bottom:1px solid rgba(127,127,127,0.35)">
        <th style="padding:4px 18px 4px 0">Per frame, 512&times;512</th>
        <th style="padding:4px 18px 4px 0">Here &mdash; free CPU</th>
        <th style="padding:4px 18px 4px 0">Blackhole P300C, one chip</th>
      </tr>
    </thead>
    <tbody>
      <tr><td style="padding:4px 18px 4px 0">2-step schedule</td>
          <td style="padding:4px 18px 4px 0">~6.5 s</td>
          <td style="padding:4px 18px 4px 0">not measured</td></tr>
      <tr><td style="padding:4px 18px 4px 0">4-step schedule</td>
          <td style="padding:4px 18px 4px 0">~25 s</td>
          <td style="padding:4px 18px 4px 0"><strong>0.64 s</strong> &mdash; about 40&times; faster</td></tr>
      <tr><td style="padding:4px 18px 4px 0">8-step schedule</td>
          <td style="padding:4px 18px 4px 0">not offered here</td>
          <td style="padding:4px 18px 4px 0"><strong>0.92 s</strong></td></tr>
    </tbody>
  </table>

  <p style="margin:6px 0">
    <strong>So expect roughly:</strong> 2 frames at 2 steps &asymp; <strong>15 s</strong>
    of compute, 4 frames at 4 steps &asymp; <strong>1 min 40 s</strong>. Add
    <strong>60&ndash;90 s</strong> the first time you use a step count &mdash; it downloads
    a 908&nbsp;MB checkpoint and loads the pipeline. Switching between 2 and 4 steps pays
    that once each.
  </p>
  <p style="margin:6px 0">
    A long wait is the queue working, not a hang. <strong>Start with 2 frames at 2 steps</strong>
    &mdash; the 4&times;4 job runs for several minutes and can outlast the connection.
  </p>
  <p style="margin:6px 0 0;font-size:0.86em;opacity:0.8">
    CPU figures measured in this Space's own logs (2 frames at 2 steps; 4 frames at 4 steps).
    Blackhole figures from the repo's committed benchmark &mdash; 8 frames, 512&times;512,
    warm, median of 3, <a href="https://github.com/tenstorrent/tt-animatediff/blob/main/docs/measurements/serving-benchmark.json">serving-benchmark.json</a>.
    Same job shape, different checkpoint: the CPU path uses a distilled Lightning adapter to
    make 2&ndash;4 steps viable, Blackhole runs the standard one, so this compares time for a
    given frames&nbsp;&times;&nbsp;steps job rather than identical output.
  </p>
</div>
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
    """Pre-rendered Blackhole GIFs staged into gallery/ at publish time.

    Empty is a real possibility, not just a theoretical one: the bundle is
    assembled by scripts/build_space_artifact.py, so a GALLERY_SOURCES
    regression would ship a Space with nothing here. _gallery_is_empty() below
    turns that into something a visitor (and we) can see.
    """
    if not GALLERY_DIR.is_dir():
        return []
    return sorted(str(p) for p in GALLERY_DIR.glob("*.gif"))


def _gallery_is_empty() -> bool:
    return not _gallery_items()


with gr.Blocks(title="tt-animatediff") as demo:
    gr.Markdown("# tt-animatediff — AnimateDiff on Tenstorrent Blackhole")
    gr.HTML(BANNER)

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
    if _gallery_is_empty():
        # Say so rather than rendering an empty strip: this only happens if the
        # staged bundle lost its gallery, and a blank area looks like a styling
        # bug instead of the packaging regression it actually is.
        gr.Markdown(
            "_The pre-rendered gallery is missing from this deployment — see "
            "`GALLERY_SOURCES` in `scripts/build_space_artifact.py`. The "
            "[repo gallery](https://tenstorrent.github.io/tt-animatediff/gallery.html) "
            "has the same output._"
        )
    else:
        gr.Markdown(
            "Generated on a Blackhole P300C at 512×512 — what the hardware path "
            "actually produces, without waiting for the CPU demo above."
        )
        gr.Gallery(value=_gallery_items(), columns=3, height="auto", label=None)

demo.queue(max_size=8)

if __name__ == "__main__":
    demo.launch()
