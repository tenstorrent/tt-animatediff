#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Gradio UI for tt-animatediff.

Local (Blackhole hardware):
    pip install -e ".[ui]"
    python app.py

HuggingFace Spaces (ttsim, no hardware):
    Deployed from spaces/ — sets SPACE_MODE=sim automatically.

The UI exposes the same parameters as generate.py.  Backend functions are
imported lazily inside the generate callback so the module loads without
tt-metal present.
"""

import os
import sys
import tempfile
from pathlib import Path

import gradio as gr

# Default mode: env var SPACE_MODE overrides (set to "sim" on HF Spaces)
DEFAULT_MODE = os.environ.get("SPACE_MODE", "blackhole")

# Add repo root to path so animatediff_ttnn is importable
sys.path.insert(0, str(Path(__file__).parent))

NEG_DEFAULT = "blurry, low quality, distorted, text, people, faces, modern buildings"

_cpu_pipe = None       # cached CPU pipeline (slow to load)
_bh_device = None      # cached Blackhole/sim MeshDevice
_bh_models = None      # cached (ttnn_model, torch_vae, config, torch_time_proj)


def _ensure_cpu_pipeline():
    global _cpu_pipe
    if _cpu_pipe is None:
        from animatediff_ttnn.pipeline import create_animatediff_pipeline
        _cpu_pipe = create_animatediff_pipeline()
    return _cpu_pipe


def _ensure_bh_device(mode: str, sim_path: str):
    """Open the Blackhole or ttsim device (once per session)."""
    global _bh_device, _bh_models
    if _bh_device is not None:
        return _bh_device, _bh_models

    if mode == "sim":
        sim_so = Path(sim_path).expanduser() if sim_path else Path.home() / "sim/libttsim_bh.so"
        if not sim_so.exists():
            raise FileNotFoundError(
                f"ttsim binary not found at {sim_so}. "
                "Download from https://github.com/tenstorrent/ttsim/releases "
                "or set the Sim binary path field."
            )
        os.environ["TT_METAL_SIMULATOR"] = str(sim_so)
        os.environ.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")
        os.environ.setdefault("TT_METAL_DISABLE_SFPLOADMACRO", "1")
        os.environ.setdefault("TT_METAL_ARCH_NAME", "blackhole")

    TT_METAL_PATH = Path.home() / "tt-metal"
    sys.path.insert(0, str(TT_METAL_PATH))

    from animatediff_ttnn.ttnn_pipeline import setup_blackhole, _ensure_tt_metal_path
    import ttnn
    from models.demos.vision.generative.stable_diffusion.wormhole.common import SD_L1_SMALL_SIZE

    if mode == "sim":
        _ensure_tt_metal_path()
        _bh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(1, 1),
            physical_device_ids=[0],
            l1_small_size=SD_L1_SMALL_SIZE,
        )
    else:
        _bh_device = setup_blackhole(device_ids=[0])

    # Load models onto device
    from examples.generate import load_sd14_ttnn
    _bh_models = load_sd14_ttnn(_bh_device)
    return _bh_device, _bh_models


def generate(
    mode: str,
    prompt: str,
    negative_prompt: str,
    frames: int,
    steps: int,
    seed: int,
    temporal_alpha: float,
    sim_path: str,
):
    """Run generation and return a GIF path for Gradio to display."""
    if not prompt.strip():
        raise gr.Error("Prompt cannot be empty.")
    if not 0.0 <= temporal_alpha <= 1.0:
        raise gr.Error("Temporal alpha must be between 0 and 1.")

    out_dir = Path(tempfile.mkdtemp())
    out_path = str(out_dir / "output.gif")

    if mode == "cpu":
        from animatediff_ttnn.pipeline import generate as cpu_generate, export_gif
        pipe = _ensure_cpu_pipeline()
        frames_list = cpu_generate(
            pipe, prompt,
            negative_prompt=negative_prompt,
            num_frames=frames,
            num_inference_steps=steps,
            seed=seed,
        )
        export_gif(frames_list, out_path)

    else:  # blackhole or sim
        import torch
        from examples.generate import encode_prompt
        from animatediff_ttnn.temporal_attention import generate_frames_temporal
        from animatediff_ttnn.pipeline import export_gif

        device, (ttnn_model, torch_vae, config, torch_time_proj) = _ensure_bh_device(mode, sim_path)
        text_embeddings = encode_prompt(prompt, negative_prompt)
        frames_list = generate_frames_temporal(
            device=device,
            ttnn_model=ttnn_model,
            torch_vae=torch_vae,
            config=config,
            torch_time_proj=torch_time_proj,
            text_embeddings=text_embeddings,
            num_frames=frames,
            num_steps=steps,
            seed=seed,
            temporal_alpha=temporal_alpha,
        )
        export_gif(frames_list, out_path)

    return out_path


# ── UI layout ─────────────────────────────────────────────────────────────

_DESCRIPTION = """
**AnimateDiff on Tenstorrent Blackhole** — generate animated GIFs using SD 1.4 TTNN UNet.

- **blackhole** — runs on real Blackhole hardware (~15 s/frame on P300C)
- **sim** — runs on ttsim virtual device (bit-exact, slower)
- **cpu** — runs on CPU via diffusers AnimateDiffPipeline (~2 min/frame)

See the [prompt guide](https://tenstorrent.github.io/tt-animatediff/#prompt-guide) for tips.
"""

with gr.Blocks(title="tt-animatediff", theme=gr.themes.Base()) as demo:
    gr.Markdown("# tt-animatediff")
    gr.Markdown(_DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            mode = gr.Radio(
                choices=["blackhole", "sim", "cpu"],
                value=DEFAULT_MODE,
                label="Mode",
            )
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="swirling nebula in deep space, purple and teal gas clouds, cinematic 4K",
                lines=3,
            )
            negative_prompt = gr.Textbox(
                label="Negative prompt",
                value=NEG_DEFAULT,
                lines=2,
            )
            with gr.Row():
                frames = gr.Slider(2, 24, value=8, step=1, label="Frames")
                steps = gr.Slider(4, 50, value=25, step=1, label="Steps")
            with gr.Row():
                seed = gr.Number(value=42, label="Seed", precision=0)
                temporal_alpha = gr.Slider(0.0, 1.0, value=0.35, step=0.05,
                                           label="Temporal alpha (blackhole/sim only)")
            sim_path = gr.Textbox(
                label="Sim binary path (sim mode only)",
                placeholder="~/sim/libttsim_bh.so",
                visible=False,
            )
            mode.change(
                fn=lambda m: gr.update(visible=(m == "sim")),
                inputs=mode,
                outputs=sim_path,
            )
            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            output_gif = gr.Image(label="Output GIF", type="filepath")
            status = gr.Markdown("")

    run_btn.click(
        fn=generate,
        inputs=[mode, prompt, negative_prompt, frames, steps, seed, temporal_alpha, sim_path],
        outputs=output_gif,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
