#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Gradio UI for tt-animatediff.

Local (Blackhole hardware):
    pip install -e ".[ui]"
    python app.py

HuggingFace Spaces (ttsim, no hardware):
    Deployed from spaces/ — sets SPACE_MODE=sim automatically.

Streaming: the generate callback is a Python generator. After each denoising
step, _latent_preview() converts the current noisy latents to a colourised
preview GIF (tanh-mapped RGB, bilinear upsampled) and yields it to Gradio.
The final yield is the VAE-decoded result. No hardware overhead per preview —
preview decode is pure CPU tensor ops on the 64×64 latent grid.
"""

import os
import queue
import sys
import tempfile
import threading
from pathlib import Path

import gradio as gr

# Default mode: env var SPACE_MODE overrides (set to "sim" on HF Spaces)
_VALID_MODES = {"blackhole", "sim", "cpu"}
_raw_mode = os.environ.get("SPACE_MODE", "blackhole")
DEFAULT_MODE = _raw_mode if _raw_mode in _VALID_MODES else "blackhole"

# Add repo root to path so animatediff_ttnn is importable
sys.path.insert(0, str(Path(__file__).parent))

NEG_DEFAULT = "blurry, low quality, distorted, text, people, faces, modern buildings"

_cpu_pipes: dict = {}  # keyed by (lightning: bool, lightning_steps: int)
_bh_device = None      # cached Blackhole/sim MeshDevice
_bh_models = None      # cached (ttnn_model, ttnn_vae, config, torch_time_proj)


def _ensure_cpu_pipeline(lightning: bool = False, lightning_steps: int = 4):
    key = (lightning, lightning_steps)
    if key not in _cpu_pipes:
        from animatediff_ttnn.pipeline import create_animatediff_pipeline, create_lightning_pipeline
        if lightning:
            _cpu_pipes[key] = create_lightning_pipeline(step=lightning_steps)
        else:
            _cpu_pipes[key] = create_animatediff_pipeline()
    return _cpu_pipes[key]


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

    from animatediff_ttnn.generation_helpers import load_sd14_ttnn
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
    lightning: bool = False,
    lightning_steps: int = 4,
    chain_from: str = "",
    chain_save: str = "",
    chain_alpha: float = 0.6,
):
    """Generator: yields preview GIF paths as denoising progresses, then final GIF.

    Streaming works by running the TTNN generation in a background thread while
    the Gradio handler thread reads previews from a queue and yields them. Each
    denoising step fires on_step(), which encodes a fast CPU-side latent preview
    (no VAE, no hardware) into a GIF and enqueues it. The sentinel None signals
    that generation is complete and the final result is ready.
    """
    # Cast numeric inputs — Gradio Number/Slider can deliver floats
    frames = int(frames)
    steps = int(steps)
    seed = int(seed)
    lightning_steps = int(lightning_steps)

    if not prompt.strip():
        raise gr.Error("Prompt cannot be empty.")
    if not 0.0 <= temporal_alpha <= 1.0:
        raise gr.Error("Temporal alpha must be between 0 and 1.")

    chain_from_path = chain_from.strip() or None
    chain_save_path = chain_save.strip() or None

    out_dir = Path(tempfile.mkdtemp())
    final_path = str(out_dir / "output.gif")

    # OUTPUT size. Distinct from the preview size, which `make_step_callback`
    # picks on its own (256x256 — the latents are a 64x64 grid, so upsampling a
    # preview to full output size buys nothing). Conflating the two is how these
    # two names briefly went missing.
    height, width = 512, 512

    from animatediff_ttnn.pipeline import export_gif

    # Every mode streams. `run_fn(on_step) -> frames` is the only thing that
    # differs between them; the queue/thread/yield machinery below is shared.
    #
    # CPU used to be the exception ("CPU mode has no step-level callback hook —
    # just run and yield final"), which was true until `pipeline.generate()`
    # grew an `on_step` passthrough. Leaving it unstreamed would have left the
    # UI's own blurb — "Preview updates stream in real time as each denoising
    # step completes" — false for one of the three modes it offers.
    if mode == "cpu":
        from animatediff_ttnn.pipeline import generate as cpu_generate

        pipe = _ensure_cpu_pipeline(lightning=lightning, lightning_steps=lightning_steps)
        guidance = 1.0 if lightning else 7.5
        # CPU Lightning uses distilled checkpoints that only support 2/4/8 steps;
        # ignore the general steps slider and use the lightning_steps value instead.
        steps = lightning_steps if lightning else steps

        def run_fn(on_step):
            return cpu_generate(
                pipe, prompt,
                negative_prompt=negative_prompt,
                num_frames=frames,
                guidance_scale=guidance,
                num_inference_steps=steps,
                seed=seed,
                on_step=on_step,
            )
    else:
        from animatediff_ttnn.generation_helpers import encode_prompt
        from animatediff_ttnn.temporal_attention import generate_frames_temporal

        device, (ttnn_model, ttnn_vae, config, torch_time_proj) = _ensure_bh_device(mode, sim_path)
        text_embeddings = encode_prompt(prompt, negative_prompt)

        def run_fn(on_step):
            return generate_frames_temporal(
                device=device,
                ttnn_model=ttnn_model,
                ttnn_vae=ttnn_vae,
                config=config,
                torch_time_proj=torch_time_proj,
                text_embeddings=text_embeddings,
                num_frames=frames,
                num_steps=steps,
                guidance_scale=7.5,
                seed=seed,
                temporal_alpha=temporal_alpha,
                use_lightning=lightning,
                chain_from=chain_from_path,
                chain_save=chain_save_path,
                chain_alpha=chain_alpha,
                on_step=on_step,
                height=height,
                width=width,
            )

    # Queue carries (preview_gif_path | None, error | None)
    # None preview + None error = done, final result is ready
    step_q: queue.Queue = queue.Queue()
    result_holder = []
    error_holder = []

    preview_path = str(out_dir / "preview.gif")  # single file, overwritten each step

    # Cadence, rendering and the (atomic) write all come from the shared
    # `animatediff_ttnn.preview` module, which the CLI runner uses too — this
    # block used to hand-roll all three. Sharing it means the Space and the CLI
    # can't drift on how often previews appear or how big they are, and the
    # Space picks up the module's temp-file+rename write, so Gradio can no
    # longer be handed a half-written GIF.
    #
    # `emit` is where the two differ: the CLI prints a line to stdout for a
    # subprocess consumer to parse, while here the path goes straight onto the
    # queue the Gradio handler thread is draining.
    from animatediff_ttnn.preview import make_step_callback, parse_preview_line

    def _to_queue(line: str) -> None:
        parsed = parse_preview_line(line)
        if parsed is None:
            return
        _step, _total, path = parsed
        # main's queue contract is (preview_path | None, error | None) — a
        # 2-tuple. The step/total are parsed but not forwarded here; the Space
        # renders the image, not a step counter.
        step_q.put((path, None))

    on_step = make_step_callback(preview_path, num_steps=steps, emit=_to_queue)

    def worker():
        try:
            fl = run_fn(on_step)
            result_holder.append(fl)
        except Exception as exc:
            error_holder.append(exc)
        finally:
            step_q.put((None, None))  # sentinel

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    # Yield previews as they arrive; stop when sentinel received
    while True:
        item = step_q.get()
        preview_path, err = item
        if err is not None:
            raise gr.Error(str(err))
        if preview_path is None:
            break
        yield preview_path

    t.join()

    if error_holder:
        raise gr.Error(str(error_holder[0]))

    export_gif(result_holder[0], final_path)
    yield final_path


# ── UI layout ─────────────────────────────────────────────────────────────

_DESCRIPTION = """
**AnimateDiff on Tenstorrent Blackhole** — generate animated GIFs using SD 1.4 TTNN UNet.

- **blackhole** — runs on real Blackhole hardware (~15 s/frame on P300C)
- **sim** — runs on ttsim virtual device (bit-exact, slower)
- **cpu** — runs on CPU via diffusers AnimateDiffPipeline (~2 min/frame)

Preview updates stream in real time as each denoising step completes.
See the [prompt guide](https://tenstorrent.github.io/tt-animatediff/#prompt-guide) for tips.
"""

with gr.Blocks(title="tt-animatediff") as demo:
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
                frames_slider = gr.Slider(2, 24, value=8, step=1, label="Frames")
                steps_slider = gr.Slider(4, 50, value=25, step=1, label="Steps")
            with gr.Row():
                seed_num = gr.Number(value=42, label="Seed", precision=0)
                temporal_alpha_slider = gr.Slider(
                    0.0, 1.0, value=0.35, step=0.05,
                    label="Temporal alpha (blackhole/sim only)",
                )
            sim_path = gr.Textbox(
                label="Sim binary path (sim mode only)",
                placeholder="~/sim/libttsim_bh.so",
                visible=False,
            )

            with gr.Row():
                lightning = gr.Checkbox(
                    label="⚡ Lightning (Euler scheduler — CPU: use Lightning Steps 2/4/8 only; BH/sim: any step count)",
                    value=False,
                )
                lightning_steps = gr.Radio(
                    choices=[2, 4, 8], value=4,
                    label="Lightning steps (CPU only)",
                    info="CPU Lightning only: must match the distilled checkpoint (4 recommended)",
                )

            with gr.Accordion("Chain continuity (blackhole/sim only)", open=False):
                gr.Markdown(
                    "Thread visual DNA from one generation into the next. "
                    "Save this run's final latents with **Chain save**, then load them in the next run with **Chain from**."
                )
                with gr.Row():
                    chain_from_box = gr.Textbox(
                        label="Chain from (path to .pt)", placeholder="chain.pt"
                    )
                    chain_save_box = gr.Textbox(
                        label="Chain save (path to .pt)", placeholder="chain.pt"
                    )
                chain_alpha_slider = gr.Slider(
                    0.0, 1.0, value=0.6, step=0.05,
                    label="Chain alpha (0 = ignore, 1 = replace seed noise)",
                )

            def _update_visibility(m):
                return gr.update(visible=(m == "sim"))

            mode.change(fn=_update_visibility, inputs=mode, outputs=[sim_path])
            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            output_gif = gr.Image(label="Output (streaming preview → final GIF)", type="filepath")

    run_btn.click(
        fn=generate,
        inputs=[
            mode, prompt, negative_prompt,
            frames_slider, steps_slider, seed_num, temporal_alpha_slider,
            sim_path, lightning, lightning_steps,
            chain_from_box, chain_save_box, chain_alpha_slider,
        ],
        outputs=output_gif,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
