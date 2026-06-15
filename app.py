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
_motion_kernels: dict | None = None  # loaded once per session

DEFAULT_MOTION_MODEL = "guoyww/animatediff-motion-adapter-v1-5-2"

# ── World's Fair preset prompts ────────────────────────────────────────────
_WORLDS_FAIR_PRESETS = [
    ["Paris 1889 — Eiffel Tower Opening",
     "The Eiffel Tower at the 1889 Paris Exposition Universelle, iron lattice glowing under gas "
     "lamps at dusk, crowds of visitors in Victorian dress, the Seine reflecting amber light, "
     "Belle Époque grandeur, painterly impressionist",
     "modern, cars, neon, digital, blurry, low quality"],
    ["Chicago 1893 — White City",
     "The White City at the 1893 Chicago World's Columbian Exposition, neoclassical white marble "
     "palaces reflected in the Grand Basin, electric lights illuminating the Court of Honor at "
     "night for the first time, crowds in Gilded Age attire, moonlit clouds, majestic and ethereal",
     "modern, neon, digital, blurry, low quality"],
    ["New York 1939 — World of Tomorrow",
     "The 1939 New York World's Fair at night, the Trylon and Perisphere glowing white against a "
     "violet sky, art deco streamlined pavilions, fountains lit in vivid color, visitors in 1930s "
     "dress gazing at the future, retro-futurist wonder, cinematic",
     "modern, digital, blurry, low quality"],
    ["Brussels 1958 — Atomium",
     "The Atomium at the 1958 Brussels World Expo, nine steel spheres magnifying an iron crystal "
     "atom 165 billion times, golden evening light, Cold War era optimism, swooping modernist "
     "pavilions in the background, atomic age, silver and chrome",
     "blurry, low quality, digital artifacts"],
    ["New York 1964 — Unisphere",
     "The Unisphere at the 1964 New York World's Fair, massive stainless steel globe rising from "
     "Flushing Meadows, three orbital rings glinting in summer sunlight, IBM and Ford pavilions "
     "in background, mid-century modern optimism, Space Age, cinematic wide shot, warm afternoon light",
     "blurry, low quality, modern, digital"],
    ["Osaka 1970 — Tower of the Sun",
     "The Tower of the Sun by Taro Okamoto at Expo '70 Osaka, massive expressionist sculpture "
     "with golden face and white face, festival plaza teeming with visitors, futuristic space-frame "
     "roof structure, Japan's economic miracle, vivid colors, 1970s psychedelic energy, cinematic",
     "blurry, low quality, modern, digital"],
    ["2064 — Lunar World's Fair",
     "The 2064 World's Fair on the Moon, glass dome pavilions in the Sea of Tranquility, Earth "
     "rising on the horizon, bioluminescent architecture glowing against the lunar regolith, "
     "delegations from forty nations and three orbital habitats, low gravity fountains arcing "
     "impossibly high, retro-futurist meets deep future, awe-inspiring",
     "blurry, low quality, Earth gravity, trees, clouds"],
    ["2064 — Pacific Deepwater Expo",
     "The 2064 Pacific Deepwater World Expo at 500 meters depth, coral-lattice arcologies lit by "
     "bioluminescent algae, submersibles docking at crystal pavilions, whale song translated into "
     "light, kelp forests framing the grand promenade, ethereal blue-green dreamscape, wonder and "
     "reverence for the ocean",
     "blurry, low quality, surface, sky, land"],
    ["2064 — Orbital Ring World's Fair",
     "The 2064 World's Fair aboard the Orbital Ring station, a gleaming torus circling Earth at "
     "36,000 km, nations of the world building spinning pavilions along the inner hull, Earth a "
     "blue marble through vast observation windows, zero-gravity dancers in the atrium, humanity's "
     "greatest achievement on display, cinematic 4K",
     "blurry, low quality, gravity, surface, ugly"],
]


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


def _ensure_motion_kernels(num_frames: int):
    """Load MotionAdapter weights once per session."""
    global _motion_kernels
    if _motion_kernels is not None:
        return _motion_kernels
    from animatediff_ttnn.motion_weights import load_motion_modules
    _motion_kernels = load_motion_modules(DEFAULT_MOTION_MODEL)
    return _motion_kernels



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
    motion_adapter: bool = False,
    chain_from: str = "",
    chain_save: str = "",
    chain_alpha: float = 0.35,
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

    if mode == "cpu":
        # CPU mode has no step-level callback hook — just run and yield final
        from animatediff_ttnn.pipeline import generate as cpu_generate, export_gif
        pipe = _ensure_cpu_pipeline(lightning=lightning, lightning_steps=lightning_steps)
        guidance = 1.0 if lightning else 7.5
        # CPU Lightning uses distilled checkpoints that only support 2/4/8 steps;
        # ignore the general steps slider and use the lightning_steps value instead.
        cpu_steps = lightning_steps if lightning else steps
        frames_list = cpu_generate(
            pipe, prompt,
            negative_prompt=negative_prompt,
            num_frames=frames,
            guidance_scale=guidance,
            num_inference_steps=cpu_steps,
            seed=seed,
        )
        export_gif(frames_list, final_path)
        yield final_path
        return

    # Blackhole / sim — stream per-step previews then yield final
    from animatediff_ttnn.generation_helpers import encode_prompt
    from animatediff_ttnn.temporal_attention import generate_frames_temporal, _latent_preview
    from animatediff_ttnn.pipeline import export_gif

    device, (ttnn_model, ttnn_vae, config, torch_time_proj) = _ensure_bh_device(mode, sim_path)
    text_embeddings = encode_prompt(prompt, negative_prompt)

    # Resolve MotionAdapter kernels if requested (Phase 3)
    temporal_kernels = None
    if motion_adapter and mode in ("blackhole", "sim"):
        temporal_kernels = _ensure_motion_kernels(num_frames=frames)

    # Queue carries (preview_gif_path | None, error | None)
    # None preview + None error = done, final result is ready
    step_q: queue.Queue = queue.Queue()
    result_holder = []
    error_holder = []

    # Preview cadence: emit every step for ≤10 steps, every 2 steps otherwise.
    # Avoids flooding Gradio with updates on long runs while still feeling live.
    preview_every = 1 if steps <= 10 else 2

    height, width = 512, 512

    preview_path = str(out_dir / "preview.gif")  # single file, overwritten each step

    def on_step(step_idx, num_steps, frame_latents):
        if step_idx % preview_every != 0 and step_idx != num_steps - 1:
            return
        preview_frames = _latent_preview(frame_latents, height, width)
        export_gif(preview_frames, preview_path)
        step_q.put((preview_path, None))

    def worker():
        try:
            if temporal_kernels is not None:
                from animatediff_ttnn.temporal_attention import generate_frames_motion
                fl = generate_frames_motion(
                    device=device,
                    ttnn_model=ttnn_model,
                    ttnn_vae=ttnn_vae,
                    config=config,
                    torch_time_proj=torch_time_proj,
                    text_embeddings=text_embeddings,
                    temporal_kernels=temporal_kernels,
                    num_frames=frames,
                    num_steps=steps,
                    guidance_scale=1.0 if lightning else 7.5,
                    seed=seed,
                    use_lightning=lightning,
                    chain_from=chain_from_path,
                    chain_save=chain_save_path,
                    chain_alpha=chain_alpha,
                    temporal_alpha=temporal_alpha,
                    on_step=on_step,
                    height=height,
                    width=width,
                )
            else:
                fl = generate_frames_temporal(
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

Enable **MotionAdapter (Phase 3)** for full temporal coherence using
`guoyww/animatediff-motion-adapter-v1-5-2` weights (blackhole/sim only).
Preview updates stream in real time as each denoising step completes.
See the [World's Fair showcase](https://tenstorrent.github.io/tt-animatediff/worlds-fair.html) for examples.
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

            motion_adapter = gr.Checkbox(
                label="🎞 MotionAdapter Phase 3 (blackhole/sim only — loads guoyww/animatediff-motion-adapter-v1-5-2)",
                value=False,
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
                    0.0, 1.0, value=0.35, step=0.05,
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
            sim_path, lightning, lightning_steps, motion_adapter,
            chain_from_box, chain_save_box, chain_alpha_slider,
        ],
        outputs=output_gif,
    )

    # ── World's Fair presets ───────────────────────────────────────────────
    gr.Markdown("---\n### World's Fair Presets\nClick a row to load prompt and negative prompt.")

    def _load_preset(evt: gr.SelectData):
        row = _WORLDS_FAIR_PRESETS[evt.index[0]]
        return row[1], row[2]  # prompt, negative_prompt

    wf_table = gr.Dataframe(
        value=[[r[0]] for r in _WORLDS_FAIR_PRESETS],
        headers=["Preset"],
        datatype=["str"],
        interactive=False,
        row_count=(len(_WORLDS_FAIR_PRESETS), "fixed"),
        col_count=(1, "fixed"),
    )
    wf_table.select(fn=_load_preset, outputs=[prompt, negative_prompt])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
