# AnimateDiff on Tenstorrent Hardware

Two-phase implementation: **Phase 1** generates real, temporally coherent video
on CPU using the correct AnimateDiff architecture. **Phase 2** accelerates spatial
denoising on Blackhole hardware using the TTNN UNet.

| *"World of Tomorrow"* | *"Phosphor Horizon"* |
|---|---|
| ![world of tomorrow](docs/assets/demo_world_of_tomorrow.gif) | ![phosphor horizon](docs/assets/demo_phosphor_horizon.gif) |

Both GIFs generated on **Blackhole (P300C)** — 8 frames × 25 denoising steps, ~15 s/frame.

---

## Background: Why the Previous Implementation Was Wrong

The original implementation applied `mm_sd_v15_v2.ckpt` (AnimateDiff motion weights)
to SD 3.5's DiT transformer (2432-dim features). These weights were trained for SD 1.5's
UNet (320-dim features). They are architecturally incompatible.

This rewrite uses the correct base: **SD 1.4 UNet** + **MotionAdapter**, where temporal
attention is injected inside each UNet transformer block at the 320-dim level.

---

## Phase 1 — Correct AnimateDiff (CPU, no hardware needed)

Uses `diffusers.AnimateDiffPipeline` with `MotionAdapter`. The MotionAdapter injects
`TemporalTransformer` attention modules at each `BasicTransformerBlock` in the SD 1.4 UNet.
Motion weights operate at 320-dim features during every denoising step — frames are
temporally coherent by design, not post-hoc.

### Setup

```bash
pip install -r requirements.txt
hf download CompVis/stable-diffusion-v1-4
hf download guoyww/animatediff-motion-adapter-v1-5-2
```

### Run

```bash
python examples/generate_baseline.py
python examples/generate_baseline.py --prompt "purple phosphor glow across distant mountains at 2am, retro CRT haze, cinematic" --frames 8
```

Expected output: `output/baseline.gif` — 16 frames of temporally coherent animation.

---

## Phase 2 — Blackhole-Accelerated Frame Generation

Uses the SD 1.4 TTNN UNet from `~/tt-metal/models/demos/vision/generative/stable_diffusion/wormhole/` —
the same kernels run on Blackhole via `TT_METAL_ARCH_NAME=blackhole`. Frames are denoised
sequentially; Phase 2.5 adds cross-frame temporal attention at each step.

**Documented tradeoff:** This is TT-hardware-accelerated spatial denoising for video
frames, not full AnimateDiff temporal attention. Full integration would require injecting
`TemporalTransformer` blocks into the TTNN UNet transformer blocks.

### Requirements

- Blackhole hardware (P100 or P300c)
- `~/tt-metal` present, environment activated: `source ~/tt-metal/python_env/bin/activate`
- `hf download CompVis/stable-diffusion-v1-4` (also used by Phase 1; CLIP loads from this model's subfolders)

### Run

```bash
source ~/tt-metal/python_env/bin/activate
# Phase 2.5 (canonical, temporal attention — default)
python examples/generate.py
python examples/generate.py --prompt "1939 World's Fair imagined from the year 2099, art deco spires at golden dusk, cinematic 4K" --frames 8
```

Expected: `output/blackhole.gif` — 8 frames generated on Blackhole hardware.

---

## Code Structure

```
animatediff_ttnn/
  pipeline.py           Phase 1: thin wrapper around diffusers AnimateDiffPipeline
  ttnn_pipeline.py      Phase 2/2.5: TTNN UNet frame generation on Blackhole
  temporal_attention.py Phase 2.5: cross-frame self-attention at each denoising step
  temporal_module.py    Reference — temporal attention math (kept for study)
  __init__.py           Exports Phase 1 public API

examples/
  generate.py              Unified entry point (--mode cpu|blackhole|sim; default blackhole)
  generate_baseline.py     Phase 1 CPU (diffusers AnimateDiffPipeline, any machine)
  generate_blackhole.py    → shim to generate.py --mode blackhole
  generate_blackhole_v2.py → shim to generate.py --mode blackhole
  generate_sim.py          → shim to generate.py --mode sim

tests/
  test_pipeline.py       Phase 1 unit tests
  test_ttnn_pipeline.py  Phase 2 unit tests (hardware-mocked)

docs/
  IMPLEMENTATION_STATUS.md  Current Phase 1/2 status
  INTEGRATION_GUIDE.md      How downstream projects should consume this repo
  SIMULATOR.md              Running on ttsim without Blackhole hardware
```

---

## No hardware? Use the simulator

[ttsim](https://github.com/tenstorrent/ttsim) is a bit-exact Blackhole simulator
that runs on any Linux/x86_64 machine. See [docs/SIMULATOR.md](docs/SIMULATOR.md)
for setup and usage. Quick start:

```bash
python examples/generate.py --mode sim --frames 2 --steps 4
# or with explicit sim path:
python examples/generate.py --mode sim --sim ~/sim/libttsim_bh.so --frames 2 --steps 4
```

## Interactive UI

Launch the Gradio web interface for point-and-click generation:

```bash
pip install -e ".[ui]"
source ~/tt-metal/python_env/bin/activate  # for blackhole/sim modes
python app.py
# Open http://localhost:7860
```

Also available on [HuggingFace Spaces](https://huggingface.co/spaces/tenstorrent/tt-animatediff)
(runs on ttsim — no hardware required, slower).

See [docs/UI.md](docs/UI.md) for full usage.

## Integration

For guidance on how to consume this project from other repos — as a toolkit
lesson, an application plugin, or a Python library — see
[docs/INTEGRATION_GUIDE.md](docs/INTEGRATION_GUIDE.md).

## Testing

```bash
pip install pytest
python -m pytest tests/ -v
```

All tests mock hardware dependencies and run on any machine.

---

## Prompt Guide

This pipeline runs **SD 1.4 + TTNN UNet** (Phase 2/2.5) or **SD 1.4 + MotionAdapter** (Phase 1 CPU). Both differ from modern SD 2.x/SDXL — knowing the model's pedigree helps you write prompts that land.

### Model pedigree

| Property | Value |
|---|---|
| Base model | CompVis/stable-diffusion-v1-4 (2022) |
| Resolution | 512 × 512 (native), upscale externally if needed |
| CLIP text encoder | ViT-L/14, 77-token max, no long-prompt support |
| Training data | LAION-400M subset (pre-NSFW filter era) |
| Temporal coherence (Phase 2.5) | Cross-frame self-attention blend (`--temporal-alpha`, default 0.35) |
| Temporal coherence (Phase 1 CPU) | Full AnimateDiff MotionAdapter — genuine cross-frame attention |

### What SD 1.4 does well

- **Natural scenes:** forests, mountains, oceans, deserts, sky, fire, water
- **Painterly / artistic styles:** oil painting, watercolor, impressionism, concept art
- **Cinematic lighting:** golden hour, neon, moonlight, candlelight, dramatic shadows
- **Architecture:** temples, ruins, castles, sci-fi structures (avoid photo-realism)
- **Cosmic / abstract:** nebulae, galaxies, aurora, energy fields, geometric patterns
- **Retro aesthetics:** CRT glow, vintage film grain, vaporwave, cyberpunk

### What to avoid

- **Photorealistic people / faces** — SD 1.4 struggles with anatomy; faces drift frame-to-frame
- **Text in the image** — SD 1.4 cannot render legible text
- **Specific named real places** — results are impressionistic, not faithful
- **Very long prompts** — CLIP truncates at 77 tokens; keep your prompt under ~60 words

### Prompt patterns that work

```
# Style before subject → model weights the style heavily
"watercolor painting of ancient ruins at sunset, soft brushstrokes, muted palette"

# Cinematic lighting descriptors unlock quality
"cinematic 4K, dramatic side lighting, volumetric fog, depth of field"

# Cosmic + architecture = reliable sweet spot
"Mayan pyramid under a swirling nebula, starfield, bioluminescent jungle"

# Motion-friendly subjects for animation
"crackling campfire", "ocean waves", "swirling clouds", "aurora borealis",
"shifting cosmos", "flowing lava", "drifting smoke"
```

### `--temporal-alpha` tuning (Phase 2.5)

`--temporal-alpha` blends cross-frame attention into each denoising step.

| Value | Effect |
|---|---|
| `0.0` | No cross-frame mixing — each frame denoised independently (shared noise only) |
| `0.2–0.3` | Subtle coherence, slight variation frame-to-frame |
| `0.35` | **Default** — good balance for most motion subjects |
| `0.5–0.7` | Strong coherence; background stabilizes but detail may flatten |
| `1.0` | Maximum blending — frames are very similar, low motion |

For subjects with fast motion (fire, water) use `0.2–0.35`. For architectural scenes with slow drift (cosmos, aurora) use `0.4–0.6`.

### `--steps` vs quality

PNDM scheduler; minimum effective is ~4 steps (sim/fast preview). 25 steps is the sweet spot for silicon. Beyond 30 gives diminishing returns and increases per-frame time linearly.

---

## AnimateDiff Architecture Reference

```
SD 1.4 UNet without MotionAdapter:
  Noise → [Down blocks] → [Mid block] → [Up blocks] → Denoised latent
           each block has BasicTransformerBlock(spatial attention)

SD 1.4 UNet WITH MotionAdapter (Phase 1):
  Noise → [Down blocks] → [Mid block] → [Up blocks] → Denoised latent
           each BasicTransformerBlock now has:
             spatial attention (unchanged)
             + TemporalTransformer(cross-frame attention, 320-dim)
                                              ↑
                              This is where mm_sd_v15_v2.ckpt weights live
```

For full AnimateDiff on Blackhole, the TTNN UNet transformer blocks would need
TemporalTransformer layers inserted — a deeper integration than Phase 2 attempts.

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on:

- Reporting bugs via GitHub Issues
- Suggesting enhancements
- Submitting pull requests
- Development guidelines and code style

Pull requests are typically reviewed on a weekly basis.

---

## License

**Overall license for this project, except where specified:**
- [Apache License 2.0](LICENSE)
- See [LICENSE_understanding.txt](LICENSE_understanding.txt) for clarification on how Apache 2.0 applies to this project

**Third-party dependencies** are listed in the [NOTICE](NOTICE) file with their respective licenses.
