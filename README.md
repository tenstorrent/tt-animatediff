# AnimateDiff on Tenstorrent Hardware

Three-phase implementation on Tenstorrent Blackhole P300C.
**Phase 1** generates temporally coherent video on CPU using the full AnimateDiff
MotionAdapter. **Phase 2** accelerates spatial denoising on Blackhole with the TTNN
UNet. **Phase 3** injects AnimateDiff MotionAdapter temporal attention directly into
the Blackhole denoising loop — no distillation required, weights loaded straight from
`guoyww/animatediff-motion-adapter-v1-5-2`.

---

## Gallery

### Blackhole (P300C) — 8 frames × 25 steps, ~15 s/frame

| *"World of Tomorrow"* | *"Phosphor Horizon"* | *"Mayan Temple"* |
|---|---|---|
| ![world of tomorrow](docs/assets/demo_world_of_tomorrow.gif) | ![phosphor horizon](docs/assets/demo_phosphor_horizon.gif) | ![mayan temple](docs/assets/mayan_temple.gif) |

| *"Neon Dystopia"* | *"Ocean"* |
|---|---|
| ![neon dystopia](docs/assets/neon_dystopia.gif) | ![ocean](docs/assets/ocean.gif) |

### AnimateDiff-Lightning on Blackhole — 8 frames × 25 steps, Euler · CFG=7.5

Lightning mode runs the TTNN UNet with `EulerDiscreteScheduler` (trailing, linear).
On Blackhole the base SD 1.4 TTNN UNet is used — CFG=7.5 retained for full guidance.
Same step count (25) as standard PNDM, different solver.

See the [full 10-prompt gallery](https://tenstorrent.github.io/tt-animatediff/gallery.html) for standard vs Lightning side-by-side.

| *"Aurora"* | *"Mandala"* | *"Mycelium"* |
|---|---|---|
| ![lightning aurora](docs/assets/study/lightning-aurora.gif) | ![lightning mandala](docs/assets/study/lightning-mandala.gif) | ![lightning mycelium](docs/assets/study/lightning-mycelium.gif) |

---

## Quick Start

```bash
pip install -r requirements.txt
hf download CompVis/stable-diffusion-v1-4
hf download guoyww/animatediff-motion-adapter-v1-5-2

# CPU — any machine, no hardware required
python examples/generate.py --mode cpu --prompt "ocean waves at sunset, cinematic"

# Blackhole hardware (default, ~15 s/frame)
source ~/tt-metal/python_env/bin/activate
python examples/generate.py --prompt "aurora borealis over a frozen lake, cinematic 4K"

# Blackhole + Lightning (Euler scheduler, 25 steps, CFG=7.5)
python examples/generate.py --lightning

# CPU + Lightning (~20 s/frame, 4-step distilled adapter, no hardware required)
python examples/generate.py --mode cpu --lightning --lightning-steps 4

# Blackhole + MotionAdapter Phase 3 full (7 injection pts, ~52 s/frame)
python examples/generate.py --motion-adapter --frames 8

# Blackhole + MotionAdapter Phase 3 fast (skip up1+up2, ~7.7 s/frame — faster than Phase 2.5)
python examples/generate.py --motion-adapter --motion-adapter-skip up1 up2 --frames 8

# Simulator — no hardware, bit-exact Blackhole
python examples/generate.py --mode sim --frames 2 --steps 4
```

---

## Hugging Face

The pipeline is published as a weights-free diffusers custom pipeline:

```python
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "episod/tt-animatediff",
    custom_pipeline="episod/tt-animatediff",
    trust_remote_code=True,
)
frames = pipe("a swirling nebula, teal and gold").frames
print(pipe.resolved_mode)  # "blackhole" or "cpu"
```

- Model repo: [`episod/tt-animatediff`](https://huggingface.co/episod/tt-animatediff)
  — no weights; SD 1.4 and the MotionAdapter are resolved from upstream at generation time.
- Demo Space: [`episod/tt-animatediff-demo`](https://huggingface.co/spaces/episod/tt-animatediff-demo)
  — capped CPU-Lightning reference (4 frames, 2 or 4 steps, 512×512). Not representative
  of Blackhole performance; a 4-frame run takes several minutes on free-tier CPU.

Rebuild and republish with `scripts/build_hf_artifact.py` (model artifact),
`scripts/build_space_artifact.py` (Space bundle) and
`scripts/publish_to_hub.py`; the Hub copy is generated from this checkout, never edited
on the Hub.

---

## Lightning Mode

On **Blackhole/sim**: `--lightning` switches to `EulerDiscreteScheduler` (trailing, linear)
with the base SD 1.4 TTNN UNet — no distilled weights loaded, CFG=7.5 retained, any step count.

On **CPU**: loads `ByteDance/AnimateDiff-Lightning` (genuine 4-step distilled adapter, CFG=1.0
baked in). Use `--lightning-steps 2|4|8` to match the distillation checkpoint.

LCM distillation (4-run attempt, flat LR on sharp loss landscape) was closed — broken weights
archived as `weights/*.broken`. Use Lightning mode for fast inference.

---

## Gradio UI

Launch a web interface for point-and-click generation.

### Local — Blackhole hardware

```bash
pip install -e ".[ui]"
source ~/tt-metal/python_env/bin/activate
python app.py
# Open http://localhost:7860
```

Models are cached across generations — only the first run pays the load cost
(~7 s) and kernel compilation (~2–3 min, cached after first run).

### Local — CPU only (no hardware)

```bash
pip install -e ".[ui]"
python app.py
# Switch mode to "cpu" in the UI dropdown
```

### Local — ttsim simulator (no hardware)

```bash
mkdir -p ~/sim
wget -O ~/sim/libttsim_bh.so \
    https://github.com/tenstorrent/ttsim/releases/download/v1.7.0/libttsim_bh.so

pip install -e ".[ui]"
source ~/tt-metal/python_env/bin/activate
python app.py
# Switch mode to "sim" in the UI, set sim binary path if needed
```

### HuggingFace Spaces

The `spaces/` directory contains deployment files for hosting on HuggingFace Spaces (ttsim mode, no hardware required). To deploy:

1. Create a new Space (SDK: Gradio)
2. Copy `spaces/` contents into the Space repo root
3. Copy `app.py` and `animatediff_ttnn/` into the Space repo root
4. The Space picks up `SPACE_MODE=sim` automatically

Note: ttsim requires a Linux x86\_64 runner. Upload `libttsim_bh.so` as a Space
file or add a setup script to download it at startup.

### UI Parameters

| Parameter | Range | Default | Notes |
|---|---|---|---|
| Mode | cpu / blackhole / sim | blackhole | use sim for HF Spaces deployment |
| Prompt | text | — | See [Prompt Guide](#prompt-guide) |
| Negative prompt | text | standard exclusions | |
| Frames | 2–24 | 8 | 2–4 recommended for sim |
| Steps | 4–50 | 25 | 4 for Lightning / sim |
| Seed | integer | 42 | |
| Temporal alpha | 0.0–1.0 | 0.35 | Blackhole/sim only |
| Lightning | checkbox | off | Euler solver on Blackhole/sim (same steps, different trajectory); ~6× faster on CPU |
| MotionAdapter Phase 3 | checkbox | off | Full AnimateDiff MotionAdapter on Blackhole/sim; adds CPU round-trip per step |
| Sim binary path | file path | ~/sim/libttsim_bh.so | sim mode only |

---

## Python API

The `animatediff_ttnn` package is importable as a Python library. The high-level
`generate_animation()` entry point manages device lifetime, mode selection, and the
CPU pipeline cache internally — no setup required.

```python
from animatediff_ttnn import generate_animation, export_mp4, export_gif

# Auto mode: picks Blackhole if tt-metal is importable, CPU otherwise
frames = generate_animation(
    prompt="swirling nebula, teal and gold, cinematic",
    num_frames=8,
    num_steps=25,
    seed=42,
)
export_mp4(frames, "output.mp4", fps=8)
export_gif(frames, "output.gif")
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `prompt` | (required) | Text description of the animation |
| `negative_prompt` | `""` | Features to suppress |
| `num_frames` | `8` | Frame count |
| `num_steps` | `25` | Denoising steps (use `4` for sim preview) |
| `guidance_scale` | `7.5` | CFG scale — use `1.0` with CPU Lightning |
| `seed` | `42` | Random seed |
| `temporal_alpha` | `0.35` | Cross-frame attention blend (Blackhole/sim only, 0–1) |
| `mode` | `"auto"` | `"auto"` · `"blackhole"` · `"sim"` · `"cpu"` |
| `use_lightning` | `False` | Euler scheduler instead of PNDM |
| `lightning_steps` | `4` | CPU Lightning checkpoint step count (2, 4, or 8) |
| `chain_from` | `None` | Path to `.pt` latent file from a previous run for visual continuity |
| `chain_save` | `None` | Save this run's final latents for use as `chain_from` next time |
| `chain_alpha` | `0.6` | Blend weight for `chain_from` latents |
| `on_step` | `None` | Callback `(step_idx, num_steps, frame_latents)` per denoising step — Blackhole/sim only |

Returns `list[PIL.Image]`, one per frame. See the
[full Python API reference](https://tenstorrent.github.io/tt-animatediff/usage.html#api)
for all parameters and usage examples.

---

## Modes Reference

| Mode | Hardware | Speed (8 fr, 512²) | Temporal attention |
|---|---|---|---|
| `cpu` | None | ~2 min/frame | Full AnimateDiff MotionAdapter ✓ |
| `cpu --lightning` | None | ~20 s/frame | Full AnimateDiff MotionAdapter ✓ |
| `blackhole` | Blackhole P300C | **~12.5 s/frame** (25 steps, PNDM) | Cross-frame blend (temporal-alpha) |
| `blackhole --lightning` | Blackhole P300C | **~12.0 s/frame** (25 steps, Euler, CFG=7.5) | Cross-frame blend (temporal-alpha) |
| `blackhole --motion-adapter` | Blackhole P300C | **~52 s/frame** (7 injection pts, batched D→H) | Full MotionAdapter Phase 3 ✓ |
| `blackhole --motion-adapter --motion-adapter-skip up1 up2` | Blackhole P300C | **~7.7 s/frame** (5 injection pts) | Full MotionAdapter Phase 3 ✓ |
| `sim` | None (ttsim) | ~10–100× slower than silicon | Cross-frame blend (temporal-alpha) |

All timings measured on a QB2 board (4 × P300C), 8 frames at 512×512, warm model (TTNN JIT already compiled).
See [docs/benchmarks.html](https://tenstorrent.github.io/tt-animatediff/benchmarks.html) for the full timing breakdown and comparison GIFs.

Lightning on Blackhole uses `EulerDiscreteScheduler` (trailing, linear) with the base SD 1.4 TTNN UNet — different solver, not fewer steps, CFG=7.5 retained.
CPU Lightning uses the real 4-step distilled adapter (CFG=1.0 baked in) and is ~6× faster than CPU standard.
Phase 3 `--motion-adapter` runs `AnimateDiffTransformer3D.forward()` at 7 UNet injection points per denoising step. A batched D→H transfer (all N frames pulled in one `ttnn.concat → ttnn.to_torch` call) delivers a 1.94× speedup over the naive per-frame implementation.
`--motion-adapter-skip up1 up2` bypasses the two costliest decoder injection points (up1: 32×32 C=1280, up2: 64×64 C=640), dropping from ~52 s/frame to ~7.7 s/frame — faster than Phase 2.5 — with a minor reduction in decoder-side temporal coherence.

---

## No Hardware? Use the Simulator

[ttsim](https://github.com/tenstorrent/ttsim) is a bit-exact Blackhole simulator
that runs on any Linux/x86\_64 machine. See [docs/SIMULATOR.md](docs/SIMULATOR.md)
for full setup. Quick start:

```bash
python examples/generate.py --mode sim --frames 2 --steps 4
python examples/generate.py --mode sim --sim ~/sim/libttsim_bh.so --frames 2 --steps 4
```

---

## Implementation Status

| Phase | Status | Notes |
|---|---|---|
| Phase 1 — CPU baseline | ✅ Complete | `diffusers.AnimateDiffPipeline` + MotionAdapter |
| Phase 1 — Lightning (CPU) | ✅ Complete | `ByteDance/AnimateDiff-Lightning`, ~20 s/frame |
| Phase 2 — Blackhole denoising | ✅ Complete | TTNN UNet, ~15 s/frame on P300C |
| Phase 2.5 — Cross-frame temporal | ✅ Complete | `--temporal-alpha` blend during denoising |
| Phase 2.5 — Lightning (Blackhole) | ✅ Complete | `TtEulerScheduler`, Euler solver · 25 steps · CFG=7.5 |
| Phase 3 — MotionAdapter on Blackhole | ✅ Complete | `--motion-adapter` · 7 injection points · no distillation |
| TT-Lang temporal attention sim | ✅ Complete | Functional simulator, dual P300c HW smoke test |
| Gradio UI | ✅ Complete | Local + HF Spaces · MotionAdapter + Phase 3 + World's Fair presets |

For full details see [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md).

### Phase 3 — MotionAdapter on Blackhole

The `guoyww/animatediff-motion-adapter-v1-5-2` motion weights are loaded and injected
at 7 points (down0/1/2, mid, up0/1/2) in the SD 1.4 TTNN UNet without modifying tt-metal
source. After each TTNN cross-attention block, hidden states are round-tripped to CPU,
passed through the full `AnimateDiffTransformer3D.forward()` (GroupNorm, proj_in/out,
LayerNorm×3, positional embedding, GEGLU feedforward, output projection) per injection
point, then returned to Blackhole. Enable with `--motion-adapter`.

**Speed optimizations (measured on QB2, 8 frames, 25 steps):**

| Configuration | s/frame | Total (8fr) | Notes |
|---|---|---|---|
| Baseline (per-frame D→H) | ~101 | ~806s | Original implementation |
| Batched D→H (`ttnn.concat → to_torch`) | ~52 | **~416s** | 1.94× speedup — current default |
| Skip up1+up2 (`--motion-adapter-skip up1 up2`) | **~7.7** | **~62s** | 6.75× vs full; faster than Phase 2.5 |

The two decoder injection points (up1 32×32 C=1280, up2 64×64 C=640) account for ~80% of the
CPU transformer cost. Skipping them retains encoder and mid-block temporal attention with
only a minor reduction in decoder-side coherence. See the
[benchmark page](https://tenstorrent.github.io/tt-animatediff/benchmarks.html) and
[Maya glyph comparison](https://tenstorrent.github.io/tt-animatediff/mayan-glyphs.html)
for measured numbers and side-by-side visual comparisons.

---

## Code Structure

```
animatediff_ttnn/
  pipeline.py               Phase 1: thin wrapper around diffusers AnimateDiffPipeline
  ttnn_pipeline.py          Phase 2/2.5: TTNN UNet frame generation on Blackhole
  temporal_attention.py     Phase 2.5/3: cross-frame blend + generate_frames_motion()
  tt_euler_scheduler.py     TtEulerScheduler — Euler wrapper for Lightning on Blackhole
  generation_helpers.py     Shared load_sd14_ttnn / encode_prompt
  motion_weights.py         Phase 3: load MotionAdapter weights → AnimateDiffTransformer3D modules
  ttnn_motion_pipeline.py   Phase 3: _apply_temporal() + forward_unet_staged()
  temporal_module.py        Reference — temporal attention math (kept for study)
  ttlang/                   TT-Lang sim kernel track (TemporalAttentionKernel + 3 DSL kernels)
  __init__.py               Exports Phase 1 public API

examples/
  generate.py               Unified entry point (--mode cpu|blackhole|sim; --motion-adapter)
  generate_baseline.py      Phase 1 CPU (diffusers AnimateDiffPipeline, any machine)
  generate_blackhole.py     → shim to generate.py --mode blackhole
  generate_blackhole_v2.py  → shim to generate.py --mode blackhole
  generate_sim.py           → shim to generate.py --mode sim

scripts/
  generate_worlds_fair.py   9 World's Fair prompts × 3 tiers, 4-chip parallel + Unisphere chain
  generate_comparison_grid.py  Multi-mode comparison grid (A–F)
  generate_gallery.py       Gallery GIF batch generation
  distill_lcm.py            LCM UNet distillation (experimental, closed)
  distill_motion_adapter.py LCM MotionAdapter distillation (experimental, closed)

app.py                      Gradio UI (local; spaces/ contains HF Spaces deployment files)
spaces/                     HuggingFace Spaces deployment files

tests/
  test_pipeline.py               Phase 1 unit tests
  test_ttnn_pipeline.py          Phase 2 unit tests (hardware-mocked)
  test_tt_euler_scheduler.py     TtEulerScheduler unit tests
  test_temporal_attention.py     Cross-frame attention unit tests
  test_motion_weights.py         Phase 3 weight loader tests
  test_ttnn_motion_pipeline.py   Phase 3 pipeline tests
  test_ttlang_temporal_attention.py  TT-Lang sim kernel tests (9 tests, PCC > 0.999)
  test_app.py                    Gradio UI smoke tests

docs/
  IMPLEMENTATION_STATUS.md    Current phase status
  INTEGRATION_GUIDE.md        Consuming this repo from downstream projects
  SIMULATOR.md                ttsim setup and usage
  UI.md                       Gradio UI full documentation
  assets/worlds-fair/         World's Fair GIFs (generated by generate_worlds_fair.py)
```

---

## Prompt Guide

This pipeline runs **SD 1.4 + MotionAdapter** (Phase 1/CPU) or
**SD 1.4 + TTNN UNet** (Phase 2/Blackhole). Both use SD 1.4 — knowing
its characteristics helps write prompts that land.

### Model pedigree

| Property | Value |
|---|---|
| Base model | CompVis/stable-diffusion-v1-4 (2022) |
| Resolution | 512 × 512 native |
| CLIP text encoder | ViT-L/14, 77-token max |
| Temporal coherence (cpu) | Full AnimateDiff MotionAdapter |
| Temporal coherence (blackhole/sim, default) | Cross-frame blend (`--temporal-alpha`, default 0.35) |
| Temporal coherence (blackhole/sim, `--motion-adapter`) | Full AnimateDiff MotionAdapter Phase 3 ✓ |

### What SD 1.4 does well

- **Natural scenes:** forests, mountains, oceans, sky, fire, water
- **Painterly / artistic styles:** oil painting, watercolor, impressionism, concept art
- **Cinematic lighting:** golden hour, neon, moonlight, candlelight, dramatic shadows
- **Architecture:** temples, ruins, castles, sci-fi structures
- **Cosmic / abstract:** nebulae, galaxies, aurora, energy fields, geometric patterns
- **Retro aesthetics:** CRT glow, vintage film grain, vaporwave, cyberpunk

### What to avoid

- **Photorealistic people / faces** — anatomy drifts frame-to-frame
- **Text in the image** — SD 1.4 cannot render legible text
- **Prompts over ~60 words** — CLIP truncates at 77 tokens

### Prompt patterns that work

```
# Style before subject
"watercolor painting of ancient ruins at sunset, soft brushstrokes, muted palette"

# Cinematic lighting descriptors
"cinematic 4K, dramatic side lighting, volumetric fog, depth of field"

# Motion-friendly subjects
"crackling campfire", "ocean waves", "swirling clouds", "aurora borealis",
"shifting cosmos", "flowing lava", "drifting smoke", "mycelium network pulsing"
```

### `--temporal-alpha` tuning (Blackhole/sim)

| Value | Effect |
|---|---|
| `0.0` | No cross-frame mixing — frames denoised independently |
| `0.2–0.3` | Subtle coherence, slight variation frame-to-frame |
| `0.35` | **Default** — good balance for most subjects |
| `0.5–0.7` | Strong coherence; background stabilizes, detail may flatten |
| `1.0` | Maximum blending — frames very similar, low motion |

Fast motion (fire, water): `0.2–0.35`. Slow drift (cosmos, aurora): `0.4–0.6`.

### `--steps` vs quality

| Mode | Minimum | Sweet spot | Notes |
|---|---|---|---|
| PNDM (standard, blackhole/sim) | 4 (preview/sim) | 25 on silicon | Diminishing returns beyond 30 |
| Euler (Lightning, blackhole/sim) | 4 (preview/sim) | **25** | Base TTNN UNet — same CFG=7.5, different solver |
| Euler (Lightning, cpu) | 2 | **4** | Real distilled adapter — CFG=1.0 baked in; more steps degrades |

CPU Lightning `--lightning-steps` must match the distillation checkpoint (2, 4, or 8).
Blackhole/sim Lightning ignores `--lightning-steps` and uses `--steps` (default 25).

---

## Execution Flow

```mermaid
flowchart TD
    P([Prompt + seed]) --> ENC["CLIP encode — CPU"]
    ENC --> MODE{Mode?}

    MODE -->|cpu| CPU_SCHED["PNDM or Euler Lightning\n+ full AnimateDiff MotionAdapter\n~2 min/frame"]
    CPU_SCHED --> GIF([GIF])

    MODE -->|blackhole / sim| SCHED["Scheduler — CPU\nPNDM standard · Euler Lightning"]
    SCHED --> LOOP["denoising loop"]
    LOOP --> BH_UNET["TTNN UNet2D — SD 1.4\nBlackhole P300C · ~0.5 s/call"]
    BH_UNET --> PHASE{"--motion-adapter?"}

    PHASE -->|no — Phase 2.5\n~12.5 s/frame| CFA["cross_frame_attention\nnoise blend α=0.35 — CPU"]
    PHASE -->|yes — Phase 3| SKIP{"--motion-adapter-skip?"}

    SKIP -->|no — full\n~52 s/frame| MA_FULL["7 × AnimateDiffTransformer3D\nbatched D→H transfer\nCPU · ~4 s each"]
    SKIP -->|up1 up2 — fast\n~7.7 s/frame| MA_SKIP["5 × AnimateDiffTransformer3D\ndown0/1/2, mid, up0 only\nCPU · encoder points"]

    MA_FULL --> CFA
    MA_SKIP --> CFA
    CFA --> STEP["scheduler.step — CPU"]
    STEP --> LOOP
    LOOP -->|done| BH_VAE["TTNN VAE decode — Blackhole"]
    BH_VAE --> GIF

    style BH_UNET fill:#0f2a35,stroke:#4fd1c5,color:#e8f0f2
    style MA_FULL fill:#0f2a35,stroke:#ec96b8,color:#e8f0f2
    style MA_SKIP fill:#0f2a35,stroke:#27ae60,color:#e8f0f2
    style CFA fill:#0f2a35,stroke:#4fd1c5,color:#e8f0f2
```

## Architecture Reference: Original vs Current

The original implementation used architecturally incompatible components:

```mermaid
flowchart TB
    subgraph WRONG["❌ Original — silent failure"]
        W1["SD 3.5 DiT · 2432-dim features"] -->|motion weights injected| W2["mm_sd_v15_v2.ckpt\ntrained for SD 1.5 UNet · 320-dim"]
        W2 --> W3["Dimension mismatch\nNo temporal attention actually applied"]
    end

    subgraph RIGHT["✅ Current"]
        R1["SD 1.4 UNet · 320-dim — matching architecture"]
        R1 -->|CPU| R2["MotionAdapter TemporalTransformer\nFull AnimateDiff ✓"]
        R1 -->|Blackhole Phase 2.5| R3["TTNN UNet2D\nCross-frame attention blend"]
        R1 -->|Blackhole Phase 3| R4["forward_unet_staged()\n7 × MotionAdapter injection\nno tt-metal modifications"]
    end

    WRONG -.->|"fix: use matching architecture"| RIGHT
```

---

## Testing

```bash
pip install pytest
python -m pytest tests/ -v
```

All tests mock hardware dependencies and run on any machine.

---

## Integration

For guidance on consuming this project from other repos — as a toolkit lesson,
application plugin, or Python library — see
[docs/INTEGRATION_GUIDE.md](docs/INTEGRATION_GUIDE.md).

---

## Changelog

### v0.9.0 — 2026-06-15
- **Phase 3 batched D→H transfer** — `_apply_temporal` now pulls all N frame tensors from
  device in a single `ttnn.concat → ttnn.to_torch` call instead of N separate transfers.
  Measured speedup: **1.94×** (806s → 416s, 8 frames × 25 steps on QB2). H→D stays per-frame
  (`ttnn.split` produces parent-buffer views incompatible with the downstream resnet reshard kernel).
  `torch.compile` removed — hits 8-recompile guard limit on attention processor object ID changes.
- **`--motion-adapter-skip up1 up2` fast path** — skipping the two costliest decoder injection
  points (up1 32×32 C=1280, up2 64×64 C=640) drops wall-clock from ~52 s/frame to **~7.7 s/frame**,
  a 6.75× speedup over full Phase 3 and faster than Phase 2.5 (12.5 s/frame). Measured on QB2.
  Lightning + MotionAdapter tested and confirmed no benefit (~50.6 s/frame, ≈ same as 25-step
  PNDM) — CPU bridge calls per step dominate, not step count.
- **Maya glyph Q3/Q4 tiers** — `generate_mayan_glyphs.py` adds Q3 (full MotionAdapter) and Q4
  (skip up1+up2) tiers with `--sample` flag (4 representative glyphs). Side-by-side comparison
  section added to `docs/mayan-glyphs.html` showing Q2 / Q4 / Q3 stacked per glyph.
- **Benchmarks page updated** — new bar, table rows, speedup cards, and observation cards for
  skip and Lightning+MA results. Mode diagram reordered fastest→slowest.

### v0.8.0 — 2026-06-14
- **Phase 3 bug fixes** — two root-cause fixes for energy explosion that made all Phase 3
  output pure noise:
  1. **Weight transpose** — `nn.Linear` stores `[out, in]`; `load_weights()` now applies
     `.T.contiguous()` so `x @ w` gets the correct `[in, out]` projection. Square `[C,C]`
     matrices hid the bug until energy was measured.
  2. **Full diffusers module** — `TemporalAttentionKernel` only implemented QKV+residual,
     missing GroupNorm, `proj_in/proj_out` (trained, not identity), LayerNorm×3, positional
     embedding, and GEGLU feedforward. Replaced with direct `AnimateDiffTransformer3D.forward()`.
     Energy ratios dropped from >2.0 to <1.25.
- **`--motion-adapter-skip KEY...`** — skip injection points by name (e.g., `--motion-adapter-skip up1 up2`).
  Skipping the two highest-resolution up-blocks gives ~6× speedup with negligible quality change.
- **World's Fair Q1/Q2 regenerated** — all 9 prompts × 2 tiers (plus Unisphere chain) re-run
  with corrected Phase 3 weights.

### v0.7.0 — 2026-06-13
- **Phase 3 — MotionAdapter on Blackhole** — `--motion-adapter` flag injects
  `guoyww/animatediff-motion-adapter-v1-5-2` at 7 UNet cross-attention points via CPU
  round-trip. No tt-metal source modifications required.
  `forward_unet_staged()` replicates the TTNN UNet `__call__` orchestration in-repo,
  calling the same block objects and inserting `_apply_temporal()` hooks between them.
- **TT-Lang temporal attention** — `animatediff_ttnn/ttlang/` implements QKV projection,
  SDPA, and output projection as TT-Lang DSL kernels verified in the functional simulator
  (9 tests, all PCC > 0.999). Hardware smoke test on dual P300c: PCC > 0.99 at all dims.
- **World's Fair showcase** — `docs/worlds-fair.html`: 9 prompts × 3 quality tiers
  (Q1: best/16fr/4-chip, Q2: Lightning/8fr/4-chip, Q3: 1-chip) + Unisphere 100-year
  chain. `scripts/generate_worlds_fair.py` orchestrates parallel generation.
- **Gradio MotionAdapter + presets** — `app.py` adds MotionAdapter checkbox, routes to
  `generate_frames_motion()`, and shows 9 World's Fair prompts as selectable presets.
- **`--device-id INT`** — pin generation to a specific Blackhole chip (0-based), enabling
  multi-process parallel dispatch across chips.
- **LCM distillation (closed)** — 4 distillation runs attempted; all failed due to flat
  LR on sharp loss landscape. Broken weights archived as `weights/*.broken`.

### v0.6.0 — 2026-06-07
- **TTNN VAE on Blackhole** — VAE decode now runs fully on Blackhole hardware (no CPU fallback). Root cause of previous OOM identified: live UNet L1 tensors were not deallocated before VAE decode. Fix mirrors the official `sd_helper_funcs.py::run()` deallocation pattern. `load_sd14_ttnn()` now returns a TTNN `Vae` instance alongside the UNet.
- **`--chain` stateful latent threading** — `--chain-save <path>` persists final denoised latents; `--chain-from <path>` blends them into the next run's seed noise at configurable `--chain-alpha` (default 0.6). Creates visual narrative continuity across independent prompts without explicit conditioning.
- **Grid refresh** — four grid cells (`p3-crystal-data`, `p4-silicon`, `p4-neural-const`, `p4-chip-city`) regenerated with evolved prompts (pinks, purples, recognizable Tensix-emergent phenomena) using Lightning mode. Grid nodes display a ⚡ pip and pink glow for Lightning cells.
- **Mermaid architecture diagrams** — execution flow and original-vs-current comparison diagrams added to README and website; fixed `direction TB` syntax error in nested subgraphs (invalid in Mermaid v11).
- **`generate.py` default-steps fix** — Blackhole Lightning defaulted to 4 steps (distillation constraint); now correctly defaults to 25. CPU Lightning still uses `--lightning-steps`.
- **bfloat16→float32 cast fix** — `ttnn.to_torch()` returns bfloat16; added `.float()` before numpy conversion to fix `unsupported ScalarType BFloat16` crash.
- Home-page Lightning samples replaced with gallery GIFs generated post CFG-fix.

### v0.5.0 — 2026-06-07
- **Lightning CFG fix** — Blackhole/sim Lightning path now uses CFG=7.5 (was incorrectly 1.0); CFG=1.0 only applies to the real distilled CPU adapter
- **10-prompt comparison gallery** — [gallery.html](https://tenstorrent.github.io/tt-animatediff/gallery.html): standard PNDM vs Euler Lightning, all prompts, 25 steps each
- **`app.py` bug fix** — Gradio UI Blackhole/sim path now passes `guidance_scale=7.5` for Lightning
- Gallery labels and README/index corrected throughout

### v0.4.0 — 2026-06-07
- **Lightning on Blackhole** — `--lightning` now works in all modes (blackhole, sim, cpu)
- `TtEulerScheduler` — wraps `EulerDiscreteScheduler` for `build_tlist()` compatibility
- Measured **~4.3 s/frame** on P300C (8 frames, 4 steps) vs ~15 s/frame standard
- Lightning GIF examples regenerated on real Blackhole hardware
- `TtEulerScheduler` unit tests (13 tests, all modes verified)

### v0.3.0 — 2026-06-06
- **Gradio UI** (`app.py`) — local launch and HuggingFace Spaces deployment
- **AnimateDiff-Lightning** support — `--lightning` flag for CPU generation
- HuggingFace Space files in `spaces/`
- UI smoke tests (`tests/test_app.py`)
- Gradio 6 compatibility fix (theme outside Blocks constructor)

### v0.2.0 — 2026-05-14
- **Phase 2.5** — cross-frame temporal attention blend (`--temporal-alpha`)
- Unified entry point `examples/generate.py` with `--mode cpu|blackhole|sim`
- ttsim simulator support (`--mode sim`)
- `generate_sim.py`, `generate_blackhole.py`, `generate_blackhole_v2.py` shims
- Public release compliance report, SECURITY.md, CODE_OF_CONDUCT.md
- Hardware compatibility docs (`HARDWARE_COMPAT.md`)

### v0.1.0 — 2026-05-07
- Initial public release
- Phase 1: CPU AnimateDiff baseline (`diffusers.AnimateDiffPipeline` + MotionAdapter)
- Phase 2: Blackhole TTNN UNet frame generation
- `temporal_module.py` reference implementation
- Repository scaffolding: CI, tests, integration guide, simulator docs

---

## Background: Why the Previous Implementation Was Wrong

The original implementation applied `mm_sd_v15_v2.ckpt` motion weights to SD 3.5's
DiT transformer (2432-dim features). These weights were trained for SD 1.5's UNet
(320-dim features). They are architecturally incompatible — no temporal attention was
actually being applied.

The fix: use **SD 1.4 UNet + MotionAdapter**, where temporal attention is injected at
the 320-dim level that the motion weights expect.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). PRs reviewed weekly.

---

## License

**Overall:** [Apache License 2.0](LICENSE) — see [LICENSE_understanding.txt](LICENSE_understanding.txt).

**Third-party dependencies:** [NOTICE](NOTICE).
