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

# Blackhole + MotionAdapter Phase 3 (full temporal attention, 16 frames)
python examples/generate.py --motion-adapter --frames 16

# Blackhole + MotionAdapter + Lightning (fast path, 8 frames)
python examples/generate.py --motion-adapter --lightning --frames 8

# Simulator — no hardware, bit-exact Blackhole
python examples/generate.py --mode sim --frames 2 --steps 4
```

---

## Lightning Mode

On **Blackhole/sim**: `--lightning` switches to `EulerDiscreteScheduler` (trailing, linear)
with the base SD 1.4 TTNN UNet — no distilled weights loaded, CFG=7.5 retained, any step count.

On **CPU**: loads `ByteDance/AnimateDiff-Lightning` (genuine 4-step distilled adapter, CFG=1.0
baked in). Use `--lightning-steps 2|4|8` to match the distillation checkpoint.

LCM distillation (4-run attempt, flat LR on sharp loss landscape) was closed — broken weights
archived as `weights/*.broken`. Use Lightning mode for fast inference.

---

## Phase 2 — Blackhole-Accelerated Frame Generation

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

Also available at [huggingface.co/spaces/tenstorrent/tt-animatediff](https://huggingface.co/spaces/tenstorrent/tt-animatediff) — runs on ttsim, no hardware required.

To self-host:

1. Create a new Space (SDK: Gradio)
2. Copy `spaces/` contents into the Space repo root
3. Copy `app.py` and `animatediff_ttnn/` into the Space repo root
4. The Space picks up `SPACE_MODE=sim` automatically

Note: ttsim requires a Linux x86\_64 runner. Upload `libttsim_bh.so` as a Space
file or add a setup script to download it at startup.

### UI Parameters

| Parameter | Range | Default | Notes |
|---|---|---|---|
| Mode | cpu / blackhole / sim | blackhole | sim on HF Spaces |
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

## Modes Reference

| Mode | Hardware | Speed | Temporal attention |
|---|---|---|---|
| `cpu` | None | ~2 min/frame | Full AnimateDiff MotionAdapter ✓ |
| `cpu --lightning` | None | ~20 s/frame | Full AnimateDiff MotionAdapter ✓ |
| `blackhole` | Blackhole P100/P300C | ~15 s/frame (25 steps, PNDM) | Cross-frame blend (temporal-alpha) |
| `blackhole --lightning` | Blackhole P100/P300C | ~15 s/frame (25 steps, Euler) | Cross-frame blend (temporal-alpha) |
| `blackhole --motion-adapter` | Blackhole P100/P300C | ~15 s/frame + CPU round-trip | Full MotionAdapter Phase 3 ✓ |
| `blackhole --motion-adapter --lightning` | Blackhole P100/P300C | ~15 s/frame + CPU round-trip | Full MotionAdapter Phase 3 ✓ |
| `sim` | None (ttsim) | ~10–100× slower than silicon | Cross-frame blend (temporal-alpha) |

Both standard and Lightning on Blackhole use 25 steps and CFG=7.5 with the base SD 1.4 TTNN UNet.
Lightning uses `EulerDiscreteScheduler` (trailing, linear) rather than `PNDMScheduler` — a different solver trajectory, not fewer steps.
CPU Lightning (`--mode cpu --lightning`) uses the real 4-step distilled adapter (CFG=1.0 baked in) and is genuinely ~6× faster than CPU standard.
Phase 3 (`--motion-adapter`) adds a CPU round-trip after each TTNN cross-attention block — ~50–200 ms extra per step depending on frame count and injection point.

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
passed through the full `AnimateDiffTransformer3D.forward()` (norm, proj_in/out, positional
embedding, GEGLU feedforward) per injection point, then returned to Blackhole.
Enable with `--motion-adapter` (any number of chips, any step count).

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

app.py                      Gradio UI (local + HF Spaces, MotionAdapter + World's Fair presets)
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
    P([Prompt + seed]) --> ENC["CLIP encode\n(CPU, always)"]
    ENC --> MODE{Mode?}

    MODE -->|cpu| CPU_SCHED["PNDMScheduler\nor EulerDiscreteScheduler\n(Lightning)"]
    CPU_SCHED --> CPU_UNET["diffusers UNet2DConditionModel\n+ MotionAdapter\n(CPU — full temporal attention)"]
    CPU_UNET --> CPU_VAE["VAE decode (CPU — diffusers)"]
    CPU_VAE --> GIF([GIF])

    MODE -->|blackhole / sim| BH_SCHED["PNDMScheduler (standard)\nor EulerDiscreteScheduler (Lightning)\n— one per frame"]
    BH_SCHED --> LOOP["For each step t:"]
    LOOP --> BH_UNET["TTNN UNet2D — SD 1.4\nBlackhole P300C\n~15 s/frame"]
    BH_UNET -->|--motion-adapter| MA["forward_unet_staged()\n7 × AnimateDiffTransformer3D\nCPU round-trip per block"]
    BH_UNET -->|standard| CFA
    MA --> CFA["cross_frame_attention()\nnoise_preds blended\n(CPU, tiny)"]
    CFA --> STEP["scheduler.step()\n— one per frame"]
    STEP -->|Lightning: +latent blend| LAT["cross_frame_attention()\nprev_sample blended\nα × 0.4, cosine decay"]
    LAT --> LOOP
    STEP -->|PNDM| LOOP
    LOOP -->|done| BH_VAE["TTNN VAE decode\nBlackhole · L1 freed before decode"]
    BH_VAE --> GIF

    style BH_UNET fill:#0f2a35,stroke:#4fd1c5,color:#e8f0f2
    style MA fill:#0f2a35,stroke:#ec96b8,color:#e8f0f2
    style CFA fill:#0f2a35,stroke:#4fd1c5,color:#e8f0f2
    style LAT fill:#0f2a35,stroke:#81e6d9,color:#e8f0f2
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
