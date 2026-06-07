# AnimateDiff Implementation Status

**Last Updated:** 2026-06-07
**Version:** v0.6.0

---

## Phase 1 — Correct AnimateDiff (CPU Baseline)

**Status: ✅ Complete**

Uses `diffusers.AnimateDiffPipeline` + `MotionAdapter("guoyww/animatediff-motion-adapter-v1-5-2")`
on `CompVis/stable-diffusion-v1-4`. MotionAdapter injects temporal attention at each UNet
transformer block (320-dim features, matching mm_sd_v15_v2.ckpt). No TT hardware needed.

**Run:** `python examples/generate_baseline.py`

**Produces:** Real, temporally coherent GIF animation.

---

## Phase 2 — Blackhole-Accelerated Frame Generation

**Status: ✅ Code complete, hardware validation pending**

TTNN UNet (`UNet2D` from tt-metal SD 1.4 demo) denoises frames sequentially on Blackhole.
Temporal coherence via shared base noise. `TT_METAL_ARCH_NAME=blackhole` required.

**Run:** `python examples/generate_blackhole.py` (requires Blackhole hardware + ~/tt-metal)

**Known tradeoff:** Temporal attention is NOT applied during TTNN denoising. For full
AnimateDiff on TT hardware, TemporalTransformer blocks would need to be added to the
TTNN UNet — that is out of scope here and would require modifying tt-metal source.

---

## Phase 2.5 — Lightning Mode on Blackhole

**Status: ✅ Complete (v0.5.0)**

`--lightning` flag uses `EulerDiscreteScheduler` (trailing, linear) instead of PNDM.
On Blackhole/sim the base SD 1.4 TTNN UNet is used — CFG=7.5 retained.
Cross-frame two-point blend: noise_pred blend before `scheduler.step()` + latent blend
after, both with cosine-decay alpha.

**CFG clarification:** The "CFG=1.0 required" constraint applies only to the real
AnimateDiff-Lightning distilled MotionAdapter weights (ByteDance CPU path). Our TTNN
path uses the undistilled SD 1.4 UNet, which requires CFG=7.5 for meaningful guidance.

**Run:** `python examples/generate.py --lightning`

---

## Gradio UI

**Status: ✅ Complete (v0.3.0)**

`app.py` — browser interface for all three modes (Blackhole, ttsim, CPU) with Lightning
controls. Device and model are cached across generations.

**Run:** `python app.py` → `http://localhost:7860`

---

## TTNN VAE on Blackhole

**Status: ✅ Complete (v0.6.0)**

VAE decode now runs fully on Blackhole hardware. The original OOM on `conv_out`
was caused by live UNet L1 tensors remaining allocated during VAE decode — the
same issue fixed in the official `sd_helper_funcs.py::run()`. Fix:
- `load_sd14_ttnn()` returns a TTNN `Vae` instance alongside the UNet
- `generate_frames()` and `generate_frames_temporal()` explicitly deallocate all
  UNet L1 tensors (latent_model_input, UNet output, guided noise_pred) before decode
- VAE input permuted NCHW→NHWC; output cast to float32 before numpy

---

## Stateful Latent Chaining (`--chain`)

**Status: ✅ Complete (v0.6.0)**

```bash
# Run A: save latents
python examples/generate.py --prompt "coral reef" --chain-save chain.pt

# Run B: thread visual DNA from A into B
python examples/generate.py --prompt "deep space nebula" --chain-from chain.pt
```

`--chain-from` blends the mean of previous run's denoised latents (normalised to
unit std) into the seed noise at `--chain-alpha` (default 0.6). Creates narrative
visual continuity across independent prompts without explicit conditioning.

---

## Comparison Gallery

**Status: ✅ Complete (v0.5.0, updated v0.6.0)**

10-prompt × 2-scheduler gallery at `docs/gallery.html`. All GIFs generated on
real Blackhole P300C silicon, 16 frames × 25 steps × 512×512.

v0.6.0 update: home-page Lightning preview samples replaced with gallery GIFs
generated post CFG-fix. Grid cells p3-crystal-data, p4-silicon, p4-neural-const,
p4-chip-city refreshed with evolved pink/purple Tensix-emergent prompts via Lightning.

---

## Root Cause of Original Implementation Failure

The original `examples/generate_with_sd35.py` attempted to use:
- **SD 3.5 DiT** (Diffusion Transformer, 2432-dim features)
- with **mm_sd_v15_v2.ckpt** motion weights trained for SD 1.5 UNet (320-dim features)

These are architecturally incompatible. The motion weights expect transformer blocks
with 320-dim hidden states; the DiT operates at 2432-dim. This mismatch meant no
temporal attention was actually being applied. Additionally, `_generate_frame_latents()`
and `_decode_latent()` were both placeholder implementations returning synthetic data.

The fix: use SD 1.4/1.5 (same UNet architecture as training) via diffusers MotionAdapter.
