---
license: creativeml-openrail-m
base_model: CompVis/stable-diffusion-v1-4
tags:
  - animatediff
  - text-to-video
  - tenstorrent
  - blackhole
  - lcm
  - consistency-distillation
pipeline_tag: text-to-video
---

# tt-animatediff LCM Distilled Weights

LCM-distilled AnimateDiff weights for fast inference on Tenstorrent Blackhole hardware.
Produced by running [`scripts/distill_lcm.py`](https://github.com/tenstorrent/tt-animatediff/blob/main/scripts/distill_lcm.py)
and [`scripts/distill_motion_adapter.py`](https://github.com/tenstorrent/tt-animatediff/blob/main/scripts/distill_motion_adapter.py)
on a Blackhole P300C (CPU training, Blackhole inference validation).

## Files

| File | Size | Description |
|------|------|-------------|
| `unet_lcm_4step.pt` | ~3.3 GB | SD 1.4 UNet distilled to 4 denoising steps |
| `unet_lcm_8step.pt` | ~3.3 GB | SD 1.4 UNet distilled to 8 denoising steps |
| `motion_adapter_lcm_4step.pt` | ~150 MB | AnimateDiff MotionAdapter distilled to 4 steps |
| `motion_adapter_lcm_8step.pt` | ~150 MB | AnimateDiff MotionAdapter distilled to 8 steps |

## Usage

```bash
# Download via the helper script in the repo
bash weights/download_weights.sh

# Or manually with huggingface_hub
python3 -c "
from huggingface_hub import hf_hub_download
for f in ['unet_lcm_4step.pt', 'unet_lcm_8step.pt',
          'motion_adapter_lcm_4step.pt', 'motion_adapter_lcm_8step.pt']:
    hf_hub_download('tenstorrent/tt-animatediff-lcm', f, local_dir='weights/')
"
```

Then run inference:

```bash
python examples/generate.py \
    --mode blackhole \
    --prompt "aurora borealis over arctic ice, cinematic 4K" \
    --lcm-unet weights/unet_lcm_8step.pt \
    --lcm-adapter weights/motion_adapter_lcm_8step.pt \
    --steps 8 \
    --frames 8
```

## Distillation method

Latent Consistency Model (LCM) distillation — student UNet learns to match the teacher's
clean-image predictions while skipping timesteps. Two phases:

1. **Phase 1**: SD 1.4 UNet distilled against itself (teacher frozen). 5000 gradient steps,
   AdamW lr=1e-5, CPU PyTorch. SNR-weighted consistency loss. Skip window `w ∈ [2, T/steps]`.

2. **Phase 2**: AnimateDiff MotionAdapter distilled against the original adapter, with the
   Phase 1 UNet frozen. 3000 gradient steps, same schedule.

Full details and beginner-friendly explanation:
[docs/DISTILLATION_GUIDE.md](https://github.com/tenstorrent/tt-animatediff/blob/main/docs/DISTILLATION_GUIDE.md)

## Base model license

These weights are derivatives of:
- **SD 1.4** (`CompVis/stable-diffusion-v1-4`) — [CreativeML Open RAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license)
- **AnimateDiff MotionAdapter** (`guoyww/animatediff-motion-adapter-v1-5-2`) — [Apache 2.0](https://github.com/guoyww/AnimateDiff/blob/main/LICENSE)

These distilled weights are released under the same **CreativeML Open RAIL-M** license as SD 1.4.
You must comply with its use restrictions (no harmful content generation, etc.) when using these weights.

## Hardware

Distilled on: 4× Tenstorrent Blackhole P300C · 16-core x86 · 249 GB RAM  
Inference validated on: same hardware via `scripts/validate_parallel.py`

## Citation

```bibtex
@misc{tt-animatediff-lcm,
  author       = {Tenstorrent AI ULC},
  title        = {tt-animatediff LCM Distilled Weights},
  year         = {2026},
  howpublished = {Hugging Face Hub},
  url          = {https://huggingface.co/tenstorrent/tt-animatediff-lcm}
}
```
