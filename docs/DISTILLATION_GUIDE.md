# AnimateDiff-Lightning on Tenstorrent Blackhole — Distillation Guide

> **Who this is for:** Anyone curious how AI video models are trained and optimized —
> including people who have never used PyTorch or run ML code before.

---

## 1. What is distillation?

Imagine you have a very thorough teacher who solves every math problem by working through
25 careful steps. You watch them long enough that you start to see patterns — and you
learn to get the same answer in 4 steps by skipping the intermediate work.

That is literally what we are doing here. The "teacher" is an AI model that generates
video by slowly removing noise from a random image over 25 steps. The "student" is an
identical model that we train to get the same result in 4 or 8 steps.

The training signal is simple: if the teacher and student both look at the same noisy
image, their guesses about what the clean image looks like should match. We call this
the **consistency constraint**.

```
Noise level:    [HIGH ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← NONE]
                 T=1000                                   T=0
                   │                                        │
Teacher: ──────────┤──────┤──────┤──────┤──────┤──────┤───►│ (25 steps)
                   │      │      │      │      │      │
Student: ──────────┤──────┴──────┴──────┴──────┴──────┴───►│ (4 steps)
                   │                                        │
                 z_t                                      clean
```

After enough practice, the student "learns to skip" without losing quality.

---

## 2. What is AnimateDiff-Lightning?

[AnimateDiff](https://github.com/guoyww/AnimateDiff) generates video by running an
image diffusion model once per frame, but with a special module (the **MotionAdapter**)
that watches all frames simultaneously and enforces smooth motion between them.

AnimateDiff-Lightning is the distilled version — same model, 4–8 steps instead of 25.

**Two phases of distillation:**

1. **Phase 1 — LCM UNet:** Distill the spatial denoising (the part that handles what
   each frame looks like as an image).
2. **Phase 2 — MotionAdapter:** Distill the temporal attention (the part that handles
   how frames connect in time).

> **Sidebar: The `diffusers` shortcut (Option C)**
>
> Hugging Face's `diffusers` library ships an official LCM training script
> (`train_lcm_distill_sd_wikiart_lora.py`) that handles Phase 1 for you. If you want
> to use it:
>
> ```bash
> pip install diffusers[training]
> python -m diffusers.examples.consistency_distillation.train_lcm_distill_sd \
>     --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4 \
>     --output_dir=weights/unet_lcm_hf
> ```
>
> This produces weights identical to what Phase 1 below produces. The advantage of
> the from-scratch approach in this guide is that you understand every line — which
> matters when you need to debug, modify, or port to new hardware like Blackhole.

---

## 3. Prerequisites

**Hardware:**
- 1–4 Tenstorrent Blackhole chips (P100 or P300)
- Tested on: 4× Blackhole, 249 GB RAM, 16-core x86

**Software:**
```bash
# Python 3.10+ required
python3 --version

# Clone this repo
git clone https://github.com/tenstorrent/tt-animatediff
cd tt-animatediff
pip install -r requirements.txt

# Install tt-metal (for Blackhole inference validation)
# See docs/HARDWARE_COMPAT.md for full instructions
source ~/tt-metal/python_env/bin/activate
```

**Pre-download model weights** (saves time during training):
```bash
python3 -c "
from diffusers import UNet2DConditionModel, DDPMScheduler, MotionAdapter
UNet2DConditionModel.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='unet')
DDPMScheduler.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='scheduler')
MotionAdapter.from_pretrained('guoyww/animatediff-motion-adapter-v1-5-2')
print('All weights cached.')
"
```

**Estimated time:**
| Phase | Description | Estimated time |
|-------|-------------|---------------|
| Phase 1 (4-step) | LCM UNet distillation | ~1.5 hours |
| Phase 1 (8-step) | LCM UNet distillation | ~1.5 hours |
| Phase 2 (adapters) | MotionAdapter distillation ×2 | ~45 minutes |
| Validation | 4-chip parallel inference | ~15 minutes |
| **Total** | | **~4 hours** |

---

## 4. Phase 1: LCM UNet Distillation

> **New to UNets?** A UNet is a neural network shaped like the letter U — it compresses
> an image down (the left side of the U) to understand its overall structure, then
> expands it back up (the right side) to produce the output. In diffusion models, the
> UNet's job is to look at a noisy image and predict what noise was added so we can
> remove it. Do this enough times and you get a clean image.

**Run Phase 1:**
```bash
# Train the 4-step UNet (~1.5 hours on CPU)
python scripts/distill_lcm.py --steps 4 --num_train_steps 5000

# Train the 8-step UNet (~1.5 hours on CPU)
python scripts/distill_lcm.py --steps 8 --num_train_steps 5000
```

**What the code is doing** (`scripts/distill_lcm.py`):

1. **Load the teacher UNet** (line ~130): `UNet2DConditionModel.from_pretrained(...)` —
   this is the 859M-parameter SD 1.4 UNet, frozen (we never update its weights).

2. **Copy it to make the student** (line ~160): `copy.deepcopy(teacher)` — the student
   starts identical to the teacher and slowly diverges as it learns to skip steps.

3. **Training loop**: for each step:
   - Sample a clean random latent `z0` and Gaussian noise
   - Pick a timestep pair `(t_student, t_teacher)` where `t_teacher > t_student`
     — the gap is how many steps the student learns to skip
   - Add noise to `z0` at level `t_teacher` to get `z_noisy`
   - Teacher (frozen): denoise `z_noisy` from `t_teacher` to `t_student`, then predict
     clean image `x0_teacher`
   - Student: predict clean image `x0_student` directly from `z_noisy` at `t_teacher`
   - Loss: `mean((x0_student - x0_teacher)²)` — these should be the same!
   - Backpropagate loss into student only, update with AdamW

4. **Save student weights** when done: `weights/unet_lcm_{N}step.pt`

**Key hyperparameters and what they mean:**

| Param | Default | What it controls |
|-------|---------|-----------------|
| `--steps` | 8 | Target inference steps (4 or 8) |
| `--num_train_steps` | 5000 | How many gradient updates to run |
| `--lr` | 1e-5 | How fast the student adapts (too high → unstable) |
| `w_min` | 2 | Minimum skip window |
| `w_max` | T/steps | Maximum skip window (auto-calculated) |

**Expected output:**
```
LCM distill → 4-step:   0%|          | 0/5000 [00:00<?, ?it/s]
LCM distill → 4-step:   2%|▏         | 100/5000 [02:14<1:52:01, loss=0.0431]
...
Saved distilled 4-step UNet → weights/unet_lcm_4step.pt
```

Loss should decrease from ~0.04 to ~0.01 over 5000 steps. If it diverges (goes up),
reduce `--lr` to `5e-6`.

---

## 5. Phase 2: MotionAdapter Distillation

> **New to MotionAdapters?** The MotionAdapter is a set of attention layers plugged into
> the UNet. "Attention" means the model can look at multiple positions at once and decide
> which ones are related. The MotionAdapter adds attention across frames (instead of just
> within a frame), which is what creates smooth video motion.

**Run Phase 2:**
```bash
python scripts/distill_motion_adapter.py --steps 8 --unet weights/unet_lcm_8step.pt
python scripts/distill_motion_adapter.py --steps 4 --unet weights/unet_lcm_4step.pt
```

The process is the same as Phase 1, but only the MotionAdapter's ~40M parameters are
updated. The UNet from Phase 1 stays frozen. Because there are fewer parameters to train,
this runs ~4× faster.

---

## 6. Validation on Blackhole

```bash
# Runs all 4 configs in parallel, one chip each
python scripts/validate_parallel.py
```

**Chip assignments:**

| Chip | Config | What it tests |
|------|--------|--------------|
| 0 | 4-step UNet + original adapter | Is the spatial distillation working? |
| 1 | 8-step UNet + original adapter | Same, at higher quality |
| 2 | 8-step UNet + 8-step adapter | Full lightning, balanced |
| 3 | 4-step UNet + 4-step adapter | Full lightning, maximum speed |

**Reading `tt-smi` output while it runs:**
```bash
watch -n 2 tt-smi -s
```
Watch the `AICLK` field — it should be 0x320 (800MHz) for all active chips.
If a chip shows AICLK=0, it may be in the ARC hang state — see Troubleshooting.

---

## 7. Results & Benchmarks

*(Fill in after running validation on this hardware)*

| Config | Steps | Time (s) | Quality |
|--------|-------|----------|---------|
| Teacher (CPU baseline) | 25 | — | Reference |
| spatial-fast-4step | 4 | — | — |
| spatial-balanced-8step | 8 | — | — |
| lightning-8step | 8 | — | — |
| lightning-4step | 4 | — | — |

### Recordings

![Distillation process](assets/recordings/distill.gif)
*Full distillation run: training loop (left) + hardware monitor (right)*

![Inference demo](assets/recordings/inference.gif)
*Lightning 8-step inference demo*

---

## 8. Troubleshooting

**ARC firmware hang (chip shows anomalous temperature or power readings)**

This is a known issue with board 0000046131924055 (chip 3 on this system). The validation
script uses hwmon sentinel checks and will warn if a chip appears dead. If this happens:
1. AC power-cycle the machine (hold power button until fans stop, wait 10s, restart)
2. Verify all chips appear healthy: `tt-smi -s`
3. Re-run validation

**VAE OOM on Blackhole L1**

TTNN's VAE `conv_out` layer OOMs on the Blackhole L1 SRAM grid. This is a known
limitation — VAE decode is intentionally left on CPU in all configurations. Do not
try to move VAE to the Blackhole.

**Loss diverges during distillation**

If training loss increases instead of decreasing:
1. Reduce learning rate: `--lr 5e-6` or even `1e-6`
2. Reduce `--num_train_steps` to 2000 and check the loss trend at step 100

**`ImportError: cannot import name 'generate_frames_ttnn'`**

The validation script requires tt-metal to be activated:
```bash
source ~/tt-metal/python_env/bin/activate
python scripts/validate_parallel.py
```
