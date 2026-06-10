# AnimateDiff-Lightning Distillation on Tenstorrent Blackhole — Design Spec

**Date:** 2026-06-10
**Branch:** model-resolute
**Status:** Approved, ready for implementation planning

---

## Summary

Distill AnimateDiff (SD 1.4 + MotionAdapter) into a "lightning" variant that generates
video in 4 or 8 denoising steps instead of 25–50. The process runs entirely on CPU using
PyTorch (no CUDA required), with 4× Blackhole chips used for parallel inference validation
after each phase. The full process is recorded with asciinema + VHS and documented for
readers who have never used PyTorch.

---

## Goals

- Produce `unet_lcm_4step.pt`, `unet_lcm_8step.pt` (Phase 1)
- Produce `motion_adapter_lcm_4step.pt`, `motion_adapter_lcm_8step.pt` (Phase 2)
- Record both training phases and inference validation as embeddable GIF/MP4
- Write a complete beginner-friendly distillation guide (`docs/DISTILLATION_GUIDE.md`)
- Make full use of: 16 CPU cores, 249 GB RAM, 4× Blackhole chips

---

## Architecture

### Phase 1 — LCM UNet Distillation (CPU)

```
teacher: SD 1.4 UNet (frozen, full 25-step)
student: SD 1.4 UNet (same arch, being trained)
loss:    consistency loss — student at step T must match teacher at step T-k
output:  weights/unet_lcm_4step.pt
         weights/unet_lcm_8step.pt
```

The training data is synthetic — (noise, timestep, denoised target) tuples are generated
on the fly from the teacher. No external dataset required. The teacher *is* the dataset.

Hyperparameters:
- `w_min / w_max`: consistency skipping window (e.g. 2–10 steps)
- `learning_rate`: 1e-5 to 1e-4 (documented with rationale)
- `batch_size` + `gradient_accumulation_steps`: tuned for 249 GB RAM / 16 cores
- `num_train_steps`: estimated wall-clock ~3 hours for Phase 1

Loss: consistency loss only (no adversarial). Adversarial extension noted in docs as
optional improvement.

### Phase 2 — Motion Adapter Distillation (CPU)

```
base:    unet_lcm_8step.pt (frozen, from Phase 1)
teacher: base + original MotionAdapter (guoyww/animatediff-motion-adapter-v1-5-2, 25 steps)
student: base + new MotionAdapter (same arch, being trained)
loss:    same consistency loss applied to temporal features
output:  weights/motion_adapter_lcm_4step.pt
         weights/motion_adapter_lcm_8step.pt
```

The MotionAdapter is ~40M parameters vs ~860M for the UNet. This phase runs much faster
(estimated ~45 minutes).

### Validation — 4× Blackhole Chips (Parallel)

After both phases complete, all 4 chips run inference simultaneously:

| Chip | UNet              | MotionAdapter             | Purpose                         |
|------|-------------------|---------------------------|---------------------------------|
| 0    | unet_lcm_4step    | original (25-step)        | spatial-only fast               |
| 1    | unet_lcm_8step    | original (25-step)        | spatial-only balanced           |
| 2    | unet_lcm_8step    | motion_adapter_lcm_8step  | full lightning, balanced        |
| 3    | unet_lcm_4step    | motion_adapter_lcm_4step  | full lightning, maximum speed   |

Each chip produces a GIF. Side-by-side comparison printed to terminal.

---

## Recording Strategy

### Two VHS recordings

**`scripts/record/distill.tape`** (~15–20 min playback)
- tmux split: left = training loop with live loss/step counter; right = `tt-smi -s` refreshing
- VHS `Type` commands print explanatory narration before each major step
- Milestone banners: "Phase 1 complete — LCM UNet saved", etc.
- Ends with 4-chip parallel validation, all 4 GIFs appear in output directory

**`scripts/record/inference.tape`** (~3 min playback)
- Clean demo: load distilled weights, generate video
- Terminal comparison table: `Teacher: 25 steps, 47s | 8-step: 14s | 4-step: 8s`
  (actual measured times filled in after the run)

### Why VHS over raw asciinema

VHS produces `.gif`/`.mp4` that embed directly in GitHub README. Asciinema requires a
player. Strategy: record the real run with asciinema (preserves exact timing/output),
replay through VHS for the clean embeddable artifact. Both committed.

### Directory layout

```
scripts/record/
  distill.tape          ← VHS script (drives run_distill.sh)
  inference.tape        ← VHS script (drives run_inference.sh)
  run_distill.sh        ← actual training script
  run_inference.sh      ← inference demo
  recordings/
    distill.gif         ← embeddable output
    inference.gif
    distill.cast        ← raw asciinema archive
```

---

## Documentation Structure (`docs/DISTILLATION_GUIDE.md`)

Audience: external ML practitioners and people completely new to PyTorch.

| Section | Content |
|---------|---------|
| 1. What is distillation? | Student/teacher analogy, ASCII timestep-skipping diagram, no math |
| 2. What is AnimateDiff-Lightning? | Fewer steps: cost/benefit, two-phase structure, link to C sidebar |
| 3. Prerequisites | Hardware (Blackhole P100/P300), software (tt-metal, Python 3.10+), estimated time |
| 4. Phase 1: LCM UNet Distillation | Line-by-line annotated code, "What is a UNet?" sidebar, **C sidebar** here |
| 5. Phase 2: Motion Adapter Distillation | Same concept at smaller scale, "What is a MotionAdapter?" sidebar |
| 6. Validation on Blackhole | 4-chip assignment, reading `tt-smi`, interpreting GIF quality differences |
| 7. Results & Benchmarks | Steps × quality × time table (measured), embedded recordings |
| 8. Troubleshooting | ARC firmware hang (chip 3 history), L1 OOM, loss divergence |

Code files (`scripts/distill_lcm.py`, `scripts/distill_motion_adapter.py`) are heavily
commented. Guide references line numbers so readers can follow both simultaneously.

### The "Option C" Sidebar

A callout box in Section 4 after the Phase 1 loop is explained:

> **Shortcut: the `diffusers` LCM training script**
> If you'd rather not write the consistency distillation loop yourself, the Hugging Face
> `diffusers` library ships an official LCM training script that produces identical weights.
> Here's how to use it, and here's what it's doing internally — mapped to the code you
> just read above. We recommend the from-scratch approach for learning, but for production
> use the official script is well-tested.

---

## File Layout (new files this work produces)

```
scripts/
  distill_lcm.py                   ← Phase 1 training script
  distill_motion_adapter.py        ← Phase 2 training script
  validate_parallel.py             ← 4-chip parallel inference validation
  record/
    distill.tape
    inference.tape
    run_distill.sh
    run_inference.sh
    recordings/  (gitignored large files, except .cast)
weights/
  unet_lcm_4step.pt                (gitignored, generated)
  unet_lcm_8step.pt                (gitignored, generated)
  motion_adapter_lcm_4step.pt      (gitignored, generated)
  motion_adapter_lcm_8step.pt      (gitignored, generated)
docs/
  DISTILLATION_GUIDE.md            ← beginner guide
  superpowers/specs/
    2026-06-10-lightning-distillation-design.md  ← this file
```

---

## Known Constraints

- VAE decode stays on CPU — TTNN VAE conv_out OOMs on Blackhole L1 grid (existing constraint)
- Chip 3 (board 0000046131924055) has ARC firmware hang history — validation script uses
  existing hwmon sentinel check before assigning work to it
- No CUDA on this machine — all training is CPU PyTorch; estimated times reflect that
- The TTNN UNet does not yet have temporal attention blocks — validation uses the same
  spatial-only TTNN path as Phase 2 in the existing codebase

---

## Success Criteria

- [ ] Both distillation phases run to completion on this hardware
- [ ] All 4 GIF outputs from parallel validation are non-trivial (no blank/noise frames)
- [ ] `distill.gif` and `inference.gif` embed cleanly in the README
- [ ] A reader with no PyTorch background can follow the guide from start to finish
- [ ] Actual benchmark numbers (steps/time/quality) are filled into the guide
