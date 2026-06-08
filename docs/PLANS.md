# tt-animatediff — Plans

## Distillation Training on Blackhole

### What we want

Train a distilled MotionAdapter that produces high-quality AnimateDiff outputs in 4 steps
instead of 25, with the student model running inference on Blackhole via the existing TTNN
UNet pipeline. The ByteDance AnimateDiff-Lightning paper showed this is achievable with
progressive adversarial distillation; we want to do it with weights trained on or for BH.

---

### Background: what distillation means here

AnimateDiff-Lightning distillation is a two-phase process:

1. **Teacher rollouts** — the full SD 1.4 + MotionAdapter runs 25-step PNDM/DDIM
   trajectories, producing (noise, denoised latent) pairs at each step.
2. **Student training** — a second network (same architecture) learns to reproduce
   those trajectories in 16 → 8 → 4 → 2 steps via progressive adversarial loss.
   The adversarial discriminator ensures the student's shorter trajectories land in the
   same distribution as the teacher's full ones, not just a blurry mean.

The student's MotionAdapter weights are what gets saved as the distilled checkpoint
(ByteDance's `mm_sd_v15_v2.ckpt` equivalent at each step count).

---

### Paths to get there

#### Path A — Train on CPU/GPU, deploy on BH (most practical near-term)

1. Assemble a text-image-video dataset (LAION subsets, WebVid, or synthetic rollouts
   from the existing CPU pipeline).
2. Run progressive adversarial fine-tuning on a GPU node (A100 or H100):
   - Stage 1: 16-step student from 25-step teacher
   - Stage 2: 8-step student from 16-step student
   - Stage 3: 4-step student (matches ByteDance's released checkpoints)
   - Stage 4 (optional): 2-step student
3. Save `.safetensors` checkpoints after each stage.
4. Load the 4-step checkpoint into `create_lightning_pipeline()` (CPU path) and verify
   quality parity with ByteDance's released weights.
5. Adapt inference to run through the TTNN UNet on BH — the student's MotionAdapter
   weights inject into the same UNet blocks already compiled for BH.

**Effort:** 2–4 weeks of GPU time depending on dataset size. BH inference is already
wired; only the training loop needs a GPU. This is the path that ships first.

#### Path B — TTNN backward pass (training on BH)

The TTNN UNet forward pass is compiled. Training requires autograd through those ops —
currently not supported in the tt-animatediff stack.

What would need to exist:
- TTNN op gradients for Conv2d, GroupNorm, SiLU, Attention registered with PyTorch autograd
- A mixed-precision training loop that keeps weights on BH L1 between steps
- Gradient checkpointing to fit the UNet activations in L1 across the full rollout

This is a significant project but one Tenstorrent's core team would likely be interested
in — training large models on BH is on the roadmap. This path would make tt-animatediff
a reference training workload, not just an inference demo.

**Effort:** 3–6 months, requires coordination with tt-metal team on op gradient support.

#### Path C — Chain-threaded curriculum distillation (novel approach, BH-native)

This is the most interesting path architecturally. It uses the `--chain` latent threading
we built as the distillation signal itself:

**Concept:**
- Teacher: 25-step generation, chain-save final latents
- Student: N-step generation, chain-from teacher latents at alpha=0.8
- Loss: MSE between student's final latents and teacher's, plus perceptual loss on decoded frames
- Curriculum: progressively lower `--chain-alpha` from 0.8 → 0.0 as the student improves
  (the student learns to match the teacher without needing the latent crutch)

This is a form of **latent-space curriculum learning** — the teacher's denoised state
provides a warm initialisation that the student is trained to not need. Because the
chain mechanism operates at the latent level (not inside UNet blocks), this approach
works entirely in Python with PyTorch autograd on the MotionAdapter weights — no TTNN
backward pass required.

**Why this is only possible here:** The `--chain` primitive captures hardware-resident
denoised state across generation calls. This kind of persistent-state curriculum signal
doesn't exist in standard diffusers because latents don't persist between pipeline calls.
The Blackhole L1 residency is what makes it efficient at scale.

**Effort:** 6–8 weeks with a GPU for the actual gradient steps. The infrastructure
(chain save/load, latent blending) is already built.

---

### Recommended sequence

1. **Now:** Collect teacher rollouts using existing CPU pipeline → build a dataset of
   (prompt, 25-step latent trajectories). This can run while other work happens.
2. **Near-term:** Path A — GPU-based progressive distillation → 4-step checkpoint →
   deploy on BH via existing TTNN pipeline. Validates the full training→deployment loop.
3. **Medium-term:** Path C — chain-threaded curriculum distillation → produces a
   checkpoint that required BH's stateful inference to train. First publishable result
   that is architecturally native to Tenstorrent hardware.
4. **Long-term:** Path B — native TTNN training — coordinate with tt-metal team.
   Makes BH a first-class training accelerator for video diffusion.

---

### What we already have

- Full 25-step teacher inference pipeline on BH (`generate_frames_temporal`)
- `--chain` latent save/load with configurable blend alpha
- CPU Lightning path (`create_lightning_pipeline`) for loading distilled checkpoints
- `generate_study.py` and `generate_gallery.py` for automated batch generation
  (can be adapted to produce training rollout datasets)
- TTNN UNet forward pass compiled and running on P300C

---

### Open questions

- Dataset licensing: LAION subset vs synthetic rollouts from our own pipeline
- Whether to use ByteDance's adversarial discriminator design or a simpler consistency
  loss (consistency models approach — no discriminator needed, potentially more stable)
- Step count target: 4-step for BH inference (matches ByteDance) vs 8-step (higher quality)
- Whether Path C produces checkpoints that generalise beyond the chain-demo prompts or
  overfit to the training subject matter

---

## Persist Prompt Guidance to Chained Generations

### Problem

The `--chain` mechanism carries coarse composition (subject silhouette, layout) from the
previous run's latents into the next run's seed noise. At `chain_alpha=0.20` with a 9px
low-pass blur, hop 1→2 is visually convincing — the subject persists while the scene
changes. Hops beyond 2 currently fail: the chained composition signal degrades each hop,
and by hop 3 the prompt takes over without any recognisable continuity.

Additionally, the prompts at hops 2+ are at a disadvantage: the composition bias from
hop N-1 partially counteracts the text guidance, so distinct-world prompts (ocean, forest,
circuit) converge toward a similar neutral-gray appearance rather than producing their
intended colour palettes.

### Root cause

The chain blending operates only at the **noise seed** level, before any text conditioning.
CLIP text embeddings are applied uniformly across all denoising steps — they have no way
to "reinforce" themselves against the structural prior coming from the chained latent.
By the middle of the diffusion trajectory the text guidance loses ground to the implicit
layout prior, and by hop 3 the latent fingerprint has drifted far enough from the prompt
that the two signals are incoherent.

### Proposed approach: prompt-weighted chain injection

Two complementary directions:

#### A — Adaptive chain_alpha decay

Use a per-hop decay: `effective_alpha = chain_alpha * decay^hop_index`. A 10% decay per
hop (`decay=0.9`) would give alpha=0.20, 0.18, 0.16, … — the composition nudge weakens
gracefully rather than staying constant while the accumulated drift grows.

Implementation: thread `hop_index` through to `generate_frames_temporal`, compute
`effective_alpha = chain_alpha * (decay ** hop_index)`, apply.

#### B — Text embedding re-injection at mid-trajectory ("prompt anchoring")

The standard pipeline applies text embeddings at every step but treats them as fixed.
We can amplify them at specific steps where the structural prior is strongest:

1. Identify the "transition steps" — the middle third of the PNDM/Euler schedule where
   the image structure is being settled (roughly steps N//3 to 2*N//3).
2. At those steps, scale the text embedding contribution up by a factor (e.g. ×1.5)
   while scaling the unconditional embedding down correspondingly — effectively a
   step-local CFG boost targeted at the compositional transition zone.
3. This costs nothing at inference (no extra forward passes) and requires only a
   modification to the guidance scaling in `generate_frames_temporal`'s denoising loop.

#### C — Carry the prompt embedding, not just the latent

Save the CLIP text embedding alongside the latent `.pt` file (`--chain-save` stores
both). On the next hop, blend a small fraction (e.g. 5-10%) of the previous run's text
embedding into the current run's — creating a "soft subject memory" at the conditioning
level, not just the noise level.

Risk: may cause text-embedding drift across many hops (same accumulation problem as the
original latent approach). Needs per-channel normalisation of the embedding space, or a
projection step that preserves the new prompt's direction while allowing a small pull
toward the previous subject concept.

### Recommended sequence

1. **First**: implement A (adaptive alpha decay) — trivial change, immediate test.
2. **Second**: implement B (mid-trajectory prompt anchoring) — high leverage, no extra
   compute, should directly fix the "prompt ignored" symptom on hops 2+.
3. **Third**: evaluate C (embedding carry) after B is validated — only needed if B alone
   doesn't provide enough subject persistence across 4+ hops.

### Success criteria

- Hop 3 and beyond maintain visually distinct colour palettes per scene (ocean=turquoise,
  forest=green, circuit=violet) rather than converging to neutral gray.
- Subject silhouette remains recognisable across at least 4 consecutive hops.
- The chain-demo script produces a compelling 6-scene sequence suitable for the website.
