#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Quick validation probe for LCM distilled weights.

The KEY quality signal is prompt-diversity: a correctly conditioned model
produces different images for semantically different prompts, even with the
same seed.  A broken/unconditioned model (trained with torch.randn() instead
of real CLIP embeddings) ignores the prompt and produces nearly identical
noisy output for every prompt.

We generate two frames at the same seed with opposite prompts and compute:
  - L2 distance between the two images (higher = more prompt-responsive)
  - Per-channel std of each image (sanity-check for degenerate collapse)

Real images from a correctly conditioned model:   L2 ≈ 0.30–0.60
Broken model (random conditioning during training): L2 ≈ 0.00–0.05

A threshold of 0.15 gives clear separation.

The probe also optionally runs the base SD 1.4 model at 25 steps so you can
see the expected L2 for a healthy reference model.

Usage:
    # Probe distilled weights — fast (~25s total for 2 × 4-step generations)
    python scripts/probe_distill.py --unet weights/unet_lcm_4step.pt --steps 4

    # Skip the 25-step base-model reference (saves ~2 min, lose the baseline)
    python scripts/probe_distill.py --unet weights/unet_lcm_4step.pt \\
        --steps 4 --skip-baseline

    # Probe base model itself (sanity-check the probe setup)
    python scripts/probe_distill.py --steps 25 --skip-baseline

Exit codes: 0 = HEALTHY, 1 = SUSPECT.
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Two semantically opposite prompts — same seed should give very different
# images for a correctly conditioned model.
PROMPT_A = "aurora borealis over arctic tundra, vivid green and purple curtains of light"
PROMPT_B = "desert sandstorm at noon, arid orange dust, flat horizon, bleached sky"


def generate_frame(pipe, prompt: str, num_steps: int, seed: int,
                   lcm: bool = True) -> torch.Tensor:
    """Return a single decoded frame tensor (C, H, W) in [-1, 1]."""
    if lcm:
        from diffusers import EulerDiscreteScheduler
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
            beta_schedule="linear",
        )
        guidance = 1.0
    else:
        from diffusers import DDIMScheduler
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        guidance = 7.5

    generator = torch.Generator().manual_seed(seed)
    result = pipe(
        prompt,
        num_frames=1,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        generator=generator,
        output_type="pt",
    )
    frame = result.frames[0][0]
    if frame.dim() == 4:
        frame = frame[0]
    return frame * 2.0 - 1.0  # [0,1] → [-1,1]


def prompt_diversity_l2(frame_a: torch.Tensor, frame_b: torch.Tensor) -> float:
    """Mean L2 distance per pixel between two image tensors.

    Range: 0 (identical) to ~2.83 (max possible for [-1,1] range).
    A correctly conditioned model should score > 0.15 for semantically
    opposite prompts.  An unconditioned model scores < 0.05.
    """
    return (frame_a - frame_b).pow(2).mean().sqrt().item()


def channel_std(frame: torch.Tensor) -> float:
    """Mean per-channel std — sanity check for degenerate collapse (all grey)."""
    return frame.std(dim=[1, 2]).mean().item()


def save_thumbnail(frame: torch.Tensor, path: Path) -> None:
    try:
        from PIL import Image
        img_np = ((frame.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).numpy()
        path.parent.mkdir(exist_ok=True)
        Image.fromarray(img_np).save(str(path))
    except Exception:
        pass


def load_pipe(unet_path=None, adapter_path=None):
    from animatediff_ttnn.pipeline import create_animatediff_pipeline
    pipe = create_animatediff_pipeline()
    if unet_path:
        p = Path(unet_path)
        if not p.exists():
            print(f"ERROR: UNet weights not found: {p}")
            sys.exit(1)
        state = torch.load(p, map_location="cpu", weights_only=True)
        missing, unexpected = pipe.unet.load_state_dict(state, strict=False)
        non_motion_missing = [k for k in missing if "motion_modules" not in k]
        print(f"  Loaded UNet: {len(state)} keys, {len(missing)} missing "
              f"({len(non_motion_missing)} non-motion-module), {len(unexpected)} unexpected")
        if non_motion_missing:
            print(f"  WARN unexpected missing keys: {non_motion_missing[:5]}")
    if adapter_path:
        p = Path(adapter_path)
        if p.exists():
            state = torch.load(p, map_location="cpu", weights_only=True)
            pipe.motion_adapter.load_state_dict(state, strict=False)
            print(f"  Loaded adapter: {len(state)} keys")
    return pipe


def run_diversity_probe(pipe, steps: int, seed: int, lcm: bool, label: str):
    """Generate frames for PROMPT_A and PROMPT_B, return diversity metrics dict."""
    print(f"  [{label}] frame A @ {steps} steps: {PROMPT_A[:60]}...")
    fa = generate_frame(pipe, PROMPT_A, steps, seed, lcm=lcm)
    print(f"  [{label}] frame B @ {steps} steps: {PROMPT_B[:60]}...")
    fb = generate_frame(pipe, PROMPT_B, steps, seed, lcm=lcm)
    return {
        "frame_a": fa,
        "frame_b": fb,
        "l2": prompt_diversity_l2(fa, fb),
        "std_a": channel_std(fa),
        "std_b": channel_std(fb),
    }


def main():
    parser = argparse.ArgumentParser(description="Probe LCM distilled weights for validity")
    parser.add_argument("--unet", default=None, metavar="PATH",
                        help="Path to distilled UNet .pt (omit to test base model at 25 steps)")
    parser.add_argument("--adapter", default=None, metavar="PATH",
                        help="Path to distilled MotionAdapter .pt (optional)")
    parser.add_argument("--steps", type=int, default=4,
                        help="Inference steps (default 4 for distilled, 25 for base)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed — both prompts use the same seed")
    # Threshold: L2 diversity below this = unconditioned model (broken)
    # Healthy conditioned model: L2 ≈ 0.30–0.60
    # Broken unconditioned model: L2 ≈ 0.00–0.05
    # Threshold at 0.15 gives wide margin between the two regimes.
    parser.add_argument("--l2-threshold", type=float, default=0.15,
                        help="Min L2 diversity between prompt A/B to pass (default 0.15)")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip base SD 1.4 @ 25-step reference generation")
    args = parser.parse_args()

    is_lcm = args.unet is not None
    steps = args.steps if args.steps != 4 or is_lcm else (4 if is_lcm else 25)
    mode = Path(args.unet).stem if args.unet else "base_no_distill"

    print(f"\nLoading pipeline ({mode})...")
    pipe = load_pipe(args.unet, args.adapter)

    print(f"\nRunning prompt-diversity probe ({steps} steps, seed {args.seed}):")
    result = run_diversity_probe(pipe, steps, args.seed, lcm=is_lcm, label=mode)

    # Save thumbnails
    thumb_a = REPO_ROOT / f"generated/probe_{mode}_{steps}step_A.png"
    thumb_b = REPO_ROOT / f"generated/probe_{mode}_{steps}step_B.png"
    save_thumbnail(result["frame_a"], thumb_a)
    save_thumbnail(result["frame_b"], thumb_b)

    print(f"\n╔══════════════════════════════════════════════════════════")
    print(f"║  Probe: {mode} @ {steps} steps")
    print(f"╠══════════════════════════════════════════════════════════")
    print(f"║  L2 prompt diversity : {result['l2']:.4f}  (threshold: ≥{args.l2_threshold})")
    print(f"║  std frame A         : {result['std_a']:.4f}")
    print(f"║  std frame B         : {result['std_b']:.4f}")
    print(f"║  thumbnail A         : {thumb_a.name}  [{PROMPT_A[:40]}...]")
    print(f"║  thumbnail B         : {thumb_b.name}  [{PROMPT_B[:40]}...]")

    # Optional: baseline comparison using real base model
    baseline_l2 = None
    if not args.skip_baseline and is_lcm:
        print(f"║")
        print(f"║  Running baseline (SD 1.4 @ 25 steps) for reference...")
        base_pipe = load_pipe()  # no weights — clean base model
        base_result = run_diversity_probe(base_pipe, 25, args.seed, lcm=False, label="baseline")
        baseline_l2 = base_result["l2"]
        save_thumbnail(base_result["frame_a"], REPO_ROOT / "generated/probe_baseline_25step_A.png")
        save_thumbnail(base_result["frame_b"], REPO_ROOT / "generated/probe_baseline_25step_B.png")
        print(f"║  Baseline L2         : {baseline_l2:.4f}  (SD 1.4 @ 25 steps)")
        if baseline_l2 > 0:
            ratio = result["l2"] / baseline_l2
            print(f"║  Ratio vs baseline   : {ratio:.2f}  (distilled / baseline)")

    # Verdict
    print(f"╠══════════════════════════════════════════════════════════")
    healthy = True
    reasons = []

    if result["l2"] < args.l2_threshold:
        healthy = False
        reasons.append(
            f"L2 diversity {result['l2']:.4f} < {args.l2_threshold} — "
            "model produces nearly identical output regardless of prompt "
            "(text conditioning not working)"
        )

    # Sanity: if either image has very low std, the model collapsed to a constant
    for label, std_val in [("frame A", result["std_a"]), ("frame B", result["std_b"])]:
        if std_val < 0.05:
            healthy = False
            reasons.append(f"{label} std {std_val:.4f} < 0.05 — output collapsed to near-constant")

    if healthy:
        print("║  VERDICT: ✓ HEALTHY — model responds to text conditioning")
    else:
        print("║  VERDICT: ✗ SUSPECT — model does not respond to text conditioning")
        for r in reasons:
            print(f"║    • {r}")
        print("║  Likely cause: trained with torch.randn() embeddings instead of real CLIP")
    print(f"╚══════════════════════════════════════════════════════════")
    sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()
