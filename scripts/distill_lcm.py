#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Phase 1: LCM (Latent Consistency Model) distillation of the SD 1.4 UNet.

What this file does
-------------------
Takes the full 25-step SD 1.4 UNet (the "teacher") and trains a copy of it
(the "student") to produce the same denoised output in just 4 or 8 steps.

The technique is called consistency distillation:
- We pick a random noisy latent z_t at timestep t.
- The teacher denoises it one step to get z_{t-k} (a less-noisy latent).
- We then ask the student: "given z_t, predict the fully clean image directly."
- We also ask the teacher: "given z_{t-k}, predict the fully clean image."
- These two predictions should be the same — that's the consistency constraint.
- We backpropagate the difference (consistency_loss) into the student only.

After enough steps the student learns to "skip" the intermediate timesteps.

Usage
-----
Run directly:
    python scripts/distill_lcm.py --steps 4 --num_train_steps 5000
    python scripts/distill_lcm.py --steps 8 --num_train_steps 5000

Outputs
-------
    weights/unet_lcm_4step.pt
    weights/unet_lcm_8step.pt
"""

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DConditionModel
from tqdm import tqdm

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Core loss primitives (tested independently in tests/test_distill_lcm.py)
# ---------------------------------------------------------------------------

def consistency_loss(
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Weighted MSE between student and teacher x0 predictions.

    Args:
        student_pred: Student's prediction of the clean image, shape (B, C, H, W).
        teacher_pred: Teacher's prediction of the clean image, shape (B, C, H, W).
                      Detached from the computation graph — teacher is frozen.
        weight: Per-sample loss weight, shape (B,). Higher weight = more emphasis.

    Returns:
        Scalar loss tensor.
    """
    # MSE per sample: mean over C, H, W dimensions, keeping batch dim
    per_sample = F.mse_loss(student_pred, teacher_pred.detach(), reduction="none")
    per_sample = per_sample.mean(dim=[1, 2, 3])  # (B,)
    # Apply per-sample weight and average across batch
    return (weight * per_sample).mean()


def sample_timestep_pairs(
    batch_size: int,
    num_timesteps: int,
    w_min: int,
    w_max: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (t_student, t_teacher) pairs where t_teacher = t_student + gap.

    The student denoises from t_teacher all the way to 0 in one step.
    The teacher denoises from t_teacher to t_student (one small step),
    then from t_student to 0 (another small step).
    The gap is the "skip window" — how many timesteps the student learns to skip.

    Args:
        batch_size: Number of pairs to sample.
        num_timesteps: Total number of diffusion timesteps (usually 1000).
        w_min: Minimum skip window (inclusive).
        w_max: Maximum skip window (inclusive).

    Returns:
        t_student: Timesteps where the student starts, shape (batch_size,).
        t_teacher: Timesteps where both teacher and student start, shape (batch_size,).
                   Always t_teacher = t_student + gap, gap in [w_min, w_max].
    """
    assert 0 < w_min <= w_max < num_timesteps, (
        f"w_min/w_max must satisfy 0 < w_min <= w_max < num_timesteps, "
        f"got w_min={w_min}, w_max={w_max}, num_timesteps={num_timesteps}"
    )
    # Sample student timesteps leaving room for the gap
    t_student = torch.randint(0, num_timesteps - w_max, (batch_size,))
    # Sample a random gap for each sample in [w_min, w_max]
    gap = torch.randint(w_min, w_max + 1, (batch_size,))
    t_teacher = t_student + gap
    return t_student, t_teacher


def compute_loss_weight(
    timesteps: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """SNR-based loss weight: higher noise timesteps contribute less to the loss.

    The signal-to-noise ratio (SNR) at timestep t tells us how much signal
    is left in the noisy latent. At low t (little noise), SNR is high and
    we trust the prediction more. At high t (lots of noise), SNR is low
    and the prediction is harder — we down-weight those samples.

    Weight = 1 / (1 + 1/SNR(t)) = alpha_t^2 / (alpha_t^2 + sigma_t^2)
    where alpha_t = sqrt(alphas_cumprod[t]), sigma_t = sqrt(1 - alphas_cumprod[t]).

    Args:
        timesteps: Batch of timestep indices, shape (B,).
        alphas_cumprod: Cumulative product of (1 - beta_t) for each timestep,
                        shape (num_timesteps,). Available from DDPMScheduler.

    Returns:
        Per-sample loss weights, shape (B,), all positive.
    """
    alpha_sq = alphas_cumprod[timesteps].clamp(min=1e-8)  # shape (B,) — clamped to avoid division by zero
    sigma_sq = 1.0 - alpha_sq                      # shape (B,)
    snr = alpha_sq / sigma_sq.clamp(min=1e-8)      # shape (B,) — SNR = alpha^2/sigma^2
    # Down-weight high-noise (low SNR) timesteps
    weight = snr / (snr + 1.0)
    return weight
