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


# ---------------------------------------------------------------------------
# Noise / denoising helpers
# ---------------------------------------------------------------------------

def add_noise(
    latent: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """Forward diffusion: add noise to a clean latent at given timesteps.

    The noisy latent at timestep t is:
        z_t = sqrt(alpha_t) * z_0 + sqrt(1 - alpha_t) * epsilon

    where epsilon is Gaussian noise and alpha_t is the noise schedule value
    at timestep t.

    Args:
        latent:          Clean latent tensor, shape (B, C, H, W).
        noise:           Gaussian noise, same shape as latent.
        timesteps:       Timestep indices for each sample, shape (B,).
        alphas_cumprod:  Noise schedule, shape (num_timesteps,).

    Returns:
        Noisy latent z_t, same shape as latent.
    """
    # Broadcast to (B, 1, 1, ...) matching latent's number of dims (4D or 5D).
    extra_dims = (1,) * (latent.dim() - 1)
    sqrt_alpha = alphas_cumprod[timesteps].sqrt().view(-1, *extra_dims)
    sqrt_one_minus_alpha = (1.0 - alphas_cumprod[timesteps]).sqrt().view(-1, *extra_dims)
    return sqrt_alpha * latent + sqrt_one_minus_alpha * noise


def predict_x0(
    unet: torch.nn.Module,
    noisy_latent: torch.Tensor,
    timesteps: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """Ask the UNet to predict the clean image from a noisy latent.

    The UNet predicts the noise epsilon that was added (noise prediction).
    We invert the forward diffusion formula to recover x0:
        x0 = (z_t - sqrt(1 - alpha_t) * predicted_noise) / sqrt(alpha_t)

    Args:
        unet:                   UNet model (teacher or student).
        noisy_latent:           Noisy input z_t, shape (B, C, H, W).
        timesteps:              Timestep indices, shape (B,).
        encoder_hidden_states:  Text embeddings, shape (B, seq_len, hidden_dim).
        alphas_cumprod:         Noise schedule, shape (num_timesteps,).

    Returns:
        Predicted clean latent x0, shape (B, C, H, W).
    """
    # Use return_dict=True so both real diffusers UNets (which return a named
    # object) and test stubs (which attach .sample directly) work uniformly.
    noise_pred = unet(
        noisy_latent,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        return_dict=True,
    ).sample

    extra_dims = (1,) * (noisy_latent.dim() - 1)
    sqrt_alpha = alphas_cumprod[timesteps].sqrt().view(-1, *extra_dims)
    sqrt_one_minus_alpha = (1.0 - alphas_cumprod[timesteps]).sqrt().view(-1, *extra_dims)
    x0_pred = (noisy_latent - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha.clamp(min=1e-8)
    return x0_pred


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_distillation(
    teacher_unet: torch.nn.Module,
    alphas_cumprod: torch.Tensor,
    num_train_steps: int,
    target_steps: int,
    output_path: Path,
    latent_shape: tuple,
    encoder_hidden_states: torch.Tensor,
    learning_rate: float = 1e-5,
    w_min: int = 2,
    w_max: int = 10,
) -> None:
    """Distill teacher_unet into a student that converges in target_steps steps.

    Creates a deep copy of teacher_unet as the student. Trains with
    consistency loss only (no adversarial term). Saves student state dict
    to output_path when finished.

    Args:
        teacher_unet:            Frozen teacher model.
        alphas_cumprod:          Noise schedule from DDPMScheduler, shape (T,).
        num_train_steps:         Number of gradient update steps.
        target_steps:            Inference steps the student should need (4 or 8).
        output_path:             Where to save the distilled weights (.pt file).
        latent_shape:            Shape of one training latent (B, C, H, W).
        encoder_hidden_states:   Text embeddings repeated for batch, shape (B, seq, hid).
        learning_rate:           Adam learning rate.
        w_min:                   Minimum timestep skip gap.
        w_max:                   Maximum timestep skip gap.
    """
    teacher_unet.eval()
    for p in teacher_unet.parameters():
        p.requires_grad_(False)

    student_unet = copy.deepcopy(teacher_unet)
    student_unet.train()
    # Re-enable gradients on student parameters — deepcopy preserves the
    # teacher's requires_grad=False state, so we must explicitly flip it back.
    for p in student_unet.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(student_unet.parameters(), lr=learning_rate)

    num_timesteps = len(alphas_cumprod)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(range(num_train_steps), desc=f"LCM distill → {target_steps}-step")

    for step in pbar:
        B = latent_shape[0]
        z0 = torch.randn(*latent_shape)
        noise = torch.randn_like(z0)

        t_student, t_teacher = sample_timestep_pairs(B, num_timesteps, w_min, w_max)
        z_teacher = add_noise(z0, noise, t_teacher, alphas_cumprod)

        with torch.no_grad():
            # return_dict=True gives a named object with .sample; works for both
            # real diffusers UNets and lightweight test stubs.
            eps_teacher = teacher_unet(
                z_teacher, t_teacher,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=True,
            ).sample
            sqrt_alpha_t = alphas_cumprod[t_teacher].sqrt().view(-1, 1, 1, 1)
            sqrt_one_minus_t = (1 - alphas_cumprod[t_teacher]).sqrt().view(-1, 1, 1, 1)
            sqrt_alpha_s = alphas_cumprod[t_student].sqrt().view(-1, 1, 1, 1)
            sqrt_one_minus_s = (1 - alphas_cumprod[t_student]).sqrt().view(-1, 1, 1, 1)
            x0_from_teacher_t = (z_teacher - sqrt_one_minus_t * eps_teacher) / sqrt_alpha_t.clamp(min=1e-8)
            # Fresh noise required — reusing the eps that built z_teacher would bias teacher_x0.
            eps_renoise = torch.randn_like(z0)
            z_student_level = sqrt_alpha_s * x0_from_teacher_t + sqrt_one_minus_s * eps_renoise
            teacher_x0 = predict_x0(teacher_unet, z_student_level, t_student,
                                     encoder_hidden_states, alphas_cumprod)

        student_x0 = predict_x0(student_unet, z_teacher, t_teacher,
                                 encoder_hidden_states, alphas_cumprod)

        weight = compute_loss_weight(t_teacher, alphas_cumprod)
        loss = consistency_loss(student_x0, teacher_x0, weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    torch.save(student_unet.state_dict(), output_path)
    print(f"\nSaved distilled {target_steps}-step UNet → {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LCM UNet distillation")
    parser.add_argument("--steps", type=int, choices=[4, 8], default=8,
                        help="Target inference steps (4 or 8)")
    parser.add_argument("--num_train_steps", type=int, default=5000,
                        help="Number of gradient update steps per run")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="AdamW learning rate")
    parser.add_argument("--model_id", type=str,
                        default="CompVis/stable-diffusion-v1-4")
    args = parser.parse_args()

    print(f"Loading teacher UNet from {args.model_id}...")
    teacher = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet")
    scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    alphas_cumprod = scheduler.alphas_cumprod

    # Fixed random text embedding for distillation. LCM consistency distillation does
    # not require real prompts — the self-consistency constraint holds regardless of
    # the conditioning signal. A real deployment would use a CLIPTextModel here for
    # prompt-conditional generation quality.
    encoder_hs = torch.randn(1, 77, 768)
    latent_shape = (1, 4, 64, 64)
    num_timesteps = len(alphas_cumprod)
    w_max = num_timesteps // args.steps

    out = REPO_ROOT / "weights" / f"unet_lcm_{args.steps}step.pt"
    run_distillation(
        teacher_unet=teacher,
        alphas_cumprod=alphas_cumprod,
        num_train_steps=args.num_train_steps,
        target_steps=args.steps,
        output_path=out,
        latent_shape=latent_shape,
        encoder_hidden_states=encoder_hs,
        learning_rate=args.lr,
        w_min=2,
        w_max=w_max,
    )


if __name__ == "__main__":
    main()
