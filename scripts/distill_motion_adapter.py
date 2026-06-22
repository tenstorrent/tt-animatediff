#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Phase 2: Distill the AnimateDiff MotionAdapter to work in fewer steps.

What this file does
-------------------
After Phase 1 produced a fast-denoising UNet, this phase makes the temporal
attention (the part that creates smooth video motion) equally fast.

The MotionAdapter is the module that makes AnimateDiff different from a
static image generator. It adds attention blocks between UNet layers that
look at multiple frames at once and make sure motion is smooth.

We freeze the LCM UNet from Phase 1 and apply the same consistency
distillation approach to the MotionAdapter:
- Teacher: frozen LCM UNet + original 25-step MotionAdapter
- Student: frozen LCM UNet + new MotionAdapter (being trained)
- Loss:    same consistency MSE from distill_lcm.py

The MotionAdapter is only ~40M parameters (vs 859M for the UNet),
so this phase trains much faster — roughly 45 minutes.

Usage
-----
    python scripts/distill_motion_adapter.py --steps 8 --unet weights/unet_lcm_8step.pt
    python scripts/distill_motion_adapter.py --steps 4 --unet weights/unet_lcm_4step.pt

Outputs
-------
    weights/motion_adapter_lcm_8step.pt
    weights/motion_adapter_lcm_4step.pt
"""

import argparse
import copy
import sys
from pathlib import Path

import torch
from diffusers import AnimateDiffPipeline, DDPMScheduler, MotionAdapter
from tqdm import tqdm

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.distill_lcm import (
    add_noise,
    consistency_loss,
    compute_loss_weight,
    predict_x0,
    sample_timestep_pairs,
)


def build_temporal_pipeline(
    unet_weights_path,
    model_id: str = "CompVis/stable-diffusion-v1-4",
    adapter_id: str = "guoyww/animatediff-motion-adapter-v1-5-2",
    load_unet_weights: bool = True,
) -> AnimateDiffPipeline:
    """Load AnimateDiffPipeline and optionally replace UNet with distilled weights.

    Args:
        unet_weights_path:  Path to a .pt state dict from Phase 1, or None.
        model_id:           HuggingFace model ID for SD 1.4.
        adapter_id:         HuggingFace ID for the AnimateDiff MotionAdapter.
        load_unet_weights:  If True, load unet_weights_path into the pipeline UNet.

    Returns:
        AnimateDiffPipeline with the UNet replaced by the distilled version.
    """
    pipe = AnimateDiffPipeline.from_pretrained(
        model_id,
        motion_adapter=MotionAdapter.from_pretrained(adapter_id),
        torch_dtype=torch.float32,
    )
    if load_unet_weights and unet_weights_path is not None:
        state = torch.load(unet_weights_path, map_location="cpu", weights_only=True)
        pipe.unet.load_state_dict(state, strict=False)
        print(f"Loaded distilled UNet weights from {unet_weights_path}")
    return pipe


def run_adapter_distillation(
    teacher_unet: torch.nn.Module,
    student_unet: torch.nn.Module,
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
    """Distill the motion modules so they converge in target_steps denoising steps.

    Both teacher_unet and student_unet are full UNetMotionModel instances
    (LCM backbone + MotionAdapter fused). The teacher is completely frozen.
    In the student, only the motion_modules parameters are trainable — the
    backbone weights are shared from Phase 1 and stay frozen.

    We use the same consistency distillation loss as Phase 1:
      student x0(z_student, t_student) ≈ teacher x0(z_teacher, t_teacher)
    where t_student < t_teacher (student skips steps, teacher takes the full path).

    Args:
        teacher_unet:            Fully frozen reference UNet (original motion modules).
        student_unet:            UNet with only motion_modules trainable.
        alphas_cumprod:          Noise schedule, shape (T,).
        num_train_steps:         Gradient update steps.
        target_steps:            Inference steps the student should need.
        output_path:             Where to save the distilled motion module weights.
        latent_shape:            (B, C, F, H, W) — frames at dim 2 (UNetMotionModel convention).
        encoder_hidden_states:   Text embeddings, shape (B*F, seq, hid).
        learning_rate:           AdamW learning rate.
        w_min:                   Minimum skip-step gap.
        w_max:                   Maximum skip-step gap.
    """
    teacher_unet.eval()
    for p in teacher_unet.parameters():
        p.requires_grad_(False)

    # Freeze backbone; only motion_modules stay trainable.
    for name, p in student_unet.named_parameters():
        p.requires_grad_("motion_modules" in name)
    student_unet.train()

    trainable = [p for p in student_unet.parameters() if p.requires_grad]
    print(f"Trainable motion params: {sum(p.numel() for p in trainable):,}")
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)

    num_timesteps = len(alphas_cumprod)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(range(num_train_steps), desc=f"Adapter distill → {target_steps}-step")

    for step in pbar:
        B = latent_shape[0]
        z0 = torch.randn(*latent_shape)
        noise = torch.randn_like(z0)
        t_student, t_teacher = sample_timestep_pairs(B, num_timesteps, w_min, w_max)
        z_student = add_noise(z0, noise, t_student, alphas_cumprod)
        z_teacher = add_noise(z0, noise, t_teacher, alphas_cumprod)

        with torch.no_grad():
            teacher_x0 = predict_x0(teacher_unet, z_teacher, t_teacher,
                                     encoder_hidden_states, alphas_cumprod)

        student_x0 = predict_x0(student_unet, z_student, t_student,
                                 encoder_hidden_states, alphas_cumprod)

        weight = compute_loss_weight(t_teacher, alphas_cumprod)
        loss = consistency_loss(student_x0, teacher_x0, weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    # Save only the motion module weights — that's what gets swapped at inference.
    motion_state = {k: v for k, v in student_unet.state_dict().items()
                    if "motion_modules" in k}
    torch.save(motion_state, output_path)
    print(f"\nSaved distilled {target_steps}-step motion modules → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MotionAdapter distillation")
    parser.add_argument("--steps", type=int, choices=[4, 8], default=8)
    parser.add_argument("--unet", type=str,
                        default="weights/unet_lcm_8step.pt",
                        help="Path to distilled UNet .pt from Phase 1")
    parser.add_argument("--num_train_steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--model_id", type=str,
                        default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--adapter_id", type=str,
                        default="guoyww/animatediff-motion-adapter-v1-5-2")
    args = parser.parse_args()

    unet_path = REPO_ROOT / args.unet

    # Teacher: LCM UNet backbone + original (25-step) motion modules. Fully frozen.
    print(f"Loading teacher pipeline (distilled UNet: {unet_path})")
    teacher_pipe = build_temporal_pipeline(
        unet_weights_path=unet_path,
        model_id=args.model_id,
        adapter_id=args.adapter_id,
    )

    # Student: same backbone weights, fresh copy of motion modules (will be trained).
    print("Loading student pipeline (same backbone, fresh motion modules)")
    student_pipe = build_temporal_pipeline(
        unet_weights_path=unet_path,
        model_id=args.model_id,
        adapter_id=args.adapter_id,
    )

    scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    alphas_cumprod = scheduler.alphas_cumprod
    num_timesteps = len(alphas_cumprod)
    w_max = num_timesteps // args.steps

    # UNetMotionModel input: (batch, channels, frames, height, width).
    # num_frames = sample.shape[2] in the model, so frames is dim 2.
    # Use 16×16 spatial (vs 64×64 inference) — motion modules are purely temporal
    # so spatial resolution doesn't affect what they learn, only training speed.
    # encoder_hidden_states pre-tiled to (B*F, 77, 768) as AnimateDiffPipeline does.
    num_frames = 8
    encoder_hs = torch.randn(1, 77, 768).repeat_interleave(repeats=num_frames, dim=0)
    latent_shape = (1, 4, num_frames, 16, 16)

    out = REPO_ROOT / "weights" / f"motion_adapter_lcm_{args.steps}step.pt"
    run_adapter_distillation(
        teacher_unet=teacher_pipe.unet,
        student_unet=student_pipe.unet,
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
