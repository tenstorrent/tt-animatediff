# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for LCM consistency loss and timestep sampling."""

import torch
import pytest
from scripts.distill_lcm import (
    consistency_loss,
    sample_timestep_pairs,
    compute_loss_weight,
)


def test_consistency_loss_zero_when_predictions_match():
    """Loss must be zero when student and teacher predict identically."""
    B, C, H, W = 2, 4, 8, 8
    student_pred = torch.randn(B, C, H, W)
    teacher_pred = student_pred.clone()
    loss = consistency_loss(student_pred, teacher_pred, weight=torch.ones(B))
    assert loss.item() == 0.0


def test_consistency_loss_positive_when_predictions_differ():
    B, C, H, W = 2, 4, 8, 8
    student_pred = torch.randn(B, C, H, W)
    teacher_pred = torch.randn(B, C, H, W)
    loss = consistency_loss(student_pred, teacher_pred, weight=torch.ones(B))
    assert loss.item() > 0


def test_consistency_loss_scales_with_weight():
    B, C, H, W = 1, 4, 8, 8
    student_pred = torch.zeros(B, C, H, W)
    teacher_pred = torch.ones(B, C, H, W)
    loss_w1 = consistency_loss(student_pred, teacher_pred, weight=torch.tensor([1.0]))
    loss_w2 = consistency_loss(student_pred, teacher_pred, weight=torch.tensor([2.0]))
    assert abs(loss_w2.item() / loss_w1.item() - 2.0) < 1e-5


def test_sample_timestep_pairs_shape():
    """sample_timestep_pairs returns (t_student, t_teacher) both of length batch_size."""
    t_student, t_teacher = sample_timestep_pairs(
        batch_size=8, num_timesteps=1000, w_min=2, w_max=10
    )
    assert t_student.shape == (8,)
    assert t_teacher.shape == (8,)


def test_sample_timestep_pairs_gap_in_range():
    """Gap between t_teacher and t_student must always be in [w_min, w_max]."""
    for _ in range(50):
        t_student, t_teacher = sample_timestep_pairs(
            batch_size=16, num_timesteps=1000, w_min=2, w_max=10
        )
        gaps = (t_teacher - t_student).float()
        assert (gaps >= 2).all()
        assert (gaps <= 10).all()
        assert (t_teacher < 1000).all(), "t_teacher must be a valid timestep index"
        assert (t_student >= 0).all(), "t_student must be non-negative"


def test_compute_loss_weight_returns_positive_tensor():
    timesteps = torch.randint(0, 1000, (8,))
    alphas_cumprod = torch.rand(1000)
    w = compute_loss_weight(timesteps, alphas_cumprod)
    assert w.shape == (8,)
    assert (w > 0).all()
    assert (w <= 1.0).all(), "loss weight must not exceed 1.0"


def test_add_noise_output_shape():
    """add_noise returns a tensor with the same shape as the input."""
    from scripts.distill_lcm import add_noise
    latent = torch.randn(2, 4, 8, 8)
    noise = torch.randn_like(latent)
    alphas_cumprod = torch.rand(1000)
    timesteps = torch.randint(0, 1000, (2,))
    noisy = add_noise(latent, noise, timesteps, alphas_cumprod)
    assert noisy.shape == latent.shape


def test_predict_x0_output_shape():
    """predict_x0 returns a clean-image estimate with same spatial shape as input."""
    from scripts.distill_lcm import predict_x0
    class TinyUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, sample, timestep, encoder_hidden_states, return_dict=False):
            class R:
                pass
            r = R()
            r.sample = torch.zeros_like(sample)
            return r
    unet = TinyUNet()
    noisy = torch.randn(1, 4, 8, 8)
    timesteps = torch.tensor([500])
    encoder_hs = torch.randn(1, 77, 768)
    alphas_cumprod = torch.rand(1000)
    x0 = predict_x0(unet, noisy, timesteps, encoder_hs, alphas_cumprod)
    assert x0.shape == noisy.shape


def test_run_distillation_saves_checkpoint(tmp_path):
    """run_distillation completes 2 steps and saves a .pt file."""
    from scripts.distill_lcm import run_distillation

    class TinyUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)
            self.config = type("C", (), {"in_channels": 4})()
        def forward(self, sample, timestep, encoder_hidden_states, return_dict=False):
            class R:
                pass
            r = R()
            r.sample = torch.zeros_like(sample)
            return r

    unet = TinyUNet()
    alphas_cumprod = torch.rand(1000)
    out_path = tmp_path / "unet_test.pt"
    run_distillation(
        teacher_unet=unet,
        alphas_cumprod=alphas_cumprod,
        num_train_steps=2,
        target_steps=4,
        output_path=out_path,
        latent_shape=(1, 4, 8, 8),
        encoder_hidden_states=torch.randn(1, 77, 768),
        learning_rate=1e-4,
        w_min=2,
        w_max=4,
    )
    assert out_path.exists()
    state = torch.load(out_path, map_location="cpu", weights_only=True)
    assert "linear.weight" in state
