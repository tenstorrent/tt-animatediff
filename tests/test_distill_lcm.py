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
