# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for MotionAdapter consistency distillation."""

import torch
import pytest
from pathlib import Path
from scripts.distill_motion_adapter import (
    build_temporal_pipeline,
    run_adapter_distillation,
)


def _tiny_unet_state():
    """Return a minimal state dict that looks like a saved LCM UNet."""
    import torch.nn as nn
    m = nn.Linear(4, 4)
    return m.state_dict()


def test_build_temporal_pipeline_returns_pipeline():
    """build_temporal_pipeline returns an object with a unet and motion_adapter."""
    from unittest.mock import MagicMock, patch
    mock_pipe = MagicMock()
    mock_pipe.unet = MagicMock()
    mock_pipe.motion_adapter = MagicMock()
    with patch("scripts.distill_motion_adapter.AnimateDiffPipeline") as MockPipeline:
        MockPipeline.from_pretrained.return_value = mock_pipe
        pipe = build_temporal_pipeline(
            unet_weights_path=None,
            model_id="CompVis/stable-diffusion-v1-4",
            adapter_id="guoyww/animatediff-motion-adapter-v1-5-2",
            load_unet_weights=False,
        )
    assert hasattr(pipe, "unet")
    assert hasattr(pipe, "motion_adapter")


def test_run_adapter_distillation_saves_checkpoint(tmp_path):
    """run_adapter_distillation completes 2 steps and saves motion_modules weights."""

    class TinyUNet(torch.nn.Module):
        """Minimal stub with one motion_modules param so we can verify the saved keys."""
        def __init__(self):
            super().__init__()
            # Named to match the "motion_modules" filter used when saving weights.
            self.motion_modules = torch.nn.Linear(4, 4)

        def forward(self, sample, timestep, encoder_hidden_states, return_dict=True):
            class R:
                pass
            r = R()
            # Route through motion_modules so gradients flow in the student.
            # Takes the first channel scalar through the linear and adds it back.
            delta = self.motion_modules(sample.flatten()[:4]).mean() * 0.0
            r.sample = sample + delta
            return r

    out_path = tmp_path / "adapter_test.pt"
    alphas_cumprod = torch.rand(1000)

    # 5D latent: (B, C, F, H, W) — frames at dim 2 per UNetMotionModel convention.
    run_adapter_distillation(
        teacher_unet=TinyUNet(),
        student_unet=TinyUNet(),
        alphas_cumprod=alphas_cumprod,
        num_train_steps=2,
        target_steps=8,
        output_path=out_path,
        latent_shape=(1, 4, 2, 4, 4),
        encoder_hidden_states=torch.randn(2, 77, 768),
        learning_rate=1e-4,
        w_min=2,
        w_max=4,
    )
    assert out_path.exists()
    state = torch.load(out_path, map_location="cpu", weights_only=True)
    # Only motion_modules keys are saved.
    assert all("motion_modules" in k for k in state)
    assert len(state) > 0
