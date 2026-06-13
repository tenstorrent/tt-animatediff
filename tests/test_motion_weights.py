# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for MotionAdapter weight loading — no hardware, HuggingFace cache only."""
import pytest
import torch


def test_load_motion_kernels_returns_all_keys():
    """load_motion_kernels returns dict with all 7 injection point keys."""
    from animatediff_ttnn.motion_weights import load_motion_kernels
    kernels = load_motion_kernels()
    expected_keys = {"down0", "down1", "down2", "mid", "up0", "up1", "up2"}
    assert set(kernels.keys()) == expected_keys


def test_load_motion_kernels_list_lengths():
    """down keys have 2 kernels each; up keys have 3 each; mid has 1.

    Verified against the actual guoyww/animatediff-motion-adapter-v1-5-2
    state dict: down_blocks 0-2 have 2 motion_modules each, up_blocks 0-2
    have 3 motion_modules each, mid_block has 1.
    """
    from animatediff_ttnn.motion_weights import load_motion_kernels
    kernels = load_motion_kernels()
    for key in ("down0", "down1", "down2"):
        assert len(kernels[key]) == 2, f"{key} should have 2 motion modules"
    for key in ("up0", "up1", "up2"):
        assert len(kernels[key]) == 3, f"{key} should have 3 motion modules"
    assert len(kernels["mid"]) == 1, "mid should have 1 motion module"


def test_load_motion_kernels_weights_loaded():
    """Every kernel has non-None w_q, w_k, w_v, w_o."""
    from animatediff_ttnn.motion_weights import load_motion_kernels
    kernels = load_motion_kernels()
    for key, kernel_list in kernels.items():
        for j, k in enumerate(kernel_list):
            assert k.w_q is not None, f"{key}[{j}].w_q is None"
            assert k.w_k is not None, f"{key}[{j}].w_k is None"
            assert k.w_v is not None, f"{key}[{j}].w_v is None"
            assert k.w_o is not None, f"{key}[{j}].w_o is None"


def test_load_motion_kernels_channel_dims():
    """Kernel dims match expected channel sizes for each injection point.

    Channel dims verified from actual to_q.weight shapes in the
    guoyww/animatediff-motion-adapter-v1-5-2 checkpoint:
      down_blocks 0→320, 1→640, 2→1280
      mid_block       →1280
      up_blocks  0→1280, 1→1280, 2→640
    """
    from animatediff_ttnn.motion_weights import load_motion_kernels
    kernels = load_motion_kernels()
    expected_dims = {"down0": 320, "down1": 640, "down2": 1280,
                     "mid": 1280, "up0": 1280, "up1": 1280, "up2": 640}
    for key, dim in expected_dims.items():
        for j, k in enumerate(kernels[key]):
            assert k.dim == dim, f"{key}[{j}].dim: expected {dim}, got {k.dim}"
            assert k.w_q.shape == (dim, dim), f"{key}[{j}].w_q.shape wrong: {k.w_q.shape}"


def test_load_motion_kernels_weight_dtype():
    """All weights are float32 (load_weights converts on load)."""
    from animatediff_ttnn.motion_weights import load_motion_kernels
    kernels = load_motion_kernels()
    for key, kernel_list in kernels.items():
        for j, k in enumerate(kernel_list):
            assert k.w_q.dtype == torch.float32, f"{key}[{j}].w_q dtype: {k.w_q.dtype}"


def test_load_motion_kernels_different_modules_have_different_weights():
    """Adjacent motion modules per block have distinct weights (not copies)."""
    from animatediff_ttnn.motion_weights import load_motion_kernels
    kernels = load_motion_kernels()
    # down blocks have 2 modules each — compare module 0 vs 1
    for key in ("down0", "down1", "down2"):
        k0, k1 = kernels[key][0], kernels[key][1]
        assert not torch.allclose(k0.w_q, k1.w_q), f"{key}: modules 0 and 1 have identical w_q"
    # up blocks have 3 modules each — compare module 0 vs 1
    for key in ("up0", "up1", "up2"):
        k0, k1 = kernels[key][0], kernels[key][1]
        assert not torch.allclose(k0.w_q, k1.w_q), f"{key}: modules 0 and 1 have identical w_q"
