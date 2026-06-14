# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for MotionAdapter weight loading — no hardware, HuggingFace cache only."""
import pytest
import torch


def test_load_motion_modules_returns_all_keys():
    """load_motion_modules returns dict with all 7 injection point keys."""
    from animatediff_ttnn.motion_weights import load_motion_modules
    modules = load_motion_modules()
    expected_keys = {"down0", "down1", "down2", "mid", "up0", "up1", "up2"}
    assert set(modules.keys()) == expected_keys


def test_load_motion_modules_list_lengths():
    """down keys have 2 modules each; up keys have 3 each; mid has 1.

    Verified against the actual guoyww/animatediff-motion-adapter-v1-5-2
    state dict: down_blocks 0-2 have 2 motion_modules each, up_blocks 0-2
    have 3 motion_modules each, mid_block has 1.
    """
    from animatediff_ttnn.motion_weights import load_motion_modules
    modules = load_motion_modules()
    for key in ("down0", "down1", "down2"):
        assert len(modules[key]) == 2, f"{key} should have 2 motion modules"
    for key in ("up0", "up1", "up2"):
        assert len(modules[key]) == 3, f"{key} should have 3 motion modules"
    assert len(modules["mid"]) == 1, "mid should have 1 motion module"


def test_load_motion_modules_are_transformer3d():
    """Every module is an AnimateDiffTransformer3D (has the forward method and num_attention_heads)."""
    from animatediff_ttnn.motion_weights import load_motion_modules
    modules = load_motion_modules()
    for key, module_list in modules.items():
        for j, m in enumerate(module_list):
            assert hasattr(m, "forward"), f"{key}[{j}] missing .forward"
            assert hasattr(m, "num_attention_heads"), f"{key}[{j}] missing .num_attention_heads"


def test_load_motion_modules_eval_mode():
    """All modules are in eval mode (training=False)."""
    from animatediff_ttnn.motion_weights import load_motion_modules
    modules = load_motion_modules()
    for key, module_list in modules.items():
        for j, m in enumerate(module_list):
            assert not m.training, f"{key}[{j}] is still in training mode"


def test_load_motion_modules_different_modules_have_different_weights():
    """Adjacent motion modules per block have distinct weights (not copies)."""
    from animatediff_ttnn.motion_weights import load_motion_modules
    modules = load_motion_modules()
    # down blocks have 2 modules each — compare proj_in weights of module 0 vs 1
    for key in ("down0", "down1", "down2"):
        m0, m1 = modules[key][0], modules[key][1]
        w0 = m0.proj_in.weight.data
        w1 = m1.proj_in.weight.data
        assert not torch.allclose(w0, w1), f"{key}: modules 0 and 1 have identical proj_in weights"


def test_get_injection_point_info_dims():
    """get_injection_point_info returns correct spatial dims for all 7 keys."""
    from animatediff_ttnn.motion_weights import get_injection_point_info
    expected = {
        "down0": (320, 32, 32),
        "down1": (640, 16, 16),
        "down2": (1280, 8, 8),
        "mid":   (1280, 8, 8),
        "up0":   (1280, 16, 16),
        "up1":   (1280, 32, 32),
        "up2":   (640, 64, 64),
    }
    for key, (dim, h, w) in expected.items():
        ip = get_injection_point_info(key)
        assert ip.dim == dim, f"{key} dim: expected {dim}, got {ip.dim}"
        assert ip.spatial_h == h, f"{key} spatial_h: expected {h}, got {ip.spatial_h}"
        assert ip.spatial_w == w, f"{key} spatial_w: expected {w}, got {ip.spatial_w}"
