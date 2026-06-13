# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for _apply_temporal bridge — CPU-only mock (no hardware)."""
import pytest
import torch
from unittest.mock import patch, MagicMock

from animatediff_ttnn.ttlang.temporal_attention_kernel import TemporalAttentionKernel


def _make_kernel(C: int, N: int = 8) -> TemporalAttentionKernel:
    """Create a kernel with random non-zero weights."""
    torch.manual_seed(42)
    kernel = TemporalAttentionKernel(dim=C, num_frames=N, use_ttlang=False)
    kernel.load_weights(
        torch.randn(C, C) * 0.02,
        torch.randn(C, C) * 0.02,
        torch.randn(C, C) * 0.02,
        torch.randn(C, C) * 0.02,
    )
    return kernel


def _make_fake_ttnn_tensors(N: int, S: int, C: int):
    """Create N fake 'TTNN' tensors as plain CPU torch tensors for mocking."""
    return [torch.randn(2, S, C) for _ in range(N)]


def test_apply_temporal_output_count():
    """_apply_temporal returns the same number of tensors as input."""
    from animatediff_ttnn.ttnn_motion_pipeline import _apply_temporal

    N, S, C = 8, 64, 320
    samples = _make_fake_ttnn_tensors(N, S, C)
    kernel = _make_kernel(C)

    mock_device = MagicMock()
    with patch("animatediff_ttnn.ttnn_motion_pipeline.ttnn") as mock_ttnn, \
         patch("animatediff_ttnn.ttnn_motion_pipeline.to_device") as mock_to_device:
        mock_ttnn.to_torch.side_effect = lambda t: t
        mock_to_device.side_effect = lambda t, *a, **kw: t
        for s in samples:
            s.deallocate = MagicMock()

        result = _apply_temporal(samples, [kernel], mock_device, N, C)

    assert len(result) == N


def test_apply_temporal_output_shape():
    """_apply_temporal output tensors have same shape as input."""
    from animatediff_ttnn.ttnn_motion_pipeline import _apply_temporal

    N, S, C = 8, 64, 320
    samples = _make_fake_ttnn_tensors(N, S, C)
    kernel = _make_kernel(C)

    mock_device = MagicMock()
    with patch("animatediff_ttnn.ttnn_motion_pipeline.ttnn") as mock_ttnn, \
         patch("animatediff_ttnn.ttnn_motion_pipeline.to_device") as mock_to_device:
        mock_ttnn.to_torch.side_effect = lambda t: t
        mock_to_device.side_effect = lambda t, *a, **kw: t
        for s in samples:
            s.deallocate = MagicMock()

        result = _apply_temporal(samples, [kernel], mock_device, N, C)

    for i, t in enumerate(result):
        assert t.shape == (2, S, C), f"result[{i}].shape: expected (2, {S}, {C}), got {t.shape}"


def test_apply_temporal_modifies_values():
    """_apply_temporal output differs from input (attention is doing something)."""
    from animatediff_ttnn.ttnn_motion_pipeline import _apply_temporal

    N, S, C = 8, 64, 320
    torch.manual_seed(99)
    samples = _make_fake_ttnn_tensors(N, S, C)
    original_values = [s.clone() for s in samples]
    kernel = _make_kernel(C)

    mock_device = MagicMock()
    with patch("animatediff_ttnn.ttnn_motion_pipeline.ttnn") as mock_ttnn, \
         patch("animatediff_ttnn.ttnn_motion_pipeline.to_device") as mock_to_device:
        mock_ttnn.to_torch.side_effect = lambda t: t
        mock_to_device.side_effect = lambda t, *a, **kw: t
        for s in samples:
            s.deallocate = MagicMock()

        result = _apply_temporal(samples, [kernel], mock_device, N, C)

    changed = sum(not torch.allclose(result[i], original_values[i]) for i in range(N))
    assert changed > 0, "All output tensors are identical to input — attention not applied"


def test_apply_temporal_two_modules_applied_in_sequence():
    """Two kernels are applied sequentially (module 0 then module 1)."""
    from animatediff_ttnn.ttnn_motion_pipeline import _apply_temporal

    N, S, C = 4, 32, 320
    torch.manual_seed(7)
    original = _make_fake_ttnn_tensors(N, S, C)

    k0 = _make_kernel(C)
    k1 = TemporalAttentionKernel(dim=C, num_frames=N, use_ttlang=False)
    torch.manual_seed(999)
    k1.load_weights(
        torch.randn(C, C) * 0.02,
        torch.randn(C, C) * 0.02,
        torch.randn(C, C) * 0.02,
        torch.randn(C, C) * 0.02,
    )

    mock_device = MagicMock()

    # Apply [k0, k1] — two modules
    samples2 = [s.clone() for s in original]
    with patch("animatediff_ttnn.ttnn_motion_pipeline.ttnn") as mock_ttnn, \
         patch("animatediff_ttnn.ttnn_motion_pipeline.to_device") as mock_to_device:
        mock_ttnn.to_torch.side_effect = lambda t: t
        mock_to_device.side_effect = lambda t, *a, **kw: t
        for s in samples2:
            s.deallocate = MagicMock()
        result_two = _apply_temporal(samples2, [k0, k1], mock_device, N, C)

    # Apply just k0 (single module)
    samples1 = [s.clone() for s in original]
    with patch("animatediff_ttnn.ttnn_motion_pipeline.ttnn") as mock_ttnn, \
         patch("animatediff_ttnn.ttnn_motion_pipeline.to_device") as mock_to_device:
        mock_ttnn.to_torch.side_effect = lambda t: t
        mock_to_device.side_effect = lambda t, *a, **kw: t
        for s in samples1:
            s.deallocate = MagicMock()
        result_one = _apply_temporal(samples1, [k0], mock_device, N, C)

    diffs = sum(not torch.allclose(result_two[i], result_one[i]) for i in range(N))
    assert diffs > 0, "Two-module and one-module results are identical — second module not applied"
