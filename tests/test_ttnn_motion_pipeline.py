# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for _apply_temporal bridge — CPU-only mock (no hardware).

_apply_temporal now calls AnimateDiffTransformer3D.forward(stacked, num_frames=N).
Tests use MagicMock objects as fake TTNN tensors, with ttnn.to_torch returning
plain CPU tensors. AnimateDiffTransformer3D is replaced by tiny stub modules.
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock


class _IdentityModule(nn.Module):
    """Drop-in for AnimateDiffTransformer3D that returns the input unchanged."""

    def forward(self, x, num_frames=None):
        return x


class _ShiftModule(nn.Module):
    """Drop-in that adds a constant offset so output != input."""

    def __init__(self, shift: float = 0.1):
        super().__init__()
        self.shift = shift

    def forward(self, x, num_frames=None):
        return x + self.shift


def _make_fake_ttnn_tensors(N: int, spatial_h: int, spatial_w: int, C: int):
    """Create N MagicMock 'TTNN' tensors backed by plain CPU tensors.

    ttnn.to_torch will be mocked to return the underlying .cpu_data attribute.
    """
    S = spatial_h * spatial_w
    fakes = []
    for _ in range(N):
        m = MagicMock()
        m.cpu_data = torch.randn(1, 1, 2 * S, C)
        m.dtype = torch.float32   # MagicMock supports attribute assignment
        m.deallocate = MagicMock()
        fakes.append(m)
    return fakes


def _run_apply_temporal(fake_samples, real_tensors, module_list, N, C, H, W, alpha=1.0):
    """Call _apply_temporal with fully mocked ttnn/to_device.

    fake_samples: MagicMock objects passed as samples (each has .cpu_data torch tensor)
    real_tensors: unused — kept for call-site compatibility
    """
    from animatediff_ttnn.ttnn_motion_pipeline import _apply_temporal

    mock_device = MagicMock()

    def mock_to_torch(t):
        # Works for both individual fake samples and the concatenated mock.
        return t.cpu_data

    def mock_concat(tensors, dim=0):
        # Concatenate the underlying cpu_data tensors, wrap in a mock.
        joined = torch.cat([t.cpu_data for t in tensors], dim=dim)
        m = MagicMock()
        m.cpu_data = joined
        m.deallocate = MagicMock()
        return m

    def mock_to_device_fn(t, *a, **kw):
        # Return the tensor directly. The output of to_device in the H→D path
        # is stored in `out` and returned — it is never .deallocate()d by the
        # production code (only the original samples[] are deallocated).
        return t

    def mock_split(tensor, num_splits, dim=0):
        # tensor is our MagicMock wrapper; extract the actual torch data.
        actual = getattr(tensor, "torch_data", tensor)
        chunk_size = actual.shape[dim] // num_splits
        # Return plain tensors — to_memory_config mock will wrap them.
        return list(torch.split(actual, chunk_size, dim=dim))

    def mock_to_memory_config(t, config):
        # Pass-through: in tests, memory configs are not meaningful.
        # Return whatever we got (torch.Tensor or MagicMock).
        return t

    with patch("animatediff_ttnn.ttnn_motion_pipeline.ttnn") as mock_ttnn, \
         patch("animatediff_ttnn.ttnn_motion_pipeline.to_device") as mock_to_device:
        mock_ttnn.to_torch.side_effect = mock_to_torch
        mock_ttnn.concat.side_effect = mock_concat
        mock_ttnn.split.side_effect = mock_split
        mock_ttnn.to_memory_config.side_effect = mock_to_memory_config
        # get_memory_config → not sharded
        mc = MagicMock()
        mc.is_sharded.return_value = False
        mock_ttnn.get_memory_config.return_value = mc
        # to_device: wrap in a mock that supports .deallocate()
        mock_to_device.side_effect = mock_to_device_fn

        result = _apply_temporal(fake_samples, module_list, mock_device, N, C, H, W, alpha)
    return result


def test_apply_temporal_output_count():
    """_apply_temporal returns the same number of tensors as input."""
    N, H, W, C = 8, 8, 8, 320
    fakes = _make_fake_ttnn_tensors(N, H, W, C)
    real = [f.cpu_data for f in fakes]
    result = _run_apply_temporal(fakes, real, [_IdentityModule()], N, C, H, W)
    assert len(result) == N


def test_apply_temporal_output_shape():
    """_apply_temporal output tensors have the same [1,1,2*S,C] shape as input."""
    N, H, W, C = 8, 8, 8, 320
    S = H * W
    fakes = _make_fake_ttnn_tensors(N, H, W, C)
    real = [f.cpu_data for f in fakes]
    result = _run_apply_temporal(fakes, real, [_IdentityModule()], N, C, H, W)
    for i, t in enumerate(result):
        assert t.shape == (1, 1, 2 * S, C), (
            f"result[{i}].shape: expected (1, 1, {2*S}, {C}), got {t.shape}"
        )


def test_apply_temporal_identity_module_no_change():
    """An identity module should produce output equal to input (residual path)."""
    N, H, W, C = 4, 8, 8, 320
    torch.manual_seed(0)
    fakes = _make_fake_ttnn_tensors(N, H, W, C)
    originals = [f.cpu_data.clone() for f in fakes]
    result = _run_apply_temporal(fakes, originals, [_IdentityModule()], N, C, H, W)
    for i in range(N):
        assert torch.allclose(result[i], originals[i]), (
            f"result[{i}] changed with identity module"
        )


def test_apply_temporal_shift_module_modifies_values():
    """A non-identity module must produce output that differs from input."""
    N, H, W, C = 8, 8, 8, 320
    torch.manual_seed(1)
    fakes = _make_fake_ttnn_tensors(N, H, W, C)
    originals = [f.cpu_data.clone() for f in fakes]
    result = _run_apply_temporal(fakes, originals, [_ShiftModule(0.5)], N, C, H, W)
    changed = sum(not torch.allclose(result[i], originals[i]) for i in range(N))
    assert changed > 0, "All output tensors identical to input — module not applied"


def test_apply_temporal_two_modules_applied_in_sequence():
    """Two modules applied sequentially produce different output than one module alone."""
    N, H, W, C = 4, 8, 8, 320
    torch.manual_seed(7)

    fakes_one = _make_fake_ttnn_tensors(N, H, W, C)
    fakes_two = [MagicMock() for _ in range(N)]
    for i, f in enumerate(fakes_two):
        f.cpu_data = fakes_one[i].cpu_data.clone()
        f.dtype = torch.float32
        f.deallocate = MagicMock()

    result_one = _run_apply_temporal(fakes_one, None, [_ShiftModule(0.1)], N, C, H, W)
    result_two = _run_apply_temporal(fakes_two, None, [_ShiftModule(0.1), _ShiftModule(0.2)], N, C, H, W)

    diffs = sum(not torch.allclose(result_two[i], result_one[i]) for i in range(N))
    assert diffs > 0, "Two-module and one-module results are identical — second module not applied"


def test_apply_temporal_alpha_zero_is_noop():
    """injection_alpha=0.0 should return input unchanged regardless of module."""
    N, H, W, C = 4, 8, 8, 320
    torch.manual_seed(2)
    fakes = _make_fake_ttnn_tensors(N, H, W, C)
    originals = [f.cpu_data.clone() for f in fakes]
    result = _run_apply_temporal(fakes, originals, [_ShiftModule(10.0)], N, C, H, W, alpha=0.0)
    for i in range(N):
        assert torch.allclose(result[i], originals[i]), (
            f"alpha=0.0 should be a no-op but result[{i}] changed"
        )


def test_apply_temporal_alpha_half_blend():
    """injection_alpha=0.5 blends original and attended 50/50."""
    N, H, W, C = 4, 8, 8, 320
    torch.manual_seed(3)
    fakes = _make_fake_ttnn_tensors(N, H, W, C)
    originals = [f.cpu_data.clone() for f in fakes]

    # With shift=2.0 and alpha=0.5, result should be original + 1.0 (half of 2.0 shift)
    result = _run_apply_temporal(fakes, originals, [_ShiftModule(2.0)], N, C, H, W, alpha=0.5)
    for i in range(N):
        expected = originals[i] + 1.0
        assert torch.allclose(result[i], expected, atol=1e-5), (
            f"result[{i}] alpha-blend mismatch"
        )
