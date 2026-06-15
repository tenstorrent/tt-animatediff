# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Unit tests for shard_frames_to_device and gather_frames_from_device.

Run without Blackhole hardware — ttnn is fully mocked.
"""
import sys
import torch
import pytest
from unittest.mock import MagicMock, patch


def _make_mesh_device(num_chips: int):
    """Return a MagicMock that looks like a ttnn.MeshDevice with num_chips chips."""
    dev = MagicMock()
    dev.__class__.__name__ = "MeshDevice"
    dev.get_num_devices.return_value = num_chips
    return dev


def _make_ttnn_mock():
    """Return a MagicMock for the ttnn module with the minimal surface area used."""
    m = MagicMock()
    m.MeshDevice = MagicMock  # isinstance check uses the class
    # ShardTensorToMesh and ConcatMeshToTensor are called as constructors
    m.ShardTensorToMesh = MagicMock(return_value=MagicMock())
    m.ConcatMeshToTensor = MagicMock(return_value=MagicMock())
    m.bfloat16 = "bfloat16"
    m.TILE_LAYOUT = "TILE_LAYOUT"
    # from_torch returns a sentinel TTNN tensor
    m.from_torch.return_value = MagicMock(name="ttnn_tensor")
    # to_torch returns the stacked CPU tensor (simulate gather)
    return m


def test_shard_frames_to_device_shape():
    """shard_frames_to_device stacks N [2,4,lh,lw] frames and sends [2N,4,lh,lw]."""
    from animatediff_ttnn.ttnn_pipeline import shard_frames_to_device

    N, lh, lw = 4, 8, 8
    frames = [torch.randn(2, 4, lh, lw) for _ in range(N)]
    device = _make_mesh_device(num_chips=2)
    ttnn_mock = _make_ttnn_mock()

    # The helpers use lazy `import ttnn` inside the function body, so we
    # intercept via sys.modules rather than patching a module-level attribute.
    with patch.dict(sys.modules, {"ttnn": ttnn_mock}):
        shard_frames_to_device(frames, device)

    # from_torch called once with the stacked tensor
    assert ttnn_mock.from_torch.call_count == 1
    actual_tensor = ttnn_mock.from_torch.call_args[0][0]
    assert actual_tensor.shape == (2 * N, 4, lh, lw)
    # ShardTensorToMesh used as mesh_mapper
    ttnn_mock.ShardTensorToMesh.assert_called_once_with(device, dim=0)
    call_kwargs = ttnn_mock.from_torch.call_args[1]
    assert "mesh_mapper" in call_kwargs, "mesh_mapper must be passed as kwarg to from_torch"


def test_shard_frames_to_device_single_chip():
    """shard_frames_to_device on a 1-chip device still uses ShardTensorToMesh."""
    from animatediff_ttnn.ttnn_pipeline import shard_frames_to_device

    frames = [torch.randn(2, 4, 8, 8)]
    device = _make_mesh_device(num_chips=1)
    ttnn_mock = _make_ttnn_mock()

    with patch.dict(sys.modules, {"ttnn": ttnn_mock}):
        shard_frames_to_device(frames, device)

    ttnn_mock.ShardTensorToMesh.assert_called_once_with(device, dim=0)


def test_gather_frames_from_device_splits_correctly():
    """gather_frames_from_device splits [2N,4,lh,lw] into N tensors of [2,4,lh,lw]."""
    from animatediff_ttnn.ttnn_pipeline import gather_frames_from_device

    N, lh, lw = 4, 8, 8
    stacked = torch.randn(2 * N, 4, lh, lw)
    ttnn_mock = _make_ttnn_mock()
    ttnn_mock.to_torch.return_value = stacked

    device = _make_mesh_device(num_chips=2)
    fake_ttnn_tensor = MagicMock()

    with patch.dict(sys.modules, {"ttnn": ttnn_mock}):
        result = gather_frames_from_device(fake_ttnn_tensor, device, num_frames=N)

    assert len(result) == N
    for t in result:
        assert t.shape == (2, 4, lh, lw)
    ttnn_mock.ConcatMeshToTensor.assert_called_once_with(device, dim=0)


def test_gather_frames_from_device_values():
    """gather_frames_from_device preserves tensor values after split."""
    from animatediff_ttnn.ttnn_pipeline import gather_frames_from_device

    N = 3
    stacked = torch.arange(N * 2 * 4 * 8 * 8, dtype=torch.float32).reshape(2 * N, 4, 8, 8)
    ttnn_mock = _make_ttnn_mock()
    ttnn_mock.to_torch.return_value = stacked
    device = _make_mesh_device(num_chips=1)

    with patch.dict(sys.modules, {"ttnn": ttnn_mock}):
        result = gather_frames_from_device(MagicMock(), device, num_frames=N)

    assert torch.allclose(result[0], stacked[0:2])
    assert torch.allclose(result[1], stacked[2:4])
    assert torch.allclose(result[2], stacked[4:6])
