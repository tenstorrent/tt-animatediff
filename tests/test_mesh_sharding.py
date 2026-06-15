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


def test_num_frames_not_divisible_raises_temporal():
    """generate_frames_temporal raises ValueError if num_frames % num_chips != 0."""
    ttnn_mock = _make_ttnn_mock()
    # Make isinstance(device, ttnn.MeshDevice) return True
    ttnn_mock.MeshDevice = type("MeshDevice", (), {})
    device = ttnn_mock.MeshDevice()
    device.get_num_devices = MagicMock(return_value=4)

    with patch.dict(sys.modules, {"ttnn": ttnn_mock}):
        from animatediff_ttnn.temporal_attention import generate_frames_temporal
        with pytest.raises(ValueError, match="num_frames.*divisible"):
            generate_frames_temporal(
                device=device,
                ttnn_model=MagicMock(),
                ttnn_vae=MagicMock(),
                config=MagicMock(),
                torch_time_proj=MagicMock(),
                text_embeddings=torch.zeros(2, 96, 768),
                num_frames=7,   # 7 % 4 != 0
                num_steps=1,
            )


def test_num_frames_divisible_does_not_raise_guard():
    """generate_frames_temporal does NOT raise for num_frames divisible by num_chips."""
    ttnn_mock = _make_ttnn_mock()
    ttnn_mock.MeshDevice = type("MeshDevice", (), {})
    device = ttnn_mock.MeshDevice()
    device.get_num_devices = MagicMock(return_value=4)

    with patch.dict(sys.modules, {"ttnn": ttnn_mock}):
        from animatediff_ttnn.temporal_attention import generate_frames_temporal
        # Should NOT raise ValueError for the divisibility guard.
        # It will raise something else (ttnn call fails on mock), but not our guard.
        try:
            generate_frames_temporal(
                device=device,
                ttnn_model=MagicMock(),
                ttnn_vae=MagicMock(),
                config=MagicMock(),
                torch_time_proj=MagicMock(),
                text_embeddings=torch.zeros(2, 96, 768),
                num_frames=8,   # 8 % 4 == 0
                num_steps=1,
            )
        except ValueError as e:
            assert "divisible" not in str(e), f"Guard should not fire: {e}"
        except Exception:
            pass  # other errors from mocked ttnn are expected


def test_gather_frames_single_batch_per_frame():
    """gather_frames_from_device with batch_per_frame=1 splits [N,C,H,W] into N [1,C,H,W]."""
    from animatediff_ttnn.ttnn_pipeline import gather_frames_from_device

    N, lh, lw = 4, 8, 8
    stacked = torch.randn(N, 4, lh, lw)   # N frames, NOT CFG-doubled
    ttnn_mock = _make_ttnn_mock()
    ttnn_mock.to_torch.return_value = stacked
    device = _make_mesh_device(num_chips=4)

    with patch.dict(sys.modules, {"ttnn": ttnn_mock}):
        result = gather_frames_from_device(MagicMock(), device, num_frames=N, batch_per_frame=1)

    assert len(result) == N
    for t in result:
        assert t.shape == (1, 4, lh, lw)


def test_forward_unet_staged_guard_raises():
    """forward_unet_staged raises ValueError if num_frames % num_chips != 0."""
    import importlib.util
    import os

    ttnn_mock = _make_ttnn_mock()
    ttnn_mock.MeshDevice = type("MeshDevice", (), {})
    device = ttnn_mock.MeshDevice()
    device.get_num_devices = MagicMock(return_value=4)

    # Load ttnn_motion_pipeline.py directly by file path to bypass
    # animatediff_ttnn/__init__.py (which imports pipeline.py → diffusers
    # AnimateDiffPipeline → CLIPImageProcessor, unavailable in CI).
    module_path = os.path.join(
        os.path.dirname(__file__), "..", "animatediff_ttnn", "ttnn_motion_pipeline.py"
    )
    spec = importlib.util.spec_from_file_location("ttnn_motion_pipeline_direct", module_path)
    mod = importlib.util.module_from_spec(spec)

    # Stub out the module-level ttnn import and to_device so the module loads cleanly.
    stub_modules = {
        "ttnn": ttnn_mock,
        "animatediff_ttnn.ttnn_pipeline": MagicMock(),
    }
    with patch.dict(sys.modules, stub_modules):
        spec.loader.exec_module(mod)

    forward_unet_staged = mod.forward_unet_staged

    with patch.dict(sys.modules, {"ttnn": ttnn_mock}):
        # Patch the module-level ttnn reference used by the guard inside forward_unet_staged.
        mod.ttnn = ttnn_mock
        with pytest.raises(ValueError, match="num_frames.*divisible"):
            forward_unet_staged(
                ttnn_model=MagicMock(),
                frame_samples=[MagicMock()] * 5,  # 5 % 4 != 0
                timestep=MagicMock(),
                encoder_hidden_states=MagicMock(),
                config=MagicMock(),
                temporal_kernels={},
                device=device,
                num_frames=5,
            )
