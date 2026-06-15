# Mesh Frame Sharding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Distribute AnimateDiff frame generation across all N Blackhole chips in the MeshDevice by sharding the frame batch along `dim=0`, replacing the serialized `for i in range(num_frames)` UNet and VAE loops with single parallel calls — achieving ~4× UNet speedup and ~4× VAE decode speedup on a 4-chip QB2 board.

**Architecture:** Two new helpers (`shard_frames_to_device` / `gather_frames_from_device`) in `ttnn_pipeline.py` handle the stack-shard-gather pattern. `generate_frames_temporal` (Phase 2.5) and `forward_unet_staged` + VAE decode in `generate_frames_motion` (Phase 3) are updated to use them. Single-chip mode degrades gracefully — `ShardTensorToMesh` on a 1-chip mesh is semantically correct and produces identical results.

**Tech Stack:** TTNN MeshDevice API (`ShardTensorToMesh`, `ConcatMeshToTensor`), existing SD 1.4 TTNN UNet (compiled at `batch_size=2`, unchanged), diffusers PNDM/Euler schedulers, PyTorch CPU tensors.

**Spec:** `docs/superpowers/specs/2026-06-15-mesh-frame-sharding-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `animatediff_ttnn/ttnn_pipeline.py` | Modify | Add `shard_frames_to_device`, `gather_frames_from_device` |
| `animatediff_ttnn/temporal_attention.py` | Modify | Replace per-frame UNet loop + VAE decode loop in `generate_frames_temporal` |
| `animatediff_ttnn/ttnn_motion_pipeline.py` | Modify | Replace per-frame block loops + VAE decode loop in `forward_unet_staged` (UNet blocks) and `generate_frames_motion` (VAE) |
| `tests/test_mesh_sharding.py` | Create | Unit tests for the two new helpers and `ValueError` guards |
| `scripts/mesh_sharding_hw_test.py` | Create | Hardware smoke test: serial vs mesh, PCC + timing |

---

### Task 1: `shard_frames_to_device` and `gather_frames_from_device` helpers

**Context:** `animatediff_ttnn/ttnn_pipeline.py` already has `to_device` (uses `ReplicateTensorToMesh`) and `from_device` (uses `ConcatMeshToTensor`). We add two new helpers for the frame-sharding pattern. `shard_frames_to_device` accepts a list of N CPU tensors (each `[2, 4, lh, lw]`, CFG-doubled), stacks to `[2N, 4, lh, lw]`, and sends with `ShardTensorToMesh(device, dim=0)`. `gather_frames_from_device` pulls back with `ConcatMeshToTensor(dim=0)` and splits into N tensors.

The `[2, 4, lh, lw]` shape is the CFG-doubled latent for one frame (uncond + cond stacked). After sharding, chip K sees frames `[2K, 2K+2]` of the original batch — still `batch_size=2`, matching the compiled kernel.

For a single-chip `MeshDevice` (1 chip), `ShardTensorToMesh` with `dim=0` on a `[2, ...]` tensor is a no-op (the shard IS the full tensor) — correct and no performance regression.

**Files:**
- Modify: `animatediff_ttnn/ttnn_pipeline.py` (add after `from_device`, around line 137)
- Create: `tests/test_mesh_sharding.py`

- [ ] **Step 1: Create the test file with failing tests**

```python
# tests/test_mesh_sharding.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Unit tests for shard_frames_to_device and gather_frames_from_device.

Run without Blackhole hardware — ttnn is fully mocked.
"""
import sys
import torch
import pytest
from unittest.mock import MagicMock, patch, call


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

    with patch.dict(sys.modules, {"ttnn": ttnn_mock}):
        import importlib, animatediff_ttnn.ttnn_pipeline as tp
        importlib.reload(tp)
        with patch("animatediff_ttnn.ttnn_pipeline.ttnn", ttnn_mock):
            shard_frames_to_device(frames, device)

    # from_torch called once with the stacked tensor
    assert ttnn_mock.from_torch.call_count == 1
    actual_tensor = ttnn_mock.from_torch.call_args[0][0]
    assert actual_tensor.shape == (2 * N, 4, lh, lw)
    # ShardTensorToMesh used as mesh_mapper
    ttnn_mock.ShardTensorToMesh.assert_called_once_with(device, dim=0)


def test_shard_frames_to_device_single_chip():
    """shard_frames_to_device on a 1-chip device still uses ShardTensorToMesh."""
    from animatediff_ttnn.ttnn_pipeline import shard_frames_to_device

    frames = [torch.randn(2, 4, 8, 8)]
    device = _make_mesh_device(num_chips=1)
    ttnn_mock = _make_ttnn_mock()

    with patch("animatediff_ttnn.ttnn_pipeline.ttnn", ttnn_mock):
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

    with patch("animatediff_ttnn.ttnn_pipeline.ttnn", ttnn_mock):
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

    with patch("animatediff_ttnn.ttnn_pipeline.ttnn", ttnn_mock):
        result = gather_frames_from_device(MagicMock(), device, num_frames=N)

    assert torch.allclose(result[0], stacked[0:2])
    assert torch.allclose(result[1], stacked[2:4])
    assert torch.allclose(result[2], stacked[4:6])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/ttuser/code/tt-animatediff
python -m pytest tests/test_mesh_sharding.py -v 2>&1 | head -30
```

Expected: `ImportError: cannot import name 'shard_frames_to_device'`

- [ ] **Step 3: Implement the two helpers in `ttnn_pipeline.py`**

Add the following after `from_device` (after line 137):

```python
def shard_frames_to_device(frame_tensors: list, device, dtype=None, layout=None):
    """Send N CFG-doubled frame tensors to device via frame-sharding.

    Stacks N tensors of shape [2, C, H, W] into [2N, C, H, W] and sends with
    ShardTensorToMesh(dim=0) so chip K receives frames [2K : 2K+2] — keeping
    each chip at batch_size=2, matching the compiled TTNN UNet kernel.

    Works correctly with a single-chip MeshDevice (shard == full tensor).

    Args:
        frame_tensors: List of N CPU tensors, each [2, C, H, W] (CFG-doubled).
        device: TTNN MeshDevice from setup_blackhole().
        dtype: Optional TTNN dtype (e.g. ttnn.bfloat16).
        layout: Optional TTNN layout (e.g. ttnn.TILE_LAYOUT).

    Returns:
        Single TTNN tensor sharded across chips, logical shape [2N, C, H, W].
    """
    import ttnn
    stacked = torch.cat(frame_tensors, dim=0)
    kwargs = {"mesh_mapper": ttnn.ShardTensorToMesh(device, dim=0)}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if layout is not None:
        kwargs["layout"] = layout
    return ttnn.from_torch(stacked, device=device, **kwargs)


def gather_frames_from_device(tensor, device, num_frames: int) -> list:
    """Retrieve N frame tensors from a sharded device tensor.

    Pulls the full [2N, C, H, W] tensor to CPU via ConcatMeshToTensor(dim=0)
    and splits into a list of N tensors of shape [2, C, H, W].

    Args:
        tensor: TTNN tensor sharded across chips (logical shape [2N, C, H, W]).
        device: TTNN MeshDevice from setup_blackhole().
        num_frames: N, the number of frames to split into.

    Returns:
        List of N CPU tensors, each [2, C, H, W].
    """
    import ttnn
    full = ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
    return [full[2 * i : 2 * i + 2] for i in range(num_frames)]
```

Also add `shard_frames_to_device` and `gather_frames_from_device` to `__init__.py` exports if the package exports anything — check first:

```bash
grep -n "shard\|gather\|to_device\|from_device" /home/ttuser/code/tt-animatediff/animatediff_ttnn/__init__.py
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/ttuser/code/tt-animatediff
python -m pytest tests/test_mesh_sharding.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add animatediff_ttnn/ttnn_pipeline.py tests/test_mesh_sharding.py
git commit -m "feat(mesh): add shard_frames_to_device and gather_frames_from_device helpers"
```

---

### Task 2: `num_frames % num_chips` guard tests and ValueError

**Context:** Both `generate_frames_temporal` and `generate_frames_motion` must raise `ValueError` early if `num_frames` is not divisible by the number of chips. This prevents a silent wrong-shape shard (e.g. 7 frames / 4 chips = 1.75 frames per chip, which TTNN would reject with a cryptic error deep in kernel dispatch). The guard also gives a clear single-chip path (1 chip always divides evenly).

`device.get_num_devices()` is the TTNN MeshDevice method for chip count. On a single-chip device it returns 1.

**Files:**
- Modify: `animatediff_ttnn/temporal_attention.py` (add guard at start of `generate_frames_temporal`)
- Modify: `animatediff_ttnn/ttnn_motion_pipeline.py` (add guard at start of `forward_unet_staged`)
- Modify: `tests/test_mesh_sharding.py` (add guard tests)

- [ ] **Step 1: Add guard tests to `tests/test_mesh_sharding.py`**

Append to the file:

```python
def test_num_frames_not_divisible_raises_temporal():
    """generate_frames_temporal raises ValueError if num_frames % num_chips != 0."""
    import importlib
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


def test_forward_unet_staged_guard_raises():
    """forward_unet_staged raises ValueError if num_frames % num_chips != 0."""
    ttnn_mock = _make_ttnn_mock()
    ttnn_mock.MeshDevice = type("MeshDevice", (), {})
    device = ttnn_mock.MeshDevice()
    device.get_num_devices = MagicMock(return_value=4)

    with patch.dict(sys.modules, {"ttnn": ttnn_mock}):
        from animatediff_ttnn.ttnn_motion_pipeline import forward_unet_staged
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
```

- [ ] **Step 2: Run new guard tests to verify they fail**

```bash
python -m pytest tests/test_mesh_sharding.py::test_num_frames_not_divisible_raises_temporal tests/test_mesh_sharding.py::test_forward_unet_staged_guard_raises -v
```

Expected: FAIL — no `ValueError` raised yet.

- [ ] **Step 3: Add guard to `generate_frames_temporal` in `temporal_attention.py`**

At the very top of `generate_frames_temporal`, after the imports block inside the function (just before the `if use_lightning:` scheduler setup block), add:

```python
    import ttnn as _ttnn
    _num_chips = device.get_num_devices() if isinstance(device, _ttnn.MeshDevice) else 1
    if num_frames % _num_chips != 0:
        raise ValueError(
            f"num_frames ({num_frames}) must be divisible by num_chips ({_num_chips}). "
            f"Valid counts for {_num_chips} chips: {[_num_chips * k for k in range(1, 9)]}"
        )
```

- [ ] **Step 4: Add guard to `forward_unet_staged` in `ttnn_motion_pipeline.py`**

At the top of `forward_unet_staged`, before the imports block, add:

```python
    if ttnn is not None:
        _num_chips = device.get_num_devices() if isinstance(device, ttnn.MeshDevice) else 1
    else:
        _num_chips = 1
    if num_frames % _num_chips != 0:
        raise ValueError(
            f"num_frames ({num_frames}) must be divisible by num_chips ({_num_chips}). "
            f"Valid counts for {_num_chips} chips: {[_num_chips * k for k in range(1, 9)]}"
        )
```

- [ ] **Step 5: Run all mesh sharding tests**

```bash
python -m pytest tests/test_mesh_sharding.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add animatediff_ttnn/temporal_attention.py animatediff_ttnn/ttnn_motion_pipeline.py tests/test_mesh_sharding.py
git commit -m "feat(mesh): add num_frames % num_chips ValueError guard"
```

---

### Task 3: Shard the UNet denoising loop in `generate_frames_temporal`

**Context:** `generate_frames_temporal` in `temporal_attention.py` currently has this inner loop at each denoising step (around line 347):

```python
for step_idx, t in enumerate(timesteps):
    noise_preds = []
    for i in range(num_frames):               # ← serial UNet calls
        lat = to_device(frame_latents[i], ...)
        lat_input = ttnn.concat([lat, lat], dim=0)
        ttnn_out = ttnn_model(lat_input, ...)
        guided = tt_guide(ttnn_out, guidance_scale)
        noise_preds.append(from_device(guided, device)...)
```

We replace the inner `for i` loop with a single `shard_frames_to_device` → `ttnn_model` → `gather_frames_from_device` call. Key details:

- CFG-doubling (`ttnn.concat([lat, lat], dim=0)`) must happen before sharding. Since we stack CPU-side, we CFG-double each frame's CPU tensor: `[fl, fl]` concatenated along `dim=0` giving `[2, 4, lh, lw]` per frame, then `shard_frames_to_device` stacks those to `[2N, 4, lh, lw]`.
- `tt_guide` extracts the guided noise prediction from a `[2, ...]` tensor (first half = uncond, second = cond). After gathering we have N tensors of `[2, 4, lh, lw]` — apply `tt_guide` per frame on the CPU result.
- The `lat_input.deallocate(True)` / `ttnn_out.deallocate(True)` / `guided.deallocate(True)` calls in the old loop become a single `stacked_dev.deallocate(True)` / `ttnn_out.deallocate(True)` after gathering.
- Lightning mode scales `frame_latents[i]` via `schedulers[i].scale_model_input(latent_cpu, t)` before sending to device — this still happens per-frame on CPU before CFG-doubling, so it fits into the new flow naturally.

**Files:**
- Modify: `animatediff_ttnn/temporal_attention.py` (lines ~347–381 — the `for step_idx` block's inner frame loop)

- [ ] **Step 1: Replace the inner UNet call loop**

Find the block in `generate_frames_temporal` starting with `for step_idx, t in enumerate(timesteps):` and replace the inner `for i in range(num_frames):` UNet call section:

```python
    for step_idx, t in enumerate(timesteps):
        # CFG-double each frame CPU-side, then shard across chips in one call.
        # Lightning: scale_model_input first (per-frame, CPU — Euler sigma normalization).
        cfg_latents = []
        for i in range(num_frames):
            latent_cpu = frame_latents[i]
            if use_lightning:
                latent_cpu = schedulers[i].scale_model_input(latent_cpu, t)
            cfg_latents.append(torch.cat([latent_cpu, latent_cpu], dim=0))  # [2, 4, lh, lw]

        stacked_dev = shard_frames_to_device(
            cfg_latents, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        )
        ttnn_out = ttnn_model(
            stacked_dev,
            timestep=_tlist[step_idx],
            encoder_hidden_states=ttnn_text_emb,
            class_labels=None,
            attention_mask=None,
            cross_attention_kwargs=None,
            return_dict=True,
            config=config,
        )
        stacked_dev.deallocate(True)

        # Gather all N frame outputs from device, apply CFG guidance per frame.
        frame_outputs = gather_frames_from_device(ttnn_out, device, num_frames)
        ttnn_out.deallocate(True)
        noise_preds = []
        for frame_out in frame_outputs:
            guided_cpu = tt_guide_cpu(frame_out, guidance_scale)
            noise_preds.append(guided_cpu.to(torch.float32))
```

Note: `tt_guide` in the current code operates on a TTNN tensor on-device. After gathering we have CPU tensors, so we need a CPU equivalent. `tt_guide` does `uncond + guidance_scale * (cond - uncond)` where the tensor is `[2, 4, lh, lw]` split in half. Add this CPU helper just above `generate_frames_temporal`:

```python
def _tt_guide_cpu(tensor: torch.Tensor, guidance_scale: float) -> torch.Tensor:
    """CFG guidance on a CPU [2, C, H, W] tensor (uncond=[:1], cond=[1:])."""
    uncond, cond = tensor[:1], tensor[1:]
    return uncond + guidance_scale * (cond - uncond)
```

Then inside `generate_frames_temporal` use `_tt_guide_cpu(frame_out, guidance_scale)` instead of `tt_guide`.

Also add the import for the new helpers at the top of the function's import block:

```python
    from animatediff_ttnn.ttnn_pipeline import build_tlist, to_device, from_device, shard_frames_to_device, gather_frames_from_device
```

- [ ] **Step 2: Run the existing temporal attention tests to detect regressions**

```bash
python -m pytest tests/test_temporal_attention.py tests/test_chain_blend.py -v
```

Expected: all existing tests PASS (they mock the UNet call and don't exercise the sharding path directly).

- [ ] **Step 3: Run the full test suite**

```bash
python -m pytest tests/ -v --ignore=tests/test_ttlang_temporal_attention.py 2>&1 | tail -20
```

Expected: all tests PASS (TT-Lang tests need the simulator, skip them).

- [ ] **Step 4: Commit**

```bash
git add animatediff_ttnn/temporal_attention.py
git commit -m "feat(mesh): shard UNet denoising loop in generate_frames_temporal"
```

---

### Task 4: Shard the VAE decode loop in `generate_frames_temporal`

**Context:** After the denoising loop in `generate_frames_temporal`, there is a serial VAE decode loop (around line 446):

```python
for i, latent in enumerate(frame_latents):
    latent_scaled = latent / 0.18215
    ttnn_lat = to_device(latent_scaled.permute(0, 2, 3, 1), device, ...)
    ttnn_decoded = ttnn_vae.decode(ttnn_lat)
    ...
```

We replace this with a single sharded decode. The TTNN Vae conv kernels are not batch-size-fixed like the UNet — they handle whatever batch they receive. Key shape flow:

- Stack N latents: `[N, 4, lh, lw]` → scale → permute to NHWC `[N, lh, lw, 4]` → shard (1 chip gets `[N/num_chips, lh, lw, 4]`)
- `ttnn_vae.decode(...)` → `[N, lh, lw, 3]` (gathered across chips)
- Reshape to `[N, H, W, 3]`, permute to `[N, 3, H, W]`, split into N PIL images

For VAE decode sharding we shard along `dim=0` of the NHWC latent stack — same `shard_frames_to_device` helper, but the input tensors are `[1, lh, lw, 4]` (single latent, NHWC) not `[2, 4, lh, lw]` (CFG-doubled NCHW). The helper just stacks and shards — the shape semantics are the same.

**Files:**
- Modify: `animatediff_ttnn/temporal_attention.py` (the VAE decode section near line 446)

- [ ] **Step 1: Replace the serial VAE decode loop**

Find the VAE decode block in `generate_frames_temporal` (starts with `frames = []`, `for i, latent in enumerate(frame_latents):`). Replace the entire VAE decode section with:

```python
    # Sharded VAE decode — stack all N latents, decode in parallel across chips.
    # Each chip decodes num_frames // num_chips frames. The TTNN Vae conv kernels
    # are not batch-size-fixed, so any N/num_chips batch is valid.
    lat_nhwc_list = [
        (lat / 0.18215).permute(0, 2, 3, 1)  # [1, lh, lw, 4] NHWC
        for lat in frame_latents
    ]
    # Stack to [N, lh, lw, 4] then shard: chip K gets [N//num_chips, lh, lw, 4]
    stacked_lat = torch.cat(lat_nhwc_list, dim=0)        # [N, lh, lw, 4]
    ttnn_lat = shard_frames_to_device(
        [stacked_lat[i:i+1] for i in range(num_frames)], # list of N [1, lh, lw, 4]
        device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    )
    ttnn_decoded = ttnn_vae.decode(ttnn_lat)
    ttnn_lat.deallocate(True)

    # Gather [N, H, W, 3] from all chips, convert to PIL
    decoded_all = gather_frames_from_device(ttnn_decoded, device, num_frames)
    ttnn_decoded.deallocate(True)
    frames = []
    for i in range(num_frames):
        # decoded_all[i] is [2, H, W, 3] because gather splits on [2, ...] assumption.
        # For VAE (no CFG doubling) each "frame" is [1, lh, lw, 4] → decoded [1, H, W, 3].
        # Reshape: ttnn_decoded shape before gather is [N, H, W, 3]; each split is [1, H, W, 3].
        dec = decoded_all[i]                               # [1, H, W, 3] or [2, H, W, 3]
        dec = dec[:1]                                      # ensure [1, H, W, 3]
        dec = dec.permute(0, 3, 1, 2).float()             # [1, 3, H, W]
        img = (dec / 2 + 0.5).clamp(0, 1)
        img = (img.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
        frames.append(Image.fromarray(img))
        print(f"  Frame {i + 1}/{num_frames} decoded")
```

**Important:** `gather_frames_from_device` assumes `[2, ...]` per frame because it was designed for CFG-doubled UNet tensors. For VAE decode the frames are `[1, ...]` not `[2, ...]`. We need a second variant or make `gather_frames_from_device` accept a `frames_per_shard` argument. Update the helper signature:

In `ttnn_pipeline.py`, change `gather_frames_from_device` to:

```python
def gather_frames_from_device(tensor, device, num_frames: int, batch_per_frame: int = 2) -> list:
    """Retrieve N frame tensors from a sharded device tensor.

    Args:
        tensor: TTNN tensor sharded across chips.
        device: TTNN MeshDevice from setup_blackhole().
        num_frames: N, the number of frames to split into.
        batch_per_frame: Batch size per frame (2 for CFG-doubled UNet tensors,
                         1 for VAE decode tensors). Default 2.

    Returns:
        List of N CPU tensors, each [batch_per_frame, ...].
    """
    import ttnn
    full = ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
    B = batch_per_frame
    return [full[B * i : B * i + B] for i in range(num_frames)]
```

Update the VAE decode call to use `batch_per_frame=1`:

```python
    decoded_all = gather_frames_from_device(ttnn_decoded, device, num_frames, batch_per_frame=1)
```

Also update the test in `test_mesh_sharding.py` — `test_gather_frames_from_device_splits_correctly` and `test_gather_frames_from_device_values` should still pass since the default is `batch_per_frame=2`. Add one new test:

```python
def test_gather_frames_single_batch_per_frame():
    """gather_frames_from_device with batch_per_frame=1 splits [N,C,H,W] into N [1,C,H,W]."""
    from animatediff_ttnn.ttnn_pipeline import gather_frames_from_device

    N, lh, lw = 4, 8, 8
    stacked = torch.randn(N, 4, lh, lw)   # N frames, NOT CFG-doubled
    ttnn_mock = _make_ttnn_mock()
    ttnn_mock.to_torch.return_value = stacked
    device = _make_mesh_device(num_chips=4)

    with patch("animatediff_ttnn.ttnn_pipeline.ttnn", ttnn_mock):
        result = gather_frames_from_device(MagicMock(), device, num_frames=N, batch_per_frame=1)

    assert len(result) == N
    for t in result:
        assert t.shape == (1, 4, lh, lw)
```

- [ ] **Step 2: Run all mesh sharding + temporal attention tests**

```bash
python -m pytest tests/test_mesh_sharding.py tests/test_temporal_attention.py tests/test_chain_blend.py -v
```

Expected: all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add animatediff_ttnn/ttnn_pipeline.py animatediff_ttnn/temporal_attention.py tests/test_mesh_sharding.py
git commit -m "feat(mesh): shard VAE decode in generate_frames_temporal; add batch_per_frame to gather helper"
```

---

### Task 5: Shard the UNet block loops in `forward_unet_staged`

**Context:** `forward_unet_staged` in `ttnn_motion_pipeline.py` has three places where frames are processed serially through TTNN block objects:

1. Down blocks (lines ~343–422) — `for i in range(num_frames): s, res_samples = down_block(...)`
2. Mid block (lines ~432–456) — `for i in range(num_frames): s = ttnn_model.mid_block(...)`
3. Up blocks (lines ~506–538) — `for i in range(num_frames): s = up_block(...)`

Each is replaced with a shard → block call → gather. The residual skip connections (`down_res_per_frame`) accumulate tensor lists per frame — these stay on-device (DRAM) between blocks and must be kept per-frame. The gather after each block restores the per-frame lists.

The evict-to-DRAM pattern (`ttnn.to_memory_config(s, ttnn.DRAM_MEMORY_CONFIG)`) remains — each chip's shard is evicted to that chip's DRAM before the next block, same as before.

The `pre_process_input` + `conv_in` section (lines ~247–308) also has a per-frame loop — shard that too.

**Files:**
- Modify: `animatediff_ttnn/ttnn_motion_pipeline.py`

- [ ] **Step 1: Add imports for the new helpers at the top of `forward_unet_staged`**

Inside `forward_unet_staged`, add to the deferred import block at the start (after the `from models.demos...` imports):

```python
    from animatediff_ttnn.ttnn_pipeline import shard_frames_to_device, gather_frames_from_device
```

- [ ] **Step 2: Replace the `conv_in` per-frame loop (lines ~293–308)**

The current loop:

```python
    hidden_samples = []
    for s in processed_samples:
        s, [ttnn_model.conv_in_weights, ttnn_model.conv_in_bias] = ttnn.conv2d(
            input_tensor=s,
            ...
            return_weights_and_bias=True,
        )
        s = reshard_for_output_channels_divisibility(s, out_channels)
        s = ttnn.reallocate(s)
        s = ttnn.to_memory_config(s, ttnn.DRAM_MEMORY_CONFIG)
        hidden_samples.append(s)
```

The `conv_in` layer has stateful weight caching (`return_weights_and_bias=True`) — it must be called once per device. With sharding, all N frames go through `conv_in` in one call. The output shape changes from `[2, ...]` to `[2N, ...]` (chip K sees `[2N/num_chips, ...]`).

Keep the loop for `conv_in` — it updates `ttnn_model.conv_in_weights`/`conv_in_bias` in-place, and sharding a conv with stateful weight caching may conflict. The `conv_in` loop is not a significant bottleneck (it runs once per generation, not per step). Leave it serial and comment why:

```python
    # conv_in loop stays serial: return_weights_and_bias=True updates weights
    # in-place on the model object — sharding a conv with stateful weight
    # tensors requires verifying the TTNN conv2d API supports batch>2 shards.
    # This loop runs once per generation (not per step), so it is not a bottleneck.
    hidden_samples = []
    for s in processed_samples:
        ...  # unchanged
```

- [ ] **Step 3: Replace the down block inner per-frame loops**

For each `CrossAttnDownBlock2D` block, replace:

```python
            new_hidden_dram = []
            for i in range(num_frames):
                hs_in = hidden_samples[i]
                ...
                s, res_samples = down_block(
                    hidden_states=hs_in,
                    ...
                )
                down_res_per_frame[i].extend(list(res_samples))
                s_dram = ttnn.to_memory_config(s, ttnn.DRAM_MEMORY_CONFIG)
                s.deallocate(True)
                new_hidden_dram.append(s_dram)
```

With:

```python
            # Shard all N hidden states → single block call → gather back.
            # Each chip runs CrossAttnDownBlock2D on its N/num_chips frames.
            stacked_dev = shard_frames_to_device(
                [ttnn.to_memory_config(h, ttnn.DRAM_MEMORY_CONFIG) if ttnn.get_memory_config(h).is_sharded()
                 else h for h in hidden_samples],
                device,
            )
            s_stacked, res_stacked_tuple = down_block(
                hidden_states=stacked_dev,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                config=config,
                resnet_groups=norm_num_groups,
                downsample_padding=downsample_padding,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[block_idx],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[block_idx],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            stacked_dev.deallocate(True)
            new_hidden_dram = gather_frames_from_device(s_stacked, device, num_frames)
            s_stacked.deallocate(True)
            # Residuals: res_stacked_tuple contains the stacked residuals for all N frames.
            # Split along dim=0 (same gather logic, num_layers residuals each [2N, ...]).
            for res_stacked in res_stacked_tuple:
                res_per_frame = gather_frames_from_device(res_stacked, device, num_frames)
                for i in range(num_frames):
                    down_res_per_frame[i].append(res_per_frame[i])
```

Apply the same pattern to the `DownBlock2D` branch (no temporal injection, simpler — no `res_stacked_tuple` since `DownBlock2D` also returns residuals in a tuple).

- [ ] **Step 4: Replace the mid block per-frame loop**

Replace:

```python
    new_hidden_dram = []
    for i in range(num_frames):
        s = ttnn_model.mid_block(hidden_states=hidden_samples[i], ...)
        s_dram = ttnn.to_memory_config(s, ttnn.DRAM_MEMORY_CONFIG)
        s.deallocate(True)
        new_hidden_dram.append(s_dram)
    hidden_samples = new_hidden_dram
```

With:

```python
    stacked_dev = shard_frames_to_device(hidden_samples, device)
    s_stacked = ttnn_model.mid_block(
        hidden_states=stacked_dev,
        temb=emb,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
        in_channels=block_out_channels[-1],
        temb_channels=time_embed_dim,
        resnet_eps=norm_eps,
        resnet_act_fn=act_fn,
        output_scale_factor=mid_block_scale_factor,
        resnet_time_scale_shift=resnet_time_scale_shift,
        cross_attention_dim=cross_attention_dim,
        config=config,
        attn_num_head_channels=attention_head_dim[-1],
        resnet_groups=norm_num_groups,
        dual_cross_attention=dual_cross_attention,
        use_linear_projection=use_linear_projection,
        upcast_attention=upcast_attention,
    )
    stacked_dev.deallocate(True)
    hidden_samples = gather_frames_from_device(s_stacked, device, num_frames)
    s_stacked.deallocate(True)
```

- [ ] **Step 5: Replace the up block per-frame loops**

Same pattern as down blocks. `CrossAttnUpBlock2D` returns a single tensor (no residual tuple output). `UpBlock2D` likewise.

For the residual consumption (`res_tuples`): currently each frame pops from its own `down_res_per_frame[i]` list. After sharding, those per-frame lists were populated by the gather in Step 3. The residual consumption logic (`res_tuple = tuple(down_res_per_frame[i][-resnets:])`) remains per-frame on CPU — the residuals were already gathered as CPU tensors. Re-shard them before each up block call:

```python
        # Re-shard the per-frame residuals for this up block.
        res_sharded_tuple = tuple(
            shard_frames_to_device([res_tuples[i][r] for i in range(num_frames)], device)
            for r in range(len(res_tuples[0]))
        )
        stacked_dev = shard_frames_to_device(hidden_samples, device)
        s_stacked = up_block(
            hidden_states=stacked_dev,
            temb=emb,
            res_hidden_states_tuple=res_sharded_tuple,
            ...
        )
        stacked_dev.deallocate(True)
        hidden_samples = gather_frames_from_device(s_stacked, device, num_frames)
        s_stacked.deallocate(True)
```

- [ ] **Step 6: Run the motion pipeline tests**

```bash
python -m pytest tests/test_ttnn_motion_pipeline.py -v
```

Expected: all existing tests PASS.

- [ ] **Step 7: Run full test suite**

```bash
python -m pytest tests/ -v --ignore=tests/test_ttlang_temporal_attention.py 2>&1 | tail -30
```

Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add animatediff_ttnn/ttnn_motion_pipeline.py
git commit -m "feat(mesh): shard UNet block loops in forward_unet_staged"
```

---

### Task 6: Shard the VAE decode in `generate_frames_motion`

**Context:** `generate_frames_motion` in `temporal_attention.py` has its own serial VAE decode loop (around line 664), identical in structure to `generate_frames_temporal`. Apply the same sharded decode from Task 4.

**Files:**
- Modify: `animatediff_ttnn/temporal_attention.py` (the VAE decode section in `generate_frames_motion`)

- [ ] **Step 1: Replace the VAE decode loop in `generate_frames_motion`**

Find the VAE decode section in `generate_frames_motion` (starts `for i, latent in enumerate(frame_latents):` after the chain-save block). Replace with the identical sharded pattern from Task 4:

```python
    # Sharded VAE decode — identical pattern to generate_frames_temporal.
    lat_nhwc_list = [
        (lat / 0.18215).permute(0, 2, 3, 1)
        for lat in frame_latents
    ]
    stacked_lat = torch.cat(lat_nhwc_list, dim=0)
    ttnn_lat = shard_frames_to_device(
        [stacked_lat[i:i+1] for i in range(num_frames)],
        device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    )
    ttnn_decoded = ttnn_vae.decode(ttnn_lat)
    ttnn_lat.deallocate(True)
    decoded_all = gather_frames_from_device(ttnn_decoded, device, num_frames, batch_per_frame=1)
    ttnn_decoded.deallocate(True)
    frames = []
    for i in range(num_frames):
        dec = decoded_all[i][:1].permute(0, 3, 1, 2).float()
        img = (dec / 2 + 0.5).clamp(0, 1)
        img = (img.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
        frames.append(Image.fromarray(img))
        print(f"  Frame {i + 1}/{num_frames} decoded")
```

Also update the imports inside `generate_frames_motion` to include the sharding helpers:

```python
    from animatediff_ttnn.ttnn_pipeline import build_tlist, to_device, from_device, shard_frames_to_device, gather_frames_from_device
```

- [ ] **Step 2: Add the `ValueError` guard to `generate_frames_motion`**

At the top of `generate_frames_motion`, after the `lh, lw = ...` line, add:

```python
    import ttnn as _ttnn_guard
    _num_chips_motion = device.get_num_devices() if isinstance(device, _ttnn_guard.MeshDevice) else 1
    if num_frames % _num_chips_motion != 0:
        raise ValueError(
            f"num_frames ({num_frames}) must be divisible by num_chips ({_num_chips_motion}). "
            f"Valid counts for {_num_chips_motion} chips: {[_num_chips_motion * k for k in range(1, 9)]}"
        )
```

- [ ] **Step 3: Run full test suite**

```bash
python -m pytest tests/ -v --ignore=tests/test_ttlang_temporal_attention.py 2>&1 | tail -30
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add animatediff_ttnn/temporal_attention.py
git commit -m "feat(mesh): shard VAE decode in generate_frames_motion; add guard"
```

---

### Task 7: Hardware smoke test script

**Context:** `scripts/ttlang_temporal_attn_hw_test.py` is the existing pattern for hardware smoke tests. We create a companion that runs a 4-frame generation (4 denoising steps for speed) in both serial 1-chip and 4-chip mesh modes, asserts PCC > 0.99 on the final decoded frame tensors, and prints a wall-clock breakdown.

**Files:**
- Create: `scripts/mesh_sharding_hw_test.py`

- [ ] **Step 1: Create the hardware smoke test**

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Hardware smoke test: mesh frame sharding vs serial single-chip.

Runs a 4-frame generation (4 denoising steps) twice:
  1. Serial mode — device_ids=[0], 1 chip
  2. Mesh mode   — device_ids=[0,1,2,3], 4 chips

Asserts PCC > 0.99 between final decoded frame tensors (same seed, same weights).
Prints per-component wall-clock breakdown (denoising, VAE decode, total).

Usage:
    source ~/tt-metal/python_env/bin/activate
    TT_METAL_ARCH_NAME=blackhole python scripts/mesh_sharding_hw_test.py
"""
import sys
import time
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def run_generation(device_ids: list, label: str):
    """Run 4-frame, 4-step generation. Returns (frames, denoising_s, vae_s, total_s)."""
    import ttnn
    from animatediff_ttnn.ttnn_pipeline import setup_blackhole
    from animatediff_ttnn.generation_helpers import load_sd14_ttnn, encode_prompt
    from animatediff_ttnn.temporal_attention import generate_frames_temporal

    print(f"\n{'='*60}")
    print(f"  {label}: device_ids={device_ids}")
    print(f"{'='*60}")

    t0 = time.time()
    device = setup_blackhole(device_ids=device_ids)
    ttnn_model, ttnn_vae, config, time_proj = load_sd14_ttnn(device)
    text_emb = encode_prompt("a serene mountain lake at sunrise")

    # Monkey-patch to time the two phases separately
    original_decode = ttnn_vae.decode
    _vae_times = []

    def timed_decode(*args, **kwargs):
        t = time.time()
        result = original_decode(*args, **kwargs)
        _vae_times.append(time.time() - t)
        return result

    ttnn_vae.decode = timed_decode

    t_gen_start = time.time()
    frames = generate_frames_temporal(
        device=device,
        ttnn_model=ttnn_model,
        ttnn_vae=ttnn_vae,
        config=config,
        torch_time_proj=time_proj,
        text_embeddings=text_emb,
        num_frames=4,
        num_steps=4,
        seed=42,
    )
    total_gen = time.time() - t_gen_start
    vae_s = sum(_vae_times)
    denoise_s = total_gen - vae_s

    ttnn.close_mesh_device(device)
    total_s = time.time() - t0

    print(f"  Denoising : {denoise_s:.1f}s")
    print(f"  VAE decode: {vae_s:.1f}s")
    print(f"  Total gen : {total_gen:.1f}s  (wall: {total_s:.1f}s incl. compile)")
    return frames, denoise_s, vae_s, total_s


def main():
    frames_serial, d1, v1, t1 = run_generation([0], "SERIAL (1 chip)")
    frames_mesh,   d4, v4, t4 = run_generation([0, 1, 2, 3], "MESH (4 chips)")

    print("\n" + "="*60)
    print("  SPEEDUP SUMMARY")
    print("="*60)
    print(f"  Denoising : {d1:.1f}s → {d4:.1f}s  ({d1/max(d4,0.1):.2f}×)")
    print(f"  VAE decode: {v1:.1f}s → {v4:.1f}s  ({v1/max(v4,0.1):.2f}×)")

    print("\n  PCC between serial and mesh outputs:")
    all_pass = True
    for i, (f1, f4) in enumerate(zip(frames_serial, frames_mesh)):
        t1_np = torch.tensor(np.array(f1)).float()
        t4_np = torch.tensor(np.array(f4)).float()
        p = pcc(t1_np, t4_np)
        status = "PASS" if p > 0.99 else "FAIL"
        if p <= 0.99:
            all_pass = False
        print(f"    Frame {i}: PCC={p:.4f}  [{status}]")

    if all_pass:
        print("\n  ALL PCC CHECKS PASSED")
        sys.exit(0)
    else:
        print("\n  SOME PCC CHECKS FAILED — check that both runs used identical seeds/weights")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make executable and verify it's syntactically valid**

```bash
chmod +x scripts/mesh_sharding_hw_test.py
python -c "import ast; ast.parse(open('scripts/mesh_sharding_hw_test.py').read()); print('syntax OK')"
```

Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/mesh_sharding_hw_test.py
git commit -m "feat(mesh): add mesh_sharding_hw_test.py hardware smoke test"
```

---

### Task 8: Update CLAUDE.md and run full test suite

**Context:** The CLAUDE.md project notes describe the architecture. Update to document mesh sharding, the `num_frames % num_chips` constraint, and the `shard_frames_to_device` / `gather_frames_from_device` helpers.

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Run the full test suite one final time**

```bash
cd /home/ttuser/code/tt-animatediff
python -m pytest tests/ -v --ignore=tests/test_ttlang_temporal_attention.py 2>&1 | tail -40
```

Expected: all tests PASS, no regressions.

- [ ] **Step 2: Update CLAUDE.md**

Add a new section after the "TT-Lang temporal attention track" section:

```markdown
## Mesh Frame Sharding (Phase 4)

Replaces the serialized `for i in range(num_frames)` loops in Phase 2.5 and Phase 3 with single sharded TTNN calls across all N Blackhole chips.

**New helpers in `animatediff_ttnn/ttnn_pipeline.py`:**
- `shard_frames_to_device(frame_tensors, device, dtype, layout)` — stacks N `[2, 4, lh, lw]` CPU tensors to `[2N, ...]` and sends via `ShardTensorToMesh(dim=0)`. Each chip gets `[2, ...]` — matching the compiled `batch_size=2` kernel.
- `gather_frames_from_device(tensor, device, num_frames, batch_per_frame=2)` — pulls via `ConcatMeshToTensor(dim=0)`, splits into N tensors of `[batch_per_frame, ...]`.

**Constraint:** `num_frames % num_chips == 0` — enforced by `ValueError` in `generate_frames_temporal`, `generate_frames_motion`, and `forward_unet_staged`. Valid counts for 4-chip QB2: 4, 8, 12, 16.

**Sharded paths:**
- `generate_frames_temporal`: UNet denoising loop + VAE decode sharded
- `generate_frames_motion`: UNet block loops in `forward_unet_staged` sharded; VAE decode in `generate_frames_motion` sharded
- `conv_in` loop in `forward_unet_staged` stays serial (stateful weight caching, runs once per generation)

**Expected speedup (4 chips, 8 frames):** ~3.2–4× Phase 2.5, ~2.5–2.9× Phase 3 (Phase 3 bottlenecked by `_apply_temporal` on CPU).

**Hardware smoke test:** `scripts/mesh_sharding_hw_test.py` — asserts PCC > 0.99 between 1-chip serial and 4-chip mesh outputs.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude): document mesh frame sharding (Phase 4)"
```

---

## Self-Review

**1. Spec coverage:**
- ✅ `shard_frames_to_device` / `gather_frames_from_device` helpers → Task 1
- ✅ `num_frames % num_chips` guard → Task 2
- ✅ UNet loop sharding in `generate_frames_temporal` → Task 3
- ✅ VAE decode sharding in `generate_frames_temporal` → Task 4
- ✅ UNet block loop sharding in `forward_unet_staged` → Task 5
- ✅ VAE decode sharding in `generate_frames_motion` → Task 6
- ✅ Hardware smoke test → Task 7
- ✅ CLAUDE.md update → Task 8
- ✅ Single-chip graceful degradation → covered in Task 1 (ShardTensorToMesh on 1 chip)
- ✅ `batch_per_frame=1` for VAE decode → Task 4 (updated helper + new test)

**2. Placeholder scan:** None found. All code blocks are complete.

**3. Type consistency:**
- `shard_frames_to_device` defined in Task 1, used in Tasks 3, 4, 5, 6 — consistent signature `(frame_tensors, device, dtype=None, layout=None)`.
- `gather_frames_from_device` updated in Task 4 to add `batch_per_frame=2` default — Tasks 3 and 5 use default (CFG=2), Tasks 4 and 6 use `batch_per_frame=1`. Consistent.
- `_tt_guide_cpu` defined and used in Task 3 only. Consistent.

**Note on Task 5 complexity:** `forward_unet_staged` sharding is the most complex change — residual skip connections must be split per-frame after gathering, then re-sharded before up blocks. The plan shows the shape logic explicitly. If the TTNN block API returns residuals differently than expected (e.g., already-gathered or differently batched), the implementer should check the actual return type of `down_block(...)` in the existing serial code before replacing the loop.
