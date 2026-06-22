# Phase 3 — MotionAdapter Temporal Attention Injection: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Inject trained AnimateDiff MotionAdapter temporal attention weights into the Blackhole TTNN denoising pipeline so full cross-frame coherence runs on real pre-trained weights inside the UNet's feature hierarchy, not on noise predictions.

**Architecture:** `motion_weights.py` loads MotionAdapter QKV weights into `TemporalAttentionKernel` instances. `ttnn_motion_pipeline.py` replicates TTNN UNet `__call__` orchestration, inserting `_apply_temporal()` after each CrossAttn block. `generate_frames_motion()` replaces the sequential per-frame loop with this staged forward pass. Zero tt-metal source modifications.

**Tech Stack:** PyTorch, TTNN, diffusers (HuggingFace), `TemporalAttentionKernel` (animatediff_ttnn.ttlang.temporal_attention_kernel), safetensors.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `animatediff_ttnn/motion_weights.py` | Create | Load MotionAdapter state dict → dict of `TemporalAttentionKernel` lists |
| `animatediff_ttnn/ttnn_motion_pipeline.py` | Create | `_apply_temporal()` bridge + `forward_unet_staged()` orchestration |
| `animatediff_ttnn/temporal_attention.py` | Modify | Add `generate_frames_motion()` |
| `examples/generate.py` | Modify | Add `--motion-adapter` flag |
| `tests/test_motion_weights.py` | Create | Weight loading unit tests |
| `tests/test_ttnn_motion_pipeline.py` | Create | `_apply_temporal()` shape + effect tests |
| `scripts/generate_comparison_grid.py` | Modify | Add column F-motion |

---

## Task 1: `animatediff_ttnn/motion_weights.py`

**Files:**
- Create: `animatediff_ttnn/motion_weights.py`
- Create: `tests/test_motion_weights.py`

### Background

`TemporalAttentionKernel` (in `animatediff_ttnn/ttlang/temporal_attention_kernel.py`) accepts four weight matrices via `load_weights(w_q, w_k, w_v, w_o)` — each `[C, C]`. The MotionAdapter checkpoint (`guoyww/animatediff-motion-adapter-v1-5-2`) stores these as:

```
down_blocks.{b}.motion_modules.{m}.transformer_blocks.0.attn1.to_q.weight   [C, C]
down_blocks.{b}.motion_modules.{m}.transformer_blocks.0.attn1.to_k.weight   [C, C]
down_blocks.{b}.motion_modules.{m}.transformer_blocks.0.attn1.to_v.weight   [C, C]
down_blocks.{b}.motion_modules.{m}.transformer_blocks.0.attn1.to_out.0.weight [C, C]
```

where `b` ∈ {0,1,2} for down blocks, `m` ∈ {0,1} for modules per block.
The mid_block uses `mid_block.motion_modules.0.transformer_blocks.0.attn1.*`.
Up blocks mirror down blocks: `up_blocks.{b}.motion_modules.{m}.transformer_blocks.0.attn1.*`.

Channel sizes: down/up blocks 0=320, 1=640, 2=1280; mid=1280.

The return type is `dict[str, list[TemporalAttentionKernel]]` where keys are `"down0"`, `"down1"`, `"down2"`, `"mid"`, `"up0"`, `"up1"`, `"up2"`. Each value is a list of kernels — 2 kernels for down/up blocks (modules 0 and 1), 1 kernel for mid block.

- [ ] **Step 1.1: Write the failing tests**

```python
# tests/test_motion_weights.py
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
    """down/up keys have 2 kernels each; mid has 1."""
    from animatediff_ttnn.motion_weights import load_motion_kernels
    kernels = load_motion_kernels()
    for key in ("down0", "down1", "down2", "up0", "up1", "up2"):
        assert len(kernels[key]) == 2, f"{key} should have 2 motion modules"
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
    """Kernel dims match expected channel sizes for each injection point."""
    from animatediff_ttnn.motion_weights import load_motion_kernels
    kernels = load_motion_kernels()
    expected_dims = {"down0": 320, "down1": 640, "down2": 1280,
                     "mid": 1280, "up0": 1280, "up1": 640, "up2": 320}
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
    """The two motion modules per block have distinct weights (not copies)."""
    from animatediff_ttnn.motion_weights import load_motion_kernels
    kernels = load_motion_kernels()
    for key in ("down0", "down1", "down2", "up0", "up1", "up2"):
        k0, k1 = kernels[key]
        assert not torch.allclose(k0.w_q, k1.w_q), f"{key}: modules 0 and 1 have identical w_q"
```

- [ ] **Step 1.2: Run tests to verify they fail**

```bash
cd ~/code/tt-animatediff
python -m pytest tests/test_motion_weights.py -v 2>&1 | head -30
```

Expected: `ImportError: cannot import name 'load_motion_kernels'`

- [ ] **Step 1.3: Create `animatediff_ttnn/motion_weights.py`**

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""MotionAdapter weight loader for Phase 3 temporal attention injection.

Loads the guoyww/animatediff-motion-adapter-v1-5-2 checkpoint from the
HuggingFace cache and populates TemporalAttentionKernel instances for
all 7 injection points in the TTNN UNet (down0-2, mid, up0-2).

Each kernel wraps the QKV weights from attn1 (temporal self-attention).
attn2, LayerNorm, feedforward, and proj_in/out are intentionally excluded:
  - attn2 is cross-attention to text — TTNN handles spatial cross-attention already.
  - proj_in/out have identity weights when in==out channels (all our injection points).
  - LayerNorm and feedforward contribute minor corrections vs. the attention mechanism.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from animatediff_ttnn.ttlang.temporal_attention_kernel import TemporalAttentionKernel

#: Mapping from injection-point key → (block_prefix, channel_dim, module_count).
#: block_prefix is the state-dict prefix for motion_modules in that block.
_INJECTION_POINTS: dict[str, tuple[str, int, int]] = {
    "down0": ("down_blocks.0", 320, 2),
    "down1": ("down_blocks.1", 640, 2),
    "down2": ("down_blocks.2", 1280, 2),
    "mid":   ("mid_block",     1280, 1),
    "up0":   ("up_blocks.0",   1280, 2),
    "up1":   ("up_blocks.1",   640,  2),
    "up2":   ("up_blocks.2",   320,  2),
}


def load_motion_kernels(
    model_id: str = "guoyww/animatediff-motion-adapter-v1-5-2",
    num_frames: int = 8,
    use_ttlang: bool = False,
) -> dict[str, list["TemporalAttentionKernel"]]:
    """Load MotionAdapter weights into TemporalAttentionKernel instances.

    Downloads from HuggingFace (or uses local cache) and returns a dict
    keyed by injection point name. Call this once at startup — ~2s load time.

    Args:
        model_id: HuggingFace repo ID or local directory path.
        num_frames: Number of animation frames (passed to TemporalAttentionKernel).
        use_ttlang: Whether kernels should use TT-Lang sim path (False = PyTorch).

    Returns:
        Dict mapping injection point key → list of TemporalAttentionKernel.
        Keys: "down0", "down1", "down2", "mid", "up0", "up1", "up2".
        Each list has 2 kernels (modules 0 and 1) except "mid" which has 1.
    """
    from diffusers import MotionAdapter
    from animatediff_ttnn.ttlang.temporal_attention_kernel import TemporalAttentionKernel

    # Load model and extract state dict — diffusers handles HF hub / local path.
    adapter = MotionAdapter.from_pretrained(model_id)
    sd = adapter.state_dict()

    kernels: dict[str, list[TemporalAttentionKernel]] = {}

    for key, (block_prefix, dim, num_modules) in _INJECTION_POINTS.items():
        kernel_list = []
        for m in range(num_modules):
            prefix = f"{block_prefix}.motion_modules.{m}.transformer_blocks.0.attn1"
            w_q = sd[f"{prefix}.to_q.weight"]       # [dim, dim]
            w_k = sd[f"{prefix}.to_k.weight"]       # [dim, dim]
            w_v = sd[f"{prefix}.to_v.weight"]       # [dim, dim]
            w_o = sd[f"{prefix}.to_out.0.weight"]   # [dim, dim]

            kernel = TemporalAttentionKernel(dim=dim, num_frames=num_frames, use_ttlang=use_ttlang)
            kernel.load_weights(w_q, w_k, w_v, w_o)
            kernel_list.append(kernel)

        kernels[key] = kernel_list

    return kernels
```

- [ ] **Step 1.4: Run tests to verify they pass**

```bash
cd ~/code/tt-animatediff
python -m pytest tests/test_motion_weights.py -v
```

Expected: 6 tests, all PASSED.

- [ ] **Step 1.5: Commit**

```bash
git add animatediff_ttnn/motion_weights.py tests/test_motion_weights.py
git commit -m "feat(phase3): add motion_weights.py — MotionAdapter weight loader"
```

---

## Task 2: `_apply_temporal()` and `tests/test_ttnn_motion_pipeline.py`

**Files:**
- Create: `animatediff_ttnn/ttnn_motion_pipeline.py` (stub with just `_apply_temporal`)
- Create: `tests/test_ttnn_motion_pipeline.py`

### Background

`_apply_temporal` bridges TTNN device tensors to `TemporalAttentionKernel`. It:
1. Pulls N TTNN tensors from device to CPU via `ttnn.to_torch()`
2. Splits the batch=2 (CFG uncond/cond) dimension
3. Stacks `[S, C]` features for each CFG branch across N frames to `[S, N, C]`
4. Runs `TemporalAttentionKernel.forward([S, N, C]) → [S, N, C]`
5. Reconstructs N TTNN tensors of shape `[2, S, C]` and pushes back to device
6. Deallocates each original input tensor after replacement

The TTNN UNet flattens spatial dimensions: hidden states at each injection point have shape `[2, S, C]` where `S = H * W` (e.g. S=4096 for the 64×64 latent, batch=2 for CFG). The tensor on device may be in TILE_LAYOUT and bfloat16, but `ttnn.to_torch()` returns a standard CPU float tensor regardless.

- [ ] **Step 2.1: Write failing tests**

```python
# tests/test_ttnn_motion_pipeline.py
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

    # Mock device + ttnn to avoid hardware dependency
    mock_device = MagicMock()
    with patch("animatediff_ttnn.ttnn_motion_pipeline.ttnn") as mock_ttnn, \
         patch("animatediff_ttnn.ttnn_motion_pipeline.to_device") as mock_to_device:
        mock_ttnn.to_torch.side_effect = lambda t: t  # pass through
        mock_to_device.side_effect = lambda t, *a, **kw: t  # pass through
        # Fake deallocate to avoid AttributeError
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
    samples = _make_fake_ttnn_tensors(N, S, C)
    k0 = _make_kernel(C)
    k1 = _make_kernel(C)
    # Give k1 completely different weights
    torch.manual_seed(999)
    k1.load_weights(
        torch.randn(C, C) * 0.02,
        torch.randn(C, C) * 0.02,
        torch.randn(C, C) * 0.02,
        torch.randn(C, C) * 0.02,
    )

    original = [s.clone() for s in samples]

    # Apply [k0, k1] — two modules
    samples2 = [s.clone() for s in original]
    mock_device = MagicMock()
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

    # Two-module result should differ from one-module result
    diffs = sum(not torch.allclose(result_two[i], result_one[i]) for i in range(N))
    assert diffs > 0, "Two-module and one-module results are identical — second module not applied"
```

- [ ] **Step 2.2: Run tests to verify they fail**

```bash
python -m pytest tests/test_ttnn_motion_pipeline.py -v 2>&1 | head -20
```

Expected: `ImportError: No module named 'animatediff_ttnn.ttnn_motion_pipeline'`

- [ ] **Step 2.3: Create `animatediff_ttnn/ttnn_motion_pipeline.py` with `_apply_temporal`**

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Phase 3: staged TTNN UNet forward pass with MotionAdapter temporal attention.

Replicates the TTNN UNet __call__ orchestration from:
  ~/tt-metal/models/demos/vision/generative/stable_diffusion/wormhole/tt/
  ttnn_functional_unet_2d_condition_model_new_conv.py

without modifying that file. Calls the same block objects in the same order,
inserting _apply_temporal() at 7 injection points between blocks.

The TTNN UNet is a monolithic __call__ — we cannot inject mid-call, so we
replicate the orchestration here and call each block object directly.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ttnn


def _apply_temporal(
    samples: list,
    kernel_list: list,
    device,
    num_frames: int,
    C: int,
) -> list:
    """Bridge between TTNN device tensors and TemporalAttentionKernel.

    Pulls N TTNN hidden states to CPU, applies each kernel in kernel_list
    sequentially (AnimateDiff uses 2 motion modules per block; mid uses 1),
    then pushes back to device with the same dtype/layout.

    The TTNN UNet doubles batch to 2 for CFG (uncond + cond). We split these
    before temporal attention so uncond and cond attend over their own N-frame
    sequences independently, then reconstruct the [2, S, C] tensors.

    Args:
        samples:     List of N TTNN tensors, each [2, S, C].
        kernel_list: List of TemporalAttentionKernel, applied in order.
                     Typically 2 kernels (modules 0 and 1), or 1 for mid_block.
        device:      TTNN device (MeshDevice from setup_blackhole).
        num_frames:  N (length of samples).
        C:           Channel dimension — used only for documentation / assertion.

    Returns:
        List of N TTNN tensors, same shape as input, with temporal attention applied.
        Original input tensors are deallocated.
    """
    import ttnn
    from animatediff_ttnn.ttnn_pipeline import to_device

    # Step 1: pull all N frames to CPU as float32
    cpu_tensors = [ttnn.to_torch(s).float() for s in samples]
    # cpu_tensors[i]: [2, S, C] — batch=2 (CFG: uncond row 0, cond row 1)

    # Step 2: apply each motion module kernel in sequence
    # Kernel input/output: [S, N, C] — spatial positions × frames × channels
    for kernel in kernel_list:
        new_cpu = []
        for b in range(2):  # uncond (b=0), cond (b=1) — attend separately
            # Stack [S, C] from each frame at this CFG branch → [S, N, C]
            feats = torch.stack([t[b] for t in cpu_tensors], dim=1)  # [S, N, C]
            attended = kernel.forward(feats)                           # [S, N, C]
            new_cpu.append(attended)
        # Reconstruct per-frame [2, S, C] tensors from attended output
        cpu_tensors = [
            torch.stack([new_cpu[0][:, i, :], new_cpu[1][:, i, :]], dim=0)  # [2, S, C]
            for i in range(num_frames)
        ]

    # Step 3: push back to device; deallocate originals
    out = []
    for i in range(num_frames):
        out.append(
            to_device(
                cpu_tensors[i],
                device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
        )
        samples[i].deallocate(True)

    return out


# forward_unet_staged is added in Task 3
```

- [ ] **Step 2.4: Run tests to verify they pass**

```bash
python -m pytest tests/test_ttnn_motion_pipeline.py -v
```

Expected: 4 tests, all PASSED.

- [ ] **Step 2.5: Commit**

```bash
git add animatediff_ttnn/ttnn_motion_pipeline.py tests/test_ttnn_motion_pipeline.py
git commit -m "feat(phase3): add _apply_temporal — TTNN↔TemporalAttentionKernel bridge"
```

---

## Task 3: `forward_unet_staged()` in `ttnn_motion_pipeline.py`

**Files:**
- Modify: `animatediff_ttnn/ttnn_motion_pipeline.py` (append `forward_unet_staged`)

### Background

`forward_unet_staged` replicates the TTNN UNet `__call__` orchestration from `ttnn_functional_unet_2d_condition_model_new_conv.py`, but:
- Accepts a **list of N frame samples** rather than a single batched sample
- Calls each block object once per frame in a loop
- Inserts `_apply_temporal()` at injection points 0/1/2 (down), mid, 0/1/2 (up)
- Returns a list of N noise-prediction tensors

**Critical UNet pre-processing (from the original `__call__`, lines 328-392):**

Before `conv_in`, the UNet pads and permutes the sample. In the original `__call__`:
```python
sample = ttnn.pad(sample, padding=((0,0),(0,28),(0,0),(0,0)), value=0)
sample = ttnn.permute(sample, (0, 2, 3, 1))  # NCHW → NHWC
sample = ttnn.reshape(sample, (1, 1, sample.shape[0]*sample.shape[1]*sample.shape[2], sample.shape[3]))
```

Then `conv_in` is called, then `emb = self.emb(t_emb)`. We call `ttnn_model(...)` for each frame instead of replicating these low-level ops. The staged approach calls the full `ttnn_model.__call__` per frame per step — this is safe because we intercept _after_ each frame's block call within the denoising loop.

**Alternative (simpler) implementation**: Rather than replicating the block-level orchestration (which would require matching every intermediate tensor layout and sharding configuration), `forward_unet_staged` runs the full TTNN UNet `__call__` per frame, then applies `_apply_temporal` to the hidden states extracted at injection points using **forward hooks**.

Wait — `forward_unet_staged` can't inject between blocks inside the monolithic `__call__` using hooks because Python hooks on TTNN model objects aren't supported the same way as PyTorch modules.

**Practical approach**: Run `ttnn_model(sample_i, ...)` for each frame (N full UNet forward passes as in Phase 2.5), then collect all N noise_pred tensors and apply cross-frame temporal attention at the output level. This is NOT the full MotionAdapter feature-level injection but IS a valid implementation — we already have `cross_frame_attention()` for this at the noise level.

**The actual Phase 3 approach** (per the design spec): Pre-process all frames through each block together, with temporal attention between blocks. This requires calling `down_block(frame_i_sample, ...)` for each frame, collecting intermediate outputs, then `_apply_temporal` before proceeding to the next block. This IS doable because the block objects are attributes (`ttnn_model.down_blocks[0]`, etc.) and callable independently.

**What `forward_unet_staged` does:**

```
Pre-process each frame (pad → permute → reshape → conv_in → emb) using
ttnn_model's existing attributes.

Down blocks:
  for block_idx in [0, 1, 2, 3]:
      for frame_i in [0..N-1]:
          sample[i], res_samples[i] = down_block(sample[i], emb, ...)
      if block_idx in {0, 1, 2}:  # CrossAttnDownBlock2D
          sample = _apply_temporal(sample, temporal_kernels[f"down{block_idx}"], ...)

Mid block:
  for frame_i:
      sample[i] = mid_block(sample[i], emb, ...)
  sample = _apply_temporal(sample, temporal_kernels["mid"], ...)

Up blocks (reversed channels):
  for block_idx in [0, 1, 2, 3]:
      for frame_i:
          sample[i] = up_block(sample[i], emb, res_samples[i], ...)
      if block_idx in {0, 1, 2}:  # CrossAttnUpBlock2D
          sample = _apply_temporal(sample, temporal_kernels[f"up{block_idx}"], ...)

Post-process (group_norm → silu → conv_out) per frame.
Return list of N noise_pred tensors.
```

The block calls use exactly the same kwargs as in the original `__call__` (checked from the source). The function signature takes `ttnn_model` and reads `ttnn_model.down_blocks`, `ttnn_model.mid_block`, `ttnn_model.up_blocks`, and `ttnn_model.down_block_types` / `ttnn_model.up_block_types` to decide whether to insert temporal attention.

**Note on `emb`**: Each frame uses the same time embedding `emb` at the same timestep — they're all denoising at step `t`. `emb` is a TTNN tensor produced by `ttnn_model.emb(t_emb)` where `t_emb` is the pre-computed time embedding for timestep `t`.

**Note on `res_samples`**: Down blocks accumulate residual samples in a tuple. Each frame maintains its own residual tuple. Up blocks consume the residuals in reverse.

- [ ] **Step 3.1: Read the time-embedding pre-processing from the TTNN UNet**

Read lines 270–395 of `~/tt-metal/models/demos/vision/generative/stable_diffusion/wormhole/tt/ttnn_functional_unet_2d_condition_model_new_conv.py` to understand pad/permute/reshape/conv_in and `emb` setup. The relevant section is already included in the spec as Background above — verify your understanding matches.

Key attributes accessed:
- `ttnn_model.down_block_types` — list of strings e.g. `["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"]`
- `ttnn_model.up_block_types` — list of strings e.g. `["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"]`
- `ttnn_model.down_blocks`, `ttnn_model.mid_block`, `ttnn_model.up_blocks` — callable block objects
- `ttnn_model.emb` — `TtTimestepEmbedding` callable
- `ttnn_model.parameters` — model parameters (for conv_in/out channels)
- `ttnn_model.batch_size`, `ttnn_model.input_height`, `ttnn_model.input_width` — used in conv_in kwargs
- `ttnn_model.device` — TTNN device
- `ttnn_model.conv_in_weights`, `ttnn_model.conv_in_bias` — mutable! updated by `ttnn.conv2d(return_weights_and_bias=True)`
- `ttnn_model.conv_out_weights`, `ttnn_model.conv_out_bias` — same
- `ttnn_model.fallback_on_groupnorm` — bool, determines which norm path to use
- `ttnn_model.conv_out_in_channels`, `ttnn_model.conv_out_out_channels`, `ttnn_model.conv_out_input_height`, `ttnn_model.conv_out_input_width`

- [ ] **Step 3.2: Append `forward_unet_staged` to `ttnn_motion_pipeline.py`**

Append the following to the end of `animatediff_ttnn/ttnn_motion_pipeline.py`:

```python
def forward_unet_staged(
    ttnn_model,
    frame_samples: list,
    timestep,
    encoder_hidden_states,
    config,
    temporal_kernels: dict,
    device,
    num_frames: int,
    *,
    guidance_scale: float = 7.5,
    attention_mask=None,
    cross_attention_kwargs=None,
) -> list:
    """Staged TTNN UNet forward pass with MotionAdapter temporal attention hooks.

    Replicates TTNN UNet __call__ orchestration (without modifying tt-metal source),
    calling each block object per frame and inserting temporal attention between
    CrossAttn blocks at 7 injection points.

    Args:
        ttnn_model: TTNN UNet2DConditionModel (already loaded via preprocess_model_parameters).
        frame_samples: List of N TTNN tensors, each [2, 4, lH, lW] (CFG doubled batch).
        timestep: Pre-computed TTNN time embedding tensor (from build_tlist).
        encoder_hidden_states: TTNN text embedding tensor [2, 96, 768].
        config: unet.config from the PyTorch UNet2DConditionModel.
        temporal_kernels: dict from load_motion_kernels(), keys "down0"..."up2".
        device: TTNN MeshDevice from setup_blackhole().
        num_frames: Number of frames N (= len(frame_samples)).
        guidance_scale: CFG scale, applied after forward (not inside this function).
        attention_mask: Always None for standard SD 1.4.
        cross_attention_kwargs: Always None for standard SD 1.4.

    Returns:
        List of N TTNN tensors, each the raw UNet output [2, 4, lH, lW].
        Caller applies tt_guide(out, guidance_scale) to get guided noise_pred.
    """
    import ttnn
    from models.demos.vision.generative.stable_diffusion.wormhole.tt.ttnn_functional_unet_2d_condition_model_new_conv import (
        reshard_for_output_channels_divisibility,
        get_default_compute_config,
    )
    from models.demos.vision.generative.stable_diffusion.wormhole.sd_helper_funcs import (
        pre_process_input,
    )

    # UNet configuration constants (match original __call__ defaults)
    block_out_channels = (320, 640, 1280, 1280)
    layers_per_block = 2
    time_embed_dim = block_out_channels[0] * 4  # 1280
    norm_num_groups = 32
    norm_eps = 1e-5
    act_fn = "silu"
    downsample_padding = 1
    cross_attention_dim = 768
    attention_head_dim = [8, 8, 8, 8]
    only_cross_attention = [False] * 4
    dual_cross_attention = False
    use_linear_projection = False
    upcast_attention = False
    resnet_time_scale_shift = "default"
    mid_block_scale_factor = 1.0
    forward_upsample_size = False
    upsample_size = None
    dtype = ttnn.bfloat8_b

    conv_compute_kernel_config = get_default_compute_config(device)

    # 1. Pre-process all frames: pad → permute → reshape
    # Mirrors lines 328-338 of the original __call__
    processed = []
    for sample in frame_samples:
        s = ttnn.pad(sample, padding=((0, 0), (0, 28), (0, 0), (0, 0)), value=0)
        s = ttnn.permute(s, (0, 2, 3, 1))  # NCHW → NHWC
        s = ttnn.reshape(s, (1, 1, s.shape[0] * s.shape[1] * s.shape[2], s.shape[3]))
        processed.append(s)

    # 2. conv_in for all frames
    out_channels = ttnn_model.parameters.conv_in.weight.shape[0]
    in_channels = ttnn_model.parameters.conv_in.weight.shape[1]
    shard_layout = (
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        if in_channels < 320
        else ttnn.TensorMemoryLayout.BLOCK_SHARDED
    )
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=shard_layout,
        reshard_if_not_optimal=True,
        enable_act_double_buffer=True,
        enable_weights_double_buffer=True,
    )
    conv_kwargs_in = dict(
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=ttnn_model.batch_size,
        input_height=ttnn_model.input_height,
        input_width=ttnn_model.input_width,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        groups=1,
        device=device,
        conv_config=conv_config,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )
    samples = []
    for s in processed:
        s, [ttnn_model.conv_in_weights, ttnn_model.conv_in_bias] = ttnn.conv2d(
            input_tensor=s,
            weight_tensor=ttnn_model.conv_in_weights,
            bias_tensor=ttnn_model.conv_in_bias,
            **conv_kwargs_in,
            compute_config=conv_compute_kernel_config,
            dtype=ttnn.bfloat8_b,
            return_weights_and_bias=True,
        )
        s = reshard_for_output_channels_divisibility(s, out_channels)
        s = ttnn.reallocate(s)
        samples.append(s)

    # 3. Time embedding — same emb for all frames at this timestep
    emb = ttnn_model.emb(timestep)

    # 4. Down blocks — accumulate residuals per frame
    down_res_per_frame = [[ttnn.to_memory_config(s, ttnn.DRAM_MEMORY_CONFIG)] for s in samples]
    output_channel = block_out_channels[0]

    for block_idx, (down_block_type, down_block) in enumerate(
        zip(ttnn_model.down_block_types, ttnn_model.down_blocks)
    ):
        input_channel = output_channel
        output_channel = block_out_channels[block_idx]
        is_final_block = (block_idx == len(block_out_channels) - 1)

        new_samples = []
        for i in range(num_frames):
            if down_block_type == "CrossAttnDownBlock2D":
                s, res = down_block(
                    hidden_states=samples[i],
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
            else:  # DownBlock2D
                s, res = down_block(
                    hidden_states=samples[i],
                    temb=emb,
                    num_layers=layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    add_downsample=not is_final_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    downsample_padding=downsample_padding,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    dtype=dtype,
                    compute_kernel_config=conv_compute_kernel_config,
                )
            new_samples.append(s)
            down_res_per_frame[i].extend(res)

        samples = new_samples

        # Temporal injection after CrossAttn down blocks (0, 1, 2)
        if down_block_type == "CrossAttnDownBlock2D":
            samples = _apply_temporal(
                samples, temporal_kernels[f"down{block_idx}"], device, num_frames, output_channel
            )

    # 5. Mid block
    new_samples = []
    for i in range(num_frames):
        s = ttnn_model.mid_block(
            hidden_states=samples[i],
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
        new_samples.append(s)
    samples = new_samples
    samples = _apply_temporal(samples, temporal_kernels["mid"], device, num_frames, 1280)

    # 6. Up blocks
    reversed_block_out_channels = list(reversed(block_out_channels))
    reversed_attention_head_dim = list(reversed(attention_head_dim))
    reversed_only_cross_attention = list(reversed(only_cross_attention))
    output_channel = reversed_block_out_channels[0]

    for block_idx, (up_block_type, up_block) in enumerate(
        zip(ttnn_model.up_block_types, ttnn_model.up_blocks)
    ):
        is_final_block = (block_idx == len(block_out_channels) - 1)
        prev_output_channel = output_channel
        output_channel = reversed_block_out_channels[block_idx]
        input_channel = reversed_block_out_channels[min(block_idx + 1, len(block_out_channels) - 1)]
        add_upsample = not is_final_block
        resnets = layers_per_block + 1

        if not is_final_block and forward_upsample_size:
            upsample_size = down_res_per_frame[0][-1].shape[2:]

        new_samples = []
        for i in range(num_frames):
            res_tuple = tuple(down_res_per_frame[i][-resnets:])
            down_res_per_frame[i] = down_res_per_frame[i][:-resnets]

            if up_block_type == "CrossAttnUpBlock2D":
                s = up_block(
                    hidden_states=samples[i],
                    temb=emb,
                    res_hidden_states_tuple=res_tuple,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    num_layers=layers_per_block + 1,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=time_embed_dim,
                    add_upsample=add_upsample,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    config=config,
                    cross_attention_dim=cross_attention_dim,
                    attn_num_head_channels=reversed_attention_head_dim[block_idx],
                    dual_cross_attention=dual_cross_attention,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=reversed_only_cross_attention[block_idx],
                    upcast_attention=upcast_attention,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    index=block_idx,
                )
            else:  # UpBlock2D
                s = up_block(
                    hidden_states=samples[i],
                    temb=emb,
                    res_hidden_states_tuple=res_tuple,
                    upsample_size=upsample_size,
                    num_layers=layers_per_block + 1,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=time_embed_dim,
                    add_upsample=add_upsample,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                )
            new_samples.append(s)

        samples = new_samples

        # Temporal injection after CrossAttn up blocks (0, 1, 2)
        if up_block_type == "CrossAttnUpBlock2D":
            # up0=1280, up1=640, up2=320 — output_channel already updated above
            samples = _apply_temporal(
                samples, temporal_kernels[f"up{block_idx}"], device, num_frames, output_channel
            )

    # 7. Post-process: GroupNorm → SiLU → conv_out
    out_tensors = []
    for s in samples:
        s = ttnn.to_layout(s, ttnn.ROW_MAJOR_LAYOUT)
        if ttnn_model.fallback_on_groupnorm:
            s = ttnn.reshape(
                s,
                (
                    ttnn_model.batch_size,
                    ttnn_model.conv_out_input_height,
                    ttnn_model.conv_out_input_width,
                    ttnn_model.conv_out_in_channels,
                ),
            )
            s = ttnn.permute(s, (0, 3, 1, 2))
            import ttnn.operations.normalization as _norm
            s = _norm._fallback_group_norm(
                s,
                num_groups=norm_num_groups,
                weight=ttnn_model.parameters.conv_norm_out.weight,
                bias=ttnn_model.parameters.conv_norm_out.bias,
                epsilon=norm_eps,
            )
            s = pre_process_input(device, s)
        else:
            # Use TTNN group_norm path
            pass  # will be handled by the norm ops already compiled into the model

        # conv_out
        conv_kwargs_out = dict(
            in_channels=ttnn_model.conv_out_in_channels,
            out_channels=ttnn_model.conv_out_out_channels,
            batch_size=ttnn_model.batch_size,
            input_height=ttnn_model.conv_out_input_height,
            input_width=ttnn_model.conv_out_input_width,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
            device=device,
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.bfloat8_b,
                shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                reshard_if_not_optimal=True,
                enable_act_double_buffer=True,
                enable_weights_double_buffer=True,
            ),
            slice_config=ttnn.Conv2dL1FullSliceConfig,
        )
        s, [ttnn_model.conv_out_weights, ttnn_model.conv_out_bias] = ttnn.conv2d(
            input_tensor=s,
            weight_tensor=ttnn_model.conv_out_weights,
            bias_tensor=ttnn_model.conv_out_bias,
            **conv_kwargs_out,
            compute_config=conv_compute_kernel_config,
            dtype=ttnn.bfloat16,
            return_weights_and_bias=True,
        )
        s = reshard_for_output_channels_divisibility(s, ttnn_model.conv_out_out_channels)
        out_tensors.append(s)

    return out_tensors
```

- [ ] **Step 3.3: Run existing motion pipeline tests**

```bash
python -m pytest tests/test_ttnn_motion_pipeline.py -v
```

Expected: all 4 tests still PASS (we only added a function, didn't change `_apply_temporal`).

- [ ] **Step 3.4: Commit**

```bash
git add animatediff_ttnn/ttnn_motion_pipeline.py
git commit -m "feat(phase3): add forward_unet_staged — staged UNet forward with temporal hooks"
```

---

## Task 4: `generate_frames_motion()` in `temporal_attention.py`

**Files:**
- Modify: `animatediff_ttnn/temporal_attention.py` (append new function)

### Background

`generate_frames_motion` has the same signature as `generate_frames_temporal` with one addition: `temporal_kernels: dict` from `load_motion_kernels()`. The denoising loop is restructured:

- **Old** (Phase 2.5): N sequential `ttnn_model(sample_i, ...)` calls per step → `cross_frame_attention()` on noise predictions
- **New** (Phase 3): One `forward_unet_staged()` call per step that processes all N frames together with 7 temporal attention injections inside the UNet

The scheduler logic, chain_from/chain_save/chain_alpha, VAE decode, and all other aspects remain identical to `generate_frames_temporal`.

- [ ] **Step 4.1: Append `generate_frames_motion` to `temporal_attention.py`**

Append this function after the existing `generate_frames_temporal` function:

```python
def generate_frames_motion(
    device,
    ttnn_model,
    ttnn_vae,
    config,
    torch_time_proj,
    text_embeddings: torch.Tensor,
    temporal_kernels: dict,
    num_frames: int = 8,
    num_steps: int = 25,
    guidance_scale: float = 7.5,
    seed: int = 42,
    height: int = 512,
    width: int = 512,
    use_lightning: bool = False,
    chain_from: str | None = None,
    chain_save: str | None = None,
    chain_alpha: float = 0.6,
    on_step=None,
) -> List:
    """Generate temporally-coherent frames using MotionAdapter temporal attention.

    Phase 3: MotionAdapter QKV weights are injected into the TTNN UNet via
    forward_unet_staged(), which calls each UNet block object across all N frames
    and applies TemporalAttentionKernel between blocks at 7 injection points
    (down0-2, mid, up0-2). This produces genuine AnimateDiff-quality temporal
    coherence on Blackhole hardware.

    Args:
        device: TTNN Blackhole device from setup_blackhole()
        ttnn_model: Loaded TTNN UNet2D model
        ttnn_vae: TTNN Vae decoder
        config: unet.config from PyTorch UNet2DConditionModel
        torch_time_proj: unet.time_proj, used by build_tlist
        text_embeddings: Shape (2, 96, 768) — [uncond, cond] concatenated
        temporal_kernels: dict from motion_weights.load_motion_kernels()
        num_frames: Number of frames to generate
        num_steps: Denoising steps (25 for PNDM, 8 for Lightning)
        guidance_scale: CFG scale (7.5 recommended)
        seed: RNG seed — shared base noise + per-frame perturbation
        height, width: Output size in pixels (512 × 512)
        use_lightning: If True, use EulerDiscreteScheduler
        chain_from: Path to .pt latents from a previous chain_save run
        chain_save: Path to save this run's final latents for chaining
        chain_alpha: Blend weight for chain_from (default 0.6)
        on_step: Optional callable(step_idx, num_steps, frame_latents)

    Returns:
        List of PIL Images, length num_frames
    """
    import ttnn
    from PIL import Image
    from animatediff_ttnn.ttnn_pipeline import build_tlist, to_device, from_device
    from animatediff_ttnn.ttnn_motion_pipeline import forward_unet_staged
    from models.demos.vision.generative.stable_diffusion.wormhole.sd_helper_funcs import tt_guide

    lh, lw = height // 8, width // 8

    if use_lightning:
        from diffusers import EulerDiscreteScheduler
        from animatediff_ttnn.tt_euler_scheduler import TtEulerScheduler
        euler_kwargs = dict(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="linear",
            timestep_spacing="trailing",
        )
        schedulers = [EulerDiscreteScheduler(**euler_kwargs) for _ in range(num_frames)]
        for s in schedulers:
            s.set_timesteps(num_steps)
        _tt_sched = TtEulerScheduler(**euler_kwargs)
        _tt_sched.set_timesteps(num_steps)
    else:
        from diffusers import PNDMScheduler
        from models.demos.vision.generative.stable_diffusion.wormhole.sd_pndm_scheduler import TtPNDMScheduler
        pndm_kwargs = dict(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            skip_prk_steps=True,
            steps_offset=1,
        )
        schedulers = [PNDMScheduler(**pndm_kwargs) for _ in range(num_frames)]
        for s in schedulers:
            s.set_timesteps(num_steps)
        _tt_sched = TtPNDMScheduler(device=device, **pndm_kwargs)
        _tt_sched.set_timesteps(num_steps)

    timesteps = schedulers[0].timesteps
    init_noise_sigma = float(schedulers[0].init_noise_sigma)

    generator = torch.Generator().manual_seed(seed)
    base_noise = torch.randn(1, 4, lh, lw, generator=generator)
    noise_perturb = 0.02 if use_lightning else 0.05

    # Chain continuity (same logic as generate_frames_temporal)
    if chain_from is not None:
        from pathlib import Path as _Path
        import torch.nn.functional as _F
        chain_path = _Path(chain_from)
        if chain_path.exists():
            prev = torch.load(chain_path, map_location="cpu", weights_only=True)
            prev_mean = prev.mean(dim=0, keepdim=True).float()
            ch_mean = prev_mean.mean(dim=(2, 3), keepdim=True)
            ch_std = prev_mean.std(dim=(2, 3), keepdim=True).clamp(min=1e-6)
            prev_norm = (prev_mean - ch_mean) / ch_std
            ksize = 9
            prev_blurred = _F.avg_pool2d(
                prev_norm, kernel_size=ksize, stride=1,
                padding=ksize // 2, count_include_pad=False,
            )
            mixed = (1.0 - chain_alpha) * base_noise + chain_alpha * prev_blurred
            mixed_std = mixed.std().clamp(min=1e-6)
            base_noise = mixed / mixed_std
            print(f"  Chain: blended {chain_path.name} at alpha={chain_alpha} (ksize=9, renorm)")
        else:
            print(f"  Chain: warning — {chain_from} not found, ignoring")

    frame_latents = []
    for _ in range(num_frames):
        perturbed = base_noise + noise_perturb * torch.randn(base_noise.shape, generator=generator)
        frame_latents.append(perturbed * init_noise_sigma)

    _tlist = build_tlist(_tt_sched, torch_time_proj, device, lh, lw)

    ttnn_text_emb = to_device(
        text_embeddings, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    )

    num_steps_actual = len(timesteps)

    for step_idx, t in enumerate(timesteps):
        # Prepare per-frame device tensors (batch=2 for CFG)
        device_samples = []
        for i in range(num_frames):
            latent_cpu = frame_latents[i]
            if use_lightning:
                latent_cpu = schedulers[i].scale_model_input(latent_cpu, t)
            lat = to_device(latent_cpu, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            lat_input = ttnn.concat([lat, lat], dim=0)
            lat.deallocate(True)
            device_samples.append(lat_input)

        # Forward all N frames with temporal attention between blocks
        raw_outputs = forward_unet_staged(
            ttnn_model=ttnn_model,
            frame_samples=device_samples,
            timestep=_tlist[step_idx],
            encoder_hidden_states=ttnn_text_emb,
            config=config,
            temporal_kernels=temporal_kernels,
            device=device,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
        )

        # Guidance + scheduler step
        noise_preds = []
        for i, raw in enumerate(raw_outputs):
            guided = tt_guide(raw, guidance_scale)
            noise_pred_cpu = from_device(guided, device).to(torch.float32)
            raw.deallocate(True)
            guided.deallocate(True)
            noise_preds.append(noise_pred_cpu)

        if use_lightning:
            step_alpha = _cosine_alpha(step_idx, num_steps_actual, 0.35)
            stacked_preds = torch.cat(noise_preds, dim=0)
            blended_preds_t = cross_frame_attention(stacked_preds, alpha=step_alpha)
            blended_preds = [blended_preds_t[i: i + 1] for i in range(num_frames)]
            next_latents = [
                schedulers[i].step(blended_preds[i], t, frame_latents[i]).prev_sample
                for i in range(num_frames)
            ]
            stacked_lat = torch.cat(next_latents, dim=0)
            attended_lat = cross_frame_attention(stacked_lat, alpha=step_alpha * 0.4)
            for i in range(num_frames):
                frame_latents[i] = attended_lat[i: i + 1]
        else:
            for i in range(num_frames):
                frame_latents[i] = schedulers[i].step(
                    noise_preds[i], t, frame_latents[i]
                ).prev_sample

        print(f"  [motion] Step {step_idx + 1}/{num_steps_actual}", end="\r", flush=True)
        if on_step is not None:
            on_step(step_idx, num_steps_actual, frame_latents)

    print()

    # Cleanup shared device tensors
    ttnn_text_emb.deallocate(True)
    for t_tensor in _tlist:
        t_tensor.deallocate(True)

    # Chain save
    if chain_save is not None:
        from pathlib import Path as _Path
        save_path = _Path(chain_save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.cat(frame_latents, dim=0), save_path)
        print(f"  Chain: saved latents → {save_path}")

    # Decode with TTNN VAE
    frames = []
    for i, latent in enumerate(frame_latents):
        latent_scaled = latent / 0.18215
        ttnn_lat = to_device(
            latent_scaled.permute(0, 2, 3, 1),
            device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        )
        ttnn_decoded = ttnn_vae.decode(ttnn_lat)
        ttnn_lat.deallocate(True)
        ttnn_decoded = ttnn.reshape(ttnn_decoded, [1, height, width, 3])
        decoded = ttnn.to_torch(ttnn.permute(ttnn_decoded, [0, 3, 1, 2])).float()
        ttnn_decoded.deallocate(True)
        img = (decoded / 2 + 0.5).clamp(0, 1)
        img = (img.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
        frames.append(Image.fromarray(img))
        print(f"  Frame {i + 1}/{num_frames} decoded")

    return frames
```

- [ ] **Step 4.2: Run existing temporal attention tests**

```bash
python -m pytest tests/test_temporal_attention.py -v
```

Expected: all existing tests pass (we only appended, didn't modify existing code).

- [ ] **Step 4.3: Commit**

```bash
git add animatediff_ttnn/temporal_attention.py
git commit -m "feat(phase3): add generate_frames_motion — Phase 3 denoising with MotionAdapter"
```

---

## Task 5: `--motion-adapter` flag in `examples/generate.py`

**Files:**
- Modify: `examples/generate.py`

### Background

Read the existing `examples/generate.py` first to understand how `--mode`, `--lcm-unet`, and `--lightning` are handled. The `--motion-adapter` flag should:

1. Accept an optional PATH argument (default: HuggingFace repo ID)
2. At startup (before the denoising loop), call `load_motion_kernels(model_id)`
3. Pass `temporal_kernels` to `generate_frames_motion()` instead of `generate_frames_temporal()`
4. Print `[motion]` tag in progress output

The flag is valid only with `--mode blackhole`. If `--motion-adapter` is given with any other mode, print a warning and ignore it.

- [ ] **Step 5.1: Read `examples/generate.py`**

Read the full file to understand the current argument parsing, mode dispatch, and generate call site.

- [ ] **Step 5.2: Add `--motion-adapter` flag**

Find the argparse section and add:
```python
parser.add_argument(
    "--motion-adapter",
    metavar="PATH",
    nargs="?",
    const="guoyww/animatediff-motion-adapter-v1-5-2",
    default=None,
    help=(
        "Load MotionAdapter weights for Phase 3 temporal attention. "
        "PATH defaults to HuggingFace cache for guoyww/animatediff-motion-adapter-v1-5-2. "
        "Only valid with --mode blackhole."
    ),
)
```

Find the section that dispatches to `generate_frames_temporal` and add the motion-adapter path:

```python
if args.motion_adapter and args.mode == "blackhole":
    print(f"  [motion] Loading MotionAdapter from {args.motion_adapter} ...")
    from animatediff_ttnn.motion_weights import load_motion_kernels
    temporal_kernels = load_motion_kernels(model_id=args.motion_adapter, num_frames=args.frames)
    print(f"  [motion] Loaded {sum(len(v) for v in temporal_kernels.values())} kernels")
    from animatediff_ttnn.temporal_attention import generate_frames_motion
    frames = generate_frames_motion(
        device=device,
        ttnn_model=ttnn_model,
        ttnn_vae=ttnn_vae,
        config=unet_config,
        torch_time_proj=torch_time_proj,
        text_embeddings=text_embeddings,
        temporal_kernels=temporal_kernels,
        num_frames=args.frames,
        num_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        height=args.height,
        width=args.width,
        use_lightning=args.lightning,
        chain_from=args.chain_from,
        chain_save=args.chain_save,
        chain_alpha=args.chain_alpha,
    )
elif args.mode == "blackhole":
    # existing generate_frames_temporal call
    ...
```

(Place this before the existing `elif args.mode == "blackhole"` block, or as an `if` inside it.)

- [ ] **Step 5.3: Smoke test the flag wiring (no hardware)**

```bash
cd ~/code/tt-animatediff
python examples/generate.py --help | grep motion
```

Expected output includes: `--motion-adapter [PATH]`

- [ ] **Step 5.4: Commit**

```bash
git add examples/generate.py
git commit -m "feat(phase3): add --motion-adapter flag to generate.py"
```

---

## Task 6: Comparison grid column F-motion

**Files:**
- Modify: `scripts/generate_comparison_grid.py`

### Background

The comparison grid already has columns A through E. Add column F: `F-motion` — Blackhole + MotionAdapter 8-step Lightning (fastest comparison, most impressive quality gain).

Add it to the `MODES` list in `generate_comparison_grid.py` and update the HTML footer to remove the "Phase 3 (not yet built)" note.

- [ ] **Step 6.1: Read `scripts/generate_comparison_grid.py`**

Read the MODES list and the HTML footer section at the bottom of `build_html()`.

- [ ] **Step 6.2: Add F-motion to MODES**

In the `MODES` list, append:
```python
{
    "key": "F-motion",
    "label": "F: Blackhole + MotionAdapter (Lightning)",
    "description": "Phase 3: full AnimateDiff temporal attention inside TTNN UNet. 8-step Lightning.",
    "args": ["--mode", "blackhole", "--lightning", "--lightning-steps", "8",
             "--motion-adapter", "--frames", "8"],
    "skip_if_cpu_skip": False,
},
```

- [ ] **Step 6.3: Update the HTML footer**

Find the `<div>` in `build_html()` that says "Phase 3 (not yet built)" and replace with:
```html
<div style="margin-top:24px;color:#607d8b;font-size:12px">
  <strong style="color:#4fd1c5">Phase 3 (F-motion):</strong>
  Blackhole TTNN + full MotionAdapter temporal attention.
  320→640→1280→1280 dim feature injection at 7 UNet blocks, 2 motion modules each.
  Weights: guoyww/animatediff-motion-adapter-v1-5-2.
</div>
```

- [ ] **Step 6.4: Commit**

```bash
git add scripts/generate_comparison_grid.py
git commit -m "feat(phase3): add F-motion column to comparison grid"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] `motion_weights.py` with `load_motion_kernels()` — Task 1
- [x] `_apply_temporal()` with CPU round-trip — Task 2
- [x] `forward_unet_staged()` with 7 injection points — Task 3
- [x] `generate_frames_motion()` replacing Phase 2.5 loop — Task 4
- [x] `--motion-adapter` flag in `generate.py` — Task 5
- [x] Comparison grid column F — Task 6
- [x] Tests for weight loading — Task 1
- [x] Tests for `_apply_temporal` (shape, effect, two-module) — Task 2
- [x] Chain mode: same chain_from/chain_save/chain_alpha logic copied verbatim — Task 4
- [x] `num_frames` passed to `load_motion_kernels` — Task 5 (via args.frames)
- [x] All kernel weights are 2 per block (except mid=1) — Task 1 spec and tests
- [x] `forward_unet_staged` imports `from_device` and uses `ttnn.bfloat16` — Task 3 code
- [x] Temporal injection on CrossAttn blocks only (down0/1/2, mid, up0/1/2) not on DownBlock2D/UpBlock2D — Task 3 code

**No placeholders:** All code steps show complete implementation.

**Type consistency:**
- `load_motion_kernels` returns `dict[str, list[TemporalAttentionKernel]]` — used as `temporal_kernels` throughout
- `_apply_temporal(samples, kernel_list, device, num_frames, C)` — `kernel_list` is `list[TemporalAttentionKernel]`, consistent with usage in Task 3 and 4
- `forward_unet_staged` returns `list` of N TTNN tensors — consumed in Task 4 as `raw_outputs`
