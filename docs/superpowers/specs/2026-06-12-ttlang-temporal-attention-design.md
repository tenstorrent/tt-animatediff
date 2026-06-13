# TT-Lang Temporal Attention Kernel — Design Spec

## Goal

Write a fused AnimateDiff temporal attention module in TT-Lang DSL, verified in
the functional simulator, that replaces the CPU `cross_frame_attention()` hotspot
with a Tenstorrent-native kernel operating on real UNet feature dimensions. Scope
ends at a simulator-verified, hardware-ready wrapper; UNet surgery (injection into
the live TTNN forward pass) is a separate, future step.

## Architecture

### New directory: `animatediff_ttnn/ttlang/`

Lives alongside the existing `animatediff_ttnn/` package. No existing files are
modified.

```
animatediff_ttnn/ttlang/
    __init__.py                      # exports TemporalAttentionKernel
    temporal_attention_kernel.py     # three @ttl.operation kernels + wrapper class
    sim_helpers.py                   # tensor_to_block / block_to_tensor utilities

tests/
    test_ttlang_temporal_attention.py   # simulator-only, pure pytest, no hardware
```

### What this replaces (eventually)

The existing `temporal_attention.py::cross_frame_attention()` operates on UNet
**outputs** (noise predictions, shape `[N, C, H, W]` with C=4). That is Phase 2.5 —
a CPU approximation.

This kernel targets the **AnimateDiff motion module** attention that runs *inside*
the UNet between spatial ResBlocks, operating on UNet **intermediate features**:

```
Input shape: [S, N, C]
  S = spatial positions  (e.g. 4096 = 64×64 at shallowest UNet depth)
  N = frames             (e.g. 8)
  C = channel dim        (320 / 640 / 1280 depending on UNet depth)
```

This is real temporal coherence — the same mechanism AnimateDiff uses in its
motion modules with the MotionAdapter — but written as a custom TT-Lang kernel
rather than dispatched through the PyTorch MotionAdapter weights.

---

## Tile dimensions

All kernels use 32×32 tiles (standard Tenstorrent tile size).

```python
TILE_SIZE = 32
N_TILES   = 1    # N=8 frames padded to one 32-wide tile
C_TILES   = 10   # C=320 / 32  (shallowest motion module)
```

S loops row-by-row: each kernel iteration processes one `[1, N_TILES, C_TILES]`
slice, keeping L1 usage under ~500 KB (same streaming strategy as
`skyreels-ttlang/kernels/rmsnorm_skyreels`).

For deeper UNet layers (C=640, C=1280): C_TILES scales to 20 / 40. The tile loop
structure is identical; only C_TILES changes.

---

## Kernel signatures

### Kernel 1 — QKV projection: `temporal_qkv_kernel`

```python
@ttl.operation(grid=(1, 1))
def temporal_qkv_kernel(x, w_q, w_k, w_v, q_out, k_out, v_out):
    """Project [S, N, C] → Q, K, V each [S, N, C].

    Streams one [1, N_TILES, C_TILES] row per iteration.
    w_q/w_k/w_v are [C_TILES, C_TILES] weight matrices loaded once into L1.
    """
```

DFB layout per iteration:
- `x_row_dfb`   (1, C_TILES) — input row, block_count=2
- `w_q/k/v_dfb` (C_TILES, C_TILES) — weight, block_count=1 (loaded once, reused)
- `q/k/v_dfb`   (1, C_TILES) — output rows, block_count=2

### Kernel 2 — Scaled dot-product attention: `temporal_sdpa_kernel`

```python
@ttl.operation(grid=(1, 1))
def temporal_sdpa_kernel(q, k, v, scale, attn_out):
    """Self-attention across N frames at each spatial position.

    For each spatial row s:
      scores = q[s] @ k[s].T * scale     [N_TILES, N_TILES]
      attn   = softmax(scores)            [N_TILES, N_TILES]
      out[s] = attn @ v[s]               [N_TILES, C_TILES]
    """
```

DFB layout per spatial row:
- `q_row_dfb`, `k_row_dfb`, `v_row_dfb`  (1, C_TILES) — block_count=2
- `k_t_dfb`                               (C_TILES, 1) — transposed K
- `scores_dfb`                            (1, 1) — attention logits [N×N, padded]
- `scale_dfb`                             (1, 1) — scalar scale tile
- `softmax_dfb`                           (1, 1) — attention weights
- `out_row_dfb`                           (1, C_TILES) — block_count=2

The attention matrix is `[N, N]` = `[8, 8]` — fits entirely in a single 32×32
tile. No tiling needed on the sequence dimension.

### Kernel 3 — Output projection + residual: `temporal_out_proj_kernel`

```python
@ttl.operation(grid=(1, 1))
def temporal_out_proj_kernel(attn_out, x_residual, w_o, out):
    """out = x_residual + attn_out @ w_o

    Streams one [1, N_TILES, C_TILES] row per iteration.
    """
```

DFB layout per row:
- `attn_dfb`, `res_dfb`    (1, C_TILES) — block_count=2
- `w_o_dfb`                (C_TILES, C_TILES) — loaded once
- `proj_dfb`               (1, C_TILES) — intermediate
- `out_dfb`                (1, C_TILES) — block_count=2

---

## Wrapper class

```python
class TemporalAttentionKernel:
    """AnimateDiff temporal attention module backed by TT-Lang sim kernels.

    Args:
        dim:         Channel dimension C (320, 640, or 1280).
        num_frames:  Number of frames N (default 8).
        use_ttlang:  If True, dispatch through sim kernels.
                     If False, use pure-PyTorch reference (default).
    """

    def __init__(self, dim: int, num_frames: int = 8, use_ttlang: bool = False): ...

    def load_weights(self, w_q, w_k, w_v, w_o): ...
    # Accepts [C, C] PyTorch tensors; converts to simulator Blocks.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [S, N, C]  →  out: [S, N, C]
        # Both paths produce the same result to PCC > 0.999.
```

The PyTorch reference path implements:
```python
q = x @ w_q; k = x @ w_k; v = x @ w_v
scale = C ** -0.5
scores = (q @ k.transpose(-1, -2)) * scale   # [S, N, N]
attn = softmax(scores, dim=-1)
out = attn @ v                                # [S, N, C]
return x + out @ w_o                          # residual
```

---

## Simulator setup

Uses `from sim import ttl` and `from sim.dfb import Block` — identical to
`tt-lang/examples/wan_rmsnorm.py`. Requires:

```bash
source /home/ttuser/code/tt-lang/build/env/activate
PYTHONPATH=/home/ttuser/code/tt-lang/python pytest tests/test_ttlang_temporal_attention.py -v
```

The test file itself must include a `sys.path.insert` guard so it works without
manual env vars when run from the repo root:

```python
import sys, os
TT_LANG_PYTHON = os.path.expanduser("~/code/tt-lang/python")
if TT_LANG_PYTHON not in sys.path:
    sys.path.insert(0, TT_LANG_PYTHON)
```

No hardware. No `ttnn.open_device()`. No `TT_METAL_ARCH_NAME`.

---

## Test plan

All tests in `tests/test_ttlang_temporal_attention.py`:

1. **`test_sim_helpers_roundtrip`** — `tensor_to_block → block_to_tensor` is lossless.
2. **`test_qkv_kernel_pcc`** — PCC(sim Q, ref Q) > 0.999 at (S=64, N=8, C=320).
3. **`test_sdpa_kernel_pcc`** — PCC(sim attn_out, ref attn_out) > 0.999.
4. **`test_out_proj_kernel_pcc`** — PCC(sim out, ref out) > 0.999.
5. **`test_full_forward_pytorch`** — `use_ttlang=False` path, trivially correct.
6. **`test_full_forward_ttlang`** — `use_ttlang=True` end-to-end, PCC vs PyTorch > 0.999.
7. **`test_wrapper_shape_preserved`** — output shape matches input shape.
8. **`test_wrapper_n1_passthrough`** — N=1 frame: attention is trivially identity.
9. **`test_dim_640`** — same structure at C=640 (N_TILES=1, C_TILES=20).

---

## What is NOT in scope

- UNet surgery / injection into `ttnn_pipeline.py` or the TTNN UNet forward pass
- Hardware dispatch (`ttnn.open_device`, real Blackhole chips)
- Weight loading from actual AnimateDiff MotionAdapter checkpoint
- Multi-chip sharding (`(1,4)` mesh, `ShardTensorToMesh`)
- Training / distillation

These are all explicitly Phase 4 and beyond.

---

## File structure summary

| File | Status | Notes |
|------|--------|-------|
| `animatediff_ttnn/ttlang/__init__.py` | Create | exports `TemporalAttentionKernel` |
| `animatediff_ttnn/ttlang/sim_helpers.py` | Create | `tensor_to_block`, `block_to_tensor` |
| `animatediff_ttnn/ttlang/temporal_attention_kernel.py` | Create | 3 kernels + wrapper |
| `tests/test_ttlang_temporal_attention.py` | Create | 9 simulator tests |
| `animatediff_ttnn/temporal_attention.py` | **Unchanged** | existing Phase 2.5 code |
| `animatediff_ttnn/ttnn_pipeline.py` | **Unchanged** | existing Phase 2 pipeline |

---

## Integration handoff (future)

When hardware integration is ready, the injection point in `ttnn_pipeline.py` is:

```python
# After each spatial ResBlock in the UNet forward pass, at the motion module
# injection points: down_block depths 1, 2 and up_block depths 1, 2
# (UNet blocks with channel dims 320, 640, 1280, 640, 320).
temporal_attn = TemporalAttentionKernel(dim=C, use_ttlang=True)
features = temporal_attn.forward(features)  # [S, N, C] → [S, N, C]
```

The wrapper's `use_ttlang` flag means hardware validation can proceed
incrementally: start with `use_ttlang=False` (PyTorch, known correct) and flip
to `True` one injection point at a time.
