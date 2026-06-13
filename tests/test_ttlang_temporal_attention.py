# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for TT-Lang temporal attention kernel (simulator-only, no hardware)."""
import sys
import os

_TT_LANG_PYTHON = os.path.expanduser("~/code/tt-lang/python")
if _TT_LANG_PYTHON not in sys.path:
    sys.path.insert(0, _TT_LANG_PYTHON)

import pytest
import torch

pytest.importorskip("sim.ttnnsim")  # skip whole file if tt-lang sim not installed


def test_sim_helpers_roundtrip():
    """tensor_to_block → block_to_tensor is lossless."""
    from animatediff_ttnn.ttlang.sim_helpers import tensor_to_block, block_to_tensor

    TILE = 32
    t = torch.randn(2 * TILE, 3 * TILE)
    block = tensor_to_block(t, shape=(2, 3))
    out = block_to_tensor(block)
    assert out.shape == t.shape
    assert torch.allclose(out, t, atol=1e-5)


from animatediff_ttnn.ttlang.temporal_attention_kernel import TemporalAttentionKernel

TILE = 32
N_TILES = 1   # 8 frames → 1 tile of 32
C_TILES = 10  # 320/32
S_ROWS  = 4   # 4 × 32 = 128 spatial positions (small for fast tests)

S = S_ROWS * TILE
N = N_TILES * TILE  # 32 (8 real frames, 24 padded to zero)
C = C_TILES * TILE  # 320


def _make_weights(C):
    """Random QKV+out weights, [C, C] each."""
    torch.manual_seed(0)
    return (
        torch.randn(C, C) * 0.02,
        torch.randn(C, C) * 0.02,
        torch.randn(C, C) * 0.02,
        torch.randn(C, C) * 0.02,
    )


def test_wrapper_shape_preserved():
    """Output shape == input shape for use_ttlang=False."""
    w_q, w_k, w_v, w_o = _make_weights(C)
    mod = TemporalAttentionKernel(dim=C, num_frames=N, use_ttlang=False)
    mod.load_weights(w_q, w_k, w_v, w_o)
    x = torch.randn(S, N, C)
    out = mod.forward(x)
    assert out.shape == (S, N, C)


def test_wrapper_n1_passthrough():
    """N=1 (single frame): shape is preserved."""
    w_q, w_k, w_v, w_o = _make_weights(C)
    mod = TemporalAttentionKernel(dim=C, num_frames=1, use_ttlang=False)
    mod.load_weights(w_q, w_k, w_v, w_o)
    x = torch.randn(S, 1, C)
    out = mod.forward(x)
    assert out.shape == (S, 1, C)


def test_full_forward_pytorch():
    """use_ttlang=False forward matches manual reference implementation."""
    w_q, w_k, w_v, w_o = _make_weights(C)
    mod = TemporalAttentionKernel(dim=C, num_frames=N, use_ttlang=False)
    mod.load_weights(w_q, w_k, w_v, w_o)
    torch.manual_seed(1)
    x = torch.randn(S, N, C)

    # Manual reference
    scale = C ** -0.5
    q = x.float() @ w_q
    k = x.float() @ w_k
    v = x.float() @ w_v
    scores = (q @ k.transpose(-1, -2)) * scale   # [S, N, N]
    attn = torch.softmax(scores, dim=-1)
    attn_out = attn @ v                           # [S, N, C]
    ref = x.float() + attn_out @ w_o             # [S, N, C]

    out = mod.forward(x)
    pcc = torch.corrcoef(torch.stack([out.float().flatten(), ref.float().flatten()]))[0, 1].item()
    assert pcc > 0.9999, f"PyTorch path PCC {pcc:.6f} < 0.9999"


def test_dim_640():
    """TemporalAttentionKernel works at C=640 (deeper UNet layer)."""
    C2 = 640
    w_q2, w_k2, w_v2, w_o2 = _make_weights(C2)
    mod = TemporalAttentionKernel(dim=C2, num_frames=N, use_ttlang=False)
    mod.load_weights(w_q2, w_k2, w_v2, w_o2)
    x = torch.randn(S, N, C2)
    out = mod.forward(x)
    assert out.shape == (S, N, C2)
