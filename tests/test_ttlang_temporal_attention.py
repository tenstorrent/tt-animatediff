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
    """use_ttlang=False forward matches manual reference implementation.

    _make_weights() returns [C, C] matrices that simulate nn.Linear storage
    (i.e. [out_features, in_features]).  load_weights() transposes them on
    load, so the reference must also transpose before the @ operator.
    """
    w_q, w_k, w_v, w_o = _make_weights(C)
    mod = TemporalAttentionKernel(dim=C, num_frames=N, use_ttlang=False)
    mod.load_weights(w_q, w_k, w_v, w_o)
    torch.manual_seed(1)
    x = torch.randn(S, N, C)

    # Reference uses transposed weights, matching nn.Linear convention.
    scale = C ** -0.5
    q = x.float() @ w_q.T
    k = x.float() @ w_k.T
    v = x.float() @ w_v.T
    scores = (q @ k.transpose(-1, -2)) * scale   # [S, N, N]
    attn = torch.softmax(scores, dim=-1)
    attn_out = attn @ v                           # [S, N, C]
    ref = x.float() + attn_out @ w_o.T           # [S, N, C]

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


def _pcc(a, b):
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return torch.corrcoef(torch.stack([a_f, b_f]))[0, 1].item()


def test_qkv_kernel_pcc():
    """Sim QKV projection matches float32 reference to PCC > 0.999."""
    from animatediff_ttnn.ttlang.temporal_attention_kernel import _qkv_kernel_sim

    torch.manual_seed(42)
    SN = S_ROWS * N_TILES   # tile-rows (S spatial rows × N frame-tiles)
    x_t   = torch.randn(SN * TILE, C_TILES * TILE)
    w_q_t = torch.randn(C_TILES * TILE, C_TILES * TILE) * 0.02
    w_k_t = torch.randn(C_TILES * TILE, C_TILES * TILE) * 0.02
    w_v_t = torch.randn(C_TILES * TILE, C_TILES * TILE) * 0.02

    q_ref = x_t @ w_q_t
    k_ref = x_t @ w_k_t
    v_ref = x_t @ w_v_t

    q_sim, k_sim, v_sim = _qkv_kernel_sim(
        x_t, w_q_t, w_k_t, w_v_t, sn_tiles=SN, c_tiles=C_TILES
    )

    assert _pcc(q_sim, q_ref) > 0.999, f"Q PCC {_pcc(q_sim, q_ref):.4f}"
    assert _pcc(k_sim, k_ref) > 0.999, f"K PCC {_pcc(k_sim, k_ref):.4f}"
    assert _pcc(v_sim, v_ref) > 0.999, f"V PCC {_pcc(v_sim, v_ref):.4f}"


def test_sdpa_kernel_pcc():
    """Sim SDPA matches float32 reference to PCC > 0.999."""
    from animatediff_ttnn.ttlang.temporal_attention_kernel import _sdpa_kernel_sim

    torch.manual_seed(7)
    SN = S_ROWS * N_TILES
    scale = (C_TILES * TILE) ** -0.5
    q_t = torch.randn(SN * TILE, C_TILES * TILE)
    k_t = torch.randn(SN * TILE, C_TILES * TILE)
    v_t = torch.randn(SN * TILE, C_TILES * TILE)

    # Reference: reshape to [S, N, C], compute attention, flatten back.
    q = q_t.reshape(S_ROWS, N_TILES * TILE, C_TILES * TILE)
    k = k_t.reshape(S_ROWS, N_TILES * TILE, C_TILES * TILE)
    v = v_t.reshape(S_ROWS, N_TILES * TILE, C_TILES * TILE)
    scores = (q @ k.transpose(-1, -2)) * scale  # [S, N, N]
    attn = torch.softmax(scores, dim=-1)
    ref = (attn @ v).reshape(SN * TILE, C_TILES * TILE)  # [SN*TILE, C*TILE]

    sim_out = _sdpa_kernel_sim(q_t, k_t, v_t, s_rows=S_ROWS, n_tiles=N_TILES, c_tiles=C_TILES)

    pcc = _pcc(sim_out, ref)
    assert pcc > 0.999, f"SDPA sim PCC {pcc:.6f} < 0.999"


def test_out_proj_kernel_pcc():
    """Sim out-proj + residual matches float32 reference to PCC > 0.999."""
    from animatediff_ttnn.ttlang.temporal_attention_kernel import _out_proj_kernel_sim

    torch.manual_seed(11)
    SN = S_ROWS * N_TILES
    attn_out_t = torch.randn(SN * TILE, C_TILES * TILE)
    x_res_t    = torch.randn(SN * TILE, C_TILES * TILE)
    w_o_t      = torch.randn(C_TILES * TILE, C_TILES * TILE) * 0.02

    ref = x_res_t + attn_out_t @ w_o_t

    sim_out = _out_proj_kernel_sim(
        attn_out_t, x_res_t, w_o_t,
        s_rows=S_ROWS, n_tiles=N_TILES, c_tiles=C_TILES
    )

    pcc = _pcc(sim_out, ref)
    assert pcc > 0.999, f"out_proj sim PCC {pcc:.6f} < 0.999"


def test_full_forward_ttlang():
    """End-to-end use_ttlang=True matches use_ttlang=False to PCC > 0.999."""
    w_q, w_k, w_v, w_o = _make_weights(C)

    mod_py = TemporalAttentionKernel(dim=C, num_frames=N, use_ttlang=False)
    mod_py.load_weights(w_q, w_k, w_v, w_o)

    mod_tt = TemporalAttentionKernel(dim=C, num_frames=N, use_ttlang=True)
    mod_tt.load_weights(w_q, w_k, w_v, w_o)

    torch.manual_seed(3)
    x = torch.randn(S, N, C)

    ref = mod_py.forward(x)
    out = mod_tt.forward(x)

    assert out.shape == ref.shape, f"shape mismatch: {out.shape} vs {ref.shape}"
    pcc = _pcc(out, ref)
    assert pcc > 0.999, f"full ttlang forward PCC {pcc:.6f} < 0.999"
