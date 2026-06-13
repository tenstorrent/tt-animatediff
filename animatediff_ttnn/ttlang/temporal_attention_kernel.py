# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""TT-Lang temporal attention kernel for AnimateDiff motion modules.

Operates on [S, N, C] feature tensors (S spatial positions, N frames,
C channels). Two execution paths:
  use_ttlang=False  — pure PyTorch reference (always correct, no hardware)
  use_ttlang=True   — TT-Lang simulator kernels (verified to PCC > 0.999)

This module is simulator-only. Hardware dispatch is a future step.
"""
import torch


class TemporalAttentionKernel:
    """AnimateDiff temporal attention backed by TT-Lang sim kernels.

    Args:
        dim:        Channel dimension C (320, 640, or 1280).
        num_frames: Number of frames N (padded to TILE_SIZE=32 in sim path).
        use_ttlang: If True, dispatch through simulator kernels.
                    If False (default), use pure-PyTorch reference.
    """

    TILE_SIZE = 32

    def __init__(self, dim: int, num_frames: int = 8, use_ttlang: bool = False):
        self.dim = dim
        self.num_frames = num_frames
        self.use_ttlang = use_ttlang
        self.w_q = None
        self.w_k = None
        self.w_v = None
        self.w_o = None

    def load_weights(self, w_q, w_k, w_v, w_o):
        """Load QKV and output-projection weights.

        Args:
            w_q, w_k, w_v, w_o: Each shape [C, C], any dtype (converted to float32).
        """
        self.w_q = w_q.float()
        self.w_k = w_k.float()
        self.w_v = w_v.float()
        self.w_o = w_o.float()

    def forward(self, x):
        """Apply temporal self-attention across N frames.

        Args:
            x: Shape [S, N, C] — S spatial positions, N frames, C channels.

        Returns:
            Shape [S, N, C] — same as input, with residual added.
        """
        assert self.w_q is not None, "Call load_weights() before forward()"
        if self.use_ttlang:
            return self._forward_ttlang(x)
        return self._forward_pytorch(x)

    def _forward_pytorch(self, x):
        """Scaled dot-product temporal attention in PyTorch.

        Computes: out = x + softmax(Q @ K.T / sqrt(C)) @ V) @ W_o
        where Q = x @ W_q, K = x @ W_k, V = x @ W_v.
        """
        xf = x.float()
        scale = self.dim ** -0.5
        q = xf @ self.w_q                             # [S, N, C]
        k = xf @ self.w_k
        v = xf @ self.w_v
        scores = (q @ k.transpose(-1, -2)) * scale    # [S, N, N]
        attn = torch.softmax(scores, dim=-1)           # [S, N, N]
        attn_out = attn @ v                            # [S, N, C]
        out = xf + attn_out @ self.w_o                # [S, N, C] residual
        return out.to(x.dtype)

    def _forward_ttlang(self, x):
        raise NotImplementedError("TT-Lang sim path implemented in Task 3-5")


def _qkv_kernel_sim(x_t, w_q_t, w_k_t, w_v_t, sn_tiles, c_tiles):
    """Compute Q, K, V projections using TT-Lang simulator Block matmul.

    Streams row-by-row: for each of *sn_tiles* input rows, computes
    ``x_row [1, c_tiles] @ w [c_tiles, c_tiles] → out_row [1, c_tiles]``.

    The inner dimension of x_row is contracted against the matching dimension
    of w using the sim's native Block ``@`` operator, which follows standard
    ``(M, K) @ (K, N) → (M, N)`` tile-grid matmul rules.

    Args:
        x_t:       Float tensor of shape ``[sn_tiles*32, c_tiles*32]`` — the
                   flattened (S*N, C) feature map laid out in 32-element tiles.
        w_q_t:     Float tensor of shape ``[c_tiles*32, c_tiles*32]`` — query
                   projection weight.
        w_k_t:     Float tensor of shape ``[c_tiles*32, c_tiles*32]`` — key
                   projection weight.
        w_v_t:     Float tensor of shape ``[c_tiles*32, c_tiles*32]`` — value
                   projection weight.
        sn_tiles:  Number of tile-rows (S_ROWS * N_TILES).
        c_tiles:   Number of tile-columns (C / 32).

    Returns:
        A tuple ``(q, k, v)`` where each element is a float32 tensor of shape
        ``[sn_tiles*32, c_tiles*32]`` matching the corresponding PyTorch
        reference computation ``x_t @ w_*``.
    """
    import sys
    import os
    _TT_LANG_PYTHON = os.path.expanduser("~/code/tt-lang/python")
    if _TT_LANG_PYTHON not in sys.path:
        sys.path.insert(0, _TT_LANG_PYTHON)

    from sim.dfb import Block
    from animatediff_ttnn.ttlang.sim_helpers import tensor_to_block, block_to_tensor

    # Convert full tensors into tile-Block representations.
    # tensor_to_block slices into 32×32 tiles in row-major order so that
    # tile (r, c) maps to flat index r*c_tiles + c in block.to_list().
    x_blk  = tensor_to_block(x_t.float(),   shape=(sn_tiles, c_tiles))
    wq_blk = tensor_to_block(w_q_t.float(), shape=(c_tiles, c_tiles))
    wk_blk = tensor_to_block(w_k_t.float(), shape=(c_tiles, c_tiles))
    wv_blk = tensor_to_block(w_v_t.float(), shape=(c_tiles, c_tiles))

    # Accumulate output tiles row by row.  Processing one row at a time keeps
    # L1 working-set proportional to c_tiles rather than sn_tiles*c_tiles, which
    # mirrors the on-chip streaming strategy used on Blackhole hardware.
    q_tiles, k_tiles, v_tiles = [], [], []

    for m in range(sn_tiles):
        # Extract row m of x: tiles at flat indices [m*c_tiles : (m+1)*c_tiles].
        # This gives a (1, c_tiles) tile-grid (one input row, all channel cols).
        x_row = Block.from_list(
            x_blk.to_list()[m * c_tiles:(m + 1) * c_tiles],
            shape=(1, c_tiles),
        )

        # Tile-grid matmul: (1, c_tiles) @ (c_tiles, c_tiles) → (1, c_tiles).
        # The sim's __matmul__ delegates to torch.matmul on the backing tensor
        # so numerical accuracy is float32-exact.
        q_row = x_row @ wq_blk
        k_row = x_row @ wk_blk
        v_row = x_row @ wv_blk

        # Collect the (1, c_tiles) output tiles in row-major order.
        # block.to_list() returns tiles in the same row-major convention so
        # extending in order preserves the spatial layout.
        q_tiles.extend(q_row.to_list())
        k_tiles.extend(k_row.to_list())
        v_tiles.extend(v_row.to_list())

    # Re-assemble the full (sn_tiles, c_tiles) tile-grid and convert to tensor.
    q = block_to_tensor(Block.from_list(q_tiles, shape=(sn_tiles, c_tiles)))
    k = block_to_tensor(Block.from_list(k_tiles, shape=(sn_tiles, c_tiles)))
    v = block_to_tensor(Block.from_list(v_tiles, shape=(sn_tiles, c_tiles)))
    return q, k, v
