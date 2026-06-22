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
            w_q, w_k, w_v, w_o: Each shape [C, C] from nn.Linear (stored as
                [out_features, in_features]).  Transposed here so the forward
                pass can compute ``x @ w`` (not ``x @ w.T``).
        """
        # nn.Linear stores weights as [out, in]; transpose to [in, out] so that
        # x @ self.w_{q,k,v,o} matches the correct linear-projection direction.
        self.w_q = w_q.float().T.contiguous()
        self.w_k = w_k.float().T.contiguous()
        self.w_v = w_v.float().T.contiguous()
        self.w_o = w_o.float().T.contiguous()

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
        """Full TT-Lang sim path: QKV proj → SDPA → out proj + residual.

        Reshapes the 3-D input [S, N, C] to the 2-D tile-kernel layout
        [S*N, C], runs each stage through the simulator Block ops, then
        reshapes the result back to [S, N, C].

        Args:
            x: Float or half tensor of shape [S, N, C].

        Returns:
            Tensor of shape [S, N, C] matching _forward_pytorch to PCC > 0.999.
        """
        S, N, C = x.shape
        TILE = self.TILE_SIZE

        # Both N and C must be tile-aligned for the Block matmul kernels.
        assert N % TILE == 0, f"N={N} must be multiple of TILE={TILE}"
        assert C % TILE == 0, f"C={C} must be multiple of TILE={TILE}"

        n_tiles = N // TILE    # frame tiles (typically 1 when N=32)
        c_tiles = C // TILE    # channel tiles (e.g. 10 for C=320)
        # Each spatial position is one "row" of tile-rows in the 2-D layout.
        # The flattened layout has s_rows * n_tiles tile-rows total.
        s_rows = S

        xf = x.float()
        # Flatten spatial + frame dims: [S, N, C] → [S*N, C].
        # Kernel convention: sn_tiles = s_rows * n_tiles tile-rows of height TILE.
        x_2d = xf.reshape(S * N, C)

        # Stage 1: QKV projection — x_2d @ w_{q,k,v}
        q_2d, k_2d, v_2d = _qkv_kernel_sim(
            x_2d, self.w_q, self.w_k, self.w_v,
            sn_tiles=s_rows * n_tiles, c_tiles=c_tiles,
        )

        # Stage 2: Scaled dot-product attention per spatial position.
        attn_out_2d = _sdpa_kernel_sim(
            q_2d, k_2d, v_2d,
            s_rows=s_rows, n_tiles=n_tiles, c_tiles=c_tiles,
        )

        # Stage 3: Output projection + residual: out = x_2d + attn_out @ w_o
        out_2d = _out_proj_kernel_sim(
            attn_out_2d, x_2d, self.w_o,
            s_rows=s_rows, n_tiles=n_tiles, c_tiles=c_tiles,
        )

        # Restore [S, N, C] shape and cast back to the input dtype.
        return out_2d.reshape(S, N, C).to(x.dtype)


def _sdpa_kernel_sim(q_t, k_t, v_t, s_rows, n_tiles, c_tiles):
    """Scaled dot-product attention using TT-Lang simulator Block ops.

    For each spatial position s:
      scores = q[s] @ k[s].T * scale     [n_tiles, n_tiles] tile-grid
      attn   = softmax(scores)            [n_tiles, n_tiles] (stable: subtract row-max)
      out[s] = attn @ v[s]               [n_tiles, c_tiles] tile-grid

    Stable softmax pattern:
      1. row_max  = reduce_max(scores, scaler, dims=[1])      → (n_tiles, 1)
      2. max_bcast= broadcast(row_max, output_hint=scores, dims=[1]) → (n_tiles, n_tiles)
      3. exp_s    = exp(scores - max_bcast)                   → (n_tiles, n_tiles)
      4. row_sum  = reduce_sum(exp_s, scaler, dims=[1])       → (n_tiles, 1)
      5. sum_bcast= broadcast(row_sum, output_hint=exp_s, dims=[1]) → (n_tiles, n_tiles)
      6. attn     = exp_s * recip(sum_bcast)                  → (n_tiles, n_tiles)

    K transpose is handled by re-ordering the tile list and transposing each
    tile's backing data.  Tile (r, c) of k becomes tile (c, r) of k^T with
    transposed data, giving the correct (c_tiles, n_tiles) block.

    The scaler block is a single (1, 1) tile of 1.0s, required by reduce_max
    and reduce_sum — it acts as a multiplicative identity (scale factor = 1).

    Args:
        q_t, k_t, v_t: Float32 tensors [s_rows*n_tiles*32, c_tiles*32]
                        — output of _qkv_kernel_sim, flattened (S*N, C).
        s_rows:   Number of spatial rows (S).
        n_tiles:  Frame tiles (N/32, typically 1).
        c_tiles:  Channel tiles (C/32, typically 10).

    Returns:
        attn_out: Float32 tensor [s_rows*n_tiles*32, c_tiles*32]
                  matching PyTorch reference: softmax(Q@K.T/sqrt(C)) @ V.
    """
    import sys
    import os
    _TT_LANG_PYTHON = os.path.expanduser("~/code/tt-lang/python")
    if _TT_LANG_PYTHON not in sys.path:
        sys.path.insert(0, _TT_LANG_PYTHON)

    from sim.dfb import Block
    from sim.ttnnsim import Tensor
    import sim.math as ttl_math
    from animatediff_ttnn.ttlang.sim_helpers import tensor_to_block, block_to_tensor

    # Scale factor: 1 / sqrt(C) where C = c_tiles * 32.
    scale = (c_tiles * 32) ** -0.5

    # sn_tiles: total tile-rows in the flattened (S*N) layout.
    sn_tiles = s_rows * n_tiles

    # Convert full QKV tensors to tile-Block representations.
    # Each has shape (sn_tiles, c_tiles) in the tile grid.
    q_blk = tensor_to_block(q_t.float(), shape=(sn_tiles, c_tiles))
    k_blk = tensor_to_block(k_t.float(), shape=(sn_tiles, c_tiles))
    v_blk = tensor_to_block(v_t.float(), shape=(sn_tiles, c_tiles))

    # One-tile scaler block of all 1.0s: required by reduce_max / reduce_sum.
    # Shape (1, 1) in the tile grid — a single 32×32 tile filled with ones.
    scaler_tile = Tensor(torch.ones(32, 32, dtype=torch.float32))
    scaler_blk = Block.from_list([scaler_tile], shape=(1, 1))

    out_tiles = []  # accumulates (n_tiles, c_tiles) output tiles per spatial row

    for s in range(s_rows):
        # --- Extract one spatial position: rows [s*n_tiles : (s+1)*n_tiles] ---
        # Flat tile indices in row-major order: row r, col c → r*c_tiles + c.
        # q_row, k_row, v_row each have tile-grid shape (n_tiles, c_tiles).
        q_flat = q_blk.to_list()
        k_flat = k_blk.to_list()
        v_flat = v_blk.to_list()

        base = s * n_tiles  # first tile-row index for this spatial position

        q_row = Block.from_list(
            q_flat[base * c_tiles:(base + n_tiles) * c_tiles],
            shape=(n_tiles, c_tiles),
        )
        # k_row (n_tiles, c_tiles) will be transposed to (c_tiles, n_tiles).
        k_row_tiles = k_flat[base * c_tiles:(base + n_tiles) * c_tiles]
        v_row = Block.from_list(
            v_flat[base * c_tiles:(base + n_tiles) * c_tiles],
            shape=(n_tiles, c_tiles),
        )

        # --- Transpose K: (n_tiles, c_tiles) → (c_tiles, n_tiles) ---
        # Tile (r, c) in k_row (row-major flat index r*c_tiles + c) becomes
        # tile (c, r) in k_row_T (flat index c*n_tiles + r), with its data
        # transposed to flip the 32×32 element layout as well.
        k_T_tiles = [None] * (c_tiles * n_tiles)
        for r in range(n_tiles):
            for c in range(c_tiles):
                src_tile = k_row_tiles[r * c_tiles + c]
                # to_torch() gives the 32×32 backing data; .T transposes it.
                transposed_data = Tensor(src_tile.to_torch().T.contiguous())
                k_T_tiles[c * n_tiles + r] = transposed_data
        k_row_T = Block.from_list(k_T_tiles, shape=(c_tiles, n_tiles))

        # --- Stage 1: Scaled scores = q_row @ k_row_T * scale ----------------
        # (n_tiles, c_tiles) @ (c_tiles, n_tiles) → (n_tiles, n_tiles)
        scores_raw = q_row @ k_row_T  # Block (n_tiles, n_tiles)

        # Scale each tile's backing data by the scalar factor.
        # Block arithmetic operators require two Blocks of the same shape, so
        # we apply the scale directly to the backing tensors for simplicity.
        scores_tiles = [
            Tensor(t.to_torch() * scale) for t in scores_raw.to_list()
        ]
        scores = Block.from_list(scores_tiles, shape=(n_tiles, n_tiles))

        # --- Stage 2: Stable row-wise softmax ---------------------------------
        # Step 2a: per-row max — reduce along cols (dim 1).
        # reduce_max(shape (n_tiles, n_tiles), dims=[1]) → (n_tiles, 1).
        row_max = ttl_math.reduce_max(scores, scaler_blk, dims=[1])

        # Step 2b: broadcast (n_tiles, 1) → (n_tiles, n_tiles) along cols.
        bcast_max = ttl_math.broadcast(row_max, output_hint=scores, dims=[1])

        # Step 2c: shift by max and exponentiate (two copies for sum + normalise).
        shifted = scores - bcast_max
        exp_s1 = ttl_math.exp(shifted)  # copy 1: fed to reduce_sum
        exp_s2 = ttl_math.exp(shifted)  # copy 2: fed to final multiply

        # Step 2d: per-row sum — reduce along cols.
        # reduce_sum(shape (n_tiles, n_tiles), dims=[1]) → (n_tiles, 1).
        row_sum = ttl_math.reduce_sum(exp_s1, scaler_blk, dims=[1])

        # Step 2e: broadcast (n_tiles, 1) → (n_tiles, n_tiles) along cols.
        bcast_sum = ttl_math.broadcast(row_sum, output_hint=exp_s2, dims=[1])

        # Step 2f: normalise — element-wise multiply by reciprocal of sum.
        # recip(bcast_sum) gives 1/sum; * exp_s2 gives the softmax weights.
        attn = exp_s2 * ttl_math.recip(bcast_sum)  # (n_tiles, n_tiles)

        # --- Stage 3: out = attn @ v_row -------------------------------------
        # (n_tiles, n_tiles) @ (n_tiles, c_tiles) → (n_tiles, c_tiles).
        out_row = attn @ v_row  # Block (n_tiles, c_tiles)

        # Collect tiles in row-major order for final reassembly.
        out_tiles.extend(out_row.to_list())

    # Reassemble all spatial positions into (sn_tiles, c_tiles) and convert.
    out_blk = Block.from_list(out_tiles, shape=(sn_tiles, c_tiles))
    return block_to_tensor(out_blk)


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


def _out_proj_kernel_sim(attn_out_t, x_residual_t, w_o_t, s_rows, n_tiles, c_tiles):
    """Output projection + residual: out = x_residual + attn_out @ w_o.

    Uses the same row-streaming Block ``@`` pattern as ``_qkv_kernel_sim``:
    for each tile-row *m*, computes ``attn_row [1, c_tiles] @ w_o → proj_row``
    then adds the corresponding ``x_residual`` row tile-by-tile.

    The addition uses the sim Block ``+`` operator (element-wise, same shape),
    which is overloaded on the Block class.

    Args:
        attn_out_t:   Float32 [s_rows*n_tiles*32, c_tiles*32] — output of
                      _sdpa_kernel_sim.
        x_residual_t: Float32 [s_rows*n_tiles*32, c_tiles*32] — original x
                      (for the residual connection).
        w_o_t:        Float32 [c_tiles*32, c_tiles*32] — output projection weight.
        s_rows:       Spatial rows S.
        n_tiles:      Frame tiles N/32 (typically 1).
        c_tiles:      Channel tiles C/32.

    Returns:
        out: Float32 tensor [s_rows*n_tiles*32, c_tiles*32]
             equal to x_residual + attn_out @ w_o.
    """
    import sys
    import os
    _TT_LANG_PYTHON = os.path.expanduser("~/code/tt-lang/python")
    if _TT_LANG_PYTHON not in sys.path:
        sys.path.insert(0, _TT_LANG_PYTHON)

    from sim.dfb import Block
    from animatediff_ttnn.ttlang.sim_helpers import tensor_to_block, block_to_tensor

    # Total tile-rows in the flattened (S*N) layout.
    sn_tiles = s_rows * n_tiles

    # Convert inputs to tile-Block representations.
    attn_blk = tensor_to_block(attn_out_t.float(),   shape=(sn_tiles, c_tiles))
    xres_blk = tensor_to_block(x_residual_t.float(), shape=(sn_tiles, c_tiles))
    wo_blk   = tensor_to_block(w_o_t.float(),        shape=(c_tiles, c_tiles))

    out_tiles = []

    for m in range(sn_tiles):
        # Extract row m from attn_out and x_residual: each is (1, c_tiles).
        attn_row = Block.from_list(
            attn_blk.to_list()[m * c_tiles:(m + 1) * c_tiles],
            shape=(1, c_tiles),
        )
        xres_row = Block.from_list(
            xres_blk.to_list()[m * c_tiles:(m + 1) * c_tiles],
            shape=(1, c_tiles),
        )

        # Output projection: (1, c_tiles) @ (c_tiles, c_tiles) → (1, c_tiles).
        proj_row = attn_row @ wo_blk

        # Residual addition: (1, c_tiles) + (1, c_tiles) → (1, c_tiles).
        # Block.__add__ performs element-wise addition tile-by-tile.
        out_row = xres_row + proj_row

        # Collect output tiles in row-major order.
        out_tiles.extend(out_row.to_list())

    # Reassemble the full (sn_tiles, c_tiles) block and convert to tensor.
    out_blk = Block.from_list(out_tiles, shape=(sn_tiles, c_tiles))
    return block_to_tensor(out_blk)
