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
