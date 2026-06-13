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
#:
#: The guoyww/animatediff-motion-adapter-v1-5-2 checkpoint mirrors the full
#: SD 1.x UNet layout: 4 down blocks (indices 0-3) and 4 up blocks (0-3).
#: Module counts per block (verified from actual state-dict):
#:   down_blocks 0-2: 2 modules each (down_block 3 has 2 as well)
#:   up_blocks 0-2:   3 modules each (up_block 3 has 3 as well)
#:   mid_block:       1 module
#: Channel dims (from actual to_q.weight shapes):
#:   down 0→320, 1→640, 2→1280, 3→1280
#:   mid  →1280
#:   up   0→1280, 1→1280, 2→640, 3→320
#: The 7 injection-point keys (down0-2, mid, up0-2) target blocks 0-2
#: on each side, skipping the outermost blocks (down3 / up3) which carry
#: the full spatial resolution and are not used for temporal injection.
_INJECTION_POINTS: dict[str, tuple[str, int, int]] = {
    "down0": ("down_blocks.0", 320,  2),
    "down1": ("down_blocks.1", 640,  2),
    "down2": ("down_blocks.2", 1280, 2),
    "mid":   ("mid_block",     1280, 1),
    "up0":   ("up_blocks.0",   1280, 3),
    "up1":   ("up_blocks.1",   1280, 3),
    "up2":   ("up_blocks.2",   640,  3),
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
