# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""MotionAdapter module loader for Phase 3 temporal attention injection.

Loads the guoyww/animatediff-motion-adapter-v1-5-2 checkpoint from the
HuggingFace cache and returns the AnimateDiffTransformer3D modules for all 7
injection points in the TTNN UNet (down0-2, mid, up0-2).

Each module is the full diffusers AnimateDiffTransformer3D (with norm,
proj_in/out, LayerNorm, QKV attn, feedforward, positional embedding) run on
CPU during the TTNN denoising loop.

We use the diffusers modules directly rather than a partial reimplementation to
ensure correctness: earlier partial QKV-only kernels missed LayerNorm, bias,
positional embedding, and feedforward, causing energy explosion.
"""
from __future__ import annotations

from typing import NamedTuple


class InjectionPoint(NamedTuple):
    block_prefix: str   # state-dict prefix for this block's motion_modules
    dim: int            # channel dimension C
    num_modules: int    # number of motion modules in this block
    spatial_h: int      # spatial height H at this UNet stage (64×64 input → SD1.x)
    spatial_w: int      # spatial width  W at this UNet stage


#: Injection points for the 7 TTNN UNet locations.
#:
#: Spatial resolutions for a 64×64 latent input to SD 1.x UNet:
#:   down0: 32×32 (after first down-sampling)
#:   down1: 16×16
#:   down2:  8×8
#:   mid:    8×8  (no spatial change through mid block)
#:   up0:   16×16 (after first up-sampling)
#:   up1:   32×32
#:   up2:   64×64
_INJECTION_POINTS: dict[str, InjectionPoint] = {
    "down0": InjectionPoint("down_blocks.0", 320,  2, 32, 32),
    "down1": InjectionPoint("down_blocks.1", 640,  2, 16, 16),
    "down2": InjectionPoint("down_blocks.2", 1280, 2,  8,  8),
    "mid":   InjectionPoint("mid_block",     1280, 1,  8,  8),
    "up0":   InjectionPoint("up_blocks.0",   1280, 3, 16, 16),
    "up1":   InjectionPoint("up_blocks.1",   1280, 3, 32, 32),
    "up2":   InjectionPoint("up_blocks.2",   640,  3, 64, 64),
}


def load_motion_modules(
    model_id: str = "guoyww/animatediff-motion-adapter-v1-5-2",
) -> dict[str, list]:
    """Load MotionAdapter modules for all 7 injection points.

    Returns the actual AnimateDiffTransformer3D PyTorch modules (eval mode,
    on CPU), keyed by injection point.  Callers pass TTNN activations through
    these modules on each denoising step.

    Args:
        model_id: HuggingFace repo ID or local directory path.

    Returns:
        Dict mapping injection point key → list of AnimateDiffTransformer3D.
        Keys: "down0", "down1", "down2", "mid", "up0", "up1", "up2".
        Each list has 2 modules (motion_modules.0 and 1) except "mid" (1 module).
    """
    from diffusers import MotionAdapter

    adapter = MotionAdapter.from_pretrained(model_id)
    adapter.eval()

    modules: dict[str, list] = {}
    for key, ip in _INJECTION_POINTS.items():
        # Access the actual MotionAdapter sub-modules by block name.
        block_obj = _get_block(adapter, ip.block_prefix)
        module_list = list(block_obj.motion_modules)[: ip.num_modules]
        modules[key] = module_list

    return modules


def get_injection_point_info(key: str) -> InjectionPoint:
    """Return the InjectionPoint metadata for a given key."""
    return _INJECTION_POINTS[key]


def _get_block(adapter, prefix: str):
    """Navigate from the adapter root to the named sub-block via attribute path."""
    obj = adapter
    for part in prefix.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj
