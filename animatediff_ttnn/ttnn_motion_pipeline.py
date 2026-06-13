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

# Import ttnn at module level so tests can patch
# 'animatediff_ttnn.ttnn_motion_pipeline.ttnn'. In environments where the
# ttnn wheel is not installed (CI, unit-test runners) the import will fail;
# we fall back to None so the module still loads. _apply_temporal guards
# against this via the mock in tests.
try:
    import ttnn
except ModuleNotFoundError:
    ttnn = None  # type: ignore[assignment]

# Import to_device at module level for the same patchability reason.
try:
    from animatediff_ttnn.ttnn_pipeline import to_device
except Exception:
    to_device = None  # type: ignore[assignment]


def _apply_temporal(
    samples: list,
    kernel_list: list,
    device,
    num_frames: int,
    C: int,
) -> list:
    """Bridge between TTNN device tensors and TemporalAttentionKernel.

    Pulls N TTNN hidden states to CPU, applies each kernel in kernel_list
    sequentially (AnimateDiff uses 2-3 motion modules per block; mid uses 1),
    then pushes back to device with the same dtype/layout.

    The TTNN UNet doubles batch to 2 for CFG (uncond + cond). We split these
    before temporal attention so uncond and cond attend over their own N-frame
    sequences independently, then reconstruct the [2, S, C] tensors.

    Args:
        samples:     List of N TTNN tensors, each [2, S, C].
        kernel_list: List of TemporalAttentionKernel, applied in order.
                     Typically 2-3 kernels (motion modules), or 1 for mid_block.
        device:      TTNN device (MeshDevice from setup_blackhole).
        num_frames:  N (length of samples).
        C:           Channel dimension — used only for documentation / assertion.

    Returns:
        List of N TTNN tensors, same shape as input, with temporal attention applied.
        Original input tensors are deallocated.
    """
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
