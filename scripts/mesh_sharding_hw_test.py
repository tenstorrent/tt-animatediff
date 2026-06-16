#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Hardware smoke test: submesh-parallel frame generation vs serial single-chip.

The tt-metal SD demo UNet (wormhole) is a single-device model.  Its __init__
calls ttnn.to_torch(weight) without a mesh_composer, which fails on any tensor
distributed across N>1 chips (buffers.size() == N, not 1).  Attempting to load
the UNet via preprocess_model_parameters(device=4-chip-mesh) fails at the first
permute_conv_weights() call.

Correct multi-chip strategy: N×(1×1 submeshes), one UNet per chip, one frame
per chip.  This is data-parallelism at the frame level, not tensor-parallelism
within a single UNet call.  The SD demo UNet would need to be rewritten to
support MeshDevice natively for the latter.

Test structure
--------------
Serial run  — 1-chip MeshDevice(1×1), 4 frames, full generate_frames_temporal.
Submesh run — 4-chip MeshDevice(1×4), partitioned into 4 1×1 submeshes.
              Each submesh generates 1 frame with num_frames=1, no cross-frame
              attention.  Frames are generated in frame-index order (not truly
              parallel — TTNN is not thread-safe) but each on a different chip.

PCC comparison: since both runs use the same seed and no cross-frame attention
is possible in the submesh path, we compare each submesh frame against the
corresponding single-frame result from a separate reference run, not the full
4-frame temporal run (which blends noise_preds).

Usage:
    source ~/tt-metal/python_env/bin/activate
    TT_METAL_ARCH_NAME=blackhole python scripts/mesh_sharding_hw_test.py
"""
import sys
import time
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def run_single_chip(chip_id: int, num_frames: int = 1, num_steps: int = 4, seed: int = 42,
                    label: str | None = None):
    """Run num_frames on a single 1×1 MeshDevice. Returns (frames, gen_s, total_s)."""
    import ttnn
    from animatediff_ttnn.ttnn_pipeline import setup_blackhole
    from animatediff_ttnn.generation_helpers import load_sd14_ttnn, encode_prompt
    from animatediff_ttnn.temporal_attention import generate_frames_temporal

    label = label or f"chip {chip_id}"
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    t0 = time.time()
    device = setup_blackhole(device_ids=[chip_id])
    ttnn_model, ttnn_vae, config, time_proj = load_sd14_ttnn(device)
    text_emb = encode_prompt("a serene mountain lake at sunrise")

    t_gen = time.time()
    frames = generate_frames_temporal(
        device=device,
        ttnn_model=ttnn_model,
        ttnn_vae=ttnn_vae,
        config=config,
        torch_time_proj=time_proj,
        text_embeddings=text_emb,
        num_frames=num_frames,
        num_steps=num_steps,
        seed=seed,
    )
    gen_s = time.time() - t_gen
    ttnn.close_mesh_device(device)
    total_s = time.time() - t0

    print(f"  Generation: {gen_s:.1f}s  (wall: {total_s:.1f}s incl. compile)")
    return frames, gen_s, total_s


def run_submesh_round_robin(device_ids: list, num_frames: int = 4, num_steps: int = 4,
                            seed: int = 42):
    """Run num_frames using N 1×1 submeshes in round-robin.

    Opens a full 1×N MeshDevice, partitions into N 1×1 submeshes, loads one
    UNet+VAE per submesh, then generates frames in round-robin order (frame i
    on chip i % N).  Sequential, not truly parallel — TTNN is not thread-safe.

    Each chip runs num_frames=1, so the result is N independent single-frame
    outputs with no cross-frame attention.  This demonstrates that all N chips
    can load and run the UNet correctly.
    """
    import ttnn
    from animatediff_ttnn.ttnn_pipeline import setup_blackhole
    from animatediff_ttnn.generation_helpers import load_sd14_ttnn, encode_prompt
    from animatediff_ttnn.temporal_attention import generate_frames_temporal

    n = len(device_ids)
    print(f"\n{'='*60}")
    print(f"  SUBMESH ROUND-ROBIN ({n} chips): device_ids={device_ids}")
    print(f"  Each chip: 1×1 submesh, independent UNet, 1 frame at a time")
    print(f"{'='*60}")

    t0 = time.time()
    full_mesh = setup_blackhole(device_ids=device_ids)
    submeshes = full_mesh.create_submeshes(ttnn.MeshShape(1, 1))
    assert len(submeshes) == n, f"Expected {n} submeshes, got {len(submeshes)}"
    print(f"  Opened {n} submeshes from MeshDevice(1×{n})")

    text_emb = encode_prompt("a serene mountain lake at sunrise")

    # Load one model per submesh
    print(f"  Loading {n} UNet+VAE instances...")
    models = []
    for i, sub in enumerate(submeshes):
        print(f"    Chip {i} (device_id={device_ids[i]}): loading UNet...")
        ttnn_model_i, ttnn_vae_i, config_i, time_proj_i = load_sd14_ttnn(sub)
        models.append((sub, ttnn_model_i, ttnn_vae_i, config_i, time_proj_i))

    t_gen = time.time()
    all_frames = []
    for fi in range(num_frames):
        chip_i = fi % n
        sub, ttnn_model_i, ttnn_vae_i, config_i, time_proj_i = models[chip_i]
        print(f"  Frame {fi}: chip {chip_i} (device_id={device_ids[chip_i]})")
        # Use seed so noise matches a reference single-chip run with same seed
        frames_i = generate_frames_temporal(
            device=sub,
            ttnn_model=ttnn_model_i,
            ttnn_vae=ttnn_vae_i,
            config=config_i,
            torch_time_proj=time_proj_i,
            text_embeddings=text_emb,
            num_frames=1,
            num_steps=num_steps,
            seed=seed,
        )
        all_frames.append(frames_i[0])

    gen_s = time.time() - t_gen

    for sub in submeshes:
        ttnn.close_mesh_device(sub)
    ttnn.close_mesh_device(full_mesh)
    total_s = time.time() - t0

    print(f"  Generation: {gen_s:.1f}s  (wall: {total_s:.1f}s incl. compile)")
    return all_frames, gen_s, total_s


def main():
    # Reference: chip 0 only, 1 frame, 4 steps — same seed as submesh run
    ref_frames, ref_gen, ref_total = run_single_chip(
        0, num_frames=1, num_steps=4, seed=42, label="REFERENCE (chip 0, 1 frame)"
    )

    # Submesh run: 4 chips, 4 frames (each chip generates the same seed→same noise)
    sub_frames, sub_gen, sub_total = run_submesh_round_robin(
        [0, 1, 2, 3], num_frames=4, num_steps=4, seed=42
    )

    print("\n" + "="*60)
    print("  RESULTS")
    print("="*60)
    print(f"  Reference (1 chip, 1 frame): {ref_gen:.1f}s gen, {ref_total:.1f}s wall")
    print(f"  Submesh   (4 chips, 4 frames): {sub_gen:.1f}s gen, {sub_total:.1f}s wall")
    print()
    print("  NOTE: Each submesh frame uses the same seed → same base noise as")
    print("  the reference. All 4 outputs should be PCC≈1.0 against ref_frames[0].")
    print()

    print("  PCC (each submesh frame vs reference frame 0):")
    ref_t = torch.tensor(np.array(ref_frames[0])).float()
    all_pass = True
    for i, f4 in enumerate(sub_frames):
        f4_t = torch.tensor(np.array(f4)).float()
        p = pcc(ref_t, f4_t)
        status = "PASS" if p > 0.99 else "FAIL"
        if p <= 0.99:
            all_pass = False
        chip_id = i % 4
        print(f"    Frame {i} (chip {chip_id}): PCC={p:.4f}  [{status}]")

    if all_pass:
        print("\n  ALL SUBMESH CHIPS PASSED — each chip produces identical output to chip 0.")
        print("  To use N chips in production: open MeshDevice, create_submeshes(MeshShape(1,1)),")
        print("  load N UNet instances, dispatch 1 frame per chip (sequential or via threads).")
        sys.exit(0)
    else:
        print("\n  SOME PCC CHECKS FAILED — check seed/noise alignment between serial and submesh.")
        sys.exit(1)


if __name__ == "__main__":
    main()
