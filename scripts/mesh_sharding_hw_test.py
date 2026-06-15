#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Hardware smoke test: mesh frame sharding vs serial single-chip.

Runs a 4-frame generation (4 denoising steps) twice:
  1. Serial mode — device_ids=[0], 1 chip
  2. Mesh mode   — device_ids=[0,1,2,3], 4 chips

Asserts PCC > 0.99 between final decoded frame tensors (same seed, same weights).
Prints per-component wall-clock breakdown (denoising, VAE decode, total).

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


def run_generation(device_ids: list, label: str):
    """Run 4-frame, 4-step generation. Returns (frames, denoising_s, vae_s, total_s)."""
    import ttnn
    from animatediff_ttnn.ttnn_pipeline import setup_blackhole
    from animatediff_ttnn.generation_helpers import load_sd14_ttnn, encode_prompt
    from animatediff_ttnn.temporal_attention import generate_frames_temporal

    print(f"\n{'='*60}")
    print(f"  {label}: device_ids={device_ids}")
    print(f"{'='*60}")

    t0 = time.time()
    device = setup_blackhole(device_ids=device_ids)
    ttnn_model, ttnn_vae, config, time_proj = load_sd14_ttnn(device)
    text_emb = encode_prompt("a serene mountain lake at sunrise")

    # Monkey-patch ttnn_vae.decode to time the VAE decode phase separately.
    original_decode = ttnn_vae.decode
    _vae_times = []

    def timed_decode(*args, **kwargs):
        t = time.time()
        result = original_decode(*args, **kwargs)
        _vae_times.append(time.time() - t)
        return result

    ttnn_vae.decode = timed_decode

    t_gen_start = time.time()
    frames = generate_frames_temporal(
        device=device,
        ttnn_model=ttnn_model,
        ttnn_vae=ttnn_vae,
        config=config,
        torch_time_proj=time_proj,
        text_embeddings=text_emb,
        num_frames=4,
        num_steps=4,
        seed=42,
    )
    total_gen = time.time() - t_gen_start
    vae_s = sum(_vae_times)
    denoise_s = total_gen - vae_s

    ttnn.close_mesh_device(device)
    total_s = time.time() - t0

    print(f"  Denoising : {denoise_s:.1f}s")
    print(f"  VAE decode: {vae_s:.1f}s")
    print(f"  Total gen : {total_gen:.1f}s  (wall: {total_s:.1f}s incl. compile)")
    return frames, denoise_s, vae_s, total_s


def main():
    frames_serial, d1, v1, t1 = run_generation([0], "SERIAL (1 chip)")
    frames_mesh,   d4, v4, t4 = run_generation([0, 1, 2, 3], "MESH (4 chips)")

    print("\n" + "="*60)
    print("  SPEEDUP SUMMARY")
    print("="*60)
    print(f"  Denoising : {d1:.1f}s → {d4:.1f}s  ({d1/max(d4, 0.1):.2f}×)")
    print(f"  VAE decode: {v1:.1f}s → {v4:.1f}s  ({v1/max(v4, 0.1):.2f}×)")

    print("\n  PCC between serial and mesh outputs:")
    all_pass = True
    for i, (f1, f4) in enumerate(zip(frames_serial, frames_mesh)):
        t1_np = torch.tensor(np.array(f1)).float()
        t4_np = torch.tensor(np.array(f4)).float()
        p = pcc(t1_np, t4_np)
        status = "PASS" if p > 0.99 else "FAIL"
        if p <= 0.99:
            all_pass = False
        print(f"    Frame {i}: PCC={p:.4f}  [{status}]")

    if all_pass:
        print("\n  ALL PCC CHECKS PASSED")
        sys.exit(0)
    else:
        print("\n  SOME PCC CHECKS FAILED — check that both runs used identical seeds/weights")
        sys.exit(1)


if __name__ == "__main__":
    main()
