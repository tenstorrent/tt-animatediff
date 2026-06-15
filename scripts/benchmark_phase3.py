#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Phase 3 timing benchmark — measures wall-clock with and without the
batched-transfer + torch.compile optimisation.

Usage:
    source ~/tt-metal/python_env/bin/activate
    TT_METAL_ARCH_NAME=blackhole python scripts/benchmark_phase3.py [--frames 8] [--steps 10]

Prints per-step, per-injection-point breakdown and total wall-clock time.
"""
import argparse
import time
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 wall-clock benchmark")
    p.add_argument("--frames", type=int, default=8,
                   help="Number of animation frames (must be divisible by num chips)")
    p.add_argument("--steps",  type=int, default=10,
                   help="Denoising steps (fewer → faster, less quality)")
    p.add_argument("--prompt", type=str,
                   default="a majestic waterfall in a lush rainforest, cinematic lighting, 4K",
                   help="Generation prompt")
    p.add_argument("--alpha",  type=float, default=1.0,
                   help="Temporal injection alpha (0=bypass, 1=full)")
    p.add_argument("--output", type=str, default="/tmp/benchmark_phase3.gif",
                   help="Output GIF path")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 64)
    print("Phase 3 MotionAdapter benchmark")
    print(f"  prompt : {args.prompt[:60]}...")
    print(f"  frames : {args.frames}")
    print(f"  steps  : {args.steps}")
    print(f"  alpha  : {args.alpha}")
    print("=" * 64)

    # ── hardware init ─────────────────────────────────────────────────────────
    t_init_start = time.perf_counter()
    from animatediff_ttnn.ttnn_pipeline import setup_blackhole
    device = setup_blackhole()
    t_init = time.perf_counter() - t_init_start
    print(f"\n[init] Blackhole setup: {t_init:.1f}s")

    # ── model + weight load ───────────────────────────────────────────────────
    t_load_start = time.perf_counter()
    from animatediff_ttnn.motion_weights import load_motion_modules
    temporal_kernels = load_motion_modules(
        model_id="guoyww/animatediff-motion-adapter-v1-5-2"
    )
    t_load = time.perf_counter() - t_load_start
    print(f"[load] MotionAdapter weights: {t_load:.1f}s  ({len(temporal_kernels)} injection points)")

    # ── TTNN UNet model load ──────────────────────────────────────────────────
    t_model_start = time.perf_counter()
    from animatediff_ttnn.ttnn_pipeline import build_unet_model
    ttnn_model = build_unet_model(device)
    t_model = time.perf_counter() - t_model_start
    print(f"[load] TTNN UNet model: {t_model:.1f}s")

    # ── VAE + tokenizer + text encoder (CPU) ─────────────────────────────────
    t_cpu_start = time.perf_counter()
    from diffusers import AutoencoderKL, CLIPTextModel, CLIPTokenizer
    model_id = "stable-diffusion-v1-4"
    vae        = AutoencoderKL.from_pretrained(f"CompVis/{model_id}", subfolder="vae")
    tokenizer  = CLIPTokenizer.from_pretrained(f"CompVis/{model_id}", subfolder="tokenizer")
    text_enc   = CLIPTextModel.from_pretrained(f"CompVis/{model_id}", subfolder="text_encoder")
    vae.eval(); text_enc.eval()
    t_cpu = time.perf_counter() - t_cpu_start
    print(f"[load] CPU components (VAE/tokenizer/text_enc): {t_cpu:.1f}s")

    # ── generation ────────────────────────────────────────────────────────────
    from animatediff_ttnn.temporal_attention import generate_frames_motion
    import torch

    # Encode prompt
    tokens = tokenizer([args.prompt, ""], padding="max_length",
                       max_length=tokenizer.model_max_length,
                       return_tensors="pt")
    with torch.no_grad():
        enc_hidden = text_enc(tokens.input_ids)[0]  # [2, 77, 768]

    print(f"\n[run] Starting Phase 3 generation ({args.frames} frames × {args.steps} steps)...")
    t_gen_start = time.perf_counter()

    frames = generate_frames_motion(
        ttnn_model=ttnn_model,
        prompt_embeds=enc_hidden,
        device=device,
        temporal_kernels=temporal_kernels,
        num_frames=args.frames,
        num_steps=args.steps,
        guidance_scale=7.5,
        injection_alpha=args.alpha,
    )

    t_gen = time.perf_counter() - t_gen_start
    total_calls = args.frames * args.steps * 7  # 7 injection points

    print("\n" + "=" * 64)
    print("Results")
    print(f"  generation wall-clock : {t_gen:.1f}s")
    print(f"  per-frame             : {t_gen / args.frames:.1f}s")
    print(f"  per-step              : {t_gen / args.steps:.1f}s")
    print(f"  total _apply_temporal calls: {total_calls} ({args.steps} steps × 7 pts × 1 device)")
    print("=" * 64)

    # ── decode + save ─────────────────────────────────────────────────────────
    t_vae_start = time.perf_counter()
    import numpy as np
    from PIL import Image

    decoded = []
    for lat in frames:
        with torch.no_grad():
            lat_in = lat.to(dtype=torch.float32) / vae.config.scaling_factor
            img_t = vae.decode(lat_in.unsqueeze(0)).sample  # [1, 3, H, W]
        img_t = (img_t.clamp(-1, 1) + 1) / 2
        img_np = (img_t.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        decoded.append(Image.fromarray(img_np))

    t_vae = time.perf_counter() - t_vae_start
    print(f"  VAE decode            : {t_vae:.1f}s")

    decoded[0].save(
        args.output,
        save_all=True,
        append_images=decoded[1:],
        loop=0,
        duration=125,
    )
    print(f"  Saved → {args.output}")
    print(f"\nBaseline reference: ~806s / 8 frames / 25 steps (before optimisation)")
    print(f"This run: {t_gen:.1f}s / {args.frames} frames / {args.steps} steps")
    # Scale to equivalent 8-frame 25-step run for comparison
    equiv = t_gen * (8 / args.frames) * (25 / args.steps)
    print(f"Scaled to 8fr/25step equivalent: ~{equiv:.0f}s  (speedup vs baseline: {806/equiv:.2f}x)")


if __name__ == "__main__":
    main()
