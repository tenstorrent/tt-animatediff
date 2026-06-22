#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Sequential validation: run all 4 distilled weight combos and benchmark them.

What this file does
-------------------
After distillation, we want to compare four configurations:
  Config 0: 4-step UNet + original MotionAdapter  (spatial-only fast)
  Config 1: 8-step UNet + original MotionAdapter  (spatial-only balanced)
  Config 2: 8-step UNet + 8-step MotionAdapter    (full lightning, balanced)
  Config 3: 4-step UNet + 4-step MotionAdapter    (full lightning, maximum speed)

Each configuration runs inference sequentially and saves a GIF. At the end
we print a benchmark table comparing steps, elapsed time, and output path.

NOTE: previously used multiprocessing.Pool across 4 chips, but Blackhole's
PCIe lock (CHIP_IN_USE_0_PCIe) is process-wide — parallel processes all block
on chip 0's lock regardless of which chip_id they request. Sequential is correct.

Usage
-----
    python scripts/validate_parallel.py

    # Or specify output dir:
    python scripts/validate_parallel.py --out_dir output/validation

Requirements
------------
    - All 4 weight files present in weights/
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


def build_validation_configs(
    unet_4step: Path,
    unet_8step: Path,
    adapter_4step: Path,
    adapter_8step: Path,
) -> list[dict]:
    """Return the 4 chip-config dicts to run in parallel.

    Each dict has:
        chip_id:        Physical Blackhole device ID (0-3)
        label:          Human-readable name for this config
        unet_path:      Path to distilled UNet weights (.pt)
        adapter_path:   Path to MotionAdapter weights (.pt), or None for original
        num_steps:      Inference denoising steps
    """
    return [
        {
            "chip_id": 0,
            "label": "spatial-fast-4step",
            "unet_path": unet_4step,
            "adapter_path": None,
            "num_steps": 4,
        },
        {
            "chip_id": 1,
            "label": "spatial-balanced-8step",
            "unet_path": unet_8step,
            "adapter_path": None,
            "num_steps": 8,
        },
        {
            "chip_id": 2,
            "label": "lightning-8step",
            "unet_path": unet_8step,
            "adapter_path": adapter_8step,
            "num_steps": 8,
        },
        {
            "chip_id": 3,
            "label": "lightning-4step",
            "unet_path": unet_4step,
            "adapter_path": adapter_4step,
            "num_steps": 4,
        },
    ]


def format_benchmark_table(results: list[dict]) -> str:
    """Format a human-readable benchmark comparison table.

    Args:
        results: List of dicts with keys: label, elapsed_s, gif_path.

    Returns:
        Multi-line string ready to print.
    """
    lines = [
        "",
        "╔══════════════════════════════════════════════════════════════",
        "║  Validation Results",
        "╠══════════════════════════════════════════════════════════════",
        f"║  {'Config':<30} {'Time (s)':>10}  {'Output'}",
        "╠══════════════════════════════════════════════════════════════",
    ]
    for r in results:
        lines.append(
            f"║  {r['label']:<30} {r['elapsed_s']:>10.1f}  {r['gif_path']}"
        )
    lines.append("╚══════════════════════════════════════════════════════════════")
    return "\n".join(lines)


def _run_one_config(config: dict, out_dir: Path, prompt: str) -> dict:
    """Run inference for one config and return timing + output path.

    Args:
        config:   One entry from build_validation_configs().
        out_dir:  Directory to save the output GIF.
        prompt:   Text prompt for generation.

    Returns:
        Dict with label, elapsed_s, gif_path — ready for format_benchmark_table.
    """
    from animatediff_ttnn.pipeline import create_animatediff_pipeline, generate
    import torch

    label = config["label"]
    print(f"\n[{label}] Starting...")

    pipe = create_animatediff_pipeline()
    if config["unet_path"] and Path(config["unet_path"]).exists():
        state = torch.load(config["unet_path"], map_location="cpu", weights_only=True)
        pipe.unet.load_state_dict(state, strict=False)

    if config["adapter_path"] and Path(config["adapter_path"]).exists():
        adapter_state = torch.load(config["adapter_path"], map_location="cpu",
                                   weights_only=True)
        pipe.motion_adapter.load_state_dict(adapter_state, strict=False)

    t0 = time.perf_counter()
    frames = generate(
        pipe=pipe,
        prompt=prompt,
        num_frames=16,
        num_inference_steps=config["num_steps"],
        seed=42,
    )
    elapsed = time.perf_counter() - t0

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = out_dir / f"{label}.gif"
    frames[0].save(
        str(gif_path),
        save_all=True,
        append_images=frames[1:],
        duration=125,
        loop=0,
    )

    print(f"[{label}] done in {elapsed:.1f}s → {gif_path}")
    return {"label": label, "elapsed_s": elapsed, "gif_path": gif_path}


def main():
    parser = argparse.ArgumentParser(description="Sequential 4-config validation")
    parser.add_argument("--out_dir", type=str, default="output/validation")
    parser.add_argument("--prompt", type=str,
                        default="a campfire burning in a dark forest, cinematic")
    args = parser.parse_args()

    weights = REPO_ROOT / "weights"
    configs = build_validation_configs(
        unet_4step=weights / "unet_lcm_4step.pt",
        unet_8step=weights / "unet_lcm_8step.pt",
        adapter_4step=weights / "motion_adapter_lcm_4step.pt",
        adapter_8step=weights / "motion_adapter_lcm_8step.pt",
    )

    out_dir = REPO_ROOT / args.out_dir
    results = []

    print(f"Running {len(configs)} configs sequentially...")
    for cfg in configs:
        result = _run_one_config(cfg, out_dir, args.prompt)
        results.append(result)

    print(format_benchmark_table(results))


if __name__ == "__main__":
    main()
