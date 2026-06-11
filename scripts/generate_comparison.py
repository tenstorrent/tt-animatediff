#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Generate the three Lightning Mode comparison GIFs shown on the website.

Regenerates docs/assets/lcm-*.gif using our own LCM-distilled weights.
Also generates the existing docs/assets/lightning-*.gif (Blackhole 8-step Euler)
if they don't already exist.

GIFs produced
-------------
  docs/assets/lcm-aurora.gif       LCM-4step (our weights, CPU)
  docs/assets/lcm-mandala.gif      LCM-4step (our weights, CPU)
  docs/assets/lcm-mycelium.gif     LCM-4step (our weights, CPU)
  docs/assets/lightning-aurora.gif     Blackhole 8-step Euler (regenerate if missing)
  docs/assets/lightning-mandala.gif    Blackhole 8-step Euler (regenerate if missing)
  docs/assets/lightning-mycelium.gif   Blackhole 8-step Euler (regenerate if missing)

Usage:
    source ~/tt-metal/python_env/bin/activate
    python scripts/generate_comparison.py [--only lcm|lightning|all]

LCM modes require weights in weights/:
    weights/unet_lcm_4step.pt
    weights/motion_adapter_lcm_4step.pt
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).parent.parent
ASSETS = REPO / "docs" / "assets"
WEIGHTS = REPO / "weights"

# The three prompts that appear in the website comparison section.
# Each dict maps: name, prompt, seed, alpha
COMPARISON_SUBJECTS = [
    {
        "name": "aurora",
        "prompt": "aurora borealis dancing over arctic ice, green and violet ribbons, starfield, cinematic",
        "negative": "blurry, low quality, text, people, faces, tropical",
        "seed": 42,
        "alpha": 0.35,
    },
    {
        "name": "mandala",
        "prompt": "sacred mandala blooming from starfield, geometric petals, gold and violet, cosmic, cinematic",
        "negative": "blurry, low quality, text, people, faces",
        "seed": 42,
        "alpha": 0.4,
    },
    {
        "name": "mycelium",
        "prompt": "mycelium network glowing with bioluminescent spores, threads of light connecting nodes, cosmic forest",
        "negative": "blurry, low quality, text, people, faces",
        "seed": 42,
        "alpha": 0.4,
    },
]


def run_cmd(cmd, label):
    """Run a subprocess command and return True on success."""
    print(f"\n  → {label}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    for line in (result.stdout + result.stderr).splitlines():
        if any(k in line for k in ("Done in", "Saved", "Error", "Traceback", "frame", "Loaded")):
            print(f"    {line}")

    if result.returncode != 0:
        print(f"  FAILED ({elapsed:.0f}s)")
        print((result.stdout + result.stderr)[-800:])
        return False

    print(f"  Done in {elapsed:.0f}s")
    return True


def generate_lcm(subject):
    """Generate LCM-4step GIF using our distilled weights (CPU mode)."""
    unet_path = WEIGHTS / "unet_lcm_4step.pt"
    adapter_path = WEIGHTS / "motion_adapter_lcm_4step.pt"
    out = ASSETS / f"lcm-{subject['name']}.gif"

    if not unet_path.exists():
        print(f"  SKIP lcm-{subject['name']} — missing {unet_path}")
        return False

    cmd = [
        sys.executable, str(REPO / "examples" / "generate.py"),
        "--mode", "cpu",
        "--prompt", subject["prompt"],
        "--negative-prompt", subject["negative"],
        "--frames", "8",
        "--steps", "4",
        "--seed", str(subject["seed"]),
        "--output", str(out),
        "--lcm-unet", str(unet_path),
    ]
    if adapter_path.exists():
        cmd += ["--lcm-adapter", str(adapter_path)]

    return run_cmd(cmd, f"lcm-{subject['name']}.gif  (4-step LCM, CPU)")


def generate_lightning(subject):
    """Generate Blackhole 8-step Euler GIF."""
    out = ASSETS / f"lightning-{subject['name']}.gif"

    cmd = [
        sys.executable, str(REPO / "examples" / "generate.py"),
        "--mode", "blackhole",
        "--prompt", subject["prompt"],
        "--negative-prompt", subject["negative"],
        "--frames", "8",
        "--steps", "8",
        "--seed", str(subject["seed"]),
        "--temporal-alpha", str(subject["alpha"]),
        "--output", str(out),
        "--lightning",
    ]
    return run_cmd(cmd, f"lightning-{subject['name']}.gif  (8-step Euler, Blackhole)")


def main():
    parser = argparse.ArgumentParser(description="Regenerate Lightning Mode comparison GIFs")
    parser.add_argument(
        "--only",
        choices=["lcm", "lightning", "all"],
        default="all",
        help="Which set to generate (default: all)",
    )
    parser.add_argument(
        "--subject",
        choices=["aurora", "mandala", "mycelium"],
        default=None,
        help="Generate only one subject (default: all three)",
    )
    args = parser.parse_args()

    ASSETS.mkdir(parents=True, exist_ok=True)

    subjects = [s for s in COMPARISON_SUBJECTS if args.subject is None or s["name"] == args.subject]
    run_lcm = args.only in ("lcm", "all")
    run_lightning = args.only in ("lightning", "all")

    total = len(subjects) * (run_lcm + run_lightning)
    done = 0

    for s in subjects:
        if run_lcm:
            print(f"\n{'='*55}")
            print(f"  LCM · {s['name']}")
            print(f"{'='*55}")
            generate_lcm(s)
            done += 1
            print(f"  [{done}/{total}]")

        if run_lightning:
            out = ASSETS / f"lightning-{s['name']}.gif"
            if out.exists() and out.stat().st_size > 0:
                print(f"\n  skip lightning-{s['name']}.gif (already exists)")
            else:
                print(f"\n{'='*55}")
                print(f"  Lightning · {s['name']}")
                print(f"{'='*55}")
                generate_lightning(s)
            done += 1
            print(f"  [{done}/{total}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
