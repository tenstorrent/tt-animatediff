#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Gallery generation script — 10 prompts × 4 modes.

Modes
-----
standard   Blackhole TTNN UNet, PNDMScheduler, 25 steps, CFG=7.5
lightning  Blackhole TTNN UNet, EulerDiscreteScheduler (trailing/linear), 25 steps, CFG=7.5
lcm-8step  CPU AnimateDiffPipeline + our LCM-distilled UNet (8-step, CFG=1.0)
lcm-4step  CPU AnimateDiffPipeline + our LCM-distilled UNet + MotionAdapter (4-step, CFG=1.0)

Usage:
    source ~/tt-metal/python_env/bin/activate
    python scripts/generate_gallery.py [--only <slug>] [--mode standard|lightning|lcm-8step|lcm-4step|all]

LCM modes require distilled weight files in weights/:
    weights/unet_lcm_8step.pt
    weights/unet_lcm_4step.pt
    weights/motion_adapter_lcm_4step.pt
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).parent.parent
OUT = REPO / "docs" / "assets" / "gallery"
MANIFEST = OUT / "manifest.json"
WEIGHTS = REPO / "weights"

PROMPTS = [
    {
        "slug": "desert-storm",
        "prompt": "sweeping desert sandstorm at sunset, dunes and swirling ochre dust, Sahara, golden hour, cinematic 4K",
        "negative": "blurry, low quality, text, people, faces",
        "seed": 101,
    },
    {
        "slug": "lava-flow",
        "prompt": "molten lava flowing into the ocean at night, glowing orange cracks, steam explosions, dramatic macro photography",
        "negative": "blurry, low quality, text, people, faces",
        "seed": 202,
    },
    {
        "slug": "japanese-garden",
        "prompt": "cherry blossom petals falling over a Japanese zen garden, koi pond reflections, soft spring light, watercolor painting",
        "negative": "blurry, low quality, text, people, faces, modern buildings",
        "seed": 303,
    },
    {
        "slug": "deep-sea",
        "prompt": "deep sea hydrothermal vent, alien bioluminescent ecosystem, creatures drifting in darkness, BBC nature documentary",
        "negative": "blurry, low quality, text, people, faces, bright light",
        "seed": 404,
    },
    {
        "slug": "cathedral",
        "prompt": "stained glass light flooding a Gothic cathedral nave, dust motes floating, volumetric light beams, oil painting style",
        "negative": "blurry, low quality, text, people, faces, modern",
        "seed": 505,
    },
    {
        "slug": "supernova",
        "prompt": "supernova explosion in deep space, shockwave rippling through nebula, purple and orange gas clouds, Hubble telescope",
        "negative": "blurry, low quality, text, planets, stars too sharp",
        "seed": 606,
    },
    {
        "slug": "crystal-cave",
        "prompt": "crystal cave interior, hexagonal amethyst and quartz formations, shimmering light refractions, macro photography",
        "negative": "blurry, low quality, text, people, faces",
        "seed": 707,
    },
    {
        "slug": "silk-road",
        "prompt": "merchant caravan crossing the Silk Road through dramatic mountain passes, storm clouds gathering, cinematic dusk, oil painting",
        "negative": "blurry, low quality, text, modern vehicles, people closeup",
        "seed": 808,
    },
    {
        "slug": "clockwork",
        "prompt": "intricate clockwork mechanism close-up, brass gears and cogs turning, steam engine aesthetic, steampunk macro, warm amber light",
        "negative": "blurry, low quality, text, people, faces, plastic",
        "seed": 909,
    },
    {
        "slug": "arctic-wave",
        "prompt": "massive arctic wave cresting, translucent blue-green glacial ice, spray catching low winter sun, cinematic slow motion",
        "negative": "blurry, low quality, text, people, faces, tropical",
        "seed": 1010,
    },
]


def load_manifest():
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def save_manifest(m):
    MANIFEST.write_text(json.dumps(m, indent=2))


def run_one(slug, prompt, negative, seed, mode_label, frames=16):
    """Run inference for one slug+mode and return True on success.

    mode_label: "standard", "lightning", "lcm-8step", or "lcm-4step"
    """
    out_path = OUT / f"{slug}-{mode_label}.gif"

    if mode_label in ("lcm-8step", "lcm-4step"):
        # CPU pipeline with our distilled weights
        unet_steps = "8" if mode_label == "lcm-8step" else "4"
        unet_path = WEIGHTS / f"unet_lcm_{unet_steps}step.pt"
        if not unet_path.exists():
            print(f"  SKIP {slug} [{mode_label}] — weights not found: {unet_path}")
            return None  # not a failure, just missing weights

        cmd = [
            sys.executable,
            str(REPO / "examples" / "generate.py"),
            "--mode", "cpu",
            "--prompt", prompt,
            "--negative-prompt", negative,
            "--frames", str(frames),
            "--seed", str(seed),
            "--output", str(out_path),
            "--lcm-unet", str(unet_path),
            "--steps", unet_steps,
        ]
        if mode_label == "lcm-4step":
            # Full LCM: distilled UNet + distilled MotionAdapter
            adapter_path = WEIGHTS / "motion_adapter_lcm_4step.pt"
            if adapter_path.exists():
                cmd += ["--lcm-adapter", str(adapter_path)]
    else:
        # Blackhole TTNN backend (standard or lightning solver)
        cmd = [
            sys.executable,
            str(REPO / "examples" / "generate.py"),
            "--mode", "blackhole",
            "--prompt", prompt,
            "--negative-prompt", negative,
            "--frames", str(frames),
            "--seed", str(seed),
            "--temporal-alpha", "0.35",
            "--output", str(out_path),
        ]
        if mode_label == "lightning":
            # 25 Euler steps — same count as standard PNDM for apples-to-apples.
            # TTNN path uses base SD 1.4 UNet (no distilled adapter); --lightning
            # switches to EulerDiscreteScheduler (trailing, linear), CFG=7.5.
            cmd += ["--lightning", "--steps", "25"]
        else:
            cmd += ["--steps", "25"]

    icon = {"standard": "◆", "lightning": "⚡", "lcm-8step": "🔵", "lcm-4step": "🟢"}.get(mode_label, "◆")
    print(f"\n{icon} {slug} [{mode_label}]")
    print(f"  → {out_path.name}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    # Print lines that matter (suppress UMD noise)
    for line in (result.stdout + result.stderr).splitlines():
        if any(k in line for k in ("Done in", "Saved", "Error", "Traceback", "frame")):
            print(" ", line)

    if result.returncode != 0:
        print(f"  FAILED ({elapsed:.0f}s)")
        print((result.stdout + result.stderr)[-800:])
        return False

    print(f"  Done in {elapsed:.0f}s")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", metavar="SLUG", help="Run only this slug")
    parser.add_argument(
        "--mode",
        choices=["standard", "lightning", "lcm-8step", "lcm-4step", "all"],
        default="all",
        help="Which mode(s) to run (default: all)",
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    prompts = [p for p in PROMPTS if not args.only or p["slug"] == args.only]
    if not prompts:
        print(f"No prompt with slug '{args.only}'")
        sys.exit(1)

    all_modes = ["standard", "lightning", "lcm-8step", "lcm-4step"]
    modes = all_modes if args.mode == "all" else [args.mode]

    total = len(prompts) * len(modes)
    done = 0

    for p in prompts:
        for mode_label in modes:
            key = f"{p['slug']}-{mode_label}"
            if manifest.get(key) == "done":
                print(f"  skip {key} (already done)")
                done += 1
                continue

            manifest[key] = "running"
            save_manifest(manifest)

            ok = run_one(p["slug"], p["prompt"], p["negative"], p["seed"], mode_label)
            if ok is None:
                # Weights missing — skip without marking failed
                manifest.pop(key, None)
            else:
                manifest[key] = "done" if ok else "failed"
            save_manifest(manifest)

            done += 1
            print(f"  Progress: {done}/{total}")

    print("\nAll done.")
    failed = [k for k, v in manifest.items() if v == "failed"]
    if failed:
        print(f"  Failed: {failed}")
        print(f"  Retry with: python scripts/generate_gallery.py --only <slug> --mode <mode>")


if __name__ == "__main__":
    main()
