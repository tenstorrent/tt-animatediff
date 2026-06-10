#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Gallery generation script — 10 prompts × 2 modes (standard + Lightning).

Runs on Blackhole hardware. Standard: 16 frames, 25 steps, alpha=0.35.
Lightning: 16 frames, 4 steps, alpha=0.35.

Usage:
    source ~/tt-metal/python_env/bin/activate
    python scripts/generate_gallery.py [--only <slug>] [--mode standard|lightning|both]
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


def run_one(slug, prompt, negative, seed, lightning, frames=16):
    mode_label = "lightning" if lightning else "standard"
    out_path = OUT / f"{slug}-{mode_label}.gif"

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
    if lightning:
        # 25 Euler steps — same count as standard PNDM for apples-to-apples quality.
        # Our TTNN path doesn't use Lightning's motion adapter, so any step count is
        # valid; we keep --lightning for the Euler scheduler + CFG=1.0 behaviour.
        cmd += ["--lightning", "--steps", "25"]
    else:
        cmd += ["--steps", "25"]

    print(f"\n{'⚡' if lightning else '◆'} {slug} [{mode_label}]")
    print(f"  → {out_path.name}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    # Print stderr lines that matter (suppress UMD noise)
    for line in result.stderr.splitlines():
        if any(k in line for k in ("Done in", "Saved", "Error", "Traceback", "frame")):
            print(" ", line)

    if result.returncode != 0:
        print(f"  FAILED ({elapsed:.0f}s)")
        print(result.stderr[-800:])
        return False

    print(f"  Done in {elapsed:.0f}s")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", metavar="SLUG", help="Run only this slug")
    parser.add_argument("--mode", choices=["standard", "lightning", "both"], default="both")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    prompts = [p for p in PROMPTS if not args.only or p["slug"] == args.only]
    if not prompts:
        print(f"No prompt with slug '{args.only}'")
        sys.exit(1)

    run_standard = args.mode in ("standard", "both")
    run_lightning = args.mode in ("lightning", "both")

    total = len(prompts) * (run_standard + run_lightning)
    done = 0

    for p in prompts:
        for lightning in ([False] if run_standard else []) + ([True] if run_lightning else []):
            key = f"{p['slug']}-{'lightning' if lightning else 'standard'}"
            if manifest.get(key) == "done":
                print(f"  skip {key} (already done)")
                done += 1
                continue

            manifest[key] = "running"
            save_manifest(manifest)

            ok = run_one(p["slug"], p["prompt"], p["negative"], p["seed"], lightning)
            manifest[key] = "done" if ok else "failed"
            save_manifest(manifest)

            done += 1
            print(f"  Progress: {done}/{total}")

    print("\nAll done.")
    failed = [k for k, v in manifest.items() if v == "failed"]
    if failed:
        print(f"  Failed: {failed}")
        print(f"  Retry with: python scripts/generate_gallery.py --only <slug>")


if __name__ == "__main__":
    main()
