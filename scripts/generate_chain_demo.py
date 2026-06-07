#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Chain continuity demo — retro 3D glasses across contexts and styles.

Demonstrates --chain latent threading: each scene saves its final denoised
latents and the next scene blends them into its seed noise, creating visual
DNA continuity across independent prompts.

Subject: retro red-and-cyan anaglyphic 3D glasses, a recognizable shape that
threads through wildly different contexts and colour palettes.

Usage:
    source ~/tt-metal/python_env/bin/activate
    python scripts/generate_chain_demo.py

Skips scenes whose output GIF already exists (safe to resume after Ctrl-C).
"""

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).parent.parent
OUT = REPO / "docs" / "assets" / "chain"
CHAIN_DIR = OUT / "latents"
MANIFEST = OUT / "manifest.json"
GENERATE = REPO / "examples" / "generate.py"
PYTHON = Path.home() / "tt-metal" / "python_env" / "bin" / "python"

NEG = "blurry, low quality, distorted, text, people, faces, modern buildings"

# Each scene lists: slug, prompt, negative, seed, alpha, lightning
# The chain_alpha is tuned per scene — earlier scenes need tighter coupling
# to establish the latent fingerprint; later scenes can loosen it for variety.
SCENES = [
    {
        "slug": "glasses-neon",
        "prompt": "retro red-and-cyan 3D glasses resting on a neon-lit diner counter at night, "
                  "reflections in the lenses showing a glowing jukebox, chromatic aberration, "
                  "cinematic 35mm film photography",
        "seed": 111,
        "alpha": 0.35,
        "lightning": False,
        "chain_alpha": None,  # first scene — no chain input
    },
    {
        "slug": "glasses-cosmic",
        "prompt": "retro red-and-cyan 3D glasses floating in deep space, lenses filled with "
                  "swirling violet and magenta nebulae, starfield reflection, macro photography, "
                  "cinematic 4K",
        "seed": 222,
        "alpha": 0.40,
        "lightning": True,
        "chain_alpha": 0.55,
    },
    {
        "slug": "glasses-forest",
        "prompt": "retro red-and-cyan 3D glasses perched on a mossy ancient tree root in a "
                  "bioluminescent forest, teal spores drifting, pink fungi glowing, macro, "
                  "nature photography",
        "seed": 333,
        "alpha": 0.40,
        "lightning": False,
        "chain_alpha": 0.50,
    },
    {
        "slug": "glasses-circuit",
        "prompt": "retro red-and-cyan 3D glasses dissolving into a glowing circuit board, "
                  "lens outlines traced in violet and pink data streams, macro semiconductor "
                  "photography, Tenstorrent aesthetic",
        "seed": 444,
        "alpha": 0.45,
        "lightning": True,
        "chain_alpha": 0.55,
    },
    {
        "slug": "glasses-watercolor",
        "prompt": "retro red-and-cyan 3D glasses rendered in flowing watercolor washes, "
                  "wet-on-wet blooms of crimson and cerulean, white paper texture, "
                  "studio art, ultra-detailed",
        "seed": 555,
        "alpha": 0.35,
        "lightning": False,
        "chain_alpha": 0.45,
    },
    {
        "slug": "glasses-ocean",
        "prompt": "retro red-and-cyan 3D glasses resting on coral in a shallow tropical reef, "
                  "fish passing through lenses, caustic light patterns, underwater macro "
                  "photography, vivid saturated colors",
        "seed": 666,
        "alpha": 0.40,
        "lightning": True,
        "chain_alpha": 0.50,
    },
]


def load_manifest():
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def save_manifest(m):
    MANIFEST.write_text(json.dumps(m, indent=2))


def scene_done(slug, manifest):
    p = OUT / f"{slug}.gif"
    return manifest.get(slug) == "done" and p.exists() and p.stat().st_size > 0


def run_scene(scene, prev_slug, manifest, idx, total):
    slug = scene["slug"]
    out_gif = OUT / f"{slug}.gif"
    chain_pt = CHAIN_DIR / f"{slug}.pt"
    prev_pt = CHAIN_DIR / f"{prev_slug}.pt" if prev_slug else None

    lt = " ⚡" if scene["lightning"] else ""
    print(f"\n{'='*60}")
    print(f"  [{idx}/{total}] {slug}{lt}")
    if prev_pt:
        print(f"  chain_from={prev_pt.name}  chain_alpha={scene['chain_alpha']}")
    print(f"  {scene['prompt'][:80]}...")
    print(f"{'='*60}")
    sys.stdout.flush()

    manifest[slug] = "running"
    save_manifest(manifest)

    cmd = [
        str(PYTHON), str(GENERATE),
        "--mode", "blackhole",
        "--prompt", scene["prompt"],
        "--negative-prompt", NEG,
        "--frames", "8",
        "--steps", "25",
        "--seed", str(scene["seed"]),
        "--temporal-alpha", str(scene["alpha"]),
        "--output", str(out_gif),
        "--chain-save", str(chain_pt),
    ]
    if scene["lightning"]:
        cmd.append("--lightning")
    if prev_pt and prev_pt.exists():
        cmd += ["--chain-from", str(prev_pt), "--chain-alpha", str(scene["chain_alpha"])]

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(REPO))
    elapsed = time.time() - t0

    if result.returncode == 0 and out_gif.exists() and out_gif.stat().st_size > 0:
        manifest[slug] = "done"
        print(f"  ✓ done in {elapsed:.0f}s → {out_gif.name}")
    else:
        manifest[slug] = "failed"
        print(f"  ✗ FAILED (exit {result.returncode}) after {elapsed:.0f}s")
        print(f"    Retry: python scripts/generate_chain_demo.py --only {slug}")

    save_manifest(manifest)
    return manifest[slug] == "done"


def main():
    only = None
    if "--only" in sys.argv:
        idx = sys.argv.index("--only")
        only = sys.argv[idx + 1]
        if not any(s["slug"] == only for s in SCENES):
            print(f"Error: '{only}' not in scene list.", file=sys.stderr)
            sys.exit(1)

    if "--list" in sys.argv:
        for s in SCENES:
            lt = " ⚡" if s["lightning"] else ""
            print(f"  {s['slug']:30s}  alpha={s['alpha']}{lt}")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    CHAIN_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    targets = [s for s in SCENES if only is None or s["slug"] == only]
    pending = [s for s in targets if not scene_done(s["slug"], manifest)]
    done_count = sum(1 for s in SCENES if scene_done(s["slug"], manifest))

    print(f"Chain demo — {len(SCENES)} scenes")
    print(f"  Done: {done_count}  Pending: {len(pending)}")
    if not pending:
        print("  All done!")
        return

    for scene in pending:
        scene_idx = SCENES.index(scene)
        prev_slug = SCENES[scene_idx - 1]["slug"] if scene_idx > 0 else None
        run_scene(scene, prev_slug, manifest, scene_idx + 1, len(SCENES))

    done_count = sum(1 for s in SCENES if scene_done(s["slug"], manifest))
    failed = [s["slug"] for s in SCENES if manifest.get(s["slug"]) == "failed"]
    print(f"\n{'='*60}")
    print(f"  Done: {done_count}/{len(SCENES)}")
    if failed:
        print(f"  Failed: {failed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
