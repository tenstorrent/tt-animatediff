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

NEG = (
    "blurry, low quality, distorted, text, watermark, people, faces, "
    "missing glasses, glasses removed, bare counter, empty scene"
)

# Canonical SD 1.4 prompt style:
#   - Lead token = most important visual noun
#   - (token:weight) attention syntax to emphasise subject
#   - Scene context after subject, not before
#   - Colour descriptors placed close to their subjects
#   - Style/quality tags at the end
#
# chain_alpha 0.20: blurred low-pass signal is sparse; 20% is enough for
# composition bias without drowning the text conditioning.
SCENES = [
    {
        "slug": "glasses-neon",
        "prompt": "(retro anaglyphic 3D glasses:1.5), chromatic lens flare, "
                  "pink neon reflection in left lens, teal neon in right lens, "
                  "wet formica diner counter, rain on window, night scene, "
                  "35mm Kodak film grain, shallow depth of field, cinematic",
        "seed": 111,
        "alpha": 0.65,
        "lightning": False,
        "chain_alpha": None,  # first scene — no chain input
    },
    {
        "slug": "glasses-cosmic",
        "prompt": "(retro anaglyphic 3D glasses:1.5), floating in zero gravity, "
                  "violet nebula fills left lens, indigo starfield in right lens, "
                  "deep space background, cold blue-white stars, "
                  "macro product photography, 4K cinematic",
        "seed": 222,
        "alpha": 0.68,
        "lightning": False,
        "chain_alpha": 0.20,
    },
    {
        "slug": "glasses-forest",
        "prompt": "(retro anaglyphic 3D glasses:1.5), resting on a moss-covered log, "
                  "emerald green forest, golden god-rays through canopy, "
                  "bioluminescent blue spores in bokeh background, "
                  "macro nature photography, morning mist",
        "seed": 333,
        "alpha": 0.65,
        "lightning": False,
        "chain_alpha": 0.20,
    },
    {
        "slug": "glasses-circuit",
        "prompt": "(retro anaglyphic 3D glasses:1.5), on a glowing PCB, "
                  "fuchsia data-stream traces, violet solder points, "
                  "teal LED backlighting, dark studio, "
                  "macro semiconductor photography, sharp focus",
        "seed": 444,
        "alpha": 0.68,
        "lightning": False,
        "chain_alpha": 0.20,
    },
    {
        "slug": "glasses-watercolor",
        "prompt": "(retro anaglyphic 3D glasses:1.5), loose watercolor painting style, "
                  "cerulean wash, gold wet-on-wet blooms, ink linework, "
                  "white paper grain showing through, studio art, "
                  "ultra-detailed illustration",
        "seed": 555,
        "alpha": 0.65,
        "lightning": False,
        "chain_alpha": 0.20,
    },
    {
        "slug": "glasses-ocean",
        "prompt": "(retro anaglyphic 3D glasses:1.5), submerged in shallow coral reef, "
                  "turquoise caustic light patterns, orange clownfish nearby, "
                  "purple sea anemone, crystal clear tropical water, "
                  "underwater macro photography, vivid saturated colors",
        "seed": 666,
        "alpha": 0.65,
        "lightning": False,
        "chain_alpha": 0.20,
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
