#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Resumable batch generation script for the Cosmic Study website.

Run from the repo root inside the tt-metal python env:
    source ~/tt-metal/python_env/bin/activate
    python scripts/generate_study.py

Skips GIFs that already exist and are non-zero. Safe to Ctrl-C and re-run.

Options:
    --only <name>   Run a single GIF by name (e.g. --only p1-nebula)
    --list          Print all GIF names and exit
"""

import json
import sys
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
OUT_DIR = REPO_ROOT / "docs" / "assets" / "study"
MANIFEST = OUT_DIR / "manifest.json"
GENERATE = REPO_ROOT / "examples" / "generate.py"
PYTHON = Path.home() / "tt-metal" / "python_env" / "bin" / "python"

NEG = "blurry, low quality, distorted, text, people, faces, modern buildings"

GIFS = [
    # Phase 1 — Pure cosmic (teal, alpha 0.35)
    dict(name="p1-nebula",        row=0, col=0,  phase=1, alpha=0.35, seed=42,
         prompt="swirling nebula in deep space, purple and teal gas clouds, stars, cinematic 4K"),
    dict(name="p1-galaxy",        row=0, col=2,  phase=1, alpha=0.35, seed=42,
         prompt="spiral galaxy seen from above, golden core, trailing arms of blue starlight, cinematic"),
    dict(name="p1-aurora",        row=0, col=4,  phase=1, alpha=0.35, seed=42,
         prompt="aurora borealis dancing over arctic ice, green and violet ribbons, starfield, cinematic"),
    dict(name="p1-starfield",     row=0, col=6,  phase=1, alpha=0.35, seed=42,
         prompt="infinite starfield rotating slowly, Milky Way band, depth of field, cosmic, cinematic 4K"),
    dict(name="p1-solar-flare",   row=0, col=8,  phase=1, alpha=0.35, seed=42,
         prompt="solar flare erupting from sun surface, plasma arcs, golden orange, cinematic"),
    dict(name="p1-gas-cloud",     row=0, col=10, phase=1, alpha=0.35, seed=42,
         prompt="interstellar gas cloud glowing teal and pink, cosmic dust, nebular light, cinematic 4K"),
    dict(name="p1-crystal-cave",  row=1, col=1,  phase=1, alpha=0.35, seed=42,
         prompt="crystal cave glowing with bioluminescent cosmic light, amethyst formations, starlight within"),
    dict(name="p1-deep-ocean",    row=1, col=3,  phase=1, alpha=0.35, seed=42,
         prompt="deep ocean at night, bioluminescent creatures, stars reflected on black water, cinematic"),
    dict(name="p1-lunar-halo",    row=1, col=6,  phase=1, alpha=0.35, seed=42,
         prompt="full moon with atmospheric halo, aurora behind it, deep blue sky, cinematic 4K"),
    dict(name="p1-north-lights",  row=1, col=8,  phase=1, alpha=0.35, seed=42,
         prompt="northern lights over ancient pine forest, green and magenta, slow swirl, cinematic"),
    dict(name="p1-milky-way",     row=2, col=0,  phase=1, alpha=0.35, seed=42,
         prompt="Milky Way over desert landscape, rock formations silhouetted, cosmic scale, cinematic 4K"),
    dict(name="p1-fractal-tunnel", row=2, col=3, phase=1, alpha=0.35, seed=42,
         prompt="fractal space tunnel, recursive geometry, teal and gold, infinite zoom, cinematic"),
    # Phase 2 — Intertwining oneness (pink, alpha 0.4)
    dict(name="p2-mandala",       row=2, col=7,  phase=2, alpha=0.4, seed=42,
         prompt="sacred mandala blooming from starfield, geometric petals, gold and violet, cosmic, cinematic"),
    dict(name="p2-mycelium",      row=2, col=10, phase=2, alpha=0.4, seed=42,
         prompt="mycelium network glowing with bioluminescent spores, threads of light connecting nodes, cosmic forest"),
    dict(name="p2-dna",           row=3, col=1,  phase=2, alpha=0.4, seed=42,
         prompt="DNA double helix rotating in cosmic void, glowing teal, stars inside, cinematic 4K"),
    dict(name="p2-roots-cosmos",  row=3, col=5,  phase=2, alpha=0.4, seed=42,
         prompt="ancient tree roots intertwining with stars and galaxies, roots become light filaments, cinematic"),
    dict(name="p2-consciousness",  row=3, col=9, phase=2, alpha=0.4, seed=42,
         prompt="threads of consciousness connecting glowing nodes, neural web across dark void, teal and gold"),
    dict(name="p2-sacred-geo",    row=4, col=0,  phase=2, alpha=0.4, seed=42,
         prompt="sacred geometry unfolding in cosmic space, flower of life, metatrons cube, golden ratio spirals"),
    dict(name="p2-forest-mind",   row=4, col=4,  phase=2, alpha=0.4, seed=42,
         prompt="forest canopy seen from below as neural network, branches are synapses, bioluminescent, cinematic"),
    dict(name="p2-reef",          row=4, col=7,  phase=2, alpha=0.4, seed=42,
         prompt="living coral reef as collective mind, tendrils of light, pulsing bioluminescence, cosmic ocean"),
    dict(name="p2-meridians",     row=4, col=10, phase=2, alpha=0.4, seed=42,
         prompt="energy meridians flowing across a body silhouette, acupuncture lines glowing gold, cosmic backdrop"),
    # Phase 3 — Liminal threshold (gold, alpha 0.4)
    dict(name="p3-temple",        row=5, col=2,  phase=3, alpha=0.4, seed=42,
         prompt="ancient Mayan temple under shifting cosmos, nebula behind it, jungle emerging, cinematic 4K"),
    dict(name="p3-cave-circuit",  row=5, col=6,  phase=3, alpha=0.4, seed=42,
         prompt="cave paintings of animals slowly morphing into circuit traces, orange firelight to teal digital glow"),
    dict(name="p3-standing-stones", row=5, col=9, phase=3, alpha=0.4, seed=42,
         prompt="stone circle at dawn with aurora, ancient megaliths glowing with interior light, cosmic sky"),
    dict(name="p3-crystal-data",  row=6, col=1,  phase=3, alpha=0.4, seed=42, lightning=True,
         prompt="amethyst and rose quartz crystal formations fused with glowing circuit traces, violet light refracting through silicon faces, pink data streams flowing along crystal edges, macro crystallography, deep dark background, cinematic 4K"),
    dict(name="p3-ley-fiber",     row=6, col=5,  phase=3, alpha=0.4, seed=42,
         prompt="ancient ley lines across landscape glowing gold, slowly becoming fiber optic cables, aerial view"),
    dict(name="p3-mayan-grid",    row=6, col=10, phase=3, alpha=0.4, seed=42,
         prompt="Mayan calendar stone dissolving into computational grid, stone becomes silicon, cosmic light"),
    # Phase 4 — Tech emergence (green, alpha 0.45)
    dict(name="p4-circuit-moss",  row=7, col=0,  phase=4, alpha=0.45, seed=42,
         prompt="circuit board growing like moss and lichen, organic circuitry, green teal glow, macro, cinematic"),
    dict(name="p4-silicon",       row=7, col=4,  phase=4, alpha=0.45, seed=42, lightning=True,
         prompt="Tensix compute core dissolving into visible logic: rippling pink and violet waveforms cascading across a dark wafer, each wave a matrix multiply completing, the arithmetic made visible as color, macro semiconductor photography"),
    dict(name="p4-neural-const",  row=7, col=8,  phase=4, alpha=0.45, seed=42, lightning=True,
         prompt="swarm of luminous butterflies emerging from a circuit board, each wing printed with logic gates, fuchsia and magenta iridescence, dissolving back into silicon at the edges, macro nature photography, dark background"),
    dict(name="p4-server-aurora", row=8, col=2,  phase=4, alpha=0.45, seed=42,
         prompt="server room bathed in aurora borealis light through glass ceiling, racks of machines, cinematic"),
    dict(name="p4-chip-city",     row=8, col=7,  phase=4, alpha=0.45, seed=42, lightning=True,
         prompt="aerial view of a glowing city at night where every street is a data bus and every building is a compute tile, purple and magenta neon, tight urban grid with Tensix-style hexagonal districts, cinematic 4K overhead shot"),
    # Phase 5 — Full integration (bright teal, alpha 0.5)
    dict(name="p5-chip-cosmos",   row=9, col=0,  phase=5, alpha=0.5, seed=42,
         prompt="Tenstorrent Blackhole chip glowing with embedded cosmos, galaxies visible inside the die, cinematic"),
    dict(name="p5-hand-galaxy",   row=9, col=5,  phase=5, alpha=0.5, seed=42,
         prompt="human hand touching a circuit board that blooms into a galaxy on contact, light erupting, cinematic"),
    dict(name="p5-grid-is-all",   row=9, col=10, phase=5, alpha=0.5, seed=42,
         prompt="computational grid slowly revealed to be the fabric of the universe itself, zoom out from chip to cosmos"),
]


def load_manifest():
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def save_manifest(manifest):
    MANIFEST.write_text(json.dumps(manifest, indent=2))


def gif_done(name, manifest):
    path = OUT_DIR / f"{name}.gif"
    return manifest.get(name) == "done" and path.exists() and path.stat().st_size > 0


def run_gif(g, manifest, index, total):
    name = g["name"]
    out = OUT_DIR / f"{name}.gif"
    lt = " ⚡ lightning" if g.get("lightning") else ""
    print(f"\n{'='*60}")
    print(f"  [{index}/{total}] {name}{lt}")
    print(f"  phase={g['phase']}  alpha={g['alpha']}  seed={g['seed']}")
    print(f"  {g['prompt'][:80]}...")
    print(f"{'='*60}")
    sys.stdout.flush()

    manifest[name] = "running"
    save_manifest(manifest)

    cmd = [
        str(PYTHON), str(GENERATE),
        "--mode", "blackhole",
        "--prompt", g["prompt"],
        "--negative-prompt", NEG,
        "--frames", "8",
        "--steps", "25",
        "--seed", str(g["seed"]),
        "--temporal-alpha", str(g["alpha"]),
        "--output", str(out),
    ]
    if g.get("lightning"):
        cmd.append("--lightning")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    elapsed = time.time() - t0

    if result.returncode == 0 and out.exists() and out.stat().st_size > 0:
        manifest[name] = "done"
        print(f"  ✓ done in {elapsed:.0f}s → {out.name}")
    else:
        manifest[name] = "failed"
        print(f"  ✗ FAILED (exit {result.returncode}) after {elapsed:.0f}s")
        print(f"    Retry: python scripts/generate_study.py --only {name}")

    save_manifest(manifest)
    return manifest[name] == "done"


def main():
    if "--list" in sys.argv:
        for g in GIFS:
            print(f"  {g['name']:25s}  phase={g['phase']}  ({g['row']},{g['col']})")
        return

    only = None
    if "--only" in sys.argv:
        idx = sys.argv.index("--only")
        if idx + 1 >= len(sys.argv):
            print("Error: --only requires a name argument", file=sys.stderr)
            sys.exit(1)
        only = sys.argv[idx + 1]
        if not any(g["name"] == only for g in GIFS):
            print(f"Error: '{only}' not in GIF list. Run --list to see names.", file=sys.stderr)
            sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    targets = [g for g in GIFS if only is None or g["name"] == only]
    pending = [g for g in targets if not gif_done(g["name"], manifest)]
    done_count = sum(1 for g in GIFS if gif_done(g["name"], manifest))

    print(f"Cosmic Study generation")
    print(f"  Total: {len(GIFS)}  Done: {done_count}  Pending: {len(pending)}")
    if not pending:
        print("  All done!")
        return

    for g in pending:
        global_idx = GIFS.index(g) + 1
        run_gif(g, manifest, global_idx, len(GIFS))

    done_count = sum(1 for g in GIFS if gif_done(g["name"], manifest))
    failed = [g["name"] for g in GIFS if manifest.get(g["name"]) == "failed"]
    print(f"\n{'='*60}")
    print(f"  Done: {done_count}/{len(GIFS)}")
    if failed:
        print(f"  Failed: {failed}")
        print(f"  Retry each: python scripts/generate_study.py --only <name>")
        print(f"  If hardware hung: tt-smi -r 0 1 2 3 && sleep 8, then re-run")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
