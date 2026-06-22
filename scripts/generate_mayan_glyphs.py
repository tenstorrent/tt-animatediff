#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Mayan glyph benchmark suite for tt-animatediff.

Generates animated GIFs for all 20 Maya day glyphs of the tzolk'in (260-day
sacred calendar) on a 4-chip Blackhole QB2 board.  Each glyph is rendered at
best quality: PNDM 25-step, 8 frames, 4-chip mesh sharding.

The 20 day glyphs (in traditional Yucatec Maya order):
  Imix, Ik', Ak'b'al, K'an, Chikchan, Kimi, Manik', Lamat, Muluk, Ok,
  Chuwen, Eb', B'en, Ix, Men, K'ib', Kaban, Etz'nab', Kawak, Ajaw

Usage:
  source ~/tt-metal/python_env/bin/activate
  python scripts/generate_mayan_glyphs.py [--dry-run] [--tier Q1|Q2|Q3|Q4|all] [--glyph <slug>] [--sample]
  python3 -m http.server 8080 --directory docs/
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).parent.parent
OUT = REPO / "docs" / "assets" / "mayan-glyphs"
GENERATE = REPO / "examples" / "generate.py"

# ── 20 Maya day glyphs (tzolk'in calendar) ─────────────────────────────────
# Prompts are written to produce rich, cinematic, animated still-life glyphs:
# the stone glyph itself in motion — lit, atmospheric, alive.

GLYPHS = [
    {
        "slug": "imix",
        "name": "Imix",
        "number": 1,
        "meaning": "Crocodile / Primordial Water",
        "prompt": (
            "Ancient Maya stone glyph of Imix, the primordial crocodile and first day of "
            "creation, carved deep into jade-green limestone, water rippling across the "
            "glyph surface, dark jungle cenote behind, bioluminescent algae drifting, "
            "torch smoke curling upward, hyper-detailed Maya iconography, cinematic"
        ),
        "negative": "blurry, low quality, modern, letters, text, western",
        "seed": 1,
    },
    {
        "slug": "ik",
        "name": "Ik'",
        "number": 2,
        "meaning": "Wind / Breath",
        "prompt": (
            "Ancient Maya stone glyph of Ik', the wind and breath of life, carved in obsidian-black "
            "volcanic rock, visible wind lines swirling off the surface, jungle canopy bending in "
            "the breeze behind, cacao leaves spinning, twilight blue-violet sky, "
            "hyper-detailed Maya iconography, cinematic"
        ),
        "negative": "blurry, low quality, modern, letters, text, western",
        "seed": 2,
    },
    {
        "slug": "akbal",
        "name": "Ak'b'al",
        "number": 3,
        "meaning": "Night / Darkness",
        "prompt": (
            "Ancient Maya stone glyph of Ak'b'al, the house of darkness and night, "
            "carved in deep obsidian stone, Milky Way stars visible through an open temple doorway, "
            "stars reflected on the glyph surface, fireflies drifting past, "
            "moonlight rim-lighting the carved edges, "
            "hyper-detailed Maya iconography, cinematic chiaroscuro"
        ),
        "negative": "blurry, low quality, modern, letters, text, western",
        "seed": 3,
    },
    {
        "slug": "kan",
        "name": "K'an",
        "number": 4,
        "meaning": "Seed / Corn / Abundance",
        "prompt": (
            "Ancient Maya stone glyph of K'an, the yellow corn seed and abundance, "
            "carved in warm golden sandstone, maize cobs woven into the border relief, "
            "morning sunlight raking across the carved surface, pollen drifting through the air, "
            "terraced cornfields of the Yucatan behind, "
            "hyper-detailed Maya iconography, golden hour cinematic"
        ),
        "negative": "blurry, low quality, modern, letters, text, western",
        "seed": 4,
    },
    {
        "slug": "chikchan",
        "name": "Chikchan",
        "number": 5,
        "meaning": "Serpent / Life Force",
        "prompt": (
            "Ancient Maya stone glyph of Chikchan, the celestial serpent and life force, "
            "carved in red volcanic stone, a feathered serpent sinuously winding around the glyph, "
            "scales shimmering in firelight, jungle rain beginning, "
            "Chichen Itza temple visible in mist behind, "
            "hyper-detailed Maya iconography, cinematic"
        ),
        "negative": "blurry, low quality, modern, letters, text, western",
        "seed": 5,
    },
    {
        "slug": "kimi",
        "name": "Kimi",
        "number": 6,
        "meaning": "Death / Transformation",
        "prompt": (
            "Ancient Maya stone glyph of Kimi, the god of death and transformation, "
            "carved in pale white limestone with skull motifs in the border, "
            "incense smoke coiling through the underworld darkness, "
            "jade death mask leaning against the stone, flickering copal flame, "
            "hyper-detailed Maya iconography, dramatic cinematic lighting"
        ),
        "negative": "blurry, low quality, modern, letters, text, western",
        "seed": 6,
    },
    {
        "slug": "manik",
        "name": "Manik'",
        "number": 7,
        "meaning": "Deer / Hand",
        "prompt": (
            "Ancient Maya stone glyph of Manik', the deer and the hand of offering, "
            "carved in warm terracotta stone, a white-tailed deer stepping past in the jungle background, "
            "dappled forest light through the canopy above, morning mist rising, "
            "hyper-detailed Maya iconography, cinematic forest atmosphere"
        ),
        "negative": "blurry, low quality, modern, letters, text, western",
        "seed": 7,
    },
    {
        "slug": "lamat",
        "name": "Lamat",
        "number": 8,
        "meaning": "Venus / Star / Rabbit",
        "prompt": (
            "Ancient Maya stone glyph of Lamat, the star of Venus, "
            "carved in midnight-blue lapis lazuli inlaid stone, Venus gleaming at dawn behind, "
            "star-shaped carvings catching first light, morning dew on the carved relief, "
            "Yucatan horizon glowing rose and gold, "
            "hyper-detailed Maya iconography, celestial cinematic"
        ),
        "negative": "blurry, low quality, modern, letters, text, western",
        "seed": 8,
    },
    {
        "slug": "muluk",
        "name": "Muluk",
        "number": 9,
        "meaning": "Jade / Water / Rain",
        "prompt": (
            "Ancient Maya stone glyph of Muluk, jade and the sacred rain, "
            "carved in deep jade-green serpentine stone, monsoon rain falling across the carved surface, "
            "water streaming down the glyphs channels, jade beads arranged at the base, "
            "Chac the rain god mask visible in the temple niche behind, "
            "hyper-detailed Maya iconography, tropical storm cinematic"
        ),
        "negative": "blurry, low quality, modern, letters, text, western",
        "seed": 9,
    },
    {
        "slug": "ok",
        "name": "Ok",
        "number": 10,
        "meaning": "Dog / Loyalty / Guide",
        "prompt": (
            "Ancient Maya stone glyph of Ok, the loyal dog who guides souls through Xibalba, "
            "carved in smooth river granite, a tan xolo dog curled beside the stone in firelight, "
            "underworld torch flames reflected on the carved surface, "
            "cave stalactites above, "
            "hyper-detailed Maya iconography, warm cave cinematic"
        ),
        "negative": "blurry, low quality, modern, letters, text, western",
        "seed": 10,
    },
    {
        "slug": "chuwen",
        "name": "Chuwen",
        "number": 11,
        "meaning": "Monkey / Arts / Time Weaver",
        "prompt": (
            "Ancient Maya stone glyph of Chuwen, the howler monkey god of arts and the weaver of time, "
            "carved in warm brown sandstone, a spider monkey hanging playfully from jungle vines above, "
            "golden afternoon light, painted Maya codex pages scattered at the base, "
            "hyper-detailed Maya iconography, playful cinematic"
        ),
        "negative": "blurry, low quality, modern, letters, text, western",
        "seed": 11,
    },
    {
        "slug": "eb",
        "name": "Eb'",
        "number": 12,
        "meaning": "Grass / Road / Skull",
        "prompt": (
            "Ancient Maya stone glyph of Eb', the road and the human skull, "
            "carved in grey limestone by the edge of the sacbe white road, "
            "tall jungle grass swaying at the stone's edges, "
            "white road stretching into misty distance, "
            "dusk light, fireflies awakening, "
            "hyper-detailed Maya iconography, atmospheric road cinematic"
        ),
        "negative": "blurry, low quality, modern, letters, text, western",
        "seed": 12,
    },
    {
        "slug": "ben",
        "name": "B'en",
        "number": 13,
        "meaning": "Reed / Corn Stalk / Staff",
        "prompt": (
            "Ancient Maya stone glyph of B'en, the growing corn stalk and reed staff, "
            "carved in pale cream limestone, tall reed grasses rustling in warm wind around it, "
            "bright midday Yucatan sun, corn silk catching the light, "
            "ancient milpa cornfield behind, "
            "hyper-detailed Maya iconography, harvest cinematic"
        ),
        "negative": "blurry, low quality, modern, letters, text, western",
        "seed": 13,
    },
    {
        "slug": "ix",
        "name": "Ix",
        "number": 14,
        "meaning": "Jaguar / Earth Magic",
        "prompt": (
            "Ancient Maya stone glyph of Ix, the jaguar shaman and earth magic, "
            "carved in spotted black-and-gold stone, a jaguar's eyes gleaming from the jungle darkness behind, "
            "moonlight filtered through dense canopy, "
            "shaman's mirror of obsidian beside the glyph, "
            "hyper-detailed Maya iconography, night jungle cinematic"
        ),
        "negative": "blurry, low quality, modern, letters, text, western",
        "seed": 14,
    },
    {
        "slug": "men",
        "name": "Men",
        "number": 15,
        "meaning": "Eagle / Moon / Wisdom",
        "prompt": (
            "Ancient Maya stone glyph of Men, the great eagle and lunar wisdom, "
            "carved in cool grey basalt on a high temple platform, "
            "a harpy eagle soaring past against a full moon, "
            "clouds drifting through midnight blue sky, "
            "temple pinnacles below stretching into the jungle canopy, "
            "hyper-detailed Maya iconography, celestial cinematic"
        ),
        "negative": "blurry, low quality, modern, letters, text, western",
        "seed": 15,
    },
    {
        "slug": "kib",
        "name": "K'ib'",
        "number": 16,
        "meaning": "Wax / Owl / Vulture / Soul",
        "prompt": (
            "Ancient Maya stone glyph of K'ib', the wax candle and soul-keeper owl, "
            "carved in dark wax-smooth obsidian, a great horned owl perched on the stone in candlelight, "
            "beeswax candles burning in clay pots below, "
            "flickering golden light on carved relief, incense smoke, "
            "hyper-detailed Maya iconography, candlelit cinematic"
        ),
        "negative": "blurry, low quality, modern, letters, text, western",
        "seed": 16,
    },
    {
        "slug": "kaban",
        "name": "Kaban",
        "number": 17,
        "meaning": "Earth / Earthquake / Thought",
        "prompt": (
            "Ancient Maya stone glyph of Kaban, the earth and the quaking ground, "
            "carved in rough raw volcanic rock, cracks splitting the earth around the base, "
            "molten light rising from below, dust and ash drifting, "
            "ancient jungle roots breaking through stone, "
            "hyper-detailed Maya iconography, seismic cinematic"
        ),
        "negative": "blurry, low quality, modern, letters, text, western",
        "seed": 17,
    },
    {
        "slug": "etznab",
        "name": "Etz'nab'",
        "number": 18,
        "meaning": "Flint / Mirror / Knife",
        "prompt": (
            "Ancient Maya stone glyph of Etz'nab', the obsidian sacrificial knife and mirror, "
            "carved in razor-sharp black obsidian, the glyph surface reflecting the sky like a dark mirror, "
            "lightning flashing in storm clouds above, "
            "ritual obsidian blades arranged at the base like an offering, "
            "hyper-detailed Maya iconography, electric storm cinematic"
        ),
        "negative": "blurry, low quality, modern, letters, text, western",
        "seed": 18,
    },
    {
        "slug": "kawak",
        "name": "Kawak",
        "number": 19,
        "meaning": "Storm / Rain Cloud / Thunder",
        "prompt": (
            "Ancient Maya stone glyph of Kawak, the storm and the thunder cloud, "
            "carved in dark storm-grey andesite, tropical lightning striking behind the temple, "
            "rain cascading over the carved relief, Chac rain serpent coiling in the clouds, "
            "dramatic purple-black storm sky, "
            "hyper-detailed Maya iconography, epic storm cinematic"
        ),
        "negative": "blurry, low quality, modern, letters, text, western",
        "seed": 19,
    },
    {
        "slug": "ajaw",
        "name": "Ajaw",
        "number": 20,
        "meaning": "Lord / Sun / Completion",
        "prompt": (
            "Ancient Maya stone glyph of Ajaw, the sun lord and completion of the sacred cycle, "
            "carved in warm golden limestone at the summit of a great pyramid, "
            "the rising sun directly behind the glyph, "
            "golden rays radiating through carved relief channels, "
            "full solar corona visible, smoke from copal offering rising into light, "
            "hyper-detailed Maya iconography, transcendent sunrise cinematic"
        ),
        "negative": "blurry, low quality, modern, letters, text, western",
        "seed": 20,
    },
]

# ── Tier config ────────────────────────────────────────────────────────────

TIERS = {
    "Q1": {
        "label": "Best Quality",
        "badge": "Q1 · Best",
        "desc": "4-chip Blackhole · PNDM 25-step · 8 frames · 4 glyphs in parallel",
        "frames": 8,
        "steps": 25,
        "lightning": False,
        "temporal_alpha": 0.35,
        "motion_adapter": None,
        "motion_adapter_skip": [],
        "chips": [0, 1, 2, 3],
        "color": "#4fd1c5",
    },
    "Q2": {
        "label": "Fast",
        "badge": "Q2 · Fast",
        "desc": "4-chip Blackhole · Lightning 8-step · 8 frames · 4 glyphs in parallel",
        "frames": 8,
        "steps": 8,
        "lightning": True,
        "temporal_alpha": 0.35,
        "motion_adapter": None,
        "motion_adapter_skip": [],
        "chips": [0, 1, 2, 3],
        "color": "#f4c471",
    },
    # Phase 3 tiers — full MotionAdapter and skip variant for side-by-side comparison
    "Q3": {
        "label": "Phase 3 · Full MotionAdapter",
        "badge": "Q3 · Phase 3",
        "desc": "4-chip Blackhole · PNDM 25-step · 8 frames · full AnimateDiff · ~52 s/frame",
        "frames": 8,
        "steps": 25,
        "lightning": False,
        "temporal_alpha": 1.0,
        "motion_adapter": "guoyww/animatediff-motion-adapter-v1-5-2",
        "motion_adapter_skip": [],
        "chips": [0, 1, 2, 3],
        "color": "#ec96b8",
    },
    "Q4": {
        "label": "Phase 3 · Skip up1+up2",
        "badge": "Q4 · Skip up1+up2",
        "desc": "4-chip Blackhole · PNDM 25-step · 8 frames · 5 injection points · ~7.7 s/frame",
        "frames": 8,
        "steps": 25,
        "lightning": False,
        "temporal_alpha": 1.0,
        "motion_adapter": "guoyww/animatediff-motion-adapter-v1-5-2",
        "motion_adapter_skip": ["up1", "up2"],
        "chips": [0, 1, 2, 3],
        "color": "#27ae60",
    },
}


# ── Generation helpers ─────────────────────────────────────────────────────

def _build_cmd(glyph: dict, tier_key: str, chip: int) -> tuple[list[str], Path]:
    """Build generate.py command for one glyph pinned to a specific chip."""
    tier = TIERS[tier_key]
    slug = glyph["slug"]
    out_path = OUT / tier_key / f"{slug}.gif"
    cmd = [
        sys.executable, str(GENERATE),
        "--mode", "blackhole",
        "--prompt", glyph["prompt"],
        "--negative-prompt", glyph["negative"],
        "--seed", str(glyph["seed"]),
        "--frames", str(tier["frames"]),
        "--steps", str(tier["steps"]),
        "--temporal-alpha", str(tier["temporal_alpha"]),
        "--device-id", str(chip),
        "--output", str(out_path),
    ]
    if tier["lightning"]:
        cmd += ["--lightning", "--lightning-steps", str(tier["steps"])]
    if tier.get("motion_adapter"):
        cmd += ["--motion-adapter", tier["motion_adapter"]]
    if tier.get("motion_adapter_skip"):
        cmd += ["--motion-adapter-skip"] + tier["motion_adapter_skip"]
    return cmd, out_path


def run_tier(tier_key: str, glyphs: list, dry_run: bool = False, stagger: int = 60,
             live_results: dict | None = None):
    """Run all glyphs in a tier, 4 at a time (one per chip, parallel).

    The TTNN SD 1.4 UNet (wormhole-targeted) calls to_torch() internally without
    a mesh_composer, so it cannot run sharded across multiple chips in a single
    process. Instead, each chip runs a separate process, using all 4 chips in
    parallel — 4 glyphs simultaneously, each on its own dedicated Blackhole chip.

    stagger: seconds between process launches to avoid TTNN JIT cache file-lock
    races when multiple processes hit the compiled kernel cache simultaneously.

    live_results: shared dict passed from main() so the manifest is written after
    every batch, not just at the end of the full run.
    """
    tier = TIERS[tier_key]
    chips = tier["chips"]
    (OUT / tier_key).mkdir(parents=True, exist_ok=True)

    print(f"\n{'═' * 60}")
    print(f"║  Tier {tier_key}: {tier['label']}")
    print(f"║  {tier['desc']}")
    if stagger:
        print(f"║  Stagger: {stagger}s between launches (JIT cache lock avoidance)")
    print(f"{'═' * 60}")

    pending = list(glyphs)
    results = {}

    while pending:
        batch = pending[:len(chips)]
        pending = pending[len(chips):]

        procs = []
        for i, g in enumerate(batch):
            chip = chips[i % len(chips)]
            cmd, out_path = _build_cmd(g, tier_key, chip)

            if out_path.exists():
                print(f"  [skip] {out_path.name} already exists")
                results[f"{g['slug']}-{tier_key}"] = {"path": str(out_path), "elapsed": None, "skipped": True}
                continue

            print(f"  [chip {chip}] Starting: {g['slug']} ({g['name']})")
            if dry_run:
                print(f"    cmd tail: {' '.join(cmd[-8:])}")
                results[f"{g['slug']}-{tier_key}"] = {"path": str(out_path), "elapsed": 0}
                continue

            procs.append((g, chip, out_path, subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            ), time.time()))

            if stagger and i < len(batch) - 1:
                print(f"  [stagger] waiting {stagger}s before next launch…")
                time.sleep(stagger)

        for g, chip, out_path, proc, t0 in procs:
            stdout, stderr = proc.communicate()
            elapsed = time.time() - t0
            if proc.returncode == 0:
                size = out_path.stat().st_size // 1024 if out_path.exists() else 0
                print(f"  [chip {chip}] Done: {g['slug']} — {elapsed:.0f}s → {size} KB")
                results[f"{g['slug']}-{tier_key}"] = {"path": str(out_path), "elapsed": elapsed}
            else:
                print(f"  [chip {chip}] ERROR: {g['slug']} (exit {proc.returncode})")
                print(stderr[-300:] if stderr else "  (no stderr)")
                results[f"{g['slug']}-{tier_key}"] = {"path": None, "elapsed": elapsed, "error": stderr[-120:]}

        # Write manifest after each batch so the page reflects live progress
        if live_results is not None and not dry_run:
            live_results.update(results)
            build_manifest(live_results)

    return results


def build_manifest(results: dict):
    # Merge with existing manifest so partial-tier runs don't clobber other tiers
    manifest_path = OUT / "manifest.json"
    existing_results = {}
    if manifest_path.exists():
        try:
            existing_results = json.loads(manifest_path.read_text()).get("results", {})
        except Exception:
            pass
    merged = {**existing_results, **results}

    manifest = {
        "glyphs": {g["slug"]: {
            "name": g["name"],
            "number": g["number"],
            "meaning": g["meaning"],
            "prompt": g["prompt"],
        } for g in GLYPHS},
        "tiers": TIERS,
        "results": {k: {"path": str(v["path"]) if v.get("path") else None, "elapsed": v.get("elapsed")}
                    for k, v in merged.items()},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest: {manifest_path}")


# Canonical 4-glyph sample set for quick comparison runs (one full parallel batch).
# Covers diverse visuals: water/jungle, serpent/fire, jaguar/night, sun/pyramid.
SAMPLE_SLUGS = ["imix", "chikchan", "ix", "ajaw"]


def main():
    parser = argparse.ArgumentParser(description="Generate Maya day glyph benchmark suite")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--tier", choices=["Q1", "Q2", "Q3", "Q4", "all"], default="Q1",
                        help="Quality tier to run (default: Q1)")
    parser.add_argument("--glyph", help="Run only this glyph slug (e.g. imix, ajaw)")
    parser.add_argument("--sample", action="store_true",
                        help=f"Run only the canonical 4-glyph sample set: {SAMPLE_SLUGS}")
    parser.add_argument("--stagger", type=int, default=60,
                        help="Seconds between chip launches (default: 60, avoids JIT cache races)")
    args = parser.parse_args()

    tiers_to_run = list(TIERS.keys()) if args.tier == "all" else [args.tier]
    if args.sample:
        glyphs_to_run = [g for g in GLYPHS if g["slug"] in SAMPLE_SLUGS]
    elif args.glyph:
        glyphs_to_run = [g for g in GLYPHS if g["slug"] == args.glyph]
    else:
        glyphs_to_run = list(GLYPHS)

    if not glyphs_to_run:
        slug = args.glyph or "(sample)"
        print(f"No glyph found with slug '{slug}'. Valid: {[g['slug'] for g in GLYPHS]}")
        sys.exit(1)

    OUT.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for tier_key in tiers_to_run:
        results = run_tier(tier_key, glyphs_to_run, dry_run=args.dry_run,
                           stagger=args.stagger, live_results=all_results)
        all_results.update(results)

    build_manifest(all_results)

    total = len(all_results)
    ok = sum(1 for v in all_results.values() if v.get("path") and not v.get("error"))
    print(f"\n{'═' * 60}")
    print(f"║  Complete: {ok}/{total} succeeded")
    print(f"║  HTML: docs/mayan-glyphs.html")
    print(f"║  View: python3 -m http.server 8080 --directory docs/")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
