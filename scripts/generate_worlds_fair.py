#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""World's Fair prompt suite for tt-animatediff quality showcase.

Runs 9 World's Fair prompts (6 historical + 3 future 2064) across 3 quality tiers
on the 4-chip Blackhole system, then a bonus Unisphere continuity chain across
six eras using --chain-from/--chain-save.

Quality tiers
─────────────
  Q1  best         4 chips × PNDM 25-step, 16 frames, MotionAdapter Phase 3
  Q2  fast         4 chips × Lightning 8-step, 8 frames, Phase 3
  Q3  single       1 chip  × PNDM 25-step, 8 frames, cross-frame alpha=0.35

Each tier runs 4 prompts in parallel (one per chip) using subprocess, cycling
chips round-robin until all prompts are done.

Unisphere chain (bonus)
───────────────────────
  6 sequential runs on chip 0, each --chain-from the previous, PNDM 25-step.
  Simulates the Unisphere at the 1964 World's Fair through 100 years of change.

Usage
─────
  source ~/tt-metal/python_env/bin/activate
  python scripts/generate_worlds_fair.py [--dry-run] [--skip-chain] [--tier Q1|Q2|Q3|all]
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
OUT = REPO / "docs" / "assets" / "worlds-fair"
GENERATE = REPO / "examples" / "generate.py"
CHAIN_DIR = REPO / "output" / "worlds-fair-chain"

# ── World's Fair prompts ───────────────────────────────────────────────────

HISTORICAL_FAIRS = [
    {
        "slug": "paris-1889",
        "title": "Paris 1889 — Eiffel Tower Opening",
        "year": 1889,
        "prompt": (
            "The Eiffel Tower at the 1889 Paris Exposition Universelle, iron lattice "
            "glowing under gas lamps at dusk, crowds of visitors in Victorian dress, "
            "the Seine reflecting amber light, Belle Époque grandeur, painterly impressionist"
        ),
        "negative": "modern, cars, neon, digital, blurry, low quality",
        "seed": 1889,
    },
    {
        "slug": "chicago-1893",
        "title": "Chicago 1893 — White City",
        "year": 1893,
        "prompt": (
            "The White City at the 1893 Chicago World's Columbian Exposition, "
            "neoclassical white marble palaces reflected in the Grand Basin, "
            "electric lights illuminating the Court of Honor at night for the first time, "
            "crowds in Gilded Age attire, moonlit clouds, majestic and ethereal"
        ),
        "negative": "modern, neon, digital, blurry, low quality",
        "seed": 1893,
    },
    {
        "slug": "nyc-1939",
        "title": "New York 1939 — World of Tomorrow",
        "year": 1939,
        "prompt": (
            "The 1939 New York World's Fair at night, the Trylon and Perisphere glowing white "
            "against a violet sky, art deco streamlined pavilions, fountains lit in vivid color, "
            "visitors in 1930s dress gazing at the future, retro-futurist wonder, cinematic"
        ),
        "negative": "modern, digital, blurry, low quality",
        "seed": 1939,
    },
    {
        "slug": "brussels-1958",
        "title": "Brussels 1958 — Atomium",
        "year": 1958,
        "prompt": (
            "The Atomium at the 1958 Brussels World Expo, nine steel spheres magnifying an iron "
            "crystal atom 165 billion times, golden evening light, Cold War era optimism, "
            "swooping modernist pavilions in the background, atomic age, silver and chrome"
        ),
        "negative": "blurry, low quality, digital artifacts",
        "seed": 1958,
    },
    {
        "slug": "nyc-1964",
        "title": "New York 1964 — Unisphere",
        "year": 1964,
        "prompt": (
            "The Unisphere at the 1964 New York World's Fair, massive stainless steel globe "
            "rising from Flushing Meadows, three orbital rings glinting in summer sunlight, "
            "IBM and Ford pavilions in background, mid-century modern optimism, Space Age, "
            "cinematic wide shot, warm afternoon light"
        ),
        "negative": "blurry, low quality, modern, digital",
        "seed": 1964,
    },
    {
        "slug": "osaka-1970",
        "title": "Osaka 1970 — Tower of the Sun",
        "year": 1970,
        "prompt": (
            "The Tower of the Sun by Taro Okamoto at Expo '70 Osaka, massive expressionist "
            "sculpture with golden face and white face, festival plaza teeming with visitors, "
            "futuristic space-frame roof structure, Japan's economic miracle, vivid colors, "
            "1970s psychedelic energy, cinematic"
        ),
        "negative": "blurry, low quality, modern, digital",
        "seed": 1970,
    },
]

FUTURE_2064_FAIRS = [
    {
        "slug": "2064-lunar-basin",
        "title": "2064 — Lunar World's Fair",
        "year": 2064,
        "prompt": (
            "The 2064 World's Fair on the Moon, glass dome pavilions in the Sea of Tranquility, "
            "Earth rising on the horizon, bioluminescent architecture glowing against the lunar "
            "regolith, delegations from forty nations and three orbital habitats, low gravity "
            "fountains arcing impossibly high, retro-futurist meets deep future, awe-inspiring"
        ),
        "negative": "blurry, low quality, Earth gravity, trees, clouds",
        "seed": 2064,
    },
    {
        "slug": "2064-deepwater",
        "title": "2064 — Pacific Deepwater Expo",
        "year": 2064,
        "prompt": (
            "The 2064 Pacific Deepwater World Expo at 500 meters depth, coral-lattice arcologies "
            "lit by bioluminescent algae, submersibles docking at crystal pavilions, whale song "
            "translated into light, kelp forests framing the grand promenade, ethereal blue-green "
            "dreamscape, wonder and reverence for the ocean"
        ),
        "negative": "blurry, low quality, surface, sky, land",
        "seed": 2065,
    },
    {
        "slug": "2064-orbital",
        "title": "2064 — Orbital Ring World's Fair",
        "year": 2064,
        "prompt": (
            "The 2064 World's Fair aboard the Orbital Ring station, a gleaming torus circling "
            "Earth at 36,000 km, nations of the world building spinning pavilions along the "
            "inner hull, Earth a blue marble through vast observation windows, zero-gravity "
            "dancers in the atrium, humanity's greatest achievement on display, cinematic 4K"
        ),
        "negative": "blurry, low quality, gravity, surface, ugly",
        "seed": 2066,
    },
]

ALL_PROMPTS = HISTORICAL_FAIRS + FUTURE_2064_FAIRS

# ── Unisphere through 100 years ────────────────────────────────────────────

UNISPHERE_CHAIN = [
    {
        "slug": "unisphere-1964",
        "era": "1964 — Opening Day",
        "prompt": (
            "The Unisphere at the 1964 New York World's Fair, gleaming stainless steel "
            "globe in Flushing Meadows, opening day crowds, summer sunshine, "
            "Space Age optimism, wide shot, cinematic"
        ),
        "seed": 1964,
    },
    {
        "slug": "unisphere-1980",
        "era": "1980 — Urban Decay",
        "prompt": (
            "The Unisphere in Flushing Meadows 1980, a little weathered but still magnificent, "
            "autumn leaves swirling at its base, young skateboarders circling the reflecting pool, "
            "New York City skyline behind, overcast sky, cinematic melancholy"
        ),
        "seed": 1980,
    },
    {
        "slug": "unisphere-2000",
        "era": "2000 — Millennium",
        "prompt": (
            "The Unisphere at midnight December 31 1999, fireworks exploding around the globe, "
            "millennium crowds filling Flushing Meadows, reflection rippling in the pool, "
            "city lights, Y2K celebration, triumphant and electric"
        ),
        "seed": 2000,
    },
    {
        "slug": "unisphere-2026",
        "era": "2026 — America 250",
        "prompt": (
            "The Unisphere in 2026 during America's 250th anniversary, freshly restored and gleaming, "
            "drone light show forming the shape of the Declaration of Independence above it, "
            "diverse crowds celebrating, summer evening, golden hour light"
        ),
        "seed": 2026,
    },
    {
        "slug": "unisphere-2050",
        "era": "2050 — Climate Restored",
        "prompt": (
            "The Unisphere in 2050, surrounded by restored wetland parkland, "
            "solar ribbon trees arching overhead, autonomous quiet transit gliding past, "
            "children from all nations playing at its base, the air crystalline clear, "
            "hopeful solarpunk future, warm golden light"
        ),
        "seed": 2050,
    },
    {
        "slug": "unisphere-2064",
        "era": "2064 — World's Fair Centennial",
        "prompt": (
            "The Unisphere in 2064, the centennial of the New York World's Fair, "
            "holographic rings added to the original three orbital bands, visitors from "
            "Moon colonies attending the centennial ceremony, Earth and Moon both visible "
            "in the sky behind it, one hundred years of human progress, cinematic grandeur"
        ),
        "seed": 2064,
    },
]

# ── Quality tier configs ───────────────────────────────────────────────────

TIERS = {
    "Q1": {
        "label": "Best Quality",
        "badge": "Q1 · Best",
        "desc": "4-chip Blackhole · PNDM 25-step · 16 frames · MotionAdapter Phase 3",
        "frames": 16,
        "steps": 25,
        "lightning": False,
        "motion_adapter": True,
        "temporal_alpha": 0.35,
        "chips": [0, 1, 2, 3],
        "color": "#4fd1c5",
    },
    "Q2": {
        "label": "Fast + Quality",
        "badge": "Q2 · Fast",
        "desc": "4-chip Blackhole · Lightning 8-step · 8 frames · MotionAdapter Phase 3",
        "frames": 8,
        "steps": 8,
        "lightning": True,
        "motion_adapter": True,
        "temporal_alpha": 0.35,
        "chips": [0, 1, 2, 3],
        "color": "#f4c471",
    },
    "Q3": {
        "label": "Single Chip",
        "badge": "Q3 · 1 Chip",
        "desc": "1-chip Blackhole · PNDM 25-step · 8 frames · cross-frame α=0.35",
        "frames": 8,
        "steps": 25,
        "lightning": False,
        "motion_adapter": False,
        "temporal_alpha": 0.35,
        "chips": [0],
        "color": "#ec96b8",
    },
}


# ── Generation helpers ─────────────────────────────────────────────────────

def _build_cmd(prompt_cfg: dict, tier_key: str, chip: int) -> tuple[list[str], Path]:
    tier = TIERS[tier_key]
    slug = prompt_cfg["slug"]
    out_path = OUT / tier_key / f"{slug}.gif"
    cmd = [
        sys.executable, str(GENERATE),
        "--mode", "blackhole",
        "--prompt", prompt_cfg["prompt"],
        "--negative-prompt", prompt_cfg["negative"],
        "--seed", str(prompt_cfg["seed"]),
        "--frames", str(tier["frames"]),
        "--steps", str(tier["steps"]),
        "--temporal-alpha", str(tier["temporal_alpha"]),
        "--device-id", str(chip),
        "--output", str(out_path),
    ]
    if tier["lightning"]:
        cmd += ["--lightning", "--lightning-steps", str(tier["steps"])]
    if tier["motion_adapter"]:
        cmd += ["--motion-adapter"]
        # Skip large-spatial up-block injections: 6× faster with negligible quality loss.
        # up1 (32×32) and up2 (64×64) account for 85% of CPU round-trip time.
        if tier.get("skip_up_blocks", True):
            cmd += ["--motion-adapter-skip", "up1", "up2"]
    return cmd, out_path


def run_tier(tier_key: str, prompts: list, dry_run: bool = False, stagger: int = 360):
    """Run all prompts in a tier, up to 4 at a time (one per chip).

    stagger: seconds to wait between launching each subprocess.  The TTNN JIT
    cache and model-weight loader use file locks that deadlock when 4 processes
    hit them simultaneously.  A stagger of ~360s (6 min) lets each process
    complete model loading before the next one starts.  Pass 0 to disable.
    """
    tier = TIERS[tier_key]
    chips = tier["chips"]
    (OUT / tier_key).mkdir(parents=True, exist_ok=True)

    print(f"\n{'═' * 60}")
    print(f"║  Tier {tier_key}: {tier['label']}")
    print(f"║  {tier['desc']}")
    if stagger:
        print(f"║  Stagger: {stagger}s between launches to avoid JIT lock contention")
    print(f"{'═' * 60}")

    pending = list(prompts)
    results = {}

    while pending:
        # Launch a batch — one per available chip, up to len(pending)
        batch = pending[:len(chips)]
        pending = pending[len(chips):]

        procs = []
        for i, p in enumerate(batch):
            chip = chips[i % len(chips)]
            cmd, out_path = _build_cmd(p, tier_key, chip)

            if out_path.exists():
                print(f"  [skip] {out_path.name} already exists")
                results[f"{p['slug']}-{tier_key}"] = {"path": out_path, "elapsed": None, "skipped": True}
                continue

            print(f"  [chip {chip}] Starting: {p['slug']}")
            if dry_run:
                print(f"    {' '.join(cmd[-10:])}")
                results[f"{p['slug']}-{tier_key}"] = {"path": out_path, "elapsed": 0}
                continue

            procs.append((p, chip, out_path, subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            ), time.time()))

            # Stagger: wait before launching the next process so each one
            # finishes TTNN JIT compilation and model loading before the next
            # one starts — simultaneous loads deadlock on the JIT cache lock.
            if stagger and i < len(batch) - 1:
                print(f"  [stagger] waiting {stagger}s before next launch…")
                time.sleep(stagger)

        for p, chip, out_path, proc, t0 in procs:
            stdout, stderr = proc.communicate()
            elapsed = time.time() - t0
            if proc.returncode == 0:
                size = out_path.stat().st_size // 1024 if out_path.exists() else 0
                print(f"  [chip {chip}] Done: {p['slug']} — {elapsed:.0f}s → {size} KB")
                results[f"{p['slug']}-{tier_key}"] = {"path": out_path, "elapsed": elapsed}
            else:
                print(f"  [chip {chip}] ERROR: {p['slug']} (exit {proc.returncode})")
                print(stderr[-300:] if stderr else "  (no stderr)")
                results[f"{p['slug']}-{tier_key}"] = {"path": None, "elapsed": elapsed, "error": stderr[-120:]}

    return results


def run_unisphere_chain(dry_run: bool = False):
    """Run the 6-era Unisphere continuity chain using ChainSession.

    Keeps the compiled TTNN UNet resident on chip 0 across all hops —
    each hop pays only CLIP encode + denoising time (~2 min), not 30s recompile.
    chain_alpha=0.35: ~15% layout correlation (enough for subject continuity,
    won't overpower text guidance the way 0.55 did).
    """
    CHAIN_DIR.mkdir(parents=True, exist_ok=True)
    (OUT / "chain").mkdir(parents=True, exist_ok=True)

    print(f"\n{'═' * 60}")
    print("║  Unisphere — 100-year continuity chain")
    print("║  6 eras · PNDM 25-step · 16 frames · MotionAdapter · chip 0")
    print("║  chain_alpha=0.35 (fixed — was 0.55 which killed subject identity)")
    print(f"{'═' * 60}")

    CHAIN_ALPHA = 0.35
    NEG = "blurry, low quality, ugly"

    if dry_run:
        for era in UNISPHERE_CHAIN:
            slug = era["slug"]
            out_path = OUT / "chain" / f"{slug}.gif"
            chain_save = CHAIN_DIR / f"{slug}.pt"
            print(f"  [dry-run] {era['era']} → {out_path.name}  chain_alpha={CHAIN_ALPHA}")
        return {e["slug"]: {"path": None, "elapsed": 0} for e in UNISPHERE_CHAIN}

    sys.path.insert(0, str(REPO))
    from animatediff_ttnn.generation_helpers import ChainSession
    from animatediff_ttnn.pipeline import export_gif
    from animatediff_ttnn.motion_weights import load_motion_modules
    from animatediff_ttnn.temporal_attention import generate_frames_motion

    results = {}

    with ChainSession(device_ids=[0]) as sess:
        # Load MotionAdapter once — stays in memory for all hops
        print("  Loading MotionAdapter weights...")
        temporal_kernels = load_motion_modules()
        print(f"  Loaded {sum(len(v) for v in temporal_kernels.values())} modules\n")

        prev_chain = None

        for era in UNISPHERE_CHAIN:
            slug = era["slug"]
            out_path = OUT / "chain" / f"{slug}.gif"
            chain_save = CHAIN_DIR / f"{slug}.pt"

            print(f"  [{era['era']}] {'(chain α='+str(CHAIN_ALPHA)+')' if prev_chain else '(seed)'}")

            t0 = time.time()
            try:
                from animatediff_ttnn.generation_helpers import encode_prompt
                text_embeddings = encode_prompt(era["prompt"], NEG)

                frames = generate_frames_motion(
                    device=sess.device,
                    ttnn_model=sess._ttnn_model,
                    ttnn_vae=sess._ttnn_vae,
                    config=sess._config,
                    torch_time_proj=sess._time_proj,
                    text_embeddings=text_embeddings,
                    temporal_kernels=temporal_kernels,
                    num_frames=16,
                    num_steps=25,
                    seed=era["seed"],
                    temporal_alpha=0.35,
                    chain_from=prev_chain,
                    chain_save=str(chain_save),
                    chain_alpha=CHAIN_ALPHA,
                )
                export_gif(frames, str(out_path))
                elapsed = time.time() - t0
                size = out_path.stat().st_size // 1024 if out_path.exists() else 0
                print(f"    Done in {elapsed:.0f}s → {size} KB")
                results[slug] = {"path": str(out_path), "elapsed": elapsed}
                prev_chain = str(chain_save)
            except Exception as exc:
                elapsed = time.time() - t0
                print(f"    ERROR after {elapsed:.0f}s: {exc}")
                results[slug] = {"path": None, "elapsed": elapsed, "error": str(exc)}
                if chain_save.exists():
                    prev_chain = str(chain_save)

    return results


def build_manifest(all_results: dict):
    """Write a JSON manifest of all generated GIFs for the HTML page to consume."""
    manifest = {
        "prompts": {p["slug"]: {
            "title": p["title"],
            "year": p["year"],
            "prompt": p["prompt"],
        } for p in ALL_PROMPTS},
        "tiers": TIERS,
        "chain": {era["slug"]: {"era": era["era"], "prompt": era["prompt"]} for era in UNISPHERE_CHAIN},
        "results": {k: {"path": str(v["path"]) if v.get("path") else None, "elapsed": v.get("elapsed")}
                    for k, v in all_results.items()},
    }
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="World's Fair generation suite")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--skip-chain", action="store_true", help="Skip Unisphere chain")
    parser.add_argument("--chain-only", action="store_true", help="Run only Unisphere chain (skip all tiers)")
    parser.add_argument("--tier", choices=["Q1", "Q2", "Q3", "all"], default="all",
                        help="Which tier to run (default: all)")
    parser.add_argument("--only-prompt", type=int, default=None,
                        help="Run only prompt index 0-8 (for testing)")
    parser.add_argument("--stagger", type=int, default=360,
                        help="Seconds between parallel chip launches (default 360). "
                             "Prevents JIT cache deadlock when 4 processes load models simultaneously. "
                             "Use 0 to disable.")
    args = parser.parse_args()

    prompts = ALL_PROMPTS if args.only_prompt is None else [ALL_PROMPTS[args.only_prompt]]
    tiers = [] if args.chain_only else (["Q1", "Q2", "Q3"] if args.tier == "all" else [args.tier])

    all_results = {}

    for tier_key in tiers:
        results = run_tier(tier_key, prompts, dry_run=args.dry_run, stagger=args.stagger)
        all_results.update(results)

    if not args.skip_chain:
        chain_results = run_unisphere_chain(dry_run=args.dry_run)
        all_results.update(chain_results)

    build_manifest(all_results)

    print(f"\nAll done. GIFs in: {OUT}")
    print(f"To view: python3 -m http.server 8080 --directory {REPO}/docs/")


if __name__ == "__main__":
    main()
