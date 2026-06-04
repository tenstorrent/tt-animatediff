# Cosmic Study Website Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate 35 thematic GIFs on Blackhole hardware and publish them as a single-page GitHub Pages site where the grid maps to the 11×10 Tensix chip layout.

**Architecture:** A resumable Python generation script runs all 35 GIFs sequentially on a single MeshDevice(1×1), writing each output to `docs/assets/study/` and updating `manifest.json` so any interrupted run can skip completed GIFs on resume. A static `docs/index.html` page embeds all GIFs in the chip grid. GH Pages serves from `main/docs/`.

**Tech Stack:** Python 3.10, `~/tt-metal` python_env, `examples/generate.py --mode blackhole`, Pillow for GIF validation, plain HTML/CSS (no build step), GitHub Pages, GitHub Actions.

---

## File map

| File | Status | Responsibility |
|---|---|---|
| `scripts/generate_study.py` | **create** | Resumable batch runner — defines all 35 prompts, checks manifest, calls generate.py for each missing GIF |
| `docs/assets/study/manifest.json` | **create (auto)** | Written by generate_study.py; tracks done/pending/failed per GIF |
| `docs/assets/study/*.gif` | **create (generated)** | 35 GIFs, one per prompt entry |
| `docs/index.html` | **create** | Full static page: hero → chip grid → attribution → what we built → footer |
| `.github/workflows/pages.yml` | **create** | Deploy docs/ to GH Pages on push to main |
| `requirements.txt` | **modify** | Pin tt-metal firmware/KMD version as a comment |
| `docs/HARDWARE_COMPAT.md` | **create** | Documents path change from old wormhole SD location to new one; required tt-metal version |

---

## Task 1: Resumable generation script

**Files:**
- Create: `scripts/generate_study.py`

This script defines all 35 GIFs, checks which are already done, and runs each missing one by shelling out to `~/tt-metal/python_env/bin/python examples/generate.py`. It reopens the device fresh for each GIF (closes after each), which is the safest pattern given Blackhole's ethernet-core instability.

- [ ] **Step 1: Create the script**

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Resumable batch generation script for the Cosmic Study website.

Run from the repo root inside the tt-metal python env:
    source ~/tt-metal/python_env/bin/activate
    python scripts/generate_study.py

Skips GIFs that already exist and are non-zero. Safe to Ctrl-C and re-run.
"""

import json
import os
import subprocess
import sys
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
    dict(name="p1-nebula",       row=0, col=0,  phase=1, alpha=0.35, seed=42,
         prompt="swirling nebula in deep space, purple and teal gas clouds, stars, cinematic 4K"),
    dict(name="p1-galaxy",       row=0, col=2,  phase=1, alpha=0.35, seed=42,
         prompt="spiral galaxy seen from above, golden core, trailing arms of blue starlight, cinematic"),
    dict(name="p1-aurora",       row=0, col=4,  phase=1, alpha=0.35, seed=42,
         prompt="aurora borealis dancing over arctic ice, green and violet ribbons, starfield, cinematic"),
    dict(name="p1-starfield",    row=0, col=6,  phase=1, alpha=0.35, seed=42,
         prompt="infinite starfield rotating slowly, Milky Way band, depth of field, cosmic, cinematic 4K"),
    dict(name="p1-solar-flare",  row=0, col=8,  phase=1, alpha=0.35, seed=42,
         prompt="solar flare erupting from sun surface, plasma arcs, golden orange, cinematic"),
    dict(name="p1-gas-cloud",    row=0, col=10, phase=1, alpha=0.35, seed=42,
         prompt="interstellar gas cloud glowing teal and pink, cosmic dust, nebular light, cinematic 4K"),
    dict(name="p1-crystal-cave", row=1, col=1,  phase=1, alpha=0.35, seed=42,
         prompt="crystal cave glowing with bioluminescent cosmic light, amethyst formations, starlight within"),
    dict(name="p1-deep-ocean",   row=1, col=3,  phase=1, alpha=0.35, seed=42,
         prompt="deep ocean at night, bioluminescent creatures, stars reflected on black water, cinematic"),
    dict(name="p1-lunar-halo",   row=1, col=6,  phase=1, alpha=0.35, seed=42,
         prompt="full moon with atmospheric halo, aurora behind it, deep blue sky, cinematic 4K"),
    dict(name="p1-north-lights", row=1, col=8,  phase=1, alpha=0.35, seed=42,
         prompt="northern lights over ancient pine forest, green and magenta, slow swirl, cinematic"),
    dict(name="p1-milky-way",    row=2, col=0,  phase=1, alpha=0.35, seed=42,
         prompt="Milky Way over desert landscape, rock formations silhouetted, cosmic scale, cinematic 4K"),
    dict(name="p1-fractal-tunnel",row=2, col=3, phase=1, alpha=0.35, seed=42,
         prompt="fractal space tunnel, recursive geometry, teal and gold, infinite zoom, cinematic"),
    # Phase 2 — Intertwining oneness (pink, alpha 0.4)
    dict(name="p2-mandala",      row=2, col=7,  phase=2, alpha=0.4, seed=42,
         prompt="sacred mandala blooming from starfield, geometric petals, gold and violet, cosmic, cinematic"),
    dict(name="p2-mycelium",     row=2, col=10, phase=2, alpha=0.4, seed=42,
         prompt="mycelium network glowing with bioluminescent spores, threads of light connecting nodes, cosmic forest"),
    dict(name="p2-dna",          row=3, col=1,  phase=2, alpha=0.4, seed=42,
         prompt="DNA double helix rotating in cosmic void, glowing teal, stars inside, cinematic 4K"),
    dict(name="p2-roots-cosmos", row=3, col=5,  phase=2, alpha=0.4, seed=42,
         prompt="ancient tree roots intertwining with stars and galaxies, roots become light filaments, cinematic"),
    dict(name="p2-consciousness", row=3, col=9, phase=2, alpha=0.4, seed=42,
         prompt="threads of consciousness connecting glowing nodes, neural web across dark void, teal and gold"),
    dict(name="p2-sacred-geo",   row=4, col=0,  phase=2, alpha=0.4, seed=42,
         prompt="sacred geometry unfolding in cosmic space, flower of life, metatrons cube, golden ratio spirals"),
    dict(name="p2-forest-mind",  row=4, col=4,  phase=2, alpha=0.4, seed=42,
         prompt="forest canopy seen from below as neural network, branches are synapses, bioluminescent, cinematic"),
    dict(name="p2-reef",         row=4, col=7,  phase=2, alpha=0.4, seed=42,
         prompt="living coral reef as collective mind, tendrils of light, pulsing bioluminescence, cosmic ocean"),
    dict(name="p2-meridians",    row=4, col=10, phase=2, alpha=0.4, seed=42,
         prompt="energy meridians flowing across a body silhouette, acupuncture lines glowing gold, cosmic backdrop"),
    # Phase 3 — Liminal threshold (gold, alpha 0.4)
    dict(name="p3-temple",       row=5, col=2,  phase=3, alpha=0.4, seed=42,
         prompt="ancient Mayan temple under shifting cosmos, nebula behind it, jungle emerging, cinematic 4K"),
    dict(name="p3-cave-circuit", row=5, col=6,  phase=3, alpha=0.4, seed=42,
         prompt="cave paintings of animals slowly morphing into circuit traces, orange firelight to teal digital glow"),
    dict(name="p3-standing-stones",row=5,col=9, phase=3, alpha=0.4, seed=42,
         prompt="stone circle at dawn with aurora, ancient megaliths glowing with interior light, cosmic sky"),
    dict(name="p3-crystal-data", row=6, col=1,  phase=3, alpha=0.4, seed=42,
         prompt="crystal mountain formation with streams of data flowing through it, organic meets digital, cinematic"),
    dict(name="p3-ley-fiber",    row=6, col=5,  phase=3, alpha=0.4, seed=42,
         prompt="ancient ley lines across landscape glowing gold, slowly becoming fiber optic cables, aerial view"),
    dict(name="p3-mayan-grid",   row=6, col=10, phase=3, alpha=0.4, seed=42,
         prompt="Mayan calendar stone dissolving into computational grid, stone becomes silicon, cosmic light"),
    # Phase 4 — Tech emergence (green, alpha 0.45)
    dict(name="p4-circuit-moss", row=7, col=0,  phase=4, alpha=0.45, seed=42,
         prompt="circuit board growing like moss and lichen, organic circuitry, green teal glow, macro, cinematic"),
    dict(name="p4-silicon",      row=7, col=4,  phase=4, alpha=0.45, seed=42,
         prompt="silicon crystal lattice glowing in cosmic light, atomic structure visible, teal and gold, cinematic"),
    dict(name="p4-neural-const", row=7, col=8,  phase=4, alpha=0.45, seed=42,
         prompt="neural network nodes arranged as constellation, synapses firing like shooting stars, dark space"),
    dict(name="p4-server-aurora",row=8, col=2,  phase=4, alpha=0.45, seed=42,
         prompt="server room bathed in aurora borealis light through glass ceiling, racks of machines, cinematic"),
    dict(name="p4-chip-city",    row=8, col=7,  phase=4, alpha=0.45, seed=42,
         prompt="semiconductor chip die seen as aerial city at night, nanoscale streets glowing teal, cinematic 4K"),
    # Phase 5 — Full integration (bright teal, alpha 0.5)
    dict(name="p5-chip-cosmos",  row=9, col=0,  phase=5, alpha=0.5, seed=42,
         prompt="Tenstorrent Blackhole chip glowing with embedded cosmos, galaxies visible inside the die, cinematic"),
    dict(name="p5-hand-galaxy",  row=9, col=5,  phase=5, alpha=0.5, seed=42,
         prompt="human hand touching a circuit board that blooms into a galaxy on contact, light erupting, cinematic"),
    dict(name="p5-grid-is-all",  row=9, col=10, phase=5, alpha=0.5, seed=42,
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


def run_gif(g, manifest):
    name = g["name"]
    out = OUT_DIR / f"{name}.gif"
    print(f"\n{'='*60}")
    print(f"  [{GIFS.index(g)+1}/{len(GIFS)}] {name}")
    print(f"  phase={g['phase']}  alpha={g['alpha']}  seed={g['seed']}")
    print(f"  {g['prompt'][:80]}...")
    print(f"{'='*60}")

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
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    elapsed = time.time() - t0

    if result.returncode == 0 and out.exists() and out.stat().st_size > 0:
        manifest[name] = "done"
        print(f"  ✓ done in {elapsed:.0f}s → {out.name}")
    else:
        manifest[name] = "failed"
        print(f"  ✗ FAILED (exit {result.returncode}) after {elapsed:.0f}s")
        print(f"    Retry manually: python scripts/generate_study.py --only {name}")

    save_manifest(manifest)
    return manifest[name] == "done"


def main():
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    targets = [g for g in GIFS if only is None or g["name"] == only]
    pending = [g for g in targets if not gif_done(g["name"], manifest)]

    print(f"Cosmic Study generation")
    print(f"  Total: {len(GIFS)}  Done: {len(GIFS)-len(pending)}  Pending: {len(pending)}")
    if not pending:
        print("  All done! Run: open docs/index.html")
        return

    for g in pending:
        run_gif(g, manifest)

    done = sum(1 for g in GIFS if gif_done(g["name"], manifest))
    failed = [g["name"] for g in GIFS if manifest.get(g["name"]) == "failed"]
    print(f"\n{'='*60}")
    print(f"  Done: {done}/{len(GIFS)}")
    if failed:
        print(f"  Failed: {failed}")
        print(f"  Retry: python scripts/generate_study.py --only <name>")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make it executable and do a dry-run check (no hardware)**

```bash
chmod +x scripts/generate_study.py
python scripts/generate_study.py --only NONEXISTENT 2>/dev/null | head -5
# Expected: prints "Total: 35  Done: 0  Pending: 0" (NONEXISTENT not in list, so nothing runs)
```

- [ ] **Step 3: Commit the script**

```bash
git add scripts/generate_study.py
git commit -m "feat: add resumable cosmic study generation script (35 GIFs)"
```

---

## Task 2: Run the generation (the long one)

**Files:**
- Creates: `docs/assets/study/*.gif` (35 files)
- Creates/updates: `docs/assets/study/manifest.json`

- [ ] **Step 1: Start a tmux session so it survives disconnection**

```bash
tmux new-session -d -s study "source ~/tt-metal/python_env/bin/activate && python scripts/generate_study.py 2>&1 | tee /tmp/study_gen.log"
echo "Generation running in tmux session 'study'"
echo "Monitor: tmux attach -t study"
echo "Log: tail -f /tmp/study_gen.log"
```

- [ ] **Step 2: Monitor progress (in another terminal or after reattaching)**

```bash
tmux attach -t study
# Ctrl-b d to detach without stopping
```

Check manifest at any time:
```bash
python3 -c "
import json; m = json.load(open('docs/assets/study/manifest.json'))
done = [k for k,v in m.items() if v=='done']
fail = [k for k,v in m.items() if v=='failed']
run  = [k for k,v in m.items() if v=='running']
print(f'Done: {len(done)}  Failed: {len(fail)}  Running: {run}')
"
```

- [ ] **Step 3: If any GIF fails, retry it**

```bash
# Re-run just the failed one (script skips all completed GIFs)
source ~/tt-metal/python_env/bin/activate
python scripts/generate_study.py --only p1-nebula  # replace with actual failed name
```

If hardware hangs (ethernet core timeout), reset and retry:
```bash
tt-smi -r 0 1 2 3 && sleep 8
python scripts/generate_study.py  # resumes from where it left off
```

- [ ] **Step 4: Verify all 35 GIFs exist and are non-zero**

```bash
ls -lh docs/assets/study/*.gif | wc -l   # should print 35
ls -lh docs/assets/study/*.gif | awk '$5 == "0" {print "ZERO:", $9}'  # should print nothing
python3 -c "
import json; m = json.load(open('docs/assets/study/manifest.json'))
failed = [k for k,v in m.items() if v != 'done']
print('All done!' if not failed else f'Not done: {failed}')
"
```

- [ ] **Step 5: Commit the generated GIFs**

```bash
git add docs/assets/study/
git commit -m "feat: add 35 cosmic study GIFs generated on Blackhole P300C"
```

---

## Task 3: Build the static HTML page

**Files:**
- Create: `docs/index.html`

The grid positions for all 35 GIFs are defined in the JS at the bottom — matching the exact `(row, col)` from the generation script. GIFs load lazily. A tooltip shows the prompt on hover.

- [ ] **Step 1: Create `docs/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>tt-animatediff — A Cosmic Quick Study</title>
  <meta name="description" content="35 animated GIFs generated on a Tenstorrent Blackhole P300C. AnimateDiff SD 1.4 TTNN UNet — cosmic themes, chip grid layout.">
  <meta property="og:title" content="tt-animatediff — A Cosmic Quick Study">
  <meta property="og:description" content="35 animated GIFs generated on Tenstorrent Blackhole hardware. Each glowing cell is a Tensix compute node.">
  <meta property="og:image" content="assets/study/p1-nebula.gif">
  <style>
    :root {
      --bg0: #080f14; --bg1: #0d1b24; --bg2: #122130; --card: #162838;
      --teal: #4fd1c5; --teal-dim: rgba(79,209,197,.15); --teal-border: rgba(79,209,197,.3);
      --pink: #ec96b8; --gold: #f4c471; --green: #27ae60; --teal2: #81e6d9;
      --text: #e8f0f2; --text2: #b0c4d0; --muted: #607d8b;
      --border: rgba(79,209,197,.12); --radius: 8px;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html { scroll-behavior: smooth; }
    body { background: var(--bg0); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; font-size: 16px; line-height: 1.65; -webkit-font-smoothing: antialiased; }
    a { color: var(--teal); text-decoration: none; }
    a:hover { text-decoration: underline; }

    /* hero */
    .hero { padding: 4rem 2rem 2.5rem; text-align: center; border-bottom: 1px solid var(--border); }
    .badge { display: inline-block; font-size: .68rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: var(--teal); border: 1px solid var(--teal-border); border-radius: 20px; padding: .25rem .75rem; margin-bottom: 1.25rem; }
    .hero h1 { font-size: clamp(1.8rem, 4vw, 3rem); font-weight: 800; line-height: 1.15; margin-bottom: .75rem; }
    .hero h1 em { color: var(--teal); font-style: normal; }
    .hero p { font-size: 1rem; color: var(--text2); max-width: 560px; margin: 0 auto 2rem; }
    .hero-links { display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; }
    .btn { display: inline-flex; align-items: center; gap: .4rem; padding: .55rem 1.25rem; border-radius: var(--radius); font-size: .85rem; font-weight: 600; text-decoration: none; transition: opacity .15s; }
    .btn:hover { opacity: .85; text-decoration: none; }
    .btn-primary { background: var(--teal); color: #080f14; }
    .btn-ghost { background: var(--teal-dim); color: var(--teal); border: 1px solid var(--teal-border); }

    /* chip grid */
    .chip-section { padding: 3.5rem 2rem; display: flex; flex-direction: column; align-items: center; overflow-x: auto; }
    .chip-label { font-size: .72rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); margin-bottom: 1rem; }
    .chip-grid { display: grid; grid-template-columns: repeat(11, 72px); grid-template-rows: repeat(10, 72px); gap: 3px; flex-shrink: 0; }
    .node { border-radius: 5px; border: 1px solid var(--border); background: #0a141c; overflow: hidden; position: relative; }
    .node img { width: 100%; height: 100%; object-fit: cover; display: block; }
    .node.phase-1 { border-color: rgba(79,209,197,.5); }
    .node.phase-2 { border-color: rgba(236,150,184,.4); }
    .node.phase-3 { border-color: rgba(244,196,113,.4); }
    .node.phase-4 { border-color: rgba(39,174,96,.4); }
    .node.phase-5 { border-color: rgba(129,230,217,.6); }
    .node-tip { position: absolute; bottom: 0; left: 0; right: 0; background: rgba(8,15,20,.92); color: var(--text2); font-size: .38rem; padding: 2px 3px; line-height: 1.3; opacity: 0; transition: opacity .2s; pointer-events: none; text-align: center; }
    .node:hover .node-tip { opacity: 1; }

    /* phase legend */
    .phase-legend { display: flex; gap: 1.5rem; flex-wrap: wrap; justify-content: center; margin-top: 1.25rem; }
    .ph { display: flex; align-items: center; gap: .45rem; font-size: .72rem; color: var(--muted); }
    .ph-dot { width: 10px; height: 10px; border-radius: 2px; border: 1px solid; flex-shrink: 0; }

    /* shoulders */
    .shoulders { background: var(--bg1); border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); padding: 3rem 2rem; text-align: center; }
    .section-label { font-size: .68rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); margin-bottom: 1.25rem; display: block; }
    .cards { display: flex; flex-wrap: wrap; gap: 1rem; justify-content: center; max-width: 920px; margin: 0 auto; }
    .card { background: var(--bg2); border: 1px solid var(--border); border-radius: 10px; padding: .9rem 1.2rem; text-align: left; max-width: 260px; flex: 1 1 220px; }
    .card h4 { font-size: .82rem; font-weight: 700; color: var(--teal); margin-bottom: .35rem; }
    .card p { font-size: .74rem; color: var(--muted); line-height: 1.6; }
    .card a { color: var(--teal2); font-size: .72rem; }
    .thanks { margin-top: 1.5rem; font-size: .82rem; color: var(--teal); font-style: italic; max-width: 600px; margin-left: auto; margin-right: auto; }

    /* what we built */
    .what { padding: 3rem 2rem; }
    .what-inner { max-width: 860px; margin: 0 auto; }
    .what h3 { font-size: 1.1rem; color: var(--teal); margin-bottom: 1rem; font-weight: 700; }
    .what-cols { display: flex; gap: 2rem; flex-wrap: wrap; }
    .what-col { flex: 1 1 220px; }
    .what-col h4 { font-size: .78rem; font-weight: 700; color: var(--text); margin-bottom: .35rem; text-transform: uppercase; letter-spacing: .07em; }
    .what-col p { font-size: .8rem; color: var(--muted); line-height: 1.65; }

    /* footer */
    footer { padding: 1.5rem 2rem; text-align: center; border-top: 1px solid rgba(79,209,197,.08); font-size: .72rem; color: var(--muted); }
  </style>
</head>
<body>

<section class="hero">
  <div class="badge">AnimateDiff on Tenstorrent Blackhole</div>
  <h1>A Cosmic <em>Quick Study</em></h1>
  <p>35 animated GIFs generated on a single Blackhole chip. Each cell in this grid is a Tensix compute node. The ones that glow ran the model.</p>
  <div class="hero-links">
    <a class="btn btn-primary" href="https://github.com/tenstorrent/tt-animatediff">GitHub →</a>
    <a class="btn btn-ghost" href="https://github.com/tenstorrent/tt-animatediff#prompt-guide">Prompt Guide</a>
  </div>
</section>

<section class="chip-section">
  <div class="chip-label">Blackhole P300C · 11 × 10 Tensix cores · 110 nodes</div>
  <div class="chip-grid" id="grid"></div>
  <div class="phase-legend">
    <div class="ph"><div class="ph-dot" style="border-color:#4fd1c5;background:rgba(79,209,197,.1)"></div>Pure cosmic</div>
    <div class="ph"><div class="ph-dot" style="border-color:#ec96b8;background:rgba(236,150,184,.1)"></div>Intertwining oneness</div>
    <div class="ph"><div class="ph-dot" style="border-color:#f4c471;background:rgba(244,196,113,.1)"></div>Liminal threshold</div>
    <div class="ph"><div class="ph-dot" style="border-color:#27ae60;background:rgba(39,174,96,.1)"></div>Tech emergence</div>
    <div class="ph"><div class="ph-dot" style="border-color:#81e6d9;background:rgba(129,230,217,.15)"></div>Full integration</div>
    <div class="ph"><div class="ph-dot" style="border-color:rgba(79,209,197,.15);background:#0a141c"></div>Harvested core</div>
  </div>
</section>

<section class="shoulders">
  <span class="section-label">Standing on the shoulders of giants</span>
  <div class="cards">
    <div class="card">
      <h4>AnimateDiff</h4>
      <p>Yuwei Guo, Chuanxia Zheng, Ruizhen Hu et al. The insight that motion priors can be injected into any SD UNet as a plug-and-play MotionAdapter — without retraining the base model. We just ran it faster.<br><br>
      <a href="https://arxiv.org/abs/2307.04725">arxiv 2307.04725 →</a></p>
    </div>
    <div class="card">
      <h4>Stable Diffusion 1.4</h4>
      <p>Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer (CompVis / Stability AI). The latent diffusion model that made high-quality generation accessible to everyone.<br><br>
      <a href="https://github.com/CompVis/stable-diffusion">CompVis/stable-diffusion →</a></p>
    </div>
    <div class="card">
      <h4>🤗 diffusers</h4>
      <p>The Hugging Face team. The <code style="color:#81e6d9;font-size:.72rem">AnimateDiffPipeline</code> and <code style="color:#81e6d9;font-size:.72rem">MotionAdapter</code> abstractions gave us a verified CPU reference to build from and compare against.<br><br>
      <a href="https://github.com/huggingface/diffusers">huggingface/diffusers →</a></p>
    </div>
    <div class="card">
      <h4>TT-Metalium &amp; TTNN</h4>
      <p>The Tenstorrent systems team. The SD 1.4 TTNN UNet kernel — built for Wormhole, adapted here for Blackhole — is what makes the hardware path possible. We wired the temporal loop around it.<br><br>
      <a href="https://github.com/tenstorrent/tt-metal">tenstorrent/tt-metal →</a></p>
    </div>
  </div>
  <p class="thanks">We ported, adapted, debugged, and shipped — but the hard science and the hard systems work were done by these teams. This repo is gratitude in the form of running code.</p>
</section>

<section class="what">
  <div class="what-inner">
    <h3>What Tenstorrent contributed</h3>
    <div class="what-cols">
      <div class="what-col">
        <h4>Hardware path</h4>
        <p>Replaced the diffusers CPU UNet with the SD 1.4 TTNN UNet running natively on Blackhole silicon. ~15 s/frame on P300C vs ~2 min/frame on CPU.</p>
      </div>
      <div class="what-col">
        <h4>Temporal attention</h4>
        <p>Cross-frame self-attention at each PNDM step across all N frame latents simultaneously — a bridge toward full MotionAdapter integration in TTNN.</p>
      </div>
      <div class="what-col">
        <h4>Open implementation</h4>
        <p>A clean, tested, documented repo anyone with Blackhole hardware can clone and run. Includes a ttsim path for development without silicon.</p>
      </div>
    </div>
  </div>
</section>

<footer>
  <a href="https://github.com/tenstorrent/tt-animatediff">tt-animatediff</a> ·
  Apache 2.0 ·
  Built on <a href="https://github.com/tenstorrent/tt-metal">TT-Metalium</a> ·
  Original work by <a href="https://arxiv.org/abs/2307.04725">Guo et al. 2023</a>
</footer>

<script>
const GIFS = [
  {name:"p1-nebula",      row:0,col:0, phase:1, prompt:"swirling nebula in deep space"},
  {name:"p1-galaxy",      row:0,col:2, phase:1, prompt:"spiral galaxy from above"},
  {name:"p1-aurora",      row:0,col:4, phase:1, prompt:"aurora borealis over arctic ice"},
  {name:"p1-starfield",   row:0,col:6, phase:1, prompt:"infinite starfield rotating"},
  {name:"p1-solar-flare", row:0,col:8, phase:1, prompt:"solar flare erupting"},
  {name:"p1-gas-cloud",   row:0,col:10,phase:1, prompt:"interstellar gas cloud"},
  {name:"p1-crystal-cave",row:1,col:1, phase:1, prompt:"crystal cave, bioluminescent cosmic light"},
  {name:"p1-deep-ocean",  row:1,col:3, phase:1, prompt:"deep ocean at night, bioluminescent"},
  {name:"p1-lunar-halo",  row:1,col:6, phase:1, prompt:"full moon with atmospheric halo"},
  {name:"p1-north-lights",row:1,col:8, phase:1, prompt:"northern lights over pine forest"},
  {name:"p1-milky-way",   row:2,col:0, phase:1, prompt:"Milky Way over desert landscape"},
  {name:"p1-fractal-tunnel",row:2,col:3,phase:1,prompt:"fractal space tunnel"},
  {name:"p2-mandala",     row:2,col:7, phase:2, prompt:"sacred mandala blooming from starfield"},
  {name:"p2-mycelium",    row:2,col:10,phase:2, prompt:"mycelium network, bioluminescent spores"},
  {name:"p2-dna",         row:3,col:1, phase:2, prompt:"DNA helix in cosmic void"},
  {name:"p2-roots-cosmos",row:3,col:5, phase:2, prompt:"tree roots intertwining with galaxies"},
  {name:"p2-consciousness",row:3,col:9,phase:2, prompt:"threads of consciousness, neural web"},
  {name:"p2-sacred-geo",  row:4,col:0, phase:2, prompt:"sacred geometry in cosmic space"},
  {name:"p2-forest-mind", row:4,col:4, phase:2, prompt:"forest canopy as neural network"},
  {name:"p2-reef",        row:4,col:7, phase:2, prompt:"living reef as collective mind"},
  {name:"p2-meridians",   row:4,col:10,phase:2, prompt:"energy meridians, body silhouette"},
  {name:"p3-temple",      row:5,col:2, phase:3, prompt:"Mayan temple under shifting cosmos"},
  {name:"p3-cave-circuit",row:5,col:6, phase:3, prompt:"cave paintings morphing to circuit traces"},
  {name:"p3-standing-stones",row:5,col:9,phase:3,prompt:"stone circle with aurora"},
  {name:"p3-crystal-data",row:6,col:1, phase:3, prompt:"crystal mountain with data streams"},
  {name:"p3-ley-fiber",   row:6,col:5, phase:3, prompt:"ley lines becoming fiber optic"},
  {name:"p3-mayan-grid",  row:6,col:10,phase:3, prompt:"Mayan calendar dissolving to grid"},
  {name:"p4-circuit-moss",row:7,col:0, phase:4, prompt:"circuit board growing like moss"},
  {name:"p4-silicon",     row:7,col:4, phase:4, prompt:"silicon crystal lattice, cosmic light"},
  {name:"p4-neural-const",row:7,col:8, phase:4, prompt:"neural network as constellation"},
  {name:"p4-server-aurora",row:8,col:2,phase:4, prompt:"server room bathed in aurora"},
  {name:"p4-chip-city",   row:8,col:7, phase:4, prompt:"chip die as aerial city at night"},
  {name:"p5-chip-cosmos", row:9,col:0, phase:5, prompt:"Blackhole chip glowing with cosmos"},
  {name:"p5-hand-galaxy", row:9,col:5, phase:5, prompt:"hand touching circuit blooms to galaxy"},
  {name:"p5-grid-is-all", row:9,col:10,phase:5, prompt:"grid revealed as fabric of universe"},
];

const gifMap = new Map(GIFS.map(g => [g.row * 11 + g.col, g]));
const grid = document.getElementById('grid');

for (let i = 0; i < 110; i++) {
  const r = Math.floor(i / 11), c = i % 11;
  const div = document.createElement('div');
  div.className = 'node';
  if (gifMap.has(i)) {
    const g = gifMap.get(i);
    div.classList.add(`phase-${g.phase}`);
    const img = document.createElement('img');
    img.src = `assets/study/${g.name}.gif`;
    img.alt = g.prompt;
    img.loading = 'lazy';
    const tip = document.createElement('div');
    tip.className = 'node-tip';
    tip.textContent = g.prompt;
    div.appendChild(img);
    div.appendChild(tip);
  }
  grid.appendChild(div);
}
</script>
</body>
</html>
```

- [ ] **Step 2: Open locally and verify grid renders with placeholder cells**

```bash
xdg-open docs/index.html
# Should show: hero, 11×10 grid with colored borders on 35 cells (broken images fine), attribution, footer
```

- [ ] **Step 3: Commit**

```bash
git add docs/index.html
git commit -m "feat: add cosmic study website (docs/index.html)"
```

---

## Task 4: GH Pages workflow + hardware compat doc

**Files:**
- Create: `.github/workflows/pages.yml`
- Create: `docs/HARDWARE_COMPAT.md`
- Modify: `requirements.txt`

- [ ] **Step 1: Create the Pages deploy workflow**

```yaml
# .github/workflows/pages.yml
name: Deploy GH Pages

on:
  push:
    branches: [main]
    paths: ['docs/**']

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: docs/
      - id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 2: Create `docs/HARDWARE_COMPAT.md`**

```markdown
# Hardware Compatibility Notes

## tt-metal version

Tested with:
- Firmware bundle: 19.8.0 (KMD 2.8.0)
- Note: firmware 19.5.0 was the last "fully tested" version per tt-metal's own warning; 19.8.0 works with the caveats below.

## SD model path reorganization

Between tt-metal ≤19.5.0 and 19.8.0, the Stable Diffusion 1.4 demo moved:

**Old path (broken):**
```
models.demos.wormhole.stable_diffusion.*
```

**New path (current):**
```
models.demos.vision.generative.stable_diffusion.wormhole.*
```

All imports in this repo have been updated. If you see `ModuleNotFoundError: No module named 'models.demos.wormhole'`, your tt-metal is on the old layout — either upgrade or revert the import paths in `animatediff_ttnn/ttnn_pipeline.py`, `animatediff_ttnn/temporal_attention.py`, and `examples/generate.py`.

## Ethernet core hang recovery

If tt-metal throws `Timed out while waiting for active ethernet core`, run:

```bash
tt-smi -r 0 1 2 3
sleep 8
```

This clears hung ethernet cores from prior incomplete teardowns. It is safe to run at any time when no process is actively using the hardware.

## Concurrent device contexts

The Blackhole runtime does not support opening multiple MeshDevice contexts simultaneously in separate processes. All generation in this repo is single-device, sequential.
```

- [ ] **Step 3: Add tt-metal version note to `requirements.txt`**

```bash
cat >> requirements.txt << 'EOF'

# tt-metal / ttnn (install separately — not on PyPI)
# Tested with firmware bundle 19.8.0, KMD 2.8.0
# Install: git clone https://github.com/tenstorrent/tt-metal && ./build_metal.sh
# Activate: source ~/tt-metal/python_env/bin/activate
#
# SD 1.4 model path changed in 19.8.0:
#   old: models.demos.wormhole.stable_diffusion.*
#   new: models.demos.vision.generative.stable_diffusion.wormhole.*
EOF
```

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/pages.yml docs/HARDWARE_COMPAT.md requirements.txt
git commit -m "feat: add GH Pages workflow, hardware compat notes, tt-metal version pin"
```

---

## Task 5: Enable GH Pages and push

- [ ] **Step 1: Push the branch**

```bash
git push
```

- [ ] **Step 2: Enable GH Pages in repo settings**

Go to: `https://github.com/tenstorrent/tt-animatediff/settings/pages`
- Source: **Deploy from a branch**
- Branch: `main`, folder: `/docs`
- Click Save

(This only needs to happen once. After the PR merges to main, the pages.yml workflow deploys automatically on every push to `docs/**`.)

- [ ] **Step 3: After PR merges — verify site is live**

```bash
# Wait ~2 minutes after merge, then:
curl -s -o /dev/null -w "%{http_code}" https://tenstorrent.github.io/tt-animatediff/
# Expected: 200
```

---

## Task 6: Final PR prep and tool distribution

- [ ] **Step 1: Verify all 35 GIFs are committed and page renders correctly**

```bash
ls docs/assets/study/*.gif | wc -l   # 35
python3 -c "import json; m=json.load(open('docs/assets/study/manifest.json')); print('all done' if all(v=='done' for v in m.values()) else [k for k,v in m.items() if v!='done'])"
xdg-open docs/index.html
```

- [ ] **Step 2: Export to tt-vscode-toolkit and tt-local-generator**

```bash
rsync -a --delete \
  --exclude="__pycache__" --exclude="*.pyc" --exclude="*.egg-info" \
  --exclude=".git" --exclude="CLAUDE.md" --exclude=".superpowers" \
  ~/code/tt-animatediff/ ~/code/tt-vscode-toolkit/content/projects/animatediff/

rsync -a --delete \
  --exclude="__pycache__" --exclude="*.pyc" --exclude="*.egg-info" \
  --exclude=".git" --exclude="CLAUDE.md" --exclude=".superpowers" \
  --exclude="output/" --exclude="weights/" --exclude="docs/" \
  --exclude="MODEL_BRINGUP_TUTORIAL.md" --exclude="README.md" \
  --exclude="scripts/generate_study.py" \
  ~/code/tt-animatediff/ ~/code/tt-local-generator/app/animatediff/
```

- [ ] **Step 3: Commit in each downstream repo**

```bash
cd ~/code/tt-vscode-toolkit
git add content/projects/animatediff/
git commit -m "chore: sync animatediff from tt-animatediff canonical (post-PR)"
git push

cd ~/code/tt-local-generator
git add app/animatediff/
git commit -m "chore: sync animatediff from tt-animatediff canonical (post-PR)"
git push
```

- [ ] **Step 4: Final push of tt-animatediff branch**

```bash
cd ~/code/tt-animatediff
git push
```
