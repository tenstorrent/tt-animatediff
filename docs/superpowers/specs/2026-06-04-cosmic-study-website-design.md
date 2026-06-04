# Cosmic Study Website — Design Spec
_2026-06-04_

## Overview

A single-page static website (`docs/index.html`) deployed via GitHub Pages showcasing 35 animated GIFs generated on a Tenstorrent Blackhole P300C. The page is structured as a chip diagram — 11×10 Tensix nodes — where 35 nodes glow with generated animation and 75 remain dark (harvested cores). The grid is the hero; text is secondary.

Page order: **hero → chip grid → attribution → what we built → footer.**

---

## Generation plan

### GIFs to generate: 35 total

All generated with `examples/generate.py --mode blackhole --frames 8 --steps 25 --temporal-alpha <per-phase>`.

Narrative arc reads left→right, top→bottom across the grid.

#### Phase 1 — Pure cosmic (12 GIFs, teal, `--temporal-alpha 0.35`)

| # | Grid position (row, col) | Filename | Prompt |
|---|---|---|---|
| 1 | 0,0 | `p1-nebula.gif` | `swirling nebula in deep space, purple and teal gas clouds, stars, cinematic 4K` |
| 2 | 0,2 | `p1-galaxy.gif` | `spiral galaxy seen from above, golden core, trailing arms of blue starlight, cinematic` |
| 3 | 0,4 | `p1-aurora.gif` | `aurora borealis dancing over arctic ice, green and violet ribbons, starfield, cinematic` |
| 4 | 0,6 | `p1-starfield.gif` | `infinite starfield rotating slowly, Milky Way band, depth of field, cosmic, cinematic 4K` |
| 5 | 0,8 | `p1-solar-flare.gif` | `solar flare erupting from sun surface, plasma arcs, golden orange, cinematic` |
| 6 | 0,10 | `p1-gas-cloud.gif` | `interstellar gas cloud glowing teal and pink, cosmic dust, nebular light, cinematic 4K` |
| 7 | 1,1 | `p1-crystal-cave.gif` | `crystal cave glowing with bioluminescent cosmic light, amethyst formations, starlight within` |
| 8 | 1,3 | `p1-deep-ocean.gif` | `deep ocean at night, bioluminescent creatures, stars reflected on black water, cinematic` |
| 9 | 1,6 | `p1-lunar-halo.gif` | `full moon with atmospheric halo, aurora behind it, deep blue sky, cinematic 4K` |
| 10 | 1,8 | `p1-north-lights.gif` | `northern lights over ancient pine forest, green and magenta, slow swirl, cinematic` |
| 11 | 2,0 | `p1-milky-way.gif` | `Milky Way over desert landscape, rock formations silhouetted, cosmic scale, cinematic 4K` |
| 12 | 2,3 | `p1-fractal-tunnel.gif` | `fractal space tunnel, recursive geometry, teal and gold, infinite zoom, cinematic` |

#### Phase 2 — Intertwining oneness (9 GIFs, pink, `--temporal-alpha 0.4`)

| # | Grid position (row, col) | Filename | Prompt |
|---|---|---|---|
| 13 | 2,7 | `p2-mandala.gif` | `sacred mandala blooming from starfield, geometric petals, gold and violet, cosmic, cinematic` |
| 14 | 2,10 | `p2-mycelium.gif` | `mycelium network glowing with bioluminescent spores, threads of light connecting nodes, cosmic forest` |
| 15 | 3,1 | `p2-dna.gif` | `DNA double helix rotating in cosmic void, glowing teal, stars inside, cinematic 4K` |
| 16 | 3,5 | `p2-roots-cosmos.gif` | `ancient tree roots intertwining with stars and galaxies, roots become light filaments, cinematic` |
| 17 | 3,9 | `p2-consciousness.gif` | `threads of consciousness connecting glowing nodes, neural web across dark void, teal and gold` |
| 18 | 4,0 | `p2-sacred-geo.gif` | `sacred geometry unfolding in cosmic space, flower of life, metatrons cube, golden ratio spirals` |
| 19 | 4,4 | `p2-forest-mind.gif` | `forest canopy seen from below as neural network, branches are synapses, bioluminescent, cinematic` |
| 20 | 4,7 | `p2-reef.gif` | `living coral reef as collective mind, tendrils of light, pulsing bioluminescence, cosmic ocean` |
| 21 | 4,10 | `p2-meridians.gif` | `energy meridians flowing across a body silhouette, acupuncture lines glowing gold, cosmic backdrop` |

#### Phase 3 — Liminal threshold (6 GIFs, gold, `--temporal-alpha 0.4`)

| # | Grid position (row, col) | Filename | Prompt |
|---|---|---|---|
| 22 | 5,2 | `p3-temple.gif` | `ancient Mayan temple under shifting cosmos, nebula behind it, jungle emerging, cinematic 4K` |
| 23 | 5,6 | `p3-cave-circuit.gif` | `cave paintings of animals slowly morphing into circuit traces, orange firelight to teal digital glow` |
| 24 | 5,9 | `p3-standing-stones.gif` | `stone circle at dawn with aurora, ancient megaliths glowing with interior light, cosmic sky` |
| 25 | 6,1 | `p3-crystal-data.gif` | `crystal mountain formation with streams of data flowing through it, organic meets digital, cinematic` |
| 26 | 6,5 | `p3-ley-fiber.gif` | `ancient ley lines across landscape glowing gold, slowly becoming fiber optic cables, aerial view` |
| 27 | 6,10 | `p3-mayan-grid.gif` | `Mayan calendar stone dissolving into computational grid, stone becomes silicon, cosmic light` |

#### Phase 4 — Tech emergence (5 GIFs, green, `--temporal-alpha 0.45`)

| # | Grid position (row, col) | Filename | Prompt |
|---|---|---|---|
| 28 | 7,0 | `p4-circuit-moss.gif` | `circuit board growing like moss and lichen, organic circuitry, green teal glow, macro, cinematic` |
| 29 | 7,4 | `p4-silicon.gif` | `silicon crystal lattice glowing in cosmic light, atomic structure visible, teal and gold, cinematic` |
| 30 | 7,8 | `p4-neural-const.gif` | `neural network nodes arranged as constellation, synapses firing like shooting stars, dark space` |
| 31 | 8,2 | `p4-server-aurora.gif` | `server room bathed in aurora borealis light through glass ceiling, racks of machines, cinematic` |
| 32 | 8,7 | `p4-chip-city.gif` | `semiconductor chip die seen as aerial city at night, nanoscale streets glowing teal, cinematic 4K` |

#### Phase 5 — Full integration (3 GIFs, bright teal, `--temporal-alpha 0.5`)

| # | Grid position (row, col) | Filename | Prompt |
|---|---|---|---|
| 33 | 9,0 | `p5-chip-cosmos.gif` | `Tenstorrent Blackhole chip glowing with embedded cosmos, galaxies visible inside the die, cinematic` |
| 34 | 9,5 | `p5-hand-galaxy.gif` | `human hand touching a circuit board that blooms into a galaxy on contact, light erupting, cinematic` |
| 35 | 9,10 | `p5-grid-is-all.gif` | `computational grid slowly revealed to be the fabric of the universe itself, zoom out from chip to cosmos` |

### Negative prompt (all GIFs)
`blurry, low quality, distorted, text, people, faces, modern buildings`

### Seeds
Use seed 42 for all. If a result is poor, retry with seed 77, then 123.

### Resumability
All GIFs saved to `docs/assets/study/`. Generation script writes a `docs/assets/study/manifest.json` tracking which GIFs are done. On resume, script skips files that already exist and are non-zero size.

### Parallelism
Run sequentially — one MeshDevice(1×1) open at a time, one GIF at a time. The Blackhole runtime does not support concurrent device contexts. Each GIF takes ~3 min (8 frames × 25 steps × ~15s/frame includes kernel compile on first run, cached after). Total wall-clock: ~2 hours after first-run kernel compile (~20 min). The generation script runs in a tmux session so it survives disconnection.

---

## Page structure

### `docs/index.html` — single static file

**Section order:**

1. **Hero** — badge, h1 "A Cosmic Quick Study", one-paragraph tagline, two buttons (GitHub, Prompt Guide)
2. **Chip grid** — 11×10 node grid, phase legend below
3. **Attribution** — "Standing on the shoulders of giants" with four cards: AnimateDiff (Guo et al.), SD 1.4 (CompVis), diffusers (HuggingFace), TT-Metalium. Closing line: *"This repo is gratitude in the form of running code."*
4. **What we built** — three-column strip: Hardware path / Temporal attention / Open implementation
5. **Footer** — repo link, license, original paper credit

### Grid behavior

- 110 nodes in 11×10 CSS grid
- GIF nodes: `<img src="assets/study/p1-nebula.gif" loading="lazy">` fills the cell
- Dark nodes: empty div with dark background
- Phase color encoded in border-color (teal / pink / gold / green / bright teal)
- Node size: `72px × 72px` with `3px` gap — grid is ~830px wide, fits on 1080p without scroll
- Placeholder shown while GIF loads (dark with spinner or subtle gradient)

### GH Pages deployment

- Source: `docs/` folder, branch `taylor/phase2-blackhole-implementation` → merge to `main` → Pages serves from `main/docs/`
- Enable Pages in repo Settings → Pages → Source: Deploy from branch `main`, folder `/docs`
- During development, preview by opening `docs/index.html` directly in browser (all assets are relative paths)
- No build step — pure static HTML + GIFs committed to the repo

---

## tt-metal version pinning

Update `requirements.txt` and `README.md` to record the tt-metal version currently installed (firmware 19.8.0, KMD 2.8.0). Add a `docs/HARDWARE_COMPAT.md` noting the path reorganization from `models.demos.wormhole.stable_diffusion` → `models.demos.vision.generative.stable_diffusion.wormhole`.

---

## Files created / modified

```
docs/
  index.html                    ← new, the website
  assets/
    study/
      manifest.json             ← generation state tracker
      p1-nebula.gif             ← generated GIFs (35 total)
      p1-galaxy.gif
      ... (33 more)
  HARDWARE_COMPAT.md            ← new
  superpowers/specs/
    2026-06-04-cosmic-study-website-design.md  ← this file

scripts/
  generate_study.py             ← new, resumable batch generation script

.github/workflows/
  pages.yml                     ← new, GH Pages deploy workflow
```

---

## Out of scope

- JavaScript GIF player / lightbox (nice-to-have, not now)
- Multiple pages
- Any backend / server
- Analytics
- Mobile optimization beyond `overflow-x: auto` on the grid
