---
license: apache-2.0
library_name: diffusers
pipeline_tag: text-to-video
tags:
  - tenstorrent
  - blackhole
  - animatediff
  - ttnn
  - tt-metal
  - stable-diffusion
  - video-generation
base_model:
  - CompVis/stable-diffusion-v1-4
  - guoyww/animatediff-motion-adapter-v1-5-2
---

# tt-animatediff

**A port, not a checkpoint.** This repository ships no model weights. It is an
implementation of AnimateDiff that runs the SD 1.4 UNet on **Tenstorrent Blackhole**
hardware through TTNN, with cross-frame temporal attention for motion coherence. The
weights it needs — SD 1.4 and the AnimateDiff MotionAdapter — are resolved from their
upstream repositories the first time you generate.

Version 0.9.0 · [GitHub](https://github.com/tenstorrent/tt-animatediff) ·
[Docs](https://docs.tenstorrent.com/tt-animatediff/) ·
[How it was built](https://tsingletarytt.github.io/writing/2026/06/23/animatediff-on-tt-hardware-the-full-story/)

## Usage

`trust_remote_code=True` is required: this is a custom pipeline, so loading it executes
`pipeline.py` from this repo.

```python
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "episod/tt-animatediff",
    custom_pipeline="episod/tt-animatediff",
    trust_remote_code=True,
)

# mode="auto": Blackhole if the ttnn runtime is importable, CPU otherwise.
frames = pipe("a swirling nebula, teal and gold, cinematic").frames
frames[0].save("out.gif", save_all=True, append_images=frames[1:], duration=125, loop=0)

print(pipe.resolved_mode)  # "blackhole" or "cpu" — what actually ran
```

Loading is offline-safe and opens no device; nothing is fetched until you call the
pipeline.

`model_index.json`'s `base_model`, `motion_adapter`, and `lightning_repo` are
declarative metadata, not configurable inputs: they record the upstream weights this
pipeline resolves, but `__call__` never reads them, so passing a different value at
`from_pretrained()` time (e.g. `base_model="other/model"`) is accepted, persists in
`pipe.config`, and changes nothing about what actually runs.

**On a Blackhole box** (tt-metal built and its `python_env` active), `mode="blackhole"`
requires the hardware rather than falling back, so a missing runtime is an error instead
of a silent 100× slowdown:

```python
frames = pipe("a swirling nebula", mode="blackhole", num_frames=8, num_steps=25).frames
```

**On any machine, no hardware**, distilled 4-step Lightning weights make CPU tolerable:

```python
frames = pipe(
    "a swirling nebula", mode="cpu", use_lightning=True, lightning_steps=4,
    num_frames=4, guidance_scale=1.0,
).frames
```

The delegate package is imported if installed, and otherwise fetched from this repo
automatically. To install it explicitly:

```bash
pip install 'animatediff-ttnn @ git+https://github.com/tenstorrent/tt-animatediff'
```

## Implementation phases

| Phase | What runs where | Motion mechanism |
|---|---|---|
| 1 | CPU, diffusers `AnimateDiffPipeline` | Full MotionAdapter |
| 2.5 | **TTNN UNet on Blackhole** ← this pipeline's default | Cross-frame temporal attention (`temporal_alpha`) |
| 3 | TTNN UNet + MotionAdapter injected into the denoising loop | Full MotionAdapter, 7 injection points |

This pipeline runs Phase 2.5 on Blackhole and Phase 1 on CPU. Phase 3 is reached through
this repo's CLI (`--motion-adapter`), not through this pipeline.

## Measured performance

| Configuration | Hardware | Throughput |
|---|---|---|
| `mode="blackhole"`, 25 steps, PNDM | Blackhole P300C | **~12.5 s/frame** |
| `mode="blackhole"`, 25 steps, Euler, CFG 7.5 | Blackhole P300C | **~12.0 s/frame** |
| `mode="cpu"` | CPU (any machine) | ~2 min/frame |
| `mode="cpu"`, Lightning 4-step | CPU (any machine) | ~20 s/frame |

Blackhole figures are 8 frames at 512×512 on a single P300C chip (QB2 board, 4 × P300C), warm model — TTNN JIT already compiled. CPU figures are the reference path, not a target.

## Limitations

- The TTNN path needs a Blackhole board **and** a local tt-metal build (`ttnn` is not on
  PyPI). Most users of this repo will only ever exercise the CPU path.
- Frame count must be a multiple of the chip count on multi-chip boards (mesh frame
  sharding); 8 works on 1, 2, and 4 chips.
- The LCM distillation track is closed — all four runs failed and no distilled weights
  ship here. `use_lightning=True` on CPU uses ByteDance's published checkpoint.
- `mode="sim"` (ttsim virtual Blackhole) is bit-exact but 10–100× slower per op.

## Licensing — read this before redistributing output

**The code in this repository is Apache-2.0.** The weights it downloads at runtime are
not, and this repository cannot grant you their terms:

| Artifact | License |
|---|---|
| This repo's code | Apache-2.0 |
| `CompVis/stable-diffusion-v1-4` | CreativeML Open RAIL-M |
| `ByteDance/AnimateDiff-Lightning` | CreativeML Open RAIL-M |
| `guoyww/animatediff-motion-adapter-v1-5-2` | **Undeclared upstream** |

Using this pipeline means accepting the RAIL-M use restrictions on the weights it
fetches, even though this repository does not carry them. The MotionAdapter's terms are
not stated by its publisher; if that matters to your use, resolve it with the upstream
author rather than inferring permission from this repo's Apache-2.0 header.
