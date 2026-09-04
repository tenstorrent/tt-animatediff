# Hugging Face Model Repo (`episod/tt-animatediff`) — Design Spec

## Overview

Publish tt-animatediff to the Hugging Face Hub as `episod/tt-animatediff`: a
**diffusers custom pipeline** that loads with one `from_pretrained` call, selects the
TTNN Blackhole backend when `ttnn` is importable, and falls back to the existing CPU
`AnimateDiffPipeline` path otherwise. A companion Space
(`episod/tt-animatediff-demo`) offers a capped CPU-Lightning try-it.

The repo carries **no model weights**. It resolves SD 1.4 and the AnimateDiff
MotionAdapter from upstream repos at load time, which keeps the repo Apache-2.0 (a
few hundred KB) and avoids redistributing weights whose terms we cannot grant.

This is a packaging and distribution change. No generation behaviour changes; the
pipeline is a thin adapter over the already-tested `generate_animation()`.

### Why a custom pipeline rather than a code drop

The alternatives considered were a thin pointer repo (card plus a loader shim) and a
self-contained code mirror. Both leave the repo as "a README with code next to it" —
nothing on HF can actually load them. The custom pipeline is the only option where
`episod/tt-animatediff` earns the model label, and where the Space becomes a few lines
instead of a bespoke application.

## Verified loading mechanics

Both facts below were confirmed empirically against the installed `diffusers 0.39.0`
before this spec was written. They are the load-bearing assumptions of the design.

1. **A weights-free repo loads.** `DiffusionPipeline.from_pretrained(repo,
   custom_pipeline=repo, trust_remote_code=True)` against a directory containing only
   `model_index.json` and `pipeline.py` resolves the custom class, reports
   `Loading pipeline components...: 0it`, and returns an instance. No component
   subfolders are required.
2. **Scalar config keys reach `__init__`.** Extra top-level keys in `model_index.json`
   (`base_model`, `motion_adapter`, `temporal_alpha`) are passed as keyword arguments
   to the pipeline class constructor and persist in `pipe.config`.

`CUSTOM_PIPELINE_FILE_NAME` is `pipeline.py`
(`diffusers/pipelines/pipeline_loading_utils.py:67`), so the module must be named
exactly that at the repo root.

## Repo layout — `episod/tt-animatediff`

```
model_index.json      # _class_name, upstream model ids, generation defaults
pipeline.py           # TTAnimateDiffPipeline — the custom_pipeline entry point
animatediff_ttnn/     # vendored from this checkout by scripts/build_hf_artifact.py
app.py                # Gradio interface for local Blackhole or CPU use
requirements.txt      # copied from hf/requirements.txt; excludes ttnn/tt-metal
README.md             # the model card, copied from docs/model-card.md
LICENSE               # Apache-2.0
```

Nothing in this tree is hand-authored on the Hub. Every file is produced by the build
script from this repository, so the Hub copy cannot drift from GitHub.

### `model_index.json`

```json
{
  "_class_name": "TTAnimateDiffPipeline",
  "_diffusers_version": "0.39.0",
  "base_model": "CompVis/stable-diffusion-v1-4",
  "motion_adapter": "guoyww/animatediff-motion-adapter-v1-5-2",
  "lightning_repo": "ByteDance/AnimateDiff-Lightning",
  "temporal_alpha": 0.35,
  "num_frames": 8,
  "num_steps": 25,
  "guidance_scale": 7.5
}
```

Upstream ids live in config rather than hardcoded in `pipeline.py` so a future
adapter swap is a config edit, not a code change.

`_diffusers_version` records the version the artifact was built and verified against;
it is informational to diffusers. The enforced floor lives in `hf/requirements.txt` as
`diffusers>=0.32.1`, matching `setup.py`'s `install_requires`. The mechanics in
"Verified loading mechanics" were confirmed on 0.39.0, so `--verify` is the gate that
catches a future diffusers release changing custom-pipeline resolution.

## The pipeline contract

`pipeline.py` defines `TTAnimateDiffPipeline(DiffusionPipeline)`.

**`__init__(base_model, motion_adapter, lightning_repo, temporal_alpha, num_frames,
num_steps, guidance_scale)`** stores configuration via `register_to_config` and loads
nothing. `from_pretrained` therefore stays fast and works offline; no device is opened
and no weights are fetched until generation is requested.

**`__call__(prompt, negative_prompt="", num_frames=None, num_steps=None,
guidance_scale=None, seed=42, temporal_alpha=None, height=512, width=512,
mode="auto", use_lightning=False, lightning_steps=4, chain_from=None,
chain_save=None, output_type="pil")`** delegates to
`animatediff_ttnn.generate_animation()`. Arguments left as `None` fall back to the
config defaults, so `pipe("a nebula")` works with no further arguments.

This delegation is the central design decision. Backend selection, the CPU fallback,
the device singleton, and the chain-mode plumbing are all reached through
`generate_animation()`, which the suite already covers. **The CPU fallback is not new
code** — it is the existing tested path. `pipeline.py` adds no generation logic of its
own, only argument marshalling and the diffusers-shaped return value.

**Return value.** A `TTAnimateDiffPipelineOutput` dataclass with a `frames` attribute
(a list of PIL Images), so the object substitutes for diffusers'
`AnimateDiffPipelineOutput`. `output_type="pil"` returns PIL images;
`output_type="np"` converts to a stacked `numpy` array.

**Backend reporting.** A `resolved_mode` property exposes what `mode="auto"` chose, so
a caller — notably the Space — can state which backend actually ran rather than
guessing.

### Error handling

- `ttnn` missing with `mode="blackhole"` explicitly requested: raise `RuntimeError`
  naming the missing runtime and pointing at the docs. Do **not** silently fall back —
  a caller who asked for hardware needs to know they did not get it. `mode="auto"` is
  the only path that falls back silently, which is its documented purpose.
- Device open or weight-load failure: propagate. `session.ensure_blackhole()` already
  clears its cached device on failure so a retry is possible.
- `trust_remote_code` not passed: diffusers raises on its own; the card documents the
  flag and why it is required.

## Publish tooling

Both scripts live in **this** repository, not on the Hub.

### `scripts/build_hf_artifact.py`

Assembles `build/hf/` from the checkout: copies `animatediff_ttnn/` (excluding
`__pycache__`, and excluding `invokeai/` which is not carried on this branch), copies
`app.py` and `LICENSE`, copies `hf/pipeline.py` to the artifact root as
`pipeline.py`, copies `hf/requirements.txt`, copies `docs/model-card.md` to
`README.md`, and writes `model_index.json` from a literal in the script. Rebuilt from
scratch each run so a removed file cannot survive as a stale artifact.

`model_index.json` is the only generated file; everything else is a copy, so there is
exactly one place to look when the artifact is wrong.

Exits non-zero if the assembled tree fails an import check, so a broken artifact is
never offered to the publish step.

### `scripts/publish_to_hub.py`

Uploads `build/hf/` to `episod/tt-animatediff` and applies `docs/model-card.md` as the
card. Modelled on `tt-tnt/scripts/publish_to_hub.py`, with the same safety rules:

- Creates the repo **private**; there is no `--public` flag. Flipping visibility is a
  separate, explicitly-confirmed action outside this script.
- Any Hub write requires `--yes`. `--dry-run` never touches the Hub regardless.
- `--verify` is read-only and round-trips the **published** copy through
  `DiffusionPipeline.from_pretrained`, so it proves what a downstream user receives
  rather than re-checking local state.
- Re-runnable by design: `create_repo(..., private=True, exist_ok=True)` does not
  flip an existing repo's visibility.

## The Space — `episod/tt-animatediff-demo`

A separate Space repo, so the model repo stays loadable without demo dependencies.

- Loads `episod/tt-animatediff` via the custom pipeline and runs **CPU-Lightning
  only**. Blackhole is unreachable from HF infrastructure.
- Hard caps to keep it usable on free-tier CPU (2 vCPU): **4 frames, 4 steps, 384×384,
  one concurrent job** via `queue(max_size=...)` and `concurrency_limit=1`.
- A persistent banner stating that this is a CPU reference running a distilled
  4-step checkpoint, that it is not representative of Blackhole performance, and
  linking the measured numbers.
- Also presents a gallery of pre-rendered Blackhole output (World's Fair chain, Maya
  glyphs) so a visitor sees real quality even if they never wait for a generation.

Built and pushed only after the model repo is published and `--verify` passes.

Pushed by `scripts/publish_to_hub.py --space`, which uploads `space/` to a `space`-type
repo. It reuses the same safety rules as the model path (private on create, `--yes`
required to write, `--dry-run` honoured) so there is one publishing entry point with one
set of guardrails rather than two scripts that could drift apart.

## Model card

`docs/model-card.md` in this repo is the source of truth; the build script copies it
to the artifact's `README.md`.

**Front matter:** `license: apache-2.0`, `library_name: diffusers`,
`pipeline_tag: text-to-video`, `tags: [tenstorrent, blackhole, animatediff, ttnn,
tt-metal, stable-diffusion, video-generation]`, and `base_model` relations to
`CompVis/stable-diffusion-v1-4` and `guoyww/animatediff-motion-adapter-v1-5-2`.

**Prose sections:**

- What this is: the TTNN port, not a new set of weights. Stated in the first
  paragraph so nobody arrives expecting a checkpoint.
- Usage: both backends, including the `trust_remote_code=True` requirement.
- The three phases, and which one the pipeline runs by default (Phase 2.5).
- Measured performance, quoting the figures the repo already publishes: Blackhole
  P300C ~12.5 s/frame at 25 steps PNDM and ~12.0 s/frame at 8-step Euler; CPU
  ~2 min/frame, ~20 s/frame with Lightning. Each labelled with the hardware it was
  measured on.
- Limitations: the TTNN path needs a Blackhole board plus a tt-metal build, so most
  HF users will only ever exercise the CPU path; the LCM distillation track is closed
  and no distilled weights ship.
- **Licensing split** — this repo's code is Apache-2.0, but the weights it fetches at
  runtime are `creativeml-openrail-m` (SD 1.4, AnimateDiff-Lightning) and the
  MotionAdapter's terms are **undeclared** upstream. Users inherit those restrictions
  even though this repo does not carry them. This section exists because the
  distinction is invisible from the front matter alone.
- Links: `github.com/tenstorrent/tt-animatediff`,
  `docs.tenstorrent.com/tt-animatediff/`, and the writeup
  `tsingletarytt.github.io/writing/2026/06/23/animatediff-on-tt-hardware-the-full-story/`.

Version reported as `0.9.0`, matching `VERSION`.

## Testing

`tests/test_hf_pipeline.py` — hermetic, CPU-only, no network and no hardware:

- The built artifact loads via `from_pretrained(dir, custom_pipeline=dir,
  trust_remote_code=True)` and yields a `TTAnimateDiffPipeline`.
- Config passthrough: `base_model`, `motion_adapter`, and the generation defaults
  reach `__init__` and persist in `pipe.config`.
- `__init__` loads nothing: constructing the pipeline opens no device and fetches no
  weights (asserted by patching `generate_animation` and the session and checking
  neither is touched).
- `__call__` delegates to `generate_animation()` with the right arguments, including
  config defaults filling in for omitted ones, and chain-mode arguments forwarded.
- Backend selection with `ttnn` mocked present and absent; `mode="blackhole"` with
  `ttnn` absent raises rather than falling back.
- Output object shape: `.frames` is a list of PIL images; `output_type="np"` converts.

`tests/test_build_hf_artifact.py`:

- The artifact contains every required file, `model_index.json` is valid JSON with a
  `_class_name`, and no `__pycache__` or `invokeai/` leaks in.
- A rebuild over a dirty output directory removes stale files.

**On-hardware smoke run**, once, before publishing: build the artifact, load it, and
generate 4 frames on Blackhole under a `gozer` lease
(`gozer run --chips 1 --who "claude:hf-publish" --reason "verify TTNN branch"`). This
is the only way the TTNN branch gets real coverage; the mocked tests cannot prove it.

## Files created / modified

**Created in this repo:**

| File | Purpose |
|---|---|
| `scripts/build_hf_artifact.py` | Assemble `build/hf/` from the checkout |
| `scripts/publish_to_hub.py` | Upload the artifact and apply the card |
| `hf/pipeline.py` | Source of the custom pipeline the build script ships |
| `hf/requirements.txt` | Runtime deps copied into the artifact |
| `docs/model-card.md` | Model card, source of truth |
| `tests/test_hf_pipeline.py` | Custom pipeline tests |
| `tests/test_build_hf_artifact.py` | Artifact completeness tests |
| `space/app.py` | Space Gradio app (capped CPU-Lightning) |
| `space/README.md` | Space card with the required HF Space front matter |
| `space/requirements.txt` | Space runtime deps |

**Modified:**

| File | Change |
|---|---|
| `.gitignore` | Ignore `build/` |
| `README.md` | Link the HF model repo and Space |
| `CLAUDE.md` | Record the HF publishing track and the publish workflow |

## Out of scope

- Mirroring or re-hosting any model weights.
- The InvokeAI node (unproven end-to-end; stays on `feature/invokeai-api`).
- Writing the blog post — it already exists and is linked.
- Flipping either Hub repo public. Both are created private; visibility is a separate
  explicitly-confirmed step.
- Any change to generation behaviour, schedulers, or the TTNN kernels.
- Publishing a dataset repo of generated animations. Reconsider separately if the
  gallery outgrows the Space.
