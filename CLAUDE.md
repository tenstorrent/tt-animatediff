# tt-animatediff — project notes for Claude

## What this project is
Canonical three-phase implementation of AnimateDiff on Tenstorrent Blackhole hardware.
- Phase 1: CPU AnimateDiff (full MotionAdapter, any machine)
- Phase 2/2.5: TTNN UNet on Blackhole, cross-frame temporal blend (--temporal-alpha)
- Phase 3: MotionAdapter injected into Blackhole denoising loop (--motion-adapter)
  via forward_unet_staged() + _apply_temporal() CPU round-trip at 7 injection points

## Export workflow
Public repo (v0.1.0+). Consumers vendor via git submodule — no rsync needed.
See ~/CLAUDE.md "tt-animatediff" section for details.

## Architecture phases
- Phase 1 (generate_baseline.py): CPU AnimateDiff with MotionAdapter
- Phase 2 (generate_blackhole.py): TTNN UNet on Blackhole, sequential frames
- Phase 2.5 (generate_blackhole_v2.py): TTNN UNet + cross-frame temporal attention ← default Blackhole
- Phase 3 (generate.py --motion-adapter): forward_unet_staged() replaces TTNN __call__,
  inserts _apply_temporal() after each cross-attention block (7 injection points, CPU round-trip)

## Key known issues
- ARC firmware hang on chip 3 (P300C board 0000046131924055) — see ~/qb2-debug/
  setup_blackhole() now reads hwmon sentinel values to skip dead chips
- ttnn_pipeline.py uses open_mesh_device (not open_device) — all chips claimed upfront
- VAE decode: Phase 4 mesh sharding sends VAE latents to TTNN device. If TTNN VAE conv_out OOMs on L1, fall back to CPU decode via `ttnn.to_torch` + diffusers VAE.

## Phase 3 MotionAdapter — two bugs fixed (2026-06-14)

**Bug 1:** `nn.Linear` weights are stored `[out, in]` but `TemporalAttentionKernel`
computed `x @ w` (expecting `[in, out]`). Silent shape match due to square [C,C]
matrices. Fixed by transposing in `load_weights()`. Tests in
`tests/test_ttlang_temporal_attention.py` updated to use `.T` reference.

**Bug 2 (root cause of noisy output):** `TemporalAttentionKernel` only implemented
QKV self-attention + residual. The full `AnimateDiffTransformer3D` module also applies:
GroupNorm, proj_in/proj_out (trained, not identity), LayerNorm ×3, positional embedding
(pe, norm=143 at C=1280), output projection bias, and GEGLU feedforward.
These missing components caused energy explosion (ratio up to 2.5×) at injection points.

**Fix:** `_apply_temporal()` in `ttnn_motion_pipeline.py` now calls
`AnimateDiffTransformer3D.forward()` directly via diffusers on CPU.
Tensors are unfolded `[1,1,2*S,C] → [B*N,C,H,W]`, passed through the full module,
then refolded. Spatial dims (H,W) per injection point stored in `InjectionPoint`
NamedTuple in `motion_weights.py`. Energy ratios after fix: all < 1.25.
Visual result: clear photographic Eiffel Tower vs previous noise.

## TT-Lang temporal attention track

New package `animatediff_ttnn/ttlang/` implementing AnimateDiff motion-module
temporal attention (`[S, N, C]` input) as TT-Lang DSL kernels, verified in the
functional simulator.

### Files
- `animatediff_ttnn/ttlang/__init__.py` — exports `TemporalAttentionKernel`
- `animatediff_ttnn/ttlang/sim_helpers.py` — `tensor_to_block` / `block_to_tensor`
- `animatediff_ttnn/ttlang/temporal_attention_kernel.py` — three kernels + wrapper:
  - `_qkv_kernel_sim` — QKV projection, row-streaming Block @
  - `_sdpa_kernel_sim` — scaled dot-product attention with stable softmax
  - `_out_proj_kernel_sim` — output projection + residual add
  - `TemporalAttentionKernel` — wrapper class (use_ttlang=False PyTorch, use_ttlang=True sim)
- `tests/test_ttlang_temporal_attention.py` — 9 simulator tests, all PCC > 0.999
- `scripts/ttlang_temporal_attn_hw_test.py` — dual P300c hardware smoke test

### Run simulator tests
```bash
PYTHONPATH=/home/ttuser/code/tt-lang/python python -m pytest tests/test_ttlang_temporal_attention.py -v
```

### Run hardware smoke test
```bash
source ~/tt-metal/python_env/bin/activate
TT_METAL_ARCH_NAME=blackhole python scripts/ttlang_temporal_attn_hw_test.py
```
Hardware results (dual P300c): C=320 PCC=0.9998, C=640 PCC=0.9989, C=1280 PCC=0.9949. All > 0.99.

## Mesh Frame Sharding (Phase 4)

Replaces the serialized `for i in range(num_frames)` UNet loop in Phase 2.5 with sharded TTNN calls across the Blackhole mesh. Frames are sharded one-per-chip in chunks of `num_chips` frames, so `num_frames > num_chips` runs `ceil(num_frames / num_chips)` sharded passes (driven by `plan_frame_sharding`).

**New helpers in `animatediff_ttnn/ttnn_pipeline.py`:**
- `shard_frames_to_device(frame_tensors, device, dtype, layout)` — stacks N same-shaped CPU tensors along `dim=0` and sends via `ShardTensorToMesh(dim=0)`. UNet use: N `[2, 4, lh, lw]` CFG-doubled tensors → chip K gets `[2, 4, lh, lw]` matching the compiled `batch_size=2` kernel. VAE use: N `[1, lh, lw, 4]` NHWC tensors.
- `gather_frames_from_device(tensor, device, num_frames, batch_per_frame=2)` — pulls via `ConcatMeshToTensor(dim=0)`, splits into N tensors of `[batch_per_frame, ...]`. Use `batch_per_frame=1` for VAE decode (not CFG-doubled).

**Constraint:** `num_frames % num_chips == 0` — enforced by `plan_frame_sharding` (Phase 2.5) and the `ValueError` guard in `generate_frames_motion`. A partial final chunk would place fewer than `num_chips` frames on the mesh, producing a mis-sized shard the batch=2 kernel rejects. Valid frame counts for QB2 (4 chips): 4, 8, 12, 16 (8/12/16 run 2/3/4 sharded passes).

**What's sharded:**
- Phase 2.5 UNet denoising loop (`generate_frames_temporal`) — chunked sharding, ~4× speedup on 4 chips when `num_frames` is a multiple of `num_chips`
- VAE decode (both phases) — NOT sharded; serial per-frame (the TTNN VAE takes batch=1 NHWC input, so frames are decoded one at a time)
- Phase 3 UNet block loops (`forward_unet_staged`) — NOT sharded (blocks use on-device TTNN tensors; CPU-side shard helper incompatible)

**Expected speedup (4-chip QB2, 8 frames, 25 steps):**
- Phase 2.5 UNet denoising: ~3.2–4× vs 1-chip serial (VAE decode stays serial, so overall wall-clock gain is lower)
- Phase 3 (`generate_frames_motion`): no mesh sharding — `forward_unet_staged` and VAE both run serial per-frame; the divisibility guard exists only for forward parity with Phase 2.5

**Hardware smoke test:** `scripts/mesh_sharding_hw_test.py` — see hardware findings below.

**HARDWARE FINDING (2026-06-16): ShardTensorToMesh fails on SD demo UNet.**
The tt-metal SD demo UNet (wormhole) is a single-device model. Its `__init__`
calls `ttnn.to_torch(weight)` without a `mesh_composer` inside `permute_conv_weights()`.
When `preprocess_model_parameters(device=4-chip-mesh)` puts weights on all 4 chips,
this call fails with `TT_FATAL: buffers.size() == 1`.

**Correct approach: `create_submeshes(MeshShape(1,1))`.**
Open a `MeshDevice(1×4)`, call `.create_submeshes(MeshShape(1,1))` to get 4
independent 1×1 MeshDevices (one per chip), load one `UNet2D` per submesh.
Dispatch one frame per chip. Verified 2026-06-16:
- All 4 chips pass PCC=1.0 vs reference (chip 0 single-chip run)
- Each chip produces identical output — deterministic seeded noise
- Model load per chip: ~9s (from compile cache), 1 frame/chip

`scripts/mesh_sharding_hw_test.py` updated to use this approach.

`shard_frames_to_device` / `gather_frames_from_device` remain useful utilities
for any future TTNN ops that natively support mesh input (e.g. custom kernels
not using the SD demo UNet wrapper).

## Phase 3 batched D→H transfer (2026-06-15)

`_apply_temporal` now pulls all N frame tensors from device in a single `ttnn.concat → ttnn.to_torch` call instead of N separate transfers. H→D path stays per-frame (`to_device` per frame) — `ttnn.split` produces parent-buffer views whose memory config is incompatible with the downstream resnet reshard kernel.

**Measured speedup (QB2, 4×P300c, 8 frames, 25 steps):**
- Before: ~806s (~101 s/frame)
- After:   416s (~52 s/frame)
- Speedup: 1.94×

`torch.compile` was removed — `AnimateDiffTransformer3D` triggers the recompile guard limit (8) on every unique attention processor object ID, falling back to eager with warnings. Batched D→H accounts for the full measured speedup.

**Key correctness constraints for future changes to `_apply_temporal`:**
- Module forward expects `[2*N, C, H, W]` with layout `[uncond_fr0, …, uncond_frN-1, cond_fr0, …, cond_frN-1]`.
  Stack: `torch.stack(frames, dim=1)` (not dim=0) → `[2, N, C, H, W]` → `.reshape(2*N, C, H, W)`.
- Unstack: `attended.reshape(2, N, C, H, W).permute(1,0,2,3,4)` → `[N, 2, C, H, W]`.
- Output inverse reshape: `[2,C,H,W].reshape(2,C,S).permute(0,2,1).reshape(1,1,2*S,C)` (NOT permute(0,2,3,1)).

### LCM distillation (closed)
All four distillation runs failed (flat LR without warmup on sharp loss landscape).
Broken weights archived as `weights/*.broken`. Distillation track is closed.

## Container serving: two traps found by an actual image build (2026-09-08)

The image layer was unverified through #9. A colleague's session built and served it for
real, which turned up two things no host serve can see:

**tt-metal v0.77.0 ships `import pytest` at module scope.** In
`models/common/utility_functions.py` (line 12), and that file ships in this package's
allowlist -- the SD demo's `common.py` does
`from models.common.utility_functions import is_blackhole`. So the container's lifespan dies
with `ModuleNotFoundError: No module named 'pytest'` the moment it opens the device. A host
serve never sees it, because tt-metal's `python_env` has pytest installed: the same
works-on-host, dies-in-image shape as the under-shipped `models/demos` allowlist.

Fixed upstream in tt-metal `fdcb5cf2ec6c` (#55079, merged 2026-09-03), which moves the
import inside the one function using it -- verified by reading the file at that commit. **No
release tag carries it**: v0.78.0 still has the module-level import. So the choice was ship
pytest or move `source.tt_metal.ref` 58 days past a tag onto a bare commit. `pytest` is in
`tt_model_package.lock` for now, with the reasoning inline and a test
(`test_the_lock_ships_pytest_only_while_tt_metal_needs_it`) that fails when `tt_metal.ref`
moves, so the workaround gets removed rather than inherited. The bump itself is verified to
build and serve (~137 s for a 512x512 GIF) but its effect on the SD demo's numerics is
unchecked, and every measurement in this repo was taken against v0.77.0.

**`~/.cache/huggingface/hub/` on this box symlinks some models to `/mnt/bonus/models/`.**
The container bind-mounts only `~/.cache/huggingface`, so a symlinked model resolves to a
dangling link inside it -- hit through the hardcoded `CompVis/stable-diffusion-v1-4` VAE
load. For a container serve here, use `HF_HOME=<fresh dir> tt-model serve ...` and let it
fetch over the network. A box property, not a repo bug, but it will look like one.

## Accepted security finding: accelerate (2026-09-08)

`Cycode: Vulnerable Dependencies` fails on PR #9 and **cannot be made green by a version
bump**. GHSA-4j2p-28q2-5m79 / CVE-2026-69112 (MEDIUM, published 2026-08-10) covers
`accelerate <= 1.14.0` — the whole release history — with `first_patched_version: null`.

Accepted rather than fixed, on the user's call. The reasoning, so nobody re-litigates it
from scratch:

* Nothing here imports accelerate. diffusers uses it for `low_cpu_mem_usage` and falls
  back to `False` with a warning when it is absent (`if low_cpu_mem_usage and not
  is_accelerate_available()`), so removing it costs peak memory at load, not correctness.
* That cost lands exactly where it hurts: the CPU pipeline is ~7.8 GB resident and the
  free-tier Space has 16 GB, which has already OOMed once. Trading a MEDIUM for a
  reintroduced OOM is the wrong direction.
* Exploiting it needs an attacker-controlled **sharded checkpoint**; every load path in
  this repo names a pinned upstream repo, and a caller who can redirect that is already
  running `trust_remote_code=True`.

Declared in four places, all equally exposed and none fixable by version:
`hf/requirements.txt`, `spaces/requirements.txt`, `tt_model_package.yaml`, `setup.py`.
The first two carry the rationale inline.

**The acceptance is conditional.** Revisit if a patched accelerate ships (bump at once),
or if any load path starts taking a caller-supplied checkpoint id — the argument rests on
those ids being pinned, not on the advisory being harmless. There is no in-repo Cycode
config and the bot offers no ignore command, so the suppression itself has to be recorded
in the Cycode console by someone with access; the check stays red until it is.

## Hugging Face publishing track (2026-08-19)

`episod/tt-animatediff` is a **weights-free diffusers custom pipeline**; the Space
`episod/tt-animatediff-demo` is a capped CPU-Lightning demo. Both are built from this
checkout by `scripts/build_hf_artifact.py` and uploaded by `scripts/publish_to_hub.py`
(private on create, `--yes` required to write, `--dry-run` inert, `--verify` read-only).
Never hand-edit either repo on the Hub — the next build overwrites it.

Spec: `docs/superpowers/specs/2026-08-19-hf-model-repo-design.md`.
Plan: `docs/superpowers/plans/2026-08-19-hf-model-repo.md`.
Release steps, including the one the tests cannot enforce (repinning
`tt_model_package.yaml`'s `extra_code` ref to the new tag): `docs/RELEASING.md`.

### Two diffusers traps that make `hf/pipeline.py` look wrong

Both were measured, and both bite silently on the **older** supported diffusers:

1. **Never write `import animatediff_ttnn` in `hf/pipeline.py`** — not even indented
   inside a method. diffusers' `check_imports` regex-scans the file (`^\s*import`), and
   on **0.32.1 it raises ImportError at load time** for any module not installed, so
   every user without the package pip-installed would be unable to load the pipeline at
   all. 0.39.0 only warns. Use `importlib.import_module(PACKAGE_NAME)`.
   `tests/test_hf_pipeline.py::test_pipeline_py_never_imports_the_package_literally`
   guards this.
2. **`__init__` must take named parameters with defaults and no `**kwargs`.** diffusers
   derives its expected-component list from the signature, so a `**kwargs`-only
   `__init__` fails with `ValueError: Pipeline ... expected ['kwargs']`.

Also measured: diffusers executes `pipeline.py` out of
`~/.cache/huggingface/modules/diffusers_modules/`, so the vendored `animatediff_ttnn/`
is **not** a sibling of `__file__`. `resolve_package()` finds it via
`config._name_or_path`, falling back to `snapshot_download(code_repo,
allow_patterns=["animatediff_ttnn/**"])` — which is why `code_repo` is in
`model_index.json`.

The Space upload is **staged**, not committed. `scripts/build_space_artifact.py` assembles `build/space/` from `spaces/` plus six gallery GIFs copied out of `docs/assets/` (see `GALLERY_SOURCES` there); `scripts/publish_to_hub.py --space` calls it and uploads the result. `spaces/gallery/` and `build/space/` are git-ignored. The GIFs total ~14 MB and already live in this repo, so committing copies into `spaces/gallery/` would have added them to git history permanently for no benefit. Consequence to remember: a file dropped into `spaces/` by hand reaches the Space, but a new gallery GIF does not unless it is added to `GALLERY_SOURCES`.

**Visibility, and what actually blocks the Space (2026-09-04).** `episod/tt-animatediff` is
now **public** — verified anonymously (`GET /api/models/...` and a `model_index.json` fetch with
no token both 200). That closes the private-repo trap: a Space gets no implicit credential for a
*private* model repo, so while it stayed private the Space would have built, reached "Running",
and then 401'd on a visitor's first click (its `from_pretrained(MODEL_REPO, ...)` had nothing to
authenticate with). If that repo is ever made private again, the Space needs an `HF_TOKEN`
secret instead. Noted in `spaces/README.md` too.

**The Space's dependency pins: four caps, then none (2026-09-07).** gradio was pinned at
4.44.1, and four more pins existed only because a deploy had failed without them —
`python_version: "3.12"` (3.13 dropped stdlib `audioop`, which gradio's pydub imports),
`huggingface_hub<1` (gradio's own oauth.py imports `HfFolder`, removed in 1.0),
`pydantic<2.11` (gradio-client 1.3.0's schema walker assumes `additionalProperties` is a
dict) and `starlette<1` (gradio calls `TemplateResponse(name, context)` in the pre-0.29
positional order). Each was real and measured.

**They also froze the stack below its security floors, which is what killed them.** Cycode
flagged starlette 0.52.1 (CVE-2026-54283, -48818 HIGH; -48710, -48817 MEDIUM) and
transformers 4.57.6 (CVE-2026-9856, -5241, -4372 HIGH; -1839 MEDIUM), and **no version
inside either cap is clean** — every fix ships in starlette 1.x and transformers 5.x. So
the caps were buying gradio-4 compatibility with known vulnerabilities. Moving to
**gradio 6.26.0** removed all four at once: it *requires* `starlette>=1.0.1` and
`huggingface-hub>=1.16.0`, accepts `pydantic>=2.0`, and pulls `audioop-lts` on 3.13.
`tests/test_build_space_artifact.py` now guards the opposite of what it used to — CVE
floors (`starlette>=1.3.1`, `transformers>=5.10`) plus a test that fails if any of the
four caps is reintroduced, because re-adding one is the obvious wrong fix for the next
gradio breakage.

**Two failure modes to remember from the gradio-4 era**, because they generalise: the
pydantic and starlette breakages both presented as a **RUNNING Space serving nothing** —
the crash was per request inside gradio's own route, not at startup. "stage: RUNNING" is
never evidence a Space works.

**And the check has to be the right one.** `import app.py`, `demo.get_api_info()` and a
`GET /config` all passed while starlette 1.x was breaking every page load, because none of
them renders a template. Verify a change to `spaces/requirements.txt` by installing it in
a clean venv, LAUNCHING `build/space/app.py`, fetching **`/`**, and running one real
generation — the page and the generate path fail independently. Then fetch `/`
anonymously after deploying.

**The CPU pipeline cache is bounded, and that is a memory limit not a style choice.**
`animatediff_ttnn._cpu_pipe_cache` is keyed on `(use_lightning, lightning_steps)` — a
value chosen from the Space's dropdown — and used to never evict, so a visitor who tried
both step counts left two pipelines resident and OOMed the 16 GB box. Measured with no
mocks: cap=1 gives 7.77 GB resident / 10.98 GB peak; cap=2 gives 14.37 GB / 17.58 GB.
`ANIMATEDIFF_CPU_PIPE_CACHE` raises it where there is memory to spare. Eviction happens
*before* the replacement is built, and a test asserts that ordering from inside the
builder — evicting afterwards still satisfies "the cache holds one" and still dies at the
peak.

**When measuring that, do not use `unittest.mock.patch` for the stub.** It records
`call_args_list`, which holds a reference to every argument including the pipeline. Two
runs reported the fix as not working and one reported a reference leak; all three were the
probe measuring itself. A plain stub function gave the true answer, and `malloc_trim(0)`
confirmed there was no allocator-retention problem to solve.

**The Space is published and public** (`episod/tt-animatediff-demo`, 2026-09-04). It got there
only after PRO was enabled on the account: `create_repo(repo_type="space")` had been returning
**402 Payment Required** — "Static Spaces are free for everyone, but hosting Gradio and Docker
Spaces on free cpu-basic requires a PRO subscription". Keep that mapping in mind, because the
402 arrives from `create_repo` and reads like a script bug: if `whoami()["isPro"]` is False,
`publish_to_hub.py --space --yes` cannot succeed, and the alternatives are an org with the
entitlement or a **static** gallery-only Space (which is not the demo this repo built).

## Pinned upstream weights revisions (2026-09-27, v0.11.1)

**Prompt.** A packaging-hygiene pass ahead of repackaging the v6 thin bundle with pinned
weights. No hardware. Add pinned revisions for SD 1.4 and for the motion adapter where it
is loaded, and record the HF repo collision described below.

**What changed.**
- New `animatediff_ttnn/weights_pins.py` holds one sha per upstream repo (HF API `sha`
  on 2026-09-27; all three repos were last modified in 2023 or early 2024).
- `revision_for(repo_id)` decides the revision for each load:
  - SD 1.4 gets `$TT_MODEL_WEIGHTS_REVISION` or the pin. tt-model-manager's v6 `run.sh`
    exports that variable from the manifest.
  - The adapter and Lightning get their own fixed pins. The env var is deliberately
    ignored for them, because it names SD 1.4's revision and would not resolve in
    another repo.
  - A local directory or an unknown repo id gets `None`, as before.
- Every hub load in the package now passes `revision=`. That is 10 calls: 4 in
  `generation_helpers.py` (the served path), 5 in `pipeline.py` and 1 in
  `motion_weights.py`.

**Why the code, not just the manifest.** `tt-model package-thin --weights-revision` only
changes what `tt-model pull` pre-fetches. The server calls `from_pretrained` by repo id
and would still resolve `main`, so a manifest-only pin is ineffective.

**Guard.** `tests/test_weights_pins.py` checks three things:
- the rules;
- the runtime wiring of the CPU-runnable loaders, with stubs;
- an AST sweep over `animatediff_ttnn/` that fails on any `from_pretrained` /
  `hf_hub_download` / `snapshot_download` without `revision=`.

I watched it fail against the pre-change `generation_helpers.py` and `motion_weights.py`
(5 unpinned loads listed by file:line), then pass once they were restored. I also
updated `test_the_manifest_declares_the_weights_the_served_path_actually_loads`, which
grepped for the repo as a string literal: it now checks the declaration against
`weights_pins.SD14_REPO`.

I ran the full CPU suite with `ttnn` blocked at import: 359 passed. The 7
`test_ttnn_motion_pipeline.py` failures are the block itself (those tests import ttnn),
not this change.

**Not pinned here.** The `scripts/` utilities (`distill_lcm.py` and others) still load
SD 1.4 by id alone. They are dev tools, not shipped in the wheel.

**Open issue, recorded and not acted on: one HF repo, two package formats.**
`tt_model_package.yaml` on `main` (the v5.1 container manifest) has
`repo: episod/tt-animatediff`, the same HF repo that now holds the v6 thin bundle
(`tt_kernel_manifest.json`, `run.sh`, `wheels/`) alongside the diffusers custom pipeline.
A `tt-model package` / push from the yaml would write a container manifest over the thin
bundle. Proposal: retarget the yaml's `repo:` to `episod/tt-animatediff-container` (or
retire the yaml) and drop the stale `tt-model-container` tag from the card. That is left
for a maintainer decision, not changed in this branch.

**Repackage needs:**
- a new `animatediff_ttnn-0.11.1` wheel;
- `package-thin --weights-revision 133a221b8aa7292a167afc5127cb63fb5005638b`;
- ideally allow-patterns limited to the SD 1.4 subfolders the server reads
  (`unet/`, `vae/`, `tokenizer/`, `text_encoder/`, plus `model_index.json`). A pull of
  the whole SD 1.4 repo is ~22 GB.

### 2026-09-28 — v0.11.2: review fix on PR #12
Copilot review: `motion_weights.load_motion_modules` defaulted to a copied adapter repo string
rather than `weights_pins.MOTION_ADAPTER_REPO`, so a change to the canonical id would leave the
default behind and `revision_for()` would silently return None (the default branch). It now
imports the constant. The regression test checks the default expression by name (AST), because
a value check passes on the old code too: the copy is equal today. It was seen failing on the
old code. Bumped to 0.11.2, since 0.11.1's wheel is already published on HF and this changes a
file inside it. `docs/model-card.md`'s version line must follow VERSION too
(`test_the_model_card_version_matches_VERSION`; the card becomes the artifact's README): the
first push missed it and failed CI, because the suite had been run BEFORE the VERSION bump, not
after it. Re-run the suite on the exact tree being pushed. Suite on the fixed tree: 367 passed
with ttnn import-blocked.

**Repackage from here needs the `animatediff_ttnn-0.11.2` wheel**, not 0.11.1 (the 0.11.1 section
above is the release already published). Everything else in that repackage list is unchanged.

*Superseded 2026-10-16:* the thin bundle is now built by `packaging/package-thin.sh` and ships no
`animatediff_ttnn` wheel. See "v6 thin bundle: canonical recipe" below.

## LoRA support and served features (v0.12 to v0.13, PR #15)

Numbers, request fields and how-to are in `docs/LORA.md`, `docs/features.html` and the README
changelog. This section keeps only what a new session would trip over.

**Design.** `animatediff_ttnn/lora.py` merges kohya-format `.safetensors` UNet LoRAs (SD 1.x) into the
CPU UNet before `preprocess_model_parameters`, so a LoRA costs nothing per step. The UNet half goes through
diffusers (`load_lora_into_unet`, `fuse_lora`, `unload_lora`); the text-encoder half is merged by hand
(`_merge_text_encoder`) because diffusers' loader finds no modules on the transformers 5.x CLIP layout.
`inspect_lora` + `check_lora_compatible` read the header first and refuse SDXL, LoHa, LoKr and DoRA files
before any chip is claimed (CLI and the server lifespan both call them before the device opens).

**The LCM decision lives in one place.** `variant_is_lcm()` in `lora.py` (a spec marked `#lcm`, or the
published LCM-LoRA repo id) is used by the server, by `load_sd14_variant` and by `examples/generate.py`. Do not
re-derive it from a flag or a repo name. LCM steps are capped at `LCM_ORIGINAL_STEPS` (50, in `weights_pins.py`),
which is also the scheduler's `original_inference_steps`. The post-step latent blend is skipped for LCM
because its `prev_sample` carries per-frame noise that blending averages away (on-chip A/B, 2026-10-14:
sharper, but 30 to 60% more frame-to-frame change).

**Served path** (`animatediff_ttnn/server/app.py`). Request fields: `lora`, `sampler`, `preview`,
`preview_every`, `request_id`, `chain_save`/`chain_from`/`chain_alpha`, `prompt_schedule`, `motion_adapter*`.
Checks that can fail run before the device lock (`request_conflicts`, `chain_conflict`) and `chain_conflict`
runs again under the lock. The preview store is one global slot; `request_id` is how a client tells whose
preview it is looking at, so the generator and this server release together with no compatibility path.

**Environment.** Use the repo `.venv` (diffusers 0.39+, peft, ttnn). `~/.tenstorrent-venv` has diffusers 0.32.1
and cannot import `diffusers.loaders`. `peft` must stay in `tt_model_package.yaml` and its lock: diffusers
imports it lazily, so the image build passes without it and every LoRA request then fails.

**Hardware.**
* One device-opening process at a time. Three at once left the chips unable to open, even singly.
* Never run `tt-smi -r` by hand on a leased board. Doing so on a leased board wedged it until a host reboot;
  `gozer release` does the reset. `gozer run` holds the lease about 30 s after the command exits while it
  resets, so wait for FREE before the next run.
* A reboot clears `/tmp`. Keep builds in `~/tt-model-builds-*` and logs in `~/tt-serve-logs/`.
* Every server run ends with a tt-metal `UmdException` abort after shutdown. It predates this work.
* `pkill -f` matches its own shell. Use a pid file.

**Packaging and releasing.**
* The container package is its own Hub repo, `episod/tt-animatediff-container-v5.1`.
  `scripts/publish_to_hub.py` targets the thin bundle `episod/tt-animatediff` and would replace its README and
  root code, so it is not run for container work.
* `tt_model_package.yaml`'s `extra_code.ref` must stay at the tag `main` ships while a PR is open:
  `test_the_pinned_extra_code_ref_ships_the_package_we_are_developing` compares trees and fails in CI
  otherwise (a commit pin on the branch did exactly that). Package from a copy of the manifest with the commit
  pin (`docs/RELEASING.md`), and repin to the release tag after the squash.
* `tt-model serve` takes Hub repo ids only (no local path, no `--env`; env goes through serve profiles in the
  manifest). `tt-model stop` takes the repo id. A `tt-model push` of a rebuilt package printed "no files have been
  modified" for a changed tree, so check what the Hub holds after a push.
* The generated card says "Runs on p150 or p150" when two profiles share hardware. It is a tt-model-manager
  quirk and cannot be fixed from this manifest.
* Release in lockstep with tt-local-generator and update `docs/LORA.md`, the README changelog and
  `docs/features.html` with each served feature.

**Measuring.**
* Phase 2.5, 8 frames, 512x512, 25 steps, one chip: about 15.5 s warm (1.94 s/frame). A fresh `generate.py`
  process takes about 21 s because the first call warms up. The older ~100 s figure was stale. The sharding
  timings (16 frames ~86 s and others) rest on that stale baseline and were not re-measured.
* A timing taken while other agents load the box is not evidence. A 38 s plain run against a 15.3 s baseline
  was machine load (load average 7 to 10). Compare under equal load, alternating A and B, and record the load.
* `benchmark_modes.py` reads `loras` and `motion_adapter` beside `data`, at the top level of `/v1/models`.
  Its first request is a compile and is excluded from every statistic.
* A chaining check with the same seed for source, control and chained runs proves nothing, and a pixel
  difference metric scored the chained runs as less similar while the composition visibly carried over. Use a
  different seed and look at the frames.
* The "Pixel Art" words in a prompt ask for a grid. An LCM run with grid lines was the prompt, not the LoRA.

## v6 thin bundle: canonical recipe (2026-10-16, v0.13.3)

**Prompt.** Repackage the v6 thin bundle in the canonical shape documented in tt-model-manager
PR #183 (`docs/E2E_RECIPE_V6.md`, the `tt-model-thin-package` skill). Stage only, no hardware, no
push. Leave a committed, rerunnable script.

**What the published bundle was.** `episod/tt-animatediff` at 0.11.2 (kind `tt-dit-server`): stock
`ttnn==0.77.0`, plus two `--models-wheel` wheels: `animatediff-ttnn` (which also carried `tests/` and
`scripts/` as top-level packages) and `tt_animatediff_models_closure` (about 150 files of tt-metal
`models/common`, including other models' code, to run one model). No constraints file, no install
guards, no provenance. Its requirements file was named `tt-model-requirements.txt`, because an older
`tt-model` allowed that name. Current `tt-model` always writes `requirements.txt`. The CLAUDE.md
text above mentions a `kernel_patch/` directory. No such directory exists in this repo or in the
published bundle.

**What it is now** (`packaging/`, run `packaging/package-thin.sh`; `--install-test`, `--relock`):
- `extra_code/` (`--extra-code`) holds this package and the 32 tt-metal v0.77.0 files it imports,
  found by reading the source (`build_inputs.py`). The files are byte-identical to the ones in the
  published closure wheel. Empty `__init__.py` files are added where tt-metal has none, so nothing
  else named `models` can shadow the slice.
- `requirements.txt` holds `ttnn==0.77.0`, `torch==2.11.0` (tt-metal v0.77.0's pin) and the HTTP
  stack. Nothing else, because `install.sh` step 1 puts the PyTorch CPU index above PyPI. That index
  replaces `certifi`, `urllib3`, `requests` and `idna` with old releases. A resolve that reaches them
  fails under constraints and silently downgrades without.
- The pip dependencies (diffusers, transformers, peft, accelerate, huggingface_hub, pytest) travel in
  `tt-animatediff-deps`, a metadata-only wheel built by the script and passed with `--ops-wheel`.
  `install.sh` installs `--ops-wheel` wheels in step 3 from PyPI only. Delete it when tt-model scopes
  the CPU index to torch (see `runtime-deps.txt`).
- `constraints.txt` is generated by `--relock` in two resolves (step 1 as install.sh does it, then
  step 3 with those pins fixed). `validated-pins.txt` holds the model-library versions the container
  served on a card; a test checks them against `tt_model_package.lock`. `idna==3.4` stays pinned
  because fastapi reaches it in step 1 and the CPU index serves only that version.
- `--weights-allow` limits the pull to the fp32 safetensors for the UNet, VAE and text encoder plus
  the tokenizer: about 4.3 GB, not 22 GB. A test fails if the code starts loading another SD 1.4
  subfolder.
- Five `--verify` guards in `install-guards.txt` (stock ttnn, model code present, `models` from
  `extra_code/`, runtime packages present, `import model` without loading ttnn). Each has a test that
  runs it against a layout it must accept and one it must reject.
- `--provenance` writes the source commit, the tt-metal tag and commit, and hashes.
- `--tool-parser` and `--reasoning-parser` do not apply (not an LLM). There is no ops wheel (every
  kernel is a stock ttnn op) and no tt-transformers (the model imports none), so `deps.custom_ops`
  is empty and no tooling wheel ships.

**What a future session should not redo.** `--models-wheel` is gone on purpose. Do not put
diffusers or transformers back into `requirements.txt`. Do not hand-edit `constraints.txt`.
If `animatediff_ttnn` starts importing a new tt-metal module, the closure picks it up. If it starts
loading a new SD 1.4 subfolder, `test_the_weights_allow_list_covers_every_subfolder_the_server_loads`
fails until `WEIGHTS_ALLOW` follows.

**Open issue: do not push this bundle to `episod/tt-animatediff` as it stands.** That repo holds the
diffusers pipeline, whose root `requirements.txt` has the same name as the bundle's. `tt-model`
refuses to overwrite foreign files only in a repo that is not yet a bundle; this one is, so the push
would replace the pipeline's file silently. Decide first: a separate repo for the bundle, or a
renamed pipeline requirements file (`scripts/build_hf_artifact.py` and `hf/pipeline.py`).

**Verified on 2026-10-16 (no card):** staging with `tt-model package-thin --out` (no repo id);
`run.sh` argv (expanded through bash: 10 tokens, `model:app` is one), env,
`PYTHONPATH` (`extra_code` first), `deps.custom_ops` empty, `deps.models_wheels` empty; static import
resolution of every first-party import; `install.sh` in a copy (exit 0, "All installed packages are
compatible", all five guards ran, certifi 2026.7.22 / requests 2.34.2 / urllib3 2.8.0 installed);
`find_spec` for every third-party import in the installed venv; an offline CPU load of the UNet, VAE,
text encoder and tokenizer from only the allowed files; the CPU suite with ttnn blocked (549 passed).
Mutation checks: eleven deliberate breakages of the script, the walker, the inspector, the guards and
the committed inputs each turned a test red.

**Not verified:** serving on a card (`run.sh`, `Application startup complete`, `GET /v1/models`, a
generation, comparison against the container's output); that `ttnn==0.77.0` from PyPI behaves like the
image's tt-metal build on a card; the SFPI check against this host (7.87 installed, 7.69 declared, a
warning only); a pull of the published copy; the bundle with the LoRA catalog or motion adapter
(the default is the plain model, as before; set `ANIMATEDIFF_LORAS` in the environment to use a
catalog). A full pytest run made earlier the same day, without ttnn blocked, imported ttnn from
`~/tt-metal` through a `.pth` file in the repo `.venv`; use the blocked run above.
