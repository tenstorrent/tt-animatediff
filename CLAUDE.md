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

## LoRA support (2026-10-07, v0.12.0)

**Prompt.** "Commit the LoRA changes into a new branch, open a PR, then test the LoRAs on TT
hardware and put the results in the docs. Use extra spooky and mysterious prompts." Then, mid-run:
"do a code review too".

**What changed my mind.** The first commit (PR #15, draft) was a PEFT-on-MotionAdapter design and
its notes marked three LoRAs "Verified". Looking at the repos showed none is a PEFT adapter: all are
single kohya-format `.safetensors` UNet LoRAs for SD 1.5, and one (Krea-2) targets another base model.
That design could not load any of them. The PR was reworked on the same branch rather than closed.

**Design.** `animatediff_ttnn/lora.py` merges into the CPU UNet before `preprocess_model_parameters`.
UNet half: diffusers `load_lora_into_unet` + `fuse_lora` + `unload_lora`. Text-encoder half: merged by
hand, since diffusers' text-encoder loader finds no modules on the transformers 5.x CLIP layout
(IndexError on an empty rank map). `merge_loras` raises if a LoRA changes no UNet weight.

**Environment trap.** `~/.tenstorrent-venv` has diffusers 0.32.1 with transformers 5.x and cannot import
`diffusers.loaders`. Use the repo's `.venv` (diffusers 0.39, peft 0.21, ttnn).

**Hardware trap.** Opening three devices at once (three processes, `--device-id 0/1/2`) failed in
`risc_firmware_initializer` and left the chips unable to open even singly. `gozer release` reset them.
Run one device-opening process at a time. Also: `pkill -f` killed my own shell.

**Review fixes.** `hf_hub_download` without `revision=` failed `tests/test_weights_pins.py`; the module-level
CLIP cache would carry a merged LoRA into later loads (now reset in `load_sd14_ttnn`); `peft` was undeclared.

### 2026-10-13 — Copilot review on PR #15 (v0.12.1)
Four findings, all fixed in `animatediff_ttnn/lora.py`. Bumped to 0.12.1 (VERSION, model card, README changelog).
- **Non-finite scale.** `@1e999` matched the scale regex and `float()` gave inf, which fuse would write into every
  targeted weight. `LoraSpec.__post_init__` now rejects inf and nan, so specs built in code are covered too.
- **`~` paths.** `os.path.isfile` does not expand `~`, so an existing `~/x.safetensors` was reported missing.
  `_resolve_file` expands the source first.
- **Memory.** `merge_loras` cloned the whole UNet state dict per LoRA to count changed tensors. It now snapshots only the
  layers the loader just wrapped (`_lora_targets`, peft `base_layer` of modules holding the adapter). Checked against real
  peft on a tiny UNet: 24 of 24 targets found and changed, none left wrapped after unload.
- **Tests.** `TestMergeLoras` runs the UNet path on a fake UNet with a stubbed diffusers loader: fuse scaling, stacking,
  cleanup order, the zero-match error, scale 0, and that `state_dict()` is never called. 8 of the new tests fail on the old
  `lora.py` (scale, tilde and snapshot ones) and pass now. The fake mirrors peft's `base_layer` / `lora_A` shape, so the
  real-peft check above is the guard if peft changes it.
Full suite: 400 passed, 1 failed. The failure is `test_the_pinned_extra_code_ref_ships_the_package_we_are_developing`, which
compares the package tree with `main`'s; it fails for any branch that changes `animatediff_ttnn/` and is resolved by the
release repin in `docs/RELEASING.md`.

### 2026-10-13 — LoRA features for the first release (v0.13.0)
**Prompt.** After the review fixes: "work on some of these useful and faster items so our shipped version starts
on the best foot", then "how do we make our full feature set available when served via tt-model?".

**What was built.** Header pre-check (`inspect_lora`, `check_lora_compatible`), `scripts/lora_device_check.py`,
served LoRA variants (`ANIMATEDIFF_LORAS` catalog, `"lora"` request field, resident per variant, shared VAE),
`--lcm-lora` plus `use_lcm` in `generate_frames_temporal`, `prompt_schedule` in the served request, and `peft`
in the container manifest and lock. Details and numbers are in `docs/LORA.md`.

**Things that were wrong until checked.**
* The container lock had no `peft`. diffusers imports it lazily, so the build-time verify passed and every
  LoRA request in the image would have failed. Added to `tt_model_package.yaml`; the lock was recompiled from
  its own top-level requirements plus peft, and the diff is only peft and its `# via` lines. The LoRA tests and a
  real three-LoRA merge then ran in a bare venv built from the lock (diffusers 0.40, not the 0.39 used before).
* I assumed the LCM-LoRA was diffusers-format and would be refused by the new pre-check. Reading its header showed
  kohya keys with conv targets (a LoCon). The assumption was wrong, and the header read is why it was caught.
* The LCM-only run showed grid lines. Alpha 0 and 8 steps did not remove them; a neutral prompt did. The
  "Pixel Art" in the prompt asked for the grid. Without that control it would have been logged as an LCM bug.
* A port the box already used (8765) failed the first serve attempt; pick a free port.

**Hardware traps.** `gozer run` leaves the lease held for ~30 s after the server exits while it resets chips;
`gozer status` reports unreadable serials during that window. Wait for FREE before the next run.

**Not done.** Nothing was run through `tt-model serve` itself, only uvicorn on the chip. `tt_model_package.yaml`'s
`extra_code.ref` still points at v0.10.0 and must move to a tag that contains this release. The
`test_the_pinned_extra_code_ref_ships_the_package_we_are_developing` failure is that repin. A LoRA catalog needs
the files in the container's HF cache or network access at startup.

### 2026-10-13 — served previews, sampler, chaining, Phase 3 (v0.13.0, same branch)
**Prompt.** The tt-local-generator session asked for served previews; the user then said the two projects
release together ("code as if we are in lock step"), to keep docs and the site current per feature, and to notify
that agent per feature. Notes for later sessions: update `docs/LORA.md`, the README changelog and
`docs/features.html` (matrix row, section, request reference) with each served feature, then message the peer.

**Served request additions** (all in `animatediff_ttnn/server/app.py`): `preview` / `preview_every` plus
`GET /v1/videos/preview`; `sampler: euler`; `chain_save` / `chain_from` / `chain_alpha` backed by `ChainStore`;
`motion_adapter` (+ `_alpha`, `_skip`) enabled by `ANIMATEDIFF_MOTION_ADAPTER`; `request_conflicts()` returns the
400 reasons before the device lock; `/v1/models` reports `loras` and `motion_adapter`.

**Measured, not assumed.**
* Preview GIF rendering costs ~0.4 s per render on one chip (13.5-13.9 s -> 17.8-20.2 s for 8 frames x 20 steps),
  which is why the GIF is opt-in per request while the step counter is always on.
* The first chaining check used the same seed for source, control and chained runs, so the control already shared the
  source's noise; and a pixel-difference metric scored the chained runs as *less* similar even though the tower and
  cliff visibly carried over (colour dominates the metric). The check was redone with a different seed and judged by
  eye. The instrument, not the feature, was wrong both times.
* `tt-model`-level serving is still untested; everything was run through uvicorn on one chip.
* Every server run ends with a tt-metal `UmdException` abort after "Application shutdown complete" (exit non-zero);
  it predates this work.

### 2026-10-14 — container serve check and all-mode benchmark (v0.13.0, same branch)
**Prompt.** "go ahead and test the package serving after a tt-smi -r. benchmark all the modes again", then, after a
reboot, "fix docs error. continue".

**What happened.**
* `tt-smi -r 0 1` on the two chips of a leased board failed (`Reset failed`, warm_reset.cpp:337) and left the board
  unreadable: `tt_serial` returned "No data available", `tt-smi` found no device 0, and `gozer` refused every command
  for all four chips, including the healthy board. Only a host reboot recovered it. Next time let `gozer release`
  do the reset; a hand-run reset on a leased board is what broke it.
* The reboot cleared `/tmp`, including the scratchpad with the staged package. Keep builds out of `/tmp`.
* `tt-model serve` has no `--env`. It forwards only `HF_TOKEN`. My docs said `tt-model serve --env ANIMATEDIFF_LORAS=...`.
  The catalog goes in `serve.env` or a serve profile in the manifest; docs corrected.
* `tt-model serve <path>` does not take a local package directory, so the real `tt-model serve` run of this branch is
  still untested. The image itself was run with the `docker run` that `tt-model serve --print` composes.
* The image has `peft` and imports `animatediff_ttnn.lora` and the server module: the lock fix holds in a real image.
* The benchmark script first read `loras` from the wrong level of `/v1/models` and reported the LoRA and adapter modes as
  skipped. The server was right and the instrument was wrong.
* A 5.6 s motion-adapter figure from 2026-10-13 was a first request. Steady state is 3.4 s with `up1`,`up2` skipped, and
  about 24 s with nothing skipped, at alpha 0 as well.

### 2026-10-14 (later) — container package published, verified through `tt-model serve`
**Prompt.** "yes push both commits, and let's go ahead and publish the model on HF even before we review and approve
those PRs", then, after I explained that `publish_to_hub.py` would overwrite the thin bundle's README and root code,
"let's do option 2, episod/tt-animatediff-container-v5.1 ?".
* The v5.1 manifest's `repo:` is now `episod/tt-animatediff-container-v5.1`, with a `card:` block (numbers from the
  mode benchmark) and `extra_code.ref` pinned to `fb9ef60` (tree `ce376a9`, identical to the branch head's). That ref is
  not reachable from main until PR #15 merges and is squashed: repin to the tag or squash commit afterwards
  (docs/RELEASING.md step 5), or the next `tt-model package` fails once the branch is deleted.
* Pushed private, run through `tt-model serve episod/tt-animatediff-container-v5.1 --device-id 0` under a gozer lease
  (benchmark within 0.3 s of the docker-run numbers, frames byte-identical), then made public with
  `update_repo_settings(private=False)` and read back anonymously.
* The generated card says "Runs on **p150** or **p150**" when two profiles share hardware. That is a
  tt-model-manager card-generator quirk, not fixable from this manifest.
* `tt-model stop` takes the repo id (`episod/tt-animatediff-container-v5.1`), not the package name.
* A monitor that tested `pgrep -x gozer` reported "exited" while the serve was running (the process is python3). Wait on a
  pid you recorded.
* `scripts/publish_to_hub.py` (the diffusers pipeline repo `episod/tt-animatediff`) was NOT run. It would replace the live
  README with the generic model card and the root `animatediff_ttnn/` that the thin bundle's `run.sh` loads.
* `tt-model serve` is now the primary run path on the site's top page. `docs/index.html` has a `#run` section after the hero
  (the hero's main button points at it), `docs/usage.html` has `#install-tt-model` first, and the README Quick Start leads
  with it. User's instruction: "tt-model should now be presented as the primary way to deploy/run the model on the
  top-level part of the website". Docs only, so VERSION is unchanged. `docs/index.html`, `usage.html` and `benchmarks.html`
  still carry the older ~12.5 s/frame and ~100 s Phase 2.5 figures, while the README says ~1.94 s/frame and the measured
  served runs say 15.3 s per 8 frames; the CLI path has not been re-timed to settle which is right.
* **Phase 2.5 timing settled (2026-10-14).** User question: how do the new numbers compare to the spooky-loops blog post
  (~100 s per clip)? Under a gozer lease, `generate.py` (one chip, 8 frames, 512x512, 25 steps, seed 42) printed "Done in"
  21.2 s x3 (PNDM) and 20.6 s x3 (`--lightning`, Euler). Four `generate_animation` calls in one process took 28.3 s, then
  15.7 s, 15.8 s, 15.6 s. So 21 s is a first call, 15.5 s is warm (matches the model card and `tt-model serve`), and ~100 s
  was stale. My first blog edits assumed the blog's table was CLI-timed and nearly used 2.7 s/frame; the one-process run is
  what showed the CLI figure carries first-call warmup. The blog's sharding rows (16 frames ~86 s, 4 clips ~105 s) and
  `benchmarks.html`'s sharding text still rest on the stale single-chip baseline and are not re-measured. The Google Doc
  edits were made as suggestions (`writeMode: SUGGEST`); the doc's JSON marks them with `suggestedInsertionIds`.


### 2026-10-14 — second Copilot review on PR #15 (v0.13.1)
**Prompt.** "watch for another batch of reviews on this PR and tt-local-generator's please", then "GO AHEAD FOR BOTH 15 AND 29".
Four findings, each checked against the code before fixing, each with a test seen failing first.
* **Local LCM-LoRA** was only recognised by repo id. `LoraSpec.lcm` and the `#lcm` catalog marker add an explicit designation;
  `variant_is_lcm()` is the one decision, and a wiring test fails if `load_sd14_variant` compares repo names itself.
* **LCM steps above 50** were a 500 inside the loop. I ran `LCMScheduler.set_timesteps(51)` with `original_inference_steps=50`
  to confirm it raises, then made the limit one constant (`LCM_ORIGINAL_STEPS`) shared by the scheduler and `request_conflicts`.
* **Chain pruned while queued**: `chain_conflict()` runs before the lock and again under it. The test replaces the lock with one that
  deletes the chain on acquire, which is when a pruning generation would have finished. That test first failed by reaching generation.
* **Benchmark script** no longer guesses LCM from the name; it sends no step count for any variant.
Not repackaged: the published container package is still built from `fb9ef60` (0.13.0).
* **Preview ownership (from the tt-local-generator #29 review).** One global preview plus a queue on the device lock let a waiting
  client show another client's step-20 frame. The fix is in both repos, released together: `request_id` on the request,
  `PreviewStore.begin(total, request_id)`, echoed in `GET /v1/videos/preview`; the generator sends a fresh id and ignores previews
  with another one. No compatibility path: a server without the field gives the generator no previews, which it already treats as
  "no preview endpoint".
