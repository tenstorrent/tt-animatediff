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

## Hugging Face publishing track (2026-08-19)

`episod/tt-animatediff` is a **weights-free diffusers custom pipeline**; the Space
`episod/tt-animatediff-demo` is a capped CPU-Lightning demo. Both are built from this
checkout by `scripts/build_hf_artifact.py` and uploaded by `scripts/publish_to_hub.py`
(private on create, `--yes` required to write, `--dry-run` inert, `--verify` read-only).
Never hand-edit either repo on the Hub — the next build overwrites it.

Spec: `docs/superpowers/specs/2026-08-19-hf-model-repo-design.md`.
Plan: `docs/superpowers/plans/2026-08-19-hf-model-repo.md`.

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

**Private-repo trap:** a Space gets no implicit credential for a **private** model repo. While
`episod/tt-animatediff` stays private, the Space would build, reach "Running", and then 401 on
a visitor's first click (its `from_pretrained(MODEL_REPO, ...)` call has nothing to authenticate
with). Publishing the Space for real needs either an `HF_TOKEN` secret set on the Space, or the
model repo made public first. Noted in `spaces/README.md` too.
