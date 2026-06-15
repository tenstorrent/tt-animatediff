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

Replaces serialized `for i in range(num_frames)` loops in Phase 2.5 and Phase 3 with single sharded TTNN calls across all N Blackhole chips.

**New helpers in `animatediff_ttnn/ttnn_pipeline.py`:**
- `shard_frames_to_device(frame_tensors, device, dtype, layout)` — stacks N same-shaped CPU tensors along `dim=0` and sends via `ShardTensorToMesh(dim=0)`. UNet use: N `[2, 4, lh, lw]` CFG-doubled tensors → chip K gets `[2, 4, lh, lw]` matching the compiled `batch_size=2` kernel. VAE use: N `[1, lh, lw, 4]` NHWC tensors.
- `gather_frames_from_device(tensor, device, num_frames, batch_per_frame=2)` — pulls via `ConcatMeshToTensor(dim=0)`, splits into N tensors of `[batch_per_frame, ...]`. Use `batch_per_frame=1` for VAE decode (not CFG-doubled).

**Constraint:** `num_frames % num_chips == 0` — enforced by `ValueError` guards at the start of `generate_frames_temporal` and `generate_frames_motion`. Valid frame counts for QB2 (4 chips): 4, 8, 12, 16.

**What's sharded:**
- Phase 2.5 UNet denoising loop (`generate_frames_temporal`) — ~4× speedup on 4 chips
- Phase 2.5 VAE decode (`generate_frames_temporal`) — ~4× speedup
- Phase 3 VAE decode (`generate_frames_motion`) — ~4× speedup
- Phase 3 UNet block loops (`forward_unet_staged`) — NOT sharded (blocks use on-device TTNN tensors; CPU-side shard helper incompatible)

**Expected speedup (4-chip QB2, 8 frames, 25 steps):**
- Phase 2.5 overall: ~3.2–4× vs 1-chip serial
- Phase 3 overall: ~2.5–2.9× (bottlenecked by `_apply_temporal` CPU calls at 7 injection points)

**Hardware smoke test:** `scripts/mesh_sharding_hw_test.py` — compares 1-chip vs 4-chip on 4-frame / 4-step generation, asserts PCC > 0.99, prints timing breakdown.

### LCM distillation (closed)
All four distillation runs failed (flat LR without warmup on sharp loss landscape).
Broken weights archived as `weights/*.broken`. Distillation track is closed.
