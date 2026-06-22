# Mesh Frame Sharding — Implementation Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the serialized `for i in range(num_frames)` loops in Phase 2.5 and Phase 3 — both the UNet denoising calls and the VAE decode calls — with single sharded calls that distribute frames across all N Blackhole chips in the MeshDevice, achieving ~N× speedup on a 4-chip QB2 board.

**Architecture:** Stack all N CFG-doubled frame latents into a `[2N, 4, lh, lw]` tensor, shard along `dim=0` via `ShardTensorToMesh` so each chip receives `[2, 4, lh, lw]` (matching the already-compiled `batch_size=2` kernel), run one TTNN UNet call per step instead of N, then reassemble with `ConcatMeshToTensor(dim=0)`. Apply the same shard-and-gather pattern to VAE decode: stack N latents, shard so each chip decodes `N/num_chips` frames in parallel, gather the decoded images.

**Tech Stack:** TTNN MeshDevice API (`ShardTensorToMesh`, `ConcatMeshToTensor`), existing SD 1.4 TTNN UNet, diffusers schedulers, PyTorch CPU tensors.

---

## Constraint: `num_frames % num_chips == 0`

With 4 chips and 8 frames each chip gets `batch_size = 2*num_frames/num_chips = 2` — identical to the compiled kernel's `batch_size`. This is a hard requirement: both `generate_frames_temporal` and `generate_frames_motion` must raise `ValueError` at the start if `num_frames % num_chips != 0`. Valid frame counts for QB2: 4, 8, 12, 16. Single-chip mode (default) has no constraint.

## Data Flow

```
Current (ReplicateTensorToMesh — all chips run same computation):
  for frame i in [0..N]:
    CPU [2, 4, lh, lw] → chip 0 [2,4,lh,lw], chip 1 [2,4,lh,lw], ... (N-1 chips wasted)
    UNet(chip 0 result) → noise_pred_i

New (ShardTensorToMesh — each chip runs different frames):
  stack → CPU [2N, 4, lh, lw]
  shard → chip 0: [2, 4, lh, lw] (frames 0-1)
        → chip 1: [2, 4, lh, lw] (frames 2-3)
        → chip 2: [2, 4, lh, lw] (frames 4-5)
        → chip 3: [2, 4, lh, lw] (frames 6-7)
  UNet (one call, all chips in parallel)
  gather → CPU [2N, 4, lh, lw] → split → N CPU tensors

VAE decode (same pattern):
  stack → CPU [N, 4, lh, lw]
  shard → chip 0: [N/4, 4, lh, lw] (frames 0-1)
        → chip 1: [N/4, 4, lh, lw] (frames 2-3)
        → chip 2: [N/4, 4, lh, lw] (frames 4-5)
        → chip 3: [N/4, 4, lh, lw] (frames 6-7)
  ttnn_vae.decode (one call, all chips in parallel)
  gather → CPU [N, 3, H, W] → split → N decoded images
```

Text embeddings and time embeddings remain replicated (`ReplicateTensorToMesh`) — same value needed on every chip.

Weights are already replicated on all chips via `preprocess_model_parameters` with `ReplicateTensorToMesh`. Sharded inputs + replicated weights = correct, zero additional weight transfer.

## What Changes

### `animatediff_ttnn/ttnn_pipeline.py`

Add two helpers alongside `to_device`/`from_device`:

**`shard_frames_to_device(frame_tensors, device, dtype, layout)`**
- Input: list of N tensors, each `[2, 4, lh, lw]` (CFG-doubled, CPU)
- Stacks to `[2N, 4, lh, lw]`
- When `device` is a `MeshDevice` with M chips: calls `ttnn.from_torch(..., mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0))`
- When `device` is single-chip: falls back to `to_device` (no sharding needed; `[2, 4, lh, lw]` is the full tensor)
- Returns a single TTNN tensor distributed across chips

**`gather_frames_from_device(tensor, device, num_frames)`**
- When `device` is a `MeshDevice`: calls `ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))`
- Returns `[2N, 4, lh, lw]` CPU tensor, splits along `dim=0` into N tensors of `[2, 4, lh, lw]`
- When single-chip: returns `[tensor]` (list of one)

Single-chip degrades gracefully — `ShardTensorToMesh` on a 1-chip mesh with a `[2, ...]` tensor is semantically correct and produces identical results to the current path.

### `animatediff_ttnn/temporal_attention.py` — `generate_frames_temporal()`

Replace the per-frame UNet call block:

```python
# OLD
for i in range(num_frames):
    lat = to_device(frame_latents[i], ...)
    lat_input = ttnn.concat([lat, lat], dim=0)
    ttnn_out = ttnn_model(lat_input, ...)
    noise_preds.append(from_device(guided, device))
```

With a single sharded call:

```python
# NEW
# CFG-double each frame: [2, 4, lh, lw] per frame
cfg_latents = [ttnn.concat([lat, lat], dim=0) for lat in per_frame_device_tensors]
# But actually: stack CPU-side before sending
stacked_cpu = torch.cat([torch.cat([fl, fl], dim=0) for fl in frame_latents], dim=0)  # [2N, 4, lh, lw]
stacked_dev = shard_frames_to_device(stacked_cpu, device, ...)
ttnn_out = ttnn_model(stacked_dev, timestep=_tlist[step_idx],
                      encoder_hidden_states=ttnn_text_emb, ...)
noise_preds_stacked = gather_frames_from_device(ttnn_out, device, num_frames)
# noise_preds is now a list of N [2, 4, lh, lw] tensors; apply tt_guide per frame
```

Cross-frame attention and scheduler steps are unchanged — they already operate on CPU tensors.

Add `ValueError` guard at function entry:
```python
num_chips = device.get_num_devices() if isinstance(device, ttnn.MeshDevice) else 1
if num_frames % num_chips != 0:
    raise ValueError(f"num_frames ({num_frames}) must be divisible by num_chips ({num_chips})")
```

### `animatediff_ttnn/ttnn_motion_pipeline.py` — `forward_unet_staged()`

Same pattern applied to every block loop. Currently each block has:

```python
for i in range(num_frames):
    s, res_samples = down_block(hidden_states=hidden_samples[i], ...)
    new_hidden_dram.append(...)
```

Replace with:

```python
# Stack all N hidden states → shard → one block call → gather → unstack
stacked = shard_frames_to_device(hidden_samples_cpu, device, ...)
s_stacked, res_stacked = down_block(hidden_states=stacked, ...)
hidden_samples = gather_frames_from_device(s_stacked, device, num_frames)
# res_stacked: split residuals per-frame for the up-block accumulator
```

The evict-to-DRAM pattern between blocks is preserved — same reason (static CB conflict); now it applies to the `[2N/num_chips, ...]` shard on each chip rather than a `[2, ...]` single-frame tensor.

`_apply_temporal()` is unchanged — it already accepts and returns `list[ttnn_tensor]`, pulls each to CPU via `ttnn.to_torch`, runs diffusers CPU modules, pushes back. The only change: the list now has length N where N was previously N sequential calls.

Add the same `ValueError` guard at `forward_unet_staged` entry.

### VAE decode — `generate_frames_temporal()` and `generate_frames_motion()`

Both functions end with a serial decode loop:

```python
# OLD
for i, latent in enumerate(frame_latents):
    ttnn_lat = to_device(latent_scaled.permute(0, 2, 3, 1), device, ...)
    ttnn_decoded = ttnn_vae.decode(ttnn_lat)
    ...
```

Replace with a single sharded decode:

```python
# NEW — add a helper shard_latents_to_device / gather_decoded_from_device
# Stack N latents (already scaled, permuted NHWC): [N, lh, lw, 4]
stacked_lat = torch.cat([fl.permute(0, 2, 3, 1) / 0.18215 for fl in frame_latents], dim=0)  # [N, lh, lw, 4]
ttnn_lat = shard_frames_to_device(stacked_lat, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
ttnn_decoded = ttnn_vae.decode(ttnn_lat)                          # each chip decodes N/num_chips frames
ttnn_lat.deallocate(True)
ttnn_decoded = ttnn.reshape(ttnn_decoded, [num_frames, height, width, 3])
decoded_all = ttnn.to_torch(ttnn.permute(ttnn_decoded, [0, 3, 1, 2])).float()  # [N, 3, H, W]
ttnn_decoded.deallocate(True)
# Split N decoded images into individual PIL frames
frames = []
for i in range(num_frames):
    img = (decoded_all[i] / 2 + 0.5).clamp(0, 1)
    img = (img.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    frames.append(Image.fromarray(img))
```

The VAE decode sharding uses `shard_frames_to_device` with the latent batch dimension (not CFG-doubled — VAE takes 1 latent per frame, not 2). The NHWC permutation and `/0.18215` scaling happen CPU-side before sharding, same as before. The `ttnn_vae.decode()` call operates on whatever batch size it receives per chip — no compiled batch constraint here since TTNN Vae convolutions are spatially laid out, not batch-fixed.

## What Does NOT Change

- `motion_weights.py` — weight loading, `InjectionPoint`, `get_injection_point_info`
- `generation_helpers.py` — `ChainSession`, `encode_prompt`, `load_sd14_ttnn`
- `generate.py`, `app.py` — CLI flags and Gradio UI
- `_apply_temporal()` — CPU-side diffusers module application
- `cross_frame_attention()` — CPU-side self-attention

## Performance Model

With 4 chips and 8 frames (Phase 2.5):

| Component | Old | New | Speedup |
|---|---|---|---|
| UNet denoising (25 steps × 8 frames) | 200 serial calls | 25 calls (4× parallel) | ~4× |
| VAE decode (8 frames) | ~80s serial | ~20s sharded | ~4× |
| CLIP encode | ~2s | ~2s | 1× |
| **Overall** | **~800s** | **~200–250s** | **~3.2–4×** |

With 4 chips and 16 frames (Phase 3, MotionAdapter):

| Component | Old | New | Speedup |
|---|---|---|---|
| UNet denoising (25 steps × 16 frames) | 400 serial calls | 25 calls (4× parallel) | ~4× |
| `_apply_temporal` (7 pts × 25 steps, CPU) | ~150s | ~150s | 1× |
| VAE decode (16 frames) | ~160s serial | ~40s sharded | ~4× |
| **Overall** | **~800s** | **~280–320s** | **~2.5–2.9×** |

The remaining bottleneck in Phase 3 is `_apply_temporal` on CPU — 7 injection points × 25 steps of `AnimateDiffTransformer3D.forward()`. Accelerating that would require moving temporal attention itself onto Blackhole, a separate future project.

## Testing

### `tests/test_mesh_sharding.py` (unit tests, no hardware)

- `test_shard_frames_to_device_shape`: mocked MeshDevice, verify stacked shape `[2N, 4, 64, 64]` and `ShardTensorToMesh` mapper is used
- `test_gather_frames_from_device_split`: verify N tensors returned, each `[2, 4, 64, 64]`
- `test_singleship_passthrough`: single-chip device uses `to_device` path, no sharding mapper
- `test_num_frames_not_divisible_raises`: 7 frames / 4 chips → `ValueError`
- `test_num_frames_divisible_passes`: 8 frames / 4 chips → no error

### `scripts/mesh_sharding_hw_test.py` (hardware smoke test)

Runs a 4-frame generation (8 denoising steps, fixed seed) in both modes:
1. Serial mode: `device_ids=[0]`, single chip
2. Mesh mode: `device_ids=[0,1,2,3]`, 4 chips

Asserts PCC > 0.99 between the two final decoded frame tensors (determinism check — same weights, same input, same schedule, different dispatch).

Prints wall-clock breakdown (denoising time, VAE decode time, total) for both runs so the per-component speedup is directly visible.
