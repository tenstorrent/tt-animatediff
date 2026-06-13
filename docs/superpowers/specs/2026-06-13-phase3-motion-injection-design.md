# Phase 3 — MotionAdapter Temporal Attention Injection: Design Spec

## Goal

Inject trained AnimateDiff MotionAdapter temporal attention into the Blackhole TTNN
denoising pipeline, replacing Phase 2.5's identity-weight cross-frame attention with
real learned QKV weights. No distillation needed — weights are loaded directly from
`guoyww/animatediff-motion-adapter-v1-5-2`. Zero modifications to tt-metal.

## Architecture

### New and modified files

```
animatediff_ttnn/
  ttnn_motion_pipeline.py   # NEW: staged UNet forward pass with temporal hooks
  motion_weights.py         # NEW: MotionAdapter weight loader
  temporal_attention.py     # MODIFIED: add generate_frames_motion()

examples/
  generate.py               # MODIFIED: add --motion-adapter flag
```

### How it works

The TTNN UNet model's block objects (`ttnn_model.down_blocks`, `ttnn_model.mid_block`,
`ttnn_model.up_blocks`) are Python attributes accessible after `preprocess_model_parameters`.
The monolithic `__call__` is orchestration code. We replicate that orchestration in
`forward_unet_staged()` — calling the same block objects in the same order with the same
tensor shapes — but insert a temporal attention step after each MotionAdapter injection
point, across all N frames before continuing to the next block.

This is architecturally identical to how AnimateDiff's MotionAdapter patches the diffusers
UNet: it wraps each ResBlock+Transformer pair with a TemporalTransformer that operates on
the same hidden states.

---

## Component 1: `animatediff_ttnn/motion_weights.py`

Loads the MotionAdapter checkpoint and returns a dict of `TemporalAttentionKernel`
instances keyed by injection point name.

```python
def load_motion_kernels(
    model_id: str = "guoyww/animatediff-motion-adapter-v1-5-2",
    num_frames: int = 8,
    use_ttlang: bool = False,
) -> dict[str, TemporalAttentionKernel]:
    """Load MotionAdapter weights → dict of TemporalAttentionKernel.

    Returns:
        Dict mapping injection point name → TemporalAttentionKernel with weights loaded.
        Keys: "down0", "down1", "down2", "mid", "up0", "up1", "up2"
    """
```

**Injection point → MotionAdapter key mapping:**

| Key | MotionAdapter prefixes (both modules) | C |
|---|---|---|
| `down0` | `down_blocks.0.motion_modules.[0,1].transformer_blocks.0.attn1` | 320 |
| `down1` | `down_blocks.1.motion_modules.[0,1].transformer_blocks.0.attn1` | 640 |
| `down2` | `down_blocks.2.motion_modules.[0,1].transformer_blocks.0.attn1` | 1280 |
| `mid` | `mid_block.motion_modules.0.transformer_blocks.0.attn1` | 1280 |
| `up0` | `up_blocks.0.motion_modules.[0,1].transformer_blocks.0.attn1` | 1280 |
| `up1` | `up_blocks.1.motion_modules.[0,1].transformer_blocks.0.attn1` | 640 |
| `up2` | `up_blocks.2.motion_modules.[0,1].transformer_blocks.0.attn1` | 320 |

`load_motion_kernels` returns a list of kernels per key: `down0 → [kernel_m0, kernel_m1]`,
applied in order at each injection point. `mid` has only one module so `mid → [kernel_m0]`.

Each block has 2 motion modules (layers_per_block=2). We run both in sequence — motion
module 0 then motion module 1 — matching how AnimateDiff applies them after each ResBlock.
This doubles temporal attention calls per block but is architecturally correct; the two
modules have distinct weights and process different features. If speed is a concern, a
future option is to average their weights into a single kernel (documented trade-off).

**Weight extraction per kernel:**
```python
w_q = sd[f"{prefix}.to_q.weight"]        # [C, C]
w_k = sd[f"{prefix}.to_k.weight"]        # [C, C]
w_v = sd[f"{prefix}.to_v.weight"]        # [C, C]
w_o = sd[f"{prefix}.to_out.0.weight"]    # [C, C]
kernel = TemporalAttentionKernel(dim=C, num_frames=num_frames, use_ttlang=use_ttlang)
kernel.load_weights(w_q, w_k, w_v, w_o)
```

---

## Component 2: `animatediff_ttnn/ttnn_motion_pipeline.py`

### `forward_unet_staged()`

```python
def forward_unet_staged(
    ttnn_model,
    frame_samples: list,          # N TTNN tensors, each [2, C, lH, lW] (CFG doubled)
    emb,                          # pre-computed time embedding TTNN tensor
    encoder_hidden_states,        # TTNN text embedding tensor
    config,                       # unet.config
    temporal_kernels: dict,       # from load_motion_kernels()
    device,
    num_frames: int,
    block_out_channels=(320, 640, 1280, 1280),
    layers_per_block=2,
    **unet_kwargs,
) -> list:                        # N TTNN tensors, each [2, 4, lH, lW] noise predictions
```

**Pseudocode:**

```
# conv_in: per frame (same emb for all frames at same timestep)
for i in range(N):
    sample[i] = conv_in(frame_samples[i])
    sample[i], emb[i] = setup(sample[i], emb)   # time embedding injection

# down blocks
down_res_samples = [[] for _ in range(N)]
for block_idx, down_block in enumerate(ttnn_model.down_blocks):
    C = block_out_channels[block_idx]
    for i in range(N):
        sample[i], res[i] = down_block(sample[i], emb, encoder_hidden_states, ...)
        down_res_samples[i].extend(res[i])

    if block_idx in {0, 1, 2}:   # CrossAttn blocks with MotionAdapter
        key = f"down{block_idx}"
        sample = _apply_temporal(sample, temporal_kernels[key], device, N, C)

# mid block
for i in range(N):
    sample[i] = mid_block(sample[i], emb, encoder_hidden_states, ...)
sample = _apply_temporal(sample, temporal_kernels["mid"], device, N, 1280)

# up blocks (block_out_channels reversed: 1280, 1280, 640, 320)
reversed_channels = list(reversed(block_out_channels))   # [1280, 1280, 640, 320]
for block_idx, up_block in enumerate(ttnn_model.up_blocks):
    C = reversed_channels[block_idx]
    res_tuple = [down_res_samples[i][-layers_per_block-1:] for i in range(N)]
    for i in range(N):
        sample[i] = up_block(sample[i], emb, res_tuple[i], encoder_hidden_states, ...)
    if block_idx in {0, 1, 2}:   # CrossAttn blocks (up[3] is plain UpBlock)
        up_key_idx = block_idx    # up0=1280, up1=640, up2=320 → keys up0/up1/up2
        sample = _apply_temporal(sample, temporal_kernels[f"up{block_idx}"], device, N, C)

# conv_out: per frame
for i in range(N):
    sample[i] = conv_out(sample[i])

return sample
```

### `_apply_temporal()`

The bridge between TTNN device tensors and `TemporalAttentionKernel`:

```python
def _apply_temporal(samples, kernel, device, num_frames, C):
    """Pull N TTNN hidden states to CPU, apply temporal attention, push back.

    samples: list of N TTNN tensors, each [2*B, S, C] or [2*B, H, W, C] (NHWC layout)
    Returns: list of N TTNN tensors with same shape.
    """
    import ttnn, torch
    from animatediff_ttnn.ttnn_pipeline import to_device, from_device

    # 1. To CPU — one ttnn.to_torch per frame
    cpu_tensors = [ttnn.to_torch(s).float() for s in samples]

    # 2. The TTNN UNet uses batch=2 (CFG doubled). Split uncond/cond, attend each.
    # Each tensor shape: [2, seq_len, C] where seq_len = S (spatial positions flattened)
    # We attend across the N frames dimension while keeping B=2 (uncond/cond) separate.

    results = []
    for b in range(2):   # uncond, then cond
        # Collect [S, C] for each frame at this CFG branch
        feats = torch.stack([t[b] for t in cpu_tensors], dim=1)  # [S, N, C]
        attended = kernel.forward(feats)                           # [S, N, C]
        results.append(attended)

    # 3. Reconstruct per-frame TTNN tensors
    out = []
    for i in range(num_frames):
        # Stack uncond[i] and cond[i] back to [2, S, C]
        frame_cpu = torch.stack([results[0][:, i, :], results[1][:, i, :]], dim=0)
        out.append(to_device(frame_cpu, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT))
        # Deallocate original
        samples[i].deallocate(True)

    return out
```

---

## Component 3: `generate_frames_motion()` in `temporal_attention.py`

Same signature as `generate_frames_temporal()` plus one new parameter:

```python
def generate_frames_motion(
    device,
    ttnn_model,
    ttnn_vae,
    config,
    torch_time_proj,
    text_embeddings,
    temporal_kernels: dict,         # from motion_weights.load_motion_kernels()
    num_frames: int = 8,
    num_steps: int = 25,
    guidance_scale: float = 7.5,
    seed: int = 42,
    height: int = 512,
    width: int = 512,
    use_lightning: bool = False,
    chain_from: str | None = None,
    chain_save: str | None = None,
    chain_alpha: float = 0.6,
    on_step=None,
) -> list:
```

**Denoising loop change**: replaces the per-frame sequential TTNN calls + `cross_frame_attention()` with a single `forward_unet_staged()` call per step that processes all N frames and applies temporal attention internally. The scheduler step logic, chain_from/chain_save blending, and VAE decode remain identical.

**Chain mode benefit**: Because temporal attention now runs at every UNet block depth
(320→640→1280→1280 down, 1280→1280→640→320 up), the final denoised latents saved by
`--chain-save` carry richer structural information than Phase 2.5 produced. The next
chained run inherits better compositional continuity at no extra cost.

---

## Component 4: `generate.py` — `--motion-adapter` flag

```
--motion-adapter [PATH]   Load MotionAdapter weights for full temporal attention.
                          PATH defaults to HuggingFace cache for
                          guoyww/animatediff-motion-adapter-v1-5-2.
                          Enables generate_frames_motion() instead of
                          generate_frames_temporal().
```

When `--motion-adapter` is set, the generate script:
1. Calls `load_motion_kernels(model_id)` at startup (one-time, ~2s)
2. Passes `temporal_kernels` to `generate_frames_motion()`
3. Adds `[motion]` tag to progress output

---

## Injection points and spatial shapes

The TTNN UNet operates in NHWC layout. Hidden states at each injection point:

| Key | C | Spatial (H×W) | S = H*W | TTNN shape (batch=2) |
|---|---|---|---|---|
| down0 | 320 | 64×64 | 4096 | [2, 4096, 320] |
| down1 | 640 | 32×32 | 1024 | [2, 1024, 640] |
| down2 | 1280 | 16×16 | 256 | [2, 256, 1280] |
| mid | 1280 | 8×8 | 64 | [2, 64, 1280] |
| up0 | 1280 | 16×16 | 256 | [2, 256, 1280] |
| up1 | 640 | 32×32 | 1024 | [2, 1024, 640] |
| up2 | 320 | 64×64 | 4096 | [2, 4096, 320] |

**Tile alignment**: All S values (4096, 1024, 256, 64) are multiples of TILE_SIZE=32.
All C values (320, 640, 1280) are multiples of 32. `TemporalAttentionKernel` handles all
of these without modification — S acts as `s_rows`, N_TILES=1 (8 frames in one tile).

---

## Performance estimate

Per denoising step (N=8 frames, 7 injection points):
- 7 × N = 56 `ttnn.to_torch` calls (hidden state → CPU)
- 7 temporal attention forward passes on CPU (tiny: [S, 8, C] matmuls)
- 7 × N = 56 `to_device` calls (CPU → device)
- Estimated overhead: ~20–35s per run on top of current ~65s

Total for a 25-step run: ~85–100s vs ~65s (Phase 2.5). Quality improvement justifies this.
Lightning mode (8 steps) narrows the gap: ~75s vs ~65s.

---

## Testing plan

1. **`test_motion_weights.py`**: `load_motion_kernels()` returns 7 kernels with correct
   dims, all weights loaded (no None).

2. **`test_forward_unet_staged.py`**: Smoke test that `forward_unet_staged()` returns
   a list of N tensors with shape `[2, 4, lH, lW]` without crashing. Uses real hardware.

3. **`test_apply_temporal.py`**: `_apply_temporal` on synthetic TTNN tensors produces
   output with same shape and different values from input (attention is doing something).

4. **Comparison column `F-motion`** added to `generate_comparison_grid.py`: runs
   `generate.py --motion-adapter` on all 3 prompts, adds column to the HTML viewer.

---

## What is NOT in scope

- Using `attn2` (cross-attention to text) from MotionAdapter — only `attn1` (temporal
  self-attention) is used. `attn2` conditions on text; we use the existing TTNN spatial
  cross-attention for that.
- LayerNorm / feedforward from MotionAdapter transformer blocks — `TemporalAttentionKernel`
  handles only the QKV attention, which is the dominant temporal coherence mechanism.
- `proj_in` / `proj_out` from MotionAdapter — these are no-ops when the input and output
  channel dimensions match (identity weight init), which they do at all injection points.
- Multi-chip sharding of temporal attention — remains single-CPU for now.
- `use_ttlang=True` path in `TemporalAttentionKernel` — use the PyTorch path; the TT-Lang
  sim path is Phase 4.

---

## File summary

| File | Status | Changes |
|---|---|---|
| `animatediff_ttnn/motion_weights.py` | Create | `load_motion_kernels()` |
| `animatediff_ttnn/ttnn_motion_pipeline.py` | Create | `forward_unet_staged()`, `_apply_temporal()` |
| `animatediff_ttnn/temporal_attention.py` | Modify | Add `generate_frames_motion()` |
| `examples/generate.py` | Modify | Add `--motion-adapter` flag |
| `tests/test_motion_weights.py` | Create | Weight loading tests |
| `tests/test_ttnn_motion_pipeline.py` | Create | Staged forward smoke test |
| `animatediff_ttnn/ttlang/temporal_attention_kernel.py` | Unchanged | |
| `animatediff_ttnn/ttnn_pipeline.py` | Unchanged | |
| tt-metal source files | Unchanged | Zero modifications |
