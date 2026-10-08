---

# LoRA Support in tt-animatediff

**Status:** ✅ Implementation complete

**Date:** 2026-10-07

## Overview

This document describes how LoRA (Low-Rank Adaptation) support was implemented in
tt-animatediff to enable more creative control through fine-tuned models.

---

## What is LoRA?

LoRA (Low-Rank Adaptation) is a technique for efficiently fine-tuning large models
by injecting low-rank update matrices into target layers. Instead of training all
weights, only small adapter matrices are trained, then merged back into the base
model.

### Key advantages for video generation:

1. **Fast adaptation** — Train new styles/poses in hours instead of days
2. **Small artifacts** — Adapter weights are ~1-10 MB vs 2-7 GB base model
3. **Modular** — Swap adapters without re-downloading the base model
4. **Memory efficient** — Only the low-rank matrices need to be loaded for inference

---

## Current tt-animatediff Architecture

tt-animatediff has two main execution paths:

### Phase 1 (CPU)
- Uses `diffusers.AnimateDiffPipeline` with `MotionAdapter`
- Base model: `CompVis/stable-diffusion-v1-4`
- Motion weights: `guoyww/animatediff-motion-adapter-v1-5-2` (567 keys, 320-dim features)
- All processing on CPU
- Uses `AnimateDiffTransformer3D` blocks at each injection point

### Phase 2/2.5 (Blackhole)
- TTNN UNet2D accelerated spatial denoising
- Base model: `CompVis/stable-diffusion-v1-4` TTNN UNet
- Temporal coherence via cross-frame attention blend (`temporal_alpha`)
- Fast but limited to the base model's capabilities

### Phase 3 (Blackhole + MotionAdapter)
- Full AnimateDiff Transformer3D injected at 7 UNet points
- CPU-side temporal processing with batched transfers
- Best quality but slower (~7-52 s/frame depending on configuration)

---

## Hardware Considerations

### Where LoRA Runs

LoRA adapters are **CPU-bound** during the merge phase in tt-animatediff, but this has minimal impact:

| Component | Hardware | Notes |
|-----------|----------|-------|
| **TTNN UNet** | Blackhole | Spatial denoising (main acceleration) |
| **LoRA matrices** | CPU | Small (~1-10 MB), fast to load |
| **LoRA merge** | CPU | One-time per adapter, cached |
| **Temporal blend** | CPU | Tiny overhead (~1-3%) |

### Why CPU is Acceptable

1. **LoRA adapters are tiny** — ~1-10 MB vs base model ~2 GB
2. **Merge is one-time** — Done at load time, not per-step
3. **Cache reuse** — Merged adapter stored for subsequent generations
4. **Minimal overhead** — CPU transfer of 4-channel 64×64 tensors is negligible

### Hybrid Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    tt-animatediff                       │
├─────────────────────┬───────────────────────────────────┤
│   Blackhole (GPU)   │        CPU                            │
│                     │                                     │
│  • TTNN UNet2D      │  • Base MotionAdapter (loaded once) │
│  • Spatial denoise  │  • LoRA merge (cached)              │
│  • ~1.94 s/frame    │  • Temporal blend (α=0.35)            │
│                     │  • LoRA matrices (1-10 MB)            │
└─────────────────────┴───────────────────────────────────┘
          │                           │
          └───────────┬───────────────┘
                      ▼
              (merged, then on Blackhole)
```

### Performance Impact

| Operation | Latency | Notes |
|-----------|---------|-------|
| Base TTNN UNet | ~1.94 s/frame | Main acceleration |
| LoRA load/merge | ~100-500 ms | One-time, cached |
| Temporal blend | ~1-5 ms | Per-step, negligible |

**Net effect:** <1% overhead on top of existing ~1.94 s/frame baseline.

---

## Implementation Status

The LoRA implementation is complete with the following components:

### Core Module: `animatediff_ttnn/lora.py`

**Location:** `animatediff_ttnn/lora.py`

**Functions:**
- `load_lora_adapter(base_adapter_id, lora_adapter_id, lora_alpha=1.0, device="cpu")`
- `merge_and_cache(base_adapter_id, lora_adapter_id, output_path=None, lora_alpha=1.0)`
- `get_cached_adapters()`
- `clear_cache()`
- `get_peft_config(r=16, lora_alpha=32, target_modules=None, lora_dropout=0.0, bias="none")`

### Tests: `tests/test_lora.py`

**Location:** `tests/test_lora.py`

**Tests:**
- Module loading tests (6 tests)
- PEFT config tests (6 tests)
- Cache directory tests (5 tests)

### Documentation: `lora/README.md`

**Location:** `lora/README.md`

**Sections:**
- Usage examples
- Available functions
- Use cases
- Performance characteristics

### Design Document: `LORA_DESIGN.md`

This document — describes the implementation approach and decisions.

---

## How LoRA Is Integrated

### Option 1: LoRA for MotionAdapters (Implemented)

Apply LoRA to the MotionAdapter itself, creating style-specific or pose-specific
motion adapters that can be merged at runtime.

#### Technical Details

The MotionAdapter contains `AnimateDiffTransformer3D` blocks at 7 injection points:
- `down_blocks.0.motion_modules.0` (first downblock)
- `down_blocks.0.motion_modules.1` (second downblock)
- `down_blocks.0.motion_modules.2` (third downblock)
- `mid_block.motion_modules.0` (mid block)
- `up_blocks.0.motion_modules.0` (first upblock)
- `up_blocks.0.motion_modules.1` (second upblock)
- `up_blocks.0.motion_modules.2` (third upblock)

Each `AnimateDiffTransformer3D` contains transformer blocks with attention layers
(`to_q`, `to_k`, `to_v`, `to_out`) that are ideal targets for LoRA injection.

A typical LoRA config for MotionAdapter:
```python
from peft import LoraConfig

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.0,
    bias="none",
)
```

Adapter size: ~1-10 MB vs base ~2 GB (99%+ reduction)

#### Implementation

```python
from animatediff_ttnn import generate_animation
from animatediff_ttnn.lora import load_lora_adapter

# Load a LoRA adapter (e.g., anime style)
adapter = load_lora_adapter(
    base_adapter_id="guoyww/animatediff-motion-adapter-v1-5-2",
    lora_adapter_id="my-org/anime-motion-lora"
)

frames = generate_animation(
    prompt="a girl running in anime style",
    motion_adapter=adapter
)
```

#### Use cases:

- **Style adapters** — Anime, oil painting, claymation, stop-motion
- **Motion adapters** — Slow drift, fast action, smooth pan, jittery camera
- **Pose adapters** — Specific body poses, dance moves, facial expressions

#### Cache Location:

Merged adapters are stored in:
```
~/.cache/tt-animatediff/lora/
```

This can be overridden with the `LORA_CACHE_DIR` environment variable.

---

## API Reference

### `load_lora_adapter()`

```python
from animatediff_ttnn.lora import load_lora_adapter

adapter = load_lora_adapter(
    base_adapter_id="guoyww/animatediff-motion-adapter-v1-5-2",
    lora_adapter_id="my-org/anime-motion-lora",
    lora_alpha=1.0,
    device="cpu"
)
```

**Parameters:**
- `base_adapter_id`: HuggingFace repo ID for the base MotionAdapter
- `lora_adapter_id`: HuggingFace repo ID or local path for the LoRA adapter
- `lora_alpha`: LoRA alpha parameter (scaling factor, default: 1.0)
- `device`: Device to load adapters on (default: "cpu")

**Returns:** Merged `MotionAdapter` ready for use

**Example:**
```python
from animatediff_ttnn import generate_animation
from animatediff_ttnn.lora import load_lora_adapter

adapter = load_lora_adapter(
    base_adapter_id="guoyww/animatediff-motion-adapter-v1-5-2",
    lora_adapter_id="my-org/anime-motion-lora"
)

frames = generate_animation(
    prompt="a girl running in anime style",
    motion_adapter=adapter
)
```

### `merge_and_cache()`

```python
from animatediff_ttnn.lora import merge_and_cache

merged_path = merge_and_cache(
    base_adapter_id="guoyww/animatediff-motion-adapter-v1-5-2",
    lora_adapter_id="my-org/anime-motion-lora",
    output_path="/custom/path"
)
```

**Parameters:**
- `base_adapter_id`: HuggingFace repo ID for the base MotionAdapter
- `lora_adapter_id`: HuggingFace repo ID or local path for the LoRA adapter
- `output_path`: Optional custom output path (defaults to cache directory)
- `lora_alpha`: LoRA alpha parameter (scaling factor, default: 1.0)

**Returns:** `Path` to the merged adapter

### `get_peft_config()`

```python
from animatediff_ttnn.lora import get_peft_config

config = get_peft_config(
    r=16,
    lora_alpha=32,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.0,
    bias="none"
)
```

**Parameters:**
- `r`: LoRA rank (default: 16)
- `lora_alpha`: LoRA alpha parameter (default: 32)
- `target_modules`: List of modules to apply LoRA to
- `lora_dropout`: Dropout probability (default: 0.0)
- `bias`: Bias handling ("none", "all", or "lora_only", default: "none")

**Returns:** `peft.LoraConfig` for use with `get_peft_model()`

**Example for training:**
```python
from animatediff_ttnn.lora import get_peft_config, merge_and_cache
from diffusers import MotionAdapter
from peft import get_peft_model

config = get_peft_config(r=16, lora_alpha=32)
base = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
lora = get_peft_model(base, config)
# Train lora on your dataset...
lora.save_pretrained("my-lora-adapter")
```

### `get_cached_adapters()`

```python
from animatediff_ttnn.lora import get_cached_adapters

adapters = get_cached_adapters()  # Returns list of adapter names
```

**Returns:** List of cached adapter names (cache keys)

### `clear_cache()`

```python
from animatediff_ttnn.lora import clear_cache

clear_cache()  # Removes all cached adapters
```

---

## Testing

Run the LoRA tests with:

```bash
pytest tests/test_lora.py -v
```

All 17 tests pass.

---

## Future Enhancements

- [ ] Multi-adapter blending (combine 2+ adapters)
- [ ] Real-time adapter switching
- [ ] Adapter mixing ratios
- [ ] Adapter training from user data
- [ ] Cloud-based adapter sharing

## See Also

- `lora/README.md` — User-facing documentation
- `tests/test_lora.py` — Unit tests
- `examples/lora_examples.py` — Example code
