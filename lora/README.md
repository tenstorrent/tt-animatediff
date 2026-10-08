# LoRA Support for tt-animatediff

This directory contains the LoRA (Low-Rank Adaptation) implementation for
tt-animatediff, enabling style-specific motion adapters.

## What is LoRA?

LoRA is a technique for efficiently fine-tuning large models by injecting
low-rank update matrices. Instead of training all weights (2-7 GB), only small
adapter matrices (~1-10 MB) are trained and merged into the base model.

## Usage

### Basic Example

```python
from animatediff_ttnn import generate_animation
from animatediff_ttnn.lora import load_lora_adapter, merge_and_cache

# Load a LoRA adapter (e.g., anime style)
adapter = load_lora_adapter(
    base_adapter_id="guoyww/animatediff-motion-adapter-v1-5-2",
    lora_adapter_id="my-org/anime-motion-lora"
)

# Generate with LoRA-enhanced adapter
frames = generate_animation(
    prompt="a girl running in anime style",
    motion_adapter=adapter
)
```

### Available Functions

- `load_lora_adapter(base_adapter_id, lora_adapter_id, lora_alpha=1.0, device="cpu")` — Load and merge a LoRA adapter
- `merge_and_cache(base_adapter_id, lora_adapter_id, output_path=None, lora_alpha=1.0)` — Merge and save to cache
- `get_cached_adapters()` — List all cached adapters
- `clear_cache()` — Clear all cached adapters
- `get_peft_config(r=16, lora_alpha=32, target_modules=None, lora_dropout=0.0, bias="none")` — Create PEFT config for training

### Advanced Usage

```python
from animatediff_ttnn.lora import get_peft_config, merge_and_cache

# Create a custom PEFT config for training
config = get_peft_config(r=8, lora_alpha=16, target_modules=["to_q", "to_k"])

# Merge and cache an adapter
merged_path = merge_and_cache(
    base_adapter_id="guoyww/animatediff-motion-adapter-v1-5-2",
    lora_adapter_id="my-org/anime-motion-lora",
    output_path="/custom/path"
)
```

## Use Cases

### 1. Style Transfer
Apply different artistic styles to your videos:
- Anime / Manga
- Oil painting / Impressionism
- Claymation / Stop-motion
- Pixel art
- Sketch / Cartoon

### 2. Motion Control
Control the motion characteristics:
- Fast action (quick cuts, rapid movement)
- Slow drift (gentle, flowing motion)
- Smooth pan (camera movement)
- Jittery camera (handheld style)

### 3. Custom Adaptations
Train or download adapters for specific needs:
- Character-specific poses
- Studio-specific animation style
- Genre-specific motion (e.g., horror suspense)

## Integration Points

### Pipeline Integration

```python
from animatediff_ttnn.pipeline import create_animatediff_pipeline_with_lora

pipe = create_animatediff_pipeline_with_lora(
    lora_adapter_id="my-org/anime-motion-lora"
)
```

### CLI Integration

```bash
python examples/generate.py \
    --prompt "swirling nebula in anime style" \
    --motion-lora anime-motion-lora
```

### Gradio UI Integration

The UI will show a dropdown of available LoRA adapters:

```python
lora_selector = gr.Dropdown(
    choices=get_cached_adapters(),
    value="anime-motion-lora",
    label="Motion LoRA"
)
```

## Performance

### Memory Usage

- Base MotionAdapter: ~2 GB
- LoRA adapter: ~1-10 MB
- Merged adapter: ~2 GB (base + LoRA)

### Inference Speed

- Without LoRA: ~1.94 s/frame (Blackhole, 25 steps)
- With LoRA: ~1.94-2.1 s/frame (Blackhole, 25 steps)

LoRA adds minimal overhead when used as designed (pre-merge into base model).

## Storage

Cached adapters are stored in:
```
~/.cache/tt-animatediff/lora/
```

This can be overridden with the `LORA_CACHE_DIR` environment variable.

## Compatible AnimateDiff LoRAs

The following open-source LoRAs are compatible with tt-animatediff:

### Recommended (3 Verified Styles)

| Repository | Style | Use Case |
|------------|-------|----------|
| `artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5` | Pixel art | Retro 8-bit aesthetics |
| `ilkerzgi/krea-2-chrome-iridescent-vaporwave-lora` | Chrome/Iridescent | Trippy vaporwave styles |
| `neonforestmist/sd15-storybook-anime-lora` | Anime/Storybook | Anime-style animations |

### Motion-Specific (from same creator as MotionAdapter)

- `guoyww/animatediff-motion-lora-rolling-anticlockwise` - Rotation
- `guoyww/animatediff-motion-lora-pan-right` - Pan right
- `guoyww/animatediff-motion-lora-pan-left` - Pan left
- `guoyww/animatediff-motion-lora-tilt-down` - Tilt down
- `guoyww/animatediff-motion-lora-zoom-in` - Zoom in
- `guoyww/animatediff-motion-lora-zoom-out` - Zoom out

For the complete list and details, see `ANIMATEDIFF_LORAS.md`.

## Future Enhancements

- [ ] Multi-adapter blending (combine 2+ adapters)
- [ ] Real-time adapter switching
- [ ] Adapter mixing ratios
- [ ] Adapter training from user data
- [ ] Cloud-based adapter sharing

## See Also

- `LORA_DESIGN.md` — Full design document
- `ANIMATEDIFF_LORAS.md` — List of compatible AnimateDiff LoRAs
- `examples/lora_examples.py` — Example code
- `animatediff_ttnn/lora.py` — Implementation
