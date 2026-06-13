# tt-animatediff — project notes for Claude

## What this project is
Canonical implementation of AnimateDiff on Tenstorrent Blackhole hardware.
TTNN UNet denoising on Blackhole; VAE decode on CPU; cross-frame temporal
attention in Phase 2.5. Exported by copy to:
  - ~/code/tt-vscode-toolkit/content/projects/animatediff/
  - ~/code/tt-local-generator/app/animatediff/ (via _BUNDLED_DIR in animatediff.py)

## Export workflow (until this is a public vendorable repo)
After changes here, run:
  rsync -a --exclude="__pycache__" --exclude="*.pyc" --exclude="*.egg-info" \
    ~/code/tt-animatediff/ ~/code/tt-vscode-toolkit/content/projects/animatediff/
  rsync -a --exclude="__pycache__" --exclude="*.pyc" --exclude="*.egg-info" \
    ~/code/tt-animatediff/ ~/code/tt-local-generator/app/animatediff/
Then commit in each downstream repo separately.

## Architecture phases
- Phase 1 (generate_baseline.py): CPU AnimateDiff with MotionAdapter
- Phase 2 (generate_blackhole.py): TTNN UNet on Blackhole, sequential frames
- Phase 2.5 (generate_blackhole_v2.py): TTNN UNet + cross-frame temporal attention

## Key known issues
- ARC firmware hang on chip 3 (P300C board 0000046131924055) — see ~/qb2-debug/
  setup_blackhole() now reads hwmon sentinel values to skip dead chips
- ttnn_pipeline.py uses open_mesh_device (not open_device) — all chips claimed upfront
- VAE decode must stay on CPU: TTNN VAE conv_out OOMs on Blackhole L1 grid

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

### LCM distillation (closed)
All four distillation runs failed (flat LR without warmup on sharp loss landscape).
Broken weights archived as `weights/*.broken`. Distillation track is closed.
