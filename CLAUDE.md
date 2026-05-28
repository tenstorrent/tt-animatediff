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
