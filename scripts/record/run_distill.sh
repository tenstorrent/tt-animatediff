#!/usr/bin/env bash
# Run both distillation phases sequentially with visible progress.
# Used by distill.tape — all output goes to the terminal for recording.
# Skips any phase whose output weights already exist on disk.
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

_banner() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════"
    echo "║  $1"
    echo "╚══════════════════════════════════════════════════════════════"
}

_skip() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════"
    echo "║  SKIP — $1 already exists"
    echo "╚══════════════════════════════════════════════════════════════"
}

_banner "AnimateDiff-Lightning Distillation"

# Phase 1 — LCM UNet (4-step)
if [ -f weights/unet_lcm_4step.pt ]; then
    _skip "weights/unet_lcm_4step.pt"
else
    _banner "Phase 1: LCM UNet (4-step)"
    python3 scripts/distill_lcm.py --steps 4 --num_train_steps 5000
fi

# Phase 1 — LCM UNet (8-step)
if [ -f weights/unet_lcm_8step.pt ]; then
    _skip "weights/unet_lcm_8step.pt"
else
    _banner "Phase 1: LCM UNet (8-step)"
    python3 scripts/distill_lcm.py --steps 8 --num_train_steps 5000
fi

# Phase 2 — MotionAdapter (8-step)
if [ -f weights/motion_adapter_lcm_8step.pt ]; then
    _skip "weights/motion_adapter_lcm_8step.pt"
else
    _banner "Phase 2: MotionAdapter (8-step)"
    python3 scripts/distill_motion_adapter.py --steps 8 --unet weights/unet_lcm_8step.pt
fi

# Phase 2 — MotionAdapter (4-step)
if [ -f weights/motion_adapter_lcm_4step.pt ]; then
    _skip "weights/motion_adapter_lcm_4step.pt"
else
    _banner "Phase 2: MotionAdapter (4-step)"
    python3 scripts/distill_motion_adapter.py --steps 4 --unet weights/unet_lcm_4step.pt
fi

_banner "All weights done — running post_distill.sh (validate + suspend)"
bash scripts/post_distill.sh
