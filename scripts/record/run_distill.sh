#!/usr/bin/env bash
# Run both distillation phases sequentially with visible progress.
# Used by distill.tape — all output goes to the terminal for recording.
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

echo ""
echo "╔══════════════════════════════════════════════════════════════"
echo "║  AnimateDiff-Lightning Distillation"
echo "║  Phase 1: LCM UNet (4-step)"
echo "╚══════════════════════════════════════════════════════════════"
python3 scripts/distill_lcm.py --steps 4 --num_train_steps 5000

echo ""
echo "╔══════════════════════════════════════════════════════════════"
echo "║  Phase 1: LCM UNet (8-step)"
echo "╚══════════════════════════════════════════════════════════════"
python3 scripts/distill_lcm.py --steps 8 --num_train_steps 5000

echo ""
echo "╔══════════════════════════════════════════════════════════════"
echo "║  Phase 2: MotionAdapter (8-step)"
echo "╚══════════════════════════════════════════════════════════════"
python3 scripts/distill_motion_adapter.py --steps 8 --unet weights/unet_lcm_8step.pt

echo ""
echo "╔══════════════════════════════════════════════════════════════"
echo "║  Phase 2: MotionAdapter (4-step)"
echo "╚══════════════════════════════════════════════════════════════"
python3 scripts/distill_motion_adapter.py --steps 4 --unet weights/unet_lcm_4step.pt

echo ""
echo "╔══════════════════════════════════════════════════════════════"
echo "║  Validation: launching 4 chips in parallel"
echo "╚══════════════════════════════════════════════════════════════"
python3 scripts/validate_parallel.py
