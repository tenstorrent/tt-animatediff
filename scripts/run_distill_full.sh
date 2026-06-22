#!/usr/bin/env bash
# Full distillation pipeline — runs all 4 phases sequentially.
# Phase 1a: UNet 4-step  (resume from 1000-step checkpoint)
# Phase 1b: UNet 8-step  (fresh from teacher)
# Phase 2a: MotionAdapter 4-step (requires unet_lcm_4step.pt)
# Phase 2b: MotionAdapter 8-step (requires unet_lcm_8step.pt)
#
# Usage: bash scripts/run_distill_full.sh 2>&1 | tee /tmp/distill.log

set -e
cd "$(dirname "$0")/.."

echo "============================================================"
echo " Phase 1a: UNet 4-step (resuming from 1000-step checkpoint)"
echo "============================================================"
python scripts/distill_lcm.py \
    --steps 4 \
    --num_train_steps 5000 \
    --resume weights/unet_lcm_4step.pt

echo "============================================================"
echo " Phase 1b: UNet 8-step (fresh start)"
echo "============================================================"
python scripts/distill_lcm.py \
    --steps 8 \
    --num_train_steps 5000

echo "============================================================"
echo " Phase 2a: MotionAdapter 4-step"
echo "============================================================"
python scripts/distill_motion_adapter.py \
    --steps 4 \
    --unet weights/unet_lcm_4step.pt

echo "============================================================"
echo " Phase 2b: MotionAdapter 8-step"
echo "============================================================"
python scripts/distill_motion_adapter.py \
    --steps 8 \
    --unet weights/unet_lcm_8step.pt

echo "============================================================"
echo " All phases complete."
echo "============================================================"
