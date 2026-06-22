#!/usr/bin/env bash
# Post-distillation: validate on Blackhole, capture perf notes, then suspend.
# Runs automatically after run_distill.sh completes.
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
LOG="$REPO/generated/validation_$(date +%Y%m%d_%H%M%S).log"
mkdir -p generated

_banner() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════"
    echo "║  $1"
    echo "╚══════════════════════════════════════════════════════════════"
}

exec > >(tee "$LOG") 2>&1

_banner "Post-distillation validation — $(date)"

# Hardware snapshot before validation
_banner "Hardware state (tt-smi -s)"
tt-smi -s 2>/dev/null || echo "tt-smi unavailable"

# Weight inventory
_banner "Weights produced"
ls -lh weights/*.pt weights/*.ckpt 2>/dev/null || echo "No weight files found"

# Validate on TT hardware
_banner "Blackhole inference validation (all 4 chips)"
START=$(date +%s)
python3 scripts/validate_parallel.py --out_dir generated/validation 2>&1
END=$(date +%s)
echo ""
echo "Total validation wall time: $((END - START))s"

# Per-chip GIF sizes as a proxy for successful output
_banner "Output GIFs"
ls -lh generated/validation/*.gif 2>/dev/null || echo "No GIFs produced"

# tt-smi after (chip temps, power draw post-inference)
_banner "Hardware state post-inference"
tt-smi -s 2>/dev/null || echo "tt-smi unavailable"

_banner "Validation complete — log saved to $LOG"
echo "Suspending system in 10 seconds..."
sleep 10
systemctl suspend
