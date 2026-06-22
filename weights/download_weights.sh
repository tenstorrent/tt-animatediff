#!/usr/bin/env bash
# Download tt-animatediff LCM distilled weights from Hugging Face Hub.
# Also downloads the AnimateDiff motion module needed for baseline inference.
set -e

HF_REPO="tenstorrent/tt-animatediff-lcm"
WEIGHTS_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "╔══════════════════════════════════════════════════════════════"
echo "║  tt-animatediff weight downloader"
echo "╚══════════════════════════════════════════════════════════════"
echo ""
echo "Destination: $WEIGHTS_DIR"
echo ""

if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "❌ huggingface_hub not found. Install with: pip install huggingface_hub"
    exit 1
fi

_download() {
    local filename="$1"
    local repo="$2"
    local dest="$WEIGHTS_DIR/$filename"
    if [ -f "$dest" ]; then
        echo "  ✓ $filename (already exists, skipping)"
        return
    fi
    echo "  ↓ $filename ..."
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('$repo', '$filename', local_dir='$WEIGHTS_DIR')
"
    echo "  ✓ $filename"
}

echo "LCM distilled weights:"
_download unet_lcm_4step.pt         "$HF_REPO"
_download unet_lcm_8step.pt         "$HF_REPO"
_download motion_adapter_lcm_4step.pt  "$HF_REPO"
_download motion_adapter_lcm_8step.pt  "$HF_REPO"

echo ""
echo "AnimateDiff baseline motion module:"
_download mm_sd_v15_v2.ckpt "guoyww/animatediff"

echo ""
echo "╔══════════════════════════════════════════════════════════════"
echo "║  All weights ready in $WEIGHTS_DIR"
echo "╚══════════════════════════════════════════════════════════════"
echo ""
echo "Run baseline (25-step):"
echo "  python examples/generate.py --mode blackhole --steps 25 --frames 8 \\"
echo "    --prompt \"your prompt here\""
echo ""
echo "Run Lightning (8-step):"
echo "  python examples/generate.py --mode blackhole --steps 8 --frames 8 \\"
echo "    --lcm-unet weights/unet_lcm_8step.pt \\"
echo "    --lcm-adapter weights/motion_adapter_lcm_8step.pt \\"
echo "    --prompt \"your prompt here\""
