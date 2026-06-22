#!/usr/bin/env bash
# Short inference demo — loads distilled weights, generates one video, prints timing.
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

echo ""
echo "╔══════════════════════════════════════════════════════════════"
echo "║  AnimateDiff-Lightning — Inference Demo"
echo "╚══════════════════════════════════════════════════════════════"
echo ""
echo "Loading lightning-8step model..."
mkdir -p output
time python3 -c "
from animatediff_ttnn.pipeline import create_animatediff_pipeline, generate
import torch, time

pipe = create_animatediff_pipeline()
state = torch.load('weights/unet_lcm_8step.pt', map_location='cpu', weights_only=True)
pipe.unet.load_state_dict(state, strict=False)

t0 = time.perf_counter()
frames = generate(pipe, 'a campfire burning in a dark forest, cinematic',
                  num_inference_steps=8, seed=42)
elapsed = time.perf_counter() - t0
frames[0].save('output/lightning_8step_demo.gif', save_all=True,
               append_images=frames[1:], duration=125, loop=0)
print(f'8-step inference: {elapsed:.1f}s -> output/lightning_8step_demo.gif')
"
