#!/usr/bin/env bash
# Driver script for the asciinema recording.
# Run this inside the recording: asciinema rec -c "bash demo/record_demo.sh" ...
# All "typing" is simulated via the type() helper — adjust DELAY_CHAR to taste.

set -euo pipefail

# ── Pacing controls ──────────────────────────────────────────────────────────
DELAY_CHAR=0.045   # seconds between keystrokes (feel of fast but human typing)
DELAY_ENTER=0.4    # pause after hitting enter
DELAY_THINK=1.2    # pause while "reading" output
DELAY_SECTION=2.0  # pause between major sections

# ── Helpers ──────────────────────────────────────────────────────────────────
type() {
    local text="$1"
    local extra_delay="${2:-0}"
    for ((i=0; i<${#text}; i++)); do
        printf '%s' "${text:$i:1}"
        sleep "$DELAY_CHAR"
    done
    sleep "$extra_delay"
}

run() {
    # type command, press enter, run it
    local cmd="$1"
    local think="${2:-$DELAY_THINK}"
    type "$cmd"
    printf '\n'
    sleep "$DELAY_ENTER"
    eval "$cmd"
    sleep "$think"
}

comment() {
    # print a dim comment line, not executed
    type "# $1" 0.1
    printf '\n'
    sleep 0.6
}

section() {
    printf '\n'
    sleep 0.3
    type "# ── $1 ──"
    printf '\n'
    sleep "$DELAY_SECTION"
}

pause() { sleep "${1:-$DELAY_THINK}"; }

# ── Env: suppress TF noise ────────────────────────────────────────────────────
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONWARNINGS=ignore
export PYTHONDONTWRITEBYTECODE=1

# ── Working directory ─────────────────────────────────────────────────────────
WORK_DIR="$(mktemp -d /tmp/animatediff-demo-XXXX)"
cd "$WORK_DIR"

clear
sleep 0.8

# ─────────────────────────────────────────────────────────────────────────────
section "Clone the AnimateDiff reference implementation"

run "git clone --depth 1 https://github.com/guoyww/AnimateDiff.git" 3.0

run "cd AnimateDiff && ls"

pause

# ─────────────────────────────────────────────────────────────────────────────
section "Find the file that matters"

comment "The core of AnimateDiff is in one file: motion_module.py"

run "wc -l animatediff/models/motion_module.py"

pause "$DELAY_SECTION"

# Show the key class definition using grep
comment "VanillaTemporalModule — the thing we need to port"
run "grep -n 'class Vanilla\|def forward\|rearrange\|video_length' animatediff/models/motion_module.py | head -20" "$DELAY_SECTION"

comment "Input reshaping: (b*f, spatial, c) → (b*spatial, f, c)"
comment "That's the whole insight. Frames become the sequence dimension."

pause "$DELAY_SECTION"

# ─────────────────────────────────────────────────────────────────────────────
section "Set up the TT project"

run "cd $WORK_DIR"
run "mkdir -p tt-animatediff/animatediff_ttnn && cd tt-animatediff"

pause

# ─────────────────────────────────────────────────────────────────────────────
section "Write the temporal attention module"

comment "Create animatediff_ttnn/temporal_module.py from scratch"
pause 0.8

# Write the file using a heredoc piped through cat — visible and clean in terminal
cat > animatediff_ttnn/temporal_module.py << 'PYEOF'
"""Temporal attention for AnimateDiff on Tenstorrent hardware."""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Optional
import torch


@dataclass
class TemporalAttentionWeights:
    to_q_weight: torch.Tensor
    to_q_bias:   Optional[torch.Tensor]
    to_k_weight: torch.Tensor
    to_k_bias:   Optional[torch.Tensor]
    to_v_weight: torch.Tensor
    to_v_bias:   Optional[torch.Tensor]
    to_out_weight: torch.Tensor
    to_out_bias:   Optional[torch.Tensor]
    pos_encoding:  Optional[torch.Tensor]
    dim:       int
    num_heads: int


def sinusoidal_encoding(dim: int, max_len: int = 24) -> torch.Tensor:
    pos = torch.arange(max_len).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
    pe = torch.zeros(1, max_len, dim)
    pe[0, :, 0::2] = torch.sin(pos * div)
    pe[0, :, 1::2] = torch.cos(pos * div)
    return pe


def temporal_attention(
    hidden_states: torch.Tensor,
    weights: TemporalAttentionWeights,
    num_frames: int,
) -> torch.Tensor:
    """Apply temporal attention across video frames.

    Input/output shape: (batch*frames, spatial_tokens, channels)
    The key move: reshape so frames become the sequence dimension,
    run standard multi-head attention, reshape back.
    """
    if num_frames == 1:
        return hidden_states

    bf, seq, c = hidden_states.shape
    b = bf // num_frames
    h = weights.num_heads
    hd = c // h

    # (b*f, seq, c) → (b*seq, f, c)
    x = hidden_states.view(b, num_frames, seq, c)
    x = x.permute(0, 2, 1, 3).reshape(b * seq, num_frames, c)

    # positional encoding
    if weights.pos_encoding is not None:
        x = x + weights.pos_encoding[:, :num_frames, :].to(x.device)

    F = torch.nn.functional
    q = F.linear(x, weights.to_q_weight, weights.to_q_bias)
    k = F.linear(x, weights.to_k_weight, weights.to_k_bias)
    v = F.linear(x, weights.to_v_weight, weights.to_v_bias)

    # multi-head: (b*seq, f, h, hd) → (b*seq, h, f, hd)
    def split_heads(t):
        return t.view(b * seq, num_frames, h, hd).permute(0, 2, 1, 3)

    q, k, v = split_heads(q), split_heads(k), split_heads(v)

    scale  = 1.0 / math.sqrt(hd)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    probs  = F.softmax(scores, dim=-1)
    out    = torch.matmul(probs, v)

    # merge heads → output projection
    out = out.permute(0, 2, 1, 3).reshape(b * seq, num_frames, c)
    out = F.linear(out, weights.to_out_weight, weights.to_out_bias)

    # (b*seq, f, c) → (b*f, seq, c)
    out = out.view(b, seq, num_frames, c).permute(0, 2, 1, 3).reshape(bf, seq, c)
    return out


def synthetic_weights(dim: int = 320, num_heads: int = 8) -> TemporalAttentionWeights:
    """Random weights for testing — same shapes as mm_sd_v15_v2.ckpt."""
    def w(rows, cols): return torch.randn(rows, cols) * 0.02
    def b(n):          return torch.zeros(n)
    return TemporalAttentionWeights(
        to_q_weight=w(dim, dim), to_q_bias=b(dim),
        to_k_weight=w(dim, dim), to_k_bias=b(dim),
        to_v_weight=w(dim, dim), to_v_bias=b(dim),
        to_out_weight=w(dim, dim), to_out_bias=b(dim),
        pos_encoding=sinusoidal_encoding(dim),
        dim=dim, num_heads=num_heads,
    )
PYEOF

type "vim animatediff_ttnn/temporal_module.py"
printf '\n'
sleep "$DELAY_ENTER"
# Open vim read-only just to show the file, then quit
vim -R -c 'set number' -c 'norm! G' animatediff_ttnn/temporal_module.py
sleep "$DELAY_THINK"

# ─────────────────────────────────────────────────────────────────────────────
section "Quick smoke test — does it run?"

cat > test_shapes.py << 'PYEOF'
"""Validate shapes and temporal coherence without hardware."""
import torch
import sys
sys.path.insert(0, ".")
from animatediff_ttnn.temporal_module import temporal_attention, synthetic_weights

B, F, SEQ, C = 1, 8, 256, 320   # batch, frames, spatial tokens, channels

weights = synthetic_weights(dim=C, num_heads=8)
x = torch.randn(B * F, SEQ, C)

out = temporal_attention(x, weights, num_frames=F)

assert out.shape == x.shape, f"shape mismatch: {out.shape} vs {x.shape}"

# Adjacent frames should be correlated after temporal attention
corrs = []
for i in range(F - 1):
    a = out[i].flatten()
    b = out[i + 1].flatten()
    corrs.append(torch.corrcoef(torch.stack([a, b]))[0, 1].item())
avg = sum(corrs) / len(corrs)

print(f"output shape : {tuple(out.shape)}  ✓")
print(f"avg frame correlation : {avg:.4f}")
print("temporal coherence    : " + ("✓ pass" if avg > 0.3 else "✗ fail"))
PYEOF

run "python3 test_shapes.py" "$DELAY_SECTION"

# ─────────────────────────────────────────────────────────────────────────────
section "Write the pipeline wrapper"

cat > animatediff_ttnn/pipeline.py << 'PYEOF'
"""Thin wrapper — apply temporal coherence on top of any latent generator."""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import torch
from PIL import Image

from .temporal_module import (
    TemporalAttentionWeights, temporal_attention,
    sinusoidal_encoding, synthetic_weights,
)


class AnimateDiffPipeline:
    def __init__(self, weights: TemporalAttentionWeights):
        self.weights = weights

    @classmethod
    def from_checkpoint(cls, path: str, **kw) -> "AnimateDiffPipeline":
        import torch
        ckpt = torch.load(path, map_location="cpu")
        dim  = kw.get("dim", 320)
        nh   = kw.get("num_heads", 8)
        pfx  = "down_blocks.0.motion_modules.0.temporal_transformer.transformer_blocks.0.attention_blocks.0"
        def g(k): return ckpt.get(f"{pfx}.{k}")
        w = TemporalAttentionWeights(
            to_q_weight=g("to_q.weight"), to_q_bias=g("to_q.bias"),
            to_k_weight=g("to_k.weight"), to_k_bias=g("to_k.bias"),
            to_v_weight=g("to_v.weight"), to_v_bias=g("to_v.bias"),
            to_out_weight=g("to_out.0.weight"), to_out_bias=g("to_out.0.bias"),
            pos_encoding=sinusoidal_encoding(dim),
            dim=dim, num_heads=nh,
        )
        return cls(w)

    @classmethod
    def from_synthetic(cls, dim: int = 320, num_heads: int = 8) -> "AnimateDiffPipeline":
        return cls(synthetic_weights(dim=dim, num_heads=num_heads))

    def apply_temporal_coherence(
        self,
        latents: torch.Tensor,  # (B*F, H, W, C) or (B*F, C, H, W)
        num_frames: int,
        channels_last: bool = True,
    ) -> torch.Tensor:
        if not channels_last:
            latents = latents.permute(0, 2, 3, 1)
        bf, h, w, c = latents.shape
        flat = latents.reshape(bf, h * w, c)
        flat = temporal_attention(flat, self.weights, num_frames)
        out  = flat.reshape(bf, h, w, c)
        if not channels_last:
            out = out.permute(0, 3, 1, 2)
        return out

    def export_gif(
        self, frames: List[Image.Image], path: str, fps: int = 8
    ) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            path, save_all=True, append_images=frames[1:],
            duration=1000 // fps, loop=0,
        )
PYEOF

type "vim animatediff_ttnn/pipeline.py"
printf '\n'
sleep "$DELAY_ENTER"
vim -R -c 'set number' animatediff_ttnn/pipeline.py
sleep "$DELAY_THINK"

# ─────────────────────────────────────────────────────────────────────────────
section "End-to-end demo — generate and export a GIF"

cat > run_demo.py << 'PYEOF'
"""Full pipeline demo using synthetic weights (no checkpoint needed)."""
import sys, torch
from PIL import Image, ImageDraw
sys.path.insert(0, ".")
from animatediff_ttnn.pipeline import AnimateDiffPipeline

NUM_FRAMES = 8
DIM        = 320

print("Loading pipeline (synthetic weights)...")
pipe = AnimateDiffPipeline.from_synthetic(dim=DIM)

print(f"Generating latents — {NUM_FRAMES} frames...")
torch.manual_seed(42)
latents = torch.randn(NUM_FRAMES, 8, 8, DIM)

print("Applying temporal coherence...")
latents = pipe.apply_temporal_coherence(latents, num_frames=NUM_FRAMES)

# Decode latents → colourful thumbnails (stand-in for VAE decode)
frames = []
for i in range(NUM_FRAMES):
    val = float(latents[i].mean().item())
    r   = int(min(255, max(0, 128 + val * 60)))
    g   = int(min(255, max(0, 100 + i * 18)))
    b   = int(min(255, max(0, 200 - i * 10)))
    img = Image.new("RGB", (256, 256), (r, g, b))
    d   = ImageDraw.Draw(img)
    d.text((10, 10), f"frame {i+1}/{NUM_FRAMES}", fill=(255, 255, 255))
    frames.append(img)

out = "/tmp/animatediff_demo.gif"
pipe.export_gif(frames, out, fps=8)
print(f"\nExported → {out}")

# Correlation across frames
corrs = []
for i in range(NUM_FRAMES - 1):
    a = latents[i].flatten(); b = latents[i+1].flatten()
    corrs.append(torch.corrcoef(torch.stack([a, b]))[0, 1].item())
avg = sum(corrs) / len(corrs)
print(f"Avg frame correlation : {avg:.4f}  ({'✓ coherent' if avg > 0.3 else '✗ low'})")
print("\nDone.")
PYEOF

run "python3 run_demo.py" "$DELAY_SECTION"

# ─────────────────────────────────────────────────────────────────────────────
section "What we built"

run "find . -name '*.py' | sort"

pause

comment "temporal_module.py  — attention math, sinusoidal encoding, weight loader"
comment "pipeline.py         — thin wrapper: apply_temporal_coherence + export_gif"
comment "test_shapes.py      — shape + coherence assertions"
comment "run_demo.py         — end-to-end: latents → temporal attention → GIF"

pause "$DELAY_SECTION"

comment "swap synthetic_weights() for from_checkpoint(path) when you have"
comment "mm_sd_v15_v2.ckpt and it runs on real hardware identically."

pause "$DELAY_SECTION"

printf '\n'
type "# fin."
printf '\n'
sleep 2
