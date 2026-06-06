#!/usr/bin/env bash
# Pre-writes all demo Python files into /tmp/tt-animatediff-demo/
# Called from the VHS tape's Hide block before the visible session starts.

set -euo pipefail

DEMO_DIR="/tmp/tt-animatediff-demo"
rm -rf "$DEMO_DIR"
mkdir -p "$DEMO_DIR/animatediff_ttnn"
touch "$DEMO_DIR/animatediff_ttnn/__init__.py"

# ── temporal_module.py ───────────────────────────────────────────────────────
cat > "$DEMO_DIR/animatediff_ttnn/temporal_module.py" << 'EOF'
"""Temporal attention for AnimateDiff on Tenstorrent hardware."""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Optional
import torch


@dataclass
class TemporalAttentionWeights:
    to_q_weight:   torch.Tensor
    to_q_bias:     Optional[torch.Tensor]
    to_k_weight:   torch.Tensor
    to_k_bias:     Optional[torch.Tensor]
    to_v_weight:   torch.Tensor
    to_v_bias:     Optional[torch.Tensor]
    to_out_weight: torch.Tensor
    to_out_bias:   Optional[torch.Tensor]
    pos_encoding:  Optional[torch.Tensor]
    dim:           int
    num_heads:     int


def sinusoidal_encoding(dim: int, max_len: int = 24) -> torch.Tensor:
    pos = torch.arange(max_len).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
    pe  = torch.zeros(1, max_len, dim)
    pe[0, :, 0::2] = torch.sin(pos * div)
    pe[0, :, 1::2] = torch.cos(pos * div)
    return pe


def temporal_attention(
    hidden_states: torch.Tensor,
    weights: TemporalAttentionWeights,
    num_frames: int,
) -> torch.Tensor:
    """Apply temporal attention across video frames.

    Input/output: (batch*frames, spatial_tokens, channels)
    Frames become the sequence dimension — standard attention
    then attends across time instead of space.
    """
    if num_frames == 1:
        return hidden_states

    bf, seq, c = hidden_states.shape
    b  = bf // num_frames
    h  = weights.num_heads
    hd = c // h

    # (b*f, seq, c) → (b*seq, f, c)
    x = hidden_states.view(b, num_frames, seq, c)
    x = x.permute(0, 2, 1, 3).reshape(b * seq, num_frames, c)

    if weights.pos_encoding is not None:
        x = x + weights.pos_encoding[:, :num_frames, :].to(x.device)

    F = torch.nn.functional
    q = F.linear(x, weights.to_q_weight, weights.to_q_bias)
    k = F.linear(x, weights.to_k_weight, weights.to_k_bias)
    v = F.linear(x, weights.to_v_weight, weights.to_v_bias)

    def split_heads(t):
        return t.view(b * seq, num_frames, h, hd).permute(0, 2, 1, 3)

    q, k, v = split_heads(q), split_heads(k), split_heads(v)

    scale  = 1.0 / math.sqrt(hd)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    probs  = F.softmax(scores, dim=-1)
    out    = torch.matmul(probs, v)

    out = out.permute(0, 2, 1, 3).reshape(b * seq, num_frames, c)
    out = F.linear(out, weights.to_out_weight, weights.to_out_bias)

    # (b*seq, f, c) → (b*f, seq, c)
    out = out.view(b, seq, num_frames, c).permute(0, 2, 1, 3).reshape(bf, seq, c)
    return out


def synthetic_weights(dim: int = 320, num_heads: int = 8) -> TemporalAttentionWeights:
    """Random weights matching mm_sd_v15_v2.ckpt shapes — for testing."""
    def w(r, c): return torch.randn(r, c) * 0.02
    def b(n):    return torch.zeros(n)
    return TemporalAttentionWeights(
        to_q_weight=w(dim, dim), to_q_bias=b(dim),
        to_k_weight=w(dim, dim), to_k_bias=b(dim),
        to_v_weight=w(dim, dim), to_v_bias=b(dim),
        to_out_weight=w(dim, dim), to_out_bias=b(dim),
        pos_encoding=sinusoidal_encoding(dim),
        dim=dim, num_heads=num_heads,
    )
EOF

# ── pipeline.py ──────────────────────────────────────────────────────────────
cat > "$DEMO_DIR/animatediff_ttnn/pipeline.py" << 'EOF'
"""Thin wrapper — apply temporal coherence on top of any latent generator."""
from __future__ import annotations
from pathlib import Path
from typing import List
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
    def from_checkpoint(cls, path: str, dim: int = 320, num_heads: int = 8):
        ckpt = torch.load(path, map_location="cpu")
        pfx  = ("down_blocks.0.motion_modules.0"
                ".temporal_transformer.transformer_blocks.0.attention_blocks.0")
        def g(k): return ckpt.get(f"{pfx}.{k}")
        w = TemporalAttentionWeights(
            to_q_weight=g("to_q.weight"), to_q_bias=g("to_q.bias"),
            to_k_weight=g("to_k.weight"), to_k_bias=g("to_k.bias"),
            to_v_weight=g("to_v.weight"), to_v_bias=g("to_v.bias"),
            to_out_weight=g("to_out.0.weight"), to_out_bias=g("to_out.0.bias"),
            pos_encoding=sinusoidal_encoding(dim),
            dim=dim, num_heads=num_heads,
        )
        return cls(w)

    @classmethod
    def from_synthetic(cls, dim: int = 320, num_heads: int = 8):
        return cls(synthetic_weights(dim=dim, num_heads=num_heads))

    def apply_temporal_coherence(
        self, latents: torch.Tensor, num_frames: int
    ) -> torch.Tensor:
        bf, h, w, c = latents.shape
        flat = latents.reshape(bf, h * w, c)
        flat = temporal_attention(flat, self.weights, num_frames)
        return flat.reshape(bf, h, w, c)

    def export_gif(self, frames: List[Image.Image], path: str, fps: int = 8):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(path, save_all=True, append_images=frames[1:],
                       duration=1000 // fps, loop=0)
EOF

# ── test_shapes.py ───────────────────────────────────────────────────────────
cat > "$DEMO_DIR/test_shapes.py" << 'EOF'
import torch, sys
sys.path.insert(0, ".")
from animatediff_ttnn.temporal_module import temporal_attention, synthetic_weights

B, F, SEQ, C = 1, 8, 256, 320
weights = synthetic_weights(dim=C)
x   = torch.randn(B * F, SEQ, C)
out = temporal_attention(x, weights, num_frames=F)

assert out.shape == x.shape
corrs = []
for i in range(F - 1):
    a = out[i].flatten(); b = out[i+1].flatten()
    corrs.append(torch.corrcoef(torch.stack([a, b]))[0, 1].item())
avg = sum(corrs) / len(corrs)

print(f"output shape          : {tuple(out.shape)}  ✓")
print(f"avg frame correlation : {avg:.4f}")
print("temporal coherence    : " + ("✓  pass" if avg > 0.3 else "✗  fail"))
EOF

# ── run_demo.py ──────────────────────────────────────────────────────────────
cat > "$DEMO_DIR/run_demo.py" << 'EOF'
import sys, torch
from PIL import Image, ImageDraw
sys.path.insert(0, ".")
from animatediff_ttnn.pipeline import AnimateDiffPipeline

NUM_FRAMES, DIM = 8, 320
pipe    = AnimateDiffPipeline.from_synthetic(dim=DIM)
torch.manual_seed(42)
latents = torch.randn(NUM_FRAMES, 8, 8, DIM)

print(f"Applying temporal coherence across {NUM_FRAMES} frames...")
latents = pipe.apply_temporal_coherence(latents, num_frames=NUM_FRAMES)

frames = []
for i in range(NUM_FRAMES):
    val = float(latents[i].mean())
    r   = int(min(255, max(0, 80  + val * 60)))
    g   = int(min(255, max(0, 180 - i * 12)))
    b   = int(min(255, max(0, 220 - i *  8)))
    img = Image.new("RGB", (320, 200), (r, g, b))
    d   = ImageDraw.Draw(img)
    d.text((12, 12), f"AnimateDiff  frame {i+1}/{NUM_FRAMES}", fill=(255, 255, 255))
    d.text((12, 36), f"TT hardware port — temporal coherence",  fill=(200, 240, 240))
    frames.append(img)

pipe.export_gif(frames, "/tmp/animatediff_tt.gif", fps=8)

corrs = []
for i in range(NUM_FRAMES - 1):
    a = latents[i].flatten(); b = latents[i+1].flatten()
    corrs.append(torch.corrcoef(torch.stack([a, b]))[0, 1].item())
avg = sum(corrs) / len(corrs)
print(f"avg frame correlation : {avg:.4f}  ✓")
print(f"exported              : /tmp/animatediff_tt.gif")
print("done.")
EOF

echo "Demo files written to $DEMO_DIR"
ls -1 "$DEMO_DIR/animatediff_ttnn/"
