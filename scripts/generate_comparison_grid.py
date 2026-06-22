#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Generate a 4-mode comparison grid for visual evaluation.

Runs 4 modes on 3 prompts and produces an HTML viewer:
  A: Blackhole TTNN, no temporal attention   (fastest, incoherent)
  B: Blackhole TTNN + Phase 2.5 cross-frame  (coherent, 25 steps)
  C: Blackhole Lightning 8-step              (fast coherent)
  D: CPU AnimateDiff Phase 1                 (best quality, slow)

Usage:
    source ~/tt-metal/python_env/bin/activate
    python scripts/generate_comparison_grid.py [--skip-cpu] [--only-prompt 0]

Output: output/comparison/
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).parent.parent
OUT = REPO / "output" / "comparison"
GENERATE = REPO / "examples" / "generate.py"

PROMPTS = [
    {
        "slug": "aurora",
        "prompt": "aurora borealis dancing over arctic ice, green and violet ribbons of light, starfield, cinematic 4K",
        "negative": "blurry, low quality, text, people, faces, tropical",
        "seed": 42,
    },
    {
        "slug": "lava",
        "prompt": "molten lava flowing into the ocean at night, glowing orange cracks in black rock, steam explosions, dramatic macro",
        "negative": "blurry, low quality, text, people, faces",
        "seed": 77,
    },
    {
        "slug": "mycelium",
        "prompt": "mycelium network glowing with bioluminescent spores, threads of light connecting nodes, cosmic forest underground",
        "negative": "blurry, low quality, text, people, faces",
        "seed": 99,
    },
]

MODES = [
    {
        "key": "A-noattn",
        "label": "A: Blackhole, no temporal attn",
        "description": "Fastest. Frames denoised independently — no inter-frame coherence.",
        "args": ["--mode", "blackhole", "--temporal-alpha", "0", "--frames", "8", "--steps", "25"],
        "skip_if_cpu_skip": False,
    },
    {
        "key": "B-phase25",
        "label": "B: Blackhole + Phase 2.5 cross-frame",
        "description": "25-step PNDM with cross-frame attention on noise predictions. Current best hardware quality.",
        "args": ["--mode", "blackhole", "--temporal-alpha", "0.35", "--frames", "8", "--steps", "25"],
        "skip_if_cpu_skip": False,
    },
    {
        "key": "C-lightning",
        "label": "C: Blackhole Lightning 8-step",
        "description": "Euler scheduler, 8 steps. ~3× faster than B with cross-frame attention.",
        "args": ["--mode", "blackhole", "--lightning", "--lightning-steps", "8", "--temporal-alpha", "0.35", "--frames", "8"],
        "skip_if_cpu_skip": False,
    },
    {
        "key": "D-highalpha",
        "label": "D: Blackhole high-α (0.75)",
        "description": "Phase 2.5 with stronger cross-frame blend (α=0.75). More coherent but less per-frame variety.",
        "args": ["--mode", "blackhole", "--temporal-alpha", "0.75", "--frames", "8", "--steps", "25"],
        "skip_if_cpu_skip": False,
    },
    {
        "key": "E-16frames",
        "label": "E: Blackhole 16 frames",
        "description": "Standard Phase 2.5 with 16 frames instead of 8. Longer animation, more motion.",
        "args": ["--mode", "blackhole", "--temporal-alpha", "0.35", "--frames", "16", "--steps", "25"],
        "skip_if_cpu_skip": False,
    },
    {
        "key": "F-motion",
        "label": "F: Blackhole + MotionAdapter (Lightning)",
        "description": "Phase 3: full AnimateDiff temporal attention inside TTNN UNet. 8-step Lightning.",
        "args": ["--mode", "blackhole", "--lightning", "--lightning-steps", "8",
                 "--motion-adapter", "--frames", "8"],
        "skip_if_cpu_skip": False,
    },
]


def run_mode(mode, prompt_cfg, dry_run=False):
    out_path = OUT / f"{prompt_cfg['slug']}-{mode['key']}.gif"
    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists")
        return {"path": out_path, "elapsed": None, "skipped": True}

    cmd = [
        sys.executable, str(GENERATE),
        "--prompt", prompt_cfg["prompt"],
        "--negative-prompt", prompt_cfg["negative"],
        "--seed", str(prompt_cfg["seed"]),
        "--output", str(out_path),
    ] + mode["args"]

    print(f"  Running: {' '.join(cmd[-6:])}")
    if dry_run:
        print("  [dry-run] skipping actual execution")
        return {"path": out_path, "elapsed": 0, "skipped": False}

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [ERROR] exit {result.returncode}")
        print(result.stderr[-500:] if result.stderr else "  (no stderr)")
        return {"path": None, "elapsed": elapsed, "error": result.stderr[-200:]}

    print(f"  Done in {elapsed:.1f}s → {out_path.name} ({out_path.stat().st_size // 1024} KB)")
    return {"path": out_path, "elapsed": elapsed, "skipped": False}


def build_html(results):
    """Build a side-by-side comparison HTML page."""

    rows_html = ""
    for p in PROMPTS:
        slug = p["slug"]
        cells = ""
        for m in MODES:
            key = m["key"]
            r = results.get(f"{slug}-{key}", {})
            path = r.get("path")
            elapsed = r.get("elapsed")
            skipped = r.get("skipped", False)
            error = r.get("error")

            if path and path.exists():
                rel = path.relative_to(OUT)
                time_str = f"{elapsed:.0f}s" if elapsed else "(cached)"
                if skipped and r.get("elapsed") is None:
                    time_str = "(cached)"
                cell_content = f"""
                <img src="{rel}" alt="{slug} {key}" loading="lazy" style="width:100%;border-radius:4px">
                <div class="timing">{time_str}</div>"""
            elif error:
                cell_content = f'<div class="error">ERROR<br><small>{error[:80]}</small></div>'
            else:
                cell_content = '<div class="missing">not generated</div>'

            cells += f'<td class="cell">{cell_content}</td>\n'

        rows_html += f"""
        <tr>
          <td class="prompt-label"><strong>{slug}</strong><br><small>{p["prompt"][:80]}…</small></td>
          {cells}
        </tr>"""

    mode_headers = "\n".join(
        f'<th><div class="mode-label">{m["label"]}</div>'
        f'<div class="mode-desc">{m["description"]}</div></th>'
        for m in MODES
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AnimateDiff Mode Comparison</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #0f2a35; color: #e8f0f2; margin: 0; padding: 20px; }}
  h1 {{ color: #4fd1c5; margin-bottom: 4px; }}
  .subtitle {{ color: #607d8b; margin-bottom: 24px; font-size: 14px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th {{ background: #1a3c47; padding: 12px 8px; text-align: center; border: 1px solid #2d3142; vertical-align: top; }}
  td {{ border: 1px solid #2d3142; padding: 8px; vertical-align: top; }}
  .cell {{ background: #1a3c47; text-align: center; min-width: 200px; }}
  .prompt-label {{ background: #0f2a35; padding: 12px; min-width: 160px; font-size: 13px; color: #b0c4de; }}
  .timing {{ font-size: 11px; color: #4fd1c5; margin-top: 4px; }}
  .mode-label {{ color: #4fd1c5; font-weight: bold; font-size: 13px; }}
  .mode-desc {{ color: #607d8b; font-size: 11px; margin-top: 4px; }}
  .error {{ color: #ff6b6b; padding: 20px; font-size: 12px; }}
  .missing {{ color: #607d8b; padding: 20px; }}
  img {{ display: block; margin: 0 auto; }}
</style>
</head>
<body>
<h1>AnimateDiff — Hardware Mode Comparison</h1>
<div class="subtitle">Dual P300c Blackhole · {time.strftime("%Y-%m-%d %H:%M")}</div>
<table>
  <thead>
    <tr>
      <th>Prompt</th>
      {mode_headers}
    </tr>
  </thead>
  <tbody>
    {rows_html}
  </tbody>
</table>
<div style="margin-top:24px;color:#607d8b;font-size:12px">
  <strong style="color:#4fd1c5">Phase 3 (F-motion):</strong>
  Blackhole TTNN + full MotionAdapter temporal attention.
  320→640→1280→1280 dim feature injection at 7 UNet blocks, 2-3 motion modules each.
  Weights: guoyww/animatediff-motion-adapter-v1-5-2.
</div>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cpu", action="store_true", help="Skip CPU mode (slow)")
    parser.add_argument("--only-prompt", type=int, default=None, help="Run only prompt index 0/1/2")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    prompts = PROMPTS if args.only_prompt is None else [PROMPTS[args.only_prompt]]
    results = {}

    for p in prompts:
        print(f"\n=== Prompt: {p['slug']} ===")
        for m in MODES:
            if args.skip_cpu and m["skip_if_cpu_skip"]:
                print(f"  [skip-cpu] {m['key']}")
                continue
            print(f"\n  Mode: {m['label']}")
            r = run_mode(m, p, dry_run=args.dry_run)
            results[f"{p['slug']}-{m['key']}"] = r

    html = build_html(results)
    html_path = OUT / "index.html"
    html_path.write_text(html)
    print(f"\nComparison viewer: {html_path}")
    print(f"Open with: python3 -m http.server 8080 --directory {OUT}")


if __name__ == "__main__":
    main()
