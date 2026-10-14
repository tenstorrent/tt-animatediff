#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Time every served generation mode against an already-running server.

``benchmark_serving.py`` sweeps steps and concurrency for the plain model. This script
times each *mode* the request accepts, one after another, on the same prompt and seed so
the numbers can be compared:

    plain 25 steps, plain 12 steps, no cross-frame blend, Euler sampler, every LoRA variant
    in the catalog (an LCM variant runs at its own defaults), prompt schedule, per-step
    preview, chaining (save, then continue from it), and the Phase 3 motion adapter.

Modes the server does not offer are listed as skipped, with the reason, instead of being
left out: ``/v1/models`` is read first, so a catalog with no LoRAs or a server started
without ``ANIMATEDIFF_MOTION_ADAPTER`` still produces a complete report.

The first request of the whole process compiles kernels, so it is made once, timed on its
own and kept out of every statistic. Each mode then runs ``--repeats`` times and the report
gives the median and the full range. The first repeat of each mode is saved as a GIF so a
fast mode that returns garbage shows up when someone looks.

    python scripts/benchmark_modes.py --base-url http://127.0.0.1:20000 --out output/bench
"""
from __future__ import annotations

import argparse
import base64
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

PROMPT = "a moonlit crypt beneath a drowned chapel, ghostly mist, haunting, cinematic"
NEGATIVE = "blurry, low quality, watermark"
FRAMES = 8
STEPS = 25


def base_payload(**over) -> dict:
    p = {
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE,
        "num_frames": FRAMES,
        "num_inference_steps": STEPS,
        "seed": 42,
    }
    p.update(over)
    return p


def post(base_url: str, payload: dict, timeout: float) -> tuple:
    """Return (seconds, gif_bytes). Raises on a non-200 so a refused mode is not timed."""
    t0 = time.perf_counter()
    r = requests.post(f"{base_url}/v1/videos/generations", json=payload, timeout=timeout)
    dt = time.perf_counter() - t0
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return dt, base64.b64decode(r.json()["data"][0]["b64_json"])


def build_modes(models: dict) -> tuple:
    """Return (modes, skipped). A mode is (name, payload, setup_payload_or_None)."""
    modes: List[tuple] = []
    skipped: List[tuple] = []
    modes.append(("plain, 25 steps", base_payload(), None))
    modes.append(("plain, 20 steps (baseline for preview and chaining)", base_payload(num_inference_steps=20), None))
    modes.append(("plain, 12 steps", base_payload(num_inference_steps=12), None))
    modes.append(("no cross-frame blend (temporal_alpha 0)", base_payload(temporal_alpha=0.0), None))
    modes.append(("Euler sampler, 25 steps", base_payload(sampler="euler"), None))
    modes.append((
        "prompt schedule (frames 0 and 7)",
        base_payload(prompt_schedule=[
            {"frame": 0, "prompt": PROMPT},
            {"frame": FRAMES - 1, "prompt": "a sunlit meadow of wildflowers, bright, cheerful, cinematic"},
        ]),
        None,
    ))
    modes.append(("per-step preview on, 20 steps",
                  base_payload(num_inference_steps=20, preview=True, preview_every=2), None))
    modes.append((
        "chaining: save, then continue (alpha 0.4)",
        base_payload(num_inference_steps=20, seed=7, chain_from="bench-chain", chain_alpha=0.4),
        base_payload(num_inference_steps=20, chain_save="bench-chain"),
    ))

    loras = models.get("loras") or []
    if not loras:
        skipped.append(("LoRA variants", "the server has no ANIMATEDIFF_LORAS catalog"))
    for name in loras:
        # An LCM variant defaults to 4 steps and guidance 1.5 on its own; the others take
        # the same 25 steps as the plain model so the comparison is like for like.
        fast = "fast" in name or "lcm" in name.lower()
        payload = base_payload(lora=name)
        if fast:
            payload.pop("num_inference_steps")
        modes.append((f"LoRA variant {name!r}" + (" (its own defaults)" if fast else ", 25 steps"), payload, None))

    ma = models.get("motion_adapter")
    if not ma:
        skipped.append(("Phase 3 motion adapter",
                        "the server was started without ANIMATEDIFF_MOTION_ADAPTER"))
    else:
        skip = ["up1", "up2"]
        modes.append(("motion adapter, 2 frames, 6 steps, skip up1+up2",
                      base_payload(num_frames=2, num_inference_steps=6, motion_adapter=True,
                                   motion_adapter_skip=skip), None))
        modes.append(("motion adapter alpha 0 (adapter bypassed), same shape",
                      base_payload(num_frames=2, num_inference_steps=6, motion_adapter=True,
                                   motion_adapter_alpha=0.0, motion_adapter_skip=skip), None))
    return modes, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="output/bench-modes")
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()
    base_url = args.base_url.rstrip("/")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    live = requests.get(f"{base_url}/tt-liveness", timeout=10).json()
    if live.get("status") != "alive":
        raise SystemExit(f"server is not alive yet: {live}")
    models = requests.get(f"{base_url}/v1/models", timeout=10).json()
    # `loras` and `motion_adapter` sit beside `data`, at the top level of the response.
    modes, skipped = build_modes(models)

    print("cold start (first request of the process, excluded from every statistic)...")
    cold, _ = post(base_url, base_payload(num_frames=4, num_inference_steps=4), args.timeout)
    print(f"  {cold:.1f}s")

    results: List[dict] = []
    for i, (name, payload, setup) in enumerate(modes):
        slug = f"{i:02d}"
        try:
            if setup is not None:
                post(base_url, setup, args.timeout)  # the producer run, not timed
            times: List[float] = []
            for r in range(args.repeats):
                dt, gif = post(base_url, payload, args.timeout)
                times.append(dt)
                if r == 0:
                    (out / f"{slug}.gif").write_bytes(gif)
            row = {"mode": name, "median_s": statistics.median(times), "min_s": min(times),
                   "max_s": max(times), "runs_s": times, "gif": f"{slug}.gif"}
            print(f"  {name:62s} median {row['median_s']:6.1f}s  range {row['min_s']:.1f}-{row['max_s']:.1f}")
        except Exception as exc:  # a mode that fails is a result, not a reason to stop the sweep
            row = {"mode": name, "error": str(exc)}
            print(f"  {name:62s} FAILED: {exc}")
        results.append(row)

    report = {"base_url": base_url, "frames": FRAMES, "steps": STEPS, "repeats": args.repeats,
              "cold_start_s": cold, "results": results,
              "skipped": [{"mode": m, "why": w} for m, w in skipped]}
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    for m, w in skipped:
        print(f"  skipped {m}: {w}")
    print(f"wrote {out/'results.json'}")


if __name__ == "__main__":
    main()
