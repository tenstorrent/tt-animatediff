#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Benchmark the ASGI server, against an already-running instance.

Follows tt-inference-server's performance_tests conventions: poll ``/tt-liveness`` for
``status == "alive"`` before timing anything, and drive an EXTERNAL server rather than
starting one, so the thing measured is a real serve rather than this script's own setup.

WHAT IT SEPARATES, AND WHY
--------------------------
**Cold start from steady state.** The first generation compiles kernels. Averaging it in
reports the compiler, not the model. The first request at each shape is timed separately
and excluded from the steady-state statistics.

Measured, and it corrected this script's own assumption: only the very FIRST call of the
whole process is cold. At 8 frames the first 4-step call took 11.45s against a 5.15s warm
median (2.22x), while the first 8- and 16-step calls came in at 1.02x and 0.99x of theirs.
So kernel compilation is per-process, not per-shape, and the per-shape framing below is
kept only because it costs one extra call and would catch the opposite being true on a
different model.

**Fixed overhead from per-step cost.** Latency is timed across a step sweep so the two can
be separated by a fit rather than assumed. A single (frames, steps) point cannot tell you
whether 4 steps costs 4 units or 1 unit plus overhead, and the difference decides whether
fewer steps is worth anything.

**Serialisation from concurrency.** The server holds one device behind a lock, so
concurrent requests queue. The concurrency probe checks that wall-clock for N simultaneous
requests is close to N sequential ones -- confirming the queueing is real. A server that
appeared to run them in parallel would be interleaving denoise loops on one device, which
is the failure the lock exists to prevent.

Reports median and full range, never a bare mean: with n=3 a mean hides an outlier that
the range makes obvious.

    python scripts/benchmark_serving.py --base-url http://127.0.0.1:8000 --repeats 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
from urllib.parse import urlsplit

import requests


def validated_base_url(raw: str) -> str:
    """Check --base-url is an http(s) address, and return it without a trailing slash.

    Every request this script makes is built from this one operator-supplied value:
    there is no request context here and no untrusted caller, so this is argument
    validation rather than an SSRF control. It earns its place by failing on a
    malformed value with a message that names the problem -- ``127.0.0.1:8000`` with
    the scheme left off otherwise reaches requests as a relative URL and comes back
    as a MissingSchema traceback pointing at the library, not at the flag.
    """
    parts = urlsplit(raw)
    if parts.scheme not in ("http", "https"):
        raise SystemExit(
            f"--base-url must start with http:// or https://, got {raw!r}"
        )
    if not parts.hostname:
        raise SystemExit(f"--base-url has no host: {raw!r}")
    return raw.rstrip("/")


def wait_for_ready(base_url: str, timeout: int = 600) -> float:
    """Poll /tt-liveness the way tt-media-server's harness does. Returns seconds waited."""
    start = time.time()
    url = f"{base_url.rstrip('/')}/tt-liveness"
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200 and r.json().get("status") == "alive":
                if r.json().get("model_ready"):
                    return time.time() - start
        except requests.exceptions.RequestException:
            pass
        time.sleep(1.0)
    raise SystemExit(f"server at {base_url} never became ready within {timeout}s")


def generate(base_url: str, *, frames: int, steps: int, seed: int = 0,
             timeout: int = 1800) -> Dict:
    """One generation. Returns timing plus the payload size, never the payload."""
    body = {"prompt": "a lighthouse in a storm, cinematic",
            "num_frames": frames, "num_inference_steps": steps, "seed": seed}
    t0 = time.perf_counter()
    r = requests.post(f"{base_url.rstrip('/')}/v1/videos/generations",
                      json=body, timeout=timeout)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    payload = r.json()
    return {"seconds": dt, "frames": frames, "steps": steps,
            "b64_len": len(payload["data"][0]["b64_json"])}


def summarise(times: List[float]) -> Dict:
    return {
        "n": len(times),
        "median_s": round(statistics.median(times), 3),
        "min_s": round(min(times), 3),
        "max_s": round(max(times), 3),
        # Reported alongside, never instead of, the range.
        "mean_s": round(statistics.fmean(times), 3),
    }


def sweep(base_url: str, *, frames: int, step_values: List[int], repeats: int) -> List[Dict]:
    rows = []
    for steps in step_values:
        cold = generate(base_url, frames=frames, steps=steps, seed=0)["seconds"]
        warm = [generate(base_url, frames=frames, steps=steps, seed=i + 1)["seconds"]
                for i in range(repeats)]
        row = {"frames": frames, "steps": steps,
               # NOT necessarily cold: only the process's first call is. See the
               # module docstring for the measurement.
               "first_call_at_this_shape_s": round(cold, 3), **summarise(warm)}
        # Reported because it is the obvious thing to ask for, and immediately
        # misleading: it divides the FIXED cost by the step count, so it falls from 1.287
        # to 0.749 across this sweep while the marginal cost is flat. Fit the line
        # (see --fit output) before quoting a per-step number.
        row["s_per_step_naive"] = round(row["median_s"] / steps, 4)
        row["frames_per_s"] = round(frames / row["median_s"], 3)
        rows.append(row)
        print(f"  frames={frames:>2} steps={steps:>2}  cold {cold:7.2f}s  "
              f"warm median {row['median_s']:7.2f}s  "
              f"[{row['min_s']:.2f}-{row['max_s']:.2f}]  "
              f"{row['s_per_step_naive']:.3f} s/step (naive)")
    return rows


def concurrency_probe(base_url: str, *, frames: int, steps: int, n: int = 3) -> Dict:
    """N simultaneous requests against a one-device server, which must serialise."""
    one = generate(base_url, frames=frames, steps=steps, seed=99)["seconds"]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(lambda i: generate(base_url, frames=frames, steps=steps, seed=100 + i),
                    range(n)))
    wall = time.perf_counter() - t0
    return {"n_concurrent": n, "single_request_s": round(one, 3),
            "wall_clock_s": round(wall, 3),
            "ratio_to_serial": round(wall / (one * n), 3),
            "reading": ("serialised as designed" if wall > one * n * 0.7
                        else "FASTER than serial -- the device lock is not holding")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--steps", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--frame-sweep", type=int, nargs="*", default=[16])
    ap.add_argument("--skip-concurrency", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    base_url = validated_base_url(args.base_url)

    waited = wait_for_ready(base_url)
    print(f"server ready (waited {waited:.1f}s)\n")

    print("step sweep:")
    rows = sweep(base_url, frames=args.frames, step_values=args.steps,
                 repeats=args.repeats)

    for f in args.frame_sweep or []:
        print(f"\nframe sweep (steps={args.steps[len(args.steps)//2]}):")
        rows += sweep(base_url, frames=f,
                      step_values=[args.steps[len(args.steps) // 2]], repeats=args.repeats)

    conc: Optional[Dict] = None
    if not args.skip_concurrency:
        print("\nconcurrency probe:")
        conc = concurrency_probe(base_url, frames=args.frames, steps=args.steps[0])
        print(f"  {conc['n_concurrent']} at once: {conc['wall_clock_s']}s vs "
              f"{conc['single_request_s']}s single -> {conc['reading']}")

    report = {"base_url": base_url, "rows": rows, "concurrency": conc}
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
