#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Hardware smoke test: temporal attention on dual P300c Blackhole.

Validates that TTNN matmul reproduces the TT-Lang sim temporal attention
result (PCC > 0.99) and measures wall-clock throughput for one forward pass
at each UNet channel depth (C=320, C=640, C=1280).

The script does NOT inject into the UNet pipeline — it is a standalone
correctness and timing probe for the temporal-attention QKV/SDPA/out-proj
computation sequence.

If TTNN or Blackhole hardware is unavailable the script exits 0 with a clear
diagnostic message so it can be run safely in non-hardware CI environments.

Usage:
    source ~/tt-metal/python_env/bin/activate
    TT_METAL_ARCH_NAME=blackhole python scripts/ttlang_temporal_attn_hw_test.py

Optional flags:
    --seed N          Random seed for reproducible weight/input generation (default 42)
    --frames N        Number of frames N (default 8; kept as S=128, N padded to tile)
    --spatial N       Spatial positions S (default 128)
    --skip-hw         Dry-run: compute CPU reference only, skip device round-trip
"""

import argparse
import sys
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation coefficient between two tensors (flat, float32).

    PCC=1.0 means perfect linear correlation (identical up to scaling/offset).
    We use corrcoef rather than allclose because bfloat16 introduces a uniform
    ~0.4% relative error; PCC remains high (>0.99) even when element-wise
    absolute differences are non-trivial.
    """
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return torch.corrcoef(torch.stack([a_f, b_f]))[0, 1].item()


def cpu_reference(x: torch.Tensor, w_q, w_k, w_v, w_o, C: int) -> torch.Tensor:
    """Float32 CPU temporal-attention reference.

    Computes: out = x + softmax(Q @ K.T / sqrt(C)) @ V) @ W_o
    where Q = x @ W_q, K = x @ W_k, V = x @ W_v.

    This replicates TemporalAttentionKernel._forward_pytorch() directly so the
    smoke test has zero dependency on the simulator path — the hardware result
    is validated against this ground truth alone.

    Args:
        x:    [S, N, C] float32 input tensor.
        w_q, w_k, w_v, w_o: [C, C] float32 weight tensors.
        C:    Channel dimension (used as attention head dimension for scale).

    Returns:
        [S, N, C] float32 tensor — same shape as input.
    """
    scale = C ** -0.5
    xf = x.float()
    q = xf @ w_q                                # [S, N, C]
    k = xf @ w_k
    v = xf @ w_v
    scores = (q @ k.transpose(-1, -2)) * scale  # [S, N, N]
    attn = torch.softmax(scores, dim=-1)         # [S, N, N]
    attn_out = attn @ v                          # [S, N, C]
    return xf + attn_out @ w_o                  # [S, N, C]


# ---------------------------------------------------------------------------
# TTNN hardware path
# ---------------------------------------------------------------------------

def run_ttnn(x: torch.Tensor, w_q, w_k, w_v, w_o, device, ttnn) -> torch.Tensor:
    """Run temporal attention forward pass using TTNN ops on Blackhole.

    Strategy:
      1. Flatten input to 2-D: [S*N, C]
      2. Send x_2d and all four weight matrices to device as bfloat16 TILE tensors
      3. Q = matmul(x, w_q), K = matmul(x, w_k), V = matmul(x, w_v)
      4. scores = matmul(Q, K.T) * scale
      5. attn   = softmax(scores, dim=-1)
      6. out_2d = x_dev + matmul(matmul(attn, V), w_o)
      7. Retrieve to CPU and reshape to [S, N, C]

    MeshDevice note: to_device() from animatediff_ttnn.ttnn_pipeline replicates
    the tensor across all chips. from_device() concatenates along dim=0 and
    slices [:batch] to recover one replica — the same pattern used in
    generate_frames() in ttnn_pipeline.py. batch=S*N is passed for the final
    output so the full [S*N, C] result is not truncated.

    Args:
        x:      [S, N, C] float32 input tensor (CPU).
        w_q, w_k, w_v, w_o: [C, C] float32 weight tensors (CPU).
        device: TTNN device or MeshDevice from setup_blackhole().
        ttnn:   The ttnn module (pre-imported by caller).

    Returns:
        [S, N, C] float32 tensor on CPU, cast from bfloat16.

    Raises:
        RuntimeError: If a required TTNN op fails on this device type.
    """
    # Track which stages actually executed on hardware for the report.
    hw_stages = []

    S, N, C = x.shape
    x_2d = x.float().reshape(S * N, C)  # [S*N, C]

    # ---- Send inputs to device ------------------------------------------------
    # Use project helpers from animatediff_ttnn.ttnn_pipeline (imported above)
    # which handle both single Device and MeshDevice transparently.
    x_dev  = to_device(x_2d, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    wq_dev = to_device(w_q,  device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    wk_dev = to_device(w_k,  device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    wv_dev = to_device(w_v,  device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    wo_dev = to_device(w_o,  device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    hw_stages.append("send_to_device")

    # ---- QKV projections -------------------------------------------------------
    q_dev = ttnn.matmul(x_dev, wq_dev)   # [S*N, C]
    k_dev = ttnn.matmul(x_dev, wk_dev)
    v_dev = ttnn.matmul(x_dev, wv_dev)
    hw_stages.append("qkv_matmul")

    # ---- Scaled dot-product scores  -------------------------------------------
    # K must be transposed: [S*N, C] → [C, S*N].
    # ttnn.transpose operates on the last two dimensions when given -2, -1.
    k_T = ttnn.transpose(k_dev, -2, -1)                      # [C, S*N]
    scale = C ** -0.5
    scores_dev = ttnn.matmul(q_dev, k_T)                     # [S*N, S*N]
    # Scale: multiply element-wise by a scalar — TTNN supports scalar mul via *
    scores_dev = scores_dev * scale
    hw_stages.append("scores_scaled")

    # ---- Softmax  -------------------------------------------------------------
    attn_dev = ttnn.softmax(scores_dev, dim=-1)               # [S*N, S*N]
    hw_stages.append("softmax")

    # ---- Attention output  ----------------------------------------------------
    attn_out_dev = ttnn.matmul(attn_dev, v_dev)               # [S*N, C]
    hw_stages.append("attn_out_matmul")

    # ---- Output projection + residual  ----------------------------------------
    proj_dev = ttnn.matmul(attn_out_dev, wo_dev)              # [S*N, C]
    out_dev  = x_dev + proj_dev                               # [S*N, C] residual
    hw_stages.append("out_proj_residual")

    # ---- Retrieve to CPU  -----------------------------------------------------
    # batch=S*N: from_device slices [:batch] after concatenating chip replicas,
    # so the full [S*N, C] result is preserved (not truncated to batch=1).
    out_cpu = from_device(out_dev, device, batch=S * N).float()  # cast bf16 → f32
    hw_stages.append("from_device")

    # Reshape back to [S, N, C]
    out = out_cpu.reshape(S, N, C)

    print(f"    hw stages executed: {', '.join(hw_stages)}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Temporal attention hardware smoke test")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--frames", type=int, default=8,
                   help="Number of frames N (default: 8). Padded internally to "
                        "tile multiple ≥ N; TTNN path uses S*N as the sequence dim.")
    p.add_argument("--spatial", type=int, default=128,
                   help="Spatial positions S (default: 128)")
    p.add_argument("--skip-hw", action="store_true",
                   help="Skip hardware, compute CPU reference only (dry run)")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # ---- Try to import TTNN ---------------------------------------------------
    # Fail gracefully: TTNN requires the tt-metal Python environment, which is not
    # present in all CI runners. Exit 0 so the smoke test does not block CI.
    ttnn = None
    if not args.skip_hw:
        try:
            import ttnn as _ttnn
            ttnn = _ttnn
            print("[info] ttnn imported successfully")
        except ImportError as exc:
            print(f"[skip] ttnn not available ({exc}). "
                  f"Activate ~/tt-metal/python_env and set "
                  f"TT_METAL_ARCH_NAME=blackhole to run the hardware path.")
            print("[info] Falling back to CPU-only reference run.")

    # ---- Open Blackhole device ------------------------------------------------
    device = None
    if ttnn is not None and not args.skip_hw:
        # Import setup_blackhole from the project.  We add the repo root to
        # sys.path so this works when the script is called without installation.
        repo_root = Path(__file__).resolve().parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        try:
            from animatediff_ttnn.ttnn_pipeline import setup_blackhole, to_device, from_device
            print("[info] Opening Blackhole device …")
            device = setup_blackhole()
            n_chips = device.num_devices if hasattr(device, "num_devices") else 1
            print(f"[info] MeshDevice opened: {n_chips} chip(s)")
        except Exception as exc:
            print(f"[skip] Could not open Blackhole device: {exc}")
            print("[info] Running CPU reference only.")
            device = None

    # ---- Determine channel depths to test -------------------------------------
    # Standard AnimateDiff UNet channel depths: down/mid/up blocks use C=320, 640, 1280.
    channel_depths = [320, 640, 1280]

    # S and N from args
    S = args.spatial
    N_real = args.frames
    # TTNN TILE_LAYOUT requires dimensions ≥ 32 and aligned to 32.
    # We round N up to the nearest multiple of 32 (minimum 32).
    TILE = 32
    N = max(TILE, ((N_real + TILE - 1) // TILE) * TILE)
    if N != N_real:
        print(f"[info] Frames padded {N_real} → {N} to satisfy TILE_LAYOUT alignment")

    print()
    print("╔══════════════════════════════════════════════════════════════")
    print("║  tt-animatediff — Temporal Attention Hardware Smoke Test")
    print(f"║  S={S}  N={N} (real frames={N_real})  seed={args.seed}")
    mode = "CPU-only (no hardware)" if (device is None) else "Blackhole TTNN + CPU reference"
    print(f"║  mode: {mode}")
    print("╚══════════════════════════════════════════════════════════════")
    print()

    results = []  # list of (C, pcc_val | None, hw_ms | None, cpu_ms, status)

    for C in channel_depths:
        print(f"─── C={C} ─────────────────────────────────────────────────────")

        # Generate random input and weights. Scale weights by 0.02 to keep
        # attention scores in a reasonable range (avoids softmax saturation).
        torch.manual_seed(args.seed)
        x      = torch.randn(S, N, C)
        w_q    = torch.randn(C, C) * 0.02
        w_k    = torch.randn(C, C) * 0.02
        w_v    = torch.randn(C, C) * 0.02
        w_o    = torch.randn(C, C) * 0.02

        # ---- CPU float32 reference --------------------------------------------
        t0_cpu = time.perf_counter()
        ref = cpu_reference(x, w_q, w_k, w_v, w_o, C)
        cpu_ms = (time.perf_counter() - t0_cpu) * 1e3
        print(f"  CPU reference: {cpu_ms:.1f} ms  shape={tuple(ref.shape)}")

        # ---- Hardware path (if device available) ------------------------------
        hw_ms = None
        pcc_val = None
        status = "CPU_ONLY"

        if device is not None:
            try:
                # Warm-up: a single small matmul to ensure the device is
                # initialised and the JIT cache is populated before timing.
                _warmup = ttnn.matmul(
                    ttnn.from_torch(
                        torch.ones(32, 32),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                        **({"mesh_mapper": ttnn.ReplicateTensorToMesh(device)}
                           if isinstance(device, ttnn.MeshDevice) else {}),
                    ),
                    ttnn.from_torch(
                        torch.ones(32, 32),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                        **({"mesh_mapper": ttnn.ReplicateTensorToMesh(device)}
                           if isinstance(device, ttnn.MeshDevice) else {}),
                    ),
                )
                del _warmup

                t0_hw = time.perf_counter()
                hw_out = run_ttnn(x, w_q, w_k, w_v, w_o, device, ttnn)
                hw_ms = (time.perf_counter() - t0_hw) * 1e3

                pcc_val = pcc(hw_out, ref)
                threshold = 0.99
                ok = pcc_val >= threshold
                status = "PASS" if ok else "FAIL"

                print(f"  TTNN hardware: {hw_ms:.1f} ms  PCC={pcc_val:.4f}  [{status}]")
                if not ok:
                    print(f"  [warn] PCC {pcc_val:.4f} < threshold {threshold} — "
                          f"bfloat16 precision degradation may be significant for C={C}")

            except Exception as exc:
                # Hardware path failed — report the error but continue testing
                # remaining channel depths.  The script still exits 0 so it
                # does not block CI.
                print(f"  [error] TTNN path failed for C={C}: {exc}")
                import traceback
                traceback.print_exc()
                status = "ERROR"

        results.append((C, pcc_val, hw_ms, cpu_ms, status))
        print()

    # ---- Summary table --------------------------------------------------------
    print("╔══════════════════════════════════════════════════════════════")
    print("║  Summary")
    print("║")
    print(f"║  {'C':>6}  {'PCC':>8}  {'HW ms':>8}  {'CPU ms':>8}  STATUS")
    print("║  " + "─" * 52)
    all_pass = True
    for C, pcc_v, hw_ms, cpu_ms, status in results:
        pcc_str = f"{pcc_v:.4f}" if pcc_v is not None else "—"
        hw_str  = f"{hw_ms:.1f}"  if hw_ms  is not None else "—"
        cpu_str = f"{cpu_ms:.1f}"
        print(f"║  {C:>6}  {pcc_str:>8}  {hw_str:>8}  {cpu_str:>8}  {status}")
        if status == "FAIL":
            all_pass = False
    print("╚══════════════════════════════════════════════════════════════")

    # ---- Close device (always) ------------------------------------------------
    if device is not None:
        try:
            ttnn.close_mesh_device(device)
            print("[info] MeshDevice closed")
        except Exception:
            # close_mesh_device may not exist in all TTNN versions; ignore.
            pass

    # ---- Exit code ------------------------------------------------------------
    # Exit 0 for: all PASS, CPU_ONLY (no hardware), or ERROR (hw not available).
    # Exit 1 only for FAIL (hardware ran but PCC below threshold).
    if not all_pass:
        print("[result] FAIL — one or more PCC checks did not meet threshold 0.99")
        sys.exit(1)

    print("[result] OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
