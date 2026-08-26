#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""AnimateDiff video generation — unified entry point.

Modes
-----
blackhole  (default) — TTNN UNet on Blackhole hardware + cross-frame temporal
                       attention. Most performant on silicon. ~15 s/frame.
cpu        — diffusers AnimateDiffPipeline with MotionAdapter, CPU only.
             Full AnimateDiff temporal attention. ~2 min/frame. No hardware.
sim        — Same as blackhole but against a ttsim virtual device.
             Any Linux/x86_64 machine; no silicon required.

Requirements
------------
All modes:
    pip install -e ".[dev]"
    hf download CompVis/stable-diffusion-v1-4

cpu mode also needs the motion adapter:
    hf download guoyww/animatediff-motion-adapter-v1-5-2

blackhole / sim modes also need tt-metal:
    source ~/tt-metal/python_env/bin/activate

sim mode also needs ttsim:
    mkdir -p ~/sim
    wget -O ~/sim/libttsim_bh.so \\
        https://github.com/tenstorrent/ttsim/releases/download/v1.7.0/libttsim_bh.so

Usage
-----
    # Blackhole hardware (default, most performant)
    python examples/generate.py --prompt "ocean waves, cinematic 4K" --frames 8

    # CPU only (no hardware required)
    python examples/generate.py --mode cpu --frames 16

    # ttsim simulator (no hardware, slower)
    python examples/generate.py --mode sim --frames 2 --steps 4
    python examples/generate.py --mode sim --sim ~/sim/libttsim_bh.so --frames 2

    # Disable temporal attention blending
    python examples/generate.py --temporal-alpha 0
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch

# ── parse args first — sim mode needs --sim before env bootstrap ───────────
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AnimateDiff — Blackhole TTNN UNet (default), CPU, or ttsim"
    )
    parser.add_argument(
        "--mode",
        choices=["blackhole", "cpu", "sim"],
        default="blackhole",
        help="Execution backend (default: blackhole)",
    )
    parser.add_argument(
        "--prompt",
        default="1939 World's Fair imagined from the year 2099, art deco spires at golden dusk, retro-futurist optimism, cinematic 4K",
    )
    parser.add_argument(
        "--negative-prompt", default="blurry, low quality", dest="negative_prompt"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Frames to generate (default: 16 for cpu, 8 for blackhole/sim)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Denoising steps (default: 25 for blackhole/cpu, 4 for sim)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=None,
        help="Output GIF path (default: output/<mode>.gif)",
    )
    parser.add_argument(
        "--temporal-alpha",
        type=float,
        default=0.35,
        dest="temporal_alpha",
        help=(
            "Cross-frame attention blend 0–1 (blackhole/sim only; default 0.35). "
            "In Lightning mode a cosine decay schedule is applied automatically: "
            "alpha decays from this value to ~15%% of it across the denoising steps."
        ),
    )
    parser.add_argument(
        "--sim",
        default=None,
        metavar="PATH",
        help="Path to libttsim_bh.so for sim mode (overrides TT_METAL_SIMULATOR)",
    )
    parser.add_argument(
        "--lightning",
        action="store_true",
        help=(
            "Use EulerDiscreteScheduler instead of PNDM — different solver, same CFG=7.5. "
            "In cpu mode also loads AnimateDiff-Lightning distilled weights (requires CFG=1.0). "
            "In blackhole/sim modes the base SD 1.4 TTNN UNet is used regardless."
        ),
    )
    parser.add_argument(
        "--lightning-steps",
        type=int,
        default=4,
        choices=[2, 4, 8],
        dest="lightning_steps",
        help="Lightning distillation step count: 2, 4, or 8 (default 4)",
    )
    parser.add_argument(
        "--lcm-unet",
        default=None,
        dest="lcm_unet",
        metavar="PATH",
        help="Load our own LCM-distilled UNet weights (.pt from scripts/distill_lcm.py). "
             "Works with --mode cpu. Sets --steps to 4 or 8 automatically if not specified.",
    )
    parser.add_argument(
        "--lcm-adapter",
        default=None,
        dest="lcm_adapter",
        metavar="PATH",
        help="Load our own LCM-distilled MotionAdapter weights (.pt from scripts/distill_motion_adapter.py). "
             "Use together with --lcm-unet for full LCM inference.",
    )
    parser.add_argument(
        "--chain-from",
        default=None,
        dest="chain_from",
        metavar="PATH",
        help="Load latents saved by a previous --chain-save run and blend into seed noise "
             "for visual narrative continuity across prompts (blackhole/sim only).",
    )
    parser.add_argument(
        "--chain-save",
        default=None,
        dest="chain_save",
        metavar="PATH",
        help="Save this run's final denoised latents to PATH (.pt) for use by --chain-from.",
    )
    parser.add_argument(
        "--chain-alpha",
        type=float,
        default=0.6,
        dest="chain_alpha",
        help="Blend weight for --chain-from latents (0=ignore, 1=replace; default 0.6).",
    )
    parser.add_argument(
        "--motion-adapter",
        metavar="PATH",
        nargs="?",
        const="guoyww/animatediff-motion-adapter-v1-5-2",
        default=None,
        dest="motion_adapter",
        help=(
            "Load MotionAdapter weights for Phase 3 temporal attention. "
            "PATH defaults to HuggingFace cache for guoyww/animatediff-motion-adapter-v1-5-2. "
            "Only valid with --mode blackhole."
        ),
    )
    parser.add_argument(
        "--motion-adapter-alpha",
        type=float,
        default=1.0,
        dest="motion_adapter_alpha",
        help=(
            "Injection blend weight for MotionAdapter temporal attention (0.0–1.0). "
            "0.0 = bypass (no-op, useful for debugging forward_unet_staged in isolation). "
            "1.0 = full injection (default). Only used with --motion-adapter."
        ),
    )
    parser.add_argument(
        "--motion-adapter-skip",
        nargs="+",
        default=[],
        dest="motion_adapter_skip",
        metavar="KEY",
        help=(
            "Injection-point keys to skip (space-separated). "
            "Valid keys: down0 down1 down2 mid up0 up1 up2. "
            "Skipping up1/up2 cuts ~85%% of CPU overhead (large spatial dims) "
            "with minimal quality impact. Example: --motion-adapter-skip up1 up2"
        ),
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=None,
        dest="device_id",
        metavar="ID",
        help=(
            "Blackhole chip index to use (0-based). Defaults to all available chips "
            "(typically 4 on a quad-P300c system). Use to pin a generation run to "
            "a specific chip for parallel multi-process batch jobs."
        ),
    )
    parser.add_argument(
        "--preview-path",
        type=str,
        default=None,
        dest="preview_path",
        metavar="GIF",
        help=(
            "Write a rolling preview GIF of the in-flight latents to GIF as "
            "denoising proceeds, and print a 'PREVIEW: <step>/<total> <path>' "
            "line each time. Lets a GUI draining this process's stdout show the "
            "animation forming. The preview is CPU-side (no VAE, no device "
            "work), so it adds no latency to the hardware pipeline."
        ),
    )
    parser.add_argument(
        "--preview-every",
        type=int,
        default=None,
        dest="preview_every",
        metavar="N",
        help=(
            "Emit a preview every N steps (default: every step for runs of "
            "<=10 steps, every 2nd step otherwise). The final step always "
            "emits regardless."
        ),
    )
    return parser

args = _build_parser().parse_args()

# Apply mode-specific defaults now that we know the mode
if args.frames is None:
    args.frames = 16 if args.mode == "cpu" else 8
if args.steps is None:
    if args.mode == "sim":
        args.steps = 4
    elif args.lightning and args.mode == "cpu":
        # CPU Lightning uses real distilled adapter — step count must match the
        # checkpoint (2, 4, or 8). Blackhole/sim Lightning uses base TTNN UNet;
        # 25 steps is correct there (no distillation constraint).
        args.steps = args.lightning_steps
    elif args.lcm_unet:
        # LCM distilled UNet — default to 8 steps (balanced quality/speed).
        # Pass --steps 4 explicitly for maximum speed.
        args.steps = 8
    else:
        args.steps = 25
if args.output is None:
    args.output = f"output/{args.mode}.gif"
if not 0.0 <= args.temporal_alpha <= 1.0:
    _build_parser().error(f"--temporal-alpha must be in [0, 1], got {args.temporal_alpha}")
if args.lightning and args.mode != "cpu":
    # On Blackhole/sim, --lightning switches to TtEulerScheduler (trailing,
    # linear) and runs the base TTNN UNet — no distilled adapter is loaded.
    # CFG stays at 7.5 (distilled CFG=1.0 constraint doesn't apply here).
    # This is intentional and supported; no error.
    pass

# ── sim: resolve ttsim path and configure env before tt-metal loads ────────
if args.mode == "sim":
    _DEFAULT_SIM = Path.home() / "sim" / "libttsim_bh.so"
    if args.sim:
        _sim_so = Path(args.sim)
        if not _sim_so.exists():
            print(f"ERROR: --sim path not found: {_sim_so}", file=sys.stderr)
            sys.exit(1)
        os.environ["TT_METAL_SIMULATOR"] = str(_sim_so)
    elif os.environ.get("TT_METAL_SIMULATOR"):
        _sim_so = Path(os.environ["TT_METAL_SIMULATOR"])
        if not _sim_so.exists():
            print(
                f"ERROR: TT_METAL_SIMULATOR path not found: {_sim_so}\n"
                "Update the env var or pass --sim /path/to/libttsim_bh.so",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        _sim_so = _DEFAULT_SIM
        if not _sim_so.exists():
            print(
                f"ERROR: ttsim binary not found at {_sim_so}\n"
                "Download it from https://github.com/tenstorrent/ttsim/releases\n"
                "or pass --sim /path/to/libttsim_bh.so",
                file=sys.stderr,
            )
            sys.exit(1)
        os.environ["TT_METAL_SIMULATOR"] = str(_sim_so)

    # Required simulator env before tt-metal dispatch initialises
    os.environ.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")
    os.environ.setdefault("TT_METAL_DISABLE_SFPLOADMACRO", "1")
    os.environ.setdefault("TT_METAL_ARCH_NAME", "blackhole")

# ── project / tt-metal paths ───────────────────────────────────────────────
TT_METAL_PATH = Path.home() / "tt-metal"
sys.path.insert(0, str(Path(__file__).parent.parent))
if args.mode in ("blackhole", "sim"):
    sys.path.insert(0, str(TT_METAL_PATH))


# ══════════════════════════════════════════════════════════════════════════
# Shared helpers (blackhole + sim share load_sd14_ttnn / encode_prompt)
# ══════════════════════════════════════════════════════════════════════════

# Helpers live in animatediff_ttnn.generation_helpers so they can be
# imported by app.py without triggering this module's arg-parsing side effect.
from animatediff_ttnn.generation_helpers import load_sd14_ttnn, encode_prompt


# ══════════════════════════════════════════════════════════════════════════
# Mode: cpu
# ══════════════════════════════════════════════════════════════════════════

def run_cpu():
    from animatediff_ttnn.pipeline import (
        create_animatediff_pipeline, create_lightning_pipeline, generate, export_gif,
    )
    import torch as _torch

    if args.lcm_unet:
        lcm_tag = f"LCM-{args.steps}step"
        if args.lcm_adapter:
            label = f"AnimateDiff LCM — {lcm_tag} UNet + MotionAdapter (our distilled weights)"
        else:
            label = f"AnimateDiff LCM — {lcm_tag} UNet only (our distilled weights)"
        guidance = 1.0
    elif args.lightning:
        label = f"AnimateDiff-Lightning ({args.lightning_steps}-step distilled, ~6× faster) — CPU"
        # Real distilled adapter: guidance is baked in, CFG=1.0 required
        guidance = 1.0
    else:
        label = "AnimateDiff — CPU mode (diffusers AnimateDiffPipeline + MotionAdapter)"
        guidance = 7.5

    print(label)
    print(f"  Prompt  : {args.prompt}")
    print(f"  Frames  : {args.frames}  Steps: {args.steps}  Seed: {args.seed}")
    print()

    print("Loading pipeline (first run downloads weights)...")
    t0 = time.time()
    if args.lcm_unet:
        pipe = create_animatediff_pipeline()
        unet_state = _torch.load(args.lcm_unet, map_location="cpu", weights_only=True)
        pipe.unet.load_state_dict(unet_state, strict=False)
        print(f"  Loaded LCM UNet weights from {args.lcm_unet}")
        if args.lcm_adapter:
            adapter_state = _torch.load(args.lcm_adapter, map_location="cpu", weights_only=True)
            pipe.motion_adapter.load_state_dict(adapter_state, strict=False)
            print(f"  Loaded LCM MotionAdapter weights from {args.lcm_adapter}")
        from diffusers import EulerDiscreteScheduler
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
            beta_schedule="linear",
        )
    elif args.lightning:
        pipe = create_lightning_pipeline(step=args.lightning_steps)
    else:
        pipe = create_animatediff_pipeline()
    print(f"  Loaded in {time.time() - t0:.1f}s\n")

    print(f"Generating {args.frames} frames...")
    t1 = time.time()
    frames = generate(
        pipe, args.prompt,
        negative_prompt=args.negative_prompt,
        num_frames=args.frames,
        guidance_scale=guidance,
        num_inference_steps=args.steps,
        seed=args.seed,
        on_step=_preview_callback(args),
    )
    elapsed = time.time() - t1
    print(f"  Done in {elapsed:.1f}s ({elapsed / args.frames:.1f}s/frame)\n")

    export_gif(frames, args.output)
    print(f"Saved {len(frames)} frames → {args.output}")
    if args.lightning:
        print(f"\nNote: AnimateDiff-Lightning {args.lightning_steps}-step distilled weights (ByteDance).")
        print("      CFG disabled (guidance_scale=1.0) — required for Lightning.")
    else:
        print("\nNote: MotionAdapter injected temporal attention into every UNet block.")
        print("      Each denoising step attends across all frames simultaneously.")


# ══════════════════════════════════════════════════════════════════════════
# Mode: blackhole / sim
# ══════════════════════════════════════════════════════════════════════════

def _open_device():
    """Open a MeshDevice — real Blackhole or ttsim virtual device."""
    from animatediff_ttnn.ttnn_pipeline import setup_blackhole, _ensure_tt_metal_path
    import ttnn
    from models.demos.vision.generative.stable_diffusion.wormhole.common import SD_L1_SMALL_SIZE

    if args.mode == "sim":
        _ensure_tt_metal_path()
        return ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(1, 1),
            physical_device_ids=[0],
            l1_small_size=SD_L1_SMALL_SIZE,
        )
    else:
        # TTNN SD 1.4 UNet (wormhole-targeted) calls ttnn.to_torch() internally
        # without a mesh_composer — crashes if tensor is sharded across >1 chip.
        # Multi-chip throughput is achieved by running separate processes with
        # --device-id 0/1/2/3 in parallel (one process per chip).
        chip = [args.device_id] if args.device_id is not None else [0]
        return setup_blackhole(device_ids=chip)


def _preview_callback(args):
    """Build the per-step preview callback, or None when --preview-path is unset.

    Kept trivially thin on purpose: the logic (cadence, atomic write, the line
    format consumers parse) lives in `animatediff_ttnn.preview`, which is
    importable and unit-tested without torch or hardware. Fail-soft — if the
    preview module can't be imported for any reason, the run proceeds blind
    rather than not at all.
    """
    if not getattr(args, "preview_path", None):
        return None
    try:
        from animatediff_ttnn.preview import make_step_callback

        return make_step_callback(
            args.preview_path,
            num_steps=args.steps,
            every=getattr(args, "preview_every", None),
        )
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  (previews unavailable: {exc})", file=sys.stderr)
        return None


def run_ttnn():
    from animatediff_ttnn.temporal_attention import generate_frames_temporal
    from animatediff_ttnn.pipeline import export_gif

    backend = "ttsim simulator" if args.mode == "sim" else "Blackhole hardware"
    lightning_tag = " ⚡ Lightning (Euler)" if args.lightning else ""
    print(f"AnimateDiff{lightning_tag} — {backend} (TTNN UNet + cross-frame temporal attention)")
    if args.mode == "sim":
        print(f"  Simulator      : {os.environ.get('TT_METAL_SIMULATOR', '?')}")
    print(f"  Prompt         : {args.prompt}")
    print(f"  Frames         : {args.frames}  Steps: {args.steps}  Seed: {args.seed}")
    print(f"  Temporal alpha : {args.temporal_alpha}")
    if args.lightning:
        print(f"  Scheduler      : EulerDiscrete (trailing, linear)")
        print(f"  CFG            : 7.5 (TTNN path uses base SD 1.4 UNet, not distilled adapter)")
    else:
        print(f"  Scheduler      : PNDM (scaled_linear, skip_prk)")
    if args.chain_from:
        print(f"  Chain from     : {args.chain_from}  (alpha={args.chain_alpha})")
    if args.chain_save:
        print(f"  Chain save     : {args.chain_save}")
    if args.mode == "sim":
        print(f"\n  Note: ttsim is 10–100× slower than silicon.")
    print()

    guidance = 7.5

    print(f"Opening {'simulated ' if args.mode == 'sim' else ''}Blackhole device...")
    device = _open_device()
    print()

    try:
        print("Loading SD 1.4 models...")
        t0 = time.time()
        ttnn_model, ttnn_vae, config, torch_time_proj = load_sd14_ttnn(device)
        print(f"  Loaded in {time.time() - t0:.1f}s\n")

        print("Encoding prompts with CLIP...")
        text_embeddings = encode_prompt(args.prompt, args.negative_prompt)
        print(f"  Embeddings: {text_embeddings.shape}\n")

        print(f"Generating {args.frames} frame(s)...")
        t1 = time.time()
        if args.motion_adapter and args.mode == "blackhole":
            # Validate --motion-adapter-skip keys early so user gets a clear error.
            _VALID_SKIP_KEYS = {"down0", "down1", "down2", "mid", "up0", "up1", "up2"}
            bad_keys = set(args.motion_adapter_skip) - _VALID_SKIP_KEYS
            if bad_keys:
                parser.error(
                    f"--motion-adapter-skip: unknown keys {sorted(bad_keys)}. "
                    f"Valid keys: {sorted(_VALID_SKIP_KEYS)}"
                )
            # Phase 3: MotionAdapter-injected temporal attention
            print(f"  [motion] Loading MotionAdapter from {args.motion_adapter} ...")
            from animatediff_ttnn.motion_weights import load_motion_modules
            temporal_kernels = load_motion_modules(model_id=args.motion_adapter)
            print(f"  [motion] Loaded {sum(len(v) for v in temporal_kernels.values())} modules")
            from animatediff_ttnn.temporal_attention import generate_frames_motion
            frames = generate_frames_motion(
                device=device,
                ttnn_model=ttnn_model,
                ttnn_vae=ttnn_vae,
                config=config,
                torch_time_proj=torch_time_proj,
                text_embeddings=text_embeddings,
                temporal_kernels=temporal_kernels,
                num_frames=args.frames,
                num_steps=args.steps,
                guidance_scale=guidance,
                seed=args.seed,
                use_lightning=args.lightning,
                chain_from=args.chain_from,
                chain_save=args.chain_save,
                chain_alpha=args.chain_alpha,
                injection_alpha=args.motion_adapter_alpha,
                skip_keys=set(args.motion_adapter_skip),
                on_step=_preview_callback(args),
            )
        else:
            # Default path: cross-frame temporal attention (no MotionAdapter)
            frames = generate_frames_temporal(
                device=device,
                ttnn_model=ttnn_model,
                ttnn_vae=ttnn_vae,
                config=config,
                torch_time_proj=torch_time_proj,
                text_embeddings=text_embeddings,
                num_frames=args.frames,
                num_steps=args.steps,
                guidance_scale=guidance,
                seed=args.seed,
                temporal_alpha=args.temporal_alpha,
                use_lightning=args.lightning,
                chain_from=args.chain_from,
                chain_save=args.chain_save,
                chain_alpha=args.chain_alpha,
                on_step=_preview_callback(args),
            )
        elapsed = time.time() - t1
        print(f"  Done in {elapsed:.1f}s ({elapsed / args.frames:.1f}s/frame)\n")
    finally:
        import ttnn
        ttnn.close_mesh_device(device)
        print("Device closed.\n")

    export_gif(frames, args.output)
    print(f"Saved {len(frames)} frame(s) → {args.output}")
    print(f"\nBackend: TTNN UNet spatial denoising on {backend}")
    print(f"         Cross-frame temporal attention (alpha={args.temporal_alpha}): CPU")
    if args.lightning:
        print(f"         Scheduler: EulerDiscrete (trailing, linear) — base TTNN UNet, CFG=7.5")
    print(f"         VAE decode: Blackhole TTNN VAE")


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def main():
    if args.mode == "cpu":
        run_cpu()
    else:
        run_ttnn()


if __name__ == "__main__":
    main()
