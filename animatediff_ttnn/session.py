# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Module-level Blackhole device singleton for library callers.

This module manages the lifetime of the TTNN device and loaded SD 1.4
weights so that callers (e.g. an InvokeAI node, a test harness, or a
script) get a single shared device across repeated generate_animation()
calls without paying the 2-3 minute UNet compile cost more than once.

Usage::

    from animatediff_ttnn.session import ensure_blackhole
    device, (ttnn_model, ttnn_vae, config, time_proj) = ensure_blackhole()

Direct callers should prefer generate_animation() in __init__.py, which
calls ensure_blackhole() automatically.

Thread safety: the initialization lock guarantees that concurrent first
calls (e.g. two InvokeAI nodes racing at startup) serialize correctly.
After the first call returns, subsequent calls are lock-free reads.

Process lifetime: the device is held open until close() is called or the
process exits. TTNN does not support opening the same device twice in one
process, so callers must not open their own device independently when
this module is in use. (app.py manages its own device separately and is
not affected — it runs in its own process.)
"""

import os
import sys
import threading
from pathlib import Path
from typing import Optional, Tuple

_lock = threading.Lock()
_device = None
_models: Optional[Tuple] = None  # (ttnn_model, ttnn_vae, config, torch_time_proj)


def ensure_blackhole(
    mode: str = "blackhole",
    sim_so: Optional[str] = None,
) -> Tuple:
    """Open the Blackhole device and load SD 1.4 model weights (once per process).

    Args:
        mode: "blackhole" — real Blackhole hardware via setup_blackhole();
              "sim"       — ttsim virtual device (requires sim_so path).
        sim_so: Path to libttsim_bh.so. Required when mode="sim".
                Defaults to ~/sim/libttsim_bh.so if not given.

    Returns:
        (device, (ttnn_model, ttnn_vae, config, torch_time_proj))

    Raises:
        FileNotFoundError: sim_so path does not exist.
        RuntimeError: device open failed (hardware not present, driver error, etc.).
    """
    global _device, _models
    if _device is not None:
        return _device, _models

    with _lock:
        # Re-check inside the lock — another thread may have initialized while
        # we were waiting.
        if _device is not None:
            return _device, _models

        if mode == "sim":
            _setup_sim_env(sim_so)

        _ensure_tt_metal_on_path()

        if mode == "sim":
            _device = _open_sim_device()
        else:
            from animatediff_ttnn.ttnn_pipeline import setup_blackhole
            _device = setup_blackhole(device_ids=[0])

        try:
            from animatediff_ttnn.generation_helpers import load_sd14_ttnn
            _models = load_sd14_ttnn(_device)
        except Exception:
            # Reset so future calls don't hit the fast-path and return (device, None).
            _device = None
            raise

        return _device, _models


def close() -> None:
    """Close the TTNN device and release model weights.

    Rarely needed — process exit reclaims all TTNN resources automatically.
    Call this only when you need to release hardware mid-process (e.g. to
    hand the chip back to another owner) and will not call generate_animation()
    again in this process.
    """
    global _device, _models
    with _lock:
        if _device is None:
            return
        try:
            import ttnn
            ttnn.close_mesh_device(_device)
        except Exception:
            pass
        _device = None
        _models = None


# ── private helpers ────────────────────────────────────────────────────────────

def _setup_sim_env(sim_so: Optional[str]) -> None:
    """Validate sim binary path and set required env vars before ttnn import."""
    so = Path(sim_so).expanduser() if sim_so else Path.home() / "sim" / "libttsim_bh.so"
    if not so.exists():
        raise FileNotFoundError(
            f"ttsim binary not found at {so}. "
            "Download a release from https://github.com/tenstorrent/ttsim/releases "
            "or pass sim_so= with the correct path."
        )
    os.environ["TT_METAL_SIMULATOR"] = str(so)
    os.environ.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")
    os.environ.setdefault("TT_METAL_DISABLE_SFPLOADMACRO", "1")
    os.environ.setdefault("TT_METAL_ARCH_NAME", "blackhole")


def _ensure_tt_metal_on_path() -> None:
    """Add ~/tt-metal to sys.path if not already present."""
    tt_metal = Path.home() / "tt-metal"
    if str(tt_metal) not in sys.path:
        sys.path.insert(0, str(tt_metal))


def _open_sim_device():
    """Open a 1×1 MeshDevice against the ttsim virtual chip."""
    import ttnn
    from animatediff_ttnn.ttnn_pipeline import _ensure_tt_metal_path
    from models.demos.vision.generative.stable_diffusion.wormhole.common import SD_L1_SMALL_SIZE

    _ensure_tt_metal_path()
    return ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1),
        physical_device_ids=[0],
        l1_small_size=SD_L1_SMALL_SIZE,
    )
