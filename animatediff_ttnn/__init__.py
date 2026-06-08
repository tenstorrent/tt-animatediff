# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""AnimateDiff for Tenstorrent hardware.

Phase 1 (CPU baseline):
    from animatediff_ttnn.pipeline import create_animatediff_pipeline, generate, export_gif

Phase 2 (Blackhole hardware):
    from animatediff_ttnn.ttnn_pipeline import setup_blackhole, build_tlist, generate_frames
"""

try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("animatediff-ttnn")
    except PackageNotFoundError:
        # Editable / source checkout: fall back to repo-root VERSION file
        from pathlib import Path
        __version__ = (Path(__file__).parent.parent / "VERSION").read_text().strip()
except Exception:
    __version__ = "unknown"

from .pipeline import create_animatediff_pipeline, generate, export_gif

__all__ = [
    "create_animatediff_pipeline",
    "generate",
    "export_gif",
]
