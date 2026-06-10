# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""HuggingFace Space entry point.

Sets SPACE_MODE=sim so the UI defaults to ttsim mode, then runs the
shared app.py from the repo root.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("SPACE_MODE", "sim")

# Add repo root (parent of spaces/) to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import demo  # noqa: E402 — must come after sys.path manipulation

if __name__ == "__main__":
    demo.launch()
