#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Shim — delegates to generate.py --mode sim.

Kept for backward compatibility with docs that reference this script by name.
Use generate.py directly for new workflows:
    python examples/generate.py --mode sim --frames 2 --steps 4
"""

import sys
from pathlib import Path

sys.argv = [sys.argv[0], "--mode", "sim"] + sys.argv[1:]
exec(compile(open(Path(__file__).parent / "generate.py").read(), "generate.py", "exec"))
