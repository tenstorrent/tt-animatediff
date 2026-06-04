#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Shim — delegates to generate.py --mode blackhole.

Kept for backward compatibility with VS Code toolkit commands and docs that
reference this script by name. Use generate.py directly for new workflows.
"""

import runpy
import sys
from pathlib import Path

sys.argv = [sys.argv[0], "--mode", "blackhole"] + sys.argv[1:]
runpy.run_path(str(Path(__file__).parent / "generate.py"), run_name="__main__")
