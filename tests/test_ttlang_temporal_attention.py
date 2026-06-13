# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for TT-Lang temporal attention kernel (simulator-only, no hardware)."""
import sys
import os

_TT_LANG_PYTHON = os.path.expanduser("~/code/tt-lang/python")
if _TT_LANG_PYTHON not in sys.path:
    sys.path.insert(0, _TT_LANG_PYTHON)

import pytest
import torch

pytest.importorskip("sim.ttnnsim")  # skip whole file if tt-lang sim not installed


def test_sim_helpers_roundtrip():
    """tensor_to_block → block_to_tensor is lossless."""
    from animatediff_ttnn.ttlang.sim_helpers import tensor_to_block, block_to_tensor

    TILE = 32
    t = torch.randn(2 * TILE, 3 * TILE)
    block = tensor_to_block(t, shape=(2, 3))
    out = block_to_tensor(block)
    assert out.shape == t.shape
    assert torch.allclose(out, t, atol=1e-5)
