# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# temporal_attention_kernel is built in later tasks; guard the import so that
# tests which only use sim_helpers can import this package without error.
try:
    from .temporal_attention_kernel import TemporalAttentionKernel
    __all__ = ["TemporalAttentionKernel"]
except ImportError:
    __all__ = []
