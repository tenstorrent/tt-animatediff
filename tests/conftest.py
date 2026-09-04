# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixtures, plus a guard against tt-lang's global torch dtype rebinding.

Why this file exists
--------------------
``tests/test_ttlang_temporal_attention.py`` calls
``pytest.importorskip("sim.ttnnsim")`` at module scope. Importing tt-lang's
functional simulator has a *process-wide* side effect: ``sim/ttnnsim.py`` runs

    for _attr in _PROMOTABLE_FLOAT_DTYPES:      # {"bfloat16", "float16"}
        setattr(torch, _attr, torch.float32)

at import time, so that native PyTorch calls such as
``torch.randn(dtype=torch.bfloat16)`` transparently use float32 on hosts without
hardware bfloat16. That is reasonable for the simulator, but it rebinds
attributes on the shared ``torch`` module for every other test in the session.

The concrete damage is to ``torch.load``. ``torch.storage._dtype_to_storage_type_map()``
is a dict literal keyed by dtype:

    {..., torch.float: "FloatStorage", ..., torch.bfloat16: "BFloat16Storage", ...}

Once ``torch.bfloat16 is torch.float32``, those two entries collapse onto one key
and the later one wins, so the map has 16 entries instead of 17 and
``"FloatStorage"`` is *gone*. Both that map and its inverse are ``lru_cache``d, so
the corruption is sticky. Any subsequent ``torch.load()`` of a float32 tensor then
dies with ``KeyError: 'FloatStorage'``.

pytest imports every test module during collection, before running any test, so
this used to fail 10 tests in ``test_chain_blend.py`` and ``test_distill_*.py``
even though those files are collected *first* and pass in isolation.

Note we cannot simply call tt-lang's own ``set_disable_float32_promotion(True)``
to undo this: when tt-lang's optional ``greenlet`` dependency is missing the
import raises *after* the rebinding loop has already run, and Python then drops
the half-initialised module from ``sys.modules`` — the rebinding survives but the
function that would reverse it is unreachable. So we snapshot the real dtypes
here instead. This module is imported by pytest before any test module, which is
what makes the snapshot trustworthy.
"""

import pytest
import torch

#: The genuine dtype objects, captured before any test module (and therefore
#: before tt-lang's simulator) has had a chance to rebind them.
_NATIVE_FLOAT_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

#: The one module that legitimately wants tt-lang's float32 promotion in effect:
#: it is testing the simulator itself, so leave its environment alone.
_SIMULATOR_TEST_MODULE = "test_ttlang_temporal_attention"


def _restore_native_float_dtypes() -> bool:
    """Undo tt-lang's ``torch.bfloat16``/``torch.float16`` rebinding.

    Returns True if anything actually had to be repaired, in which case the
    ``lru_cache``d storage-type maps in ``torch.storage`` are also dropped so
    they get rebuilt from the corrected attributes.
    """
    repaired = False
    for name, native in _NATIVE_FLOAT_DTYPES.items():
        if getattr(torch, name, native) is not native:
            setattr(torch, name, native)
            repaired = True

    if repaired:
        for cached in (
            torch.storage._dtype_to_storage_type_map,
            torch.storage._storage_type_to_dtype_map,
        ):
            cache_clear = getattr(cached, "cache_clear", None)
            if cache_clear is not None:
                cache_clear()

    return repaired


@pytest.fixture(autouse=True)
def native_torch_float_dtypes(request):
    """Ensure each test sees real ``torch.bfloat16``/``torch.float16``.

    Skipped for the tt-lang simulator tests, which are exercising the promotion
    behaviour on purpose.
    """
    if request.node.module.__name__.endswith(_SIMULATOR_TEST_MODULE):
        yield
        return

    _restore_native_float_dtypes()
    yield
