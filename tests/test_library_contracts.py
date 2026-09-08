# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for the two library-level findings from #9's review.

Both are about a value the package trusts without checking: an env var parsed at import
time, and a mode string passed through unvalidated.
"""
import subprocess
import sys

import pytest


@pytest.mark.parametrize("value", ["", "auto", "1.5", " ", "-"])
def test_importing_the_package_survives_a_junk_cache_env_var(value):
    """`import animatediff_ttnn` must not raise because of an env var.

    ANIMATEDIFF_CPU_PIPE_CACHE was parsed with a bare int() at module scope, so an empty
    or non-numeric value made the IMPORT raise ValueError. That breaks things far from the
    CPU cache it configures: tt-model-manager's verify step imports
    animatediff_ttnn.server.app at image build time, hf/pipeline.py's resolve_package
    catches ImportError and would see a ValueError instead, and the CLI dies on import.
    A Dockerfile or a Space settings panel exporting an empty value is enough.

    Run in a subprocess because the module is already imported in this process.
    """
    code = "import animatediff_ttnn; print(animatediff_ttnn.__name__)"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "ANIMATEDIFF_CPU_PIPE_CACHE": value,
                            "PYTHONPATH": ".", "HOME": "/tmp"})
    assert r.returncode == 0, (
        f"ANIMATEDIFF_CPU_PIPE_CACHE={value!r} broke the import:\n{r.stderr[-1500:]}"
    )


def test_a_junk_cache_env_var_falls_back_to_the_documented_default():
    """Falling back silently is right here, but it must land on the documented 1."""
    from animatediff_ttnn import _cpu_pipe_cache_max

    assert _cpu_pipe_cache_max({"ANIMATEDIFF_CPU_PIPE_CACHE": ""}) == 1
    assert _cpu_pipe_cache_max({"ANIMATEDIFF_CPU_PIPE_CACHE": "auto"}) == 1
    assert _cpu_pipe_cache_max({"ANIMATEDIFF_CPU_PIPE_CACHE": "0"}) == 1, "floor is 1"
    assert _cpu_pipe_cache_max({"ANIMATEDIFF_CPU_PIPE_CACHE": "-3"}) == 1
    assert _cpu_pipe_cache_max({"ANIMATEDIFF_CPU_PIPE_CACHE": "4"}) == 4
    assert _cpu_pipe_cache_max({}) == 1


@pytest.mark.parametrize("mode", ["CPU", "cuda", "", "blackhole ", "sim2", "Auto"])
def test_an_unknown_mode_is_refused_rather_than_treated_as_blackhole(mode):
    """An unrecognised mode must not fall through to opening hardware.

    _resolve_mode returned any non-"auto" string unchanged, generate_animation only
    special-cased "cpu" and ensure_blackhole only "sim" -- so mode="CPU" (wrong case) or
    "cuda" skipped the CPU branch and went on to claim chip 0 and start a multi-minute
    Blackhole run. On a shared box that takes a chip nobody meant to take. hf/pipeline.py
    validates its own VALID_MODES precisely because the library did not.
    """
    from animatediff_ttnn import _resolve_mode

    with pytest.raises(ValueError) as exc:
        _resolve_mode(mode)
    assert repr(mode) in str(exc.value) or mode in str(exc.value)
    assert "auto" in str(exc.value) and "blackhole" in str(exc.value)


@pytest.mark.parametrize("mode", ["cpu", "blackhole", "sim"])
def test_the_documented_modes_still_resolve(mode):
    from animatediff_ttnn import _resolve_mode

    assert _resolve_mode(mode) == mode
