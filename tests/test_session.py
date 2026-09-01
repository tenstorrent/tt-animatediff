# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for the animatediff_ttnn.session device singleton — no hardware.

session.ensure_blackhole() holds the TTNN device and compiled SD 1.4 weights for
the lifetime of the process, because opening the device and compiling the UNet
costs tens of seconds. That makes it process-global mutable state, and the two
things that go wrong with process-global state are both covered here:

  1. Caching. A second call must reuse the open device and must not re-open it
     or reload weights.
  2. Failure recovery. If weight loading raises *after* the device opened, the
     module must not be left holding a device with no models. The original bug
     returned (device, None) from every subsequent call -- a poisoned singleton
     that survived until process exit, so one transient load failure broke
     generation permanently.

Every test resets the module globals via the `clean_session` fixture, so the
suite never leaks a fake device into a later test.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from animatediff_ttnn import session


@pytest.fixture(autouse=True)
def clean_session():
    """Reset the module-level device/model globals around every test."""
    session._device = None
    session._models = None
    yield
    session._device = None
    session._models = None


@pytest.fixture
def fake_backend():
    """Stub out device opening and weight loading.

    ensure_blackhole() imports both lazily inside the function, so the patches
    target the modules that define them.
    """
    device = MagicMock(name="device")
    models = (MagicMock(name="unet"), MagicMock(name="vae"),
              MagicMock(name="config"), MagicMock(name="time_proj"))

    with patch("animatediff_ttnn.ttnn_pipeline.setup_blackhole",
               return_value=device) as setup, \
         patch("animatediff_ttnn.generation_helpers.load_sd14_ttnn",
               return_value=models) as load, \
         patch.object(session, "_ensure_tt_metal_on_path"):
        yield {"setup": setup, "load": load, "device": device, "models": models}


# ---------------------------------------------------------------------------
# Happy path and caching
# ---------------------------------------------------------------------------

def test_first_call_opens_device_and_loads_models(fake_backend):
    device, models = session.ensure_blackhole(mode="blackhole")

    assert device is fake_backend["device"]
    assert models is fake_backend["models"]
    fake_backend["setup"].assert_called_once_with(device_ids=[0])
    fake_backend["load"].assert_called_once_with(fake_backend["device"])


def test_second_call_reuses_the_open_device(fake_backend):
    """Re-opening the device or recompiling the UNet would cost ~10 s per call."""
    first = session.ensure_blackhole(mode="blackhole")
    second = session.ensure_blackhole(mode="blackhole")

    assert first[0] is second[0]
    assert first[1] is second[1]
    assert fake_backend["setup"].call_count == 1
    assert fake_backend["load"].call_count == 1


def test_cached_call_ignores_a_changed_mode(fake_backend):
    """Once a device is open the mode argument cannot change it.

    This documents real behaviour rather than endorsing it: a caller who opens
    blackhole and then asks for sim in the same process gets the blackhole
    device back. Switching backends requires close() first.
    """
    session.ensure_blackhole(mode="blackhole")
    device, _ = session.ensure_blackhole(mode="sim", sim_so="/nonexistent.so")

    assert device is fake_backend["device"]
    assert fake_backend["setup"].call_count == 1


# ---------------------------------------------------------------------------
# Failure recovery -- the poisoned-singleton regression
# ---------------------------------------------------------------------------

def test_model_load_failure_does_not_leave_a_poisoned_device(fake_backend):
    """Regression guard: a failed load must clear _device, not cache it.

    Before the fix, _device stayed set while _models stayed None, so the
    fast-path at the top of ensure_blackhole() returned (device, None) forever
    and every later generate_animation() call crashed unpacking that None.
    """
    fake_backend["load"].side_effect = RuntimeError("L1 allocation failed")

    with pytest.raises(RuntimeError, match="L1 allocation failed"):
        session.ensure_blackhole(mode="blackhole")

    assert session._device is None, "device must be cleared so the next call retries"
    assert session._models is None


def test_retry_after_a_load_failure_succeeds(fake_backend):
    """A transient load failure must not be permanent."""
    fake_backend["load"].side_effect = [
        RuntimeError("transient OOM"),
        fake_backend["models"],
    ]

    with pytest.raises(RuntimeError, match="transient OOM"):
        session.ensure_blackhole(mode="blackhole")

    device, models = session.ensure_blackhole(mode="blackhole")
    assert device is fake_backend["device"]
    assert models is fake_backend["models"]
    assert fake_backend["setup"].call_count == 2, "device must be re-opened on retry"


def test_load_failure_propagates_the_original_exception(fake_backend):
    """The caller needs the real error, not a wrapped or swallowed one."""
    boom = ValueError("bad weights checksum")
    fake_backend["load"].side_effect = boom

    with pytest.raises(ValueError) as exc:
        session.ensure_blackhole(mode="blackhole")

    assert exc.value is boom


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------

def test_close_resets_state(fake_backend):
    session.ensure_blackhole(mode="blackhole")
    ttnn_mock = MagicMock()

    with patch.dict(sys.modules, {"ttnn": ttnn_mock}):
        session.close()

    assert session._device is None
    assert session._models is None
    ttnn_mock.close_mesh_device.assert_called_once_with(fake_backend["device"])


def test_close_is_a_noop_when_no_device_is_open():
    ttnn_mock = MagicMock()
    with patch.dict(sys.modules, {"ttnn": ttnn_mock}):
        session.close()  # must not raise
    ttnn_mock.close_mesh_device.assert_not_called()


def test_close_still_clears_state_if_ttnn_close_raises(fake_backend):
    """A failing close must not strand the globals -- otherwise the process is
    stuck with a device it believes is open and cannot reopen."""
    session.ensure_blackhole(mode="blackhole")
    ttnn_mock = MagicMock()
    ttnn_mock.close_mesh_device.side_effect = RuntimeError("device already gone")

    with patch.dict(sys.modules, {"ttnn": ttnn_mock}):
        session.close()  # swallows the error by design

    assert session._device is None
    assert session._models is None


def test_reopen_after_close(fake_backend):
    session.ensure_blackhole(mode="blackhole")
    with patch.dict(sys.modules, {"ttnn": MagicMock()}):
        session.close()
    session.ensure_blackhole(mode="blackhole")

    assert fake_backend["setup"].call_count == 2


# ---------------------------------------------------------------------------
# sim mode validation
# ---------------------------------------------------------------------------

def test_sim_mode_rejects_a_missing_binary(tmp_path):
    """Fail before touching the device, with an actionable message."""
    missing = tmp_path / "libttsim_bh.so"

    with pytest.raises(FileNotFoundError) as exc:
        session.ensure_blackhole(mode="sim", sim_so=str(missing))

    message = str(exc.value)
    assert str(missing) in message
    assert "github.com/tenstorrent/ttsim/releases" in message
    assert session._device is None


def test_sim_mode_sets_env_vars_for_an_existing_binary(tmp_path, monkeypatch):
    """The simulator needs these set *before* ttnn is imported."""
    so = tmp_path / "libttsim_bh.so"
    so.write_bytes(b"\x7fELF fake")

    for var in ("TT_METAL_SIMULATOR", "TT_METAL_SLOW_DISPATCH_MODE",
                "TT_METAL_DISABLE_SFPLOADMACRO", "TT_METAL_ARCH_NAME"):
        monkeypatch.delenv(var, raising=False)

    session._setup_sim_env(str(so))

    import os
    assert os.environ["TT_METAL_SIMULATOR"] == str(so)
    assert os.environ["TT_METAL_SLOW_DISPATCH_MODE"] == "1"
    assert os.environ["TT_METAL_DISABLE_SFPLOADMACRO"] == "1"
    assert os.environ["TT_METAL_ARCH_NAME"] == "blackhole"


def test_sim_env_does_not_clobber_a_caller_set_arch(tmp_path, monkeypatch):
    """The optional vars use setdefault, so an explicit choice must survive."""
    so = tmp_path / "libttsim_bh.so"
    so.write_bytes(b"\x7fELF fake")
    monkeypatch.setenv("TT_METAL_ARCH_NAME", "wormhole_b0")

    session._setup_sim_env(str(so))

    import os
    assert os.environ["TT_METAL_ARCH_NAME"] == "wormhole_b0"


def test_sim_mode_defaults_to_home_sim_path(monkeypatch, tmp_path):
    """With no sim_so, the default is ~/sim/libttsim_bh.so."""
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

    with pytest.raises(FileNotFoundError) as exc:
        session._setup_sim_env(None)

    assert str(tmp_path / "sim" / "libttsim_bh.so") in str(exc.value)


# ---------------------------------------------------------------------------
# sys.path handling
# ---------------------------------------------------------------------------

def test_tt_metal_is_added_to_sys_path_once(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    tt_metal = str(tmp_path / "tt-metal")
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != tt_metal])

    session._ensure_tt_metal_on_path()
    session._ensure_tt_metal_on_path()

    assert sys.path.count(tt_metal) == 1
