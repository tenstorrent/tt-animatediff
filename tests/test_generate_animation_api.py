# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for the generate_animation() public API — CPU-only, no hardware.

generate_animation() is the single entry point external callers use, so its
*routing* behaviour is the contract that matters: which backend a given mode
selects, and which arguments get forwarded to that backend. The heavy lifting
(the actual denoising) is covered by the phase-specific tests; here we mock the
backends out entirely and assert on the dispatch.

What these tests lock down:

  1. Mode resolution — "auto" consults TTNN availability, explicit modes are
     passed through untouched.
  2. CPU routing translates Lightning settings correctly. This is easy to get
     wrong because the Lightning checkpoint has guidance_scale baked in, so
     generate_animation() must override num_steps *and* guidance_scale rather
     than forwarding the caller's values.
  3. Blackhole/sim routing forwards every chain-mode and temporal argument.
     A dropped kwarg here silently disables chain continuity, which is the
     kind of bug that only shows up as "the animation doesn't look chained".
  4. The CPU pipeline cache is keyed on (use_lightning, lightning_steps) so a
     second call with the same settings reuses the ~60 s model load, and a call
     with different settings does not collide with it.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

import animatediff_ttnn
from animatediff_ttnn import _resolve_mode, _ttnn_available, generate_animation


# ---------------------------------------------------------------------------
# Mode resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("explicit", ["cpu", "blackhole", "sim"])
def test_explicit_mode_is_passed_through(explicit):
    """Only "auto" is resolved; a named backend must never be second-guessed."""
    assert _resolve_mode(explicit) == explicit


def test_auto_resolves_to_blackhole_when_ttnn_importable():
    with patch.object(animatediff_ttnn, "_ttnn_available", return_value=True):
        assert _resolve_mode("auto") == "blackhole"


def test_auto_resolves_to_cpu_when_ttnn_missing():
    with patch.object(animatediff_ttnn, "_ttnn_available", return_value=False):
        assert _resolve_mode("auto") == "cpu"


def test_ttnn_available_false_when_import_raises():
    """_ttnn_available() must report False rather than propagating ImportError."""
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def _no_ttnn(name, *args, **kwargs):
        if name == "ttnn":
            raise ImportError("no ttnn here")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_no_ttnn):
        assert _ttnn_available() is False


def test_ttnn_available_true_when_import_succeeds():
    with patch.dict(sys.modules, {"ttnn": MagicMock()}):
        assert _ttnn_available() is True


# ---------------------------------------------------------------------------
# CPU routing
# ---------------------------------------------------------------------------

def test_cpu_mode_routes_to_generate_cpu():
    sentinel = ["frame0", "frame1"]
    with patch.object(animatediff_ttnn, "_generate_cpu", return_value=sentinel) as gen_cpu:
        out = generate_animation(prompt="a nebula", mode="cpu", num_frames=2)

    assert out is sentinel
    assert gen_cpu.call_args.kwargs["prompt"] == "a nebula"
    assert gen_cpu.call_args.kwargs["num_frames"] == 2


def test_cpu_mode_forwards_plain_steps_and_guidance_without_lightning():
    with patch.object(animatediff_ttnn, "_generate_cpu", return_value=[]) as gen_cpu:
        generate_animation(
            prompt="p", mode="cpu", num_steps=25, guidance_scale=7.5, use_lightning=False
        )

    kwargs = gen_cpu.call_args.kwargs
    assert kwargs["num_steps"] == 25
    assert kwargs["guidance_scale"] == 7.5


def test_cpu_lightning_overrides_steps_and_guidance():
    """The Lightning checkpoint bakes in guidance_scale=1.0 and a fixed step count.

    Forwarding the caller's num_steps=25 / guidance_scale=7.5 would degrade the
    output badly, so generate_animation() must substitute lightning_steps and 1.0.
    """
    with patch.object(animatediff_ttnn, "_generate_cpu", return_value=[]) as gen_cpu:
        generate_animation(
            prompt="p",
            mode="cpu",
            num_steps=25,
            guidance_scale=7.5,
            use_lightning=True,
            lightning_steps=4,
        )

    kwargs = gen_cpu.call_args.kwargs
    assert kwargs["num_steps"] == 4, "Lightning must use lightning_steps, not num_steps"
    assert kwargs["guidance_scale"] == 1.0, "Lightning has CFG baked in; must force 1.0"
    assert kwargs["lightning_steps"] == 4


# ---------------------------------------------------------------------------
# Blackhole / sim routing
# ---------------------------------------------------------------------------

@pytest.fixture
def mocked_blackhole_backend():
    """Patch out the device session, prompt encoder, and denoising loop.

    generate_animation() imports these lazily *inside* the function body, so the
    patches must target the defining modules rather than animatediff_ttnn.
    """
    frames = ["bh-frame"]
    device = MagicMock(name="device")
    models = (MagicMock(name="unet"), MagicMock(name="vae"), MagicMock(name="config"),
              MagicMock(name="time_proj"))

    with patch("animatediff_ttnn.session.ensure_blackhole",
               return_value=(device, models)) as ensure, \
         patch("animatediff_ttnn.generation_helpers.encode_prompt",
               return_value="EMBEDDINGS") as encode, \
         patch("animatediff_ttnn.temporal_attention.generate_frames_temporal",
               return_value=frames) as gen:
        yield {"ensure": ensure, "encode": encode, "gen": gen,
               "frames": frames, "device": device, "models": models}


def test_blackhole_mode_routes_to_temporal_generator(mocked_blackhole_backend):
    out = generate_animation(prompt="a nebula", mode="blackhole")

    assert out is mocked_blackhole_backend["frames"]
    mocked_blackhole_backend["ensure"].assert_called_once_with(
        mode="blackhole", sim_so=None
    )


def test_sim_mode_passes_sim_so_through(mocked_blackhole_backend):
    generate_animation(prompt="p", mode="sim", sim_so="/tmp/libttsim_bh.so")

    mocked_blackhole_backend["ensure"].assert_called_once_with(
        mode="sim", sim_so="/tmp/libttsim_bh.so"
    )


def test_blackhole_mode_encodes_prompt_and_negative_prompt(mocked_blackhole_backend):
    generate_animation(prompt="a nebula", negative_prompt="blurry", mode="blackhole")

    mocked_blackhole_backend["encode"].assert_called_once_with("a nebula", "blurry")


def test_blackhole_mode_forwards_chain_and_temporal_arguments(mocked_blackhole_backend):
    """Every chain/temporal knob must reach generate_frames_temporal().

    A silently dropped kwarg here would not raise -- it would just produce an
    un-chained or un-blended animation, which is very hard to spot by eye.
    """
    on_step = object()
    generate_animation(
        prompt="p",
        mode="blackhole",
        num_frames=8,
        num_steps=25,
        guidance_scale=7.5,
        seed=1234,
        temporal_alpha=0.42,
        height=384,
        width=640,
        use_lightning=True,
        chain_from="/tmp/prev.pt",
        chain_save="/tmp/next.pt",
        chain_alpha=0.55,
        on_step=on_step,
    )

    kwargs = mocked_blackhole_backend["gen"].call_args.kwargs
    assert kwargs["num_frames"] == 8
    assert kwargs["num_steps"] == 25
    assert kwargs["guidance_scale"] == 7.5
    assert kwargs["seed"] == 1234
    assert kwargs["temporal_alpha"] == 0.42
    assert kwargs["height"] == 384
    assert kwargs["width"] == 640
    assert kwargs["use_lightning"] is True
    assert kwargs["chain_from"] == "/tmp/prev.pt"
    assert kwargs["chain_save"] == "/tmp/next.pt"
    assert kwargs["chain_alpha"] == 0.55
    assert kwargs["on_step"] is on_step


def test_blackhole_mode_passes_loaded_models_through(mocked_blackhole_backend):
    """The session's cached device/models must be handed to the generator as-is."""
    generate_animation(prompt="p", mode="blackhole")

    kwargs = mocked_blackhole_backend["gen"].call_args.kwargs
    ttnn_model, ttnn_vae, config, torch_time_proj = mocked_blackhole_backend["models"]
    assert kwargs["device"] is mocked_blackhole_backend["device"]
    assert kwargs["ttnn_model"] is ttnn_model
    assert kwargs["ttnn_vae"] is ttnn_vae
    assert kwargs["config"] is config
    assert kwargs["torch_time_proj"] is torch_time_proj
    assert kwargs["text_embeddings"] == "EMBEDDINGS"


def test_blackhole_mode_does_not_touch_the_cpu_path(mocked_blackhole_backend):
    with patch.object(animatediff_ttnn, "_generate_cpu") as gen_cpu:
        generate_animation(prompt="p", mode="blackhole")
    gen_cpu.assert_not_called()


# ---------------------------------------------------------------------------
# CPU pipeline cache
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_cpu_cache():
    """Isolate the module-level CPU pipeline cache for one test."""
    saved = dict(animatediff_ttnn._cpu_pipe_cache)
    animatediff_ttnn._cpu_pipe_cache.clear()
    yield animatediff_ttnn._cpu_pipe_cache
    animatediff_ttnn._cpu_pipe_cache.clear()
    animatediff_ttnn._cpu_pipe_cache.update(saved)


def test_cpu_pipeline_is_built_once_and_reused(clean_cpu_cache):
    """A second call with identical settings must not rebuild the ~60 s pipeline."""
    with patch("animatediff_ttnn.pipeline.create_animatediff_pipeline") as create, \
         patch("animatediff_ttnn.pipeline.generate", return_value=[]):
        create.return_value = MagicMock(name="pipe")
        generate_animation(prompt="p", mode="cpu")
        generate_animation(prompt="p", mode="cpu")

    assert create.call_count == 1
    assert list(clean_cpu_cache) == [(False, 4)]


def test_cpu_cache_separates_lightning_from_standard(clean_cpu_cache):
    """Lightning and standard pipelines are different objects and must not collide."""
    with patch("animatediff_ttnn.pipeline.create_animatediff_pipeline") as create_std, \
         patch("animatediff_ttnn.pipeline.create_lightning_pipeline") as create_light, \
         patch("animatediff_ttnn.pipeline.generate", return_value=[]):
        create_std.return_value = MagicMock(name="std")
        create_light.return_value = MagicMock(name="light")
        generate_animation(prompt="p", mode="cpu", use_lightning=False)
        generate_animation(prompt="p", mode="cpu", use_lightning=True, lightning_steps=8)

    assert create_std.call_count == 1
    assert create_light.call_count == 1
    create_light.assert_called_once_with(step=8)
    assert set(clean_cpu_cache) == {(False, 4), (True, 8)}


# ---------------------------------------------------------------------------
# Exported surface
# ---------------------------------------------------------------------------

def test_public_api_names_are_exported():
    """__all__ is the documented contract for `from animatediff_ttnn import *`."""
    for name in ("generate_animation", "export_gif", "export_mp4",
                 "create_animatediff_pipeline", "generate"):
        assert name in animatediff_ttnn.__all__, f"{name} missing from __all__"
        assert hasattr(animatediff_ttnn, name), f"{name} not importable"


# ── _ttnn_available robustness (PR #7 review) ──────────────────────────────


def test_ttnn_available_survives_non_import_errors():
    """A broken tt-metal install must read as "no backend", not crash mode="auto".

    A missing shared library surfaces as OSError, not ImportError. Catching only
    ImportError made generate_animation(mode="auto") propagate that instead of
    falling back to CPU — the opposite of its documented contract.
    """
    import builtins

    import animatediff_ttnn

    real_import = builtins.__import__

    def _raise_for_ttnn(name, *args, **kwargs):
        if name == "ttnn":
            raise OSError("libtt_metal.so: cannot open shared object file")
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", _raise_for_ttnn):
        assert animatediff_ttnn._ttnn_available() is False


def test_ttnn_available_still_false_on_plain_import_error():
    """The ordinary "not installed" case must keep working."""
    import builtins

    import animatediff_ttnn

    real_import = builtins.__import__

    def _raise_for_ttnn(name, *args, **kwargs):
        if name == "ttnn":
            raise ImportError("No module named 'ttnn'")
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", _raise_for_ttnn):
        assert animatediff_ttnn._ttnn_available() is False


def test_auto_mode_falls_back_to_cpu_when_ttnn_import_raises_oserror():
    """End-to-end: mode="auto" routes to the CPU path, not an OSError traceback."""
    import builtins

    import animatediff_ttnn

    real_import = builtins.__import__

    def _raise_for_ttnn(name, *args, **kwargs):
        if name == "ttnn":
            raise OSError("libtt_metal.so: cannot open shared object file")
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", _raise_for_ttnn):
        assert animatediff_ttnn._resolve_mode("auto") == "cpu"
