# SPDX-License-Identifier: Apache-2.0
"""Streaming behaviour of the Gradio app's `generate()` generator.

`app.py` had **no test coverage at all**, which is how a refactor briefly
deleted `height, width = 512, 512` while leaving two references to them — a
NameError on every blackhole/sim generation that `py_compile` cannot see and no
test existed to catch. These tests exercise both branches of `generate()` with
the heavy pieces (the diffusers pipeline, the TTNN device) faked out, so they
run anywhere in milliseconds.

`gradio` is not installed in every environment this suite runs in, so it is
stubbed rather than imported — nothing here needs the real thing.
"""

import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# Imported at module scope, like every other test file here: importing these
# lazily inside a test re-enters torch's package init and trips a circular
# import.
import animatediff_ttnn.pipeline as pipeline  # noqa: E402
import animatediff_ttnn.temporal_attention as ta  # noqa: E402
import animatediff_ttnn.generation_helpers as helpers  # noqa: E402


@pytest.fixture()
def app_mod(monkeypatch):
    """Import `app.py` with `gradio` stubbed out."""
    if "gradio" not in sys.modules:
        gr = types.ModuleType("gradio")

        class _Error(Exception):
            pass

        gr.Error = _Error
        # `app.py` builds its UI at import time; give the builders enough shape
        # to be constructed and discarded.
        class _Any:
            def __init__(self, *a, **k): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def __getattr__(self, _n): return _Any()
            def __call__(self, *a, **k): return _Any()

        # Any attribute the UI reaches for resolves to the same inert widget,
        # so this stub can't drift as app.py's layout changes.
        gr.__getattr__ = lambda _name: _Any
        gr.themes = types.SimpleNamespace(Soft=_Any, Base=_Any)
        monkeypatch.setitem(sys.modules, "gradio", gr)

    # Import once — reloading re-enters torch's package init and trips a
    # circular import, and the gradio stub is already in place above.
    import importlib
    return importlib.import_module("app")


def _drain(gen, limit=200):
    out = []
    for i, item in enumerate(gen):
        out.append(item)
        if i > limit:
            break
    return out


# ── CPU branch ───────────────────────────────────────────────────────────────

def test_cpu_mode_streams_previews_then_the_final_gif(app_mod, monkeypatch, tmp_path):
    """CPU used to run blind and yield once. `pipeline.generate()` now takes an
    `on_step`, so every mode streams — and the UI's own "preview updates stream
    in real time" blurb is true for all three."""
    monkeypatch.setattr(app_mod, "_ensure_cpu_pipeline", lambda **k: object())

    def fake_generate(pipe, prompt, *, num_inference_steps, on_step=None, **kw):
        for i in range(num_inference_steps):
            if on_step is not None:
                on_step(i, num_inference_steps, ["latent"])
        return ["frame"]

    monkeypatch.setattr(pipeline, "generate", fake_generate)
    monkeypatch.setattr(pipeline, "export_gif",
                        lambda frames, path: Path(path).write_bytes(b"GIF89a"))
    # The preview render is exercised by tests/test_preview.py; keep it cheap.
    monkeypatch.setattr("animatediff_ttnn.preview._latent_preview_fn",
                        lambda: (lambda fl, h, w: ["f"]))
    monkeypatch.setattr("animatediff_ttnn.preview._export_gif_fn",
                        lambda: (lambda frames, path: Path(path).write_bytes(b"GIF89a")))

    yielded = _drain(app_mod.generate(
        mode="cpu", prompt="a koi pond", negative_prompt="", frames=2, steps=4,
        seed=1, temporal_alpha=0.35, sim_path="",
    ))

    assert len(yielded) > 1, "CPU mode must stream, not yield only the final GIF"
    assert yielded[-1].endswith("output.gif")
    assert all(p.endswith(".gif") for p in yielded)


def test_cpu_lightning_overrides_the_step_count(app_mod, monkeypatch):
    """Distilled CPU checkpoints only support 2/4/8 steps, so the slider is
    ignored in favour of lightning_steps — preserved through the refactor."""
    seen = {}
    monkeypatch.setattr(app_mod, "_ensure_cpu_pipeline", lambda **k: object())

    def fake_generate(pipe, prompt, *, num_inference_steps, on_step=None, **kw):
        seen["steps"] = num_inference_steps
        return ["frame"]

    monkeypatch.setattr(pipeline, "generate", fake_generate)
    monkeypatch.setattr(pipeline, "export_gif", lambda frames, path: None)

    _drain(app_mod.generate(
        mode="cpu", prompt="x", negative_prompt="", frames=2, steps=25, seed=1,
        temporal_alpha=0.35, sim_path="", lightning=True, lightning_steps=4,
    ))
    assert seen["steps"] == 4


# ── Blackhole / sim branch ───────────────────────────────────────────────────

def test_blackhole_branch_resolves_every_name_it_uses(app_mod, monkeypatch):
    """THE regression this file exists for.

    A refactor removed `height, width = 512, 512` and left `height=height` /
    `width=width` in the generate_frames_temporal call. `py_compile` is blind to
    it and nothing else touched this path, so it would have shipped as a
    NameError on every Space (sim) generation.
    """

    monkeypatch.setattr(app_mod, "_ensure_bh_device",
                        lambda mode, sim_path: (object(), (1, 2, 3, 4)))
    monkeypatch.setattr(helpers, "encode_prompt", lambda p, n: "emb")

    seen = {}

    def fake_gft(**kw):
        seen.update(kw)
        on_step = kw.get("on_step")
        if on_step is not None:
            on_step(0, kw["num_steps"], ["latent"])
        return ["frame"]

    monkeypatch.setattr(ta, "generate_frames_temporal", fake_gft)
    monkeypatch.setattr(pipeline, "export_gif",
                        lambda frames, path: Path(path).write_bytes(b"GIF89a"))
    monkeypatch.setattr("animatediff_ttnn.preview._latent_preview_fn",
                        lambda: (lambda fl, h, w: ["f"]))
    monkeypatch.setattr("animatediff_ttnn.preview._export_gif_fn",
                        lambda: (lambda frames, path: Path(path).write_bytes(b"GIF89a")))

    yielded = _drain(app_mod.generate(
        mode="sim", prompt="a koi pond", negative_prompt="", frames=2, steps=2,
        seed=1, temporal_alpha=0.35, sim_path="/tmp/libttsim.so",
    ))

    assert yielded, "sim mode produced nothing"
    # The OUTPUT size, which is what these two names are for — distinct from the
    # preview size, which the preview module chooses for itself.
    assert seen["height"] == 512 and seen["width"] == 512
    assert seen["on_step"] is not None
