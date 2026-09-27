# SPDX-License-Identifier: Apache-2.0
"""Tests for the pinned upstream weights revisions (animatediff_ttnn/weights_pins.py).

These tests are pure CPU and offline. They never import ttnn and never download
anything: every hub call is replaced by a recorder.

Three layers, because a pin can fail at each one:

1. ``revision_for``'s rules: which repo gets which sha, and when the env var applies.
2. The runtime wiring for the loaders that can run without ttnn: the CLIP encoder in
   generation_helpers, and both diffusers pipeline builders. A stub records what each
   ``from_pretrained`` / ``hf_hub_download`` actually received.
3. A source-level sweep over the whole package. Every ``from_pretrained`` and
   ``hf_hub_download`` call must pass ``revision=``. This covers the loads that sit
   behind a ttnn import (``load_sd14_ttnn``) and can't be executed here. It also
   catches a new load added later without a pin.
"""

import ast
from pathlib import Path

import pytest

from animatediff_ttnn import weights_pins as wp

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "animatediff_ttnn"


# ---- 1. the rules ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "sha", [wp.SD14_REVISION, wp.MOTION_ADAPTER_REVISION, wp.LIGHTNING_REVISION]
)
def test_every_pin_is_a_full_40_char_sha(sha):
    assert len(sha) == 40
    int(sha, 16)


def test_sd14_uses_the_pin_when_the_env_var_is_unset(monkeypatch):
    monkeypatch.delenv(wp.WEIGHTS_REVISION_ENV, raising=False)
    assert wp.revision_for(wp.SD14_REPO) == wp.SD14_REVISION


def test_an_empty_env_var_does_not_become_an_empty_revision(monkeypatch):
    monkeypatch.setenv(wp.WEIGHTS_REVISION_ENV, "")
    assert wp.revision_for(wp.SD14_REPO) == wp.SD14_REVISION


def test_the_env_var_overrides_the_sd14_pin(monkeypatch):
    monkeypatch.setenv(wp.WEIGHTS_REVISION_ENV, "f" * 40)
    assert wp.revision_for(wp.SD14_REPO) == "f" * 40


def test_the_env_var_does_not_leak_onto_the_adapter_repos(monkeypatch):
    """The bundle's env var names SD 1.4's revision; applying it to another repo would
    fail to resolve there."""
    monkeypatch.setenv(wp.WEIGHTS_REVISION_ENV, "f" * 40)
    assert wp.revision_for(wp.MOTION_ADAPTER_REPO) == wp.MOTION_ADAPTER_REVISION
    assert wp.revision_for(wp.LIGHTNING_REPO) == wp.LIGHTNING_REVISION


def test_local_dirs_and_unknown_repos_get_no_revision(tmp_path):
    assert wp.revision_for(str(tmp_path)) is None
    assert wp.revision_for("someone/other-model") is None


# ---- 2. runtime wiring (loaders that run without ttnn) --------------------------------


class _Recorder:
    """Stands in for a class with ``from_pretrained``; records every call."""

    def __init__(self, calls, name):
        self.calls, self.name = calls, name

    def from_pretrained(self, repo, **kwargs):
        self.calls.append((self.name, repo, kwargs.get("revision")))
        return _Anything()


class _Anything:
    """Absorbs whatever the loader does with a loaded object."""

    config = {}
    model_max_length = 77

    def __getattr__(self, _):
        return _Anything()

    def __call__(self, *a, **k):
        return _Anything()


@pytest.fixture
def no_env(monkeypatch):
    monkeypatch.delenv(wp.WEIGHTS_REVISION_ENV, raising=False)


def test_the_clip_encoder_loads_sd14_at_the_pinned_revision(monkeypatch, no_env):
    import transformers

    from animatediff_ttnn import generation_helpers as gh

    calls = []
    monkeypatch.setattr(transformers, "CLIPTokenizer", _Recorder(calls, "tokenizer"))
    monkeypatch.setattr(transformers, "CLIPTextModel", _Recorder(calls, "text_encoder"))
    # Force a fresh load: the helpers cache both objects in module globals.
    monkeypatch.setattr(gh, "_clip_tokenizer", None)
    monkeypatch.setattr(gh, "_clip_text_encoder", None)
    try:
        gh._encode_one("a cat")
    except Exception:
        pass  # the stub output isn't a real tensor; only the loads matter here
    assert calls == [
        ("tokenizer", wp.SD14_REPO, wp.SD14_REVISION),
        ("text_encoder", wp.SD14_REPO, wp.SD14_REVISION),
    ]


def test_the_cpu_animatediff_pipeline_pins_both_repos(monkeypatch, no_env):
    from animatediff_ttnn import pipeline as p

    calls = []
    monkeypatch.setattr(p, "MotionAdapter", _Recorder(calls, "adapter"))
    monkeypatch.setattr(p, "DDIMScheduler", _Recorder(calls, "scheduler"))
    monkeypatch.setattr(p, "AnimateDiffPipeline", _Recorder(calls, "pipeline"))
    p.create_animatediff_pipeline()
    assert calls == [
        ("adapter", wp.MOTION_ADAPTER_REPO, wp.MOTION_ADAPTER_REVISION),
        ("scheduler", wp.SD14_REPO, wp.SD14_REVISION),
        ("pipeline", wp.SD14_REPO, wp.SD14_REVISION),
    ]


def test_the_lightning_pipeline_pins_the_checkpoint_and_the_base(monkeypatch, no_env):
    import huggingface_hub
    import safetensors.torch

    from animatediff_ttnn import pipeline as p

    calls = []

    def fake_hf_hub_download(repo, filename, **kwargs):
        calls.append(("lightning", repo, kwargs.get("revision")))
        return "/nonexistent.safetensors"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(safetensors.torch, "load_file", lambda path: {})
    monkeypatch.setattr(p, "MotionAdapter", lambda: _Anything())
    monkeypatch.setattr(p, "AnimateDiffPipeline", _Recorder(calls, "pipeline"))
    monkeypatch.setattr(p, "EulerDiscreteScheduler", _Anything())
    p.create_lightning_pipeline(step=4)
    assert calls == [
        ("lightning", wp.LIGHTNING_REPO, wp.LIGHTNING_REVISION),
        ("pipeline", wp.SD14_REPO, wp.SD14_REVISION),
    ]


# ---- 3. source sweep: no unpinned load anywhere in the package ------------------------


def _hub_calls_without_revision():
    """(file, line, call) for every from_pretrained/hf_hub_download/snapshot_download
    call in animatediff_ttnn/ that has no ``revision=`` keyword."""
    missing, seen = [], 0
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name not in {"from_pretrained", "hf_hub_download", "snapshot_download"}:
                continue
            seen += 1
            if not any(kw.arg == "revision" for kw in node.keywords):
                missing.append(f"{path.relative_to(PACKAGE_DIR.parent)}:{node.lineno} {name}")
    return seen, missing


def test_every_hub_load_in_the_package_passes_a_revision():
    seen, missing = _hub_calls_without_revision()
    # Sanity: the sweep really found the loads. A sweep that matches nothing would
    # pass vacuously. The package has 10 today; a lower bound tolerates additions.
    assert seen >= 10, f"sweep found only {seen} hub calls; has the AST match broken?"
    assert missing == [], "unpinned weights loads:\n  " + "\n  ".join(missing)
