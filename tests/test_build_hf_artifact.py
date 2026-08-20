# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for scripts/build_hf_artifact.py — artifact completeness and hygiene."""

import json
from pathlib import Path

import pytest

from scripts.build_hf_artifact import DEFAULT_OUT, MODEL_INDEX, build_artifact

ROOT = Path(__file__).resolve().parent.parent


def test_artifact_contains_every_required_file(tmp_path):
    out = build_artifact(out_dir=tmp_path / "hf", load_check=False)
    for relative in (
        "model_index.json",
        "pipeline.py",
        "requirements.txt",
        "README.md",
        "LICENSE",
        "app.py",
        "animatediff_ttnn/__init__.py",
    ):
        assert (out / relative).is_file(), f"missing {relative}"


def test_model_index_is_valid_json_naming_the_pipeline_class(tmp_path):
    out = build_artifact(out_dir=tmp_path / "hf", load_check=False)
    index = json.loads((out / "model_index.json").read_text())
    assert index["_class_name"] == "TTAnimateDiffPipeline"
    assert index["base_model"] == "CompVis/stable-diffusion-v1-4"
    assert index["motion_adapter"] == "guoyww/animatediff-motion-adapter-v1-5-2"
    assert index["code_repo"] == "episod/tt-animatediff"
    assert index["num_frames"] == 8
    assert index == MODEL_INDEX


def test_model_index_keys_all_reach_the_pipeline_signature(tmp_path):
    """Every non-underscore key must be an __init__ parameter, or diffusers
    silently drops it and the default wins instead."""
    import importlib.util
    import inspect
    import sys

    name = "tt_hf_pipeline_for_index_check"
    spec = importlib.util.spec_from_file_location(name, ROOT / "hf" / "pipeline.py")
    module = importlib.util.module_from_spec(spec)
    # The registration is required, not incidental: hf/pipeline.py combines
    # `from __future__ import annotations` with a @dataclass BaseOutput
    # subclass, and dataclass field resolution looks the module up in
    # sys.modules — without this, exec_module raises AttributeError on None.
    # try/finally so the entry does not outlive the test and leak into the
    # rest of the session.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        params = set(
            inspect.signature(module.TTAnimateDiffPipeline.__init__).parameters
        ) - {"self"}
    finally:
        sys.modules.pop(name, None)

    config_keys = {k for k in MODEL_INDEX if not k.startswith("_")}
    assert config_keys <= params, f"unbacked keys: {config_keys - params}"


def test_no_pycache_and_no_invokeai_leak(tmp_path):
    out = build_artifact(out_dir=tmp_path / "hf", load_check=False)
    assert list(out.rglob("__pycache__")) == []
    assert list(out.rglob("*.pyc")) == []
    assert list(out.rglob("invokeai")) == []


def test_no_weights_in_artifact(tmp_path):
    """The repo's Apache-2.0 claim depends on carrying no weights."""
    out = build_artifact(out_dir=tmp_path / "hf", load_check=False)
    for pattern in ("*.pt", "*.ckpt", "*.safetensors", "*.bin"):
        assert list(out.rglob(pattern)) == [], f"weight file matched {pattern}"


def test_rebuild_removes_stale_files(tmp_path):
    out_dir = tmp_path / "hf"
    build_artifact(out_dir=out_dir, load_check=False)
    stale = out_dir / "stale_leftover.py"
    stale.write_text("# removed in a previous version\n")
    build_artifact(out_dir=out_dir, load_check=False)
    assert not stale.exists()


def test_default_out_is_inside_build():
    assert DEFAULT_OUT.parent.name == "build"


def test_built_artifact_loads_through_diffusers(tmp_path):
    """The gate that matters: what we ship is what from_pretrained accepts."""
    from diffusers import DiffusionPipeline

    # load_check left ON here: this is the gate the other tests deliberately skip.
    out = str(build_artifact(out_dir=tmp_path / "hf"))
    pipe = DiffusionPipeline.from_pretrained(
        out, custom_pipeline=out, trust_remote_code=True
    )
    assert type(pipe).__name__ == "TTAnimateDiffPipeline"
    assert pipe.config["temporal_alpha"] == 0.35


def test_missing_copy_source_names_the_file(tmp_path, monkeypatch):
    """A build that cannot find a required source must say which one.

    The copy list is the one place a rename in this repo silently breaks the
    artifact, so the error has to be actionable rather than a bare traceback.
    """
    from scripts import build_hf_artifact

    monkeypatch.setattr(
        build_hf_artifact,
        "COPIES",
        (("does/not/exist.md", "README.md"),),
    )
    with pytest.raises(FileNotFoundError, match="does/not/exist.md"):
        build_hf_artifact.build_artifact(out_dir=tmp_path / "hf", load_check=False)


def test_load_check_reports_through_the_callback_not_stdout(tmp_path, capsys):
    """build_artifact() does no I/O of its own; the CLI opts in via on_load_check."""
    seen = []
    build_artifact(out_dir=tmp_path / "hf", on_load_check=seen.append)

    assert seen and "TTAnimateDiffPipeline" in seen[0]
    captured = capsys.readouterr()
    assert "load check OK" not in captured.out, (
        "build_artifact printed the load check itself; that belongs to the caller"
    )
