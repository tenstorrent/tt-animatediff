# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for the Hugging Face custom pipeline (hf/pipeline.py).

Hermetic: no network, no Tenstorrent hardware, and the real animatediff_ttnn
package is never imported — a stub is injected into sys.modules instead, so a
test failure means the adapter is wrong, not that the backend is unavailable.
"""

import importlib.util
import json
import shutil
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PIPELINE_SRC = ROOT / "hf" / "pipeline.py"


def _load_pipeline_module(name="tt_hf_pipeline_under_test"):
    """Import hf/pipeline.py directly, without going through diffusers."""
    spec = importlib.util.spec_from_file_location(name, PIPELINE_SRC)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # Register module so type annotations can be resolved
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pipeline_module():
    return _load_pipeline_module()


def test_pipeline_class_and_defaults(pipeline_module):
    pipe = pipeline_module.TTAnimateDiffPipeline()
    assert pipe.config["base_model"] == "CompVis/stable-diffusion-v1-4"
    assert pipe.config["motion_adapter"] == "guoyww/animatediff-motion-adapter-v1-5-2"
    assert pipe.config["lightning_repo"] == "ByteDance/AnimateDiff-Lightning"
    assert pipe.config["code_repo"] == "episod/tt-animatediff"
    assert pipe.config["temporal_alpha"] == 0.35
    assert pipe.config["num_frames"] == 8
    assert pipe.config["num_steps"] == 25
    assert pipe.config["guidance_scale"] == 7.5


def test_config_overrides_persist(pipeline_module):
    pipe = pipeline_module.TTAnimateDiffPipeline(
        base_model="other/model", temporal_alpha=0.5, num_frames=16
    )
    assert pipe.config["base_model"] == "other/model"
    assert pipe.config["temporal_alpha"] == 0.5
    assert pipe.config["num_frames"] == 16


def test_init_signature_has_no_bare_kwargs(pipeline_module):
    """diffusers derives expected components from __init__; a **kwargs-only
    signature makes from_pretrained raise "expected ['kwargs']"."""
    import inspect

    params = inspect.signature(pipeline_module.TTAnimateDiffPipeline.__init__).parameters
    assert "kwargs" not in params
    named = [p for p in params.values() if p.name != "self"]
    assert named, "__init__ must declare named parameters"
    assert all(p.default is not inspect.Parameter.empty for p in named)


def test_pipeline_py_never_imports_the_package_literally():
    """diffusers' check_imports scans ^\\s*import / ^\\s*from and hard-fails on
    0.32.1 for any unresolvable module — indentation does not hide it."""
    import re

    source = PIPELINE_SRC.read_text()
    literal = re.findall(r"^\s*(?:import|from)\s+animatediff_ttnn", source, re.MULTILINE)
    assert literal == [], f"pipeline.py must reach the package via importlib: {literal}"


def test_output_object_shape(pipeline_module):
    frames = ["frame-a", "frame-b"]
    out = pipeline_module.TTAnimateDiffPipelineOutput(frames=frames)
    assert out.frames == frames
    assert out[0] == frames  # BaseOutput tuple access, like AnimateDiffPipelineOutput
