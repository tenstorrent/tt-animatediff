# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for the Hugging Face custom pipeline (hf/pipeline.py).

Hermetic: no network, no Tenstorrent hardware, and the real animatediff_ttnn
package is never imported — a stub is injected into sys.modules instead, so a
test failure means the adapter is wrong, not that the backend is unavailable.
"""

import importlib.util
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


def _stub_package(monkeypatch, name="animatediff_ttnn"):
    """Install a stub animatediff_ttnn that records generate_animation calls."""
    mod = types.ModuleType(name)
    mod.calls = []

    def generate_animation(**kwargs):
        mod.calls.append(kwargs)
        n = kwargs.get("num_frames") or 1
        return [f"pil-frame-{i}" for i in range(n)]

    mod.generate_animation = generate_animation
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def test_resolve_package_prefers_installed(pipeline_module, monkeypatch):
    stub = _stub_package(monkeypatch)
    assert pipeline_module.resolve_package(None, None) is stub


def test_resolve_package_falls_back_to_load_source(pipeline_module, monkeypatch, tmp_path):
    """When the package is not installed, the directory from_pretrained loaded
    from (config._name_or_path) is put on sys.path."""
    pkg = tmp_path / "animatediff_ttnn"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("MARKER = 'vendored'\n")

    def not_installed():
        exc = ModuleNotFoundError("No module named 'animatediff_ttnn'")
        exc.name = "animatediff_ttnn"
        raise exc

    monkeypatch.setattr(pipeline_module, "_import_package", not_installed)
    monkeypatch.setattr(pipeline_module, "_import_from_root", lambda root: root)

    assert pipeline_module.resolve_package(None, str(tmp_path)) == tmp_path


def test_resolve_package_downloads_vendored_code_as_last_resort(
    pipeline_module, monkeypatch, tmp_path
):
    snapshot = tmp_path / "snap"
    (snapshot / "animatediff_ttnn").mkdir(parents=True)
    (snapshot / "animatediff_ttnn" / "__init__.py").write_text("MARKER = 'snapshot'\n")

    def not_installed():
        exc = ModuleNotFoundError("No module named 'animatediff_ttnn'")
        exc.name = "animatediff_ttnn"
        raise exc

    monkeypatch.setattr(pipeline_module, "_import_package", not_installed)
    monkeypatch.setattr(pipeline_module, "_import_from_root", lambda root: root)
    seen = {}

    def fake_snapshot_download(repo_id, allow_patterns=None, **kw):
        seen["repo_id"] = repo_id
        seen["allow_patterns"] = allow_patterns
        return str(snapshot)

    monkeypatch.setattr(pipeline_module, "_snapshot_download", fake_snapshot_download)

    assert pipeline_module.resolve_package("episod/tt-animatediff", None) == snapshot
    assert seen["repo_id"] == "episod/tt-animatediff"
    assert seen["allow_patterns"] == ["animatediff_ttnn/**"]


def test_resolve_package_raises_with_install_hint(pipeline_module, monkeypatch):
    def not_installed():
        exc = ModuleNotFoundError("No module named 'animatediff_ttnn'")
        exc.name = "animatediff_ttnn"
        raise exc

    monkeypatch.setattr(pipeline_module, "_import_package", not_installed)
    monkeypatch.setattr(pipeline_module, "_import_from_root", lambda root: None)
    monkeypatch.setattr(
        pipeline_module,
        "_snapshot_download",
        lambda *a, **k: (_ for _ in ()).throw(OSError("offline")),
    )
    with pytest.raises(ImportError) as excinfo:
        pipeline_module.resolve_package("episod/tt-animatediff", None)
    assert "pip install" in str(excinfo.value)
    assert "tt-animatediff" in str(excinfo.value)


def test_resolve_package_propagates_import_error_from_transitive_dep(
    pipeline_module, monkeypatch
):
    """If animatediff_ttnn is installed but has a broken transitive dependency,
    that ModuleNotFoundError (naming the *dependency*, not the package) must
    propagate — layers 2 and 3 cannot fix an installed-but-broken package.
    Verify the error is re-raised, not swallowed and fallen through to _snapshot_download."""

    def broken_import():
        # Simulate: animatediff_ttnn imports ttnn; ttnn import fails
        exc = ModuleNotFoundError("No module named 'ttnn'")
        exc.name = "ttnn"  # The failing module is ttnn, not animatediff_ttnn
        raise exc

    monkeypatch.setattr(pipeline_module, "_import_package", broken_import)
    fail_if_called = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("_snapshot_download should not be called")
    )
    monkeypatch.setattr(pipeline_module, "_snapshot_download", fail_if_called)

    # The error from the transitive dependency should propagate out,
    # not be swallowed so that layer 3 gets called.
    with pytest.raises(ModuleNotFoundError) as excinfo:
        pipeline_module.resolve_package("episod/tt-animatediff", None)
    assert str(excinfo.value) == "No module named 'ttnn'"


def test_call_delegates_with_config_defaults(pipeline_module, monkeypatch):
    stub = _stub_package(monkeypatch)
    monkeypatch.setattr(pipeline_module, "_ttnn_available", lambda: False)
    pipe = pipeline_module.TTAnimateDiffPipeline()

    out = pipe("a swirling nebula")

    assert len(stub.calls) == 1
    call = stub.calls[0]
    assert call["prompt"] == "a swirling nebula"
    assert call["num_frames"] == 8          # from config
    assert call["num_steps"] == 25          # from config
    assert call["guidance_scale"] == 7.5    # from config
    assert call["temporal_alpha"] == 0.35   # from config
    assert call["mode"] == "cpu"            # auto with no ttnn
    assert call["seed"] == 42
    assert out.frames == [f"pil-frame-{i}" for i in range(8)]


def test_call_explicit_arguments_win_over_config(pipeline_module, monkeypatch):
    stub = _stub_package(monkeypatch)
    monkeypatch.setattr(pipeline_module, "_ttnn_available", lambda: False)
    pipe = pipeline_module.TTAnimateDiffPipeline()

    pipe("x", num_frames=4, num_steps=6, guidance_scale=1.0, temporal_alpha=0.9, seed=7)

    call = stub.calls[0]
    assert (call["num_frames"], call["num_steps"]) == (4, 6)
    assert call["guidance_scale"] == 1.0
    assert call["temporal_alpha"] == 0.9
    assert call["seed"] == 7


def test_call_forwards_chain_and_lightning_arguments(pipeline_module, monkeypatch):
    stub = _stub_package(monkeypatch)
    monkeypatch.setattr(pipeline_module, "_ttnn_available", lambda: True)
    pipe = pipeline_module.TTAnimateDiffPipeline()

    pipe(
        "x",
        negative_prompt="blurry",
        use_lightning=True,
        lightning_steps=8,
        chain_from="prev.pt",
        chain_save="next.pt",
        chain_alpha=0.4,
        height=384,
        width=384,
    )

    call = stub.calls[0]
    assert call["negative_prompt"] == "blurry"
    assert call["use_lightning"] is True
    assert call["lightning_steps"] == 8
    assert call["chain_from"] == "prev.pt"
    assert call["chain_save"] == "next.pt"
    assert call["chain_alpha"] == 0.4
    assert (call["height"], call["width"]) == (384, 384)


def test_auto_selects_blackhole_when_ttnn_present(pipeline_module, monkeypatch):
    stub = _stub_package(monkeypatch)
    monkeypatch.setattr(pipeline_module, "_ttnn_available", lambda: True)
    pipe = pipeline_module.TTAnimateDiffPipeline()

    pipe("x")

    assert stub.calls[0]["mode"] == "blackhole"
    assert pipe.resolved_mode == "blackhole"


def test_explicit_blackhole_without_ttnn_raises(pipeline_module, monkeypatch):
    stub = _stub_package(monkeypatch)
    monkeypatch.setattr(pipeline_module, "_ttnn_available", lambda: False)
    pipe = pipeline_module.TTAnimateDiffPipeline()

    with pytest.raises(RuntimeError, match="ttnn"):
        pipe("x", mode="blackhole")
    assert stub.calls == [], "must not silently fall back to CPU"


def test_invalid_mode_raises_value_error(pipeline_module, monkeypatch):
    _stub_package(monkeypatch)
    pipe = pipeline_module.TTAnimateDiffPipeline()
    with pytest.raises(ValueError, match="mode"):
        pipe("x", mode="cuda")


def test_resolved_mode_is_none_before_first_call(pipeline_module):
    assert pipeline_module.TTAnimateDiffPipeline().resolved_mode is None


def test_output_type_np_stacks_frames(pipeline_module, monkeypatch):
    stub = _stub_package(monkeypatch)
    monkeypatch.setattr(pipeline_module, "_ttnn_available", lambda: False)

    def generate_animation(**kwargs):
        from PIL import Image

        stub.calls.append(kwargs)
        return [Image.new("RGB", (8, 8)) for _ in range(2)]

    stub.generate_animation = generate_animation
    pipe = pipeline_module.TTAnimateDiffPipeline()

    out = pipe("x", num_frames=2, output_type="np")

    assert out.frames.shape == (2, 8, 8, 3)


def test_call_forwards_only_real_generate_animation_kwargs():
    """__call__'s whole job is forwarding kwargs into the real
    animatediff_ttnn.generate_animation(). Every other test in this file
    stubs that package with a **kwargs-only fake, so a misspelled or renamed
    kwarg would pass the entire suite and only raise TypeError on a real
    user's first call.

    This test imports the REAL animatediff_ttnn (in-repo, no device opened,
    no network touched — see CLAUDE.md) and cross-checks its actual
    inspect.signature() against the keyword arguments hf/pipeline.py's
    __call__ actually passes to generate_animation(...).

    The forwarded set is derived by parsing pipeline.py's source with `ast`
    and reading the keywords off the real generate_animation(...) call node
    — not by re-typing the argument list into this test. A hand-copied list
    is exactly the kind of thing that goes stale the next time __call__
    changes; parsing the actual call cannot drift out of sync with it. If
    someone renames e.g. ``temporal_alpha=`` to ``temporal_alpha_value=`` in
    the call, or a future animatediff_ttnn drops/renames a parameter, the
    forwarded set and the real signature stop overlapping and this test
    fails — whichever side changed.
    """
    import ast
    import inspect

    import animatediff_ttnn

    tree = ast.parse(PIPELINE_SRC.read_text())

    call_node = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "generate_animation"
        ):
            call_node = node
            break
    assert call_node is not None, (
        "could not find a `....generate_animation(...)` call in hf/pipeline.py "
        "— has __call__ stopped delegating to it?"
    )

    forwarded = {kw.arg for kw in call_node.keywords if kw.arg is not None}
    assert forwarded, "found the generate_animation(...) call but it passes no keyword arguments"

    real_params = set(inspect.signature(animatediff_ttnn.generate_animation).parameters)
    unbacked = forwarded - real_params
    assert not unbacked, (
        f"hf/pipeline.py forwards keyword(s) {sorted(unbacked)} that are not "
        f"parameters of the real animatediff_ttnn.generate_animation "
        f"signature ({sorted(real_params)}). A caller's first real generation "
        "would fail with TypeError even though the (stub-based) test suite "
        "is green."
    )

    # A floor, not just a non-empty check: catches the call being replaced by
    # something that forwards a nearly-empty, technically-non-empty set.
    # hf/pipeline.py's __call__ forwards 15 keyword arguments today
    # (prompt, negative_prompt, num_frames, num_steps, guidance_scale, seed,
    # temporal_alpha, height, width, mode, use_lightning, lightning_steps,
    # chain_from, chain_save, chain_alpha). Update this count if that set
    # deliberately changes.
    assert len(forwarded) == 15, (
        f"expected 15 forwarded keyword arguments, found {len(forwarded)}: "
        f"{sorted(forwarded)}"
    )


def test_init_opens_no_device_and_generates_nothing(pipeline_module, monkeypatch):
    """Constructing the pipeline must not import ttnn, open a device, or generate."""
    stub = _stub_package(monkeypatch)
    touched = []
    monkeypatch.setattr(pipeline_module, "_ttnn_available", lambda: touched.append("ttnn"))
    monkeypatch.setattr(
        pipeline_module, "resolve_package", lambda *a: touched.append("resolve") or stub
    )

    pipeline_module.TTAnimateDiffPipeline()

    assert touched == []
    assert stub.calls == []
