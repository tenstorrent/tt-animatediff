# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Guards for the v6 thin-bundle recipe in ``packaging/``.

The recipe is ``packaging/package-thin.sh`` plus ``packaging/build_inputs.py``. These tests run
without a card, without ``tt-model`` and without network. They cover three things:

* the parts of the recipe that are code (closure walk, import check, dependency wheel, bundle
  inspection), against small synthetic trees;
* the committed inputs (requirements, constraints, guards), against each other and against the
  container's lock, so the two package formats keep serving the same library versions;
* the wiring of the shell script, by running it with a stand-in ``tt-model`` that records the
  arguments it was given. A test of the arithmetic would pass while the script called the tool
  wrongly, so the assertions are on the command the script actually builds.

Nothing here imports ``ttnn``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "packaging"

sys.path.insert(0, str(PKG))
import build_inputs as bi  # noqa: E402


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def _write(root: Path, rel: str, text: str = "") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def _metal_tree(root: Path) -> Path:
    """A miniature tt-metal ``models/`` tree. ``models/`` has no __init__.py, as upstream."""
    _write(root, "models/common/util.py", "import pytest\nimport torch\n\ndef is_blackhole(): ...\n")
    _write(root, "models/common/unused.py", "import requests\n")
    _write(root, "models/demos/sd/__init__.py", "")
    _write(
        root,
        "models/demos/sd/common.py",
        "from models.common.util import is_blackhole\nfrom . import helpers\nimport ttnn\n\nX = 1\n",
    )
    _write(root, "models/demos/sd/helpers.py", "from models.demos.sd.tt import unet\n")
    _write(root, "models/demos/sd/tt/unet.py", "import loguru\nfrom models.common import util\n")
    _write(root, "models/demos/sd/other.py", "import numpy\n")
    return root


def _package(root: Path, body: str) -> Path:
    pkg = root / "mypkg"
    _write(pkg, "__init__.py", "")
    _write(pkg, "run.py", body)
    _write(pkg, "__pycache__/x.pyc", "junk")
    return pkg


# ---------------------------------------------------------------------------------------------
# closure walk
# ---------------------------------------------------------------------------------------------


def test_the_closure_follows_imports_inside_functions_and_ancestors(tmp_path):
    metal = _metal_tree(tmp_path / "metal")
    pkg = _package(tmp_path / "src", "def f():\n    from models.demos.sd.common import X\n")
    version = _write(tmp_path / "src", "VERSION", "1.2.3\n")
    out = tmp_path / "extra"

    report = bi.build_extra_code(pkg, version, metal, out)

    shipped = set(report["models_files"])
    # Reached through a function-level import, a relative import and an absolute one in a dependency.
    assert "models/demos/sd/common.py" in shipped
    assert "models/demos/sd/helpers.py" in shipped
    assert "models/demos/sd/tt/unet.py" in shipped
    assert "models/common/util.py" in shipped
    # Not reached, so not shipped. This is the point of the closure: no other model's code.
    assert "models/common/unused.py" not in shipped
    assert "models/demos/sd/other.py" not in shipped
    assert not (out / "models/common/unused.py").exists()


def test_a_namespace_directory_gets_an_empty_init_and_a_regular_one_is_copied(tmp_path):
    metal = _metal_tree(tmp_path / "metal")
    pkg = _package(tmp_path / "src", "from models.demos.sd.common import X\n")
    version = _write(tmp_path / "src", "VERSION", "1.2.3\n")
    out = tmp_path / "extra"

    report = bi.build_extra_code(pkg, version, metal, out)

    # models/, models/common, models/demos, models/demos/sd/tt have no __init__.py upstream.
    for rel in ("models", "models/common", "models/demos", "models/demos/sd/tt"):
        assert (out / rel / "__init__.py").is_file(), rel
        assert f"{rel}/__init__.py" in report["synthesized_inits"]
    # models/demos/sd has a real one, and it is copied rather than replaced.
    assert "models/demos/sd/__init__.py" in report["models_files"]
    assert "models/demos/sd/__init__.py" not in report["synthesized_inits"]


def test_extra_code_carries_VERSION_and_leaves_out_bytecode(tmp_path):
    metal = _metal_tree(tmp_path / "metal")
    pkg = _package(tmp_path / "src", "from models.common.util import is_blackhole\n")
    version = _write(tmp_path / "src", "VERSION", "9.9.9\n")
    out = tmp_path / "extra"

    bi.build_extra_code(pkg, version, metal, out)

    assert (out / "VERSION").read_text() == "9.9.9\n"
    assert (out / "mypkg/run.py").is_file()
    assert not list(out.rglob("*.pyc"))
    assert not list(out.rglob("__pycache__"))


def test_an_import_missing_from_the_metal_tree_stops_the_build(tmp_path):
    metal = _metal_tree(tmp_path / "metal")
    pkg = _package(tmp_path / "src", "from models.demos.sd.gone import X\n")
    version = _write(tmp_path / "src", "VERSION", "1\n")

    with pytest.raises(bi.ClosureError, match=r"models\.demos\.sd\.gone"):
        bi.build_extra_code(pkg, version, metal, tmp_path / "extra")


def test_third_party_imports_of_the_closure_are_reported(tmp_path):
    metal = _metal_tree(tmp_path / "metal")
    pkg = _package(tmp_path / "src", "from models.demos.sd.common import X\n")
    version = _write(tmp_path / "src", "VERSION", "1\n")

    report = bi.build_extra_code(pkg, version, metal, tmp_path / "extra")

    third = report["third_party_from_models"]
    assert {"pytest", "torch", "ttnn", "loguru"} <= set(third)
    assert "requests" not in third  # only imported by the file that is not shipped
    assert "numpy" not in third


def test_the_extra_code_digest_changes_when_a_shipped_file_changes(tmp_path):
    a = tmp_path / "a"
    _write(a, "x/f.py", "1")
    before = bi.tree_digest(a)
    _write(a, "x/f.py", "2")
    assert bi.tree_digest(a) != before


# ---------------------------------------------------------------------------------------------
# Step 2c: the import check
# ---------------------------------------------------------------------------------------------


def _extra_code_with(tmp_path: Path, files: dict) -> Path:
    root = tmp_path / "extra_code"
    for rel, text in files.items():
        _write(root, rel, text)
    return root


def test_check_reports_a_first_party_import_that_does_not_exist(tmp_path):
    root = _extra_code_with(
        tmp_path,
        {"pkg/__init__.py": "", "pkg/a.py": "from pkg.missing import thing\nimport os\n", "pkg/b.py": ""},
    )
    model = _write(tmp_path, "model.py", "from pkg.b import x\n")

    r = bi.check_imports(root, model, venv_python=None)

    assert any("pkg.missing" in u for u in r["unresolved_first_party"])
    # model.py imports a name pkg.b does not define.
    assert any("pkg.b.x" in u for u in r["unresolved_first_party"])


def test_check_accepts_relative_imports_names_and_submodules(tmp_path):
    root = _extra_code_with(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "from . import b\nfrom .b import thing\nfrom pkg import c\nimport json\n",
            "pkg/b.py": "thing = 1\n",
            "pkg/c.py": "",
        },
    )
    model = _write(tmp_path, "model.py", "from pkg.b import thing\n")

    r = bi.check_imports(root, model, venv_python=None)

    assert r["unresolved_first_party"] == []
    assert r["third_party"] == {}


def test_check_looks_third_party_names_up_in_the_given_interpreter(tmp_path):
    root = _extra_code_with(tmp_path, {"pkg/__init__.py": "", "pkg/a.py": "import json_not_a_real_module_xyz\nimport sys\n"})
    model = _write(tmp_path, "model.py", "")

    r = bi.check_imports(root, model, venv_python=Path(sys.executable))

    assert "json_not_a_real_module_xyz" in r["third_party_not_installed"]


def test_check_treats_the_simulator_imports_as_optional(tmp_path):
    root = _extra_code_with(tmp_path, {"pkg/__init__.py": "", "pkg/a.py": "import sim\n"})
    model = _write(tmp_path, "model.py", "")

    r = bi.check_imports(root, model, venv_python=Path(sys.executable))

    assert "sim" in r["third_party_optional"]
    assert r["third_party_not_installed"] == {}


def test_check_never_imports_what_it_scans(tmp_path):
    """The scan reads source. A module that explodes on import must not be executed."""
    marker = tmp_path / "executed"
    root = _extra_code_with(tmp_path, {"pkg/__init__.py": f"open({str(marker)!r}, 'w').close()\n", "pkg/a.py": "import pkg\n"})
    model = _write(tmp_path, "model.py", "import pkg\n")

    bi.check_imports(root, model, venv_python=None)

    assert not marker.exists()


# ---------------------------------------------------------------------------------------------
# the dependency wheel
# ---------------------------------------------------------------------------------------------


def test_the_deps_wheel_declares_requirements_and_ships_no_code(tmp_path):
    whl = bi.build_deps_wheel("tt-animatediff-deps", "1.2.3", ["diffusers>=0.32,<0.41", "peft>=0.15,<1"], tmp_path)

    assert whl.name == "tt_animatediff_deps-1.2.3-py3-none-any.whl"
    with zipfile.ZipFile(whl) as z:
        names = z.namelist()
        meta = z.read("tt_animatediff_deps-1.2.3.dist-info/METADATA").decode()
    assert all(n.startswith("tt_animatediff_deps-1.2.3.dist-info/") for n in names), names
    assert "Requires-Dist: diffusers>=0.32,<0.41" in meta
    assert "Requires-Dist: peft>=0.15,<1" in meta


def test_the_deps_wheel_is_byte_for_byte_reproducible(tmp_path):
    a = bi.build_deps_wheel("tt-animatediff-deps", "1.0", ["x>=1"], tmp_path / "a")
    b = bi.build_deps_wheel("tt-animatediff-deps", "1.0", ["x>=1"], tmp_path / "b")
    assert hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(b.read_bytes()).digest()


def test_the_deps_wheel_record_matches_its_contents(tmp_path):
    import base64

    whl = bi.build_deps_wheel("tt-animatediff-deps", "1.0", ["x>=1"], tmp_path)
    with zipfile.ZipFile(whl) as z:
        record = z.read("tt_animatediff_deps-1.0.dist-info/RECORD").decode().splitlines()
        for line in record:
            path, digest, size = line.split(",")
            if not digest:
                continue
            body = z.read(path)
            want = base64.urlsafe_b64encode(hashlib.sha256(body).digest()).rstrip(b"=").decode()
            assert digest == f"sha256={want}" and int(size) == len(body), path


def test_a_dependency_list_refuses_option_lines():
    with pytest.raises(SystemExit):
        bi.parse_requirements("diffusers>=1\n--extra-index-url https://example.invalid/simple\n")


def test_a_dependency_list_drops_comments_and_blanks():
    assert bi.parse_requirements("# c\n\nfoo>=1  # why\nbar\n") == ["foo>=1", "bar"]


# ---------------------------------------------------------------------------------------------
# the committed inputs
# ---------------------------------------------------------------------------------------------


def _pins(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s#;]+)", line)
        if m:
            out[m.group(1).lower().replace("_", "-")] = m.group(2)
    return out


def _script_var(name: str) -> str:
    m = re.search(rf'^{name}="([^"]*)"', (PKG / "package-thin.sh").read_text(), re.M)
    assert m, f"{name} not found in package-thin.sh"
    return m.group(1)


def test_requirements_pin_stock_ttnn_and_carry_no_index_options():
    text = (PKG / "requirements.txt").read_text()
    live = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
    assert not [l for l in live if l.startswith("-")], "index or other option lines in requirements.txt"
    assert f"ttnn=={_script_var('TTNN_VERSION')}" in live
    ttnn = [l for l in live if l.startswith("ttnn")]
    assert len(ttnn) == 1 and "+" not in ttnn[0], "ttnn must be a stock release, not a +local build"


def test_requirements_stay_free_of_the_packages_the_cpu_index_replaces():
    """Anything with a wide closure goes through the deps wheel, not step 1."""
    names = {re.split(r"[<>=!~\[ ]", l.strip(), 1)[0].lower() for l in (PKG / "requirements.txt").read_text().splitlines() if l.strip() and not l.strip().startswith("#")}
    assert not names & {"diffusers", "transformers", "huggingface-hub", "huggingface_hub", "requests", "certifi", "urllib3", "peft", "accelerate"}


def test_constraints_pin_every_runtime_dependency_and_stock_ttnn():
    pins = _pins(PKG / "constraints.txt")
    wanted = {re.split(r"[<>=!~\[ ]", l.strip(), 1)[0].lower().replace("_", "-") for l in (PKG / "runtime-deps.txt").read_text().splitlines() if l.strip() and not l.strip().startswith("#")}
    assert not wanted - set(pins), f"not pinned: {sorted(wanted - set(pins))}"
    assert pins["ttnn"] == _script_var("TTNN_VERSION")
    assert pins["torch"].endswith("+cpu"), "torch must be the CPU build the CPU index serves"
    live = [l for l in (PKG / "constraints.txt").read_text().splitlines() if l and not l.startswith("#")]
    assert not [l for l in live if l.startswith("-")], "option lines in constraints.txt"


def test_constraints_do_not_enshrine_the_stale_networking_stack():
    """certifi 2022.12.7, requests 2.28.1 and urllib3 1.26.13 are what the CPU index serves.

    They are what a resolve without this file installs silently. Pinning them here would make
    the known bug permanent.
    """
    pins = _pins(PKG / "constraints.txt")
    assert pins["certifi"] != "2022.12.7"
    assert pins["requests"] != "2.28.1"
    assert pins["urllib3"] != "1.26.13"


def test_validated_pins_agree_with_constraints_and_with_the_container_lock():
    validated = _pins(PKG / "validated-pins.txt")
    final = _pins(PKG / "constraints.txt")
    lock = _pins(ROOT / "tt_model_package.lock")
    assert validated, "validated-pins.txt is empty"
    for name, version in validated.items():
        assert final.get(name) == version, f"constraints.txt has {name}=={final.get(name)}, validated pin is {version}"
        assert lock.get(name) == version, f"tt_model_package.lock has {name}=={lock.get(name)}, validated pin is {version}"
    # torch is the one the container lock names with the same local tag.
    assert lock["torch"] == final["torch"]


def test_pytest_is_shipped_only_while_tt_metal_needs_it():
    """Same wart as the container lock, with the same removal reminder.

    tt-metal v0.77.0's models/common/utility_functions.py has ``import pytest`` at module scope
    and the Stable Diffusion demo imports it. Without pytest the server dies in its lifespan the
    moment the device opens, which no host test sees. When TT_METAL_REF moves to a tag that
    carries tt-metal fdcb5cf2ec6c, delete pytest from runtime-deps.txt and this test.
    """
    ref = _script_var("TT_METAL_REF")
    deps = (PKG / "runtime-deps.txt").read_text().splitlines()
    assert "pytest" in [l.strip() for l in deps], "pytest missing from runtime-deps.txt"
    assert ref in ("v0.77.0", "v0.78.0"), (
        f"TT_METAL_REF moved to {ref}. If that tag carries tt-metal fdcb5cf2ec6c, drop pytest from "
        "runtime-deps.txt, re-run --relock and delete this test. Otherwise extend this list."
    )


def test_the_weights_allow_list_covers_every_subfolder_the_server_loads():
    """If the code starts reading a new SD 1.4 subfolder, the pull must fetch it too.

    A missing file does not fail at install. It fails on the first request, on a card.
    """
    loaded = set()
    for py in (ROOT / "animatediff_ttnn").rglob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "from_pretrained":
                args = [ast.unparse(a) for a in node.args]
                if args and args[0] == "SD14_REPO":
                    for kw in node.keywords:
                        if kw.arg == "subfolder" and isinstance(kw.value, ast.Constant):
                            loaded.add(kw.value.value)
    assert loaded >= {"unet", "vae", "text_encoder", "tokenizer"}, loaded

    script = (PKG / "package-thin.sh").read_text()
    allow = re.search(r"WEIGHTS_ALLOW=\(\n(.*?)\n\)", script, re.S).group(1)
    patterns = re.findall(r'"([^"]+)"', allow)
    for sub in loaded:
        assert any(p.startswith(f"{sub}/") for p in patterns), f"{sub}/ is loaded by the server but not in WEIGHTS_ALLOW"
    # No variant of the big files that the server does not read.
    assert not [p for p in patterns if "fp16" in p or "non_ema" in p or p.endswith(".bin") or p.endswith(".ckpt")]


def test_the_closure_the_code_needs_is_what_the_install_guards_check_for():
    """The guard for models/ names a file the SD demo cannot run without."""
    guards = (PKG / "install-guards.txt").read_text()
    assert "stable_diffusion/wormhole/common.py" in guards
    # Every first-party package in extra_code/ is one a guard can see.
    assert "extra_code/animatediff_ttnn/server/app.py" in guards


# ---------------------------------------------------------------------------------------------
# the install guards, each against a layout it must accept and one it must reject
# ---------------------------------------------------------------------------------------------


def _guards(ttnn_version: str = "0.77.0") -> list:
    out = []
    for line in (PKG / "install-guards.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            out.append(line.replace("@TTNN_VERSION@", ttnn_version))
    return out


def _fake_bundle(tmp_path: Path) -> Path:
    """<bundle>/venv (a real, empty venv) plus the files the guards look for."""
    bundle = tmp_path / "bundle"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(bundle / "venv")], check=True)
    for rel in (
        "model.py",
        "extra_code/animatediff_ttnn/server/app.py",
        "extra_code/models/demos/vision/generative/stable_diffusion/wormhole/common.py",
    ):
        _write(bundle, rel, "")
    return bundle


def _run_guard(bundle: Path, stmt: str, extra_env: dict | None = None, pythonpath: str = "") -> subprocess.CompletedProcess:
    env = {"PATH": os.environ["PATH"], "PYTHONPATH": pythonpath}
    env.update(extra_env or {})
    return subprocess.run([str(bundle / "venv/bin/python"), "-c", stmt], capture_output=True, text=True, env=env, cwd=bundle)


def _site(bundle: Path) -> Path:
    return next((bundle / "venv/lib").glob("python*/site-packages"))


def _fake_dist(site: Path, name: str, version: str) -> None:
    d = site / f"{name}-{version}.dist-info"
    d.mkdir(parents=True, exist_ok=True)
    (d / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")


def test_guard_ttnn_version_accepts_the_stock_release_and_rejects_a_local_build(tmp_path):
    bundle = _fake_bundle(tmp_path)
    guard = _guards()[0]
    _fake_dist(_site(bundle), "ttnn", "0.77.0")
    assert _run_guard(bundle, guard).returncode == 0

    _fake_dist(_site(bundle), "ttnn", "0.77.0+local")
    shutil.rmtree(_site(bundle) / "ttnn-0.77.0.dist-info")
    bad = _run_guard(bundle, guard)
    assert bad.returncode != 0 and "stock" in bad.stderr


def test_guard_ttnn_version_rejects_a_missing_ttnn(tmp_path):
    bundle = _fake_bundle(tmp_path)
    assert _run_guard(bundle, _guards()[0]).returncode != 0


def test_guard_model_code_present_accepts_a_full_bundle_and_rejects_a_gap(tmp_path):
    bundle = _fake_bundle(tmp_path)
    guard = _guards()[1]
    assert _run_guard(bundle, guard).returncode == 0

    (bundle / "extra_code/models/demos/vision/generative/stable_diffusion/wormhole/common.py").unlink()
    bad = _run_guard(bundle, guard)
    assert bad.returncode != 0 and "common.py" in bad.stderr


def test_guard_models_comes_from_extra_code_and_tt_transformers_is_absent(tmp_path):
    bundle = _fake_bundle(tmp_path)
    guard = _guards()[2]
    _write(bundle, "extra_code/models/__init__.py", "")
    ok = _run_guard(bundle, guard, pythonpath=str(bundle / "extra_code"))
    assert ok.returncode == 0, ok.stderr

    # A different `models` wins on the path: reject.
    other = tmp_path / "other"
    _write(other, "models/__init__.py", "")
    shadowed = _run_guard(bundle, guard, pythonpath=f"{other}{os.pathsep}{bundle / 'extra_code'}")
    assert shadowed.returncode != 0 and "extra_code" in shadowed.stderr

    # tt_transformers installed: reject.
    _write(_site(bundle), "tt_transformers/__init__.py", "")
    tt = _run_guard(bundle, guard, pythonpath=str(bundle / "extra_code"))
    assert tt.returncode != 0 and "tt_transformers" in tt.stderr


def test_guard_third_party_names_accepts_a_complete_env_and_names_what_is_missing(tmp_path):
    bundle = _fake_bundle(tmp_path)
    guard = _guards()[3]
    names = re.search(r"need = \[(.*?)\]", guard).group(1)
    for n in re.findall(r"'([^']+)'", names):
        _write(_site(bundle), f"{n}/__init__.py", "")
    assert _run_guard(bundle, guard).returncode == 0

    shutil.rmtree(_site(bundle) / "tt_lib")
    bad = _run_guard(bundle, guard)
    assert bad.returncode != 0 and "tt_lib" in bad.stderr


def test_guard_import_model_accepts_a_clean_import_and_rejects_one_that_loads_ttnn(tmp_path):
    bundle = _fake_bundle(tmp_path)
    guard = _guards()[4]
    _write(bundle, "model.py", "def app(*a): ...\n")
    assert _run_guard(bundle, guard).returncode == 0

    _write(bundle, "model.py", "import sys, types\nsys.modules['ttnn'] = types.ModuleType('ttnn')\ndef app(*a): ...\n")
    bad = _run_guard(bundle, guard)
    assert bad.returncode != 0 and "ttnn" in bad.stderr


def test_every_guard_statement_is_one_line_of_valid_python():
    for g in _guards():
        compile(g, "<guard>", "exec")
        assert "\n" not in g


# ---------------------------------------------------------------------------------------------
# inspecting a staged bundle
# ---------------------------------------------------------------------------------------------

_APP = "model:app"
_WHEEL = "tt_animatediff_deps-0.13.3-py3-none-any.whl"
_REV = "133a221b8aa7292a167afc5127cb63fb5005638b"
_EXPECT = {
    "kind": "tt-dit-server",
    "app": _APP,
    "custom_ops": [],
    "wheels": ["wheels/" + _WHEEL],
    "wheel_names": [_WHEEL],
    "weights_repo": "CompVis/stable-diffusion-v1-4",
    "weights_revision": _REV,
    "must_exist": ["model.py", "constraints.txt", "extra_code/animatediff_ttnn/server/app.py"],
    "must_not_exist": ["wheels/tt_animatediff_models_closure-0.77.0-py3-none-any.whl", "vllm_models", "metal"],
}

_RUN_SH = """#!/usr/bin/env bash
export LD_PRELOAD="x"
export PYTHONPATH="$HERE/extra_code:$HERE:${PYTHONPATH:-}"
CMD=("$PYBIN" -m uvicorn --host 0.0.0.0 --port "${PORT:-8000}" --lifespan on model:app "$@")
"""


def _staged_bundle(root: Path) -> Path:
    """What `tt-model package-thin` stages for this model, written by hand."""
    b = root / "bundle"
    manifest = {
        "schema_version": "6",
        "name": "tt-animatediff",
        "weights": {"repo_id": _EXPECT["weights_repo"], "revision": _REV, "allow_patterns": ["unet/*"]},
        "env": {"ANIMATEDIFF_MESH_SHAPE": "1x1"},
        "entrypoint": None,
        "deps": {
            "python": "3.12.12",
            "requirements": "requirements.txt",
            "wheels": ["wheels/" + _WHEEL],
            "models_wheels": [],
            "vllm": None,
            "extra_code_dir": "extra_code",
            "verify": ["pass"],
            "kind": "tt-dit-server",
            "app": _APP,
        },
    }
    _write(b, "tt_kernel_manifest.json", json.dumps(manifest))
    _write(b, "run.sh", _RUN_SH)
    _write(b, "install.sh", "")
    _write(b, "requirements.txt", "# c\nttnn==0.77.0\ntorch==2.11.0\n")
    _write(b, "constraints.txt", "ttnn==0.77.0\n")
    _write(b, "PROVENANCE.json", "{}")
    _write(b, "model.py", "")
    _write(b, "extra_code/animatediff_ttnn/server/app.py", "")
    _write(b, "wheels/" + _WHEEL, "x")
    return b


def test_inspect_passes_the_canonical_bundle(tmp_path):
    b = _staged_bundle(tmp_path)
    facts, problems = bi.inspect_bundle(b, _EXPECT)
    assert problems == []
    assert facts["deps"]["custom_ops"] is None or facts["deps"]["custom_ops"] == []


@pytest.mark.parametrize(
    "mutate, expect_in_problem",
    [
        (lambda b, m: m["deps"].update(models_wheels=["wheels/tt_animatediff_models_closure-0.77.0-py3-none-any.whl"]), "models_wheels"),
        (lambda b, m: m["deps"].update(custom_ops=["wheels/" + _WHEEL]), "custom_ops"),
        (lambda b, m: m["deps"].update(extra_code_dir=None), "extra_code_dir"),
        (lambda b, m: m["deps"].update(verify=[]), "--verify"),
        (lambda b, m: m["weights"].update(allow_patterns=None), "allow_patterns"),
        (lambda b, m: m["weights"].update(revision="main"), "weights.revision"),
        (lambda b, m: m["deps"].update(app="animatediff_ttnn.server.app:app"), "deps.app"),
    ],
)
def test_inspect_rejects_a_manifest_that_leaves_the_canonical_shape(tmp_path, mutate, expect_in_problem):
    b = _staged_bundle(tmp_path)
    mp = b / "tt_kernel_manifest.json"
    m = json.loads(mp.read_text())
    mutate(b, m)
    mp.write_text(json.dumps(m))

    _, problems = bi.inspect_bundle(b, _EXPECT)

    assert any(expect_in_problem in p for p in problems), problems


def test_inspect_rejects_an_index_option_in_requirements(tmp_path):
    b = _staged_bundle(tmp_path)
    (b / "requirements.txt").write_text("ttnn==0.77.0\n--extra-index-url https://download.pytorch.org/whl/cpu\n")
    _, problems = bi.inspect_bundle(b, _EXPECT)
    assert any("option line" in p for p in problems), problems


def test_inspect_rejects_a_forked_ttnn_pin(tmp_path):
    b = _staged_bundle(tmp_path)
    (b / "requirements.txt").write_text("ttnn==0.77.0+local\n")
    _, problems = bi.inspect_bundle(b, _EXPECT)
    assert any("stock ttnn" in p for p in problems), problems


def test_inspect_rejects_extra_code_that_is_not_first_on_pythonpath(tmp_path):
    b = _staged_bundle(tmp_path)
    (b / "run.sh").write_text(_RUN_SH.replace('"$HERE/extra_code:$HERE:', '"$HERE:$HERE/extra_code:'))
    _, problems = bi.inspect_bundle(b, _EXPECT)
    assert any("PYTHONPATH" in p for p in problems), problems


def test_inspect_rejects_the_old_closure_wheel(tmp_path):
    b = _staged_bundle(tmp_path)
    _write(b, "wheels/tt_animatediff_models_closure-0.77.0-py3-none-any.whl", "x")
    _, problems = bi.inspect_bundle(b, _EXPECT)
    assert any("closure" in p for p in problems), problems


def test_expanding_run_sh_keeps_a_json_argument_as_one_token(tmp_path):
    run = tmp_path / "run.sh"
    run.write_text(
        '#!/usr/bin/env bash\n'
        'CMD=("$PYBIN" serve --additional-config \'{"tt": {"a": 1, "b": "x y"}}\' --port "${PORT:-8000}" "$@")\n'
    )
    assert bi.expand_cmd(run) == ["PY", "serve", "--additional-config", '{"tt": {"a": 1, "b": "x y"}}', "--port", "8000"]


# ---------------------------------------------------------------------------------------------
# the script's wiring, with a stand-in tt-model
# ---------------------------------------------------------------------------------------------

_STUB = r'''#!/usr/bin/env python3
"""Stand-in for tt-model. Records its argv, then lays down the bundle the real one stages."""
import json, os, shutil, sys
from pathlib import Path
argv = sys.argv[1:]
Path(os.environ["STUB_RECORD"]).write_text(json.dumps(argv))
out = Path(argv[argv.index("--out") + 1])
shutil.copytree(os.environ["STUB_BUNDLE"], out)
# The real tool copies --provenance and --extra-code into the bundle; do the same so inspect sees them.
shutil.copy(argv[argv.index("--provenance") + 1], out / "PROVENANCE.json")
'''

_BOOLEAN_FLAGS = {"--no-vllm"}


def _flag_values(argv: list, flag: str) -> list:
    return [argv[i + 1] for i, t in enumerate(argv) if t == flag]


def _positionals(argv: list) -> list:
    """Tokens that are neither a flag nor the value of one. A repo id would show up here."""
    out, i = [], 1  # argv[0] is the subcommand
    while i < len(argv):
        t = argv[i]
        if t.startswith("--"):
            i += 1 if t in _BOOLEAN_FLAGS else 2
        else:
            out.append(t)
            i += 1
    return out


@pytest.fixture()
def script_run(tmp_path):
    """Run packaging/package-thin.sh against a synthetic tt-metal checkout and the stub."""
    if shutil.which("git") is None or shutil.which("uv") is None:
        pytest.skip("git and uv are needed to run the script")
    metal = _metal_tree(tmp_path / "metal")
    # The script wants a real tt-metal commit. Give it a git repo and tell it which one to expect.
    subprocess.run(["git", "init", "-q"], cwd=metal, check=True)
    subprocess.run(["git", "add", "-A"], cwd=metal, check=True)
    subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "m"], cwd=metal, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=metal, check=True, capture_output=True, text=True).stdout.strip()

    # A copy of the repo's package so the script's closure walk sees the synthetic tree, not tt-metal.
    # Only the files the script reads are copied. animatediff_ttnn is replaced by a tiny stand-in.
    repo = tmp_path / "repo"
    shutil.copytree(PKG, repo / "packaging", ignore=shutil.ignore_patterns("__pycache__"))
    _write(repo, "VERSION", "0.13.3\n")
    pkg = repo / "animatediff_ttnn"
    _write(pkg, "__init__.py", "")
    _write(pkg, "server/__init__.py", "")
    _write(pkg, "server/app.py", "from models.demos.sd.common import X\napp = None\n")
    shutil.copy(ROOT / "animatediff_ttnn/weights_pins.py", pkg / "weights_pins.py")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "m"], cwd=repo, check=True)

    # The bundle the stub lays down is the canonical one for the hand-written fixture.
    staged = _staged_bundle(tmp_path / "stub")
    _write(staged, "extra_code/models/demos/vision/generative/stable_diffusion/wormhole/common.py", "")
    _write(staged, "extra_code/VERSION", "0.13.3\n")
    # The deps wheel's name carries the version the script is given.
    stub = tmp_path / "bin" / "tt-model"
    _write(stub.parent, "tt-model", _STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    work = tmp_path / "work"
    record = tmp_path / "argv.json"
    env = dict(
        os.environ,
        WORK=str(work),
        TT_METAL_SRC=str(metal),
        TT_METAL_COMMIT=commit,
        TT_MODEL=str(stub),
        STUB_RECORD=str(record),
        STUB_BUNDLE=str(staged),
    )

    def run(*args: str, **env_over) -> subprocess.CompletedProcess:
        e = dict(env, **env_over)
        return subprocess.run(
            ["bash", str(repo / "packaging/package-thin.sh"), *args], capture_output=True, text=True, env=e, cwd=repo
        )

    run.record = record
    run.work = work
    run.repo = repo
    return run


def test_the_script_stages_without_a_repo_id_and_with_every_canonical_flag(script_run):
    r = script_run()
    assert r.returncode == 0, r.stdout + r.stderr
    argv = json.loads(script_run.record.read_text())

    assert argv[0] == "package-thin"
    # Stage only: a positional repo id would push.
    assert _positionals(argv) == [], _positionals(argv)
    assert "--out" in argv

    # Canonical shape, one flag at a time.
    assert _flag_values(argv, "--kind") == ["tt-dit-server"]
    assert _flag_values(argv, "--app") == ["model:app"]
    assert len(_flag_values(argv, "--extra-code")) == 1
    assert len(_flag_values(argv, "--constraints")) == 1
    assert len(_flag_values(argv, "--provenance")) == 1
    assert len(_flag_values(argv, "--requirements")) == 1
    assert _flag_values(argv, "--weights") == ["CompVis/stable-diffusion-v1-4"]
    assert _flag_values(argv, "--weights-revision") == ["133a221b8aa7292a167afc5127cb63fb5005638b"]
    assert _flag_values(argv, "--arch") == ["blackhole"]
    assert _flag_values(argv, "--tt-metal-version") == ["0.77.0"]
    assert _flag_values(argv, "--env") == ["ANIMATEDIFF_MESH_SHAPE=1x1"]
    assert len(_flag_values(argv, "--weights-allow")) >= 8
    assert len(_flag_values(argv, "--verify")) == len(_guards())

    # One --ops-wheel, and it is the dependency carrier. No models wheel, no plugin or vLLM wheel.
    ops = _flag_values(argv, "--ops-wheel")
    assert len(ops) == 1 and Path(ops[0]).name.startswith("tt_animatediff_deps-0.13.3-")
    assert "--models-wheel" not in argv
    assert "--plugin-wheel" not in argv and "--vllm-wheel" not in argv
    assert "--no-vllm" in argv


def test_the_script_hands_tt_model_the_inputs_it_built_from_this_checkout(script_run):
    r = script_run()
    assert r.returncode == 0, r.stdout + r.stderr
    argv = json.loads(script_run.record.read_text())

    extra = Path(_flag_values(argv, "--extra-code")[0])
    assert (extra / "animatediff_ttnn/server/app.py").is_file()
    assert (extra / "models/demos/sd/common.py").is_file()
    assert (extra / "VERSION").read_text().strip() == "0.13.3"
    assert not (extra / "models/common/unused.py").exists()

    prov = json.loads(Path(_flag_values(argv, "--provenance")[0]).read_text())
    assert prov["version"] == "0.13.3"
    assert prov["tt_metal_code"]["tag"] == "v0.77.0"
    assert prov["weights"]["revision"] == "133a221b8aa7292a167afc5127cb63fb5005638b"
    assert len(prov["source_commit"]) == 40

    constraints = Path(_flag_values(argv, "--constraints")[0])
    assert constraints.read_text() == (PKG / "constraints.txt").read_text()

    # The guards that reach tt-model have the ttnn version filled in.
    verify = _flag_values(argv, "--verify")
    assert not [v for v in verify if "@TTNN_VERSION@" in v]
    assert any("== '0.77.0'" in v for v in verify)


@pytest.mark.parametrize("arg", ["episod/tt-animatediff", "--push", "someone/else"])
def test_the_script_refuses_anything_that_looks_like_a_repo_id(script_run, arg):
    r = script_run(arg)
    assert r.returncode == 2
    assert not script_run.record.exists(), "tt-model must not be called"


def test_the_script_refuses_a_tt_metal_checkout_at_the_wrong_commit(script_run):
    r = script_run(TT_METAL_COMMIT="0" * 40)
    assert r.returncode != 0
    assert "expected" in r.stderr
    assert not script_run.record.exists()


def test_the_script_stops_when_an_import_does_not_resolve(script_run):
    app = script_run.repo / "animatediff_ttnn/server/app.py"
    app.write_text(app.read_text() + "from animatediff_ttnn.gone import X\n")
    r = script_run()
    assert r.returncode != 0
    assert "does not resolve" in r.stderr
    assert not script_run.record.exists()


def test_the_script_stops_when_the_staged_bundle_is_not_canonical(script_run, tmp_path):
    # Break the bundle the stub lays down: a models wheel in the manifest.
    staged = tmp_path / "stub" / "bundle"
    mp = staged / "tt_kernel_manifest.json"
    m = json.loads(mp.read_text())
    m["deps"]["models_wheels"] = ["wheels/tt_animatediff_models_closure-0.77.0-py3-none-any.whl"]
    mp.write_text(json.dumps(m))

    r = script_run()

    assert r.returncode != 0
    assert "not canonical" in r.stderr or "PROBLEM" in r.stderr


def test_the_script_never_names_a_push_command():
    text = (PKG / "package-thin.sh").read_text()
    code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    assert "hf upload" not in code
    assert "tt-model push" not in code
    assert "huggingface-cli upload" not in code
    assert "gozer" not in text.lower()
