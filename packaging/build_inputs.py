#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Helpers for ``packaging/package-thin.sh``: the parts of the v6 thin recipe that are Python.

The shell script is the recipe. This file holds the steps that need real parsing, so they can be
unit-tested (``tests/test_packaging_thin.py``) instead of living as untested shell:

``extra-code``   Lay out ``extra_code/``: this repo's ``animatediff_ttnn`` package, plus the smallest
                 slice of tt-metal's ``models/`` tree that package imports (the Stable Diffusion
                 demo UNet and VAE). This replaces the old ``tt_animatediff_models_closure`` wheel,
                 which carried all of ``models/common`` (about 150 files, including other models'
                 code) to serve one model.
``check``        Prove every import under ``extra_code/`` and ``model.py`` resolves. This is Step 2c
                 of the ``tt-model-thin-package`` skill. It reads source only and never imports
                 ``ttnn`` or opens a device.
``deps-wheel``   Build the small metadata-only wheel that carries this model's pip dependencies
                 (diffusers, transformers, and so on). See ``package-thin.sh`` for why it exists.
``provenance``   Write ``PROVENANCE.json`` for ``--provenance``.
``inspect``      Read a staged bundle and fail if it is not the canonical shape (Step 3).

Standard library only, so it runs before any venv exists.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# The tt-metal package this model borrows code from. Everything under it is first-party for the
# purpose of closure, so it must be found in the tt-metal tree or the build stops.
METAL_TOP = "models"

# Imports that exist only for development and never run on the serve path. Each one is guarded or
# lives in a function that serving does not reach. `check` reports them as "optional" instead of
# "unresolved". Keep this list short and give a reason for every entry.
OPTIONAL_THIRD_PARTY = {
    # tt-lang's functional simulator. animatediff_ttnn/ttlang/ is the simulator track, covered by
    # simulator tests only. Nothing the server imports reaches it.
    "sim": "tt-lang functional simulator (animatediff_ttnn.ttlang, test-only)",
    "ttl": "tt-lang DSL (animatediff_ttnn.ttlang, test-only)",
}


# --------------------------------------------------------------------------------------------
# Import scanning
# --------------------------------------------------------------------------------------------


class Import:
    """One import statement, resolved to an absolute dotted name where possible."""

    __slots__ = ("module", "names", "path", "line", "module_level")

    def __init__(self, module: str, names: List[str], path: Path, line: int, module_level: bool):
        self.module = module  # absolute dotted module ("a.b"); "" if unresolvable
        self.names = names  # names after `from module import ...`; [] for a plain import
        self.path = path
        self.line = line
        self.module_level = module_level  # True when it runs at import time (not in a function)


def _package_parts(py: Path, root: Path) -> List[str]:
    """The dotted package a file belongs to, as parts, relative to ``root``."""
    rel = py.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    return parts[:-1]  # a module's package is its directory, for __init__ too


def scan_imports(py: Path, root: Path) -> List[Import]:
    """Every import in ``py``, including ones inside functions and ``try`` blocks.

    Relative imports are resolved against ``root`` so the caller always sees absolute names.
    ``module_level`` is True for imports that execute when the module is imported, which is what
    decides whether a missing dependency breaks ``import`` or only a later call.
    """
    tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
    pkg = _package_parts(py, root)
    out: List[Import] = []

    def walk(node: ast.AST, in_func: bool) -> None:
        for child in ast.iter_child_nodes(node):
            nested = in_func or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            if isinstance(child, ast.Import):
                for alias in child.names:
                    out.append(Import(alias.name, [], py, child.lineno, not in_func))
            elif isinstance(child, ast.ImportFrom):
                if child.level:
                    base = pkg[: len(pkg) - (child.level - 1)] if child.level - 1 <= len(pkg) else []
                    mod = ".".join(base + ([child.module] if child.module else []))
                else:
                    mod = child.module or ""
                out.append(Import(mod, [a.name for a in child.names], py, child.lineno, not in_func))
            walk(child, nested)

    walk(tree, False)
    return out


def _module_file(root: Path, dotted: str) -> Optional[Path]:
    """The file backing ``dotted`` under ``root``, or None. A bare directory counts as a package."""
    base = root.joinpath(*dotted.split("."))
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    if base.is_dir():
        return base  # namespace package: no file of its own
    return None


# --------------------------------------------------------------------------------------------
# extra-code: the model's own code, laid out for --extra-code
# --------------------------------------------------------------------------------------------


class ClosureError(Exception):
    """An import the closure cannot resolve. Packaging stops rather than shipping a gap."""


def models_closure(metal_root: Path, seeds: Iterable[Import]) -> Tuple[Set[Path], Dict[str, Set[str]], Set[Path]]:
    """The files of tt-metal's ``models/`` tree reachable from ``seeds``.

    Returns ``(files, third_party, synthesized_inits)``:

    * ``files``: source files to copy, as paths under ``metal_root``.
    * ``third_party``: top-level non-tt-metal modules the closure imports, mapped to the
      closure files that import them. The recipe turns this into install guards.
    * ``synthesized_inits``: package directories that have no ``__init__.py`` upstream. tt-metal's
      ``models/`` is a namespace tree. A shipped copy gets empty ``__init__.py`` files so no other
      ``models`` on ``sys.path`` can merge with it or shadow it.

    Importing ``a.b.c`` runs ``a/__init__``, ``a/b/__init__`` and ``a/b/c``, so every ancestor of a
    reached module is part of the closure. ``from m import x`` pulls in ``m.x`` when that is a
    module.
    """
    files: Set[Path] = set()
    third_party: Dict[str, Set[str]] = {}
    inits: Set[Path] = set()
    todo: List[str] = []
    seen_modules: Set[str] = set()

    def want(dotted: str, origin: str) -> None:
        if dotted in seen_modules:
            return
        if _module_file(metal_root, dotted) is None:
            raise ClosureError(f"{origin}: imports {dotted!r}, which is not in the tt-metal tree {metal_root}")
        seen_modules.add(dotted)
        todo.append(dotted)

    for imp in seeds:
        if imp.module.split(".")[0] != METAL_TOP:
            continue
        where = f"{imp.path.name}:{imp.line}"
        want(imp.module, where)
        for name in imp.names:
            if _module_file(metal_root, f"{imp.module}.{name}") is not None:
                want(f"{imp.module}.{name}", where)

    while todo:
        dotted = todo.pop()
        parts = dotted.split(".")
        # Every ancestor package is imported on the way in.
        for i in range(1, len(parts)):
            anc = ".".join(parts[:i])
            if anc not in seen_modules:
                seen_modules.add(anc)
                todo.append(anc)
        f = _module_file(metal_root, dotted)
        if f is None:
            continue
        if f.is_dir():  # namespace package upstream
            inits.add(f)
            continue
        files.add(f)
        for imp in scan_imports(f, metal_root):
            top = imp.module.split(".")[0] if imp.module else ""
            if top == METAL_TOP:
                where = f"{f.relative_to(metal_root)}:{imp.line}"
                want(imp.module, where)
                for name in imp.names:
                    if _module_file(metal_root, f"{imp.module}.{name}") is not None:
                        want(f"{imp.module}.{name}", where)
            elif top and top not in sys.stdlib_module_names:
                third_party.setdefault(top, set()).add(str(f.relative_to(metal_root)))
    return files, third_party, inits


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_extra_code(package_dir: Path, version_file: Path, metal_root: Path, out: Path) -> dict:
    """Write ``out`` (the directory handed to ``--extra-code``) and return a report.

    ``out`` ends up with ``animatediff_ttnn/`` (copied from this repo), ``models/`` (the closure of
    what ``animatediff_ttnn`` imports from tt-metal) and ``VERSION``. ``VERSION`` is there because
    ``animatediff_ttnn/__init__.py`` falls back to ``<package parent>/VERSION`` when no
    ``animatediff-ttnn`` distribution is installed, and a shipped source tree has none.
    """
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; the recipe builds it from scratch so it can be rebuilt exactly")
    out.mkdir(parents=True, exist_ok=True)

    # 1. The model's own package. Skip byte-code and caches: they are build debris and make the
    #    staged tree depend on whichever interpreter ran last.
    pkg_name = package_dir.name
    copied: List[str] = []
    for src in sorted(package_dir.rglob("*")):
        rel = src.relative_to(package_dir)
        if src.is_dir() or "__pycache__" in rel.parts or src.suffix == ".pyc":
            continue
        dst = out / pkg_name / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        copied.append(f"{pkg_name}/{rel.as_posix()}")

    # 2. What the package imports from tt-metal, found by reading its source.
    seeds: List[Import] = []
    for py in sorted(package_dir.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        seeds.extend(scan_imports(py, package_dir.parent))
    files, third_party, inits = models_closure(metal_root, seeds)

    model_files: List[str] = []
    for f in sorted(files):
        rel = f.relative_to(metal_root)
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(f.read_bytes())
        model_files.append(rel.as_posix())
    synthesized: List[str] = []
    for d in sorted(inits):
        rel = d.relative_to(metal_root)
        init = out / rel / "__init__.py"
        init.parent.mkdir(parents=True, exist_ok=True)
        init.write_text("# Empty on purpose: tt-metal ships this directory as a namespace package.\n")
        synthesized.append((rel / "__init__.py").as_posix())

    (out / "VERSION").write_text(version_file.read_text())

    manifest = {
        "package_files": len(copied),
        "models_files": sorted(model_files),
        "synthesized_inits": sorted(synthesized),
        "third_party_from_models": {k: sorted(v) for k, v in sorted(third_party.items())},
        "tree_sha256": tree_digest(out),
    }
    return manifest


def tree_digest(root: Path) -> str:
    """One sha256 over every file's relative path and bytes, in sorted order."""
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(root).as_posix().encode())
            h.update(b"\0")
            h.update(p.read_bytes())
            h.update(b"\0")
    return h.hexdigest()


# --------------------------------------------------------------------------------------------
# check: Step 2c, every import resolves
# --------------------------------------------------------------------------------------------


def check_imports(extra_code: Path, model_py: Path, venv_python: Optional[Path]) -> dict:
    """Walk every import in ``extra_code/`` and ``model.py`` and sort it into a bucket.

    * ``stdlib``: standard library.
    * ``first_party``: a top-level package that lives in ``extra_code/``. The target module must
      exist there, or it is ``unresolved``. A name imported from a module is accepted when it is a
      submodule file or is assigned/defined in the target (checked by reading the target's AST).
    * ``third_party``: anything else. With ``venv_python`` set, each top-level name is looked up
      with ``importlib.util.find_spec`` inside that interpreter. A top-level ``find_spec`` finds a
      package without importing it, so ``ttnn`` is located and never loaded.

    ``model.py`` is scanned with ``extra_code`` as the first-party root.
    """
    first_party = {p.name for p in extra_code.iterdir() if p.is_dir() and not p.name.startswith(".")}
    unresolved: List[str] = []
    third: Dict[str, Set[str]] = {}

    def defines(target: Path, name: str) -> bool:
        """True if module file ``target`` defines or imports ``name`` at any level."""
        try:
            tree = ast.parse(target.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            return False
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                return True
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == name:
                        return True
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
                return True
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    if (a.asname or a.name.split(".")[0]) == name:
                        return True
        return False

    py_files = sorted(p for p in extra_code.rglob("*.py") if "__pycache__" not in p.parts) + [model_py]
    for py in py_files:
        for imp in scan_imports(py, extra_code if py != model_py else model_py.parent):
            top = imp.module.split(".")[0] if imp.module else ""
            if not top or top in sys.stdlib_module_names:
                continue
            where = f"{py.name}:{imp.line}"
            if top in first_party:
                target = _module_file(extra_code, imp.module)
                if target is None:
                    unresolved.append(f"{where}: {imp.module}")
                    continue
                for name in imp.names:
                    if name == "*" or _module_file(extra_code, f"{imp.module}.{name}") is not None:
                        continue
                    if target.is_file() and not defines(target, name):
                        unresolved.append(f"{where}: {imp.module}.{name}")
            else:
                third.setdefault(top, set()).add(where)

    missing: Dict[str, List[str]] = {}
    optional: Dict[str, str] = {}
    if venv_python is not None and third:
        code = (
            "import importlib.util, json, sys\n"
            "names = json.loads(sys.argv[1])\n"
            "print(json.dumps({n: importlib.util.find_spec(n) is not None for n in names}))\n"
        )
        res = subprocess.run(
            [str(venv_python), "-I", "-c", code, json.dumps(sorted(third))],
            check=True, capture_output=True, text=True,
        )
        found = json.loads(res.stdout)
        for name, ok in found.items():
            if ok:
                continue
            if name in OPTIONAL_THIRD_PARTY:
                optional[name] = OPTIONAL_THIRD_PARTY[name]
            else:
                missing[name] = sorted(third[name])[:3]
    return {
        "first_party": sorted(first_party),
        "unresolved_first_party": unresolved,
        "third_party": {k: len(v) for k, v in sorted(third.items())},
        "third_party_not_installed": missing,
        "third_party_optional": optional,
        "third_party_checked_in_venv": venv_python is not None,
    }


# --------------------------------------------------------------------------------------------
# deps-wheel: a metadata-only wheel that carries the pip dependencies
# --------------------------------------------------------------------------------------------


def parse_requirements(text: str) -> List[str]:
    """Requirement lines of a requirements file: comments and blanks dropped, index options refused."""
    reqs: List[str] = []
    for raw in text.splitlines():
        line = raw.split(" #")[0].strip() if " #" in raw else raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            raise SystemExit(f"{line!r}: option lines do not belong in a dependency list")
        reqs.append(line)
    return reqs


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


def build_deps_wheel(name: str, version: str, requirements: List[str], out_dir: Path) -> Path:
    """Write ``<name>-<version>-py3-none-any.whl`` with ``Requires-Dist`` lines and no modules.

    The wheel is deterministic: fixed timestamps, sorted entries, no host data.
    """
    dist = _norm(name)
    info = f"{dist}-{version}.dist-info"
    metadata = "\n".join(
        [
            "Metadata-Version: 2.1",
            f"Name: {name}",
            f"Version: {version}",
            "Summary: Dependency set for the tt-animatediff v6 bundle. Contains no code.",
            "Requires-Python: >=3.10",
            *[f"Requires-Dist: {r}" for r in requirements],
            "",
        ]
    )
    wheel_meta = "Wheel-Version: 1.0\nGenerator: tt-animatediff packaging/build_inputs.py\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    entries = {f"{info}/METADATA": metadata, f"{info}/WHEEL": wheel_meta}
    record_lines = []
    for path, body in sorted(entries.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(body.encode()).digest()).rstrip(b"=").decode()
        record_lines.append(f"{path},sha256={digest},{len(body.encode())}")
    record_lines.append(f"{info}/RECORD,,")
    entries[f"{info}/RECORD"] = "\n".join(record_lines) + "\n"

    out_dir.mkdir(parents=True, exist_ok=True)
    wheel = out_dir / f"{dist}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(entries):
            zi = zipfile.ZipInfo(path, date_time=(2026, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, entries[path])
    return wheel


# --------------------------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------------------------


def write_provenance(out: Path, fields: Dict[str, object]) -> None:
    out.write_text(json.dumps(fields, indent=2, sort_keys=True) + "\n")


# --------------------------------------------------------------------------------------------
# inspect: Step 3, is the staged bundle the canonical shape
# --------------------------------------------------------------------------------------------


def expand_cmd(run_sh: Path) -> List[str]:
    """The argv ``run.sh`` would launch, expanded through bash exactly as run.sh does.

    Pulls out the ``CMD=(...)`` line, evaluates it in a bash with stand-in values for ``PYBIN`` and
    ``PORT``, and prints one argv token per record. A JSON argument must come back as one token.
    """
    line = next((ln for ln in run_sh.read_text().splitlines() if ln.startswith("CMD=(")), None)
    if line is None:
        raise SystemExit(f"{run_sh}: no CMD=(...) line")
    script = f'PYBIN=PY; PORT=8000; set --; {line}\nprintf "%s\\0" "${{CMD[@]}}"\n'
    res = subprocess.run(["bash", "-c", script], check=True, capture_output=True)
    return [t.decode() for t in res.stdout.split(b"\0")[:-1]]


def inspect_bundle(bundle: Path, expect: dict) -> Tuple[dict, List[str]]:
    """Check a staged bundle against ``expect``. Returns ``(facts, problems)``.

    ``problems`` is empty when the bundle is canonical for this model. Nothing here runs the
    bundle; it reads files.
    """
    problems: List[str] = []
    manifest = json.loads((bundle / "tt_kernel_manifest.json").read_text())
    deps = manifest.get("deps") or {}
    run_sh = (bundle / "run.sh").read_text()
    reqs = (bundle / "requirements.txt").read_text()
    argv = expand_cmd(bundle / "run.sh")

    def need(cond: bool, msg: str) -> None:
        if not cond:
            problems.append(msg)

    need(manifest.get("schema_version") == "6", "schema_version is not 6")
    need(deps.get("kind") == expect["kind"], f"deps.kind is {deps.get('kind')!r}")
    need(deps.get("app") == expect["app"], f"deps.app is {deps.get('app')!r}")
    need(deps.get("extra_code_dir") == "extra_code", "extra_code_dir is not 'extra_code'")
    need(not deps.get("models_wheels"), f"models_wheels must be empty, got {deps.get('models_wheels')}")
    need(deps.get("custom_ops", []) == expect["custom_ops"], f"deps.custom_ops is {deps.get('custom_ops')}")
    need(deps.get("wheels") == expect["wheels"], f"deps.wheels is {deps.get('wheels')}")
    need(deps.get("vllm") is None, "deps.vllm must be null for a tt-dit-server bundle")
    need(manifest.get("entrypoint") is None, "entrypoint must be null for a tt-dit-server bundle")
    need((bundle / "constraints.txt").is_file(), "constraints.txt is missing")
    need((bundle / "PROVENANCE.json").is_file(), "PROVENANCE.json is missing")
    need(not (bundle / "vllm_models").exists(), "vllm_models/ must not exist for a tt-dit-server bundle")
    for line in reqs.splitlines():
        s = line.strip()
        need(not s.startswith("-"), f"requirements.txt has an option line: {s!r}")
    ttnn_pin = [ln for ln in reqs.splitlines() if re.match(r"\s*ttnn\s*==", ln)]
    need(len(ttnn_pin) == 1 and "+" not in ttnn_pin[0], f"requirements.txt must pin stock ttnn exactly: {ttnn_pin}")
    w = manifest.get("weights") or {}
    need(w.get("repo_id") == expect["weights_repo"], f"weights.repo_id is {w.get('repo_id')!r}")
    need(w.get("revision") == expect["weights_revision"], f"weights.revision is {w.get('revision')!r}")
    need(bool(w.get("allow_patterns")), "weights.allow_patterns is empty: the pull would fetch ~22 GB")
    need((manifest.get("env") or {}).get("ANIMATEDIFF_MESH_SHAPE") == "1x1", "ANIMATEDIFF_MESH_SHAPE is not 1x1")
    need(bool(deps.get("verify")), "no --verify guards recorded")

    # run.sh: the app is served by uvicorn, as one argv per token, and extra_code is first on PYTHONPATH.
    need(argv[:3] == ["PY", "-m", "uvicorn"], f"run.sh argv does not start with python -m uvicorn: {argv[:3]}")
    need(expect["app"] in argv, f"run.sh argv does not name {expect['app']}")
    pp = next((ln for ln in run_sh.splitlines() if ln.startswith("export PYTHONPATH=")), "")
    need(bool(re.search(r'PYTHONPATH="?\$HERE/extra_code[:"]', pp)), f"extra_code is not first on PYTHONPATH: {pp!r}")
    need("LD_PRELOAD" in run_sh, "run.sh does not preload _ttnncpp.so")

    for rel in expect["must_exist"]:
        need((bundle / rel).exists(), f"missing from the bundle: {rel}")
    for rel in expect["must_not_exist"]:
        need(not (bundle / rel).exists(), f"must not be in the bundle: {rel}")
    for whl in (bundle / "wheels").glob("*.whl"):
        need(whl.name in expect["wheel_names"], f"unexpected wheel: {whl.name}")

    facts = {
        "argv": argv,
        "pythonpath_line": pp,
        "deps": {k: deps.get(k) for k in ("python", "wheels", "custom_ops", "models_wheels", "kind", "app", "extra_code_dir")},
        "weights": w,
        "env": manifest.get("env"),
        "tt_metal_version": manifest.get("tt_metal_version"),
        "verify_count": len(deps.get("verify") or []),
    }
    return facts, problems


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extra-code")
    p.add_argument("--package-dir", type=Path, required=True)
    p.add_argument("--version-file", type=Path, required=True)
    p.add_argument("--metal-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)

    p = sub.add_parser("check")
    p.add_argument("--extra-code", type=Path, required=True)
    p.add_argument("--model-py", type=Path, required=True)
    p.add_argument("--venv-python", type=Path)

    p = sub.add_parser("deps-wheel")
    p.add_argument("--name", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--requirements", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)

    p = sub.add_parser("provenance")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--json", required=True, help="JSON object to write")

    p = sub.add_parser("inspect")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--expect", type=Path, required=True, help="JSON file of expected values")

    a = ap.parse_args(argv)

    if a.cmd == "extra-code":
        report = build_extra_code(a.package_dir, a.version_file, a.metal_root, a.out)
        a.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"extra_code: {report['package_files']} package files, {len(report['models_files'])} models files "
              f"(+{len(report['synthesized_inits'])} empty __init__), tree sha256 {report['tree_sha256'][:16]}")
        print("third-party imports in the models closure: " + ", ".join(report["third_party_from_models"]))
        return 0
    if a.cmd == "check":
        r = check_imports(a.extra_code, a.model_py, a.venv_python)
        print(json.dumps(r, indent=2))
        bad = r["unresolved_first_party"] or r["third_party_not_installed"]
        return 1 if bad else 0
    if a.cmd == "deps-wheel":
        reqs = parse_requirements(a.requirements.read_text())
        whl = build_deps_wheel(a.name, a.version, reqs, a.out_dir)
        print(whl)
        return 0
    if a.cmd == "provenance":
        write_provenance(a.out, json.loads(a.json))
        return 0
    if a.cmd == "inspect":
        facts, problems = inspect_bundle(a.bundle, json.loads(a.expect.read_text()))
        print(json.dumps(facts, indent=2))
        for pr in problems:
            print(f"PROBLEM: {pr}", file=sys.stderr)
        return 1 if problems else 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
