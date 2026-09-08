#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Assemble the Hugging Face artifact for episod/tt-animatediff.

Everything the Hub repo contains is produced from this checkout by this script,
so the published copy cannot drift from GitHub. ``model_index.json`` is the only
generated file; every other entry is a straight copy, which means there is
exactly one place to look when the artifact is wrong.

The output directory is rebuilt from scratch on every run: a file deleted from
this repo must not survive as a stale artifact entry that still gets uploaded.

Usage:
    python scripts/build_hf_artifact.py                  # build build/hf
    python scripts/build_hf_artifact.py --out /tmp/hf     # build elsewhere
    python scripts/build_hf_artifact.py --skip-load-check  # assemble only
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "build" / "hf"

#: The delegate package, vendored so the Hub copy is readable and so
#: resolve_package() in pipeline.py can fetch it without any weights.
PACKAGE = "animatediff_ttnn"

#: Never copied into the artifact. ``invokeai`` is not carried on this branch
#: and would ship an unproven integration; ``__pycache__`` is build noise that
#: would also leak local absolute paths.
EXCLUDED = ("__pycache__", "*.pyc", "invokeai")

#: (source relative to repo root, destination relative to artifact root).
COPIES = (
    ("hf/pipeline.py", "pipeline.py"),
    ("hf/requirements.txt", "requirements.txt"),
    ("docs/model-card.md", "README.md"),
    ("LICENSE", "LICENSE"),
    ("app.py", "app.py"),
    ("VERSION", "VERSION"),
)

#: The generated config. base_model / motion_adapter / lightning_repo are
#: declarative metadata, NOT injection points: they mirror the upstream ids
#: that are hardcoded in animatediff_ttnn/pipeline.py and
#: animatediff_ttnn/generation_helpers.py, so the Hub page records what the
#: pipeline resolves. hf/pipeline.py's __call__ never reads them, so
#: overriding them at from_pretrained() time has no effect on generation — an
#: actual adapter swap requires a code change in animatediff_ttnn, not a
#: config edit here.
#: Every non-underscore key must be an __init__ parameter of
#: TTAnimateDiffPipeline — diffusers silently ignores keys that are not, so an
#: unbacked key looks like it works while the default quietly wins.
#: tests/test_build_hf_artifact.py asserts that correspondence.
#:
#: _diffusers_version records what this artifact was built and verified
#: against; it is informational to diffusers. The enforced floor is
#: diffusers>=0.32.1 in hf/requirements.txt, and the custom-pipeline load was
#: measured working on both 0.32.1 and 0.39.0.
MODEL_INDEX = {
    "_class_name": "TTAnimateDiffPipeline",
    "_diffusers_version": "0.39.0",
    "base_model": "CompVis/stable-diffusion-v1-4",
    "motion_adapter": "guoyww/animatediff-motion-adapter-v1-5-2",
    "lightning_repo": "ByteDance/AnimateDiff-Lightning",
    "code_repo": "episod/tt-animatediff",
    "temporal_alpha": 0.35,
    "num_frames": 8,
    "num_steps": 25,
    "guidance_scale": 7.5,
}


def build_artifact(
    out_dir: Path = DEFAULT_OUT,
    root: Path = ROOT,
    load_check: bool = True,
    on_load_check=None,
) -> Path:
    """Rebuild the artifact tree and return its path.

    Args:
        out_dir: Destination; removed and recreated.
        root: Repo checkout to build from.
        load_check: Round-trip the result through DiffusionPipeline in a
            subprocess. Off only for tests that build many trees.
        on_load_check: Optional callable receiving the load check's stdout.
            Lets the CLI show "load check OK" without this function doing I/O
            of its own; None keeps the build silent.

    Raises:
        FileNotFoundError: a required source file is missing.
        RuntimeError: the assembled tree does not load.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    shutil.copytree(
        root / PACKAGE,
        out_dir / PACKAGE,
        ignore=shutil.ignore_patterns(*EXCLUDED),
    )

    for source_rel, dest_rel in COPIES:
        source = root / source_rel
        if not source.is_file():
            raise FileNotFoundError(
                f"{source_rel} is required for the artifact but does not exist"
            )
        shutil.copy2(source, out_dir / dest_rel)

    (out_dir / "model_index.json").write_text(
        json.dumps(MODEL_INDEX, indent=2) + "\n"
    )

    if load_check:
        output = _assert_artifact_loads(out_dir)
        if on_load_check is not None and output:
            on_load_check(output)
    return out_dir


def _assert_artifact_loads(out_dir: Path) -> str:
    """Load the tree through diffusers in a subprocess, returning its output.

    A subprocess, because a custom pipeline is executed out of the shared
    ``~/.cache/huggingface/modules`` directory: loading in-process would leave
    this build's pipeline.py cached under a module name a later load reuses.
    """
    probe = (
        "from diffusers import DiffusionPipeline;"
        f"p = DiffusionPipeline.from_pretrained({str(out_dir)!r},"
        f" custom_pipeline={str(out_dir)!r}, trust_remote_code=True);"
        "assert type(p).__name__ == 'TTAnimateDiffPipeline', type(p).__name__;"
        "print('load check OK:', type(p).__name__)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            "assembled artifact failed its load check — refusing to offer it "
            f"for publish:\n{result.stdout}\n{result.stderr}"
        )
    # Return the probe's output rather than printing it: this function asserts,
    # and a caller that wants the "load check OK" line on the terminal can say
    # so. main() does.
    return result.stdout


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-load-check", action="store_true")
    args = parser.parse_args(argv)

    out = build_artifact(
        out_dir=args.out,
        load_check=not args.skip_load_check,
        on_load_check=lambda text: print(text, end=""),
    )
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    print(f"built {out} ({len(files)} files)")
    for name in files[:12]:
        print(f"  {name}")
    if len(files) > 12:
        print(f"  ... and {len(files) - 12} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
