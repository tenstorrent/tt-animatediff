#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Publish the tt-animatediff artifact (or its Space) to the Hugging Face Hub.

Safety rules are baked in here rather than left to the caller's discipline:

* Repos are created **private**. There is no --public flag. Flipping visibility
  is a separate, explicitly-confirmed action and does not belong in a
  re-runnable publish script. ``create_repo(..., private=True, exist_ok=True)``
  cannot flip an existing repo either way — per huggingface_hub's docs the
  value "is ignored if the repo already exists" — so re-running is safe.
* Any Hub write requires --yes. --dry-run never touches the Hub regardless.
* --verify is read-only and round-trips the **published** copy through
  DiffusionPipeline, so it proves what a downstream user receives rather than
  re-checking local state.

Usage:
    python scripts/publish_to_hub.py --dry-run          # preview the model publish
    python scripts/publish_to_hub.py --yes              # publish the model repo
    python scripts/publish_to_hub.py --verify           # read-only round trip
    python scripts/publish_to_hub.py --space --dry-run  # preview the Space
    python scripts/publish_to_hub.py --space --yes      # publish the Space
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_hf_artifact import DEFAULT_OUT, build_artifact  # noqa: E402

MODEL_REPO = "episod/tt-animatediff"
SPACE_REPO = "episod/tt-animatediff-demo"
SPACE_DIR = ROOT / "spaces"
# No CARD_PATH here on purpose: build_hf_artifact.py copies docs/model-card.md to the
# artifact's README.md, so upload_folder already applies the card. A second
# metadata/card write from this script would be a second source of truth.

# Gallery sources: (source_path, destination_filename in staging bundle)
GALLERY_SOURCES = [
    (ROOT / "docs" / "assets" / "gallery" / "arctic-wave-standard.gif", "arctic-wave-standard.gif"),
    (ROOT / "docs" / "assets" / "gallery" / "cathedral-standard.gif", "cathedral-standard.gif"),
    (ROOT / "docs" / "assets" / "gallery" / "crystal-cave-standard.gif", "crystal-cave-standard.gif"),
    (ROOT / "docs" / "assets" / "chain" / "glasses-cosmic.gif", "glasses-cosmic.gif"),
    (ROOT / "docs" / "assets" / "chain" / "glasses-ocean.gif", "glasses-ocean.gif"),
    (ROOT / "docs" / "assets" / "mayan-glyphs" / "Q1" / "imix.gif", "mayan-imix.gif"),
]

#: What --verify must find in the published config. If diffusers ever changes
#: custom-pipeline resolution, this is the check that catches it.
EXPECTED_CONFIG = {
    "base_model": "CompVis/stable-diffusion-v1-4",
    "motion_adapter": "guoyww/animatediff-motion-adapter-v1-5-2",
    "temporal_alpha": 0.35,
    "num_frames": 8,
}


def stage_space() -> Path:
    """Stage Space files and gallery assets into build/space/.

    Copies everything from spaces/ plus the six gallery GIFs into build/space/,
    rebuilding from scratch each call to prevent stale uploads.

    Returns: Path to the staged build/space/ directory.
    Raises: FileNotFoundError if any gallery source is missing.
    """
    build_space = ROOT / "build" / "space"

    # Rebuild from scratch
    if build_space.exists():
        shutil.rmtree(build_space)
    build_space.mkdir(parents=True, exist_ok=True)

    # Copy all files from spaces/ (including README.md, app.py, requirements.txt)
    for item in SPACE_DIR.iterdir():
        if item.is_file():
            shutil.copy2(item, build_space / item.name)

    # Create gallery directory and copy GIFs
    gallery_dir = build_space / "gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    for source, dest_name in GALLERY_SOURCES:
        if not source.exists():
            raise FileNotFoundError(f"Gallery source missing: {source}")
        shutil.copy2(source, gallery_dir / dest_name)

    return build_space


def publish(
    repo_id: str,
    folder: Path,
    repo_type: str,
    yes: bool,
    dry_run: bool,
) -> int:
    """Create the repo (private) if needed and upload ``folder``."""
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    total_kb = sum(p.stat().st_size for p in files) / 1024
    print(f"repo:      {repo_id} ({repo_type}, private on create)")
    print(f"folder:    {folder}")
    print(f"contents:  {len(files)} files, {total_kb:.0f} KB")

    # Show full list when actually writing, truncate for preview (dry-run).
    show_all = yes and not dry_run
    limit = len(files) if show_all else 15
    for path in files[:limit]:
        print(f"  {path.relative_to(folder).as_posix()}")
    if len(files) > limit:
        print(f"  ... and {len(files) - limit} more")

    for path in files:
        if path.suffix in {".pt", ".ckpt", ".safetensors", ".bin", ".pth", ".gguf", ".onnx", ".h5"}:
            print(f"REFUSING: weight-shaped file in the upload: {path}")
            return 1

    if dry_run:
        print("\n--dry-run: nothing was sent to the Hub.")
        return 0
    if not yes:
        print("\nRefusing to write to the Hub without --yes.")
        return 1

    from huggingface_hub import HfApi

    api = HfApi()
    # Spaces require space_sdk at creation time per huggingface_hub.
    # The Hub refuses create_repo(..., repo_type="space") without it.
    create_kwargs = {"repo_id": repo_id, "repo_type": repo_type, "private": True, "exist_ok": True}
    if repo_type == "space":
        create_kwargs["space_sdk"] = "gradio"

    api.create_repo(**create_kwargs)

    try:
        api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type=repo_type)
    except Exception as e:
        print(f"\nupload_folder failed: {e}")
        print(f"WARNING: {repo_id} was created (private) but upload did not complete.")
        print("Re-running with --yes is safe (create_repo uses exist_ok=True).")
        return 1

    print(f"\nuploaded to https://huggingface.co/{repo_id} (private)")
    return 0


def verify(repo_id: str = MODEL_REPO) -> int:
    """Round-trip the published copy. Read-only."""
    from diffusers import DiffusionPipeline

    print(f"loading {repo_id} from the Hub ...")
    pipe = DiffusionPipeline.from_pretrained(
        repo_id, custom_pipeline=repo_id, trust_remote_code=True
    )
    name = type(pipe).__name__
    print(f"class:  {name}")
    if name != "TTAnimateDiffPipeline":
        print(f"FAIL: expected TTAnimateDiffPipeline, got {name}")
        return 1
    for key, expected in EXPECTED_CONFIG.items():
        actual = pipe.config.get(key)
        status = "ok" if actual == expected else "FAIL"
        print(f"config: {key} = {actual!r} ({status}, expected {expected!r})")
        if actual != expected:
            return 1
    print("verify: PASS")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", action="store_true", help="publish the Space instead")

    # --dry-run, --yes, and --verify are mutually exclusive.
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument("--dry-run", action="store_true", help="preview without sending to Hub")
    action_group.add_argument("--yes", action="store_true", help="publish to Hub (requires --space or model)")
    action_group.add_argument("--verify", action="store_true", help="read-only round-trip of published model")

    args = parser.parse_args(argv)

    if args.space and args.verify:
        print("--space --verify: cannot verify a Space via DiffusionPipeline.from_pretrained.")
        print("Check the Space's build log and runtime status on the Hub instead.")
        return 1

    if args.verify:
        return verify()

    if args.space:
        if not SPACE_DIR.is_dir():
            print(f"missing {SPACE_DIR}")
            return 1
        print("staging Space files and gallery assets ...")
        try:
            folder = stage_space()
        except FileNotFoundError as e:
            print(f"FAILED: {e}")
            return 1
        return publish(SPACE_REPO, folder, "space", args.yes, args.dry_run)

    print("building the artifact ...")
    folder = build_artifact(out_dir=DEFAULT_OUT)
    return publish(MODEL_REPO, folder, "model", args.yes, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
