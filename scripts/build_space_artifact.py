#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Assemble the Hugging Face Space bundle for episod/tt-animatediff-demo.

Sibling of ``build_hf_artifact.py`` and deliberately the same shape: the thing
that gets uploaded is *built* from this checkout, never edited in place, and is
rebuilt from scratch on every run so a file deleted from this repo cannot
survive as a stale upload entry.

Why the Space is staged rather than committed ready-to-upload: the six gallery
GIFs total ~14 MB and already exist in this repo under ``docs/assets/``.
Committing copies into ``spaces/gallery/`` would have added them to git history
permanently, for nothing. So ``spaces/`` holds only the Space's own code and
card, and this script joins it with the gallery at build time.

The consequence to remember: a file dropped into ``spaces/`` reaches the Space
automatically, but a **new gallery GIF does not** unless it is added to
``GALLERY_SOURCES`` below.

Usage:
    python scripts/build_space_artifact.py                 # build build/space
    python scripts/build_space_artifact.py --out /tmp/spc   # build elsewhere
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: The Space's own files: app, card, requirements. Uploaded to the Space root.
SPACE_DIR = ROOT / "spaces"

#: Where the assembled bundle lands. Git-ignored.
DEFAULT_OUT = ROOT / "build" / "space"

#: Pre-rendered Blackhole output shown in the Space's gallery, as
#: (source in this repo, destination filename inside ``gallery/``).
#: The app globs ``gallery/*.gif``, so a renamed destination silently changes
#: what a visitor sees — the destination names are part of the contract.
GALLERY_SOURCES = [
    (ROOT / "docs/assets/gallery/arctic-wave-standard.gif", "arctic-wave-standard.gif"),
    (ROOT / "docs/assets/gallery/cathedral-standard.gif", "cathedral-standard.gif"),
    (ROOT / "docs/assets/gallery/crystal-cave-standard.gif", "crystal-cave-standard.gif"),
    (ROOT / "docs/assets/chain/glasses-cosmic.gif", "glasses-cosmic.gif"),
    (ROOT / "docs/assets/chain/glasses-ocean.gif", "glasses-ocean.gif"),
    (ROOT / "docs/assets/mayan-glyphs/Q1/imix.gif", "mayan-imix.gif"),
]

#: Never copied out of ``spaces/``: build noise that would also leak local paths.
EXCLUDED_NAMES = {"__pycache__"}


def build_space_artifact(
    out_dir: Path = DEFAULT_OUT,
    space_dir: Path = SPACE_DIR,
    gallery_sources=GALLERY_SOURCES,
) -> Path:
    """Rebuild the Space bundle and return its path.

    Args:
        out_dir: Destination; removed and recreated.
        space_dir: The Space's own files (app, card, requirements).
        gallery_sources: (source path, destination filename) pairs.

    Returns:
        The assembled directory.

    Raises:
        FileNotFoundError: ``space_dir`` is missing, or a gallery source does
            not exist. Named explicitly, because a silently-absent GIF would
            publish a Space with a partly-empty gallery and no other signal.
        RuntimeError: the assembled bundle is missing something the Space needs
            to boot, or the gallery count does not match ``gallery_sources``.
    """
    out_dir = Path(out_dir)
    if not space_dir.is_dir():
        raise FileNotFoundError(f"{space_dir} does not exist")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # Only top-level files. A stray subdirectory under spaces/ is not part of
    # the Space's contract and would just bloat the upload.
    for item in sorted(space_dir.iterdir()):
        if item.is_file() and item.name not in EXCLUDED_NAMES:
            shutil.copy2(item, out_dir / item.name)

    gallery_dir = out_dir / "gallery"
    gallery_dir.mkdir(parents=True)
    for source, dest_name in gallery_sources:
        if not Path(source).is_file():
            raise FileNotFoundError(
                f"gallery source missing: {source} (referenced by "
                f"GALLERY_SOURCES as {dest_name!r})"
            )
        shutil.copy2(source, gallery_dir / dest_name)

    _assert_bundle_is_complete(out_dir, expected_gallery=len(gallery_sources))
    return out_dir


def _assert_bundle_is_complete(out_dir: Path, expected_gallery: int) -> None:
    """Fail the build rather than publish a Space that cannot boot.

    ``app.py`` and ``README.md`` are what the Hub needs to run and describe the
    Space; the gallery count is checked because the app degrades *silently* to
    an empty gallery, so a staging regression would otherwise ship unnoticed.
    """
    for required in ("app.py", "README.md", "requirements.txt"):
        if not (out_dir / required).is_file():
            raise RuntimeError(
                f"staged Space bundle is missing {required} — refusing to "
                f"offer it for publish"
            )

    found = sorted(p.name for p in (out_dir / "gallery").glob("*.gif"))
    if len(found) != expected_gallery:
        raise RuntimeError(
            f"staged gallery holds {len(found)} GIF(s) but GALLERY_SOURCES "
            f"names {expected_gallery}: {found}"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    out = build_space_artifact(out_dir=args.out)
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    total_kb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1024
    print(f"built {out} ({len(files)} files, {total_kb:.0f} KB)")
    for name in files:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
