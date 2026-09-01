# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for scripts/build_space_artifact.py — the staged HF Space bundle.

The Space is assembled at publish time rather than committed, because its six
gallery GIFs total ~14 MB and already live in this repo. That trade buys a
smaller history at the cost of one more moving part, so these tests pin the
part that can now drift: what actually lands in the bundle.
"""

from pathlib import Path

import pytest

from scripts.build_space_artifact import (
    GALLERY_SOURCES,
    build_space_artifact,
)

ROOT = Path(__file__).resolve().parent.parent


def test_bundle_contains_space_files_and_every_gallery_gif(tmp_path):
    out = build_space_artifact(out_dir=tmp_path / "space")

    for required in ("app.py", "README.md", "requirements.txt"):
        assert (out / required).is_file(), f"missing {required}"

    gifs = sorted(p.name for p in (out / "gallery").glob("*.gif"))
    assert gifs == sorted(dest for _, dest in GALLERY_SOURCES)


def test_every_declared_gallery_source_exists_in_the_repo():
    """GALLERY_SOURCES names tracked files; a rename must fail loudly here."""
    missing = [str(src) for src, _ in GALLERY_SOURCES if not Path(src).is_file()]
    assert not missing, f"GALLERY_SOURCES references missing files: {missing}"


def test_gallery_destination_names_are_part_of_the_contract(tmp_path):
    """The Space app globs gallery/*.gif, so destination names decide what shows."""
    out = build_space_artifact(out_dir=tmp_path / "space")
    assert (out / "gallery" / "mayan-imix.gif").is_file(), (
        "the mayan glyph is renamed on copy; that name is what the app serves"
    )


def test_rebuild_removes_stale_files(tmp_path):
    out_dir = tmp_path / "space"
    build_space_artifact(out_dir=out_dir)
    stale = out_dir / "gallery" / "removed-last-release.gif"
    stale.write_text("stale")
    build_space_artifact(out_dir=out_dir)
    assert not stale.exists()


def test_missing_gallery_source_raises_naming_the_file(tmp_path):
    """A silently-absent GIF would publish a half-empty gallery with no signal."""
    with pytest.raises(FileNotFoundError, match="nope.gif"):
        build_space_artifact(
            out_dir=tmp_path / "space",
            gallery_sources=[(ROOT / "docs" / "assets" / "nope.gif", "nope.gif")],
        )


def test_incomplete_space_dir_is_refused(tmp_path):
    """A bundle without app.py cannot boot on the Hub; fail the build instead."""
    fake_space = tmp_path / "spaces"
    fake_space.mkdir()
    (fake_space / "README.md").write_text("---\ntitle: x\n---\n")

    with pytest.raises(RuntimeError, match="app.py"):
        build_space_artifact(
            out_dir=tmp_path / "space",
            space_dir=fake_space,
            gallery_sources=[],
        )


def test_no_pycache_leaks_into_the_bundle(tmp_path):
    out = build_space_artifact(out_dir=tmp_path / "space")
    assert list(out.rglob("__pycache__")) == []
    assert list(out.rglob("*.pyc")) == []
