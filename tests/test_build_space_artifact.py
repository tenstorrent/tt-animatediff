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


# ---------------------------------------------------------------------------
# The Space's dependency pins -- every one of these cost a deploy
# ---------------------------------------------------------------------------

def _card_front_matter() -> dict:
    """The Space card's YAML front matter, which is real configuration on the Hub."""
    import yaml

    text = (ROOT / "spaces" / "README.md").read_text()
    assert text.startswith("---\n"), "the Space card must open with YAML front matter"
    return yaml.safe_load(text.split("---\n")[1])


def _requirement(name: str) -> str:
    """The one uncommented requirement line for ``name``."""
    lines = [
        ln.strip()
        for ln in (ROOT / "spaces" / "requirements.txt").read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    match = [ln for ln in lines if ln.lower().replace("_", "-").startswith(name.lower().replace("_", "-"))]
    assert len(match) == 1, f"expected exactly one {name} line, found {match}"
    return match[0]


def test_the_card_sdk_version_matches_the_pinned_gradio():
    """The Hub builds against `sdk_version`; requirements.txt installs its own pin. If the
    two drift, the Space runs a different Gradio than the file claims, and nothing says so."""
    assert _requirement("gradio") == f"gradio=={_card_front_matter()['sdk_version']}"


def test_the_card_pins_a_python_that_still_has_audioop():
    """Deploy 1 died here.

    The Hub's default image is Python 3.13, where `audioop` was removed from the stdlib
    (PEP 594). gradio 4.44.1 pulls in pydub, which imports it, so the Space built cleanly
    and then died on `import gradio` before app.py ran a line of its own.
    """
    version = str(_card_front_matter().get("python_version", ""))
    assert version, "python_version must be pinned, or the Hub picks 3.13 and gradio cannot import"
    major, minor = (int(p) for p in version.split(".")[:2])
    assert (major, minor) < (3, 13), (
        f"python_version {version} has no stdlib audioop; gradio {_card_front_matter()['sdk_version']} "
        "needs pydub, which imports it"
    )


def test_huggingface_hub_is_capped_below_1():
    """Deploy 2 died here.

    gradio 4.44.1 declares `huggingface-hub>=0.19.3` with no upper bound, and its own
    oauth.py does `from huggingface_hub import HfFolder` -- removed in 1.0. The resolver
    took 1.30.0 and gradio failed to import. gradio's metadata cannot protect us, so the
    cap has to live in our file.
    """
    assert "<1" in _requirement("huggingface_hub")


def test_pydantic_is_capped_below_2_11():
    """Deploy 3 died here, in the worst possible shape: the Space reported RUNNING while
    returning 503 to every visitor, because the failure was per-request inside gradio's own
    route rather than at startup.

    gradio 4.44.1 pins gradio-client==1.3.0, whose json_schema_to_python_type() assumes
    `additionalProperties` is a dict; pydantic 2.11 began emitting it as a bool. Bisected
    locally: 2.9.2 OK, 2.10.6 OK, 2.11.0 TypeError, 2.12.0 TypeError.
    """
    assert "<2.11" in _requirement("pydantic")


def test_transformers_is_capped_to_the_major_this_is_exercised_against():
    """Not a measured failure -- a bound the file's own rule already asked for.

    The comment says bounds are "the versions the pipeline is actually exercised against
    ... raise them deliberately, after testing", and `<6` admitted a 5.x nothing here has
    run. A deploy duly installed transformers 5.16.1.
    """
    assert "<5" in _requirement("transformers")
