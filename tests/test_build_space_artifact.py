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


def test_the_card_pins_the_interpreter_it_was_verified_on():
    """No longer about audioop.

    Under gradio 4.44.1 this pin was mandatory: 3.13 dropped stdlib `audioop` and
    gradio's pydub imports it, so the Space built and then died on `import gradio`.
    gradio 6 ships audioop-lts for 3.13, so that reason is gone -- the pin stays only so
    the interpreter this is verified against locally is the one the Hub runs.
    """
    version = str(_card_front_matter().get("python_version", ""))
    assert version, "python_version must be pinned so local verification matches the Hub"
    major, minor = (int(part) for part in version.split(".")[:2])
    assert (major, minor) >= (3, 10), f"python_version {version} is below gradio's floor"


def test_transformers_floor_clears_its_cve_fixes():
    """CVE-2026-9856/-5241/-4372 (HIGH) and -1839 (MEDIUM) are fixed across transformers
    5.3, 5.5 and 5.10, so 5.10 is the floor that clears all four. The `<5` cap that used
    to sit here made every one of them unavoidable."""
    assert ">=5.10" in _requirement("transformers"), _requirement("transformers")


def test_starlette_floor_clears_its_cve_fixes():
    """CVE-2026-54283/-48818 (HIGH) and -48710/-48817 (MEDIUM) are fixed across starlette
    1.0.1, 1.1.0 and 1.3.1. gradio 6 only requires >=1.0.1, which still admits two of
    them, so this floor has to be ours rather than inherited."""
    assert ">=1.3.1" in _requirement("starlette"), _requirement("starlette")


def test_no_compatibility_cap_is_reintroduced_below_a_security_floor():
    """The guard that matters most here, because re-adding a cap is the obvious wrong fix.

    Four caps once lived in this file -- huggingface_hub<1, pydantic<2.11, starlette<1,
    transformers<5 -- and each one fixed a real, measured gradio 4.44.1 deploy failure.
    Together they froze the stack below its CVE fixes, and no version inside any of them
    is clean: every fix ships in the major the cap excluded. They came off with the gradio
    pin that needed them.

    If a future gradio breakage tempts someone to pin one back, this fails and names the
    trade rather than letting it pass as a compatibility fix.
    """
    lines = [ln.strip() for ln in (ROOT / "spaces" / "requirements.txt").read_text().splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    forbidden = {"huggingface-hub": "<1", "pydantic": "<2.11",
                 "starlette": "<1", "transformers": "<5"}
    offenders = []
    for ln in lines:
        norm = ln.replace(" ", "").replace("_", "-").lower()
        for name, bad in forbidden.items():
            if not norm.startswith(name):
                continue
            for spec in norm[len(name):].split(","):
                # "<1" must not match "<1.3.1" or "<2"; compare the bound exactly.
                if spec == bad:
                    offenders.append(ln)
    assert not offenders, (
        "a compatibility cap is back below a security floor -- every CVE fix for these "
        f"ships in the major it excludes: {offenders}"
    )
