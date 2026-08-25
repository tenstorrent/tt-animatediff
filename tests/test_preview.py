# SPDX-License-Identifier: Apache-2.0
"""Tests for `animatediff_ttnn.preview` — the per-step preview emitter.

The Gradio app (`app.py`) has streamed per-step latent previews since the UI
landed, but the CLI runner (`examples/generate.py`) — the entry point
tt-local-generator actually execs — never wired `on_step`, so nothing outside
the Space could see a generation as it formed. This module is the shared,
importable half of closing that gap; the runner is left thin.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from animatediff_ttnn import preview  # noqa: E402


class _FakeLatent:
    """Stands in for a (1, 4, lh, lw) tensor without importing torch."""


def _fake_latents(n=2):
    return [_FakeLatent() for _ in range(n)]


@pytest.fixture()
def wired(tmp_path, monkeypatch):
    """Capture what the callback renders and emits, with no torch/PIL involved."""
    rendered: list = []
    exported: list = []

    def fake_latent_preview(frame_latents, height, width):
        rendered.append((len(frame_latents), height, width))
        return ["frame"] * len(frame_latents)

    def fake_export_gif(frames, path):
        exported.append(path)
        Path(path).write_bytes(b"GIF89a-fake")

    monkeypatch.setattr(preview, "_latent_preview_fn", lambda: fake_latent_preview)
    monkeypatch.setattr(preview, "_export_gif_fn", lambda: fake_export_gif)
    return rendered, exported


# ── Construction ─────────────────────────────────────────────────────────────

def test_no_preview_path_means_no_callback(tmp_path):
    """Previews are opt-in — the runner must be byte-identical without them."""
    assert preview.make_step_callback(None, num_steps=25) is None
    assert preview.make_step_callback("", num_steps=25) is None


def test_returns_a_callable_when_a_path_is_given(tmp_path):
    cb = preview.make_step_callback(str(tmp_path / "p.gif"), num_steps=25)
    assert callable(cb)


# ── Cadence ──────────────────────────────────────────────────────────────────

def test_short_runs_preview_every_step(tmp_path, wired):
    """<=10 steps: every step, matching app.py's cadence rule."""
    _rendered, exported = wired
    cb = preview.make_step_callback(str(tmp_path / "p.gif"), num_steps=8)
    for i in range(8):
        cb(i, 8, _fake_latents())
    assert len(exported) == 8


def test_long_runs_preview_every_other_step(tmp_path, wired):
    _rendered, exported = wired
    cb = preview.make_step_callback(str(tmp_path / "p.gif"), num_steps=25)
    for i in range(25):
        cb(i, 25, _fake_latents())
    # steps 0,2,4,...,22 -> 12, plus the final step 24 which always emits
    assert len(exported) == 13


def test_final_step_always_emits_even_off_cadence(tmp_path, wired):
    """The last step is the closest thing to the real result — never skip it."""
    _rendered, exported = wired
    cb = preview.make_step_callback(str(tmp_path / "p.gif"), num_steps=25, every=10)
    for i in range(25):
        cb(i, 25, _fake_latents())
    # 0, 10, 20, and the final 24
    assert len(exported) == 4


def test_explicit_every_overrides_the_default_cadence(tmp_path, wired):
    _rendered, exported = wired
    cb = preview.make_step_callback(str(tmp_path / "p.gif"), num_steps=8, every=4)
    for i in range(8):
        cb(i, 8, _fake_latents())
    assert len(exported) == 3  # 0, 4, and the final 7


# ── The emitted line (tt-local-generator's transport) ────────────────────────

def test_emits_a_parseable_line_per_preview(tmp_path, wired):
    out: list = []
    p = str(tmp_path / "p.gif")
    cb = preview.make_step_callback(p, num_steps=4, emit=out.append)
    cb(0, 4, _fake_latents())
    assert out == [f"{preview.PREVIEW_PREFIX} 1/4 {p}"]


def test_line_reports_one_based_steps(tmp_path, wired):
    out: list = []
    cb = preview.make_step_callback(str(tmp_path / "p.gif"), num_steps=4, emit=out.append)
    cb(3, 4, _fake_latents())
    assert " 4/4 " in out[0], "a human-facing step count starts at 1, not 0"


def test_parse_line_roundtrips_what_the_callback_emits(tmp_path, wired):
    out: list = []
    p = str(tmp_path / "p.gif")
    cb = preview.make_step_callback(p, num_steps=25, emit=out.append)
    cb(6, 25, _fake_latents())
    parsed = preview.parse_preview_line(out[0])
    assert parsed == (7, 25, p)


def test_parse_line_ignores_unrelated_output():
    assert preview.parse_preview_line("Loading pipeline...") is None
    assert preview.parse_preview_line("  Done in 42.0s") is None
    assert preview.parse_preview_line("") is None


def test_parse_line_tolerates_surrounding_whitespace(tmp_path):
    line = f"   {preview.PREVIEW_PREFIX} 3/25 /tmp/x.gif  "
    assert preview.parse_preview_line(line) == (3, 25, "/tmp/x.gif")


def test_parse_line_accepts_a_path_containing_spaces():
    line = f"{preview.PREVIEW_PREFIX} 2/8 /tmp/my runs/preview.gif"
    assert preview.parse_preview_line(line) == (2, 8, "/tmp/my runs/preview.gif")


# ── Robustness — a preview must never break the generation ───────────────────

def test_render_failure_is_swallowed(tmp_path, monkeypatch):
    """A preview is decoration. If it raises, the run must continue."""
    def boom(*_a, **_k):
        raise RuntimeError("latent preview exploded")

    monkeypatch.setattr(preview, "_latent_preview_fn", lambda: boom)
    out: list = []
    cb = preview.make_step_callback(str(tmp_path / "p.gif"), num_steps=4, emit=out.append)
    cb(0, 4, _fake_latents())  # must not raise
    assert out == [], "nothing should be announced when nothing was written"


def test_emit_failure_is_swallowed(tmp_path, wired):
    def boom(_line):
        raise RuntimeError("stdout closed")

    cb = preview.make_step_callback(str(tmp_path / "p.gif"), num_steps=4, emit=boom)
    cb(0, 4, _fake_latents())  # must not raise


# ── Atomicity — the reader is a GUI polling the same file ───────────────────

def test_preview_file_is_written_atomically(tmp_path, monkeypatch):
    """tt-local-generator loads this GIF while the runner rewrites it. A
    half-written file would render as a broken frame, so the write must land
    via a rename rather than in place."""
    seen: list = []

    def fake_export_gif(frames, path):
        seen.append(path)
        Path(path).write_bytes(b"GIF89a-fake")

    monkeypatch.setattr(preview, "_latent_preview_fn",
                        lambda: (lambda fl, h, w: ["f"]))
    monkeypatch.setattr(preview, "_export_gif_fn", lambda: fake_export_gif)

    final = tmp_path / "p.gif"
    cb = preview.make_step_callback(str(final), num_steps=4, emit=lambda _l: None)
    cb(0, 4, _fake_latents())

    assert seen and seen[0] != str(final), (
        "export_gif must write a temp file, not the path the GUI reads"
    )
    assert final.exists(), "the temp file must then be moved into place"
    assert final.read_bytes() == b"GIF89a-fake"


# ── The diffusers (CPU) path ─────────────────────────────────────────────────
#
# The TTNN path calls `generate_frames_temporal(on_step=...)` directly. The CPU
# path goes through a diffusers `AnimateDiffPipeline`, whose per-step hook is
# `callback_on_step_end(pipe, step, timestep, kwargs)` and whose latents are a
# single (B, C, F, H, W) tensor rather than a list of per-frame tensors. This
# adapter is what lets ONE preview callback serve both, so `--preview-path`
# means the same thing in either mode instead of being silently ignored in one.

def test_diffusers_adapter_is_none_without_a_callback():
    assert preview.as_diffusers_callback(None) is None


def test_diffusers_adapter_splits_latents_per_frame():
    torch = pytest.importorskip("torch")
    seen = {}

    def on_step(step_idx, total, frame_latents):
        seen["args"] = (step_idx, total, frame_latents)

    cb = preview.as_diffusers_callback(on_step, total_steps=8)
    latents = torch.zeros(1, 4, 3, 8, 8)  # (B, C, F, H, W) — 3 frames
    out = cb(object(), 2, 0, {"latents": latents})

    step_idx, total, frame_latents = seen["args"]
    assert (step_idx, total) == (2, 8)
    assert len(frame_latents) == 3, "one (1, 4, h, w) tensor per frame"
    assert tuple(frame_latents[0].shape) == (1, 4, 8, 8)
    assert out == {}, "diffusers expects the callback to return a kwargs dict"


def test_diffusers_adapter_passes_through_missing_latents():
    cb = preview.as_diffusers_callback(lambda *_a: None, total_steps=8)
    assert cb(object(), 0, 0, {}) == {}  # must not raise


def test_diffusers_adapter_never_raises_into_the_pipeline():
    def boom(*_a):
        raise RuntimeError("preview failed")

    torch = pytest.importorskip("torch")
    cb = preview.as_diffusers_callback(boom, total_steps=8)
    assert cb(object(), 0, 0, {"latents": torch.zeros(1, 4, 2, 8, 8)}) == {}
