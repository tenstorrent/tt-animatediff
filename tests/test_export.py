# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for export_mp4() and export_gif() — CPU-only, no hardware.

export_mp4() shells out to ffmpeg over a pipe, which is exactly the sort of
plumbing that fails in ways unit tests catch and eyeballing does not:

  1. The stderr deadlock. Writing all frames to ffmpeg's stdin while stderr is
     a PIPE that nobody drains will hang forever once ffmpeg's diagnostics
     exceed the OS pipe buffer (~64 KB). export_mp4() uses communicate() to
     drain both concurrently; test_many_frames_do_not_deadlock exercises a
     frame count large enough to matter, under a timeout.
  2. Failure reporting. A non-zero ffmpeg exit must raise with ffmpeg's stderr
     attached, otherwise callers get a bare "it didn't work".
  3. Preconditions — empty frame list, and ffmpeg missing from PATH.

The tests that actually invoke ffmpeg skip cleanly when it is not installed,
so this file stays runnable on a bare checkout.
"""

import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from animatediff_ttnn import export_gif, export_mp4


HAS_FFMPEG = shutil.which("ffmpeg") is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not on PATH")


@pytest.fixture
def frames():
    """Eight small, visually distinct RGB frames."""
    return [
        Image.new("RGB", (64, 64), (i * 30 % 256, 80, 200 - i * 20 % 256))
        for i in range(8)
    ]


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------

def test_export_mp4_rejects_empty_frame_list(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        export_mp4([], str(tmp_path / "out.mp4"))


def test_export_mp4_reports_missing_ffmpeg(frames, tmp_path):
    """The error must name ffmpeg and tell the user how to install it."""
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="ffmpeg not found") as exc:
            export_mp4(frames, str(tmp_path / "out.mp4"))

    message = str(exc.value)
    assert "apt install ffmpeg" in message
    assert "brew install ffmpeg" in message


def test_export_mp4_checks_frames_before_ffmpeg(tmp_path):
    """An empty list must fail on its own merits, even with no ffmpeg installed."""
    with patch("shutil.which", return_value=None):
        with pytest.raises(ValueError, match="empty"):
            export_mp4([], str(tmp_path / "out.mp4"))


# ---------------------------------------------------------------------------
# Real encoding
# ---------------------------------------------------------------------------

@needs_ffmpeg
def test_export_mp4_writes_a_decodable_file(frames, tmp_path):
    out = tmp_path / "out.mp4"
    export_mp4(frames, str(out), fps=8)

    assert out.is_file()
    assert out.stat().st_size > 0

    # Prove it is a real video rather than a non-empty file: ask ffprobe for
    # the stream properties and check them against what we asked for.
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height,nb_read_frames",
         "-count_frames", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, timeout=60,
    )
    if probe.returncode != 0:
        pytest.skip(f"ffprobe unavailable or failed: {probe.stderr.strip()}")

    fields = probe.stdout.strip().split(",")
    assert fields[0] == "h264", f"expected h264, got {fields[0]}"
    assert (int(fields[1]), int(fields[2])) == (64, 64)
    assert int(fields[3]) == len(frames), "every frame must land in the video"


@needs_ffmpeg
def test_export_mp4_respects_fps(frames, tmp_path):
    """fps must reach ffmpeg -- a wrong rate changes playback speed silently."""
    out = tmp_path / "fast.mp4"
    export_mp4(frames, str(out), fps=24)

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, timeout=60,
    )
    if probe.returncode != 0:
        pytest.skip("ffprobe unavailable")
    assert probe.stdout.strip() == "24/1"


@needs_ffmpeg
def test_many_frames_do_not_deadlock(tmp_path):
    """Regression guard for the stderr-pipe deadlock.

    With proc.wait() plus stderr=PIPE this hangs indefinitely once ffmpeg's
    diagnostics fill the pipe buffer. 64 frames is comfortably past the point
    where the old code stalled; the timeout turns a hang into a failure.
    """
    many = [Image.new("RGB", (128, 128), (i * 4 % 256, 60, 90)) for i in range(64)]
    out = tmp_path / "many.mp4"

    export_mp4(many, str(out), fps=8)

    assert out.is_file() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Failure propagation
# ---------------------------------------------------------------------------

def test_export_mp4_raises_with_ffmpeg_stderr_on_failure(frames, tmp_path):
    """A non-zero exit must surface ffmpeg's own diagnostics, not just a code."""
    class _FailingProc:
        returncode = 1

        def communicate(self, input=None):
            return b"", b"Unknown encoder 'libx264'"

    with patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("subprocess.Popen", return_value=_FailingProc()):
        with pytest.raises(RuntimeError) as exc:
            export_mp4(frames, str(tmp_path / "out.mp4"))

    message = str(exc.value)
    assert "exited with code 1" in message
    assert "Unknown encoder 'libx264'" in message


def test_export_mp4_uses_communicate_not_wait(frames, tmp_path):
    """Pin the deadlock fix in place.

    communicate() is what drains stderr while stdin is being written. If a
    future refactor goes back to write()+wait(), this fails loudly rather than
    waiting for a large-output run to hang in production.
    """
    calls = []

    class _RecordingProc:
        returncode = 0

        def communicate(self, input=None):
            calls.append(("communicate", len(input or b"")))
            return b"", b""

        def wait(self, timeout=None):
            calls.append(("wait", 0))
            return 0

    with patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("subprocess.Popen", return_value=_RecordingProc()):
        export_mp4(frames, str(tmp_path / "out.mp4"))

    assert [name for name, _ in calls] == ["communicate"]
    # All eight PNG-encoded frames must be handed over in that single call.
    assert calls[0][1] > 0


def test_export_mp4_passes_output_path_and_pixel_format(frames, tmp_path):
    """yuv420p + libx264 is what makes the file play in browsers and previews."""
    out = tmp_path / "checked.mp4"
    captured = {}

    class _OkProc:
        returncode = 0

        def communicate(self, input=None):
            return b"", b""

    def _capture(cmd, **kwargs):
        captured["cmd"] = cmd
        return _OkProc()

    with patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("subprocess.Popen", side_effect=_capture):
        export_mp4(frames, str(out), fps=12)

    cmd = captured["cmd"]
    assert cmd[-1] == str(out), "output path must be the final ffmpeg argument"
    assert "libx264" in cmd
    assert "yuv420p" in cmd
    assert "12" in cmd, "fps must appear in the command"


# ---------------------------------------------------------------------------
# export_gif
# ---------------------------------------------------------------------------

def test_export_gif_writes_an_animated_gif(frames, tmp_path):
    out = tmp_path / "out.gif"
    export_gif(frames, str(out))

    assert out.is_file() and out.stat().st_size > 0
    with Image.open(out) as img:
        assert img.format == "GIF"
        assert getattr(img, "n_frames", 1) == len(frames), "GIF must be animated"


def test_export_gif_creates_parent_directories(frames, tmp_path):
    """Callers pass nested output paths (output/run-3/anim.gif) routinely.

    export_gif() mkdir(parents=True)s the destination's parent, so this asserts
    rather than tolerating a failure: the earlier version skipped on
    FileNotFoundError/OSError, which would have silently swallowed exactly the
    regression this test exists to catch.
    """
    out = tmp_path / "nested" / "deeper" / "out.gif"
    export_gif(frames, str(out))
    assert out.is_file()
