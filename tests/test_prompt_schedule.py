# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for prompt_schedule — pure torch, no ttnn / no hardware.

Covers the interpolation helper (prompt travel core) and the CLI schedule
parser. These run on any machine with torch installed; no Blackhole needed.
"""

import pytest

torch = pytest.importorskip("torch")

from animatediff_ttnn.prompt_schedule import interpolate_embeddings, parse_schedule


# ── interpolate_embeddings ──────────────────────────────────────────────────

def test_single_keyframe_broadcasts_identical():
    """One keyframe → every frame gets the same embedding, all identical."""
    emb = torch.ones(2, 4)
    out = interpolate_embeddings([(0, emb)], num_frames=5)
    assert len(out) == 5
    for f in out:
        assert torch.equal(f, emb)


def test_returns_num_frames_length():
    """Output length always equals num_frames regardless of keyframe count."""
    a = torch.zeros(2, 4)
    b = torch.ones(2, 4)
    out = interpolate_embeddings([(0, a), (7, b)], num_frames=8)
    assert len(out) == 8


def test_two_keyframe_endpoints_exact():
    """Frames landing on keyframe indices reproduce that keyframe exactly."""
    a = torch.zeros(2, 4)
    b = torch.ones(2, 4)
    out = interpolate_embeddings([(0, a), (4, b)], num_frames=5)
    assert torch.equal(out[0], a)
    assert torch.equal(out[4], b)


def test_two_keyframe_midpoint_is_mean():
    """The exact midpoint between two keyframes equals their mean."""
    a = torch.zeros(2, 4)
    b = torch.ones(2, 4) * 2.0
    out = interpolate_embeddings([(0, a), (2, b)], num_frames=3)
    # frame 1 is halfway between 0 and 2 → t=0.5 → mean → all ones
    assert torch.allclose(out[1], (a + b) / 2.0)
    assert torch.allclose(out[1], torch.ones(2, 4))


def test_linear_interpolation_quarter_point():
    """Interpolation weight t=(i-a)/(b-a) is applied linearly."""
    a = torch.zeros(2, 4)
    b = torch.ones(2, 4) * 4.0
    out = interpolate_embeddings([(0, a), (4, b)], num_frames=5)
    # frame 1 → t=0.25 → 0.25*4 = 1.0
    assert torch.allclose(out[1], torch.ones(2, 4) * 1.0)
    # frame 3 → t=0.75 → 0.75*4 = 3.0
    assert torch.allclose(out[3], torch.ones(2, 4) * 3.0)


def test_before_first_keyframe_clamps_to_first():
    """Frames before the first keyframe index use the first embedding."""
    a = torch.ones(2, 4) * 5.0
    b = torch.ones(2, 4) * 9.0
    out = interpolate_embeddings([(2, a), (4, b)], num_frames=6)
    assert torch.equal(out[0], a)
    assert torch.equal(out[1], a)
    assert torch.equal(out[2], a)  # endpoint exact


def test_after_last_keyframe_clamps_to_last():
    """Frames after the last keyframe index use the last embedding."""
    a = torch.zeros(2, 4)
    b = torch.ones(2, 4) * 9.0
    out = interpolate_embeddings([(0, a), (2, b)], num_frames=5)
    assert torch.equal(out[2], b)  # endpoint exact
    assert torch.equal(out[3], b)
    assert torch.equal(out[4], b)


def test_unsorted_keyframes_are_sorted():
    """Keyframes passed out of order are handled by frame index, not list order."""
    a = torch.zeros(2, 4)
    b = torch.ones(2, 4) * 8.0
    out = interpolate_embeddings([(4, b), (0, a)], num_frames=5)
    assert torch.equal(out[0], a)
    assert torch.equal(out[4], b)
    assert torch.allclose(out[2], (a + b) / 2.0)


def test_three_keyframes_piecewise():
    """Three keyframes interpolate piecewise between consecutive pairs."""
    a = torch.zeros(2, 4)
    b = torch.ones(2, 4) * 2.0
    c = torch.ones(2, 4) * 6.0
    out = interpolate_embeddings([(0, a), (2, b), (4, c)], num_frames=5)
    assert torch.equal(out[0], a)
    assert torch.equal(out[2], b)
    assert torch.equal(out[4], c)
    assert torch.allclose(out[1], torch.ones(2, 4) * 1.0)  # mid of a,b
    assert torch.allclose(out[3], torch.ones(2, 4) * 4.0)  # mid of b,c


# ── parse_schedule ──────────────────────────────────────────────────────────

def test_parse_schedule_valid():
    out = parse_schedule(["0:spring meadow", "16:snowfall"])
    assert out == [(0, "spring meadow"), (16, "snowfall")]


def test_parse_schedule_sorts_by_frame():
    out = parse_schedule(["16:snowfall", "0:spring meadow"])
    assert out == [(0, "spring meadow"), (16, "snowfall")]


def test_parse_schedule_prompt_may_contain_colon():
    out = parse_schedule(["0:a city: at night"])
    assert out == [(0, "a city: at night")]


def test_parse_schedule_missing_colon_raises():
    with pytest.raises(ValueError):
        parse_schedule(["0 spring meadow"])


def test_parse_schedule_non_int_frame_raises():
    with pytest.raises(ValueError):
        parse_schedule(["first:spring meadow"])


def test_parse_schedule_empty_frame_raises():
    with pytest.raises(ValueError):
        parse_schedule([":snowfall"])


# ── per-frame conditioning length validation ────────────────────────────────
#
# Added after PR #7 review: both generators index (and, in the chunked sharding
# path, *slice*) text_embeddings_per_frame by frame. A wrong-length list used to
# surface as an IndexError deep in the denoising loop — or worse, a silent
# truncation from the slice producing a mis-sized shard — after the device was
# open and the UNet compiled. `_validate_per_frame_text` fails up front instead.


def test_validate_per_frame_text_accepts_matching_length():
    from animatediff_ttnn.temporal_attention import _validate_per_frame_text

    _validate_per_frame_text([object(), object(), object()], num_frames=3)


def test_validate_per_frame_text_allows_none_for_shared_prompt():
    """None is the single-prompt path and is always valid."""
    from animatediff_ttnn.temporal_attention import _validate_per_frame_text

    _validate_per_frame_text(None, num_frames=8)


@pytest.mark.parametrize("given,num_frames", [(2, 4), (5, 4), (1, 8), (16, 8)])
def test_validate_per_frame_text_rejects_mismatched_length(given, num_frames):
    """Both too-short and too-long are errors: a surplus would be dropped silently."""
    from animatediff_ttnn.temporal_attention import _validate_per_frame_text

    with pytest.raises(ValueError) as excinfo:
        _validate_per_frame_text([object()] * given, num_frames=num_frames)
    message = str(excinfo.value)
    # The message must name both numbers — that is the whole point of failing here.
    assert str(given) in message and str(num_frames) in message
    assert "text_embeddings_per_frame" in message


def test_generators_validate_before_touching_hardware():
    """The check must run before the ttnn import, so a bad call fails on any machine.

    Both generators are called with MagicMock stand-ins for every hardware
    argument. If validation were still positioned after the device/scheduler
    setup, these calls would raise something other than ValueError (or hang), so
    this test pins the *ordering*, not just the message.
    """
    from unittest.mock import MagicMock

    from animatediff_ttnn.temporal_attention import (
        generate_frames_motion,
        generate_frames_temporal,
    )

    common = dict(
        device=MagicMock(),
        ttnn_model=MagicMock(),
        ttnn_vae=MagicMock(),
        config=MagicMock(),
        torch_time_proj=MagicMock(),
        text_embeddings=torch.zeros(2, 96, 768),
        num_frames=4,
        num_steps=1,
        text_embeddings_per_frame=[torch.zeros(2, 96, 768)] * 3,
    )
    # generate_frames_motion additionally requires the injected motion modules.
    for generator, extra in (
        (generate_frames_temporal, {}),
        (generate_frames_motion, {"temporal_kernels": {}}),
    ):
        with pytest.raises(ValueError, match="text_embeddings_per_frame"):
            generator(**common, **extra)
