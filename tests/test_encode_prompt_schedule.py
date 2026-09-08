# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for encode_prompt's prompt-schedule mode with CLIP mocked out.

Runs without hardware and without downloading CLIP weights: the module-level
tokenizer/text-encoder caches are pre-populated with mocks so ``_encode_one``
never touches HuggingFace. QB2 validates the real CLIP + ttnn path end-to-end.
"""

import pytest

torch = pytest.importorskip("torch")

import animatediff_ttnn.generation_helpers as gh


class _FakeTokens:
    def __init__(self, input_ids):
        self.input_ids = input_ids


def _install_fake_clip(monkeypatch):
    """Wire deterministic per-text (1, 77, 768) embeddings; text -> fill value.

    Each unique prompt string maps to a constant-filled tensor so we can assert
    exactly which cond ended up on which frame.
    """
    fills = {
        "": 0.0,            # negative / uncond
        "spring": 1.0,
        "summer": 3.0,
        "winter": 9.0,
    }

    def fake_tokenizer(text, **kwargs):
        # Pass the raw string through as "input_ids" so the encoder can key on it.
        return _FakeTokens(text)

    fake_tokenizer.model_max_length = 77

    def fake_text_encoder(input_ids):
        fill = fills[input_ids]
        return [torch.full((1, 77, 768), fill)]  # [0] -> (1, 77, 768)

    monkeypatch.setattr(gh, "_clip_tokenizer", fake_tokenizer)
    monkeypatch.setattr(gh, "_clip_text_encoder", fake_text_encoder)
    return fills


def test_single_prompt_mode_unchanged(monkeypatch):
    """No schedule → one (2, 96, 768) tensor, [uncond, cond]."""
    _install_fake_clip(monkeypatch)
    out = gh.encode_prompt("spring", negative_prompt="")
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 96, 768)
    assert torch.allclose(out[0], torch.zeros(96, 768))      # uncond
    assert torch.allclose(out[1, :77], torch.ones(77, 768))  # cond (spring=1.0)


def test_schedule_mode_returns_per_frame_list(monkeypatch):
    """Schedule → list of N (2, 96, 768) tensors."""
    _install_fake_clip(monkeypatch)
    out = gh.encode_prompt(
        "unused", negative_prompt="",
        prompt_schedule=[(0, "spring"), (4, "winter")], num_frames=5,
    )
    assert isinstance(out, list)
    assert len(out) == 5
    for t in out:
        assert t.shape == (2, 96, 768)


def test_schedule_endpoints_match_keyframe_encodings(monkeypatch):
    """Frames on keyframe indices reproduce that keyframe's cond exactly."""
    _install_fake_clip(monkeypatch)
    out = gh.encode_prompt(
        "unused", negative_prompt="",
        prompt_schedule=[(0, "spring"), (4, "winter")], num_frames=5,
    )
    # cond at frame 0 == spring (1.0), at frame 4 == winter (9.0)
    assert torch.allclose(out[0][1, :77], torch.full((77, 768), 1.0))
    assert torch.allclose(out[4][1, :77], torch.full((77, 768), 9.0))
    # midpoint frame 2 == mean(1, 9) = 5.0
    assert torch.allclose(out[2][1, :77], torch.full((77, 768), 5.0))


def test_schedule_uncond_constant_across_frames(monkeypatch):
    """The uncond half is identical on every frame — only cond travels."""
    _install_fake_clip(monkeypatch)
    out = gh.encode_prompt(
        "unused", negative_prompt="",
        prompt_schedule=[(0, "spring"), (2, "summer")], num_frames=3,
    )
    for t in out:
        assert torch.allclose(t[0], torch.zeros(96, 768))


def test_schedule_requires_num_frames(monkeypatch):
    _install_fake_clip(monkeypatch)
    with pytest.raises(ValueError):
        gh.encode_prompt("x", prompt_schedule=[(0, "spring")], num_frames=None)


def test_empty_schedule_raises(monkeypatch):
    _install_fake_clip(monkeypatch)
    with pytest.raises(ValueError):
        gh.encode_prompt("x", prompt_schedule=[], num_frames=4)
