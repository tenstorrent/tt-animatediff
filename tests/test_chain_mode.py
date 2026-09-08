# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Chain-mode round-trip tests — CPU-only, no hardware.

tests/test_chain_blend.py covers chain_blend_seed() in isolation, feeding it
tensors it constructs itself. These tests cover the part that spans two runs:
the on-disk contract between the *writer* (chain_save, inside
generate_frames_temporal) and the *reader* (chain_blend_seed via chain_from).

That contract is what the World's Fair chain depends on -- unisphere-1964 →
1980 → 2000 → ... each feed the next run's noise, so a format mismatch or a
silently-dropped signal breaks the whole sequence rather than one clip.

Covered here:

  1. The exact file format the writer produces round-trips through the reader.
     The writer does `torch.save(torch.cat(frame_latents, dim=0))` over N
     tensors of shape (1, 4, lh, lw), giving (N, 4, lh, lw).
  2. Frame count is decoupled from the blend: the reader frame-averages, so an
     8-frame and a 16-frame save with the same mean behave identically.
  3. Chaining across several generations keeps a detectable signal, and the
     shipped World's Fair chain length (6 links) does not decay to noise.
  4. Real World's Fair latents on disk, when present, load and blend.
  5. The reader loads with weights_only=True, so a chain file cannot execute
     arbitrary code on load.
"""

import math
from pathlib import Path

import pytest
import torch

from animatediff_ttnn.temporal_attention import chain_blend_seed


LATENT_H = LATENT_W = 64  # 512px / 8 (VAE downscale)


def _correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation between two tensors, flattened."""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    a_c = a_flat - a_flat.mean()
    b_c = b_flat - b_flat.mean()
    denom = a_c.norm() * b_c.norm()
    return float((a_c @ b_c) / denom) if denom > 0 else 0.0


def _write_chain_file(path: Path, num_frames: int, seed: int = 0) -> torch.Tensor:
    """Write a chain file exactly the way generate_frames_temporal does.

    The generator accumulates one (1, 4, lh, lw) latent per frame and saves
    `torch.cat(frame_latents, dim=0)`. Mimicking that concatenation here is the
    point: if the writer's layout ever changes, these tests should stop matching
    the reader.
    """
    torch.manual_seed(seed)
    frame_latents = [torch.randn(1, 4, LATENT_H, LATENT_W) * 0.9 for _ in range(num_frames)]
    stacked = torch.cat(frame_latents, dim=0)
    assert stacked.shape == (num_frames, 4, LATENT_H, LATENT_W)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stacked, path)
    return stacked


@pytest.fixture
def base_noise():
    torch.manual_seed(1234)
    return torch.randn(1, 4, LATENT_H, LATENT_W)


# ---------------------------------------------------------------------------
# Writer → reader round trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_frames", [4, 8, 16])
def test_writer_format_round_trips_through_reader(tmp_path, base_noise, num_frames):
    """Every frame count the mesh-sharding guard allows must load and blend."""
    path = tmp_path / "chain" / "prev.pt"
    _write_chain_file(path, num_frames)

    out = chain_blend_seed(base_noise.clone(), str(path), alpha=0.35)

    assert out.shape == base_noise.shape
    assert torch.isfinite(out).all()
    assert not torch.allclose(out, base_noise), "the chain signal must change the noise"


def test_blend_uses_the_frame_mean_not_the_first_frame(tmp_path, base_noise):
    """The reader averages across frames; a file whose mean is zero is a no-op
    in the signal direction even though individual frames are large."""
    path = tmp_path / "antisymmetric.pt"
    frame = torch.randn(1, 4, LATENT_H, LATENT_W) * 2.0
    # Two frames that cancel exactly -> frame-mean is all zeros.
    torch.save(torch.cat([frame, -frame], dim=0), path)

    out = chain_blend_seed(base_noise.clone(), str(path), alpha=0.5)

    # With a zero mean, blending only rescales the base noise, so direction is
    # preserved almost perfectly.
    assert _correlation(out, base_noise) > 0.999


def test_frame_count_does_not_change_the_result_for_an_equal_mean(tmp_path, base_noise):
    """8 frames and 16 copies of the same mean must blend identically.

    This is what lets chain files be reused across runs with different
    num_frames, as the World's Fair chain does.
    """
    mean_frame = torch.randn(1, 4, LATENT_H, LATENT_W) * 0.9

    p8 = tmp_path / "eight.pt"
    p16 = tmp_path / "sixteen.pt"
    torch.save(mean_frame.repeat(8, 1, 1, 1), p8)
    torch.save(mean_frame.repeat(16, 1, 1, 1), p16)

    out8 = chain_blend_seed(base_noise.clone(), str(p8), alpha=0.35)
    out16 = chain_blend_seed(base_noise.clone(), str(p16), alpha=0.35)

    assert torch.allclose(out8, out16, atol=1e-6)


# ---------------------------------------------------------------------------
# Multi-generation chains
# ---------------------------------------------------------------------------

def test_signal_survives_a_six_link_chain(tmp_path, base_noise):
    """The shipped World's Fair chain is 6 links (1964 → 2064).

    Each link re-blends the *previous run's output* into fresh noise, so the
    original layout signal attenuates geometrically. This pins that a 6-link
    chain still carries a detectable trace of link 1 rather than decaying into
    pure noise -- the property the sequence's visual continuity relies on.
    """
    origin = tmp_path / "link0.pt"
    origin_stack = _write_chain_file(origin, 8, seed=7)
    origin_mean = origin_stack.mean(dim=0, keepdim=True)

    latents = base_noise.clone()
    current = origin
    for link in range(6):
        latents = chain_blend_seed(latents, str(current), alpha=0.35)
        # Feed this run's result forward as the next run's chain file, the way
        # chain_save → chain_from does between generations.
        current = tmp_path / f"link{link + 1}.pt"
        torch.save(latents.repeat(8, 1, 1, 1), current)

    corr = _correlation(latents, origin_mean)
    assert corr > 0.05, f"6-link chain lost the origin signal (corr={corr:.4f})"
    assert torch.isfinite(latents).all()


def test_chain_output_stays_unit_std_across_links(tmp_path, base_noise):
    """Renormalisation must hold at every link.

    The scheduler's sigma scaling assumes unit-std noise at t=T. If std drifted
    per link, later clips in a chain would be progressively over- or
    under-denoised.
    """
    latents = base_noise.clone()
    current = tmp_path / "start.pt"
    _write_chain_file(current, 8, seed=3)

    for link in range(5):
        latents = chain_blend_seed(latents, str(current), alpha=0.35)
        assert math.isclose(float(latents.std()), 1.0, rel_tol=0.05), (
            f"link {link} drifted to std={float(latents.std()):.4f}"
        )
        current = tmp_path / f"l{link}.pt"
        torch.save(latents.repeat(8, 1, 1, 1), current)


# ---------------------------------------------------------------------------
# Defaults and documented ranges
# ---------------------------------------------------------------------------

def test_api_default_chain_alpha_matches_the_documented_value():
    """generate_animation()'s chain_alpha default is 0.6.

    Note this sits above the 0.20-0.55 "effective range" chain_blend_seed's own
    docstring recommends. The test records the shipped value so a change is
    deliberate rather than accidental; it does not endorse 0.6 as optimal.
    """
    import inspect

    from animatediff_ttnn import generate_animation

    default = inspect.signature(generate_animation).parameters["chain_alpha"].default
    assert default == 0.6


def test_higher_alpha_increases_similarity_to_the_chain_file(tmp_path, base_noise):
    """Monotonicity across the range the API can actually request."""
    path = tmp_path / "prev.pt"
    stack = _write_chain_file(path, 8, seed=11)
    prev_mean = stack.mean(dim=0, keepdim=True)

    corrs = [
        _correlation(chain_blend_seed(base_noise.clone(), str(path), alpha=a), prev_mean)
        for a in (0.2, 0.35, 0.6, 0.9)
    ]

    assert corrs == sorted(corrs), f"correlation must rise with alpha, got {corrs}"


# ---------------------------------------------------------------------------
# Safety and failure modes
# ---------------------------------------------------------------------------

def test_chain_file_cannot_execute_code_on_load(tmp_path, base_noise):
    """The reader uses weights_only=True, so a hostile chain file is inert.

    Chain files are ordinary paths a user can point at, so a pickle that runs
    code on load would be a real hazard.
    """
    class _Exploit:
        def __reduce__(self):
            return (print, ("chain file executed code",))

    path = tmp_path / "hostile.pt"
    torch.save({"payload": _Exploit()}, path)

    with pytest.raises(Exception) as exc:
        chain_blend_seed(base_noise.clone(), str(path), alpha=0.35)

    # Any refusal is acceptable; silently executing the payload is not.
    assert "weights_only" in str(exc.value) or "Unsupported" in str(exc.value) \
        or "GLOBAL" in str(exc.value) or isinstance(exc.value, (AttributeError, TypeError))


def test_resolution_mismatch_is_not_silently_ignored(tmp_path):
    """A chain file from a different output size must not blend into a
    mis-shaped seed. Broadcasting the wrong way would corrupt the latent."""
    path = tmp_path / "small.pt"
    torch.save(torch.randn(8, 4, 32, 32), path)
    base = torch.randn(1, 4, LATENT_H, LATENT_W)

    with pytest.raises(RuntimeError):
        chain_blend_seed(base, str(path), alpha=0.35)


def test_missing_chain_file_is_a_warning_not_a_crash(tmp_path, base_noise, capsys):
    """A chain sequence whose first link has no predecessor must still run."""
    out = chain_blend_seed(base_noise.clone(), str(tmp_path / "absent.pt"), alpha=0.35)

    assert torch.equal(out, base_noise)
    assert "not found" in capsys.readouterr().out


def test_alpha_zero_short_circuits_before_reading_the_file(tmp_path, base_noise):
    """alpha=0 means "no chaining", so a present-but-unreadable file is fine."""
    path = tmp_path / "garbage.pt"
    path.write_bytes(b"not a torch file at all")

    out = chain_blend_seed(base_noise.clone(), str(path), alpha=0.0)

    assert torch.equal(out, base_noise)


# ---------------------------------------------------------------------------
# The real shipped chain assets
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SHIPPED_CHAIN_LATENTS = sorted((REPO_ROOT / "docs/assets/chain/latents").glob("*.pt"))


@pytest.mark.skipif(not SHIPPED_CHAIN_LATENTS, reason="chain latents not present")
@pytest.mark.parametrize("latent_path", SHIPPED_CHAIN_LATENTS,
                         ids=lambda p: p.stem)
def test_shipped_chain_latents_load_and_blend(latent_path, base_noise):
    """The committed chain latents must stay loadable by the current reader.

    These files are the reproducibility record for the chain demo. If a torch
    upgrade or a format change makes them unreadable, that is a real break and
    should fail here rather than at demo time.
    """
    stack = torch.load(latent_path, map_location="cpu", weights_only=True)

    assert stack.ndim == 4, f"expected (F, 4, lh, lw), got {tuple(stack.shape)}"
    assert stack.shape[1] == 4, "latents must have 4 channels"
    assert torch.isfinite(stack).all()

    seed = torch.randn(1, *stack.shape[1:])
    out = chain_blend_seed(seed.clone(), str(latent_path), alpha=0.35)

    assert out.shape == seed.shape
    assert torch.isfinite(out).all()
    assert math.isclose(float(out.std()), 1.0, rel_tol=0.05)
