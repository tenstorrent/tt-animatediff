# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for chain_blend_seed() — CPU-only, no hardware.

chain_blend_seed() is the extracted helper that blends previous-run denoised
latents into the current seed noise, biasing the denoiser toward the same
coarse composition.  These tests verify:

  1. The chain signal survives the blend + renorm at a perceptually meaningful
     level (≥8% correlation at alpha=0.35).
  2. The output remains unit-std so the scheduler's sigma scaling is correct.
  3. alpha=0.0 is a no-op (pure noise, no chain bias).
  4. alpha=1.0 fully replaces the noise (100% chain bias after renorm).
  5. The blend is monotone: higher alpha → higher correlation with previous latents.
  6. The function handles a .pt file path correctly (round-trips save/load).
  7. Spatial structure propagates: a high-energy blob in one quadrant of the
     previous latent creates a measurable spatial bias in the same quadrant.

The minimum correlation threshold (≥8% at alpha=0.35) is derived from
diffusion-model perceptual sensitivity: signals below ~5% correlation in
noise space are statistically undetectable across a 25-step denoising run.
"""

import math
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Import the function under test
# ---------------------------------------------------------------------------

from animatediff_ttnn.temporal_attention import chain_blend_seed


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_noise():
    torch.manual_seed(0)
    return torch.randn(1, 4, 64, 64)


@pytest.fixture
def prev_latents():
    """Realistic denoised latents: 8 frames, SD-range std (~0.5–1.5)."""
    torch.manual_seed(1)
    return torch.randn(8, 4, 64, 64) * 0.9


def _correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation between two tensors (flattened)."""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    stacked = torch.stack([a_flat, b_flat])
    return torch.corrcoef(stacked)[0, 1].item()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_chain_blend_signal_survives_at_alpha_035(base_noise, prev_latents):
    """At alpha=0.35 the chain signal must survive blend+renorm at >=8% correlation.

    Previous implementation used per-channel normalisation + ksize=9 avg_pool,
    which reduced signal to <1% — perceptually invisible.  This test enforces
    the minimum threshold for a detectable layout bias.
    """
    with tempfile.TemporaryDirectory() as d:
        pt_path = Path(d) / "prev.pt"
        torch.save(prev_latents, pt_path)
        result = chain_blend_seed(base_noise.clone(), str(pt_path), alpha=0.35)

    prev_mean = prev_latents.mean(dim=0, keepdim=True)
    corr = _correlation(result, prev_mean)
    assert corr >= 0.08, (
        f"Chain signal too weak: corr={corr:.4f} < 0.08 at alpha=0.35. "
        "Per-channel norm + large ksize destroys the signal before blending."
    )


def test_chain_blend_signal_survives_at_alpha_020(base_noise, prev_latents):
    """Even at alpha=0.20 the chain signal must be detectable (>=2% correlation)."""
    with tempfile.TemporaryDirectory() as d:
        pt_path = Path(d) / "prev.pt"
        torch.save(prev_latents, pt_path)
        result = chain_blend_seed(base_noise.clone(), str(pt_path), alpha=0.20)

    prev_mean = prev_latents.mean(dim=0, keepdim=True)
    corr = _correlation(result, prev_mean)
    assert corr >= 0.02, (
        f"Chain signal too weak at alpha=0.20: corr={corr:.4f} < 0.02"
    )


def test_output_is_unit_std(base_noise, prev_latents):
    """Output std must be 1.0 ± 0.05 so the scheduler sigma scaling is correct."""
    with tempfile.TemporaryDirectory() as d:
        pt_path = Path(d) / "prev.pt"
        torch.save(prev_latents, pt_path)
        result = chain_blend_seed(base_noise.clone(), str(pt_path), alpha=0.35)

    std = result.std().item()
    assert abs(std - 1.0) < 0.05, (
        f"Output std={std:.4f} — renorm failed or wasn't applied"
    )


def test_alpha_zero_returns_unchanged_noise(base_noise, prev_latents):
    """alpha=0.0 must return base_noise unchanged (no chain influence)."""
    with tempfile.TemporaryDirectory() as d:
        pt_path = Path(d) / "prev.pt"
        torch.save(prev_latents, pt_path)
        result = chain_blend_seed(base_noise.clone(), str(pt_path), alpha=0.0)

    assert torch.allclose(result, base_noise, atol=1e-5), (
        "alpha=0.0 should be a no-op but output differs from base_noise"
    )


def test_alpha_one_fully_replaces_noise(base_noise, prev_latents):
    """alpha=1.0 must produce output that is highly correlated with prev latents."""
    with tempfile.TemporaryDirectory() as d:
        pt_path = Path(d) / "prev.pt"
        torch.save(prev_latents, pt_path)
        result = chain_blend_seed(base_noise.clone(), str(pt_path), alpha=1.0)

    prev_mean = prev_latents.mean(dim=0, keepdim=True)
    corr = _correlation(result, prev_mean)
    assert corr >= 0.60, (
        f"alpha=1.0 should be dominated by prev latents but corr={corr:.4f} < 0.60"
    )


def test_higher_alpha_means_higher_correlation(base_noise, prev_latents):
    """Correlation with prev latents must be monotone increasing in alpha."""
    corrs = []
    with tempfile.TemporaryDirectory() as d:
        pt_path = Path(d) / "prev.pt"
        torch.save(prev_latents, pt_path)
        prev_mean = prev_latents.mean(dim=0, keepdim=True)
        for alpha in (0.10, 0.25, 0.40, 0.60):
            result = chain_blend_seed(base_noise.clone(), str(pt_path), alpha=alpha)
            corrs.append(_correlation(result, prev_mean))

    for i in range(len(corrs) - 1):
        assert corrs[i] < corrs[i + 1], (
            f"Correlation not monotone: alpha sequence produced {corrs}"
        )


def test_missing_file_returns_base_noise_unchanged(base_noise):
    """If the .pt path doesn't exist, return base_noise unchanged (soft fail)."""
    result = chain_blend_seed(base_noise.clone(), "/nonexistent/path/chain.pt", alpha=0.5)
    assert torch.allclose(result, base_noise, atol=1e-6), (
        "Missing chain file should silently return base_noise unchanged"
    )


def test_spatial_structure_propagates(base_noise):
    """High-energy region in prev latents creates measurable bias in same spatial region.

    Build a prev latent where the top-left quadrant has 5× the energy of the
    rest.  After blending at alpha=0.5, the top-left quadrant of the output
    should have higher mean-squared energy than the bottom-right quadrant.
    """
    torch.manual_seed(7)
    prev = torch.zeros(8, 4, 64, 64)
    # Strong positive signal in top-left quadrant only
    prev[:, :, :32, :32] = 3.0
    prev += torch.randn_like(prev) * 0.1  # small noise everywhere else

    with tempfile.TemporaryDirectory() as d:
        pt_path = Path(d) / "prev.pt"
        torch.save(prev, pt_path)
        result = chain_blend_seed(base_noise.clone(), str(pt_path), alpha=0.5)

    tl = result[:, :, :32, :32].pow(2).mean().item()
    br = result[:, :, 32:, 32:].pow(2).mean().item()
    assert tl > br, (
        f"Spatial bias not propagated: top-left energy {tl:.4f} <= bottom-right {br:.4f}"
    )


def test_output_shape_preserved(base_noise, prev_latents):
    """Output shape must match base_noise shape."""
    with tempfile.TemporaryDirectory() as d:
        pt_path = Path(d) / "prev.pt"
        torch.save(prev_latents, pt_path)
        result = chain_blend_seed(base_noise.clone(), str(pt_path), alpha=0.35)

    assert result.shape == base_noise.shape, (
        f"Shape mismatch: expected {base_noise.shape}, got {result.shape}"
    )


def test_output_is_finite(base_noise, prev_latents):
    """No NaN or Inf in output — renorm must not divide by zero."""
    with tempfile.TemporaryDirectory() as d:
        pt_path = Path(d) / "prev.pt"
        torch.save(prev_latents, pt_path)
        result = chain_blend_seed(base_noise.clone(), str(pt_path), alpha=0.35)

    assert result.isfinite().all(), "Output contains NaN or Inf"
