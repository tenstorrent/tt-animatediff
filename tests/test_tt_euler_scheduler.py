# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for TtEulerScheduler — runs without Blackhole hardware (CPU only).

Verifies that TtEulerScheduler exposes the interface that build_tlist() and
generate_frames_temporal() rely on.
"""

import torch
import pytest

from animatediff_ttnn.tt_euler_scheduler import TtEulerScheduler


def test_set_timesteps_sets_num_inference_steps():
    sched = TtEulerScheduler()
    sched.set_timesteps(4)
    assert sched.num_inference_steps == 4


def test_timesteps_is_tensor_after_set_timesteps():
    sched = TtEulerScheduler()
    sched.set_timesteps(4)
    assert isinstance(sched.timesteps, torch.Tensor)


def test_timesteps_length_matches_steps():
    """Euler trailing schedule produces exactly num_steps timestep values."""
    sched = TtEulerScheduler()
    sched.set_timesteps(4)
    # EulerDiscreteScheduler with trailing spacing yields exactly num_steps entries
    assert len(sched.timesteps) == 4


def test_timesteps_descending():
    """Timesteps run from high → low (standard denoising direction)."""
    sched = TtEulerScheduler()
    sched.set_timesteps(8)
    ts = sched.timesteps.tolist()
    assert ts == sorted(ts, reverse=True), f"Timesteps not descending: {ts}"


def test_init_noise_sigma_positive():
    sched = TtEulerScheduler()
    sched.set_timesteps(4)
    assert sched.init_noise_sigma > 0.0


def test_init_noise_sigma_is_float():
    sched = TtEulerScheduler()
    sched.set_timesteps(4)
    assert isinstance(sched.init_noise_sigma, float)


def test_step_returns_prev_sample():
    """step() returns an object with a .prev_sample attribute."""
    sched = TtEulerScheduler()
    sched.set_timesteps(4)
    t = sched.timesteps[0]
    noise_pred = torch.randn(1, 4, 8, 8)
    latent = torch.randn(1, 4, 8, 8) * sched.init_noise_sigma
    result = sched.step(noise_pred, t, latent)
    assert hasattr(result, "prev_sample")
    assert result.prev_sample.shape == latent.shape


def test_timesteps_differ_from_pndm():
    """Euler trailing timesteps should differ from PNDM scaled_linear timesteps.

    This is the whole point: Lightning needs different spacing than PNDM.
    """
    from diffusers import PNDMScheduler

    euler_sched = TtEulerScheduler(
        beta_schedule="linear", timestep_spacing="trailing"
    )
    euler_sched.set_timesteps(4)

    pndm_sched = PNDMScheduler(
        beta_schedule="scaled_linear",
        skip_prk_steps=True,
        steps_offset=1,
    )
    pndm_sched.set_timesteps(4)

    euler_ts = set(euler_sched.timesteps.tolist())
    pndm_ts = set(pndm_sched.timesteps.tolist())
    assert euler_ts != pndm_ts, (
        f"Euler and PNDM timesteps should differ but both gave: {euler_ts}"
    )


def test_set_timesteps_idempotent():
    """Calling set_timesteps twice produces the same result as once."""
    sched = TtEulerScheduler()
    sched.set_timesteps(4)
    ts1 = sched.timesteps.clone()
    sched.set_timesteps(4)
    ts2 = sched.timesteps.clone()
    assert torch.equal(ts1, ts2)


def test_device_arg_ignored():
    """TtEulerScheduler accepts device= kwarg for interface parity but ignores it."""
    sched = TtEulerScheduler(device="this_is_ignored")
    sched.set_timesteps(4)
    assert len(sched.timesteps) == 4


@pytest.mark.parametrize("step", [2, 4, 8])
def test_lightning_step_counts(step):
    """All Lightning-valid step counts (2, 4, 8) produce correct-length timesteps."""
    sched = TtEulerScheduler()
    sched.set_timesteps(step)
    assert len(sched.timesteps) == step
