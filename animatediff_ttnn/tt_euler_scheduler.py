# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Minimal Euler scheduler wrapper for building TTNN time-embedding lists.

This wraps diffusers' EulerDiscreteScheduler to expose the same interface
that build_tlist() in ttnn_pipeline.py consumes:
  - .timesteps  (torch.Tensor of integer timestep values)
  - .init_noise_sigma  (float)
  - .num_inference_steps  (int, set by set_timesteps())

AnimateDiff-Lightning requires EulerDiscreteScheduler with
timestep_spacing="trailing" and beta_schedule="linear" — the scheduler
pre-computes sigma values under those constraints, so the distilled weights
produce coherent output. Using PNDM timestep spacing with Lightning weights
yields degraded/corrupted frames.

This class does NOT wrap the Euler step() logic into TTNN ops — the actual
denoising step stays on CPU via a plain diffusers EulerDiscreteScheduler.
The Tenstorrent silicon is used only for UNet spatial denoising, same as the
standard Blackhole path.
"""

from diffusers import EulerDiscreteScheduler


class TtEulerScheduler:
    """Thin wrapper around EulerDiscreteScheduler for Lightning on Blackhole.

    Compatible with build_tlist() in ttnn_pipeline.py.  All scheduler math
    (the Euler step equation) runs on CPU via plain diffusers — no TTNN ops.

    Args:
        num_train_timesteps: Training timesteps of the base model (1000 for SD 1.4).
        beta_start: Noise schedule start (Lightning uses linear 0.00085).
        beta_end: Noise schedule end (Lightning uses linear 0.012).
        beta_schedule: Must be "linear" for Lightning distilled weights.
        timestep_spacing: Must be "trailing" for Lightning distilled weights.
        device: Ignored — kept for interface parity with TtPNDMScheduler.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "linear",
        timestep_spacing: str = "trailing",
        device=None,
    ):
        self._cpu = EulerDiscreteScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            timestep_spacing=timestep_spacing,
        )
        # These are set by set_timesteps() below
        self.timesteps = None
        self.num_inference_steps = None

    @property
    def init_noise_sigma(self) -> float:
        """Maximum sigma — used to scale the initial latent before denoising.

        For Euler schedulers this is the largest sigma in the schedule, which
        is also the value of self._cpu.sigmas.max().
        """
        return float(self._cpu.init_noise_sigma)

    def set_timesteps(self, num_inference_steps: int, device=None):
        """Set the denoising schedule.

        Delegates to EulerDiscreteScheduler.set_timesteps() and caches
        .timesteps as integer values (matching TtPNDMScheduler's interface).
        """
        self._cpu.set_timesteps(num_inference_steps, device=device)
        # EulerDiscreteScheduler.timesteps are already integer-valued scalars
        self.timesteps = self._cpu.timesteps
        self.num_inference_steps = num_inference_steps

    def step(self, model_output, timestep, sample, return_dict=True):
        """CPU Euler step — delegates to diffusers EulerDiscreteScheduler.

        Accepts torch tensors (not TTNN tensors). The temporal_attention.py
        loop calls this with CPU tensors, so no TTNN ops are needed here.
        """
        return self._cpu.step(model_output, timestep, sample, return_dict=return_dict)
