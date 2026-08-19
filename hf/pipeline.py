# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Diffusers custom pipeline for tt-animatediff.

Loaded from the Hugging Face Hub with::

    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        "episod/tt-animatediff",
        custom_pipeline="episod/tt-animatediff",
        trust_remote_code=True,
    )
    frames = pipe("a swirling nebula, teal and gold").frames

This module carries NO generation logic. It marshals arguments into
``animatediff_ttnn.generate_animation()`` — the tested entry point that owns
backend selection, the CPU fallback, the device singleton and chain mode — and
wraps the result in a diffusers-shaped output object.

Two constraints here are imposed by how diffusers loads custom pipelines, and
both are easy to "fix" back into breakage:

1. ``__init__`` declares named parameters with defaults and takes no
   ``**kwargs``. diffusers derives its expected-component list from this
   signature; a ``**kwargs``-only signature fails the load outright.
2. The string ``animatediff_ttnn`` never appears in an ``import`` statement,
   at any indentation. diffusers' ``check_imports`` regex-scans this file and,
   on diffusers 0.32.x, raises ImportError at *load* time for any module that
   is not installed — which would lock out every user who has not already
   pip-installed this package, i.e. exactly the audience the CPU path serves.
   The package is reached through ``importlib.import_module`` instead.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput

# numpy and torch are imported here rather than in the methods that use them
# (output_type="np" and the @torch.no_grad() decorator, both added in Task 3):
# check_imports resolves them fine because diffusers depends on both, and a
# lazy import inside a method would be scanned identically anyway.

#: Name of the package this pipeline delegates to. Never written as an import.
PACKAGE_NAME = "animatediff_ttnn"

#: Compute backends accepted by ``__call__``; forwarded to generate_animation().
VALID_MODES = ("auto", "blackhole", "cpu", "sim")


@dataclass
class TTAnimateDiffPipelineOutput(BaseOutput):
    """Stand-in for diffusers' ``AnimateDiffPipelineOutput``.

    ``frames`` is a list of PIL Images, or a stacked numpy array when
    ``output_type="np"`` was requested.
    """

    frames: Any


class TTAnimateDiffPipeline(DiffusionPipeline):
    """AnimateDiff on Tenstorrent Blackhole, with a CPU fallback.

    Construction loads nothing: no device is opened and no weights are fetched
    until ``__call__``. That keeps ``from_pretrained`` fast and offline-safe.
    """

    def __init__(
        self,
        base_model: str = "CompVis/stable-diffusion-v1-4",
        motion_adapter: str = "guoyww/animatediff-motion-adapter-v1-5-2",
        lightning_repo: str = "ByteDance/AnimateDiff-Lightning",
        code_repo: str = "episod/tt-animatediff",
        temporal_alpha: float = 0.35,
        num_frames: int = 8,
        num_steps: int = 25,
        guidance_scale: float = 7.5,
    ):
        super().__init__()
        self.register_to_config(
            base_model=base_model,
            motion_adapter=motion_adapter,
            lightning_repo=lightning_repo,
            code_repo=code_repo,
            temporal_alpha=temporal_alpha,
            num_frames=num_frames,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
        )
        self._resolved_mode: Optional[str] = None

    @property
    def resolved_mode(self) -> Optional[str]:
        """Backend the last ``__call__`` actually ran on, or None before one.

        Lets a caller state which backend ran instead of guessing — notably the
        Space, where ``mode="auto"`` always lands on CPU.
        """
        return self._resolved_mode
