# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Phase 1: AnimateDiff baseline using diffusers AnimateDiffPipeline + MotionAdapter.

Uses CompVis/stable-diffusion-v1-4 with guoyww/animatediff-motion-adapter-v1-5-2.
The MotionAdapter injects temporal attention inside each SD 1.4 UNet transformer
block at 320-dim features — the correct location for mm_sd_v15_v2.ckpt weights.
No TT hardware required.

Lightning mode uses ByteDance/AnimateDiff-Lightning distilled checkpoints
(arXiv:2403.12706). 4-step Lightning ≈ 25-step standard quality, ~6× faster.
Requires EulerDiscreteScheduler with timestep_spacing="trailing" and CFG=1.0.
Supported step counts: 2, 4, 8 (each has a distinct distilled checkpoint).
"""

from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from diffusers import AnimateDiffPipeline, DDIMScheduler, EulerDiscreteScheduler, MotionAdapter

# LIGHTNING_REPO is re-exported from here for existing importers. Every repo this module
# loads goes through revision_for(), which pins the three known repos and returns None
# (default branch, as before) for a caller-supplied model_id it has no sha for.
from animatediff_ttnn.weights_pins import (  # noqa: F401  (re-export)
    LIGHTNING_REPO,
    MOTION_ADAPTER_REPO,
    SD14_REPO,
    revision_for,
)
LIGHTNING_STEPS = (2, 4, 8)  # 1-step is research-only


def create_lightning_pipeline(
    step: int = 4,
    model_id: str = SD14_REPO,
    torch_dtype: torch.dtype = torch.float32,
) -> AnimateDiffPipeline:
    """Create an AnimateDiff-Lightning pipeline for fast CPU generation.

    Uses ByteDance's distilled motion adapter weights (arXiv:2403.12706).
    4-step Lightning ≈ 25-step standard AnimateDiff quality, ~6× faster.

    Args:
        step: Distillation step count — must be 2, 4, or 8 (distinct checkpoints).
              Use 4 (default) for the best quality/speed tradeoff; 2 for fastest.
        model_id: Base SD model. Lightning works best with stylised bases.
        torch_dtype: torch.float32 (CPU) or torch.float16 (GPU).

    Returns:
        AnimateDiffPipeline configured with EulerDiscreteScheduler (required).
    """
    if step not in LIGHTNING_STEPS:
        raise ValueError(f"step must be one of {LIGHTNING_STEPS}, got {step}")

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
    adapter = MotionAdapter()
    adapter.load_state_dict(
        load_file(hf_hub_download(LIGHTNING_REPO, ckpt, revision=revision_for(LIGHTNING_REPO))),
        strict=True,
    )
    adapter = adapter.to(dtype=torch_dtype)

    pipe = AnimateDiffPipeline.from_pretrained(
        model_id,
        motion_adapter=adapter,
        torch_dtype=torch_dtype,
        revision=revision_for(model_id),
    )
    # Lightning requires EulerDiscreteScheduler with these exact settings
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
        beta_schedule="linear",
    )
    return pipe


def create_animatediff_pipeline(
    model_id: str = SD14_REPO,
    adapter_id: str = MOTION_ADAPTER_REPO,
    torch_dtype: torch.dtype = torch.float32,
) -> AnimateDiffPipeline:
    """Create a diffusers AnimateDiffPipeline with MotionAdapter.

    Downloads from HuggingFace cache on first call. No TT hardware required.
    Run `hf download CompVis/stable-diffusion-v1-4` and
    `hf download guoyww/animatediff-motion-adapter-v1-5-2` beforehand.

    Returns:
        diffusers.AnimateDiffPipeline ready for inference
    """
    adapter = MotionAdapter.from_pretrained(
        adapter_id, torch_dtype=torch_dtype, revision=revision_for(adapter_id)
    )
    scheduler = DDIMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        revision=revision_for(model_id),
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )
    pipe = AnimateDiffPipeline.from_pretrained(
        model_id,
        motion_adapter=adapter,
        scheduler=scheduler,
        torch_dtype=torch_dtype,
        revision=revision_for(model_id),
    )
    return pipe


def generate(
    pipe: AnimateDiffPipeline,
    prompt: str,
    negative_prompt: str = "",
    num_frames: int = 16,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 25,
    seed: int = 42,
    on_step=None,
) -> List[Image.Image]:
    """Generate an animated sequence from a text prompt.

    Args:
        pipe: AnimateDiffPipeline from create_animatediff_pipeline()
        prompt: Text prompt describing the animation
        negative_prompt: What to avoid in the output
        num_frames: Number of frames (8 or 16 recommended)
        guidance_scale: CFG scale — higher = more prompt-adherent (7.5 is standard)
        num_inference_steps: Denoising steps (25 balances speed and quality)
        seed: Random seed for reproducibility
        on_step: Optional ``on_step(step_idx, total_steps, frame_latents)``
            called as denoising proceeds — the same signature
            ``generate_frames_temporal`` takes, so one preview callback serves
            both the CPU and TTNN paths. Adapted to diffusers'
            ``callback_on_step_end`` shape by
            ``animatediff_ttnn.preview.as_diffusers_callback``.

    Returns:
        List of PIL Images, one per frame, 512x512
    """
    kwargs = {}
    if on_step is not None:
        from animatediff_ttnn.preview import as_diffusers_callback

        kwargs["callback_on_step_end"] = as_diffusers_callback(
            on_step, total_steps=num_inference_steps
        )

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator().manual_seed(seed),
        **kwargs,
    )
    return output.frames[0]


def export_gif(frames: List[Image.Image], output_path: str, fps: int = 8) -> None:
    """Save a list of PIL Images as an animated GIF.

    Args:
        frames: List of PIL Images (all same size)
        output_path: Destination file path, e.g. 'output/result.gif'
        fps: Frames per second (duration = 1000 // fps ms per frame)
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // fps,
        loop=0,
    )
