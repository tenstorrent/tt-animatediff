# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Cross-frame temporal attention for AnimateDiff Phase 2.5 on Blackhole.

Applies self-attention across N frames at each denoising step, giving genuine
temporal coherence without requiring the MotionAdapter TemporalTransformer
(which operates on 320-dim UNet intermediate features not accessible from
the standalone TTNN UNet pipeline).

Architecture:
    For step t in [T, T-1, ..., 0]:
        For frame i in [0, N-1]:
            noise_pred[i] = TTNN_UNet(latent[i], t, text_emb)   # Blackhole
        noise_preds = cross_frame_attention(stack(noise_pred))    # CPU (tiny)
        For frame i in [0, N-1]:
            latent[i] = scheduler.step(noise_preds[i], t)        # CPU

Why this works: noise predictions at each step carry structural information
(edges, shapes, motion directions). Attending across frames causes the network
to agree on structure before the scheduler commits to a latent direction.

Phase 2.5 vs Phase 1 (MotionAdapter):
    Phase 1 temporal attention runs INSIDE UNet blocks on 320-dim features —
    full AnimateDiff but CPU-only. Phase 2.5 attention runs at the 4-dim
    noise-prediction level — approximate but Blackhole-accelerated.

Lightning mode (use_lightning=True):
    Uses EulerDiscreteScheduler with timestep_spacing="trailing" and
    beta_schedule="linear". CFG=7.5 is retained — the "CFG=1.0 required"
    constraint applies only to real AnimateDiff-Lightning distilled adapter
    weights (ByteDance); our TTNN path uses the base SD 1.4 UNet, which
    benefits fully from guidance amplification. Cross-frame attention
    applied at two points per step: (1) blend noise_preds before
    scheduler.step(), and (2) blend prev_sample latents after step()
    with 0.4× lower alpha. Both use cosine-decay alpha to prioritise
    coarse structure early and preserve per-frame variety late.
"""

import math
import sys
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def cross_frame_attention(tensors: torch.Tensor, alpha: float = 0.35) -> torch.Tensor:
    """Self-attention across frames on stacked latent tensors.

    Reshapes [N, C, H, W] to [H*W, N, C] so each spatial position attends
    across all N frames, then blends attended and original.

    Args:
        tensors: Shape [N, C, H, W] — stacked tensors for all frames.
                 For PNDM: noise predictions before scheduler step.
                 For Lightning/Euler: prev_sample latents after scheduler step.
        alpha: Blend weight (0 = no effect, 1 = full attention).

    Returns:
        Blended tensor of same shape [N, C, H, W]
    """
    N, C, H, W = tensors.shape
    if N == 1:
        return tensors

    x = tensors.permute(2, 3, 0, 1).reshape(H * W, N, C).float()

    # Scaled dot-product self-attention (Q = K = V = x)
    scale = C ** -0.5
    attn = torch.bmm(x, x.transpose(-2, -1)) * scale  # [H*W, N, N]
    attn = torch.softmax(attn, dim=-1)
    attended = torch.bmm(attn, x)                      # [H*W, N, C]

    attended = attended.reshape(H, W, N, C).permute(2, 3, 0, 1)
    blended = (1.0 - alpha) * tensors + alpha * attended
    return blended.to(tensors.dtype)


def _cosine_alpha(step_idx: int, num_steps: int, alpha_max: float, alpha_min_frac: float = 0.15) -> float:
    """Cosine decay schedule for temporal alpha over denoising steps.

    Starts at alpha_max (early steps: coarse structure, where cross-frame
    agreement matters most) and decays to alpha_max * alpha_min_frac
    (late steps: fine detail, where variety should be preserved).

    Used for Lightning/Euler where few large steps each carry huge structural
    weight — a constant alpha at the wrong magnitude destroys the trajectory.
    """
    t = step_idx / max(num_steps - 1, 1)  # 0.0 → 1.0
    cosine = 0.5 * (1.0 + math.cos(math.pi * t))  # 1.0 → 0.0
    return alpha_max * (alpha_min_frac + (1.0 - alpha_min_frac) * cosine)


def generate_frames_temporal(
    device,
    ttnn_model,
    torch_vae,
    config,
    torch_time_proj,
    text_embeddings: torch.Tensor,
    num_frames: int = 8,
    num_steps: int = 25,
    guidance_scale: float = 7.5,
    seed: int = 42,
    height: int = 512,
    width: int = 512,
    temporal_alpha: float = 0.35,
    use_lightning: bool = False,
) -> List:
    """Generate temporally-coherent frames on Blackhole with cross-frame attention.

    Frames are denoised sequentially (one TTNN UNet call per frame per step).
    Cross-frame attention is applied to the stacked noise predictions at each step
    before the scheduler commits to the next latent. Total TTNN UNet calls:
    num_frames × num_steps (same as Phase 2).

    Args:
        device: TTNN Blackhole device from setup_blackhole()
        ttnn_model: Loaded TTNN UNet2D model (from preprocess_model_parameters)
        torch_vae: CPU PyTorch AutoencoderKL for latent → pixel decode
        config: unet.config from PyTorch UNet2DConditionModel
        torch_time_proj: unet.time_proj, used by build_tlist for timestep embeddings
        text_embeddings: Shape (2, 96, 768) — [uncond, cond] concatenated,
                         padded from 77 to 96 tokens
        num_frames: Number of frames to generate
        num_steps: Denoising steps (25 recommended for both Lightning and standard;
                   Lightning with fewer steps is faster but lower quality)
        guidance_scale: CFG scale. Use 7.5 for both standard and Lightning on TTNN.
        seed: RNG seed — shared base noise + per-frame perturbation
        height, width: Output size in pixels (512 × 512 recommended)
        temporal_alpha: Cross-frame attention blend (0 → Phase 2 shared noise,
                        1 → full attention; default 0.35)
        use_lightning: If True, use EulerDiscreteScheduler instead of PNDM.
                       CFG=7.5 still applies (base UNet, no distilled adapter).

    Returns:
        List of PIL Images, length num_frames, with temporal coherence
    """
    import ttnn
    from PIL import Image
    from animatediff_ttnn.ttnn_pipeline import build_tlist, to_device, from_device
    from models.demos.vision.generative.stable_diffusion.wormhole.sd_helper_funcs import tt_guide

    lh, lw = height // 8, width // 8

    if use_lightning:
        # Lightning: EulerDiscreteScheduler with trailing timesteps, linear betas.
        # One scheduler per frame — each maintains independent step-index state.
        from diffusers import EulerDiscreteScheduler
        from animatediff_ttnn.tt_euler_scheduler import TtEulerScheduler

        euler_kwargs = dict(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="linear",
            timestep_spacing="trailing",
        )
        schedulers = [
            EulerDiscreteScheduler(**euler_kwargs) for _ in range(num_frames)
        ]
        for s in schedulers:
            s.set_timesteps(num_steps)

        # TtEulerScheduler only used to build _tlist — same timesteps as CPU schedulers
        _tt_sched = TtEulerScheduler(**euler_kwargs)
        _tt_sched.set_timesteps(num_steps)
    else:
        from diffusers import PNDMScheduler
        from models.demos.vision.generative.stable_diffusion.wormhole.sd_pndm_scheduler import TtPNDMScheduler

        pndm_kwargs = dict(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            skip_prk_steps=True,
            steps_offset=1,
        )
        schedulers = [PNDMScheduler(**pndm_kwargs) for _ in range(num_frames)]
        for s in schedulers:
            s.set_timesteps(num_steps)

        _tt_sched = TtPNDMScheduler(device=device, **pndm_kwargs)
        _tt_sched.set_timesteps(num_steps)

    timesteps = schedulers[0].timesteps
    init_noise_sigma = float(schedulers[0].init_noise_sigma)

    # Shared base noise — same starting point for all frames.
    # Lightning uses tighter per-frame perturbation (0.02 vs 0.05): with only
    # 4 Euler steps, frames have less time to diverge from their start point,
    # so tighter initial correlation translates more directly into coherence.
    generator = torch.Generator().manual_seed(seed)
    base_noise = torch.randn(1, 4, lh, lw, generator=generator)
    noise_perturb = 0.02 if use_lightning else 0.05

    frame_latents = []
    for _ in range(num_frames):
        perturbed = base_noise + noise_perturb * torch.randn(base_noise.shape, generator=generator)
        frame_latents.append(perturbed * init_noise_sigma)

    # Build TTNN time embeddings once — timesteps are identical across all frames
    _tlist = build_tlist(_tt_sched, torch_time_proj, device, lh, lw)

    # Text embeddings to device — same tensor reused for every frame at every step
    ttnn_text_emb = to_device(
        text_embeddings, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    )

    # Phase 2.5 note: the frame loop below is serialized in Python — one frame
    # per TTNN call regardless of how many chips the MeshDevice contains. On a
    # multi-chip system each `to_device` replicates the tensor to all chips, but
    # only chip 0 produces the output used here, so extra chips pay replication
    # cost without contributing throughput. Phase 3 will replace this with a
    # ShardTensorToMesh mapper that dispatches N distinct frames to N chips in a
    # single batched call. For now, use a single-chip MeshDevice (1×1) on QB2.
    num_steps_actual = len(timesteps)

    for step_idx, t in enumerate(timesteps):
        # Collect TTNN noise predictions for all frames at timestep t
        noise_preds = []
        for i in range(num_frames):
            latent_cpu = frame_latents[i]
            if use_lightning:
                # Euler schedulers require scaling the latent by 1/sqrt(sigma^2+1)
                # before each UNet call (PNDM's sigma is always 1.0 so it's a no-op
                # there, but Euler's sigma starts at ~25 and must be normalized).
                latent_cpu = schedulers[i].scale_model_input(latent_cpu, t)
            lat = to_device(
                latent_cpu, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            )
            # TTNN UNet expects batch=2 for CFG (unconditional + conditional)
            lat_input = ttnn.concat([lat, lat], dim=0)
            ttnn_out = ttnn_model(
                lat_input,
                timestep=_tlist[step_idx],
                encoder_hidden_states=ttnn_text_emb,
                class_labels=None,
                attention_mask=None,
                cross_attention_kwargs=None,
                return_dict=True,
                config=config,
            )
            guided = tt_guide(ttnn_out, guidance_scale)
            noise_preds.append(from_device(guided, device).to(torch.float32))

        if use_lightning:
            # Lightning two-point blend with cosine-decay alpha:
            #
            # Point 1: blend noise_preds before stepping. All preds at the same
            # timestep are in the same space — no normalisation needed. High alpha
            # early drives coarse structural agreement; cosine decay to low alpha
            # late preserves per-frame variety in fine detail.
            #
            # Point 2: blend prev_sample latents after stepping. Catches structural
            # drift that escaped the noise_pred blend. Lower alpha (0.4×) so it's
            # a gentle correction, not a second full alignment pass.
            step_alpha = _cosine_alpha(step_idx, num_steps_actual, temporal_alpha)

            # Point 1: noise_pred blend
            stacked_preds = torch.cat(noise_preds, dim=0)
            blended_preds_t = cross_frame_attention(stacked_preds, alpha=step_alpha)
            blended_preds = [blended_preds_t[i : i + 1] for i in range(num_frames)]

            # Step each frame with the blended noise_pred
            next_latents = []
            for i in range(num_frames):
                next_latents.append(
                    schedulers[i].step(blended_preds[i], t, frame_latents[i]).prev_sample
                )

            # Point 2: latent-space blend after stepping (gentler — 0.4× alpha)
            stacked_lat = torch.cat(next_latents, dim=0)
            attended_lat = cross_frame_attention(stacked_lat, alpha=step_alpha * 0.4)
            for i in range(num_frames):
                frame_latents[i] = attended_lat[i : i + 1]
        else:
            # PNDM: attend on noise_preds before stepping. PNDM uses a multi-step
            # error accumulation buffer (ets) that would desync if we modified
            # latents after stepping — so we blend preds then step as before.
            stacked = torch.cat(noise_preds, dim=0)
            attended = cross_frame_attention(stacked, alpha=temporal_alpha)
            for i in range(num_frames):
                frame_latents[i] = schedulers[i].step(
                    attended[i : i + 1], t, frame_latents[i]
                ).prev_sample

        print(f"  Step {step_idx + 1}/{num_steps_actual}", end="\r", flush=True)

    print()

    # Decode all latents with CPU VAE (TTNN VAE conv_out OOMs on Blackhole)
    frames = []
    for i, latent in enumerate(frame_latents):
        latent_scaled = latent / 0.18215
        with torch.no_grad():
            decoded = torch_vae.decode(latent_scaled).sample  # (1, 3, H, W) in [-1, 1]
        img = (decoded / 2 + 0.5).clamp(0, 1)
        img = (img.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
        frames.append(Image.fromarray(img))
        print(f"  Frame {i + 1}/{num_frames} decoded")

    return frames
