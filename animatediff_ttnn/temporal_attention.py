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
import animatediff_ttnn.ttnn_pipeline as _tp  # module-ref so patch.object works in tests

sys.path.insert(0, str(Path(__file__).parent.parent))


def chain_blend_seed(
    base_noise: torch.Tensor,
    chain_from: str,
    alpha: float = 0.35,
) -> torch.Tensor:
    """Blend previous-run denoised latents into the current seed noise.

    This is the core of chain mode: the denoised latents from the previous
    generation are used to bias this run's starting noise toward the same
    coarse spatial layout, so the new prompt inherits subject position and
    rough composition while controlling all content and colour.

    Args:
        base_noise: (1, 4, lh, lw) float32 unit-std noise tensor.
        chain_from: Path to a .pt file saved by chain_save (shape: (F, 4, lh, lw)).
                    If the file doesn't exist, base_noise is returned unchanged.
        alpha: Blend weight (0 = pure noise, 1 = fully replaced by prev layout).
               Effective range: 0.20–0.55. Values >0.6 suppress prompt guidance.

    Returns:
        (1, 4, lh, lw) float32 tensor, renormalised to unit std so the
        scheduler's sigma scaling sees the expected noise distribution at t=T.

    Design notes:
        The previous latents are frame-averaged then gently low-pass filtered
        (ksize=5) to keep coarse layout (silhouette, position) while attenuating
        fine texture that would fight the new prompt.

        Critically: we do NOT per-channel-normalise before blurring.  The old
        code did (prev_mean - ch_mean) / ch_std before avg_pool, which reduced
        the blurred signal std from ~0.28 to ~0.03 — only 0.08–1.85% of the
        final mixture by variance after renorm.  At that level the chain signal
        is perceptually invisible regardless of alpha.  Without per-channel
        norm, alpha=0.35 produces ~15% correlation with prev layout, which is
        detectable in a 25-step denoising run.
    """
    from pathlib import Path as _Path

    chain_path = _Path(chain_from)
    if not chain_path.exists():
        print(f"  Chain: warning — {chain_from} not found, ignoring")
        return base_noise

    if alpha == 0.0:
        return base_noise

    prev = torch.load(chain_path, map_location="cpu", weights_only=True)  # (F, 4, lh, lw)
    # Average across frames: reduces per-frame noise while preserving layout signal.
    # At 64×64 latent resolution the frame-mean is already "coarse" — no additional
    # spatial blur needed.  Blurring here (ksize=5+) reduces signal std ~5-9×, leaving
    # only 1-4% of variance after the final renorm: perceptually invisible.
    prev_mean = prev.mean(dim=0, keepdim=True).float()                     # (1, 4, lh, lw)

    # Blend, then renorm to unit std so the scheduler sigma scaling is correct.
    mixed = (1.0 - alpha) * base_noise + alpha * prev_mean
    mixed_std = mixed.std().clamp(min=1e-6)
    result = mixed / mixed_std
    print(f"  Chain: blended {chain_path.name} at alpha={alpha} (frame-mean, renorm)")
    return result


def cross_frame_attention(tensors: torch.Tensor, alpha: float = 0.35) -> torch.Tensor:
    """Self-attention across frames on stacked latent tensors.

    Reshapes [N, C, H, W] to [H*W, N, C] so each spatial position attends
    across all N frames, then blends attended and original.

    Args:
        tensors: Shape [N, C, H, W] — stacked tensors for all frames.
                 Called at two points per Lightning step:
                   Point 1 — stacked noise_preds before scheduler.step()
                   Point 2 — stacked prev_sample latents after scheduler.step()
                 Also called once per PNDM step on noise_preds.
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


def _latent_preview(frame_latents: list, height: int, width: int):
    """Fast CPU-side preview of current denoising latents — no VAE, no hardware.

    Maps 3 of the 4 latent channels to RGB via tanh soft-clipping and bilinear
    upsample to full resolution. Shows structure emerging from noise in real time
    without adding any latency to the hardware pipeline.

    Args:
        frame_latents: List of (1, 4, lh, lw) CPU tensors — current denoised latents.
        height, width: Target image size in pixels.

    Returns:
        List of PIL Images, one per frame — approximate colourised preview.
    """
    from PIL import Image as _Image
    import numpy as _np

    frames = []
    for lat in frame_latents:
        rgb = lat[0, :3].float()                   # (3, lh, lw) — first 3 channels as RGB proxy
        rgb = torch.tanh(rgb * 0.5) * 0.5 + 0.5   # soft-clip wide dynamic range to [0, 1]
        rgb_up = torch.nn.functional.interpolate(
            rgb.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
        )[0]                                        # (3, H, W)
        arr = (rgb_up.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(_np.uint8)
        frames.append(_Image.fromarray(arr))
    return frames


def _tt_guide_cpu(tensor: torch.Tensor, guidance_scale: float) -> torch.Tensor:
    """CFG guidance on a CPU [2, C, H, W] tensor (uncond=[:1], cond=[1:])."""
    uncond, cond = tensor[:1], tensor[1:]
    return uncond + guidance_scale * (cond - uncond)


def generate_frames_temporal(
    device,
    ttnn_model,
    ttnn_vae,
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
    chain_from: str | None = None,
    chain_save: str | None = None,
    chain_alpha: float = 0.6,
    on_step=None,
    text_embeddings_per_frame: List | None = None,
) -> List:
    """Generate temporally-coherent frames on Blackhole with cross-frame attention.

    Frames are denoised sequentially (one TTNN UNet call per frame per step).
    Cross-frame attention is applied to the stacked noise predictions at each step
    before the scheduler commits to the next latent. Total TTNN UNet calls:
    num_frames × num_steps (same as Phase 2).

    Args:
        device: TTNN Blackhole device from setup_blackhole()
        ttnn_model: Loaded TTNN UNet2D model (from preprocess_model_parameters)
        ttnn_vae: TTNN Vae decoder (from load_sd14_ttnn). Runs on Blackhole.
                  All UNet L1 tensors are explicitly deallocated before decoding
                  so the VAE has sufficient L1 headroom.
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
        chain_from: Path to a .pt file saved by a previous run's chain_save.
                    The stored latents are blended into this run's base_noise at
                    chain_alpha weight — visual continuity across prompts without
                    explicit conditioning.
        chain_save: Path to save this run's final denoised latents as a .pt file
                    so the next run can use them via chain_from.
        chain_alpha: Blend weight for chain_from latents (0 = ignore, 1 = replace).
                     Default 0.6 — dominant influence but not overriding fresh noise.
        on_step: Optional callable(step_idx, num_steps, frame_latents) called after
                 each complete denoising step. frame_latents is the current list of
                 (1, 4, lh, lw) CPU tensors — pass them to _latent_preview() for a
                 fast in-progress visual. Called in the TTNN thread; must be thread-safe.
        text_embeddings_per_frame: Optional "prompt travel" conditioning — a list of
                 num_frames tensors, each (2, 96, 768) = [uncond, cond_i], produced by
                 encode_prompt(..., prompt_schedule=...). When provided, frame i is
                 conditioned on text_embeddings_per_frame[i] instead of the shared
                 text_embeddings tensor, so content morphs across frames. When None
                 (default), behaviour is byte-identical to before: the single
                 text_embeddings tensor is broadcast to every frame.

    Returns:
        List of PIL Images, length num_frames, with temporal coherence
    """
    import ttnn as _ttnn
    from animatediff_ttnn.ttnn_pipeline import plan_frame_sharding
    # Plan how frames map onto chips. On a mesh we shard one CFG-doubled frame per
    # chip per pass, running ceil(num_frames / num_chips) passes for num_frames >
    # num_chips (each chip is compiled for batch=2, so it must receive exactly one
    # frame's uncond+cond rows). On a single chip we fall back to the serial
    # per-frame path. plan_frame_sharding raises ValueError (with valid counts) when
    # num_frames is not a multiple of num_chips — a partial chunk would produce a
    # mis-sized shard the batch=2 kernel rejects.
    _num_chips = device.get_num_devices() if isinstance(device, _ttnn.MeshDevice) else 1
    _use_sharding, _chunk = plan_frame_sharding(num_frames, _num_chips)

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
    # Lightning uses tighter per-frame perturbation (0.02 vs 0.05): each Euler
    # step covers a larger sigma interval than PNDM, so the initial correlation
    # has proportionally more leverage on the final structure.
    generator = torch.Generator().manual_seed(seed)
    base_noise = torch.randn(1, 4, lh, lw, generator=generator)
    noise_perturb = 0.02 if use_lightning else 0.05

    # Chain continuity: blend a low-pass version of the previous run's latents into
    # this run's seed noise, biasing the denoiser toward the same coarse composition
    # (subject position, rough silhouette) while leaving colour, texture, and scene
    # content entirely to the new prompt.
    #
    # Key invariant: the blended base_noise must remain unit-std so the scheduler's
    # sigma scaling is correct.  We enforce this with a final re-normalisation.
    #
    # Low-pass kernel choice: 9px at 64×64 latent ≈ keeping structure at >~14px in
    # pixel space — enough for silhouette, not enough for any recognisable detail.
    # 19px (prev value) was too aggressive: the blurred map retained enough energy
    # to override the prompt's colour guidance entirely.
    if chain_from is not None:
        base_noise = chain_blend_seed(base_noise, chain_from, alpha=chain_alpha)

    frame_latents = []
    for _ in range(num_frames):
        perturbed = base_noise + noise_perturb * torch.randn(base_noise.shape, generator=generator)
        frame_latents.append(perturbed * init_noise_sigma)

    # Build TTNN time embeddings once — timesteps are identical across all frames
    _tlist = build_tlist(_tt_sched, torch_time_proj, device, lh, lw)

    # Text embeddings to device.
    #
    # Default (single-prompt) path — UNCHANGED, byte-identical to before: one
    # (2, 96, 768) tensor replicated to every chip and reused for every frame at
    # every step.
    #
    # Prompt-travel path (text_embeddings_per_frame given): each frame i is
    # conditioned on its own (2, 96, 768) embedding. How that reaches the device
    # depends on the frame-distribution strategy:
    #   * serial path (single chip): pre-upload one replicated device tensor per
    #     frame; frame i uses ttnn_text_embs[i]. Low risk — same shape/layout as
    #     the shared tensor, just a different tensor object per UNet call.
    #   * sharded mesh path: each chunk of _chunk frames runs in ONE UNet call
    #     with the frames sharded across chips, so the encoder_hidden_states must
    #     be sharded the SAME way — chunk's [2*_chunk, 96, 768] via
    #     ShardTensorToMesh(dim=0) so chip K sees its frame's [2, 96, 768]. This
    #     mirrors the latent sharding exactly (shard_frames_to_device) but is
    #     ⚠️ UNVALIDATED ON HARDWARE — needs QB2 to confirm the SD-demo UNet's
    #     cross-attention consumes a sharded encoder_hidden_states correctly.
    _per_frame_text = text_embeddings_per_frame is not None
    ttnn_text_emb = None
    ttnn_text_embs_serial = None
    sharded_text_by_chunk = None
    if not _per_frame_text:
        ttnn_text_emb = to_device(
            text_embeddings, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        )
    elif _use_sharding:
        # Pre-build one sharded encoder_hidden_states per chunk (text is constant
        # across denoising steps, so build once).  ⚠️ needs QB2 validation.
        sharded_text_by_chunk = {}
        for c0 in range(0, num_frames, _chunk):
            chunk_list = text_embeddings_per_frame[c0 : c0 + _chunk]
            sharded_text_by_chunk[c0] = _tp.shard_frames_to_device(
                chunk_list, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            )
    else:
        # Serial single-chip path: one replicated device tensor per frame.
        ttnn_text_embs_serial = [
            to_device(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            for t in text_embeddings_per_frame
        ]

    # Denoising loop: on a mesh, frames are CFG-doubled and sharded across chips via
    # ShardTensorToMesh in chunks of _chunk (== num_chips) frames — one frame per
    # chip per pass, batch=2 each. num_frames > num_chips runs multiple sharded
    # passes. On a single chip, fall back to the serial per-frame path — the TTNN
    # UNet is compiled for batch=2 and would corrupt L1 if fed a larger batch.
    num_steps_actual = len(timesteps)

    for step_idx, t in enumerate(timesteps):
        if _use_sharding:
            # Shard in chunks of _chunk (== num_chips) frames: one CFG-doubled frame
            # per chip per pass, ceil(num_frames / _chunk) passes total. Frames are
            # processed in order so noise_preds stays frame-aligned.
            noise_preds = []
            for c0 in range(0, num_frames, _chunk):
                # CFG-double each frame in this chunk CPU-side, then shard across chips.
                # Lightning: scale_model_input first (per-frame, CPU — Euler sigma norm).
                cfg_latents = []
                for i in range(c0, c0 + _chunk):
                    latent_cpu = frame_latents[i]
                    if use_lightning:
                        # Euler schedulers require scaling the latent by 1/sqrt(sigma^2+1)
                        # before each UNet call (PNDM's sigma is always 1.0 so it's a no-op
                        # there, but Euler's sigma starts at ~25 and must be normalized).
                        latent_cpu = schedulers[i].scale_model_input(latent_cpu, t)
                    cfg_latents.append(torch.cat([latent_cpu, latent_cpu], dim=0))  # [2, 4, lh, lw]

                # Shard this chunk's frames to device in a single call.
                # shard_frames_to_device stacks the list to [2*_chunk, 4, lh, lw] and maps
                # each pair to a distinct chip via ShardTensorToMesh — replacing _chunk
                # serial to_device calls and _chunk serial ttnn.concat operations.
                stacked_dev = _tp.shard_frames_to_device(
                    cfg_latents, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                )
                # Prompt travel: use this chunk's sharded per-frame conditioning
                # (⚠️ needs QB2 validation); else the shared replicated tensor.
                _enc = sharded_text_by_chunk[c0] if _per_frame_text else ttnn_text_emb
                ttnn_out = ttnn_model(
                    stacked_dev,
                    timestep=_tlist[step_idx],
                    encoder_hidden_states=_enc,
                    class_labels=None,
                    attention_mask=None,
                    cross_attention_kwargs=None,
                    return_dict=True,
                    config=config,
                )
                stacked_dev.deallocate(True)

                # Gather this chunk's frame outputs, apply CFG guidance per frame.
                # gather_frames_from_device pulls [2*_chunk, 4, lh, lw] back to CPU and
                # splits it into _chunk tensors of shape [2, 4, lh, lw] — one per frame.
                frame_outputs = _tp.gather_frames_from_device(ttnn_out, device, _chunk)
                ttnn_out.deallocate(True)
                for frame_out in frame_outputs:
                    # _tt_guide_cpu applies CFG on a CPU [2, C, H, W] tensor; result is
                    # [1, 4, lh, lw] — the guided noise prediction for this frame.
                    noise_preds.append(_tt_guide_cpu(frame_out, guidance_scale).to(torch.float32))
        else:
            # Serial path: one UNet call per frame, batch=2 CFG-doubled.
            # Used on a single chip (num_chips == 1); meshes take the sharded path above.
            noise_preds = []
            for i in range(num_frames):
                latent_cpu = frame_latents[i]
                if use_lightning:
                    latent_cpu = schedulers[i].scale_model_input(latent_cpu, t)
                cfg_lat = torch.cat([latent_cpu, latent_cpu], dim=0)  # [2, 4, lh, lw]
                lat_dev = to_device(cfg_lat, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                # Prompt travel: frame i uses its own conditioning; else shared.
                _enc = ttnn_text_embs_serial[i] if _per_frame_text else ttnn_text_emb
                ttnn_out_i = ttnn_model(
                    lat_dev,
                    timestep=_tlist[step_idx],
                    encoder_hidden_states=_enc,
                    class_labels=None,
                    attention_mask=None,
                    cross_attention_kwargs=None,
                    return_dict=True,
                    config=config,
                )
                lat_dev.deallocate(True)
                frame_out = from_device(ttnn_out_i, device, batch=2)
                ttnn_out_i.deallocate(True)
                noise_preds.append(_tt_guide_cpu(frame_out, guidance_scale).to(torch.float32))

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
        if on_step is not None:
            on_step(step_idx, num_steps_actual, frame_latents)

    print()

    # Deallocate all live UNet device tensors before VAE decode.
    # The TTNN VAE needs substantial L1 for its conv layers; if the UNet's
    # output tensors are still occupying L1 the VAE conv_out will OOM.
    # Free shared tensors that outlive the loop before VAE decode needs L1.
    if ttnn_text_emb is not None:
        ttnn_text_emb.deallocate(True)
    if ttnn_text_embs_serial is not None:
        for t_tensor in ttnn_text_embs_serial:
            t_tensor.deallocate(True)
    if sharded_text_by_chunk is not None:
        for t_tensor in sharded_text_by_chunk.values():
            t_tensor.deallocate(True)
    for t_tensor in _tlist:
        t_tensor.deallocate(True)

    # Chain save: persist final denoised latents for the next chained run.
    if chain_save is not None:
        from pathlib import Path as _Path
        save_path = _Path(chain_save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.cat(frame_latents, dim=0), save_path)
        print(f"  Chain: saved latents → {save_path}")

    # VAE decode — serial per frame. The TTNN VAE only supports batch=1 input
    # ([1, lh, lw, 4] NHWC). Output is [1, 1, H*W, 3] flat; reshape before PIL.
    frames = []
    for i, latent in enumerate(frame_latents):
        latent_scaled = latent / 0.18215
        ttnn_lat = to_device(
            latent_scaled.permute(0, 2, 3, 1),  # [1, lh, lw, 4] NHWC
            device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        )
        ttnn_decoded = ttnn_vae.decode(ttnn_lat)
        ttnn_lat.deallocate(True)
        ttnn_decoded = ttnn.reshape(ttnn_decoded, [1, height, width, 3])
        ttnn_decoded_perm = ttnn.permute(ttnn_decoded, [0, 3, 1, 2])
        ttnn_decoded.deallocate(True)
        decoded = from_device(ttnn_decoded_perm, device, batch=1).float()
        ttnn_decoded_perm.deallocate(True)
        img = (decoded / 2 + 0.5).clamp(0, 1)
        img = (img.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
        frames.append(Image.fromarray(img))
        print(f"  Frame {i + 1}/{num_frames} decoded")

    return frames


def generate_frames_motion(
    device,
    ttnn_model,
    ttnn_vae,
    config,
    torch_time_proj,
    text_embeddings: torch.Tensor,
    temporal_kernels: dict,
    num_frames: int = 8,
    num_steps: int = 25,
    guidance_scale: float = 7.5,
    seed: int = 42,
    height: int = 512,
    width: int = 512,
    use_lightning: bool = False,
    chain_from: str | None = None,
    chain_save: str | None = None,
    chain_alpha: float = 0.6,
    temporal_alpha: float = 0.35,
    on_step=None,
    injection_alpha: float = 1.0,
    skip_keys: set | None = None,
    text_embeddings_per_frame: List | None = None,
) -> List:
    """Generate temporally-coherent frames using MotionAdapter temporal attention.

    Phase 3: MotionAdapter modules are injected into the TTNN UNet via
    forward_unet_staged(), which calls each UNet block object across all N frames
    and applies full AnimateDiffTransformer3D.forward() between blocks at 7
    injection points (down0-2, mid, up0-2). This produces genuine AnimateDiff-quality
    temporal coherence on Blackhole hardware.

    The denoising loop differs from generate_frames_temporal: instead of N
    sequential ttnn_model() calls per step followed by cross_frame_attention(),
    forward_unet_staged() processes all N frames in a single staged pass with
    temporal attention applied between blocks at feature-level.

    Args:
        device: TTNN Blackhole device from setup_blackhole()
        ttnn_model: Loaded TTNN UNet2D model
        ttnn_vae: TTNN Vae decoder
        config: unet.config from PyTorch UNet2DConditionModel
        torch_time_proj: unet.time_proj, used by build_tlist
        text_embeddings: Shape (2, 96, 768) — [uncond, cond] concatenated
        temporal_kernels: dict from motion_weights.load_motion_modules()
        num_frames: Number of frames to generate
        num_steps: Denoising steps (25 for PNDM, 8 for Lightning)
        guidance_scale: CFG scale (7.5 recommended)
        seed: RNG seed — shared base noise + per-frame perturbation
        height, width: Output size in pixels (512 × 512)
        use_lightning: If True, use EulerDiscreteScheduler
        chain_from: Path to .pt latents from a previous chain_save run
        chain_save: Path to save this run's final latents for chaining
        chain_alpha: Blend weight for chain_from (default 0.6)
        on_step: Optional callable(step_idx, num_steps, frame_latents)

    Returns:
        List of PIL Images, length num_frames
    """
    import ttnn as _ttnn_guard
    _num_chips_motion = device.get_num_devices() if isinstance(device, _ttnn_guard.MeshDevice) else 1
    if num_frames % _num_chips_motion != 0:
        raise ValueError(
            f"num_frames ({num_frames}) must be divisible by num_chips ({_num_chips_motion}). "
            f"Valid counts for {_num_chips_motion} chips: {[_num_chips_motion * k for k in range(1, 9)]}"
        )

    import ttnn
    from PIL import Image
    from animatediff_ttnn.ttnn_pipeline import build_tlist, to_device, from_device
    from animatediff_ttnn.ttnn_motion_pipeline import forward_unet_staged
    from models.demos.vision.generative.stable_diffusion.wormhole.sd_helper_funcs import tt_guide

    lh, lw = height // 8, width // 8

    # Scheduler setup — identical to generate_frames_temporal
    if use_lightning:
        from diffusers import EulerDiscreteScheduler
        from animatediff_ttnn.tt_euler_scheduler import TtEulerScheduler
        euler_kwargs = dict(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="linear",
            timestep_spacing="trailing",
        )
        schedulers = [EulerDiscreteScheduler(**euler_kwargs) for _ in range(num_frames)]
        for s in schedulers:
            s.set_timesteps(num_steps)
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

    generator = torch.Generator().manual_seed(seed)
    base_noise = torch.randn(1, 4, lh, lw, generator=generator)
    noise_perturb = 0.02 if use_lightning else 0.05

    if chain_from is not None:
        base_noise = chain_blend_seed(base_noise, chain_from, alpha=chain_alpha)

    frame_latents = []
    for _ in range(num_frames):
        perturbed = base_noise + noise_perturb * torch.randn(base_noise.shape, generator=generator)
        frame_latents.append(perturbed * init_noise_sigma)

    _tlist = build_tlist(_tt_sched, torch_time_proj, device, lh, lw)

    # Text conditioning. Default (single-prompt) path is UNCHANGED: one shared
    # (2, 96, 768) tensor for every frame. Prompt-travel path: one replicated
    # device tensor per frame, threaded into forward_unet_staged which conditions
    # frame i on text_embeddings_per_frame[i]. The motion path calls the UNet
    # blocks once per frame, so per-frame conditioning is a clean tensor swap
    # (same shape/layout) — no mesh-shard reshaping needed here.
    _per_frame_text = text_embeddings_per_frame is not None
    if _per_frame_text:
        ttnn_text_emb = [
            to_device(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            for t in text_embeddings_per_frame
        ]
    else:
        ttnn_text_emb = to_device(
            text_embeddings, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        )

    num_steps_actual = len(timesteps)

    for step_idx, t in enumerate(timesteps):
        # Build per-frame CFG-doubled device tensors
        device_samples = []
        for i in range(num_frames):
            latent_cpu = frame_latents[i]
            if use_lightning:
                latent_cpu = schedulers[i].scale_model_input(latent_cpu, t)
            lat = to_device(latent_cpu, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            lat_input = ttnn.concat([lat, lat], dim=0)
            lat.deallocate(True)
            device_samples.append(lat_input)

        # Staged UNet forward — all N frames, 7 temporal injection points
        raw_outputs = forward_unet_staged(
            ttnn_model=ttnn_model,
            frame_samples=device_samples,
            timestep=_tlist[step_idx],
            encoder_hidden_states=ttnn_text_emb,
            config=config,
            temporal_kernels=temporal_kernels,
            device=device,
            num_frames=num_frames,
            injection_alpha=injection_alpha,
            skip_keys=skip_keys or set(),
        )

        # Apply CFG guidance and collect noise predictions
        noise_preds = []
        for i, raw in enumerate(raw_outputs):
            guided = tt_guide(raw, guidance_scale)
            noise_pred_cpu = from_device(guided, device).to(torch.float32)
            raw.deallocate(True)
            guided.deallocate(True)
            noise_preds.append(noise_pred_cpu)

        # Scheduler step — same logic as generate_frames_temporal
        if use_lightning:
            step_alpha = _cosine_alpha(step_idx, num_steps_actual, temporal_alpha)
            stacked_preds = torch.cat(noise_preds, dim=0)
            blended_preds_t = cross_frame_attention(stacked_preds, alpha=step_alpha)
            blended_preds = [blended_preds_t[i: i + 1] for i in range(num_frames)]
            next_latents = [
                schedulers[i].step(blended_preds[i], t, frame_latents[i]).prev_sample
                for i in range(num_frames)
            ]
            stacked_lat = torch.cat(next_latents, dim=0)
            attended_lat = cross_frame_attention(stacked_lat, alpha=step_alpha * 0.4)
            for i in range(num_frames):
                frame_latents[i] = attended_lat[i: i + 1]
        else:
            for i in range(num_frames):
                frame_latents[i] = schedulers[i].step(
                    noise_preds[i], t, frame_latents[i]
                ).prev_sample

        print(f"  [motion] Step {step_idx + 1}/{num_steps_actual}", end="\r", flush=True)
        if on_step is not None:
            on_step(step_idx, num_steps_actual, frame_latents)

    print()

    # Cleanup shared device tensors (single tensor, or per-frame list for prompt travel)
    if _per_frame_text:
        for t_tensor in ttnn_text_emb:
            t_tensor.deallocate(True)
    else:
        ttnn_text_emb.deallocate(True)
    for t_tensor in _tlist:
        t_tensor.deallocate(True)

    # Chain save
    if chain_save is not None:
        from pathlib import Path as _Path
        save_path = _Path(chain_save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.cat(frame_latents, dim=0), save_path)
        print(f"  Chain: saved latents → {save_path}")

    # VAE decode — serial per frame. The TTNN VAE only supports batch=1 input
    # ([1, lh, lw, 4] NHWC). Output is [1, 1, H*W, 3] flat; reshape before PIL.
    frames = []
    for i, latent in enumerate(frame_latents):
        latent_scaled = latent / 0.18215
        ttnn_lat = to_device(
            latent_scaled.permute(0, 2, 3, 1),  # [1, lh, lw, 4] NHWC
            device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        )
        ttnn_decoded = ttnn_vae.decode(ttnn_lat)
        ttnn_lat.deallocate(True)
        ttnn_decoded = ttnn.reshape(ttnn_decoded, [1, height, width, 3])
        ttnn_decoded_perm = ttnn.permute(ttnn_decoded, [0, 3, 1, 2])
        ttnn_decoded.deallocate(True)
        decoded = from_device(ttnn_decoded_perm, device, batch=1).float()
        ttnn_decoded_perm.deallocate(True)
        img = (decoded / 2 + 0.5).clamp(0, 1)
        img = (img.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
        frames.append(Image.fromarray(img))
        print(f"  Frame {i + 1}/{num_frames} decoded")

    return frames
