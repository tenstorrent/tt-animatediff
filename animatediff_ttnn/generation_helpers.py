# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Shared generation helpers used by both generate.py and app.py.

These functions have no module-level side effects and are safe to import
without tt-metal present (imports inside functions are lazy).
"""

import torch


def load_sd14_ttnn(device):
    """Load SD 1.4 TTNN UNet and TTNN VAE onto device.

    Returns (ttnn_model, ttnn_vae, config, time_proj).

    The TTNN VAE runs on Blackhole — the OOM that previously forced CPU fallback
    was caused by L1 exhaustion from live UNet tensors, not a BH incompatibility.
    The fix: deallocate UNet L1 tensors before invoking the VAE (see
    generate_frames_temporal). The Vae object only allocates L1 during decode(),
    so it is safe to initialise here alongside the UNet.
    """
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from ttnn.model_preprocessing import preprocess_model_parameters
    from models.demos.vision.generative.stable_diffusion.wormhole.custom_preprocessing import custom_preprocessor
    from models.demos.vision.generative.stable_diffusion.wormhole.tt.ttnn_functional_unet_2d_condition_model_new_conv import (
        UNet2DConditionModel as UNet2D,
    )
    from models.demos.vision.generative.stable_diffusion.wormhole.tt.vae.ttnn_vae import Vae

    print("  Loading PyTorch VAE weights...")
    torch_vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    torch_vae.eval()

    print("  Initialising TTNN VAE decoder (weights prepared, kernels compiled on first decode)...")
    ttnn_vae = Vae(torch_vae=torch_vae, device=device)

    print("  Loading PyTorch UNet (config + time_proj)...")
    torch_unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

    print("  Building TTNN UNet (~2-3 min first run, cached after)...")
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_unet,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    ttnn_model = UNet2D(device, parameters, 2, 64, 64)
    return ttnn_model, ttnn_vae, torch_unet.config, torch_unet.time_proj


_clip_tokenizer = None
_clip_text_encoder = None


def _encode_one(text: str) -> torch.Tensor:
    """CLIP-encode a single string to a (1, 96, 768) embedding.

    Pads 77 → 96 tokens to match the TTNN UNet's expected sequence length.
    Tokenizer and text encoder are cached in module globals after first use —
    loading from HuggingFace Hub takes ~2s and is wasteful to repeat.

    Shared by both the single-prompt path (``encode_prompt``) and the
    prompt-schedule path so keyframe prompts are encoded identically to the
    single prompt (guaranteeing endpoints match the non-scheduled result).
    """
    global _clip_tokenizer, _clip_text_encoder
    from transformers import CLIPTokenizer, CLIPTextModel

    if _clip_tokenizer is None:
        _clip_tokenizer = CLIPTokenizer.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="tokenizer"
        )
    if _clip_text_encoder is None:
        _clip_text_encoder = CLIPTextModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="text_encoder"
        )
        _clip_text_encoder.eval()

    tokens = _clip_tokenizer(
        text, padding="max_length", max_length=_clip_tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    with torch.no_grad():
        embeds = _clip_text_encoder(tokens.input_ids)[0]
    return torch.nn.functional.pad(embeds, (0, 0, 0, 19))  # 77 → 96 tokens


def encode_prompt(
    prompt: str,
    negative_prompt: str = "",
    prompt_schedule=None,
    num_frames: int | None = None,
):
    """Encode text conditioning for the UNet — single prompt or a prompt schedule.

    Two return modes, selected by ``prompt_schedule``:

    **Single prompt** (``prompt_schedule is None``, the default — UNCHANGED
    behaviour, byte-identical to the original):
        Returns one ``(2, 96, 768)`` tensor: ``[uncond, cond]`` concatenated on
        dim 0, where ``uncond`` is the negative prompt's embedding and ``cond``
        is ``prompt``'s. This is exactly what the frame loops broadcast to every
        frame today.

    **Prompt schedule** (``prompt_schedule`` given — "prompt travel"):
        ``prompt_schedule`` is a list of ``(frame_index, prompt_text)`` keyframes
        (as produced by ``prompt_schedule.parse_schedule``). Each keyframe prompt
        is CLIP-encoded, its ``cond`` half is interpolated across ``num_frames``
        via ``interpolate_embeddings``, and the returned value is a **list of
        ``num_frames`` tensors, each ``(2, 96, 768)`` = ``[uncond, cond_i]``**.

        The ``uncond`` (negative-prompt) embedding is constant across frames —
        only ``cond`` travels. The per-frame ``[uncond, cond_i]`` layout is
        identical in shape to the single-prompt tensor, so a frame loop can feed
        ``text_embeddings[i]`` to frame ``i`` exactly where it used to feed the
        one shared tensor. Frames landing on a keyframe index reproduce that
        keyframe's ``cond`` exactly (endpoints exact).

    Args:
        prompt: Positive prompt (used only in single-prompt mode).
        negative_prompt: Negative/uncond prompt (used in both modes).
        prompt_schedule: Optional list of ``(frame_index, prompt_text)`` keyframes.
            When None, single-prompt mode.
        num_frames: Required when ``prompt_schedule`` is given — length N of the
            returned per-frame list.

    Returns:
        ``torch.Tensor`` of shape ``(2, 96, 768)`` in single-prompt mode, or a
        ``list[torch.Tensor]`` of length ``num_frames`` (each ``(2, 96, 768)``)
        in schedule mode.

    Raises:
        ValueError: if ``prompt_schedule`` is given without a valid ``num_frames``,
            or the schedule is empty.
    """
    if prompt_schedule is None:
        # ── single-prompt path — unchanged, byte-identical to the original ──
        return torch.cat([_encode_one(negative_prompt), _encode_one(prompt)], dim=0)  # (2, 96, 768)

    # ── prompt-schedule path ("prompt travel") ─────────────────────────────
    from animatediff_ttnn.prompt_schedule import interpolate_embeddings

    if not prompt_schedule:
        raise ValueError("prompt_schedule is empty — provide at least one keyframe.")
    if num_frames is None or num_frames < 1:
        raise ValueError(
            f"num_frames must be a positive int when prompt_schedule is given, got {num_frames!r}."
        )

    # Constant uncond embedding — only cond travels across frames.
    uncond = _encode_one(negative_prompt)  # (1, 96, 768)

    # Encode each keyframe prompt to its cond embedding (1, 96, 768) and build
    # (frame_index, cond) keyframes for interpolation.
    keyframes = [(frame_index, _encode_one(text)) for frame_index, text in prompt_schedule]
    per_frame_cond = interpolate_embeddings(keyframes, num_frames)  # list of (1, 96, 768)

    # Assemble per-frame [uncond, cond_i] → each (2, 96, 768), matching the
    # single-prompt tensor's shape so the frame loops can index frame-by-frame.
    return [torch.cat([uncond, cond_i], dim=0) for cond_i in per_frame_cond]
