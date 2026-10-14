# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Shared generation helpers used by both generate.py and app.py.

These functions have no module-level side effects and are safe to import
without tt-metal present (imports inside functions are lazy).
"""

from typing import NamedTuple

import torch

# Every SD 1.4 load below resolves ONE pinned revision (or $TT_MODEL_WEIGHTS_REVISION),
# so the UNet, VAE, tokenizer and text encoder cannot come from different upstream
# commits. See animatediff_ttnn/weights_pins.py for the rules.
from animatediff_ttnn.weights_pins import SD14_REPO, revision_for


def load_sd14_ttnn(device, loras=()):
    """Load SD 1.4 TTNN UNet and TTNN VAE onto device.

    ``loras`` is an iterable of ``animatediff_ttnn.lora.LoraSpec``. They are merged into
    the CPU UNet before TTNN preprocessing and into the cached CLIP text encoder, so the
    device runs ordinary SD weights and inference cost is unchanged. See lora.py.

    Returns (ttnn_model, ttnn_vae, config, time_proj).

    The TTNN VAE runs on Blackhole — the OOM that previously forced CPU fallback
    was caused by L1 exhaustion from live UNet tensors, not a BH incompatibility.
    The fix: deallocate UNet L1 tensors before invoking the VAE (see
    generate_frames_temporal). The Vae object only allocates L1 during decode(),
    so it is safe to initialise here alongside the UNet.
    """
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from models.demos.vision.generative.stable_diffusion.wormhole.tt.vae.ttnn_vae import Vae

    print("  Loading PyTorch VAE weights...")
    torch_vae = AutoencoderKL.from_pretrained(
        SD14_REPO, subfolder="vae", revision=revision_for(SD14_REPO)
    )
    torch_vae.eval()

    print("  Initialising TTNN VAE decoder (weights prepared, kernels compiled on first decode)...")
    ttnn_vae = Vae(torch_vae=torch_vae, device=device)

    print("  Loading PyTorch UNet (config + time_proj)...")
    torch_unet = UNet2DConditionModel.from_pretrained(
        SD14_REPO, subfolder="unet", revision=revision_for(SD14_REPO)
    )

    loras = list(loras)
    # The CLIP encoder is a process-wide cache. Reload it when a LoRA is about to be
    # merged (so the delta is not applied twice) or when it still carries one (so it
    # cannot leak into a plain load). A plain load after a plain load keeps the warm cache.
    global _clip_text_encoder, _clip_has_lora
    if loras or _clip_has_lora:
        _clip_text_encoder = None
    _clip_has_lora = bool(loras)
    if loras:
        from animatediff_ttnn.lora import merge_loras

        for spec, n_unet, n_te in merge_loras(torch_unet, _get_clip()[1], loras):
            print(f"  LoRA {spec.source} @ {spec.scale}: {n_unet} UNet + {n_te} text-encoder tensors merged")

    return (_build_ttnn_unet(device, torch_unet), ttnn_vae, torch_unet.config, torch_unet.time_proj)


def _build_ttnn_unet(device, torch_unet):
    """Upload a CPU UNet's weights and return the TTNN UNet built from them.

    The compiled kernels depend on shapes, not weight values, so a second UNet built on the
    same device (a LoRA variant) reuses the first one's program cache: it costs the weight
    upload and preprocessing, not a recompile.
    """
    from ttnn.model_preprocessing import preprocess_model_parameters
    from models.demos.vision.generative.stable_diffusion.wormhole.custom_preprocessing import custom_preprocessor
    from models.demos.vision.generative.stable_diffusion.wormhole.tt.ttnn_functional_unet_2d_condition_model_new_conv import (
        UNet2DConditionModel as UNet2D,
    )

    print("  Building TTNN UNet (~2-3 min first run, cached after)...")
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_unet,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    return UNet2D(device, parameters, 2, 64, 64)


class Sd14Variant(NamedTuple):
    """A resident model set for one LoRA recipe, built by ``load_sd14_variant``.

    ``models`` has the same shape ``load_sd14_ttnn`` returns. ``clip`` is the
    ``(tokenizer, text_encoder)`` pair carrying that recipe's text-encoder merge, to pass to
    ``encode_prompt(clip=...)``. Prompts must be encoded with the variant's own encoder: the
    UNet and the text encoder were adapted together.
    """

    models: tuple
    clip: tuple
    #: True when the recipe includes the LCM-LoRA. Such a variant must be sampled with the
    #: LCM scheduler at a few steps and low guidance, or it produces noise.
    lcm: bool = False


def load_sd14_variant(device, loras, vae):
    """Build a UNet with ``loras`` merged, next to an already-loaded model set.

    Unlike ``load_sd14_ttnn(loras=...)`` this touches no process-wide state: the merged text
    encoder is private to the returned variant, and the plain model set's cached encoder is
    left alone. That is what lets a server keep a plain model and several LoRA variants
    resident at once and pick one per request at no cost.

    ``vae`` is the TTNN VAE from the plain model set. It holds no LoRA-dependent weights
    (``lora_unet_*`` and ``lora_te_*`` never touch it), so variants share it and do not pay for
    a second copy on the device.
    """
    from diffusers import UNet2DConditionModel
    from transformers import CLIPTextModel

    from animatediff_ttnn.lora import merge_loras

    loras = list(loras)
    if not loras:
        raise ValueError("load_sd14_variant needs at least one LoRA; the plain set comes from load_sd14_ttnn")
    torch_unet = UNet2DConditionModel.from_pretrained(
        SD14_REPO, subfolder="unet", revision=revision_for(SD14_REPO)
    )
    tokenizer = _get_clip()[0]
    text_encoder = CLIPTextModel.from_pretrained(
        SD14_REPO, subfolder="text_encoder", revision=revision_for(SD14_REPO)
    ).eval()
    for spec, n_unet, n_te in merge_loras(torch_unet, text_encoder, loras):
        print(f"  LoRA {spec.source} @ {spec.scale}: {n_unet} UNet + {n_te} text-encoder tensors merged")
    models = (_build_ttnn_unet(device, torch_unet), vae, torch_unet.config, torch_unet.time_proj)
    from animatediff_ttnn.lora import variant_is_lcm

    return Sd14Variant(
        models=models,
        clip=(tokenizer, text_encoder),
        lcm=variant_is_lcm(loras),
    )


_clip_tokenizer = None
_clip_text_encoder = None
_clip_has_lora = False  # True while the cached encoder carries a merged LoRA


def _encode_one(text: str, clip=None) -> torch.Tensor:
    """CLIP-encode a single string to a (1, 96, 768) embedding.

    Pads 77 → 96 tokens to match the TTNN UNet's expected sequence length.
    Tokenizer and text encoder are cached in module globals after first use —
    loading from HuggingFace Hub takes ~2s and is wasteful to repeat.

    Shared by both the single-prompt path (``encode_prompt``) and the
    prompt-schedule path so keyframe prompts are encoded identically to the
    single prompt (guaranteeing endpoints match the non-scheduled result).
    """
    tokenizer, text_encoder = clip if clip is not None else _get_clip()
    tokens = tokenizer(
        text, padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    with torch.no_grad():
        embeds = text_encoder(tokens.input_ids)[0]
    return torch.nn.functional.pad(embeds, (0, 0, 0, 19))  # 77 → 96 tokens


def _get_clip():
    """Return the cached (tokenizer, text_encoder), loading them on first use."""
    global _clip_tokenizer, _clip_text_encoder
    from transformers import CLIPTokenizer, CLIPTextModel

    if _clip_tokenizer is None:
        _clip_tokenizer = CLIPTokenizer.from_pretrained(
            SD14_REPO, subfolder="tokenizer", revision=revision_for(SD14_REPO)
        )
    if _clip_text_encoder is None:
        _clip_text_encoder = CLIPTextModel.from_pretrained(
            SD14_REPO, subfolder="text_encoder", revision=revision_for(SD14_REPO)
        )
        _clip_text_encoder.eval()
    return _clip_tokenizer, _clip_text_encoder


def encode_prompt(
    prompt: str,
    negative_prompt: str = "",
    prompt_schedule=None,
    num_frames: int | None = None,
    clip=None,
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
        clip: Optional ``(tokenizer, text_encoder)`` to encode with, instead of the
            process-wide cached pair. A LoRA variant passes its own merged encoder here.

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
        return torch.cat([_encode_one(negative_prompt, clip), _encode_one(prompt, clip)], dim=0)  # (2, 96, 768)

    # ── prompt-schedule path ("prompt travel") ─────────────────────────────
    from animatediff_ttnn.prompt_schedule import interpolate_embeddings

    if not prompt_schedule:
        raise ValueError("prompt_schedule is empty — provide at least one keyframe.")
    if num_frames is None or num_frames < 1:
        raise ValueError(
            f"num_frames must be a positive int when prompt_schedule is given, got {num_frames!r}."
        )

    # Constant uncond embedding — only cond travels across frames.
    uncond = _encode_one(negative_prompt, clip)  # (1, 96, 768)

    # Encode each keyframe prompt to its cond embedding (1, 96, 768) and build
    # (frame_index, cond) keyframes for interpolation.
    keyframes = [(frame_index, _encode_one(text, clip)) for frame_index, text in prompt_schedule]
    per_frame_cond = interpolate_embeddings(keyframes, num_frames)  # list of (1, 96, 768)

    # Assemble per-frame [uncond, cond_i] → each (2, 96, 768), matching the
    # single-prompt tensor's shape so the frame loops can index frame-by-frame.
    return [torch.cat([uncond, cond_i], dim=0) for cond_i in per_frame_cond]
