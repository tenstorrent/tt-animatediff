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


def encode_prompt(prompt: str, negative_prompt: str = "") -> torch.Tensor:
    """Encode text prompt pair to (2, 96, 768) CLIP embeddings.

    Pads 77 → 96 tokens to match TTNN UNet's expected sequence length.
    Tokenizer and text encoder are cached after first call — loading from
    HuggingFace Hub takes ~2s and is wasteful to repeat per generation.
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

    tokenizer = _clip_tokenizer
    text_encoder = _clip_text_encoder

    def encode(text):
        tokens = tokenizer(
            text, padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            embeds = text_encoder(tokens.input_ids)[0]
        return torch.nn.functional.pad(embeds, (0, 0, 0, 19))  # 77 → 96 tokens

    return torch.cat([encode(negative_prompt), encode(prompt)], dim=0)  # (2, 96, 768)
