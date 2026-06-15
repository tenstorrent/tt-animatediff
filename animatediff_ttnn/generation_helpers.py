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


class ChainSession:
    """Persistent Blackhole device session for chained multi-hop generation.

    Opens the device and compiles the TTNN UNet *once*, then runs any number of
    generation hops without re-opening or re-compiling between them.  On GPU
    this is trivial (model stays in VRAM); on Blackhole it requires explicit
    L1 lifecycle management, which this class handles.

    Usage::

        with ChainSession(device_ids=[0]) as sess:
            for scene in scenes:
                frames = sess.run_hop(
                    prompt=scene["prompt"],
                    chain_from=prev_pt,
                    chain_save=cur_pt,
                    chain_alpha=0.35,
                )
                export_gif(frames, scene["out"])
                prev_pt = cur_pt

    The device is closed automatically on context exit (or on exception).
    """

    def __init__(self, device_ids=None, mode="blackhole"):
        self._device_ids = device_ids or [0]
        self._mode = mode
        self.device = None
        self._ttnn_model = None
        self._ttnn_vae = None
        self._config = None
        self._time_proj = None

    def __enter__(self):
        import time
        from animatediff_ttnn.ttnn_pipeline import setup_blackhole, _ensure_tt_metal_path
        import ttnn
        from models.demos.vision.generative.stable_diffusion.wormhole.common import SD_L1_SMALL_SIZE

        if self._mode == "sim":
            _ensure_tt_metal_path()
            self.device = ttnn.open_mesh_device(
                mesh_shape=ttnn.MeshShape(1, 1),
                physical_device_ids=[0],
                l1_small_size=SD_L1_SMALL_SIZE,
            )
        else:
            self.device = setup_blackhole(device_ids=self._device_ids)

        print("ChainSession: loading SD 1.4 models (one-time compile)...")
        t0 = time.time()
        self._ttnn_model, self._ttnn_vae, self._config, self._time_proj = load_sd14_ttnn(self.device)
        print(f"ChainSession: models ready in {time.time() - t0:.1f}s")
        return self

    def __exit__(self, *_):
        if self.device is not None:
            import ttnn
            ttnn.close_mesh_device(self.device)
            print("ChainSession: device closed.")
            self.device = None

    def run_hop(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_frames: int = 8,
        num_steps: int = 25,
        seed: int = 42,
        temporal_alpha: float = 0.35,
        use_lightning: bool = False,
        chain_from=None,
        chain_save=None,
        chain_alpha: float = 0.35,
    ):
        """Run one generation hop using the resident device and compiled model.

        Args:
            prompt: Text prompt for this hop.
            negative_prompt: Negative guidance text.
            num_frames: Number of animation frames to generate.
            num_steps: Denoising steps.
            seed: RNG seed.
            temporal_alpha: Cross-frame attention blend strength.
            use_lightning: Use Euler scheduler instead of PNDM.
            chain_from: Path to .pt latents from the previous hop (or None).
            chain_save: Path to save this hop's latents for the next hop.
            chain_alpha: Blend weight for chain_from (0=ignore, 1=full bias).

        Returns:
            List of PIL Images (one per frame).
        """
        from animatediff_ttnn.temporal_attention import generate_frames_temporal

        text_embeddings = encode_prompt(prompt, negative_prompt)
        return generate_frames_temporal(
            device=self.device,
            ttnn_model=self._ttnn_model,
            ttnn_vae=self._ttnn_vae,
            config=self._config,
            torch_time_proj=self._time_proj,
            text_embeddings=text_embeddings,
            num_frames=num_frames,
            num_steps=num_steps,
            seed=seed,
            temporal_alpha=temporal_alpha,
            use_lightning=use_lightning,
            chain_from=chain_from,
            chain_save=chain_save,
            chain_alpha=chain_alpha,
        )


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
