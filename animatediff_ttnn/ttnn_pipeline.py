# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Phase 2: Blackhole-accelerated video frame generation using TTNN UNet.

Uses the SD 1.4 TTNN UNet from ~/tt-metal (same code runs on Blackhole via
TT_METAL_ARCH_NAME=blackhole). Frames denoised sequentially; temporal
coherence from shared base noise initialization (0.05 per-frame perturbation).

Documented tradeoff: this is TT-hardware-accelerated spatial denoising, not
full AnimateDiff temporal attention. Full integration would require injecting
TemporalTransformer blocks into the TTNN UNet transformer blocks.

Requirements:
    ~/tt-metal present and activated: source ~/tt-metal/python_env/bin/activate
    Blackhole hardware (P100 or P300c)
"""

import os
import sys
from pathlib import Path
from typing import List

import torch
from PIL import Image


TT_METAL_PATH = Path.home() / "tt-metal"


def _ensure_tt_metal_path() -> None:
    """Add ~/tt-metal to sys.path so SD demo module imports work."""
    if not TT_METAL_PATH.exists():
        raise RuntimeError(
            f"~/tt-metal not found. Activate the tt-metal environment first:\n"
            f"  cd ~/tt-metal && source python_env/bin/activate"
        )
    tt_metal_str = str(TT_METAL_PATH)
    if tt_metal_str not in sys.path:
        sys.path.insert(0, tt_metal_str)


def _constant_prop_time_embeddings(timesteps, sample, time_proj):
    """Compute time embeddings for a scalar timestep.

    Equivalent to the function defined in SD demo.py. Accepts a scalar
    timestep tensor, expands to batch size, runs through UNet time_proj.
    """
    timesteps = timesteps[None]
    timesteps = timesteps.expand(sample.shape[0])
    return time_proj(timesteps)


def setup_blackhole(device_ids: list[int] | None = None):
    """Open all available Blackhole chips as a 1×N MeshDevice.

    Uses open_mesh_device with explicit physical_device_ids so every chip is
    claimed upfront. This prevents the ARC on un-claimed chips from entering a
    degraded state mid-run and avoids the implicit device_id=0 assumption that
    breaks on multi-card systems where PCIe enumeration order is not guaranteed.

    Args:
        device_ids: Physical device IDs to open (default: all available chips).

    Returns the open MeshDevice — compatible with preprocess_model_parameters,
    UNet2D, and ttnn.from_torch(..., device=mesh_device).
    """
    os.environ.setdefault("TT_METAL_ARCH_NAME", "blackhole")
    _ensure_tt_metal_path()

    import ttnn
    from models.demos.vision.generative.stable_diffusion.wormhole.common import SD_L1_SMALL_SIZE

    if device_ids is None:
        # Warn if any hwmon entry shows the ARC-dead sentinel (temp > 1000°C /
        # power = 4294W). We can't reliably map hwmon enumeration order to TTNN
        # physical device IDs, so we do not filter the id list — that could
        # exclude a healthy chip. Instead we warn early so the user knows to
        # AC power-cycle before TTNN's own enumeration times out.
        import glob as _glob
        import warnings
        dead_hwmon = []
        for hwmon in sorted(_glob.glob("/sys/class/hwmon/hwmon*")):
            try:
                if open(f"{hwmon}/name").read().strip() != "blackhole":
                    continue
                temp_mc = int(open(f"{hwmon}/temp1_input").read().strip())
                if temp_mc > 1_000_000:  # sentinel: ARC dead
                    dead_hwmon.append(hwmon)
            except (OSError, ValueError):
                pass
        if dead_hwmon:
            warnings.warn(
                f"hwmon entries {dead_hwmon} show ARC-dead sentinel temperatures "
                f"(> 1000°C). One or more chips may be unresponsive. If TTNN "
                f"enumeration hangs, AC power-cycle the system and retry.",
                RuntimeWarning, stacklevel=2,
            )
        device_ids = list(range(ttnn.GetNumAvailableDevices()))

    n = len(device_ids)
    return ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, n),
        physical_device_ids=device_ids,
        l1_small_size=SD_L1_SMALL_SIZE,
    )


def to_device(tensor, device, dtype=None, layout=None):
    """Send a torch tensor to device (single or mesh) with automatic replication.

    For a MeshDevice every chip gets an identical copy of the tensor — correct
    for data-parallel SD inference where all chips run the same model.
    """
    import ttnn
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if layout is not None:
        kwargs["layout"] = layout
    if isinstance(device, ttnn.MeshDevice):
        kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
    return ttnn.from_torch(tensor, device=device, **kwargs)


def from_device(tensor, device, batch: int = 1):
    """Retrieve a tensor from device (single or mesh) back to CPU torch.

    For a MeshDevice, ConcatMeshToTensor(dim=0) stacks all N chip replicas into
    a single tensor whose first dimension is batch*N. We take [:batch] to recover
    one replica. Pass the actual batch size if it differs from 1 — the default
    covers all current call sites (guided noise_pred and latents are both batch=1).
    """
    import ttnn
    if isinstance(device, ttnn.MeshDevice):
        return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))[:batch]
    return ttnn.to_torch(tensor)


def build_tlist(ttnn_scheduler, torch_time_proj, device, latent_h: int = 64, latent_w: int = 64) -> list:
    """Build pre-computed time embeddings for each denoising timestep.

    sd_helper_funcs.run() expects _tlist[i] to be constant_prop_time_embeddings(t_i)
    already converted to a TTNN bfloat16 tensor on device, shape permuted for UNet.

    Args:
        ttnn_scheduler: TtPNDMScheduler with set_timesteps() already called
        torch_time_proj: unet.time_proj from the PyTorch UNet2DConditionModel
        device: TTNN device from setup_blackhole()
        latent_h: Latent height (image_height // 8; 64 for 512px images)
        latent_w: Latent width (image_width // 8; 64 for 512px images)

    Returns:
        List of TTNN tensors, one per timestep
    """
    import ttnn

    dummy = to_device(
        torch.zeros(2, 4, latent_h, latent_w),
        device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    )

    _tlist = []
    for t in ttnn_scheduler.timesteps:
        _t = _constant_prop_time_embeddings(t, dummy, torch_time_proj)
        _t = _t.unsqueeze(0).unsqueeze(0)
        _t = _t.permute(2, 0, 1, 3)
        _t = to_device(_t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        _tlist.append(_t)
    return _tlist


def generate_frames(
    device,
    ttnn_model,
    ttnn_vae,
    config,
    ttnn_scheduler,
    torch_time_proj,
    text_embeddings: torch.Tensor,
    num_frames: int = 8,
    guidance_scale: float = 7.5,
    seed: int = 42,
    height: int = 512,
    width: int = 512,
) -> List[Image.Image]:
    """Generate video frames using TTNN UNet on Blackhole.

    UNet denoising runs on Blackhole via TTNN. VAE decode also runs on Blackhole
    via the TTNN Vae — UNet L1 tensors are deallocated before decode to free L1.

    Args:
        device: TTNN Blackhole device from setup_blackhole()
        ttnn_model: UNet2D TTNN model loaded with preprocess_model_parameters
        ttnn_vae: TTNN Vae decoder (from load_sd14_ttnn)
        config: unet.config from PyTorch UNet2DConditionModel
        ttnn_scheduler: TtPNDMScheduler (set_timesteps already called once)
        torch_time_proj: unet.time_proj from PyTorch UNet (used to build _tlist)
        text_embeddings: Shape (2, 96, 768) torch tensor — [uncond, cond] concatenated,
                         padded from 77 to 96 tokens with torch.nn.functional.pad(..., (0,0,0,19))
        num_frames: Number of frames to generate
        guidance_scale: CFG scale (7.5 standard)
        seed: Random seed for shared base noise
        height: Output image height in pixels (512 recommended for single Blackhole)
        width: Output image width in pixels (512 recommended for single Blackhole)

    Returns:
        List of PIL Images, one per frame

    Note:
        Temporal coherence comes from shared base noise initialization (0.05
        per-frame perturbation). This is not full AnimateDiff temporal attention —
        see Phase 1 (generate_baseline.py) for that.
    """
    import ttnn
    from models.demos.vision.generative.stable_diffusion.wormhole.sd_helper_funcs import tt_guide

    num_steps = ttnn_scheduler.num_inference_steps
    lh, lw = height // 8, width // 8

    # Convert text embeddings to TTNN device tensor once; reused every frame
    ttnn_text_embeddings = to_device(
        text_embeddings, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    )

    # Shared base noise for inter-frame temporal coherence
    generator = torch.Generator().manual_seed(seed)
    base_noise = torch.randn(1, 4, lh, lw, generator=generator)

    # Build time embeddings once — timesteps are the same for every frame
    ttnn_scheduler.set_timesteps(num_steps)
    time_step = ttnn_scheduler.timesteps.tolist()
    _tlist = build_tlist(ttnn_scheduler, torch_time_proj, device, lh, lw)

    frames = []
    for frame_idx in range(num_frames):
        # Reset PNDM scheduler state (counter, ets buffer) before each frame
        ttnn_scheduler.set_timesteps(num_steps)

        # Per-frame perturbation uses the seeded generator so runs are reproducible
        frame_noise = base_noise + 0.05 * torch.randn(base_noise.shape, generator=generator)
        ttnn_latents = to_device(
            frame_noise * ttnn_scheduler.init_noise_sigma,
            device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        )

        # TTNN UNet denoising loop on Blackhole
        ttnn_latent_model_input = None
        ttnn_output = None
        noise_pred = None
        for index in range(len(time_step)):
            ttnn_latent_model_input = ttnn.concat([ttnn_latents, ttnn_latents], dim=0)
            _t = _tlist[index]
            t = time_step[index]
            ttnn_output = ttnn_model(
                ttnn_latent_model_input,
                timestep=_t,
                encoder_hidden_states=ttnn_text_embeddings,
                class_labels=None,
                attention_mask=None,
                cross_attention_kwargs=None,
                return_dict=True,
                config=config,
            )
            noise_pred = tt_guide(ttnn_output, guidance_scale)
            ttnn_latents = ttnn_scheduler.step(noise_pred, t, ttnn_latents).prev_sample

        # Deallocate live UNet L1 tensors before VAE decode — same pattern as
        # sd_helper_funcs.py::run() in tt-metal SD demo. Without this the VAE
        # conv layers OOM because UNet output buffers still occupy L1.
        if ttnn_latent_model_input is not None:
            ttnn_latent_model_input.deallocate(True)
        if ttnn_output is not None:
            ttnn_output.deallocate(True)
        if noise_pred is not None:
            noise_pred.deallocate(True)

        # Decode with TTNN VAE on Blackhole — permute to NHWC for conv layout
        latent_scaled = from_device(ttnn_latents, device).to(torch.float32) / 0.18215
        ttnn_latents.deallocate(True)
        ttnn_lat = to_device(
            latent_scaled.permute(0, 2, 3, 1),
            device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        )
        ttnn_decoded = ttnn_vae.decode(ttnn_lat)
        ttnn_lat.deallocate(True)
        ttnn_decoded = ttnn.reshape(ttnn_decoded, [1, height, width, 3])
        decoded = ttnn.to_torch(ttnn.permute(ttnn_decoded, [0, 3, 1, 2])).float()
        ttnn_decoded.deallocate(True)
        img = (decoded / 2 + 0.5).clamp(0, 1)
        img = (img.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
        frames.append(Image.fromarray(img))

        print(f"  Frame {frame_idx + 1}/{num_frames} done")

    return frames
