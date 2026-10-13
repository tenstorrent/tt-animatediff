# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Does a merged LoRA survive the trip onto the device?

Visual results (docs/LORA.md) show a LoRA changes the picture. They do not show that the
device UNet computes what the CPU UNet would with the same merge. The TTNN weights are
bfloat16 and a LoRA delta is small, so quantisation could erase part of it.

This runs ONE UNet step (same latent, timestep and text embedding) four ways:

    cpu_base   fp32 CPU UNet, no LoRA              (reference)
    cpu_lora   fp32 CPU UNet, LoRA merged          (reference with the effect)
    dev_base   TTNN UNet, no LoRA
    dev_lora   TTNN UNet, same LoRA merged

and reports, between noise predictions:

    PCC(dev_base, cpu_base)    the device's own error with no LoRA: the floor
    PCC(dev_lora, cpu_lora)    the device's error with the LoRA
    PCC(dev_lora - dev_base, cpu_lora - cpu_base)
                               do the device's LoRA effects match the CPU's?

The last number is the one that can fail when the instrument is blind: the first two can
be high even if the LoRA did nothing on the device, because the LoRA is a small change to a
large signal. Only the difference of differences isolates the effect.

Needs one chip. Run under a lease:
    gozer run --chips 1 --who "claude:lora-check" --reason "lora device numeric check" -- \\
        python scripts/lora_device_check.py --lora "SOURCE[:FILE][@SCALE]" [--device-id 0]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

PROMPT = "a moonlit crypt beneath a drowned chapel, ghostly mist, haunting"
TIMESTEP = 500  # mid-trajectory: the UNet's prediction is neither near-noise nor near-image


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def cpu_unet_step(unet, latent, t, emb) -> torch.Tensor:
    with torch.no_grad():
        return unet(latent, t, encoder_hidden_states=emb).sample


def device_unet_step(device, bundle, latent, t, emb) -> torch.Tensor:
    import ttnn
    from animatediff_ttnn.ttnn_pipeline import build_tlist, to_device

    ttnn_model, _vae, config, time_proj = bundle
    # build_tlist makes the on-device time embedding exactly as the denoise loop does.

    class _Sched:  # build_tlist only reads .timesteps
        timesteps = torch.tensor([float(t)])

    (_t,) = build_tlist(_Sched, time_proj, device)
    out = ttnn_model(
        to_device(latent, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        timestep=_t,
        encoder_hidden_states=to_device(emb, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        class_labels=None,
        attention_mask=None,
        cross_attention_kwargs=None,
        return_dict=True,
        config=config,
    )
    host = ttnn.to_torch(out).float()
    out.deallocate(True)
    return host


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lora", required=True, help="SOURCE[:FILE][@SCALE]")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", help="write the numbers here")
    args = ap.parse_args()

    from diffusers import UNet2DConditionModel
    from animatediff_ttnn.generation_helpers import encode_prompt, load_sd14_ttnn
    from animatediff_ttnn.lora import merge_loras, parse_lora_spec
    from animatediff_ttnn.ttnn_pipeline import setup_blackhole
    from animatediff_ttnn.weights_pins import SD14_REPO, revision_for

    spec = parse_lora_spec(args.lora)
    g = torch.Generator().manual_seed(args.seed)
    latent1 = torch.randn(1, 4, 64, 64, generator=g)
    latent = torch.cat([latent1, latent1], dim=0)  # batch 2 (uncond, cond) as the device UNet expects
    emb = encode_prompt(PROMPT, "")  # (2, 96, 768); built BEFORE any LoRA touches the cached CLIP

    def cpu_unet():
        return UNet2DConditionModel.from_pretrained(
            SD14_REPO, subfolder="unet", revision=revision_for(SD14_REPO)
        ).eval()

    print("CPU references...")
    base = cpu_unet()
    cpu_base = cpu_unet_step(base, latent, TIMESTEP, emb)
    merge_loras(base, None, [spec])  # UNet half only: the text embedding is held fixed
    cpu_lora = cpu_unet_step(base, latent, TIMESTEP, emb)
    del base

    device = setup_blackhole(device_ids=[args.device_id])
    try:
        print("Device, no LoRA...")
        dev_base = device_unet_step(device, load_sd14_ttnn(device), latent, TIMESTEP, emb)
        print("Device, LoRA merged...")
        dev_lora = device_unet_step(device, load_sd14_ttnn(device, loras=[spec]), latent, TIMESTEP, emb)
    finally:
        import ttnn

        ttnn.close_mesh_device(device)

    dev_base, dev_lora = dev_base[: cpu_base.shape[0]], dev_lora[: cpu_base.shape[0]]
    effect_cpu, effect_dev = cpu_lora - cpu_base, dev_lora - dev_base
    res = {
        "lora": args.lora,
        "pcc_dev_base_vs_cpu_base": pcc(dev_base, cpu_base),
        "pcc_dev_lora_vs_cpu_lora": pcc(dev_lora, cpu_lora),
        "pcc_effect_dev_vs_cpu": pcc(effect_dev, effect_cpu),
        "effect_norm_cpu": float(effect_cpu.norm()),
        "effect_norm_dev": float(effect_dev.norm()),
        "effect_to_signal_cpu": float(effect_cpu.norm() / cpu_base.norm()),
    }
    print(json.dumps(res, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
