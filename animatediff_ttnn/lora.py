# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Style LoRA support: merge Stable Diffusion LoRAs into the weights the TTNN UNet is built from.

Why merge on the CPU, before the device sees anything
-----------------------------------------------------
The TTNN UNet is built by ``preprocess_model_parameters`` from a CPU
``UNet2DConditionModel``. A LoRA is a low-rank delta on selected weight matrices,
``W' = W + scale * (alpha / rank) * (up @ down)``. Folding that into the CPU weights
*before* preprocessing means the device runs the ordinary SD UNet with different numbers
in it. There is no extra kernel, no extra op and no per-step cost. A merge takes seconds
and happens once per process, at model load.

What this module does and does not load
---------------------------------------
* Style LoRAs for SD 1.x in kohya format (``lora_unet_*`` / ``lora_te_*`` keys), which is
  what most Civitai / Hub style LoRAs ship. They patch the UNet and, if present, the CLIP
  text encoder. Diffusers converts the key names; this module does not parse them.
* It does NOT load LoRAs trained against a different base model (SDXL, Flux, Krea-2...).
  Those have different layer shapes and fail with a shape or key mismatch.
* It does NOT load the ``guoyww/animatediff-motion-lora-*`` files. Those patch the motion
  modules, which live in the MotionAdapter and not in the SD UNet.

SD 1.4 versus SD 1.5: the LoRAs on the Hub are mostly trained on 1.5, and this repo's
base is 1.4. The two share one architecture, so the shapes match and the merge succeeds.
Whether a 1.5 style transfers cleanly onto 1.4 weights is a visual question that tests
cannot answer; docs/LORA.md records what was observed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class LoraSpec:
    """One LoRA to merge: where it lives, which file inside it, and how strongly.

    ``source`` is a Hub repo id or a local path (file or directory).
    ``weight_name`` selects a file when the repo holds several (or a directory is given).
    ``scale`` multiplies the delta; 1.0 is the strength the author trained for.
    """

    source: str
    weight_name: Optional[str] = None
    scale: float = 1.0


# "repo/id[:file.safetensors][@0.8]". The scale is peeled off the right first so a
# Windows-style or colon-bearing path in the middle cannot be mistaken for it.
_SCALE_RE = re.compile(r"@([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)$")


def parse_lora_spec(text: str) -> LoraSpec:
    """Parse the CLI form ``SOURCE[:WEIGHT_NAME][@SCALE]`` into a LoraSpec.

    >>> parse_lora_spec("org/style:style.safetensors@0.8")
    LoraSpec(source='org/style', weight_name='style.safetensors', scale=0.8)
    """
    scale = 1.0
    m = _SCALE_RE.search(text)
    if m:
        scale = float(m.group(1))
        text = text[: m.start()]
    source, _, weight_name = text.partition(":")
    if not source:
        raise ValueError(f"empty LoRA source in {text!r}")
    if "@" in weight_name or (not weight_name and "@" in source):
        raise ValueError(f"could not read a scale from {text!r}; write it as ...@0.8")
    return LoraSpec(source=source, weight_name=weight_name or None, scale=scale)


def _resolve_file(spec: LoraSpec) -> str:
    """Return a local .safetensors path for ``spec`` (downloading from the Hub if needed)."""
    import os

    name = spec.weight_name or spec.source
    if not name.endswith(".safetensors"):
        raise ValueError(f"LoRA files must be .safetensors, got {name!r}")
    if os.path.isfile(spec.source):
        return spec.source
    if os.path.isdir(spec.source):
        if not spec.weight_name:
            raise ValueError(f"{spec.source!r} is a directory; give a weight file name")
        path = os.path.join(spec.source, spec.weight_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return path
    if spec.source.startswith((".", "/", "~")) or spec.source.endswith(".safetensors"):
        raise FileNotFoundError(f"LoRA file not found: {spec.source}")
    from huggingface_hub import hf_hub_download

    from animatediff_ttnn.weights_pins import revision_for

    if not spec.weight_name:
        raise ValueError(
            f"{spec.source!r} is a Hub repo; name the file, e.g. {spec.source}:file.safetensors"
        )
    # revision_for() is None for any repo this package does not pin, which resolves to
    # the repo's default branch. A user-chosen LoRA is by definition not pinned here;
    # the call still goes through revision_for so the package-wide pinning guard
    # (tests/test_weights_pins.py) holds and a future pin takes effect automatically.
    return hf_hub_download(spec.source, spec.weight_name, revision=revision_for(spec.source))


# kohya names a text-encoder layer lora_te_text_model_encoder_layers_<N>_<module with _ for .>
_TE_KEY = re.compile(r"^lora_te_text_model_encoder_layers_(\d+)_(self_attn|mlp)_(\w+)\.lora_down\.weight$")


def _merge_text_encoder(path: str, text_encoder, scale: float) -> int:
    """Fold the ``lora_te_*`` part of a kohya file into a CLIP text encoder. Returns tensors changed.

    Done by hand because diffusers matches LoRA keys to
    module names by string, and that match finds nothing on the transformers 5.x
    CLIPTextModel layout (it raises IndexError on an empty rank map). The format is
    small: per target, ``W += scale * (alpha / rank) * (up @ down)``.
    """
    import torch
    from safetensors import safe_open

    modules = dict(text_encoder.named_modules())
    by_suffix = {}
    for n in modules:
        by_suffix.setdefault(n.split('encoder.', 1)[-1], []).append(n)
    changed = 0
    with safe_open(path, "pt") as f:
        keys = set(f.keys())
        for k in sorted(keys):
            m = _TE_KEY.match(k)
            if not m:
                continue
            layer, block, leaf = m.groups()
            base = k[: -len(".lora_down.weight")]
            hits = by_suffix.get(f"layers.{layer}.{block}.{leaf}", [])
            if len(hits) != 1:
                raise ValueError(f"{len(hits)} text-encoder modules match LoRA key {k!r}; expected exactly 1")
            target = modules[hits[0]]
            down, up = f.get_tensor(k).float(), f.get_tensor(f"{base}.lora_up.weight").float()
            alpha = f.get_tensor(f"{base}.alpha").item() if f"{base}.alpha" in keys else down.shape[0]
            delta = scale * (alpha / down.shape[0]) * (up @ down)
            with torch.no_grad():
                target.weight += delta.to(target.weight.dtype)
            changed += 1
    return changed


def merge_loras(unet, text_encoder, specs: Iterable[LoraSpec]) -> List[Tuple[LoraSpec, int, int]]:
    """Fuse each LoRA into ``unet`` and ``text_encoder`` in place.

    The UNet part goes through diffusers' kohya converter and fuse; the adapter layers
    are removed afterwards, so ``preprocess_model_parameters`` sees a plain SD UNet with
    modified weights. The text-encoder part is merged by ``_merge_text_encoder``. LoRAs
    apply in order and their deltas add.

    Returns ``[(spec, n_unet_tensors_changed, n_text_encoder_tensors_changed), ...]``.

    Raises:
        ValueError: a LoRA changed no UNet weight. That is the signature of a LoRA for
            the wrong base model, and returning quietly would look like success.
    """
    import torch

    results = []
    from diffusers.loaders import StableDiffusionLoraLoaderMixin as Loader

    for i, spec in enumerate(specs):
        name = f"tt_lora_{i}"
        path = _resolve_file(spec)
        before = {k: v.detach().clone() for k, v in unet.state_dict().items()}
        # UNet half only: the stock loader's text-encoder half cannot see the
        # transformers 5.x CLIP layout, so that half is _merge_text_encoder's job.
        state, alphas = Loader.lora_state_dict(path)
        Loader.load_lora_into_unet(state, alphas, unet, adapter_name=name)
        unet.set_adapters([name], [spec.scale])
        unet.fuse_lora()
        unet.unload_lora()
        after = unet.state_dict()
        changed = sum(1 for k, v in before.items() if not torch.equal(v, after[k]))
        te_changed = _merge_text_encoder(path, text_encoder, spec.scale) if text_encoder is not None else 0
        if changed == 0 and te_changed == 0 and spec.scale != 0:
            raise ValueError(
                f"LoRA {spec.source!r} loaded but changed no weight. It was most "
                "likely trained for a different base model."
            )
        results.append((spec, changed, te_changed))
    return results
