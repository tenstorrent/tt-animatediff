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

import json
import math
import re
from dataclasses import dataclass, field
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

    def __post_init__(self):
        # Checked here, not only in parse_lora_spec, so a spec built in code gets the same
        # guard. "@1e999" matches the scale regex and float() turns it into inf; fusing
        # inf (or nan) writes it into every targeted weight and corrupts inference.
        if not math.isfinite(self.scale):
            raise ValueError(f"LoRA scale must be a finite number, got {self.scale!r} for {self.source!r}")


# The published LCM-LoRA, as a spec. Merged alongside a style LoRA (their deltas add) and
# paired with the LCM sampler, it cuts 20-25 steps to about 4. See docs/LORA.md.
def lcm_lora_spec(scale: float = 1.0) -> "LoraSpec":
    from animatediff_ttnn.weights_pins import LCM_LORA_FILE, LCM_LORA_REPO

    return LoraSpec(LCM_LORA_REPO, LCM_LORA_FILE, scale)


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


@dataclass(frozen=True)
class LoraInfo:
    """What a LoRA file says about itself, read from its safetensors header only.

    Nothing here loads a tensor, so inspecting a multi-hundred-MB file is instant and needs
    neither a device nor a UNet. ``base`` and the trigger words come from kohya's ``ss_*``
    metadata, which is optional and written by the trainer, not verified: treat them as
    hints. The key-name fields (``family``, ``kind``) come from the tensor names and are
    what the compatibility decision uses.
    """

    path: str
    family: str  # "sd1" | "sdxl" | "unknown"
    kind: str  # "lora" | "loha" | "lokr" | "unknown"
    has_dora: bool = False
    has_conv: bool = False  # LoCon-style: the LoRA also targets conv layers
    rank: Optional[int] = None
    base_hint: Optional[str] = None  # trainer-reported base model, if any
    trigger_words: Tuple[str, ...] = field(default_factory=tuple)
    n_unet_modules: int = 0
    n_text_encoder_modules: int = 0


def _trigger_words(meta: dict, limit: int = 8) -> Tuple[str, ...]:
    """Most frequent training tags from kohya's ``ss_tag_frequency``, as activation hints.

    The field is ``{dataset: {tag: count}}`` serialised as JSON. The top tags are the words
    the model saw most, which is where a style or character's trigger phrase usually lives,
    but a LoRA need not have one, so this is a suggestion and never a requirement.
    """
    raw = meta.get("ss_tag_frequency")
    if not raw:
        return ()
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return ()
    counts: dict = {}
    for tags in data.values() if isinstance(data, dict) else ():
        if isinstance(tags, dict):
            for tag, n in tags.items():
                if isinstance(n, (int, float)):
                    counts[tag.strip()] = counts.get(tag.strip(), 0) + n
    ranked = sorted(counts, key=lambda t: (-counts[t], t))
    return tuple(t for t in ranked if t)[:limit]


def inspect_lora(path: str) -> LoraInfo:
    """Read a ``.safetensors`` LoRA's header and classify it. No tensor data is loaded."""
    from safetensors import safe_open

    with safe_open(path, "pt") as f:
        keys = list(f.keys())
        meta = f.metadata() or {}
        rank = None
        for k in keys:
            if k.endswith(".lora_down.weight"):
                rank = f.get_slice(k).get_shape()[0]
                break

    unet = [k for k in keys if k.startswith("lora_unet_")]
    te = [k for k in keys if k.startswith(("lora_te_", "lora_te1_", "lora_te2_"))]
    # SDXL has two text encoders, so kohya names them lora_te1_/lora_te2_. SD 1.x has the
    # single lora_te_. The metadata agrees when present; the key names win because they
    # are what the loader actually sees.
    sdxl_meta = (meta.get("ss_base_model_version", "") + meta.get("modelspec.architecture", "")).lower()
    if any(k.startswith(("lora_te1_", "lora_te2_")) for k in te) or "sdxl" in sdxl_meta:
        family = "sdxl"
    elif unet or any(k.startswith("lora_te_") for k in te):
        family = "sd1"
    else:
        family = "unknown"

    if any(".hada_w" in k for k in keys):
        kind = "loha"
    elif any(".lokr_w" in k for k in keys):
        kind = "lokr"
    elif any(k.endswith(".lora_down.weight") for k in keys):
        kind = "lora"
    else:
        kind = "unknown"

    return LoraInfo(
        path=path,
        family=family,
        kind=kind,
        has_dora=any(k.endswith(".dora_scale") for k in keys),
        # SD 1.x conv targets in kohya naming: resnets, downsamplers, upsamplers, conv_in/out.
        has_conv=any(re.search(r"_(resnets|downsamplers|upsamplers)_|_conv_(in|out)\.", k) for k in unet),
        rank=rank,
        base_hint=meta.get("ss_sd_model_name") or meta.get("ss_base_model_version") or None,
        trigger_words=_trigger_words(meta),
        n_unet_modules=len({k.split(".")[0] for k in unet}),
        n_text_encoder_modules=len({k.split(".")[0] for k in te}),
    )


def check_lora_compatible(info: LoraInfo) -> None:
    """Raise ValueError, naming the reason, for a file this module cannot merge into SD 1.x.

    Runs before any model loads, so a wrong file fails in milliseconds and before a chip is
    claimed. Without it the same mistakes surface only after the UNet has loaded, and as the
    vaguer "changed no weight".
    """
    name = info.path
    if info.family == "sdxl":
        raise ValueError(f"{name}: this is an SDXL LoRA; this model is Stable Diffusion 1.x")
    if info.kind in ("loha", "lokr"):
        raise ValueError(
            f"{name}: this is a {info.kind.upper()} (LyCORIS) file. Only plain LoRA and LoCon "
            "files (lora_down / lora_up weights) can be merged here"
        )
    if info.kind == "unknown" or info.family == "unknown":
        raise ValueError(
            f"{name}: no kohya-format LoRA tensors found (expected lora_unet_* / lora_te_* "
            "keys with lora_down / lora_up weights). Diffusers-format and motion LoRAs are not supported"
        )
    if info.has_dora:
        raise ValueError(f"{name}: this is a DoRA file; its magnitude vectors are not merged here")


def _resolve_file(spec: LoraSpec) -> str:
    """Return a local .safetensors path for ``spec`` (downloading from the Hub if needed)."""
    import os

    name = spec.weight_name or spec.source
    if not name.endswith(".safetensors"):
        raise ValueError(f"LoRA files must be .safetensors, got {name!r}")
    # The shell expands "~" only when it is unquoted and leads a word, so a value like
    # --lora ~/loras/x.safetensors@0.8 or one read from a config arrives with the tilde intact.
    # Expand before testing, otherwise an existing file is reported as missing.
    source = os.path.expanduser(spec.source)
    if os.path.isfile(source):
        return source
    if os.path.isdir(source):
        if not spec.weight_name:
            raise ValueError(f"{spec.source!r} is a directory; give a weight file name")
        path = os.path.join(source, spec.weight_name)
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


def _lora_targets(unet, adapter_name: str) -> list:
    """Return the base layers that carry adapter ``adapter_name`` (peft ``LoraLayer`` wrappers).

    Each returned module is the original ``nn.Linear`` / ``nn.Conv2d``, whose ``weight``
    is what ``fuse_lora`` overwrites. The same objects stay in the UNet after
    ``unload_lora`` unwraps them, so their ``weight`` can be read again after the fuse.
    """
    return [
        m.base_layer
        for m in unet.modules()
        if hasattr(m, "base_layer") and adapter_name in getattr(m, "lora_A", {})
    ]


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
        check_lora_compatible(inspect_lora(path))
        # UNet half only: the stock loader's text-encoder half cannot see the
        # transformers 5.x CLIP layout, so that half is _merge_text_encoder's job.
        state, alphas = Loader.lora_state_dict(path)
        Loader.load_lora_into_unet(state, alphas, unet, adapter_name=name)
        # Count changed tensors by snapshotting only the layers this LoRA targets (the
        # loader has just wrapped them), not the whole UNet. A LoRA touches the attention
        # projections, a fraction of the model, and the snapshot is dropped after each LoRA.
        targets = _lora_targets(unet, name)
        before = [t.weight.detach().clone() for t in targets]
        unet.set_adapters([name], [spec.scale])
        unet.fuse_lora()
        unet.unload_lora()
        changed = sum(1 for t, b in zip(targets, before) if not torch.equal(t.weight, b))
        del before
        te_changed = _merge_text_encoder(path, text_encoder, spec.scale) if text_encoder is not None else 0
        if changed == 0 and te_changed == 0 and spec.scale != 0:
            raise ValueError(
                f"LoRA {spec.source!r} loaded but changed no weight. It was most "
                "likely trained for a different base model."
            )
        results.append((spec, changed, te_changed))
    return results
