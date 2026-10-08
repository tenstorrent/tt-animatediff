# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""CPU tests for animatediff_ttnn.lora (no device, no network).

What these cannot show: that a real Hub LoRA merges into the real SD 1.4 UNet. That was
checked by hand on 2026-10-07 and is recorded in docs/LORA.md.
"""
import ast
import inspect
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from animatediff_ttnn import lora
from animatediff_ttnn.lora import LoraSpec, parse_lora_spec

REPO = Path(__file__).resolve().parent.parent


class TestParse:
    def test_repo_only(self):
        assert parse_lora_spec("org/style") == LoraSpec("org/style", None, 1.0)

    def test_file_and_scale(self):
        assert parse_lora_spec("org/style:s.safetensors@0.8") == LoraSpec("org/style", "s.safetensors", 0.8)

    def test_scale_without_file(self):
        assert parse_lora_spec("/tmp/x.safetensors@-0.5") == LoraSpec("/tmp/x.safetensors", None, -0.5)

    def test_empty_source_rejected(self):
        with pytest.raises(ValueError):
            parse_lora_spec(":file.safetensors")


class _TinyClip(torch.nn.Module):
    """Two CLIP-shaped layers: just enough names for the kohya key mapping to resolve."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            torch.nn.ModuleDict(
                {
                    "self_attn": torch.nn.ModuleDict({"q_proj": torch.nn.Linear(8, 8, bias=False)}),
                    "mlp": torch.nn.ModuleDict({"fc1": torch.nn.Linear(8, 16, bias=False)}),
                }
            )
            for _ in range(2)
        )


def _write_te_lora(path, rank=4, alpha=2.0):
    g = torch.Generator().manual_seed(0)
    t = {}
    for layer, block, leaf, out in [(1, "self_attn", "q_proj", 8), (0, "mlp", "fc1", 16)]:
        base = f"lora_te_text_model_encoder_layers_{layer}_{block}_{leaf}"
        t[f"{base}.lora_down.weight"] = torch.randn(rank, 8, generator=g)
        t[f"{base}.lora_up.weight"] = torch.randn(out, rank, generator=g)
        t[f"{base}.alpha"] = torch.tensor(alpha)
    save_file(t, str(path))
    return t


class TestTextEncoderMerge:
    def test_delta_matches_the_lora_formula(self, tmp_path):
        f = tmp_path / "l.safetensors"
        t = _write_te_lora(f, rank=4, alpha=2.0)
        te = _TinyClip()
        before = te.layers[1]["self_attn"]["q_proj"].weight.detach().clone()
        n = lora._merge_text_encoder(str(f), te, scale=0.5)
        k = "lora_te_text_model_encoder_layers_1_self_attn_q_proj"
        expect = 0.5 * (2.0 / 4) * (t[f"{k}.lora_up.weight"] @ t[f"{k}.lora_down.weight"])
        assert n == 2
        assert torch.allclose(te.layers[1]["self_attn"]["q_proj"].weight - before, expect, atol=1e-6)

    def test_scale_zero_changes_nothing(self, tmp_path):
        f = tmp_path / "l.safetensors"
        _write_te_lora(f)
        te = _TinyClip()
        before = te.layers[0]["mlp"]["fc1"].weight.detach().clone()
        lora._merge_text_encoder(str(f), te, scale=0.0)
        assert torch.equal(te.layers[0]["mlp"]["fc1"].weight, before)

    def test_unmatched_key_raises(self, tmp_path):
        f = tmp_path / "l.safetensors"
        _write_te_lora(f)
        te = _TinyClip()
        te.layers = te.layers[:1]  # layer 1 no longer exists
        with pytest.raises(ValueError, match="no text-encoder module"):
            lora._merge_text_encoder(str(f), te, scale=1.0)


class TestResolve:
    def test_local_file(self, tmp_path):
        f = tmp_path / "a.safetensors"
        f.write_bytes(b"")
        assert lora._resolve_file(LoraSpec(str(f))) == str(f)

    def test_hub_repo_without_file_name_is_refused(self):
        with pytest.raises(ValueError, match="name the file"):
            lora._resolve_file(LoraSpec("org/repo"))


class TestWiring:
    """Guard the layer that fails: the merge must happen BEFORE TTNN preprocessing.

    Merging after preprocess_model_parameters would still run, report success and leave
    the device with the unmodified weights.
    """

    def test_load_sd14_ttnn_merges_before_preprocessing(self):
        from animatediff_ttnn import generation_helpers as g

        src = inspect.getsource(g.load_sd14_ttnn)
        assert "merge_loras(" in src
        assert src.index("merge_loras(") < src.index("preprocess_model_parameters(")

    def test_cli_passes_loras_to_the_loader(self):
        tree = ast.parse((REPO / "examples" / "generate.py").read_text())
        calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "load_sd14_ttnn"
        ]
        assert calls and all(any(k.arg == "loras" for k in c.keywords) for c in calls)
