# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""CPU tests for animatediff_ttnn.lora (no device, no network).

What these cannot show: that a real Hub LoRA merges into the real SD 1.4 UNet. That was
checked by hand on 2026-10-07 and is recorded in docs/LORA.md.
"""
import ast
import inspect
import json
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

    @pytest.mark.parametrize("text,scale", [("a/b:f.safetensors@.8", 0.8), ("a/b:f.safetensors@1.", 1.0),
                                            ("a/b:f.safetensors@+0.5", 0.5), ("a/b:f.safetensors@1e-1", 0.1)])
    def test_scale_forms(self, text, scale):
        assert parse_lora_spec(text) == LoraSpec("a/b", "f.safetensors", scale)

    def test_garbled_scale_is_rejected_not_folded_into_the_filename(self):
        with pytest.raises(ValueError, match="scale"):
            parse_lora_spec("a/b:f.safetensors@abc")

    @pytest.mark.parametrize("text", ["a/b:f.safetensors@1e999", "a/b:f.safetensors@-1e999"])
    def test_overflowing_scale_is_rejected(self, text):
        # float("1e999") is inf; fused into the weights it corrupts every targeted layer.
        with pytest.raises(ValueError, match="finite"):
            parse_lora_spec(text)

    @pytest.mark.parametrize("scale", [float("inf"), float("-inf"), float("nan")])
    def test_non_finite_scale_rejected_on_direct_construction(self, scale):
        with pytest.raises(ValueError, match="finite"):
            LoraSpec("a/b", "f.safetensors", scale)

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

    def test_ambiguous_target_raises(self, tmp_path):
        f = tmp_path / "l.safetensors"
        _write_te_lora(f)
        te = _TinyClip()
        te.a_encoder = torch.nn.ModuleDict({"layers": __import__("copy").deepcopy(te.layers)})  # same suffix under a second name
        with pytest.raises(ValueError, match="expected exactly 1"):
            lora._merge_text_encoder(str(f), te, scale=1.0)

    def test_unmatched_key_raises(self, tmp_path):
        f = tmp_path / "l.safetensors"
        _write_te_lora(f)
        te = _TinyClip()
        te.layers = te.layers[:1]  # layer 1 no longer exists
        with pytest.raises(ValueError, match="expected exactly 1"):
            lora._merge_text_encoder(str(f), te, scale=1.0)


def _save(path, keys, metadata=None):
    """Write a safetensors file whose tensors are tiny but whose names/shapes are as given."""
    save_file({k: torch.zeros(*shape) for k, shape in keys.items()}, str(path), metadata=metadata)
    return str(path)


_SD1 = {
    "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight": (4, 8),
    "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight": (8, 4),
    "lora_te_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight": (4, 8),
}


class TestInspect:
    def test_sd1_lora(self, tmp_path):
        info = lora.inspect_lora(_save(tmp_path / "a.safetensors", _SD1))
        assert (info.family, info.kind, info.rank) == ("sd1", "lora", 4)
        assert (info.n_unet_modules, info.n_text_encoder_modules) == (1, 1)
        assert not info.has_conv and not info.has_dora
        lora.check_lora_compatible(info)  # does not raise

    def test_sdxl_detected_by_two_text_encoders(self, tmp_path):
        keys = dict(_SD1)
        keys["lora_te2_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight"] = (4, 8)
        info = lora.inspect_lora(_save(tmp_path / "x.safetensors", keys))
        assert info.family == "sdxl"
        with pytest.raises(ValueError, match="SDXL"):
            lora.check_lora_compatible(info)

    def test_sdxl_detected_by_metadata(self, tmp_path):
        info = lora.inspect_lora(_save(tmp_path / "x.safetensors", _SD1, {"ss_base_model_version": "sdxl_base_v1-0"}))
        assert info.family == "sdxl"

    @pytest.mark.parametrize("marker,kind", [("hada_w1_a", "loha"), ("lokr_w1", "lokr")])
    def test_lycoris_variants_are_refused_by_name(self, tmp_path, marker, kind):
        keys = {"lora_unet_down_blocks_0_attentions_0_proj_in." + marker: (4, 4)}
        info = lora.inspect_lora(_save(tmp_path / "x.safetensors", keys))
        assert info.kind == kind
        with pytest.raises(ValueError, match=kind.upper()):
            lora.check_lora_compatible(info)

    def test_dora_is_refused(self, tmp_path):
        keys = dict(_SD1)
        keys["lora_unet_down_blocks_0_attentions_0_proj_in.dora_scale"] = (1, 4)
        with pytest.raises(ValueError, match="DoRA"):
            lora.check_lora_compatible(lora.inspect_lora(_save(tmp_path / "x.safetensors", keys)))

    def test_diffusers_format_file_is_refused_with_a_reason(self, tmp_path):
        keys = {"unet.down_blocks.0.attentions.0.to_q.lora_A.weight": (4, 8)}
        with pytest.raises(ValueError, match="no kohya-format"):
            lora.check_lora_compatible(lora.inspect_lora(_save(tmp_path / "x.safetensors", keys)))

    def test_conv_targets_are_flagged(self, tmp_path):
        keys = dict(_SD1)
        keys["lora_unet_down_blocks_0_resnets_0_conv1.lora_down.weight"] = (4, 8, 3, 3)
        assert lora.inspect_lora(_save(tmp_path / "x.safetensors", keys)).has_conv

    def test_trigger_words_come_from_tag_frequency_most_common_first(self, tmp_path):
        meta = {"ss_tag_frequency": json.dumps({"d1": {"pixel art": 30, "castle": 5}, "d2": {"castle": 9, "sky": 1}})}
        info = lora.inspect_lora(_save(tmp_path / "x.safetensors", _SD1, meta))
        assert info.trigger_words[:3] == ("pixel art", "castle", "sky")

    def test_malformed_tag_metadata_is_ignored(self, tmp_path):
        info = lora.inspect_lora(_save(tmp_path / "x.safetensors", _SD1, {"ss_tag_frequency": "{not json"}))
        assert info.trigger_words == ()

    def test_no_tensor_is_loaded(self, tmp_path, monkeypatch):
        # The point of header-only inspection: it must not read tensor data.
        import safetensors

        path = _save(tmp_path / "x.safetensors", _SD1)
        orig = safetensors.safe_open

        class Spy:
            def __init__(self, *a, **k):
                self.f = orig(*a, **k)

            def __enter__(self):
                self.f.__enter__()
                return self

            def __exit__(self, *a):
                return self.f.__exit__(*a)

            def get_tensor(self, *a):
                raise AssertionError("inspect_lora read a tensor")

            def __getattr__(self, n):
                return getattr(self.f, n)

        monkeypatch.setattr(safetensors, "safe_open", Spy)
        lora.inspect_lora(path)


class TestResolve:
    def test_local_file(self, tmp_path):
        f = tmp_path / "a.safetensors"
        f.write_bytes(b"")
        assert lora._resolve_file(LoraSpec(str(f))) == str(f)

    def test_tilde_path_is_expanded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "a.safetensors").write_bytes(b"")
        assert lora._resolve_file(LoraSpec("~/a.safetensors")) == str(tmp_path / "a.safetensors")

    def test_tilde_directory_with_weight_name(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "d").mkdir()
        (tmp_path / "d" / "w.safetensors").write_bytes(b"")
        got = lora._resolve_file(LoraSpec("~/d", "w.safetensors"))
        assert got == str(tmp_path / "d" / "w.safetensors")

    def test_missing_tilde_path_is_file_not_found(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        with pytest.raises(FileNotFoundError):
            lora._resolve_file(LoraSpec("~/nope.safetensors"))

    def test_hub_repo_without_file_name_is_refused(self):
        with pytest.raises(ValueError, match=".safetensors"):
            lora._resolve_file(LoraSpec("org/repo"))

    def test_non_safetensors_is_refused(self):
        with pytest.raises(ValueError, match=".safetensors"):
            lora._resolve_file(LoraSpec("org/repo", "w.ckpt"))

    def test_mistyped_local_path_is_a_file_not_found_not_a_hub_lookup(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            lora._resolve_file(LoraSpec(str(tmp_path / "nope.safetensors")))


class _FakeLoraLayer(torch.nn.Module):
    """Stands in for a peft LoRA-wrapped Linear: holds the base layer plus an adapter."""

    def __init__(self, base, adapter, up_down):
        super().__init__()
        self.base_layer = base
        self.lora_A = {adapter: up_down}  # only the key set is read by merge_loras


class _FakeUnet(torch.nn.Module):
    """Tiny UNet with the three calls merge_loras makes: set_adapters, fuse_lora, unload_lora.

    ``deltas`` maps a layer name to the matrix the fuse adds, per unit scale. The stubbed
    loader (see ``_stub_loader``) wraps those layers the way load_lora_into_unet would.
    """

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(4, 4, bias=False)
        self.b = torch.nn.Linear(4, 4, bias=False)
        self.untouched = torch.nn.Linear(4, 4, bias=False)
        self.pending = {}
        self.wrapped = {}
        self.scales = None
        self.log = []

    def set_adapters(self, names, scales):
        self.log.append("set")
        self.scales = dict(zip(names, scales))

    def fuse_lora(self):
        self.log.append("fuse")
        (name, scale), = self.scales.items()
        with torch.no_grad():
            for layer, delta in self.pending.items():
                getattr(self, layer).weight += scale * delta

    def unload_lora(self):
        self.log.append("unload")
        for layer in list(self.wrapped):
            setattr(self, f"_wrapper_{layer}", None)
        self.wrapped.clear()
        self.pending = {}


def _stub_loader(monkeypatch, unet_deltas):
    """Replace the diffusers loader so merge_loras runs without a real kohya file."""
    from diffusers.loaders import StableDiffusionLoraLoaderMixin as Loader

    monkeypatch.setattr(Loader, "lora_state_dict", staticmethod(lambda path: ({}, {})))

    def load_into_unet(state, alphas, unet, adapter_name):
        unet.pending = dict(unet_deltas)
        for layer in unet_deltas:
            wrapper = _FakeLoraLayer(getattr(unet, layer), adapter_name, None)
            unet.wrapped[layer] = wrapper
            setattr(unet, f"_wrapper_{layer}", wrapper)

    monkeypatch.setattr(Loader, "load_lora_into_unet", staticmethod(load_into_unet))


@pytest.fixture
def lora_file(tmp_path):
    # A real (tiny) SD 1.x kohya header: merge_loras inspects it before loading anything.
    return _save(tmp_path / "x.safetensors", _SD1)


class TestMergeLoras:
    """merge_loras' UNet path, run against a fake UNet and a stubbed diffusers loader."""

    def test_fuse_adds_scaled_delta_and_cleans_up(self, monkeypatch, lora_file):
        unet = _FakeUnet()
        delta = torch.ones(4, 4)
        _stub_loader(monkeypatch, {"a": delta})
        before_a = unet.a.weight.detach().clone()
        before_other = unet.untouched.weight.detach().clone()
        res = lora.merge_loras(unet, None, [LoraSpec(lora_file, scale=0.5)])
        assert torch.allclose(unet.a.weight, before_a + 0.5 * delta)
        assert torch.equal(unet.untouched.weight, before_other)
        assert unet.log == ["set", "fuse", "unload"]
        assert unet.scales == {"tt_lora_0": 0.5}
        assert not unet.wrapped  # adapter layers removed again
        assert [(c, t) for _, c, t in res] == [(1, 0)]

    def test_counts_only_layers_whose_weight_changed(self, monkeypatch, lora_file):
        unet = _FakeUnet()
        # "b" is targeted but its delta is zero, so it is not counted as changed.
        _stub_loader(monkeypatch, {"a": torch.ones(4, 4), "b": torch.zeros(4, 4)})
        (_, changed, _), = lora.merge_loras(unet, None, [LoraSpec(lora_file)])
        assert changed == 1

    def test_stacking_adds_both_deltas_in_order(self, monkeypatch, lora_file):
        unet = _FakeUnet()
        delta = torch.ones(4, 4)
        _stub_loader(monkeypatch, {"a": delta})
        before = unet.a.weight.detach().clone()
        res = lora.merge_loras(unet, None, [LoraSpec(lora_file, scale=1.0), LoraSpec(lora_file, scale=0.25)])
        assert torch.allclose(unet.a.weight, before + 1.25 * delta)
        assert unet.log == ["set", "fuse", "unload"] * 2
        assert len(res) == 2

    def test_incompatible_file_is_refused_before_the_loader_runs(self, monkeypatch, tmp_path):
        from diffusers.loaders import StableDiffusionLoraLoaderMixin as Loader

        called = []
        monkeypatch.setattr(Loader, "lora_state_dict", staticmethod(lambda p: called.append(p) or ({}, {})))
        keys = dict(_SD1)
        keys["lora_te2_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight"] = (4, 8)
        f = _save(tmp_path / "xl.safetensors", keys)
        with pytest.raises(ValueError, match="SDXL"):
            lora.merge_loras(_FakeUnet(), None, [LoraSpec(f)])
        assert called == []

    def test_zero_match_raises(self, monkeypatch, lora_file):
        # A LoRA for another base model matches nothing: it must not look like success.
        unet = _FakeUnet()
        _stub_loader(monkeypatch, {})
        with pytest.raises(ValueError, match="changed no weight"):
            lora.merge_loras(unet, None, [LoraSpec(lora_file)])

    def test_scale_zero_with_no_change_is_allowed(self, monkeypatch, lora_file):
        unet = _FakeUnet()
        _stub_loader(monkeypatch, {"a": torch.ones(4, 4)})
        (_, changed, _), = lora.merge_loras(unet, None, [LoraSpec(lora_file, scale=0.0)])
        assert changed == 0

    def test_merge_does_not_snapshot_the_whole_unet(self, monkeypatch, lora_file):
        # Guards the review finding: state_dict() was cloned in full before every LoRA.
        unet = _FakeUnet()
        _stub_loader(monkeypatch, {"a": torch.ones(4, 4)})
        calls = []
        orig = unet.state_dict
        monkeypatch.setattr(unet, "state_dict", lambda *a, **k: calls.append(1) or orig(*a, **k))
        lora.merge_loras(unet, None, [LoraSpec(lora_file)])
        assert calls == []


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
