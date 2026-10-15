# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""scripts/benchmark_modes.py decides each LoRA variant's request from the catalog it reads."""

from scripts import benchmark_modes as bm


def _payloads(loras):
    modes, _ = bm.build_modes({"loras": loras})
    return {name: payload for name, payload, _ in modes if name.startswith("LoRA variant")}


def test_no_lora_variant_gets_an_explicit_step_count():
    # Review finding: catalog names are arbitrary, so an LCM variant called "turbo" was sent
    # 25 steps. The server already defaults style variants to 25 and LCM variants to 4, so the
    # benchmark sends neither and measures what a caller who leaves steps out gets.
    got = _payloads(["pixel", "turbo", "pixel-fast", "lcm"])
    assert len(got) == 4
    for name, payload in got.items():
        assert "num_inference_steps" not in payload, name
    assert {p["lora"] for p in got.values()} == {"pixel", "turbo", "pixel-fast", "lcm"}
