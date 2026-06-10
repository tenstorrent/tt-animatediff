# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for parallel Blackhole validation script."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from scripts.validate_parallel import (
    build_validation_configs,
    format_benchmark_table,
)


def test_build_validation_configs_returns_four_entries():
    configs = build_validation_configs(
        unet_4step=Path("weights/unet_lcm_4step.pt"),
        unet_8step=Path("weights/unet_lcm_8step.pt"),
        adapter_4step=Path("weights/motion_adapter_lcm_4step.pt"),
        adapter_8step=Path("weights/motion_adapter_lcm_8step.pt"),
    )
    assert len(configs) == 4


def test_build_validation_configs_chip_ids_are_unique():
    configs = build_validation_configs(
        unet_4step=Path("w/u4.pt"),
        unet_8step=Path("w/u8.pt"),
        adapter_4step=Path("w/a4.pt"),
        adapter_8step=Path("w/a8.pt"),
    )
    chip_ids = [c["chip_id"] for c in configs]
    assert len(set(chip_ids)) == 4


def test_build_validation_configs_contains_expected_labels():
    configs = build_validation_configs(
        unet_4step=Path("w/u4.pt"),
        unet_8step=Path("w/u8.pt"),
        adapter_4step=Path("w/a4.pt"),
        adapter_8step=Path("w/a8.pt"),
    )
    labels = {c["label"] for c in configs}
    assert "spatial-fast-4step" in labels
    assert "spatial-balanced-8step" in labels
    assert "lightning-8step" in labels
    assert "lightning-4step" in labels


def test_format_benchmark_table_contains_all_labels():
    results = [
        {"label": "spatial-fast-4step",     "elapsed_s": 8.1,  "gif_path": Path("a.gif")},
        {"label": "spatial-balanced-8step", "elapsed_s": 14.3, "gif_path": Path("b.gif")},
        {"label": "lightning-8step",        "elapsed_s": 14.1, "gif_path": Path("c.gif")},
        {"label": "lightning-4step",        "elapsed_s": 8.0,  "gif_path": Path("d.gif")},
    ]
    table = format_benchmark_table(results)
    assert "spatial-fast-4step" in table
    assert "lightning-4step" in table
    assert "8.0" in table or "8.1" in table
