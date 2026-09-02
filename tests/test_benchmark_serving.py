# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for scripts/benchmark_serving.py — the HTTP benchmark harness.

The harness itself can only be exercised against a live server on hardware, so
what is testable here is the part that silently rots: the shape of what it
writes. docs/measurements/serving-benchmark.json is a committed sample of that
output, annotated by hand with the hardware, the fit and the limitations. Once
the numbers and the schema live in two places, a key can be renamed in one and
not the other -- which had already happened (``cold_first_call_s`` against the
script's ``first_call_at_this_shape_s``), leaving the file's own note referring
to a key the file did not contain.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import benchmark_serving as bs

SAMPLE = (Path(__file__).resolve().parent.parent
          / "docs" / "measurements" / "serving-benchmark.json")


def _row_from_the_script():
    """One sweep row, with the HTTP call stubbed out to a fixed latency."""
    with patch.object(bs, "generate",
                      side_effect=lambda url, *, frames, steps, seed=0: {
                          "seconds": 5.0 + seed * 0.1, "frames": frames,
                          "steps": steps, "b64_len": 10}):
        rows = bs.sweep("http://127.0.0.1:8000", frames=8, step_values=[4], repeats=3)
    assert len(rows) == 1
    return rows[0]


def test_committed_sample_rows_use_the_scripts_own_key_names():
    """A fresh run must be diffable against the committed sample, key for key."""
    expected = list(_row_from_the_script())
    sample = json.loads(SAMPLE.read_text())

    for i, row in enumerate(sample["rows"]):
        assert list(row) == expected, (
            f"row {i} of {SAMPLE.name} does not match the schema "
            f"sweep() emits: {list(row)} != {expected}"
        )


def test_the_samples_own_notes_reference_keys_the_sample_contains():
    """The annotations explain specific fields; they must name real ones."""
    sample = json.loads(SAMPLE.read_text())
    row_keys = set(sample["rows"][0])

    assert "s_per_step_naive" in sample["fit"]["note"], (
        "the fit note exists to warn about the naive per-step number"
    )
    assert "s_per_step_naive" in row_keys


# ---------------------------------------------------------------------------
# --base-url validation
# ---------------------------------------------------------------------------

def test_base_url_keeps_a_valid_address_and_drops_a_trailing_slash():
    assert bs.validated_base_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000"
    assert bs.validated_base_url("https://box.local:8000/") == "https://box.local:8000"


@pytest.mark.parametrize("bad, reason", [
    ("127.0.0.1:8000", "no scheme -- requests would read this as a relative URL"),
    ("file:///etc/passwd", "not an HTTP address"),
    ("http://", "no host"),
    ("", "empty"),
])
def test_base_url_rejects_a_malformed_value(bad, reason):
    with pytest.raises(SystemExit) as exc:
        bs.validated_base_url(bad)
    assert "--base-url" in str(exc.value), reason
