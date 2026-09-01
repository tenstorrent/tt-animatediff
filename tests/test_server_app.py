# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""The ASGI app, tested on CPU with no card and no tt-metal.

Every assertion here is one that a wrong answer makes expensive on hardware:

* importing the module must not touch ttnn -- tt-model-manager's ``verify_lines`` imports
  the ASGI attribute at IMAGE BUILD time, on a machine with no device, purely to prove the
  code allowlist shipped the server. A module-scope ``import ttnn`` turns that check into a
  build failure, and a stray one is invisible until then;
* the mesh shape must come from the environment or fail loudly -- converting against the
  wrong mesh produces bad frames, not an error;
* the readiness contract must hold: no device work outside the lifespan, because the
  supervisor calls the server ready the moment uvicorn logs "Application startup complete".

The lifespan is deliberately never entered: ``TestClient(app)`` as a plain object does not
run it, and entering it would open a device.
"""
import subprocess
import sys
import textwrap

import pytest
from fastapi.testclient import TestClient

from animatediff_ttnn.server.app import (
    MESH_SHAPE_ENV,
    VideoGenerationRequest,
    app,
    mesh_shape_from_env,
)


def test_importing_the_server_does_not_import_ttnn():
    """THE BUILD-TIME PROPERTY. Run in a subprocess so this process's own imports -- which
    may include ttnn from another test -- cannot make it pass for the wrong reason."""
    code = textwrap.dedent(
        """
        import sys
        import animatediff_ttnn.server.app  # noqa: F401
        bad = sorted(m for m in sys.modules if m == "ttnn" or m.startswith("ttnn."))
        print(",".join(bad))
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "", f"importing the server pulled in: {out.stdout.strip()}"


def test_the_asgi_attribute_is_what_uvicorn_will_load():
    """`runtime.app: animatediff_ttnn.server.app:app` names this attribute; uvicorn calls
    it with (scope, receive, send)."""
    assert callable(app)


def test_the_routes_are_the_ones_the_manifest_promises():
    paths = {r.path for r in app.routes}
    assert {"/health", "/tt-liveness", "/v1/models",
            "/v1/videos/generations"} <= paths


# ---- mesh shape ------------------------------------------------------------------------


def test_mesh_shape_defaults_to_one_chip_when_unset():
    assert mesh_shape_from_env({}) == (1, 1)


@pytest.mark.parametrize("raw,expected", [("1x4", (1, 4)), ("2x2", (2, 2)), ("1X1", (1, 1))])
def test_mesh_shape_is_read_from_the_environment(raw, expected):
    assert mesh_shape_from_env({MESH_SHAPE_ENV: raw}) == expected


@pytest.mark.parametrize("raw", ["four", "1x", "0x4", "-1x2", "1,4"])
def test_a_malformed_mesh_shape_raises_rather_than_defaulting(raw):
    """Falling back to (1,1) would convert against the wrong mesh and report success."""
    with pytest.raises(ValueError):
        mesh_shape_from_env({MESH_SHAPE_ENV: raw})


# ---- the request contract ---------------------------------------------------------------


def test_prompt_is_required_and_may_not_be_empty():
    with pytest.raises(Exception):
        VideoGenerationRequest(prompt="")


def test_defaults_match_the_pipeline_rather_than_being_invented():
    r = VideoGenerationRequest(prompt="a cat")
    assert (r.num_frames, r.num_inference_steps, r.guidance_scale, r.seed) == (16, 25, 7.5, 42)


@pytest.mark.parametrize("field,value", [
    ("num_frames", 0), ("num_frames", 65), ("num_inference_steps", 0),
    ("guidance_scale", -1.0), ("temporal_alpha", 1.5), ("height", 32),
])
def test_out_of_range_parameters_are_refused_at_the_edge(field, value):
    """A device-side failure deep in a denoise loop is far more expensive to diagnose than
    a 422 from the request model."""
    with pytest.raises(Exception):
        VideoGenerationRequest(prompt="a cat", **{field: value})


# ---- behaviour before the lifespan has run -----------------------------------------------


def test_health_is_a_READINESS_probe_and_refuses_traffic_before_the_model_is_warm():
    """THE CORRECTION. This returned 200 {"status": "starting"} until the semantics were
    checked against tt-inference-server's tt-media-server.

    Despite the names, /health is the READINESS probe (gate traffic on it) and
    /tt-liveness is the LIVENESS probe (restart on it). A readiness probe answering 200
    while the pipeline is still loading tells a load balancer to route work to a server
    that cannot do it -- reintroducing, one layer up, the exact failure the lifespan
    design exists to prevent."""
    r = TestClient(app).get("/health")
    assert r.status_code == 503


def test_health_body_is_vllm_compatible_when_ready():
    """tt-media-server returns an empty dict when healthy, matching vLLM. Clients written
    against one should not need a branch for the other."""
    app.state.engine = {"hf_model": "x", "mesh_device": "P150", "mesh_shape": (1, 1)}
    try:
        r = TestClient(app).get("/health")
        assert r.status_code == 200 and r.json() == {}
    finally:
        del app.state.engine


def test_tt_liveness_reports_alive_while_the_model_is_still_warming():
    """A warming model is ALIVE, not broken. Reporting it as failure would have an
    orchestrator restart a server that is loading correctly -- which is the whole reason
    liveness and readiness are separate probes."""
    r = TestClient(app).get("/tt-liveness")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "alive"
    assert body["model_ready"] is False


def test_tt_liveness_carries_the_model_payload_once_warm():
    """Matches tt-media-server's {"status": "alive", **check_is_model_ready()}."""
    app.state.engine = {"hf_model": "guoyww/animatediff", "mesh_device": "P150",
                        "mesh_shape": (1, 1)}
    try:
        body = TestClient(app).get("/tt-liveness").json()
        assert body == {"status": "alive", "model_ready": True,
                        "model": "guoyww/animatediff", "mesh_device": "P150",
                        "mesh_shape": "1x1"}
    finally:
        del app.state.engine


def test_the_two_probes_cannot_disagree_about_readiness():
    """Both read one _readiness(). Two hand-maintained copies drift, and a liveness probe
    that says ready while readiness says no is worse than either alone."""
    from animatediff_ttnn.server.app import _readiness

    c = TestClient(app)
    assert _readiness()["model_ready"] is False
    assert c.get("/tt-liveness").json()["model_ready"] is False
    assert c.get("/health").status_code == 503

    app.state.engine = {"hf_model": "x", "mesh_device": "P150", "mesh_shape": (1, 1)}
    try:
        assert _readiness()["model_ready"] is True
        assert c.get("/tt-liveness").json()["model_ready"] is True
        assert c.get("/health").status_code == 200
    finally:
        del app.state.engine


def test_generation_is_refused_while_still_starting():
    """503, not a crash and not a queue: the supervisor should see an honest 'not yet'."""
    r = TestClient(app).post("/v1/videos/generations", json={"prompt": "a cat"})
    assert r.status_code == 503


def test_an_unsupported_response_format_is_refused():
    """The server has no file store, so a URL response would point at nothing."""
    app.state.engine = {"hf_model": "x", "mesh_shape": (1, 1)}
    try:
        r = TestClient(app).post(
            "/v1/videos/generations",
            json={"prompt": "a cat", "response_format": "url"},
        )
        assert r.status_code == 400
        assert "b64_json" in r.json()["detail"]
    finally:
        del app.state.engine


# ---- the manifest and the server must agree ----------------------------------------------


def _manifest() -> dict:
    import yaml
    from pathlib import Path

    return yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "tt_model_package.yaml").read_text()
    )


def test_the_manifest_points_at_the_asgi_attribute_that_exists():
    """`runtime.app` is resolved by uvicorn, in the image, long after anyone would notice a
    typo here. Split it and import it the same way."""
    import importlib

    module, attr = _manifest()["runtime"]["app"].split(":", 1)
    assert getattr(importlib.import_module(module), attr) is app


def test_the_manifest_ships_the_package_that_holds_the_server():
    """tt-dit-server refuses an app whose top-level package no allowlist entry ships. That
    check runs at package time; this one runs now."""
    m = _manifest()
    top = m["runtime"]["app"].split(":", 1)[0].split(".")[0]
    shipped = list(m["source"]["code"])
    for extra in m["source"].get("extra_code", []):
        shipped += extra["paths"]
    assert any(p.split("/")[0] == top for p in shipped), (
        f"{top!r} is shipped by neither source.code nor source.extra_code"
    )


def test_the_manifest_mesh_shape_env_matches_the_name_the_server_reads():
    """THE SILENT DRIFT. The launcher exports whatever `mesh_shape_env` names; the server
    reads MESH_SHAPE_ENV. If they diverge the server sees nothing, silently falls back to a
    single chip, and returns frames converted against the wrong mesh -- with no error
    anywhere. Renaming either constant must break this test."""
    assert _manifest()["runtime"]["mesh_shape_env"] == MESH_SHAPE_ENV


def test_the_manifest_declares_the_diffusion_kind():
    """A vLLM kind would demand max_num_seqs and block_size, which mean nothing here."""
    assert _manifest()["kind"] == "tt-dit-server"


def test_the_manifest_declares_a_single_chip_because_the_model_is_single_device():
    """Found by serving, not by review.

    The manifest first declared P300x2 (four chips). The server started, opened the mesh
    and loaded the weights, then died in the lifespan with:

        TT_FATAL: Can't convert a tensor distributed on MeshShape([1, 4]) mesh to
        row-major logical tensor. Supply a mesh composer to concatenate multi-device shards.

    animatediff_ttnn's working path hardcodes ``device_ids=[0]`` and mesh frame-sharding is
    still a plan document. A four-chip declaration is therefore a promise the model cannot
    keep, and it fails at serve time on someone else's machine rather than here.
    """
    mesh = _manifest()["serve"]["mesh_device"]
    single = {"N150", "P100", "P150"}
    assert mesh in single, (
        f"mesh_device={mesh!r} claims more than one chip; animatediff_ttnn is single-device "
        f"until docs/superpowers/plans/2026-06-15-mesh-frame-sharding.md lands"
    )
