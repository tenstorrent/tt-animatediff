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
import asyncio
import os
import subprocess
import sys
import textwrap
from unittest.mock import MagicMock, patch

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
            "/v1/videos/generations", "/v1/videos/preview"} <= paths


# ---- mesh shape ------------------------------------------------------------------------


def test_mesh_shape_defaults_to_one_chip_when_unset():
    assert mesh_shape_from_env({}) == (1, 1)


@pytest.mark.parametrize("raw,expected", [("1x1", (1, 1)), ("1X1", (1, 1))])
def test_mesh_shape_is_read_from_the_environment(raw, expected):
    """Parsing, including the case-insensitive separator.

    This used to parametrize ("1x4", (1, 4)) and ("2x2", (2, 2)) as valid, which
    documented a capability the model does not have -- parsing them was never the
    question, and accepting them let a stray value claim every chip on the box. The
    refusal is asserted in test_a_mesh_shape_the_model_cannot_use_is_refused.
    """
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
    """Derived from the library, not retyped as literals.

    This test used to assert `(16, 25, 7.5, 42)` as bare numbers while its name claimed
    they matched the pipeline. They partly did not -- temporal_alpha was 0.5 against the
    library's 0.35 -- and a literal cannot notice that, so the name was a claim the body
    never checked. Now the shared ones are read from generate_animation's own signature.
    """
    import inspect

    from animatediff_ttnn import generate_animation

    lib = inspect.signature(generate_animation).parameters
    r = VideoGenerationRequest(prompt="a cat")

    assert r.num_inference_steps == lib["num_steps"].default
    assert r.guidance_scale == lib["guidance_scale"].default
    assert r.seed == lib["seed"].default
    assert r.temporal_alpha == lib["temporal_alpha"].default
    assert (r.height, r.width) == (lib["height"].default, lib["width"].default)

    # num_frames is the one that deliberately does NOT track the library: 16 here against
    # generate_animation's 8. It is a serving choice about clip length rather than a
    # calibrated constant, so it is pinned as a literal on purpose -- and pinned, so that
    # changing the API default for every HTTP client stays a deliberate edit.
    assert r.num_frames == 16
    assert lib["num_frames"].default == 8


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


def _models_imports_in_the_package() -> set:
    """Every ``models.*`` module animatediff_ttnn imports, at any nesting.

    Static rather than by importing: these modules import ttnn at module scope, so
    resolving them for real needs a card, and the question here is only which names the
    image's ``models`` tree has to be able to satisfy.
    """
    import re
    from pathlib import Path

    pattern = re.compile(
        r"^\s*(?:from\s+(models\.[A-Za-z0-9_.]*)|import\s+(models\.[A-Za-z0-9_.]*))", re.M
    )
    root = Path(__file__).resolve().parents[1] / "animatediff_ttnn"
    found = set()
    for py in root.rglob("*.py"):
        for a, b in pattern.findall(py.read_text()):
            found.add(a or b)
    return found


def test_the_allowlist_ships_every_tt_metal_module_the_package_imports():
    """THE ONE VERIFY CANNOT CATCH.

    The image deletes tt-metal's own ``models/`` and copies ``source.code`` in as the only
    ``models`` package there is, so a name missing from the allowlist is a
    ModuleNotFoundError inside the container. The kind's verify step is supposed to catch
    that by importing ``runtime.app`` -- but app.py defers every ttnn and model import into
    a function so the module stays importable on a machine with no card. Nothing touches
    ``models.demos`` until the lifespan warms the pipeline, i.e. on a consumer's first boot.

    Found exactly that way: the allowlist shipped ``models/common``, which nothing in this
    package imports directly, and omitted the SD 1.4 demo tree, which every working path
    needs. On this box the import is rescued by ~/tt-metal being on sys.path, so serving it
    here could never have caught it either.
    """
    shipped = [entry.split("/") for entry in _manifest()["source"]["code"]]

    def is_shipped(module: str) -> bool:
        parts = module.split(".")
        return any(parts[: len(e)] == e for e in shipped)

    missing = sorted(m for m in _models_imports_in_the_package() if not is_shipped(m))
    assert not missing, (
        "source.code does not ship these, so the server's lifespan would raise "
        f"ModuleNotFoundError in the image: {missing}"
    )


def test_the_allowlist_does_not_ship_trees_nothing_needs():
    """The allowlist is the image's whole models tree; a stale entry is dead weight a
    reader has to reason about. models/common stays because the SD demo imports it."""
    imported = _models_imports_in_the_package()
    covered = {
        entry
        for entry in _manifest()["source"]["code"]
        if any(m.split(".")[: len(entry.split("/"))] == entry.split("/") for m in imported)
    }
    unused = sorted(set(_manifest()["source"]["code"]) - covered)
    # models/common is imported by the demo tree, not by this package, so it is expected
    # here -- named explicitly rather than silently tolerated.
    assert unused == ["models/common"], (
        f"unexpected allowlist entries no import reaches: {unused}"
    )


def test_the_pinned_extra_code_ref_actually_contains_what_it_ships():
    """A ref that predates the code it is supposed to ship.

    ``extra_code`` pins a ref so the package is reproducible, and staging clones THAT ref
    rather than the working tree. The pin sat at v0.10.0, which predates
    animatediff_ttnn/server entirely -- so a consumer staging this manifest would have got
    a code tree with no ``runtime.app`` in it, and the kind's verify step would have failed
    in their image build with an import error naming a module this repo does have.
    """
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    for extra in _manifest()["source"].get("extra_code", []):
        root = extra["root"]
        if not isinstance(root, dict) or "ref" not in root:
            continue  # a local path root ships the working tree; nothing to pin
        ref = root["ref"]
        if subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", "--quiet",
                           f"{ref}^{{commit}}"], capture_output=True).returncode != 0:
            pytest.skip(
                f"extra_code ref {ref} is not in this clone (shallow checkout); "
                "cannot verify its contents offline"
            )
        for rel in extra["paths"]:
            missing = subprocess.run(
                ["git", "-C", str(repo), "cat-file", "-e", f"{ref}:{rel}"],
                capture_output=True,
            ).returncode != 0
            assert not missing, f"extra_code ref {ref} has no {rel}"
        # The point of the pin is the server, so name it rather than trusting the
        # directory to imply it.
        app_rel = _manifest()["runtime"]["app"].split(":", 1)[0].replace(".", "/") + ".py"
        assert subprocess.run(
            ["git", "-C", str(repo), "cat-file", "-e", f"{ref}:{app_rel}"],
            capture_output=True,
        ).returncode == 0, (
            f"extra_code ref {ref} does not contain {app_rel}, which runtime.app names"
        )


def test_the_pinned_extra_code_ref_ships_the_package_we_are_developing():
    """The pin must match what MAIN ships, not what this branch is proposing.

    Compared against main deliberately, and this was wrong once in the other direction.
    Comparing against HEAD failed every PR that touched animatediff_ttnn/ -- which is most
    of them -- because a pin can only name a commit that already exists, so it cannot
    possibly name the tree a PR is still proposing. The consequence was a guard that cried
    wolf on healthy branches and had to be argued with.

    Against main it says the thing that actually matters: the ref a CONSUMER stages is the
    package main released. A PR in flight is not released, so it passes; once it merges and
    main's tree moves, this goes red until someone repins -- which is exactly the prompt
    docs/RELEASING.md step 5 describes, arriving at the moment it is actionable.
    """
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]

    def rev(spec):
        r = subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", spec],
                           capture_output=True, text=True)
        return r.stdout.strip() or None

    main = next((rev(f"{n}^{{commit}}") for n in ("origin/main", "main") if rev(f"{n}^{{commit}}")), None)
    if not main:
        pytest.skip("no main ref in this clone; cannot compare the pin against what main ships")

    for extra in _manifest()["source"].get("extra_code", []):
        root = extra["root"]
        if not isinstance(root, dict) or "ref" not in root:
            continue
        ref = root["ref"]
        if not rev(f"{ref}^{{commit}}"):
            pytest.skip(
                f"extra_code ref {ref} is not in this clone (shallow checkout); "
                "cannot compare its tree offline"
            )
        for rel in extra["paths"]:
            pinned, shipped = rev(f"{ref}:{rel}"), rev(f"{main}:{rel}")
            assert pinned and pinned == shipped, (
                f"extra_code ref {ref} ships {rel} at tree {pinned or '<missing>'}, but main "
                f"has {shipped}. Repin -- consumers stage an older package than main released."
            )

def test_the_server_request_default_matches_the_librarys_calibrated_one():
    """One default, three places, and the server had picked its own number.

    `temporal_alpha` is the cross-frame blend weight. generate_animation() and
    hf/pipeline.py both default it to 0.35, and so does generate_frames_temporal --
    the function this server actually calls. The server's request model said 0.5, which
    matched none of them, so an HTTP client omitting the field got quietly different
    motion from every other entry point into the same code.

    Asserted against the library rather than against a literal, so the two cannot drift
    apart again in either direction.
    """
    import inspect

    from animatediff_ttnn import generate_animation
    from animatediff_ttnn.temporal_attention import generate_frames_temporal

    library = inspect.signature(generate_animation).parameters["temporal_alpha"].default
    called = inspect.signature(generate_frames_temporal).parameters["temporal_alpha"].default
    served = VideoGenerationRequest.model_fields["temporal_alpha"].default

    assert library == called, (
        f"the library API ({library}) and the function it calls ({called}) already disagree"
    )
    assert served == library, (
        f"server default temporal_alpha={served} but the library calibrated {library}; "
        "an HTTP client omitting the field would get different motion"
    )


# ---- the shapes the compiled kernel can actually serve -------------------------------


@pytest.mark.parametrize("field,value", [
    ("height", 768), ("width", 768), ("height", 500), ("width", 64), ("height", 1024),
])
def test_a_resolution_the_kernel_cannot_serve_is_refused_at_the_edge(field, value):
    """Refused by validation, not by a TT_FATAL mid-denoise.

    The TTNN UNet is compiled once at a fixed 64x64 latent
    (generation_helpers.py: `UNet2D(device, parameters, 2, 64, 64)`), so this server can
    only produce 512x512. height/width were merely bounded 64..1024, which is
    pydantic-valid and kernel-invalid: the request passed validation, took the device
    lock, and then raised deep inside the kernel as an uncaught 500 -- having held the
    one device for the duration.
    """
    with pytest.raises(Exception) as exc:
        VideoGenerationRequest(prompt="a cat", **{field: value})
    assert "512" in str(exc.value), f"the refusal should name the only supported size: {exc.value}"


def test_the_supported_resolution_is_still_accepted():
    r = VideoGenerationRequest(prompt="a cat", height=512, width=512)
    assert (r.height, r.width) == (512, 512)


def test_the_gif_encode_does_not_run_on_the_event_loop():
    """The probes are only lock-free if nothing else blocks the loop.

    _frames_to_gif_b64 does PIL GIF palette quantization plus base64 of a multi-MB
    payload. It ran inline in the endpoint coroutine -- outside the device lock, but ON
    the event loop -- so for its duration no coroutine could run, including /health and
    /tt-liveness. That contradicts this module's own docstring and is exactly the failure
    the lock-free probes exist to prevent: an orchestrator with a 1-2 s liveness timeout
    restarts a server that is merely busy encoding.

    Asserted by recording which thread the encode runs on: awaited through to_thread it
    lands on an anyio worker; called inline it would run on the loop's own thread, the
    same one serving every other request.
    """
    import threading

    from animatediff_ttnn.server import app as mod

    seen = {}

    def fake_encode(frames):
        seen["encode_thread"] = threading.current_thread().name
        seen["encode_ident"] = threading.get_ident()
        return "Zm9v"

    class _ClockOnTheLoop:
        """time.time() is called in the response construction, i.e. ON the event loop
        after both awaits -- so it is a reliable way to capture the loop's own thread
        from inside the endpoint, which is what the encode must NOT share."""

        @staticmethod
        def time():
            seen["loop_ident"] = threading.get_ident()
            return 1234.0

    # Plain TestClient(app), per this module's convention -- entering it as a context
    # manager would run the real lifespan and try to open a device. The lifespan is also
    # where device_lock is created, so this test supplies one.
    app.state.engine = {"device": object(), "models": object(), "ready": True,
                        "model": "guoyww/animatediff", "mesh_device": "P150",
                        "mesh_shape": "1x1"}
    app.state.device_lock = asyncio.Lock()
    try:
        with patch.object(mod, "_frames_to_gif_b64", fake_encode), \
             patch.object(mod, "time", _ClockOnTheLoop), \
             patch.object(mod, "_generate", return_value=[object()]):
            r = TestClient(app).post("/v1/videos/generations", json={"prompt": "a cat"})
        assert r.status_code == 200, r.text
        assert seen.get("encode_ident"), "the encode never ran"
        assert seen.get("loop_ident"), "never captured the loop thread"
        assert seen["encode_ident"] != seen["loop_ident"], (
            f"the encode ran on the event loop's own thread ({seen['encode_thread']!r}), "
            "so /health and /tt-liveness were blocked for its duration"
        )
        assert seen["encode_ident"] != threading.get_ident(), (
            "the encode ran on the thread that issued the request"
        )
    finally:
        del app.state.engine


def test_the_lifespan_warms_the_text_encoder_too():
    """Readiness is a claim about ALL the models, not just the ones on the device.

    load_sd14_ttnn brings up the UNet and VAE. CLIP's tokenizer and text encoder were
    downloaded lazily by the first encode_prompt() -- inside _generate, under the device
    lock, after /health had answered 200. A container with only unet/vae cached and no
    egress reported "Application startup complete" and then failed every request, which
    is precisely the failure the lifespan design exists to prevent.
    """
    from animatediff_ttnn.server import app as mod

    with patch("animatediff_ttnn.ttnn_pipeline.setup_blackhole", return_value="dev") as setup, \
         patch("animatediff_ttnn.generation_helpers.load_sd14_ttnn",
               return_value=("unet", "vae", "cfg", "proj")) as load, \
         patch("animatediff_ttnn.generation_helpers.encode_prompt") as encode:
        device, models = mod._open_device_and_models((1, 1))

    setup.assert_called_once_with(device_ids=[0])
    load.assert_called_once_with("dev")
    assert encode.called, (
        "the text encoder was not warmed in the lifespan, so its first download happens "
        "under the device lock after /health has already returned 200"
    )
    assert device == "dev" and models == ("unet", "vae", "cfg", "proj")


# ---- the image's dependency set ------------------------------------------------------


def _lock_pins() -> dict:
    """The lock's `name==version` pins, ignoring uv's `# via` annotations."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    name = _manifest()["runtime"]["lock"]
    pins = {}
    for line in (root / name).read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#") and "==" in s:
            pkg, _, ver = s.partition("==")
            pins[pkg.strip().lower().replace("_", "-")] = ver.strip()
    return pins


def test_the_manifests_lock_exists_and_is_the_one_it_names():
    """`stage()` copies `manifest_path.parent / runtime.lock`, and RAISES if it is absent
    -- so a renamed or deleted lock is a failed package run, not a fallback to ranges."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    name = _manifest()["runtime"]["lock"]
    assert (root / name).is_file(), f"runtime.lock names {name}, which does not exist"
    assert _lock_pins(), f"{name} contains no pins"


def test_the_lock_pins_the_torch_the_image_verifies():
    """The kind injects tt-metal's torch pin ONLY when there is no lock; with one, the
    lock is the whole install. Its verify step then asserts torch is exactly tt-metal's
    version and a +cpu build, so getting this wrong fails the image build."""
    torch = _lock_pins().get("torch")
    assert torch, "the lock does not pin torch at all; the kind will not inject it either"
    assert torch.endswith("+cpu"), f"torch=={torch} is not a CPU build"
    assert torch.split("+")[0] == "2.11.0", (
        f"torch=={torch} but tt-metal v0.77.0 (source.tt_metal.ref) pins 2.11.0, which is "
        "what the kind's verify step asserts"
    )


def test_the_lock_carries_no_cuda_wheels():
    """This is a CPU-only serving image; the accelerators are Tenstorrent.

    Measured reason this is a test and not a preference: resolving the manifest's ranges
    WITHOUT the lock pulls torch 2.14.0 (accelerate depends on torch and nothing bounds
    it) and 19 CUDA/NVIDIA wheels -- gigabytes of CUDA runtime in an image that will never
    see an NVIDIA device. Compiling against the PyTorch CPU index is what avoids it.
    """
    cuda = sorted(p for p in _lock_pins()
                  if p.startswith(("nvidia-", "cuda-")) or p == "triton")
    assert not cuda, f"CUDA wheels in a CPU-only image: {cuda}"


def test_the_lock_covers_the_http_stack_the_kind_would_otherwise_add():
    """With runtime.lock set, the kind's DEFAULT_PACKAGES are never installed -- so
    anything the server imports has to be in the lock. A missing fastapi is an image that
    builds and then cannot import its own ASGI app."""
    pins = _lock_pins()
    for required in ("fastapi", "uvicorn", "pydantic", "pillow",
                     "diffusers", "transformers", "safetensors"):
        assert required in pins, f"{required} is not pinned in the lock"


# ---- follow-ups from the #9 approval -------------------------------------------------


@pytest.mark.parametrize("raw", ["1x4", "2x2", "1x2", "4x1"])
def test_a_mesh_shape_the_model_cannot_use_is_refused(raw):
    """Refuse it, rather than claim every chip on the box and fail later.

    `_open_device_and_models` passes `device_ids=None` for any shape whose product is not
    1, and setup_blackhole(None) opens EVERY available chip as a 1xN mesh. The model
    cannot shard -- that is why the manifest declares P150, proven by a serve that died
    with `Can't convert a tensor distributed on MeshShape([1, 4])` -- so a 1x4 value takes
    four chips out of circulation on a shared box and then fails in the lifespan anyway.
    """
    with pytest.raises(ValueError) as exc:
        mesh_shape_from_env({MESH_SHAPE_ENV: raw})
    assert "1x1" in str(exc.value)
    assert "shard" in str(exc.value).lower()


@pytest.mark.parametrize("raw,expected", [("", (1, 1)), ("1x1", (1, 1))])
def test_the_one_shape_the_model_can_use_is_still_accepted(raw, expected):
    assert mesh_shape_from_env({MESH_SHAPE_ENV: raw} if raw else {}) == expected


def test_a_failed_warm_closes_the_device_the_lifespan_opened():
    """Same leak that was fixed in session.py, in the server's own open path.

    setup_blackhole succeeds, then load_sd14_ttnn or the warm-up encode raises. Nothing
    closed the chip, so a container that fails to start keeps a Tenstorrent device claimed
    -- and on a restart loop it claims one per attempt while TTNN refuses to reopen it.
    """
    from animatediff_ttnn.server import app as mod

    ttnn_mock = MagicMock()
    with patch.dict(sys.modules, {"ttnn": ttnn_mock}), \
         patch("animatediff_ttnn.ttnn_pipeline.setup_blackhole", return_value="dev"), \
         patch("animatediff_ttnn.generation_helpers.load_sd14_ttnn",
               side_effect=RuntimeError("L1 allocation failed")):
        with pytest.raises(RuntimeError, match="L1 allocation failed"):
            mod._open_device_and_models((1, 1))

    ttnn_mock.close_mesh_device.assert_called_once_with("dev")


def test_the_manifest_declares_the_weights_the_served_path_actually_loads():
    """HF_MODEL comes from `weights:`, and /v1/models reports it.

    It declared `guoyww/animatediff`, which is a repo of raw mm_sd_v*.ckpt motion modules
    -- not diffusers format, and not what this server loads. The served path is Phase 2.5
    (generate_frames_temporal): SD 1.4's UNet, VAE and tokenizer, and NO motion adapter,
    because the cross-frame motion comes from temporal_alpha. So the declaration was wrong
    in both directions: a pre-caching deploy would fetch gigabytes it never uses and still
    miss the repo it needs, and /v1/models named a model that was never loaded.

    Since the pinned-revision change, generation_helpers.py names the repo through
    ``weights_pins.SD14_REPO`` instead of a string literal. So this checks both halves:
    the declaration equals that constant, and every SD 1.4 load in generation_helpers
    goes through it (no stray literal naming some other repo).
    """
    from pathlib import Path

    from animatediff_ttnn.weights_pins import SD14_REPO

    root = Path(__file__).resolve().parents[1]
    declared = _manifest()["weights"]
    loader = (root / "animatediff_ttnn" / "generation_helpers.py").read_text()
    assert declared == SD14_REPO, (
        f"weights: {declared!r} is not SD14_REPO ({SD14_REPO!r}), which is what the "
        "served path loads; HF_MODEL and /v1/models would report a model nobody loads"
    )
    # Every load names the constant, and no hardcoded repo literal is left behind.
    # tests/test_weights_pins.py checks the revision= half.
    # (Counted as equal, not as a fixed number: adding a load is fine, adding one that does
    # not name the constant is the thing to catch.)
    n_loads = loader.count("from_pretrained(")
    assert n_loads == loader.count("SD14_REPO, subfolder=") >= 4
    assert '"CompVis/' not in loader


def test_the_pinned_extra_code_ref_is_reachable_from_head():
    """A pin can be tree-correct and still unfetchable, which is a total failure.

    #9 was squash-merged, so every commit that was on the branch is now dangling. The pin
    it merged with (d89a90c) resolved only because the branch had not been pruned yet --
    and stage() does `git clone --filter=blob:none` then `git checkout <ref>`, so the day
    someone deletes that branch, every consumer's `tt-model package` fails on a ref the
    server no longer has.

    The tree-equality guard cannot see this: it passed on main with a pin that was one
    branch-deletion away from breaking. So this asserts reachability separately, which is
    what "pin a commit on main, or a tag" actually means.
    """
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    for extra in _manifest()["source"].get("extra_code", []):
        root = extra["root"]
        if not isinstance(root, dict) or "ref" not in root:
            continue
        ref = root["ref"]
        if subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", "--quiet",
                           f"{ref}^{{commit}}"], capture_output=True).returncode != 0:
            pytest.skip(
                f"extra_code ref {ref} is not in this clone (shallow checkout); "
                "reachability cannot be checked offline"
            )
        reachable = subprocess.run(
            ["git", "-C", str(repo), "merge-base", "--is-ancestor", ref, "HEAD"],
            capture_output=True,
        ).returncode == 0
        assert reachable, (
            f"extra_code ref {ref} is not an ancestor of HEAD, so it lives only on a "
            "branch. When that branch is deleted the ref becomes unfetchable and every "
            "consumer's `tt-model package` fails at the clone. Pin a commit on main, or a tag."
        )


def test_the_lock_ships_pytest_only_while_tt_metal_needs_it():
    """A deliberate wart, with its own removal reminder.

    tt-metal v0.77.0's models/common/utility_functions.py has `import pytest` at MODULE
    scope, and that file ships in this package's allowlist -- the SD demo's common.py does
    `from models.common.utility_functions import is_blackhole`. So without pytest in the
    image the container's lifespan dies with ModuleNotFoundError the moment it opens the
    device. A host serve never sees it: tt-metal's python_env has pytest installed. Same
    works-on-host, dies-in-image shape as the under-shipped allowlist.

    Fixed upstream in tt-metal fdcb5cf2ec6c (#55079), which moves the import into the one
    function that uses it -- but no release tag carries it yet (v0.78.0 still has the
    module-level import), and source.tt_metal.ref points at a tag on purpose.

    This test exists to make the wart visible rather than permanent: when tt_metal.ref
    moves to a ref that carries the fix, drop pytest from the lock and delete this test.
    It fails loudly in both directions.
    """
    pins = _lock_pins()
    ref = _manifest()["source"]["tt_metal"]
    ref = ref if isinstance(ref, str) else ref["ref"]

    assert pins.get("pytest"), (
        f"the lock does not ship pytest, but source.tt_metal.ref is {ref!r}, whose "
        "models/common/utility_functions.py imports pytest at module scope -- the "
        "container lifespan would die with ModuleNotFoundError"
    )
    assert ref in ("v0.77.0", "v0.78.0"), (
        f"source.tt_metal.ref moved to {ref!r}. If that ref carries tt-metal fdcb5cf2ec6c "
        "(pytest import moved inside its function), drop pytest/pluggy/iniconfig from "
        "lock.in, regenerate the lock, and delete this test -- the workaround is no longer "
        "needed. If it does not carry the fix, extend this list."
    )


# ---- LoRA variants ----------------------------------------------------------------------


def test_the_lora_catalog_is_empty_when_unset():
    from animatediff_ttnn.server.app import lora_catalog_from_env

    assert lora_catalog_from_env({}) == {}
    assert lora_catalog_from_env({"ANIMATEDIFF_LORAS": "  "}) == {}


def test_the_lora_catalog_parses_names_specs_and_stacks():
    import json

    from animatediff_ttnn.lora import LoraSpec
    from animatediff_ttnn.server.app import lora_catalog_from_env

    raw = json.dumps({"pixel": "org/p:p.safetensors@0.8", "both": ["org/a:a.safetensors", "org/b:b.safetensors@0.5"]})
    cat = lora_catalog_from_env({"ANIMATEDIFF_LORAS": raw})
    assert cat["pixel"] == [LoraSpec("org/p", "p.safetensors", 0.8)]
    assert cat["both"] == [LoraSpec("org/a", "a.safetensors", 1.0), LoraSpec("org/b", "b.safetensors", 0.5)]


@pytest.mark.parametrize("raw", [
    "{not json", "[]", '{"x": 3}', '{"x": []}', '{"": "a/b:f.safetensors"}', '{"base": "a/b:f.safetensors"}',
    '{"../etc": "a/b:f.safetensors"}', '{"x": "a/b:f.safetensors@1e999"}',
])
def test_a_malformed_lora_catalog_raises_rather_than_dropping_entries(raw):
    from animatediff_ttnn.server.app import lora_catalog_from_env

    with pytest.raises(ValueError):
        lora_catalog_from_env({"ANIMATEDIFF_LORAS": raw})


def test_an_unknown_lora_is_refused_before_the_device_lock():
    """400 names what IS offered; a refusal must not wait behind a running generation."""
    app.state.engine = {"hf_model": "x", "mesh_shape": (1, 1), "variants": {"pixel": object()}}
    app.state.device_lock = asyncio.Lock()
    try:
        with patch("animatediff_ttnn.server.app._generate") as gen:
            r = TestClient(app).post("/v1/videos/generations", json={"prompt": "a cat", "lora": "nope"})
        assert r.status_code == 400
        assert "pixel" in r.json()["detail"]
        gen.assert_not_called()
    finally:
        del app.state.engine


def test_a_lora_request_without_any_catalog_says_so():
    app.state.engine = {"hf_model": "x", "mesh_shape": (1, 1), "variants": {}}
    try:
        r = TestClient(app).post("/v1/videos/generations", json={"prompt": "a cat", "lora": "pixel"})
        assert r.status_code == 400 and "ANIMATEDIFF_LORAS" in r.json()["detail"]
    finally:
        del app.state.engine


def test_the_request_picks_the_variants_models_and_text_encoder():
    """The wiring: a named LoRA must run on ITS UNet and encode with ITS text encoder.

    Using the plain UNet with a merged encoder (or the reverse) would still produce a
    picture, just not the one asked for, so assert on what _generate hands the pipeline.
    """
    from animatediff_ttnn.generation_helpers import Sd14Variant
    from animatediff_ttnn.server import app as mod

    plain = ("p_unet", "vae", "p_cfg", "p_proj")
    variant = Sd14Variant(models=("v_unet", "vae", "v_cfg", "v_proj"), clip=("tok", "v_te"))
    state = {"device": "dev", "models": plain, "variants": {"pixel": variant}}

    def run(lora):
        with patch("animatediff_ttnn.generation_helpers.encode_prompt", return_value="emb") as enc, \
             patch("animatediff_ttnn.temporal_attention.generate_frames_temporal", return_value=[]) as gen:
            mod._generate(state, VideoGenerationRequest(prompt="a cat", lora=lora))
        return enc.call_args.kwargs.get("clip"), gen.call_args.kwargs["ttnn_model"]

    assert run("pixel") == (("tok", "v_te"), "v_unet")
    assert run(None) == (None, "p_unet")  # plain model: the process-wide encoder, no override


def test_a_failed_variant_load_closes_the_device():
    """Same leak as a failed warm: a container that dies starting must release its chip."""
    import json

    from animatediff_ttnn.server import app as mod

    async def start():
        async with mod.lifespan(mod.app):
            pass

    ttnn_mock = MagicMock()
    env = {"ANIMATEDIFF_LORAS": json.dumps({"pixel": "org/p:p.safetensors"})}
    with patch.dict(sys.modules, {"ttnn": ttnn_mock}), patch.dict("os.environ", env), \
         patch.object(mod, "_open_device_and_models", return_value=("dev", ("u", "vae", "c", "t"))), \
         patch.object(mod, "_load_lora_variants", side_effect=ValueError("SDXL LoRA")):
        with pytest.raises(ValueError, match="SDXL"):
            asyncio.run(start())
    ttnn_mock.close_mesh_device.assert_called_once_with("dev")


def test_variants_share_the_plain_models_vae():
    """The VAE has no LoRA-dependent weights; a second copy would waste device memory."""
    from animatediff_ttnn.lora import LoraSpec
    from animatediff_ttnn.server import app as mod

    with patch("animatediff_ttnn.generation_helpers.load_sd14_variant", return_value="v") as load:
        out = mod._load_lora_variants("dev", ("u", "THE_VAE", "c", "t"), {"pixel": [LoraSpec("a/b", "f.safetensors")]})
    assert out == {"pixel": "v"}
    assert load.call_args.args[2] == "THE_VAE"


def _run_generate(state, **fields):
    """Call _generate with the model calls stubbed; return (encode calls, generate kwargs)."""
    from animatediff_ttnn.server import app as mod

    with patch("animatediff_ttnn.generation_helpers.encode_prompt", return_value="emb") as enc, \
         patch("animatediff_ttnn.temporal_attention.generate_frames_temporal", return_value=[]) as gen:
        mod._generate(state, VideoGenerationRequest(prompt="a cat", **fields))
    return enc.call_args_list, gen.call_args.kwargs


def _state(lcm):
    from animatediff_ttnn.generation_helpers import Sd14Variant

    v = Sd14Variant(models=("u", "vae", "c", "t"), clip=("tok", "te"), lcm=lcm)
    return {"device": "dev", "models": ("pu", "vae", "pc", "pt"), "variants": {"fast": v, "style": v._replace(lcm=False)}}


def test_an_lcm_variant_samples_with_lcm_and_its_own_defaults():
    """The sampler follows the LoRA: an LCM variant at 25 steps / cfg 7.5 would return noise."""
    _, kw = _run_generate(_state(True), lora="fast")
    assert kw["use_lcm"] is True
    assert (kw["num_steps"], kw["guidance_scale"]) == (4, 1.5)


def test_an_lcm_variant_honours_steps_and_guidance_the_caller_sent():
    _, kw = _run_generate(_state(True), lora="fast", num_inference_steps=8, guidance_scale=2.0)
    assert (kw["num_steps"], kw["guidance_scale"], kw["use_lcm"]) == (8, 2.0, True)


def test_a_sent_default_value_still_counts_as_sent():
    # model_fields_set, not a comparison with the default: a caller who asks for 25 gets 25.
    _, kw = _run_generate(_state(True), lora="fast", num_inference_steps=25)
    assert kw["num_steps"] == 25


def test_a_style_variant_and_the_plain_model_keep_the_normal_sampler_and_defaults():
    for lora in ("style", None):
        _, kw = _run_generate(_state(True), lora=lora)
        assert kw["use_lcm"] is False
        assert (kw["num_steps"], kw["guidance_scale"]) == (25, 7.5)


def test_a_prompt_schedule_is_encoded_per_frame_with_the_variants_encoder():
    calls, kw = _run_generate(
        _state(False), lora="style", num_frames=8,
        prompt_schedule=[{"frame": 0, "prompt": "dawn"}, {"frame": 7, "prompt": "dusk"}],
    )
    sched = [c for c in calls if "prompt_schedule" in c.kwargs]
    assert len(sched) == 1
    assert sched[0].kwargs["prompt_schedule"] == [(0, "dawn"), (7, "dusk")]
    assert sched[0].kwargs["num_frames"] == 8 and sched[0].kwargs["clip"] == ("tok", "te")
    assert kw["text_embeddings_per_frame"] == "emb"


def test_without_a_schedule_no_per_frame_embeddings_are_passed():
    _, kw = _run_generate(_state(False))
    assert kw["text_embeddings_per_frame"] is None


def test_a_keyframe_outside_the_clip_is_refused_at_the_edge():
    r = TestClient(app).post("/v1/videos/generations", json={
        "prompt": "a cat", "num_frames": 4, "prompt_schedule": [{"frame": 4, "prompt": "x"}]})
    assert r.status_code == 422 and "outside the clip" in r.text


def test_the_image_ships_peft_so_a_lora_request_cannot_fail_on_a_missing_import():
    """diffusers imports peft lazily, so a missing peft passes the build-time verify and
    fails on the first LoRA load. Assert the declaration AND the lock the image installs."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    assert any(p.startswith("peft") for p in _manifest()["runtime"]["packages"])
    assert "peft" in _lock_pins(), "tt_model_package.lock does not pin peft; the image would lack it"


# ---- per-step previews ------------------------------------------------------------------


def test_the_preview_endpoint_answers_before_the_model_is_ready():
    """A poller should not have to tell "starting" from "idle": always 200, all zeros."""
    r = TestClient(app).get("/v1/videos/preview")
    assert r.status_code == 200
    assert r.json() == {"step": 0, "total": 0, "gif_b64": None, "running": False}


def test_the_preview_store_resets_per_request_and_publishes_what_the_callback_wrote(tmp_path):
    import base64

    from animatediff_ttnn.server.app import PreviewStore

    store = PreviewStore()
    try:
        store.begin(4)
        assert store.snapshot() == {"step": 0, "total": 4, "gif_b64": None, "running": True}
        with open(store.path, "wb") as fh:
            fh.write(b"GIF89a-one")
        store.publish(f"PREVIEW: 2/4 {store.path}")
        snap = store.snapshot()
        assert (snap["step"], snap["total"]) == (2, 4)
        assert base64.b64decode(snap["gif_b64"]) == b"GIF89a-one"
        store.end()
        assert store.snapshot()["running"] is False
        assert store.snapshot()["gif_b64"], "the last frame stays readable after the run ends"
        store.begin(8)  # next request: the previous request's frame must not leak into it
        assert store.snapshot() == {"step": 0, "total": 8, "gif_b64": None, "running": True}
    finally:
        store.close()


def test_a_lost_preview_frame_keeps_the_previous_one_and_never_raises():
    from animatediff_ttnn.server.app import PreviewStore

    store = PreviewStore()
    try:
        store.begin(4)
        with open(store.path, "wb") as fh:
            fh.write(b"first")
        store.publish(f"PREVIEW: 1/4 {store.path}")
        before = store.snapshot()["gif_b64"]
        store.publish("PREVIEW: 2/4 /no/such/file.gif")  # unreadable
        store.publish("some other log line")  # not an announcement
        assert store.snapshot()["gif_b64"] == before
    finally:
        store.close()


def test_the_step_counter_runs_without_the_flag_but_no_gif_is_rendered():
    """The counter is free; the GIF render is what costs device time, so it is opt-in."""
    from animatediff_ttnn.server import app as mod

    store = mod.PreviewStore()
    try:
        with patch("animatediff_ttnn.preview.make_step_callback") as make:
            cb = mod._step_callback(store, VideoGenerationRequest(prompt="x"), 4)
            cb(0, 4, [object()])
            cb(1, 4, [object()])
        make.assert_not_called()
        assert store.snapshot()["step"] == 2
        assert store.snapshot()["gif_b64"] is None
    finally:
        store.close()


def test_the_preview_endpoint_answers_while_the_device_lock_is_held():
    """The requirement: previews are only useful if they can be read DURING the work.

    A real POST holds the device lock inside a blocked fake denoise loop. While it is blocked
    the endpoint must still answer promptly, report the step reached, and carry the frame the
    loop published. The loop is the real _generate with only the model calls and the GIF
    render stubbed, so the wiring from on_step to the endpoint is what is under test.
    """
    import base64
    import threading
    import time as _time

    from animatediff_ttnn.generation_helpers import Sd14Variant  # noqa: F401  (import check)
    from animatediff_ttnn.server import app as mod

    started, release = threading.Event(), threading.Event()

    def fake_loop(**kw):
        for i in range(3):  # three steps; the callback is the real _step_callback
            kw["on_step"](i, 3, [object()])
        started.set()
        assert release.wait(timeout=20), "test never released the fake loop"
        return [object()]

    def fake_export(frames, path):
        with open(path, "wb") as fh:
            fh.write(b"GIF89a-live")

    store = mod.PreviewStore()
    app.state.engine = {"device": "d", "models": ("u", "vae", "c", "t"), "variants": {}, "preview": store,
                        "hf_model": "x", "mesh_shape": (1, 1)}
    app.state.device_lock = asyncio.Lock()
    result = {}

    def post():
        with patch("animatediff_ttnn.generation_helpers.encode_prompt", return_value="emb"), \
             patch("animatediff_ttnn.temporal_attention.generate_frames_temporal", side_effect=lambda **kw: fake_loop(**kw)), \
             patch("animatediff_ttnn.preview._latent_preview_fn", return_value=lambda *a, **k: ["f"]), \
             patch("animatediff_ttnn.preview._export_gif_fn", return_value=fake_export), \
             patch.object(mod, "_frames_to_gif_b64", return_value="Zm9v"):
            result["r"] = TestClient(app).post("/v1/videos/generations", json={
                "prompt": "a cat", "num_inference_steps": 3, "num_frames": 4, "preview": True})

    t = threading.Thread(target=post)
    try:
        t.start()
        assert started.wait(timeout=20), "the generation never reached the loop"
        t0 = _time.monotonic()
        r = TestClient(app).get("/v1/videos/preview")  # the device lock is held right now
        elapsed = _time.monotonic() - t0
        assert r.status_code == 200
        body = r.json()
        assert body["running"] is True and (body["step"], body["total"]) == (3, 3)
        assert base64.b64decode(body["gif_b64"]) == b"GIF89a-live"
        assert elapsed < 2.0, f"the preview endpoint waited {elapsed:.1f}s behind the generation"
        assert app.state.device_lock.locked(), "the test is not holding the lock it claims to"
    finally:
        release.set()
        t.join(timeout=30)
        del app.state.engine
        store.close()
    assert result["r"].status_code == 200
    assert store.snapshot()["running"] is False


def test_preview_every_reaches_the_step_callback_builder():
    from animatediff_ttnn.server import app as mod

    store = mod.PreviewStore()
    try:
        with patch("animatediff_ttnn.preview.make_step_callback") as make:
            mod._step_callback(store, VideoGenerationRequest(prompt="x", preview=True, preview_every=5), 20)
            mod._step_callback(store, VideoGenerationRequest(prompt="x", preview=True), 20)
        assert make.call_args_list[0].kwargs["every"] == 5
        assert make.call_args_list[1].kwargs["every"] is None  # library default cadence
    finally:
        store.close()


@pytest.mark.parametrize("bad", [0, -1, 101])
def test_a_preview_cadence_out_of_range_is_refused(bad):
    r = TestClient(app).post("/v1/videos/generations", json={"prompt": "x", "preview_every": bad})
    assert r.status_code == 422


# ---- chaining, sampler choice, Phase 3 -------------------------------------------------


def _req(**kw):
    return VideoGenerationRequest(prompt="a cat", **kw)


def _engine(**kw):
    from animatediff_ttnn.generation_helpers import Sd14Variant
    from animatediff_ttnn.server.app import ChainStore

    base = {"variants": {"style": Sd14Variant(("u", "v", "c", "t"), ("tok", "te"), lcm=False),
                         "fast": Sd14Variant(("u", "v", "c", "t"), ("tok", "te"), lcm=True)},
            "motion_kernels": None, "chains": ChainStore()}
    base.update(kw)
    return base


def test_request_conflicts_accepts_a_plain_request_and_the_documented_combinations():
    from animatediff_ttnn.server.app import request_conflicts

    eng = _engine(motion_kernels={"mid": []})
    try:
        for kw in ({}, {"lora": "style", "sampler": "euler"}, {"lora": "fast"},
                   {"motion_adapter": True, "sampler": "euler", "motion_adapter_skip": ["up1", "up2"]},
                   {"motion_adapter": True, "lora": "style"}):
            assert request_conflicts(_req(**kw), eng) is None, kw
    finally:
        eng["chains"].close()


@pytest.mark.parametrize("kw,needle", [
    ({"lora": "nope"}, "unknown lora"),
    ({"lora": "fast", "sampler": "euler"}, "always samples with LCM"),
    ({"motion_adapter": True}, "ANIMATEDIFF_MOTION_ADAPTER"),
    ({"motion_adapter_skip": ["up1"]}, "need motion_adapter"),
    ({"motion_adapter_alpha": 0.5}, "need motion_adapter"),
    ({"chain_from": "nothing-saved-yet"}, "unknown chain"),
    ({"lora": "fast", "num_inference_steps": 51}, "at most 50"),
    ({"lora": "fast", "num_inference_steps": 100}, "at most 50"),
])
def test_request_conflicts_names_the_reason(kw, needle):
    from animatediff_ttnn.server.app import request_conflicts

    eng = _engine()
    try:
        assert needle in request_conflicts(_req(**kw), eng)
    finally:
        eng["chains"].close()


def test_motion_adapter_cannot_ride_an_lcm_variant():
    from animatediff_ttnn.server.app import request_conflicts

    eng = _engine(motion_kernels={"mid": []})
    try:
        assert "LCM" in request_conflicts(_req(motion_adapter=True, lora="fast"), eng)
    finally:
        eng["chains"].close()


def test_a_chain_name_cannot_point_outside_the_store():
    """Names, never paths: the pattern refuses separators and dots-only names at the edge, and the
    file name is a hash anyway."""
    import os

    from animatediff_ttnn.server.app import ChainStore

    for bad in ("../x", "a/b", "/etc/passwd", ".hidden", ""):
        r = TestClient(app).post("/v1/videos/generations", json={"prompt": "x", "chain_save": bad})
        assert r.status_code == 422, bad
    store = ChainStore()
    try:
        p = store.path("../../etc/passwd")
        assert os.path.dirname(p) == store._dir and p.endswith(".pt")
    finally:
        store.close()


def test_the_chain_store_keeps_the_newest_and_forgets_on_close():
    import os

    from animatediff_ttnn.server.app import ChainStore

    store = ChainStore()
    store.MAX_CHAINS = 3
    try:
        for i in range(5):
            with open(store.path(f"c{i}"), "wb") as fh:
                fh.write(b"x")
            os.utime(store.path(f"c{i}"), (1000 + i, 1000 + i))
        store.prune()
        assert [store.has(f"c{i}") for i in range(5)] == [False, False, True, True, True]
    finally:
        d = store._dir
        store.close()
    assert not os.path.exists(d)


def test_the_sampler_and_chain_fields_reach_the_denoise_loop():
    eng = {"device": "dev", "models": ("u", "vae", "c", "t"), "variants": {}, "preview": None}
    from animatediff_ttnn.server.app import ChainStore

    eng["chains"] = ChainStore()
    try:
        _calls, kw = _run_generate(eng, sampler="euler", chain_from="a", chain_save="b", chain_alpha=0.4)
        assert kw["use_lightning"] is True and kw["use_lcm"] is False
        assert kw["chain_from"] == eng["chains"].path("a") and kw["chain_save"] == eng["chains"].path("b")
        assert kw["chain_alpha"] == 0.4
        _calls, kw = _run_generate(eng)
        assert kw["use_lightning"] is False and kw["chain_from"] is None and kw["chain_save"] is None
    finally:
        eng["chains"].close()


def test_a_motion_adapter_request_runs_the_phase_3_loop_with_the_operators_kernels():
    from animatediff_ttnn.server import app as mod

    kernels = {"mid": ["m"]}
    eng = {"device": "dev", "models": ("u", "vae", "c", "t"), "variants": {}, "preview": None,
           "motion_kernels": kernels, "chains": None}
    with patch("animatediff_ttnn.generation_helpers.encode_prompt", return_value="emb"), \
         patch("animatediff_ttnn.temporal_attention.generate_frames_temporal") as plain, \
         patch("animatediff_ttnn.temporal_attention.generate_frames_motion", return_value=[]) as motion:
        mod._generate(eng, _req(motion_adapter=True, motion_adapter_alpha=0.5, motion_adapter_skip=["up1", "up2"]))
    plain.assert_not_called()
    kw = motion.call_args.kwargs
    assert kw["temporal_kernels"] is kernels and kw["injection_alpha"] == 0.5
    assert kw["skip_keys"] == {"up1", "up2"} and "use_lcm" not in kw


def test_saving_a_chain_prunes_the_store_afterwards():
    from animatediff_ttnn.server.app import ChainStore

    eng = {"device": "dev", "models": ("u", "vae", "c", "t"), "variants": {}, "preview": None,
           "chains": ChainStore()}
    try:
        with patch.object(ChainStore, "prune") as prune:
            _run_generate(eng, chain_save="keep")
        prune.assert_called_once()
    finally:
        eng["chains"].close()


def test_the_http_edge_refuses_unknown_chains_and_disabled_motion_before_the_lock():
    from animatediff_ttnn.server.app import ChainStore

    eng = _engine()
    eng.update({"hf_model": "x", "mesh_shape": (1, 1), "preview": None})
    app.state.engine = eng
    app.state.device_lock = asyncio.Lock()
    try:
        with patch("animatediff_ttnn.server.app._generate") as gen:
            c = TestClient(app)
            r1 = c.post("/v1/videos/generations", json={"prompt": "x", "chain_from": "missing"})
            r2 = c.post("/v1/videos/generations", json={"prompt": "x", "motion_adapter": True})
        assert r1.status_code == 400 and "unknown chain" in r1.json()["detail"]
        assert r2.status_code == 400 and "ANIMATEDIFF_MOTION_ADAPTER" in r2.json()["detail"]
        gen.assert_not_called()
    finally:
        del app.state.engine
        eng["chains"].close()


def test_motion_adapter_env_parsing():
    from animatediff_ttnn.server.app import motion_adapter_from_env
    from animatediff_ttnn.weights_pins import MOTION_ADAPTER_REPO

    assert motion_adapter_from_env({}) is None
    assert motion_adapter_from_env({"ANIMATEDIFF_MOTION_ADAPTER": "  "}) is None
    assert motion_adapter_from_env({"ANIMATEDIFF_MOTION_ADAPTER": "default"}) == MOTION_ADAPTER_REPO
    assert motion_adapter_from_env({"ANIMATEDIFF_MOTION_ADAPTER": "me/adapter"}) == "me/adapter"


def test_models_advertises_whether_motion_adapter_requests_are_accepted():
    app.state.engine = {"hf_model": "x", "variants": {}, "motion_kernels": {"mid": []}}
    try:
        assert TestClient(app).get("/v1/models").json()["motion_adapter"] is True
        app.state.engine["motion_kernels"] = None
        assert TestClient(app).get("/v1/models").json()["motion_adapter"] is False
    finally:
        del app.state.engine


def test_a_failed_motion_adapter_load_closes_the_device():
    from animatediff_ttnn.server import app as mod

    async def start():
        async with mod.lifespan(mod.app):
            pass

    ttnn_mock = MagicMock()
    with patch.dict(sys.modules, {"ttnn": ttnn_mock}), \
         patch.dict("os.environ", {"ANIMATEDIFF_MOTION_ADAPTER": "default"}), \
         patch.object(mod, "_open_device_and_models", return_value=("dev", ("u", "vae", "c", "t"))), \
         patch("animatediff_ttnn.motion_weights.load_motion_modules", side_effect=OSError("no such repo")):
        with pytest.raises(OSError, match="no such repo"):
            asyncio.run(start())
    ttnn_mock.close_mesh_device.assert_called_once_with("dev")


def test_the_serve_profiles_carry_a_catalog_the_server_can_parse():
    """`tt-model serve` has no --env, so the manifest's serve profiles are the only way an
    operator's LoRA catalog and motion-adapter switch reach the container. Parse each
    profile's env with the server's own parser: a typo here would otherwise surface as a
    lifespan failure inside a published image."""
    from animatediff_ttnn.server.app import LORAS_ENV, lora_catalog_from_env

    m = _manifest()
    profiles = {p["name"]: p for p in m.get("serve_profiles", [])}
    assert profiles, "no serve_profiles: the catalog has no way into the container"
    assert m["default_profile"] in profiles
    with_catalog = 0
    for name, p in profiles.items():
        env = {k: str(v) for k, v in (p.get("env") or {}).items()}
        catalog = lora_catalog_from_env(env)  # raises on a malformed catalog
        if catalog:
            with_catalog += 1
            assert "base" not in catalog
        if "ANIMATEDIFF_MOTION_ADAPTER" in env:
            assert env["ANIMATEDIFF_MOTION_ADAPTER"] in ("default", "repo") or "/" in env["ANIMATEDIFF_MOTION_ADAPTER"]
    assert with_catalog >= 1, f"no profile sets {LORAS_ENV}"
    assert "plain" in profiles and not (profiles["plain"].get("env") or {}), "plain must stay catalog-free"


def test_lcm_accepts_fifty_steps_and_a_style_variant_accepts_a_hundred():
    from animatediff_ttnn.server.app import request_conflicts

    eng = _engine()
    try:
        assert request_conflicts(_req(lora="fast", num_inference_steps=50), eng) is None
        assert request_conflicts(_req(lora="fast"), eng) is None  # the 4-step default
        assert request_conflicts(_req(lora="style", num_inference_steps=100), eng) is None
    finally:
        eng["chains"].close()


def test_the_lcm_step_limit_is_the_one_the_scheduler_is_built_with():
    # Review finding: 51-100 steps passed validation and then failed inside the scheduler as
    # an HTTP 500 after taking the device lock. The server's limit and the scheduler's
    # original_inference_steps must be the same number, and the scheduler must really refuse
    # one step more.
    import inspect

    from diffusers import LCMScheduler

    from animatediff_ttnn import temporal_attention
    from animatediff_ttnn.weights_pins import LCM_ORIGINAL_STEPS

    assert f"original_inference_steps={'LCM_ORIGINAL_STEPS'}" in inspect.getsource(
        temporal_attention.generate_frames_temporal).replace(" ", "")
    sched = LCMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                         original_inference_steps=LCM_ORIGINAL_STEPS)
    sched.set_timesteps(LCM_ORIGINAL_STEPS)
    with pytest.raises(ValueError):
        sched.set_timesteps(LCM_ORIGINAL_STEPS + 1)


def test_a_chain_evicted_while_the_request_waits_for_the_device_is_a_400():
    """The early check passes, the request queues, and a running generation prunes the source.

    Review finding: chain_blend_seed ignores a missing file, so the queued request used to
    return 200 with no continuity. The lock below deletes the chain at the moment it is
    granted, which is when a pruning generation would have finished.
    """
    import animatediff_ttnn.server.app as mod
    from animatediff_ttnn.server.app import ChainStore

    chains = ChainStore()
    open(chains.path("src"), "wb").write(b"x")

    class _EvictOnAcquire:
        async def __aenter__(self):
            os.unlink(chains.path("src"))

        async def __aexit__(self, *exc):
            return False

    app.state.engine = {"device": object(), "models": object(), "ready": True, "variants": {},
                        "chains": chains, "model": "m", "mesh_device": "P150", "mesh_shape": "1x1"}
    app.state.device_lock = _EvictOnAcquire()
    try:
        with patch.object(mod, "_generate", return_value=[object()]) as gen:
            r = TestClient(app).post("/v1/videos/generations", json={"prompt": "a cat", "chain_from": "src"})
        assert r.status_code == 400, r.text
        assert "unknown chain" in r.json()["detail"]
        gen.assert_not_called()
    finally:
        del app.state.engine
        chains.close()
