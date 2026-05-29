# tt-animatediff Integration Guide

How to consume this project correctly depending on your use case.

---

## The canonical source

`github.com/tenstorrent/tt-animatediff` is the **single source of truth** for this
codebase. Do not copy files from it into your project's tree and maintain them
separately — that is how the three-way divergence that led to this repo came about.

All downstream consumers should pin to a **tagged release** and upgrade
intentionally, not track `main`.

---

## For toolkit / lesson projects (e.g. tt-vscode-toolkit)

**Goal:** Teach users to run AnimateDiff on Blackhole hardware. The lesson content should
walk through setting up and running the canonical release, not ship a stale snapshot of it.

### Recommended pattern: release checkout at lesson start

Instead of rsyncing a copy of the project into the extension's `content/projects/animatediff/`
directory, the lesson should guide the user to clone a pinned release tag into their
workspace on first run:

```bash
# In the lesson's setup step or tt-gen/workspace-init hook:
ANIMATEDIFF_TAG="v0.2.0"   # update this when cutting a new lesson revision
DEST="$HOME/tt-projects/animatediff"

if [ ! -d "$DEST" ]; then
    git clone --depth 1 --branch "$ANIMATEDIFF_TAG" \
        https://github.com/tenstorrent/tt-animatediff.git "$DEST"
fi
cd "$DEST"
pip install -e ".[dev]"
bash weights/download_weights.sh
```

This gives the user a real working copy they own, not one embedded in the extension.

### What the lesson content should contain

The lesson `.md`/walkthrough files in `tt-vscode-toolkit` should:

- Reference the release tag the lesson was written against (e.g. `# tested against tt-animatediff v0.2.0`)
- Link to the canonical repo for source exploration
- Contain the commands to clone + install, not copies of the Python source
- Reference `examples/generate_baseline.py` and `examples/generate_blackhole_v2.py`
  by name — users run them from their checked-out copy, not from the extension bundle

### What the lesson should NOT contain

- Do **not** vendor a copy of `animatediff_ttnn/` inside the extension tree.
  If the lesson ships a copy, it will drift from the canonical version and
  users will hit bugs that have already been fixed upstream.
- Do **not** include `output/`, `weights/`, or any `.ckpt`/`.safetensors` files.
  The `weights/download_weights.sh` script in the repo handles weight acquisition.

### When to bump the pinned tag in the lesson

Pin to a new release when you intentionally update the lesson content. Do not
auto-track `main` — lesson walkthroughs must be stable over the lifetime of a
toolkit version. The lesson's `README` and setup step should show the user which
version they are using.

---

## For application plugins (e.g. tt-local-generator)

**Goal:** Ship AnimateDiff generation as a production feature of the application.
The plugin should be robust, pinned, and independently testable.

### Recommended pattern: git subtree or submodule at a release tag

The cleanest way to vendor this project in an application is a **git subtree** —
it avoids the stale-copy problem while keeping the dependency self-contained and
auditable:

```bash
# Initial vendor (run once, from tt-local-generator root):
git subtree add \
    --prefix app/animatediff \
    https://github.com/tenstorrent/tt-animatediff.git \
    v0.2.0 --squash

# To upgrade to a new release:
git subtree pull \
    --prefix app/animatediff \
    https://github.com/tenstorrent/tt-animatediff.git \
    v0.3.0 --squash
```

This records exactly which upstream commit is vendored, keeps the full tree available
for import, and lets `git log app/animatediff/` show the upgrade history.

### What the plugin owns vs. what is delegated

The plugin layer in `plugins/animatediff/plugin.py` should own:

- **MCP schema and argument parsing** — the shape of the external API exposed to
  callers (prompts, frame count, seed, etc.)
- **Hardware gate** — the `check_hardware()` call before attempting any TTNN work
- **Output path management** — where GIFs land, thumbnail generation
- **Error surface** — translating subprocess failures into user-readable messages

The plugin should **not** own:

- Model loading, denoising logic, scheduler parameters, or temporal attention —
  those live in `animatediff_ttnn/` and `examples/generate_blackhole_v2.py` in the
  vendored tree. Keep business logic there, keep it tested there.

### Subprocess vs. direct import

The current `run_subprocess()` pattern (running `generate_blackhole_v2.py` via
`~/tt-metal/python_env/bin/python`) is the right call for production use because:

- tt-metal's Python env is hermetic and may conflict with the application's env
- Subprocess isolation means a TTNN crash does not kill the application process
- Timeout, logging, and progress streaming are easy to implement at the subprocess boundary

Keep this pattern. Do **not** attempt to `import animatediff_ttnn` directly from
within the application process unless the application itself runs inside the
tt-metal Python env.

### Version tracking

The vendored subtree path should have a `VERSION` file (already present in the
canonical repo as `setup.py` version field) that the plugin reads at startup and
includes in log output:

```python
# In plugin startup / health check:
version_file = _BUNDLED_DIR / "setup.py"
# or read from animatediff_ttnn/__init__.py __version__
```

This makes it immediately clear in support logs which release of tt-animatediff
is running.

---

## For end users / third-party developers

If you want to use tt-animatediff in your own project:

### Install from the public repo

```bash
pip install git+https://github.com/tenstorrent/tt-animatediff.git@v0.2.0
```

Or clone and install in editable mode for development:

```bash
git clone https://github.com/tenstorrent/tt-animatediff.git
cd tt-animatediff
pip install -e ".[dev]"          # CPU/Phase 1 only
pip install -e ".[dev,video]"    # adds diffusers for GIF export
```

### What requires Blackhole hardware vs. what runs on any machine

| Feature | Hardware required |
|---|---|
| `generate_baseline.py` (Phase 1 CPU AnimateDiff) | No — any machine with 16 GB RAM |
| `generate_blackhole.py` (Phase 2 TTNN UNet) | Yes — Blackhole (P100/P150/P300c/QB2) |
| `generate_blackhole_v2.py` (Phase 2.5 temporal attention) | Yes — Blackhole |
| `examples/generate_sim.py` (ttsim simulator) | No — any Linux/x86_64 machine |
| `pytest tests/test_pipeline.py` | No |
| `pytest tests/test_ttnn_pipeline.py` | No (hardware mocked) |

### Required tt-metal setup (for Phase 2/2.5 on real hardware)

```bash
# Build tt-metal (one-time):
git clone https://github.com/tenstorrent/tt-metal.git ~/tt-metal
cd ~/tt-metal && ./build_metal.sh

# Activate its Python env every session:
source ~/tt-metal/python_env/bin/activate

# Run:
cd ~/tt-animatediff
python examples/generate_blackhole_v2.py --prompt "your prompt" --frames 8
```

### Using ttsim (no hardware required)

See [`docs/SIMULATOR.md`](SIMULATOR.md) for the full guide to running Phase 2
on the ttsim Blackhole simulator.

### Importing the Python library

```python
# Phase 1 (CPU only — works without tt-metal):
from animatediff_ttnn.pipeline import create_animatediff_pipeline, generate, export_gif

pipeline = create_animatediff_pipeline()
frames = generate(pipeline, prompt="a candle flame flickering", num_frames=8)
export_gif(frames, "output/candle.gif")

# Phase 2.5 (Blackhole device required):
from animatediff_ttnn.ttnn_pipeline import setup_blackhole, generate_frames
from animatediff_ttnn.temporal_attention import generate_frames_temporal

device = setup_blackhole()
# ... (see examples/generate_blackhole_v2.py for full setup)
```

---

## Release policy

- Releases are tagged `vMAJOR.MINOR.PATCH` on `main`.
- **PATCH** bumps: bug fixes, hardware workarounds, documentation.
- **MINOR** bumps: new generation phases, new CLI flags, API additions (backwards compatible).
- **MAJOR** bumps: breaking API changes (e.g. refactoring public function signatures).
- `main` should always pass `pytest tests/test_pipeline.py`. Hardware tests are
  gated on Blackhole CI runners and are not required to merge.

Downstream consumers (toolkit lessons, application plugins) should pin to a
release tag and upgrade at their own cadence — not track `main`.
