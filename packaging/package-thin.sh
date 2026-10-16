#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# Stage the tt-animatediff v6 THIN bundle, the canonical way. Rerunnable: every input is rebuilt
# from this checkout and from pinned sources, so two runs on the same commit give the same bundle
# (the manifest's created_at and hostname aside).
#
#   packaging/package-thin.sh                 stage the bundle into $WORK/bundle and inspect it
#   packaging/package-thin.sh --install-test  also run install.sh in a COPY and check imports there
#   packaging/package-thin.sh --relock        regenerate packaging/constraints.txt, then stop
#
# THIS SCRIPT NEVER PUSHES. It calls `tt-model package-thin --out ...` without the positional
# repo id, which stages and does nothing else. Publishing is a separate, deliberate step.
#
# Before anyone publishes the result:
#   * Do not push it to episod/tt-animatediff as it stands. That repo already holds the diffusers
#     pipeline, whose root requirements.txt (pip ranges for the pipeline) has the same name as the
#     bundle's requirements.txt (the ttnn and torch pins). tt-model refuses to overwrite a foreign
#     file only when the repo is not yet a bundle. episod/tt-animatediff already is one, so the
#     push would replace the pipeline's file without a word. The bundle also writes install.sh and
#     run.sh next to the pipeline's app.py. Decide where the bundle lives first: its own repo, or
#     a renamed pipeline requirements file.
#   * Look at what the target repo holds now (tt_kernel_manifest.json name) and its visibility.
#   * Prove the bundle on a card first (install.sh, run.sh, GET /v1/models, one generation). This
#     script proves the install and the imports only. See "What was not verified" in CLAUDE.md.
#   * Push by re-running this command with the repo id added to the package-thin call.
#
# What "canonical" means here (tt-model-manager docs/E2E_RECIPE_V6.md, the tt-model-thin-package
# skill) and how this model maps onto it:
#
#   engine         stock ttnn 0.77.0 from PyPI (requirements.txt). No fork, no +local tag.
#   custom kernels none. Every kernel is a stock ttnn op. There is no generic_op wheel, so there
#                  is no --ops-wheel for kernels and deps.custom_ops stays empty.
#   tooling        tt-transformers is NOT used. This is a Stable Diffusion UNet and VAE served by
#                  uvicorn (kind tt-dit-server). It imports nothing from tt_transformers, so there
#                  is no tooling wheel. (The recipe's tooling-only wheel is for LLM adapters.)
#   model code     --extra-code: animatediff_ttnn (this repo) plus the slice of tt-metal's
#                  models/ tree it imports (32 files, plus empty __init__.py files), copied
#                  from tag v0.77.0. This replaces the old --models-wheel
#                  tt_animatediff_models_closure, which carried all of models/common (about 150
#                  files, including other models' code) to run one model.
#   dependencies   --constraints with fully resolved pins. The pip dependencies with a wide
#                  closure travel in a metadata-only wheel passed by --ops-wheel, because
#                  install.sh resolves requirements.txt against the PyTorch CPU index, which
#                  replaces certifi, urllib3 and requests with 2022 releases. See
#                  packaging/runtime-deps.txt.
#   parsers        --tool-parser and --reasoning-parser do not apply: this is not an LLM.
#   guards         packaging/install-guards.txt, passed as --verify.
#   provenance     --provenance, a JSON file written below.
#
# Hardware: nothing here touches a device. It never imports ttnn or ttml.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

# ---- Pins. Change these deliberately, and rerun with --relock if the requirements change. ----
TTNN_VERSION="0.77.0"            # packaging/requirements.txt must pin the same release
TT_METAL_REF="v0.77.0"           # the tt-metal tag the SD demo code comes from
# What that tag points at. Checked against the checkout below. The environment can override it so
# tests/test_packaging_thin.py can run this script against a small synthetic tree.
TT_METAL_COMMIT="${TT_METAL_COMMIT:-9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9}"
PYTHON_VERSION="3.12.12"         # interpreter uv provisions inside the bundle (ttnn has a cp312 wheel)
BUNDLE_NAME="tt-animatediff"
APP="model:app"                  # packaging/model.py re-exports animatediff_ttnn.server.app:app
DEPS_WHEEL_NAME="tt-animatediff-deps"
PYTORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"

# What package-thin pulls from the weights repo. Only what animatediff_ttnn reads from SD 1.4:
# the fp32 safetensors for the UNet, VAE and text encoder, their configs, and the tokenizer.
# The repo also holds .bin copies, fp16 and non-EMA variants, a safety checker and two 4 GB
# checkpoints. A full pull is about 22 GB. This is about 4.3 GB.
WEIGHTS_ALLOW=(
  "unet/config.json"
  "unet/diffusion_pytorch_model.safetensors"
  "vae/config.json"
  "vae/diffusion_pytorch_model.safetensors"
  "text_encoder/config.json"
  "text_encoder/model.safetensors"
  "tokenizer/*"
  "model_index.json"
)

# ---- Where things go. Override with environment variables. ----
WORK="${WORK:-$REPO/build/thin}"            # git-ignored (build/ is)
METAL_SRC="${TT_METAL_SRC:-$WORK/tt-metal-src}"   # a tt-metal checkout at $TT_METAL_REF; cloned if absent
TT_MODEL="${TT_MODEL:-tt-model}"

MODE=stage
for arg in "$@"; do
  case "$arg" in
    --install-test) MODE=install-test ;;
    --relock) MODE=relock ;;
    -h|--help) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d'; exit 0 ;;
    *) echo "package-thin.sh: unknown argument '$arg'. A repo id would publish; this script never does." >&2; exit 2 ;;
  esac
done

say() { printf '\n== %s\n' "$*"; }
die() { printf 'package-thin.sh: %s\n' "$*" >&2; exit 1; }

# ============================================================================================
# --relock: regenerate constraints.txt
# ============================================================================================
# Two resolves, because install.sh resolves in two places that rank indexes differently:
#   step 1  requirements.txt, PyTorch CPU index ranked ABOVE PyPI (uv's first-index strategy)
#   step 3  the tt-animatediff-deps wheel by path, PyPI only
# A pin has to hold in whichever step reaches the package. So:
#   1. resolve requirements.txt the way step 1 does. These pins are what step 1 will install,
#      including a few the CPU index forces to old versions (idna, packaging, and similar).
#   2. resolve requirements.txt + runtime-deps.txt with those pins as constraints, plus
#      validated-pins.txt (the model-library versions the container served on a card), and the
#      best version from either index for everything else. That is where certifi, requests and
#      the rest of the diffusers closure get current releases.
# The result must agree with step 1 on every package step 1 installs. The script checks that.
if [ "$MODE" = relock ]; then
  command -v uv >/dev/null || die "uv is required"
  T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
  EXCL=()
  [ -n "${RELOCK_EXCLUDE_NEWER:-}" ] && EXCL=(--exclude-newer "$RELOCK_EXCLUDE_NEWER")
  COMMON=(--python-version "${PYTHON_VERSION%.*}" --python-platform x86_64-manylinux_2_34
          --extra-index-url "$PYTORCH_CPU_INDEX" --no-header --no-annotate "${EXCL[@]}")
  say "step 1 resolve (CPU index first, as install.sh does)"
  uv pip compile "$HERE/requirements.txt" "${COMMON[@]}" -o "$T/step1.txt"
  say "step 3 resolve (best version from either index, step 1 pins fixed)"
  uv pip compile "$HERE/requirements.txt" "$HERE/runtime-deps.txt" -c "$T/step1.txt" -c "$HERE/validated-pins.txt" \
    --index-strategy unsafe-best-match "${COMMON[@]}" -o "$T/all.txt"
  # Every step-1 pin must survive unchanged into the final file.
  while read -r pin; do
    [ -z "$pin" ] && continue
    grep -qxF -- "$pin" "$T/all.txt" || die "step 1 pin '$pin' is not in the final constraints"
  done < <(grep -v '^#' "$T/step1.txt")
  {
    echo "# Fully resolved pins for every install step of this bundle. Generated: do not hand-edit."
    echo "# Regenerate with: packaging/package-thin.sh --relock   (uv $(uv --version | awk '{print $2}'), $(date -u +%Y-%m-%d), python ${PYTHON_VERSION%.*}, manylinux_2_34 x86_64)"
    echo "#"
    echo "# Two kinds of pin live here. install.sh step 1 ranks the PyTorch CPU index above PyPI, so the"
    echo "# packages that step reaches are pinned to what the CPU index offers. Some of those are old"
    echo "# (idna, packaging, and similar). Step 3 resolves from PyPI only, so everything else, including"
    echo "# certifi, urllib3 and requests, is pinned to current releases. See packaging/requirements.txt."
    echo "#"
    echo "# pytest is pinned because tt-metal v0.77.0's models/common/utility_functions.py imports it at"
    echo "# module scope. See packaging/runtime-deps.txt."
    cat "$T/all.txt"
  } > "$HERE/constraints.txt"
  say "wrote $HERE/constraints.txt ($(grep -c '==' "$HERE/constraints.txt") pins)"
  exit 0
fi

# ============================================================================================
# Preflight
# ============================================================================================
say "preflight"
command -v "$TT_MODEL" >/dev/null || die "$TT_MODEL not found. Install tt-model-manager (pip install -e ~/code/tt-model-manager)."
# `tt` belongs to the official Tenstorrent CLI. This recipe uses tt-model only.
command -v uv >/dev/null || die "uv is required (it provisions the bundle's interpreter)"
command -v git >/dev/null || die "git is required"
PYTHON="${PYTHON:-python3}"
VERSION="$(tr -d '[:space:]' < "$REPO/VERSION")"
[ -f "$HERE/constraints.txt" ] || die "packaging/constraints.txt is missing. Run: packaging/package-thin.sh --relock"
grep -qx "ttnn==$TTNN_VERSION" "$HERE/requirements.txt" || die "packaging/requirements.txt does not pin ttnn==$TTNN_VERSION"
grep -q '^\s*-' "$HERE/requirements.txt" && die "packaging/requirements.txt has an option line. No index options in this file."

# The weights revision comes from the package, so the manifest pin and the pin the server loads
# cannot drift apart. weights_pins.py imports nothing heavy.
read -r WEIGHTS_REPO WEIGHTS_REV < <("$PYTHON" -I -c \
  "import runpy,sys; m=runpy.run_path(sys.argv[1]); print(m['SD14_REPO'], m['SD14_REVISION'])" \
  "$REPO/animatediff_ttnn/weights_pins.py")
echo "version $VERSION, weights $WEIGHTS_REPO @ ${WEIGHTS_REV:0:12}, ttnn $TTNN_VERSION, python $PYTHON_VERSION"

# ============================================================================================
# 1. tt-metal source for the Stable Diffusion demo code
# ============================================================================================
# Primary source: the tt-metal tag the container image is built from. The previously published
# bundle carried the same files in tt_animatediff_models_closure-0.77.0 (episod/tt-animatediff at
# 3c64ff6); they match this tree byte for byte, except that the wheel adds empty __init__.py files.
say "tt-metal $TT_METAL_REF source"
if [ ! -d "$METAL_SRC/models" ]; then
  rm -rf "$METAL_SRC"
  git clone -q --depth 1 --branch "$TT_METAL_REF" --filter=blob:none --sparse \
    https://github.com/tenstorrent/tt-metal "$METAL_SRC"
  git -C "$METAL_SRC" sparse-checkout set models/common models/demos/vision/generative/stable_diffusion/wormhole
fi
GOT="$(git -C "$METAL_SRC" rev-parse HEAD)"
[ "$GOT" = "$TT_METAL_COMMIT" ] || die "tt-metal at $METAL_SRC is $GOT, expected $TT_METAL_REF = $TT_METAL_COMMIT"
echo "tt-metal $TT_METAL_REF = ${GOT:0:12}"

# ============================================================================================
# 2. Lay out the inputs
# ============================================================================================
say "inputs in $WORK/inputs"
rm -rf "$WORK/inputs" "$WORK/bundle"
mkdir -p "$WORK/inputs/wheels"
IN="$WORK/inputs"

# 2b. extra_code/: the model's own code, and the slice of tt-metal it imports.
"$PYTHON" -I "$HERE/build_inputs.py" extra-code \
  --package-dir "$REPO/animatediff_ttnn" --version-file "$REPO/VERSION" \
  --metal-root "$METAL_SRC" --out "$IN/extra_code" --report "$IN/extra_code.report.json"

# 2d. model.py, requirements.txt, constraints.txt.
cp "$HERE/model.py" "$IN/model.py"
cp "$HERE/requirements.txt" "$IN/requirements.txt"
cp "$HERE/constraints.txt" "$IN/constraints.txt"

# 2c. Static import resolution, before any install. Every first-party import must exist in
# extra_code/. Third-party names are checked in the venv by --install-test below.
"$PYTHON" -I "$HERE/build_inputs.py" check --extra-code "$IN/extra_code" --model-py "$IN/model.py" \
  > "$IN/import-check.static.json" || { cat "$IN/import-check.static.json"; die "an import does not resolve (Step 2c)"; }
echo "static import check: no unresolved first-party import"

# The dependency carrier (see runtime-deps.txt). Versioned with the model so a new release is a
# new wheel name, and installed by path so the bytes are the ones tested.
DEPS_WHEEL="$("$PYTHON" -I "$HERE/build_inputs.py" deps-wheel --name "$DEPS_WHEEL_NAME" --version "$VERSION" \
  --requirements "$HERE/runtime-deps.txt" --out-dir "$IN/wheels")"
echo "deps wheel: $(basename "$DEPS_WHEEL")"

# 2f. Install guards, with the ttnn version filled in.
GUARDS=()
while IFS= read -r line; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue ;; esac
  GUARDS+=(--verify "${line//@TTNN_VERSION@/$TTNN_VERSION}")
done < "$HERE/install-guards.txt"
echo "${#GUARDS[@]} install guard arguments"

# Provenance: where each input came from, so a bundle can be traced to a commit and a tag.
GIT_COMMIT="$(git -C "$REPO" rev-parse HEAD)"
GIT_DIRTY="false"; [ -n "$(git -C "$REPO" status --porcelain -- animatediff_ttnn packaging VERSION)" ] && GIT_DIRTY="true"
export GIT_COMMIT GIT_DIRTY VERSION TTNN_VERSION TT_METAL_REF TT_METAL_COMMIT WEIGHTS_REPO WEIGHTS_REV PYTHON_VERSION
PROV_JSON="$("$PYTHON" -I - "$IN" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
inp = Path(sys.argv[1])
rep = json.loads((inp / "extra_code.report.json").read_text())
whl = next((inp / "wheels").glob("*.whl"))
print(json.dumps({
    "model": "tt-animatediff",
    "version": os.environ["VERSION"],
    "recipe": "packaging/package-thin.sh",
    "source_commit": os.environ["GIT_COMMIT"],
    "source_tree_dirty": os.environ["GIT_DIRTY"] == "true",
    "engine": {"ttnn": os.environ["TTNN_VERSION"], "source": "PyPI, stock"},
    "tt_metal_code": {
        "tag": os.environ["TT_METAL_REF"], "commit": os.environ["TT_METAL_COMMIT"],
        "files_shipped": len(rep["models_files"]),
        "note": "models/ files the model imports, copied unchanged; empty __init__.py added where upstream has none",
    },
    "extra_code_tree_sha256": rep["tree_sha256"],
    "deps_wheel": {"file": whl.name, "sha256": hashlib.sha256(whl.read_bytes()).hexdigest()},
    "weights": {"repo": os.environ["WEIGHTS_REPO"], "revision": os.environ["WEIGHTS_REV"]},
    "python": os.environ["PYTHON_VERSION"],
}, indent=2, sort_keys=True))
PY
)"
printf '%s\n' "$PROV_JSON" > "$IN/provenance.json"

# ============================================================================================
# 3. Stage. No positional repo id: this writes a folder and uploads nothing.
# ============================================================================================
# --weights-allow is repeatable, so build one flag per pattern.
ALLOW_ARGS=()
for pat in "${WEIGHTS_ALLOW[@]}"; do ALLOW_ARGS+=(--weights-allow "$pat"); done

# Flags that do not apply, and why:
#   --plugin-wheel, --vllm-*, --main-class, --arch-name   no vLLM: this kind serves an ASGI app
#   --tool-parser, --reasoning-parser                     not an LLM
#   --max-model-len, --max-num-seqs, --block-size,
#   --trace-region-bytes, --tt-config-json                 vLLM serving knobs
#   --mesh-fabric                                          one chip, no fabric
#   --readme                                               the Hub repo's README comes from
#                                                          scripts/build_hf_artifact.py
say "tt-model package-thin --out $WORK/bundle"
"$TT_MODEL" package-thin --out "$WORK/bundle" --name "$BUNDLE_NAME" \
  --kind tt-dit-server --app "$APP" --no-vllm \
  --model-py "$IN/model.py" --extra-code "$IN/extra_code" \
  --requirements "$IN/requirements.txt" --constraints "$IN/constraints.txt" \
  --python "$PYTHON_VERSION" --tt-metal-version "$TTNN_VERSION" \
  --ops-wheel "$DEPS_WHEEL" \
  --arch blackhole --mesh P150 --device-count 1 \
  --weights "$WEIGHTS_REPO" --weights-revision "$WEIGHTS_REV" "${ALLOW_ARGS[@]}" \
  --env ANIMATEDIFF_MESH_SHAPE=1x1 \
  "${GUARDS[@]}" --provenance "$IN/provenance.json"

# ============================================================================================
# 4. Inspect the staged bundle (Step 3 of the skill)
# ============================================================================================
# Reads files only. Fails if the bundle is not the canonical shape for this model.
say "inspect the staged bundle"
"$PYTHON" -I - "$WORK" "$WEIGHTS_REPO" "$WEIGHTS_REV" "$APP" "$(basename "$DEPS_WHEEL")" <<'PY' > "$WORK/expect.json"
import json, sys
work, repo, rev, app, wheel = sys.argv[1:6]
print(json.dumps({
    "kind": "tt-dit-server", "app": app,
    "custom_ops": [],                       # no generic_op wheel: every kernel is a stock ttnn op
    "wheels": ["wheels/" + wheel],          # only the dependency carrier
    "wheel_names": [wheel],
    "weights_repo": repo, "weights_revision": rev,
    "must_exist": ["model.py", "extra_code/animatediff_ttnn/server/app.py",
                   "extra_code/models/demos/vision/generative/stable_diffusion/wormhole/common.py",
                   "extra_code/VERSION", "constraints.txt", "install.sh", "run.sh", "PROVENANCE.json"],
    "must_not_exist": ["wheels/tt_animatediff_models_closure-0.77.0-py3-none-any.whl", "vllm_models", "metal"],
}))
PY
"$PYTHON" -I "$HERE/build_inputs.py" inspect --bundle "$WORK/bundle" --expect "$WORK/expect.json" > "$WORK/inspect.json" \
  || { cat "$WORK/inspect.json"; die "the staged bundle is not canonical (see PROBLEM lines above)"; }
echo "run.sh argv, one token per line:"; "$PYTHON" -I -c "import json,sys; [print('  [%s]' % t) for t in json.load(open(sys.argv[1]))['argv']]" "$WORK/inspect.json"
du -sh "$WORK/bundle" | sed 's/^/bundle size: /'

# ============================================================================================
# 5. --install-test: Step 4a/4b in a COPY. Needs network. Never touches hardware.
# ============================================================================================
# install.sh creates venv/, .python/ and caches, so it runs on a copy and the staged folder stays
# clean. The --verify guards run at the end of install.sh and include the import smoke test
# (guard 5: `import model`, and ttnn must not be loaded). Then every import is checked in the
# installed venv with find_spec, which locates ttnn without importing it.
if [ "$MODE" = install-test ]; then
  say "install.sh in a copy"
  COPY="${INSTALL_COPY:-$WORK/bundle-install-test}"
  rm -rf "$COPY"; cp -r "$WORK/bundle" "$COPY"
  bash "$COPY/install.sh" 2>&1 | tee "$WORK/install.log"
  say "import check in the installed venv"
  "$PYTHON" -I "$HERE/build_inputs.py" check --extra-code "$COPY/extra_code" --model-py "$COPY/model.py" \
    --venv-python "$COPY/venv/bin/python" | tee "$WORK/import-check.venv.json"
  echo "installed in $COPY. To serve it on a card (not done by this script): bash $COPY/run.sh --port 20000"
fi

say "staged: $WORK/bundle"
echo "Nothing was pushed. Read the header of this script before publishing."
