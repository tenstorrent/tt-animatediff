# Releasing

Short, because most of it is enforced by tests. The steps that are *not* enforced are
marked ⚠️ — those are the ones that get forgotten.

## Order matters

Every tag in this repo is reachable from `main` (`v0.1.0`, `v0.6.0`, `v0.9.0`, `v0.10.0`).
Keep it that way: **merge first, then tag**. A tag cut on a feature branch is not reachable
from `main` at all if the PR is squash-merged, and nothing in CI would tell you.

## Checklist

1. **`VERSION`** holds the version being released, with no `v` prefix (`0.11.0`). `setup.py`
   reads this file, so it is the single source of truth for the package version — there is
   no second copy to keep in sync.

2. **README changelog.** Prepare the entry as `### vX.Y.Z — unreleased` while the work is in
   review, then change `unreleased` to the release date when you cut it. Dated entries are
   the convention; an undated one that has shipped is a lie a reader cannot detect.

3. **Merge to `main`.**

4. **Tag on `main`:**
   ```bash
   git checkout main && git pull
   git tag vX.Y.Z && git push origin vX.Y.Z
   ```

5. ⚠️ **Repin `tt_model_package.yaml`.** `source.extra_code[].root.ref` tells
   tt-model-manager which commit to clone `animatediff_ttnn/` from at *package* time, so it
   must name a ref that carries the code being shipped. Move it to the tag you just cut:
   ```yaml
       - root:
           repo: https://github.com/tenstorrent/tt-animatediff
           ref: vX.Y.Z
   ```
   Two things about this pin are easy to get wrong:

   - It can never name the tag that contains itself — the manifest is *inside* the release
     it would be pinning — so it will always name the previous tag until the next release
     moves it. That is expected, and harmless, because the pin only has to be correct about
     `animatediff_ttnn/`.
   - It therefore only needs to move when `animatediff_ttnn/` itself changes, not on every
     release.

   `tests/test_server_app.py::test_the_pinned_extra_code_ref_ships_the_package_we_are_developing`
   compares the pinned ref's `animatediff_ttnn/` tree against `HEAD`'s and fails when they
   diverge. It **skips** on a shallow clone (CI checks out at `fetch-depth: 1`, so the ref's
   objects are not there), which is why this step is marked ⚠️: run the suite in a full local
   clone after repinning, and read the skips — `pytest -rs` prints them.

6. **Publish to the Hub** (only when the pipeline or model card changed):
   ```bash
   python scripts/publish_to_hub.py --dry-run   # preview
   python scripts/publish_to_hub.py --yes       # upload
   python scripts/publish_to_hub.py --verify    # read-only round-trip of the PUBLISHED copy
   ```
   Repos are created **private** and the script has no `--public` flag by design; flipping
   visibility is a separate, deliberately-confirmed action. `episod/tt-animatediff` is
   currently public. The demo Space cannot be created on this account at all —
   `create_repo(repo_type="space")` returns 402, because Gradio Spaces on free cpu-basic
   need PRO. See CLAUDE.md for the detail.

## What is already enforced, so you need not check it by hand

- the manifest's `runtime.app` resolves, and the module imports with **no ttnn**
- the allowlist covers every `models.*` import under `animatediff_ttnn/`, and carries
  nothing dead
- `runtime.mesh_shape_env` matches the constant the server reads
- `mesh_device` names a single chip, because the model cannot shard
- the committed benchmark sample matches the schema `scripts/benchmark_serving.py` emits
