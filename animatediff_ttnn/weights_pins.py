# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Pinned Hugging Face revisions for every upstream weights repo this package loads.

Why this module exists
----------------------
Every ``from_pretrained`` / ``hf_hub_download`` in this package used to load a repo by
id alone, which resolves that repo's default branch *at load time*. A published bundle
would then serve whatever the upstream owner pushed most recently. That need not be
the weights it was measured on, and the bundle manifest's own ``weights.revision``
could not change it: the manifest steers only what ``tt-model pull`` pre-fetches, not
what the server loads.

This module is the one place that decides which revision each repo resolves to. Every
weights load in ``animatediff_ttnn`` passes ``revision=revision_for(repo_id)``.

Rules (``revision_for``)
------------------------
* **SD 1.4** (``CompVis/stable-diffusion-v1-4``) is the served model: the only weights
  the tt-dit-server path loads. tt-model-manager's v6 ``run.sh`` exports
  ``TT_MODEL_WEIGHTS_REVISION`` from the bundle manifest, and that env var wins when set
  and non-empty. Otherwise the pin below applies.
* **The motion adapter and the Lightning checkpoints** are used only by the CPU
  diffusers path, the Phase 3 CLI and the Gradio UI. They get their own pins.
  ``TT_MODEL_WEIGHTS_REVISION`` names *the bundle's* weights repo (SD 1.4), and applying
  that sha to a different repo would fail to resolve, so it is deliberately ignored for
  them.
* **A local directory** has no revisions, so the result is ``None`` (no pin is implied).
* **Any other repo id** (a caller-supplied ``model_id`` override) also gets ``None`` and
  resolves its default branch, as before. There is no sha we could honestly claim for it.

The shas are the ``sha`` field from the HF model API on 2026-09-27. All three repos were
last modified in 2023 or early 2024, long before this port's measurements, so each pin
is the revision those measurements used.

Importing this module touches nothing: no torch, no ttnn, no network.
"""

from __future__ import annotations

import os
from typing import Optional

#: Env var tt-model-manager's v6 run.sh exports with the bundle manifest's weights
#: revision. Applies to SD14_REPO only (see module docstring).
WEIGHTS_REVISION_ENV = "TT_MODEL_WEIGHTS_REVISION"

#: The served model. lastModified 2023-08-23.
SD14_REPO = "CompVis/stable-diffusion-v1-4"
SD14_REVISION = "133a221b8aa7292a167afc5127cb63fb5005638b"

#: AnimateDiff v1.5.2 motion adapter (CPU diffusers path, Phase 3 CLI). lastModified
#: 2023-11-03.
MOTION_ADAPTER_REPO = "guoyww/animatediff-motion-adapter-v1-5-2"
MOTION_ADAPTER_REVISION = "6167b88ffe39b4441fdf2113e77b99a6f56b7906"

#: ByteDance AnimateDiff-Lightning distilled adapters (CPU Lightning path only).
LIGHTNING_REPO = "ByteDance/AnimateDiff-Lightning"
LIGHTNING_REVISION = "027c893eec01df7330f5d4b733bc9485ee02e8b2"

#: Fixed pins for the repos the env var does not govern.
_FIXED_PINS = {
    MOTION_ADAPTER_REPO: MOTION_ADAPTER_REVISION,
    LIGHTNING_REPO: LIGHTNING_REVISION,
}


def sd14_revision() -> str:
    """``$TT_MODEL_WEIGHTS_REVISION`` if set and non-empty, else ``SD14_REVISION``.

    ``or`` rather than a ``.get()`` default so that an exported-but-empty variable
    cannot become ``revision=""``.
    """
    return os.environ.get(WEIGHTS_REVISION_ENV) or SD14_REVISION


def revision_for(repo_id: str) -> Optional[str]:
    """The revision to pass to ``from_pretrained`` / ``hf_hub_download`` for ``repo_id``.

    Returns ``None`` for a local directory or an unrecognised repo id. ``revision=None``
    is what those functions default to, so passing it through unconditionally is safe.
    """
    if os.path.isdir(repo_id):
        return None
    if repo_id == SD14_REPO:
        return sd14_revision()
    return _FIXED_PINS.get(repo_id)
