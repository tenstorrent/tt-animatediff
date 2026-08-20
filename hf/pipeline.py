# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Diffusers custom pipeline for tt-animatediff.

Loaded from the Hugging Face Hub with::

    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        "episod/tt-animatediff",
        custom_pipeline="episod/tt-animatediff",
        trust_remote_code=True,
    )
    frames = pipe("a swirling nebula, teal and gold").frames

This module carries NO generation logic. It marshals arguments into
``animatediff_ttnn.generate_animation()`` — the tested entry point that owns
backend selection, the CPU fallback, the device singleton and chain mode — and
wraps the result in a diffusers-shaped output object.

Two constraints here are imposed by how diffusers loads custom pipelines, and
both are easy to "fix" back into breakage:

1. ``__init__`` declares named parameters with defaults and takes no
   ``**kwargs``. diffusers derives its expected-component list from this
   signature; a ``**kwargs``-only signature fails the load outright.
2. The string ``animatediff_ttnn`` never appears in an ``import`` statement,
   at any indentation. diffusers' ``check_imports`` regex-scans this file and,
   on diffusers 0.32.x, raises ImportError at *load* time for any module that
   is not installed — which would lock out every user who has not already
   pip-installed this package, i.e. exactly the audience the CPU path serves.
   The package is reached through ``importlib.import_module`` instead.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput

# numpy and torch are imported here rather than in the methods that use them
# (output_type="np" and the @torch.no_grad() decorator, both added in Task 3):
# check_imports resolves them fine because diffusers depends on both, and a
# lazy import inside a method would be scanned identically anyway.

#: Name of the package this pipeline delegates to. Never written as an import.
PACKAGE_NAME = "animatediff_ttnn"

#: Compute backends accepted by ``__call__``; forwarded to generate_animation().
VALID_MODES = ("auto", "blackhole", "cpu", "sim")


#: Where a user without the package installed is told to get it.
_INSTALL_HINT = (
    "pip install 'animatediff-ttnn @ "
    "git+https://github.com/tenstorrent/tt-animatediff'"
)


def _import_package():
    """Import the delegate package if it is already importable."""
    return importlib.import_module(PACKAGE_NAME)


def _import_from_root(root: Path):
    """Put ``root`` on sys.path and import the package from it, or return None.

    ``root`` is a directory *containing* an ``animatediff_ttnn/`` package —
    either the directory from_pretrained loaded from, or a Hub snapshot.
    """
    if not (root / PACKAGE_NAME / "__init__.py").is_file():
        return None
    # Idempotent insert. In the happy path this runs at most once per process
    # (a successful import lands the package in sys.modules, so layer 1 of
    # resolve_package short-circuits every later call), but if the import
    # itself fails the entry is already on sys.path — a retry would otherwise
    # stack a duplicate and shift import precedence.
    resolved = str(root.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    importlib.invalidate_caches()
    return importlib.import_module(PACKAGE_NAME)


def _snapshot_download(repo_id: str, allow_patterns=None, **kwargs) -> str:
    """Fetch just the vendored package from the Hub. Seam for tests."""
    hub = importlib.import_module("huggingface_hub")
    return hub.snapshot_download(
        repo_id=repo_id, allow_patterns=allow_patterns, **kwargs
    )


def _ttnn_available() -> bool:
    """True when the TTNN runtime can be imported in this process.

    Broad except on purpose: a half-installed tt-metal raises things other than
    ImportError, and every one of them means "no Blackhole backend here".
    """
    try:
        importlib.import_module("ttnn")
        return True
    except Exception:
        return False


def resolve_package(code_repo: Optional[str], source: Optional[str]):
    """Return the animatediff_ttnn module, making it importable if needed.

    Three layers, cheapest first:

    1. The installed package — the documented path for a Blackhole user, who
       has this repo checked out and installed anyway.
    2. The directory ``from_pretrained`` loaded from (``config._name_or_path``).
       diffusers executes this file out of its own module cache, so the
       vendored copy is NOT a sibling of ``__file__`` and this indirection is
       the only way to find it after a local-directory load.
    3. The Hub copy of the vendored package (a few hundred KB of pure Python,
       no weights). This is what lets ``pipe(...)`` work on a machine that has
       only diffusers installed. It is the first step here that touches the
       network, and it happens at generation time — never during
       ``from_pretrained``.

    Raises:
        ImportError: every layer failed; the message carries the install command.
    """
    try:
        return _import_package()
    except ModuleNotFoundError as exc:
        # Only treat it as "package not installed" if it's actually the animatediff_ttnn
        # package that failed to import. If exc.name is None or names a different
        # module (e.g. a transitive dependency like ttnn), it's a real import error
        # in an installed package and layers 2/3 cannot fix it — propagate instead.
        if exc.name != PACKAGE_NAME:
            raise

    if source:
        candidate = Path(source)
        if candidate.is_dir():
            module = _import_from_root(candidate)
            if module is not None:
                return module

    if code_repo:
        try:
            snapshot = _snapshot_download(
                code_repo, allow_patterns=[f"{PACKAGE_NAME}/**"]
            )
        except Exception as exc:  # network down, gated repo, bad id
            raise ImportError(
                f"{PACKAGE_NAME} is not installed and the vendored copy could not "
                f"be fetched from {code_repo!r} ({exc}). Install it with:\n"
                f"    {_INSTALL_HINT}"
            ) from exc
        module = _import_from_root(Path(snapshot))
        if module is not None:
            return module

    raise ImportError(
        f"{PACKAGE_NAME} could not be imported, found next to the loaded "
        f"pipeline, or fetched from the Hub. Install it with:\n    {_INSTALL_HINT}"
    )


@dataclass
class TTAnimateDiffPipelineOutput(BaseOutput):
    """Stand-in for diffusers' ``AnimateDiffPipelineOutput``.

    ``frames`` is a list of PIL Images, or a stacked numpy array when
    ``output_type="np"`` was requested.
    """

    frames: Any


class TTAnimateDiffPipeline(DiffusionPipeline):
    """AnimateDiff on Tenstorrent Blackhole, with a CPU fallback.

    Construction loads nothing: no device is opened and no weights are fetched
    until ``__call__``. That keeps ``from_pretrained`` fast and offline-safe.
    """

    def __init__(
        self,
        base_model: str = "CompVis/stable-diffusion-v1-4",
        motion_adapter: str = "guoyww/animatediff-motion-adapter-v1-5-2",
        lightning_repo: str = "ByteDance/AnimateDiff-Lightning",
        code_repo: str = "episod/tt-animatediff",
        temporal_alpha: float = 0.35,
        num_frames: int = 8,
        num_steps: int = 25,
        guidance_scale: float = 7.5,
    ):
        """Store config. Opens no device and fetches nothing (see class docstring).

        ``base_model``, ``motion_adapter``, and ``lightning_repo`` are
        declarative metadata only: they record which upstream weights the
        backend resolves, mirroring the hardcoded defaults in
        ``animatediff_ttnn.pipeline`` and ``animatediff_ttnn.generation_helpers``.
        ``__call__`` never reads them and passes nothing derived from them to
        ``generate_animation()``. They are NOT injection points — passing a
        different value here (e.g. ``base_model="other/model"``) is accepted,
        persists in ``self.config``, and changes no generation behaviour
        whatsoever. Swapping the actual upstream weights requires a code
        change in ``animatediff_ttnn``, not a config override here.
        """
        super().__init__()
        self.register_to_config(
            base_model=base_model,
            motion_adapter=motion_adapter,
            lightning_repo=lightning_repo,
            code_repo=code_repo,
            temporal_alpha=temporal_alpha,
            num_frames=num_frames,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
        )
        self._resolved_mode: Optional[str] = None

    @property
    def resolved_mode(self) -> Optional[str]:
        """Backend the last ``__call__`` actually ran on, or None before one.

        Lets a caller state which backend ran instead of guessing — notably the
        Space, where ``mode="auto"`` always lands on CPU.
        """
        return self._resolved_mode

    def _resolve_mode(self, mode: str) -> str:
        """Map the requested backend onto one generate_animation() accepts.

        ``auto`` is the only mode that falls back silently — that is its
        documented purpose. An explicit ``blackhole`` request raises instead,
        because a caller who asked for hardware needs to know they did not get
        it rather than quietly waiting minutes per frame on CPU.
        """
        if mode not in VALID_MODES:
            raise ValueError(
                f"mode must be one of {VALID_MODES}, got {mode!r}"
            )
        if mode == "auto":
            return "blackhole" if _ttnn_available() else "cpu"
        if mode == "blackhole" and not _ttnn_available():
            raise RuntimeError(
                "mode='blackhole' was requested but the ttnn runtime is not "
                "importable. Install tt-metal and activate its python_env "
                "(source ~/tt-metal/python_env/bin/activate), or use "
                "mode='cpu' / mode='auto'. See "
                "https://github.com/tenstorrent/tt-animatediff#modes-reference"
            )
        return mode

    def _config_default(self, value, key):
        """Fall back to the configured default when an argument was omitted."""
        return self.config[key] if value is None else value

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_frames: Optional[int] = None,
        num_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: int = 42,
        temporal_alpha: Optional[float] = None,
        height: int = 512,
        width: int = 512,
        mode: str = "auto",
        sim_so: Optional[str] = None,
        use_lightning: bool = False,
        lightning_steps: int = 4,
        chain_from: Optional[str] = None,
        chain_save: Optional[str] = None,
        chain_alpha: float = 0.6,
        output_type: str = "pil",
    ) -> TTAnimateDiffPipelineOutput:
        """Generate an animation.

        Arguments left as None fall back to the values in ``model_index.json``,
        so ``pipe("a nebula")`` is a complete call.

        ``height``, ``width``, ``temporal_alpha``, and the ``chain_*`` arguments
        (``chain_from``, ``chain_save``, ``chain_alpha``) are Blackhole/sim-only.
        The CPU backend (``animatediff_ttnn._generate_cpu()``) accepts no
        height/width parameters at all and is hardwired to 512x512, and it has
        no cross-frame temporal-attention or chain-mode support either — all
        of these are silently ignored on ``mode="cpu"``. This is a real
        API-surface gotcha: passing them on CPU does not raise, it just does
        nothing, so this docstring is the only place a caller learns it.

        Args:
            mode: "auto" (Blackhole if ttnn imports, else CPU), "blackhole"
                (require hardware), "cpu", or "sim" (ttsim virtual device).
            sim_so: Path to ``libttsim_bh.so``, used only when mode="sim".
                The backend defaults to ``~/sim/libttsim_bh.so`` when this is
                None, so mode="sim" silently depends on the binary sitting at
                that exact path unless you pass this.
            output_type: "pil" for PIL Images, "np" for a stacked
                ``[frames, H, W, 3]`` float array in [0, 1].

        Returns:
            TTAnimateDiffPipelineOutput with ``.frames``.

        Raises:
            ValueError: unknown ``mode`` or ``output_type``.
            RuntimeError: ``mode="blackhole"`` with no ttnn runtime, or a
                device/weight-load failure propagated from the backend.
            ImportError: the animatediff_ttnn package could not be resolved.
        """
        if output_type not in ("pil", "np"):
            raise ValueError(
                f"output_type must be 'pil' or 'np', got {output_type!r}"
            )

        resolved = self._resolve_mode(mode)
        package = resolve_package(
            self.config.get("code_repo"), self.config.get("_name_or_path")
        )

        frames = package.generate_animation(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=self._config_default(num_frames, "num_frames"),
            num_steps=self._config_default(num_steps, "num_steps"),
            guidance_scale=self._config_default(guidance_scale, "guidance_scale"),
            seed=seed,
            temporal_alpha=self._config_default(temporal_alpha, "temporal_alpha"),
            height=height,
            width=width,
            mode=resolved,
            sim_so=sim_so,
            use_lightning=use_lightning,
            lightning_steps=lightning_steps,
            chain_from=chain_from,
            chain_save=chain_save,
            chain_alpha=chain_alpha,
        )

        # Only record the backend once generation actually succeeded, so
        # resolved_mode never reports a run that did not happen.
        self._resolved_mode = resolved

        if output_type == "np":
            return TTAnimateDiffPipelineOutput(
                frames=np.stack([np.asarray(f, dtype=np.float32) / 255.0 for f in frames])
            )
        # list() is a deliberate copy, not redundancy: the backend owns the
        # sequence it returned, and callers mutate .frames (the Space appends
        # to it when writing a GIF). Copying keeps those two from aliasing.
        return TTAnimateDiffPipelineOutput(frames=list(frames))
