# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Entry point of the tt-animatediff v6 thin bundle.

``run.sh`` serves ``model:app`` with uvicorn (``--app model:app``). The server itself is
``animatediff_ttnn.server.app``, which ships in the bundle under ``extra_code/`` (run.sh puts
``extra_code/`` first on PYTHONPATH). This file only gives the bundle a root-level module that
``package-thin`` requires, and makes the served object the one the repo's tests exercise.

No custom-op import here, on purpose. The bundle ships no ``generic_op`` wheel: every kernel this
model runs is a stock ``ttnn`` op. ``deps.custom_ops`` is recorded from this file's imports, so
the empty list in the manifest is the accurate answer.

Importing this module must not import ``ttnn``. ``animatediff_ttnn.server.app`` defers every
``ttnn`` import into functions so the module loads on a host with no card, and the install guard in
``packaging/package-thin.sh`` checks that ``ttnn`` is still absent from ``sys.modules`` afterwards.
"""

from animatediff_ttnn.server.app import app  # noqa: F401  (the ASGI object uvicorn serves)

__all__ = ["app"]
