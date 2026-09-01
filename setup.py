#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Setup script for AnimateDiff TT-Metal integration."""

from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).parent
long_description = (here / "README.md").read_text() if (here / "README.md").exists() else ""
version = (here / "VERSION").read_text().strip()

setup(
    name="animatediff-ttnn",
    version=version,
    author="Tenstorrent Community",
    author_email="",
    description="AnimateDiff video generation on Tenstorrent Blackhole via TTNN UNet (SD 1.4)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tenstorrent/tt-animatediff",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "Pillow>=9.0.0",
        "diffusers>=0.32.1",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "huggingface_hub>=0.20.0",  # Lightning checkpoint download + VAE weights
        "safetensors>=0.4.0",        # Lightning adapter .safetensors loading
        # tt-metal and ttnn must be installed separately (not on PyPI)
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            # fastapi's TestClient is a thin wrapper over httpx; without it the server
            # tests fail at import rather than skipping.
            "httpx>=0.24.0",
            # tt-lang's functional simulator (sim.ttnnsim) imports greenlet.
            # Without it tests/test_ttlang_temporal_attention.py silently
            # importorskip()s its 9 simulator tests instead of running them.
            "greenlet>=3.0.0",
        ],
        "ui": [
            "gradio>=4.0.0",
        ],
        # The ASGI serving surface (animatediff_ttnn/server/). Matches the packages
        # tt-model-manager's tt-dit-server kind installs for this app, so a bundle and a
        # local `pip install -e .[serve]` run the same stack.
        "serve": [
            "fastapi>=0.110.0",
            "uvicorn>=0.27.0",
            "pydantic>=2.0.0",
        ],
    },
    entry_points={},
)
