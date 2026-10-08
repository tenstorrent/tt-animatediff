# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""LoRA (Low-Rank Adaptation) support for tt-animatediff.

This module enables loading and merging LoRA adapters with the MotionAdapter,
allowing for style-specific video generation without retraining the full model.

Usage:
    from animatediff_ttnn import generate_animation
    from animatediff_ttnn.lora import load_lora_adapter

    adapter = load_lora_adapter(
        base_adapter_id="guoyww/animatediff-motion-adapter-v1-5-2",
        lora_adapter_id="my-org/anime-motion-lora"
    )

    frames = generate_animation(
        prompt="a girl running in anime style",
        motion_adapter=adapter
    )
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    import torch
    from diffusers import MotionAdapter
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    torch = None
    MotionAdapter = None
    PeftModel = None

# Cache directory for merged adapters
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "tt-animatediff" / "lora"


def _get_cache_dir() -> Path:
    """Get the cache directory for merged LoRA adapters.
    
    Respects LORA_CACHE_DIR environment variable.
    """
    env_cache = os.environ.get("LORA_CACHE_DIR")
    if env_cache:
        return Path(env_cache)
    return _DEFAULT_CACHE_DIR


def _ensure_cache_dir() -> Path:
    """Ensure the cache directory exists."""
    cache_dir = _get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def load_lora_adapter(
    base_adapter_id: str,
    lora_adapter_id: str,
    lora_alpha: float = 1.0,
    device: str = "cpu",
) -> MotionAdapter:
    """Load a LoRA adapter and merge it into the base adapter.
    
    This function loads the base MotionAdapter, applies the LoRA adapter,
    merges them, and returns the combined adapter ready for use.
    
    Args:
        base_adapter_id: HuggingFace repo ID for the base MotionAdapter
            (e.g., "guoyww/animatediff-motion-adapter-v1-5-2")
        lora_adapter_id: HuggingFace repo ID or local path for the LoRA adapter
            (e.g., "my-org/anime-motion-lora" or "/path/to/lora")
        lora_alpha: LoRA alpha parameter (scaling factor)
        device: Device to load adapters on (default: "cpu")
        
    Returns:
        Merged MotionAdapter ready for use with tt-animatediff
        
    Raises:
        ImportError: If PEFT is not installed
        FileNotFoundError: If adapter cannot be loaded
        
    Example:
        ```python
        from animatediff_ttnn import generate_animation
        from animatediff_ttnn.lora import load_lora_adapter
        
        adapter = load_lora_adapter(
            base_adapter_id="guoyww/animatediff-motion-adapter-v1-5-2",
            lora_adapter_id="my-org/anime-motion-lora"
        )
        
        frames = generate_animation(
            prompt="a girl running in anime style",
            motion_lora=adapter
        )
        ```
    """
    if not PEFT_AVAILABLE:
        raise ImportError(
            "PEFT is required for LoRA support. "
            "Install with: pip install peft"
        )
    
    cache_dir = _ensure_cache_dir()
    
    # Create unique cache path for this adapter combination
    cache_key = f"{base_adapter_id.replace('/', '--')}+{lora_adapter_id.replace('/', '--')}"
    cache_path = cache_dir / cache_key
    
    # Check if already merged and cached
    if cache_path.exists():
        # Load cached merged adapter
        base = MotionAdapter.from_pretrained(
            base_adapter_id,
            torch_dtype=torch.float32,
        )
        merged = PeftModel.from_pretrained(
            base,
            str(cache_path),
            torch_dtype=torch.float32,
        ).merge_and_unload()
        return merged
    
    # Load base adapter
    base = MotionAdapter.from_pretrained(
        base_adapter_id,
        torch_dtype=torch.float32,
    )
    
    # Load LoRA adapter
    lora_model = PeftModel.from_pretrained(
        base,
        lora_adapter_id,
        torch_dtype=torch.float32,
        is_trainable=False,
    )
    
    # Merge LoRA into base
    merged = lora_model.merge_and_unload()
    
    # Save merged adapter to cache
    merged.save_pretrained(str(cache_path))
    
    return merged


def merge_and_cache(
    base_adapter_id: str,
    lora_adapter_id: str,
    output_path: Optional[str] = None,
    lora_alpha: float = 1.0,
) -> Path:
    """Merge LoRA into base adapter and save to cache.
    
    This function merges a LoRA adapter into a base MotionAdapter and saves
    the result to the specified output path (or cache directory if not specified).
    
    Args:
        base_adapter_id: HuggingFace repo ID for the base MotionAdapter
        lora_adapter_id: HuggingFace repo ID or local path for the LoRA adapter
        output_path: Optional custom output path (defaults to cache directory)
        lora_alpha: LoRA alpha parameter (scaling factor)
        
    Returns:
        Path to the merged adapter
        
    Raises:
        ImportError: If PEFT is not installed
        FileNotFoundError: If adapter cannot be loaded
    """
    if not PEFT_AVAILABLE:
        raise ImportError(
            "PEFT is required for LoRA support. "
            "Install with: pip install peft"
        )
    
    if output_path is None:
        output_path = str(_ensure_cache_dir() / 
                         f"{base_adapter_id.replace('/', '--')}+{lora_adapter_id.replace('/', '--')}")
    
    # Load and merge
    base = MotionAdapter.from_pretrained(
        base_adapter_id,
        torch_dtype=torch.float32,
    )
    
    lora_model = PeftModel.from_pretrained(
        base,
        lora_adapter_id,
        torch_dtype=torch.float32,
        is_trainable=False,
    )
    
    merged = lora_model.merge_and_unload()
    merged.save_pretrained(output_path)
    
    return Path(output_path)


def get_cached_adapters() -> list[str]:
    """List all cached LoRA adapters.
    
    Returns:
        List of adapter names (cache keys)
    """
    cache_dir = _get_cache_dir()
    if not cache_dir.exists():
        return []
    
    adapters = []
    for item in cache_dir.iterdir():
        if item.is_dir() and (item / "adapter_config.json").exists():
            adapters.append(item.name)
    
    return sorted(adapters)


def clear_cache() -> None:
    """Clear all cached LoRA adapters.
    
    WARNING: This removes all cached merged adapters.
    """
    cache_dir = _get_cache_dir()
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)


# Expose PEFT for advanced users
def get_peft_config(
    r: int = 16,
    lora_alpha: int = 32,
    target_modules: Optional[list[str]] = None,
    lora_dropout: float = 0.0,
    bias: str = "none",
) -> "peft.LoraConfig":
    """Create a PEFT LoraConfig for training custom adapters.
    
    This function is provided for users who want to train their own
    LoRA adapters. It creates a standard configuration suitable for
    MotionAdapter fine-tuning.
    
    Args:
        r: LoRA rank (lower = smaller adapter, less capacity)
        lora_alpha: LoRA alpha parameter (scaling factor)
        target_modules: List of modules to apply LoRA to (default: attention layers)
        lora_dropout: Dropout probability
        bias: Bias handling ("none", "all", or "lora_only")
        
    Returns:
        LoraConfig for use with get_peft_model()
        
    Raises:
        ImportError: If PEFT is not installed
        
    Example:
        ```python
        from animatediff_ttnn.lora import get_peft_config, train_lora_adapter
        from diffusers import MotionAdapter
        from peft import get_peft_model
        
        config = get_peft_config(r=16, lora_alpha=32)
        base = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
        lora = get_peft_model(base, config)
        # Train lora on your dataset...
        ```
    """
    if not PEFT_AVAILABLE:
        raise ImportError(
            "PEFT is required for LoRA support. "
            "Install with: pip install peft"
        )
    
    from peft import LoraConfig
    
    if target_modules is None:
        # Default: apply to attention layers
        target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
    
    return LoraConfig(
        task_type="CAUSAL_LM",
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
    )
