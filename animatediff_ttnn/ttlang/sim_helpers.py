# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Helpers to convert between PyTorch tensors and TT-Lang simulator Blocks.

Mirrors the pattern from tt-lang/examples/wan_rmsnorm.py.

The TT-Lang simulator tiles 2-D data into 32×32 chunks and stores them in a
Block (a structured list of Tensor objects).  Block.from_list() expects tiles in
row-major order: tile (r, c) comes before tile (r, c+1) which comes before
(r+1, 0).  block_to_tensor() inverts that layout exactly.
"""
import torch

# Standard Tensix hardware tile edge length in elements.
TILE_SIZE = 32


def tensor_to_block(data: torch.Tensor, shape: tuple):
    """Convert a 2-D float tensor to a sim Block with the given tile-grid shape.

    Slices *data* into (TILE_SIZE × TILE_SIZE) tiles in row-major order and
    wraps each in a sim.ttnnsim.Tensor before passing the list to
    Block.from_list().

    Args:
        data:  Float32 tensor of shape (M_tiles*TILE_SIZE, N_tiles*TILE_SIZE).
        shape: Tile-grid shape (M_tiles, N_tiles).  Must satisfy
               data.shape == (M_tiles*32, N_tiles*32).

    Returns:
        sim.dfb.Block with tile-grid shape *shape*.

    Raises:
        AssertionError: if *data* dimensions do not match *shape*.
    """
    from sim.dfb import Block
    from sim.ttnnsim import Tensor

    M_tiles, N_tiles = shape
    assert data.shape == (M_tiles * TILE_SIZE, N_tiles * TILE_SIZE), (
        f"Expected data shape {(M_tiles * TILE_SIZE, N_tiles * TILE_SIZE)}, "
        f"got {tuple(data.shape)}"
    )

    # Build tile list in row-major order (matches Block.from_list convention).
    tiles = []
    for r in range(M_tiles):
        for c in range(N_tiles):
            tile = data[
                r * TILE_SIZE:(r + 1) * TILE_SIZE,
                c * TILE_SIZE:(c + 1) * TILE_SIZE,
            ]
            # Clone so the Block owns its own storage; cast to float32.
            tiles.append(Tensor(tile.clone().float()))
    return Block.from_list(tiles, shape=shape)


def block_to_tensor(block) -> torch.Tensor:
    """Reconstruct a 2-D float tensor from a sim Block.

    Inverts the tile-major layout produced by Block.from_list():

        tile_grid (TM, TN, tile_h, tile_w)
          → permute(0, 2, 1, 3)  →  (TM, tile_h, TN, tile_w)
          → reshape               →  (TM*tile_h, TN*tile_w)

    Args:
        block: sim.dfb.Block with 2-D tile-grid shape (M_tiles, N_tiles).

    Returns:
        Float32 tensor of shape (M_tiles*TILE_SIZE, N_tiles*TILE_SIZE).
    """
    TM, TN = block._shape  # tile-grid dimensions
    raw = [t.to_torch() for t in block.to_list()]
    tile_h, tile_w = raw[0].shape
    # Stack into (TM*TN, tile_h, tile_w) then view as 4-D tile grid.
    grid = torch.stack(raw).reshape(TM, TN, tile_h, tile_w)
    # Permute so row-tiles and row-elements are adjacent, then flatten.
    return grid.permute(0, 2, 1, 3).contiguous().reshape(TM * tile_h, TN * tile_w)
