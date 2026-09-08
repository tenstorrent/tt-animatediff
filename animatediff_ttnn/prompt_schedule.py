# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Prompt travel (prompt scheduling) — pure-torch interpolation helpers.

"Prompt travel" lets the conditioning embedding morph across a video's frames
instead of every frame sharing one static prompt. The caller supplies a small
number of *keyframe* prompts pinned to specific frame indices; this module
interpolates a per-frame conditioning tensor across the whole timeline so the
generated content drifts smoothly from one keyframe to the next.

Design constraints:
    * NO ``ttnn`` import at module level — everything here is plain torch so it
      is unit-testable on any machine (no Blackhole hardware required). The
      hardware-facing wiring lives in ``temporal_attention.py`` /
      ``ttnn_motion_pipeline.py``; this module only does CPU tensor math.
    * The single-prompt path never touches this module, so backward
      compatibility is guaranteed by construction.
"""

from typing import List, Tuple

import torch


def parse_schedule(entries: List[str]) -> List[Tuple[int, str]]:
    """Parse ``FRAME:PROMPT`` CLI entries into sorted ``(frame_index, prompt)`` pairs.

    Turns e.g. ``["0:spring meadow", "16:snowfall"]`` into
    ``[(0, "spring meadow"), (16, "snowfall")]`` (sorted by frame index).

    The split is on the *first* colon only, so a prompt may itself contain
    colons (``"0:a city: at night"`` → ``(0, "a city: at night")``).

    Args:
        entries: List of ``"<int>:<prompt text>"`` strings.

    Returns:
        List of ``(frame_index, prompt)`` tuples sorted ascending by frame index.

    Raises:
        ValueError: if any entry has no colon or a non-integer frame index.
    """
    parsed: List[Tuple[int, str]] = []
    for entry in entries:
        if ":" not in entry:
            raise ValueError(
                f"Malformed prompt-schedule entry {entry!r}: expected 'FRAME:PROMPT' "
                "(missing ':')."
            )
        frame_str, _, prompt = entry.partition(":")
        frame_str = frame_str.strip()
        try:
            frame_index = int(frame_str)
        except ValueError:
            raise ValueError(
                f"Malformed prompt-schedule entry {entry!r}: frame index "
                f"{frame_str!r} is not an integer."
            )
        parsed.append((frame_index, prompt.strip()))
    parsed.sort(key=lambda kv: kv[0])
    return parsed


def interpolate_embeddings(
    keyframes: List[Tuple[int, torch.Tensor]],
    num_frames: int,
) -> List[torch.Tensor]:
    """Interpolate keyframe embeddings into one embedding per frame.

    Args:
        keyframes: List of ``(frame_index, embedding)`` pairs. ``embedding`` is
            the *per-prompt conditioning* tensor for that keyframe (e.g. the
            ``cond`` half of a CLIP encode). All embeddings must share a shape.
            The list need not be pre-sorted — it is sorted here by frame index.
        num_frames: Number of frames N in the output timeline.

    Returns:
        A list of exactly ``num_frames`` tensors, one per frame:

          * **Single keyframe** → that embedding for every frame (all identical;
            the same tensor object is reused — no copies).
          * **Before the first keyframe** → the first keyframe's embedding.
          * **After the last keyframe** → the last keyframe's embedding.
          * **Between two keyframes** at frames ``a < b`` → linear interpolation
            ``(1 - t) * emb_a + t * emb_b`` with ``t = (i - a) / (b - a)``.
            Endpoints are exact: at ``i == a`` the result is ``emb_a``, at
            ``i == b`` it is ``emb_b`` (returned as the original tensor objects,
            not a recomputed copy, so keyframe frames are byte-identical to the
            source embeddings).

    Raises:
        ValueError: if ``keyframes`` is empty or ``num_frames < 1``.
    """
    if not keyframes:
        raise ValueError("interpolate_embeddings requires at least one keyframe.")
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}.")

    # Sort defensively by frame index (callers may pass unsorted keyframes).
    ordered = sorted(keyframes, key=lambda kv: kv[0])
    indices = [idx for idx, _ in ordered]
    embeddings = [emb for _, emb in ordered]

    first_idx, last_idx = indices[0], indices[-1]

    out: List[torch.Tensor] = []
    for i in range(num_frames):
        # Clamp outside the keyframe span to the nearest endpoint (return the
        # original tensor object → exact, no float round-trip).
        if i <= first_idx:
            out.append(embeddings[0])
            continue
        if i >= last_idx:
            out.append(embeddings[-1])
            continue

        # Locate the enclosing keyframe pair (a <= i < b). indices is sorted.
        seg = 0
        while seg + 1 < len(indices) and indices[seg + 1] <= i:
            seg += 1
        a, b = indices[seg], indices[seg + 1]
        emb_a, emb_b = embeddings[seg], embeddings[seg + 1]

        if i == a:
            # Landed exactly on a keyframe → return it untouched (endpoint exact).
            out.append(emb_a)
            continue

        t = (i - a) / (b - a)
        out.append((1.0 - t) * emb_a + t * emb_b)

    return out
