# Copyright 2025 FlagScale Authors.
# Licensed under the Apache License, Version 2.0.

"""
Mask utilities for Pi05 inference.

These helpers ensure image/language masks align with tokenized prefixes,
preventing padding tokens from being treated as valid inputs.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch


def expand_image_masks(
    image_masks: Sequence[torch.Tensor],
    token_lengths: Sequence[int],
) -> torch.Tensor:
    """
    Expand per-image masks [B] to per-token masks [B, sum(T_i)].

    Args:
        image_masks: list of [B] bool masks, one per image.
        token_lengths: token length per image, same order as image_masks.

    Returns:
        [B, sum(token_lengths)] bool mask.
    """
    if len(image_masks) != len(token_lengths):
        raise ValueError("image_masks and token_lengths must have same length.")

    expanded = []
    for mask, length in zip(image_masks, token_lengths):
        if mask.dim() != 1:
            raise ValueError("Each image_mask must be shape [B].")
        expanded.append(mask[:, None].expand(mask.shape[0], length))
    return torch.cat(expanded, dim=1)


def expand_image_masks_from_tokens(
    image_masks: Sequence[torch.Tensor],
    image_tokens: Sequence[torch.Tensor],
) -> torch.Tensor:
    """
    Expand per-image masks [B] to per-token masks using image token lengths.

    Args:
        image_masks: list of [B] bool masks.
        image_tokens: list of [B, T_i, D] tokens (or any tensor with shape[1]=T_i).

    Returns:
        [B, sum(T_i)] bool mask.
    """
    lengths = [int(t.shape[1]) for t in image_tokens]
    return expand_image_masks(image_masks, lengths)
