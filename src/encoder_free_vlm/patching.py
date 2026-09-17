from __future__ import annotations

import torch


def extract_flattened_patches(pixel_values: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Extract flattened non-overlapping patches with reshape/permute only."""
    if pixel_values.ndim != 4:
        raise ValueError("pixel_values must have shape (batch, channels, height, width)")

    batch_size, channels, height, width = pixel_values.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("height and width must be divisible by patch_size")

    patches_h = height // patch_size
    patches_w = width // patch_size
    x = pixel_values.reshape(batch_size, channels, patches_h, patch_size, patches_w, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)
    return x.reshape(batch_size, patches_h * patches_w, channels * patch_size * patch_size)
