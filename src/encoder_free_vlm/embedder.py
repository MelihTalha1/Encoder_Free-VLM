from __future__ import annotations

import torch
from torch import nn

from .config import VisionConfig
from .patching import extract_flattened_patches


class VisionPatchEmbedder(nn.Module):
    """Tiny encoder-free image embedder used before the decoder."""

    def __init__(self, vision_config: VisionConfig, hidden_size: int) -> None:
        super().__init__()
        self.vision_config = vision_config
        self.hidden_size = hidden_size
        patch_dim = vision_config.flattened_patch_dim
        grid_size = vision_config.grid_size

        self.ln1 = nn.LayerNorm(patch_dim)
        self.fc = nn.Linear(patch_dim, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.y_pos_emb = nn.Parameter(torch.zeros(1, grid_size, hidden_size))
        self.x_pos_emb = nn.Parameter(torch.zeros(1, grid_size, hidden_size))
        self.ln3 = nn.LayerNorm(hidden_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.y_pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.x_pos_emb, mean=0.0, std=0.02)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        patch_size = self.vision_config.patch_size
        patches_h = height // patch_size
        patches_w = width // patch_size
        if patches_h > self.vision_config.grid_size or patches_w > self.vision_config.grid_size:
            raise ValueError("input image has more patches than the configured position tables")

        x = extract_flattened_patches(pixel_values, patch_size)
        x = self.ln1(x)
        x = self.fc(x)
        x = self.ln2(x)

        row_emb = self.y_pos_emb[:, :patches_h, :]
        col_emb = self.x_pos_emb[:, :patches_w, :]
        pos = (row_emb.unsqueeze(2) + col_emb.unsqueeze(1)).reshape(1, patches_h * patches_w, -1)
        x = x + pos.to(dtype=x.dtype, device=x.device)
        return self.ln3(x)
