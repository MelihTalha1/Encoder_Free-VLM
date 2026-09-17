from __future__ import annotations

import torch
from scripts.smoke_test import main as run_smoke_test
from encoder_free_vlm.config import VisionConfig
from encoder_free_vlm.patching import extract_flattened_patches


def test_smoke_pipeline():
    run_smoke_test()


def test_patch_extraction():
    vision = VisionConfig(image_size=64, patch_size=16, channels=3)
    pixels = torch.rand(2, 3, 64, 64)
    patches = extract_flattened_patches(pixels, vision.patch_size)
    assert patches.shape == (2, vision.num_patches, vision.flattened_patch_dim)
