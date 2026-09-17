from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch import nn

from encoder_free_vlm.config import VisionConfig
from encoder_free_vlm.data import DataCollatorForVisionText
from encoder_free_vlm.loss import build_labels
from encoder_free_vlm.model import EncoderFreeVLM
from encoder_free_vlm.patching import extract_flattened_patches


class ToyDecoder(nn.Module):
    def __init__(self, vocab_size: int = 64, hidden_size: int = 32) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, vocab_size=vocab_size, use_cache=False)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed

    def forward(self, inputs_embeds, attention_mask=None, labels=None, use_cache=None, **kwargs):
        logits = self.lm_head(inputs_embeds)
        loss = None
        if labels is not None:
            from encoder_free_vlm.loss import next_token_loss

            loss = next_token_loss(logits, labels)
        return SimpleNamespace(logits=logits, loss=loss)


def main() -> None:
    vision = VisionConfig(image_size=64, patch_size=16, channels=3)
    pixels = torch.rand(2, 3, 64, 64)
    patches = extract_flattened_patches(pixels, vision.patch_size)
    assert patches.shape == (2, vision.num_patches, vision.flattened_patch_dim)

    image_token_id = 1
    pad_token_id = 0
    text_ids = torch.tensor([[5, 6, 7, 2], [8, 9, 2, 0]], dtype=torch.long)
    image_slots = torch.full((2, vision.num_patches), image_token_id, dtype=torch.long)
    input_ids = torch.cat([image_slots, text_ids], dim=1)
    attention_mask = (input_ids != pad_token_id).long()
    labels = build_labels(input_ids, image_token_id, pad_token_id)
    assert labels[:, : vision.num_patches].eq(-100).all()
    assert labels[input_ids == pad_token_id].eq(-100).all()

    model = EncoderFreeVLM(
        decoder=ToyDecoder(),
        vision_config=vision,
        image_token_id=image_token_id,
        pad_token_id=pad_token_id,
    )
    output = model(
        input_ids=input_ids,
        pixel_values=pixels,
        attention_mask=attention_mask,
        labels=labels,
    )
    assert output.logits.shape[:2] == input_ids.shape
    assert output.loss is not None
    output.loss.backward()
    print("smoke test passed")


if __name__ == "__main__":
    main()
