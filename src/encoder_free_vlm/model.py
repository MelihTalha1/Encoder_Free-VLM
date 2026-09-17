from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import VisionConfig
from .embedder import VisionPatchEmbedder


def decoder_hidden_size(decoder: nn.Module) -> int:
    config = getattr(decoder, "config", None)
    for name in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    embeddings = decoder.get_input_embeddings()
    return int(embeddings.embedding_dim)


class EncoderFreeVLM(nn.Module):
    def __init__(
        self,
        decoder: nn.Module,
        vision_config: VisionConfig,
        image_token_id: int,
        pad_token_id: int | None,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.vision_config = vision_config
        self.image_token_id = int(image_token_id)
        self.pad_token_id = None if pad_token_id is None else int(pad_token_id)
        hidden_size = decoder_hidden_size(decoder)
        self.vision_embedder = VisionPatchEmbedder(vision_config, hidden_size)
        self.connector = nn.Linear(hidden_size, hidden_size)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_input_embeddings(self) -> nn.Module:
        return self.decoder.get_input_embeddings()

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        param = next(self.vision_embedder.parameters())
        pixel_values = pixel_values.to(device=param.device, dtype=param.dtype)
        image_embeds = self.vision_embedder(pixel_values)
        return self.connector(image_embeds)

    def build_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.to(device=self.device)
        token_embeds = self.get_input_embeddings()(input_ids)
        image_embeds = self.encode_images(pixel_values)

        mask = input_ids.eq(self.image_token_id)
        flat_image_embeds = image_embeds.reshape(-1, image_embeds.shape[-1])
        expected_slots = flat_image_embeds.shape[0]
        actual_slots = int(mask.sum().item())
        if actual_slots != expected_slots:
            raise ValueError(
                f"image-token slot mismatch: input has {actual_slots}, images need {expected_slots}"
            )

        combined = token_embeds.clone()
        combined[mask] = flat_image_embeds.to(combined.dtype)
        return combined

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        inputs_embeds = self.build_inputs_embeds(input_ids, pixel_values)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=self.device)
        if labels is not None:
            labels = labels.to(device=self.device)
        return self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=use_cache,
            **kwargs,
        )

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        decoder_path = path / "decoder"
        if hasattr(self.decoder, "save_pretrained"):
            self.decoder.save_pretrained(decoder_path)
        torch.save(
            {
                "vision_config": asdict(self.vision_config),
                "image_token_id": self.image_token_id,
                "pad_token_id": self.pad_token_id,
                "vision_embedder": self.vision_embedder.state_dict(),
                "connector": self.connector.state_dict(),
            },
            path / "vision_adapter.pt",
        )

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        if hasattr(self.decoder, "gradient_checkpointing_enable"):
            self.decoder.gradient_checkpointing_enable(**kwargs)
        if hasattr(getattr(self.decoder, "config", None), "use_cache"):
            self.decoder.config.use_cache = False

    def gradient_checkpointing_disable(self) -> None:
        if hasattr(self.decoder, "gradient_checkpointing_disable"):
            self.decoder.gradient_checkpointing_disable()

    def enable_input_require_grads(self) -> None:
        if hasattr(self.decoder, "enable_input_require_grads"):
            self.decoder.enable_input_require_grads()

    def load_vision_adapter(self, path: str | Path, strict: bool = True) -> None:
        path = Path(path)
        try:
            state = torch.load(path / "vision_adapter.pt", map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(path / "vision_adapter.pt", map_location="cpu")
        self.vision_embedder.load_state_dict(state["vision_embedder"], strict=strict)
        self.connector.load_state_dict(state["connector"], strict=strict)
