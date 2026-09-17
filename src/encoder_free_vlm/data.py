from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import torch
from torch.utils.data import Dataset, IterableDataset

from .config import VisionConfig
from .image_processing import preprocess_image
from .loss import build_labels
from .tokenization import encode_turns_with_labels, normalize_turns_from_messages


@dataclass
class VisionTextSample:
    image: Any
    turns: list[dict[str, str]]


def _minimum_score(value: Any) -> int | None:
    """Return the lowest quality score, accepting FineVision's list fields."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        values = [item for item in (_minimum_score(item) for item in value) if item is not None]
        return min(values) if values else None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def passes_quality_filter(
    raw: dict[str, Any],
    min_image_correspondence: int | None = None,
    min_visual_dependency: int | None = None,
) -> bool:
    """Keep visually grounded examples and gracefully support unscored subsets."""
    checks = (
        ("image_correspondence_min", min_image_correspondence),
        ("visual_dependency_min", min_visual_dependency),
    )
    for field, threshold in checks:
        if threshold is None:
            continue
        score = _minimum_score(raw.get(field))
        if score is not None and score < threshold:
            return False
    return True


def normalize_sample(raw: dict[str, Any]) -> VisionTextSample:
    image = raw.get("image")
    if image is None and raw.get("images"):
        images = raw["images"]
        image = images[0] if isinstance(images, (list, tuple)) else images
    if image is None and raw.get("image_path"):
        image = raw["image_path"]
    if image is None:
        raise KeyError("sample does not contain an image")

    if "texts" in raw and isinstance(raw["texts"], list):
        turns = []
        for turn in raw["texts"]:
            if "user" in turn and "assistant" in turn:
                turns.append({"user": str(turn["user"]), "assistant": str(turn["assistant"])})
        if turns:
            return VisionTextSample(image=image, turns=turns)

    if "conversations" in raw:
        turns = normalize_turns_from_messages(raw["conversations"])
        if turns:
            return VisionTextSample(image=image, turns=turns)

    if "messages" in raw:
        turns = normalize_turns_from_messages(raw["messages"])
        if turns:
            return VisionTextSample(image=image, turns=turns)

    if "question" in raw and "answer" in raw:
        return VisionTextSample(
            image=image,
            turns=[{"user": str(raw["question"]), "assistant": str(raw["answer"])}],
        )

    if "caption" in raw:
        return VisionTextSample(
            image=image,
            turns=[{"user": "Describe this image.", "assistant": str(raw["caption"])}],
        )

    raise KeyError("sample does not contain a supported conversation format")


class VisionTextDataset(Dataset):
    def __init__(
        self,
        samples: list[dict[str, Any]],
        tokenizer: Any,
        vision_config: VisionConfig,
        image_token: str,
        max_length: int,
        system_prompt: str | None = None,
        mask_user_prompt: bool = True,
        min_image_correspondence: int | None = None,
        min_visual_dependency: int | None = None,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.vision_config = vision_config
        self.image_token = image_token
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.mask_user_prompt = mask_user_prompt
        self.min_image_correspondence = min_image_correspondence
        self.min_visual_dependency = min_visual_dependency

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        raw = self.samples[index]
        if not passes_quality_filter(
            raw,
            min_image_correspondence=self.min_image_correspondence,
            min_visual_dependency=self.min_visual_dependency,
        ):
            raise ValueError("sample did not satisfy the configured visual-quality thresholds")
        return encode_sample(
            raw,
            self.tokenizer,
            self.vision_config,
            self.image_token,
            self.max_length,
            system_prompt=self.system_prompt,
            mask_user_prompt=self.mask_user_prompt,
        )


class VisionTextIterableDataset(IterableDataset):
    def __init__(
        self,
        samples: Iterable[dict[str, Any]],
        tokenizer: Any,
        vision_config: VisionConfig,
        image_token: str,
        max_length: int,
        system_prompt: str | None = None,
        mask_user_prompt: bool = True,
        min_image_correspondence: int | None = None,
        min_visual_dependency: int | None = None,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.vision_config = vision_config
        self.image_token = image_token
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.mask_user_prompt = mask_user_prompt
        self.min_image_correspondence = min_image_correspondence
        self.min_visual_dependency = min_visual_dependency

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for raw in self.samples:
            try:
                if not passes_quality_filter(
                    raw,
                    min_image_correspondence=self.min_image_correspondence,
                    min_visual_dependency=self.min_visual_dependency,
                ):
                    continue
                yield encode_sample(
                    raw,
                    self.tokenizer,
                    self.vision_config,
                    self.image_token,
                    self.max_length,
                    system_prompt=self.system_prompt,
                    mask_user_prompt=self.mask_user_prompt,
                )
            except Exception:
                continue


def encode_sample(
    raw: dict[str, Any],
    tokenizer: Any,
    vision_config: VisionConfig,
    image_token: str,
    max_length: int,
    system_prompt: str | None = None,
    mask_user_prompt: bool = True,
) -> dict[str, torch.Tensor]:
    sample = normalize_sample(raw)
    pixel_values = preprocess_image(sample.image, vision_config.image_size)
    input_ids, labels = encode_turns_with_labels(
        tokenizer=tokenizer,
        turns=sample.turns,
        image_token=image_token,
        num_patches=vision_config.num_patches,
        max_length=max_length,
        system_prompt=system_prompt,
        mask_user_prompt=mask_user_prompt,
    )
    image_token_id = tokenizer.convert_tokens_to_ids(image_token)
    actual_image_tokens = sum(1 for token_id in input_ids if token_id == image_token_id)
    if actual_image_tokens != vision_config.num_patches:
        raise ValueError(
            f"expected {vision_config.num_patches} image tokens, got {actual_image_tokens}"
        )
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "pixel_values": pixel_values,
    }


class KnapsackPacker(IterableDataset):
    """Greedy packing wrapper for pre-tokenized one-image samples."""

    def __init__(
        self,
        dataset: Iterable[dict[str, torch.Tensor]],
        max_length: int,
        pool_size: int = 128,
    ) -> None:
        self.dataset = dataset
        self.max_length = max_length
        self.pool_size = pool_size

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        iterator = iter(self.dataset)
        while True:
            pool = []
            try:
                for _ in range(self.pool_size):
                    sample = next(iterator)
                    if sample["input_ids"].numel() <= self.max_length:
                        pool.append(sample)
            except StopIteration:
                pass

            if not pool:
                return

            yield from self._pack_pool(pool)

    def _pack_pool(self, pool: list[dict[str, torch.Tensor]]) -> Iterator[dict[str, torch.Tensor]]:
        pool.sort(key=lambda item: item["input_ids"].numel(), reverse=True)
        knapsacks: list[dict[str, Any]] = []
        for sample in pool:
            length = sample["input_ids"].numel()
            placed = False
            for sack in knapsacks:
                if sack["remaining"] >= length:
                    sack["input_ids"].append(sample["input_ids"])
                    if "labels" in sample:
                        sack["labels"].append(sample["labels"])
                    sack["pixel_values"].append(sample["pixel_values"])
                    sack["remaining"] -= length
                    placed = True
                    break
            if not placed:
                knapsack_item: dict[str, Any] = {
                    "remaining": self.max_length - length,
                    "input_ids": [sample["input_ids"]],
                    "pixel_values": [sample["pixel_values"]],
                }
                if "labels" in sample:
                    knapsack_item["labels"] = [sample["labels"]]
                knapsacks.append(knapsack_item)

        for sack in knapsacks:
            item = {
                "input_ids": torch.cat(sack["input_ids"], dim=0),
                "pixel_values": torch.stack(sack["pixel_values"], dim=0),
            }
            if "labels" in sack:
                item["labels"] = torch.cat(sack["labels"], dim=0)
            yield item


class DataCollatorForVisionText:
    def __init__(
        self,
        pad_token_id: int,
        image_token_id: int,
        pad_to_multiple_of: int | None = None,
        ignore_index: int = -100,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.image_token_id = image_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
        self.ignore_index = ignore_index

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_length = max(feature["input_ids"].numel() for feature in features)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_length = ((max_length + multiple - 1) // multiple) * multiple

        batch_ids = []
        batch_labels = []
        batch_attention = []
        image_tensors = []

        has_precomputed_labels = "labels" in features[0]

        for feature in features:
            ids = feature["input_ids"]
            pad_length = max_length - ids.numel()

            if pad_length > 0:
                pad_tokens = torch.full((pad_length,), self.pad_token_id, dtype=ids.dtype)
                padded_ids = torch.cat([ids, pad_tokens], dim=0)
            else:
                padded_ids = ids

            batch_ids.append(padded_ids)
            batch_attention.append((padded_ids != self.pad_token_id).long())

            if has_precomputed_labels:
                lbls = feature["labels"]
                if pad_length > 0:
                    pad_lbls = torch.full((pad_length,), self.ignore_index, dtype=lbls.dtype)
                    padded_lbls = torch.cat([lbls, pad_lbls], dim=0)
                else:
                    padded_lbls = lbls
                batch_labels.append(padded_lbls)

            pixels = feature["pixel_values"]
            if pixels.ndim == 3:
                image_tensors.append(pixels)
            elif pixels.ndim == 4:
                image_tensors.extend(list(pixels))
            else:
                raise ValueError("pixel_values must be (C,H,W) or (N,C,H,W)")

        input_ids = torch.stack(batch_ids, dim=0)
        attention_mask = torch.stack(batch_attention, dim=0)

        if has_precomputed_labels:
            labels = torch.stack(batch_labels, dim=0)
        else:
            labels = build_labels(input_ids, self.image_token_id, self.pad_token_id, self.ignore_index)

        pixel_values = torch.stack(image_tensors, dim=0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }
