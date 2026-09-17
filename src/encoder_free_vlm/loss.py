from __future__ import annotations

import torch
import torch.nn.functional as F


def build_labels(
    input_ids: torch.Tensor,
    image_token_id: int,
    pad_token_id: int | None,
    ignore_index: int = -100,
) -> torch.Tensor:
    labels = input_ids.clone()
    labels[labels == image_token_id] = ignore_index
    if pad_token_id is not None:
        labels[labels == pad_token_id] = ignore_index
    return labels


def next_token_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )
