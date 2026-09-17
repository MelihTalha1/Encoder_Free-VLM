from __future__ import annotations

from typing import Any

import torch

from .config import VisionConfig
from .image_processing import preprocess_image
from .model import EncoderFreeVLM
from .tokenization import encode_turns


@torch.inference_mode()
def generate_answer(
    model: EncoderFreeVLM,
    tokenizer: Any,
    image: Any,
    prompt: str,
    vision_config: VisionConfig,
    image_token: str,
    system_prompt: str | None = None,
    max_length: int = 2048,
    max_new_tokens: int = 96,
    temperature: float = 0.0,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    model.eval()
    turns = [{"user": prompt, "assistant": None}]
    input_ids = encode_turns(
        tokenizer=tokenizer,
        turns=turns,
        image_token=image_token,
        num_patches=vision_config.num_patches,
        max_length=max_length,
        system_prompt=system_prompt,
        add_generation_prompt=True,
    )
    device = model.device
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids_tensor)
    pixel_values = preprocess_image(image, vision_config.image_size).unsqueeze(0)
    inputs_embeds = model.build_inputs_embeds(input_ids_tensor, pixel_values)

    # In Qwen2.5, both <|im_end|> (151645) and <|endoftext|> (151643) act as stop tokens.
    # Passing both to eos_token_id ensures min_new_tokens blocks BOTH during initial generation.
    eos_token_id: list[int] = []
    for end_name in ["<|im_end|>", "<|endoftext|>"]:
        try:
            tid = tokenizer.convert_tokens_to_ids(end_name)
            if tid is not None and isinstance(tid, int) and tid > 0 and tid not in eos_token_id:
                eos_token_id.append(tid)
        except Exception:
            pass
    if not eos_token_id and getattr(tokenizer, "eos_token_id", None) is not None:
        eos_token_id = [tokenizer.eos_token_id]

    bad_words_ids: list[list[int]] = []
    for banned_tok in [
        image_token,
        "<|image|>",
        "<|im_start|>",
        "<|image_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
    ]:
        try:
            tid = tokenizer.convert_tokens_to_ids(banned_tok)
            if tid is not None and isinstance(tid, int) and tid > 0 and [tid] not in bad_words_ids:
                bad_words_ids.append([tid])
        except Exception:
            pass

    do_sample = temperature is not None and temperature > 0.05

    generation_kwargs = {
        "input_ids": input_ids_tensor,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": max(12, min(max_new_tokens, 12)),
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id if getattr(tokenizer, "pad_token_id", None) is not None else eos_token_id[0],
        "eos_token_id": eos_token_id if len(eos_token_id) > 1 else eos_token_id[0],
        "bad_words_ids": bad_words_ids if bad_words_ids else None,
    }
    if repetition_penalty and repetition_penalty > 1.0:
        generation_kwargs["repetition_penalty"] = repetition_penalty
    if do_sample:
        generation_kwargs["temperature"] = max(temperature, 0.1)
        generation_kwargs["top_p"] = top_p

    generated = model.decoder.generate(**generation_kwargs)
    new_tokens = generated[:, input_ids_tensor.shape[1] :]
    decoded = tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
    if not decoded:
        # If skip_special_tokens stripped everything, try decoding with special tokens to inspect
        decoded = tokenizer.decode(new_tokens[0], skip_special_tokens=False).strip()
    return decoded
