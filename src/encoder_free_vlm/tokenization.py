from __future__ import annotations

from typing import Any


def ensure_image_token(tokenizer: Any, image_token: str = "<|image|>") -> int:
    vocab = tokenizer.get_vocab()
    if image_token not in vocab:
        tokenizer.add_special_tokens({"additional_special_tokens": [image_token]})
    token_id = tokenizer.convert_tokens_to_ids(image_token)
    tokenizer.image_token = image_token
    tokenizer.image_token_id = token_id
    return int(token_id)


def image_token_prefix(image_token: str, num_patches: int) -> str:
    return image_token * num_patches


def normalize_turns_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    pending_user: str | None = None
    for message in messages:
        role = str(message.get("role") or message.get("from") or "").lower()
        content = message.get("content", message.get("value", ""))
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    text_parts.append(item)
            content = "\n".join(text_parts)
        content = str(content)
        if role in {"human", "user"}:
            pending_user = content
        elif role in {"gpt", "assistant", "model"} and pending_user is not None:
            turns.append({"user": pending_user, "assistant": content})
            pending_user = None
    return turns


def render_chat(
    tokenizer: Any,
    turns: list[dict[str, str]],
    image_token: str,
    num_patches: int,
    system_prompt: str | None = None,
    add_generation_prompt: bool = False,
) -> str:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for index, turn in enumerate(turns):
        user_content = turn["user"]
        if index == 0:
            user_content = image_token_prefix(image_token, num_patches) + user_content
        messages.append({"role": "user", "content": user_content})
        assistant = turn.get("assistant")
        if assistant is not None:
            messages.append({"role": "assistant", "content": assistant})

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    eos = tokenizer.eos_token or ""
    chunks = []
    for message in messages:
        chunks.append(f"{message['role']}: {message['content']}{eos}")
    if add_generation_prompt:
        chunks.append("assistant: ")
    return "\n".join(chunks)


def encode_turns(
    tokenizer: Any,
    turns: list[dict[str, str]],
    image_token: str,
    num_patches: int,
    max_length: int,
    system_prompt: str | None = None,
    add_generation_prompt: bool = False,
) -> list[int]:
    text = render_chat(
        tokenizer=tokenizer,
        turns=turns,
        image_token=image_token,
        num_patches=num_patches,
        system_prompt=system_prompt,
        add_generation_prompt=add_generation_prompt,
    )
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    return list(encoded["input_ids"])


def encode_turns_with_labels(
    tokenizer: Any,
    turns: list[dict[str, str]],
    image_token: str,
    num_patches: int,
    max_length: int,
    system_prompt: str | None = None,
    mask_user_prompt: bool = True,
    ignore_index: int = -100,
) -> tuple[list[int], list[int]]:
    text = render_chat(
        tokenizer=tokenizer,
        turns=turns,
        image_token=image_token,
        num_patches=num_patches,
        system_prompt=system_prompt,
        add_generation_prompt=False,
    )
    if not mask_user_prompt:
        input_ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        labels = list(input_ids)
    else:
        # Training on every user turn makes the model learn to continue a
        # conversation as the *user*.  Supervise only answer text instead.
        # Offset mappings keep this correct for multi-turn chat templates.
        answer_spans: list[tuple[int, int]] = []
        search_start = 0
        for turn in turns:
            answer = turn.get("assistant")
            if not answer:
                continue
            start = text.find(answer, search_start)
            if start < 0:
                answer_spans = []
                break
            end = start + len(answer)
            answer_spans.append((start, end))
            search_start = end

        try:
            if not answer_spans:
                raise ValueError("could not locate assistant text in the rendered chat")
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=True,
            )
            input_ids = list(encoded["input_ids"])
            offsets = list(encoded["offset_mapping"])
            labels = [ignore_index] * len(input_ids)
            for index, (token_id, offset) in enumerate(zip(input_ids, offsets)):
                start, end = offset
                if end > start and any(start < span_end and end > span_start for span_start, span_end in answer_spans):
                    labels[index] = token_id
        except (TypeError, KeyError, ValueError):
            # Slow tokenizers may not provide offsets.  Preserve the original
            # first-turn masking as a safe fallback rather than failing a run.
            input_ids = encode_turns(
                tokenizer=tokenizer,
                turns=turns,
                image_token=image_token,
                num_patches=num_patches,
                max_length=max_length,
                system_prompt=system_prompt,
                add_generation_prompt=False,
            )
            labels = list(input_ids)
            prompt_text = render_chat(
                tokenizer=tokenizer,
                turns=[{"user": turns[0]["user"], "assistant": None}],
                image_token=image_token,
                num_patches=num_patches,
                system_prompt=system_prompt,
                add_generation_prompt=True,
            )
            prompt_ids = tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )["input_ids"]
            for i in range(min(len(prompt_ids), len(input_ids))):
                labels[i] = ignore_index

    # Also ensure any image_token_id is explicitly set to ignore_index
    image_token_id = tokenizer.convert_tokens_to_ids(image_token)
    for i in range(len(labels)):
        if input_ids[i] == image_token_id:
            labels[i] = ignore_index

    return input_ids, labels
