from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch

from .config import LoraConfigData, ModelConfig, VisionConfig
from .model import EncoderFreeVLM
from .tokenization import ensure_image_token


def torch_dtype_from_name(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"unsupported torch dtype: {name}")


def load_tokenizer_and_decoder(
    model_config: ModelConfig,
    lora_config: LoraConfigData | None = None,
) -> tuple[Any, torch.nn.Module, int, int]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.base_model_name,
        trust_remote_code=model_config.trust_remote_code,
    )
    image_token_id = ensure_image_token(tokenizer, model_config.image_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = int(tokenizer.pad_token_id)

    dtype = torch_dtype_from_name(model_config.torch_dtype)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": model_config.trust_remote_code,
        "torch_dtype": dtype,
    }
    if model_config.load_in_4bit:
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"

    decoder = AutoModelForCausalLM.from_pretrained(model_config.base_model_name, **model_kwargs)
    decoder.resize_token_embeddings(len(tokenizer))

    if model_config.gradient_checkpointing and hasattr(decoder, "gradient_checkpointing_enable"):
        decoder.gradient_checkpointing_enable()
        if hasattr(decoder.config, "use_cache"):
            decoder.config.use_cache = False

    if lora_config and lora_config.enabled:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

        if model_config.load_in_4bit:
            decoder = prepare_model_for_kbit_training(decoder)
        peft_config = LoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            target_modules=lora_config.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        decoder = get_peft_model(decoder, peft_config)

    return tokenizer, decoder, image_token_id, pad_token_id


def build_vlm(
    model_config: ModelConfig,
    vision_config: VisionConfig,
    lora_config: LoraConfigData | None = None,
) -> tuple[Any, EncoderFreeVLM]:
    tokenizer, decoder, image_token_id, pad_token_id = load_tokenizer_and_decoder(
        model_config,
        lora_config,
    )
    model = EncoderFreeVLM(
        decoder=decoder,
        vision_config=vision_config,
        image_token_id=image_token_id,
        pad_token_id=pad_token_id,
    )
    return tokenizer, model


def find_latest_checkpoint(output_dir: str | Path) -> str | None:
    output_path = Path(output_dir)
    search_paths = [output_path]

    # Google Drive auto-resume support in Colab
    drive_base = Path("/content/drive/MyDrive")
    if drive_base.exists():
        search_paths.append(drive_base / output_path.name)
        search_paths.append(drive_base / output_dir)

    pattern = re.compile(r"checkpoint-(\d+)$")
    candidates: list[tuple[int, Path]] = []
    for path in search_paths:
        if not path.exists():
            continue
        for child in path.iterdir():
            match = pattern.match(child.name)
            if child.is_dir() and match:
                # Ensure checkpoint has at least vision_adapter.pt, trainer_state.json, or decoder
                if (child / "vision_adapter.pt").exists() or (child / "trainer_state.json").exists() or (child / "decoder").exists():
                    candidates.append((int(match.group(1)), child))

    if not candidates:
        return None
    return str(max(candidates, key=lambda item: item[0])[1])


def mark_only_vlm_trainable(model: EncoderFreeVLM) -> None:
    for parameter in model.vision_embedder.parameters():
        parameter.requires_grad = True
    for parameter in model.connector.parameters():
        parameter.requires_grad = True
