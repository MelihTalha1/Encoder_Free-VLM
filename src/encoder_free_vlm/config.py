from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VisionConfig:
    image_size: int = 512
    patch_size: int = 32
    channels: int = 3

    @property
    def grid_size(self) -> int:
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def flattened_patch_dim(self) -> int:
        return self.channels * self.patch_size * self.patch_size


@dataclass
class ModelConfig:
    base_model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    image_token: str = "<|image|>"
    trust_remote_code: bool = True
    torch_dtype: str = "float16"
    load_in_4bit: bool = True
    gradient_checkpointing: bool = True


@dataclass
class DataConfig:
    dataset_name: str = "HuggingFaceM4/FineVision"
    dataset_subset: str | None = "LLaVA_Instruct_150K"
    # When set, several FineVision subsets are interleaved.  ``dataset_subset``
    # is retained for backward compatibility with the original notebook.
    dataset_subsets: list[str] | None = None
    dataset_weights: list[float] | None = None
    split: str = "train"
    streaming: bool = True
    max_samples: int | None = 5000
    max_length: int = 2048
    packing: bool = False
    pack_pool_size: int = 128
    pad_to_multiple_of: int | None = 8
    shuffle_buffer_size: int = 10_000
    # FineVision exposes these scores for many subsets.  Samples without a
    # score are kept so that older subsets remain usable.
    min_image_correspondence: int | None = None
    min_visual_dependency: int | None = None


@dataclass
class LoraConfigData:
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"


@dataclass
class TrainConfig:
    output_dir: str = "outputs/encoder_free_vlm"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    vision_learning_rate: float | None = None
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_torch"
    max_grad_norm: float = 1.0
    seed: int = 42
    max_steps: int = 1000
    num_train_epochs: int = 1
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    fp16: bool = True
    bf16: bool = False
    report_to: str | list[str] = "none"


@dataclass
class ProjectConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoraConfigData = field(default_factory=LoraConfigData)
    training: TrainConfig = field(default_factory=TrainConfig)


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(instance, key):
            raise KeyError(f"Unknown config key: {key}")
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_project_config(path: str | Path | None) -> ProjectConfig:
    cfg = ProjectConfig()
    if path is None:
        return cfg

    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return _merge_dataclass(cfg, data)
