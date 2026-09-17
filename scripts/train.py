from __future__ import annotations

import argparse
from itertools import islice
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from encoder_free_vlm.config import load_project_config
from encoder_free_vlm.data import (
    DataCollatorForVisionText,
    KnapsackPacker,
    VisionTextDataset,
    VisionTextIterableDataset,
)
from encoder_free_vlm.train_utils import build_vlm, find_latest_checkpoint, mark_only_vlm_trainable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/colab_t4.yaml")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--overfit-test", action="store_true", help="Run Day 19 overfit sanity test on 100 samples")
    parser.add_argument("--system-prompt", type=str, default=None, help="System prompt for instruction training")
    return parser.parse_args()


def load_training_dataset(cfg, tokenizer, system_prompt: str | None = None):
    from datasets import concatenate_datasets, interleave_datasets, load_dataset

    subsets = cfg.data.dataset_subsets
    if not subsets:
        subsets = [cfg.data.dataset_subset] if cfg.data.dataset_subset else [None]

    datasets = []
    for subset in subsets:
        load_kwargs = {
            "path": cfg.data.dataset_name,
            "split": cfg.data.split,
            "streaming": cfg.data.streaming,
        }
        if subset:
            load_kwargs["name"] = subset
        datasets.append(load_dataset(**load_kwargs))

    if len(datasets) == 1:
        dataset = datasets[0]
    elif cfg.data.streaming:
        dataset = interleave_datasets(
            datasets,
            probabilities=cfg.data.dataset_weights,
            seed=cfg.training.seed,
            stopping_strategy="all_exhausted",
        )
    else:
        dataset = concatenate_datasets(datasets).shuffle(seed=cfg.training.seed)

    if cfg.data.streaming and cfg.data.shuffle_buffer_size:
        dataset = dataset.shuffle(
            seed=cfg.training.seed,
            buffer_size=cfg.data.shuffle_buffer_size,
        )

    if cfg.data.max_samples is not None:
        if cfg.data.streaming:
            dataset = dataset.take(cfg.data.max_samples)
        else:
            dataset = dataset.select(range(min(cfg.data.max_samples, len(dataset))))

    if cfg.data.streaming:
        encoded = VisionTextIterableDataset(
            dataset,
            tokenizer=tokenizer,
            vision_config=cfg.vision,
            image_token=cfg.model.image_token,
            max_length=cfg.data.max_length,
            system_prompt=system_prompt,
            mask_user_prompt=True,
            min_image_correspondence=cfg.data.min_image_correspondence,
            min_visual_dependency=cfg.data.min_visual_dependency,
        )
    else:
        encoded = VisionTextDataset(
            list(dataset),
            tokenizer=tokenizer,
            vision_config=cfg.vision,
            image_token=cfg.model.image_token,
            max_length=cfg.data.max_length,
            system_prompt=system_prompt,
            mask_user_prompt=True,
            min_image_correspondence=cfg.data.min_image_correspondence,
            min_visual_dependency=cfg.data.min_visual_dependency,
        )

    if cfg.data.packing:
        return KnapsackPacker(
            encoded,
            max_length=cfg.data.max_length,
            pool_size=cfg.data.pack_pool_size,
        )
    return encoded


def main() -> None:
    args = parse_args()
    cfg = load_project_config(args.config)

    if getattr(args, "overfit_test", False):
        # A streaming shuffle buffer stores complete image examples in CPU RAM.
        # Do not inherit the quality-run buffer/packing settings for this tiny
        # sanity check: on free Colab that can exhaust host RAM before the
        # first optimisation step.
        print(">>> Low-RAM overfit sanity mode: cache 8 samples, run 12 steps <<<")
        cfg.data.dataset_subsets = None
        cfg.data.dataset_subset = "LLaVA_Instruct_150K"
        cfg.data.dataset_weights = None
        cfg.data.max_samples = 64
        cfg.data.max_length = 1024
        cfg.data.packing = False
        cfg.data.shuffle_buffer_size = 0
        cfg.data.min_image_correspondence = None
        cfg.data.min_visual_dependency = None
        cfg.training.max_steps = 12
        cfg.training.gradient_accumulation_steps = 1
        cfg.training.logging_steps = 1
        cfg.training.save_steps = 1_000
        cfg.training.save_total_limit = 1
        # Starting the randomly initialised image projection too aggressively
        # can overflow FP16 gradients on a T4.  Use a short warm-up and small,
        # deliberately conservative rates for this numerical-stability check.
        cfg.training.learning_rate = 5e-5
        cfg.training.vision_learning_rate = 1e-4
        cfg.training.warmup_ratio = 0.25
        cfg.training.max_grad_norm = 0.5
        cfg.training.output_dir = "outputs/overfit_test_qwen"

    tokenizer, model = build_vlm(cfg.model, cfg.vision, cfg.lora)
    mark_only_vlm_trainable(model)

    resume_from = None
    can_resume_trainer = False
    if not args.no_resume and not getattr(args, "overfit_test", False):
        resume_from = find_latest_checkpoint(cfg.training.output_dir)

    if resume_from:
        resume_path = Path(resume_from)
        # If the latest checkpoint is on Google Drive, sync it to the local workspace
        # so Hugging Face Trainer can resume natively with fast disk access.
        local_target = Path(cfg.training.output_dir) / resume_path.name
        if resume_path.resolve() != local_target.resolve():
            print(f"Syncing checkpoint {resume_path.name} from Google Drive to local workspace...")
            local_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(resume_path, local_target, dirs_exist_ok=True)
            resume_from = str(local_target)
            resume_path = local_target

        print(f"Loading VLM checkpoint weights from: {resume_from}")
        try:
            model.load_vision_adapter(resume_from)
            decoder_adapter = resume_path / "decoder"
            if decoder_adapter.exists():
                try:
                    import peft.import_utils
                    peft.import_utils.is_torchao_available = lambda: False
                except Exception:
                    pass
                if hasattr(model.decoder, "load_adapter"):
                    model.decoder.load_adapter(str(decoder_adapter), adapter_name="default", is_trainable=True)
                else:
                    from peft import PeftModel
                    model.decoder = PeftModel.from_pretrained(model.decoder, decoder_adapter, is_trainable=True)
            print("Successfully restored VLM weights from checkpoint!")
        except Exception as exc:
            print(f"Note: Could not restore adapter weights ({exc}), starting fresh.")

        state_file = resume_path / "trainer_state.json"
        if state_file.exists():
            can_resume_trainer = True
            try:
                state_data = json.loads(state_file.read_text(encoding="utf-8"))
                saved_step = int(state_data.get("global_step", 0))
                print(f">>> Found Trainer state at step {saved_step}. Resuming training seamlessly from step {saved_step}! <<<")
                if saved_step >= cfg.training.max_steps:
                    new_max = saved_step + 250
                    print(f">>> [Notice] Saved step ({saved_step}) is >= max_steps ({cfg.training.max_steps}). Automatically extending max_steps to {new_max}! <<<")
                    cfg.training.max_steps = new_max
            except Exception as e:
                print(f"Note: Error inspecting trainer_state.json ({e})")

    train_dataset = load_training_dataset(cfg, tokenizer, system_prompt=args.system_prompt)
    if args.overfit_test:
        # Materialising a handful of already preprocessed samples makes this a
        # genuine overfit check: Trainer repeatedly sees the same examples,
        # while host-RAM use stays bounded to roughly eight image tensors.
        train_dataset = list(islice(train_dataset, 8))
        if len(train_dataset) < 8:
            raise RuntimeError("could not collect eight valid samples for the overfit sanity check")
        print(f">>> Cached {len(train_dataset)} fixed samples for the low-RAM sanity check <<<")
    collator = DataCollatorForVisionText(
        pad_token_id=tokenizer.pad_token_id,
        image_token_id=tokenizer.image_token_id,
        pad_to_multiple_of=cfg.data.pad_to_multiple_of,
    )

    from transformers import Trainer, TrainerCallback, TrainingArguments

    class SaveEncoderFreeVLMCallback(TrainerCallback):
        def on_save(self, args, state, control, model=None, **kwargs):
            ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if model is not None and hasattr(model, "save_pretrained"):
                model.save_pretrained(ckpt_dir)

            # Auto-backup to Google Drive if running in Colab
            drive_base = Path("/content/drive/MyDrive")
            if drive_base.exists():
                drive_parent = drive_base / Path(args.output_dir).name
                drive_target = drive_parent / f"checkpoint-{state.global_step}"
                try:
                    drive_target.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(ckpt_dir, drive_target, dirs_exist_ok=True)
                    print(f"Synced checkpoint to Google Drive: {drive_target}")

                    # Clean up older Drive checkpoints according to save_total_limit
                    if args.save_total_limit is not None and args.save_total_limit > 0:
                        drive_ckpts = []
                        for p in drive_parent.glob("checkpoint-*"):
                            if p.is_dir():
                                m = re.match(r"checkpoint-(\d+)$", p.name)
                                if m:
                                    drive_ckpts.append((int(m.group(1)), p))
                        if len(drive_ckpts) > args.save_total_limit:
                            drive_ckpts.sort(key=lambda x: x[0])
                            for _, old_p in drive_ckpts[:-args.save_total_limit]:
                                try:
                                    shutil.rmtree(old_p)
                                    print(f"Cleaned up older Drive checkpoint: {old_p.name}")
                                except Exception:
                                    pass
                except Exception as exc:
                    print(f"Failed to sync checkpoint to Drive: {exc}")
            return control

    report_to = cfg.training.report_to
    if report_to == "none":
        report_to = []

    warmup_steps = int(cfg.training.max_steps * getattr(cfg.training, "warmup_ratio", 0.03))

    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_steps=warmup_steps,
        max_steps=cfg.training.max_steps,
        num_train_epochs=cfg.training.num_train_epochs,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        fp16=cfg.training.fp16,
        report_to=report_to,
        remove_unused_columns=False,
        gradient_checkpointing=cfg.model.gradient_checkpointing,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        max_grad_norm=cfg.training.max_grad_norm,
        optim=cfg.training.optim,
        seed=cfg.training.seed,
    )

    class VLMTrainer(Trainer):
        def create_optimizer(self):
            if self.optimizer is not None:
                return self.optimizer

            visual_parameters = []
            decoder_parameters = []
            for name, parameter in self.model.named_parameters():
                if not parameter.requires_grad:
                    continue
                if name.startswith(("vision_embedder.", "connector.")):
                    visual_parameters.append(parameter)
                else:
                    decoder_parameters.append(parameter)

            parameter_groups = [
                {
                    "params": decoder_parameters,
                    "lr": cfg.training.learning_rate,
                    "weight_decay": cfg.training.weight_decay,
                },
                {
                    "params": visual_parameters,
                    "lr": cfg.training.vision_learning_rate or cfg.training.learning_rate,
                    "weight_decay": cfg.training.weight_decay,
                },
            ]
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(parameter_groups, **optimizer_kwargs)
            return self.optimizer

        def _save(self, output_dir: str | None = None, state_dict=None):
            target_dir = Path(output_dir) if output_dir is not None else Path(self.args.output_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            if hasattr(self.model, "save_pretrained"):
                self.model.save_pretrained(target_dir)
            if hasattr(self.processing_class, "save_pretrained"):
                self.processing_class.save_pretrained(target_dir)

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            target_model = model if model is not None else self.model
            ckpt_path = Path(resume_from_checkpoint)
            print(f"Restoring EncoderFreeVLM weights from: {ckpt_path}")

            # 1. Restore vision embedder & connector weights
            vision_path = ckpt_path / "vision_adapter.pt"
            if vision_path.exists() and hasattr(target_model, "load_vision_adapter"):
                try:
                    target_model.load_vision_adapter(ckpt_path)
                except Exception as exc:
                    print(f"Note: Could not restore vision adapter in Trainer ({exc})")

            # 2. Restore LoRA decoder adapter weights
            decoder_adapter = ckpt_path / "decoder"
            if decoder_adapter.exists():
                decoder = getattr(target_model, "decoder", None)
                if decoder is not None:
                    try:
                        import peft.import_utils
                        peft.import_utils.is_torchao_available = lambda: False
                    except Exception:
                        pass
                    if hasattr(decoder, "load_adapter"):
                        try:
                            decoder.load_adapter(str(decoder_adapter), adapter_name="default", is_trainable=True)
                        except Exception as exc:
                            print(f"Note: Could not load_adapter on decoder ({exc})")
                    else:
                        from peft import PeftModel
                        target_model.decoder = PeftModel.from_pretrained(decoder, decoder_adapter, is_trainable=True)

    trainer = VLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[SaveEncoderFreeVLMCallback()],
    )

    resume_checkpoint_arg = resume_from if can_resume_trainer else None
    trainer.train(resume_from_checkpoint=resume_checkpoint_arg)
    trainer.save_model(cfg.training.output_dir)


if __name__ == "__main__":
    main()
