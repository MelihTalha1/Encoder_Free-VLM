from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from encoder_free_vlm.config import LoraConfigData, ModelConfig, VisionConfig
from encoder_free_vlm.generation import generate_answer
from encoder_free_vlm.train_utils import build_vlm, find_latest_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Find the newest checkpoint in this output directory (also checks Google Drive in Colab).",
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system-prompt", default=None, help="System prompt for generation")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the same 4-bit loading mode as training (enabled by default).",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.checkpoint_dir and args.output_dir:
        raise ValueError("use either --checkpoint-dir or --output-dir, not both")

    checkpoint_path = args.checkpoint_dir
    if args.output_dir:
        # Check if merged model exists first
        merged_candidate = Path(args.output_dir) / "merged"
        if not (merged_candidate / "decoder_merged").exists():
            drive_merged = Path("/content/drive/MyDrive") / Path(args.output_dir).name / "merged"
            if (drive_merged / "decoder_merged").exists():
                merged_candidate = drive_merged

        if (merged_candidate / "decoder_merged").exists():
            checkpoint_path = str(merged_candidate)
            print(f"Using merged model from: {checkpoint_path}")
        else:
            checkpoint_path = find_latest_checkpoint(args.output_dir)
            if checkpoint_path is None:
                raise FileNotFoundError(f"no checkpoint found in {args.output_dir}")
            print(f"Using newest checkpoint: {checkpoint_path}")
    checkpoint = Path(checkpoint_path) if checkpoint_path else None

    # Check if checkpoint directory contains a merged decoder
    base_model_name = args.base_model
    if checkpoint:
        merged_decoder = checkpoint / "decoder_merged"
        if merged_decoder.exists():
            base_model_name = str(merged_decoder)

    model_cfg = ModelConfig(
        base_model_name=base_model_name,
        load_in_4bit=args.load_in_4bit,
        torch_dtype="float16",
    )
    vision_cfg = VisionConfig(image_size=args.image_size, patch_size=args.patch_size)
    tokenizer, model = build_vlm(model_cfg, vision_cfg, LoraConfigData(enabled=False))
    if not getattr(tokenizer, "chat_template", None):
        try:
            from transformers import AutoTokenizer
            orig_tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
            tokenizer.chat_template = orig_tok.chat_template
        except Exception:
            pass

    if checkpoint:
        decoder_adapter = checkpoint / "decoder"
        if decoder_adapter.exists():
            try:
                import peft.import_utils
                peft.import_utils.is_torchao_available = lambda: False
                import peft.tuners.lora.torchao
                peft.tuners.lora.torchao.is_torchao_available = lambda: False
            except Exception:
                pass
            from peft import PeftModel

            model.decoder = PeftModel.from_pretrained(model.decoder, decoder_adapter)
        vision_path = None
        candidates = [checkpoint / "vision_adapter.pt"]
        if checkpoint.name == "merged":
            candidates.append(checkpoint.parent / "vision_adapter.pt")
            candidates.append(checkpoint.parent / "checkpoint-500" / "vision_adapter.pt")
            candidates.append(Path("/content/drive/MyDrive") / checkpoint.parent.name / "checkpoint-500" / "vision_adapter.pt")
            candidates.append(Path("/content/drive/MyDrive") / checkpoint.parent.name / "merged" / "vision_adapter.pt")
        for c in candidates:
            if c.exists():
                vision_path = c
                break
        if not vision_path and args.output_dir:
            latest = find_latest_checkpoint(args.output_dir)
            if latest and (Path(latest) / "vision_adapter.pt").exists():
                vision_path = Path(latest) / "vision_adapter.pt"

        if vision_path and vision_path.exists():
            print(f"Loading vision adapter from: {vision_path}")
            model.load_vision_adapter(vision_path.parent)
        else:
            print(f"Note: No vision_adapter.pt found in {checkpoint}; vision weights remain randomly initialized.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.load_in_4bit and hasattr(model, "to"):
        model.to(device)
    else:
        model.vision_embedder.to(device)
        model.connector.to(device)

    answer = generate_answer(
        model=model,
        tokenizer=tokenizer,
        image=args.image,
        prompt=args.prompt,
        vision_config=vision_cfg,
        image_token=model_cfg.image_token,
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    print("\n" + "=" * 25 + " MODEL CEVABI " + "=" * 25, flush=True)
    if answer:
        print(answer, flush=True)
    else:
        print("(Model boş çıktı üretti)", flush=True)
    print("=" * 64 + "\n", flush=True)


if __name__ == "__main__":
    main()
