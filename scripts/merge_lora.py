from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from encoder_free_vlm.config import ModelConfig
from encoder_free_vlm.tokenization import ensure_image_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint-dir", default="outputs/encoder_free_vlm_qwen15b")
    parser.add_argument("--output-dir", default="outputs/encoder_free_vlm_qwen15b/merged")
    parser.add_argument("--image-token", default="<|image|>")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import peft.import_utils
        peft.import_utils.is_torchao_available = lambda: False
        import peft.tuners.lora.torchao
        peft.tuners.lora.torchao.is_torchao_available = lambda: False
    except Exception:
        pass
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from encoder_free_vlm.train_utils import find_latest_checkpoint

    checkpoint_dir = Path(args.checkpoint_dir)
    if not (checkpoint_dir / "decoder").exists():
        search_path = checkpoint_dir if checkpoint_dir.is_dir() else checkpoint_dir.parent
        latest = find_latest_checkpoint(search_path)
        if latest and (Path(latest) / "decoder").exists():
            checkpoint_dir = Path(latest)
        else:
            raise FileNotFoundError(f"no valid checkpoint with 'decoder' found in {args.checkpoint_dir}")
    print(f"Using checkpoint: {checkpoint_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    ensure_image_token(tokenizer, args.image_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map="auto",
    )
    base.resize_token_embeddings(len(tokenizer))
    peft_model = PeftModel.from_pretrained(base, checkpoint_dir / "decoder")
    merged = peft_model.merge_and_unload()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output / "decoder_merged")
    tokenizer.save_pretrained(output)

    vision_adapter = checkpoint_dir / "vision_adapter.pt"
    if vision_adapter.exists():
        shutil.copy2(vision_adapter, output / "vision_adapter.pt")

    # Keep a tiny config marker for downstream scripts and Spaces.
    (output / "model_config.txt").write_text(
        f"base_model_name={ModelConfig(base_model_name=args.base_model).base_model_name}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
