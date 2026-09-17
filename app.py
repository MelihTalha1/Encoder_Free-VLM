from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from encoder_free_vlm.config import LoraConfigData, ModelConfig, VisionConfig
from encoder_free_vlm.generation import generate_answer
from encoder_free_vlm.train_utils import build_vlm, find_latest_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--share", action="store_true", default=True, help="Create a public Gradio share link (essential for Colab)")
    parser.add_argument("--no-share", dest="share", action="store_false", help="Do not create a public Gradio link")
    return parser.parse_args()


def main() -> None:
    import gradio as gr

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

    base_model_name = args.base_model
    if checkpoint:
        merged_decoder = checkpoint / "decoder_merged"
        if merged_decoder.exists():
            base_model_name = str(merged_decoder)

    model_cfg = ModelConfig(base_model_name=base_model_name, load_in_4bit=args.load_in_4bit)
    vision_cfg = VisionConfig()
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

    def answer(image, prompt, system_prompt, temperature, max_new_tokens):
        if image is None:
            return "Please upload an image first."
        if not prompt.strip():
            prompt = "Describe this image in detail."
        return generate_answer(
            model=model,
            tokenizer=tokenizer,
            image=image,
            prompt=prompt,
            vision_config=vision_cfg,
            image_token=model_cfg.image_token,
            system_prompt=system_prompt if system_prompt.strip() else None,
            temperature=temperature,
            max_new_tokens=int(max_new_tokens),
        )

    with gr.Blocks(title="Encoder-Free VLM Interactive Demo") as demo:
        gr.Markdown(
            "# 🚀 Encoder-Free VLM (Vision-Language Model)\n"
            "This VLM replaces traditional Vision Encoders (like ViT) with direct RGB patch embeddings mapped into LLM hidden space."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="pil", label="Upload Image")
                prompt = gr.Textbox(label="User Question", value="Describe this image in detail.", lines=2)
                system_prompt = gr.Textbox(
                    label="System Prompt (Optional)",
                    value="You are a helpful and precise vision assistant.",
                    lines=2,
                )
                with gr.Accordion("Advanced Parameters", open=False):
                    temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Temperature")
                    max_new_tokens = gr.Slider(16, 512, value=64, step=16, label="Max New Tokens")
                submit = gr.Button("Submit Question", variant="primary")
            with gr.Column(scale=1):
                output = gr.Textbox(label="VLM Answer", lines=10)

        submit.click(
            answer,
            inputs=[image, prompt, system_prompt, temperature, max_new_tokens],
            outputs=output,
        )

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
