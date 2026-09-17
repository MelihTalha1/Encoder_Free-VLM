# Encoder-Free VLM Internship Project

This repo implements the complete 30-day code path from `train-your-own-encoder-free-vlm-in-100.pdf`
as an end-to-end, free Colab T4 GPU friendly AI engineering project:

- **Vision Preprocessing & Patching**: 512x512 image resize/crop to 256 non-overlapping 32x32 RGB patches (`3072` flat vector dim)
- **Vision Patch Embedder**: `LayerNorm -> Linear(3072 -> hidden_size) -> LayerNorm -> Row/Col Positional Embeddings -> LayerNorm`
- **LLM Integration**: 256 `<|image|>` placeholder slots mapped directly into base LLM embedding sequences
- **Loss Masking**: Image patch tokens, padding tokens, and **User Question / System Prompts** are masked with `-100` (`ignore_index`) for response-only SFT loss calculation
- **MLOps & Auto-Resume**: 4-bit QLoRA training with automatic Google Drive checkpoint syncing (`/content/drive/MyDrive/...`) for Colab disconnect protection
- **Day 19 Sanity Test**: `--overfit-test` mode for 1-minute 100-sample verification
- **Inference & UI**: Pure PyTorch inference pipeline (`infer.py`), LoRA weight merging (`merge_lora.py`), and interactive Gradio interface (`app.py`)

Important practical note: tiny decoders such as `SmolLM2-135M` are ideal for fast Colab T4 experimentation and overfit tests. For production quality visual grounding, use a 1.5B–3B instruction decoder (e.g. `Qwen/Qwen2.5-1.5B-Instruct`).

---

## ⚡ Quick Start On Google Colab

You can run the full 30-day program directly via the interactive notebook [`notebooks/Encoder_Free_VLM_Colab.ipynb`](notebooks/Encoder_Free_VLM_Colab.ipynb) or in a terminal:

```bash
git clone <your-repo-url>
cd VLM_staj
pip install -r requirements.txt
huggingface-cli login
```

### 1. Run CPU/GPU Smoke Test (Day 7)
```bash
python scripts/smoke_test.py
```

### 2. Overfit Sanity Test (Day 19)
Verify loss drops on 100 samples in ~1 minute:
```bash
python scripts/train.py --config configs/colab_t4.yaml --overfit-test
```

### 3. Main QLoRA Training (Day 21)
```bash
python scripts/train.py --config configs/colab_t4.yaml
```
*Note: The trainer automatically searches for existing checkpoints and resumes if Colab disconnects. If Google Drive is mounted at `/content/drive/MyDrive`, checkpoints auto-sync to Drive.*

### 4. Merge LoRA Weights (Day 24)
```bash
python scripts/merge_lora.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --checkpoint-dir outputs/encoder_free_vlm_qwen15b \
  --output-dir outputs/encoder_free_vlm_qwen15b/merged
```

### 5. Run VQA Inference (Day 25 & 28)
```bash
python scripts/infer.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --checkpoint-dir outputs/encoder_free_vlm_qwen15b/checkpoint-500 \
  --image path/to/image.jpg \
  --prompt "What is in this image?" \
  --system-prompt "You are a helpful visual assistant."
```

### 6. Launch Gradio Demo (Day 29)
```bash
python app.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --checkpoint-dir outputs/encoder_free_vlm_qwen15b/checkpoint-500
```

---

## 📂 Repository Structure

- `src/encoder_free_vlm/`: Core model architecture, embedder, patching, tokenization, data collator, and generation modules
- `scripts/smoke_test.py`: Standalone forward & backward pass shape test
- `scripts/train.py`: Hugging Face Trainer entrypoint supporting 4-bit QLoRA, `--overfit-test`, and Colab Drive auto-resume
- `scripts/infer.py`: Standalone inference pipeline with system prompt support
- `scripts/merge_lora.py`: Script to permanently merge LoRA weights into base decoder
- `app.py`: Gradio web UI supporting system prompts, temperature sliders, and merged models
- `notebooks/Encoder_Free_VLM_Colab.ipynb`: 1-click step-by-step Google Colab notebook
- `configs/colab_t4.yaml`: Optimized hyperparameters for free Colab T4 GPU
- `docs/30_day_code_map.md`: Day-by-day mapping of the internship curriculum to code

