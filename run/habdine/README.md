# Prot2Text Data Pipeline

This directory contains scripts to process and train on the Prot2Text-Data dataset from Hugging Face.

## Scripts

### 1. `01_download_preprocess.py`
Downloads the Prot2Text-Data dataset from Hugging Face and converts it to standardized format.

**Usage:**
```bash
python run/habdine/01_download_preprocess.py
```

**Output:**
- `data/habdine/standardized_prot2text_train.json`
- `data/habdine/standardized_prot2text_validation.json`
- `data/habdine/standardized_prot2text_test.json`
- `data/habdine/standardized_prot2text.json` (combined file)

### 2. `02_extract_embeddings.py`
Extracts embeddings for Prot2Text data:
- ESM-2 sequence embeddings (mean-pooled)
- ESMFold per-residue embeddings → structure tokens (using existing k-means codebook)
- Text embeddings (for CLIP alignment training)

**Usage:**
```bash
# Process a specific split (single GPU)
python run/habdine/02_extract_embeddings.py --model llama --k 256 --split train --batch-size 32

# Parallel processing (recommended - distributes split across 8 GPUs)
bash run/habdine/run_extract_embeddings.sh --model llama --k 256 --split train --batch-size 32
bash run/habdine/run_extract_embeddings.sh --model llama --k 256 --split validation --batch-size 32
bash run/habdine/run_extract_embeddings.sh --model llama --k 256 --split test --batch-size 32
```

**Arguments:**
- `--model`: Model type (llama, qwen, deepseek-v2)
- `--k`: Number of k-means clusters (must match existing codebook)
- `--split`: Which split to process - **required**: 'train', 'validation', or 'test'
- `--batch-size`: Batch size for processing (default: 32)
- `--start-idx`: Start index for parallelization (used by parallel script)
- `--end-idx`: End index for parallelization (used by parallel script)

**Output:**
- `data/habdine/triplet_embeddings/{model}/K{k}/triplet_embeddings_{split}_batch_*.npz` (split = train/validation/test)
- `data/habdine/triplet_embeddings/{model}/K{k}/triplet_metadata_{split}_batch_*.json`

**Requirements:**
- Existing k-means codebook from `run/02_train_kmeans_codebook.py`
- Standardized Prot2Text data from script 01

### 3. `03_train.py`
Trains CLIP alignment and LoRA fine-tuning using Prot2Text template format.

**Usage:**
```bash
# Train both CLIP alignment and LoRA
python run/habdine/03_train.py --model llama --k 256 --epochs-clip 10 --epochs-lora 3

# Skip CLIP training (use existing alignment model)
python run/habdine/03_train.py --model llama --k 256 --skip-clip --epochs-lora 3
```

**Arguments:**
- `--model`: Model type (llama, qwen, deepseek-v2)
- `--k`: Number of k-means clusters
- `--epochs-clip`: Epochs for CLIP alignment training (default: 10)
- `--epochs-lora`: Epochs for LoRA fine-tuning (default: 3)
- `--skip-clip`: Skip CLIP training and use existing alignment model

**Output:**
- `data/habdine/clip_alignment/{model}_K{k}/best_alignment_K{k}.pt`
- `data/habdine/clip_alignment/{model}_K{k}/config_K{k}.json`
- `data/habdine/lora/{model}/K{k}/final_lora_K{k}/`
- `data/habdine/evaluation/{model}/K{k}/test_data.json` (test set for evaluation)

**Template Format:**
The script uses the Prot2Text template format:
- **System**: "You are a scientific assistant specialized in protein function predictions..."
- **User**: [Optional metadata: Protein name, taxonomy] + "Sequence embeddings: <embeddings>"
- **Assistant**: Function description

During training, metadata fields are randomly dropped with 50% probability for robustness.

## Pipeline Flow

1. **Download & Preprocess**: `01_download_preprocess.py`
   - Downloads Prot2Text-Data from Hugging Face
   - Converts to standardized format (saves train/validation/test separately)

2. **Extract Embeddings**: `02_extract_embeddings.py` or `run_extract_embeddings.sh`
   - Extracts sequence, structure, and text embeddings
   - Uses existing k-means codebook for structure tokens
   - **Parallel script**: `run_extract_embeddings.sh` processes all 3 splits in parallel on 3 GPUs

3. **Train**: `03_train.py`
   - Trains CLIP alignment (sequence + structure → text)
   - Fine-tunes with LoRA using Prot2Text template

## Notes

- The pipeline uses the existing k-means codebook trained on your main dataset
- ESMFold is used for inference to get structure tokens (not for training a new codebook)
- The Prot2Text template includes optional metadata fields (name, taxonomy) that are subject to dropout during training
- All outputs are saved to `data/habdine/` to keep them separate from the main pipeline
