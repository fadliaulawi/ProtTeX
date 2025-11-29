# 🔧 Tokenizer Creation Pipeline

This document explains how to recreate the tokenizer from scratch using the provided scripts.

## 📋 Pipeline Overview

The tokenizer is created in 5 steps:

```
01_fetch_sample_data.py
    ↓ Fetches proteins from ProteinLMBench
    ↓ Outputs: data/sample_proteins.json (100 proteins)
    
02_extract_esm_embeddings.py
    ↓ Runs ESM-2 inference on protein sequences
    ↓ Outputs: data/esm_embeddings.npy (1280D per residue)
    
03_train_kmeans_codebook.py
    ↓ Clusters embeddings into K discrete tokens
    ↓ Outputs: data/structure_codebook_K512.pkl + visualizations
    
04_tokenize_and_demo.py
    ↓ Tests the tokenizer on sample proteins
    ↓ Outputs: Tokenization examples + statistics
    
05_tokenizer_usage_example.py
    ↓ Shows comprehensive API usage
    ↓ Outputs: Documentation examples
```

## 🚀 Quick Run (All Steps)

```bash
# Activate environment
conda activate esm_tokenizer

# Run full pipeline (on GPU node)
cd /lustrefs/shared/mohammad.sayeed/Prot2Text/esmfold_tokenizer

python 01_fetch_sample_data.py        # ~10 seconds
python 02_extract_esm_embeddings.py   # ~5 minutes (GPU)
python 03_train_kmeans_codebook.py    # ~2 minutes
python 04_tokenize_and_demo.py        # ~1 minute
python 05_tokenizer_usage_example.py  # ~30 seconds
```

Or use the automated script:

```bash
bash run_gpu_pipeline.sh  # Runs all steps
```

## 📝 Detailed Step-by-Step

### Step 1: Fetch Sample Data

**Script**: `01_fetch_sample_data.py`

**What it does**:
- Downloads proteins from ProteinLMBench dataset
- Extracts 100 diverse proteins (various lengths)
- Cleans sequences and validates
- Saves metadata (sequence, text description, length)

**Output**:
```
data/sample_proteins.json    # 100 proteins with sequences + descriptions
data/sequences.txt           # Plain text sequences
data/metadata.txt            # Dataset statistics
```

**Key parameters**:
```python
NUM_SAMPLES = 100           # Number of proteins to fetch
DATASET = "ProteinLMBench"  # Dataset to use
```

**Command**:
```bash
python 01_fetch_sample_data.py
```

---

### Step 2: Extract ESM-2 Embeddings

**Script**: `02_extract_esm_embeddings.py`

**What it does**:
- Loads ESM-2 model (facebook/esm2_t33_650M_UR50D)
- Processes sequences in batches
- Extracts per-residue embeddings (1280D)
- Saves embeddings as numpy array

**Output**:
```
data/esm_embeddings.npy           # Shape: (total_residues, 1280)
data/embedding_metadata.json      # Residue mapping info
```

**Key parameters**:
```python
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"  # ESM-2 650M model
BATCH_SIZE = 4                                # Adjust for GPU memory
MAX_LENGTH = 1024                             # Max sequence length
```

**Requirements**:
- **GPU required** (takes ~5 min on H100, much slower on CPU)
- ~3GB GPU memory

**Command**:
```bash
python 02_extract_esm_embeddings.py
```

---

### Step 3: Train K-Means Codebook

**Script**: `03_train_kmeans_codebook.py`

**What it does**:
- Loads ESM-2 embeddings from Step 2
- Performs k-means clustering (K=512 clusters)
- Learns discrete codebook (512 centroids)
- Generates PCA visualizations
- Saves trained codebook

**Output**:
```
data/structure_codebook_K512.pkl        # Trained kmeans model + metadata
data/codebook_centroids.png             # PCA visualization of centroids
data/clustering_visualization.png       # 2D cluster distribution
data/codebook_summary_K512.json         # Statistics
```

**Key parameters**:
```python
K = 512                    # Number of structure tokens
N_INIT = 20                # K-means initializations (for stability)
MAX_ITER = 300             # Max k-means iterations
RANDOM_STATE = 42          # For reproducibility
```

**Command**:
```bash
python 03_train_kmeans_codebook.py
```

**Customization**:
To change the number of structure tokens:
```python
# In 03_train_kmeans_codebook.py
K = 256   # Smaller codebook (faster, less expressive)
K = 1024  # Larger codebook (slower, more expressive)
```

---

### Step 4: Test Tokenization

**Script**: `04_tokenize_and_demo.py`

**What it does**:
- Loads trained codebook
- Tokenizes sample proteins
- Shows token distributions
- Validates round-trip encoding/decoding

**Output**:
- Console output with examples
- Token statistics
- Validation results

**Command**:
```bash
python 04_tokenize_and_demo.py
```

---

### Step 5: Usage Examples

**Script**: `05_tokenizer_usage_example.py`

**What it does**:
- Demonstrates full API
- Shows batch processing
- Shows save/load functionality
- Shows integration with PyTorch

**Command**:
```bash
python 05_tokenizer_usage_example.py
```

---

## 🔄 Retraining the Codebook

To retrain with different settings:

### More Proteins (Better Codebook)

```python
# In 01_fetch_sample_data.py
NUM_SAMPLES = 500  # Instead of 100
```

Then rerun steps 1-3.

### Different Codebook Size

```python
# In 03_train_kmeans_codebook.py
K = 1024  # Instead of 512
```

Then rerun step 3 (no need to re-extract embeddings).

### Different ESM Model

```python
# In 02_extract_esm_embeddings.py
MODEL_NAME = "facebook/esm2_t36_3B_UR50D"  # Larger model (better but slower)
# or
MODEL_NAME = "facebook/esm2_t12_35M_UR50D"  # Smaller model (faster but worse)
```

Then rerun steps 2-3.

---

## 📊 Expected Output Sizes

| File | Size | Description |
|------|------|-------------|
| `sample_proteins.json` | ~130KB | 100 protein sequences + metadata |
| `esm_embeddings.npy` | ~208MB | 1280D embeddings for all residues |
| `structure_codebook_K512.pkl` | ~2.7MB | Trained k-means model |
| `codebook_centroids.png` | ~450KB | PCA visualization |
| `clustering_visualization.png` | ~575KB | Cluster distribution |

---

## ⚙️ System Requirements

### Minimum:
- Python 3.8+
- 16GB RAM
- GPU with 4GB VRAM (for ESM-2 inference)

### Recommended:
- Python 3.10
- 32GB RAM
- GPU with 8GB+ VRAM (H100, A100, RTX 4090, etc.)

### Dependencies:
See `requirements.txt`:
- torch >= 2.0
- transformers >= 4.30
- scikit-learn >= 1.3
- numpy, matplotlib, etc.

---

## 🐛 Troubleshooting

### Out of Memory (GPU)

```python
# In 02_extract_esm_embeddings.py
BATCH_SIZE = 1  # Reduce batch size
MAX_LENGTH = 512  # Reduce max sequence length
```

### Slow k-means

```python
# In 03_train_kmeans_codebook.py
N_INIT = 10  # Reduce from 20 (less stable but faster)
```

### Empty dataset

If `01_fetch_sample_data.py` returns 0 samples:
- Check internet connection (needs to download from HuggingFace)
- Check dataset name/path
- See error logs in terminal

---

## 📦 What Gets Included in GitHub

**Essential** (users need these):
1. ✅ Trained codebook: `data/structure_codebook_K512.pkl`
2. ✅ Main class: `protein_structure_tokenizer.py`
3. ✅ Demo: `demo_tokenizer_class.py`

**Optional** (for retraining):
4. ✅ All pipeline scripts: `01_*.py` through `05_*.py`
5. ✅ Run script: `run_gpu_pipeline.sh`
6. ✅ Sample data: `data/sample_proteins.json` (optional)

**Don't include** (too large):
- ❌ `data/esm_embeddings.npy` (208MB - regenerate if needed)
- ❌ Raw dataset files

---

## 🎯 Quick Reference Commands

```bash
# Full pipeline (from scratch)
bash run_gpu_pipeline.sh

# Or step by step:
python 01_fetch_sample_data.py
python 02_extract_esm_embeddings.py
python 03_train_kmeans_codebook.py

# Just test existing tokenizer:
python demo_tokenizer_class.py
bash RUN_DEMO.sh
```

---

## 📚 Further Customization

### Using Your Own Protein Dataset

Replace `01_fetch_sample_data.py` with your own data loader:

```python
import json

# Your proteins
proteins = [
    {"sequence": "MKTAYIAKQR", "text": "Some description"},
    {"sequence": "MAEGEITTFT", "text": "Another description"},
    # ... more
]

# Save in the expected format
with open('data/sample_proteins.json', 'w') as f:
    json.dump(proteins, f, indent=2)
```

Then continue with step 2.

### Using Different Clustering Methods

Currently uses k-means. To try alternatives:

```python
# In 03_train_kmeans_codebook.py, replace:
from sklearn.cluster import MiniBatchKMeans  # Faster for large data
# or
from sklearn.cluster import DBSCAN  # Density-based
```

---

**Questions?** Check `TOKENIZER_USAGE.md` for API details or `READY_TO_PUSH.md` for GitHub prep.

