# GPU Commands - ESM-2 Tokenizer Pipeline

Complete guide for running on GPU nodes with SLURM.

---

## 🚀 Quick Start (Recommended)

### Run Full Pipeline (All 4 Steps)

```bash
cd /lustrefs/shared/mohammad.sayeed/Prot2Text/esmfold_tokenizer
sbatch run_gpu_pipeline.sh
```

**What it does:**
- ✅ Fetches 100 proteins from ProteinLMBench
- ✅ Extracts ESM-2 embeddings (1280-dim)
- ✅ Trains k-means codebook (512 clusters)
- ✅ Shows tokenization demo

**Time:** ~1 hour on A100, ~2 hours on V100  
**Resources:** 1 GPU, 64GB RAM, 8 CPUs

---

## 📋 Individual Steps (Advanced)

### Step 1: Fetch Data (No GPU needed)

```bash
cd /lustrefs/shared/mohammad.sayeed/Prot2Text/esmfold_tokenizer
python 01_fetch_sample_data.py
```

**Time:** 5-10 minutes  
**Output:** `data/sample_proteins.json`

---

### Step 2: Extract ESM-2 Embeddings (GPU REQUIRED)

```bash
# Interactive GPU session
srun --partition=gpu --gres=gpu:1 --mem=32G --time=1:00:00 --pty bash

cd /lustrefs/shared/mohammad.sayeed/Prot2Text/esmfold_tokenizer
python 02_extract_esm_embeddings.py
```

**Or submit as job:**
```bash
sbatch --partition=gpu --gres=gpu:1 --mem=32G --time=1:00:00 \
  --wrap="cd /lustrefs/shared/mohammad.sayeed/Prot2Text/esmfold_tokenizer && python 02_extract_esm_embeddings.py"
```

**Time:** 30-45 minutes  
**Output:** `data/esm_embeddings.npy` (~80 MB)

---

### Step 3: Train k-means Codebook (CPU OK, but faster on GPU node)

```bash
cd /lustrefs/shared/mohammad.sayeed/Prot2Text/esmfold_tokenizer
python 03_train_kmeans_codebook.py
```

**Time:** 5-10 minutes  
**Output:** 
- `data/structure_codebook_K512.pkl`
- `data/clustering_visualization.png`

---

### Step 4: Tokenization Demo (GPU REQUIRED for ESM-2)

```bash
# Interactive GPU session
srun --partition=gpu --gres=gpu:1 --mem=16G --time=30:00 --pty bash

cd /lustrefs/shared/mohammad.sayeed/Prot2Text/esmfold_tokenizer
python 04_tokenize_and_demo.py
```

**Time:** 5-10 minutes  
**Output:** Demonstration of tokenization results

---

## 🔍 Monitoring Jobs

### Check Job Status
```bash
squeue -u $USER
```

### View Job Output (real-time)
```bash
tail -f slurm_esm_tokenizer_*.out
```

### Check GPU Usage (in job)
```bash
nvidia-smi
watch -n 1 nvidia-smi  # Live monitoring
```

---

## 📊 Expected Results

### After Step 1 (Data):
```
data/
├── sample_proteins.json (100 proteins)
├── sequences.txt (FASTA format)
└── metadata.txt
```

### After Step 2 (Embeddings):
```
data/
├── esm_embeddings.npy (~80 MB, shape: [~8000, 1280])
└── embedding_metadata.json
```

### After Step 3 (Codebook):
```
data/
├── structure_codebook_K512.pkl (~13 MB)
├── codebook_summary_K512.json
└── clustering_visualization.png
```

### After Step 4 (Demo):
```
Console output showing:
- Tokenization of 3 example proteins
- Structure token assignments
- LLM input format
- Usage instructions
```

---

## ⚙️ Resource Requirements

### Minimal (100 proteins):
- **GPU:** 1x any CUDA GPU (8GB+ VRAM)
- **RAM:** 32GB
- **CPUs:** 4
- **Time:** ~1 hour
- **Storage:** ~200 MB

### Recommended (100 proteins):
- **GPU:** 1x A100 or V100
- **RAM:** 64GB
- **CPUs:** 8
- **Time:** 30-45 minutes
- **Storage:** ~500 MB (with visualizations)

### For Full Dataset (10K+ proteins):
- **GPU:** 1x A100 (40GB)
- **RAM:** 128GB
- **CPUs:** 16
- **Time:** 4-6 hours
- **Storage:** ~50 GB

---

## 🐛 Troubleshooting

### Problem: "CUDA out of memory"
**Solution:**
```bash
# Reduce number of samples
# Edit 01_fetch_sample_data.py, line 13:
NUM_SAMPLES = 50  # Instead of 100
```

### Problem: "Model download timeout"
**Solution:**
```bash
# Pre-download model on login node
python -c "from transformers import EsmModel; EsmModel.from_pretrained('facebook/esm2_t33_650M_UR50D')"
```

### Problem: "Job pending too long"
**Solution:**
```bash
# Check queue
squeue

# Try different partition
sbatch --partition=gpu-long run_gpu_pipeline.sh

# Or request specific GPU
sbatch --gres=gpu:a100:1 run_gpu_pipeline.sh
```

### Problem: "datasets library not found"
**Solution:**
```bash
pip install datasets huggingface-hub transformers torch biopython scikit-learn matplotlib tqdm
```

---

## 🎯 Validation Checklist

After running, verify:

```bash
cd /lustrefs/shared/mohammad.sayeed/Prot2Text/esmfold_tokenizer/data

# ✓ Check data
ls -lh sample_proteins.json
python -c "import json; data=json.load(open('sample_proteins.json')); print(f'Proteins: {len(data)}')"

# ✓ Check embeddings
ls -lh esm_embeddings.npy
python -c "import numpy as np; e=np.load('esm_embeddings.npy'); print(f'Shape: {e.shape}')"

# ✓ Check codebook
ls -lh structure_codebook_K512.pkl
python -c "import pickle; c=pickle.load(open('structure_codebook_K512.pkl','rb')); print(f'Clusters: {c[\"n_clusters\"]}')"

# ✓ Check visualization
ls -lh clustering_visualization.png
file clustering_visualization.png
```

**Expected:**
- ✅ sample_proteins.json exists (~200 KB)
- ✅ esm_embeddings.npy shape: [~8000, 1280]
- ✅ codebook has 512 clusters
- ✅ visualization is valid PNG image

---

## 📈 Performance Tips

### Speed up embeddings extraction:
```python
# Edit 02_extract_esm_embeddings.py
# Use smaller model for testing:
MODEL_NAME = "facebook/esm2_t30_150M_UR50D"  # 3x faster
```

### Speed up k-means:
```python
# Edit 03_train_kmeans_codebook.py
# Reduce max_iter:
max_iter=50  # Instead of 100
```

### Process more proteins:
```python
# Edit 01_fetch_sample_data.py
NUM_SAMPLES = 500  # Or 1000, etc.
```

---

## 🔄 Rerunning Specific Steps

### Rerun embeddings only:
```bash
# Delete old embeddings
rm data/esm_embeddings.npy data/embedding_metadata.json

# Rerun
sbatch --partition=gpu --gres=gpu:1 --wrap="cd /lustrefs/shared/mohammad.sayeed/Prot2Text/esmfold_tokenizer && python 02_extract_esm_embeddings.py"
```

### Rerun k-means with different K:
```bash
# Edit 03_train_kmeans_codebook.py
# Change N_CLUSTERS = 256  # or 1024

python 03_train_kmeans_codebook.py
```

---

## 📞 Getting Help

### Check logs:
```bash
# SLURM output
cat slurm_esm_tokenizer_*.out

# Python errors
cat slurm_esm_tokenizer_*.err

# Last 50 lines
tail -50 slurm_esm_tokenizer_*.out
```

### Test individual components:
```bash
# Test imports
python -c "import torch; print(torch.cuda.is_available())"
python -c "from transformers import EsmModel; print('OK')"
python -c "from datasets import load_dataset; print('OK')"

# Test GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## 🎓 Understanding the Pipeline

```
1. Fetch Data (01_*.py)
   ↓
   ProteinLMBench → 100 proteins with sequences + descriptions
   
2. Extract Embeddings (02_*.py)
   ↓
   ESM-2 650M → 1280-dim vector per residue (~8K residues total)
   
3. Train Codebook (03_*.py)
   ↓
   k-means → 512 clusters = 512 "structure tokens"
   
4. Tokenize & Demo (04_*.py)
   ↓
   Each residue → assigned to 1 of 512 structure tokens
   Shows how to use for LLM training
```

---

## ✅ Success Criteria

You're done when:

1. ✅ All 4 scripts run without errors
2. ✅ `data/structure_codebook_K512.pkl` exists
3. ✅ Cluster utilization > 95%
4. ✅ Visualization shows clear clustering
5. ✅ Demo shows tokenization working

**Then you're ready to scale to full dataset and integrate with LLM!**




