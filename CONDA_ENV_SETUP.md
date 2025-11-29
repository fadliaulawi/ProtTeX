# Conda Environment Setup

---

## ✅ RECOMMENDED: Use `protein_env` (Already Set Up!)

You already have a perfect conda environment with all required packages:

```bash
conda activate protein_env
```

### What's Included:
- ✅ PyTorch 2.7.0 (with CUDA)
- ✅ Transformers 4.53.2
- ✅ scikit-learn 1.6.1
- ✅ BioPython (needed for PDB parsing)
- ✅ All other dependencies

**This environment is ALREADY configured in the pipeline scripts!**

---

## 🚀 Quick Start (No Setup Needed)

```bash
cd /lustrefs/shared/mohammad.sayeed/Prot2Text/esmfold_tokenizer

# Environment is automatically activated by run_gpu_pipeline.sh
sbatch run_gpu_pipeline.sh
```

The SLURM script automatically activates `protein_env`!

---

## 🔧 Manual Setup (Only If Needed)

### If you want to create a fresh environment:

```bash
# Create new environment
conda create -n prot_tokenizer python=3.10 -y
conda activate prot_tokenizer

# Install dependencies
pip install torch transformers datasets huggingface-hub biopython scikit-learn matplotlib tqdm numpy
```

### Or use requirements.txt:

```bash
conda activate prot_tokenizer
pip install -r requirements.txt
```

---

## 📋 Verify Environment

```bash
# Activate environment
conda activate protein_env

# Test imports
python -c "
import torch
import transformers
import sklearn
import Bio
import datasets
print('✅ All packages available!')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'scikit-learn: {sklearn.__version__}')
print(f'GPU available: {torch.cuda.is_available()}')
"
```

**Expected output:**
```
✅ All packages available!
PyTorch: 2.7.0+cu126
Transformers: 4.53.2
scikit-learn: 1.6.1
GPU available: True
```

---

## 🎯 For Interactive Sessions

```bash
# Get GPU node
srun --partition=gpu --gres=gpu:1 --mem=64G --time=2:00:00 --pty bash

# Activate environment
conda activate protein_env

# Navigate to directory
cd /lustrefs/shared/mohammad.sayeed/Prot2Text/esmfold_tokenizer

# Run scripts
python 01_fetch_sample_data.py
python 02_extract_esm_embeddings.py
# etc.
```

---

## 🔍 Troubleshooting

### "conda: command not found"
```bash
# Initialize conda in your shell
source ~/miniconda3/etc/profile.d/conda.sh
```

### "Environment not found"
```bash
# List all environments
conda env list

# Make sure protein_env is there
# If not, create it:
conda create -n protein_env python=3.10 -y
conda activate protein_env
pip install -r requirements.txt
```

### "Package not found"
```bash
# Activate environment
conda activate protein_env

# Install missing package
pip install <package_name>
```

---

## 📦 Required Packages (Already in protein_env)

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.7.0+ | PyTorch for ESM-2 model |
| transformers | 4.30.0+ | HuggingFace ESM-2 model |
| datasets | 2.14.0+ | ProteinLMBench data |
| biopython | 1.81+ | PDB file parsing |
| scikit-learn | 1.3.0+ | k-means clustering |
| numpy | 1.24.0+ | Array operations |
| matplotlib | 3.7.0+ | Visualizations |
| tqdm | 4.65.0+ | Progress bars |

---

## ✅ Bottom Line

**You don't need to do anything!**

The `protein_env` environment is already perfect and will be used automatically when you run:

```bash
sbatch run_gpu_pipeline.sh
```

Everything is already configured! 🎉




