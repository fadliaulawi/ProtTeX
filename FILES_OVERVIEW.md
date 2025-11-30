# 📁 Complete File Overview

## 🎯 What to Push to GitHub

### ✅ MUST INCLUDE (Core Functionality)

```
protein_structure_tokenizer.py       # Main tokenizer class
demo_tokenizer_class.py              # Working examples & demo
requirements.txt                     # Python dependencies
RUN_DEMO.sh                          # One-command demo runner
data/structure_codebook_K512.pkl     # Trained codebook (2.7MB)
```

### ✅ HIGHLY RECOMMENDED (Documentation)

```
README.md                            # Main readme (rename TOKENIZER_CLASS_README.md)
PIPELINE_GUIDE.md                    # How to recreate the tokenizer
TOKENIZER_USAGE.md                   # Detailed API usage
READY_TO_PUSH.md                     # This summary document
```

### ✅ NICE TO HAVE (Retraining Scripts)

```
01_fetch_sample_data.py              # Step 1: Get proteins
02_extract_esm_embeddings.py         # Step 2: ESM-2 inference
03_train_kmeans_codebook.py          # Step 3: Train codebook
04_tokenize_and_demo.py              # Step 4: Test tokenizer
05_tokenizer_usage_example.py        # Step 5: API examples
run_gpu_pipeline.sh                  # Run all steps
```

### ✅ OPTIONAL (Visualization)

```
data/codebook_centroids.png          # PCA of codebook (450KB)
data/clustering_visualization.png    # Cluster distribution (575KB)
```

### ❌ DON'T INCLUDE (Too Large / Generated)

```
data/esm_embeddings.npy              # 208MB - regenerate if needed
data/sample_proteins.json            # 130KB - can regenerate
*.log files                          # Temporary logs
__pycache__/                         # Python cache
.ipynb_checkpoints/                  # Jupyter checkpoints
```

---

## 📂 Current Directory Structure

```
esmfold_tokenizer/
├── Core Files (MUST HAVE)
│   ├── protein_structure_tokenizer.py
│   ├── demo_tokenizer_class.py
│   ├── requirements.txt
│   └── RUN_DEMO.sh
│
├── Documentation (RECOMMENDED)
│   ├── TOKENIZER_CLASS_README.md → rename to README.md
│   ├── PIPELINE_GUIDE.md
│   ├── TOKENIZER_USAGE.md
│   ├── READY_TO_PUSH.md
│   ├── CONDA_ENV_SETUP.md
│   ├── COMMANDS_TO_RUN.md
│   └── QUICK_REFERENCE.txt
│
├── Pipeline Scripts (NICE TO HAVE)
│   ├── 01_fetch_sample_data.py
│   ├── 02_extract_esm_embeddings.py
│   ├── 03_train_kmeans_codebook.py
│   ├── 04_tokenize_and_demo.py
│   ├── 05_tokenizer_usage_example.py
│   ├── run_gpu_pipeline.sh
│   ├── RUN_INTERACTIVE.sh
│   └── GPU_COMMANDS.md
│
└── Data (REQUIRED + OPTIONAL)
    ├── structure_codebook_K512.pkl      ✅ REQUIRED (2.7MB)
    ├── codebook_centroids.png           ⭐ OPTIONAL (450KB)
    ├── clustering_visualization.png     ⭐ OPTIONAL (575KB)
    ├── codebook_summary_K512.json       ⭐ OPTIONAL (779B)
    ├── esm_embeddings.npy               ❌ SKIP (208MB)
    ├── sample_proteins.json             ❌ SKIP (130KB)
    ├── sequences.txt                    ❌ SKIP
    └── metadata.txt                     ❌ SKIP
```

---

## 📊 File Descriptions

### Core Files

| File | Size | Purpose |
|------|------|---------|
| `protein_structure_tokenizer.py` | 11KB | Main tokenizer class with encode/decode/save/load |
| `demo_tokenizer_class.py` | 11KB | Comprehensive demo showing all features |
| `requirements.txt` | 490B | pip dependencies |
| `RUN_DEMO.sh` | 1KB | Bash script to run demo |

### Documentation Files

| File | Size | Purpose |
|------|------|---------|
| `TOKENIZER_CLASS_README.md` | ~5KB | GitHub-ready README (rename to README.md) |
| `PIPELINE_GUIDE.md` | ~10KB | How to recreate tokenizer from scratch |
| `TOKENIZER_USAGE.md` | ~8KB | Detailed API usage examples |
| `READY_TO_PUSH.md` | ~7KB | Summary of everything (this doc) |

### Pipeline Scripts

| File | Size | Purpose | Time |
|------|------|---------|------|
| `01_fetch_sample_data.py` | 6.8KB | Download proteins from ProteinLMBench | ~10s |
| `02_extract_esm_embeddings.py` | 6.1KB | ESM-2 inference on sequences | ~5min |
| `03_train_kmeans_codebook.py` | 11KB | K-means clustering + visualization | ~2min |
| `04_tokenize_and_demo.py` | 8.4KB | Test tokenization | ~1min |
| `05_tokenizer_usage_example.py` | 11KB | API usage examples | ~30s |

### Data Files

| File | Size | Keep? | Purpose |
|------|------|-------|---------|
| `structure_codebook_K512.pkl` | 2.7MB | ✅ YES | Trained k-means model (REQUIRED) |
| `codebook_centroids.png` | 450KB | ⭐ YES | PCA visualization |
| `clustering_visualization.png` | 575KB | ⭐ YES | Cluster distribution |
| `codebook_summary_K512.json` | 779B | ⭐ YES | Statistics |
| `esm_embeddings.npy` | 208MB | ❌ NO | Too large (regenerate if needed) |
| `sample_proteins.json` | 130KB | ❌ NO | Can regenerate |

---

## 🚀 Suggested GitHub Repo Structure

```
protein-esm-tokenizer/           # Repo name
├── README.md                    # Main documentation
├── LICENSE                      # Add MIT or Apache 2.0
├── requirements.txt             
├── setup.py                     # For pip install (optional)
│
├── src/                         # Source code
│   ├── __init__.py
│   └── protein_structure_tokenizer.py
│
├── examples/
│   ├── demo.py                  # Rename demo_tokenizer_class.py
│   └── run_demo.sh              # RUN_DEMO.sh
│
├── data/
│   ├── codebook/
│   │   └── structure_codebook_K512.pkl
│   └── visualizations/
│       ├── codebook_centroids.png
│       └── clustering_visualization.png
│
├── scripts/                     # Retraining pipeline
│   ├── 01_fetch_sample_data.py
│   ├── 02_extract_esm_embeddings.py
│   ├── 03_train_kmeans_codebook.py
│   ├── 04_tokenize_and_demo.py
│   ├── 05_tokenizer_usage_example.py
│   └── run_pipeline.sh
│
└── docs/
    ├── USAGE.md                 # TOKENIZER_USAGE.md
    ├── PIPELINE.md              # PIPELINE_GUIDE.md
    └── SETUP.md                 # CONDA_ENV_SETUP.md
```

---

## 📋 Pre-Push Checklist

- [ ] Rename `TOKENIZER_CLASS_README.md` → `README.md`
- [ ] Add LICENSE file (MIT recommended)
- [ ] Test demo runs: `bash RUN_DEMO.sh`
- [ ] Verify codebook file exists: `data/structure_codebook_K512.pkl`
- [ ] Check file sizes (nothing > 100MB for GitHub)
- [ ] Remove any `.pyc`, `__pycache__`, `.log` files
- [ ] Create `.gitignore` file
- [ ] Add requirements.txt
- [ ] Test fresh install in new environment

---

## 📝 .gitignore Template

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints/

# Data (regeneratable)
data/esm_embeddings.npy
data/sample_proteins.json
data/sequences.txt
data/metadata.txt

# Logs
*.log

# Environment
.env
venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
```

---

## 🎯 Quick Commands

### Create clean directory for GitHub:

```bash
cd /lustrefs/shared/mohammad.sayeed/Prot2Text/esmfold_tokenizer

# Create repo structure
mkdir -p github_repo/{src,examples,data/codebook,data/visualizations,scripts,docs}

# Copy essential files
cp protein_structure_tokenizer.py github_repo/src/
cp demo_tokenizer_class.py github_repo/examples/demo.py
cp RUN_DEMO.sh github_repo/examples/
cp requirements.txt github_repo/

# Copy data
cp data/structure_codebook_K512.pkl github_repo/data/codebook/
cp data/*.png github_repo/data/visualizations/

# Copy pipeline scripts
cp 0*.py github_repo/scripts/
cp run_gpu_pipeline.sh github_repo/scripts/

# Copy docs
cp TOKENIZER_CLASS_README.md github_repo/README.md
cp PIPELINE_GUIDE.md github_repo/docs/PIPELINE.md
cp TOKENIZER_USAGE.md github_repo/docs/USAGE.md
cp CONDA_ENV_SETUP.md github_repo/docs/SETUP.md
```

### Check sizes:

```bash
cd github_repo
find . -type f -exec ls -lh {} \; | awk '{print $5 "\t" $9}' | sort -h
```

---

**Ready to push!** 🚀
