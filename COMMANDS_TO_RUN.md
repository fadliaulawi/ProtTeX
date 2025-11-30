# 🚀 COMMANDS TO RUN - Copy & Paste Ready

Everything you need to run the complete pipeline.

---

## ⚡ FASTEST WAY (Recommended)

```bash
# Navigate to directory
cd /lustrefs/shared/mohammad.sayeed/Prot2Text/esmfold_tokenizer

# Submit GPU job (runs everything)
sbatch run_gpu_pipeline.sh

# Monitor progress
tail -f slurm_esm_tokenizer_*.out

# Check job status
squeue -u $USER
```

**That's it!** Check back in ~1 hour.

**Note:** The script automatically uses `protein_env` conda environment (already has all packages).

---

## 📊 WHAT GETS CREATED

After the job finishes:

```bash
# Check outputs
cd data/
ls -lh

# Expected files:
# - sample_proteins.json (100 proteins)
# - esm_embeddings.npy (ESM-2 embeddings)
# - structure_codebook_K512.pkl (trained tokenizer)
# - codebook_centroids.png (CENTROID PLOT) ⭐
# - clustering_visualization.png
# - tokenizer_saved/ (ready to push)
```

---

## 🎨 VIEW RESULTS

```bash
cd /lustrefs/shared/mohammad.sayeed/Prot2Text/esmfold_tokenizer/data

# View centroid plot (codebook visualization)
display codebook_centroids.png

# Or copy to local machine
scp mohammad.sayeed@login-node:/lustrefs/shared/mohammad.sayeed/Prot2Text/esmfold_tokenizer/data/codebook_centroids.png .
```

---

## 🧪 TEST TOKENIZER

```bash
cd /lustrefs/shared/mohammad.sayeed/Prot2Text/esmfold_tokenizer

# Run all 7 usage examples
python 05_tokenizer_usage_example.py
```

**Shows:**
- Basic encoding/decoding
- Different output formats
- Batch processing
- LLM integration
- Save/load for sharing

---

## 🐛 IF JOB FAILS

### Check logs:
```bash
cat slurm_esm_tokenizer_*.err
tail -50 slurm_esm_tokenizer_*.out
```

### Run steps individually:
```bash
# Get interactive GPU session
srun --partition=gpu --gres=gpu:1 --mem=64G --time=2:00:00 --pty bash

# Activate conda environment
conda activate protein_env

# Navigate to directory
cd /lustrefs/shared/mohammad.sayeed/Prot2Text/esmfold_tokenizer

# Step 1: Fetch data (5 min, no GPU needed)
python 01_fetch_sample_data.py

# Step 2: Extract embeddings (30-45 min, GPU required)
python 02_extract_esm_embeddings.py

# Step 3: Train codebook + plot centroids (5 min)
python 03_train_kmeans_codebook.py

# Step 4: Demo (5 min, GPU required)
python 04_tokenize_and_demo.py

# Step 5: Usage examples (5 min, GPU required)
python 05_tokenizer_usage_example.py
```

---

## 💾 SAVE FOR LATER

### Option 1: Save tokenizer for sharing

```bash
cd /lustrefs/shared/mohammad.sayeed/Prot2Text/esmfold_tokenizer

# Tokenizer already saved to data/tokenizer_saved/
# Copy to your project directory
cp -r data/tokenizer_saved ~/my_project/protein_tokenizer

# Or tar for transfer
tar -czf protein_tokenizer.tar.gz data/tokenizer_saved/
```

### Option 2: Load and use tokenizer

```python
from protein_structure_tokenizer import ProteinStructureTokenizer

# Load saved tokenizer
tokenizer = ProteinStructureTokenizer.from_pretrained(
    "data/tokenizer_saved/"
)

# Use it
tokens = tokenizer.encode("MALWMRLLPLLA")
print(tokens)
```

---

## 📦 PUSH TO GITHUB

```bash
# Create repo structure
mkdir protein-tokenizer
cd protein-tokenizer

# Copy files
cp ../protein_structure_tokenizer.py .
cp -r ../data/tokenizer_saved/* .
cp ../TOKENIZER_CLASS_README.md README.md
cp ../requirements.txt .

# Initialize repo
git init
git add .
git commit -m "Initial commit: ProteinStructureTokenizer with trained codebook"

# Push
git remote add origin https://github.com/YOUR_USERNAME/protein-tokenizer.git
git push -u origin main
```

---

## 🤗 PUSH TO HUGGINGFACE

```bash
# Install CLI
pip install huggingface-hub

# Login
huggingface-cli login

# Upload
cd /lustrefs/shared/mohammad.sayeed/Prot2Text/esmfold_tokenizer
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='data/tokenizer_saved/',
    repo_id='YOUR_USERNAME/protein-structure-tokenizer',
    repo_type='model'
)
print('✅ Uploaded to HuggingFace!')
"
```

---

## ✅ VALIDATION CHECKLIST

After running, verify:

```bash
cd data/

# 1. Check data fetched
test -f sample_proteins.json && echo "✅ Data" || echo "❌ Data missing"

# 2. Check embeddings extracted
test -f esm_embeddings.npy && echo "✅ Embeddings" || echo "❌ Embeddings missing"

# 3. Check codebook trained
test -f structure_codebook_K512.pkl && echo "✅ Codebook" || echo "❌ Codebook missing"

# 4. Check visualizations
test -f codebook_centroids.png && echo "✅ Centroid plot" || echo "❌ Plot missing"

# 5. Check tokenizer saved
test -d tokenizer_saved && echo "✅ Tokenizer saved" || echo "❌ Not saved"

# All checks pass? You're done! ✅
```

---

## 🔍 CHECK RESULTS

### Codebook quality:
```bash
python -c "
import json
s = json.load(open('data/codebook_summary_K512.json'))
print(f'Clusters: {s[\"n_clusters\"]}')
print(f'Utilization: {s[\"utilization\"]*100:.1f}%')
print(f'Residues: {s[\"n_residues\"]:,}')
"
```

**Expected:**
- Clusters: 512
- Utilization: >95%
- Residues: ~8,000

### Tokenizer test:
```bash
python -c "
import sys
sys.path.insert(0, '.')
from protein_structure_tokenizer import ProteinStructureTokenizer

tokenizer = ProteinStructureTokenizer(
    codebook_path='data/structure_codebook_K512.pkl'
)

sequence = 'MALW'
tokens = tokenizer.encode(sequence)
decoded = tokenizer.decode(tokens)

print(f'Input: {sequence}')
print(f'Tokens: {tokens}')
print(f'Decoded: {decoded}')
print(f'Match: {sequence == decoded}')
"
```

**Expected:**
- Match: True ✅

---

## 📞 QUICK HELP

### Job stuck pending?
```bash
squeue -u $USER  # Check queue
scancel JOB_ID   # Cancel if needed
```

### Out of memory?
```bash
# Edit 01_fetch_sample_data.py
# Change NUM_SAMPLES = 50  (instead of 100)
```

### Need different GPU?
```bash
sbatch --gres=gpu:a100:1 run_gpu_pipeline.sh
# or
sbatch --partition=gpu-long run_gpu_pipeline.sh
```

---

## 🎯 SUCCESS METRICS

You're successful when:

1. ✅ Job completes without errors
2. ✅ `codebook_centroids.png` exists and looks good
3. ✅ Tokenizer encodes/decodes correctly
4. ✅ Utilization >95%
5. ✅ Tokenizer saved to `data/tokenizer_saved/`

**Then you're ready to integrate with LLM!** 🚀

---

## 📚 MORE INFO

- **COMPLETE_SUMMARY.md** - What was created
- **TOKENIZER_CLASS_README.md** - Class documentation  
- **GPU_COMMANDS.md** - Detailed GPU commands
- **PIPELINE_README.md** - Technical guide

---

## ⚡ TL;DR

```bash
cd /lustrefs/shared/mohammad.sayeed/Prot2Text/esmfold_tokenizer
sbatch run_gpu_pipeline.sh
tail -f slurm_esm_tokenizer_*.out
```

**Done!** ✅

