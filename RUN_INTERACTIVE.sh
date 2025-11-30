#!/bin/bash
# Interactive GPU Session Commands
# Copy-paste these commands one by one

# ============================================================================
# STEP 1: Get GPU Node (run this first)
# ============================================================================
srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=2:00:00 --pty bash

# ============================================================================
# Once you're on the GPU node, run these:
# ============================================================================

# STEP 2: Activate conda environment
conda activate protein_env

# STEP 3: Navigate to directory
cd /lustrefs/shared/mohammad.sayeed/Prot2Text/esmfold_tokenizer

# STEP 4: Verify GPU
nvidia-smi
echo ""
echo "GPU ready! Starting pipeline..."
echo ""

# ============================================================================
# OPTION A: Run all steps sequentially
# ============================================================================
echo "=== Step 1/5: Fetch Data ==="
python 01_fetch_sample_data.py

echo ""
echo "=== Step 2/5: Extract ESM-2 Embeddings ==="
python 02_extract_esm_embeddings.py

echo ""
echo "=== Step 3/5: Train k-means Codebook ==="
python 03_train_kmeans_codebook.py

echo ""
echo "=== Step 4/5: Tokenization Demo ==="
python 04_tokenize_and_demo.py

echo ""
echo "=== Step 5/5: Usage Examples ==="
python 05_tokenizer_usage_example.py

echo ""
echo "✅ ALL DONE!"
echo ""
echo "Check outputs:"
echo "  ls -lh data/"
echo "  display data/codebook_centroids.png"

# ============================================================================
# OPTION B: Run steps one by one (if you want to check each step)
# ============================================================================

# Step 1 (5 min):
# python 01_fetch_sample_data.py

# Step 2 (30-45 min):
# python 02_extract_esm_embeddings.py

# Step 3 (5 min):
# python 03_train_kmeans_codebook.py

# Step 4 (5 min):
# python 04_tokenize_and_demo.py

# Step 5 (5 min):
# python 05_tokenizer_usage_example.py




# Interactive GPU Session Commands
# Copy-paste these commands one by one

# ============================================================================
# STEP 1: Get GPU Node (run this first)
# ============================================================================
srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=2:00:00 --pty bash

# ============================================================================
# Once you're on the GPU node, run these:
# ============================================================================

# STEP 2: Activate conda environment
conda activate protein_env

# STEP 3: Navigate to directory
cd /lustrefs/shared/mohammad.sayeed/Prot2Text/esmfold_tokenizer

# STEP 4: Verify GPU
nvidia-smi
echo ""
echo "GPU ready! Starting pipeline..."
echo ""

# ============================================================================
# OPTION A: Run all steps sequentially
# ============================================================================
echo "=== Step 1/5: Fetch Data ==="
python 01_fetch_sample_data.py

echo ""
echo "=== Step 2/5: Extract ESM-2 Embeddings ==="
python 02_extract_esm_embeddings.py

echo ""
echo "=== Step 3/5: Train k-means Codebook ==="
python 03_train_kmeans_codebook.py

echo ""
echo "=== Step 4/5: Tokenization Demo ==="
python 04_tokenize_and_demo.py

echo ""
echo "=== Step 5/5: Usage Examples ==="
python 05_tokenizer_usage_example.py

echo ""
echo "✅ ALL DONE!"
echo ""
echo "Check outputs:"
echo "  ls -lh data/"
echo "  display data/codebook_centroids.png"

# ============================================================================
# OPTION B: Run steps one by one (if you want to check each step)
# ============================================================================

# Step 1 (5 min):
# python 01_fetch_sample_data.py

# Step 2 (30-45 min):
# python 02_extract_esm_embeddings.py

# Step 3 (5 min):
# python 03_train_kmeans_codebook.py

# Step 4 (5 min):
# python 04_tokenize_and_demo.py

# Step 5 (5 min):
# python 05_tokenizer_usage_example.py




