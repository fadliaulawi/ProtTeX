#!/bin/bash
#SBATCH --job-name=esm_tokenizer
#SBATCH --output=slurm_esm_tokenizer_%j.out
#SBATCH --error=slurm_esm_tokenizer_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# ESM-2 Tokenizer Full Pipeline
# Runs all steps: fetch data -> extract embeddings -> train codebook -> demo

set -e  # Exit on error

echo "======================================================================"
echo "ESM-2 TOKENIZER PIPELINE - GPU JOB"
echo "======================================================================"
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo ""

# Navigate to directory
cd /lustrefs/shared/mohammad.sayeed/Prot2Text/esmfold_tokenizer

# Activate conda environment
source activate esm_tokenizer

# Check GPU
echo "======================================================================"
echo "GPU INFO"
echo "======================================================================"
nvidia-smi
echo ""

# Step 1: Fetch data
echo "======================================================================"
echo "STEP 1/4: Fetch Sample Data"
echo "======================================================================"
echo "⏱️  Estimated time: 5-10 minutes"
echo ""

python 01_fetch_sample_data.py

echo ""
echo "✅ Data fetching complete"
echo ""

# Step 2: Extract embeddings
echo "======================================================================"
echo "STEP 2/4: Extract ESM-2 Embeddings"
echo "======================================================================"
echo "⏱️  Estimated time: 30-45 minutes (depends on GPU)"
echo ""

python 02_extract_esm_embeddings.py

echo ""
echo "✅ Embedding extraction complete"
echo ""

# Step 3: Train codebook
echo "======================================================================"
echo "STEP 3/4: Train k-means Codebook"
echo "======================================================================"
echo "⏱️  Estimated time: 5-10 minutes"
echo ""

python 03_train_kmeans_codebook.py

echo ""
echo "✅ Codebook training complete"
echo ""

# Step 4: Demo
echo "======================================================================"
echo "STEP 4/5: Tokenization Demo"
echo "======================================================================"
echo ""

python 04_tokenize_and_demo.py

echo ""
echo "======================================================================"
echo "STEP 5/5: Tokenizer Class Usage Examples"
echo "======================================================================"
echo ""

python 05_tokenizer_usage_example.py

echo ""
echo "======================================================================"
echo "✅ PIPELINE COMPLETE!"
echo "======================================================================"
echo ""
echo "Finished: $(date)"
echo ""
echo "📁 Output files in: data/"
echo "   - sample_proteins.json"
echo "   - esm_embeddings.npy"
echo "   - structure_codebook_K512.pkl"
echo "   - clustering_visualization.png"
echo "   - codebook_centroids.png ⭐"
echo "   - tokenizer_saved/ (ready to push!)"
echo ""
echo "📊 Check visualizations:"
echo "   display data/clustering_visualization.png"
echo "   display data/codebook_centroids.png"
echo ""
echo "🚀 Tokenizer class ready:"
echo "   - protein_structure_tokenizer.py"
echo "   - See 05_tokenizer_usage_example.py for usage"
echo ""

