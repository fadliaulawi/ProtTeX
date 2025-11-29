#!/bin/bash
# Quick script to run the tokenizer demo

echo "======================================================================"
echo "Running Protein Structure Tokenizer Demo"
echo "======================================================================"
echo ""
echo "This will demonstrate:"
echo "  - Character-level amino acid tokenization"
echo "  - Structure tokenization using ESM-2 embeddings"
echo "  - Combined multimodal token stream generation"
echo ""
echo "Requirements: GPU with CUDA (for ESM-2 inference)"
echo ""
echo "======================================================================"
echo ""

cd /lustrefs/shared/mohammad.sayeed/Prot2Text/esmfold_tokenizer

# Check GPU
echo "Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo ""

# Run demo using conda environment's python directly
CONDA_ENV_PYTHON=/lustrefs/shared/mohammad.sayeed/caches/conda/envs/esm_tokenizer/bin/python
$CONDA_ENV_PYTHON demo_tokenizer_class.py

echo ""
echo "======================================================================"
echo "Demo complete! Check TOKENIZER_USAGE.md for more details."
echo "======================================================================"
