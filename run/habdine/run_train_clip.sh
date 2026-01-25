#!/bin/bash
# Launch CLIP alignment training (03a, single GPU)
# Data: data/habdine/triplet_embeddings/{model}/K{k} (pre-split: train/validation)
# Output: data/habdine/clip_alignment/{model}_K{k}
# Requires: --model, --k

# Check if WANDB_API_KEY is set
if [ -z "$WANDB_API_KEY" ]; then
    echo "⚠️  WARNING: WANDB_API_KEY is not set!"
    echo ""
    echo "Please export your WANDB API key first:"
    echo "  export WANDB_API_KEY=your_api_key_here"
    echo ""
    echo "You can get your API key from: https://wandb.ai/authorize"
    echo ""
    read -p "Do you want to continue without wandb logging? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Please set WANDB_API_KEY and try again."
        exit 1
    fi
fi

# Check if model and k arguments are provided
if [ $# -eq 0 ] || [[ ! "$*" =~ --model ]]; then
    echo "Error: --model argument is required"
    echo ""
    echo "Usage: $0 --model <model_type> --k <k_clusters> [--epochs <epochs>] [--batch-size <batch_size>]"
    echo ""
    echo "Available models: llama, qwen, deepseek"
    echo ""
    echo "Examples:"
    echo "  $0 --model llama --k 1024"
    echo "  $0 --model qwen --k 1024 --epochs 50 --batch-size 1024"
    exit 1
fi

if [[ ! "$*" =~ --k ]]; then
    echo "Error: --k argument is required"
    echo ""
    echo "Usage: $0 --model <model_type> --k <k_clusters> [--epochs <epochs>] [--batch-size <batch_size>]"
    echo ""
    echo "Examples:"
    echo "  $0 --model llama --k 1024"
    echo "  $0 --model qwen --k 1024 --epochs 50"
    exit 1
fi

# Extract model name and K value from arguments for log filename
MODEL_NAME=""
K_VALUE=""
prev_arg=""
for arg in "$@"; do
    if [[ "$prev_arg" == "--model" ]]; then
        MODEL_NAME="$arg"
    elif [[ "$prev_arg" == "--k" ]]; then
        K_VALUE="$arg"
    fi
    prev_arg="$arg"
done

# Create logs directory if it doesn't exist
mkdir -p logs/habdine/clip

# Run CLIP alignment training (single GPU)
cd "$(dirname "$0")/../.."
python run/habdine/03a_train_clip_alignment.py \
    "$@" \
    > logs/habdine/clip/clip_${MODEL_NAME}_K${K_VALUE}_$(date +%Y%m%d_%H%M%S).log 2>&1
