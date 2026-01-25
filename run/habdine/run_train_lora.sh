#!/bin/bash
# Launch LoRA training (03b) with torchrun for 8 GPUs
# Data: data/habdine/triplet_embeddings, data/habdine/clip_alignment (from 03a)
# Output: data/habdine/lora/{model}/K{k}, data/habdine/evaluation/{model}/K{k}/test_data.json
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
    echo "Usage: $0 --model <model_type> --k <k_clusters> [--epochs <epochs>]"
    echo ""
    echo "Available models: llama, qwen, deepseek"
    echo ""
    echo "Examples:"
    echo "  $0 --model llama --k 1024"
    echo "  $0 --model qwen --k 1024 --epochs 5"
    exit 1
fi

if [[ ! "$*" =~ --k ]]; then
    echo "Error: --k argument is required"
    echo ""
    echo "Usage: $0 --model <model_type> --k <k_clusters> [--epochs <epochs>]"
    echo ""
    echo "Examples:"
    echo "  $0 --model llama --k 1024"
    echo "  $0 --model qwen --k 1024 --epochs 5"
    exit 1
fi

# Memory optimization for large models
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Use a higher port to avoid conflicts (29500 is default and often in use)
# If this port is also busy, change to 29503, 29504, etc.
MASTER_PORT=29505

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
mkdir -p logs/habdine/train

# Change to project root
cd "$(dirname "$0")/../.."

# Launch with torchrun (replaces torch.distributed.launch)
torchrun \
    --nproc_per_node=8 \
    --master_port=$MASTER_PORT \
    run/habdine/03b_train_lora.py \
    "$@" \
    > logs/habdine/train/train_lora_${MODEL_NAME}_K${K_VALUE}_$(date +%Y%m%d_%H%M%S).log 2>&1
