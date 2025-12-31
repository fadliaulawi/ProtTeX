#!/bin/bash
# Launch DDP training with torchrun for 8 GPUs
# Requires --model argument (llama, qwen, qwen2.7, deepseek-v2, deepseek-r1, deepseek-r1-distill)

# Check if model argument is provided
if [ $# -eq 0 ] || [[ ! "$*" =~ --model ]]; then
    echo "Error: --model argument is required"
    echo ""
    echo "Usage: $0 --model <model_type> [other_args...]"
    echo ""
    echo "Available models:"
    echo "  llama              - Meta Llama 3.1 8B Instruct"
    echo "  qwen               - Qwen 2.5 7B Instruct"
    echo "  qwen2.7            - Qwen 2.7 7B Instruct"
    echo "  deepseek-v2        - DeepSeek-V2-Chat"
    echo "  deepseek-r1        - DeepSeek-R1"
    echo "  deepseek-r1-distill - DeepSeek-R1-Distill-Qwen-7B"
    echo ""
    echo "Examples:"
    echo "  $0 --model llama --k 128 --epochs 3"
    echo "  $0 --model qwen --k 128 --epochs 3"
    echo "  $0 --model deepseek-v2 --k 128 --epochs 3"
    exit 1
fi

# Memory optimization for large models
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Use a higher port to avoid conflicts (29500 is default and often in use)
# If this port is also busy, change to 29503, 29504, etc.
MASTER_PORT=29505

# Extract model name from arguments for log filename
MODEL_NAME=""
prev_arg=""
for arg in "$@"; do
    if [[ "$prev_arg" == "--model" ]]; then
        MODEL_NAME="$arg"
        break
    fi
    prev_arg="$arg"
done

# Fallback if model name not found (shouldn't happen due to earlier check)
if [ -z "$MODEL_NAME" ]; then
    MODEL_NAME="unknown"
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Launch with torchrun (replaces torch.distributed.launch)
torchrun \
    --nproc_per_node=8 \
    --master_port=$MASTER_PORT \
    run/05_train_lora.py \
    "$@" \
    > logs/ddp/run_ddp_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log 2>&1

# Alternative with python -m torch.distributed.launch (older style):
# python -m torch.distributed.launch \
#     --nproc_per_node=8 \
#     --master_port=29501 \
#     run/05_train_lora.py \
#     "$@"
