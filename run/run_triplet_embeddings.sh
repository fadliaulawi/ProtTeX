#!/bin/bash

# Parallel Triplet Extraction Script
# Automatically detects available ESM embedding batch files and distributes them across 8 GPUs
# Requires --model argument (llama, qwen, deepseek) and --k argument

# Parse arguments
MODEL=""
K=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --k)
            K="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            echo "Usage: $0 --model <model_type> --k <K>"
            echo ""
            echo "Available models: llama, qwen, deepseek"
            echo ""
            echo "Examples:"
            echo "  $0 --model llama --k 128"
            echo "  $0 --model qwen --k 128"
            echo "  $0 --model deepseek --k 128"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL" ]; then
    echo "Error: --model argument is required"
    echo ""
    echo "Usage: $0 --model <model_type> --k <K>"
    echo ""
    echo "Available models: llama, qwen, deepseek"
    echo ""
    echo "Examples:"
    echo "  $0 --model llama --k 128"
    echo "  $0 --model qwen --k 128"
    exit 1
fi

if [ -z "$K" ]; then
    echo "Error: --k argument is required"
    echo ""
    echo "Usage: $0 --model <model_type> --k <K>"
    echo ""
    echo "Example: $0 --model llama --k 128"
    exit 1
fi

# Get script directory and determine data directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"
EMBEDDINGS_DIR="$DATA_DIR/esm_embeddings"

# Check if embeddings directory exists
if [ ! -d "$EMBEDDINGS_DIR" ]; then
    echo "❌ Error: Embeddings directory not found: $EMBEDDINGS_DIR"
    echo "   Run 01_extract_esm_embeddings.py first!"
    exit 1
fi

# Find all ESM embedding batch files and extract batch numbers
BATCH_FILES=($(find "$EMBEDDINGS_DIR" -name "esm_embeddings_batch_*.npy" | sort -V))

if [ ${#BATCH_FILES[@]} -eq 0 ]; then
    echo "❌ Error: No ESM embedding batch files found in $EMBEDDINGS_DIR"
    echo "   Run 01_extract_esm_embeddings.py first!"
    exit 1
fi

# Extract batch numbers from filenames
BATCH_NUMS=()
for file in "${BATCH_FILES[@]}"; do
    # Extract batch number from filename: esm_embeddings_batch_123.npy -> 123
    batch_num=$(basename "$file" | sed -n 's/esm_embeddings_batch_\([0-9]*\)\.npy/\1/p')
    if [[ "$batch_num" =~ ^[0-9]+$ ]]; then
        BATCH_NUMS+=($batch_num)
    fi
done

# Sort batch numbers numerically
IFS=$'\n' BATCH_NUMS=($(sort -n <<<"${BATCH_NUMS[*]}"))
unset IFS

NUM_BATCHES=${#BATCH_NUMS[@]}
MIN_BATCH=${BATCH_NUMS[0]}
MAX_BATCH=${BATCH_NUMS[-1]}
NUM_GPUS=8

# Calculate total batch range (assuming consecutive batches from min to max)
# If batches are not consecutive, we'll still distribute the range evenly
TOTAL_BATCH_RANGE=$((MAX_BATCH - MIN_BATCH + 1))
BATCHES_PER_GPU=$(( TOTAL_BATCH_RANGE / NUM_GPUS ))
REMAINDER=$(( TOTAL_BATCH_RANGE % NUM_GPUS ))

echo "=============================================="
echo "PARALLEL TRIPLET EXTRACTION - Model: $MODEL, K=$K"
echo "=============================================="
echo "Embeddings directory: $EMBEDDINGS_DIR"
echo "Found $NUM_BATCHES batch files (batch $MIN_BATCH to $MAX_BATCH)"
echo "Total batch range: $TOTAL_BATCH_RANGE (from $MIN_BATCH to $MAX_BATCH)"
echo "Number of GPUs: $NUM_GPUS"
echo "Batches per GPU: ~$BATCHES_PER_GPU"
if [ $REMAINDER -gt 0 ]; then
    echo "Extra batches: $REMAINDER (will be distributed to first $REMAINDER GPUs)"
fi
echo ""

# Create logs directory
mkdir -p logs/triplet

# Distribute batches across GPUs
CURRENT_BATCH=$MIN_BATCH
GPU_PIDS=()
GPU_INFO=()

for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    # Calculate batches for this GPU
    if [ $gpu -lt $REMAINDER ]; then
        # First REMAINDER GPUs get one extra batch
        GPU_BATCHES=$((BATCHES_PER_GPU + 1))
    else
        GPU_BATCHES=$BATCHES_PER_GPU
    fi
    
    START_BATCH=$CURRENT_BATCH
    END_BATCH=$((CURRENT_BATCH + GPU_BATCHES))
    
    # Don't exceed max batch (end_batch is exclusive in Python, so we use MAX_BATCH + 1)
    if [ $END_BATCH -gt $((MAX_BATCH + 1)) ]; then
        END_BATCH=$((MAX_BATCH + 1))
    fi
    
    if [ $START_BATCH -le $MAX_BATCH ]; then
        # Launch GPU process and capture PID
        echo "Starting GPU $gpu (batches $START_BATCH-$((END_BATCH - 1)))..."
        CUDA_VISIBLE_DEVICES=$gpu python -u run/03_extract_triplet_embeddings.py \
            --model $MODEL --k $K \
            --batch-start $START_BATCH --batch-end $END_BATCH \
            > logs/triplet/${MODEL}_gpu${gpu}_K${K}.log 2>&1 &
        PID=$!
        GPU_PIDS+=($PID)
        GPU_INFO+=("$gpu:$START_BATCH:$((END_BATCH - 1))")
        CURRENT_BATCH=$END_BATCH
        
        # Break if we've assigned all batches
        if [ $CURRENT_BATCH -gt $MAX_BATCH ]; then
            break
        fi
    fi
done

echo ""
echo "All ${#GPU_PIDS[@]} processes started!"
for i in "${!GPU_PIDS[@]}"; do
    OLD_IFS=$IFS
    IFS=':' read -r gpu_id start_b end_b <<< "${GPU_INFO[$i]}"
    IFS=$OLD_IFS
    echo "  GPU $gpu_id (PID ${GPU_PIDS[$i]}): batches $start_b-$end_b -> logs/triplet/${MODEL}_gpu${gpu_id}_K${K}.log"
done
echo ""
echo "Monitor progress:"
for i in "${!GPU_PIDS[@]}"; do
    OLD_IFS=$IFS
    IFS=':' read -r gpu_id start_b end_b <<< "${GPU_INFO[$i]}"
    IFS=$OLD_IFS
    echo "  tail -f logs/triplet/${MODEL}_gpu${gpu_id}_K${K}.log"
done
echo ""
echo "Waiting for all processes to complete..."

# Wait for all background processes
for pid in "${GPU_PIDS[@]}"; do
    wait $pid
done

echo ""
echo "=============================================="
echo "✅ ALL TRIPLET EXTRACTION COMPLETE!"
echo "=============================================="
echo "Processed $NUM_BATCHES batches (batch $MIN_BATCH to $MAX_BATCH)"
echo ""
echo "Check logs for details:"
for i in "${!GPU_PIDS[@]}"; do
    OLD_IFS=$IFS
    IFS=':' read -r gpu_id start_b end_b <<< "${GPU_INFO[$i]}"
    IFS=$OLD_IFS
    echo "  logs/triplet/${MODEL}_gpu${gpu_id}_K${K}.log"
done
