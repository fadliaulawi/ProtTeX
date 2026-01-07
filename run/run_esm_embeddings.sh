#!/bin/bash
# Flexible script to run ESM embeddings across 8 GPUs in parallel
# Usage: ./run_esm_embeddings.sh <num_samples> [batch_size]
# Example: ./run_esm_embeddings.sh 3788499 10000

# Check if number of samples is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <num_samples> [batch_size]"
    echo "  num_samples: Total number of samples to process"
    echo "  batch_size:  Number of samples per batch (default: 10000)"
    echo ""
    echo "Example: $0 3788499 10000"
    exit 1
fi

NUM_SAMPLES=$1
BATCH_SIZE=${2:-10000}  # Default to 10000 if not provided
NUM_GPUS=8

# Calculate number of batches (ceiling division)
NUM_BATCHES=$(( (NUM_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE ))

# Calculate batches per GPU (distribute evenly)
BATCHES_PER_GPU=$(( NUM_BATCHES / NUM_GPUS ))
REMAINDER=$(( NUM_BATCHES % NUM_GPUS ))

# Create logs directory
mkdir -p logs/esm

# Clear existing logs
rm -f logs/esm/gpu*.log

echo "=========================================="
echo "ESM Embeddings Parallel Processing"
echo "=========================================="
echo "Total samples:     $NUM_SAMPLES"
echo "Batch size:        $BATCH_SIZE"
echo "Total batches:     $NUM_BATCHES"
echo "Number of GPUs:    $NUM_GPUS"
echo "Batches per GPU:   $BATCHES_PER_GPU"
if [ $REMAINDER -gt 0 ]; then
    echo "Extra batches:     $REMAINDER (will be distributed to first $REMAINDER GPUs)"
fi
echo "=========================================="
echo ""
echo "Starting parallel processing on $NUM_GPUS GPUs..."
echo "Monitor progress with: tail -f logs/esm/gpu*.log"
echo ""

# Function to launch GPU processing
launch_gpu() {
    local gpu_id=$1
    local start_batch=$2
    local end_batch=$3
    
    (
        for i in $(seq $start_batch $end_batch); do
            echo "GPU $gpu_id: Processing batch $i" >> logs/esm/gpu${gpu_id}.log
            CUDA_VISIBLE_DEVICES=$gpu_id python -u run/01_extract_esm_embeddings.py $i >> logs/esm/gpu${gpu_id}.log 2>&1
        done
        echo "GPU $gpu_id: DONE!" >> logs/esm/gpu${gpu_id}.log
    ) &
}

# Distribute batches across GPUs
CURRENT_BATCH=0
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    # Calculate batches for this GPU
    if [ $gpu -lt $REMAINDER ]; then
        # First REMAINDER GPUs get one extra batch
        GPU_BATCHES=$((BATCHES_PER_GPU + 1))
    else
        GPU_BATCHES=$BATCHES_PER_GPU
    fi
    
    START_BATCH=$CURRENT_BATCH
    END_BATCH=$((CURRENT_BATCH + GPU_BATCHES - 1))
    
    # Don't exceed total number of batches
    if [ $END_BATCH -ge $NUM_BATCHES ]; then
        END_BATCH=$((NUM_BATCHES - 1))
    fi
    
    if [ $START_BATCH -lt $NUM_BATCHES ]; then
        echo "GPU $gpu: batches $START_BATCH-$END_BATCH ($GPU_BATCHES batches)"
        launch_gpu $gpu $START_BATCH $END_BATCH
        CURRENT_BATCH=$((END_BATCH + 1))
    fi
done

echo ""
echo "All GPUs launched. Waiting for completion..."
echo ""

# Wait for all GPUs to finish
wait

echo ""
echo "=========================================="
echo "✅ All $NUM_BATCHES batches completed!"
echo "=========================================="
echo "Check logs in: logs/esm/gpu*.log"
