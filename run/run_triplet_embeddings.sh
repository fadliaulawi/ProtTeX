#!/bin/bash

# Parallel Triplet Extraction Script
# Distributes 72 batches across 8 GPUs (9 batches per GPU)
# Requires --model argument (llama, qwen, qwen2.7, deepseek-v2, deepseek-r1, deepseek-r1-distill)

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
            echo "Available models: llama, qwen, qwen2.7, deepseek-v2, deepseek-r1, deepseek-r1-distill"
            echo ""
            echo "Examples:"
            echo "  $0 --model llama --k 128"
            echo "  $0 --model qwen --k 128"
            echo "  $0 --model deepseek-v2 --k 128"
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
    echo "Available models: llama, qwen, qwen2.7, deepseek-v2, deepseek-r1, deepseek-r1-distill"
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

echo "=============================================="
echo "PARALLEL TRIPLET EXTRACTION - Model: $MODEL, K=$K"
echo "=============================================="
echo "Distributing 72 batches across 8 GPUs"
echo ""

# Create logs directory
mkdir -p logs

# GPU 0: batches 0-8
echo "Starting GPU 0 (batches 0-8)..."
CUDA_VISIBLE_DEVICES=0 python -u run/03_extract_triplet_embeddings.py --model $MODEL --k $K --batch-start 0 --batch-end 9 > logs/triplet_${MODEL}_gpu0_K${K}.log 2>&1 &
PID0=$!

# GPU 1: batches 9-17
echo "Starting GPU 1 (batches 9-17)..."
CUDA_VISIBLE_DEVICES=1 python -u run/03_extract_triplet_embeddings.py --model $MODEL --k $K --batch-start 9 --batch-end 18 > logs/triplet_${MODEL}_gpu1_K${K}.log 2>&1 &
PID1=$!

# GPU 2: batches 18-26
echo "Starting GPU 2 (batches 18-26)..."
CUDA_VISIBLE_DEVICES=2 python -u run/03_extract_triplet_embeddings.py --model $MODEL --k $K --batch-start 18 --batch-end 27 > logs/triplet_${MODEL}_gpu2_K${K}.log 2>&1 &
PID2=$!

# GPU 3: batches 27-35
echo "Starting GPU 3 (batches 27-35)..."
CUDA_VISIBLE_DEVICES=3 python -u run/03_extract_triplet_embeddings.py --model $MODEL --k $K --batch-start 27 --batch-end 36 > logs/triplet_${MODEL}_gpu3_K${K}.log 2>&1 &
PID3=$!

# GPU 4: batches 36-44
echo "Starting GPU 4 (batches 36-44)..."
CUDA_VISIBLE_DEVICES=4 python -u run/03_extract_triplet_embeddings.py --model $MODEL --k $K --batch-start 36 --batch-end 45 > logs/triplet_${MODEL}_gpu4_K${K}.log 2>&1 &
PID4=$!

# GPU 5: batches 45-53
echo "Starting GPU 5 (batches 45-53)..."
CUDA_VISIBLE_DEVICES=5 python -u run/03_extract_triplet_embeddings.py --model $MODEL --k $K --batch-start 45 --batch-end 54 > logs/triplet_${MODEL}_gpu5_K${K}.log 2>&1 &
PID5=$!

# GPU 6: batches 54-62
echo "Starting GPU 6 (batches 54-62)..."
CUDA_VISIBLE_DEVICES=6 python -u run/03_extract_triplet_embeddings.py --model $MODEL --k $K --batch-start 54 --batch-end 63 > logs/triplet_${MODEL}_gpu6_K${K}.log 2>&1 &
PID6=$!

# GPU 7: batches 63-71
echo "Starting GPU 7 (batches 63-71)..."
CUDA_VISIBLE_DEVICES=7 python -u run/03_extract_triplet_embeddings.py --model $MODEL --k $K --batch-start 63 --batch-end 72 > logs/triplet_${MODEL}_gpu7_K${K}.log 2>&1 &
PID7=$!

echo ""
echo "All 8 processes started!"
echo "  GPU 0 (PID $PID0): batches 0-8    -> logs/triplet_${MODEL}_gpu0_K${K}.log"
echo "  GPU 1 (PID $PID1): batches 9-17   -> logs/triplet_${MODEL}_gpu1_K${K}.log"
echo "  GPU 2 (PID $PID2): batches 18-26  -> logs/triplet_${MODEL}_gpu2_K${K}.log"
echo "  GPU 3 (PID $PID3): batches 27-35  -> logs/triplet_${MODEL}_gpu3_K${K}.log"
echo "  GPU 4 (PID $PID4): batches 36-44  -> logs/triplet_${MODEL}_gpu4_K${K}.log"
echo "  GPU 5 (PID $PID5): batches 45-53  -> logs/triplet_${MODEL}_gpu5_K${K}.log"
echo "  GPU 6 (PID $PID6): batches 54-62  -> logs/triplet_${MODEL}_gpu6_K${K}.log"
echo "  GPU 7 (PID $PID7): batches 63-71  -> logs/triplet_${MODEL}_gpu7_K${K}.log"
echo ""
echo "Monitor progress:"
echo "  tail -f logs/triplet_${MODEL}_gpu0_K${K}.log"
echo "  tail -f logs/triplet_${MODEL}_gpu1_K${K}.log"
echo "  tail -f logs/triplet_${MODEL}_gpu2_K${K}.log"
echo "  tail -f logs/triplet_${MODEL}_gpu3_K${K}.log"
echo "  tail -f logs/triplet_${MODEL}_gpu4_K${K}.log"
echo "  tail -f logs/triplet_${MODEL}_gpu5_K${K}.log"
echo "  tail -f logs/triplet_${MODEL}_gpu6_K${K}.log"
echo "  tail -f logs/triplet_${MODEL}_gpu7_K${K}.log"
echo ""
echo "Waiting for all processes to complete..."

# Wait for all background processes
wait $PID0
wait $PID1
wait $PID2
wait $PID3
wait $PID4
wait $PID5
wait $PID6
wait $PID7

echo ""
echo "=============================================="
echo "✅ ALL TRIPLET EXTRACTION COMPLETE!"
echo "=============================================="
echo "Check logs for details:"
echo "  logs/triplet_${MODEL}_gpu0_K${K}.log"
echo "  logs/triplet_${MODEL}_gpu1_K${K}.log"
echo "  logs/triplet_${MODEL}_gpu2_K${K}.log"
echo "  logs/triplet_${MODEL}_gpu3_K${K}.log"
echo "  logs/triplet_${MODEL}_gpu4_K${K}.log"
echo "  logs/triplet_${MODEL}_gpu5_K${K}.log"
echo "  logs/triplet_${MODEL}_gpu6_K${K}.log"
echo "  logs/triplet_${MODEL}_gpu7_K${K}.log"
