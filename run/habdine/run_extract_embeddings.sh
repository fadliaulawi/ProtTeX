#!/bin/bash

# Parallel Embedding Extraction Script for Prot2Text Data
# Distributes a single split across 8 GPUs
# Requires --model, --k, and --split arguments

# Parse arguments
MODEL=""
K=""
SPLIT=""

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
        --split)
            SPLIT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            echo "Usage: $0 --model <model_type> --k <K> --split <split>"
            echo ""
            echo "Available models: llama, qwen, deepseek"
            echo "Available splits: train, validation, test"
            echo ""
            echo "Examples:"
            echo "  $0 --model llama --k 128 --split train"
            echo "  $0 --model qwen --k 256 --split validation"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL" ]; then
    echo "Error: --model argument is required"
    echo ""
    echo "Usage: $0 --model <model_type> --k <K> --split <split>"
    exit 1
fi

if [ -z "$K" ]; then
    echo "Error: --k argument is required"
    echo ""
    echo "Usage: $0 --model <model_type> --k <K> --split <split>"
    exit 1
fi

if [ -z "$SPLIT" ]; then
    echo "Error: --split argument is required"
    echo ""
    echo "Usage: $0 --model <model_type> --k <K> --split <split>"
    echo "Available splits: train, validation, test"
    exit 1
fi

if [[ ! "$SPLIT" =~ ^(train|validation|test)$ ]]; then
    echo "Error: Invalid split '$SPLIT'"
    echo "Available splits: train, validation, test"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$ROOT_DIR/data/habdine"

# Check if input file exists
INPUT_FILE="$DATA_DIR/standardized_prot2text_${SPLIT}.json"

if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Error: Input file not found: $INPUT_FILE"
    echo "   Run 01_download_preprocess.py first!"
    exit 1
fi

# Check codebook
CODEBOOK_FILE="$ROOT_DIR/data/codebooks/structure_codebook_K${K}.pkl"
if [ ! -f "$CODEBOOK_FILE" ]; then
    echo "❌ Error: Codebook not found: $CODEBOOK_FILE"
    echo "   Run 02_train_kmeans_codebook.py first!"
    exit 1
fi

# Count total proteins in the split (using Python for JSON parsing)
TOTAL_PROTEINS=$(python3 -c "import json; data = json.load(open('$INPUT_FILE')); print(len(data))" 2>/dev/null)

if [ -z "$TOTAL_PROTEINS" ] || [ "$TOTAL_PROTEINS" -eq 0 ]; then
    echo "❌ Error: Could not determine number of proteins in $INPUT_FILE"
    exit 1
fi

# Calculate total number of batches (1000 proteins per batch)
BATCH_SIZE=1000
TOTAL_BATCHES=$(( (TOTAL_PROTEINS + BATCH_SIZE - 1) / BATCH_SIZE ))
MIN_BATCH=0
MAX_BATCH=$((TOTAL_BATCHES - 1))

NUM_GPUS=8
BATCHES_PER_GPU=$((TOTAL_BATCHES / NUM_GPUS))
REMAINDER=$((TOTAL_BATCHES % NUM_GPUS))

echo "=============================================="
echo "PARALLEL EMBEDDING EXTRACTION"
echo "=============================================="
echo "Model: $MODEL"
echo "K: $K"
echo "Split: $SPLIT"
echo "Input file: $INPUT_FILE"
echo "Total proteins: $TOTAL_PROTEINS"
echo "Total batches: $TOTAL_BATCHES (batch $MIN_BATCH to $MAX_BATCH)"
echo "Number of GPUs: $NUM_GPUS"
echo "Batches per GPU: ~$BATCHES_PER_GPU"
if [ $REMAINDER -gt 0 ]; then
    echo "Extra batches: $REMAINDER (will be distributed to first $REMAINDER GPUs)"
fi
echo ""

# Create logs directory
mkdir -p logs/habdine/extract_embeddings

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
        CUDA_VISIBLE_DEVICES=$gpu python -u "$SCRIPT_DIR/02_extract_embeddings.py" \
            --model $MODEL --k $K --split $SPLIT \
            --batch-start $START_BATCH --batch-end $END_BATCH \
            > logs/habdine/extract_embeddings/${MODEL}_${SPLIT}_gpu${gpu}_K${K}.log 2>&1 &
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
    echo "  GPU $gpu_id (PID ${GPU_PIDS[$i]}): batches $start_b-$end_b -> logs/habdine/extract_embeddings/${MODEL}_${SPLIT}_gpu${gpu_id}_K${K}.log"
done
echo ""
echo "Monitor progress:"
for i in "${!GPU_PIDS[@]}"; do
    OLD_IFS=$IFS
    IFS=':' read -r gpu_id start_b end_b <<< "${GPU_INFO[$i]}"
    IFS=$OLD_IFS
    echo "  tail -f logs/habdine/extract_embeddings/${MODEL}_${SPLIT}_gpu${gpu_id}_K${K}.log"
done
echo ""
echo "Waiting for all processes to complete..."

# Wait for all background processes
FAILED=0
for pid in "${GPU_PIDS[@]}"; do
    wait $pid
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "⚠️  Warning: Process $pid exited with code $EXIT_CODE"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=============================================="
if [ $FAILED -eq 0 ]; then
    echo "✅ ALL EMBEDDING EXTRACTION COMPLETE!"
else
    echo "⚠️  COMPLETED WITH $FAILED FAILED PROCESS(ES)"
fi
echo "=============================================="
echo "Processed $TOTAL_BATCHES batches (batch $MIN_BATCH to $MAX_BATCH) from $SPLIT split"
echo ""
echo "Check logs for details:"
for i in "${!GPU_PIDS[@]}"; do
    OLD_IFS=$IFS
    IFS=':' read -r gpu_id start_b end_b <<< "${GPU_INFO[$i]}"
    IFS=$OLD_IFS
    echo "  logs/habdine/extract_embeddings/${MODEL}_${SPLIT}_gpu${gpu_id}_K${K}.log"
done

# Check output files
OUTPUT_DIR="$DATA_DIR/triplet_embeddings/$MODEL/K${K}"
echo ""
echo "Output directory: $OUTPUT_DIR"
if [ -d "$OUTPUT_DIR" ]; then
    echo "Generated files for $SPLIT split:"
    ls -lh "$OUTPUT_DIR" | grep -E "triplet_embeddings_${SPLIT}_batch_" | head -5
    echo "  ... (showing first 5 files)"
fi
