#!/bin/bash

# Parallel Inference (04) across 8 GPUs
# Input: JSON list of {QA: {sequence, question, answer}, metadata: {id, type, subset, ...}}
# Output: data/habdine/evaluation/{model}/K{K}/{input_stem}_predictions.json
# Requires: --model, --k, --input

# Parse arguments
MODEL=""
K=""
INPUT=""
MAX_TOKENS=256
TEMPERATURE=0.7
TOP_P=0.9

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
        --input)
            INPUT="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top-p)
            TOP_P="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            echo "Usage: $0 --model <model_type> --k <K> --input <input_file> [--max-tokens <N>] [--temperature <T>] [--top-p <P>]"
            echo ""
            echo "Input: JSON list of {QA: {sequence, question, answer}, metadata: {id, type, subset, ...}}"
            echo "Output: data/habdine/evaluation/{model}/K{K}/{input_stem}_predictions.json"
            echo ""
            echo "Examples:"
            echo "  $0 --model llama --k 1024 --input data/habdine/evaluation/llama/K1024/test_data.json"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL" ]; then
    echo "Error: --model argument is required"
    exit 1
fi

if [ -z "$K" ]; then
    echo "Error: --k argument is required"
    exit 1
fi

if [ -z "$INPUT" ]; then
    echo "Error: --input argument is required"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Check if input file exists
INPUT_FILE="$ROOT_DIR/$INPUT"
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Error: Input file not found: $INPUT_FILE"
    exit 1
fi

# Generate output directory and filename
INPUT_STEM="$(basename "$INPUT_FILE" .json)"
OUTPUT_DIR="${ROOT_DIR}/data/habdine/evaluation/${MODEL}/K${K}"
OUTPUT_FILENAME="${INPUT_STEM}_predictions.json"
OUTPUT="${OUTPUT_DIR}/${OUTPUT_FILENAME}"

# Count total proteins in the input file
TOTAL_PROTEINS=$(python3 -c "import json; data = json.load(open('$INPUT_FILE')); print(len(data))" 2>/dev/null)

if [ -z "$TOTAL_PROTEINS" ] || [ "$TOTAL_PROTEINS" -eq 0 ]; then
    echo "❌ Error: Could not determine number of proteins in $INPUT_FILE"
    exit 1
fi

NUM_GPUS=8
PROTEINS_PER_GPU=$((TOTAL_PROTEINS / NUM_GPUS))
REMAINDER=$((TOTAL_PROTEINS % NUM_GPUS))

echo "=============================================="
echo "PARALLEL INFERENCE"
echo "=============================================="
echo "Model: $MODEL"
echo "K: $K"
echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT"
echo "Total proteins: $TOTAL_PROTEINS"
echo "Number of GPUs: $NUM_GPUS"
echo "Proteins per GPU: ~$PROTEINS_PER_GPU"
if [ $REMAINDER -gt 0 ]; then
    echo "Extra proteins: $REMAINDER (will be distributed to first $REMAINDER GPUs)"
fi
echo ""

# Create logs directory and run from project root (relative --input, logs)
mkdir -p "$ROOT_DIR/logs/habdine/evaluation"
cd "$ROOT_DIR"

# Distribute proteins across GPUs
CURRENT_IDX=0
GPU_PIDS=()
GPU_INFO=()
OUTPUT_FILES=()

for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    # Calculate protein range for this GPU
    if [ $gpu -lt $REMAINDER ]; then
        # First REMAINDER GPUs get one extra protein
        GPU_PROTEINS=$((PROTEINS_PER_GPU + 1))
    else
        GPU_PROTEINS=$PROTEINS_PER_GPU
    fi
    
    START_IDX=$CURRENT_IDX
    END_IDX=$((CURRENT_IDX + GPU_PROTEINS))
    
    # Don't exceed total proteins (end_idx is exclusive in Python)
    if [ $END_IDX -gt $TOTAL_PROTEINS ]; then
        END_IDX=$TOTAL_PROTEINS
    fi
    
    if [ $START_IDX -lt $TOTAL_PROTEINS ]; then
        # Calculate GPU-specific temp output filename (for combining later)
        OUTPUT_BASE=$(basename "$OUTPUT" .json)
        OUTPUT_DIR_PATH=$(dirname "$OUTPUT")
        GPU_OUTPUT="${OUTPUT_DIR_PATH}/${OUTPUT_BASE}_gpu${gpu}.json"
        OUTPUT_FILES+=("$GPU_OUTPUT")
        
        # Launch GPU process and capture PID
        echo "Starting GPU $gpu (proteins $START_IDX-$((END_IDX - 1)))..."
        
        # Build command: 04_inference uses --start-index/--end-index
        CMD="CUDA_VISIBLE_DEVICES=$gpu python -u $SCRIPT_DIR/04_inference.py \
            --model $MODEL --k $K \
            --input $INPUT --output $GPU_OUTPUT \
            --start-index $START_IDX --end-index $END_IDX \
            --max-tokens $MAX_TOKENS --temperature $TEMPERATURE --top-p $TOP_P"
        
        eval "$CMD > logs/habdine/evaluation/${MODEL}_K${K}_gpu${gpu}.log 2>&1 &"
        PID=$!
        GPU_PIDS+=($PID)
        GPU_INFO+=("$gpu:$START_IDX:$((END_IDX - 1))")
        CURRENT_IDX=$END_IDX
        
        # Break if we've assigned all proteins
        if [ $CURRENT_IDX -ge $TOTAL_PROTEINS ]; then
            break
        fi
    fi
done

echo ""
echo "All ${#GPU_PIDS[@]} processes started!"
for i in "${!GPU_PIDS[@]}"; do
    OLD_IFS=$IFS
    IFS=':' read -r gpu_id start_idx end_idx <<< "${GPU_INFO[$i]}"
    IFS=$OLD_IFS
    echo "  GPU $gpu_id (PID ${GPU_PIDS[$i]}): proteins $start_idx-$end_idx -> logs/habdine/evaluation/${MODEL}_K${K}_gpu${gpu_id}.log"
done
echo ""
echo "Monitor progress:"
for i in "${!GPU_PIDS[@]}"; do
    OLD_IFS=$IFS
    IFS=':' read -r gpu_id start_idx end_idx <<< "${GPU_INFO[$i]}"
    IFS=$OLD_IFS
    echo "  tail -f logs/habdine/evaluation/${MODEL}_K${K}_gpu${gpu_id}.log"
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
    echo "✅ ALL INFERENCE COMPLETE!"
else
    echo "⚠️  COMPLETED WITH $FAILED FAILED PROCESS(ES)"
fi
echo "=============================================="
echo "Processed $TOTAL_PROTEINS proteins"
echo ""

# Combine output files
echo "Combining output files..."

# Build Python array string for output files (properly quoted)
OUTPUT_FILES_STR=""
for file in "${OUTPUT_FILES[@]}"; do
    OUTPUT_FILES_STR="${OUTPUT_FILES_STR}'${file}', "
done
OUTPUT_FILES_STR="[${OUTPUT_FILES_STR%, }]"

python3 << EOF
import json
import sys
from pathlib import Path

output_files = $OUTPUT_FILES_STR
root_dir = Path('$ROOT_DIR')

all_predictions = []
for output_file in output_files:
    path = Path(output_file)
    if path.exists():
        with open(path, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'results' in data:
            all_predictions.extend(data['results'])
        elif isinstance(data, list):
            all_predictions.extend(data)
        else:
            print(f'⚠️  Warning: Unexpected format in {path}', file=sys.stderr)
    else:
        print(f'⚠️  Warning: Output file not found: {path}', file=sys.stderr)

# Sort by id or protein_id
try:
    all_predictions.sort(key=lambda x: x.get('id', x.get('protein_id', '')))
except Exception:
    pass

# Save combined output: {results, evaluation_summary} to match 04_inference
output_path = Path('$OUTPUT')
output_path.parent.mkdir(parents=True, exist_ok=True)
combined = {'results': all_predictions, 'evaluation_summary': {}}
with open(output_path, 'w') as f:
    json.dump(combined, f, indent=2)

print(f'✅ Combined {len(all_predictions)} predictions into {output_path}')

# Clean up temporary GPU output files
print('Cleaning up temporary files...')
for output_file in output_files:
    file_path = root_dir / output_file
    if file_path.exists():
        file_path.unlink()
        print(f'  Deleted: {file_path}')
EOF

echo ""
echo "Final output: $OUTPUT"
echo "Check logs for details:"
for i in "${!GPU_PIDS[@]}"; do
    OLD_IFS=$IFS
    IFS=':' read -r gpu_id start_idx end_idx <<< "${GPU_INFO[$i]}"
    IFS=$OLD_IFS
    echo "  logs/habdine/evaluation/${MODEL}_K${K}_gpu${gpu_id}.log"
done
