#!/bin/bash

# Parallel Inference Script
# Distributes inference workload across 8 GPUs
# Requires --model, --input, and --output arguments
# Optionally accepts --k, --variants, --test-batches

# Parse arguments
MODEL=""
INPUT=""
OUTPUT=""
K="128"
VARIANTS="all"
TEST_BATCHES="4"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --input)
            INPUT="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --k)
            K="$2"
            shift 2
            ;;
        --variants)
            VARIANTS="$2"
            shift 2
            ;;
        --test-batches)
            TEST_BATCHES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            echo "Usage: $0 --model <model_type> --input <input_file> --output <output_file> [--k <K>] [--variants <variants>] [--test-batches <N>]"
            echo ""
            echo "Available models: llama, qwen, deepseek"
            echo ""
            echo "Examples:"
            echo "  $0 --model deepseek --input run/input_inference.json --output run/output_inference.json"
            echo "  $0 --model qwen --input run/input_inference.json --output run/output_inference.json --k 128 --variants 1,5"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL" ]; then
    echo "Error: --model argument is required"
    echo ""
    echo "Usage: $0 --model <model_type> --input <input_file> --output <output_file> [other_args...]"
    exit 1
fi

if [ -z "$INPUT" ]; then
    echo "Error: --input argument is required"
    echo ""
    echo "Usage: $0 --model <model_type> --input <input_file> --output <output_file> [other_args...]"
    exit 1
fi

if [ -z "$OUTPUT" ]; then
    echo "Error: --output argument is required"
    echo ""
    echo "Usage: $0 --model <model_type> --input <input_file> --output <output_file> [other_args...]"
    exit 1
fi

# Get total number of items in input JSON
TOTAL_ITEMS=$(python3 -c "import json; data = json.load(open('$INPUT')); print(len(data) if isinstance(data, list) else 0)")
CHUNK_SIZE=$((TOTAL_ITEMS / 8))
REMAINDER=$((TOTAL_ITEMS % 8))

echo "=============================================="
echo "PARALLEL INFERENCE - Model: $MODEL, K=$K"
echo "=============================================="
echo "Total items: $TOTAL_ITEMS"
echo "Distributing across 8 GPUs (chunk size: $CHUNK_SIZE, remainder: $REMAINDER)"
echo "Input: $INPUT"
echo "Output: $OUTPUT"
echo "Variants: $VARIANTS"
echo ""

# Extract output directory and base filename for GPU-specific outputs
OUTPUT_DIR=$(dirname "$OUTPUT")
OUTPUT_BASE=$(basename "$OUTPUT")

# GPU 0: items 0 to CHUNK_SIZE-1
START0=0
END0=$CHUNK_SIZE
if [ $REMAINDER -gt 0 ]; then
    END0=$((END0 + 1))
    REMAINDER=$((REMAINDER - 1))
fi
echo "Starting GPU 0 (items $START0-$((END0-1)))..."
CUDA_VISIBLE_DEVICES=0 python -u run/06_inference.py --model $MODEL --k $K --input "$INPUT" --output "$OUTPUT" --variants "$VARIANTS" --test-batches $TEST_BATCHES --start-index $START0 --end-index $END0 --gpu-id 0 > logs/inference/${MODEL}_gpu0_K${K}.log 2>&1 &
PID0=$!

# GPU 1
START1=$END0
END1=$((START1 + CHUNK_SIZE))
if [ $REMAINDER -gt 0 ]; then
    END1=$((END1 + 1))
    REMAINDER=$((REMAINDER - 1))
fi
echo "Starting GPU 1 (items $START1-$((END1-1)))..."
CUDA_VISIBLE_DEVICES=1 python -u run/06_inference.py --model $MODEL --k $K --input "$INPUT" --output "$OUTPUT" --variants "$VARIANTS" --test-batches $TEST_BATCHES --start-index $START1 --end-index $END1 --gpu-id 1 > logs/inference/${MODEL}_gpu1_K${K}.log 2>&1 &
PID1=$!

# GPU 2
START2=$END1
END2=$((START2 + CHUNK_SIZE))
if [ $REMAINDER -gt 0 ]; then
    END2=$((END2 + 1))
    REMAINDER=$((REMAINDER - 1))
fi
echo "Starting GPU 2 (items $START2-$((END2-1)))..."
CUDA_VISIBLE_DEVICES=2 python -u run/06_inference.py --model $MODEL --k $K --input "$INPUT" --output "$OUTPUT" --variants "$VARIANTS" --test-batches $TEST_BATCHES --start-index $START2 --end-index $END2 --gpu-id 2 > logs/inference/${MODEL}_gpu2_K${K}.log 2>&1 &
PID2=$!

# GPU 3
START3=$END2
END3=$((START3 + CHUNK_SIZE))
if [ $REMAINDER -gt 0 ]; then
    END3=$((END3 + 1))
    REMAINDER=$((REMAINDER - 1))
fi
echo "Starting GPU 3 (items $START3-$((END3-1)))..."
CUDA_VISIBLE_DEVICES=3 python -u run/06_inference.py --model $MODEL --k $K --input "$INPUT" --output "$OUTPUT" --variants "$VARIANTS" --test-batches $TEST_BATCHES --start-index $START3 --end-index $END3 --gpu-id 3 > logs/inference/${MODEL}_gpu3_K${K}.log 2>&1 &
PID3=$!

# GPU 4
START4=$END3
END4=$((START4 + CHUNK_SIZE))
if [ $REMAINDER -gt 0 ]; then
    END4=$((END4 + 1))
    REMAINDER=$((REMAINDER - 1))
fi
echo "Starting GPU 4 (items $START4-$((END4-1)))..."
CUDA_VISIBLE_DEVICES=4 python -u run/06_inference.py --model $MODEL --k $K --input "$INPUT" --output "$OUTPUT" --variants "$VARIANTS" --test-batches $TEST_BATCHES --start-index $START4 --end-index $END4 --gpu-id 4 > logs/inference/${MODEL}_gpu4_K${K}.log 2>&1 &
PID4=$!

# GPU 5
START5=$END4
END5=$((START5 + CHUNK_SIZE))
if [ $REMAINDER -gt 0 ]; then
    END5=$((END5 + 1))
    REMAINDER=$((REMAINDER - 1))
fi
echo "Starting GPU 5 (items $START5-$((END5-1)))..."
CUDA_VISIBLE_DEVICES=5 python -u run/06_inference.py --model $MODEL --k $K --input "$INPUT" --output "$OUTPUT" --variants "$VARIANTS" --test-batches $TEST_BATCHES --start-index $START5 --end-index $END5 --gpu-id 5 > logs/inference/${MODEL}_gpu5_K${K}.log 2>&1 &
PID5=$!

# GPU 6
START6=$END5
END6=$((START6 + CHUNK_SIZE))
if [ $REMAINDER -gt 0 ]; then
    END6=$((END6 + 1))
    REMAINDER=$((REMAINDER - 1))
fi
echo "Starting GPU 6 (items $START6-$((END6-1)))..."
CUDA_VISIBLE_DEVICES=6 python -u run/06_inference.py --model $MODEL --k $K --input "$INPUT" --output "$OUTPUT" --variants "$VARIANTS" --test-batches $TEST_BATCHES --start-index $START6 --end-index $END6 --gpu-id 6 > logs/inference/${MODEL}_gpu6_K${K}.log 2>&1 &
PID6=$!

# GPU 7
START7=$END6
END7=$TOTAL_ITEMS
echo "Starting GPU 7 (items $START7-$((END7-1)))..."
CUDA_VISIBLE_DEVICES=7 python -u run/06_inference.py --model $MODEL --k $K --input "$INPUT" --output "$OUTPUT" --variants "$VARIANTS" --test-batches $TEST_BATCHES --start-index $START7 --end-index $END7 --gpu-id 7 > logs/inference/${MODEL}_gpu7_K${K}.log 2>&1 &
PID7=$!

echo ""
echo "All 8 processes started!"
echo "  GPU 0 (PID $PID0): items $START0-$((END0-1)) -> logs/inference/${MODEL}_gpu0_K${K}.log"
echo "  GPU 1 (PID $PID1): items $START1-$((END1-1)) -> logs/inference/${MODEL}_gpu1_K${K}.log"
echo "  GPU 2 (PID $PID2): items $START2-$((END2-1)) -> logs/inference/${MODEL}_gpu2_K${K}.log"
echo "  GPU 3 (PID $PID3): items $START3-$((END3-1)) -> logs/inference/${MODEL}_gpu3_K${K}.log"
echo "  GPU 4 (PID $PID4): items $START4-$((END4-1)) -> logs/inference/${MODEL}_gpu4_K${K}.log"
echo "  GPU 5 (PID $PID5): items $START5-$((END5-1)) -> logs/inference/${MODEL}_gpu5_K${K}.log"
echo "  GPU 6 (PID $PID6): items $START6-$((END6-1)) -> logs/inference/${MODEL}_gpu6_K${K}.log"
echo "  GPU 7 (PID $PID7): items $START7-$((END7-1)) -> logs/inference/${MODEL}_gpu7_K${K}.log"
echo ""
echo "Monitor progress:"
echo "  tail -f logs/inference/${MODEL}_gpu0_K${K}.log"
echo "  tail -f logs/inference/${MODEL}_gpu1_K${K}.log"
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
echo "✅ ALL INFERENCE PROCESSES COMPLETE!"
echo "=============================================="
echo ""

# Merge results from all GPUs
echo "Merging results from all GPUs..."
OUTPUT_DIR_ABS=$(cd "$(dirname "$OUTPUT")" && pwd)
OUTPUT_BASE=$(basename "$OUTPUT")
OUTPUT_BASE_NO_EXT="${OUTPUT_BASE%.*}"
OUTPUT_EXT="${OUTPUT_BASE##*.}"

# Merge JSON files using Python
python3 << EOF
import json
import sys
from pathlib import Path

output_dir = Path("$OUTPUT_DIR_ABS")
output_base = "$OUTPUT_BASE_NO_EXT"
output_ext = "$OUTPUT_EXT"

# Load all GPU results
all_results = []
all_rouge_scores = {
    'variant1_plain_seq': {'rouge1': [], 'rouge2': [], 'rougeL': []},
    'variant2_plain_seq_struct': {'rouge1': [], 'rouge2': [], 'rougeL': []},
    'variant3_plain_embeddings': {'rouge1': [], 'rouge2': [], 'rougeL': []},
    'variant4_finetuned_struct': {'rouge1': [], 'rouge2': [], 'rougeL': []},
    'variant5_finetuned_embeddings': {'rouge1': [], 'rouge2': [], 'rougeL': []}
}

# Load results from each GPU
for gpu_id in range(8):
    gpu_file = output_dir / f"{output_base}_gpu{gpu_id}.{output_ext}"
    if gpu_file.exists():
        with open(gpu_file, 'r') as f:
            gpu_data = json.load(f)
            all_results.extend(gpu_data.get('results', []))
            # Collect ROUGE scores for recomputation
            for result in gpu_data.get('results', []):
                for variant_num in range(1, 6):
                    variant_key = f'variant{variant_num}'
                    rouge_key = f'rouge_variant{variant_num}'
                    if rouge_key in result and result[rouge_key] is not None:
                        if variant_num == 1:
                            all_rouge_scores['variant1_plain_seq']['rouge1'].append(result[rouge_key]['rouge1'])
                            all_rouge_scores['variant1_plain_seq']['rouge2'].append(result[rouge_key]['rouge2'])
                            all_rouge_scores['variant1_plain_seq']['rougeL'].append(result[rouge_key]['rougeL'])
                        elif variant_num == 2:
                            all_rouge_scores['variant2_plain_seq_struct']['rouge1'].append(result[rouge_key]['rouge1'])
                            all_rouge_scores['variant2_plain_seq_struct']['rouge2'].append(result[rouge_key]['rouge2'])
                            all_rouge_scores['variant2_plain_seq_struct']['rougeL'].append(result[rouge_key]['rougeL'])
                        elif variant_num == 3:
                            all_rouge_scores['variant3_plain_embeddings']['rouge1'].append(result[rouge_key]['rouge1'])
                            all_rouge_scores['variant3_plain_embeddings']['rouge2'].append(result[rouge_key]['rouge2'])
                            all_rouge_scores['variant3_plain_embeddings']['rougeL'].append(result[rouge_key]['rougeL'])
                        elif variant_num == 4:
                            all_rouge_scores['variant4_finetuned_struct']['rouge1'].append(result[rouge_key]['rouge1'])
                            all_rouge_scores['variant4_finetuned_struct']['rouge2'].append(result[rouge_key]['rouge2'])
                            all_rouge_scores['variant4_finetuned_struct']['rougeL'].append(result[rouge_key]['rougeL'])
                        elif variant_num == 5:
                            all_rouge_scores['variant5_finetuned_embeddings']['rouge1'].append(result[rouge_key]['rouge1'])
                            all_rouge_scores['variant5_finetuned_embeddings']['rouge2'].append(result[rouge_key]['rouge2'])
                            all_rouge_scores['variant5_finetuned_embeddings']['rougeL'].append(result[rouge_key]['rougeL'])
        print(f"✅ Loaded {len(gpu_data.get('results', []))} results from GPU {gpu_id}")
    else:
        print(f"⚠️  GPU {gpu_id} output file not found: {gpu_file}")

# Compute merged evaluation summary
import numpy as np

def compute_avg(variant_scores):
    return {
        'avg_rouge1': float(np.mean(variant_scores['rouge1'])) if variant_scores['rouge1'] else 0,
        'avg_rouge2': float(np.mean(variant_scores['rouge2'])) if variant_scores['rouge2'] else 0,
        'avg_rougeL': float(np.mean(variant_scores['rougeL'])) if variant_scores['rougeL'] else 0,
    }

evaluation_metrics = {
    'variant1_plain_seq': {
        'n_samples': len(all_rouge_scores['variant1_plain_seq']['rouge1']),
        **compute_avg(all_rouge_scores['variant1_plain_seq'])
    },
    'variant2_plain_seq_struct': {
        'n_samples': len(all_rouge_scores['variant2_plain_seq_struct']['rouge1']),
        **compute_avg(all_rouge_scores['variant2_plain_seq_struct'])
    },
    'variant3_plain_embeddings': {
        'n_samples': len(all_rouge_scores['variant3_plain_embeddings']['rouge1']),
        **compute_avg(all_rouge_scores['variant3_plain_embeddings'])
    },
    'variant4_finetuned_struct': {
        'n_samples': len(all_rouge_scores['variant4_finetuned_struct']['rouge1']),
        **compute_avg(all_rouge_scores['variant4_finetuned_struct'])
    },
    'variant5_finetuned_embeddings': {
        'n_samples': len(all_rouge_scores['variant5_finetuned_embeddings']['rouge1']),
        **compute_avg(all_rouge_scores['variant5_finetuned_embeddings'])
    }
}

# Save merged results
merged_output = output_dir / output_base
merged_data = {
    'results': all_results,
    'evaluation_summary': evaluation_metrics
}

with open(merged_output, 'w') as f:
    json.dump(merged_data, f, indent=2)

print(f"\n✅ Merged {len(all_results)} total results")
print(f"✅ Saved merged results to: {merged_output}")

# Print summary
print(f"\n📊 Merged Evaluation Summary:")
print(f"   Total samples: {len(all_results)}")
for variant_key, metrics in evaluation_metrics.items():
    print(f"   {variant_key}: {metrics['n_samples']} samples")
    print(f"      ROUGE-1: {metrics['avg_rouge1']:.4f}")
    print(f"      ROUGE-2: {metrics['avg_rouge2']:.4f}")
    print(f"      ROUGE-L: {metrics['avg_rougeL']:.4f}")
EOF

echo ""
echo "✅ Merging complete!"
echo "Final merged output: $OUTPUT"

# Clean up temporary GPU-specific output files
echo ""
echo "🧹 Cleaning up temporary GPU output files..."
OUTPUT_DIR_ABS=$(cd "$(dirname "$OUTPUT")" && pwd)
OUTPUT_BASE_CLEANUP=$(basename "$OUTPUT")
OUTPUT_BASE_NO_EXT_CLEANUP="${OUTPUT_BASE_CLEANUP%.*}"
OUTPUT_EXT_CLEANUP="${OUTPUT_BASE_CLEANUP##*.}"

for gpu_id in {0..7}; do
    gpu_file="${OUTPUT_DIR_ABS}/${OUTPUT_BASE_NO_EXT_CLEANUP}_gpu${gpu_id}.${OUTPUT_EXT_CLEANUP}"
    if [ -f "$gpu_file" ]; then
        rm "$gpu_file"
        echo "  ✅ Deleted: ${OUTPUT_BASE_NO_EXT_CLEANUP}_gpu${gpu_id}.${OUTPUT_EXT_CLEANUP}"
    else
        echo "  ⚠️  File not found (may have been deleted already): ${OUTPUT_BASE_NO_EXT_CLEANUP}_gpu${gpu_id}.${OUTPUT_EXT_CLEANUP}"
    fi
done

echo "✅ Cleanup complete!"

