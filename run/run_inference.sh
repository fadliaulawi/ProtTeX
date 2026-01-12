#!/bin/bash

# Parallel Inference Script
# Distributes inference workload across 8 GPUs
# Requires --model, --input, --output, --k, and --variants arguments

# Parse arguments
MODEL=""
INPUT=""
OUTPUT=""
K=""
VARIANTS=""

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
        *)
            echo "Unknown option: $1"
            echo ""
            echo "Usage: $0 --model <model_type> --input <input_file> --output <output_file> --k <K> --variants <variants>"
            echo ""
            echo "Available models: llama, qwen, deepseek"
            echo ""
            echo "Examples:"
            echo "  $0 --model deepseek --input data/evaluation/deepseek/K128/test_data.json --output run/output_inference.json --k 128 --variants all"
            echo "  $0 --model qwen --input data/evaluation/qwen/K1024/test_data.json --output run/output_inference.json --k 1024 --variants 1,5"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL" ]; then
    echo "Error: --model argument is required"
    echo ""
    echo "Usage: $0 --model <model_type> --input <input_file> --output <output_file> --k <K> --variants <variants>"
    exit 1
fi

if [ -z "$INPUT" ]; then
    echo "Error: --input argument is required"
    echo ""
    echo "Usage: $0 --model <model_type> --input <input_file> --output <output_file> --k <K> --variants <variants>"
    exit 1
fi

if [ -z "$OUTPUT" ]; then
    echo "Error: --output argument is required"
    echo ""
    echo "Usage: $0 --model <model_type> --input <input_file> --output <output_file> --k <K> --variants <variants>"
    exit 1
fi

if [ -z "$K" ]; then
    echo "Error: --k argument is required"
    echo ""
    echo "Usage: $0 --model <model_type> --input <input_file> --output <output_file> --k <K> --variants <variants>"
    exit 1
fi

if [ -z "$VARIANTS" ]; then
    echo "Error: --variants argument is required"
    echo ""
    echo "Usage: $0 --model <model_type> --input <input_file> --output <output_file> --k <K> --variants <variants>"
    exit 1
fi

# Ensure logs directory exists (model-specific with K subdirectory)
LOG_DIR="logs/inference/${MODEL}/K${K}"
mkdir -p "$LOG_DIR"

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
CUDA_VISIBLE_DEVICES=0 python -u run/06_inference.py --model $MODEL --k $K --input "$INPUT" --output "$OUTPUT" --variants "$VARIANTS" --start-index $START0 --end-index $END0 --gpu-id 0 > ${LOG_DIR}/gpu0.log 2>&1 &
PID0=$!

# GPU 1
START1=$END0
END1=$((START1 + CHUNK_SIZE))
if [ $REMAINDER -gt 0 ]; then
    END1=$((END1 + 1))
    REMAINDER=$((REMAINDER - 1))
fi
echo "Starting GPU 1 (items $START1-$((END1-1)))..."
CUDA_VISIBLE_DEVICES=1 python -u run/06_inference.py --model $MODEL --k $K --input "$INPUT" --output "$OUTPUT" --variants "$VARIANTS" --start-index $START1 --end-index $END1 --gpu-id 1 > ${LOG_DIR}/gpu1.log 2>&1 &
PID1=$!

# GPU 2
START2=$END1
END2=$((START2 + CHUNK_SIZE))
if [ $REMAINDER -gt 0 ]; then
    END2=$((END2 + 1))
    REMAINDER=$((REMAINDER - 1))
fi
echo "Starting GPU 2 (items $START2-$((END2-1)))..."
CUDA_VISIBLE_DEVICES=2 python -u run/06_inference.py --model $MODEL --k $K --input "$INPUT" --output "$OUTPUT" --variants "$VARIANTS" --start-index $START2 --end-index $END2 --gpu-id 2 > ${LOG_DIR}/gpu2.log 2>&1 &
PID2=$!

# GPU 3
START3=$END2
END3=$((START3 + CHUNK_SIZE))
if [ $REMAINDER -gt 0 ]; then
    END3=$((END3 + 1))
    REMAINDER=$((REMAINDER - 1))
fi
echo "Starting GPU 3 (items $START3-$((END3-1)))..."
CUDA_VISIBLE_DEVICES=3 python -u run/06_inference.py --model $MODEL --k $K --input "$INPUT" --output "$OUTPUT" --variants "$VARIANTS" --start-index $START3 --end-index $END3 --gpu-id 3 > ${LOG_DIR}/gpu3.log 2>&1 &
PID3=$!

# GPU 4
START4=$END3
END4=$((START4 + CHUNK_SIZE))
if [ $REMAINDER -gt 0 ]; then
    END4=$((END4 + 1))
    REMAINDER=$((REMAINDER - 1))
fi
echo "Starting GPU 4 (items $START4-$((END4-1)))..."
CUDA_VISIBLE_DEVICES=4 python -u run/06_inference.py --model $MODEL --k $K --input "$INPUT" --output "$OUTPUT" --variants "$VARIANTS" --start-index $START4 --end-index $END4 --gpu-id 4 > ${LOG_DIR}/gpu4.log 2>&1 &
PID4=$!

# GPU 5
START5=$END4
END5=$((START5 + CHUNK_SIZE))
if [ $REMAINDER -gt 0 ]; then
    END5=$((END5 + 1))
    REMAINDER=$((REMAINDER - 1))
fi
echo "Starting GPU 5 (items $START5-$((END5-1)))..."
CUDA_VISIBLE_DEVICES=5 python -u run/06_inference.py --model $MODEL --k $K --input "$INPUT" --output "$OUTPUT" --variants "$VARIANTS" --start-index $START5 --end-index $END5 --gpu-id 5 > ${LOG_DIR}/gpu5.log 2>&1 &
PID5=$!

# GPU 6
START6=$END5
END6=$((START6 + CHUNK_SIZE))
if [ $REMAINDER -gt 0 ]; then
    END6=$((END6 + 1))
    REMAINDER=$((REMAINDER - 1))
fi
echo "Starting GPU 6 (items $START6-$((END6-1)))..."
CUDA_VISIBLE_DEVICES=6 python -u run/06_inference.py --model $MODEL --k $K --input "$INPUT" --output "$OUTPUT" --variants "$VARIANTS" --start-index $START6 --end-index $END6 --gpu-id 6 > ${LOG_DIR}/gpu6.log 2>&1 &
PID6=$!

# GPU 7
START7=$END6
END7=$TOTAL_ITEMS
echo "Starting GPU 7 (items $START7-$((END7-1)))..."
CUDA_VISIBLE_DEVICES=7 python -u run/06_inference.py --model $MODEL --k $K --input "$INPUT" --output "$OUTPUT" --variants "$VARIANTS" --start-index $START7 --end-index $END7 --gpu-id 7 > ${LOG_DIR}/gpu7.log 2>&1 &
PID7=$!

echo ""
echo "All 8 processes started!"
echo "  GPU 0 (PID $PID0): items $START0-$((END0-1)) -> ${LOG_DIR}/gpu0.log"
echo "  GPU 1 (PID $PID1): items $START1-$((END1-1)) -> ${LOG_DIR}/gpu1.log"
echo "  GPU 2 (PID $PID2): items $START2-$((END2-1)) -> ${LOG_DIR}/gpu2.log"
echo "  GPU 3 (PID $PID3): items $START3-$((END3-1)) -> ${LOG_DIR}/gpu3.log"
echo "  GPU 4 (PID $PID4): items $START4-$((END4-1)) -> ${LOG_DIR}/gpu4.log"
echo "  GPU 5 (PID $PID5): items $START5-$((END5-1)) -> ${LOG_DIR}/gpu5.log"
echo "  GPU 6 (PID $PID6): items $START6-$((END6-1)) -> ${LOG_DIR}/gpu6.log"
echo "  GPU 7 (PID $PID7): items $START7-$((END7-1)) -> ${LOG_DIR}/gpu7.log"
echo ""
echo "Monitor progress:"
echo "  tail -f ${LOG_DIR}/gpu*.log"
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
all_metric_scores = {
    'plain_seq': {'rouge1': [], 'rouge2': [], 'rougeL': [], 'bleu': [], 'emji': []},
    'plain_seq_struct': {'rouge1': [], 'rouge2': [], 'rougeL': [], 'bleu': [], 'emji': []},
    'plain_embeddings': {'rouge1': [], 'rouge2': [], 'rougeL': [], 'bleu': [], 'emji': []},
    'finetuned_struct': {'rouge1': [], 'rouge2': [], 'rougeL': [], 'bleu': [], 'emji': []},
    'full_model': {'rouge1': [], 'rouge2': [], 'rougeL': [], 'bleu': [], 'emji': []},
    'molinst_protein': {'rouge1': [], 'rouge2': [], 'rougeL': [], 'bleu': [], 'emji': []},
    'protex': {'rouge1': [], 'rouge2': [], 'rougeL': [], 'bleu': [], 'emji': []}
}

# Map result keys to variant keys (support both old rouge_* and new metrics_* naming)
variant_mapping = {
    'metrics_plain_seq': 'plain_seq',
    'metrics_plain_seq_struct': 'plain_seq_struct',
    'metrics_plain_embeddings': 'plain_embeddings',
    'metrics_finetuned_struct': 'finetuned_struct',
    'metrics_full_model': 'full_model',
    'metrics_molinst_protein': 'molinst_protein',
    'metrics_protex': 'protex',
    # Backward compatibility with old naming
    'rouge_plain_seq': 'plain_seq',
    'rouge_plain_seq_struct': 'plain_seq_struct',
    'rouge_plain_embeddings': 'plain_embeddings',
    'rouge_finetuned_struct': 'finetuned_struct',
    'rouge_full_model': 'full_model',
    'rouge_molinst_protein': 'molinst_protein',
    'rouge_protex': 'protex'
}

# Load results from each GPU
for gpu_id in range(8):
    gpu_file = output_dir / f"{output_base}_gpu{gpu_id}.{output_ext}"
    if gpu_file.exists():
        with open(gpu_file, 'r') as f:
            gpu_data = json.load(f)
            all_results.extend(gpu_data.get('results', []))
            # Collect metric scores for recomputation
            for result in gpu_data.get('results', []):
                for result_key, variant_key in variant_mapping.items():
                    if result_key in result and result[result_key]:
                        scores = result[result_key]
                        if 'rouge1' in scores:
                            all_metric_scores[variant_key]['rouge1'].append(scores['rouge1'])
                        if 'rouge2' in scores:
                            all_metric_scores[variant_key]['rouge2'].append(scores['rouge2'])
                        if 'rougeL' in scores:
                            all_metric_scores[variant_key]['rougeL'].append(scores['rougeL'])
                        if 'bleu' in scores:
                            all_metric_scores[variant_key]['bleu'].append(scores['bleu'])
                        if 'emji' in scores:
                            all_metric_scores[variant_key]['emji'].append(scores['emji'])
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
        'avg_bleu': float(np.mean(variant_scores['bleu'])) if variant_scores['bleu'] else 0,
        'avg_emji': float(np.mean(variant_scores['emji'])) if variant_scores['emji'] else 0
    }

evaluation_metrics = {
    'plain_seq': {
        'n_samples': len(all_metric_scores['plain_seq']['rouge1']),
        **compute_avg(all_metric_scores['plain_seq'])
    },
    'plain_seq_struct': {
        'n_samples': len(all_metric_scores['plain_seq_struct']['rouge1']),
        **compute_avg(all_metric_scores['plain_seq_struct'])
    },
    'plain_embeddings': {
        'n_samples': len(all_metric_scores['plain_embeddings']['rouge1']),
        **compute_avg(all_metric_scores['plain_embeddings'])
    },
    'finetuned_struct': {
        'n_samples': len(all_metric_scores['finetuned_struct']['rouge1']),
        **compute_avg(all_metric_scores['finetuned_struct'])
    },
    'full_model': {
        'n_samples': len(all_metric_scores['full_model']['rouge1']),
        **compute_avg(all_metric_scores['full_model'])
    },
    'molinst_protein': {
        'n_samples': len(all_metric_scores['molinst_protein']['rouge1']),
        **compute_avg(all_metric_scores['molinst_protein'])
    },
    'protex': {
        'n_samples': len(all_metric_scores['protex']['rouge1']),
        **compute_avg(all_metric_scores['protex'])
    }
}

# Save merged results
merged_output = output_dir / f"{output_base}.{output_ext}"
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
    print(f"      BLEU:    {metrics['avg_bleu']:.4f}")
    print(f"      EMJI:    {metrics['avg_emji']:.4f}")
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

