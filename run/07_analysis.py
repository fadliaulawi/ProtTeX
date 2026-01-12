#!/usr/bin/env python3
"""
Analysis script for evaluation results by protein type.

This script computes average metrics for PFUD and PSAD types across all variants.

Usage:
    python run/07_analysis.py --input data/evaluation/llama/K1024/output_test_data.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


def load_results(input_path: Path) -> List[Dict]:
    """Load results from JSON file."""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'results' in data:
        return data['results']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Invalid JSON format")


def compute_type_averages(results: List[Dict], target_type: str) -> Dict:
    """
    Compute average metrics for a specific protein type.
    
    Args:
        results: List of result dictionaries
        target_type: Protein type to filter by (e.g., 'PFUD', 'PSAD')
    
    Returns:
        Dictionary with average metrics for each variant
    """
    # Filter results by type
    type_results = [r for r in results if r.get('type') == target_type]
    
    if not type_results:
        print(f"⚠️  No results found for type: {target_type}")
        return {}
    
    # Variant names mapping
    variants = {
        'plain_seq': 'Variant 1: Plain (seq)',
        'plain_seq_struct': 'Variant 2: Plain (seq+struct)',
        'plain_embeddings': 'Variant 3: Plain (embeddings)',
        'finetuned_struct': 'Variant 4: Fine-tuned (struct)',
        'full_model': 'Variant 5: Full Model',
        'molinst_protein': 'Variant 6: MolInst-Protein',
        'protex': 'Variant 7: ProtTeX'
    }
    
    # Collect metrics for each variant
    variant_metrics = {}
    
    for variant_key, variant_name in variants.items():
        metrics_key = f'metrics_{variant_key}'
        
        # Collect all metric scores for this variant
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        bleu_scores = []
        emji_scores = []
        
        for result in type_results:
            if metrics_key in result and result[metrics_key]:
                metrics = result[metrics_key]
                if 'rouge1' in metrics:
                    rouge1_scores.append(metrics['rouge1'])
                if 'rouge2' in metrics:
                    rouge2_scores.append(metrics['rouge2'])
                if 'rougeL' in metrics:
                    rougeL_scores.append(metrics['rougeL'])
                if 'bleu' in metrics:
                    bleu_scores.append(metrics['bleu'])
                if 'emji' in metrics:
                    emji_scores.append(metrics['emji'])
        
        # Compute averages
        variant_metrics[variant_key] = {
            'name': variant_name,
            'n_samples': len(rouge1_scores),
            'avg_rouge1': sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
            'avg_rouge2': sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0,
            'avg_rougeL': sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0,
            'avg_bleu': sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
            'avg_emji': sum(emji_scores) / len(emji_scores) if emji_scores else 0.0
        }
    
    return variant_metrics


def print_type_summary(type_name: str, variant_metrics: Dict):
    """Print formatted summary for a protein type."""
    print(f"\n{'='*100}")
    print(f"📊 TYPE: {type_name}")
    print(f"{'='*100}")
    
    if not variant_metrics:
        print("No data available")
        return
    
    # Print header
    print(f"\n{'Variant':<30} {'Samples':<10} {'ROUGE-1':<12} {'ROUGE-2':<12} {'ROUGE-L':<12} {'BLEU':<12} {'EMJI':<12}")
    print(f"{'-'*100}")
    
    # Print each variant
    for variant_key in ['plain_seq', 'plain_seq_struct', 'plain_embeddings', 
                        'finetuned_struct', 'full_model', 'molinst_protein', 'protex']:
        if variant_key in variant_metrics:
            metrics = variant_metrics[variant_key]
            print(f"{metrics['name']:<30} "
                  f"{metrics['n_samples']:<10} "
                  f"{metrics['avg_rouge1']:<12.4f} "
                  f"{metrics['avg_rouge2']:<12.4f} "
                  f"{metrics['avg_rougeL']:<12.4f} "
                  f"{metrics['avg_bleu']:<12.4f} "
                  f"{metrics['avg_emji']:<12.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze evaluation results by protein type (PFUD, PSAD)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to inference output JSON file (e.g., output_test_data.json)'
    )
    parser.add_argument(
        '--types',
        type=str,
        default='PFUD,PSAD',
        help='Comma-separated list of protein types to analyze (default: PFUD,PSAD)'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return
    
    print(f"\n{'='*100}")
    print(f"🔬 PROTEIN TYPE ANALYSIS")
    print(f"{'='*100}")
    print(f"Input file: {input_path}")
    
    # Load results
    print(f"\n📥 Loading results...")
    results = load_results(input_path)
    print(f"✅ Loaded {len(results)} total results")
    
    # Count types
    type_counts = defaultdict(int)
    for r in results:
        if 'type' in r:
            type_counts[r['type']] += 1
    
    print(f"\n📋 Type distribution:")
    for type_name, count in sorted(type_counts.items()):
        print(f"   {type_name}: {count} samples")
    
    # Analyze each requested type
    types_to_analyze = [t.strip() for t in args.types.split(',')]
    
    for target_type in types_to_analyze:
        variant_metrics = compute_type_averages(results, target_type)
        print_type_summary(target_type, variant_metrics)
    
    print(f"\n{'='*100}")
    print("✅ Analysis complete!")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
