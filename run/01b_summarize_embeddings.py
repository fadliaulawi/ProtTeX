#!/usr/bin/env python3
"""
Summarize ESM Embedding Metadata
Analyzes metadata from all batch files to provide statistics on sequences and residues
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_batch_metadata(metadata_dir):
    """Load all batch metadata files"""
    batch_files = sorted(metadata_dir.glob('embedding_metadata_batch_*.json'))
    
    if not batch_files:
        print(f"❌ No metadata files found in {metadata_dir}")
        return None
    
    batches = {}
    for batch_file in batch_files:
        batch_num = int(batch_file.stem.split('_')[-1])
        with open(batch_file, 'r') as f:
            batches[batch_num] = json.load(f)
    
    return batches


def analyze_batch(batch_data):
    """Analyze a single batch"""
    lengths = [item['length'] for item in batch_data]
    
    return {
        'num_proteins': len(batch_data),
        'total_residues': sum(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'mean_length': np.mean(lengths),
        'median_length': np.median(lengths),
        'std_length': np.std(lengths),
    }


def main():
    print("=" * 80)
    print("ESM EMBEDDING METADATA SUMMARY")
    print("=" * 80)
    
    # Configuration
    DATA_DIR = Path(__file__).parent.parent / 'data'
    EMBEDDINGS_DIR = DATA_DIR / 'esm_embeddings'
    
    if not EMBEDDINGS_DIR.exists():
        print(f"❌ Embeddings directory not found: {EMBEDDINGS_DIR}")
        return
    
    print(f"\n📂 Metadata directory: {EMBEDDINGS_DIR}")
    
    # Load all batch metadata
    print("\n" + "=" * 80)
    print("LOADING BATCH METADATA")
    print("=" * 80)
    
    batches = load_batch_metadata(EMBEDDINGS_DIR)
    if batches is None:
        return
    
    print(f"✅ Loaded {len(batches)} batch metadata files")
    
    # Analyze each batch
    print("\n" + "=" * 80)
    print("PER-BATCH STATISTICS")
    print("=" * 80)
    
    batch_stats = {}
    for batch_num in sorted(batches.keys()):
        stats = analyze_batch(batches[batch_num])
        batch_stats[batch_num] = stats
        
        print(f"\nBatch {batch_num}:")
        print(f"  Proteins:        {stats['num_proteins']:,}")
        print(f"  Total residues:  {stats['total_residues']:,}")
        print(f"  Min length:      {stats['min_length']:,}")
        print(f"  Max length:      {stats['max_length']:,}")
        print(f"  Mean length:     {stats['mean_length']:.1f}")
        print(f"  Median length:   {stats['median_length']:.1f}")
        print(f"  Std length:      {stats['std_length']:.1f}")
    
    # Across-batch statistics
    print("\n" + "=" * 80)
    print("ACROSS-BATCH STATISTICS")
    print("=" * 80)
    
    all_proteins_per_batch = [stats['num_proteins'] for stats in batch_stats.values()]
    all_residues_per_batch = [stats['total_residues'] for stats in batch_stats.values()]
    all_mean_lengths = [stats['mean_length'] for stats in batch_stats.values()]
    
    print(f"\nProteins per batch:")
    print(f"  Min:     {min(all_proteins_per_batch):,}")
    print(f"  Max:     {max(all_proteins_per_batch):,}")
    print(f"  Mean:    {np.mean(all_proteins_per_batch):.1f}")
    print(f"  Median:  {np.median(all_proteins_per_batch):.1f}")
    print(f"  Std:     {np.std(all_proteins_per_batch):.1f}")
    
    print(f"\nResidues per batch:")
    print(f"  Min:     {min(all_residues_per_batch):,}")
    print(f"  Max:     {max(all_residues_per_batch):,}")
    print(f"  Mean:    {np.mean(all_residues_per_batch):,.0f}")
    print(f"  Median:  {np.median(all_residues_per_batch):,.0f}")
    print(f"  Std:     {np.std(all_residues_per_batch):,.0f}")
    
    print(f"\nMean sequence length per batch:")
    print(f"  Min:     {min(all_mean_lengths):.1f}")
    print(f"  Max:     {max(all_mean_lengths):.1f}")
    print(f"  Mean:    {np.mean(all_mean_lengths):.1f}")
    print(f"  Median:  {np.median(all_mean_lengths):.1f}")
    print(f"  Std:     {np.std(all_mean_lengths):.1f}")
    
    # Total statistics
    print("\n" + "=" * 80)
    print("TOTAL STATISTICS")
    print("=" * 80)
    
    # Collect all lengths across all batches
    all_lengths = []
    for batch_data in batches.values():
        all_lengths.extend([item['length'] for item in batch_data])
    
    total_proteins = len(all_lengths)
    total_residues = sum(all_lengths)
    
    # Count sequences exceeding ESMFold limit
    ESMFOLD_MAX_LENGTH = 1022
    truncated_sequences = sum(1 for length in all_lengths if length > ESMFOLD_MAX_LENGTH)
    truncated_percentage = (truncated_sequences / total_proteins) * 100 if total_proteins > 0 else 0
    
    print(f"\nTotal proteins:      {total_proteins:,}")
    print(f"Total residues:      {total_residues:,}")
    print(f"Min length:          {min(all_lengths):,}")
    print(f"Max length:          {max(all_lengths):,}")
    print(f"Mean length:         {np.mean(all_lengths):.1f}")
    print(f"Median length:       {np.median(all_lengths):.1f}")
    print(f"Std length:          {np.std(all_lengths):.1f}")
    print(f"\nTruncated sequences: {truncated_sequences:,} ({truncated_percentage:.2f}%)")
    print(f"  (Sequences exceeding ESMFold limit of {ESMFOLD_MAX_LENGTH} residues)")
    
    # Length distribution
    print("\n" + "=" * 80)
    print("LENGTH DISTRIBUTION")
    print("=" * 80)
    
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("\nPercentiles:")
    for p in percentiles:
        val = np.percentile(all_lengths, p)
        print(f"  {p:2d}th: {val:7.1f}")
    
    # Length bins
    bins = [0, 50, 100, 200, 300, 500, 1000, 1500, float('inf')]
    bin_labels = ['<50', '50-100', '100-200', '200-300', '300-500', '500-1000', '1000-1500', '>1500']
    bin_counts = defaultdict(int)
    
    for length in all_lengths:
        for i in range(len(bins) - 1):
            if bins[i] <= length < bins[i + 1]:
                bin_counts[bin_labels[i]] += 1
                break
    
    print("\nLength distribution:")
    for label in bin_labels:
        count = bin_counts[label]
        percentage = count / total_proteins * 100
        print(f"  {label:>10s}: {count:7,} ({percentage:5.2f}%)")
    
    # Save summary to file
    print("\n" + "=" * 80)
    print("SAVING SUMMARY")
    print("=" * 80)
    
    summary = {
        'num_batches': len(batches),
        'total_proteins': total_proteins,
        'total_residues': total_residues,
        'truncated_sequences': truncated_sequences,
        'truncated_percentage': float(truncated_percentage),
        'esmfold_max_length': ESMFOLD_MAX_LENGTH,
        'overall_stats': {
            'min_length': int(min(all_lengths)),
            'max_length': int(max(all_lengths)),
            'mean_length': float(np.mean(all_lengths)),
            'median_length': float(np.median(all_lengths)),
            'std_length': float(np.std(all_lengths)),
        },
        'per_batch_stats': {
            int(batch_num): {
                'num_proteins': stats['num_proteins'],
                'total_residues': stats['total_residues'],
                'min_length': stats['min_length'],
                'max_length': stats['max_length'],
                'mean_length': float(stats['mean_length']),
                'median_length': float(stats['median_length']),
                'std_length': float(stats['std_length']),
            }
            for batch_num, stats in batch_stats.items()
        },
        'length_distribution': {label: bin_counts[label] for label in bin_labels},
        'percentiles': {int(p): float(np.percentile(all_lengths, p)) for p in percentiles}
    }
    
    output_file = EMBEDDINGS_DIR / 'embedding_summary.json'
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Summary saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("✅ SUMMARY COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
