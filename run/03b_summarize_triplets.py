#!/usr/bin/env python3
"""
Summarize Triplet Embedding Metadata
Analyzes metadata from all triplet batch files to provide statistics
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse


def load_batch_metadata(metadata_dir, k_clusters):
    """Load all batch metadata files for a specific K"""
    batch_files = sorted(metadata_dir.glob(f'triplet_metadata_K{k_clusters}_batch_*.json'))
    
    if not batch_files:
        print(f"❌ No metadata files found in {metadata_dir} for K={k_clusters}")
        return None
    
    batches = {}
    for batch_file in batch_files:
        # Extract batch number from filename
        batch_num = int(batch_file.stem.split('_')[-1])
        with open(batch_file, 'r') as f:
            batches[batch_num] = json.load(f)
    
    return batches


def main():
    print("=" * 80)
    print("TRIPLET EMBEDDING METADATA SUMMARY")
    print("=" * 80)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Summarize triplet embedding metadata')
    parser.add_argument('--k', type=int, required=True,
                       help='K-means cluster size (e.g., 128, 512)')
    args = parser.parse_args()
    
    k_clusters = args.k
    
    # Configuration
    DATA_DIR = Path(__file__).parent.parent / 'data'
    TRIPLET_DIR = DATA_DIR / 'triplet_embeddings'
    
    if not TRIPLET_DIR.exists():
        print(f"❌ Triplet embeddings directory not found: {TRIPLET_DIR}")
        return
    
    print(f"\n📂 Metadata directory: {TRIPLET_DIR}")
    print(f"📌 K-means clusters: {k_clusters}")
    
    # Load all batch metadata
    print("\n" + "=" * 80)
    print("LOADING BATCH METADATA")
    print("=" * 80)
    
    batches = load_batch_metadata(TRIPLET_DIR, k_clusters)
    if batches is None:
        return
    
    print(f"✅ Loaded {len(batches)} batch metadata files")
    
    # Analyze each batch
    print("\n" + "=" * 80)
    print("PER-BATCH STATISTICS")
    print("=" * 80)
    
    batch_stats = {}
    for batch_num in sorted(batches.keys()):
        meta = batches[batch_num]
        batch_stats[batch_num] = {
            'num_triplets': meta['num_triplets'],
            'num_skipped': meta['num_skipped'],
            'seq_embedding_dim': meta['seq_embedding_dim'],
            'text_embedding_dim': meta['text_embedding_dim'],
            'structure_tokens_length': meta['structure_tokens_length']
        }
        
        print(f"\nBatch {batch_num}:")
        print(f"  Triplets:         {meta['num_triplets']:,}")
        print(f"  Skipped:          {meta['num_skipped']:,}")
        print(f"  Seq embed dim:    {meta['seq_embedding_dim']}")
        print(f"  Text embed dim:   {meta['text_embedding_dim']}")
        print(f"  Struct tok len:   {meta['structure_tokens_length']}")
    
    # Across-batch statistics
    print("\n" + "=" * 80)
    print("ACROSS-BATCH STATISTICS")
    print("=" * 80)
    
    all_triplets_per_batch = [stats['num_triplets'] for stats in batch_stats.values()]
    all_skipped_per_batch = [stats['num_skipped'] for stats in batch_stats.values()]
    
    print(f"\nTriplets per batch:")
    print(f"  Min:     {min(all_triplets_per_batch):,}")
    print(f"  Max:     {max(all_triplets_per_batch):,}")
    print(f"  Mean:    {np.mean(all_triplets_per_batch):.1f}")
    print(f"  Median:  {np.median(all_triplets_per_batch):.1f}")
    print(f"  Std:     {np.std(all_triplets_per_batch):.1f}")
    
    print(f"\nSkipped per batch:")
    print(f"  Min:     {min(all_skipped_per_batch):,}")
    print(f"  Max:     {max(all_skipped_per_batch):,}")
    print(f"  Mean:    {np.mean(all_skipped_per_batch):.1f}")
    print(f"  Median:  {np.median(all_skipped_per_batch):.1f}")
    print(f"  Std:     {np.std(all_skipped_per_batch):.1f}")
    
    # Total statistics
    print("\n" + "=" * 80)
    print("TOTAL STATISTICS")
    print("=" * 80)
    
    total_triplets = sum(all_triplets_per_batch)
    total_skipped = sum(all_skipped_per_batch)
    total_processed = total_triplets + total_skipped
    
    print(f"\nTotal processed:     {total_processed:,}")
    print(f"Total triplets:      {total_triplets:,}")
    print(f"Total skipped:       {total_skipped:,}")
    print(f"Success rate:        {total_triplets/total_processed*100:.2f}%")
    
    # Get dimensions from first batch
    first_batch = batch_stats[min(batch_stats.keys())]
    print(f"\nEmbedding dimensions:")
    print(f"  Sequence:          {first_batch['seq_embedding_dim']}D")
    print(f"  Text:              {first_batch['text_embedding_dim']}D")
    print(f"  Structure tokens:  {first_batch['structure_tokens_length']} tokens")
    print(f"  K-means clusters:  {k_clusters}")
    
    # Save summary to file
    print("\n" + "=" * 80)
    print("SAVING SUMMARY")
    print("=" * 80)
    
    summary = {
        'k_clusters': k_clusters,
        'num_batches': len(batches),
        'total_triplets': int(total_triplets),
        'total_skipped': int(total_skipped),
        'total_processed': int(total_processed),
        'success_rate': float(total_triplets / total_processed * 100),
        'embedding_dims': {
            'sequence': first_batch['seq_embedding_dim'],
            'text': first_batch['text_embedding_dim'],
            'structure_tokens_length': first_batch['structure_tokens_length']
        },
        'per_batch_stats': {
            int(batch_num): {
                'num_triplets': stats['num_triplets'],
                'num_skipped': stats['num_skipped']
            }
            for batch_num, stats in batch_stats.items()
        },
        'across_batch_stats': {
            'triplets_per_batch': {
                'min': int(min(all_triplets_per_batch)),
                'max': int(max(all_triplets_per_batch)),
                'mean': float(np.mean(all_triplets_per_batch)),
                'median': float(np.median(all_triplets_per_batch)),
                'std': float(np.std(all_triplets_per_batch))
            },
            'skipped_per_batch': {
                'min': int(min(all_skipped_per_batch)),
                'max': int(max(all_skipped_per_batch)),
                'mean': float(np.mean(all_skipped_per_batch)),
                'median': float(np.median(all_skipped_per_batch)),
                'std': float(np.std(all_skipped_per_batch))
            }
        }
    }
    
    output_file = TRIPLET_DIR / f'triplet_summary_K{k_clusters}.json'
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Summary saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("✅ SUMMARY COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
