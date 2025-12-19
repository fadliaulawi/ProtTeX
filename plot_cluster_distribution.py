#!/usr/bin/env python3
"""
Simple script to display sorted bar charts of cluster distribution from codebook summary.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def load_codebook_data(filepath):
    """Load codebook summary JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_cluster_distribution(data, title="Cluster Distribution (K-means Codebook)", figsize=(14, 6)):
    """Plot sorted bar chart of cluster counts."""
    clusters = data['all_clusters']
    
    # Extract cluster IDs and counts
    cluster_ids = [c['cluster_id'] for c in clusters]
    counts = [c['count'] for c in clusters]
    
    # Sort by count (descending)
    sorted_data = sorted(zip(cluster_ids, counts), key=lambda x: x[1], reverse=True)
    cluster_ids_sorted, counts_sorted = zip(*sorted_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar chart
    bars = ax.bar(range(len(cluster_ids_sorted)), counts_sorted, color='steelblue', edgecolor='navy', alpha=0.7)
    
    # Customize plot
    ax.set_xlabel('Cluster Rank (sorted by count)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Format y-axis with thousands separator
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Add cluster IDs as x-tick labels (every 25th to avoid crowding)
    tick_positions = range(0, len(cluster_ids_sorted), max(1, len(cluster_ids_sorted) // 20))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"C{cluster_ids_sorted[i]}" for i in tick_positions], rotation=45, ha='right')
    
    # Save to file
    fig.savefig('cluster_distribution.png', dpi=300, bbox_inches='tight')

    plt.tight_layout()
    return fig

def main():
    # Path to codebook data
    codebook_path = '/home/fadli.ghiffari/research/ProtTeX/esmfold_tokenizer/data/codebook_summary_K512.json'
    
    # Load data
    print("Loading codebook data...")
    data = load_codebook_data(codebook_path)
    
    # Print summary
    print(f"\nCodebook Summary:")
    print(f"  Total clusters: {data['n_clusters']}")
    print(f"  Total residues: {data['n_residues']:,}")
    print(f"  Utilization: {data['utilization']:.1%}")
    
    # Plot
    print("Creating plot...")
    fig = plot_cluster_distribution(data)
    
    # Show
    plt.show()

if __name__ == '__main__':
    main()
