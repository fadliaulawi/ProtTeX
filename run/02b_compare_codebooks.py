#!/usr/bin/env python3
"""
Compare codebook summaries from different K values.

This script loads multiple codebook summary JSON files and provides
comparative analysis across different cluster sizes (K values).
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_codebook_summary(filepath):
    """Load a codebook summary JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_key_metrics(summary):
    """Extract key metrics from a codebook summary."""
    return {
        'n_clusters': summary['n_clusters'],
        'clusters_used': summary['clusters_used'],
        'utilization': summary['utilization'],
        'n_residues': summary['n_residues'],
        'embedding_dim': summary['embedding_dim'],
        'inertia': summary['inertia'],
        'timestamp': summary.get('timestamp', 'N/A'),
    }


def compute_distribution_stats(summary):
    """Compute distribution statistics from cluster data."""
    clusters = summary['all_clusters']
    counts = [c['count'] for c in clusters]
    
    # Compute perplexity (exp of entropy)
    probs = np.array(counts) / np.sum(counts)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    perplexity = np.exp(entropy)
    
    # Compute ratio (perplexity / n_clusters)
    n_clusters = summary['n_clusters']
    perplexity_ratio = perplexity / n_clusters
    
    return {
        'mean_count': np.mean(counts),
        'median_count': np.median(counts),
        'std_count': np.std(counts),
        'min_count': np.min(counts),
        'max_count': np.max(counts),
        'cv': np.std(counts) / np.mean(counts),  # Coefficient of variation
        'gini': compute_gini(counts),
        'entropy': entropy,
        'perplexity': perplexity,
        'perplexity_ratio': perplexity_ratio,
    }


def compute_gini(counts):
    """Compute Gini coefficient for cluster distribution."""
    sorted_counts = np.sort(counts)
    n = len(counts)
    cumsum = np.cumsum(sorted_counts)
    return (2 * np.sum((np.arange(1, n + 1)) * sorted_counts)) / (n * cumsum[-1]) - (n + 1) / n


def compare_codebooks(filepaths, data_dir='data'):
    """Compare multiple codebook summaries."""
    data_dir = Path(data_dir)
    
    # Load all summaries
    summaries = {}
    for filepath in filepaths:
        full_path = data_dir / filepath if not Path(filepath).is_absolute() else Path(filepath)
        if full_path.exists():
            summaries[filepath] = load_codebook_summary(full_path)
        else:
            print(f"Warning: File not found: {full_path}")
    
    if not summaries:
        print("No valid codebook summaries found!")
        return
    
    # Extract metrics
    print("\n" + "="*80)
    print("CODEBOOK COMPARISON SUMMARY")
    print("="*80)
    
    metrics_data = []
    for name, summary in summaries.items():
        metrics = extract_key_metrics(summary)
        dist_stats = compute_distribution_stats(summary)
        metrics.update(dist_stats)
        metrics['name'] = name
        metrics_data.append(metrics)
    
    df = pd.DataFrame(metrics_data)
    
    # Reorder columns
    cols = ['name', 'n_clusters', 'clusters_used', 'utilization', 'perplexity', 
            'perplexity_ratio', 'gini', 'cv', 'n_residues', 'inertia', 
            'mean_count', 'median_count', 'std_count', 'entropy',
            'min_count', 'max_count', 'timestamp']
    df = df[cols]
    
    # Display comparison table
    print("\nKey Metrics:")
    print("-" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}' if abs(x) < 1000 else f'{x:.2e}')
    print(df.to_string(index=False))
    
    # Per-cluster analysis
    print("\n" + "="*80)
    print("INERTIA AND UTILIZATION ANALYSIS")
    print("="*80)
    
    print("\nInertia (lower is better - tighter clusters):")
    for name, summary in summaries.items():
        k = summary['n_clusters']
        inertia = summary['inertia']
        inertia_per_residue = inertia / summary['n_residues']
        inertia_per_cluster = inertia / k
        print(f"  K={k:4d}: {inertia:15.2f} | Per-residue: {inertia_per_residue:.6f} | Per-cluster: {inertia_per_cluster:.2f}")
    
    print("\nUtilization (percentage of clusters actually used):")
    for name, summary in summaries.items():
        k = summary['n_clusters']
        util = summary['utilization'] * 100
        print(f"  K={k:4d}: {util:6.2f}%")
    
    # Distribution analysis
    print("\n" + "="*80)
    print("CLUSTER DISTRIBUTION ANALYSIS")
    print("="*80)
    
    print("\nCoefficient of Variation (CV - lower means more uniform):")
    for name, summary in summaries.items():
        k = summary['n_clusters']
        dist_stats = compute_distribution_stats(summary)
        cv = dist_stats['cv']
        print(f"  K={k:4d}: {cv:.4f}")
    
    print("\nGini Coefficient (0=perfect equality, 1=perfect inequality):")
    for name, summary in summaries.items():
        k = summary['n_clusters']
        dist_stats = compute_distribution_stats(summary)
        gini = dist_stats['gini']
        print(f"  K={k:4d}: {gini:.4f}")
    
    print("\nPerplexity (effective number of clusters used):")
    for name, summary in summaries.items():
        k = summary['n_clusters']
        dist_stats = compute_distribution_stats(summary)
        perp = dist_stats['perplexity']
        print(f"  K={k:4d}: {perp:.2f}")
    
    print("\nPerplexity Ratio (perplexity/K - closer to 1 is more uniform):")
    for name, summary in summaries.items():
        k = summary['n_clusters']
        dist_stats = compute_distribution_stats(summary)
        ratio = dist_stats['perplexity_ratio']
        print(f"  K={k:4d}: {ratio:.4f}")
    
    return df, summaries


def plot_comparisons(summaries, output_dir='data'):
    """Generate comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Extract K values and metrics
    k_values = []
    inertias = []
    utils = []
    cvs = []
    ginis = []
    
    for name, summary in sorted(summaries.items(), key=lambda x: x[1]['n_clusters']):
        k = summary['n_clusters']
        k_values.append(k)
        inertias.append(summary['inertia'])
        utils.append(summary['utilization'])
        dist_stats = compute_distribution_stats(summary)
        cvs.append(dist_stats['cv'])
        ginis.append(dist_stats['gini'])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Codebook Comparison Across K Values', fontsize=16, fontweight='bold')
    
    # Plot 1: Inertia vs K
    axes[0, 0].plot(k_values, inertias, marker='o', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Number of Clusters (K)', fontsize=11)
    axes[0, 0].set_ylabel('Inertia', fontsize=11)
    axes[0, 0].set_title('Inertia vs K (Lower is Better)', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale('log')
    
    # Plot 2: Utilization vs K
    axes[0, 1].plot(k_values, [u * 100 for u in utils], marker='s', 
                     linewidth=2, markersize=8, color='green')
    axes[0, 1].set_xlabel('Number of Clusters (K)', fontsize=11)
    axes[0, 1].set_ylabel('Utilization (%)', fontsize=11)
    axes[0, 1].set_title('Cluster Utilization vs K', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_ylim([0, 105])
    
    # Plot 3: Coefficient of Variation vs K
    axes[1, 0].plot(k_values, cvs, marker='^', linewidth=2, markersize=8, color='orange')
    axes[1, 0].set_xlabel('Number of Clusters (K)', fontsize=11)
    axes[1, 0].set_ylabel('Coefficient of Variation', fontsize=11)
    axes[1, 0].set_title('Distribution CV vs K (Lower is More Uniform)', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log')
    
    # Plot 4: Gini Coefficient vs K
    axes[1, 1].plot(k_values, ginis, marker='d', linewidth=2, markersize=8, color='red')
    axes[1, 1].set_xlabel('Number of Clusters (K)', fontsize=11)
    axes[1, 1].set_ylabel('Gini Coefficient', fontsize=11)
    axes[1, 1].set_title('Distribution Gini vs K (0=Equal, 1=Unequal)', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale('log')
    
    plt.tight_layout()
    output_file = output_dir / 'codebook_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {output_file}")
    plt.close()
    
    # Plot distribution comparison
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Cluster Size Distributions Across K Values', fontsize=16, fontweight='bold')
    
    for idx, (name, summary) in enumerate(sorted(summaries.items(), key=lambda x: x[1]['n_clusters'])):
        if idx >= 6:
            break
        
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        counts = [c['count'] for c in summary['all_clusters']]
        k = summary['n_clusters']
        
        ax.hist(counts, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Cluster Size (# residues)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'K={k}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        mean_val = np.mean(counts)
        median_val = np.median(counts)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.0f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.0f}')
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for idx in range(len(summaries), 6):
        row = idx // 3
        col = idx % 3
        axes[row, col].axis('off')
    
    plt.tight_layout()
    output_file = output_dir / 'codebook_distributions.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Distribution plot saved to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Compare codebook summaries from different K values',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all available codebook summaries
  python 02b_compare_codebooks.py --all
  
  # Compare specific K values
  python 02b_compare_codebooks.py -k 64 128 256 512 1024 2048
  
  # Compare with custom file paths
  python 02b_compare_codebooks.py -f data/codebook_summary_K64.json data/codebook_summary_K128.json
  
  # Compare all and generate plots
  python 02b_compare_codebooks.py --all --plot
        """
    )
    
    parser.add_argument(
        '--all', 
        action='store_true',
        help='Compare all codebook summaries found in data directory'
    )
    parser.add_argument(
        '-k', '--k-values',
        type=int,
        nargs='+',
        help='Specific K values to compare (e.g., -k 64 128 256)'
    )
    parser.add_argument(
        '-f', '--files',
        nargs='+',
        help='Specific codebook summary files to compare'
    )
    parser.add_argument(
        '--data-dir',
        default='data/codebooks',
        help='Directory containing codebook summary files (default: data/codebooks)'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate comparison plots'
    )
    parser.add_argument(
        '--output-dir',
        default='data/codebooks',
        help='Directory for output plots (default: data/codebooks)'
    )
    
    args = parser.parse_args()
    
    # Determine which files to compare
    files = []
    
    if args.files:
        files = args.files
    elif args.k_values:
        files = [f"codebook_summary_K{k}.json" for k in args.k_values]
    elif args.all:
        data_dir = Path(args.data_dir)
        files = sorted([f.name for f in data_dir.glob("codebook_summary_K*.json")])
    else:
        # Default: compare all available
        data_dir = Path(args.data_dir)
        files = sorted([f.name for f in data_dir.glob("codebook_summary_K*.json")])
    
    if not files:
        print("No codebook summary files found!")
        print(f"Try specifying files with -f, K values with -k, or use --all")
        return
    
    print(f"Comparing {len(files)} codebook summaries...")
    
    # Perform comparison
    df, summaries = compare_codebooks(files, args.data_dir)
    
    # Generate plots if requested
    if args.plot:
        print("\nGenerating comparison plots...")
        plot_comparisons(summaries, args.output_dir)
    
    # Save comparison table
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    csv_file = output_dir / 'codebook_comparison.csv'
    df.to_csv(csv_file, index=False)
    print(f"\nComparison table saved to: {csv_file}")
    
    # Save key metrics summary
    key_metrics_df = df[['name', 'n_clusters', 'utilization', 'perplexity_ratio', 'gini']].copy()
    key_metrics_df = key_metrics_df.sort_values('n_clusters')
    key_metrics_file = output_dir / 'codebook_key_metrics.txt'
    with open(key_metrics_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CODEBOOK KEY METRICS SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write("Columns:\n")
        f.write("  - n_clusters: Number of clusters (K)\n")
        f.write("  - utilization: Fraction of clusters actually used (0-1)\n")
        f.write("  - perplexity_ratio: Effective cluster usage (perplexity/K, closer to 1 is better)\n")
        f.write("  - gini: Distribution inequality (0=equal, 1=unequal)\n\n")
        f.write(key_metrics_df.to_string(index=False))
        f.write("\n\n" + "="*80 + "\n")
        f.write("INTERPRETATION GUIDE\n")
        f.write("="*80 + "\n\n")
        f.write("Utilization: Higher is better (1.0 = all clusters used)\n")
        f.write("Perplexity Ratio: Higher is better (closer to 1.0 = more uniform distribution)\n")
        f.write("Gini: Lower is better (closer to 0 = more equal distribution)\n\n")
        f.write("Ideal codebook: High utilization, high perplexity ratio, low Gini\n")
    print(f"Key metrics summary saved to: {key_metrics_file}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    # Provide recommendations
    inertias = [summaries[name]['inertia'] for name in files]
    k_values = [summaries[name]['n_clusters'] for name in files]
    
    # Find elbow point (simplified)
    if len(k_values) > 2:
        # Calculate relative improvement
        improvements = []
        for i in range(1, len(inertias)):
            improvement = (inertias[i-1] - inertias[i]) / inertias[i-1] * 100
            improvements.append(improvement)
        
        print("\nInertia improvement (% decrease from previous K):")
        for i, improvement in enumerate(improvements):
            print(f"  K={k_values[i]} -> K={k_values[i+1]}: {improvement:.2f}%")
        
        # Find where improvement drops significantly
        if len(improvements) > 1:
            avg_improvement = np.mean(improvements)
            for i, improvement in enumerate(improvements):
                if improvement < avg_improvement * 0.5 and i > 0:
                    print(f"\nSuggested K: {k_values[i]} (diminishing returns after this point)")
                    break
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
