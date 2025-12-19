#!/usr/bin/env python3
"""
Summarize Evaluation Results Across Batches
Computes average metrics for each K value and creates plots/tables
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def load_batch_metrics(k_value):
    """Load metrics from all batches for a given K value"""
    base_dir = Path(f"evaluation_results/K{k_value}")
    
    perplexity_values = []
    utilization_values = []
    gini_values = []
    perplexity_status = []
    utilization_status = []
    gini_status = []
    
    # Load all batch files
    for batch_dir in sorted(base_dir.glob("batch_*"), key=lambda x: int(x.name.split("_")[1])):
        metrics_file = batch_dir / "codebook_metrics.json"
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                
            perplexity_values.append(data['perplexity']['value'])
            utilization_values.append(data['utilization']['percent'])
            gini_values.append(data['gini']['value'])
            perplexity_status.append(data['perplexity']['status'])
            utilization_status.append(data['utilization']['status'])
            gini_status.append(data['gini']['status'])
    
    return {
        'perplexity': perplexity_values,
        'utilization': utilization_values,
        'gini': gini_values,
        'perplexity_status': perplexity_status,
        'utilization_status': utilization_status,
        'gini_status': gini_status
    }


def main():
    print("=" * 70)
    print("SUMMARIZING EVALUATION RESULTS")
    print("=" * 70)
    
    # K values to analyze
    k_values = [128, 256, 512]
    
    # Collect data
    results = {}
    for k in k_values:
        print(f"\n📊 Loading K={k}...")
        results[k] = load_batch_metrics(k)
        print(f"   Loaded {len(results[k]['perplexity'])} batches")
    
    # Compute averages
    print("\n" + "=" * 70)
    print("COMPUTING AVERAGES")
    print("=" * 70)
    
    summary_data = []
    for k in k_values:
        avg_perplexity = np.mean(results[k]['perplexity'])
        perplexity_ratio = (avg_perplexity / k) * 100
        
        avg_utilization = np.mean(results[k]['utilization'])
        
        avg_gini = np.mean(results[k]['gini'])
        
        # Get most common status for each metric
        from collections import Counter
        perp_verdict = Counter(results[k]['perplexity_status']).most_common(1)[0][0]
        util_verdict = Counter(results[k]['utilization_status']).most_common(1)[0][0]
        gini_verdict = Counter(results[k]['gini_status']).most_common(1)[0][0]
        
        summary_data.append({
            'K': k,
            'Perplexity/K (%)': perplexity_ratio,
            'Perp Verdict': perp_verdict,
            'Utilization (%)': avg_utilization,
            'Util Verdict': util_verdict,
            'Gini': avg_gini,
            'Gini Verdict': gini_verdict
        })
        
        print(f"\nK={k}:")
        print(f"  Perplexity/K: {perplexity_ratio:.2f}% [{perp_verdict}]")
        print(f"  Utilization: {avg_utilization:.2f}% [{util_verdict}]")
        print(f"  Gini: {avg_gini:.4f} [{gini_verdict}]")
    
    # Create DataFrame
    df = pd.DataFrame(summary_data)
    
    # Save table
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save CSV
    csv_file = output_dir / "summary_across_K.csv"
    df.to_csv(csv_file, index=False)
    print(f"\n✅ Saved table: {csv_file}")
    
    # Print table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(df.to_string(index=False))
    
    # Create plots
    print("\n" + "=" * 70)
    print("CREATING PLOTS")
    print("=" * 70)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Perplexity/K
    axes[0].bar(df['K'], df['Perplexity/K (%)'], color='steelblue', alpha=0.7)
    axes[0].set_xlabel('K (Number of Clusters)', fontsize=12)
    axes[0].set_ylabel('Perplexity/K (%)', fontsize=12)
    axes[0].set_title('Perplexity/K vs K', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].axhline(y=60, color='red', linestyle='--', linewidth=1, label='Target (60%)')
    axes[0].legend()
    
    # Plot 2: Utilization
    axes[1].bar(df['K'], df['Utilization (%)'], color='forestgreen', alpha=0.7)
    axes[1].set_xlabel('K (Number of Clusters)', fontsize=12)
    axes[1].set_ylabel('Utilization (%)', fontsize=12)
    axes[1].set_title('Utilization vs K', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim([0, 100])
    axes[1].axhline(y=90, color='red', linestyle='--', linewidth=1, label='Target (90%)')
    axes[1].legend()
    
    # Plot 3: Gini
    axes[2].bar(df['K'], df['Gini'], color='coral', alpha=0.7)
    axes[2].set_xlabel('K (Number of Clusters)', fontsize=12)
    axes[2].set_ylabel('Gini Coefficient', fontsize=12)
    axes[2].set_title('Gini Coefficient vs K', fontsize=14, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='Threshold (0.5)')
    axes[2].legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / "summary_plots.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved plots: {plot_file}")
    
    # Create combined line plot
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    # Use actual percentage values for comparison
    ax.plot(df['K'], df['Perplexity/K (%)'], marker='o', linewidth=2, label='Perplexity/K (%)', color='steelblue')
    ax.plot(df['K'], df['Utilization (%)'], marker='s', linewidth=2, label='Utilization (%)', color='forestgreen')
    ax.plot(df['K'], df['Gini'] * 100, marker='^', linewidth=2, label='Gini × 100', color='coral')
    
    ax.set_xlabel('K (Number of Clusters)', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Codebook Metrics Comparison Across K', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    
    combined_plot_file = output_dir / "summary_combined_plot.png"
    plt.savefig(combined_plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved combined plot: {combined_plot_file}")
    
    print("\n" + "=" * 70)
    print("✅ SUMMARY COMPLETE!")
    print("=" * 70)
    
    print("\n📊 Key Findings:")
    print(f"   Best Perplexity/K: K={df.loc[df['Perplexity/K (%)'].idxmax(), 'K']:.0f} ({df['Perplexity/K (%)'].max():.1f}%)")
    print(f"   Best Utilization: K={df.loc[df['Utilization (%)'].idxmax(), 'K']:.0f} ({df['Utilization (%)'].max():.1f}%)")
    print(f"   Lowest Gini (best balance): K={df.loc[df['Gini'].idxmin(), 'K']:.0f} ({df['Gini'].min():.4f})")


if __name__ == '__main__':
    main()
