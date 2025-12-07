#!/usr/bin/env python3
"""
Codebook Evaluation Script - Paper-Based Metrics
=================================================

Evaluates a trained codebook using metrics from published papers.

Usage:
    python evaluate_codebook.py \
        --codebook data/structure_codebook_K512.pkl \
        --embeddings data/esm_embeddings.npy \
        --proteins data/sample_proteins.json \
        --output_dir evaluation_results

Metrics & References:
    [1] PERPLEXITY - van den Oord et al., NeurIPS 2017
        https://arxiv.org/abs/1711.00937
    [2] UTILIZATION - Razavi et al., NeurIPS 2019
        https://arxiv.org/abs/1906.00446
    [3] QUANTIZATION ERROR - Shuai et al., arXiv 2024
        https://arxiv.org/abs/2405.15840
    [4] GINI COEFFICIENT - van Kempen et al., Nature Biotechnology 2024
        https://www.nature.com/articles/s41587-023-01773-0
    [5] PER-PROTEIN DIVERSITY - Lin et al., bioRxiv 2023
        https://www.biorxiv.org/content/10.1101/2023.11.27.568722
"""

import argparse
import pickle
import numpy as np
import json
import os
from collections import Counter
from scipy.stats import entropy
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def load_data(codebook_path, embeddings_path, proteins_path):
    """Load codebook, embeddings, and protein metadata"""
    print("="*70)
    print("📂 LOADING DATA")
    print("="*70)
    
    # Load codebook
    print(f"\n   Codebook: {codebook_path}")
    with open(codebook_path, 'rb') as f:
        codebook_data = pickle.load(f)
    
    # Handle both dict and direct kmeans object
    if isinstance(codebook_data, dict):
        kmeans = codebook_data['kmeans']
    else:
        kmeans = codebook_data
    
    K = kmeans.n_clusters
    print(f"   K = {K}")
    
    # Load embeddings
    print(f"   Embeddings: {embeddings_path}")
    embeddings = np.load(embeddings_path)
    print(f"   Shape: {embeddings.shape}")
    
    # Load proteins
    print(f"   Proteins: {proteins_path}")
    with open(proteins_path, 'r') as f:
        proteins = json.load(f)
    print(f"   Count: {len(proteins)}")
    
    return kmeans, K, embeddings, proteins


def tokenize_embeddings(embeddings, kmeans, proteins):
    """Tokenize all embeddings and split by protein"""
    print("\n📊 Tokenizing embeddings...")
    
    all_tokens = kmeans.predict(embeddings)
    print(f"   Total tokens: {len(all_tokens)}")
    
    # Split by protein
    protein_tokens = []
    idx = 0
    for p in proteins:
        length = p.get('length', len(p.get('sequence', '')))
        protein_tokens.append(all_tokens[idx:idx+length])
        idx += length
    
    print(f"   Proteins tokenized: {len(protein_tokens)}")
    return all_tokens, protein_tokens


# =============================================================================
# METRIC 1: PERPLEXITY
# Reference: van den Oord et al., "Neural Discrete Representation Learning"
#            NeurIPS 2017, https://arxiv.org/abs/1711.00937
# =============================================================================
def compute_perplexity(all_tokens, K):
    """
    Perplexity = 2^H where H = entropy of token distribution.
    Measures effective vocabulary size.
    Target: perplexity/K > 60%
    """
    counts = Counter(all_tokens)
    total = len(all_tokens)
    probs = np.array([counts.get(i, 0) / total for i in range(K)])
    probs_nonzero = probs[probs > 0]
    H = entropy(probs_nonzero, base=2)
    perplexity = 2 ** H
    return perplexity, H, perplexity / K


# =============================================================================
# METRIC 2: CODEBOOK UTILIZATION
# Reference: Razavi et al., "VQ-VAE-2", NeurIPS 2019
#            https://arxiv.org/abs/1906.00446
# =============================================================================
def compute_utilization(all_tokens, K):
    """
    Utilization = (unique tokens used) / K
    Target: > 90%
    """
    unique_used = len(set(all_tokens))
    utilization = unique_used / K * 100
    unused = K - unique_used
    return utilization, unique_used, unused


# =============================================================================
# METRIC 3: QUANTIZATION ERROR
# Reference: Shuai et al., "Learning the Language of Protein Structure"
#            arXiv 2024, https://arxiv.org/abs/2405.15840
# =============================================================================
def compute_quantization_error(embeddings, all_tokens, kmeans):
    """
    QE = mean L2 distance from embedding to assigned centroid.
    Proxy for reconstruction error. Lower = better.
    """
    centroids = kmeans.cluster_centers_
    distances = np.linalg.norm(embeddings - centroids[all_tokens], axis=1)
    return np.mean(distances), np.std(distances), np.min(distances), np.max(distances)


# =============================================================================
# METRIC 4: TOKEN DISTRIBUTION (GINI)
# Reference: van Kempen et al., "Foldseek", Nature Biotechnology 2024
#            https://www.nature.com/articles/s41587-023-01773-0
# =============================================================================
def compute_gini(all_tokens, K):
    """
    Gini coefficient measures inequality in token usage.
    0 = uniform, 1 = one token dominates
    Natural proteins: Gini 0.3-0.5 expected
    """
    counts = Counter(all_tokens)
    frequencies = np.array([counts.get(i, 0) for i in range(K)])
    sorted_freqs = np.sort(frequencies)
    n = len(sorted_freqs)
    if np.sum(sorted_freqs) > 0:
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_freqs))) / (n * np.sum(sorted_freqs)) - (n + 1) / n
    else:
        gini = 1.0
    return gini, frequencies, counts.most_common(10)


# =============================================================================
# METRIC 5: PER-PROTEIN DIVERSITY
# Reference: Lin et al., "ProToken", bioRxiv 2023
#            https://www.biorxiv.org/content/10.1101/2023.11.27.568722
# =============================================================================
def compute_diversity(protein_tokens):
    """
    Diversity = unique_tokens / length per protein
    Compression = length / unique_tokens
    """
    diversities = []
    compressions = []
    for tok_seq in protein_tokens:
        if len(tok_seq) > 0:
            unique = len(set(tok_seq))
            length = len(tok_seq)
            diversities.append(unique / length)
            compressions.append(length / unique if unique > 0 else 0)
    return np.mean(diversities), np.mean(compressions), np.min(compressions), np.max(compressions)


def print_results(K, total_tokens, num_proteins, perplexity, H, perp_ratio, 
                  utilization, unique_used, unused, mean_qe, std_qe, min_qe, max_qe,
                  gini, top_tokens, mean_div, mean_comp, min_comp, max_comp):
    """Print all results with paper references"""
    
    # Metric 1: Perplexity
    print("\n" + "-"*70)
    print("[1] PERPLEXITY (Effective Vocabulary Size)")
    print("    Reference: van den Oord et al., NeurIPS 2017")
    print("    Link: https://arxiv.org/abs/1711.00937")
    print("-"*70)
    print(f"   Entropy: {H:.2f} bits")
    print(f"   Perplexity: {perplexity:.1f}")
    print(f"   Max possible: {K}")
    print(f"   Perplexity ratio: {perp_ratio*100:.1f}%")
    if perp_ratio > 0.6:
        print("   ✅ GOOD: Most tokens contribute meaningfully")
    else:
        print("   ⚠️  LOW: Many tokens underutilized")
    
    # Metric 2: Utilization
    print("\n" + "-"*70)
    print("[2] CODEBOOK UTILIZATION")
    print("    Reference: Razavi et al., NeurIPS 2019")
    print("    Link: https://arxiv.org/abs/1906.00446")
    print("-"*70)
    print(f"   Tokens used: {unique_used}/{K}")
    print(f"   Utilization: {utilization:.1f}%")
    print(f"   Unused tokens: {unused}")
    if utilization > 90:
        print("   ✅ EXCELLENT: Almost all tokens in use")
    elif utilization > 70:
        print("   ✅ GOOD")
    else:
        print("   ⚠️  LOW: Consider reducing K")
    
    # Metric 3: Quantization Error
    print("\n" + "-"*70)
    print("[3] QUANTIZATION ERROR (Reconstruction Proxy)")
    print("    Reference: Shuai et al., arXiv 2024")
    print("    Link: https://arxiv.org/abs/2405.15840")
    print("-"*70)
    print(f"   Mean QE: {mean_qe:.4f}")
    print(f"   Std QE: {std_qe:.4f}")
    print(f"   Min QE: {min_qe:.4f}")
    print(f"   Max QE: {max_qe:.4f}")
    print("   (Lower = better representation)")
    
    # Metric 4: Gini
    print("\n" + "-"*70)
    print("[4] TOKEN DISTRIBUTION (Gini Coefficient)")
    print("    Reference: van Kempen et al., Nature Biotechnology 2024")
    print("    Link: https://www.nature.com/articles/s41587-023-01773-0")
    print("-"*70)
    print(f"   Gini coefficient: {gini:.3f}")
    print(f"   (0 = uniform, 1 = one token dominates)")
    print(f"\n   Top 10 most common tokens:")
    for tok, cnt in top_tokens:
        pct = cnt / total_tokens * 100
        print(f"      Token {tok:3d}: {cnt:5,} occurrences ({pct:.2f}%)")
    if 0.3 < gini < 0.6:
        print("\n   ✅ NATURAL: Expected imbalance for protein structures")
    elif gini > 0.7:
        print("\n   ⚠️  HIGH: Few tokens dominate")
    else:
        print("\n   ⚠️  LOW: Very uniform (K might be too small)")
    
    # Metric 5: Diversity
    print("\n" + "-"*70)
    print("[5] PER-PROTEIN TOKEN DIVERSITY")
    print("    Reference: Lin et al., bioRxiv 2023")
    print("    Link: https://www.biorxiv.org/content/10.1101/2023.11.27.568722")
    print("-"*70)
    print(f"   Mean diversity: {mean_div:.3f}")
    print(f"   Mean compression: {mean_comp:.2f}x")
    print(f"   Compression range: {min_comp:.1f}x - {max_comp:.1f}x")
    
    # Final Verdict
    print("\n" + "="*70)
    print("📋 FINAL VERDICT")
    print("="*70)
    
    perp_ok = perp_ratio > 0.6
    util_ok = utilization > 90
    gini_ok = 0.25 < gini < 0.65
    
    print(f"\n   Codebook K = {K}")
    print(f"   Perplexity Ratio: {perp_ratio*100:.1f}% {'✅' if perp_ok else '⚠️'}")
    print(f"   Utilization: {utilization:.1f}% {'✅' if util_ok else '⚠️'}")
    print(f"   Gini: {gini:.3f} {'✅' if gini_ok else '⚠️'}")
    print(f"   Mean QE: {mean_qe:.4f}")
    
    print("\n   " + "-"*50)
    if perp_ok and util_ok and gini_ok:
        print("   ✅ VERDICT: Codebook is GOOD!")
        print("   → Ready for use in tokenization")
    elif not util_ok:
        new_k = int(K * utilization / 100)
        print(f"   ⚠️  VERDICT: Low utilization, try K={new_k}")
    elif not perp_ok:
        print(f"   ⚠️  VERDICT: Low perplexity, try smaller K")
    else:
        print(f"   ⚠️  VERDICT: Check distribution")
    
    return perp_ok, util_ok, gini_ok


def plot_results(frequencies, K, perp_ratio, utilization, gini, compressions, output_dir):
    """Generate evaluation plots"""
    print(f"\n📈 Generating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Token frequency histogram
    axes[0,0].hist(frequencies[frequencies > 0], bins=50, color='steelblue', alpha=0.7)
    axes[0,0].axvline(np.mean(frequencies), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(frequencies):.1f}')
    axes[0,0].set_xlabel('Token Frequency')
    axes[0,0].set_ylabel('Count')
    axes[0,0].set_title('Token Frequency Distribution\n(Foldseek, Nat Biotech 2024)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Ranked frequencies
    axes[0,1].bar(range(K), sorted(frequencies, reverse=True), color='steelblue', alpha=0.7)
    axes[0,1].set_xlabel('Token Rank')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Ranked Token Frequencies\n(VQ-VAE-2, NeurIPS 2019)')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Summary bar
    summary = {'Perplexity\nRatio': perp_ratio*100, 'Utilization': utilization, '100-Gini': (1-gini)*100}
    colors = ['steelblue', 'forestgreen', 'coral']
    bars = axes[1,0].bar(summary.keys(), summary.values(), color=colors)
    axes[1,0].axhline(60, color='red', linestyle='--', alpha=0.5, label='60% threshold')
    axes[1,0].set_ylabel('Percentage (%)')
    axes[1,0].set_title('Summary Metrics (Higher=Better)')
    axes[1,0].set_ylim([0, 105])
    axes[1,0].legend()
    for bar, val in zip(bars, summary.values()):
        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                       f'{val:.1f}%', ha='center', fontweight='bold')
    
    # 4. Compression histogram
    axes[1,1].hist(compressions, bins=30, color='forestgreen', alpha=0.7)
    axes[1,1].axvline(np.mean(compressions), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(compressions):.2f}x')
    axes[1,1].set_xlabel('Compression Ratio')
    axes[1,1].set_ylabel('Count')
    axes[1,1].set_title('Per-Protein Compression\n(ProToken, bioRxiv 2023)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'codebook_evaluation.png')
    plt.savefig(output_file, dpi=150)
    print(f"   ✅ Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate codebook using paper-based metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
References:
    [1] van den Oord et al., "VQ-VAE", NeurIPS 2017
    [2] Razavi et al., "VQ-VAE-2", NeurIPS 2019  
    [3] Shuai et al., arXiv:2405.15840, 2024
    [4] van Kempen et al., "Foldseek", Nature Biotechnology 2024
    [5] Lin et al., "ProToken", bioRxiv 2023
        """
    )
    parser.add_argument('--codebook', type=str, required=True,
                        help='Path to codebook .pkl file')
    parser.add_argument('--embeddings', type=str, required=True,
                        help='Path to embeddings .npy file')
    parser.add_argument('--proteins', type=str, required=True,
                        help='Path to proteins .json file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    kmeans, K, embeddings, proteins = load_data(args.codebook, args.embeddings, args.proteins)
    
    # Tokenize
    all_tokens, protein_tokens = tokenize_embeddings(embeddings, kmeans, proteins)
    total_tokens = len(all_tokens)
    num_proteins = len(proteins)
    
    print("\n" + "="*70)
    print("📊 COMPUTING METRICS")
    print("="*70)
    
    # Compute all metrics
    perplexity, H, perp_ratio = compute_perplexity(all_tokens, K)
    utilization, unique_used, unused = compute_utilization(all_tokens, K)
    mean_qe, std_qe, min_qe, max_qe = compute_quantization_error(embeddings, all_tokens, kmeans)
    gini, frequencies, top_tokens = compute_gini(all_tokens, K)
    mean_div, mean_comp, min_comp, max_comp = compute_diversity(protein_tokens)
    
    # Get compressions for plotting
    compressions = []
    for tok_seq in protein_tokens:
        if len(tok_seq) > 0:
            unique = len(set(tok_seq))
            compressions.append(len(tok_seq) / unique if unique > 0 else 0)
    
    # Print results
    perp_ok, util_ok, gini_ok = print_results(
        K, total_tokens, num_proteins,
        perplexity, H, perp_ratio,
        utilization, unique_used, unused,
        mean_qe, std_qe, min_qe, max_qe,
        gini, top_tokens,
        mean_div, mean_comp, min_comp, max_comp
    )
    
    # Generate plots
    plot_results(frequencies, K, perp_ratio, utilization, gini, compressions, args.output_dir)
    
    # Save metrics
    metrics = {
        'K': K,
        'total_residues': total_tokens,
        'num_proteins': num_proteins,
        'perplexity': {
            'value': float(perplexity),
            'ratio': float(perp_ratio),
            'entropy_bits': float(H),
            'status': 'good' if perp_ok else 'low'
        },
        'utilization': {
            'percent': float(utilization),
            'unique_used': int(unique_used),
            'unused': int(unused),
            'status': 'good' if util_ok else 'low'
        },
        'quantization_error': {
            'mean': float(mean_qe),
            'std': float(std_qe),
            'min': float(min_qe),
            'max': float(max_qe)
        },
        'gini': {
            'value': float(gini),
            'status': 'good' if gini_ok else 'check'
        },
        'diversity': {
            'mean_diversity': float(mean_div),
            'mean_compression': float(mean_comp),
            'min_compression': float(min_comp),
            'max_compression': float(max_comp)
        }
    }
    
    output_file = os.path.join(args.output_dir, 'codebook_metrics.json')
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n   ✅ Saved: {output_file}")
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
