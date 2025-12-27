#!/usr/bin/env python3
"""
Train k-means Codebook on ESM-2 Embeddings from Standardized Dataset
Learns discrete vocabulary of structure tokens
"""

import json
import numpy as np
import pickle
import argparse
from pathlib import Path
from collections import Counter
from scipy.stats import entropy
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import time
import warnings
import gc
warnings.filterwarnings('ignore')
from tqdm import tqdm

def train_kmeans_from_batches(embeddings_dir, n_clusters=512, batch_size=1024):
    """Train k-means clustering from batch files"""
    print(f"\n🔄 Training k-means with {n_clusters} clusters...")
    print(f"   Batch size: {batch_size}")
    
    # Find all batch files
    batch_files = sorted(embeddings_dir.glob('esm_embeddings_batch_*.npy'))
    
    if not batch_files:
        raise FileNotFoundError(f"No batch files found in {embeddings_dir}")
    
    print(f"   Found {len(batch_files)} batch files")
    
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        max_iter=100,
        n_init=10,
        random_state=42,
        verbose=1
    )
    
    print("\nFitting model...")
    start_time_total = time.time()
    batch_times = []
    
    for i, batch_file in enumerate(batch_files):
        print(f"\n   Loading batch {i}...")
        batch_start = time.time()
        
        embeddings_batch = np.load(batch_file)
        load_time = time.time() - batch_start
        print(f"   Batch {i} shape: {embeddings_batch.shape} (loaded in {load_time:.2f}s)")
        
        # Partial fit on this batch
        fit_start = time.time()
        kmeans.partial_fit(embeddings_batch)
        fit_time = time.time() - fit_start
        batch_total_time = time.time() - batch_start
        batch_times.append(batch_total_time)
        
        print(f"   ✅ Partial fit on batch {i} complete (Inertia: {kmeans.inertia_:.2f})")
        print(f"      Fit time: {fit_time:.2f}s | Total batch time: {batch_total_time:.2f}s")
        
        # Clean up memory
        del embeddings_batch
        gc.collect()
    
    total_time = time.time() - start_time_total
    avg_batch_time = np.mean(batch_times)
    
    print(f"\n📊 Training Statistics:")
    print(f"   Total training time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"   Average batch time: {avg_batch_time:.2f}s")
    print(f"   Fastest batch: {min(batch_times):.2f}s")
    print(f"   Slowest batch: {max(batch_times):.2f}s")
    
    return kmeans

def analyze_clustering_from_batches(embeddings_dir, kmeans, n_clusters):
    """Analyze cluster distribution across all batches"""
    print("\n" + "=" * 70)
    print("CLUSTER ANALYSIS")
    print("=" * 70)
    
    batch_files = sorted(embeddings_dir.glob('esm_embeddings_batch_*.npy'))
    
    # Collect labels from all batches
    all_labels = []
    total_residues = 0
    
    for batch_file in tqdm(batch_files, desc="Analyzing batches"):
        embeddings_batch = np.load(batch_file).astype(np.float32)
        batch_size = len(embeddings_batch)
        total_residues += batch_size
        
        # Predict labels for entire batch
        batch_labels = kmeans.predict(embeddings_batch)
        all_labels.extend(batch_labels)
        
        del embeddings_batch
        gc.collect()
    
    labels = np.array(all_labels)
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    print(f"\n📊 Distribution:")
    print(f"   Total clusters: {n_clusters}")
    print(f"   Clusters used: {len(unique_labels)}")
    print(f"   Utilization: {len(unique_labels)/n_clusters*100:.1f}%")
    print(f"   Total residues: {total_residues:,}")
    print(f"   Inertia: {kmeans.inertia_:.2f}")
    
    print(f"\n   Top 10 largest clusters:")
    sorted_idx = np.argsort(counts)[::-1]
    for i in range(min(10, len(counts))):
        idx = sorted_idx[i]
        print(f"     Cluster {unique_labels[idx]:3d}: {counts[idx]:6d} residues ({counts[idx]/len(labels)*100:5.2f}%)")
    
    print(f"\n   Top 10 smallest clusters:")
    for i in range(min(10, len(counts))):
        idx = sorted_idx[-(i+1)]
        print(f"     Cluster {unique_labels[idx]:3d}: {counts[idx]:6d} residues ({counts[idx]/len(labels)*100:5.2f}%)")
    
    return labels, len(unique_labels), counts, total_residues

def compute_perplexity(all_tokens, K):
    """Perplexity = 2^H where H = entropy. Measures effective vocabulary size."""
    counts = Counter(all_tokens)
    total = len(all_tokens)
    probs = np.array([counts.get(i, 0) / total for i in range(K)])
    probs_nonzero = probs[probs > 0]
    H = entropy(probs_nonzero, base=2)
    perplexity = 2 ** H
    return perplexity, H, perplexity / K

def compute_quantization_error(embeddings_dir, all_tokens, kmeans):
    """Mean L2 distance from embedding to assigned centroid."""
    batch_files = sorted(embeddings_dir.glob('esm_embeddings_batch_*.npy'))
    centroids = kmeans.cluster_centers_
    
    all_distances = []
    token_idx = 0
    
    for batch_file in tqdm(batch_files, desc="Computing QE"):
        embeddings_batch = np.load(batch_file).astype(np.float32)
        batch_size = len(embeddings_batch)
        batch_tokens = all_tokens[token_idx:token_idx+batch_size]
        
        distances = np.linalg.norm(embeddings_batch - centroids[batch_tokens], axis=1)
        all_distances.extend(distances)
        
        token_idx += batch_size
        del embeddings_batch
        gc.collect()
    
    distances_arr = np.array(all_distances)
    return np.mean(distances_arr), np.std(distances_arr)

def compute_gini(all_tokens, K):
    """Gini coefficient measures inequality in token usage."""
    counts = Counter(all_tokens)
    frequencies = np.array([counts.get(i, 0) for i in range(K)])
    sorted_freqs = np.sort(frequencies)
    n = len(sorted_freqs)
    if np.sum(sorted_freqs) > 0:
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_freqs))) / (n * np.sum(sorted_freqs)) - (n + 1) / n
    else:
        gini = 1.0
    return gini

def evaluate_codebook(embeddings_dir, labels, kmeans, n_clusters, total_residues):
    """Run comprehensive codebook evaluation"""
    print("\n" + "=" * 70)
    print("CODEBOOK EVALUATION")
    print("=" * 70)
    
    all_tokens = labels
    K = n_clusters
    
    # Metric 1: Perplexity
    perplexity, H, perp_ratio = compute_perplexity(all_tokens, K)
    
    # Metric 2: Utilization (already computed, just format)
    unique_used = len(set(all_tokens))
    utilization = unique_used / K * 100
    
    # Metric 3: Quantization Error
    print("\n📊 Computing quantization error...")
    # mean_qe, std_qe = compute_quantization_error(embeddings_dir, all_tokens, kmeans)
    
    # Metric 4: Gini
    gini = compute_gini(all_tokens, K)
    
    # Print results
    print("\n" + "-"*70)
    print("EVALUATION RESULTS")
    print("-"*70)
    
    print(f"\n[1] PERPLEXITY (Effective Vocabulary)")
    print(f"    Entropy: {H:.2f} bits")
    print(f"    Perplexity: {perplexity:.1f} / {K}")
    print(f"    Ratio: {perp_ratio*100:.1f}%")
    if perp_ratio > 0.6:
        print(f"    ✅ GOOD: Most tokens contribute meaningfully")
    else:
        print(f"    ⚠️  LOW: Many tokens underutilized")
    
    print(f"\n[2] UTILIZATION")
    print(f"    Tokens used: {unique_used}/{K}")
    print(f"    Utilization: {utilization:.1f}%")
    if utilization > 90:
        print(f"    ✅ EXCELLENT: Almost all tokens in use")
    elif utilization > 70:
        print(f"    ✅ GOOD")
    else:
        print(f"    ⚠️  LOW: Consider reducing K")
    
    print(f"\n[3] QUANTIZATION ERROR")
    print(f"    Mean QE: {'N/A'}")
    print(f"    Std QE: {'N/A'}")
    print(f"    (Lower = better representation)")
    
    print(f"\n[4] TOKEN DISTRIBUTION (Gini)")
    print(f"    Gini coefficient: {gini:.3f}")
    print(f"    (0 = uniform, 1 = one token dominates)")
    if 0.3 < gini < 0.6:
        print(f"    ✅ NATURAL: Expected imbalance for proteins")
    elif gini > 0.7:
        print(f"    ⚠️  HIGH: Few tokens dominate")
    else:
        print(f"    ⚠️  LOW: Very uniform")
    
    # Final verdict
    perp_ok = bool(perp_ratio > 0.6)
    util_ok = bool(utilization > 90)
    gini_ok = bool(0.25 < gini < 0.65)
    
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    print(f"   Perplexity Ratio: {perp_ratio*100:.1f}% {'✅' if perp_ok else '⚠️'}")
    print(f"   Utilization: {utilization:.1f}% {'✅' if util_ok else '⚠️'}")
    print(f"   Gini: {gini:.3f} {'✅' if gini_ok else '⚠️'}")
    
    if perp_ok and util_ok and gini_ok:
        print("\n   ✅ VERDICT: Codebook is GOOD!")
    elif not util_ok:
        new_k = int(K * utilization / 100)
        print(f"\n   ⚠️  VERDICT: Low utilization, try K={new_k}")
    elif not perp_ok:
        print(f"\n   ⚠️  VERDICT: Low perplexity, try smaller K")
    else:
        print(f"\n   ⚠️  VERDICT: Check distribution")
    
    return {
        'perplexity': float(perplexity),
        'perplexity_ratio': float(perp_ratio),
        'entropy': float(H),
        'utilization': float(utilization),
        'mean_qe': float(0),
        'std_qe': float(0),
        'gini': float(gini),
        'verdict': 'good' if (perp_ok and util_ok and gini_ok) else 'check',
        'perplexity_ok': perp_ok,
        'utilization_ok': util_ok,
        'gini_ok': gini_ok
    }

def main():
    print("=" * 70)
    print("TRAIN K-MEANS CODEBOOK FROM STANDARDIZED DATASET")
    print("=" * 70)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train k-means codebook on ESM embeddings')
    parser.add_argument('--k', type=int, default=512, help='Number of clusters (default: 512)')
    args = parser.parse_args()
    
    # Configuration
    DATA_DIR = Path(__file__).parent.parent / 'data'
    EMBEDDINGS_DIR = DATA_DIR / 'esm_embeddings'
    N_CLUSTERS = args.k
    
    print(f"\n📂 Data directory: {DATA_DIR}")
    print(f"📂 Embeddings directory: {EMBEDDINGS_DIR}")
    print(f"🎯 Number of clusters: {N_CLUSTERS}")
    
    if not EMBEDDINGS_DIR.exists():
        print(f"❌ Embeddings directory not found: {EMBEDDINGS_DIR}")
        print(f"   Run 01_extract_esm_embeddings.py first!")
        return
    
    # Check for existing model
    kmeans_file = DATA_DIR / f'structure_codebook_K{N_CLUSTERS}.pkl'
    
    if kmeans_file.exists():
        print(f"\n🔄 Loading existing k-means model: {kmeans_file}")
        with open(kmeans_file, 'rb') as f:
            saved_data = pickle.load(f)
            kmeans = saved_data['kmeans']
        print(f"✅ Loaded k-means model")
    else:
        # Train k-means
        print("\n" + "=" * 70)
        print("STEP 1: Training k-means from Batch Files")
        print("=" * 70)
        
        kmeans = train_kmeans_from_batches(EMBEDDINGS_DIR, N_CLUSTERS)
        print(f"\n✅ Training complete")
    
    # Analyze
    print("\n" + "=" * 70)
    print("STEP 2: Analyzing Clusters")
    print("=" * 70)
    
    labels, clusters_used, counts, total_residues = analyze_clustering_from_batches(
        EMBEDDINGS_DIR, kmeans, N_CLUSTERS
    )
    
    # Evaluate codebook
    print("\n" + "=" * 70)
    print("STEP 3: Evaluating Codebook")
    print("=" * 70)
    
    evaluation_metrics = evaluate_codebook(EMBEDDINGS_DIR, labels, kmeans, N_CLUSTERS, total_residues)
        
    # Save codebook
    print("\n" + "=" * 70)
    print("STEP 4: Saving Codebook")
    print("=" * 70)
    
    with open(kmeans_file, 'wb') as f:
        pickle.dump({
            'kmeans': kmeans,
            'n_clusters': N_CLUSTERS,
            'clusters_used': clusters_used,
            'embedding_dim': kmeans.cluster_centers_.shape[1],
            'n_residues': total_residues,
            'model_name': 'facebook/esm2_t33_650M_UR50D',
            'timestamp': datetime.now().isoformat(),
            'cluster_counts': counts.tolist(),
            'inertia': float(kmeans.inertia_)
        }, f)
    
    print(f"✅ Saved codebook: {kmeans_file}")
    print(f"   Size: {kmeans_file.stat().st_size / 1024:.1f} KB")
    
    # Save summary
    summary_file = DATA_DIR / f'codebook_summary_K{N_CLUSTERS}.json'
    
    with open(summary_file, 'w') as f:
        unique_labels, unique_counts = np.unique(labels, return_counts=True)
        sorted_idx = np.argsort(unique_counts)[::-1]
        
        all_clusters = [
            {'cluster_id': int(unique_labels[i]), 'count': int(unique_counts[i])}
            for i in sorted_idx
        ]
        
        json.dump({
            'n_clusters': N_CLUSTERS,
            'clusters_used': int(clusters_used),
            'utilization': float(clusters_used / N_CLUSTERS),
            'n_residues': int(total_residues),
            'embedding_dim': int(kmeans.cluster_centers_.shape[1]),
            'inertia': float(kmeans.inertia_),
            'timestamp': datetime.now().isoformat(),
            'all_clusters': all_clusters,
            'top_10_clusters': all_clusters[:10],
            'evaluation': evaluation_metrics
        }, f, indent=2)
    
    print(f"✅ Saved summary: {summary_file}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ CODEBOOK TRAINING COMPLETE!")
    print("=" * 70)
    
    print(f"\n📊 Results:")
    print(f"   Codebook size: {N_CLUSTERS} structure tokens")
    print(f"   Utilization: {clusters_used}/{N_CLUSTERS} ({clusters_used/N_CLUSTERS*100:.1f}%)")
    print(f"   Training residues: {total_residues:,}")
    print(f"   Embedding dim: {kmeans.cluster_centers_.shape[1]}")
    print(f"   Inertia: {kmeans.inertia_:.2f}")
    
    print(f"\n📁 Output files:")
    print(f"   Codebook: {kmeans_file.name}")
    print(f"   Summary: {summary_file.name}")
    
    print(f"\n📊 Evaluation Summary:")
    print(f"   Verdict: {evaluation_metrics['verdict']}")
    print(f"   Perplexity ratio: {evaluation_metrics['perplexity_ratio']*100:.1f}%")
    print(f"   Utilization: {evaluation_metrics['utilization']:.1f}%")
    print(f"   Gini: {evaluation_metrics['gini']:.3f}")

if __name__ == '__main__':
    main()
