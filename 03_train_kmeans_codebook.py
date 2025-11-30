#!/usr/bin/env python3
"""
Train k-means Codebook on ESM-2 Embeddings
Learns discrete vocabulary of structure tokens
"""

import json
import numpy as np
import pickle
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def train_kmeans(embeddings, n_clusters=512, batch_size=1024):
    """Train k-means clustering"""
    print(f"\n🔄 Training k-means with {n_clusters} clusters...")
    print(f"   Batch size: {batch_size}")
    print(f"   Input shape: {embeddings.shape}")
    
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        max_iter=100,
        n_init=10,
        random_state=42,
        verbose=1
    )
    
    print("\nFitting model...")
    kmeans.fit(embeddings)
    
    return kmeans


def analyze_clustering(embeddings, kmeans, n_clusters):
    """Analyze cluster distribution"""
    print("\n" + "=" * 70)
    print("CLUSTER ANALYSIS")
    print("=" * 70)
    
    labels = kmeans.predict(embeddings)
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    print(f"\n📊 Distribution:")
    print(f"   Total clusters: {n_clusters}")
    print(f"   Clusters used: {len(unique_labels)}")
    print(f"   Utilization: {len(unique_labels)/n_clusters*100:.1f}%")
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
    
    return labels, len(unique_labels), counts


def visualize_clustering(embeddings, labels, n_clusters, output_file, max_points=10000):
    """Create PCA visualization"""
    print(f"\n📊 Creating PCA visualization...")
    
    # Sample points for visualization
    n_samples = min(max_points, len(embeddings))
    indices = np.random.choice(len(embeddings), n_samples, replace=False)
    
    # PCA to 2D
    print(f"   Running PCA on {n_samples} points...")
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings[indices])
    
    explained_var = pca.explained_variance_ratio_
    print(f"   Explained variance: {explained_var.sum()*100:.1f}% (PC1: {explained_var[0]*100:.1f}%, PC2: {explained_var[1]*100:.1f}%)")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Colored by cluster
    scatter1 = ax1.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels[indices],
        cmap='tab20',
        s=1,
        alpha=0.6
    )
    ax1.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% variance)', fontsize=12)
    ax1.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% variance)', fontsize=12)
    ax1.set_title(f'ESM-2 Embeddings: k-means Clustering (K={n_clusters})', fontsize=14, fontweight='bold')
    plt.colorbar(scatter1, ax=ax1, label='Cluster ID')
    
    # Right: Density plot
    h = ax2.hist2d(embeddings_2d[:, 0], embeddings_2d[:, 1], bins=100, cmap='viridis')
    ax2.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% variance)', fontsize=12)
    ax2.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% variance)', fontsize=12)
    ax2.set_title('Embedding Density', fontsize=14, fontweight='bold')
    plt.colorbar(h[3], ax=ax2, label='Count')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {output_file}")
    
    plt.close()


def visualize_centroids(kmeans, n_clusters, output_file):
    """Visualize codebook centroids in PCA space"""
    print(f"\n📊 Visualizing codebook centroids...")
    
    # Get centroids
    centroids = kmeans.cluster_centers_  # Shape: [n_clusters, embedding_dim]
    
    # PCA to 2D
    pca = PCA(n_components=2, random_state=42)
    centroids_2d = pca.fit_transform(centroids)
    
    explained_var = pca.explained_variance_ratio_
    print(f"   Centroid PCA variance: {explained_var.sum()*100:.1f}%")
    
    # Create figure with 2 subplots
    fig = plt.figure(figsize=(18, 7))
    
    # Left: Centroid scatter
    ax1 = fig.add_subplot(121)
    scatter = ax1.scatter(
        centroids_2d[:, 0],
        centroids_2d[:, 1],
        c=range(n_clusters),
        cmap='viridis',
        s=100,
        alpha=0.8,
        edgecolors='black',
        linewidth=1
    )
    ax1.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% variance)', fontsize=12)
    ax1.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% variance)', fontsize=12)
    ax1.set_title(f'Codebook Centroids (K={n_clusters})', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Structure Token ID')
    
    # Add token IDs to some centroids
    n_labels = min(50, n_clusters)  # Label first 50
    for i in range(n_labels):
        ax1.annotate(
            f'S{i:03d}',
            (centroids_2d[i, 0], centroids_2d[i, 1]),
            fontsize=6,
            alpha=0.7,
            ha='center'
        )
    
    # Right: Centroid heatmap (first 2 PCs)
    ax2 = fig.add_subplot(122)
    
    # Sort by PC1
    sorted_idx = np.argsort(centroids_2d[:, 0])
    centroids_sorted = centroids_2d[sorted_idx]
    
    # Create heatmap showing PC values
    n_show = min(100, n_clusters)  # Show first 100
    heatmap_data = centroids_sorted[:n_show].T
    
    im = ax2.imshow(heatmap_data, cmap='RdBu_r', aspect='auto')
    ax2.set_xlabel('Structure Token ID (sorted by PC1)', fontsize=12)
    ax2.set_ylabel('Principal Component', fontsize=12)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['PC1', 'PC2'])
    ax2.set_title(f'Centroid PC Values (first {n_show} tokens)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='PC Value')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {output_file}")
    
    plt.close()
    
    return centroids_2d


def main():
    print("=" * 70)
    print("TRAIN K-MEANS CODEBOOK")
    print("=" * 70)
    
    # Configuration
    DATA_DIR = Path('/lustrefs/shared/mohammad.sayeed/Prot2Text/esmfold_tokenizer/data')
    N_CLUSTERS = 512  # Standard size for structure vocabulary
    
    embeddings_file = DATA_DIR / 'esm_embeddings.npy'
    
    if not embeddings_file.exists():
        print(f"❌ Embeddings file not found: {embeddings_file}")
        print(f"   Run 02_extract_esm_embeddings.py first!")
        return
    
    print(f"\n📂 Data directory: {DATA_DIR}")
    print(f"🎯 Number of clusters: {N_CLUSTERS}")
    
    # Load embeddings
    print("\n" + "=" * 70)
    print("STEP 1: Loading Embeddings")
    print("=" * 70)
    
    print(f"\nLoading {embeddings_file.name}...")
    embeddings = np.load(embeddings_file)
    
    print(f"✅ Loaded embeddings:")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Size: {embeddings_file.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"   Memory: {embeddings.nbytes / 1024 / 1024:.1f} MB")
    
    # Train k-means
    print("\n" + "=" * 70)
    print("STEP 2: Training k-means")
    print("=" * 70)
    
    kmeans = train_kmeans(embeddings, N_CLUSTERS)
    
    print(f"\n✅ Training complete")
    
    # Analyze
    print("\n" + "=" * 70)
    print("STEP 3: Analyzing Clusters")
    print("=" * 70)
    
    labels, clusters_used, counts = analyze_clustering(embeddings, kmeans, N_CLUSTERS)
    
    # Visualize
    print("\n" + "=" * 70)
    print("STEP 4: Creating Visualizations")
    print("=" * 70)
    
    # Clustering visualization
    viz_file = DATA_DIR / 'clustering_visualization.png'
    visualize_clustering(embeddings, labels, N_CLUSTERS, viz_file)
    
    # Centroid visualization (CODEBOOK PLOT)
    centroid_file = DATA_DIR / 'codebook_centroids.png'
    centroids_2d = visualize_centroids(kmeans, N_CLUSTERS, centroid_file)
    
    # Save codebook
    print("\n" + "=" * 70)
    print("STEP 5: Saving Codebook")
    print("=" * 70)
    
    codebook_file = DATA_DIR / f'structure_codebook_K{N_CLUSTERS}.pkl'
    
    with open(codebook_file, 'wb') as f:
        pickle.dump({
            'kmeans': kmeans,
            'n_clusters': N_CLUSTERS,
            'clusters_used': clusters_used,
            'embedding_dim': embeddings.shape[1],
            'n_residues': embeddings.shape[0],
            'model_name': 'facebook/esm2_t33_650M_UR50D',
            'timestamp': datetime.now().isoformat(),
            'cluster_counts': counts.tolist(),
            'inertia': float(kmeans.inertia_)
        }, f)
    
    print(f"✅ Saved codebook: {codebook_file}")
    print(f"   Size: {codebook_file.stat().st_size / 1024:.1f} KB")
    
    # Save summary
    summary_file = DATA_DIR / f'codebook_summary_K{N_CLUSTERS}.json'
    
    with open(summary_file, 'w') as f:
        json.dump({
            'n_clusters': N_CLUSTERS,
            'clusters_used': int(clusters_used),
            'utilization': float(clusters_used / N_CLUSTERS),
            'n_residues': int(embeddings.shape[0]),
            'embedding_dim': int(embeddings.shape[1]),
            'inertia': float(kmeans.inertia_),
            'timestamp': datetime.now().isoformat(),
            'top_10_clusters': [
                {'cluster_id': int(labels[i]), 'count': int(counts[i])}
                for i in np.argsort(counts)[::-1][:10]
            ]
        }, f, indent=2)
    
    print(f"✅ Saved summary: {summary_file}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ CODEBOOK TRAINING COMPLETE!")
    print("=" * 70)
    
    print(f"\n📊 Results:")
    print(f"   Codebook size: {N_CLUSTERS} structure tokens")
    print(f"   Utilization: {clusters_used}/{N_CLUSTERS} ({clusters_used/N_CLUSTERS*100:.1f}%)")
    print(f"   Training residues: {embeddings.shape[0]:,}")
    print(f"   Embedding dim: {embeddings.shape[1]}")
    print(f"   Inertia: {kmeans.inertia_:.2f}")
    
    print(f"\n📁 Output files:")
    print(f"   Codebook: {codebook_file.name}")
    print(f"   Summary: {summary_file.name}")
    print(f"   Clustering viz: {viz_file.name}")
    print(f"   Centroid plot: {centroid_file.name} ⭐")
    
    print(f"\n🚀 Next step:")
    print(f"   python 04_tokenize_and_demo.py")


if __name__ == '__main__':
    main()

