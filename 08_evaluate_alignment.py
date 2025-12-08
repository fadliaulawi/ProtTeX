#!/usr/bin/env python3
"""
Evaluate CLIP Alignment Model
Compare cosine similarity between protein and text embeddings before and after alignment.
Load pre-trained projection model from 07_train_clip_alignment.py
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


class ProteinProjectionHead(nn.Module):
    """Project protein embeddings to text space (1600-dim)"""
    
    def __init__(self, input_dim: int, output_dim: int = 1600, hidden_dim: int = 2048):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        proj = self.proj(x)
        return F.normalize(proj, p=2, dim=-1)


class CLIPAlignmentModel(nn.Module):
    """CLIP-style alignment model - aligns protein embeddings to text space"""
    
    def __init__(self, protein_dim: int = 1280, text_dim: int = 1600, temperature: float = 0.07):
        super().__init__()
        
        self.temperature = temperature
        self.text_dim = text_dim
        
        self.protein_proj = ProteinProjectionHead(protein_dim, text_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
    
    def forward(self, protein_emb, text_emb):
        protein_proj = self.protein_proj(protein_emb)
        text_proj = F.normalize(text_emb, p=2, dim=-1)
        
        return protein_proj, text_proj


def compute_cosine_similarities(protein_embs, text_embs):
    """
    Compute cosine similarity matrix between proteins and texts.
    Both must be in the same space (normalized).
    
    Args:
        protein_embs: [N, dim] numpy array (should be same dim as text_embs)
        text_embs: [N, dim] numpy array
    
    Returns:
        diag_similarities: [N] - diagonal similarities (matching pairs)
        all_similarities: [N, N] - full similarity matrix
    """
    # Convert to tensors and normalize
    protein_embs_t = F.normalize(torch.tensor(protein_embs, dtype=torch.float32), p=2, dim=-1)
    text_embs_t = F.normalize(torch.tensor(text_embs, dtype=torch.float32), p=2, dim=-1)
    
    # Both should have same embedding dimension
    assert protein_embs_t.shape[1] == text_embs_t.shape[1], \
        f"Dimension mismatch: protein={protein_embs_t.shape}, text={text_embs_t.shape}"
    
    # Compute full similarity matrix [N, N]
    all_similarities = torch.mm(protein_embs_t, text_embs_t.t())
    
    # Extract diagonal (matching pairs)
    diag_similarities = torch.diag(all_similarities)
    
    return diag_similarities.numpy(), all_similarities.numpy()


def main():
    print("=" * 70)
    print("EVALUATE CLIP ALIGNMENT MODEL")
    print("=" * 70)
    
    # Paths
    data_dir = Path('esmfold_tokenizer/data/embedding_pairs')
    model_dir = Path('results/clip_alignment')
    output_dir = Path('results/alignment_evaluation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Data directory: {data_dir}")
    print(f"📂 Model directory: {model_dir}")
    print(f"📂 Output directory: {output_dir}")
    
    # Load model
    print("\n" + "=" * 70)
    print("STEP 1: Load Trained Model")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    
    model_path = model_dir / 'final_model.pt'
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print(f"   Run 07_train_clip_alignment.py first!")
        return
    
    # Create model and load checkpoint
    model = CLIPAlignmentModel(protein_dim=1280, text_dim=1600)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from: {model_path}")
    
    # Load sample embeddings
    print("\n" + "=" * 70)
    print("STEP 2: Load Embedding Pairs")
    print("=" * 70)
    
    batch_files = sorted(data_dir.glob("embedding_pairs_batch_*.json"))
    
    if not batch_files:
        print(f"❌ No batch files found in {data_dir}")
        return
    
    print(f"✅ Found {len(batch_files)} batch files")
    
    # Process each batch file and compute metrics
    all_diag_after = []
    all_ranks_after = []

    batch_files = batch_files[:1]
    
    print(f"\n📊 Processing all batches and computing similarities...")
    for batch_idx, batch_file in enumerate(tqdm(batch_files, desc="Evaluating batches")):
        with open(batch_file) as f:
            pairs = json.load(f)
        
        # Extract embeddings for this batch
        batch_protein_embs = np.array([p['sequence_embedding'] for p in pairs], dtype=np.float32)
        batch_text_embs = np.array([p['text_embedding'] for p in pairs], dtype=np.float32)
        
        # After alignment (with trained projection)
        protein_embs_tensor = torch.tensor(batch_protein_embs, dtype=torch.float32).to(device)
        text_embs_tensor = torch.tensor(batch_text_embs, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            protein_proj, text_proj = model(protein_embs_tensor, text_embs_tensor)
        
        protein_proj = protein_proj.cpu().numpy()
        text_proj = text_proj.cpu().numpy()
        
        diag_after, matrix_after = compute_cosine_similarities(protein_proj, text_proj)
        all_diag_after.extend(diag_after)
        
        # Compute ranks for this batch
        for i in range(len(batch_protein_embs)):
            # Rank of diagonal element in row i
            row_after = matrix_after[i]
            rank_after = np.sum(row_after > row_after[i])
            all_ranks_after.append(rank_after)
    
    all_diag_after = np.array(all_diag_after)
    ranks_after = np.array(all_ranks_after)
    
    print(f"✅ Processed {len(all_diag_after)} total samples from all batches")
    
    # Compute metrics
    print("\n" + "=" * 70)
    print("STEP 3: Compute Metrics (After Alignment)")
    print("=" * 70)
    
    # Diagonal statistics
    print(f"\n🎯 Matching pair similarities:")
    print(f"   Mean:   {all_diag_after.mean():.4f}")
    print(f"   Median: {np.median(all_diag_after):.4f}")
    print(f"   Std:    {all_diag_after.std():.4f}")
    print(f"   Min:    {all_diag_after.min():.4f}")
    print(f"   Max:    {all_diag_after.max():.4f}")
    
    # Recall metrics
    print(f"\n📊 Recall metrics (is matching pair in top-K?):")
    recall_at_1 = 100 * np.sum(ranks_after == 0) / len(ranks_after)
    recall_at_5 = 100 * np.sum(ranks_after < 5) / len(ranks_after)
    recall_at_10 = 100 * np.sum(ranks_after < 10) / len(ranks_after)
    
    print(f"   Recall@1:  {recall_at_1:.2f}%")
    print(f"   Recall@5:  {recall_at_5:.2f}%")
    print(f"   Recall@10: {recall_at_10:.2f}%")
    print(f"   Mean rank: {ranks_after.mean():.1f}")
    
    results = {
        "num_samples": len(all_diag_after),
        "metrics": {
            "diagonal": {
                "mean": float(all_diag_after.mean()),
                "median": float(np.median(all_diag_after)),
                "std": float(all_diag_after.std()),
                "min": float(all_diag_after.min()),
                "max": float(all_diag_after.max())
            },
            "recall": {
                "recall_at_1": float(recall_at_1),
                "recall_at_5": float(recall_at_5),
                "recall_at_10": float(recall_at_10),
                "mean_rank": float(ranks_after.mean())
            }
        }
    }
    
    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved results: {results_file}")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Diagonal similarities
    axes[0, 0].hist(all_diag_after, bins=30, alpha=0.7, color='green')
    axes[0, 0].set_xlabel('Cosine Similarity')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Matching Pair Similarities (After Alignment)')
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Rank distribution
    axes[0, 1].hist(ranks_after, bins=50, alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Rank of Matching Pair')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Ranking Distribution (After Alignment)')
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Recall metrics
    recall_labels = ['Recall@1', 'Recall@5', 'Recall@10']
    recall_values = [recall_at_1, recall_at_5, recall_at_10]
    colors_recall = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    axes[1, 0].bar(recall_labels, recall_values, color=colors_recall, alpha=0.7)
    axes[1, 0].set_ylabel('Recall (%)')
    axes[1, 0].set_title('Recall Metrics')
    axes[1, 0].set_ylim([0, 100])
    axes[1, 0].grid(alpha=0.3, axis='y')
    for i, v in enumerate(recall_values):
        axes[1, 0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # 4. Cumulative recall curve
    sorted_ranks = np.sort(ranks_after)
    cumulative_recall = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks) * 100
    top_k = np.arange(len(sorted_ranks)) + 1
    axes[1, 1].plot(top_k, cumulative_recall, color='green', linewidth=2)
    axes[1, 1].axvline(1, color='red', linestyle='--', alpha=0.5, label='@1')
    axes[1, 1].axvline(5, color='orange', linestyle='--', alpha=0.5, label='@5')
    axes[1, 1].axvline(10, color='blue', linestyle='--', alpha=0.5, label='@10')
    axes[1, 1].set_xlabel('Top-K')
    axes[1, 1].set_ylabel('Cumulative Recall (%)')
    axes[1, 1].set_title('Cumulative Recall Curve')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_dir / 'alignment_evaluation.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved plot: {plot_file}")
    plt.close()
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\n📈 Retrieval Performance (After Alignment):")
    print(f"   • Recall@1:  {recall_at_1:.2f}%")
    print(f"   • Recall@5:  {recall_at_5:.2f}%")
    print(f"   • Recall@10: {recall_at_10:.2f}%")
    print(f"   • Mean rank: {ranks_after.mean():.1f}")
    print(f"   • Median similarity: {np.median(all_diag_after):.4f}")
    print(f"\n📁 Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
