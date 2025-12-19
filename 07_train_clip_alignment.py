#!/usr/bin/env python3
"""
Tri-Modal Alignment Training - Sequence + Structure to Llama-3.1
Aligns ESM-2 sequence embeddings and interleaved structure tokens with Llama-3.1 text embeddings.

Based on tri-contrastive loss (A2 + B1):
- Sequence ↔ Text
- Structure ↔ Text  
- Sequence ↔ Structure (consistency term)
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, Tuple
import warnings
import sys
import argparse
warnings.filterwarnings('ignore')

import wandb

# Llama-3.1 hidden size
LLAMA_HIDDEN_DIM = 4096


class EmbeddingPairDataset(Dataset):
    """Load pre-extracted embedding pairs with interleaved structure tokens"""
    
    def __init__(self, pairs: list):
        """
        Args:
            pairs: List of dicts with:
                - sequence_embedding: ESM-2 embedding (1280-dim)
                - structure_tokens: Interleaved tokens [BOS, AA, Struct, AA, Struct, ..., EOS]
                - text_embedding: Llama-3.1 text embedding (4096-dim)
        """
        self.pairs = pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        return {
            'sequence_emb': torch.tensor(pair['sequence_embedding'], dtype=torch.float32),
            'structure_tokens': torch.tensor(pair['structure_tokens'], dtype=torch.long),
            'text_emb': torch.tensor(pair['text_embedding'], dtype=torch.float32),
            'protein_id': pair.get('protein_id', f'p_{idx}')
        }


def load_batch_file(batch_file: Path) -> Tuple[list, list]:
    """
    Load a single batch file (NPZ compressed format from script 06).
    Keeps embeddings as numpy arrays throughout (no intermediate conversions).
    
    NPZ file structure (from script 06):
    - sequence_embeddings: [n, 1280] (float32) - ESM-2 embeddings
    - text_embeddings: [n, 4096] (float32) - Llama-3.1 embeddings
    - structure_tokens: [n] (object array of lists) - Interleaved tokens
    - protein_ids: [n] (strings) - Protein identifiers
    
    Args:
        batch_file: Path to embedding_pairs_batch_*.npz
    
    Returns:
        (train_pairs, val_pairs): Lists of dicts with numpy arrays
    """
    # === STEP 1: Load NPZ file ===
    data = np.load(batch_file, allow_pickle=True)
    
    # Extract arrays (keep as numpy - no .tolist() conversions)
    seq_embeddings = data['sequence_embeddings']      # [n, 1280] float32
    text_embeddings = data['text_embeddings']         # [n, 4096] float32
    protein_ids = data['protein_ids']                 # [n] object (strings)
    structure_tokens_obj = data['structure_tokens']   # [n] object (lists of token IDs)
    
    # Reconstruct pairs with numpy arrays (single conversion point: numpy→tensor in DataLoader)
    pairs = []
    for i in range(len(seq_embeddings)):
        # Structure tokens are stored as object array - convert to numpy array for consistency
        struct_tokens_list = structure_tokens_obj[i]  # list of token IDs
        struct_tokens_arr = np.array(struct_tokens_list, dtype=np.int64)  # [seq_len]
        
        pair = {
            'sequence_embedding': seq_embeddings[i],       # numpy [1280] float32
            'text_embedding': text_embeddings[i],          # numpy [4096] float32
            'structure_tokens': struct_tokens_arr,         # numpy [seq_len] int64
            'protein_id': str(protein_ids[i])
        }
        pairs.append(pair)
    
    # === STEP 2: Shuffle ===
    shuffled_indices = np.arange(len(pairs))
    np.random.shuffle(shuffled_indices)
    pairs = [pairs[i] for i in shuffled_indices]

    # === STEP 3: Split into train/val ===
    n_val = int(len(pairs) * 0.1)
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    
    return train_pairs, val_pairs


class ProteinProjectionHead(nn.Module):
    """Project protein embeddings to Llama-3.1 space (4096-dim)"""
    
    def __init__(self, input_dim: int, output_dim: int = LLAMA_HIDDEN_DIM, hidden_dim: int = 2048):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        # Project and normalize
        proj = self.proj(x)
        return F.normalize(proj, p=2, dim=-1)


class StructureProjectionHead(nn.Module):
    """Project structure tokens to Llama-3.1 space (4096-dim)"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256, output_dim: int = LLAMA_HIDDEN_DIM, hidden_dim: int = 2048):
        super().__init__()
        # Token embedding for structure tokens
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        # Projection layers
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, token_ids):
        """
        Args:
            token_ids: [batch, seq_len] of token IDs
        
        Returns:
            structure_emb: [batch, output_dim] (mean-pooled and normalized)
        """
        # Embed tokens
        embedded = self.token_embedding(token_ids)  # [batch, seq_len, embedding_dim]
        
        # Mean pool over sequence
        pooled = embedded.mean(dim=1)  # [batch, embedding_dim]
        
        # Project to Llama space
        proj = self.proj(pooled)
        
        return F.normalize(proj, p=2, dim=-1)


class TextProjectionHead(nn.Module):
    """Project Llama-3.1 text embeddings to shared space (already in Llama space, just normalize)"""
    
    def __init__(self, input_dim: int = LLAMA_HIDDEN_DIM, output_dim: int = LLAMA_HIDDEN_DIM):
        super().__init__()
        # Simple linear to ensure correct dimension
        self.proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
    
    def forward(self, x):
        proj = self.proj(x)
        return F.normalize(proj, p=2, dim=-1)


class TriModalAlignmentModel(nn.Module):
    """Tri-modal alignment: Sequence + Structure ↔ Text (Llama-3.1)"""
    
    def __init__(self, 
                 structure_vocab_size: int,          # AA (20) + Struct (k) + Special (5), e.g., 153 or 537
                 sequence_dim: int = 1280,           # ESM-2
                 text_dim: int = LLAMA_HIDDEN_DIM,   # Llama-3.1
                 shared_dim: int = LLAMA_HIDDEN_DIM,
                 temperature: float = 0.07):
        super().__init__()
        
        self.temperature = temperature
        self.shared_dim = shared_dim
        
        # Three projection heads
        self.sequence_proj = ProteinProjectionHead(sequence_dim, shared_dim)
        self.structure_proj = StructureProjectionHead(structure_vocab_size, embedding_dim=256, output_dim=shared_dim)
        self.text_proj = TextProjectionHead(text_dim, shared_dim)
        
        # Temperature parameter (learnable)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
    
    def forward(self, sequence_emb, structure_tokens, text_emb):
        """
        Args:
            sequence_emb: [batch, 1280] (ESM-2 sequence embedding)
            structure_tokens: [batch, seq_len] (interleaved tokens)
            text_emb: [batch, 4096] (Llama-3.1 text embedding)
        
        Returns:
            seq_proj: [batch, shared_dim] (normalized)
            struct_proj: [batch, shared_dim] (normalized)
            text_proj: [batch, shared_dim] (normalized)
        """
        seq_proj = self.sequence_proj(sequence_emb)
        struct_proj = self.structure_proj(structure_tokens)
        text_proj = self.text_proj(text_emb)
        
        return seq_proj, struct_proj, text_proj


def clip_contrastive_loss(seq_proj, struct_proj, text_proj, temperature=0.07, alpha=1.0, beta=1.0, gamma=0.5, lambda_cons=0.1):
    """
    Tri-contrastive loss.
    
    Args:
        seq_proj: [batch, shared_dim] (sequence projected)
        struct_proj: [batch, shared_dim] or None (structure projected)
        text_proj: [batch, shared_dim] (text projected)
        temperature: Temperature for scaling
        alpha: Weight for seq-text loss
        beta: Weight for struct-text loss
        gamma: Weight for struct-seq consistency loss
        lambda_cons: Weight for consistency term
    
    Returns:
        loss: scalar
    """
    batch_size = seq_proj.shape[0]
    
    # === Pairwise InfoNCE Loss (seq-text) ===
    logits_seq_text = torch.mm(seq_proj, text_proj.t()) / temperature
    labels = torch.arange(batch_size, device=seq_proj.device)
    loss_seq_text = F.cross_entropy(logits_seq_text, labels)
    loss_text_seq = F.cross_entropy(logits_seq_text.t(), labels)
    loss_seq_text_total = (loss_seq_text + loss_text_seq) / 2
    
    total_loss = alpha * loss_seq_text_total
    
    # === Pairwise InfoNCE Loss (struct-text) ===
    logits_struct_text = torch.mm(struct_proj, text_proj.t()) / temperature
    loss_struct_text = F.cross_entropy(logits_struct_text, labels)
    loss_text_struct = F.cross_entropy(logits_struct_text.t(), labels)
    loss_struct_text_total = (loss_struct_text + loss_text_struct) / 2
    
    total_loss += beta * loss_struct_text_total
    
    # === Consistency Term (seq-struct) ===
    # All three modalities should collapse to same point
    consistency_loss = (
        torch.norm(seq_proj - struct_proj, p=2, dim=1).pow(2).mean() +
        torch.norm(seq_proj - text_proj, p=2, dim=1).pow(2).mean() +
        torch.norm(struct_proj - text_proj, p=2, dim=1).pow(2).mean()
    )
    total_loss += lambda_cons * consistency_loss
    
    return total_loss


class Trainer:
    """Training loop"""
    
    def __init__(self, model, device="cuda", lr=1e-4, use_wandb=True):
        self.model = model.to(device)
        self.device = device
        self.use_wandb = use_wandb
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10
        )
        
        self.history = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')
        self.step = 0
    
    def train_epoch(self, loader, batch_idx=0):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(loader, desc="Training")
        for batch in pbar:
            sequence_emb = batch['sequence_emb'].to(self.device)
            structure_tokens = batch['structure_tokens'].to(self.device) if batch['structure_tokens'] is not None else None
            text_emb = batch['text_emb'].to(self.device)
            
            # Forward
            seq_proj, struct_proj, text_proj = self.model(sequence_emb, structure_tokens, text_emb)
            
            # Loss (tri-contrastive)
            loss = clip_contrastive_loss(
                seq_proj, struct_proj, text_proj,
                temperature=self.model.temperature,
                alpha=1.0, beta=1.0, gamma=0.5, lambda_cons=0.1
            )
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            self.step += 1
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'train_loss_batch': loss.item(),
                    'batch': batch_idx,
                    'step': self.step
                })
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(loader)
        self.history['train_loss'].append(avg_loss)
        
        # Log epoch avg
        if self.use_wandb:
            wandb.log({
                'train_loss_epoch': avg_loss,
                'batch': batch_idx
            })
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self, loader, batch_idx=0):
        """Validate"""
        self.model.eval()
        total_loss = 0
        
        pbar = tqdm(loader, desc="Validation")
        for batch in pbar:
            sequence_emb = batch['sequence_emb'].to(self.device)
            structure_tokens = batch['structure_tokens'].to(self.device)
            text_emb = batch['text_emb'].to(self.device)
            
            seq_proj, struct_proj, text_proj = self.model(sequence_emb, structure_tokens, text_emb)
            loss = clip_contrastive_loss(seq_proj, struct_proj, text_proj, temperature=self.model.temperature, alpha=1.0, beta=1.0, gamma=0.5, lambda_cons=0.1)
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(loader)
        self.history['val_loss'].append(avg_loss)
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                'val_loss': avg_loss,
                'batch': batch_idx,
                'step': self.step
            })
        
        return avg_loss
    
    def save_checkpoint(self, path):
        """Save model"""
        torch.save({
            'model_state': self.model.state_dict(),
            'history': self.history,
            'optimizer_state': self.optimizer.state_dict()
        }, path)
        print(f"✅ Saved: {path}")


def main():
    print("=" * 70)
    print("TRI-MODAL ALIGNMENT: Sequence + Structure ↔ Text (Llama-3.1)")
    print("=" * 70)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train tri-modal alignment model')
    parser.add_argument('subset_name', type=str, help='Dataset subset name (e.g., UniProt_Function)')
    parser.add_argument('--k', type=int, default=128, choices=[128, 512],
                       help='Number of k-means clusters for structure tokens (default: 128)')
    
    args = parser.parse_args()
    
    subset_name = args.subset_name
    k_clusters = args.k
    
    # Calculate vocabulary size dynamically
    # Vocab: 0-19 (AA), 20-(20+k-1) (clusters), (20+k)-(20+k+4) (special)
    vocab_size = 20 + k_clusters + 5
    
    print(f"\n📌 Subset: {subset_name}")
    print(f"📌 K-means clusters: {k_clusters}")
    print(f"📌 Vocabulary size: {vocab_size} (20 AA + {k_clusters} clusters + 5 special tokens)")
    
    # Initialize wandb
    wandb.init(
        project="prottex-clip-alignment",
        name="run-tri-{}".format(datetime.now().strftime("%Y%m%d_%H%M%S")),
        config={
            'model': 'TriModalAlignmentModel',
            'sequence_encoder': 'ESM-2 (1280-dim)',
            'structure_encoder': f'Interleaved Tokens ({vocab_size} vocab)',
            'text_encoder': 'Llama-3.1 (4096-dim)',
            'loss': 'Tri-Contrastive (A2+B1)',
            'weights': {'alpha': 1.0, 'beta': 1.0, 'gamma': 0.5, 'lambda': 0.1},
            'subset': subset_name
        }
    )
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path('esmfold_tokenizer/data') / subset_name / 'embedding_pairs'
    output_dir = Path('results/trimodal_alignment') / subset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Data: {data_dir}")
    print(f"💾 Output: {output_dir}")
    print(f"🖥️  Device: {device}")
    
    # Config
    config = {
        'sequence_dim': 1280,       # ESM-2
        'k_clusters': k_clusters,   # Number of structure clusters
        'structure_vocab': vocab_size,  # AA (20) + Struct (k) + Special (5)
        'text_dim': LLAMA_HIDDEN_DIM,  # Llama-3.1
        'shared_dim': LLAMA_HIDDEN_DIM,
        'temperature': 0.07,
        'batch_size': 1024,
        'epochs': 50,
        'learning_rate': 1e-4,
        'train_test_split': 0.1,
        'loss_weights': {'alpha': 1.0, 'beta': 1.0, 'gamma': 0.5, 'lambda': 0.1} # seq-text, struct-text, seq-struct, combined
    }
    
    # Load data
    print("\n" + "=" * 70)
    print("STEP 1: Find Batch Files")
    print("=" * 70)
    
    try:
        # Get all batch files
        data_dir_pairs = data_dir / 'embedding_pairs' if (data_dir / 'embedding_pairs').exists() else data_dir
        batch_files = sorted(data_dir_pairs.glob("embedding_pairs_batch_*.npz"))
        
        if not batch_files:
            print(f"❌ No NPZ batch files found in {data_dir_pairs}")
            print(f"   Please run script 06 to generate embedding pairs first")
            return
        
        print(f"✅ Found {len(batch_files)} batch files (NPZ format)")
        print(f"   Will process: {batch_files[0].name} → {batch_files[-1].name}")
    except Exception as e:
        print(f"❌ Error finding batch files: {e}")
        return
    
    # Create model
    print("\n" + "=" * 70)
    print("STEP 2: Create Model")
    print("=" * 70)
    
    model = TriModalAlignmentModel(
        structure_vocab_size=config['structure_vocab'],
        sequence_dim=config['sequence_dim'],
        text_dim=config['text_dim'],
        shared_dim=config['shared_dim'],
        temperature=config['temperature']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✅ Model created")
    print(f"   Parameters: {total_params:,}")
    print(f"   Sequence projection: {config['sequence_dim']} → {config['shared_dim']}")
    print(f"   Structure projection: Tokens ({config['structure_vocab']}) → {config['shared_dim']}")
    print(f"      ├─ AA tokens: 0-19 (20 types)")
    print(f"      ├─ Structure clusters: 20-{19+k_clusters} ({k_clusters} clusters)")
    print(f"      └─ Special tokens: {20+k_clusters}-{20+k_clusters+4} (5 tokens)")
    print(f"   Text projection: {config['text_dim']} → {config['shared_dim']}")
    print(f"   Shared space: {config['shared_dim']} dims (Llama-3.1)")
    print(f"   Temperature: {config['temperature']}")
    print(f"   Loss weights: α={config['loss_weights']['alpha']}, β={config['loss_weights']['beta']}, " + 
          f"γ={config['loss_weights']['gamma']}, λ={config['loss_weights']['lambda']}")
    
    # Train
    print("\n" + "=" * 70)
    print("STEP 3: Training (Global Epoch Loop)")
    print("=" * 70)
    
    trainer = Trainer(model, device=device, lr=config['learning_rate'])
    
    print(f"\nTraining config:")
    print(f"   Global epochs: {config['epochs']}")
    print(f"   Batch files per epoch: {len(batch_files)}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Learning rate: {config['learning_rate']}")
    print(f"   Loss: InfoNCE (CLIP-style contrastive)\n")
    
    # Global epoch loop
    for global_epoch in range(config['epochs']):
        print(f"\n{'='*70}")
        print(f"GLOBAL EPOCH {global_epoch + 1}/{config['epochs']}")
        print(f"{'='*70}")
        
        # Shuffle batch files each epoch
        shuffled_batch_files = batch_files.copy()
        np.random.shuffle(shuffled_batch_files)
        
        epoch_train_losses = []
        epoch_val_losses = []
        
        # Loop through all batch files in this epoch
        for batch_idx, batch_file in enumerate(tqdm(shuffled_batch_files, desc=f"Epoch {global_epoch+1}")):
            try:
                # Load and split this batch file
                train_pairs, val_pairs = load_batch_file(batch_file)
                
                train_dataset = EmbeddingPairDataset(train_pairs)
                val_dataset = EmbeddingPairDataset(val_pairs)
                
                train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

                # Train on this batch file
                train_loss = trainer.train_epoch(train_loader, batch_idx=batch_idx)
                val_loss = trainer.validate(val_loader, batch_idx=batch_idx)
                
                epoch_train_losses.append(train_loss)
                epoch_val_losses.append(val_loss)
                
                # Save best model across all batches and epochs
                if val_loss < trainer.best_val_loss:
                    trainer.best_val_loss = val_loss
                    checkpoint_path = output_dir / f'best_model_global_epoch_{global_epoch}_batch_{batch_idx}.pt'
                    trainer.save_checkpoint(checkpoint_path)
                    print(f"   🎯 New best val loss: {val_loss:.4f}")
                
            except Exception as e:
                print(f"⚠️  Error processing {batch_file.name}: {e}")
                continue
        
        # Epoch summary
        avg_train_loss = np.mean(epoch_train_losses)
        avg_val_loss = np.mean(epoch_val_losses)
        
        print(f"\n📊 Global Epoch {global_epoch + 1} Summary:")
        print(f"   Avg train loss: {avg_train_loss:.4f}")
        print(f"   Avg val loss: {avg_val_loss:.4f}")
        
        trainer.scheduler.step()
    
    # Save final model
    print("\n" + "=" * 70)
    print("STEP 4: Save Results")
    print("=" * 70)
    
    model_path = output_dir / 'final_model.pt'
    trainer.save_checkpoint(model_path)
    
    config_path = output_dir / 'config.json'
    config['timestamp'] = datetime.now().isoformat()
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Saved config: {config_path}")
    
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(trainer.history, f, indent=2)
    print(f"✅ Saved history: {history_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    
    print(f"\n📊 Final Results:")
    print(f"   Best train loss: {min(trainer.history['train_loss']):.4f}")
    print(f"   Best val loss: {min(trainer.history['val_loss']):.4f}")
    
    print(f"\n🏗️  Model Architecture:")
    print(f"   Sequence (ESM-2):     1280-dim → {config['shared_dim']}-dim (projected + normalized)")
    print(f"   Structure (Tokens):   {config['structure_vocab']} vocab → {config['shared_dim']}-dim (projected + normalized)")
    print(f"      ├─ K-means clusters: {config['k_clusters']}")
    print(f"      └─ Token range: [0-19 AA | 20-{19+config['k_clusters']} struct | {20+config['k_clusters']}-{24+config['k_clusters']} special]")
    print(f"   Text (Llama-3.1):     {config['text_dim']}-dim → {config['shared_dim']}-dim (projected + normalized)")
    print(f"   Shared space:         {config['shared_dim']} dimensions (Llama-3.1 hidden)")
    
    print(f"\n🔬 Training Strategy:")
    print(f"   Loss: Tri-Contrastive (A2+B1 from paper)")
    print(f"   Components:")
    print(f"      - Sequence ↔ Text (weight α={config['loss_weights']['alpha']})")
    print(f"      - Structure ↔ Text (weight β={config['loss_weights']['beta']})")
    print(f"      - Consistency term (weight λ={config['loss_weights']['lambda']})")
    print(f"   Temperature: {config['temperature']}")
    print(f"   Freezes: Pre-trained ESM-2 and Llama-3.1 encoders")
    print(f"   Trains: All projection heads")
    
    print(f"\n📁 Outputs in {output_dir}:")
    print(f"   final_model.pt - Trained projection heads")
    print(f"   config.json - Training configuration")
    print(f"   training_history.json - Loss curves")


if __name__ == '__main__':
    main()
