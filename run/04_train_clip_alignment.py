#!/usr/bin/env python3
"""
Tri-Modal Alignment Training - Sequence + Structure to Llama-3.1
Aligns ESM-2 sequence embeddings and interleaved structure tokens with Llama-3.1 text embeddings.
Uses pre-extracted triplet embeddings from script 03.

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
import os
import wandb
warnings.filterwarnings('ignore')

# Add root directory to Python path for imports when running from root
script_dir = Path(__file__).parent
root_dir = script_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Import model configuration
from run.config import get_model_config, list_available_models

# Embedding dimensions
ESM_HIDDEN_DIM = 2560  # ESM-2 3B
# Model hidden dim will be set from config

class TripletEmbeddingDataset(Dataset):
    """Load pre-extracted triplet embeddings from script 03"""
    
    def __init__(self, triplets: list):
        """
        Args:
            triplets: List of dicts with:
                - sequence_embedding: ESM-2 3B embedding (2560-dim)
                - structure_tokens: Interleaved tokens [BOS, AA, Struct, AA, Struct, ..., EOS]
                - text_embedding: Model-specific text embedding (varies by model)
        """
        self.triplets = triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        
        return {
            'sequence_emb': torch.tensor(triplet['sequence_embedding'], dtype=torch.float32),
            'structure_tokens': torch.tensor(triplet['structure_tokens'], dtype=torch.long),
            'text_emb': torch.tensor(triplet['text_embedding'], dtype=torch.float32),
            'protein_id': triplet.get('protein_id', f'p_{idx}')
        }


def load_batch_file(batch_file: Path) -> list:
    """
    Load a single batch file (NPZ compressed format from script 03).
    
    NPZ file structure:
    - sequence_embeddings: [n, 2560] (float32) - ESM-2 3B embeddings
    - text_embeddings: [n, text_dim] (float32) - Model-specific embeddings
    - structure_tokens: [n, max_len] (int16) - Interleaved tokens
    - protein_ids: [n] (object) - Protein identifiers
    - protein_indices: [n] (int32) - Original protein indices
    
    Args:
        batch_file: Path to triplet_embeddings_K{k}_batch_*.npz
    
    Returns:
        triplets: List of dicts with numpy arrays
    """
    # Load NPZ file
    data = np.load(batch_file, allow_pickle=True)
    
    # Extract arrays
    seq_embeddings = data['sequence_embeddings']      # [n, 2560] float32
    text_embeddings = data['text_embeddings']         # [n, text_dim] float32
    structure_tokens = data['structure_tokens']       # [n, max_len] int16
    protein_ids = data['protein_ids']                 # [n] object (strings)
    
    # Reconstruct triplets
    triplets = []
    for i in range(len(seq_embeddings)):
        triplet = {
            'sequence_embedding': seq_embeddings[i],       # numpy [2560] float32
            'text_embedding': text_embeddings[i],          # numpy [text_dim] float32
            'structure_tokens': structure_tokens[i],       # numpy [max_len] int16
            'protein_id': str(protein_ids[i])
        }
        triplets.append(triplet)
    
    return triplets


def load_all_batch_files(batch_files: list) -> Tuple[list, list]:
    """
    Load all batch files and combine into single train/val split.
    
    Args:
        batch_files: List of paths to triplet_embeddings_K{k}_batch_*.npz files
    
    Returns:
        (train_triplets, val_triplets): Combined lists of all triplets
    """
    print("\n📦 Loading all batch files into memory...")
    all_triplets = []
    
    for batch_file in tqdm(batch_files, desc="Loading files"):
        try:
            triplets = load_batch_file(batch_file)
            all_triplets.extend(triplets)
        except Exception as e:
            print(f"⚠️  Error loading {batch_file.name}: {e}")
            continue
    
    print(f"✅ Loaded {len(all_triplets):,} total samples")
    
    # Shuffle all triplets
    print("🔀 Shuffling dataset...")
    shuffled_indices = np.arange(len(all_triplets))
    np.random.shuffle(shuffled_indices)
    all_triplets = [all_triplets[i] for i in shuffled_indices]
    
    # Split into train/val (90/10)
    n_val = int(len(all_triplets) * 0.1)
    val_triplets = all_triplets[:n_val]
    train_triplets = all_triplets[n_val:]
    
    print(f"   Train: {len(train_triplets):,} samples")
    print(f"   Val:   {len(val_triplets):,} samples")
    
    return train_triplets, val_triplets


class ProteinProjectionHead(nn.Module):
    """Project protein embeddings to model space"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 2048):
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
    """Project structure tokens to model space"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256, output_dim: int = 4096, hidden_dim: int = 2048):
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
    """Project text embeddings to shared space"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        # Simple linear to ensure correct dimension
        self.proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
    
    def forward(self, x):
        proj = self.proj(x)
        return F.normalize(proj, p=2, dim=-1)

class TriModalAlignmentModel(nn.Module):
    """Tri-modal alignment: Sequence + Structure ↔ Text"""
    
    def __init__(self, 
                 structure_vocab_size: int,          # AA (20) + Struct (k) + Special (5)
                 sequence_dim: int = ESM_HIDDEN_DIM,
                 text_dim: int = 4096,
                 shared_dim: int = 4096,
                 temperature: float = 0.07):
        super().__init__()
        
        self.temperature = temperature
        self.shared_dim = shared_dim
        
        # Three projection heads
        self.sequence_proj = ProteinProjectionHead(sequence_dim, shared_dim)
        self.structure_proj = StructureProjectionHead(structure_vocab_size, embedding_dim=256, output_dim=shared_dim)
        self.text_proj = TextProjectionHead(text_dim, shared_dim)
    
    def forward(self, sequence_emb, structure_tokens, text_emb):
        """
        Args:
            sequence_emb: [batch, 2560] (ESM-2 3B sequence embedding)
            structure_tokens: [batch, seq_len] (interleaved tokens)
            text_emb: [batch, text_dim] (Model-specific text embedding)
        
        Returns:
            seq_proj: [batch, shared_dim] (normalized)
            struct_proj: [batch, shared_dim] (normalized)
            text_proj: [batch, shared_dim] (normalized)
        """
        seq_proj = self.sequence_proj(sequence_emb)
        struct_proj = self.structure_proj(structure_tokens)
        text_proj = self.text_proj(text_emb)
        
        return seq_proj, struct_proj, text_proj

def clip_contrastive_loss(seq_proj, struct_proj, text_proj, temperature=0.07, alpha=1.0, beta=1.0, lambda_cons=0.1):
    """
    Tri-contrastive loss.
    
    Args:
        seq_proj: [batch, shared_dim] (sequence projected)
        struct_proj: [batch, shared_dim] (structure projected)
        text_proj: [batch, shared_dim] (text projected)
        temperature: Temperature for scaling
        alpha: Weight for seq-text loss
        beta: Weight for struct-text loss
        lambda_cons: Weight for consistency term
    
    Returns:
        loss: scalar
        loss_dict: dict with individual losses
    """
    batch_size = seq_proj.shape[0]
    labels = torch.arange(batch_size, device=seq_proj.device)
    
    # === Pairwise InfoNCE Loss (seq-text) ===
    logits_seq_text = torch.mm(seq_proj, text_proj.t()) / temperature
    loss_seq_text = F.cross_entropy(logits_seq_text, labels)
    loss_text_seq = F.cross_entropy(logits_seq_text.t(), labels)
    loss_seq_text_total = (loss_seq_text + loss_text_seq) / 2
    
    # === Pairwise InfoNCE Loss (struct-text) ===
    logits_struct_text = torch.mm(struct_proj, text_proj.t()) / temperature
    loss_struct_text = F.cross_entropy(logits_struct_text, labels)
    loss_text_struct = F.cross_entropy(logits_struct_text.t(), labels)
    loss_struct_text_total = (loss_struct_text + loss_text_struct) / 2
    
    # === Consistency Term (seq-struct-text) ===
    consistency_loss = (
        torch.norm(seq_proj - struct_proj, p=2, dim=1).pow(2).mean() +
        torch.norm(seq_proj - text_proj, p=2, dim=1).pow(2).mean() +
        torch.norm(struct_proj - text_proj, p=2, dim=1).pow(2).mean()
    )
    
    # Total loss
    total_loss = alpha * loss_seq_text_total + beta * loss_struct_text_total + lambda_cons * consistency_loss
    
    return total_loss, {
        'seq_text': loss_seq_text_total.item(),
        'struct_text': loss_struct_text_total.item(),
        'consistency': consistency_loss.item()
    }

class Trainer:
    """Training loop"""
    
    def __init__(self, model, device="cuda", lr=1e-4, alpha=1.0, beta=1.0, lambda_cons=0.1, 
                 patience=5, min_delta=1e-4):
        """
        Args:
            patience: Number of epochs without improvement before early stopping
            min_delta: Minimum change in validation loss to qualify as improvement
        """
        self.model = model.to(device)
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.lambda_cons = lambda_cons
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10
        )
        
        self.history = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')
        self.step = 0
        
        # Early stopping
        self.patience = patience
        self.min_delta = min_delta
        self.patience_counter = 0
        self.early_stop = False
    
    def train_epoch(self, loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        batch_losses = []
        
        pbar = tqdm(loader, desc="Training")
        for batch in pbar:
            sequence_emb = batch['sequence_emb'].to(self.device)
            structure_tokens = batch['structure_tokens'].to(self.device)
            text_emb = batch['text_emb'].to(self.device)
            
            # Forward
            seq_proj, struct_proj, text_proj = self.model(sequence_emb, structure_tokens, text_emb)
            
            # Loss (tri-contrastive)
            loss, loss_dict = clip_contrastive_loss(
                seq_proj, struct_proj, text_proj,
                temperature=self.model.temperature,
                alpha=self.alpha, beta=self.beta, lambda_cons=self.lambda_cons
            )
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            batch_losses.append(loss.item())
            self.step += 1
            
            # Log to wandb (every batch)
            wandb.log({
                'train/batch_loss': loss.item(),
                'train/seq_text_loss': loss_dict['seq_text'],
                'train/struct_text_loss': loss_dict['struct_text'],
                'train/consistency_loss': loss_dict['consistency'],
                'train/step': self.step,
            })
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(loader)
        self.history['train_loss'].append(avg_loss)
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self, loader):
        """Validate"""
        self.model.eval()
        total_loss = 0
        batch_losses = []
        
        pbar = tqdm(loader, desc="Validation")
        for batch in pbar:
            sequence_emb = batch['sequence_emb'].to(self.device)
            structure_tokens = batch['structure_tokens'].to(self.device)
            text_emb = batch['text_emb'].to(self.device)
            
            seq_proj, struct_proj, text_proj = self.model(sequence_emb, structure_tokens, text_emb)
            loss, loss_dict = clip_contrastive_loss(
                seq_proj, struct_proj, text_proj,
                temperature=self.model.temperature,
                alpha=self.alpha, beta=self.beta, lambda_cons=self.lambda_cons
            )
            
            total_loss += loss.item()
            batch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(loader)
        self.history['val_loss'].append(avg_loss)
        
        return avg_loss
    
    def check_early_stopping(self, val_loss, output_dir, k_clusters):
        """
        Check if training should stop early based on validation loss.
        Returns True if training should stop.
        """
        # Check if validation loss improved
        if val_loss < (self.best_val_loss - self.min_delta):
            self.best_val_loss = val_loss
            self.patience_counter = 0
            
            # Save best model
            checkpoint_path = output_dir / f'best_model_K{k_clusters}.pt'
            self.save_checkpoint(checkpoint_path)
            print(f"   🎯 New best val loss: {val_loss:.4f}")
            
            return False
        else:
            self.patience_counter += 1
            print(f"   ⚠️  No improvement for {self.patience_counter}/{self.patience} epochs")
            
            if self.patience_counter >= self.patience:
                print(f"   🛑 Early stopping triggered! Val loss hasn't improved for {self.patience} epochs")
                self.early_stop = True
                return True
            
            return False
    
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
    print("TRI-MODAL ALIGNMENT: Sequence + Structure ↔ Text")
    print("=" * 70)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train tri-modal alignment model'
    )
    parser.add_argument('--model', type=str, required=True,
                       choices=list_available_models(),
                       help=f'Model type: {", ".join(list_available_models())}')
    parser.add_argument('--k', type=int, default=128,
                       help='Number of k-means clusters for structure tokens (default: 128)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=2048,
                       help='Batch size (default: 2048)')
    
    args = parser.parse_args()
    
    # Get model configuration
    model_config = get_model_config(args.model)
    k_clusters = args.k
    MODEL_HIDDEN_DIM = model_config.hidden_dim
    
    # Fixed hyperparameters
    learning_rate = 1e-4
    alpha = 1.0
    beta = 1.0
    lambda_cons = 0.1
    patience = 5
    
    # Login to wandb with your API key (for shared resources)
    wandb_api_key = os.getenv('WANDB_API_KEY')
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
        print("🔑 Logged in to wandb with custom API key")
    else:
        raise ValueError("❌ WANDB_API_KEY environment variable not set. Please set it to log in to wandb.")
    
    # Initialize wandb
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb_run_name = f"run-tri-{args.model}-{timestamp}"
    
    wandb.init(
        project='prottex-clip-alignment',
        name=wandb_run_name,
        config={
            'k_clusters': k_clusters,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': learning_rate,
            'alpha': alpha,
            'beta': beta,
            'lambda_cons': lambda_cons,
            'patience': patience,
            'sequence_dim': ESM_HIDDEN_DIM,
            'text_dim': MODEL_HIDDEN_DIM,
            'shared_dim': MODEL_HIDDEN_DIM,
            'model': model_config.model_name,
            'model_type': args.model,
            'temperature': 0.07,
        }
    )
    
    print(f"\n🔗 Wandb: prottex-clip-alignment / {wandb_run_name}")
    
    # Calculate vocabulary size dynamically
    vocab_size = 20 + k_clusters + 5
    
    print(f"\n📌 Model: {args.model} ({model_config.model_name})")
    print(f"📌 Text embedding dimension: {MODEL_HIDDEN_DIM}")
    print(f"📌 K-means clusters: {k_clusters}")
    print(f"📌 Vocabulary size: {vocab_size} (20 AA + {k_clusters} clusters + 5 special tokens)")
    print(f"📌 Epochs: {args.epochs}")
    print(f"📌 Batch size: {args.batch_size}")
    print(f"📌 Learning rate: {learning_rate}")
    print(f"📌 Loss weights: α={alpha}, β={beta}, λ={lambda_cons}")
    print(f"📌 Early stopping patience: {patience} epochs")
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use model-specific triplet directory
    data_dir = Path(f'data/triplet_embeddings/{args.model}')
    # Use model-specific output directory
    output_dir = Path(f'data/clip_alignment/{args.model}_K{k_clusters}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Data: {data_dir}")
    print(f"💾 Output: {output_dir}")
    print(f"🖥️  Device: {device}")
    
    # Load data
    print("\n" + "=" * 70)
    print("STEP 1: Find Batch Files")
    print("=" * 70)
    
    try:
        batch_files = sorted(data_dir.glob(f"triplet_embeddings_K{k_clusters}_batch_*.npz"))
        
        if not batch_files:
            print(f"❌ No NPZ batch files found in {data_dir} for K={k_clusters}")
            print(f"   Please run script 03 to generate triplet embeddings first")
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
        structure_vocab_size=vocab_size,
        sequence_dim=ESM_HIDDEN_DIM,
        text_dim=MODEL_HIDDEN_DIM,
        shared_dim=MODEL_HIDDEN_DIM,
        temperature=0.07
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✅ Model created")
    print(f"   Parameters: {total_params:,}")
    print(f"   Sequence projection: {ESM_HIDDEN_DIM} → {MODEL_HIDDEN_DIM}")
    print(f"   Structure projection: Tokens ({vocab_size}) → {MODEL_HIDDEN_DIM}")
    print(f"      ├─ AA tokens: 0-19 (20 types)")
    print(f"      ├─ Structure clusters: 20-{19+k_clusters} ({k_clusters} clusters)")
    print(f"      └─ Special tokens: {20+k_clusters}-{20+k_clusters+4} (5 tokens)")
    print(f"   Text projection: {MODEL_HIDDEN_DIM} → {MODEL_HIDDEN_DIM}")
    print(f"   Shared space: {MODEL_HIDDEN_DIM} dims ({args.model})")
    
    # Train
    print("\n" + "=" * 70)
    print("STEP 3: Training (Global Epoch Loop)")
    print("=" * 70)
    
    trainer = Trainer(
        model, device=device, lr=learning_rate,
        alpha=alpha, beta=beta, lambda_cons=lambda_cons,
        patience=patience
    )
    
    print(f"\nTraining config:")
    print(f"   Global epochs: {args.epochs}")
    print(f"   Batch files: {len(batch_files)}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Loss: Tri-Contrastive (CLIP-style)\n")
    
    # Global epoch loop
    for global_epoch in range(args.epochs):
        print(f"\n{'='*70}")
        print(f"GLOBAL EPOCH {global_epoch + 1}/{args.epochs}")
        print(f"{'='*70}")
        
        # Load all batch files at the start of each epoch
        train_triplets, val_triplets = load_all_batch_files(batch_files)
        
        # Create datasets and loaders
        train_dataset = TripletEmbeddingDataset(train_triplets)
        val_dataset = TripletEmbeddingDataset(val_triplets)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        print(f"\n📊 Epoch {global_epoch + 1} - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # Train one full epoch
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)
        
        # Epoch summary
        avg_train_loss = train_loss
        avg_val_loss = val_loss
        
        print(f"\n📊 Global Epoch {global_epoch + 1} Summary:")
        print(f"   Avg train loss: {avg_train_loss:.4f}")
        print(f"   Avg val loss: {avg_val_loss:.4f}")
        
        # Log epoch metrics to wandb
        current_lr = trainer.optimizer.param_groups[0]['lr']
        wandb.log({
            'epoch': global_epoch + 1,
            'train/epoch_loss': avg_train_loss,
            'val/epoch_loss': avg_val_loss,
            'learning_rate': current_lr,
            'best_val_loss': trainer.best_val_loss,
        })
        
        # Check early stopping
        if trainer.check_early_stopping(avg_val_loss, output_dir, k_clusters):
            print(f"\n🛑 Early stopping at epoch {global_epoch + 1}")
            break
        
        trainer.scheduler.step()
    
    # Save final model
    print("\n" + "=" * 70)
    print("STEP 4: Save Results")
    print("=" * 70)
    
    model_path = output_dir / f'final_model_K{k_clusters}.pt'
    trainer.save_checkpoint(model_path)
    
    config = {
        'k_clusters': k_clusters,
        'vocab_size': vocab_size,
        'sequence_dim': ESM_HIDDEN_DIM,
            'text_dim': MODEL_HIDDEN_DIM,
            'shared_dim': MODEL_HIDDEN_DIM,
            'model': model_config.model_name,
            'model_type': args.model,
        'temperature': 0.07,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': learning_rate,
        'loss_weights': {'alpha': alpha, 'beta': beta, 'lambda': lambda_cons},
        'timestamp': datetime.now().isoformat()
    }
    
    config_path = output_dir / f'config_K{k_clusters}.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Saved config: {config_path}")
    
    history_path = output_dir / f'training_history_K{k_clusters}.json'
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
    
    print(f"\n📁 Outputs in {output_dir}:")
    print(f"   final_model_K{k_clusters}.pt - Trained projection heads")
    print(f"   best_model_K{k_clusters}.pt - Best checkpoint")
    print(f"   config_K{k_clusters}.json - Training configuration")
    print(f"   training_history_K{k_clusters}.json - Loss curves")
    
    # Close wandb
    wandb.finish()


if __name__ == '__main__':
    main()
