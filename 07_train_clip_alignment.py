#!/usr/bin/env python3
"""
CLIP Alignment Training - Sequence to Text
Aligns ESM-2 sequence embeddings with BioGPT text embeddings.

Based on plan:
1. Project embeddings to shared space
2. Contrastive loss (InfoNCE / CLIP-style)
3. Temperature-scaled similarity
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
warnings.filterwarnings('ignore')

import wandb


class EmbeddingPairDataset(Dataset):
    """Load pre-extracted embedding pairs from a single JSON file"""
    
    def __init__(self, pairs: list):
        """
        Args:
            pairs: List of embedding pair dicts
        """
        self.pairs = pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        return {
            'sequence_emb': torch.tensor(pair['sequence_embedding'], dtype=torch.float32),
            'text_emb': torch.tensor(pair['text_embedding'], dtype=torch.float32),
            'protein_id': pair.get('protein_id', f'p_{idx}')
        }


def load_batch_file(batch_file: Path) -> Tuple[list, list]:
    """
    Load a single batch file and split into train/val
    
    Args:
        batch_file: Path to embedding_pairs_batch_*.json
    
    Returns:
        (train_pairs, val_pairs)
    """
    with open(batch_file) as f:
        pairs = json.load(f)
    
    # Shuffle
    shuffled_indices = np.arange(len(pairs))
    np.random.shuffle(shuffled_indices)
    pairs = [pairs[i] for i in shuffled_indices]
    
    # Split 90/10
    n_val = int(len(pairs) * 0.1)
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    
    return train_pairs, val_pairs


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
        # Project to text dimension and normalize
        proj = self.proj(x)
        return F.normalize(proj, p=2, dim=-1)


class CLIPAlignmentModel(nn.Module):
    """CLIP-style alignment model - aligns protein embeddings to text space"""
    
    def __init__(self, protein_dim: int = 1280, text_dim: int = 1600, temperature: float = 0.07):
        super().__init__()
        
        self.temperature = temperature
        self.text_dim = text_dim
        
        # Only protein needs projection (to text dimension)
        self.protein_proj = ProteinProjectionHead(protein_dim, text_dim)
        
        # Temperature parameter (learnable)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
    
    def forward(self, protein_emb, text_emb):
        """
        Args:
            protein_emb: [batch, protein_dim] (e.g., 1280-dim ESM-2)
            text_emb: [batch, text_dim] (e.g., 1600-dim BioGPT - already in text space)
        
        Returns:
            protein_proj: [batch, text_dim] (normalized, projected to text space)
            text_proj: [batch, text_dim] (normalized, no projection needed)
        """
        # Project protein to text space
        protein_proj = self.protein_proj(protein_emb)      # [batch, text_dim]
        
        # Text already in correct space - just normalize
        text_proj = F.normalize(text_emb, p=2, dim=-1)     # [batch, text_dim]
        
        return protein_proj, text_proj


def clip_contrastive_loss(protein_proj, text_proj, temperature=0.07):
    """
    InfoNCE / CLIP-style contrastive loss.
    
    Args:
        protein_proj: [batch, text_dim] (protein projected to text space)
        text_proj: [batch, text_dim] (normalized text)
        temperature: Temperature parameter
    
    Returns:
        loss: scalar
    """
    batch_size = protein_proj.shape[0]
    
    # Compute similarity matrix [batch, batch]
    # logits[i,j] = protein_i · text_j
    logits = torch.mm(protein_proj, text_proj.t()) / temperature
    
    # Labels: diagonal elements (i,i) should be high
    labels = torch.arange(batch_size, device=protein_proj.device)
    
    # Loss: protein -> text and text -> protein
    loss_protein2text = F.cross_entropy(logits, labels)
    loss_text2protein = F.cross_entropy(logits.t(), labels)
    
    return (loss_protein2text + loss_text2protein) / 2


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
            protein_emb = batch['sequence_emb'].to(self.device)
            text_emb = batch['text_emb'].to(self.device)
            
            # Forward
            protein_proj, text_proj = self.model(protein_emb, text_emb)
            
            # Loss
            loss = clip_contrastive_loss(protein_proj, text_proj, self.model.temperature)
            
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
            protein_emb = batch['sequence_emb'].to(self.device)
            text_emb = batch['text_emb'].to(self.device)
            
            protein_proj, text_proj = self.model(protein_emb, text_emb)
            loss = clip_contrastive_loss(protein_proj, text_proj, self.model.temperature)
            
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
    print("CLIP ALIGNMENT: Sequence ↔ Text")
    print("=" * 70)
    
    # Initialize wandb
    wandb.init(
        project="prottex-clip-alignment",
        name="run-{}".format(datetime.now().strftime("%Y%m%d_%H%M%S")),
        config={
            'model': 'CLIPAlignmentModel',
            'protein_encoder': 'ESM-2 (1280-dim)',
            'text_encoder': 'BioGPT-Large (1600-dim)',
            'loss': 'InfoNCE'
        }
    )
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path('esmfold_tokenizer/data/embedding_pairs')
    output_dir = Path('results/clip_alignment')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Data: {data_dir}")
    print(f"💾 Output: {output_dir}")
    print(f"🖥️  Device: {device}")
    
    # Config
    config = {
        'protein_dim': 1280,    # ESM-2
        'text_dim': 1600,       # BioGPT-Large
        'temperature': 0.07,
        'batch_size': 32,
        'epochs': 20,
        'learning_rate': 1e-4,
        'train_test_split': 0.1
    }
    
    # Load data
    print("\n" + "=" * 70)
    print("STEP 1: Find Batch Files")
    print("=" * 70)
    
    try:
        # Get all batch files
        data_dir_pairs = data_dir / 'embedding_pairs' if (data_dir / 'embedding_pairs').exists() else data_dir
        batch_files = sorted(data_dir_pairs.glob("embedding_pairs_batch_*.json"))
        
        if not batch_files:
            print(f"❌ No batch files found in {data_dir_pairs}")
            return
        
        print(f"✅ Found {len(batch_files)} batch files")
        print(f"   Will process: {batch_files[0].name} → {batch_files[-1].name}")
    except Exception as e:
        print(f"❌ Error finding batch files: {e}")
        return
    
    # Create model
    print("\n" + "=" * 70)
    print("STEP 2: Create Model")
    print("=" * 70)
    
    model = CLIPAlignmentModel(
        protein_dim=config['protein_dim'],
        text_dim=config['text_dim'],
        temperature=config['temperature']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✅ Model created")
    print(f"   Parameters: {total_params:,}")
    print(f"   Protein projection: {config['protein_dim']} → {config['text_dim']}")
    print(f"   Text: {config['text_dim']} (no projection, just normalize)")
    print(f"   Shared space: {config['text_dim']} dims")
    print(f"   Temperature: {config['temperature']}")
    
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
    print(f"   Protein (ESM-2):      1280-dim → {config['text_dim']}-dim (projected + normalized)")
    print(f"   Text (BioGPT):        {config['text_dim']}-dim (normalized, no projection)")
    print(f"   Shared space:         {config['text_dim']} dimensions (text space)")
    
    print(f"\n🔬 Training Strategy:")
    print(f"   Loss: InfoNCE (CLIP-style) - symmetric contrastive")
    print(f"   Aligns: Protein embeddings ↔ Text embeddings in text space")
    print(f"   Temperature: {config['temperature']}")
    print(f"   Freezes: Pre-trained ESM-2 and BioGPT encoders")
    print(f"   Trains: Only protein projection head to text space")
    
    print(f"\n📁 Outputs in {output_dir}:")
    print(f"   final_model.pt - Trained protein projection head")
    print(f"   config.json - Training configuration")
    print(f"   training_history.json - Loss curves")
    print(f"   best_model_epoch_*.pt - Best checkpoints")


if __name__ == '__main__':
    main()
