#!/usr/bin/env python3
"""
Tri-Modal Alignment Training - Sequence + Structure to Text (Habdine)

Habdine version: pre-split data, hardcoded data/habdine paths.
- Input: data/habdine/triplet_embeddings/{model}/K{k}/triplet_embeddings_{train|validation}_batch_*.npz
- Metadata: triplet_metadata_{train|validation}_batch_*.json
- Output: data/habdine/clip_alignment/{model}_K{k}

Otherwise same as run/04: type-aware tri-contrastive loss (PFUD/PSAD: seq-text, struct-text, seq-struct, consistency; PDD/PSPD: seq-struct only).
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
from typing import Dict, Tuple, List
import warnings
import sys
import argparse
import os
import wandb
warnings.filterwarnings('ignore')

# Add root directory to Python path (run/habdine -> project root)
script_dir = Path(__file__).parent
root_dir = script_dir.parent.parent
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
                - protein_type: Type of protein (PFUD, PSAD, PDD, PSPD)
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
            'protein_id': triplet.get('protein_id', f'p_{idx}'),
            'protein_type': triplet.get('protein_type', 'PFUD')  # Default to PFUD
        }


def load_batch_file(batch_file: Path, metadata_file: Path = None) -> list:
    """
    Load a single batch file (NPZ compressed format from script 03) with metadata.
    
    NPZ file structure:
    - sequence_embeddings: [n, 2560] (float32) - ESM-2 3B embeddings
    - text_embeddings: [n, text_dim] (float32) - Model-specific embeddings
    - structure_tokens: [n, max_len] (int16) - Interleaved tokens
    - protein_ids: [n] (object) - Protein identifiers
    - protein_indices: [n] (int32) - Original protein indices
    
    Args:
        batch_file: Path to triplet_embeddings_*_batch_*.npz
        metadata_file: Path to triplet_metadata_{split}_batch_*.json (optional)
    
    Returns:
        triplets: List of dicts with numpy arrays and type information
    """
    # Load NPZ file
    data = np.load(batch_file, allow_pickle=True)
    
    # Extract arrays
    seq_embeddings = data['sequence_embeddings']      # [n, 2560] float32
    text_embeddings = data['text_embeddings']         # [n, text_dim] float32
    structure_tokens = data['structure_tokens']       # [n, max_len] int16
    protein_ids = data['protein_ids']                 # [n] object (strings)
    
    # Load metadata if available
    metadata_dict = {}
    if metadata_file and metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata_list = json.load(f)
            # Create mapping from protein_id to type
            for meta_entry in metadata_list:
                protein_id = meta_entry.get('metadata', {}).get('id', '')
                protein_type = meta_entry.get('metadata', {}).get('type', 'PFUD')
                metadata_dict[protein_id] = protein_type
        except Exception as e:
            print(f"⚠️  Warning: Could not load metadata from {metadata_file}: {e}")
    
    # Reconstruct triplets
    triplets = []
    for i in range(len(seq_embeddings)):
        protein_id = str(protein_ids[i])
        protein_type = metadata_dict.get(protein_id, 'PFUD')  # Default to PFUD if not found
        
        triplet = {
            'sequence_embedding': seq_embeddings[i],       # numpy [2560] float32
            'text_embedding': text_embeddings[i],          # numpy [text_dim] float32
            'structure_tokens': structure_tokens[i],       # numpy [max_len] int16
            'protein_id': protein_id,
            'protein_type': protein_type
        }
        triplets.append(triplet)
    
    return triplets


def load_pre_split_batch_files(data_dir: Path, split_name: str) -> List[dict]:
    """
    Load pre-split batch files for a given split (train or validation).
    Habdine: triplet_embeddings_{split}_batch_*.npz, triplet_metadata_{split}_batch_*.json
    
    Args:
        data_dir: Directory containing triplet_embeddings_{split}_batch_*.npz
        split_name: 'train' or 'validation'
    
    Returns:
        triplets: List of triplet dicts
    """
    batch_files = sorted(data_dir.glob(f"triplet_embeddings_{split_name}_batch_*.npz"))
    triplets = []
    for batch_file in tqdm(batch_files, desc=f"Loading {split_name}"):
        try:
            if 'batch_' in batch_file.stem:
                batch_part = batch_file.stem.split('batch_')[-1]
                metadata_file = data_dir / f'triplet_metadata_{split_name}_batch_{batch_part}.json'
            else:
                metadata_file = None
            triplets.extend(load_batch_file(batch_file, metadata_file))
        except Exception as e:
            print(f"⚠️  Error loading {batch_file.name}: {e}")
            continue
    # Type distribution
    type_counts = {}
    for t in triplets:
        p = t.get('protein_type', 'PFUD')
        type_counts[p] = type_counts.get(p, 0) + 1
    print(f"   {split_name}: {len(triplets):,} samples, types: {type_counts}")
    return triplets


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

def clip_contrastive_loss(seq_proj, struct_proj, text_proj, protein_types, temperature=0.07, 
                          alpha=1.0, beta=1.0, gamma=1.0, lambda_cons=0.1):
    """
    Tri-contrastive loss with type-aware handling.
    
    Args:
        seq_proj: [batch, shared_dim] (sequence projected)
        struct_proj: [batch, shared_dim] (structure projected)
        text_proj: [batch, shared_dim] (text projected)
        protein_types: [batch] (list of protein types: PFUD, PSAD, PDD, PSPD)
        temperature: Temperature for scaling
        alpha: Weight for seq-text loss (for PFUD/PSAD only)
        beta: Weight for struct-text loss (for PFUD/PSAD only)
        gamma: Weight for seq-struct loss
        lambda_cons: Weight for consistency term
    
    Returns:
        loss: scalar
        loss_dict: dict with individual losses
    """
    batch_size = seq_proj.shape[0]
    device = seq_proj.device
    labels = torch.arange(batch_size, device=device)
    
    # Create masks for different types
    # PFUD and PSAD: use text embeddings (seq-text, struct-text)
    # PDD and PSPD: only use seq-struct (no text)
    use_text_mask = torch.tensor([
        t in ['PFUD', 'PSAD'] for t in protein_types
    ], dtype=torch.bool, device=device)
    
    # Initialize losses
    loss_seq_text_total = torch.tensor(0.0, device=device)
    loss_struct_text_total = torch.tensor(0.0, device=device)
    loss_seq_struct_total = torch.tensor(0.0, device=device)
    consistency_loss = torch.tensor(0.0, device=device)
    
    n_text_samples = use_text_mask.sum().item()
    
    # === Pairwise InfoNCE Loss (seq-struct) - applies to ALL samples ===
    logits_seq_struct = torch.mm(seq_proj, struct_proj.t()) / temperature
    loss_seq_struct = F.cross_entropy(logits_seq_struct, labels)
    loss_struct_seq = F.cross_entropy(logits_seq_struct.t(), labels)
    loss_seq_struct_total = (loss_seq_struct + loss_struct_seq) / 2
    
    # === Pairwise InfoNCE Loss (seq-text) and (struct-text) - only for PFUD/PSAD ===
    if n_text_samples > 1:  # Need at least 2 samples for contrastive loss
        # Filter to samples that use text
        seq_proj_text = seq_proj[use_text_mask]
        struct_proj_text = struct_proj[use_text_mask]
        text_proj_text = text_proj[use_text_mask]
        labels_text = torch.arange(n_text_samples, device=device)
        
        # Seq-text InfoNCE loss
        logits_seq_text = torch.mm(seq_proj_text, text_proj_text.t()) / temperature
        loss_seq_text = F.cross_entropy(logits_seq_text, labels_text)
        loss_text_seq = F.cross_entropy(logits_seq_text.t(), labels_text)
        loss_seq_text_total = (loss_seq_text + loss_text_seq) / 2
        
        # Struct-text InfoNCE loss
        logits_struct_text = torch.mm(struct_proj_text, text_proj_text.t()) / temperature
        loss_struct_text = F.cross_entropy(logits_struct_text, labels_text)
        loss_text_struct = F.cross_entropy(logits_struct_text.t(), labels_text)
        loss_struct_text_total = (loss_struct_text + loss_text_struct) / 2
        
        # Consistency Term (for PFUD/PSAD with text)
        consistency_loss = (
            torch.norm(seq_proj_text - struct_proj_text, p=2, dim=1).pow(2).mean() +
            torch.norm(seq_proj_text - text_proj_text, p=2, dim=1).pow(2).mean() +
            torch.norm(struct_proj_text - text_proj_text, p=2, dim=1).pow(2).mean()
        )
    elif n_text_samples == 1:
        # Single text sample: can still compute consistency but not contrastive loss
        seq_proj_text = seq_proj[use_text_mask]
        struct_proj_text = struct_proj[use_text_mask]
        text_proj_text = text_proj[use_text_mask]
        consistency_loss = (
            torch.norm(seq_proj_text - struct_proj_text, p=2, dim=1).pow(2).mean() +
            torch.norm(seq_proj_text - text_proj_text, p=2, dim=1).pow(2).mean() +
            torch.norm(struct_proj_text - text_proj_text, p=2, dim=1).pow(2).mean()
        )
    else:
        # For seq-struct only batches (PDD/PSPD), use a simpler consistency term
        consistency_loss = torch.norm(seq_proj - struct_proj, p=2, dim=1).pow(2).mean()
    
    # Total loss - scale text losses by the fraction of samples that use them
    text_weight = n_text_samples / batch_size if batch_size > 0 else 0.0
    
    total_loss = (
        gamma * loss_seq_struct_total +  # Always apply seq-struct loss
        alpha * loss_seq_text_total * text_weight +  # Only for PFUD/PSAD
        beta * loss_struct_text_total * text_weight +  # Only for PFUD/PSAD
        lambda_cons * consistency_loss * text_weight  # Only for PFUD/PSAD
    )
    
    return total_loss, {
        'seq_text': loss_seq_text_total.item() if isinstance(loss_seq_text_total, torch.Tensor) else 0.0,
        'struct_text': loss_struct_text_total.item() if isinstance(loss_struct_text_total, torch.Tensor) else 0.0,
        'seq_struct': loss_seq_struct_total.item(),
        'consistency': consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else 0.0,
        'n_text_samples': n_text_samples,
        'n_total_samples': batch_size
    }

class Trainer:
    """Training loop"""
    
    def __init__(self, model, device="cuda", lr=1e-4, alpha=1.0, beta=1.0, gamma=1.0, lambda_cons=0.1, 
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
        self.gamma = gamma
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
        
        pbar = tqdm(loader, desc="Training")
        for batch in pbar:
            sequence_emb = batch['sequence_emb'].to(self.device)
            structure_tokens = batch['structure_tokens'].to(self.device)
            text_emb = batch['text_emb'].to(self.device)
            protein_types = batch['protein_type']  # List of strings
            
            # Forward
            seq_proj, struct_proj, text_proj = self.model(sequence_emb, structure_tokens, text_emb)
            
            # Loss (type-aware tri-contrastive)
            loss, loss_dict = clip_contrastive_loss(
                seq_proj, struct_proj, text_proj, protein_types,
                temperature=self.model.temperature,
                alpha=self.alpha, beta=self.beta, gamma=self.gamma, lambda_cons=self.lambda_cons
            )
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            self.step += 1
            
            # Log to wandb (every batch)
            wandb.log({
                'train/batch_loss': loss.item(),
                'train/seq_text_loss': loss_dict['seq_text'],
                'train/struct_text_loss': loss_dict['struct_text'],
                'train/seq_struct_loss': loss_dict['seq_struct'],
                'train/consistency_loss': loss_dict['consistency'],
                'train/n_text_samples': loss_dict['n_text_samples'],
                'train/n_total_samples': loss_dict['n_total_samples'],
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
        
        pbar = tqdm(loader, desc="Validation")
        for batch in pbar:
            sequence_emb = batch['sequence_emb'].to(self.device)
            structure_tokens = batch['structure_tokens'].to(self.device)
            text_emb = batch['text_emb'].to(self.device)
            protein_types = batch['protein_type']  # List of strings
            
            seq_proj, struct_proj, text_proj = self.model(sequence_emb, structure_tokens, text_emb)
            loss, loss_dict = clip_contrastive_loss(
                seq_proj, struct_proj, text_proj, protein_types,
                temperature=self.model.temperature,
                alpha=self.alpha, beta=self.beta, gamma=self.gamma, lambda_cons=self.lambda_cons
            )
            
            total_loss += loss.item()
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
    print("TRI-MODAL ALIGNMENT: Sequence + Structure ↔ Text (Habdine)")
    print("=" * 70)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train tri-modal alignment model (Habdine, pre-split data)'
    )
    parser.add_argument('--model', type=str, required=True,
                       choices=list_available_models(),
                       help=f'Model type: {", ".join(list_available_models())}')
    parser.add_argument('--k', type=int, required=True,
                       help='Number of k-means clusters for structure tokens')
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
    alpha = 1.0  # Weight for seq-text loss (PFUD/PSAD only)
    beta = 1.0   # Weight for struct-text loss (PFUD/PSAD only)
    gamma = 1.0  # Weight for seq-struct loss (all types)
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
    wandb_run_name = f"run-tri-{args.model}-K{k_clusters}-{timestamp}"
    
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
            'gamma': gamma,
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
    print(f"📌 Loss weights: α={alpha} (seq-text), β={beta} (struct-text), γ={gamma} (seq-struct), λ={lambda_cons} (consistency)")
    print(f"📌 Early stopping patience: {patience} epochs")
    
    # Setup - hardcoded data/habdine paths
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path('data/habdine/triplet_embeddings') / args.model / f'K{k_clusters}'
    output_dir = Path('data/habdine/clip_alignment') / f'{args.model}_K{k_clusters}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Data: {data_dir}")
    print(f"💾 Output: {output_dir}")
    print(f"🖥️  Device: {device}")
    
    # Load data (pre-split: train and validation)
    print("\n" + "=" * 70)
    print("STEP 1: Load Pre-Split Batch Files")
    print("=" * 70)
    
    print("\n📦 Loading pre-split batch files...")
    train_triplets = load_pre_split_batch_files(data_dir, 'train')
    val_triplets = load_pre_split_batch_files(data_dir, 'validation')
    
    if not train_triplets:
        print(f"❌ No train triplet files found in {data_dir}")
        print(f"   Expected: triplet_embeddings_train_batch_*.npz")
        print(f"   Run 02_extract_embeddings.py first!")
        return
    
    print(f"✅ Train: {len(train_triplets):,} samples, Val: {len(val_triplets):,} samples")
    
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
        alpha=alpha, beta=beta, gamma=gamma, lambda_cons=lambda_cons,
        patience=patience
    )
    
    print(f"\nTraining config:")
    print(f"   Global epochs: {args.epochs}")
    print(f"   Train: {len(train_triplets):,} samples, Val: {len(val_triplets):,} samples")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Loss: Type-aware Tri-Contrastive (CLIP-style)")
    print(f"      - PFUD/PSAD: seq-text + struct-text + seq-struct + consistency")
    print(f"      - PDD/PSPD: seq-struct only\n")
    
    print("✅ Dataset loaded. Train/val from pre-split files.\n")
    
    # Global epoch loop
    for global_epoch in range(args.epochs):
        print(f"\n{'='*70}")
        print(f"GLOBAL EPOCH {global_epoch + 1}/{args.epochs}")
        print(f"{'='*70}")
        
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
        'loss_weights': {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'lambda': lambda_cons},
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
