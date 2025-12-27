#!/usr/bin/env python3
"""
Train Llama-3.1 with LoRA for Protein Function Prediction (Single-Node DataParallel)
Uses trained CLIP alignment model from script 04 to project protein embeddings.

Pipeline:
1. Load pre-trained CLIP alignment model (frozen)
2. Load Llama-3.1-8B with LoRA adapters
3. Train with tri-modal inputs: Sequence embedding + Structure tokens → Llama text space

This version uses simple DataParallel for 4 GPUs on a single node.
No torchrun or distributed setup required.
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
from typing import Dict, Tuple, Optional
import warnings
import sys
import argparse
import os
import gc
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import wandb

# Embedding dimensions (must match script 04)
ESM_HIDDEN_DIM = 2560  # ESM-2 3B
LLAMA_HIDDEN_DIM = 4096  # Llama-3.1

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
        proj = self.proj(x)
        return F.normalize(proj, p=2, dim=-1)


class StructureProjectionHead(nn.Module):
    """Project structure tokens to Llama-3.1 space (4096-dim)"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256, output_dim: int = LLAMA_HIDDEN_DIM, hidden_dim: int = 2048):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, token_ids):
        embedded = self.token_embedding(token_ids)
        pooled = embedded.mean(dim=1)
        proj = self.proj(pooled)
        return F.normalize(proj, p=2, dim=-1)


class TextProjectionHead(nn.Module):
    """Project Llama-3.1 text embeddings to shared space"""
    
    def __init__(self, input_dim: int = LLAMA_HIDDEN_DIM, output_dim: int = LLAMA_HIDDEN_DIM):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
    
    def forward(self, x):
        proj = self.proj(x)
        return F.normalize(proj, p=2, dim=-1)


class TriModalAlignmentModel(nn.Module):
    """Tri-modal alignment: Sequence + Structure ↔ Text (frozen, pre-trained from script 04)"""
    
    def __init__(self, 
                 structure_vocab_size: int,
                 sequence_dim: int = ESM_HIDDEN_DIM,
                 text_dim: int = LLAMA_HIDDEN_DIM,
                 shared_dim: int = LLAMA_HIDDEN_DIM,
                 temperature: float = 0.07):
        super().__init__()
        
        self.temperature = temperature
        self.shared_dim = shared_dim
        
        self.sequence_proj = ProteinProjectionHead(sequence_dim, shared_dim)
        self.structure_proj = StructureProjectionHead(structure_vocab_size, embedding_dim=256, output_dim=shared_dim)
        self.text_proj = TextProjectionHead(text_dim, shared_dim)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
    
    def forward(self, sequence_emb, structure_tokens):
        """
        Forward pass for protein embeddings only (text not needed for Llama training)
        
        Args:
            sequence_emb: [batch, 2560] ESM-2 sequence embeddings
            structure_tokens: [batch, seq_len] Structure token IDs
        
        Returns:
            seq_proj: [batch, 4096] Projected sequence embedding
            struct_proj: [batch, 4096] Projected structure embedding
        """
        seq_proj = self.sequence_proj(sequence_emb)
        struct_proj = self.structure_proj(structure_tokens)
        return seq_proj, struct_proj


class ProteinFunctionDataset(Dataset):
    """Dataset for protein function prediction with tri-modal inputs"""
    
    def __init__(self, triplets: list, metadata: list):
        """
        Args:
            triplets: List of dicts from script 03 output with:
                - sequence_embedding: ESM-2 3B [2560]
                - structure_tokens: Interleaved tokens [seq_len]
                - text_embedding: Llama-3.1 [4096]
                - protein_id: Identifier
            metadata: List of protein metadata with raw sequences and functions
        """
        self.triplets = triplets
        # Create protein_id -> metadata mapping
        self.id_to_metadata = {m['id']: m for m in metadata} if metadata else {}
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        protein_id = triplet['protein_id']
        
        # Get metadata (sequence, question, and function text)
        if protein_id in self.id_to_metadata:
            meta = self.id_to_metadata[protein_id]
            protein_seq = meta.get('sequence', 'UNKNOWN')
            question = meta.get('question', 'What is the function of this protein?')
            function_text = meta.get('function', 'Unknown function')
        else:
            protein_seq = 'UNKNOWN'
            question = 'What is the function of this protein?'
            function_text = 'Unknown function'
        
        return {
            'sequence_emb': torch.tensor(triplet['sequence_embedding'], dtype=torch.float32),
            'structure_tokens': torch.tensor(triplet['structure_tokens'], dtype=torch.long),
            'protein_sequence': protein_seq,
            'question': question,
            'function_text': function_text,
            'protein_id': protein_id
        }


def load_triplet_batch(batch_file: Path, metadata_file: Path = None) -> Tuple[list, list]:
    """
    Load triplet embeddings from script 03 output.
    
    Args:
        batch_file: Path to triplet_embeddings_K{k}_batch_*.npz
        metadata_file: Path to metadata JSON with protein sequences and functions
    
    Returns:
        (train_triplets, val_triplets, metadata)
    """
    # Load NPZ file
    data = np.load(batch_file, allow_pickle=True)
    
    seq_embeddings = data['sequence_embeddings']      # [n, 2560]
    text_embeddings = data['text_embeddings']         # [n, 4096]
    structure_tokens = data['structure_tokens']       # [n, max_len]
    protein_ids = data['protein_ids']                 # [n]
    
    # Reconstruct triplets
    triplets = []
    for i in range(len(seq_embeddings)):
        triplet = {
            'sequence_embedding': seq_embeddings[i],
            'text_embedding': text_embeddings[i],
            'structure_tokens': structure_tokens[i],
            'protein_id': str(protein_ids[i])
        }
        triplets.append(triplet)
    
    # Load metadata if available
    metadata = []
    if metadata_file and metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    
    # Shuffle and split
    shuffled_indices = np.arange(len(triplets))
    np.random.shuffle(shuffled_indices)
    triplets = [triplets[i] for i in shuffled_indices]
    
    n_val = int(len(triplets) * 0.1)
    val_triplets = triplets[:n_val]
    train_triplets = triplets[n_val:]
    
    return train_triplets, val_triplets, metadata


def load_all_triplets(batch_files: list, k_clusters: int, metadata_dir: Path = None) -> Tuple[list, list, list]:
    """
    Load all triplet batch files at once.
    
    Args:
        batch_files: List of triplet_embeddings_K{k}_batch_*.npz paths
        k_clusters: Number of K-means clusters
        metadata_dir: Directory with metadata JSON files
    
    Returns:
        (train_triplets, val_triplets, metadata)
    """
    print("\n📦 Loading all triplet files...")
    all_triplets = []
    all_metadata = []
    
    for batch_file in tqdm(batch_files, desc="Loading triplets"):
        try:
            # Find corresponding metadata file
            batch_num = batch_file.stem.split('_')[-1]
            metadata_file = metadata_dir / f'triplet_metadata_K{k_clusters}_batch_{batch_num}.json' if metadata_dir else None
            
            # Warn if metadata file doesn't exist
            if metadata_file and not metadata_file.exists():
                print(f"\n⚠️  Metadata file not found for batch {batch_num}: {metadata_file.name}")
            
            train_triplets, val_triplets, metadata = load_triplet_batch(batch_file, metadata_file)
            all_triplets.extend(train_triplets + val_triplets)
            all_metadata.extend(metadata)
        except Exception as e:
            print(f"\n⚠️  Error loading {batch_file.name}: {e}")
            continue
    
    print(f"✅ Loaded {len(all_triplets):,} total triplets")
    print(f"✅ Loaded {len(all_metadata):,} metadata entries")
    
    # Validate metadata
    if all_metadata:
        print(f"\n📊 Metadata sample (first entry):")
        sample = all_metadata[0]
        print(f"   ID: {sample.get('id', 'N/A')}")
        print(f"   Sequence length: {sample.get('length', 'N/A')}")
        print(f"   Question: {sample.get('question', 'N/A')[:100]}...")
        print(f"   Function: {sample.get('function', 'N/A')[:100]}...")
        print(f"   Type: {sample.get('type', 'N/A')}")
        print(f"   Subset: {sample.get('subset', 'N/A')[:50]}...")
    else:
        print("⚠️  Warning: No metadata loaded! Questions will use defaults.")
    
    # Shuffle all
    print("\n🔀 Shuffling dataset...")
    
    shuffled_indices = np.arange(len(all_triplets))
    np.random.shuffle(shuffled_indices)
    all_triplets = [all_triplets[i] for i in shuffled_indices]
    
    # Split train/val
    n_val = int(len(all_triplets) * 0.1)
    val_triplets = all_triplets[:n_val]
    train_triplets = all_triplets[n_val:]
    
    print(f"   Train: {len(train_triplets):,} samples")
    print(f"   Val:   {len(val_triplets):,} samples")
    
    return train_triplets, val_triplets, all_metadata


class TriModalLlamaTrainer:
    """Trainer for Llama with LoRA using pre-trained CLIP alignment"""
    
    def __init__(self, 
                 llama_model,
                 alignment_model: TriModalAlignmentModel,
                 tokenizer,
                 device: str = "cuda",
                 lr: float = 2e-4,
                 patience: int = 3,
                 gradient_accumulation_steps: int = 1):
        """
        Args:
            llama_model: Llama model with LoRA (DataParallel wrapped)
            alignment_model: Pre-trained CLIP alignment (frozen)
            tokenizer: Llama tokenizer
            device: Device
            lr: Learning rate
            patience: Early stopping patience
            gradient_accumulation_steps: Number of steps to accumulate gradients
        """
        self.llama = llama_model
        self.alignment_model = alignment_model.to(device)
        self.alignment_model.eval()  # Frozen
        self.tokenizer = tokenizer
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Get the actual model (unwrap DataParallel if needed)
        self.base_model = llama_model.module if isinstance(llama_model, nn.DataParallel) else llama_model
        
        # Only optimize LoRA parameters
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.llama.parameters()),
            lr=lr,
            weight_decay=1e-5
        )
        
        # Use linear warmup + cosine annealing for better stability
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
        
        warmup_steps = 500  # Warmup for first 500 steps
        total_steps = 10000  # Total training steps estimate
        
        warmup_scheduler = LinearLR(
            self.optimizer, 
            start_factor=0.1,  # Start at 10% of LR
            end_factor=1.0,
            total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=lr * 0.1  # Minimum LR is 10% of initial
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
        self.history = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')
        self.step = 0
        
        # Exponential moving average for stable loss logging
        self.ema_loss = None
        self.ema_decay = 0.995  # Stronger smoothing (99.5% previous, 0.5% new) - reduces variance
        self.loss_buffer = []  # Buffer to average losses before logging
        self.log_every_n_steps = 5  # Log raw loss less frequently to reduce noise
        
        # Early stopping
        self.patience = patience
        self.patience_counter = 0
        self.early_stop = False
    
    def construct_prompt_with_embeddings(self, batch):
        """
        Construct tri-modal prompt with protein embeddings injected.
        
        Format:
        [TEXT] "{question} Sequence: {seq}" (question from metadata)
        [SEQ_EMB] 
        [TEXT] "Structure:" 
        [STRUCT_EMB] 
        [TEXT] "Answer:" 
        [TARGET] "{function_text}"
        
        Questions vary by task type (PFUD, PSAD, PDD, PSPD) and come from Script 00 metadata.
        
        Returns:
            inputs_embeds: [batch, seq_len, 4096]
            attention_mask: [batch, seq_len]
            labels: [batch, seq_len]
        """
        batch_size = len(batch['protein_sequence'])
        
        # Project protein embeddings using frozen CLIP alignment
        with torch.no_grad():
            seq_emb = batch['sequence_emb'].to(self.device)
            struct_tokens = batch['structure_tokens'].to(self.device)
            seq_proj, struct_proj = self.alignment_model(seq_emb, struct_tokens)
            # Convert to float16 to match Llama dtype
            seq_proj = seq_proj.to(torch.float16).unsqueeze(1)  # [batch, 1, 4096]
            struct_proj = struct_proj.to(torch.float16).unsqueeze(1)  # [batch, 1, 4096]
        
        # Construct prompts for each sample
        all_embeds = []
        all_masks = []
        all_labels = []
        
        for i in range(batch_size):
            protein_seq = batch['protein_sequence'][i][:1024]  # Truncate for memory efficiency
            question = batch['question'][i]
            function_text = batch['function_text'][i][:1024]  # Truncate long functions
            
            # Part 1: Question (use actual question from metadata, append sequence for context)
            # Format: "{question} Sequence: {seq}" to provide both task-specific question and sequence
            part1 = f"{question} Sequence: {protein_seq}"
            inputs1 = self.tokenizer(part1, return_tensors="pt").to(self.device)
            emb1 = self.base_model.get_input_embeddings()(inputs1['input_ids'])
            
            # Part 2: Structure introduction
            part2 = " Structure:"
            inputs2 = self.tokenizer(part2, return_tensors="pt").to(self.device)
            emb2 = self.base_model.get_input_embeddings()(inputs2['input_ids'])
            
            # Part 3: Answer introduction
            part3 = " Answer:"
            inputs3 = self.tokenizer(part3, return_tensors="pt").to(self.device)
            emb3 = self.base_model.get_input_embeddings()(inputs3['input_ids'])
            
            # Part 4: Target (function)
            part4 = f" {function_text}"
            inputs4 = self.tokenizer(part4, return_tensors="pt").to(self.device)
            emb4 = self.base_model.get_input_embeddings()(inputs4['input_ids'])
            
            # Combine: text + seq_emb + text + struct_emb + text + target
            combined_emb = torch.cat([
                emb1[0],           # [len1, 4096]
                seq_proj[i],       # [1, 4096]
                emb2[0],           # [len2, 4096]
                struct_proj[i],    # [1, 4096]
                emb3[0],           # [len3, 4096]
                emb4[0]            # [len4, 4096]
            ], dim=0)
            
            # Attention mask (all ones)
            mask = torch.ones(combined_emb.shape[0], dtype=torch.long, device=self.device)
            
            # Labels: -100 for prompt, actual tokens for target
            prompt_len = (emb1.shape[1] + 1 + emb2.shape[1] + 1 + emb3.shape[1])
            labels = torch.full((combined_emb.shape[0],), -100, dtype=torch.long, device=self.device)
            labels[prompt_len:] = inputs4['input_ids'][0]
            
            all_embeds.append(combined_emb)
            all_masks.append(mask)
            all_labels.append(labels)
        
        # Pad to max length in batch
        max_len = max(e.shape[0] for e in all_embeds)
        
        padded_embeds = []
        padded_masks = []
        padded_labels = []
        
        for emb, mask, label in zip(all_embeds, all_masks, all_labels):
            pad_len = max_len - emb.shape[0]
            if pad_len > 0:
                emb_pad = torch.zeros(pad_len, emb.shape[1], device=self.device, dtype=emb.dtype)
                mask_pad = torch.zeros(pad_len, device=self.device, dtype=mask.dtype)
                label_pad = torch.full((pad_len,), -100, device=self.device, dtype=label.dtype)
                
                emb = torch.cat([emb, emb_pad], dim=0)
                mask = torch.cat([mask, mask_pad], dim=0)
                label = torch.cat([label, label_pad], dim=0)
            
            padded_embeds.append(emb)
            padded_masks.append(mask)
            padded_labels.append(label)
        
        inputs_embeds = torch.stack(padded_embeds)  # [batch, max_len, 4096]
        attention_mask = torch.stack(padded_masks)  # [batch, max_len]
        labels = torch.stack(padded_labels)  # [batch, max_len]
        
        return inputs_embeds, attention_mask, labels
    
    def train_epoch(self, loader):
        """Train one epoch with gradient accumulation"""
        self.llama.train()
        total_loss = 0
        
        pbar = tqdm(loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            inputs_embeds, attention_mask, labels = self.construct_prompt_with_embeddings(batch)
            
            # Get logits without computing loss (DataParallel issue workaround)
            outputs = self.llama(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask
            )
            
            # Manually compute loss to ensure gradients flow properly
            logits = outputs.logits  # [batch, seq_len, vocab_size]
            
            # Shift for causal LM: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten and compute cross-entropy
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            loss = loss_fct(shift_logits, shift_labels)
            
            loss = loss / self.gradient_accumulation_steps  # Scale loss
            loss.backward()
            
            # Only update weights every gradient_accumulation_steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()  # Update learning rate
                self.optimizer.zero_grad()
                self.step += 1
                
                # Compute exponential moving average of loss for stability
                raw_loss = loss.item() * self.gradient_accumulation_steps
                
                # Update EMA (stronger smoothing)
                if self.ema_loss is None:
                    self.ema_loss = raw_loss
                else:
                    self.ema_loss = self.ema_decay * self.ema_loss + (1 - self.ema_decay) * raw_loss
                
                # Buffer losses for averaging (reduces variance)
                self.loss_buffer.append(raw_loss)
                if len(self.loss_buffer) > 10:
                    self.loss_buffer.pop(0)
                
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Compute running average of recent losses
                avg_loss = sum(self.loss_buffer) / len(self.loss_buffer)
                
                log_dict = {
                    'train/batch_loss_smooth': self.ema_loss,  # EMA smoothed (most stable)
                    'train/batch_loss_avg': avg_loss,  # Running average (less noisy)
                    'train/step': self.step,
                    'train/learning_rate': current_lr
                }
                
                # Only log raw loss occasionally to reduce noise
                if self.step % self.log_every_n_steps == 0:
                    log_dict['train/batch_loss'] = raw_loss  # Raw loss (very noisy)
                
                wandb.log(log_dict)
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            pbar.set_postfix({'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}'})
            
            # Periodic cache clearing to prevent memory fragmentation
            if batch_idx % 50 == 0:  # More frequent clearing for OOM safety
                gc.collect()
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(loader)
        self.history['train_loss'].append(avg_loss)
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self, loader):
        """Validate with strict OOM handling - must process all batches"""
        self.llama.eval()
        total_loss = 0
        
        # Clear cache before validation
        gc.collect()
        torch.cuda.empty_cache()
        
        pbar = tqdm(loader, desc="Validation")
        for batch_idx, batch in enumerate(pbar):
            try:
                inputs_embeds, attention_mask, labels = self.construct_prompt_with_embeddings(batch)
                
                # Get logits without computing loss
                outputs = self.llama(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask
                )
                
                # Manually compute loss
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss_fct = nn.CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)
                loss = loss_fct(shift_logits, shift_labels)
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # Clear cache every batch during validation (more aggressive)
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                
            except torch.cuda.OutOfMemoryError:
                print(f"\n❌ OOM in validation - this indicates batch size is too large")
                print(f"   Validation requires processing ALL batches for accurate loss")
                print(f"   Solution: Reduce batch_size in config (currently {loader.batch_size})")
                raise  # Re-raise to stop training - validation must be accurate
        
        avg_loss = total_loss / len(loader)
        self.history['val_loss'].append(avg_loss)
        
        return avg_loss
    
    def check_early_stopping(self, val_loss, output_dir, k_clusters):
        """Check early stopping"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            
            # Save best model
            checkpoint_path = output_dir / f'best_llama_K{k_clusters}'
            self.save_checkpoint(checkpoint_path)
            print(f"   🎯 New best val loss: {val_loss:.4f}")
            
            return False
        else:
            self.patience_counter += 1
            print(f"   ⚠️  No improvement for {self.patience_counter}/{self.patience} epochs")
            
            if self.patience_counter >= self.patience:
                print(f"   🛑 Early stopping triggered!")
                self.early_stop = True
                return True
            
            return False
    
    def save_checkpoint(self, path):
        """Save LoRA adapter"""
        self.base_model.save_pretrained(path)
        print(f"✅ Saved LoRA adapter: {path}")


def main():
    print("=" * 70)
    print("TRAIN LLAMA-3.1 WITH LORA: Protein Function Prediction (DataParallel)")
    print("=" * 70)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Llama with LoRA for protein function')
    parser.add_argument('--k', type=int, default=128,
                       help='K-means clusters (must match script 04)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs (default: 50)')
    parser.add_argument('--gpus', type=str, default='0,1,2,3',
                       help='Comma-separated GPU IDs (default: 0,1,2,3)')
    
    args = parser.parse_args()
    
    k_clusters = args.k
    
    # Set visible GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    num_gpus = len(args.gpus.split(','))
    
    # Fixed hyperparameters - optimized for H200 140GB GPUs
    batch_size = 16  # Total batch across all GPUs (increased for better GPU utilization)
    gradient_accumulation_steps = 8  # Increased for more stable gradients (was 4)
    learning_rate = 1.5e-4  # Slightly reduced for stability (was 2e-4)
    lora_r = 16
    lora_alpha = 32
    patience = 3
    
    # Wandb login
    wandb_api_key = os.getenv('WANDB_API_KEY')
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
        print("🔑 Logged in to wandb")
    else:
        raise ValueError("❌ WANDB_API_KEY not set")
    
    # Initialize wandb
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(
        project="prottex-llama-lora",
        name=f"run-lora-dp-{timestamp}",
        config={
            'k_clusters': k_clusters,
            'epochs': args.epochs,
            'batch_size': batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'effective_batch_size': batch_size * gradient_accumulation_steps,
            'num_gpus': num_gpus,
            'learning_rate': learning_rate,
            'lora_r': lora_r,
            'lora_alpha': lora_alpha,
            'patience': patience,
            'model': 'Llama-3.1-8B',
            'distributed': 'DataParallel'
        }
    )
    
    # Setup
    device = "cuda:0"  # Main device
    
    print(f"\n🖥️  DataParallel training: {num_gpus} GPUs")
    for i in range(num_gpus):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)")
    
    triplet_dir = Path('data/triplet_embeddings')
    alignment_dir = Path(f'data/clip_alignment/K{k_clusters}')
    output_dir = Path(f'data/llama_lora/K{k_clusters}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Triplets: {triplet_dir}")
    print(f"📂 Alignment model: {alignment_dir}")
    print(f"💾 Output: {output_dir}")
    
    # Load alignment model config
    print("\n" + "=" * 70)
    print("STEP 1: Load Pre-trained CLIP Alignment Model")
    print("=" * 70)
    
    config_path = alignment_dir / f'config_K{k_clusters}.json'
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        print("   Run script 04 first")
        return
    
    with open(config_path) as f:
        clip_config = json.load(f)
    
    alignment_model = TriModalAlignmentModel(
        structure_vocab_size=clip_config['vocab_size'],
        sequence_dim=clip_config['sequence_dim'],
        text_dim=clip_config['text_dim'],
        shared_dim=clip_config['shared_dim'],
        temperature=clip_config['temperature']
    )
    
    # Load best checkpoint
    checkpoint_path = alignment_dir / f'best_model_K{k_clusters}.pt'
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"✅ Loading: {checkpoint_path.name}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    alignment_model.load_state_dict(checkpoint['model_state'])
    alignment_model.to(device)
    alignment_model.eval()
    
    # Freeze alignment model
    for param in alignment_model.parameters():
        param.requires_grad = False
    
    print("✅ Alignment model loaded and frozen")
    
    # Load Llama with LoRA
    print("\n" + "=" * 70)
    print("STEP 2: Load Llama-3.1-8B with LoRA")
    print("=" * 70)
    
    llama_model_name = 'meta-llama/Llama-3.1-8B-Instruct'  # Using Instruct model
    
    tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading Llama model to GPU...")
    llama = AutoModelForCausalLM.from_pretrained(
        llama_model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_cache=False  # Disable KV cache to save memory during training
    )
    
    # Apply LoRA before moving to GPU
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none"
    )
    
    llama = get_peft_model(llama, lora_config)
    
    # Note: Gradient checkpointing disabled for DataParallel with inputs_embeds
    # It breaks gradient flow when using custom embeddings
    # llama.gradient_checkpointing_enable()
    
    # Move to GPU
    llama = llama.to(device)
    
    # Wrap with DataParallel for multi-GPU
    if num_gpus > 1:
        print(f"\n🚀 Wrapping with DataParallel across {num_gpus} GPUs")
        llama = nn.DataParallel(llama)
    
    llama.module.print_trainable_parameters() if isinstance(llama, nn.DataParallel) else llama.print_trainable_parameters()
    print("✅ Llama loaded with LoRA and DataParallel")
    
    # Load data
    print("\n" + "=" * 70)
    print("STEP 3: Load Triplet Data")
    print("=" * 70)
    
    batch_files = sorted(triplet_dir.glob(f"triplet_embeddings_K{k_clusters}_batch_*.npz"))
    if not batch_files:
        print(f"❌ No triplet files found")
        print("   Run script 03 first")
        return
    
    print(f"✅ Found {len(batch_files)} batch files")
    
    # Train
    print("\n" + "=" * 70)
    print("STEP 4: Training")
    print("=" * 70)
    
    trainer = TriModalLlamaTrainer(
        llama_model=llama,
        alignment_model=alignment_model,
        tokenizer=tokenizer,
        device=device,
        lr=learning_rate,
        patience=patience,
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    
    print(f"\nConfig:")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Gradient accumulation: {gradient_accumulation_steps}")
    print(f"   Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"   Number of GPUs: {num_gpus}")
    print(f"   LoRA r={lora_r}, alpha={lora_alpha}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Early stopping patience: {patience}\n")
    
    for epoch in range(args.epochs):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch + 1}/{args.epochs}")
        print(f"{'='*70}")
        
        # Load all triplets for this epoch
        train_triplets, val_triplets, metadata = load_all_triplets(batch_files, k_clusters, triplet_dir)
        
        train_dataset = ProteinFunctionDataset(train_triplets, metadata)
        val_dataset = ProteinFunctionDataset(val_triplets, metadata)
        
        # Simple DataLoader (no DistributedSampler needed)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        
        print(f"\n📊 Epoch {epoch + 1} - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)
        
        print(f"\n📊 Epoch {epoch + 1} Summary:")
        print(f"   Train loss: {train_loss:.4f}")
        print(f"   Val loss: {val_loss:.4f}")
        
        wandb.log({
            'epoch': epoch + 1,
            'train/epoch_loss': train_loss,
            'val/epoch_loss': val_loss,
            'best_val_loss': trainer.best_val_loss
        })
        
        if trainer.check_early_stopping(val_loss, output_dir, k_clusters):
            break
        
        # Note: scheduler.step() is called per training step, not per epoch
    
    # Save final
    print("\n" + "=" * 70)
    print("STEP 5: Save Results")
    print("=" * 70)
    
    final_path = output_dir / f'final_lora_K{k_clusters}'
    trainer.save_checkpoint(final_path)
    
    config_out = {
        'k_clusters': k_clusters,
        'epochs': args.epochs,
        'batch_size': batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'effective_batch_size': batch_size * gradient_accumulation_steps,
        'num_gpus': num_gpus,
        'learning_rate': learning_rate,
        'lora_r': lora_r,
        'lora_alpha': lora_alpha,
        'patience': patience,
        'best_val_loss': trainer.best_val_loss,
        'timestamp': datetime.now().isoformat(),
        'distributed': 'DataParallel'
    }
    
    config_out_path = output_dir / f'config_K{k_clusters}.json'
    with open(config_out_path, 'w') as f:
        json.dump(config_out, f, indent=2)
    
    print(f"✅ Training complete!")
    print(f"   Best val loss: {trainer.best_val_loss:.4f}")
    
    wandb.finish()


if __name__ == '__main__':
    main()

