#!/usr/bin/env python3
"""
Unified LoRA Training Script for Protein Function Prediction
Supports multiple models: Llama, Qwen, DeepSeek

Uses trained CLIP alignment model from script 04 to project protein embeddings.

Pipeline:
1. Load pre-trained CLIP alignment model (frozen)
2. Load model with LoRA adapters (configurable)
3. Train with tri-modal inputs: Sequence embedding + Structure tokens → Model text space

Usage:
    python run/05_train_lora.py --model llama --k 128 --epochs 3
    python run/05_train_lora.py --model qwen --k 128 --epochs 3
    python run/05_train_lora.py --model deepseek-v2 --k 128 --epochs 3
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
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
warnings.filterwarnings('ignore')

# Add root directory to Python path for imports when running from root
script_dir = Path(__file__).parent
root_dir = script_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import wandb

# Import model configuration
from run.config import get_model_config, list_available_models

# Embedding dimensions (must match script 04)
ESM_HIDDEN_DIM = 2560  # ESM-2 3B
MODEL_HIDDEN_DIM = 4096  # All 7B models use 4096


def setup_distributed():
    """Initialize distributed training environment"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("Not using distributed mode")
        return 0, 1, 0
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    
    return rank, world_size, local_rank

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


class ProteinProjectionHead(nn.Module):
    """Project protein embeddings to model space (4096-dim)"""
    
    def __init__(self, input_dim: int, output_dim: int = MODEL_HIDDEN_DIM, hidden_dim: int = 2048):
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
    """Project structure tokens to model space (4096-dim)"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256, output_dim: int = MODEL_HIDDEN_DIM, hidden_dim: int = 2048):
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
    """Project text embeddings to shared space"""
    
    def __init__(self, input_dim: int = MODEL_HIDDEN_DIM, output_dim: int = MODEL_HIDDEN_DIM):
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
                 text_dim: int = MODEL_HIDDEN_DIM,
                 shared_dim: int = MODEL_HIDDEN_DIM,
                 temperature: float = 0.07):
        super().__init__()
        
        self.temperature = temperature
        self.shared_dim = shared_dim
        
        self.sequence_proj = ProteinProjectionHead(sequence_dim, shared_dim)
        self.structure_proj = StructureProjectionHead(structure_vocab_size, embedding_dim=256, output_dim=shared_dim)
        self.text_proj = TextProjectionHead(text_dim, shared_dim)
    
    def forward(self, sequence_emb, structure_tokens):
        """
        Forward pass for protein embeddings only (text not needed for training)
        
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
    """Dataset for protein function prediction with tri-modal inputs (type-aware)"""
    
    def __init__(self, triplets: list, metadata: list):
        """
        Args:
            triplets: List of dicts from script 03 output with:
                - sequence_embedding: ESM-2 3B [2560]
                - structure_tokens: Interleaved tokens [seq_len]
                - text_embedding: Model-specific [4096]
                - protein_id: Identifier
            metadata: List of protein metadata with QA and metadata structure
        """
        self.triplets = triplets
        # Create protein_id -> metadata mapping
        # Metadata structure: {"QA": {...}, "metadata": {"id": "...", "type": "...", ...}}
        self.id_to_metadata = {}
        if metadata:
            for meta_entry in metadata:
                meta_section = meta_entry.get('metadata', {})
                qa_section = meta_entry.get('QA', {})
                protein_id = meta_section.get('id', '')
                if protein_id:
                    self.id_to_metadata[protein_id] = {
                        'QA': qa_section,
                        'metadata': meta_section
                    }
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        protein_id = triplet['protein_id']
        
        # Get metadata (sequence, question, and answer text)
        if protein_id in self.id_to_metadata:
            meta = self.id_to_metadata[protein_id]
            qa = meta.get('QA', {})
            meta_info = meta.get('metadata', {})
            
            protein_seq = qa.get('sequence', 'UNKNOWN')
            question = qa.get('question', 'What is the function of this protein?')
            answer = qa.get('answer', 'Unknown function')
            protein_type = meta_info.get('type', 'PFUD')
        else:
            protein_seq = 'UNKNOWN'
            question = 'What is the function of this protein?'
            answer = 'Unknown function'
            protein_type = 'PFUD'
        
        return {
            'sequence_emb': torch.tensor(triplet['sequence_embedding'], dtype=torch.float32),
            'structure_tokens': torch.tensor(triplet['structure_tokens'], dtype=torch.long),
            'protein_sequence': protein_seq,
            'question': question,
            'function_text': answer,  # Use answer as function_text
            'protein_id': protein_id,
            'protein_type': protein_type
        }


def load_triplet_batch(batch_file: Path, metadata_file: Path = None) -> Tuple[list, list]:
    """
    Load triplet embeddings from script 03 output.
    
    Args:
        batch_file: Path to triplet_embeddings_batch_*.npz
        metadata_file: Path to triplet_metadata_batch_*.json (optional)
    
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
    
    # Split sequentially (no shuffling)
    n_val = int(len(triplets) * 0.1)
    val_triplets = triplets[:n_val]
    train_triplets = triplets[n_val:]
    
    return train_triplets, val_triplets, metadata


def load_all_triplets(batch_files: list, metadata_dir: Path = None, 
                      train_ratio: float = 0.9, val_ratio: float = 0.05, test_ratio: float = 0.05, 
                      rank: int = 0) -> Tuple[list, list, list, list]:
    """
    Load all triplet batch files at once (type-aware filtering).
    
    Args:
        batch_files: List of triplet_embeddings_batch_*.npz paths
        metadata_dir: Directory with triplet_metadata_batch_*.json files
        train_ratio: Ratio of training samples (default: 0.9)
        val_ratio: Ratio of validation samples (default: 0.05)
        test_ratio: Ratio of test samples (default: 0.05)
        rank: Process rank for distributed training (only rank 0 prints)
    
    Returns:
        (train_triplets, val_triplets, test_triplets, metadata) - filtered to exclude PDD/PSPD
    """
    if rank == 0:
        print("\n📦 Loading all triplet files...")
    all_triplets = []
    all_metadata = []
    
    for batch_file in tqdm(batch_files, desc="Loading triplets", disable=(rank != 0)):
        try:
            # Find corresponding metadata file
            metadata_file = None
            if metadata_dir:
                if 'batch_' in batch_file.stem:
                    batch_part = batch_file.stem.split('batch_')[-1]  # Get '36' part
                    metadata_file = metadata_dir / f'triplet_metadata_batch_{batch_part}.json'
            
            # Warn if metadata file doesn't exist
            if metadata_file and not metadata_file.exists():
                if rank == 0:
                    print(f"\n⚠️  Metadata file not found for batch {batch_file.stem}: {metadata_file.name}")
            
            train_triplets, val_triplets, metadata = load_triplet_batch(batch_file, metadata_file)
            all_triplets.extend(train_triplets + val_triplets)
            all_metadata.extend(metadata)
        except Exception as e:
            if rank == 0:
                print(f"\n⚠️  Error loading {batch_file.name}: {e}")
            continue
    
    if rank == 0:
        print(f"✅ Loaded {len(all_triplets):,} total triplets")
        print(f"✅ Loaded {len(all_metadata):,} metadata entries")
    
    # Type-aware filtering: create protein_id -> type mapping
    # Filter out PDD/PSPD types (where answer == sequence, not meaningful text)
    protein_id_to_type = {}
    for meta_entry in all_metadata:
        meta_section = meta_entry.get('metadata', {})
        protein_id = meta_section.get('id', '')
        protein_type = meta_section.get('type', 'PFUD')
        if protein_id:
            protein_id_to_type[protein_id] = protein_type
    
    # Filter triplets: only keep PFUD/PSAD (skip PDD/PSPD)
    filtered_triplets = []
    skipped_count = 0
    type_counts = {'PFUD': 0, 'PSAD': 0, 'PDD': 0, 'PSPD': 0}
    
    for triplet in all_triplets:
        protein_id = triplet['protein_id']
        protein_type = protein_id_to_type.get(protein_id, 'PFUD')
        type_counts[protein_type] = type_counts.get(protein_type, 0) + 1
        
        # Skip PDD/PSPD types (answer is sequence, not meaningful text)
        if protein_type in ['PDD', 'PSPD']:
            skipped_count += 1
            continue
        
        filtered_triplets.append(triplet)
    
    if rank == 0:
        print(f"\n📊 Type distribution (before filtering):")
        for ptype, count in sorted(type_counts.items()):
            print(f"   {ptype}: {count:,}")
        print(f"\n🔍 Filtered out {skipped_count:,} PDD/PSPD samples (answer == sequence)")
        print(f"✅ Keeping {len(filtered_triplets):,} PFUD/PSAD samples for training")
    
    # Split train/val/test sequentially (90/5/5) - no shuffling
    total = len(filtered_triplets)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    
    train_triplets = filtered_triplets[:n_train]
    val_triplets = filtered_triplets[n_train:n_train + n_val]
    test_triplets = filtered_triplets[n_train + n_val:]
    
    if rank == 0:
        print(f"   Train: {len(train_triplets):,} samples ({100*len(train_triplets)/total:.1f}%)")
        print(f"   Val:   {len(val_triplets):,} samples ({100*len(val_triplets)/total:.1f}%)")
        print(f"   Test:  {len(test_triplets):,} samples ({100*len(test_triplets)/total:.1f}%)")
    
    return train_triplets, val_triplets, test_triplets, all_metadata


class TriModalTrainer:
    """Unified trainer for any model with LoRA using pre-trained CLIP alignment"""
    
    def __init__(self, 
                 model,
                 alignment_model: TriModalAlignmentModel,
                 tokenizer,
                 model_config,
                 device: str = "cuda",
                 lr: float = 2e-4,
                 patience: int = 3,
                 gradient_accumulation_steps: int = 1,
                 rank: int = 0,
                 world_size: int = 1):
        """
        Args:
            model: Model with LoRA (DDP wrapped)
            alignment_model: Pre-trained CLIP alignment (frozen)
            tokenizer: Model tokenizer
            model_config: ModelConfig object from 05_config
            device: Device
            lr: Learning rate
            patience: Early stopping patience
            gradient_accumulation_steps: Number of steps to accumulate gradients
            rank: Process rank for distributed training
            world_size: Total number of processes
        """
        self.model = model
        self.alignment_model = alignment_model.to(device)
        self.alignment_model.eval()  # Frozen
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.rank = rank
        self.world_size = world_size
        
        # Get the actual model (unwrap DDP if needed)
        self.base_model = model.module if isinstance(model, DDP) else model
        
        # Only optimize LoRA parameters
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
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
        self.ema_decay = 0.99  # Smoothing factor for loss
        
        # Early stopping
        self.patience = patience
        self.patience_counter = 0
        self.early_stop = False
    
    def construct_prompt_with_embeddings(self, batch):
        """
        Construct tri-modal prompt with protein embeddings injected.
        Uses model-specific prompt builder from config.
        
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
            # Convert to float16 to match model dtype
            seq_proj = seq_proj.to(torch.float16).unsqueeze(1)  # [batch, 1, 4096]
            struct_proj = struct_proj.to(torch.float16).unsqueeze(1)  # [batch, 1, 4096]
        
        # Construct prompts for each sample using model-specific format
        all_embeds = []
        all_masks = []
        all_labels = []
        
        for i in range(batch_size):
            protein_seq = batch['protein_sequence'][i][:1024]  # Truncate for memory efficiency
            question = batch['question'][i]
            function_text = batch['function_text'][i][:1024]  # Truncate long functions
            
            # Use model-specific prompt builder from config
            part1, part2, part3, part4 = self.model_config.prompt_builder(question, protein_seq, function_text)
            
            # Tokenize each part
            inputs1 = self.tokenizer(part1, return_tensors="pt", add_special_tokens=False).to(self.device)
            emb1 = self.base_model.get_input_embeddings()(inputs1['input_ids'])
            
            inputs2 = self.tokenizer(part2, return_tensors="pt", add_special_tokens=False).to(self.device)
            emb2 = self.base_model.get_input_embeddings()(inputs2['input_ids'])
            
            inputs3 = self.tokenizer(part3, return_tensors="pt", add_special_tokens=False).to(self.device)
            emb3 = self.base_model.get_input_embeddings()(inputs3['input_ids'])
            
            inputs4 = self.tokenizer(part4, return_tensors="pt", add_special_tokens=False).to(self.device)
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
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(loader, desc="Training", disable=(self.rank != 0))
        for batch_idx, batch in enumerate(pbar):
            inputs_embeds, attention_mask, labels = self.construct_prompt_with_embeddings(batch)
            
            # Get logits without computing loss (better gradient flow)
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask
            )
            
            # Manually compute loss to ensure proper gradients
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
                if self.ema_loss is None:
                    self.ema_loss = raw_loss
                else:
                    self.ema_loss = self.ema_decay * self.ema_loss + (1 - self.ema_decay) * raw_loss
                
                if self.rank == 0:  # Only log on main process
                    current_lr = self.optimizer.param_groups[0]['lr']
                    wandb.log({
                        'train/batch_loss': raw_loss,
                        'train/batch_loss_smooth': self.ema_loss,
                        'train/step': self.step,
                        'train/learning_rate': current_lr
                    })
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            pbar.set_postfix({'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}'})
            
            # Periodic cache clearing to prevent memory fragmentation
            if batch_idx % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(loader)
        self.history['train_loss'].append(avg_loss)
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self, loader):
        """Validate with strict OOM handling - must process all batches"""
        self.model.eval()
        total_loss = 0
        
        # Clear cache before validation
        gc.collect()
        torch.cuda.empty_cache()
        
        pbar = tqdm(loader, desc="Validation", disable=(self.rank != 0))
        for batch_idx, batch in enumerate(pbar):
            try:
                inputs_embeds, attention_mask, labels = self.construct_prompt_with_embeddings(batch)
                
                # Get logits without computing loss
                outputs = self.model(
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
                
                # Clear cache every batch during validation
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                
            except torch.cuda.OutOfMemoryError:
                if self.rank == 0:
                    print(f"\n❌ OOM in validation - this indicates batch size is too large")
                    print(f"   Validation requires processing ALL batches for accurate loss")
                    print(f"   Solution: Reduce batch_size in config (currently {loader.batch_size})")
                raise
        
        avg_loss = total_loss / len(loader)
        self.history['val_loss'].append(avg_loss)
        
        return avg_loss
    
    def check_early_stopping(self, val_loss, output_dir, k_clusters):
        """Check early stopping"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            
            # Save best model (only rank 0)
            if self.rank == 0:
                checkpoint_path = output_dir / f'{self.model_config.checkpoint_prefix}_K{k_clusters}'
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
    # Hardcoded train/val/test split ratios (90/5/5)
    TRAIN_RATIO = 0.9
    VAL_RATIO = 0.05
    TEST_RATIO = 0.05
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train model with LoRA for protein function prediction'
    )
    parser.add_argument('--model', type=str, required=True,
                       choices=list_available_models(),
                       help=f'Model type: {", ".join(list_available_models())}')
    parser.add_argument('--k', type=int, required=True,
                       help='K-means clusters (must match script 04)')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Training epochs (default: 3)')
    
    args = parser.parse_args()
    
    # Get model configuration
    model_config = get_model_config(args.model)
    k_clusters = args.k
    
    if rank == 0:
        print("=" * 70)
        print(f"TRAIN {args.model.upper()} WITH LORA: Protein Function Prediction (DDP)")
        print("=" * 70)
        print(f"Model: {model_config.model_name}")
    
    # Fixed hyperparameters - optimized for H200 140GB GPUs
    batch_size = 6  # Per GPU
    gradient_accumulation_steps = 4
    learning_rate = 1.5e-4
    lora_r = 16
    lora_alpha = 32
    patience = 2
    
    # Wandb login (only rank 0)
    if rank == 0:
        wandb_api_key = os.getenv('WANDB_API_KEY')
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
            print("🔑 Logged in to wandb")
        else:
            raise ValueError("❌ WANDB_API_KEY not set")
        
        # Initialize wandb
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wandb.init(
            project='prottex-lora',
            name=f"run-{args.model}-lora-K{k_clusters}-ddp-{timestamp}",
            config={
                'k_clusters': k_clusters,
                'epochs': args.epochs,
                'batch_size_per_gpu': batch_size,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'effective_batch_size': batch_size * gradient_accumulation_steps * world_size,
                'num_gpus': world_size,
                'learning_rate': learning_rate,
                'lora_r': lora_r,
                'lora_alpha': lora_alpha,
                'patience': patience,
                'model': model_config.model_name,
                'model_type': args.model,
                'distributed': 'DDP'
            }
        )
    
    # Setup - use local device for this process
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)
    
    if rank == 0:
        print(f"\n🖥️  Distributed training: {world_size} GPUs")
        for i in range(world_size):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)")
    
    # Use model-specific directories with K{k_clusters} subdirectory
    triplet_dir = Path(f'data/triplet_embeddings/{args.model}/K{k_clusters}')
    metadata_dir = triplet_dir
    alignment_dir = Path(f'data/clip_alignment/{args.model}_K{k_clusters}')
    output_dir = Path(f'data/lora/{model_config.output_dir_suffix}/K{k_clusters}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if rank == 0:
        print(f"\n📂 Triplets: {triplet_dir}")
        print(f"📂 Alignment model: {alignment_dir}")
        print(f"💾 Output: {output_dir}")
        
        # Load alignment model config
        print("\n" + "=" * 70)
        print("STEP 1: Load Pre-trained CLIP Alignment Model")
        print("=" * 70)
    
    config_path = alignment_dir / f'config_K{k_clusters}.json'
    if not config_path.exists():
        if rank == 0:
            print(f"❌ Config not found: {config_path}")
            print("   Run script 04 first")
        cleanup_distributed()
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
        if rank == 0:
            print(f"❌ Checkpoint not found: {checkpoint_path}")
        cleanup_distributed()
        return
    
    if rank == 0:
        print(f"✅ Loading: {checkpoint_path.name}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    alignment_model.load_state_dict(checkpoint['model_state'])
    alignment_model.to(device)
    alignment_model.eval()
    
    # Freeze alignment model
    for param in alignment_model.parameters():
        param.requires_grad = False
    
    if rank == 0:
        print("✅ Alignment model loaded and frozen")
        
        # Load model with LoRA
        print("\n" + "=" * 70)
        print(f"STEP 2: Load {args.model.upper()} with LoRA")
        print("=" * 70)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name,
        trust_remote_code=model_config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if rank == 0:
        print(f"Loading {args.model} model ({model_config.model_name}) to GPU...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_cache=False,
        trust_remote_code=model_config.trust_remote_code
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
    
    model = get_peft_model(model, lora_config)
    
    # Move to local GPU for this process
    model = model.to(device)
    
    # Wrap with DistributedDataParallel
    if rank == 0:
        print(f"\n🚀 Wrapping with DDP across {world_size} GPUs")
        model.print_trainable_parameters()
    
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    
    if rank == 0:
        print(f"✅ {args.model.upper()} loaded with LoRA and DDP")
    
    # Load data
    if rank == 0:
        print("\n" + "=" * 70)
        print("STEP 3: Load Triplet Data")
        print("=" * 70)
    
    all_batch_files = sorted(triplet_dir.glob(f"triplet_embeddings_batch_*.npz"))
    if not all_batch_files:
        if rank == 0:
            print(f"❌ No triplet files found")
            print("   Run script 03 first")
        cleanup_distributed()
        return
    
    batch_files = all_batch_files
    if rank == 0:
        print(f"✅ Found {len(batch_files)} batch files")
        print(f"📊 Data split: Train={TRAIN_RATIO*100:.0f}%, Val={VAL_RATIO*100:.0f}%, Test={TEST_RATIO*100:.0f}%")
    
    # Train
    if rank == 0:
        print("\n" + "=" * 70)
        print("STEP 4: Training")
        print("=" * 70)
    
    trainer = TriModalTrainer(
        model=model,
        alignment_model=alignment_model,
        tokenizer=tokenizer,
        model_config=model_config,
        device=device,
        lr=learning_rate,
        patience=patience,
        gradient_accumulation_steps=gradient_accumulation_steps,
        rank=rank,
        world_size=world_size
    )
    
    if rank == 0:
        print(f"\nConfig:")
        print(f"   Model: {model_config.model_name}")
        print(f"   Model type: {args.model}")
        print(f"   Epochs: {args.epochs}")
        print(f"   Batch size per GPU: {batch_size}")
        print(f"   Total parallel samples: {batch_size * world_size}")
        print(f"   Gradient accumulation: {gradient_accumulation_steps}")
        print(f"   Effective batch size: {batch_size * gradient_accumulation_steps * world_size}")
        print(f"   Number of GPUs: {world_size}")
        print(f"   LoRA r={lora_r}, alpha={lora_alpha}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Early stopping patience: {patience}\n")
    
    for epoch in range(args.epochs):
        if rank == 0:
            print(f"\n{'='*70}")
            print(f"EPOCH {epoch + 1}/{args.epochs}")
            print(f"{'='*70}")
        
        # Load all triplets for this epoch
        train_triplets, val_triplets, test_triplets, metadata = load_all_triplets(
            batch_files, metadata_dir, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, rank
        )
        
        # Save test data for evaluation (only in first epoch, only rank 0)
        if epoch == 0 and rank == 0:
            # Create evaluation directory
            eval_dir = Path(f'data/evaluation/{args.model}/K{k_clusters}')
            eval_dir.mkdir(parents=True, exist_ok=True)
            
            # Create protein_id -> full metadata mapping
            protein_id_to_full_metadata = {}
            for meta_entry in metadata:
                meta_section = meta_entry.get('metadata', {})
                qa_section = meta_entry.get('QA', {})
                protein_id = meta_section.get('id', '')
                if protein_id:
                    protein_id_to_full_metadata[protein_id] = {
                        'QA': qa_section,
                        'metadata': meta_section
                    }
            
            # Convert test_triplets to inference format (same format as script 06 expects)
            test_data = []
            for triplet in test_triplets:
                protein_id = triplet['protein_id']
                if protein_id in protein_id_to_full_metadata:
                    meta = protein_id_to_full_metadata[protein_id]
                    qa = meta.get('QA', {})
                    meta_info = meta.get('metadata', {})
                    
                    test_data.append({
                        'id': protein_id,
                        'sequence': qa.get('sequence', ''),
                        'question': qa.get('question', ''),
                        'function': qa.get('answer', ''),
                        'type': meta_info.get('type', 'PFUD'),
                        'subset': meta_info.get('subset', '')
                    })
            
            # Save test data
            test_data_path = eval_dir / 'test_data.json'
            with open(test_data_path, 'w') as f:
                json.dump(test_data, f, indent=2)
            
            print(f"\n💾 Saved test data: {test_data_path}")
            print(f"   Test samples: {len(test_data)}")
        
        train_dataset = ProteinFunctionDataset(train_triplets, metadata)
        val_dataset = ProteinFunctionDataset(val_triplets, metadata)
        
        # Use DistributedSampler for proper data distribution
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=2, pin_memory=True)
        
        # Set epoch for sampler (important for shuffling)
        train_sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"\n📊 Epoch {epoch + 1} - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        train_loss = trainer.train_epoch(train_loader)
        
        # Synchronize before validation
        if dist.is_initialized():
            dist.barrier()
        
        val_loss = trainer.validate(val_loader)
        
        # Average validation loss across all GPUs for accurate metric
        if dist.is_initialized():
            val_loss_tensor = torch.tensor([val_loss], device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
            val_loss = val_loss_tensor.item()
        
        if rank == 0:
            print(f"\n📊 Epoch {epoch + 1} Summary:")
            print(f"   Train loss: {train_loss:.4f}")
            print(f"   Val loss: {val_loss:.4f}")
            
            wandb.log({
                'epoch': epoch + 1,
                'train/epoch_loss': train_loss,
                'val/epoch_loss': val_loss,
                'val/best_val_loss': trainer.best_val_loss
            })
        
        if trainer.check_early_stopping(val_loss, output_dir, k_clusters):
            break
    
    # Save final (only rank 0)
    if rank == 0:
        print("\n" + "=" * 70)
        print("STEP 5: Save Results")
        print("=" * 70)
        
        final_path = output_dir / f'final_lora_K{k_clusters}'
        trainer.save_checkpoint(final_path)
        
        config_out = {
            'k_clusters': k_clusters,
            'epochs': args.epochs,
            'batch_size_per_gpu': batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'effective_batch_size': batch_size * gradient_accumulation_steps * world_size,
            'num_gpus': world_size,
            'learning_rate': learning_rate,
            'lora_r': lora_r,
            'lora_alpha': lora_alpha,
            'patience': patience,
            'best_val_loss': trainer.best_val_loss,
            'timestamp': datetime.now().isoformat(),
            'distributed': 'DDP',
            'model': model_config.model_name,
            'model_type': args.model,
            'train_batches': len(batch_files),
            'train_ratio': TRAIN_RATIO,
            'val_ratio': VAL_RATIO,
            'test_ratio': TEST_RATIO
        }
        
        config_out_path = output_dir / f'config_K{k_clusters}.json'
        with open(config_out_path, 'w') as f:
            json.dump(config_out, f, indent=2)
        
        print(f"✅ Training complete!")
        print(f"   Best val loss: {trainer.best_val_loss:.4f}")
        
        wandb.finish()
    
    # Cleanup distributed
    cleanup_distributed()

if __name__ == '__main__':
    main()

