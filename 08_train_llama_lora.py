#!/usr/bin/env python3
"""
Train Llama-3.1 with LoRA for Protein Function Prediction
Uses tri-modal reasoning: Sequence → Embedding → Text → Structure Embedding → Function
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
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import wandb

LLAMA_HIDDEN_DIM = 4096


class ProteinProjectionHead(nn.Module):
    """Project protein embeddings to Llama space"""
    
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
    """Project structure tokens to Llama space"""
    
    def __init__(self, vocab_size: int = 537, embedding_dim: int = 256, output_dim: int = LLAMA_HIDDEN_DIM, hidden_dim: int = 2048):
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


class TriModalAlignmentModel(nn.Module):
    """Tri-modal: Sequence + Structure ↔ Text (frozen, pre-trained)"""
    
    def __init__(self, 
                 sequence_dim: int = 1280,
                 structure_vocab_size: int = 537,
                 text_dim: int = LLAMA_HIDDEN_DIM,
                 shared_dim: int = LLAMA_HIDDEN_DIM):
        super().__init__()
        
        self.sequence_proj = ProteinProjectionHead(sequence_dim, shared_dim)
        self.structure_proj = StructureProjectionHead(structure_vocab_size, embedding_dim=256, output_dim=shared_dim)
        
        # Learnable temperature parameter (CLIP-style)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, sequence_emb, structure_tokens):
        seq_proj = self.sequence_proj(sequence_emb)
        struct_proj = self.structure_proj(structure_tokens)
        return seq_proj, struct_proj


class ProteinFunctionDataset(Dataset):
    """Dataset for protein function prediction with tri-modal inputs"""
    
    def __init__(self, pairs: list):
        """
        Args:
            pairs: List of dicts with:
                - sequence_embedding: ESM-2 [1280]
                - structure_tokens: Interleaved tokens [seq_len]
                - protein_sequence: Raw AA sequence string
                - function_text: Function description (target)
                - protein_id: Identifier
        """
        self.pairs = pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        return {
            'sequence_emb': torch.tensor(pair['sequence_embedding'], dtype=torch.float32),
            'structure_tokens': torch.tensor(pair['structure_tokens'], dtype=torch.long),
            'protein_sequence': pair['protein_sequence'],
            'function_text': pair['function_text'],
            'question': pair['question'],
            'protein_id': pair.get('protein_id', f'p_{idx}')
        }


def load_batch_file(batch_file: Path) -> Tuple[list, list]:
    """
    Load NPZ batch file and extract protein data with sequences and functions.
    
    Args:
        batch_file: Path to embedding_pairs_batch_*.npz
    
    Returns:
        (train_pairs, val_pairs)
    """
    # Load embeddings from NPZ
    data = np.load(batch_file, allow_pickle=True)
    
    seq_embeddings = data['sequence_embeddings']
    text_embeddings = data['text_embeddings']
    protein_ids = data['protein_ids']
    structure_tokens_obj = data['structure_tokens']
    
    # Load raw sequences and texts from JSON (script 01 output)
    # Extract batch number from filename: embedding_pairs_batch_0.npz → sample_proteins_batch_0.json
    batch_num = batch_file.stem.split('_')[-1]  # Get the batch number
    json_file = batch_file.parent.parent / 'sample_proteins' / f'sample_proteins_batch_{batch_num}.json'
    
    # Create protein ID to raw data mapping
    id_to_raw = {}
    if json_file.exists():
        with open(json_file, 'r') as f:
            raw_proteins = json.load(f)
        for p in raw_proteins:
            id_to_raw[p['id']] = {
                'sequence': p['sequence'],
                'text': p['text'],
                'question': p.get('question', 'What is the function of this protein?')
            }
    else:
        print(f"⚠️  Warning: JSON file not found: {json_file}")
        print(f"   Using dummy data for sequences and texts")
    
    # Reconstruct pairs
    pairs = []
    for i in range(len(seq_embeddings)):
        struct_tokens_list = structure_tokens_obj[i]
        struct_tokens_arr = np.array(struct_tokens_list, dtype=np.int64)
        
        protein_id = str(protein_ids[i])
        
        # Get raw sequence and text from JSON
        if protein_id in id_to_raw:
            protein_seq = id_to_raw[protein_id]['sequence']
            function_text = id_to_raw[protein_id]['text']
            question = id_to_raw[protein_id]['question']
        else:
            # Fallback to dummy data if not found
            protein_seq = f"MVHLTPEEKS{'A'*340}"
            function_text = "This protein is involved in cellular processes."
            question = "What is the function of this protein?"
        
        pair = {
            'sequence_embedding': seq_embeddings[i],
            'text_embedding': text_embeddings[i],
            'structure_tokens': struct_tokens_arr,
            'protein_sequence': protein_seq,
            'function_text': function_text,
            'question': question,
            'protein_id': protein_id
        }
        pairs.append(pair)
    
    # Shuffle and split
    shuffled_indices = np.arange(len(pairs))
    np.random.shuffle(shuffled_indices)
    pairs = [pairs[i] for i in shuffled_indices]
    
    n_val = int(len(pairs) * 0.1)
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    
    return train_pairs, val_pairs


class TriModalLlamaTrainer:
    """Trainer for Llama with LoRA using tri-modal protein embeddings"""
    
    def __init__(self, 
                 llama_model: str,
                 alignment_model: TriModalAlignmentModel,
                 tokenizer,
                 device: str = "cuda",
                 lr: float = 2e-4,
                 use_wandb: bool = True):
        """
        Args:
            llama_model: Llama model with LoRA
            alignment_model: Pre-trained projection heads (frozen)
            tokenizer: Llama tokenizer
            device: Device
            lr: Learning rate
            use_wandb: Use wandb logging
        """
        self.llama = llama_model.to(device)
        self.alignment_model = alignment_model.to(device)
        self.alignment_model.eval()  # Frozen
        self.tokenizer = tokenizer
        self.device = device
        self.use_wandb = use_wandb
        
        # Only optimize LoRA parameters
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.llama.parameters()),
            lr=lr,
            weight_decay=1e-5
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10
        )
        
        self.history = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')
        self.step = 0
    
    def construct_prompt_with_embeddings(self, batch):
        """
        Construct tri-modal prompt with embeddings injected at strategic points.
        
        Returns:
            inputs_embeds: [batch, seq_len, 4096]
            attention_mask: [batch, seq_len]
            labels: [batch, seq_len] for loss computation
        """
        batch_size = len(batch['protein_sequence'])
        
        # Project embeddings (frozen alignment model)
        with torch.no_grad():
            seq_emb = batch['sequence_emb'].to(self.device)
            struct_tokens = batch['structure_tokens'].to(self.device)
            seq_proj, struct_proj = self.alignment_model(seq_emb, struct_tokens)
            # Convert to float16 to match Llama's dtype
            seq_proj = seq_proj.to(torch.float16).unsqueeze(1)  # [batch, 1, 4096]
            struct_proj = struct_proj.to(torch.float16).unsqueeze(1)  # [batch, 1, 4096]
        
        # Construct prompts for each sample in batch
        all_embeds = []
        all_masks = []
        all_labels = []
        
        for i in range(batch_size):
            protein_name = batch['protein_id'][i]
            protein_seq = batch['protein_sequence'][i]
            function_text = batch['function_text'][i]
            question = batch['question'][i]
            
            # Part 1: Question with sequence
            part1 = f"{question} The sequence is: {protein_seq}"
            inputs1 = self.tokenizer(part1, return_tensors="pt").to(self.device)
            emb1 = self.llama.get_input_embeddings()(inputs1['input_ids'])
            
            # Part 2: [seq_embedding] + structure introduction
            part2 = " The protein has the following structure:"
            inputs2 = self.tokenizer(part2, return_tensors="pt").to(self.device)
            emb2 = self.llama.get_input_embeddings()(inputs2['input_ids'])
            
            # Part 3: [struct_embedding] + answer introduction
            part3 = " Based on the sequence and the structure, the answer is:"
            inputs3 = self.tokenizer(part3, return_tensors="pt").to(self.device)
            emb3 = self.llama.get_input_embeddings()(inputs3['input_ids'])
            
            # Part 4: Target (function description)
            part4 = f" {function_text}"
            inputs4 = self.tokenizer(part4, return_tensors="pt").to(self.device)
            emb4 = self.llama.get_input_embeddings()(inputs4['input_ids'])
            
            # Combine: text1 + seq_emb + text2 + struct_emb + text3 + target_text
            combined_emb = torch.cat([
                emb1[0],           # [len1, 4096]
                seq_proj[i],       # [1, 4096]
                emb2[0],           # [len2, 4096]
                struct_proj[i],    # [1, 4096]
                emb3[0],           # [len3, 4096]
                emb4[0]            # [len4, 4096]
            ], dim=0)  # [total_len, 4096]
            
            # Attention mask (all ones)
            mask = torch.ones(combined_emb.shape[0], dtype=torch.long, device=self.device)
            
            # Labels: -100 for prompt tokens, actual tokens for target
            prompt_len = (emb1.shape[1] + 1 + emb2.shape[1] + 1 + emb3.shape[1])
            labels = torch.full((combined_emb.shape[0],), -100, dtype=torch.long, device=self.device)
            labels[prompt_len:] = inputs4['input_ids'][0]  # Only compute loss on function text
            
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
    
    def train_epoch(self, loader, batch_idx=0):
        """Train one epoch"""
        self.llama.train()
        total_loss = 0
        
        pbar = tqdm(loader, desc="Training")
        for batch in pbar:
            # Construct tri-modal prompts with embeddings
            inputs_embeds, attention_mask, labels = self.construct_prompt_with_embeddings(batch)
            
            # Forward pass
            outputs = self.llama(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.llama.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            self.step += 1
            
            # Log
            if self.use_wandb:
                wandb.log({
                    'train_loss_batch': loss.item(),
                    'batch': batch_idx,
                    'step': self.step
                })
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(loader)
        self.history['train_loss'].append(avg_loss)
        
        if self.use_wandb:
            wandb.log({'train_loss_epoch': avg_loss, 'batch': batch_idx})
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self, loader, batch_idx=0):
        """Validate"""
        self.llama.eval()
        total_loss = 0
        
        pbar = tqdm(loader, desc="Validation")
        for batch in pbar:
            inputs_embeds, attention_mask, labels = self.construct_prompt_with_embeddings(batch)
            
            outputs = self.llama(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(loader)
        self.history['val_loss'].append(avg_loss)
        
        if self.use_wandb:
            wandb.log({'val_loss': avg_loss, 'batch': batch_idx, 'step': self.step})
        
        return avg_loss
    
    def save_checkpoint(self, path):
        """Save LoRA adapter"""
        self.llama.save_pretrained(path)
        print(f"✅ Saved LoRA adapter: {path}")


def main():
    print("=" * 70)
    print("TRAIN LLAMA-3.1 WITH LORA: TRI-MODAL PROTEIN FUNCTION")
    print("=" * 70)
    
    # Parse arguments
    if len(sys.argv) < 2:
        print(f"❌ Arguments required!")
        print(f"   Usage: python 08_train_llama_lora.py <subset_name>")
        print(f"   Example: python 08_train_llama_lora.py UniProt_Function")
        return
    
    subset_name = sys.argv[1]
    print(f"\n📌 Subset: {subset_name}")
    
    # Initialize wandb
    wandb.init(
        project="prottex-llama-lora",
        name=f"run-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            'model': 'Llama-3.1-8B',
            'method': 'LoRA',
            'reasoning': 'Tri-Modal (Seq → Emb → Text → Emb → Function)',
            'subset': subset_name
        }
    )
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path('esmfold_tokenizer/data') / subset_name / 'embedding_pairs'
    alignment_dir = Path('results/trimodal_alignment') / subset_name
    output_dir = Path('results/llama_lora') / subset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Data: {data_dir}")
    print(f"📂 Alignment model: {alignment_dir}")
    print(f"💾 Output: {output_dir}")
    print(f"🖥️  Device: {device}")
    
    # Config
    config = {
        'llama_model': 'meta-llama/Llama-3.1-8B',
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        'batch_size': 4,  # Small for memory efficiency
        'epochs': 3,
        'learning_rate': 2e-4,
        'max_length': 2048
    }
    
    # Load alignment model (frozen)
    print("\n" + "=" * 70)
    print("STEP 1: Load Pre-trained Alignment Model")
    print("=" * 70)
    
    alignment_config_path = alignment_dir / 'config.json'
    if not alignment_config_path.exists():
        print(f"❌ Alignment config not found: {alignment_config_path}")
        print("   Run script 07 first to train alignment model")
        return
    
    with open(alignment_config_path) as f:
        alignment_config = json.load(f)
    
    alignment_model = TriModalAlignmentModel(
        sequence_dim=alignment_config['sequence_dim'],
        structure_vocab_size=alignment_config['structure_vocab'],
        text_dim=alignment_config['text_dim'],
        shared_dim=alignment_config['shared_dim']
    )
    
    # Find best checkpoint
    checkpoint_files = sorted(alignment_dir.glob('best_model_global_epoch_*.pt'))
    if not checkpoint_files:
        print(f"❌ No alignment checkpoint found")
        return
    
    checkpoint_path = checkpoint_files[-1]
    print(f"✅ Loading alignment checkpoint: {checkpoint_path.name}")
    
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
    print("STEP 2: Load Llama-3.1 with LoRA")
    print("=" * 70)
    
    tokenizer = AutoTokenizer.from_pretrained(config['llama_model'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    llama = AutoModelForCausalLM.from_pretrained(
        config['llama_model'],
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Attention layers
        bias="none"
    )
    
    llama = get_peft_model(llama, lora_config)
    llama.print_trainable_parameters()
    
    print("✅ Llama loaded with LoRA")
    
    # Load data
    print("\n" + "=" * 70)
    print("STEP 3: Load Data")
    print("=" * 70)
    
    batch_files = sorted(data_dir.glob("embedding_pairs_batch_*.npz"))
    if not batch_files:
        print(f"❌ No batch files found in {data_dir}")
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
        lr=config['learning_rate']
    )
    
    print(f"\nConfig:")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   LoRA r: {config['lora_r']}, alpha: {config['lora_alpha']}")
    print(f"   Learning rate: {config['learning_rate']}\n")
    
    for epoch in range(config['epochs']):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch + 1}/{config['epochs']}")
        print(f"{'='*70}")
        
        shuffled_batch_files = batch_files.copy()
        np.random.shuffle(shuffled_batch_files)
        
        for batch_idx, batch_file in enumerate(tqdm(shuffled_batch_files[:5], desc=f"Epoch {epoch+1}")):  # Limit to 5 for demo
            try:
                train_pairs, val_pairs = load_batch_file(batch_file)
                
                train_dataset = ProteinFunctionDataset(train_pairs)
                val_dataset = ProteinFunctionDataset(val_pairs)
                
                train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
                
                train_loss = trainer.train_epoch(train_loader, batch_idx=batch_idx)
                val_loss = trainer.validate(val_loader, batch_idx=batch_idx)
                
                if val_loss < trainer.best_val_loss:
                    trainer.best_val_loss = val_loss
                    checkpoint_path = output_dir / f'best_lora_epoch_{epoch}_batch_{batch_idx}'
                    trainer.save_checkpoint(checkpoint_path)
                
            except Exception as e:
                print(f"⚠️  Error: {e}")
                continue
        
        trainer.scheduler.step()
    
    # Save final
    print("\n" + "=" * 70)
    print("STEP 5: Save Results")
    print("=" * 70)
    
    final_path = output_dir / 'final_lora'
    trainer.save_checkpoint(final_path)
    
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Training complete!")
    print(f"   Best val loss: {trainer.best_val_loss:.4f}")


if __name__ == '__main__':
    main()
