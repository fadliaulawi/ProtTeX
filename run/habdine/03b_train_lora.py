#!/usr/bin/env python3
"""
Unified LoRA Training for Protein Function Prediction (Habdine)

Habdine version: pre-split data, hardcoded data/habdine paths, CLIP alignment always used (no --no-clip-alignment).
- Input: data/habdine/triplet_embeddings/{model}/K{k}/triplet_embeddings_{train|validation|test}_batch_*.npz
- Metadata: triplet_metadata_{split}_batch_*.json
- Alignment: data/habdine/clip_alignment/{model}_K{k}
- Output: data/habdine/lora/{output_dir_suffix}/K{k}
- Test data: data/habdine/evaluation/{model}/K{k}/test_data.json

Pipeline: Load CLIP alignment (frozen) from 03a -> LoRA on tri-modal (seq_emb + structure_tokens) -> Model text space.
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
import gc
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
warnings.filterwarnings('ignore')

# Add root (run/habdine -> project root)
script_dir = Path(__file__).parent
root_dir = script_dir.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import wandb

from run.config import get_model_config, list_available_models

ESM_HIDDEN_DIM = 2560
MODEL_HIDDEN_DIM = 4096


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        return 0, 1, 0
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


class ProteinProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = MODEL_HIDDEN_DIM, hidden_dim: int = 2048):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=-1)


class StructureProjectionHead(nn.Module):
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
        return F.normalize(self.proj(pooled), p=2, dim=-1)


class TextProjectionHead(nn.Module):
    def __init__(self, input_dim: int = MODEL_HIDDEN_DIM, output_dim: int = MODEL_HIDDEN_DIM):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=-1)


class TriModalAlignmentModel(nn.Module):
    """Tri-modal alignment: Sequence + Structure (frozen, from 03a)."""

    def __init__(self, structure_vocab_size: int, sequence_dim: int = ESM_HIDDEN_DIM, text_dim: int = MODEL_HIDDEN_DIM,
                 shared_dim: int = MODEL_HIDDEN_DIM, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.shared_dim = shared_dim
        self.sequence_proj = ProteinProjectionHead(sequence_dim, shared_dim)
        self.structure_proj = StructureProjectionHead(structure_vocab_size, embedding_dim=256, output_dim=shared_dim)
        self.text_proj = TextProjectionHead(text_dim, shared_dim)

    def forward(self, sequence_emb, structure_tokens):
        seq_proj = self.sequence_proj(sequence_emb)
        struct_proj = self.structure_proj(structure_tokens)
        return seq_proj, struct_proj


class ProteinFunctionDataset(Dataset):
    def __init__(self, triplets: list, metadata: list):
        self.triplets = triplets
        self.id_to_metadata = {}
        if metadata:
            for meta_entry in metadata:
                meta_section = meta_entry.get('metadata', {})
                qa_section = meta_entry.get('QA', {})
                protein_id = meta_section.get('id', '')
                if protein_id:
                    self.id_to_metadata[protein_id] = {'QA': qa_section, 'metadata': meta_section}

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        protein_id = triplet['protein_id']
        if protein_id in self.id_to_metadata:
            meta = self.id_to_metadata[protein_id]
            qa = meta.get('QA', {})
            meta_info = meta.get('metadata', {})
            protein_seq = qa.get('sequence', 'UNKNOWN')
            question = qa.get('question', 'What is the function of this protein?')
            answer = qa.get('answer', 'Unknown function')
            protein_type = meta_info.get('type', 'PFUD')
        else:
            protein_seq, question, answer, protein_type = 'UNKNOWN', 'What is the function of this protein?', 'Unknown function', 'PFUD'
        return {
            'sequence_emb': torch.tensor(triplet['sequence_embedding'], dtype=torch.float32),
            'structure_tokens': torch.tensor(triplet['structure_tokens'], dtype=torch.long),
            'protein_sequence': protein_seq,
            'question': question,
            'function_text': answer,
            'protein_id': protein_id,
            'protein_type': protein_type
        }


def load_triplet_file(batch_file: Path, metadata_file: Path = None) -> Tuple[list, list]:
    """Load one NPZ and optional metadata. Returns (triplets, metadata). No split."""
    data = np.load(batch_file, allow_pickle=True)
    seq_embeddings = data['sequence_embeddings']
    structure_tokens = data['structure_tokens']
    protein_ids = data['protein_ids']
    text_embeddings = data['text_embeddings'] if 'text_embeddings' in data else None
    triplets = []
    for i in range(len(seq_embeddings)):
        t = {
            'sequence_embedding': seq_embeddings[i],
            'structure_tokens': structure_tokens[i],
            'protein_id': str(protein_ids[i])
        }
        if text_embeddings is not None:
            t['text_embedding'] = text_embeddings[i]
        triplets.append(t)
    metadata = []
    if metadata_file and metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    return triplets, metadata


def load_split_triplets(split_name: str, triplet_dir: Path, rank: int = 0) -> Tuple[list, list]:
    """Load pre-split triplets and metadata. triplet_embeddings_{split}_batch_*.npz, triplet_metadata_{split}_batch_*.json."""
    batch_files = sorted(triplet_dir.glob(f'triplet_embeddings_{split_name}_batch_*.npz'))
    triplets, metadata = [], []
    for batch_file in tqdm(batch_files, desc=f"Loading {split_name}", disable=(rank != 0)):
        try:
            if 'batch_' in batch_file.stem:
                part = batch_file.stem.split('batch_')[-1]
                meta_file = triplet_dir / f'triplet_metadata_{split_name}_batch_{part}.json'
            else:
                meta_file = None
            t, m = load_triplet_file(batch_file, meta_file)
            triplets.extend(t)
            metadata.extend(m)
        except Exception as e:
            if rank == 0:
                print(f"⚠️  Error loading {batch_file.name}: {e}")
    if rank == 0:
        print(f"   {split_name}: {len(triplets):,} triplets, {len(metadata):,} metadata")
    return triplets, metadata


def filter_triplets_by_type(triplets: list, all_metadata: list, rank: int = 0) -> list:
    """Exclude PDD/PSPD (answer==sequence). Same logic as run/05."""
    protein_id_to_type = {}
    for m in all_metadata:
        pid = m.get('metadata', {}).get('id', '')
        if pid:
            protein_id_to_type[pid] = m.get('metadata', {}).get('type', 'PFUD')
    filtered, skipped, type_counts = [], 0, {'PFUD': 0, 'PSAD': 0, 'PDD': 0, 'PSPD': 0}
    for t in triplets:
        pt = protein_id_to_type.get(t['protein_id'], 'PFUD')
        type_counts[pt] = type_counts.get(pt, 0) + 1
        if pt in ('PDD', 'PSPD'):
            skipped += 1
            continue
        filtered.append(t)
    if rank == 0 and (skipped or len(filtered)):
        print(f"   types: {type_counts}, filtered PDD/PSPD: {skipped}, kept: {len(filtered):,}")
    return filtered


class TriModalTrainer:
    """Trainer using pre-trained CLIP alignment (frozen) only."""

    def __init__(self, model, alignment_model: TriModalAlignmentModel, tokenizer, model_config,
                 device: str = "cuda", lr: float = 2e-4, patience: int = 3, gradient_accumulation_steps: int = 1,
                 rank: int = 0, world_size: int = 1):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.rank = rank
        self.world_size = world_size
        self.alignment_model = alignment_model.to(device)
        self.alignment_model.eval()
        for p in self.alignment_model.parameters():
            p.requires_grad = False
        self.base_model = model.module if isinstance(model, DDP) else model
        self.optimizer = torch.optim.AdamW(
            list(filter(lambda p: p.requires_grad, self.model.parameters())),
            lr=lr, weight_decay=1e-5
        )
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
        warmup, total = 500, 10000
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[
                LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup),
                CosineAnnealingLR(self.optimizer, T_max=total - warmup, eta_min=lr * 0.1)
            ],
            milestones=[warmup]
        )
        self.history = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')
        self.step = 0
        self.ema_loss = None
        self.ema_decay = 0.99
        self.patience = patience
        self.patience_counter = 0
        self.early_stop = False

    def construct_prompt_with_embeddings(self, batch):
        batch_size = len(batch['protein_sequence'])
        seq_emb = batch['sequence_emb'].to(self.device)
        struct_tokens = batch['structure_tokens'].to(self.device)
        with torch.no_grad():
            seq_proj, struct_proj = self.alignment_model(seq_emb, struct_tokens)
        seq_proj = seq_proj.to(torch.float16).unsqueeze(1)
        struct_proj = struct_proj.to(torch.float16).unsqueeze(1)
        all_embeds, all_masks, all_labels = [], [], []
        for i in range(batch_size):
            protein_seq = batch['protein_sequence'][i][:1024]
            question = batch['question'][i]
            function_text = batch['function_text'][i][:1024]
            part1, part2, part3, part4 = self.model_config.prompt_builder(question, protein_seq, function_text)
            emb1 = self.base_model.get_input_embeddings()(self.tokenizer(part1, return_tensors="pt", add_special_tokens=False).to(self.device)['input_ids'])
            emb2 = self.base_model.get_input_embeddings()(self.tokenizer(part2, return_tensors="pt", add_special_tokens=False).to(self.device)['input_ids'])
            emb3 = self.base_model.get_input_embeddings()(self.tokenizer(part3, return_tensors="pt", add_special_tokens=False).to(self.device)['input_ids'])
            emb4 = self.base_model.get_input_embeddings()(self.tokenizer(part4, return_tensors="pt", add_special_tokens=False).to(self.device)['input_ids'])
            combined = torch.cat([emb1[0], seq_proj[i], emb2[0], struct_proj[i], emb3[0], emb4[0]], dim=0)
            mask = torch.ones(combined.shape[0], dtype=torch.long, device=self.device)
            prompt_len = emb1.shape[1] + 1 + emb2.shape[1] + 1 + emb3.shape[1]
            labels = torch.full((combined.shape[0],), -100, dtype=torch.long, device=self.device)
            labels[prompt_len:] = self.tokenizer(part4, return_tensors="pt", add_special_tokens=False).to(self.device)['input_ids'][0]
            all_embeds.append(combined)
            all_masks.append(mask)
            all_labels.append(labels)
        max_len = max(e.shape[0] for e in all_embeds)
        padded_emb, padded_masks, padded_labels = [], [], []
        for emb, mask, label in zip(all_embeds, all_masks, all_labels):
            pad = max_len - emb.shape[0]
            if pad > 0:
                emb = torch.cat([emb, torch.zeros(pad, emb.shape[1], device=self.device, dtype=emb.dtype)], dim=0)
                mask = torch.cat([mask, torch.zeros(pad, device=self.device, dtype=mask.dtype)], dim=0)
                label = torch.cat([label, torch.full((pad,), -100, device=self.device, dtype=label.dtype)], dim=0)
            padded_emb.append(emb)
            padded_masks.append(mask)
            padded_labels.append(label)
        return torch.stack(padded_emb), torch.stack(padded_masks), torch.stack(padded_labels)

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        pbar = tqdm(loader, desc="Training", disable=(self.rank != 0))
        for batch_idx, batch in enumerate(pbar):
            inputs_embeds, attention_mask, labels = self.construct_prompt_with_embeddings(batch)
            outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.step += 1
                raw = loss.item() * self.gradient_accumulation_steps
                self.ema_loss = raw if self.ema_loss is None else self.ema_decay * self.ema_loss + (1 - self.ema_decay) * raw
                if self.rank == 0:
                    wandb.log({'train/batch_loss': raw, 'train/batch_loss_smooth': self.ema_loss, 'train/step': self.step, 'train/learning_rate': self.optimizer.param_groups[0]['lr']})
            total_loss += loss.item() * self.gradient_accumulation_steps
            pbar.set_postfix({'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}'})
            if batch_idx % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        self.history['train_loss'].append(total_loss / len(loader))
        return total_loss / len(loader)

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        gc.collect()
        torch.cuda.empty_cache()
        pbar = tqdm(loader, desc="Validation", disable=(self.rank != 0))
        for batch_idx, batch in enumerate(pbar):
            try:
                inputs_embeds, attention_mask, labels = self.construct_prompt_with_embeddings(batch)
                logits = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                if self.rank == 0:
                    print("\n❌ OOM in validation. Reduce batch_size.")
                raise
        self.history['val_loss'].append(total_loss / len(loader))
        return total_loss / len(loader)

    def check_early_stopping(self, val_loss, output_dir, k_clusters):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            if self.rank == 0:
                self.save_checkpoint(output_dir / f'{self.model_config.checkpoint_prefix}_K{k_clusters}')
                print(f"   🎯 New best val loss: {val_loss:.4f}")
            return False
        self.patience_counter += 1
        print(f"   ⚠️  No improvement {self.patience_counter}/{self.patience}")
        if self.patience_counter >= self.patience:
            self.early_stop = True
            return True
        return False

    def save_checkpoint(self, path):
        self.base_model.save_pretrained(path)
        if self.rank == 0:
            print(f"✅ Saved LoRA: {path}")


def main():
    rank, world_size, local_rank = setup_distributed()
    parser = argparse.ArgumentParser(description='Train LoRA for protein function (Habdine, pre-split)')
    parser.add_argument('--model', type=str, required=True, choices=list_available_models())
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()
    model_config = get_model_config(args.model)
    k_clusters = args.k

    if rank == 0:
        print("=" * 70)
        print(f"TRAIN {args.model.upper()} WITH LORA (Habdine)")
        print("=" * 70)

    batch_size = 6
    gradient_accumulation_steps = 4
    learning_rate = 1.5e-4
    lora_r, lora_alpha, patience = 16, 32, 2

    if rank == 0:
        wandb_api_key = os.getenv('WANDB_API_KEY')
        if not wandb_api_key:
            raise ValueError("❌ WANDB_API_KEY not set")
        wandb.login(key=wandb_api_key)
        wandb.init(project='prottex-lora', name=f"run-{args.model}-lora-K{k_clusters}-ddp-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                   config={'k_clusters': k_clusters, 'epochs': args.epochs, 'batch_size_per_gpu': batch_size, 'gradient_accumulation_steps': gradient_accumulation_steps,
                           'effective_batch_size': batch_size * gradient_accumulation_steps * world_size, 'num_gpus': world_size, 'learning_rate': learning_rate,
                           'lora_r': lora_r, 'lora_alpha': lora_alpha, 'patience': patience, 'model': model_config.model_name, 'model_type': args.model, 'distributed': 'DDP'})

    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)

    # Hardcoded data/habdine paths
    triplet_dir = Path('data/habdine/triplet_embeddings') / args.model / f'K{k_clusters}'
    alignment_dir = Path('data/habdine/clip_alignment') / f'{args.model}_K{k_clusters}'
    output_dir = Path('data/habdine/lora') / model_config.output_dir_suffix / f'K{k_clusters}'
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = Path('data/habdine/evaluation') / args.model / f'K{k_clusters}'

    if rank == 0:
        print(f"\n📂 Triplets: {triplet_dir}")
        print(f"📂 Alignment: {alignment_dir}")
        print(f"💾 Output: {output_dir}")

    # STEP 1: Load CLIP alignment (required)
    if rank == 0:
        print("\n" + "=" * 70)
        print("STEP 1: Load Pre-trained CLIP Alignment (03a)")
        print("=" * 70)
    config_path = alignment_dir / f'config_K{k_clusters}.json'
    if not config_path.exists():
        if rank == 0:
            print(f"❌ Config not found: {config_path}. Run 03a_train_clip_alignment.py first.")
        cleanup_distributed()
        return
    with open(config_path) as f:
        clip_config = json.load(f)
    alignment_model = TriModalAlignmentModel(
        structure_vocab_size=clip_config['vocab_size'],
        sequence_dim=clip_config['sequence_dim'],
        text_dim=clip_config['text_dim'],
        shared_dim=clip_config['shared_dim'],
        temperature=clip_config.get('temperature', 0.07)
    )
    ckpt = alignment_dir / f'best_model_K{k_clusters}.pt'
    if not ckpt.exists():
        if rank == 0:
            print(f"❌ Checkpoint not found: {ckpt}")
        cleanup_distributed()
        return
    alignment_model.load_state_dict(torch.load(ckpt, map_location=device)['model_state'])
    alignment_model.to(device)
    alignment_model.eval()
    for p in alignment_model.parameters():
        p.requires_grad = False
    if rank == 0:
        print("✅ CLIP alignment loaded and frozen")

    # STEP 2: Load model + LoRA
    if rank == 0:
        print("\n" + "=" * 70)
        print(f"STEP 2: Load {args.model.upper()} with LoRA")
        print("=" * 70)
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name, trust_remote_code=model_config.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_cache=False, trust_remote_code=model_config.trust_remote_code)
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], bias="none"))
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    if rank == 0:
        print("✅ Model + LoRA + DDP")

    # STEP 3: Load pre-split data and filter PDD/PSPD
    if rank == 0:
        print("\n" + "=" * 70)
        print("STEP 3: Load Pre-Split Triplets")
        print("=" * 70)
    train_triplets, train_meta = load_split_triplets('train', triplet_dir, rank)
    val_triplets, val_meta = load_split_triplets('validation', triplet_dir, rank)
    test_triplets, test_meta = load_split_triplets('test', triplet_dir, rank)
    all_metadata = train_meta + val_meta + test_meta
    if not train_triplets:
        if rank == 0:
            print(f"❌ No train triplets in {triplet_dir}. Run 02_extract_embeddings.py first.")
        cleanup_distributed()
        return
    if rank == 0:
        print("   Filtering PDD/PSPD...")
    train_triplets = filter_triplets_by_type(train_triplets, all_metadata, rank)
    val_triplets = filter_triplets_by_type(val_triplets, all_metadata, rank)
    test_triplets = filter_triplets_by_type(test_triplets, all_metadata, rank)
    if rank == 0:
        print(f"   Train: {len(train_triplets):,}, Val: {len(val_triplets):,}, Test: {len(test_triplets):,}")

    # Save test_data.json (epoch 0, rank 0)
    protein_id_to_meta = {}
    for m in all_metadata:
        pid = m.get('metadata', {}).get('id', '')
        if pid:
            protein_id_to_meta[pid] = m
    test_data = []
    for t in test_triplets:
        pid = t['protein_id']
        if pid in protein_id_to_meta:
            qa = protein_id_to_meta[pid].get('QA', {})
            meta = protein_id_to_meta[pid].get('metadata', {})
            test_data.append({'id': pid, 'sequence': qa.get('sequence', ''), 'question': qa.get('question', ''), 'function': qa.get('answer', ''), 'type': meta.get('type', 'PFUD'), 'subset': meta.get('subset', '')})
    if rank == 0:
        eval_dir.mkdir(parents=True, exist_ok=True)
        with open(eval_dir / 'test_data.json', 'w') as f:
            json.dump(test_data, f, indent=2)
        print(f"💾 Saved test_data: {eval_dir / 'test_data.json'} ({len(test_data)} samples)")

    # STEP 4: Training
    if rank == 0:
        print("\n" + "=" * 70)
        print("STEP 4: Training")
        print("=" * 70)
    trainer = TriModalTrainer(model=model, alignment_model=alignment_model, tokenizer=tokenizer, model_config=model_config, device=device, lr=learning_rate, patience=patience, gradient_accumulation_steps=gradient_accumulation_steps, rank=rank, world_size=world_size)

    for epoch in range(args.epochs):
        if rank == 0:
            print(f"\n{'='*70}\nEPOCH {epoch + 1}/{args.epochs}\n{'='*70}")
        train_dataset = ProteinFunctionDataset(train_triplets, all_metadata)
        val_dataset = ProteinFunctionDataset(val_triplets, all_metadata)
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=2, pin_memory=True)
        train_sampler.set_epoch(epoch)
        if rank == 0:
            print(f"   Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        train_loss = trainer.train_epoch(train_loader)
        if dist.is_initialized():
            dist.barrier()
        val_loss = trainer.validate(val_loader)
        if dist.is_initialized():
            v = torch.tensor([val_loss], device=device)
            dist.all_reduce(v, op=dist.ReduceOp.AVG)
            val_loss = v.item()
        if rank == 0:
            print(f"   Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
            wandb.log({'epoch': epoch + 1, 'train/epoch_loss': train_loss, 'val/epoch_loss': val_loss, 'val/best_val_loss': trainer.best_val_loss})
        if trainer.check_early_stopping(val_loss, output_dir, k_clusters):
            break

    if rank == 0:
        print("\n" + "=" * 70)
        print("STEP 5: Save Results")
        print("=" * 70)
        trainer.save_checkpoint(output_dir / f'final_lora_K{k_clusters}')
        with open(output_dir / f'config_K{k_clusters}.json', 'w') as f:
            json.dump({'k_clusters': k_clusters, 'epochs': args.epochs, 'batch_size_per_gpu': batch_size, 'gradient_accumulation_steps': gradient_accumulation_steps, 'effective_batch_size': batch_size * gradient_accumulation_steps * world_size, 'num_gpus': world_size, 'learning_rate': learning_rate, 'lora_r': lora_r, 'lora_alpha': lora_alpha, 'patience': patience, 'best_val_loss': trainer.best_val_loss, 'timestamp': datetime.now().isoformat(), 'distributed': 'DDP', 'model': model_config.model_name, 'model_type': args.model, 'train_samples': len(train_triplets), 'val_samples': len(val_triplets), 'test_samples': len(test_triplets)}, f, indent=2)
        print("✅ Training complete. Best val loss:", trainer.best_val_loss)
        wandb.finish()
    cleanup_distributed()


if __name__ == '__main__':
    main()
