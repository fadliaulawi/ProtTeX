#!/usr/bin/env python3
"""
End-to-End Inference: Protein Sequence + Question → Function Prediction

Takes a dictionary of {sequence: text/question} and generates function predictions
using the trained LoRA model.

Pipeline:
1. Extract ESM-2 sequence embeddings
2. Generate structure tokens using k-means codebook
3. Project embeddings via CLIP alignment model
4. Generate text using Llama-3.1 with LoRA adapters
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Union
import argparse
import warnings
from tqdm import tqdm
import pickle
import sys
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModelForCausalLM, EsmModel
from peft import PeftModel
from rouge_score import rouge_scorer

# Add root directory to Python path for imports
script_dir = Path(__file__).parent
root_dir = script_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Import model configuration
try:
    from run.config import get_model_config, list_available_models
except ImportError:
    # Fallback for direct execution
    import importlib.util
    config_path = script_dir / 'config.py'
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    get_model_config = config_module.get_model_config
    list_available_models = config_module.list_available_models

# Embedding dimensions (must match training)
ESM_HIDDEN_DIM = 2560  # ESM-2 3B
# Model hidden dim will be set from config


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
        proj = self.proj(x)
        return F.normalize(proj, p=2, dim=-1)


class StructureProjectionHead(nn.Module):
    """Project structure tokens to model space"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256, output_dim: int = 4096, hidden_dim: int = 2048):
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
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
    
    def forward(self, x):
        proj = self.proj(x)
        return F.normalize(proj, p=2, dim=-1)


class TriModalAlignmentModel(nn.Module):
    """Tri-modal alignment: Sequence + Structure → Text (frozen, pre-trained)"""
    
    def __init__(self, 
                 structure_vocab_size: int,
                 sequence_dim: int = ESM_HIDDEN_DIM,
                 text_dim: int = 4096,
                 shared_dim: int = 4096,
                 temperature: float = 0.07):
        super().__init__()
        
        self.temperature = temperature
        self.shared_dim = shared_dim
        
        self.sequence_proj = ProteinProjectionHead(sequence_dim, shared_dim)
        self.structure_proj = StructureProjectionHead(structure_vocab_size, embedding_dim=256, output_dim=shared_dim)
        self.text_proj = TextProjectionHead(text_dim, shared_dim)
    
    def forward(self, sequence_emb, structure_tokens):
        """
        Args:
            sequence_emb: [batch, 2560] ESM-2 sequence embeddings
            structure_tokens: [batch, seq_len] Structure token IDs
        
        Returns:
            seq_proj: [batch, shared_dim] Projected sequence embedding
            struct_proj: [batch, shared_dim] Projected structure embedding
        """
        seq_proj = self.sequence_proj(sequence_emb)
        struct_proj = self.structure_proj(structure_tokens)
        return seq_proj, struct_proj


class ProteinFunctionInference:
    """End-to-end inference pipeline"""
    
    def __init__(self, 
                 model_config,
                 lora_path: str = None,
                 alignment_model_path: str = None,
                 codebook_path: str = None,
                 k_clusters: int = 128,
                 device: str = "cuda"):
        """
        Initialize inference pipeline.
        
        Args:
            model_config: ModelConfig object with model-specific settings
            lora_path: Path to trained LoRA adapters (final_lora_K{k}), or None to use base model
            alignment_model_path: Path to CLIP alignment model checkpoint
            codebook_path: Path to k-means codebook (.pkl)
            k_clusters: Number of k-means clusters
            device: 'cuda' or 'cpu'
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.k_clusters = k_clusters
        self.model_config = model_config
        
        print("=" * 70)
        print("INITIALIZING INFERENCE PIPELINE")
        print("=" * 70)
        
        # Load ESM-2 model for sequence embeddings
        print("\n📥 Loading ESM-2 3B model...")
        self.esm_model_name = "facebook/esm2_t36_3B_UR50D"
        self.esm_tokenizer = AutoTokenizer.from_pretrained(self.esm_model_name)
        self.esm_model = EsmModel.from_pretrained(self.esm_model_name).to(self.device)
        self.esm_model.eval()
        print(f"✅ ESM-2 loaded (dim: {self.esm_model.config.hidden_size})")
        
        # Load k-means codebook
        print(f"\n📥 Loading k-means codebook from {codebook_path}...")
        with open(codebook_path, 'rb') as f:
            codebook_data = pickle.load(f)
        self.kmeans = codebook_data['kmeans']
        print(f"✅ Codebook loaded ({self.k_clusters} clusters)")
        
        # Load CLIP alignment model
        print(f"\n📥 Loading CLIP alignment model from {alignment_model_path}...")
        alignment_config_path = Path(alignment_model_path).parent / f"config_K{k_clusters}.json"
        with open(alignment_config_path) as f:
            clip_config = json.load(f)
        
        # Use model_config.hidden_dim for text_dim and shared_dim if not in config
        text_dim = clip_config.get('text_dim', self.model_config.hidden_dim)
        shared_dim = clip_config.get('shared_dim', self.model_config.hidden_dim)
        
        self.alignment_model = TriModalAlignmentModel(
            structure_vocab_size=clip_config['vocab_size'],
            sequence_dim=clip_config['sequence_dim'],
            text_dim=text_dim,
            shared_dim=shared_dim,
            temperature=clip_config['temperature']
        )
        
        checkpoint = torch.load(alignment_model_path, map_location=self.device)
        self.alignment_model.load_state_dict(checkpoint['model_state'])
        self.alignment_model.to(self.device)
        self.alignment_model.eval()
        print("✅ Alignment model loaded")
        
        # Load model with optional LoRA (use Instruct model to match training)
        model_name = self.model_config.model_name
        if lora_path:
            print(f"\n📥 Loading {model_name} with LoRA from {lora_path}...")
        else:
            print(f"\n📥 Loading {model_name} base model (no LoRA adapters)...")
            
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self.model_config.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map={"": self.device},
            trust_remote_code=self.model_config.trust_remote_code
        )
        
        # Try loading LoRA adapters if path provided
        if lora_path:
            try:
                self.model = PeftModel.from_pretrained(base_model, lora_path)
                print("✅ LoRA adapters loaded")
            except Exception as e:
                print(f"⚠️  Warning: Failed to load LoRA adapters: {e}")
                print("   Using base model without LoRA adapters")
                self.model = base_model
        else:
            self.model = base_model
            print("✅ Using base model (no LoRA adapters)")
        
        self.model.eval()
        self.base_model = self.model.get_base_model()
        
        # Unwrap DDP if needed (model might be saved with DDP wrapper)
        if hasattr(self.model, 'module'):
            self.model = self.model.module
        if hasattr(self.base_model, 'module'):
            self.base_model = self.base_model.module
        
        # Load separate model for plain baseline (completely separate, no LoRA)
        print(f"\n📥 Loading plain {model_name} for baseline (separate model)...")
        self.baseline_model_name = model_name
        self.baseline_tokenizer = AutoTokenizer.from_pretrained(
            self.baseline_model_name,
            trust_remote_code=self.model_config.trust_remote_code
        )
        if self.baseline_tokenizer.pad_token is None:
            self.baseline_tokenizer.pad_token = self.baseline_tokenizer.eos_token
        
        self.baseline_model = AutoModelForCausalLM.from_pretrained(
            self.baseline_model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map={"": self.device},
            trust_remote_code=self.model_config.trust_remote_code
        )
        self.baseline_model.eval()
        
        print("✅ Both models loaded (tri-modal with LoRA, plain model for baseline)")
        
        # AA vocabulary for structure tokens
        self.aa_vocab = "ACDEFGHIKLMNPQRSTVWY"
        self.aa_to_id = {aa: i for i, aa in enumerate(self.aa_vocab)}
        self.token_offset = 20  # Structure tokens start after AA tokens
        
        print("\n" + "=" * 70)
        print("✅ INFERENCE PIPELINE READY")
        print("=" * 70)
    
    def _get_stop_token_ids(self, tokenizer):
        """
        Get stop token IDs for the model, trying model-specific tokens.
        
        Returns:
            List of stop token IDs
        """
        stop_tokens = [tokenizer.eos_token_id]
        
        # Try model-specific stop tokens
        # Llama and DeepSeek use <|eot_id|>
        eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_token_id is not None and eot_token_id != tokenizer.unk_token_id:
            stop_tokens.append(eot_token_id)
        
        # Qwen uses <|im_end|>
        im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_token_id is not None and im_end_token_id != tokenizer.unk_token_id:
            stop_tokens.append(im_end_token_id)
        
        return stop_tokens
    
    def generate(self, 
                 sequence: str,
                 question: str = "What is the function of this protein?",
                 max_new_tokens: int = 256,
                 temperature: float = 0.7,
                 top_p: float = 0.9) -> str:
        """
        Variant 4: Fine-tuned Llama (question + sequence + structure + both embeddings).
        Uses embeddings with fine-tuned LoRA adapters.
        
        Args:
            sequence: Protein sequence string
            question: Question/prompt about the protein
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        
        Returns:
            Generated function text
        """
        if self.alignment_model is None or self.esm_model is None or self.kmeans is None:
            raise ValueError("Tri-modal generation requires alignment_model, esm_model, and kmeans to be loaded")
        sequence = sequence[:1024]  # Truncate for memory
        
        # Extract per-residue embeddings (used for both mean-pooling and structure tokens)
        inputs = self.esm_tokenizer([sequence], return_tensors="pt", padding=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.esm_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            per_residue_embeddings = hidden_states[0, 1:-1, :].cpu().numpy()  # [seq_len, 2560]
        
        # Extract sequence embedding (mean-pooled)
        seq_emb = per_residue_embeddings.mean(axis=0)  # [2560]
        
        # Get structure tokens (predict clusters and add token offset)
        cluster_ids = self.kmeans.predict(per_residue_embeddings)
        structure_tokens = torch.tensor(
            [self.token_offset + int(cid) for cid in cluster_ids],
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)  # [1, seq_len]
        
        # Project embeddings via alignment model
        with torch.no_grad():
            seq_emb_tensor = torch.tensor(seq_emb, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, 2560]
            seq_proj, struct_proj = self.alignment_model(seq_emb_tensor, structure_tokens)
            hidden_dim = self.model_config.hidden_dim
            seq_proj = seq_proj.to(torch.float16).unsqueeze(1)  # [1, 1, hidden_dim]
            struct_proj = struct_proj.to(torch.float16).unsqueeze(1)  # [1, 1, hidden_dim]
        
        # Construct prompt with embeddings using model-specific format
        # Use prompt builder but without function_text (for inference)
        part1, part2, part3, _ = self.model_config.prompt_builder(question, sequence, "")
        # Remove the function_text part (part4) since we're generating it
        
        inputs1 = self.tokenizer(part1, return_tensors="pt", add_special_tokens=False).to(self.device)
        emb1 = self.base_model.get_input_embeddings()(inputs1['input_ids'])
        
        inputs2 = self.tokenizer(part2, return_tensors="pt", add_special_tokens=False).to(self.device)
        emb2 = self.base_model.get_input_embeddings()(inputs2['input_ids'])
        
        inputs3 = self.tokenizer(part3, return_tensors="pt", add_special_tokens=False).to(self.device)
        emb3 = self.base_model.get_input_embeddings()(inputs3['input_ids'])
        
        # Combine embeddings (match training format)
        combined_emb = torch.cat([
            emb1[0],      # [len1, hidden_dim] - chat header + question + sequence
            seq_proj[0],  # [1, hidden_dim] - projected sequence embedding
            emb2[0],      # [len2, hidden_dim] - structure marker
            struct_proj[0],  # [1, hidden_dim] - projected structure embedding
            emb3[0]       # [len3, hidden_dim] - assistant turn start
        ], dim=0).unsqueeze(0)  # [1, total_len, hidden_dim]
        
        attention_mask = torch.ones(combined_emb.shape[1], dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Generate - get stop tokens for this model
        stop_token_ids = self._get_stop_token_ids(self.tokenizer)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=combined_emb,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=stop_token_ids,
                use_cache=True
            )
        
        # When using inputs_embeds, generate() returns ONLY the newly generated tokens
        generated_ids = outputs[0]  # All output tokens are newly generated
        
        # Filter out special tokens before decoding
        special_ids = set(stop_token_ids + [self.tokenizer.pad_token_id])
        generated_ids = torch.tensor([t for t in generated_ids.tolist() if t not in special_ids], device=self.device)
        
        if len(generated_ids) == 0:
            return ""  # No generation
        
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text.strip()
    
    def generate_plain_llama(self,
                            sequence: str,
                            question: str = "What is the function of this protein?",
                            max_new_tokens: int = 256,
                            temperature: float = 0.7,
                            top_p: float = 0.9) -> str:
        """
        Variant 1: Plain Llama (question + sequence only).
        Baseline with no structure or embeddings.
        
        Args:
            sequence: Protein sequence string
            question: Question/prompt about the protein
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        
        Returns:
            Generated function text
        """
        sequence = sequence[:1024]  # Truncate for memory
        
        # Use proper chat template (model-agnostic)
        messages = [
            {"role": "user", "content": f"{question}\n\nSequence: {sequence}"}
        ]
        prompt = self.baseline_tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.baseline_tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        
        # Get stop tokens
        stop_token_ids = self._get_stop_token_ids(self.baseline_tokenizer)
        pad_token_id = self.baseline_tokenizer.pad_token_id
        
        # Use separate baseline model (completely separate from LoRA model)
        with torch.no_grad():
            outputs = self.baseline_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(0.1, temperature),
                top_p=top_p,
                do_sample=(temperature > 0.1),
                pad_token_id=pad_token_id,
                eos_token_id=stop_token_ids,
                use_cache=True
            )
        
        # Decode generated tokens (only new tokens after prompt)
        prompt_length = inputs['input_ids'].shape[1]
        
        # Get only new tokens
        if outputs.shape[1] > prompt_length:
            generated_ids = outputs[0, prompt_length:]
            
            # Filter out special tokens
            special_ids = set(stop_token_ids + [pad_token_id])
            generated_ids = torch.tensor([t for t in generated_ids.tolist() if t not in special_ids], device=self.device)
            
            if len(generated_ids) == 0:
                return ""  # No generation
            
            generated_text = self.baseline_tokenizer.decode(generated_ids, skip_special_tokens=True)
            return generated_text.strip()
        else:
            return ""  # No generation happened
    
    def generate_plain_llama_with_structure(self,
                                           sequence: str,
                                           question: str = "What is the function of this protein?",
                                           max_new_tokens: int = 256,
                                           temperature: float = 0.7,
                                           top_p: float = 0.9) -> str:
        """
        Variant 2: Plain Llama (question + sequence + structure tokens as text).
        Adds structure information as text tokens, but no embeddings.
        
        Args:
            sequence: Protein sequence string
            question: Question/prompt about the protein
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        
        Returns:
            Generated function text
        """
        if self.esm_model is None or self.kmeans is None:
            raise ValueError("Structure tokens require esm_model and kmeans to be loaded")
        
        sequence = sequence[:1024]  # Truncate for memory
        
        # Extract per-residue embeddings to get structure tokens
        inputs = self.esm_tokenizer([sequence], return_tensors="pt", padding=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.esm_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            per_residue_embeddings = hidden_states[0, 1:-1, :].cpu().numpy()  # [seq_len, 2560]
        
        # Get structure tokens (cluster IDs)
        cluster_ids = self.kmeans.predict(per_residue_embeddings)
        # Convert to space-separated string representation
        structure_str = " ".join([str(int(cid)) for cid in cluster_ids[:512]])  # Limit length
        
        # Use proper chat template (model-agnostic)
        messages = [
            {"role": "user", "content": f"{question}\n\nSequence: {sequence}\n\nStructure: {structure_str}"}
        ]
        prompt = self.baseline_tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.baseline_tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        
        # Get stop tokens
        stop_token_ids = self._get_stop_token_ids(self.baseline_tokenizer)
        pad_token_id = self.baseline_tokenizer.pad_token_id
        
        # Use separate baseline model
        with torch.no_grad():
            outputs = self.baseline_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(0.1, temperature),
                top_p=top_p,
                do_sample=(temperature > 0.1),
                pad_token_id=pad_token_id,
                eos_token_id=stop_token_ids,
                use_cache=True
            )
        
        # Decode generated tokens
        prompt_length = inputs['input_ids'].shape[1]
        
        if outputs.shape[1] > prompt_length:
            generated_ids = outputs[0, prompt_length:]
            special_ids = set(stop_token_ids + [pad_token_id])
            generated_ids = torch.tensor([t for t in generated_ids.tolist() if t not in special_ids], device=self.device)
            
            if len(generated_ids) == 0:
                return ""
            
            generated_text = self.baseline_tokenizer.decode(generated_ids, skip_special_tokens=True)
            return generated_text.strip()
        else:
            return ""
    
    def generate_plain_llama_with_embeddings(self,
                                            sequence: str,
                                            question: str = "What is the function of this protein?",
                                            max_new_tokens: int = 256,
                                            temperature: float = 0.7,
                                            top_p: float = 0.9) -> str:
        """
        Variant 3: Plain Llama (question + sequence + structure + both embeddings).
        Uses embeddings but with plain (non-fine-tuned) model.
        
        Args:
            sequence: Protein sequence string
            question: Question/prompt about the protein
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        
        Returns:
            Generated function text
        """
        if self.alignment_model is None or self.esm_model is None or self.kmeans is None:
            raise ValueError("Embeddings require alignment_model, esm_model, and kmeans to be loaded")
        
        sequence = sequence[:1024]  # Truncate for memory
        
        # Extract per-residue embeddings
        inputs = self.esm_tokenizer([sequence], return_tensors="pt", padding=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.esm_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            per_residue_embeddings = hidden_states[0, 1:-1, :].cpu().numpy()  # [seq_len, 2560]
        
        # Extract sequence embedding (mean-pooled)
        seq_emb = per_residue_embeddings.mean(axis=0)  # [2560]
        
        # Get structure tokens
        cluster_ids = self.kmeans.predict(per_residue_embeddings)
        structure_tokens = torch.tensor(
            [self.token_offset + int(cid) for cid in cluster_ids],
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)  # [1, seq_len]
        
        # Project embeddings via alignment model
        with torch.no_grad():
            seq_emb_tensor = torch.tensor(seq_emb, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, 2560]
            seq_proj, struct_proj = self.alignment_model(seq_emb_tensor, structure_tokens)
            hidden_dim = self.model_config.hidden_dim
            seq_proj = seq_proj.to(torch.float16).unsqueeze(1)  # [1, 1, hidden_dim]
            struct_proj = struct_proj.to(torch.float16).unsqueeze(1)  # [1, 1, hidden_dim]
        
        # Construct prompt with embeddings using model-specific format (same as fine-tuned version)
        # Use prompt builder but without function_text (for inference)
        part1, part2, part3, _ = self.model_config.prompt_builder(question, sequence, "")
        # Remove the function_text part (part4) since we're generating it
        
        inputs1 = self.baseline_tokenizer(part1, return_tensors="pt", add_special_tokens=False).to(self.device)
        emb1 = self.baseline_model.get_input_embeddings()(inputs1['input_ids'])
        
        inputs2 = self.baseline_tokenizer(part2, return_tensors="pt", add_special_tokens=False).to(self.device)
        emb2 = self.baseline_model.get_input_embeddings()(inputs2['input_ids'])
        
        inputs3 = self.baseline_tokenizer(part3, return_tensors="pt", add_special_tokens=False).to(self.device)
        emb3 = self.baseline_model.get_input_embeddings()(inputs3['input_ids'])
        
        # Combine embeddings
        combined_emb = torch.cat([
            emb1[0],
            seq_proj[0],
            emb2[0],
            struct_proj[0],
            emb3[0]
        ], dim=0).unsqueeze(0)
        
        attention_mask = torch.ones(combined_emb.shape[1], dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Generate using plain baseline model (not fine-tuned)
        stop_token_ids = self._get_stop_token_ids(self.baseline_tokenizer)
        
        with torch.no_grad():
            outputs = self.baseline_model.generate(
                inputs_embeds=combined_emb,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.baseline_tokenizer.pad_token_id,
                eos_token_id=stop_token_ids,
                use_cache=True
            )
        
        generated_ids = outputs[0]
        special_ids = set(stop_token_ids + [self.baseline_tokenizer.pad_token_id])
        generated_ids = torch.tensor([t for t in generated_ids.tolist() if t not in special_ids], device=self.device)
        
        if len(generated_ids) == 0:
            return ""
        
        generated_text = self.baseline_tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text.strip()
    
    def generate_finetuned_llama_with_structure(self,
                                                sequence: str,
                                                question: str = "What is the function of this protein?",
                                                max_new_tokens: int = 256,
                                                temperature: float = 0.7,
                                                top_p: float = 0.9) -> str:
        """
        Variant 4: Fine-tuned Llama (question + sequence + structure as text, no embeddings).
        Uses fine-tuned LoRA adapters but structure as text tokens, not embeddings.
        This isolates the contribution of fine-tuning vs embeddings.
        
        Args:
            sequence: Protein sequence string
            question: Question/prompt about the protein
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        
        Returns:
            Generated function text
        """
        if self.esm_model is None or self.kmeans is None:
            raise ValueError("Structure tokens require esm_model and kmeans to be loaded")
        
        sequence = sequence[:1024]  # Truncate for memory
        
        # Extract per-residue embeddings to get structure tokens
        inputs = self.esm_tokenizer([sequence], return_tensors="pt", padding=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.esm_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            per_residue_embeddings = hidden_states[0, 1:-1, :].cpu().numpy()  # [seq_len, 2560]
        
        # Get structure tokens (cluster IDs)
        cluster_ids = self.kmeans.predict(per_residue_embeddings)
        # Convert to space-separated string representation
        structure_str = " ".join([str(int(cid)) for cid in cluster_ids[:512]])  # Limit length
        
        # Use proper chat template (model-agnostic)
        messages = [
            {"role": "user", "content": f"{question}\n\nSequence: {sequence}\n\nStructure: {structure_str}"}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        
        # Get stop tokens
        stop_token_ids = self._get_stop_token_ids(self.tokenizer)
        pad_token_id = self.tokenizer.pad_token_id
        
        # Use fine-tuned model (with LoRA adapters)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(0.1, temperature),
                top_p=top_p,
                do_sample=(temperature > 0.1),
                pad_token_id=pad_token_id,
                eos_token_id=stop_token_ids,
                use_cache=True
            )
        
        # Decode generated tokens
        prompt_length = inputs['input_ids'].shape[1]
        
        if outputs.shape[1] > prompt_length:
            generated_ids = outputs[0, prompt_length:]
            special_ids = set(stop_token_ids + [pad_token_id])
            generated_ids = torch.tensor([t for t in generated_ids.tolist() if t not in special_ids], device=self.device)
            
            if len(generated_ids) == 0:
                return ""
            
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return generated_text.strip()
        else:
            return ""

def main():
    parser = argparse.ArgumentParser(description='Protein function prediction inference')
    parser.add_argument('--model', type=str, required=True,
                       choices=list_available_models(),
                       help=f'Model type: {", ".join(list_available_models())}')
    parser.add_argument('--k', type=int, default=128, help='K-means clusters (must match training)')
    parser.add_argument('--input', type=str, default='run/input_inference.json', help='Input JSON file: List of objects with "id", "sequence", "question", "function" fields')
    parser.add_argument('--output', type=str, default='run/output_inference.json', help='Output JSON file')
    parser.add_argument('--max-tokens', type=int, default=1024, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.9, help='Nucleus sampling top-p')
    parser.add_argument('--test-base-only', action='store_true', help='Test base model without LoRA adapters')
    parser.add_argument('--test-batches', type=int, default=4,
                       help='Number of last batches held out during training (default: 4)')
    parser.add_argument('--variants', type=str, default='all',
                       help='Variants to run (default: all). Options: "all", "1", "2", "3", "4", "5", or comma-separated like "1,3,5"')
    parser.add_argument('--start-index', type=int, default=None,
                       help='Start index for processing input list (for parallelization). If None, processes all.')
    parser.add_argument('--end-index', type=int, default=None,
                       help='End index for processing input list (for parallelization). If None, processes all.')
    parser.add_argument('--gpu-id', type=int, default=None,
                       help='GPU ID for output filename (for parallelization). If None, uses base output filename.')
    
    args = parser.parse_args()
    
    # Parse variants argument
    if args.variants.lower() == 'all':
        variants_to_run = [1, 2, 3, 4, 5]
    else:
        try:
            variants_to_run = [int(v.strip()) for v in args.variants.split(',')]
            if not all(1 <= v <= 5 for v in variants_to_run):
                print("❌ Invalid variant numbers. Must be between 1 and 5.")
                return
        except ValueError:
            print(f"❌ Invalid variants format: {args.variants}. Use 'all' or comma-separated numbers like '1,3,5'")
            return
    
    # Get model configuration
    model_config = get_model_config(args.model)
    k_clusters = args.k
    
    # Paths (use model-specific paths)
    data_dir = Path('data')
    lora_path = data_dir / f'lora/{model_config.output_dir_suffix}/K{k_clusters}_test{args.test_batches}/final_lora_K{k_clusters}'
    alignment_model_path = data_dir / f'clip_alignment/{args.model}_K{k_clusters}/best_model_K{k_clusters}.pt'
    codebook_path = data_dir / f'structure_codebook_K{k_clusters}.pkl'
    
    print(f"=" * 70)
    print(f"PROTEIN FUNCTION PREDICTION INFERENCE")
    print(f"=" * 70)
    print(f"Model: {args.model} ({model_config.model_name})")
    print(f"K-means clusters: {k_clusters}")
    print(f"Test batches held out: {args.test_batches}")
    
    # Validate paths (only if not testing base model)
    if not args.test_base_only:
        if not lora_path.exists():
            print(f"❌ LoRA model not found: {lora_path}")
            print(f"   Run script 05_train_lora.py first with --model {args.model}!")
            print(f"   Or use --test-base-only to test base model without LoRA")
            return
    
    if not alignment_model_path.exists():
        print(f"❌ Alignment model not found: {alignment_model_path}")
        print(f"   Run script 04_train_clip_alignment.py first with --model {args.model}!")
        return
    
    if not codebook_path.exists():
        print(f"❌ Codebook not found: {codebook_path}")
        print(f"   Run script 02_train_kmeans_codebook.py first!")
        return
    
    # Load inputs (must be list format)
    with open(args.input, 'r') as f:
        input_data = json.load(f)
    
    if not isinstance(input_data, list):
        print(f"❌ Input must be a JSON list of objects with 'id', 'sequence', 'question', and 'function' fields")
        return
    
    # Extract data from list format
    inputs = []
    for item in input_data:
        if not isinstance(item, dict):
            print(f"❌ Each item in input list must be a JSON object")
            return
        if 'id' not in item or 'sequence' not in item or 'question' not in item:
            print(f"❌ Each item must have 'id', 'sequence', and 'question' fields")
            return
        inputs.append({
            'id': item['id'],
            'sequence': item['sequence'],
            'question': item['question'],
            'ground_truth_function': item.get('function', '')  # For comparison
        })
    
    # Slice inputs for parallelization if start/end indices provided
    total_inputs = len(inputs)
    if args.start_index is not None or args.end_index is not None:
        start_idx = args.start_index if args.start_index is not None else 0
        end_idx = args.end_index if args.end_index is not None else len(inputs)
        inputs = inputs[start_idx:end_idx]
        print(f"✅ Loaded {total_inputs} total sequences from {args.input}")
        print(f"📊 Processing subset: indices {start_idx} to {end_idx-1} ({len(inputs)} sequences)")
    else:
    print(f"✅ Loaded {len(inputs)} sequences from {args.input}")
    
    # Initialize inference pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Device: {device}")
    print(f"🤖 Model: {args.model} ({model_config.model_name})")
    
    pipeline = ProteinFunctionInference(
        model_config=model_config,
        lora_path=str(lora_path) if not args.test_base_only else None,
        alignment_model_path=str(alignment_model_path),
        codebook_path=str(codebook_path),
        k_clusters=k_clusters,
        device=device
    )
    
    # Initialize ROUGE scorer
    output_file = Path(args.output)
    # Modify output filename for parallel execution
    if args.gpu_id is not None:
        output_file = output_file.parent / f"{output_file.stem}_gpu{args.gpu_id}{output_file.suffix}"
    output_file.parent.mkdir(parents=True, exist_ok=True)  # Create output directory if needed
    output_file = str(output_file)  # Convert back to string for json.dump
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Single loop: process each input with all 5 variants, compute ROUGE, and save immediately
    results = []
    rouge_scores = {
        'variant1_plain_seq': {'rouge1': [], 'rouge2': [], 'rougeL': []},
        'variant2_plain_seq_struct': {'rouge1': [], 'rouge2': [], 'rougeL': []},
        'variant3_plain_embeddings': {'rouge1': [], 'rouge2': [], 'rougeL': []},
        'variant4_finetuned_struct': {'rouge1': [], 'rouge2': [], 'rougeL': []},
        'variant5_finetuned_embeddings': {'rouge1': [], 'rouge2': [], 'rougeL': []}
    }
    
    print(f"\n🚀 Running inference (Ablation Study)...")
    print(f"  Model: {args.model} ({model_config.model_name})")
    print(f"  Variants to run: {', '.join(map(str, variants_to_run))}")
    variant_descriptions = {
        1: "Variant 1: Plain model (question + sequence) - Baseline",
        2: "Variant 2: Plain model (question + sequence + structure as text)",
        3: "Variant 3: Plain model (question + sequence + structure + embeddings)",
        4: "Variant 4: Fine-tuned model (question + sequence + structure as text, no embeddings)",
        5: "Variant 5: Fine-tuned model (question + sequence + structure + embeddings) - Full Model"
    }
    for v in variants_to_run:
        print(f"  {variant_descriptions[v]}")
    
    for item in tqdm(inputs, desc="Processing samples"):
        result = {
            "id": item['id'],
            "sequence": item['sequence'],
            "question": item['question'],
            "ground_truth_function": item['ground_truth_function']
        }
        
        gt = result.get('ground_truth_function', '')
        
        # Variant 1: Plain model (question + sequence)
        if 1 in variants_to_run:
        try:
            pred_v1 = pipeline.generate_plain_llama(
                item['sequence'],
                question=item['question'],
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )
            result["predicted_function_variant1"] = pred_v1
            if gt and pred_v1 and not pred_v1.startswith('ERROR'):
                scores = scorer.score(gt, pred_v1)
                result['rouge_variant1'] = {
                    'rouge1': scores['rouge1'].fmeasure,
                    'rouge2': scores['rouge2'].fmeasure,
                    'rougeL': scores['rougeL'].fmeasure,
                }
                rouge_scores['variant1_plain_seq']['rouge1'].append(scores['rouge1'].fmeasure)
                rouge_scores['variant1_plain_seq']['rouge2'].append(scores['rouge2'].fmeasure)
                rouge_scores['variant1_plain_seq']['rougeL'].append(scores['rougeL'].fmeasure)
            else:
                result['rouge_variant1'] = None
        except Exception as e:
            print(f"\n⚠️  Error processing {item['id']} (variant 1): {e}")
            result["predicted_function_variant1"] = f"ERROR: {str(e)}"
                result['rouge_variant1'] = None
        else:
            result["predicted_function_variant1"] = "SKIPPED"
            result['rouge_variant1'] = None
        
        # Variant 2: Plain model (question + sequence + structure)
        if 2 in variants_to_run:
        try:
            pred_v2 = pipeline.generate_plain_llama_with_structure(
                item['sequence'],
                question=item['question'],
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )
            result["predicted_function_variant2"] = pred_v2
            if gt and pred_v2 and not pred_v2.startswith('ERROR'):
                scores = scorer.score(gt, pred_v2)
                result['rouge_variant2'] = {
                    'rouge1': scores['rouge1'].fmeasure,
                    'rouge2': scores['rouge2'].fmeasure,
                    'rougeL': scores['rougeL'].fmeasure,
                }
                rouge_scores['variant2_plain_seq_struct']['rouge1'].append(scores['rouge1'].fmeasure)
                rouge_scores['variant2_plain_seq_struct']['rouge2'].append(scores['rouge2'].fmeasure)
                rouge_scores['variant2_plain_seq_struct']['rougeL'].append(scores['rougeL'].fmeasure)
            else:
                result['rouge_variant2'] = None
        except Exception as e:
            print(f"\n⚠️  Error processing {item['id']} (variant 2): {e}")
            result["predicted_function_variant2"] = f"ERROR: {str(e)}"
                result['rouge_variant2'] = None
        else:
            result["predicted_function_variant2"] = "SKIPPED"
            result['rouge_variant2'] = None
        
        # Variant 3: Plain model (question + sequence + structure + embeddings)
        if 3 in variants_to_run:
        try:
            pred_v3 = pipeline.generate_plain_llama_with_embeddings(
                item['sequence'],
                question=item['question'],
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )
            result["predicted_function_variant3"] = pred_v3
            if gt and pred_v3 and not pred_v3.startswith('ERROR'):
                scores = scorer.score(gt, pred_v3)
                result['rouge_variant3'] = {
                    'rouge1': scores['rouge1'].fmeasure,
                    'rouge2': scores['rouge2'].fmeasure,
                    'rougeL': scores['rougeL'].fmeasure,
                }
                rouge_scores['variant3_plain_embeddings']['rouge1'].append(scores['rouge1'].fmeasure)
                rouge_scores['variant3_plain_embeddings']['rouge2'].append(scores['rouge2'].fmeasure)
                rouge_scores['variant3_plain_embeddings']['rougeL'].append(scores['rougeL'].fmeasure)
            else:
                result['rouge_variant3'] = None
        except Exception as e:
            print(f"\n⚠️  Error processing {item['id']} (variant 3): {e}")
            result["predicted_function_variant3"] = f"ERROR: {str(e)}"
                result['rouge_variant3'] = None
        else:
            result["predicted_function_variant3"] = "SKIPPED"
            result['rouge_variant3'] = None
        
        # Variant 4: Fine-tuned model (question + sequence + structure as text, no embeddings)
        if 4 in variants_to_run:
        try:
            pred_v4 = pipeline.generate_finetuned_llama_with_structure(
                item['sequence'],
                question=item['question'],
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )
            result["predicted_function_variant4"] = pred_v4
            if gt and pred_v4 and not pred_v4.startswith('ERROR'):
                scores = scorer.score(gt, pred_v4)
                result['rouge_variant4'] = {
                    'rouge1': scores['rouge1'].fmeasure,
                    'rouge2': scores['rouge2'].fmeasure,
                    'rougeL': scores['rougeL'].fmeasure,
                }
                rouge_scores['variant4_finetuned_struct']['rouge1'].append(scores['rouge1'].fmeasure)
                rouge_scores['variant4_finetuned_struct']['rouge2'].append(scores['rouge2'].fmeasure)
                rouge_scores['variant4_finetuned_struct']['rougeL'].append(scores['rougeL'].fmeasure)
            else:
                result['rouge_variant4'] = None
        except Exception as e:
            print(f"\n⚠️  Error processing {item['id']} (variant 4): {e}")
            result["predicted_function_variant4"] = f"ERROR: {str(e)}"
                result['rouge_variant4'] = None
        else:
            result["predicted_function_variant4"] = "SKIPPED"
            result['rouge_variant4'] = None
        
        # Variant 5: Fine-tuned model (question + sequence + structure + embeddings) - Full Model
        if 5 in variants_to_run:
        try:
            pred_v5 = pipeline.generate(
                item['sequence'],
                question=item['question'],
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )
            result["predicted_function_variant5"] = pred_v5
            if gt and pred_v5 and not pred_v5.startswith('ERROR'):
                scores = scorer.score(gt, pred_v5)
                result['rouge_variant5'] = {
                    'rouge1': scores['rouge1'].fmeasure,
                    'rouge2': scores['rouge2'].fmeasure,
                    'rougeL': scores['rougeL'].fmeasure,
                }
                rouge_scores['variant5_finetuned_embeddings']['rouge1'].append(scores['rouge1'].fmeasure)
                rouge_scores['variant5_finetuned_embeddings']['rouge2'].append(scores['rouge2'].fmeasure)
                rouge_scores['variant5_finetuned_embeddings']['rougeL'].append(scores['rougeL'].fmeasure)
            else:
                result['rouge_variant5'] = None
        except Exception as e:
            print(f"\n⚠️  Error processing {item['id']} (variant 5): {e}")
            result["predicted_function_variant5"] = f"ERROR: {str(e)}"
                result['rouge_variant5'] = None
        else:
            result["predicted_function_variant5"] = "SKIPPED"
            result['rouge_variant5'] = None
        
        # Add to results
        results.append(result)
        
        # Compute current averages for summary
        def compute_avg_rouge(variant_scores):
            return {
                'avg_rouge1': float(np.mean(variant_scores['rouge1'])) if variant_scores['rouge1'] else 0,
                'avg_rouge2': float(np.mean(variant_scores['rouge2'])) if variant_scores['rouge2'] else 0,
                'avg_rougeL': float(np.mean(variant_scores['rougeL'])) if variant_scores['rougeL'] else 0,
            }
        
        # Summary evaluation metrics
        evaluation_metrics = {
            'variant1_plain_seq': {
                'n_samples': len(rouge_scores['variant1_plain_seq']['rouge1']),
                **compute_avg_rouge(rouge_scores['variant1_plain_seq'])
            },
            'variant2_plain_seq_struct': {
                'n_samples': len(rouge_scores['variant2_plain_seq_struct']['rouge1']),
                **compute_avg_rouge(rouge_scores['variant2_plain_seq_struct'])
            },
            'variant3_plain_embeddings': {
                'n_samples': len(rouge_scores['variant3_plain_embeddings']['rouge1']),
                **compute_avg_rouge(rouge_scores['variant3_plain_embeddings'])
            },
            'variant4_finetuned_struct': {
                'n_samples': len(rouge_scores['variant4_finetuned_struct']['rouge1']),
                **compute_avg_rouge(rouge_scores['variant4_finetuned_struct'])
            },
            'variant5_finetuned_embeddings': {
                'n_samples': len(rouge_scores['variant5_finetuned_embeddings']['rouge1']),
                **compute_avg_rouge(rouge_scores['variant5_finetuned_embeddings'])
            }
        }
        
        # Write JSON after each iteration (overwrite)
        output_data = {
            'results': results,
            'evaluation_summary': evaluation_metrics
        }
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")
    
    # Compute final summary metrics
    def compute_final_avg(variant_scores):
        return {
            'avg_rouge1': float(np.mean(variant_scores['rouge1'])) if variant_scores['rouge1'] else 0,
            'avg_rouge2': float(np.mean(variant_scores['rouge2'])) if variant_scores['rouge2'] else 0,
            'avg_rougeL': float(np.mean(variant_scores['rougeL'])) if variant_scores['rougeL'] else 0,
        }
    
    final_eval = {
        'variant1_plain_seq': {
            'n_samples': len(rouge_scores['variant1_plain_seq']['rouge1']),
            **compute_final_avg(rouge_scores['variant1_plain_seq'])
        },
        'variant2_plain_seq_struct': {
            'n_samples': len(rouge_scores['variant2_plain_seq_struct']['rouge1']),
            **compute_final_avg(rouge_scores['variant2_plain_seq_struct'])
        },
        'variant3_plain_embeddings': {
            'n_samples': len(rouge_scores['variant3_plain_embeddings']['rouge1']),
            **compute_final_avg(rouge_scores['variant3_plain_embeddings'])
        },
        'variant4_finetuned_struct': {
            'n_samples': len(rouge_scores['variant4_finetuned_struct']['rouge1']),
            **compute_final_avg(rouge_scores['variant4_finetuned_struct'])
        },
        'variant5_finetuned_embeddings': {
            'n_samples': len(rouge_scores['variant5_finetuned_embeddings']['rouge1']),
            **compute_final_avg(rouge_scores['variant5_finetuned_embeddings'])
        }
    }
    
    # Compare predictions with ground truth
    print(f"\n📊 Comparison Summary:")
    print(f"=" * 70)
    print(f"   Total sequences: {len(results)}")
    
    variant_names = {
        'variant1_plain_seq': 'V1: Plain (seq)',
        'variant2_plain_seq_struct': 'V2: Plain (seq+struct)',
        'variant3_plain_embeddings': 'V3: Plain (embeddings)',
        'variant4_finetuned_struct': 'V4: Fine-tuned (struct)',
        'variant5_finetuned_embeddings': 'V5: Fine-tuned (embeddings)'
    }
    
    variant_pred_keys = {
        'variant1_plain_seq': 'predicted_function_variant1',
        'variant2_plain_seq_struct': 'predicted_function_variant2',
        'variant3_plain_embeddings': 'predicted_function_variant3',
        'variant4_finetuned_struct': 'predicted_function_variant4',
        'variant5_finetuned_embeddings': 'predicted_function_variant5'
    }
    
    for variant_key, variant_name in variant_names.items():
        pred_key = variant_pred_keys[variant_key]
        successful = [r for r in results if not r.get(pred_key, '').startswith('ERROR')]
        errors = len(results) - len(successful)
        print(f"\n   {variant_name}:")
        print(f"     Successful: {len(successful)}")
        print(f"     Errors: {errors}")
    
    # Print ROUGE evaluation
    print(f"\n📈 ROUGE Evaluation (Ablation Study):")
    print(f"=" * 70)
    print(f"\n   {'Metric':<12} {'V1':<10} {'V2':<10} {'V3':<10} {'V4':<10} {'V5':<10}")
    print(f"   {'':<12} {'(seq)':<10} {'(seq+st)':<10} {'(emb)':<10} {'(ft+st)':<10} {'(ft+emb)':<10}")
    print(f"   {'-'*62}")
    for metric in ['rouge1', 'rouge2', 'rougeL']:
        v1_score = final_eval['variant1_plain_seq'][f'avg_{metric}']
        v2_score = final_eval['variant2_plain_seq_struct'][f'avg_{metric}']
        v3_score = final_eval['variant3_plain_embeddings'][f'avg_{metric}']
        v4_score = final_eval['variant4_finetuned_struct'][f'avg_{metric}']
        v5_score = final_eval['variant5_finetuned_embeddings'][f'avg_{metric}']
        print(f"   {metric.upper():<12} {v1_score:.4f}{'':>4} {v2_score:.4f}{'':>4} {v3_score:.4f}{'':>4} {v4_score:.4f}{'':>4} {v5_score:.4f}{'':>4}")
    
    print(f"\n   Samples evaluated:")
    for variant_key, variant_name in variant_names.items():
        n_samples = final_eval[variant_key]['n_samples']
        print(f"     {variant_name}: {n_samples}")


if __name__ == "__main__":
    main()

