#!/usr/bin/env python3
"""Preprocess PFUD data using ProtTeX tokenizer.

This module tokenizes protein structures that were already fetched in prepare_data.py:
1. Reads prepared dataset with pdb_string (raw PDB text)
2. Tokenizes using real ProtTeX VQ encoder to get code_indices
3. Creates prompts with sequence + structure tokens for the LLM

Reference: scripts/function_inference.py expects input with:
- code_indices: VQ encoded structure tokens from encoder
- aatype: amino acid sequence tokens
"""

import sys
import json
import pickle as pkl
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import numpy as np
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

# Add ProToken to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Add workspace root
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'ProToken'))

from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.jax_utils import replicate
from model.encoder import VQ_Encoder
from model.decoder import VQ_Decoder, Protein_Decoder
from tokenizer.vector_quantization import VQTokenizer
from inference.inference_new import InferenceCell
from train.utils import make_rng_dict
from common.config_load import load_config
from data.pipeline import protoken_input_generator, protoken_input_content, protoken_feature_content
from config.global_config import GLOBAL_CONFIG
from config.train_vq_config import TRAINING_CONFIG

class PFUDPreprocessor:
    """Tokenize PFUD data using real ProtTeX VQ encoder"""
    
    def __init__(self, output_dir: str, aa_dict_path: str, protoken_dict_path: str,
                 encoder_config: str = "./ProToken/config/encoder.yaml",
                 decoder_config: str = "./ProToken/config/decoder.yaml",
                 vq_config: str = './ProToken/config/vq.yaml',
                 ckpt_path: str = './ProToken/ckpts/protoken_params_130000.pkl',
                 random_seed: int = 8888,
                 np_random_seed: int = 18888):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load character dictionaries
        self.aa_dict = self._load_aa_dict(aa_dict_path)
        self.protoken_dict = self._load_protoken_dict(protoken_dict_path)
        
        print(f"Loaded AA dict with {len(self.aa_dict)} entries")
        print(f"Loaded ProtToken dict with {len(self.protoken_dict)} entries")
        
        # Initialize ProtTeX encoder
        print("Initializing ProtTeX VQ encoder...")
        self._init_encoder(encoder_config, decoder_config, vq_config, ckpt_path, 
                          random_seed, np_random_seed)
        print("✓ ProtTeX encoder initialized")
    
    def _load_aa_dict(self, path: str) -> Dict:
        """Load amino acid character dictionary"""
        with open(path, 'rb') as f:
            return pkl.load(f)
    
    def _load_protoken_dict(self, path: str) -> List[str]:
        """Load ProtToken character dictionary"""
        with open(path, 'r') as f:
            return json.load(f)
    
    def _init_encoder(self, encoder_config: str, decoder_config: str, vq_config: str,
                     ckpt_path: str, random_seed: int, np_random_seed: int):
        """Initialize the ProtTeX VQ encoder from checkpoint"""
        
        # Load configurations
        encoder_cfg = load_config(encoder_config)
        decoder_cfg = load_config(decoder_config)
        vq_cfg = load_config(vq_config)
        
        NRES = 1024
        encoder_cfg.seq_len = NRES
        decoder_cfg.seq_len = NRES
        
        # Define modules dictionary
        modules = {
            "encoder": {"module": VQ_Encoder, 
                        "args": {"global_config": GLOBAL_CONFIG, "cfg": encoder_cfg}, 
                        "freeze": False},
            "vq_decoder": {"module": VQ_Decoder, 
                           "args": {"global_config": GLOBAL_CONFIG, "cfg": decoder_cfg, "pre_layer_norm": False}, 
                           "freeze": False},
            "protein_decoder": {"module": Protein_Decoder, 
                            "args": {"global_config": GLOBAL_CONFIG, "cfg": decoder_cfg}, 
                            "freeze": False},
            "vq_tokenizer": {"module": VQTokenizer, 
                             "args": {"config": vq_cfg, "dtype": jnp.float32}, 
                             "freeze": False},
            "project_in": {"module": nn.Dense, 
                           "args": {"features": vq_cfg.dim_code, "kernel_init": nn.initializers.lecun_normal(), "use_bias": False}, 
                           "freeze": False},
            "project_out": {"module": nn.Dense, 
                           "args": {"features": vq_cfg.dim_in, "kernel_init": nn.initializers.lecun_normal(), "use_bias": False},
                           "freeze": False},
        }
        
        # Load checkpoint
        if ckpt_path and os.path.exists(ckpt_path):
            with open(ckpt_path, "rb") as f:
                params = pkl.load(f)
                params = jax.tree_util.tree_map(lambda x: jnp.array(x), params)
        else:
            print(f"Warning: Checkpoint not found at {ckpt_path}")
            # Initialize params if checkpoint not found
            rng = jax.random.PRNGKey(random_seed)
            dummy_input = jnp.zeros((1, NRES, 20))
            params = {}
        
        # Initialize modules
        for k, v in modules.items():
            modules[k]["module"] = v["module"](**v["args"])
        
        # Create inference cell
        self.inference_cell = InferenceCell(
            global_config=GLOBAL_CONFIG,
            train_cfg=TRAINING_CONFIG,
            encoder=modules["encoder"]["module"],
            vq_tokenizer=modules["vq_tokenizer"]["module"],
            vq_decoder=modules["vq_decoder"]["module"],
            protein_decoder=modules["protein_decoder"]["module"],
            project_in=modules["project_in"]["module"],
            project_out=modules["project_out"]["module"],
            quantize=bool(vq_cfg.quantize)
        )
        
        # Replicate params for pmap
        self.params = replicate(params)
        self.rng_key = jax.random.PRNGKey(random_seed)
        np.random.seed(np_random_seed)
        
        # JIT-compile inference function
        inference_cell_jit = jax.jit(self.inference_cell.apply)
        self.inference_cell_vmap = jax.jit(jax.vmap(inference_cell_jit, in_axes=[None] + [0] * 9))
        self.inference_cell_pmap = jax.pmap(jax.jit(self.inference_cell_vmap), axis_name="i")
    
    def pdb_string_to_code_indices(self, pdb_string: str) -> np.ndarray:
        """Convert PDB string to code_indices using ProtTeX encoder.
        
        Args:
            pdb_string: Raw PDB format text
            
        Returns:
            code_indices: VQ-encoded structure tokens from encoder
        """
        try:
            # Save PDB string to temporary file for processing
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
                f.write(pdb_string)
                pdb_path = f.name
            
            # Load features from PDB
            NRES = 1024
            feat_ = {}
            for k_ in protoken_feature_content:
                feat_[k_] = []
            
            future_ = protoken_input_generator(pdb_path, NRES=NRES, crop_start_idx_preset=0)
            batched_feature_tmp, crop_start_idx_tmp, seq_len_tmp = future_
            batched_feature_tmp = jax.tree_map(lambda x: jnp.array(x), batched_feature_tmp)
            
            for k_ in protoken_feature_content:
                feat_[k_].append(batched_feature_tmp[k_])
            
            feat_ = {k_: jnp.concatenate(v_, axis=0) for k_, v_ in feat_.items()}
            
            # Generate RNG keys
            net_rng_key, self.rng_key = make_rng_dict(self.rng_key,
                                                      ["fape_clamp_key"],
                                                      num_rngs_per_key=feat_['fake_aatype'].shape[0],
                                                      squeeze=False)
            
            # Reshape inputs
            reshape_func = lambda x: x.reshape(1, x.shape[0]//1, *x.shape[1:])
            feat_reshape = jax.tree_util.tree_map(reshape_func, feat_)
            net_rng_key_reshape = jax.tree_util.tree_map(reshape_func, net_rng_key)
            
            # Prepare input features
            input_feature_tmp = [feat_reshape[name] for name in protoken_input_content]
            
            # Run inference to get code_indices
            aux_result = self.inference_cell_pmap(self.params, *input_feature_tmp, rngs=net_rng_key_reshape)
            aux_result = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), aux_result)
            
            # Extract code_indices
            code_indices = np.asarray(aux_result['code_indices']).astype(np.int32)
            
            # Clean up temp file
            os.remove(pdb_path)
            
            return code_indices[0][:seq_len_tmp]  # Return only valid sequence length
            
        except Exception as e:
            print(f"Error converting PDB to code_indices: {e}")
            return None
    
    def structure_to_tokens(self, pdb_string: Optional[str], 
                           sequence_length: int) -> str:
        """Convert PDB structure to tokens using ProtTeX VQ encoder.
        
        Args:
            pdb_string: Raw PDB format text from structure dict
            sequence_length: Length of the protein sequence
            
        Returns:
            structure_tokens: String of protoken characters representing structure
        """
        if pdb_string is None:
            raise ValueError("No PDB string found for protein")
        
        # Get code_indices from encoder
        code_indices = self.pdb_string_to_code_indices(pdb_string)
        
        if code_indices is None:
            raise ValueError("Failed to encode PDB structure")
        
        # Convert code_indices to protoken string
        # code_indices are quantized indices that map to protoken characters
        print(code_indices)
        tokens_str = "".join([self.protoken_dict[int(idx) % len(self.protoken_dict)] 
                             for idx in code_indices[:512]])
        
        return tokens_str
    
    def tokenize_sequence(self, sequence: str) -> str:
        """Convert amino acid sequence to token indices"""
        return "".join([str(self.aa_dict.get(aa, 0)) for aa in sequence])
    
    def tokenize_structure(self, structure_tokens: List[int]) -> str:
        """Tokenize structure information"""
        return "".join([self.protoken_dict[int(t) % len(self.protoken_dict)] 
                       for t in structure_tokens[:512]])
    
    def create_prompt(self, protein_data: Dict, seq_tokens: str, struct_tokens: str, question: str) -> str:
        return (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{question}\n"
            f"<protein_sequence>{seq_tokens}</protein_sequence>\n"
            f"<protein_structure>{struct_tokens}</protein_structure>\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    
    def preprocess_split(self, input_file: Path, split_name: str) -> Path:
        """Preprocess a single data split"""
        print(f"\nProcessing {split_name} split...")
        
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        tokenized_data = []
        skipped_count = 0
        
        for item in tqdm(data, desc=f"Tokenizing {split_name}"):
            accession = item['accession']
            sequence = item['sequence']
            
            seq_tokens = self.tokenize_sequence(sequence)
            
            # Get PDB structure string from prepared dataset
            structure = item.get('structure')
            if structure is None:
                skipped_count += 1
                continue
            
            pdb_string = structure.get('pdb_string') if isinstance(structure, dict) else None
            if pdb_string is None:
                skipped_count += 1
                continue
            
            # Try to convert structure to tokens using encoder
            try:
                struct_tokens = self.structure_to_tokens(pdb_string, len(sequence))
            except Exception as e:
                print(f"  Warning: Could not tokenize structure for {accession}: {e}")
                skipped_count += 1
                continue
            
            for qa_pair in item.get('qa_pairs', []):
                question = qa_pair.get('question', '')
                prompt = self.create_prompt(item, seq_tokens, struct_tokens, question)
                example = {
                    'accession': accession,
                    'prompt': prompt,
                    'answer': qa_pair['answer'],
                    'input_ids': None,
                    'labels': None
                }
                tokenized_data.append(example)
        
        output_file = self.output_dir / f"pfud_{split_name}_tokenized.pkl"
        with open(output_file, 'wb') as f:
            pkl.dump(tokenized_data, f)

        output_file = self.output_dir / f"pfud_{split_name}_tokenized.json"
        with open(output_file, 'w') as f:
            json.dump(tokenized_data, f, indent=2)

        print(f"✓ Saved {len(tokenized_data)} tokenized examples to {output_file}")
        print(f"  Skipped {skipped_count} proteins without valid structures")
        return output_file
    
    def preprocess(self, input_dir: Path, split: str = 'all'):
        """Main preprocessing pipeline
        
        Args:
            input_dir: Directory containing pfud_*.json files
            split: Which split(s) to process ('train', 'val', 'test', or 'all')
        """
        print("=" * 60)
        print("Preprocessing PFUD Dataset with ProtTeX Tokenizer")
        print("Using real PDB structures")
        print("=" * 60)
        
        splits = ['train', 'val', 'test'] if split == 'all' else [split]
        output_files = {}
        
        for split_name in splits:
            input_file = input_dir / f"pfud_{split_name}.json"
            if input_file.exists():
                output_file = self.preprocess_split(input_file, split_name)
                output_files[split_name] = str(output_file)
        
        manifest = {
            'output_files': output_files,
            'num_protoken_tokens': len(self.protoken_dict),
            'num_aa_tokens': len(self.aa_dict),
            'output_dir': str(self.output_dir),
            'use_real_encoder': True,
            'encoder_type': 'ProtTeX_VQ_Encoder'
        }
        
        manifest_file = self.output_dir / 'manifest.json'
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n✓ Preprocessing complete!")
        print(f"  Output directory: {self.output_dir}")
        return self.output_dir
