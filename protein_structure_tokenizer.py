#!/usr/bin/env python3
"""
ProteinStructureTokenizer - Production-Ready Tokenizer Class

Combines amino acid tokens with structure tokens from ESM-2 embeddings.
Ready to push to HuggingFace or GitHub.
"""

import torch
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import AutoTokenizer, EsmModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers not available. Install with: pip install transformers")


@dataclass
class TokenizerConfig:
    """Configuration for ProteinStructureTokenizer"""
    model_name: str = "facebook/esm2_t33_650M_UR50D"
    n_structure_tokens: int = 512
    embedding_dim: int = 1280
    aa_token_offset: int = 0  # AA tokens: 0-19
    structure_token_offset: int = 20  # Structure tokens: 20-531
    special_token_offset: int = 532  # Special tokens: 532+
    
    # Special tokens
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    sep_token: str = "<sep>"
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.n_structure_tokens > 0, "n_structure_tokens must be > 0"
        assert self.embedding_dim > 0, "embedding_dim must be > 0"


class ProteinStructureTokenizer:
    """
    Tokenizer for proteins with sequence + structure information.
    
    Converts protein sequences into multimodal token streams combining:
    - Amino acid tokens (0-19)
    - Structure tokens (20-531) from ESM-2 + k-means codebook
    - Special tokens (532+) for control
    
    Example:
        >>> tokenizer = ProteinStructureTokenizer.from_pretrained("path/to/codebook.pkl")
        >>> tokens = tokenizer.encode("MALWMRLLPLLA")
        >>> print(tokens)  # [12, 287, 0, 143, 11, 287, ...]
    """
    
    # Amino acid vocabulary
    AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
    
    def __init__(
        self,
        codebook_path: Optional[str] = None,
        config: Optional[TokenizerConfig] = None,
        device: str = "cuda"
    ):
        """
        Initialize tokenizer.
        
        Args:
            codebook_path: Path to trained k-means codebook (.pkl file)
            config: TokenizerConfig object
            device: 'cuda' or 'cpu'
        """
        self.config = config or TokenizerConfig()
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Build AA vocabulary
        self.aa_to_id = {aa: i for i, aa in enumerate(self.AA_VOCAB)}
        self.id_to_aa = {i: aa for aa, i in self.aa_to_id.items()}
        
        # Special tokens
        self.special_tokens = {
            self.config.pad_token: self.config.special_token_offset,
            self.config.unk_token: self.config.special_token_offset + 1,
            self.config.bos_token: self.config.special_token_offset + 2,
            self.config.eos_token: self.config.special_token_offset + 3,
            self.config.sep_token: self.config.special_token_offset + 4,
        }
        
        # Load codebook if provided
        self.kmeans = None
        if codebook_path:
            self.load_codebook(codebook_path)
        
        # Load ESM-2 model (lazy loading)
        self._esm_model = None
        self._esm_tokenizer = None
    
    @property
    def vocab_size(self) -> int:
        """Total vocabulary size"""
        return (
            len(self.AA_VOCAB) +  # 20 AA
            self.config.n_structure_tokens +  # 512 structure
            len(self.special_tokens)  # 5 special
        )
    
    @property
    def pad_token_id(self) -> int:
        return self.special_tokens[self.config.pad_token]
    
    @property
    def unk_token_id(self) -> int:
        return self.special_tokens[self.config.unk_token]
    
    @property
    def bos_token_id(self) -> int:
        return self.special_tokens[self.config.bos_token]
    
    @property
    def eos_token_id(self) -> int:
        return self.special_tokens[self.config.eos_token]
    
    @property
    def sep_token_id(self) -> int:
        return self.special_tokens[self.config.sep_token]
    
    def _load_esm_model(self):
        """Lazy load ESM-2 model"""
        if self._esm_model is None:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("transformers required for structure tokenization")
            
            print(f"Loading ESM-2 model: {self.config.model_name}")
            self._esm_tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self._esm_model = EsmModel.from_pretrained(self.config.model_name)
            self._esm_model.eval()
            self._esm_model = self._esm_model.to(self.device)
    
    def load_codebook(self, codebook_path: str):
        """Load trained k-means codebook"""
        codebook_path = Path(codebook_path)
        if not codebook_path.exists():
            raise FileNotFoundError(f"Codebook not found: {codebook_path}")
        
        with open(codebook_path, 'rb') as f:
            codebook_data = pickle.load(f)
        
        self.kmeans = codebook_data['kmeans']
        
        # Validate codebook
        if self.kmeans.n_clusters != self.config.n_structure_tokens:
            warnings.warn(
                f"Codebook has {self.kmeans.n_clusters} clusters, "
                f"config expects {self.config.n_structure_tokens}"
            )
    
    def save_pretrained(self, save_directory: str):
        """Save tokenizer to directory"""
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = save_dir / "tokenizer_config.pkl"
        with open(config_path, 'wb') as f:
            pickle.dump({
                'config': self.config,
                'aa_to_id': self.aa_to_id,
                'special_tokens': self.special_tokens,
            }, f)
        
        # Save codebook if exists
        if self.kmeans is not None:
            codebook_path = save_dir / "structure_codebook.pkl"
            with open(codebook_path, 'wb') as f:
                pickle.dump({'kmeans': self.kmeans}, f)
        
        print(f"✅ Tokenizer saved to {save_dir}")
    
    @classmethod
    def from_pretrained(cls, load_directory: str, device: str = "cuda"):
        """Load tokenizer from directory"""
        load_dir = Path(load_directory)
        
        # Load config
        config_path = load_dir / "tokenizer_config.pkl"
        with open(config_path, 'rb') as f:
            data = pickle.load(f)
        
        # Create tokenizer
        tokenizer = cls(config=data['config'], device=device)
        tokenizer.aa_to_id = data['aa_to_id']
        tokenizer.special_tokens = data['special_tokens']
        
        # Load codebook if exists
        codebook_path = load_dir / "structure_codebook.pkl"
        if codebook_path.exists():
            tokenizer.load_codebook(codebook_path)
        
        print(f"✅ Tokenizer loaded from {load_dir}")
        return tokenizer
    
    def _extract_structure_features(self, sequence: str) -> np.ndarray:
        """Extract ESM-2 embeddings for sequence"""
        self._load_esm_model()
        
        # Tokenize for ESM-2
        inputs = self._esm_tokenizer([sequence], return_tensors="pt", padding=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract embeddings
        with torch.no_grad():
            outputs = self._esm_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            # Remove <cls> and <eos> tokens
            embeddings = hidden_states[0, 1:-1, :].cpu().numpy()
        
        return embeddings
    
    def _embeddings_to_structure_tokens(self, embeddings: np.ndarray) -> List[int]:
        """Convert embeddings to structure token IDs via k-means"""
        if self.kmeans is None:
            raise ValueError("Codebook not loaded. Call load_codebook() first.")
        
        # Assign to clusters
        cluster_ids = self.kmeans.predict(embeddings)
        
        # Add offset
        structure_tokens = [self.config.structure_token_offset + int(cid) for cid in cluster_ids]
        
        return structure_tokens
    
    def encode(
        self,
        sequence: str,
        add_special_tokens: bool = False,
        return_structure_tokens: bool = True,
        return_format: str = "interleaved"
    ) -> Union[List[int], Dict[str, List[int]]]:
        """
        Encode protein sequence into tokens.
        
        Args:
            sequence: Amino acid sequence (e.g., "MALWMRLLPLLA")
            add_special_tokens: Add <bos> and <eos> tokens
            return_structure_tokens: Include structure tokens (requires codebook)
            return_format: "interleaved", "separate", or "aa_only"
        
        Returns:
            List of token IDs or dict with separate streams
        
        Example:
            >>> tokens = tokenizer.encode("MALW", return_format="interleaved")
            >>> # [12, 287, 0, 143, 11, 287, 22, 56]
            >>> #  M   S287  A  S143  L  S287  W   S56
        """
        # Convert sequence to uppercase
        sequence = sequence.upper()
        
        # Amino acid tokens
        aa_tokens = []
        for aa in sequence:
            if aa in self.aa_to_id:
                aa_tokens.append(self.aa_to_id[aa])
            else:
                aa_tokens.append(self.unk_token_id)
        
        # Structure tokens (if requested)
        structure_tokens = None
        if return_structure_tokens:
            if self.kmeans is None:
                raise ValueError("Codebook not loaded. Cannot extract structure tokens.")
            
            embeddings = self._extract_structure_features(sequence)
            structure_tokens = self._embeddings_to_structure_tokens(embeddings)
        
        # Format output
        if return_format == "aa_only" or structure_tokens is None:
            tokens = aa_tokens
        
        elif return_format == "interleaved":
            # [AA, Structure, AA, Structure, ...]
            tokens = []
            for aa_tok, struct_tok in zip(aa_tokens, structure_tokens):
                tokens.append(aa_tok)
                tokens.append(struct_tok)
        
        elif return_format == "separate":
            tokens = {
                "aa_tokens": aa_tokens,
                "structure_tokens": structure_tokens
            }
        
        else:
            raise ValueError(f"Unknown format: {return_format}")
        
        # Add special tokens
        if add_special_tokens and isinstance(tokens, list):
            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
        
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to amino acid sequence.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Skip special tokens in output
        
        Returns:
            Amino acid sequence
        """
        sequence = []
        
        for token_id in token_ids:
            # Skip special tokens
            if skip_special_tokens and token_id >= self.config.special_token_offset:
                continue
            
            # Skip structure tokens
            if token_id >= self.config.structure_token_offset:
                continue
            
            # Decode AA token
            if token_id in self.id_to_aa:
                sequence.append(self.id_to_aa[token_id])
        
        return ''.join(sequence)
    
    def batch_encode(
        self,
        sequences: List[str],
        **kwargs
    ) -> List[Union[List[int], Dict[str, List[int]]]]:
        """Encode multiple sequences"""
        return [self.encode(seq, **kwargs) for seq in sequences]
    
    def get_token_info(self, token_id: int) -> Dict[str, str]:
        """Get information about a token ID"""
        if token_id < len(self.AA_VOCAB):
            return {
                'type': 'amino_acid',
                'symbol': self.id_to_aa[token_id],
                'token_id': token_id
            }
        elif token_id < self.config.structure_token_offset + self.config.n_structure_tokens:
            cluster_id = token_id - self.config.structure_token_offset
            return {
                'type': 'structure',
                'symbol': f'S{cluster_id:03d}',
                'token_id': token_id,
                'cluster_id': cluster_id
            }
        else:
            for name, tid in self.special_tokens.items():
                if tid == token_id:
                    return {
                        'type': 'special',
                        'symbol': name,
                        'token_id': token_id
                    }
            return {
                'type': 'unknown',
                'symbol': '<unk>',
                'token_id': token_id
            }
    
    def __repr__(self) -> str:
        return (
            f"ProteinStructureTokenizer(\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  aa_tokens={len(self.AA_VOCAB)},\n"
            f"  structure_tokens={self.config.n_structure_tokens},\n"
            f"  codebook_loaded={self.kmeans is not None}\n"
            f")"
        )




