#!/usr/bin/env python3
"""
Demo: Complete Protein Structure Tokenizer with ESMFold Embeddings
==================================================================

This script demonstrates the full tokenization pipeline:
1. Character-level amino acid tokenization (20 tokens for standard amino acids)
2. Structure tokenization using learned ESM-2 embedding codebook (512 tokens)
3. Combined multimodal token stream for LLM input

Author: Mohammad Sayeed
Date: Nov 2025
"""

import torch
import pickle
import numpy as np
from transformers import AutoTokenizer, EsmModel
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class ProteinStructureTokenizer:
    """
    Multimodal Protein Tokenizer combining amino acid sequences and structure tokens.
    
    Token Space:
    - [0-19]: Amino acid tokens (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)
    - [20-531]: Structure tokens (512 codebook entries from k-means on ESM-2 embeddings)
    - [532]: PAD token
    - [533]: UNK token
    - [534]: BOS token (begin of sequence)
    - [535]: EOS token (end of sequence)
    
    Features:
    - Extracts ESM-2 embeddings for each residue
    - Maps embeddings to nearest codebook centroid
    - Generates interleaved token stream: [BOS, aa_1, struct_1, aa_2, struct_2, ..., EOS]
    """
    
    # Standard 20 amino acids
    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
    
    def __init__(
        self,
        codebook_path: str,
        esm_model_name: str = "facebook/esm2_t33_650M_UR50D",
        device: str = None
    ):
        """
        Initialize the tokenizer.
        
        Args:
            codebook_path: Path to trained k-means codebook (.pkl file)
            esm_model_name: ESM-2 model identifier from HuggingFace
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Initializing ProteinStructureTokenizer on {self.device}")
        
        # Load codebook
        print(f"📂 Loading codebook from: {codebook_path}")
        with open(codebook_path, 'rb') as f:
            self.codebook_data = pickle.load(f)
        
        # Extract centroids from kmeans object
        kmeans = self.codebook_data['kmeans']
        self.centroids = kmeans.cluster_centers_  # (K, D) array
        self.n_clusters = self.codebook_data['n_clusters']
        self.embedding_dim = self.codebook_data['embedding_dim']
        print(f"   ✓ Loaded {self.n_clusters} centroids (dim={self.embedding_dim})")
        
        # Load ESM-2 model
        print(f"🧬 Loading ESM-2 model: {esm_model_name}")
        self.esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
        self.esm_model = EsmModel.from_pretrained(esm_model_name).to(self.device)
        self.esm_model.eval()
        print(f"   ✓ Model loaded successfully")
        
        # Build amino acid vocabulary
        self.aa_to_id = {aa: i for i, aa in enumerate(self.AMINO_ACIDS)}
        self.id_to_aa = {i: aa for aa, i in self.aa_to_id.items()}
        
        # Special tokens
        self.PAD_TOKEN = self.n_clusters + 20
        self.UNK_TOKEN = self.n_clusters + 21
        self.BOS_TOKEN = self.n_clusters + 22
        self.EOS_TOKEN = self.n_clusters + 23
        
        self.vocab_size = self.n_clusters + 24  # 20 AA + 512 struct + 4 special
        
        print(f"📊 Token Space Summary:")
        print(f"   - Amino acids: [0-19]")
        print(f"   - Structure tokens: [20-{19 + self.n_clusters}]")
        print(f"   - PAD: {self.PAD_TOKEN}")
        print(f"   - UNK: {self.UNK_TOKEN}")
        print(f"   - BOS: {self.BOS_TOKEN}")
        print(f"   - EOS: {self.EOS_TOKEN}")
        print(f"   - Total vocab size: {self.vocab_size}")
        print()
    
    def _extract_esm_embeddings(self, sequence: str) -> np.ndarray:
        """
        Extract per-residue ESM-2 embeddings.
        
        Args:
            sequence: Amino acid sequence string
            
        Returns:
            embeddings: (L, D) numpy array where L = len(sequence)
        """
        # Tokenize
        inputs = self.esm_tokenizer(
            sequence,
            return_tensors="pt",
            add_special_tokens=True
        ).to(self.device)
        
        # Extract embeddings
        with torch.no_grad():
            outputs = self.esm_model(**inputs)
            # Remove BOS and EOS tokens to get per-residue embeddings
            embeddings = outputs.last_hidden_state[0, 1:-1, :].cpu().numpy()
        
        return embeddings  # (L, D)
    
    def _get_structure_tokens(self, embeddings: np.ndarray) -> List[int]:
        """
        Map embeddings to nearest codebook centroids.
        
        Args:
            embeddings: (L, D) array of per-residue embeddings
            
        Returns:
            structure_tokens: List of structure token IDs (offset by 20)
        """
        # Compute distances to all centroids
        # distances[i, j] = distance from residue i to centroid j
        distances = np.linalg.norm(
            embeddings[:, None, :] - self.centroids[None, :, :],
            axis=2
        )
        
        # Find nearest centroid for each residue
        nearest_centroids = np.argmin(distances, axis=1)
        
        # Offset by 20 to get structure token IDs
        structure_tokens = (nearest_centroids + 20).tolist()
        
        return structure_tokens
    
    def encode(
        self,
        sequence: str,
        add_special_tokens: bool = True,
        return_structure_tokens: bool = False
    ) -> Dict:
        """
        Encode a protein sequence into multimodal tokens.
        
        Args:
            sequence: Amino acid sequence (e.g., "MKTAYIAKQR")
            add_special_tokens: Whether to add BOS/EOS tokens
            return_structure_tokens: Whether to return structure tokens separately
            
        Returns:
            Dictionary containing:
                - 'tokens': Interleaved token stream [aa_1, struct_1, aa_2, struct_2, ...]
                - 'aa_tokens': Amino acid tokens only
                - 'structure_tokens': Structure tokens only (if requested)
                - 'sequence': Original sequence
        """
        # Clean sequence
        sequence = sequence.upper().strip()
        
        # Convert amino acids to tokens
        aa_tokens = []
        for aa in sequence:
            if aa in self.aa_to_id:
                aa_tokens.append(self.aa_to_id[aa])
            else:
                aa_tokens.append(self.UNK_TOKEN)
        
        # Extract ESM embeddings and get structure tokens
        embeddings = self._extract_esm_embeddings(sequence)
        structure_tokens = self._get_structure_tokens(embeddings)
        
        # Interleave amino acid and structure tokens
        interleaved_tokens = []
        if add_special_tokens:
            interleaved_tokens.append(self.BOS_TOKEN)
        
        for aa_tok, struct_tok in zip(aa_tokens, structure_tokens):
            interleaved_tokens.append(aa_tok)
            interleaved_tokens.append(struct_tok)
        
        if add_special_tokens:
            interleaved_tokens.append(self.EOS_TOKEN)
        
        result = {
            'tokens': interleaved_tokens,
            'aa_tokens': aa_tokens,
            'sequence': sequence,
            'length': len(sequence)
        }
        
        if return_structure_tokens:
            result['structure_tokens'] = structure_tokens
        
        return result
    
    def decode(self, tokens: List[int], skip_special: bool = True) -> str:
        """
        Decode tokens back to amino acid sequence.
        
        Args:
            tokens: List of token IDs
            skip_special: Whether to skip special tokens
            
        Returns:
            Decoded amino acid sequence
        """
        sequence = []
        for tok in tokens:
            if skip_special and tok in [self.BOS_TOKEN, self.EOS_TOKEN, self.PAD_TOKEN]:
                continue
            if 0 <= tok < 20:  # Amino acid token
                sequence.append(self.id_to_aa[tok])
            # Structure tokens and UNK are skipped in decoding
        
        return ''.join(sequence)
    
    def batch_encode(self, sequences: List[str], **kwargs) -> List[Dict]:
        """Encode multiple sequences."""
        return [self.encode(seq, **kwargs) for seq in sequences]
    
    def get_token_name(self, token_id: int) -> str:
        """Get human-readable name for a token ID."""
        if 0 <= token_id < 20:
            return f"AA:{self.id_to_aa[token_id]}"
        elif 20 <= token_id < 20 + self.n_clusters:
            return f"STRUCT:{token_id - 20}"
        elif token_id == self.PAD_TOKEN:
            return "PAD"
        elif token_id == self.UNK_TOKEN:
            return "UNK"
        elif token_id == self.BOS_TOKEN:
            return "BOS"
        elif token_id == self.EOS_TOKEN:
            return "EOS"
        else:
            return f"INVALID:{token_id}"


def demo_tokenization():
    """Demonstrate the tokenizer with example sequences."""
    
    print("=" * 80)
    print("PROTEIN STRUCTURE TOKENIZER DEMO")
    print("=" * 80)
    print()
    
    # Initialize tokenizer
    codebook_path = "data/structure_codebook_K512.pkl"
    tokenizer = ProteinStructureTokenizer(
        codebook_path=codebook_path,
        esm_model_name="facebook/esm2_t33_650M_UR50D"
    )
    
    # Example sequences
    examples = [
        ("Short peptide", "MKTAYIAKQR"),
        ("Medium protein", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"),
        ("Helix-rich", "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNGGHFLRILPDGTVDGTRDRSDQHIQLQLSAESVGEVYIKSTETGQYLAMDTSGLLYGSQTPSEECLFLERLEENHYNTYTSKKHAEKNWFVGLKKNGSCKRGPRTHYGQKAILFLPLPV"),
    ]
    
    print("\n" + "=" * 80)
    print("TOKENIZATION EXAMPLES")
    print("=" * 80)
    
    for i, (name, sequence) in enumerate(examples, 1):
        print(f"\n{'─' * 80}")
        print(f"Example {i}: {name}")
        print(f"{'─' * 80}")
        print(f"Sequence: {sequence}")
        print(f"Length: {len(sequence)} residues")
        
        # Encode
        result = tokenizer.encode(sequence, return_structure_tokens=True)
        
        print(f"\n📊 Tokenization Results:")
        print(f"   Total tokens: {len(result['tokens'])}")
        print(f"   AA tokens: {len(result['aa_tokens'])}")
        print(f"   Structure tokens: {len(result['structure_tokens'])}")
        
        # Show first 10 tokens
        print(f"\n🔍 First 10 tokens (interleaved):")
        for j, tok in enumerate(result['tokens'][:10]):
            print(f"   [{j}] {tok:3d} -> {tokenizer.get_token_name(tok)}")
        
        if len(result['tokens']) > 10:
            print(f"   ... ({len(result['tokens']) - 10} more tokens)")
        
        # Show token distribution
        aa_count = sum(1 for t in result['tokens'] if 0 <= t < 20)
        struct_count = sum(1 for t in result['tokens'] if 20 <= t < 20 + tokenizer.n_clusters)
        special_count = len(result['tokens']) - aa_count - struct_count
        
        print(f"\n📈 Token Distribution:")
        print(f"   Amino acid tokens: {aa_count} ({aa_count/len(result['tokens'])*100:.1f}%)")
        print(f"   Structure tokens: {struct_count} ({struct_count/len(result['tokens'])*100:.1f}%)")
        print(f"   Special tokens: {special_count} ({special_count/len(result['tokens'])*100:.1f}%)")
        
        # Decode
        decoded = tokenizer.decode(result['tokens'])
        print(f"\n✅ Decoded sequence: {decoded}")
        print(f"   Match: {decoded == sequence}")
        
        # Show structure token statistics
        struct_toks = result['structure_tokens']
        unique_struct = len(set(struct_toks))
        print(f"\n🧬 Structure Token Statistics:")
        print(f"   Unique structure tokens used: {unique_struct}/{tokenizer.n_clusters}")
        print(f"   Most common structure token: {max(set(struct_toks), key=struct_toks.count) - 20}")
        print(f"   Structure token range: [{min(struct_toks) - 20}, {max(struct_toks) - 20}]")
    
    print("\n" + "=" * 80)
    print("USAGE FOR LLM TRAINING")
    print("=" * 80)
    print("""
The tokenized sequences can be used directly for LLM training:

1. **Input to LLM**: Use the 'tokens' list as input_ids
   Example: model(input_ids=torch.tensor([result['tokens']]))

2. **Embedding Layer**: Create an embedding layer with vocab_size={vocab_size}
   Example: nn.Embedding(num_embeddings={vocab_size}, embedding_dim=768)

3. **Training Data Format**:
   {{
       'input_ids': result['tokens'],
       'attention_mask': [1] * len(result['tokens']),
       'labels': <target_text_tokens>
   }}

4. **Batch Processing**: Use tokenizer.batch_encode() for multiple sequences
   Example: results = tokenizer.batch_encode([seq1, seq2, seq3])

5. **Integration with Transformers**:
   - The tokenizer can be saved and loaded like HuggingFace tokenizers
   - Compatible with standard training loops
   - Can be pushed to HuggingFace Hub for sharing
    """.format(vocab_size=tokenizer.vocab_size))
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE! ✅")
    print("=" * 80)


if __name__ == "__main__":
    demo_tokenization()

