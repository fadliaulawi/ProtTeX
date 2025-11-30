#!/usr/bin/env python3
"""
Tokenize Proteins and Show Results
Demonstrates how the tokenizer works and how to use it
"""

import json
import numpy as np
import pickle
import torch
from pathlib import Path
from transformers import AutoTokenizer, EsmModel
import warnings
warnings.filterwarnings('ignore')


def tokenize_protein(sequence, esm_model, esm_tokenizer, kmeans, device='cuda'):
    """
    Tokenize a single protein into structure tokens
    
    Returns:
        sequence: Original amino acid sequence
        structure_tokens: Cluster IDs for each residue
        aa_tokens: Amino acid token IDs
    """
    # Extract ESM-2 embeddings
    inputs = esm_tokenizer([sequence], return_tensors="pt", padding=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = esm_model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        embeddings = hidden_states[0, 1:-1, :].cpu().numpy()  # Remove <cls>, <eos>
    
    # Assign to clusters
    structure_tokens = kmeans.predict(embeddings)
    
    # Convert amino acids to token IDs
    aa_to_id = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
    aa_tokens = [aa_to_id.get(aa, -1) for aa in sequence]
    
    return sequence, structure_tokens, aa_tokens, embeddings


def format_multimodal_sequence(sequence, structure_tokens, aa_tokens):
    """Format tokens for LLM input"""
    
    # Interleaved format: [AA, Structure, AA, Structure, ...]
    STRUCTURE_OFFSET = 20  # AAs are 0-19, structures start at 20
    
    interleaved = []
    for aa_tok, struct_tok in zip(aa_tokens, structure_tokens):
        interleaved.append(aa_tok)  # Amino acid token
        interleaved.append(STRUCTURE_OFFSET + struct_tok)  # Structure token
    
    return interleaved


def visualize_tokenization(sequence, structure_tokens, max_display=50):
    """Pretty print tokenization results"""
    
    display_len = min(len(sequence), max_display)
    
    print("\n" + "=" * 70)
    print("TOKENIZATION RESULT")
    print("=" * 70)
    
    print(f"\nProtein length: {len(sequence)} residues")
    print(f"Showing first {display_len} residues:\n")
    
    # Amino acid sequence
    print("Sequence:  ", end="")
    for i in range(display_len):
        print(f"{sequence[i]:>4}", end="")
    print(" ...")
    
    # Structure tokens
    print("Structure: ", end="")
    for i in range(display_len):
        print(f"S{structure_tokens[i]:03d}", end="")
    print(" ...")
    
    # Token statistics
    unique_tokens = len(set(structure_tokens))
    print(f"\n\nStructure token statistics:")
    print(f"  Unique tokens used: {unique_tokens}")
    print(f"  Most common: S{np.bincount(structure_tokens).argmax():03d} ({np.bincount(structure_tokens).max()} times)")
    
    # Token distribution
    from collections import Counter
    token_counts = Counter(structure_tokens)
    print(f"\n  Top 10 most frequent structure tokens:")
    for token, count in token_counts.most_common(10):
        print(f"    S{token:03d}: {count:3d} residues ({count/len(structure_tokens)*100:5.2f}%)")


def show_llm_format(sequence, structure_tokens, aa_tokens):
    """Show how tokens would be fed to LLM"""
    
    print("\n" + "=" * 70)
    print("LLM INPUT FORMAT")
    print("=" * 70)
    
    # Interleaved format
    interleaved = format_multimodal_sequence(sequence, structure_tokens, aa_tokens)
    
    print(f"\nInterleaved token stream (first 20 tokens):")
    print(f"  {interleaved[:20]}")
    print(f"\n  Total tokens: {len(interleaved)} (2x sequence length)")
    
    # Readable format
    print(f"\nReadable format (first 10 residues):")
    for i in range(min(10, len(sequence))):
        aa = sequence[i]
        aa_tok = aa_tokens[i]
        struct_tok = structure_tokens[i]
        print(f"  Position {i:2d}: {aa} (AA_token={aa_tok:2d}) + S{struct_tok:03d} (Structure_token={20+struct_tok:3d})")
    
    # Show prompt example
    print(f"\n" + "=" * 70)
    print("EXAMPLE LLM PROMPT")
    print("=" * 70)
    
    print(f"""
<protein>
  <sequence>{sequence[:30]}...</sequence>
  <structure>[S{structure_tokens[0]:03d}][S{structure_tokens[1]:03d}][S{structure_tokens[2]:03d}]...[S{structure_tokens[29]:03d}]</structure>
</protein>

Question: What is the function of this protein?

[LLM generates answer based on sequence + 3D structure tokens]
""")


def main():
    print("=" * 70)
    print("TOKENIZE PROTEINS - DEMONSTRATION")
    print("=" * 70)
    
    # Configuration
    DATA_DIR = Path('/lustrefs/shared/mohammad.sayeed/Prot2Text/esmfold_tokenizer/data')
    MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
    N_CLUSTERS = 512
    
    # Check files
    print("\n" + "=" * 70)
    print("STEP 1: Loading Resources")
    print("=" * 70)
    
    codebook_file = DATA_DIR / f'structure_codebook_K{N_CLUSTERS}.pkl'
    proteins_file = DATA_DIR / 'sample_proteins.json'
    
    if not codebook_file.exists():
        print(f"❌ Codebook not found: {codebook_file}")
        print(f"   Run 03_train_kmeans_codebook.py first!")
        return
    
    if not proteins_file.exists():
        print(f"❌ Proteins not found: {proteins_file}")
        print(f"   Run 01_fetch_sample_data.py first!")
        return
    
    # Load codebook
    print(f"\nLoading codebook...")
    with open(codebook_file, 'rb') as f:
        codebook_data = pickle.load(f)
        kmeans = codebook_data['kmeans']
    
    print(f"✅ Codebook loaded: {codebook_data['n_clusters']} structure tokens")
    
    # Load proteins
    print(f"\nLoading proteins...")
    with open(proteins_file, 'r') as f:
        proteins = json.load(f)
    
    print(f"✅ Loaded {len(proteins)} proteins")
    
    # Load ESM-2
    print(f"\nLoading ESM-2 model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME)
    model.eval()
    model = model.to(device)
    
    print(f"✅ Model loaded")
    
    # Tokenize examples
    print("\n" + "=" * 70)
    print("STEP 2: Tokenizing Example Proteins")
    print("=" * 70)
    
    for i, protein in enumerate(proteins[:3]):  # Show first 3
        print(f"\n{'=' * 70}")
        print(f"EXAMPLE {i+1}/{len(proteins[:3])}")
        print(f"{'=' * 70}")
        
        print(f"\nProtein ID: {protein['id']}")
        print(f"Length: {protein['length']} residues")
        print(f"Function: {protein['text'][:100]}...")
        
        # Tokenize
        sequence, structure_tokens, aa_tokens, embeddings = tokenize_protein(
            protein['sequence'],
            model,
            tokenizer,
            kmeans,
            device
        )
        
        # Visualize
        visualize_tokenization(sequence, structure_tokens, max_display=30)
        
        # Show LLM format
        if i == 0:  # Detailed format for first example
            show_llm_format(sequence, structure_tokens, aa_tokens)
    
    # Usage summary
    print("\n" + "=" * 70)
    print("HOW TO USE THIS TOKENIZER")
    print("=" * 70)
    
    print("""
1. **Training Phase (One-time):**
   - Extract ESM-2 embeddings from proteins
   - Train k-means codebook (512 clusters)
   - Save codebook
   
2. **Inference Phase (For each protein):**
   - Input: Protein sequence
   - Extract ESM-2 embeddings (per-residue)
   - Assign each residue to nearest cluster
   - Output: Sequence of structure tokens
   
3. **LLM Integration:**
   - Extend LLM vocabulary: +20 (AA) + 512 (structure) tokens
   - Create multimodal token stream: [AA, Structure, AA, Structure, ...]
   - Fine-tune LLM on protein-text pairs
   - Model learns structure tokens = 3D context
   
4. **Advantages Over ProtTEX:**
   - ESM-2 pre-trained on 250M sequences (vs custom encoder)
   - All 512 codes used (vs VQ-VAE collapse)
   - Modular design (upgrade embeddings easily)
   - Simpler training pipeline
""")
    
    print("\n" + "=" * 70)
    print("✅ DEMONSTRATION COMPLETE!")
    print("=" * 70)
    
    print(f"\n📊 Summary:")
    print(f"   Codebook: {N_CLUSTERS} structure tokens")
    print(f"   Embeddings: {codebook_data['embedding_dim']}-dimensional")
    print(f"   Model: ESM-2 650M")
    
    print(f"\n🚀 Next steps:")
    print(f"   1. Scale to full ProteinLMBench dataset")
    print(f"   2. Fine-tune Qwen/Kimi with structure tokens")
    print(f"   3. Benchmark on function prediction tasks")


if __name__ == '__main__':
    main()




