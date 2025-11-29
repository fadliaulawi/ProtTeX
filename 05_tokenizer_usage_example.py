#!/usr/bin/env python3
"""
Complete Example: Using ProteinStructureTokenizer

Shows all features of the tokenizer class including:
- Loading codebook
- Encoding sequences (AA + structure tokens)
- Different output formats
- Saving/loading tokenizer
- Integration with LLM
"""

import json
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from protein_structure_tokenizer import ProteinStructureTokenizer, TokenizerConfig


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70 + "\n")


def example_1_basic_usage():
    """Example 1: Basic tokenization"""
    print_section("EXAMPLE 1: Basic Usage")
    
    # Load tokenizer with trained codebook
    DATA_DIR = Path(__file__).parent / 'data'
    codebook_path = DATA_DIR / 'structure_codebook_K512.pkl'
    
    print(f"Loading tokenizer from: {codebook_path}")
    tokenizer = ProteinStructureTokenizer(codebook_path=str(codebook_path))
    
    print(f"✅ Tokenizer loaded:")
    print(tokenizer)
    
    # Encode a protein sequence
    sequence = "MALWMRLLPLLA"
    print(f"\n🧬 Encoding sequence: {sequence}")
    print(f"   Length: {len(sequence)} residues")
    
    tokens = tokenizer.encode(sequence, return_format="interleaved")
    
    print(f"\n📊 Encoded tokens: {tokens[:20]}...")
    print(f"   Total tokens: {len(tokens)} (2x sequence length)")
    
    # Decode back
    decoded = tokenizer.decode(tokens)
    print(f"\n✅ Decoded sequence: {decoded}")
    print(f"   Match: {decoded == sequence}")


def example_2_different_formats():
    """Example 2: Different output formats"""
    print_section("EXAMPLE 2: Different Output Formats")
    
    DATA_DIR = Path(__file__).parent / 'data'
    tokenizer = ProteinStructureTokenizer(
        codebook_path=str(DATA_DIR / 'structure_codebook_K512.pkl')
    )
    
    sequence = "MALW"
    print(f"Sequence: {sequence}\n")
    
    # Format 1: Interleaved (default)
    tokens_interleaved = tokenizer.encode(sequence, return_format="interleaved")
    print(f"Interleaved: {tokens_interleaved}")
    print(f"  [AA, Struct, AA, Struct, ...]")
    
    # Format 2: Separate streams
    tokens_separate = tokenizer.encode(sequence, return_format="separate")
    print(f"\nSeparate:")
    print(f"  AA tokens:     {tokens_separate['aa_tokens']}")
    print(f"  Struct tokens: {tokens_separate['structure_tokens']}")
    
    # Format 3: AA only
    tokens_aa = tokenizer.encode(sequence, return_format="aa_only")
    print(f"\nAA only: {tokens_aa}")
    
    # With special tokens
    tokens_special = tokenizer.encode(sequence, add_special_tokens=True)
    print(f"\nWith special tokens: {tokens_special[:5]}...{tokens_special[-2:]}")
    print(f"  <bos> at start, <eos> at end")


def example_3_detailed_tokenization():
    """Example 3: Detailed token breakdown"""
    print_section("EXAMPLE 3: Detailed Token Breakdown")
    
    DATA_DIR = Path(__file__).parent / 'data'
    tokenizer = ProteinStructureTokenizer(
        codebook_path=str(DATA_DIR / 'structure_codebook_K512.pkl')
    )
    
    sequence = "MALWMR"
    print(f"Sequence: {sequence}\n")
    
    tokens = tokenizer.encode(sequence, return_format="interleaved")
    
    print("Position | AA | AA_Token | Structure_Token | Cluster_ID")
    print("-" * 60)
    
    for i, aa in enumerate(sequence):
        aa_token = tokens[i*2]
        struct_token = tokens[i*2 + 1]
        cluster_id = struct_token - tokenizer.config.structure_token_offset
        
        print(f"   {i:2d}    | {aa}  |   {aa_token:3d}    |      {struct_token:3d}       |    {cluster_id:3d}")
    
    print(f"\nToken Info Examples:")
    for token_id in [tokens[0], tokens[1], tokenizer.bos_token_id]:
        info = tokenizer.get_token_info(token_id)
        print(f"  Token {token_id:3d}: {info}")


def example_4_batch_encoding():
    """Example 4: Batch encoding multiple sequences"""
    print_section("EXAMPLE 4: Batch Encoding")
    
    DATA_DIR = Path(__file__).parent / 'data'
    tokenizer = ProteinStructureTokenizer(
        codebook_path=str(DATA_DIR / 'structure_codebook_K512.pkl')
    )
    
    # Load sample proteins
    proteins_file = DATA_DIR / 'sample_proteins.json'
    with open(proteins_file, 'r') as f:
        proteins = json.load(f)
    
    # Encode first 3 proteins
    sequences = [p['sequence'][:20] for p in proteins[:3]]  # First 20 AAs
    
    print(f"Encoding {len(sequences)} sequences...\n")
    
    encoded = tokenizer.batch_encode(sequences, return_format="interleaved")
    
    for i, (seq, tokens) in enumerate(zip(sequences, encoded)):
        print(f"Protein {i+1}:")
        print(f"  Sequence: {seq[:10]}...")
        print(f"  Tokens: {len(tokens)} ({len(tokens)//2} residues × 2)")


def example_5_llm_integration():
    """Example 5: How to use with LLM"""
    print_section("EXAMPLE 5: LLM Integration")
    
    DATA_DIR = Path(__file__).parent / 'data'
    tokenizer = ProteinStructureTokenizer(
        codebook_path=str(DATA_DIR / 'structure_codebook_K512.pkl')
    )
    
    # Load a sample protein
    proteins_file = DATA_DIR / 'sample_proteins.json'
    with open(proteins_file, 'r') as f:
        protein = json.load(f)[0]
    
    sequence = protein['sequence'][:30]  # First 30 residues
    function = protein['text'][:100]  # First 100 chars
    
    print(f"Protein sequence: {sequence}")
    print(f"Function: {function}...\n")
    
    # Tokenize protein
    protein_tokens = tokenizer.encode(sequence, return_format="interleaved")
    
    print(f"LLM Training Example:")
    print(f"-" * 60)
    print(f"\n# Step 1: Tokenize protein")
    print(f"protein_tokens = tokenizer.encode(sequence)")
    print(f"# Result: {protein_tokens[:10]}...")
    
    print(f"\n# Step 2: Create prompt")
    print(f"prompt_tokens = [")
    print(f"    tokenizer.bos_token_id,  # <bos>")
    print(f"    *protein_tokens,          # Protein tokens")
    print(f"    tokenizer.sep_token_id,   # <sep>")
    print(f"    # ... text tokens for question ...")
    print(f"]")
    
    print(f"\n# Step 3: Feed to LLM")
    print(f"# The LLM learns that:")
    print(f"#   - Tokens 0-19: Amino acids")
    print(f"#   - Tokens 20-531: Structure context")
    print(f"#   - Tokens 532+: Text/special tokens")
    
    print(f"\n# Step 4: Fine-tune")
    print(f"# loss = llm(prompt_tokens, target_tokens)")
    print(f"# Model learns structure-aware protein understanding!")


def example_6_save_and_load():
    """Example 6: Save and load tokenizer"""
    print_section("EXAMPLE 6: Save and Load Tokenizer")
    
    DATA_DIR = Path(__file__).parent / 'data'
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = ProteinStructureTokenizer(
        codebook_path=str(DATA_DIR / 'structure_codebook_K512.pkl')
    )
    
    # Save to directory
    save_dir = DATA_DIR / 'tokenizer_saved'
    print(f"\nSaving to: {save_dir}")
    tokenizer.save_pretrained(str(save_dir))
    
    # Load from directory
    print(f"\nLoading from: {save_dir}")
    tokenizer_loaded = ProteinStructureTokenizer.from_pretrained(str(save_dir))
    
    # Test that it works
    sequence = "MALW"
    tokens_original = tokenizer.encode(sequence)
    tokens_loaded = tokenizer_loaded.encode(sequence)
    
    print(f"\n✅ Verification:")
    print(f"   Original: {tokens_original[:8]}")
    print(f"   Loaded:   {tokens_loaded[:8]}")
    print(f"   Match: {tokens_original == tokens_loaded}")
    
    print(f"\n💡 Now you can push '{save_dir.name}' to GitHub/HuggingFace!")


def example_7_vocab_info():
    """Example 7: Vocabulary information"""
    print_section("EXAMPLE 7: Vocabulary Information")
    
    DATA_DIR = Path(__file__).parent / 'data'
    tokenizer = ProteinStructureTokenizer(
        codebook_path=str(DATA_DIR / 'structure_codebook_K512.pkl')
    )
    
    print(f"📊 Tokenizer Vocabulary:\n")
    
    print(f"Total vocab size: {tokenizer.vocab_size}")
    print(f"\nBreakdown:")
    print(f"  Amino acids: {len(tokenizer.AA_VOCAB)} (IDs 0-19)")
    print(f"    {tokenizer.AA_VOCAB}")
    
    print(f"\n  Structure tokens: {tokenizer.config.n_structure_tokens} (IDs 20-531)")
    print(f"    S000, S001, ..., S511")
    
    print(f"\n  Special tokens: {len(tokenizer.special_tokens)} (IDs 532+)")
    for name, token_id in tokenizer.special_tokens.items():
        print(f"    {name}: {token_id}")
    
    print(f"\n📋 Example mappings:")
    print(f"  'M' → {tokenizer.aa_to_id['M']}")
    print(f"  'A' → {tokenizer.aa_to_id['A']}")
    print(f"  'L' → {tokenizer.aa_to_id['L']}")
    print(f"  'W' → {tokenizer.aa_to_id['W']}")


def main():
    """Run all examples"""
    print("=" * 70)
    print("PROTEIN STRUCTURE TOKENIZER - COMPLETE USAGE EXAMPLES")
    print("=" * 70)
    
    # Check if codebook exists
    DATA_DIR = Path(__file__).parent / 'data'
    codebook_path = DATA_DIR / 'structure_codebook_K512.pkl'
    
    if not codebook_path.exists():
        print(f"\n❌ Codebook not found: {codebook_path}")
        print(f"   Run 03_train_kmeans_codebook.py first!")
        return
    
    try:
        # Run all examples
        example_1_basic_usage()
        example_2_different_formats()
        example_3_detailed_tokenization()
        example_4_batch_encoding()
        example_5_llm_integration()
        example_6_save_and_load()
        example_7_vocab_info()
        
        # Summary
        print_section("✅ ALL EXAMPLES COMPLETE!")
        
        print("📚 What you learned:")
        print("  1. ✅ Load tokenizer with codebook")
        print("  2. ✅ Encode sequences (interleaved/separate/aa-only)")
        print("  3. ✅ Decode tokens back to sequence")
        print("  4. ✅ Batch encode multiple proteins")
        print("  5. ✅ Integrate with LLM training")
        print("  6. ✅ Save/load for sharing")
        print("  7. ✅ Understand vocabulary structure")
        
        print("\n🚀 Next steps:")
        print("  1. Use tokenizer in your LLM training pipeline")
        print("  2. Push to GitHub: github.com/your-repo/protein-tokenizer")
        print("  3. Share on HuggingFace: huggingface.co/your-name/protein-tokenizer")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()




