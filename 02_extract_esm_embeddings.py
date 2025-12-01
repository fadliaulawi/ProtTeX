#!/usr/bin/env python3
"""
Extract ESM-2 Embeddings from Sample Proteins
Uses ESM-2 650M for rich per-residue representations
"""

import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, EsmModel
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def extract_esm_embeddings(sequences, model, tokenizer, device='cuda', batch_size=1):
    """
    Extract per-residue embeddings from ESM-2
    
    Args:
        sequences: List of protein sequences
        model: ESM-2 model
        tokenizer: ESM tokenizer
        device: 'cuda' or 'cpu'
        batch_size: Batch size (keep at 1 for variable length sequences)
    
    Returns:
        all_embeddings: numpy array [total_residues, hidden_dim]
        metadata: list of dicts with protein info
    """
    all_embeddings = []
    metadata = []
    
    model.eval()
    
    for i, sequence in enumerate(tqdm(sequences, desc="Extracting embeddings")):
        # Tokenize
        inputs = tokenizer([sequence], return_tensors="pt", padding=False, truncation=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Extract embeddings
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
            # Get last hidden state
            # Shape: [1, seq_len + 2, hidden_dim] (includes <cls> and <eos>)
            hidden_states = outputs.hidden_states[-1]
            
            # Remove <cls> and <eos> tokens
            embeddings = hidden_states[0, 1:-1, :].cpu().numpy()
            
            all_embeddings.append(embeddings)
            
            metadata.append({
                'protein_idx': i,
                'length': len(sequence),
                'start_idx': sum(len(e) for e in all_embeddings[:-1]),
                'end_idx': sum(len(e) for e in all_embeddings)
            })
    
    # Stack all embeddings
    all_embeddings_stacked = np.vstack(all_embeddings)
    
    return all_embeddings_stacked, metadata


def main():
    print("=" * 70)
    print("EXTRACT ESM-2 EMBEDDINGS")
    print("=" * 70)
    
    # Configuration
    DATA_DIR = Path('esmfold_tokenizer/data')
    MODEL_NAME = "facebook/esm2_t33_650M_UR50D"  # 650M params, 1280-dim embeddings
    
    input_file = DATA_DIR / 'sample_proteins.json'
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        print(f"   Run 01_fetch_sample_data.py first!")
        return
    
    print(f"\n📂 Data directory: {DATA_DIR}")
    print(f"🤖 Model: {MODEL_NAME}")
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Device: {device}")
    
    if device == 'cpu':
        print("⚠️  WARNING: Running on CPU - will be VERY slow!")
        print("   Recommend running on GPU node with SLURM")
    else:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load data
    print("\n" + "=" * 70)
    print("STEP 1: Loading Data")
    print("=" * 70)
    
    with open(input_file, 'r') as f:
        samples = json.load(f)
    
    sequences = [s['sequence'] for s in samples]
    
    print(f"✅ Loaded {len(sequences)} sequences")
    print(f"   Total residues: {sum(len(s) for s in sequences):,}")
    
    # Load ESM-2 model
    print("\n" + "=" * 70)
    print("STEP 2: Loading ESM-2 Model")
    print("=" * 70)
    
    print(f"Loading {MODEL_NAME}...")
    print("ℹ️  First run will download model (~2.5 GB)")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME)
    model.eval()
    model = model.to(device)
    
    print(f"✅ Model loaded")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    print(f"   Hidden dimension: {model.config.hidden_size}")
    print(f"   Layers: {model.config.num_hidden_layers}")
    
    # Extract embeddings
    print("\n" + "=" * 70)
    print("STEP 3: Extracting Embeddings")
    print("=" * 70)
    
    embeddings, metadata = extract_esm_embeddings(sequences, model, tokenizer, device)
    
    print(f"\n✅ Extraction complete")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Size in memory: {embeddings.nbytes / 1024 / 1024:.1f} MB")
    
    # Statistics
    print("\n" + "=" * 70)
    print("STEP 4: Embedding Statistics")
    print("=" * 70)
    
    print(f"\nEmbedding properties:")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Mean: {embeddings.mean():.4f}")
    print(f"   Std: {embeddings.std():.4f}")
    print(f"   Min: {embeddings.min():.4f}")
    print(f"   Max: {embeddings.max():.4f}")
    
    # Save results
    print("\n" + "=" * 70)
    print("STEP 5: Saving Results")
    print("=" * 70)
    
    # Save embeddings
    embeddings_file = DATA_DIR / 'esm_embeddings.npy'
    np.save(embeddings_file, embeddings)
    print(f"✅ Saved embeddings: {embeddings_file}")
    print(f"   Size: {embeddings_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Save metadata
    metadata_file = DATA_DIR / 'embedding_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump({
            'model_name': MODEL_NAME,
            'num_proteins': len(sequences),
            'total_residues': embeddings.shape[0],
            'embedding_dim': embeddings.shape[1],
            'device': device,
            'protein_metadata': metadata
        }, f, indent=2)
    
    print(f"✅ Saved metadata: {metadata_file}")
    
    print("\n" + "=" * 70)
    print("✅ EMBEDDING EXTRACTION COMPLETE!")
    print("=" * 70)
    
    print(f"\n📊 Summary:")
    print(f"   Proteins: {len(sequences)}")
    print(f"   Total residues: {embeddings.shape[0]:,}")
    print(f"   Embedding dim: {embeddings.shape[1]}")
    print(f"   File size: {embeddings_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    print(f"\n🚀 Next step:")
    print(f"   python 03_train_kmeans_codebook.py")


if __name__ == '__main__':
    main()




