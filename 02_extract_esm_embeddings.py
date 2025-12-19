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
import sys
warnings.filterwarnings('ignore')


def extract_esm_embeddings_batch(sequences, batch_start_idx, model, tokenizer, device='cuda'):
    """
    Extract per-residue embeddings from ESM-2 for a batch of sequences
    
    Args:
        sequences: List of protein sequences for this batch
        batch_start_idx: Starting protein index for this batch
        model: ESM-2 model
        tokenizer: ESM tokenizer
        device: 'cuda' or 'cpu'
    
    Returns:
        all_embeddings: numpy array [total_residues, hidden_dim]
        metadata: list of dicts with protein info
    """
    all_embeddings = []
    metadata = []
    
    model.eval()
    
    for i, sequence in enumerate(tqdm(sequences, desc="Extracting embeddings", leave=False)):
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
                'protein_idx': batch_start_idx + i,
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
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print(f"❌ Batch number required!")
        print(f"   Usage: python 02_extract_esm_embeddings.py <batch_number>")
        print(f"   Example: python 02_extract_esm_embeddings.py 0")
        return
    
    try:
        batch_to_process = int(sys.argv[1])
        print(f"\n📌 Processing batch {batch_to_process}")
    except ValueError:
        print(f"❌ Invalid batch number: {sys.argv[1]}")
        print(f"   Usage: python 02_extract_esm_embeddings.py <batch_number>")
        return
    
    # Configuration
    DATA_DIR = Path('esmfold_tokenizer/data/UniProt_Function')
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

    # Shuffle sequences for balanced batches
    np.random.seed(42)
    indices = np.random.permutation(len(sequences))
    sequences = [sequences[i] for i in indices]
    
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
    print("STEP 3: Extracting Embeddings (Batch Processing)")
    print("=" * 70)
    
    BATCH_SIZE = 10000  # Process 10k sequences per batch
    num_batches = (len(sequences) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Processing {len(sequences):,} sequences in {num_batches} batch(es) of {BATCH_SIZE:,}")
    
    # Determine which batches to process
    if batch_to_process < 0 or batch_to_process >= num_batches:
        print(f"❌ Batch {batch_to_process} out of range [0-{num_batches-1}]")
        return
    batches_to_run = [batch_to_process]
    
    for batch_num in batches_to_run:
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(sequences))
        batch_sequences = sequences[start_idx:end_idx]
        
        print(f"\n{'='*70}")
        print(f"Batch {batch_num + 1}/{num_batches}: Processing sequences {start_idx:,}-{end_idx:,}")
        print(f"{'='*70}")
        
        # Extract embeddings for this batch
        from datetime import datetime
        start = datetime.now()

        embeddings_batch, metadata_batch = extract_esm_embeddings_batch(
            batch_sequences, start_idx, model, tokenizer, device
        )
        
        print(f"✅ Batch {batch_num + 1} extraction complete")
        print(f"   Shape: {embeddings_batch.shape}")
        print(f"   Size in memory: {embeddings_batch.nbytes / 1024 / 1024:.1f} MB")
        end = datetime.now()
        print(f"   Extraction time: {(end - start).total_seconds():.2f} seconds")
        
        # Save batch data immediately
        batch_embeddings_file = f"{DATA_DIR}/raw_esm_embeddings/esm_embeddings_batch_{batch_num}.npy"
        np.save(batch_embeddings_file, embeddings_batch)
        print(f"✅ Saved batch embeddings: {batch_embeddings_file}")
        
        batch_metadata_file = f"{DATA_DIR}/raw_esm_embeddings/embedding_metadata_batch_{batch_num}.json"
        with open(batch_metadata_file, 'w') as f:
            json.dump(metadata_batch, f, indent=2)
        print(f"✅ Saved batch metadata: {batch_metadata_file}")
    
    print(f"\n✅ Batch {batch_to_process} extraction complete!")
    print(f"\n📌 Individual batch files saved:")
    print(f"   esm_embeddings_batch_{batch_to_process}.npy")
    print(f"   embedding_metadata_batch_{batch_to_process}.json")


if __name__ == '__main__':
    main()




