#!/usr/bin/env python3
"""
Extract ESMFold Embeddings from Standardized Dataset
Uses ESMFold trunk (ESM-2) for structure-aware per-residue representations
"""

import json
import numpy as np
import torch
import gc
from pathlib import Path
from transformers import AutoTokenizer, EsmForProteinFolding
from tqdm import tqdm
import warnings
import sys
warnings.filterwarnings('ignore')


def extract_esm_embeddings_batch(sequences, batch_start_idx, model, tokenizer, device='cuda'):
    """
    Extract per-residue embeddings from ESMFold trunk for a batch of sequences
    
    Args:
        sequences: List of protein sequences for this batch
        batch_start_idx: Starting protein index for this batch
        model: ESMFold model
        tokenizer: ESMFold tokenizer
        device: 'cuda' or 'cpu'
    
    Returns:
        all_embeddings: numpy array [total_residues, hidden_dim]
        metadata: list of dicts with protein info
    """
    all_embeddings = []
    metadata = []
    
    model.eval()
    
    skipped_count = 0
    replaced_count = 0
    for i, sequence in enumerate(tqdm(sequences, desc="Extracting embeddings", leave=False)):
        # Skip invalid sequences (None, empty, or whitespace-only)
        if sequence is None or not sequence or not sequence.strip():
            skipped_count += 1
            continue
        
        # Replace non-standard amino acids with Alanine (A) - most neutral/common
        sequence = sequence.upper().strip()
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        if not all(aa in valid_aa for aa in sequence):
            # Replace non-standard AAs (X, B, Z, U, O, etc.) with A
            original_seq = sequence
            sequence = ''.join([aa if aa in valid_aa else 'A' for aa in sequence])
            replaced_count += 1
            
        # Tokenize
        # ESMFold max length is 1024 including special tokens
        sequence = sequence[:1022]  
        inputs = tokenizer([sequence], return_tensors="pt", padding=False, truncation=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Extract embeddings from ESMFold trunk (before structure prediction heads)
        with torch.no_grad():
            # Access the ESM trunk directly
            outputs = model.esm(**inputs, output_hidden_states=True)
            
            # Get last hidden state from the trunk
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
    
    if replaced_count > 0:
        print(f"⚠️  Replaced non-standard AAs with 'A' in {replaced_count} sequences")
    
    # Stack all embeddings
    all_embeddings_stacked = np.vstack(all_embeddings)    
    return all_embeddings_stacked, metadata

def main():
    print("=" * 70)
    print("EXTRACT ESMFOLD EMBEDDINGS FROM STANDARDIZED DATASET")
    print("=" * 70)
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print(f"❌ Batch number required!")
        print(f"   Usage: python 01_extract_esm_embeddings.py <batch_number>")
        print(f"   Example: python 01_extract_esm_embeddings.py 0")
        print(f"   Use -1 to process all batches sequentially")
        return
    
    try:
        batch_to_process = int(sys.argv[1])
        if batch_to_process == -1:
            print(f"\n📌 Processing ALL batches sequentially")
        else:
            print(f"\n📌 Processing batch {batch_to_process}")
    except ValueError:
        print(f"❌ Invalid batch number: {sys.argv[1]}")
        print(f"   Usage: python 01_extract_esm_embeddings.py <batch_number>")
        return
    
    # Configuration
    DATA_DIR = Path(__file__).parent.parent / 'data'
    MODEL_NAME = "facebook/esmfold_v1"  # ESMFold with structure-aware embeddings, 1280-dim
    
    input_file = DATA_DIR / 'standardized_protein_instructions.json'
    output_dir = DATA_DIR / 'esm_embeddings'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        print(f"   Run 00_standardize_datasets.py first!")
        return
    
    print(f"\n📂 Data directory: {DATA_DIR}")
    print(f"📥 Input file: {input_file}")
    print(f"📤 Output directory: {output_dir}")
    print(f"🤖 Model: {MODEL_NAME}")
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Device: {device}")
    
    if device == 'cpu':
        print("⚠️  WARNING: Running on CPU - will be VERY slow!")
        print("   Recommend running on GPU node")
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
    
    # Load ESMFold model
    print("\n" + "=" * 70)
    print("STEP 2: Loading ESMFold Model")
    print("=" * 70)
    
    print(f"Loading {MODEL_NAME}...")
    print("ℹ️  First run will download model (~3 GB)")
    print("ℹ️  Extracting embeddings from ESM trunk (before structure heads)")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmForProteinFolding.from_pretrained(MODEL_NAME)
    model.eval()
    model = model.to(device)
    
    print(f"✅ Model loaded")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    print(f"   Hidden dimension: {model.esm.config.hidden_size}")
    print(f"   Layers: {model.esm.config.num_hidden_layers}")
    
    # Extract embeddings
    print("\n" + "=" * 70)
    print("STEP 3: Extracting Embeddings (Batch Processing)")
    print("=" * 70)
    
    BATCH_SIZE = 10000  # Process 10k sequences per batch
    num_batches = (len(sequences) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Processing {len(sequences):,} sequences in {num_batches} batch(es) of {BATCH_SIZE:,}")
    
    # Determine which batches to process
    if batch_to_process == -1:
        # Process all batches
        batches_to_process = list(range(num_batches))
        print(f"Will process all {num_batches} batches")
    elif batch_to_process < 0 or batch_to_process >= num_batches:
        print(f"❌ Batch {batch_to_process} out of range [0-{num_batches-1}]")
        return
    else:
        batches_to_process = [batch_to_process]
    
    # Process batches
    from datetime import datetime
    
    for batch_num in batches_to_process:
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(sequences))
        batch_sequences = sequences[start_idx:end_idx]
        
        print(f"\n{'='*70}")
        print(f"Batch {batch_num + 1}/{num_batches}: Processing sequences {start_idx:,}-{end_idx:,}")
        print(f"{'='*70}")
        
        # Extract embeddings for this batch
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
        batch_embeddings_file = output_dir / f"esm_embeddings_batch_{batch_num}.npy"
        np.save(batch_embeddings_file, embeddings_batch)
        print(f"✅ Saved batch embeddings: {batch_embeddings_file}")
        
        batch_metadata_file = output_dir / f"embedding_metadata_batch_{batch_num}.json"
        with open(batch_metadata_file, 'w') as f:
            json.dump(metadata_batch, f, indent=2)
        print(f"✅ Saved batch metadata: {batch_metadata_file}")
        
        # Clean up memory after each batch
        del embeddings_batch
        del metadata_batch
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        print(f"🧹 Memory cleaned up after batch {batch_num}")
    
    # If all batches were processed, create combined metadata
    if batch_to_process == -1:
        print("\n" + "=" * 70)
        print("COMBINING METADATA FILES")
        print("=" * 70)
        
        print(f"📂 Combining {num_batches} metadata files...")
        combined_metadata = []
        total_proteins = 0
        total_residues = 0
        
        for batch_num in range(num_batches):
            batch_metadata_file = output_dir / f"embedding_metadata_batch_{batch_num}.json"
            if batch_metadata_file.exists():
                with open(batch_metadata_file, 'r') as f:
                    batch_metadata = json.load(f)
                combined_metadata.extend(batch_metadata)
                total_proteins += len(batch_metadata)
                total_residues += sum(m['length'] for m in batch_metadata)
        
        print(f"✅ Combined metadata:")
        print(f"   Total proteins: {total_proteins:,}")
        print(f"   Total residues: {total_residues:,}")
        
        # Save combined metadata
        combined_metadata_file = output_dir / 'embedding_metadata.json'
        with open(combined_metadata_file, 'w') as f:
            json.dump(combined_metadata, f, indent=2)
        
        print(f"✅ Saved combined metadata: {combined_metadata_file}")
        print(f"   File size: {combined_metadata_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    if batch_to_process == -1:
        print(f"\n✅ All {num_batches} batches extraction complete!")
    else:
        print(f"\n✅ Batch {batch_to_process} extraction complete!")
        print(f"\n📌 Individual batch files saved:")
        print(f"   esm_embeddings_batch_{batch_to_process}.npy")
        print(f"   embedding_metadata_batch_{batch_to_process}.json")


if __name__ == '__main__':
    main()
