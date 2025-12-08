#!/usr/bin/env python3
"""
Extract Protein-Text Embedding Pairs
Extracts sequence embeddings (ESM-2) and text embeddings (BioGPT) from paired proteins
Outputs JSON with embedding pairs for alignment training
"""

import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, EsmModel, AutoModel
from tqdm import tqdm
import warnings
import sys

warnings.filterwarnings('ignore')


def extract_esm_embeddings(sequence, model, tokenizer, device='cuda'):
    """
    Extract sequence embedding using ESM-2 model for a single sequence.
    Returns sequence embedding (mean pooling of all residues).
    
    Args:
        sequence: Protein sequence string
        model: ESM-2 model
        tokenizer: ESM tokenizer
        device: 'cuda' or 'cpu'
    
    Returns:
        embedding: numpy array [embedding_dim]
    """
    # Tokenize
    inputs = tokenizer([sequence], return_tensors="pt", padding=False, truncation=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Extract embeddings
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
        # Get last hidden state, remove <cls> and <eos> tokens
        hidden_states = outputs.hidden_states[-1]
        embeddings = hidden_states[0, 1:-1, :].cpu().numpy()
        
        # Mean pooling to get sequence-level embedding
        seq_embedding = embeddings.mean(axis=0)
    
    return seq_embedding


def extract_text_embedding(text, model, tokenizer, device='cuda'):
    """
    Extract text embedding using BioGPT model for a single text.
    
    Args:
        text: Text description string
        model: BioGPT model
        tokenizer: BioGPT tokenizer
        device: 'cuda' or 'cpu'
    
    Returns:
        embedding: numpy array [embedding_dim]
    """
    # Tokenize
    inputs = tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Extract embeddings
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
        # Get last hidden state and mean pool
        hidden_states = outputs.hidden_states[-1]
        
        # Mean pooling over sequence length
        attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_states.shape)
        masked_hidden_states = hidden_states * attention_mask
        sum_hidden_states = masked_hidden_states.sum(dim=1)
        seq_lengths = attention_mask.sum(dim=1)
        pooled = sum_hidden_states / seq_lengths
    
    return pooled.cpu().numpy()[0]

def main():
    print("=" * 70)
    print("EXTRACT PROTEIN-TEXT EMBEDDING PAIRS")
    print("=" * 70)
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print(f"❌ Batch number required!")
        print(f"   Usage: python 06_extract_text_embeddings.py <batch_number>")
        print(f"   Example: python 06_extract_text_embeddings.py 0")
        return

    try:
        batch_to_process = int(sys.argv[1])
        print(f"\n📌 Processing batch {batch_to_process}")
    except ValueError:
        print(f"❌ Invalid batch number: {sys.argv[1]}")
        return

    # Configuration
    DATA_DIR = Path('esmfold_tokenizer/data')
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
    BIOGPT_MODEL_NAME = "microsoft/BioGPT-Large"
    
    input_file = DATA_DIR / 'sample_proteins.json'
    output_dir = DATA_DIR / 'embedding_pairs'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        print(f"   Run 01_fetch_sample_data.py first!")
        return

    print(f"\n📂 Data directory: {DATA_DIR}")
    print(f"🤖 ESM Model: {ESM_MODEL_NAME}")
    print(f"🤖 BioGPT Model: {BIOGPT_MODEL_NAME}")
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Device: {device}")
    
    if device == 'cpu':
        print("⚠️  WARNING: Running on CPU - will be VERY slow!")
    else:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load data
    print("\n" + "=" * 70)
    print("STEP 1: Loading Data")
    print("=" * 70)
    
    with open(input_file, 'r') as f:
        proteins = json.load(f)
    
    sequences = [p['sequence'] for p in proteins]
    texts = [p['text'] for p in proteins]
    protein_ids = [p['id'] for p in proteins]
    
    # Shuffle with fixed seed for reproducibility
    np.random.seed(42)
    shuffle_indices = np.arange(len(proteins))
    np.random.shuffle(shuffle_indices)
    
    sequences = [sequences[i] for i in shuffle_indices]
    texts = [texts[i] for i in shuffle_indices]
    protein_ids = [protein_ids[i] for i in shuffle_indices]
    
    # Remove prefix from texts
    prefix = "The determined function(s) of the protein include(s): "
    texts = [t[len(prefix):] if t.startswith(prefix) else t for t in texts]
    
    print(f"✅ Loaded {len(proteins)} proteins (shuffled with seed=42)")
    
    # Load Models
    print("\n" + "=" * 70)
    print("STEP 2: Loading Models")
    print("=" * 70)
    
    print(f"Loading ESM-2 {ESM_MODEL_NAME}...")
    esm_tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
    esm_model = EsmModel.from_pretrained(ESM_MODEL_NAME)
    esm_model = esm_model.to(device)
    print(f"✅ ESM-2 loaded (dim: {esm_model.config.hidden_size})")
    
    print(f"Loading BioGPT {BIOGPT_MODEL_NAME}...")
    biogpt_tokenizer = AutoTokenizer.from_pretrained(BIOGPT_MODEL_NAME)
    biogpt_model = AutoModel.from_pretrained(BIOGPT_MODEL_NAME)
    biogpt_model = biogpt_model.to(device)
    print(f"✅ BioGPT loaded (dim: {biogpt_model.config.hidden_size})")
    
    # Batch processing
    print("\n" + "=" * 70)
    print("STEP 3: Extracting Embeddings")
    print("=" * 70)
    
    BATCH_SIZE = 10000  # Process 10000 proteins per batch
    num_batches = (len(sequences) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Processing {len(sequences):,} proteins in {num_batches} batch(es) of {BATCH_SIZE}")
    
    if batch_to_process < 0 or batch_to_process >= num_batches:
        print(f"❌ Batch {batch_to_process} out of range [0-{num_batches-1}]")
        return
    
    start_idx = batch_to_process * BATCH_SIZE
    end_idx = min(start_idx + BATCH_SIZE, len(sequences))
    
    batch_sequences = sequences[start_idx:end_idx]
    batch_texts = texts[start_idx:end_idx]
    batch_ids = protein_ids[start_idx:end_idx]
    
    print(f"\n{'='*70}")
    print(f"Batch {batch_to_process + 1}/{num_batches}: Processing proteins {start_idx:,}-{end_idx:,}")
    print(f"{'='*70}")
    
    # Extract embeddings for each protein pair
    print("\n📊 Extracting embedding pairs...")
    embedding_pairs = []
    
    esm_model.eval()
    biogpt_model.eval()
    
    for i, (protein_id, sequence, text) in enumerate(
        tqdm(zip(batch_ids, batch_sequences, batch_texts), total=len(batch_ids), desc="Embedding pairs")
    ):
        # Extract sequence embedding
        seq_emb = extract_esm_embeddings(sequence, esm_model, esm_tokenizer, device)
        
        # Extract text embedding
        text_emb = extract_text_embedding(text, biogpt_model, biogpt_tokenizer, device)
        
        # Create pair
        pair = {
            "protein_id": protein_id,
            "global_index": start_idx + i,
            "sequence_embedding": seq_emb.tolist(),  # ESM-2 embedding
            "text_embedding": text_emb.tolist(),      # BioGPT embedding
            "sequence_embedding_dim": len(seq_emb),
            "text_embedding_dim": len(text_emb)
        }
        embedding_pairs.append(pair)
    
    print(f"✅ Extracted {len(embedding_pairs)} embedding pairs")
    
    # Save embedding pairs
    output_file = output_dir / f'embedding_pairs_batch_{batch_to_process}.json'
    with open(output_file, 'w') as f:
        json.dump(embedding_pairs, f)
    
    print(f"✅ Saved {len(embedding_pairs)} embedding pairs to: {output_file}")
    
    # Save metadata
    metadata = {
        "batch_number": batch_to_process,
        "total_batches": num_batches,
        "num_pairs": len(embedding_pairs),
        "start_index": start_idx,
        "end_index": end_idx,
        "esm_model": ESM_MODEL_NAME,
        "biogpt_model": BIOGPT_MODEL_NAME,
        "esm_embedding_dim": len(embedding_pairs[0]["sequence_embedding"]) if embedding_pairs else 0,
        "text_embedding_dim": len(embedding_pairs[0]["text_embedding"]) if embedding_pairs else 0
    }
    
    metadata_file = output_dir / f'metadata_batch_{batch_to_process}.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved metadata to: {metadata_file}")
    
    print("\n" + "=" * 70)
    print("✅ BATCH EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Files saved:")
    print(f"  - {output_file.name}")
    print(f"  - {metadata_file.name}")


if __name__ == '__main__':
    main()
