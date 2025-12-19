#!/usr/bin/env python3
"""
Extract Protein-Text Embedding Pairs with Structure Tokens
Extracts sequence embeddings (ESM-2 3B), structure tokens (k-means clusters), 
and text embeddings (Llama-3.1) from paired proteins
Outputs JSON with embedding triplets for alignment training
"""

import json
import numpy as np
import torch
import pickle
from pathlib import Path
from transformers import AutoTokenizer, EsmModel, AutoModel
from tqdm import tqdm
import warnings
import sys
import argparse

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
    Extract text embedding using Llama-3.1 model for a single text.
    
    Args:
        text: Text description string
        model: Llama-3.1 model
        tokenizer: Llama-3.1 tokenizer
        device: 'cuda' or 'cpu'
    
    Returns:
        embedding: numpy array [embedding_dim]
    """
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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

def extract_structure_tokens(sequence, esm_model, esm_tokenizer, kmeans, k_clusters, device='cuda', token_offset=20, max_length=1024):
    """
    Extract per-residue structure tokens interleaved with AA tokens, padded to fixed length.
    
    Args:
        sequence: Protein sequence string
        esm_model: ESM-2 model
        esm_tokenizer: ESM-2 tokenizer
        kmeans: Fitted k-means model with cluster centers
        k_clusters: Number of clusters (e.g., 128 or 512)
        device: 'cuda' or 'cpu'
        token_offset: Offset for structure token IDs (default 20, after AA tokens 0-19)
        max_length: Maximum sequence length for padding (default 1024)
    
    Returns:
        interleaved_tokens: List of integer token IDs padded to max_length with pad token
    """
    # AA vocabulary and special tokens (dynamically calculated based on k_clusters)
    # Vocab structure: 0-19 (AA), 20-(20+k-1) (clusters), (20+k)-(20+k+4) (special tokens)
    aa_vocab = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_id = {aa: i for i, aa in enumerate(aa_vocab)}
    special_tokens_dict = {
        "<pad>": 20 + k_clusters,      # e.g., 148 for k=128, 532 for k=512
        "<unk>": 20 + k_clusters + 1,
        "<bos>": 20 + k_clusters + 2,
        "<eos>": 20 + k_clusters + 3,
        "<sep>": 20 + k_clusters + 4,
    }
    bos_token_id = special_tokens_dict["<bos>"]
    eos_token_id = special_tokens_dict["<eos>"]
    
    # Convert sequence to AA tokens
    sequence = sequence.upper()
    aa_tokens = [aa_to_id.get(aa, special_tokens_dict["<unk>"]) for aa in sequence]
    
    # Extract ESM-2 embeddings
    inputs = esm_tokenizer([sequence], return_tensors="pt", padding=False, truncation=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = esm_model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        # Remove <cls> and <eos> tokens
        embeddings = hidden_states[0, 1:-1, :].cpu().numpy()  # [seq_len, 1280]
    
    # Assign to nearest clusters
    cluster_ids = kmeans.predict(embeddings)
    
    # Add token offset to get proper structure token IDs (20 to 20+k_clusters-1)
    structure_tokens = [token_offset + int(cid) for cid in cluster_ids]
    
    # Interleave AA and structure tokens: [AA, Struct, AA, Struct, ...]
    interleaved_tokens = []
    for aa_tok, struct_tok in zip(aa_tokens, structure_tokens):
        interleaved_tokens.append(aa_tok)
        interleaved_tokens.append(struct_tok)
    
    # Add special tokens (BOS + tokens + EOS)
    final_tokens = [bos_token_id] + interleaved_tokens + [eos_token_id]
    
    # Pad to max_length with pad token
    pad_token_id = special_tokens_dict["<pad>"]
    if len(final_tokens) < max_length:
        final_tokens = final_tokens + [pad_token_id] * (max_length - len(final_tokens))
    else:
        # Truncate if exceeds max_length (keep BOS + content + EOS)
        final_tokens = final_tokens[:max_length]
    
    return np.array(final_tokens, dtype=np.int16)

def main():
    print("=" * 70)
    print("EXTRACT PROTEIN-TEXT EMBEDDING PAIRS")
    print("=" * 70)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract protein-text embedding pairs with structure tokens')
    parser.add_argument('subset_name', type=str, help='Dataset subset name (e.g., UniProt_Function)')
    parser.add_argument('batch_number', type=int, help='Batch number to process (use -1 for all batches)')
    parser.add_argument('--k', type=int, default=128, choices=[128, 512],
                       help='Number of k-means clusters for structure tokens (default: 128)')
    
    args = parser.parse_args()
    
    subset_name = args.subset_name
    batch_to_process = args.batch_number
    k_clusters = args.k
    
    print(f"\n📌 Subset: {subset_name}")
    print(f"📌 Processing batch {batch_to_process}")
    print(f"📌 K-means clusters: {k_clusters}")
    
    # Calculate vocabulary size dynamically
    # Vocab: 0-19 (AA), 20-(20+k-1) (clusters), (20+k)-(20+k+4) (special)
    vocab_size = 20 + k_clusters + 5
    print(f"📌 Total vocabulary size: {vocab_size} (20 AA + {k_clusters} clusters + 5 special tokens)")

    # Configuration
    DATA_DIR = Path('esmfold_tokenizer/data') / subset_name
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
    LLAMA_MODEL_NAME = "meta-llama/Llama-3.1-8B"
    
    input_file = DATA_DIR / 'sample_proteins.json'
    codebook_path = DATA_DIR / f'structure_codebook_K{k_clusters}.pkl'
    output_dir = DATA_DIR / 'embedding_pairs'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        print(f"   Run 01_fetch_sample_data.py first!")
        return
    
    if not codebook_path.exists():
        print(f"⚠️  WARNING: Codebook not found: {codebook_path}")
        print(f"   Structure tokens will NOT be extracted!")
        print(f"   Run 03_train_kmeans_codebook.py first for full functionality.")
        codebook_path = None

    print(f"\n📂 Data directory: {DATA_DIR}")
    print(f"🤖 ESM Model: {ESM_MODEL_NAME}")
    print(f"🤖 Llama Model: {LLAMA_MODEL_NAME}")
    print(f"📊 Codebook: {codebook_path if codebook_path else 'NOT AVAILABLE'}")
    
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
    
    print(f"Loading Llama-3.1 {LLAMA_MODEL_NAME}...")
    llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME)
    llama_model = AutoModel.from_pretrained(LLAMA_MODEL_NAME, device_map="auto", torch_dtype=torch.float16)
    print(f"✅ Llama-3.1 loaded (dim: {llama_model.config.hidden_size})")
    
    # Load k-means codebook if available
    kmeans = None
    n_structure_tokens = 0
    if codebook_path:
        print(f"Loading k-means codebook...")
        try:
            with open(codebook_path, 'rb') as f:
                codebook_data = pickle.load(f)
            kmeans = codebook_data['kmeans']
            n_structure_tokens = codebook_data['n_clusters']
            print(f"✅ Codebook loaded ({n_structure_tokens} clusters)")
        except Exception as e:
            print(f"❌ Failed to load codebook: {e}")
            kmeans = None
    
    # Batch processing
    print("\n" + "=" * 70)
    print("STEP 3: Extracting Embeddings")
    print("=" * 70)
    
    BATCH_SIZE = 10000  # Process 10000 proteins per batch
    num_batches = (len(sequences) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Processing {len(sequences):,} proteins in {num_batches} batch(es) of {BATCH_SIZE}")
    
    # if batch_to_process < 0 or batch_to_process >= num_batches:
    #     print(f"❌ Batch {batch_to_process} out of range [0-{num_batches-1}]")
    #     return

    # Determine which batches to process
    batches_to_process = list(range(num_batches)) if batch_to_process == -1 else [batch_to_process]
    print(f"📋 Will process {len(batches_to_process)} batch(es): {batches_to_process}")
    
    # Set models to eval mode once
    esm_model.eval()
    llama_model.eval()
    
    # Process all batches
    for batch_idx in batches_to_process:
        
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(sequences))
        
        batch_sequences = sequences[start_idx:end_idx]
        batch_texts = texts[start_idx:end_idx]
        batch_ids = protein_ids[start_idx:end_idx]
        
        print(f"\n{'='*70}")
        print(f"Batch {batch_idx + 1}/{num_batches}: Processing proteins {start_idx:,}-{end_idx:,}")
        print(f"{'='*70}")
        
        # Extract embeddings for each protein pair
        print("\n📊 Extracting embedding pairs...")
        embedding_pairs = []
        
        for i, (protein_id, sequence, text) in enumerate(
            tqdm(zip(batch_ids, batch_sequences, batch_texts), total=len(batch_ids), desc="Embedding pairs")
        ):
            # Extract sequence embedding
            seq_emb = extract_esm_embeddings(sequence, esm_model, esm_tokenizer, device)
            
            # Extract text embedding
            text_emb = extract_text_embedding(text, llama_model, llama_tokenizer, device)
            
            # Extract structure tokens (if codebook available)
            struct_tokens = None
            if kmeans is not None:
                struct_tokens = extract_structure_tokens(
                    sequence, esm_model, esm_tokenizer, kmeans, k_clusters, device
                )
            
            # Create pair
            pair = {
                "protein_id": protein_id,
                "global_index": start_idx + i,
                "sequence_embedding": seq_emb,              # Keep as numpy [1280]
                "text_embedding": text_emb,                 # Keep as numpy [4096]
                "structure_tokens": struct_tokens           # Already a list
            }
            embedding_pairs.append(pair)
        
        print(f"✅ Extracted {len(embedding_pairs)} embedding pairs")
        
        # Save embedding pairs as NPZ (compressed binary format)
        output_file = output_dir / f'embedding_pairs_batch_{batch_idx}.npz'
        
        # Stack numpy arrays (already float32, no dtype conversion needed)
        seq_embeddings = np.stack([p['sequence_embedding'] for p in embedding_pairs])
        text_embeddings = np.stack([p['text_embedding'] for p in embedding_pairs])
        protein_ids_list = np.array([p['protein_id'] for p in embedding_pairs], dtype=object)
        structure_tokens_list = np.stack([p['structure_tokens'] for p in embedding_pairs])
        
        np.savez_compressed(
            output_file,
            sequence_embeddings=seq_embeddings,
            text_embeddings=text_embeddings,
            protein_ids=protein_ids_list,
            structure_tokens=structure_tokens_list
        )
        
        print(f"✅ Saved {len(embedding_pairs)} embedding pairs to: {output_file} (NPZ format)")
        
        # Save metadata
        metadata = {
            "batch_number": batch_idx,
            "total_batches": num_batches,
            "num_pairs": len(embedding_pairs),
            "start_index": start_idx,
            "end_index": end_idx,
            "esm_model": ESM_MODEL_NAME,
            "llama_model": LLAMA_MODEL_NAME,
            "esm_embedding_dim": len(embedding_pairs[0]["sequence_embedding"]) if embedding_pairs else 0,
            "text_embedding_dim": len(embedding_pairs[0]["text_embedding"]) if embedding_pairs else 0,
            "n_structure_tokens": n_structure_tokens if kmeans else 0,
            "k_clusters": k_clusters,
            "vocab_size": vocab_size
        }
        
        metadata_file = output_dir / f'metadata_batch_{batch_idx}.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Saved metadata to: {metadata_file}")
        
        # Clear GPU cache between batches to avoid OOM
        if batch_idx < (batches_to_process[-1]):
            print("\n🧹 Clearing GPU memory...")
            del embedding_pairs
            torch.cuda.empty_cache()
    
    print("\n" + "=" * 70)
    print("✅ ALL BATCHES EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Batches processed: {len(batches_to_process)}")
    if len(batches_to_process) == 1:
        print(f"Files saved:")
        print(f"  - embedding_pairs_batch_{batches_to_process[0]}.json")
        print(f"  - metadata_batch_{batches_to_process[0]}.json")
    else:
        print(f"Files saved:")
        print(f"  - embedding_pairs_batch_0.json → embedding_pairs_batch_{batches_to_process[-1]}.json")
        print(f"  - metadata_batch_0.json → metadata_batch_{batches_to_process[-1]}.json")
    
    # Display final statistics from last batch
    if embedding_pairs:
        print(f"\n📊 Final batch statistics:")
        print(f"  ✅ Sequence embedding (ESM-2 mean-pooled): {metadata['esm_embedding_dim']}D")
        print(f"  ✅ Text embedding (Llama-3.1 mean-pooled): {metadata['text_embedding_dim']}D")
        print(f"  ✅ Structure tokens (per-residue): {metadata['n_structure_tokens']} clusters (token IDs 20-{19+k_clusters})")
        print(f"  ✅ Special tokens: pad={20+k_clusters}, unk={20+k_clusters+1}, bos={20+k_clusters+2}, eos={20+k_clusters+3}, sep={20+k_clusters+4}")
        print(f"  ✅ Total vocabulary: {vocab_size} tokens")


if __name__ == '__main__':
    main()
