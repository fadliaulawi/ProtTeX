#!/usr/bin/env python3
"""
Extract Embeddings for Prot2Text Data (same as 03_extract_triplet_embeddings)
Extracts:
1. ESM-2 sequence embeddings (mean-pooled)
2. ESMFold per-residue embeddings -> structure tokens (using existing k-means codebook)
3. Text embeddings (mean-pooled, last hidden state)
"""

import json
import numpy as np
import torch
import pickle
from pathlib import Path
from transformers import AutoTokenizer, EsmModel, AutoModel, EsmForProteinFolding
from tqdm import tqdm
import warnings
import sys
import argparse

warnings.filterwarnings('ignore')

# Add root directory to Python path for imports when running from root
script_dir = Path(__file__).parent
root_dir = script_dir.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Import model configuration
from run.config import get_model_config, list_available_models

ESM_HIDDEN_DIM = 2560  # ESM-2 3B
ESMFOLD_HIDDEN_DIM = 1280  # ESMFold trunk


def extract_esm_embeddings(sequence, model, tokenizer, device='cuda'):
    """Extract sequence embedding using ESM-2. Returns mean-pooled embedding (same as 03)."""
    inputs = tokenizer([sequence], return_tensors="pt", padding=False, truncation=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        embeddings = hidden_states[0, 1:-1, :].cpu().numpy()  # [seq_len, 2560]
        seq_embedding = embeddings.mean(axis=0)
    
    return seq_embedding


def extract_esmfold_embeddings(sequence, model, tokenizer, device='cuda'):
    """Extract per-residue embeddings from ESMFold trunk"""
    sequence = sequence[:1022]  # ESMFold max length
    inputs = tokenizer([sequence], return_tensors="pt", padding=False, truncation=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.esm(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        embeddings = hidden_states[0, 1:-1, :].cpu().numpy()
    
    return embeddings


def extract_text_embedding(text, model, tokenizer, device='cuda'):
    """Extract text embedding using last hidden state and mean pooling (same as 03)."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    inputs = tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_states.shape)
        masked_hidden_states = hidden_states * attention_mask
        
        if masked_hidden_states.dtype == torch.float16:
            masked_hidden_states = masked_hidden_states.float()
        
        sum_hidden_states = masked_hidden_states.sum(dim=1)
        seq_lengths = attention_mask.sum(dim=1)
        pooled = sum_hidden_states / seq_lengths
    
    return pooled.cpu().numpy()[0]


def extract_structure_tokens(sequence, esmfold_embeddings, kmeans, k_clusters, token_offset=20):
    """Extract structure tokens from ESMFold embeddings using k-means codebook"""
    cluster_ids = kmeans.predict(esmfold_embeddings)
    structure_tokens = np.array([token_offset + int(cid) for cid in cluster_ids], dtype=np.int16)
    return structure_tokens


def main():
    parser = argparse.ArgumentParser(description='Extract embeddings for Prot2Text data')
    parser.add_argument('--model', type=str, required=True,
                       choices=list_available_models(),
                       help=f'Model type: {", ".join(list_available_models())}')
    parser.add_argument('--k', type=int, required=True,
                       help='Number of k-means clusters (must match existing codebook)')
    parser.add_argument('--split', type=str, required=True,
                       choices=['train', 'validation', 'test'],
                       help='Which split to process (required)')
    parser.add_argument('--batch-start', type=int, default=0,
                       help='Starting batch number (default: 0)')
    parser.add_argument('--batch-end', type=int, default=None,
                       help='Ending batch number (exclusive, None = all remaining)')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Base data directory (default: <project_root>/data)')
    
    args = parser.parse_args()
    
    model_config = get_model_config(args.model)
    k_clusters = args.k
    
    # Setup directories (follow general run/ convention: data_dir as base, then compose)
    data_dir = Path(args.data_dir) if args.data_dir else (root_dir / 'data')
    habdine_dir = data_dir / 'habdine'
    codebook_path = data_dir / 'codebooks' / f'structure_codebook_K{args.k}.pkl'
    output_dir = habdine_dir / 'triplet_embeddings' / args.model / f'K{args.k}'
    output_dir.mkdir(parents=True, exist_ok=True)
    split_file = habdine_dir / f'standardized_prot2text_{args.split}.json'
    
    print("=" * 70)
    print("EXTRACT EMBEDDINGS FOR Prot2Text DATA")
    print("=" * 70)
    
    # Check input file for specified split
    if not split_file.exists():
        print(f"❌ Input file not found: {split_file}")
        print(f"   Run 01_download_preprocess.py first!")
        return
    
    if not codebook_path.exists():
        print(f"❌ Codebook not found: {codebook_path}")
        print(f"   Run 02_train_kmeans_codebook.py first!")
        return
    
    print(f"\n📌 Data directory: {data_dir}")
    print(f"📂 Input file: {split_file}")
    print(f"📊 Codebook: {codebook_path}")
    print(f"💾 Output: {output_dir}")
    print(f"🤖 Model: {args.model} ({model_config.model_name})")
    print(f"📌 K-means clusters: {k_clusters}")
    print(f"📌 Split: {args.split}")
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Device: {device}")
    if device == 'cpu':
        print("⚠️  WARNING: Running on CPU - will be VERY slow!")
    else:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data from specified split
    print("\n" + "=" * 70)
    print("STEP 1: Loading Data")
    print("=" * 70)
    
    with open(split_file, 'r') as f:
        all_proteins = json.load(f)
    
    total_proteins = len(all_proteins)
    print(f"✅ Loaded {total_proteins} {args.split} proteins")
    
    # Calculate batch ranges based on batch numbers
    batch_size = 1000
    total_batches = (total_proteins + batch_size - 1) // batch_size
    
    if args.batch_end is None:
        args.batch_end = total_batches
    
    # Calculate protein indices from batch numbers
    start_idx = args.batch_start * batch_size
    end_idx = min(args.batch_end * batch_size, total_proteins)
    
    proteins = all_proteins[start_idx:end_idx]
    print(f"📊 Processing batches {args.batch_start} to {args.batch_end-1} (proteins {start_idx} to {end_idx-1}, {len(proteins)} proteins)")
    
    # Load models
    print("\n" + "=" * 70)
    print("STEP 2: Loading Models")
    print("=" * 70)
    
    print("Loading ESM-2 3B for sequence embeddings...")
    esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
    esm_model = EsmModel.from_pretrained("facebook/esm2_t36_3B_UR50D")
    esm_model = esm_model.to(device)
    esm_model.eval()
    print(f"✅ ESM-2 3B loaded")
    
    print("Loading ESMFold for structure embeddings...")
    esmfold_tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    esmfold_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    esmfold_model = esmfold_model.to(device)
    esmfold_model.eval()
    print(f"✅ ESMFold loaded")
    
    print(f"Loading {args.model} for text embeddings...")
    text_tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name,
        trust_remote_code=model_config.trust_remote_code
    )
    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token
    
    text_model = AutoModel.from_pretrained(
        model_config.model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=model_config.trust_remote_code
    )
    text_model.eval()
    print(f"✅ {args.model} loaded")
    
    # Load k-means codebook
    print(f"Loading k-means codebook...")
    with open(codebook_path, 'rb') as f:
        codebook_data = pickle.load(f)
    kmeans = codebook_data['kmeans']
    print(f"✅ Codebook loaded ({k_clusters} clusters)")
    
    # Extract embeddings
    print("\n" + "=" * 70)
    print("STEP 3: Extracting Embeddings")
    print("=" * 70)
    
    print(f"\n📊 Processing and saving {args.split} split ({len(proteins)} proteins)...")
    
    num_batches = (len(proteins) + batch_size - 1) // batch_size
    
    total_processed = 0
    skipped = 0
    
    for local_batch_idx in tqdm(range(num_batches), desc=f"Processing {args.split} batches"):
        # Calculate global batch number
        global_batch_num = args.batch_start + local_batch_idx
        
        start_idx = local_batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(proteins))
        batch_proteins = proteins[start_idx:end_idx]
        
        batch_sequence_embeddings = []
        batch_text_embeddings = []
        batch_struct_tokens = []
        batch_protein_ids = []
        batch_protein_indices = []
        batch_metadata = []
        
        for i, protein in enumerate(batch_proteins):
            sequence = protein['QA']['sequence']
            function_text = protein['QA']['answer']
            protein_id = protein['metadata']['id']
            protein_index = (args.batch_start + local_batch_idx) * batch_size + i
            
            try:
                # Extract sequence embedding (mean-pooled, same as 03)
                seq_emb = extract_esm_embeddings(sequence, esm_model, esm_tokenizer, device)
                
                # Extract ESMFold embeddings for structure tokens
                esmfold_emb = extract_esmfold_embeddings(sequence, esmfold_model, esmfold_tokenizer, device)
                
                # Extract structure tokens using k-means
                struct_tokens = extract_structure_tokens(sequence, esmfold_emb, kmeans, k_clusters)
                
                # Extract text embedding (mean-pooled, same as 03)
                text_emb = extract_text_embedding(function_text, text_model, text_tokenizer, device)
                
                batch_sequence_embeddings.append(seq_emb)
                batch_text_embeddings.append(text_emb)
                batch_struct_tokens.append(struct_tokens)
                batch_protein_ids.append(protein_id)
                batch_protein_indices.append(protein_index)
                batch_metadata.append(protein)
                
            except Exception as e:
                print(f"⚠️  Error processing {protein_id}: {e}")
                skipped += 1
                continue
        
        # Save this batch immediately
        if batch_sequence_embeddings:
            # Pad structure tokens to same length
            pad_token_id = 20 + k_clusters
            max_struct_len = max(len(tokens) for tokens in batch_struct_tokens)
            padded_struct_tokens = []
            for tokens in batch_struct_tokens:
                # DEBUG: Check for negative tokens before padding
                if np.any(tokens < 0):
                    print(f"⚠️  WARNING: Found negative tokens in structure_tokens before padding")
                    print(f"   Min token: {tokens.min()}, Max token: {tokens.max()}")
                
                padded = np.pad(tokens, (0, max_struct_len - len(tokens)), constant_values=pad_token_id)
                padded_struct_tokens.append(padded)
            batch_struct_tokens_array = np.array(padded_struct_tokens)
            
            # DEBUG: Verify no negative tokens after padding
            if np.any(batch_struct_tokens_array < 0):
                print(f"❌ ERROR: Found negative tokens after padding!")
                print(f"   Min token: {batch_struct_tokens_array.min()}, Max token: {batch_struct_tokens_array.max()}")
                print(f"   Pad token ID used: {pad_token_id}")
                raise ValueError(f"Structure tokens contain negative values after padding (pad_token_id={pad_token_id})")
            
            # Save NPZ file with split prefix (same keys as 03_extract_triplet_embeddings)
            npz_file = output_dir / f'triplet_embeddings_{args.split}_batch_{global_batch_num}.npz'
            np.savez_compressed(
                npz_file,
                sequence_embeddings=np.array(batch_sequence_embeddings),
                text_embeddings=np.array(batch_text_embeddings),
                structure_tokens=batch_struct_tokens_array,
                protein_ids=np.array(batch_protein_ids, dtype=object),
                protein_indices=np.array(batch_protein_indices, dtype=np.int32)
            )
            
            # Save metadata
            metadata_file = output_dir / f'triplet_metadata_{args.split}_batch_{global_batch_num}.json'
            with open(metadata_file, 'w') as f:
                json.dump(batch_metadata, f, indent=2)
            
            total_processed += len(batch_sequence_embeddings)
            print(f"✅ Saved {args.split} batch {global_batch_num}: {len(batch_sequence_embeddings)} proteins")
    
    print(f"\n✅ All embeddings saved to {output_dir}")
    print(f"   {args.split.capitalize()}: {total_processed} proteins processed")
    if skipped > 0:
        print(f"   ⚠️  Skipped {skipped} proteins")


if __name__ == "__main__":
    main()
