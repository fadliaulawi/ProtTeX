#!/usr/bin/env python3
"""
Extract Protein-Text Embedding Triplets with Structure Tokens (Batch Mode)
Extracts sequence embeddings (ESM-2 3B), structure tokens (from pre-computed ESMFold embeddings + k-means), 
and text embeddings (Llama) from paired proteins.
Processes in batches and reuses pre-computed ESMFold embeddings for speed.

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

# Add root directory to Python path for imports when running from root
script_dir = Path(__file__).parent
root_dir = script_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Import model configuration
from run.config import get_model_config, list_available_models


def generate_protein_metadata(protein, protein_id):
    """
    Generate metadata dict for a single protein.
    
    Args:
        protein: Protein dict from standardized_protein_instructions.json
        protein_id: Protein identifier
    
    Returns:
        Metadata dict with id, sequence, question, function, length, type, subset
    """
    return {
        'id': protein_id,
        'sequence': protein['sequence'],
        'question': protein.get('question', 'What is the function of this protein?'),
        'function': protein['text'],  # This is the answer/target
        'length': protein.get('length', len(protein['sequence'])),
        'type': protein.get('type', 'PFUD'),
        'subset': protein.get('subset', 'unknown')
    }


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
    Extract text embedding using language model for a single text.
    
    Args:
        text: Text description string
        model: Language model (Llama, Qwen, DeepSeek, etc.)
        tokenizer: Model tokenizer
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
        max_length=1024
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
        
        # Convert to float32 before summing to prevent float16 overflow
        if masked_hidden_states.dtype == torch.float16:
            masked_hidden_states = masked_hidden_states.float()  # Convert to float32
        
        sum_hidden_states = masked_hidden_states.sum(dim=1)
        seq_lengths = attention_mask.sum(dim=1)
        pooled = sum_hidden_states / seq_lengths
    
    return pooled.cpu().numpy()[0]

def extract_structure_tokens_from_embeddings(sequence, precomputed_embeddings, kmeans, k_clusters, token_offset=20, max_length=1024):
    """
    Extract per-residue structure tokens from pre-computed embeddings.
    Interleaves AA tokens with structure tokens, padded to fixed length.
    
    Args:
        sequence: Protein sequence string
        precomputed_embeddings: Pre-computed ESMFold embeddings [seq_len, 2560]
        kmeans: Fitted k-means model with cluster centers
        k_clusters: Number of clusters (e.g., 128, 512, 2048)
        token_offset: Offset for structure token IDs (default 20, after AA tokens 0-19)
        max_length: Maximum sequence length for padding (default 1024)
    
    Returns:
        interleaved_tokens: Array of integer token IDs padded to max_length
    """
    # AA vocabulary and special tokens
    aa_vocab = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_id = {aa: i for i, aa in enumerate(aa_vocab)}
    special_tokens_dict = {
        "<pad>": 20 + k_clusters,
        "<unk>": 20 + k_clusters + 1,
        "<bos>": 20 + k_clusters + 2,
        "<eos>": 20 + k_clusters + 3,
        "<sep>": 20 + k_clusters + 4,
    }
    bos_token_id = special_tokens_dict["<bos>"]
    eos_token_id = special_tokens_dict["<eos>"]
    pad_token_id = special_tokens_dict["<pad>"]
    
    # Convert sequence to AA tokens
    sequence = sequence.upper()
    aa_tokens = [aa_to_id.get(aa, special_tokens_dict["<unk>"]) for aa in sequence]
    
    # Predict clusters from pre-computed embeddings
    cluster_ids = kmeans.predict(precomputed_embeddings)
    
    # Add token offset to get proper structure token IDs (20 to 20+k_clusters-1)
    structure_tokens = [token_offset + int(cid) for cid in cluster_ids]
    
    # Interleave AA and structure tokens: [AA, Struct, AA, Struct, ...]
    interleaved_tokens = []
    for aa_tok, struct_tok in zip(aa_tokens, structure_tokens):
        interleaved_tokens.append(aa_tok)
        interleaved_tokens.append(struct_tok)
    
    # Add special tokens (BOS + tokens + EOS)
    final_tokens = [bos_token_id] + interleaved_tokens + [eos_token_id]
    
    # Pad to max_length
    if len(final_tokens) < max_length:
        final_tokens = final_tokens + [pad_token_id] * (max_length - len(final_tokens))
    else:
        # Truncate if exceeds max_length
        final_tokens = final_tokens[:max_length]
    
    return np.array(final_tokens, dtype=np.int16)

def main():
    print("=" * 70)
    print("EXTRACT PROTEIN-TEXT EMBEDDING TRIPLETS (BATCH MODE)")
    print("=" * 70)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Extract protein-text embedding triplets with structure tokens'
    )
    parser.add_argument('--model', type=str, required=True,
                       choices=list_available_models(),
                       help=f'Model type: {", ".join(list_available_models())}')
    parser.add_argument('--k', type=int, default=128,
                       help='Number of k-means clusters for structure tokens (default: 128)')
    parser.add_argument('--batch-start', type=int, default=0,
                       help='Starting batch number (default: 0)')
    parser.add_argument('--batch-end', type=int, default=72,
                       help='Ending batch number (exclusive, default: 72)')
    
    args = parser.parse_args()
    
    # Get model configuration
    model_config = get_model_config(args.model)
    k_clusters = args.k
    batch_start = args.batch_start
    batch_end = args.batch_end
    
    data_dir = Path('data')
    embeddings_dir = data_dir / 'esm_embeddings'
    
    # Check for existing triplet embeddings to reuse (no race condition check)
    triplet_base_dir = data_dir / 'triplet_embeddings' / args.model
    reuse_source_k = None
    
    if triplet_base_dir.exists():
        # Find all existing K directories
        existing_k_dirs = [d for d in triplet_base_dir.iterdir() if d.is_dir() and d.name.startswith('K')]
        if existing_k_dirs:
            # Extract K values
            existing_ks = []
            for d in existing_k_dirs:
                try:
                    k_val = int(d.name[1:])  # Remove 'K' prefix
                    if k_val != k_clusters:  # Skip target K
                        existing_ks.append(k_val)
                except ValueError:
                    continue
            
            if existing_ks:
                # Find the largest K (excluding target)
                max_k = max(existing_ks)
                print(f"\n🔍 Found existing triplet embeddings for model '{args.model}':")
                print(f"   Available K values: {sorted(existing_ks)}")
                print(f"   Will reuse from K={max_k}")
                reuse_source_k = max_k
    
    # Use model-specific output directory
    output_dir = data_dir / 'triplet_embeddings' / args.model / f'K{k_clusters}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📌 Model: {args.model} ({model_config.model_name})")
    print(f"📌 K-means clusters: {k_clusters}")
    print(f"📌 Batch range: {batch_start} to {batch_end-1} ({batch_end - batch_start} batches)")
    print(f"📌 Data directory: {data_dir}")
    print(f"📌 Embeddings directory: {embeddings_dir}")
    print(f"📌 Output directory: {output_dir}")
    if reuse_source_k:
        print(f"📌 Reusing seq/text embeddings from K={reuse_source_k} (faster!)")
    
    # Calculate vocabulary size dynamically
    vocab_size = 20 + k_clusters + 5
    print(f"📌 Total vocabulary size: {vocab_size} (20 AA + {k_clusters} clusters + 5 special tokens)")

    # Configuration
    ESM_SEQ_MODEL_NAME = "facebook/esm2_t36_3B_UR50D"  # 3B model for sequence embeddings
    TEXT_MODEL_NAME = model_config.model_name
    
    input_file = data_dir / 'datasets' / 'standardized_protein_instructions.json'
    codebook_path = data_dir / 'codebooks' / f'structure_codebook_K{k_clusters}.pkl'
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        print(f"   Run 00_standardize_datasets.py first!")
        return
    
    if not codebook_path.exists():
        print(f"❌ Codebook not found: {codebook_path}")
        print(f"   Run 02_train_kmeans_codebook.py first!")
        return
    
    if not embeddings_dir.exists():
        print(f"❌ Embeddings directory not found: {embeddings_dir}")
        print(f"   Run 01_extract_esm_embeddings.py first!")
        return

    print(f"\n📂 Input file: {input_file}")
    print(f"🤖 ESM-2 3B (sequence): {ESM_SEQ_MODEL_NAME}")
    print(f"🤖 Pre-computed ESMFold embeddings (structure): {embeddings_dir}")
    print(f"🤖 Text Model: {TEXT_MODEL_NAME} (hidden_dim: {model_config.hidden_dim})")
    print(f"📊 Codebook: {codebook_path}")
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Device: {device}")
    
    if device == 'cpu':
        print("⚠️  WARNING: Running on CPU - will be VERY slow!")
    else:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load full dataset to get sequences and texts
    print("\n" + "=" * 70)
    print("STEP 1: Loading Data")
    print("=" * 70)
    
    with open(input_file, 'r') as f:
        proteins = json.load(f)
    
    print(f"✅ Loaded {len(proteins)} proteins from dataset")
    
    # Load Models (skip ESM-2 and text model if reusing)
    print("\n" + "=" * 70)
    print("STEP 2: Loading Models")
    print("=" * 70)
    
    if reuse_source_k:
        print(f"⚡ Skipping ESM-2 and text model (reusing embeddings from K={reuse_source_k})")
        esm_model = None
        esm_tokenizer = None
        text_model = None
        text_tokenizer = None
    else:
        print(f"Loading ESM-2 3B for sequence embeddings: {ESM_SEQ_MODEL_NAME}...")
        esm_tokenizer = AutoTokenizer.from_pretrained(ESM_SEQ_MODEL_NAME)
        esm_model = EsmModel.from_pretrained(ESM_SEQ_MODEL_NAME)
        esm_model = esm_model.to(device)
        esm_model.eval()
        print(f"✅ ESM-2 3B loaded (dim: {esm_model.config.hidden_size})")
        
        print(f"Loading {args.model} model {TEXT_MODEL_NAME}...")
        text_tokenizer = AutoTokenizer.from_pretrained(
            TEXT_MODEL_NAME,
            trust_remote_code=model_config.trust_remote_code
        )
        if text_tokenizer.pad_token is None:
            text_tokenizer.pad_token = text_tokenizer.eos_token
        text_model = AutoModel.from_pretrained(
            TEXT_MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=model_config.trust_remote_code
        )
        text_model.eval()
        print(f"✅ {args.model} loaded (dim: {text_model.config.hidden_size})")
    
    # Load k-means codebook
    print(f"Loading k-means codebook from {codebook_path}...")
    with open(codebook_path, 'rb') as f:
        codebook_data = pickle.load(f)
    kmeans = codebook_data['kmeans']
    n_structure_tokens = codebook_data['n_clusters']
    print(f"✅ Codebook loaded ({n_structure_tokens} clusters)")
    
    # Extract embeddings batch by batch
    print("\n" + "=" * 70)
    print("STEP 3: Extracting Embedding Triplets (Batch Mode)")
    print("=" * 70)
    
    # If reusing, set up source directory
    if reuse_source_k:
        source_dir = data_dir / 'triplet_embeddings' / args.model / f'K{reuse_source_k}'
        print(f"⚡ Source directory for reuse: {source_dir}")
    
    total_triplets = 0
    total_skipped = 0
    
    for batch_num in tqdm(range(batch_start, batch_end), desc="Processing batches"):
        # REUSE MODE: Load from existing K and only regenerate structure tokens
        if reuse_source_k:
            source_npz_file = source_dir / f'triplet_embeddings_batch_{batch_num}.npz'
            source_json_file = source_dir / f'triplet_metadata_batch_{batch_num}.json'
            
            if not source_npz_file.exists() or not source_json_file.exists():
                continue
            
            # Load existing embeddings and metadata
            source_data = np.load(source_npz_file, allow_pickle=True)
            with open(source_json_file, 'r') as f:
                source_metadata = json.load(f)
            
            # Reuse sequence and text embeddings
            sequence_embeddings = source_data['sequence_embeddings']
            text_embeddings = source_data['text_embeddings']
            protein_ids = source_data['protein_ids']
            protein_indices = source_data['protein_indices']
            
            # Get sequences from metadata
            sequences = [item['QA']['sequence'] for item in source_metadata]
            
            # Load ESMFold embeddings to regenerate structure tokens
            embeddings_file = embeddings_dir / f'esm_embeddings_batch_{batch_num}.npy'
            if not embeddings_file.exists():
                continue
            
            all_embeddings = np.load(embeddings_file)
            
            # Regenerate structure tokens with new K
            new_structure_tokens = []
            emb_offset = 0
            
            for seq in sequences:
                seq_length = len(seq)
                protein_embeddings = all_embeddings[emb_offset:emb_offset + seq_length]
                emb_offset += seq_length
                
                struct_tokens = extract_structure_tokens_from_embeddings(
                    seq, protein_embeddings, kmeans, k_clusters
                )
                new_structure_tokens.append(struct_tokens)
            
            structure_tokens_array = np.stack(new_structure_tokens)
            
            # Save with new K
            output_file = output_dir / f'triplet_embeddings_batch_{batch_num}.npz'
            np.savez_compressed(
                output_file,
                sequence_embeddings=sequence_embeddings,
                text_embeddings=text_embeddings,
                protein_ids=protein_ids,
                protein_indices=protein_indices,
                structure_tokens=structure_tokens_array
            )
            
            # Copy metadata
            output_json_file = output_dir / f'triplet_metadata_batch_{batch_num}.json'
            with open(output_json_file, 'w') as f:
                json.dump(source_metadata, f, indent=2)
            
            total_triplets += len(sequences)
            continue
        
        # FULL EXTRACTION MODE: Extract all embeddings from scratch
        print(f"\n{'='*70}")
        print(f"Processing Batch {batch_num}")
        print(f"{'='*70}")
        
        # Load pre-computed ESMFold embeddings for this batch
        embeddings_file = embeddings_dir / f'esm_embeddings_batch_{batch_num}.npy'
        metadata_file = embeddings_dir / f'embedding_metadata_batch_{batch_num}.json'
        
        if not embeddings_file.exists():
            print(f"⚠️  Embeddings file not found: {embeddings_file}")
            print(f"   Skipping batch {batch_num}")
            continue
        
        if not metadata_file.exists():
            print(f"⚠️  Metadata file not found: {metadata_file}")
            print(f"   Skipping batch {batch_num}")
            continue
        
        # Load metadata to get protein indices and sequences
        with open(metadata_file, 'r') as f:
            batch_metadata = json.load(f)
        
        print(f"   Loaded metadata: {len(batch_metadata)} proteins")
        
        # Load pre-computed embeddings
        print(f"   Loading pre-computed embeddings from {embeddings_file.name}...")
        all_embeddings = np.load(embeddings_file)
        print(f"   Embeddings shape: {all_embeddings.shape}")
        
        # Process each protein in this batch
        batch_triplets = []
        batch_skipped = 0
        
        # Track current position in embeddings array
        emb_offset = 0
        
        for meta in tqdm(batch_metadata, desc=f"Batch {batch_num}"):
            protein_idx = meta['protein_idx']
            seq_length = meta['length']
            
            # Get sequence and text from original dataset
            if protein_idx >= len(proteins):
                print(f"⚠️  Protein index {protein_idx} out of range")
                batch_skipped += 1
                emb_offset += seq_length
                continue
            
            protein = proteins[protein_idx]
            sequence = protein['QA']['sequence']
            answer = protein['QA']['answer']
            question = protein['QA'].get('question', '')
            
            # Skip if answer is empty
            if not answer or not answer.strip():
                batch_skipped += 1
                emb_offset += seq_length
                continue
            
            # Get pre-computed embeddings for this protein
            protein_embeddings = all_embeddings[emb_offset:emb_offset + seq_length]
            emb_offset += seq_length
            
            # Extract sequence embedding (mean-pooled, ESM-2 3B)
            seq_emb = extract_esm_embeddings(sequence, esm_model, esm_tokenizer, device)
            
            # Extract text embedding (mean-pooled)
            text_emb = extract_text_embedding(answer, text_model, text_tokenizer, device)
            
            # Extract structure tokens from pre-computed embeddings
            struct_tokens = extract_structure_tokens_from_embeddings(
                sequence, protein_embeddings, kmeans, k_clusters
            )
            
            # Create triplet preserving QA and metadata structure
            triplet = {
                "QA": {
                    "question": question,
                    "answer": answer,
                    "sequence": sequence
                },
                "metadata": {
                    "id": protein['metadata'].get('id', f'protein_{protein_idx}'),
                    "length": protein['metadata'].get('length', len(sequence)),
                    "subset": protein['metadata'].get('subset', ''),
                    "type": protein['metadata'].get('type', ''),
                    "pdb_id": protein['metadata'].get('pdb_id', ''),
                    "uniprot_id": protein['metadata'].get('uniprot_id', ''),
                    "accession_id": protein['metadata'].get('accession_id', '')
                },
                "embeddings": {
                    "sequence_embedding": seq_emb,
                    "text_embedding": text_emb,
                    "structure_tokens": struct_tokens
                },
                "batch_num": batch_num,
                "protein_index": protein_idx
            }
            batch_triplets.append(triplet)
        
        print(f"   ✅ Extracted {len(batch_triplets)} triplets from batch {batch_num}")
        if batch_skipped > 0:
            print(f"   ⚠️  Skipped {batch_skipped} proteins (empty text or errors)")
        
        total_triplets += len(batch_triplets)
        total_skipped += batch_skipped
        
        # Save batch triplets
        if len(batch_triplets) > 0:
            output_file = output_dir / f'triplet_embeddings_batch_{batch_num}.npz'
            
            # Stack numpy arrays from embeddings section
            seq_embeddings = np.stack([t['embeddings']['sequence_embedding'] for t in batch_triplets])
            text_embeddings = np.stack([t['embeddings']['text_embedding'] for t in batch_triplets])
            structure_tokens_array = np.stack([t['embeddings']['structure_tokens'] for t in batch_triplets])
            protein_ids_array = np.array([t['metadata']['id'] for t in batch_triplets], dtype=object)
            protein_indices = np.array([t['protein_index'] for t in batch_triplets], dtype=np.int32)
            
            np.savez_compressed(
                output_file,
                sequence_embeddings=seq_embeddings,
                text_embeddings=text_embeddings,
                protein_ids=protein_ids_array,
                protein_indices=protein_indices,
                structure_tokens=structure_tokens_array
            )
            
            print(f"   💾 Saved to: {output_file.name}")
        
            # Save batch metadata preserving QA and metadata structure
            batch_meta_file = output_dir / f'triplet_metadata_batch_{batch_num}.json'
            
            # Build metadata list preserving QA and metadata structure
            protein_metadata = []
            for triplet in batch_triplets:
                # Preserve QA and metadata structure, add batch info
                metadata_entry = {
                    "QA": triplet['QA'],
                    "metadata": triplet['metadata'],
                    "batch_num": triplet['batch_num'],
                    "protein_index": triplet['protein_index']
                }
                protein_metadata.append(metadata_entry)
            
            # Save as JSON (list of metadata dicts)
            with open(batch_meta_file, 'w') as f:
                json.dump(protein_metadata, f, indent=2)
            
            print(f"   📄 Saved metadata: {batch_meta_file.name} ({len(protein_metadata)} proteins)")
        
        # Clean up
        del all_embeddings
        del batch_triplets
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 70)
    print("✅ EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Processed batches: {batch_start} to {batch_end-1}")
    print(f"Total triplets extracted: {total_triplets:,}")
    print(f"Total skipped: {total_skipped:,}")
    print(f"Output directory: {output_dir}")
    
    print(f"\n📊 Summary:")
    print(f"  ✅ Sequence embedding (ESM-2 3B): 2560D")
    print(f"  ✅ Text embedding (Llama): 4096D")
    print(f"  ✅ Structure tokens: {k_clusters} clusters")
    print(f"  ✅ Total vocabulary: {vocab_size} tokens")


if __name__ == '__main__':
    main()
