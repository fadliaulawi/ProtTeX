#!/usr/bin/env python3
"""
Fetch Sample Data from ProteinLMBench
Downloads protein samples for tokenizer testing
"""

import json
import os
import sys
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def extract_samples_from_dataset(dataset, subset_name, num_samples=1000000):
    """
    Extract and process samples from a dataset.
    
    Args:
        dataset: HuggingFace dataset object
        subset_name: Name of the subset (for cleaning text)
        num_samples: Maximum number of samples to extract
    
    Returns:
        List of processed samples
    """
    # Extract samples
    print("\n" + "=" * 70)
    print("STEP 2: Extracting Samples")
    print("=" * 70)
    
    # First, inspect the dataset structure
    print("\n🔍 Inspecting dataset structure...")
    first_item = dataset[0]
    print(f"Available keys: {list(first_item.keys())}")
    print(f"\nFirst item sample:")
    for key, value in first_item.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: {value[:100]}...")
        else:
            print(f"  {key}: {value}")
    print()
    
    samples = []
    valid_count = 0
    
    for i, item in tqdm(enumerate(dataset), total=len(dataset), desc="Extracting samples"):
        if valid_count >= num_samples:
            break
        
        # Extract sequence (handle different possible field names)
        sequence = None
        for key in ['Sequence', 'sequence', 'protein_sequence', 'seq', 'input']:
            if key in item and item[key]:
                seq_candidate = item[key]
                # If it's the input field, try to extract sequence from it
                if key == 'input' and 'Sequence:' in str(seq_candidate):
                    try:
                        sequence = str(seq_candidate).split('Sequence:')[1].strip()
                    except:
                        pass
                else:
                    sequence = seq_candidate
                if sequence:
                    # Sanitize the sequence
                    import re
                    # Handle sequences wrapped in <seq> tags (may contain spaces)
                    match = re.search(r'<seq>(.*?)</seq>', str(sequence), re.DOTALL)
                    if match:
                        sequence = match.group(1)
                    break
        
        # Extract text/description
        text = None
        for key in ['output', 'text', 'function', 'description', 'annotation', 'response']:
            if key in item and item[key]:
                text = item[key]
                if subset_name == 'Enzyme_CoT':
                    text = text.replace("Based on the analysis of the provided protein sequence, the function of this protein is described as follows: ", "")
                elif subset_name == 'UniProt_Induction':
                    text = text.replace("The known inducer(s) or condition(s) leading to the expression of this protein include(s): ", "")
                elif subset_name == 'UniProt_Function':
                    text = text.replace("The determined function(s) of the protein include(s): ", "")
                elif subset_name == 'UniProt_Involvement in disease':
                    text = text.replace("This protein is associated with the following disease(s): ", "")
                elif subset_name == 'UniProt_Post-translational modification':
                    text = text.replace("The post-translational modification(s) identified in this protein include(s): ", "")
                elif subset_name == 'UniProt_Subunit structure':
                    text = text.replace("Based on the analysis, the protein's subunit structure is characterized by:", "")
                elif subset_name == 'UniProt_Tissue specificity':
                    text = text.replace("The tissue specificity of this protein is characterized by: ", "")
                break
        
        # Skip if missing required fields
        if not sequence or not text:
            continue
        
        # Clean sequence (remove any newlines or spaces)
        sequence = ''.join(sequence.split())
        
        # Filter reasonable lengths
        if len(sequence) < 50 or len(sequence) > 1000:
            continue
        
        samples.append({
            'id': f'{subset_name}_protein_{valid_count:04d}',
            'sequence': sequence,
            'text': text,
            'length': len(sequence),
            'question': item.get('instruction', 'What is the function of this protein?'),
            'subset': subset_name
        })
        
        valid_count += 1        
    
    print(f"\n✅ Extracted {len(samples)} samples")
    
    # Print 3 first and last samples for verification
    print("\nSample previews:")
    for sample in samples[:3] + samples[-3:]:
        print(f"ID: {sample['id']}, Length: {sample['length']}, Sequence: {sample['sequence'][:30]}..., Text: {sample['text'][:50]}...")
    
    seqs = [s['sequence'] for s in samples]
    # check if duplicate sequences exist
    if len(seqs) != len(set(seqs)):
        print("Number of sequences:", len(seqs))
        print("Number of unique sequences:", len(set(seqs)))
        print("⚠️  Warning: Duplicate sequences found in extracted samples!")

    # Combine duplicate sequences (merge their text descriptions)
    seen_seqs = {}
    for sample in samples:
        seq = sample['sequence']
        if seq not in seen_seqs:
            seen_seqs[seq] = {
                'id': sample['id'],
                'sequence': seq,
                'text': [sample['text']],
                'length': sample['length'],
                'question': sample['question'],
                'subset': sample.get('subset', subset_name)
            }
        else:
            seen_seqs[seq]['text'].append(sample['text'])
    
    # Reconstruct samples with combined text
    samples = []
    for seq, data in seen_seqs.items():
        samples.append({
            'id': data['id'],
            'sequence': data['sequence'],
            'text': ' | '.join(data['text']),  # Combine texts with separator
            'length': data['length'],
            'question': data['question'],
            'subset': data['subset']
        })
    
    print(f"✅ After combining duplicates: {len(samples)} unique samples")
    
    return samples

def main():
    print("=" * 70)
    print("FETCH SAMPLE DATA FROM PROTEINLMBENCH")
    print("=" * 70)
    
    # Parse arguments
    if len(sys.argv) < 2:
        subset_name = "UniProt_Function"
        print(f"⚠️  No subset specified, using default: {subset_name}")
        print(f"   Usage: python 01_fetch_sample_data.py <subset_name>")
        print(f"   Use '-1' to download and combine all available subsets")
    else:
        subset_name = sys.argv[1]
    
    # Check if user wants all subsets
    if subset_name == "-1":
        print(f"\n📌 Mode: Download and combine ALL subsets")
        
        # List of available subsets
        available_subsets = [
            "Enzyme_CoT",
            "UniProt_Function",
            "UniProt_Induction",
            "UniProt_Involvement in disease",
            "UniProt_Post-translational modification",
            "UniProt_Subunit structure",
            "UniProt_Tissue specificity"
        ]
        
        print(f"   Available subsets: {', '.join(available_subsets)}")
        
        # Output directory for combined data
        OUTPUT_DIR = Path('esmfold_tokenizer/data') / 'ALL'
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        print(f"📂 Output directory: {OUTPUT_DIR}")
        
        # Collect all samples from all subsets
        all_samples = []
        
        for subset in available_subsets:
            print(f"\n{'='*70}")
            print(f"Processing subset: {subset}")
            print(f"{'='*70}")
            
            try:
                dataset = load_dataset(
                    "tsynbio/ProteinLMBench",
                    subset,
                    split="train",
                    streaming=False
                )
                
                print(f"✅ Loaded {subset}: {len(dataset)} samples")
                
                # Extract samples from this subset
                subset_samples = extract_samples_from_dataset(dataset, subset, num_samples=1000000)
                all_samples.extend(subset_samples)
                
                print(f"✅ Extracted {len(subset_samples)} samples from {subset}")
                
            except Exception as e:
                print(f"⚠️  Error loading {subset}: {e}")
                continue
        
        samples = all_samples
        
        print(f"\n{'='*70}")
        print(f"Combined {len(samples)} samples from all subsets")
        print(f"{'='*70}")
        
        # Deduplicate across all subsets
        print("\n" + "=" * 70)
        print("CROSS-SUBSET DEDUPLICATION")
        print("=" * 70)
        
        seqs = [s['sequence'] for s in samples]
        print(f"Total sequences before deduplication: {len(seqs)}")
        print(f"Unique sequences: {len(set(seqs))}")
        
        if len(seqs) != len(set(seqs)):
            print("⚠️  Duplicate sequences found across subsets!")
            
            # Combine duplicates (merge texts and track source subsets)
            seen_seqs = {}
            for idx, sample in tqdm(enumerate(samples), total=len(samples), desc="Deduplicating"):
                seq = sample['sequence']
                if seq not in seen_seqs:
                    seen_seqs[seq] = {
                        'id': sample['id'],
                        'sequence': seq,
                        'text': [sample['text']],
                        'length': sample['length'],
                        'question': sample['question'],
                        'subsets': {sample['subset']}  # Use set for O(1) lookup
                    }
                else:
                    seen_seqs[seq]['text'].append(sample['text'])
                    seen_seqs[seq]['subsets'].add(sample['subset'])  # O(1) add
            
            print(f"   Reconstructing deduplicated samples...")
            
            # Reconstruct samples
            samples = []
            for seq, data in seen_seqs.items():
                combined_text = ' | '.join(data['text'])                
                samples.append({
                    'id': data['id'],
                    'sequence': data['sequence'],
                    'text': combined_text,
                    'length': data['length'],
                    'question': data['question'],
                    'subset': ', '.join(sorted(data['subsets']))  # Combined subset names
                })
            
            print(f"✅ After cross-subset deduplication: {len(samples)} unique samples")
        else:
            print("✅ No duplicates found across subsets")
        
    else:
        print(f"\n📌 Subset: {subset_name}")
        
        # Configuration
        OUTPUT_DIR = Path('esmfold_tokenizer/data') / subset_name
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        print(f"📂 Output directory: {OUTPUT_DIR}")
        print(f"🎯 Fetching maximum samples")
        
        # Load ProteinLMBench
        print("\n" + "=" * 70)
        print("STEP 1: Loading ProteinLMBench from HuggingFace")
        print("=" * 70)
        print("\nℹ️  This may take a few minutes on first run (downloads dataset)...")
        
        try:
            # Load specified subset
            dataset = load_dataset(
                "tsynbio/ProteinLMBench",
                subset_name,
                split="train",
                streaming=False
            )
            
            print(f"✅ Loaded ProteinLMBench/{subset_name}")
            print(f"   Total samples available: {len(dataset)}")
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            print("\nTrying alternative approach...")
            
            # Alternative: load without specifying subset
            dataset = load_dataset(
                "tsynbio/ProteinLMBench",
                split="train",
                streaming=False
            )
            print(f"✅ Loaded ProteinLMBench (full dataset)")
        
        # Extract samples
        samples = extract_samples_from_dataset(dataset, subset_name, num_samples=1000000)
    
    # Check if we got any samples
    if len(samples) == 0:
        print("\n❌ ERROR: No samples extracted!")
        print("   The dataset field names may not match expected format.")
        return
    
    # Shuffle samples
    print("\n" + "=" * 70)
    print("SHUFFLING SAMPLES")
    print("=" * 70)
    print(f"Shuffling {len(samples)} samples...")
    np.random.shuffle(samples)
    print(f"✅ Samples shuffled")
    
    # Statistics
    print("\n" + "=" * 70)
    print("STEP 3: Dataset Statistics")
    print("=" * 70)
    
    lengths = [s['length'] for s in samples]
    print(f"\nSequence lengths:")
    print(f"   Min: {min(lengths)} residues")
    print(f"   Max: {max(lengths)} residues")
    print(f"   Mean: {sum(lengths)/len(lengths):.0f} residues")
    print(f"   Total: {sum(lengths)} residues")
    
    # Save samples
    print("\n" + "=" * 70)
    print("STEP 4: Saving Samples")
    print("=" * 70)
    
    # Save as JSON
    # output_file = OUTPUT_DIR / 'sample_proteins.json'
    # with open(output_file, 'w') as f:
    #     json.dump(samples, f, indent=2)
    
    # print(f"✅ Saved samples: {output_file}")
    # print(f"   Size: {output_file.stat().st_size / 1024:.1f} KB")
    
    # Save per 10k proteins in batches
    batch_dir = OUTPUT_DIR / 'sample_proteins'
    batch_dir.mkdir(exist_ok=True)
    batch_size = 10000
    num_batches = (len(samples) + batch_size - 1) // batch_size
    
    print(f"\n📦 Saving batches of {batch_size} proteins...")
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(samples))
        batch_samples = samples[start_idx:end_idx]
        
        batch_file = batch_dir / f'sample_proteins_batch_{batch_idx}.json'
        with open(batch_file, 'w') as f:
            json.dump(batch_samples, f, indent=2)
        
        print(f"   ✅ Batch {batch_idx}: {len(batch_samples)} samples → {batch_file.name}")
    
    print(f"✅ Saved {num_batches} batch files in: {batch_dir}")
    
    # Save sequences only (for quick access)
    sequences_file = OUTPUT_DIR / 'sequences.txt'
    with open(sequences_file, 'w') as f:
        for sample in samples:
            f.write(f">{sample['id']}\n{sample['sequence']}\n")
    
    print(f"✅ Saved sequences: {sequences_file}")
    
    # Save metadata
    metadata_file = OUTPUT_DIR / 'metadata.txt'
    with open(metadata_file, 'w') as f:
        f.write(f"ProteinLMBench Sample Dataset\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Source: ProteinLMBench (tsynbio/ProteinLMBench)\n")
        if subset_name == "-1":
            f.write(f"Mode: Combined all subsets\n")
        else:
            f.write(f"Subset: {subset_name}\n")
        f.write(f"Num samples: {len(samples)}\n")
        f.write(f"Total residues: {sum(lengths)}\n")
        f.write(f"Avg length: {sum(lengths)/len(lengths):.0f} residues\n")
        f.write(f"Length range: {min(lengths)}-{max(lengths)} residues\n\n")
        f.write(f"Files:\n")
        f.write(f"  sample_proteins.json - Full data (sequence + text)\n")
        f.write(f"  sequences.txt - FASTA format sequences\n")
        f.write(f"  metadata.txt - This file\n")
    
    print(f"✅ Saved metadata: {metadata_file}")
    
    # Show examples
    print("\n" + "=" * 70)
    print("STEP 5: Sample Preview")
    print("=" * 70)
    
    for i, sample in enumerate(samples[:3]):
        print(f"\n--- Sample {i+1} ---")
        print(f"ID: {sample['id']}")
        print(f"Length: {sample['length']} residues")
        print(f"Sequence: {sample['sequence'][:50]}...")
        print(f"Text: {sample['text'][:100]}...")
    
    print("\n" + "=" * 70)
    print("✅ DATA FETCH COMPLETE!")
    print("=" * 70)
    
    print(f"\n📊 Summary:")
    print(f"   Samples: {len(samples)}")
    print(f"   Total residues: {sum(lengths):,}")
    if subset_name == "-1":
        print(f"   Mode: Combined all subsets")
    else:
        print(f"   Subset: {subset_name}")
    print(f"   Output: {OUTPUT_DIR}")
    
    print(f"\n🚀 Next step:")
    if subset_name == "-1":
        print(f"   python 02_extract_esm_embeddings.py ALL")
    else:
        print(f"   python 02_extract_esm_embeddings.py {subset_name}")


if __name__ == '__main__':
    main()

