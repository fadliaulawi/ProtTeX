#!/usr/bin/env python3
"""
Fetch Sample Data from ProteinLMBench
Downloads 100 protein samples for tokenizer testing
"""

import json
import os
from pathlib import Path
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 70)
    print("FETCH SAMPLE DATA FROM PROTEINLMBENCH")
    print("=" * 70)
    
    # Configuration
    OUTPUT_DIR = Path('esmfold_tokenizer/data')
    NUM_SAMPLES = 1000000 # MAXIMUM
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Output directory: {OUTPUT_DIR}")
    print(f"🎯 Fetching {NUM_SAMPLES} samples")
    
    # Load ProteinLMBench
    print("\n" + "=" * 70)
    print("STEP 1: Loading ProteinLMBench from HuggingFace")
    print("=" * 70)
    print("\nℹ️  This may take a few minutes on first run (downloads dataset)...")
    
    try:
        # Load UniProt_Function subset (has sequence + function description)
        dataset = load_dataset(
            "tsynbio/ProteinLMBench",
            "UniProt_Function",
            split="train",
            streaming=False
        )
        
        print(f"✅ Loaded ProteinLMBench")
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
    
    for i, item in enumerate(dataset):
        if valid_count >= NUM_SAMPLES:
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
                    break
        
        # Extract text/description
        text = None
        for key in ['output', 'text', 'function', 'description', 'annotation', 'response']:
            if key in item and item[key]:
                text = item[key]
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
            'id': f'protein_{valid_count:04d}',
            'sequence': sequence,
            'text': text,
            'length': len(sequence)
        })
        
        valid_count += 1
        
        if valid_count % 10 == 0:
            print(f"  Extracted {valid_count}/{NUM_SAMPLES} samples...")
    
    print(f"\n✅ Extracted {len(samples)} samples")
    
    # Check if we got any samples
    if len(samples) == 0:
        print("\n❌ ERROR: No samples extracted!")
        print("   The dataset field names may not match expected format.")
        print("   Check the field inspection output above.")
        return
    
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
    output_file = OUTPUT_DIR / 'sample_proteins.json'
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"✅ Saved samples: {output_file}")
    print(f"   Size: {output_file.stat().st_size / 1024:.1f} KB")
    
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
    print(f"   Output: {OUTPUT_DIR}")
    
    print(f"\n🚀 Next step:")
    print(f"   python 02_extract_esm_embeddings.py")


if __name__ == '__main__':
    main()

