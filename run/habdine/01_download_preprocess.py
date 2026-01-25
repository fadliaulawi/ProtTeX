#!/usr/bin/env python3
"""
Download and Preprocess Prot2Text-Data from Hugging Face
Downloads the dataset and converts it to standardized format compatible with the pipeline.
"""

import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import sys

# Add root directory to Python path
script_dir = Path(__file__).parent
root_dir = script_dir.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

def main():
    print("=" * 70)
    print("DOWNLOAD AND PREPROCESS Prot2Text-Data")
    print("=" * 70)
    
    # Setup directories
    data_dir = root_dir / 'data' / 'habdine'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = data_dir / 'standardized_prot2text.json'
    
    print(f"\n📥 Downloading Prot2Text-Data from Hugging Face...")
    print(f"   Dataset: habdine/Prot2Text-Data")
    
    # Load all splits from Hugging Face
    try:
        print("📥 Loading train split...")
        train_dataset = load_dataset("habdine/Prot2Text-Data", split="train")
        print(f"✅ Train: {len(train_dataset)} samples")
        
        print("📥 Loading validation split...")
        val_dataset = load_dataset("habdine/Prot2Text-Data", split="validation")
        print(f"✅ Validation: {len(val_dataset)} samples")
        
        print("📥 Loading test split...")
        test_dataset = load_dataset("habdine/Prot2Text-Data", split="test")
        print(f"✅ Test: {len(test_dataset)} samples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Process each split
    def process_split(dataset, split_name):
        """Process a single split and convert to standardized format"""
        standardized_samples = []
        skipped = 0
        
        for item in tqdm(dataset, desc=f"Processing {split_name}"):
            # Extract fields
            accession = item.get('accession', '')
            name = item.get('name', '')
            full_name = item.get('Full Name', '')
            sequence = item.get('sequence', '')
            function = item.get('function', '')
            taxon = item.get('taxon', '')
            alphafold_id = item.get('AlphaFoldDB', '')
            
            # Skip if missing essential fields
            if not sequence or not function:
                skipped += 1
                continue
            
            # Clean sequence (uppercase, remove whitespace)
            sequence = ''.join(sequence.upper().split())
            
            # Skip if sequence is invalid
            valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
            if not all(aa in valid_aa for aa in sequence):
                skipped += 1
                continue
            
            # Create standardized sample
            sample = {
                'QA': {
                    'question': 'What is the function of this protein?',
                    'answer': function.strip(),
                    'sequence': sequence
                },
                'metadata': {
                    'id': accession if accession else f'prot2text_{split_name}_{len(standardized_samples):06d}',
                    'length': len(sequence),
                    'subset': f'Prot2Text/{split_name}',
                    'type': 'PFUD',  # Protein Function Description
                    'pdb_id': '',
                    'uniprot_id': accession,
                    'accession_id': accession,
                    'name': name,
                    'full_name': full_name,
                    'taxon': taxon,
                    'alphafold_id': alphafold_id
                }
            }
            
            standardized_samples.append(sample)
        
        return standardized_samples, skipped
    
    # Process all splits
    print(f"\n📝 Converting to standardized format...")
    
    train_samples, train_skipped = process_split(train_dataset, 'train')
    val_samples, val_skipped = process_split(val_dataset, 'validation')
    test_samples, test_skipped = process_split(test_dataset, 'test')
    
    total_samples = len(train_samples) + len(val_samples) + len(test_samples)
    total_skipped = train_skipped + val_skipped + test_skipped
    
    print(f"\n✅ Processed {total_samples} samples")
    print(f"   Train: {len(train_samples):,} samples")
    print(f"   Validation: {len(val_samples):,} samples")
    print(f"   Test: {len(test_samples):,} samples")
    if total_skipped > 0:
        print(f"⚠️  Skipped {total_skipped} samples (missing sequence or function)")
    
    # Save each split separately
    print(f"\n💾 Saving splits...")
    
    train_file = data_dir / 'standardized_prot2text_train.json'
    val_file = data_dir / 'standardized_prot2text_validation.json'
    test_file = data_dir / 'standardized_prot2text_test.json'
    
    with open(train_file, 'w') as f:
        json.dump(train_samples, f, indent=2)
    print(f"✅ Saved {len(train_samples)} train samples to {train_file}")
    
    with open(val_file, 'w') as f:
        json.dump(val_samples, f, indent=2)
    print(f"✅ Saved {len(val_samples)} validation samples to {val_file}")
    
    with open(test_file, 'w') as f:
        json.dump(test_samples, f, indent=2)
    print(f"✅ Saved {len(test_samples)} test samples to {test_file}")
    
    # Also save combined file for convenience
    all_samples = train_samples + val_samples + test_samples
    with open(output_file, 'w') as f:
        json.dump(all_samples, f, indent=2)
    print(f"✅ Saved {len(all_samples)} total samples to {output_file}")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    print(f"Total samples: {total_samples:,}")
    print(f"  Train: {len(train_samples):,}")
    print(f"  Validation: {len(val_samples):,}")
    print(f"  Test: {len(test_samples):,}")
    
    # Length statistics (combined)
    all_samples = train_samples + val_samples + test_samples
    lengths = [s['metadata']['length'] for s in all_samples]
    print(f"\nSequence length - Min: {min(lengths)}, Max: {max(lengths)}, Avg: {sum(lengths)/len(lengths):.1f}")
    
    # Function length statistics
    func_lengths = [len(s['QA']['answer']) for s in all_samples]
    print(f"Function text length - Min: {min(func_lengths)}, Max: {max(func_lengths)}, Avg: {sum(func_lengths)/len(func_lengths):.1f}")
    
    # Samples with metadata
    with_name = sum(1 for s in all_samples if s['metadata'].get('name'))
    with_taxon = sum(1 for s in all_samples if s['metadata'].get('taxon'))
    with_alphafold = sum(1 for s in all_samples if s['metadata'].get('alphafold_id'))
    print(f"\nMetadata coverage:")
    print(f"  With name: {with_name:,} ({100*with_name/len(all_samples):.1f}%)")
    print(f"  With taxon: {with_taxon:,} ({100*with_taxon/len(all_samples):.1f}%)")
    print(f"  With AlphaFold ID: {with_alphafold:,} ({100*with_alphafold/len(all_samples):.1f}%)")

if __name__ == "__main__":
    main()
