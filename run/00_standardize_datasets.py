#!/usr/bin/env python3
# type: ignore
"""
Standardize ProteinLMBench and Mol-Instruction datasets into unified format
Outputs a JSON list with standardized format: {id, sequence, text, length, question, subset}
"""

import json
import re
import random
import gzip
from pathlib import Path
from tqdm import tqdm

def extract_sequence_from_input(input_text):
    """Extract protein sequence from various input formats"""
    if not input_text:
        return None
    
    # Handle <seq>...</seq> tags
    match = re.search(r'<seq>\s*(.*?)\s*</seq>', input_text, re.DOTALL)
    if match:
        sequence = match.group(1)
        # Remove all whitespace
        return ''.join(sequence.split())
    
    # Handle markdown code blocks with ```
    match = re.search(r'```\s*(.*?)\s*```', input_text, re.DOTALL)
    if match:
        sequence = match.group(1)
        # Remove all whitespace
        return ''.join(sequence.split())
    
    # If input is just the sequence (no special markers)
    # Remove all whitespace and check if it looks like a protein sequence
    cleaned = ''.join(input_text.split())
    # Basic check: should contain only valid amino acid letters
    if cleaned and all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in cleaned.upper()):
        return cleaned.upper()
    
    return None

def clean_text_by_subset(text):
    """Clean text output based on subset-specific prefixes"""
    if not text:
        return text
    
    # Remove common prefixes from ProteinLMBench
    prefixes_to_remove = [
        "Based on the analysis of the provided protein sequence, the function of this protein is described as follows: ",
        "The known inducer(s) or condition(s) leading to the expression of this protein include(s): ",
        "The determined function(s) of the protein include(s): ",
        "This protein is associated with the following disease(s): ",
        "The post-translational modification(s) identified in this protein include(s): ",
        "Based on the analysis, the protein's subunit structure is characterized by:",
        "The tissue specificity of this protein is characterized by: ",
        "The analysis of the specified protein sequence suggests its potential function as ",
        "Based on the given amino acid sequence, the protein appears to have a primary function of ",
        "Here is a summary of the protein with the given amino acid sequence:",
        "The protein sequence under consideration appears to have a primary function of ",
        "The protein with the amino acid sequence is expected to exhibit ",
        "Based on the provided protein sequence, the enzyme appears to facilitate the chemical reaction: ",
        "The protein characterized by the amino acid sequence demonstrates ",
        "Upon analysis of the specified amino acid sequence, it is evident that the protein performs ",
        "After analyzing the given sequence, the following protein domains or motifs are predicted: ",
        "After analyzing the amino acid sequence, the protein appears to be involved in ",
        "After evaluating the protein sequence provided, its predicted function is ",
        "Our bioinformatics tools have processed the sequence you provided sequence. The prediction suggests the presence of these protein domains or motifs: ",
        "Using bioinformatic tools, we have identified potential protein domains or motifs within your provided sequence: ",
        "A summary of the protein's main attributes with the input amino acid sequence reveals: ",
        "A brief overview of the protein with the provided amino acid sequence is as follows: ",
        "A concise description of the protein with the specified amino acid sequence includes: ",
        "An outline of the key aspects of the protein with the corresponding amino acid sequence is as follows: ",
        "An analysis of the protein sequence reveals that the enzyme's catalytic function corresponds to the chemical reaction: ",
        "The sequence you provided sequence has been analyzed for potential protein domains or motifs. The results are as follows: ",
        "Based on computational analysis, the provided sequence potentially contains the following protein domains or motifs: ",
        "The computational analysis of the sequence suggests the presence of the following protein domains or motifs: ",
        "The protein sequence you provided has been analyzed, and its subcellular localization is predicted to be ",
        "Upon evaluating your submitted sequence, our predictive algorithms suggest the presence of: ",
        "By examining the input protein sequence, the enzyme catalyzes the subsequent chemical reaction: ",
        "Evaluation of the protein sequence indicates that the associated enzyme exhibits catalytic activity in the form of this chemical reaction: ",
        "A short report on the protein with the given amino acid sequence highlights: ",
        "Upon reviewing the provided protein sequence, the corresponding enzyme's catalytic activity is identified as the following chemical reaction: ",
        "Our predictive analysis of the given protein sequence reveals possible domains or motifs. These include: "
    ]
    
    for prefix in prefixes_to_remove:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    
    return text.strip()

def get_dataset_type(source, subset_name):
    """Map source and subset to dataset type (PFUD, PDD, PSAD, PSPD)"""
    # ProteinLMBench mappings (from README)
    if source == 'ProteinLMBench':
        if 'UniProt Function' in subset_name or 'UniProt_Function' in subset_name:
            return 'PFUD'
        elif 'Subunit structure' in subset_name or 'Subunit_structure' in subset_name:
            return 'PSAD'
    
    # Mol-Instructions mappings (from README)
    elif source == 'Mol-Instructions':
        if subset_name in ['protein function', 'general function', 'catalytic activity']:
            return 'PFUD'
        elif subset_name == 'protein design':
            return 'PDD'
        elif subset_name == 'domain motif':
            return 'PSAD'
    
    return 'Unknown'

def process_proteinlmbench_file(filepath):
    """Process a single ProteinLMBench JSON file"""
    subset_name = filepath.stem.replace('_', ' ')
    
    print(f"\n📖 Processing ProteinLMBench: {subset_name}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    samples = []
    valid_count = 0
    skipped = 0
    
    for item in tqdm(data, desc=f"  {subset_name}"):
        # Extract sequence
        sequence = extract_sequence_from_input(item.get('input', ''))
        
        # Skip invalid sequences (None, empty, or whitespace-only)
        if not sequence or not sequence.strip():
            skipped += 1
            continue

        # Extract text/output
        text = item.get('output', '')
        
        # Clean text
        text = clean_text_by_subset(text)
        
        # Extract question/instruction
        question = item.get('instruction', 'What is the function of this protein?')
        
        # Determine dataset type
        dataset_type = get_dataset_type('ProteinLMBench', subset_name)
        
        samples.append({
            'id': f'proteinlmbench_{subset_name.replace(" ", "_")}_{valid_count:05d}',
            'sequence': sequence,
            'text': text,
            'length': len(sequence),
            'question': question,
            'subset': f'ProteinLMBench/{subset_name}',
            'type': dataset_type
        })
        
        valid_count += 1
    
    print(f"  ✅ Extracted: {valid_count} samples (skipped: {skipped})")
    return samples

def process_mol_instructions_file(filepath):
    """Process a single Mol-Instructions JSON file"""
    subset_name = filepath.stem.replace('_', ' ')
    
    print(f"\n📖 Processing Mol-Instructions: {subset_name}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    samples = []
    valid_count = 0
    skipped = 0
    
    # Special handling for protein_design
    is_protein_design = 'protein design' in subset_name.lower() or 'protein_design' in filepath.stem
    
    for item in tqdm(data, desc=f"  {subset_name}"):
        # Special case: protein_design has sequence in OUTPUT, not input
        if is_protein_design:
            # Extract sequence from output (in markdown code blocks)
            sequence = extract_sequence_from_input(item.get('output', ''))
            
            # Skip invalid sequences (None, empty, or whitespace-only)
            if not sequence or not sequence.strip():
                skipped += 1
                continue
            
            # Get specifications from input
            specifications = item.get('input', '')
            
            # Get instruction
            instruction = item.get('instruction', 'Design a protein sequence.')
            
            # Empty text
            text = ''
            
            # Combine instruction + specifications as question
            question = f"{instruction}\n{specifications}"
        else:
            # Normal case: sequence in INPUT
            sequence = extract_sequence_from_input(item.get('input', ''))
            
            # Skip invalid sequences (None, empty, or whitespace-only)
            if not sequence or not sequence.strip():
                skipped += 1
                continue
            
            # Extract text/output
            text = item.get('output', '')
            
            # Clean text
            text = clean_text_by_subset(text)
            
            # Extract question/instruction
            question = item.get('instruction', 'What is the function of this protein?')
        
        # Determine dataset type
        dataset_type = get_dataset_type('Mol-Instructions', subset_name)
        
        samples.append({
            'id': f'mol_instructions_{subset_name.replace(" ", "_")}_{valid_count:05d}',
            'sequence': sequence,
            'text': text,
            'length': len(sequence),
            'question': question,
            'subset': f'Mol-Instructions/{subset_name}',
            'type': dataset_type
        })
        
        valid_count += 1
    
    print(f"  ✅ Extracted: {valid_count} samples (skipped: {skipped})")
    return samples

def process_alphafold_pdb(filepath):
    """Extract sequence from AlphaFold PDB file"""
    try:
        # Extract UniProt ID from filename: AF-A0A009IHW8-F1-model_v4.pdb.gz
        filename = filepath.stem.replace('.pdb', '')  # Remove .pdb from .pdb.gz
        uniprot_id = filename.split('-')[1] if '-' in filename else filename
        
        with gzip.open(filepath, 'rt') as f:
            seqres_lines = []
            for line in f:
                if line.startswith('SEQRES'):
                    # SEQRES lines contain amino acid 3-letter codes
                    parts = line.split()
                    # Skip first 4 fields: SEQRES, serial, chain, count
                    amino_acids = parts[4:]
                    seqres_lines.extend(amino_acids)
        
        if not seqres_lines:
            return None
        
        # Convert 3-letter codes to 1-letter codes
        aa_map = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        
        sequence = ''.join([aa_map.get(aa, 'X') for aa in seqres_lines])
        
        return {
            'uniprot_id': uniprot_id,
            'sequence': sequence
        }
    except Exception as e:
        return None

def process_alphafold_structures(alphafold_dir, max_samples=None):
    """Process AlphaFold PDB structure files"""
    print(f"\n📖 Processing AlphaFold Structures")
    
    # Get all PDB files
    pdb_files = sorted(alphafold_dir.glob('*.pdb.gz'))
    
    if max_samples:
        pdb_files = pdb_files[:max_samples]
    
    print(f"   Found {len(pdb_files)} PDB files")
    
    samples = []
    valid_count = 0
    skipped = 0
    
    for pdb_file in tqdm(pdb_files, desc="  Processing PDB files"):
        result = process_alphafold_pdb(pdb_file)
        
        if not result:
            skipped += 1
            continue
        
        sequence = result['sequence']
        
        # Skip if sequence is empty or has unknown amino acids
        if not sequence or 'X' in sequence:
            skipped += 1
            continue
        
        samples.append({
            'id': f'alphafold_{result["uniprot_id"]}_{valid_count:05d}',
            'sequence': sequence,
            'text': '',  # Leave blank as requested
            'length': len(sequence),
            'question': '',  # Leave blank as requested
            'subset': 'AlphaFold/SwissProt_v4',
            'type': 'PSPD'
        })
        
        valid_count += 1
    
    print(f"  ✅ Extracted: {valid_count} samples (skipped: {skipped})")
    return samples

def deduplicate_by_sequence(samples):
    """Deduplicate samples by sequence, combining texts"""
    print("\n" + "=" * 70)
    print("DEDUPLICATING BY SEQUENCE")
    print("=" * 70)
    
    print(f"Total samples before deduplication: {len(samples)}")
    
    seen_seqs = {}
    
    for sample in tqdm(samples, desc="Deduplicating"):
        # Truncate to first 1022 residues for deduplication
        seq = sample['sequence'][:1022]  
        
        if seq not in seen_seqs:
            seen_seqs[seq] = {
                'id': sample['id'],
                'sequence': seq,
                'text': [sample['text']],
                'length': sample['length'],
                'question': sample['question'],
                'subsets': {sample['subset']},
                'types': {sample['type']}
            }
        else:
            # Combine information
            seen_seqs[seq]['text'].append(sample['text'])
            seen_seqs[seq]['subsets'].add(sample['subset'])
            seen_seqs[seq]['types'].add(sample['type'])
    
    # Reconstruct deduplicated samples
    deduplicated = []
    for seq, data in seen_seqs.items():
        # Combine texts with space
        combined_text = ' '.join(data['text'])
        
        deduplicated.append({
            'id': data['id'],
            'sequence': data['sequence'],
            'text': combined_text,
            'length': data['length'],
            'question': data['question'],
            'subset': ', '.join(sorted(data['subsets'])),
            'type': ', '.join(sorted(data['types']))
        })
    
    print(f"Unique sequences: {len(deduplicated)}")
    print(f"Duplicates removed: {len(samples) - len(deduplicated)}")
    
    return deduplicated

def main():
    print("=" * 70)
    print("STANDARDIZE PROTEIN INSTRUCTION DATASETS")
    print("=" * 70)
    
    # Define data paths
    data_dir = Path(__file__).parent.parent / 'data'
    proteinlmbench_dir = data_dir / 'proteinlmbench'
    mol_instructions_dir = data_dir / 'mol_instructions_hf' / 'Protein-oriented_Instructions'
    alphafold_dir = data_dir / 'alphafold' / 'swissprot_v4'
    
    # Output path
    output_file = data_dir / 'standardized_protein_instructions.json'
    
    all_samples = []
    
    # Process ProteinLMBench files (only the ones mentioned in README)
    print("\n" + "=" * 70)
    print("PROCESSING PROTEINLMBENCH")
    print("=" * 70)
    
    # Only process these two subsets as mentioned in README
    proteinlmbench_subsets = [
        'UniProt_Function.json',  # 465K → PFUD
        'UniProt_Subunit_structure.json'  # 291K → PSAD
    ]
    
    if proteinlmbench_dir.exists():
        for subset_file in proteinlmbench_subsets:
            json_file = proteinlmbench_dir / subset_file
            if json_file.exists():
                samples = process_proteinlmbench_file(json_file)
                all_samples.extend(samples)
            else:
                print(f"⚠️  File not found: {json_file}")
    else:
        print(f"⚠️  ProteinLMBench directory not found: {proteinlmbench_dir}")
    
    # Process Mol-Instructions files
    print("\n" + "=" * 70)
    print("PROCESSING MOL-INSTRUCTIONS")
    print("=" * 70)
    
    if mol_instructions_dir.exists():
        for json_file in sorted(mol_instructions_dir.glob('*.json')):
            # Skip summary/metadata files
            if json_file.stem in ['download_summary']:
                continue
            
            samples = process_mol_instructions_file(json_file)
            all_samples.extend(samples)
    else:
        print(f"⚠️  Mol-Instructions directory not found: {mol_instructions_dir}")
    
    # Process AlphaFold structures
    print("\n" + "=" * 70)
    print("PROCESSING ALPHAFOLD STRUCTURES")
    print("=" * 70)
    
    if alphafold_dir.exists():
        # Process AlphaFold (you can limit the number with max_samples parameter)
        samples = process_alphafold_structures(alphafold_dir, max_samples=None)
        all_samples.extend(samples)
    else:
        print(f"⚠️  AlphaFold directory not found: {alphafold_dir}")
    
    # Deduplicate
    all_samples = deduplicate_by_sequence(all_samples)
    
    # Shuffle samples
    print("\n" + "=" * 70)
    print("SHUFFLING")
    print("=" * 70)
    random.seed(42)  # For reproducibility
    random.shuffle(all_samples)
    print(f"✅ Shuffled {len(all_samples)} samples")
    
    # Save standardized dataset
    print("\n" + "=" * 70)
    print("SAVING STANDARDIZED DATASET")
    print("=" * 70)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_samples, f, indent=2)
    
    print(f"✅ Saved to: {output_file}")
    print(f"   Total samples: {len(all_samples)}")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    
    # Count by subset and type
    from collections import Counter
    subsets = Counter()
    types = Counter()
    
    for sample in all_samples:
        for subset in sample['subset'].split(', '):
            subsets[subset] += 1
        for dtype in sample['type'].split(', '):
            types[dtype] += 1
    
    print("\nSamples by subset:")
    for subset, count in subsets.most_common():
        print(f"  {subset}: {count}")
    
    print("\nSamples by type:")
    for dtype, count in types.most_common():
        print(f"  {dtype}: {count}")
    
    # Length statistics
    lengths = [s['length'] for s in all_samples]
    sorted_lengths = sorted(lengths)
    print(f"\nSequence length statistics:")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Mean: {sum(lengths)/len(lengths):.1f}")
    print(f"  Median: {sorted_lengths[len(lengths)//2]}")
    
    # Save statistics to metadata.json
    metadata = {
        'total_samples': len(all_samples),
        'unique_sequences': len(all_samples),
        'samples_by_subset': dict(subsets.most_common()),
        'samples_by_type': dict(types.most_common()),
        'sequence_length_stats': {
            'min': min(lengths),
            'max': max(lengths),
            'mean': round(sum(lengths)/len(lengths), 1),
            'median': sorted_lengths[len(lengths)//2]
        }
    }
    
    metadata_file = data_dir / 'standardized_protein_instructions_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Metadata saved to: {metadata_file}")
    
    # Show sample entries
    print("\n" + "=" * 70)
    print("SAMPLE ENTRIES")
    print("=" * 70)
    
    for i, sample in enumerate(all_samples[:3]):
        print(f"\nSample {i+1}:")
        print(f"  ID: {sample['id']}")
        print(f"  Type: {sample['type']}")
        print(f"  Subset: {sample['subset']}")
        print(f"  Sequence length: {sample['length']}")
        print(f"  Sequence: {sample['sequence'][:50]}...")
        print(f"  Question: {sample['question'][:80]}...")
        print(f"  Text: {sample['text'][:100]}...")
    
    print("\n" + "=" * 70)
    print("✅ STANDARDIZATION COMPLETE")
    print("=" * 70)

if __name__ == '__main__':
    main()
