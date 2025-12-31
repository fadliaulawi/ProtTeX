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
import os
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def get_optimal_workers():
    """Determine optimal number of workers based on system resources
    
    For I/O-bound tasks (file reading), we can use more workers than CPU cores.
    For CPU-bound tasks (parsing), we should use CPU count.
    Since we're doing both, use a multiplier for I/O-bound operations.
    
    Returns:
        int: Optimal number of workers
    """
    try:
        # Try to get CPU count
        cpu_count = os.cpu_count() or 4  # Default to 4 if None
        
        # For I/O-bound tasks (file reading), we can use more workers than CPU cores
        # For CPU-bound tasks (parsing), we should use CPU count
        # Since we're doing both, use a multiplier for I/O-bound operations
        if cpu_count <= 16:
            # Small systems: 2x CPU count, cap at 32
            optimal = min(cpu_count * 2, 32)
        elif cpu_count <= 64:
            # Medium systems: 1.5x CPU count, cap at 96
            optimal = min(int(cpu_count * 1.5), 96)
        else:
            # Large systems (64+ cores): Use CPU count + 32 for I/O overhead
            optimal = min(cpu_count + 32, 128)  # Cap at 128 for very large systems
        
        # DEBUG
        optimal = 32

        return optimal
    except Exception:
        return 8  # Safe default

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
        
        sequence = ''.join([aa_map[aa] for aa in seqres_lines if aa in aa_map])
        return {
            'uniprot_id': uniprot_id,
            'sequence': sequence
        }
    except Exception as e:
        return None

def process_alphafold_structures(alphafold_dir, max_samples=None, num_workers=8):
    """Process AlphaFold PDB structure files"""
    print(f"\n📖 Processing AlphaFold Structures")
    
    # Get all PDB files
    pdb_files = sorted(alphafold_dir.glob('*.pdb.gz'))
    
    if max_samples:
        pdb_files = pdb_files[:max_samples]
    
    print(f"   Found {len(pdb_files)} PDB files")
    print(f"   Using {num_workers} workers for parallel processing")
    
    samples = []
    valid_count = 0
    skipped = 0
    
    def process_single_file(pdb_file):
        """Process a single PDB file and return result"""
        result = process_alphafold_pdb(pdb_file)
        if not result:
            return None, 'skip_no_result'
        
        sequence = result['sequence']
        if not sequence or 'X' in sequence:
            return None, 'skip_invalid'
        
        return {
            'uniprot_id': result['uniprot_id'],
            'sequence': sequence,
            'length': len(sequence)
        }, 'success'
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_file, pdb_file): pdb_file for pdb_file in pdb_files}
        
        for future in tqdm(as_completed(futures), total=len(pdb_files), desc="  Processing PDB files"):
            result, status = future.result()
            
            if status == 'skip_no_result' or status == 'skip_invalid':
                skipped += 1
                continue
            
            samples.append({
                'id': f'alphafold_{result["uniprot_id"]}_{valid_count:05d}',
                'sequence': result['sequence'],
                'text': '',  # Leave blank as requested
                'length': result['length'],
                'question': '',  # Leave blank as requested
                'subset': 'AlphaFold/SwissProt_v4',
                'type': 'PSPD'
            })
            
            valid_count += 1
    
    print(f"  ✅ Extracted: {valid_count} samples (skipped: {skipped})")
    return samples

def process_pspd_alphafold_cif(filepath):
    """Extract sequence from PSPD AlphaFold CIF file (mmCIF format)"""
    try:
        # Extract UniProt ID from filename: AF-A0A0A1GRI8-F1-model_v4.cif or .cif.gz
        filename = filepath.name
        # Remove extensions
        if filename.endswith('.cif'):
            filename = filename[:-4]  # Remove .cif
        # Format: AF-A0A0A1GRI8-F1-model_v4
        parts = filename.split('-')
        uniprot_id = parts[1] if len(parts) > 1 else filename
        
        # Open file (may be gzipped or not)
        f = open(filepath, 'rt')
        
        seq_data = []
        mon_id_col = -1
        
        with f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Check if we're entering entity_poly_seq loop
                if line == 'loop_':
                    # Check next lines for entity_poly_seq column headers
                    j = i + 1
                    headers = []
                    while j < len(lines) and lines[j].strip().startswith('_entity_poly_seq.'):
                        headers.append(lines[j].strip())
                        j += 1
                    
                    # Check if this is the entity_poly_seq loop
                    if any('_entity_poly_seq.entity_id' in h or '_entity_poly_seq.num' in h for h in headers):
                        in_entity_poly_seq_loop = True
                        # Find mon_id column index
                        for idx, h in enumerate(headers):
                            if 'mon_id' in h.lower():
                                mon_id_col = idx
                                break
                        # If not found, assume it's the last column
                        if mon_id_col == -1:
                            mon_id_col = len(headers) - 1
                        
                        # Now read data lines
                        k = j
                        while k < len(lines):
                            data_line = lines[k].strip()
                            if not data_line or data_line.startswith('#'):
                                k += 1
                                continue
                            if data_line.startswith('_') or data_line.startswith('loop_'):
                                break
                            
                            parts = data_line.split()
                            if len(parts) > mon_id_col:
                                mon_id = parts[mon_id_col].strip('"\'')
                                seq_data.append(mon_id)
                            k += 1
                        
                        i = k - 1
                
                # Fallback: extract from ATOM records if _entity_poly_seq not found
                if not seq_data and line.startswith('ATOM'):
                    parts = line.split()
                    if len(parts) >= 4:
                        res_name = parts[3].strip('"\'')  # Residue name (3-letter code)
                        # Only add if different from last residue (avoid duplicates)
                        if not seq_data or seq_data[-1] != res_name:
                            seq_data.append(res_name)
                
                i += 1
        
        if not seq_data:
            return None
        
        # Convert 3-letter codes to 1-letter codes
        aa_map = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
            # Handle modified residues
            'MSE': 'M', 'SEC': 'C', 'PYL': 'K'
        }
        
        sequence = ''.join([aa_map[aa.upper()] for aa in seq_data if aa.upper() in aa_map])
        
        return {
            'uniprot_id': uniprot_id,
            'sequence': sequence
        }
    except Exception as e:
        return None

def process_pspd_alphafold_structures(pspd_alphafold_dir, max_samples=None, num_workers=8):
    """Process PSPD AlphaFold CIF structure files"""
    print(f"\n📖 Processing PSPD AlphaFold Structures")
    
    # Find all CIF files in proteome subdirectories
    cif_files = []
    for proteome_dir in sorted(pspd_alphafold_dir.glob('proteome-*')):
        if proteome_dir.is_dir():
            # Look for .cif files (ungzipped) or .cif.gz files
            cif_files.extend(sorted(proteome_dir.glob('*model_v4.cif')))
    
    if max_samples:
        cif_files = cif_files[:max_samples]
    
    print(f"   Found {len(cif_files)} CIF files")
    print(f"   Using {num_workers} workers for parallel processing")
    
    samples = []
    valid_count = 0
    skipped = 0
    
    def process_single_file(cif_file):
        """Process a single CIF file and return result"""
        result = process_pspd_alphafold_cif(cif_file)
        if not result:
            return None, 'skip_no_result', None
        
        sequence = result['sequence']
        if not sequence or 'X' in sequence:
            return None, 'skip_invalid', None
        
        # Extract proteome name from parent directory
        proteome_name = cif_file.parent.name
        
        return {
            'uniprot_id': result['uniprot_id'],
            'sequence': sequence,
            'length': len(sequence),
            'proteome_name': proteome_name
        }, 'success', proteome_name
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_file, cif_file): cif_file for cif_file in cif_files}
        
        for future in tqdm(as_completed(futures), total=len(cif_files), desc="  Processing CIF files"):
            result, status, proteome_name = future.result()
            
            if status == 'skip_no_result' or status == 'skip_invalid':
                skipped += 1
                continue
            
            samples.append({
                'id': f'pspd_alphafold_{result["uniprot_id"]}_{valid_count:05d}',
                'sequence': result['sequence'],
                'text': '',  # Leave blank as requested
                'length': result['length'],
                'question': '',  # Leave blank as requested
                'subset': f'PSPD/AlphaFold/{proteome_name}',
                'type': 'PSPD'
            })
            
            valid_count += 1
    
    print(f"  ✅ Extracted: {valid_count} samples (skipped: {skipped})")
    return samples

def process_pspd_rcsb_pdb(filepath):
    """Extract sequence from PSPD RCSB PDB file"""
    try:
        # Extract PDB ID from filename: 1A00.pdb
        pdb_id = filepath.stem.upper()
        
        with open(filepath, 'rt') as f:
            seqres_lines = []
            for line in f:
                if line.startswith('SEQRES'):
                    # SEQRES lines contain amino acid 3-letter codes
                    parts = line.split()
                    # Skip first 4 fields: SEQRES, serial, chain, count
                    if len(parts) > 4:
                        amino_acids = parts[4:]
                        seqres_lines.extend(amino_acids)
        
        if not seqres_lines:
            return None
        
        # Convert 3-letter codes to 1-letter codes
        aa_map = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
            # Handle modified residues
            'MSE': 'M', 'SEC': 'C', 'PYL': 'K'
        }
        
        sequence = ''.join([aa_map[aa] for aa in seqres_lines if aa in aa_map])
        
        return {
            'pdb_id': pdb_id,
            'sequence': sequence
        }
    except Exception as e:
        return None

def process_pspd_rcsb_structures(pspd_rcsb_dir, max_samples=None, num_workers=8):
    """Process PSPD RCSB PDB structure files"""
    print(f"\n📖 Processing PSPD RCSB Structures")
    
    # Get all PDB files
    pdb_files = sorted(pspd_rcsb_dir.glob('*.pdb'))
    
    if max_samples:
        pdb_files = pdb_files[:max_samples]
    
    print(f"   Found {len(pdb_files)} PDB files")
    print(f"   Using {num_workers} workers for parallel processing")
    
    samples = []
    valid_count = 0
    skipped = 0
    
    def process_single_file(pdb_file):
        """Process a single PDB file and return result"""
        result = process_pspd_rcsb_pdb(pdb_file)
        if not result:
            return None, 'skip_no_result'
        
        sequence = result['sequence']
        if not sequence or 'X' in sequence:
            return None, 'skip_invalid'
        
        return {
            'pdb_id': result['pdb_id'],
            'sequence': sequence,
            'length': len(sequence)
        }, 'success'

    # Process files in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_file, pdb_file): pdb_file for pdb_file in pdb_files}
        
        for future in tqdm(as_completed(futures), total=len(pdb_files), desc="  Processing PDB files"):
            result, status = future.result()
            
            if status == 'skip_no_result' or status == 'skip_invalid':
                skipped += 1
                continue
            
            samples.append({
                'id': f'pspd_rcsb_{result["pdb_id"]}_{valid_count:05d}',
                'sequence': result['sequence'],
                'text': '',  # Leave blank as requested
                'length': result['length'],
                'question': '',  # Leave blank as requested
                'subset': 'PSPD/RCSB_PDB',
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
        # Truncate to first 2048 residues for deduplication
        seq = sample['sequence'][:2048]  
        
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
    
    # Get number of workers from environment or auto-detect optimal
    num_workers = get_optimal_workers()
    cpu_count = os.cpu_count() or 'unknown'
    print(f"Auto-detected optimal workers: {num_workers} (CPU count: {cpu_count})")
    print(f"  (Set NUM_WORKERS environment variable to override)")    
    print(f"Using {num_workers} workers for parallel processing")
    
    # Define data paths
    data_dir = Path(__file__).parent.parent / 'data' / 'datasets'
    proteinlmbench_dir = data_dir / 'proteinlmbench'
    mol_instructions_dir = data_dir / 'mol_instructions_hf' / 'Protein-oriented_Instructions'
    alphafold_dir = data_dir / 'alphafold' / 'swissprot_v4'
    pspd_alphafold_dir = data_dir / 'pspd' / 'alphafold'
    pspd_rcsb_dir = data_dir / 'pspd' / 'rcsb_pdb'
    
    # Output path
    output_file = data_dir / 'standardized_protein_instructions.json'
    
    all_samples = []
    initial_stats_by_subset = {}  # Track stats before deduplication
    initial_stats_by_type = {}  # Track stats by subtype (PFUD, PSPD, PSAD, PDD)
    
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
                # Track initial stats for this subset and type
                if samples:
                    subset_name = samples[0]['subset']
                    dataset_type = samples[0]['type']
                    lengths = [s['length'] for s in samples]
                    sorted_lengths = sorted(lengths)
                    
                    # Track by subset
                    initial_stats_by_subset[subset_name] = {
                        'count': len(samples),
                        'sequence_length_stats': {
                            'min': min(lengths),
                            'max': max(lengths),
                            'mean': round(sum(lengths)/len(lengths), 1),
                            'median': sorted_lengths[len(lengths)//2]
                        }
                    }
                    
                    # Track by type (subtype)
                    if dataset_type not in initial_stats_by_type:
                        initial_stats_by_type[dataset_type] = {
                            'count': 0,
                            'lengths': []
                        }
                    initial_stats_by_type[dataset_type]['count'] += len(samples)
                    initial_stats_by_type[dataset_type]['lengths'].extend(lengths)
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
            # Track initial stats for this subset and type
            if samples:
                subset_name = samples[0]['subset']
                dataset_type = samples[0]['type']
                lengths = [s['length'] for s in samples]
                sorted_lengths = sorted(lengths)
                
                # Track by subset
                initial_stats_by_subset[subset_name] = {
                    'count': len(samples),
                    'sequence_length_stats': {
                        'min': min(lengths),
                        'max': max(lengths),
                        'mean': round(sum(lengths)/len(lengths), 1),
                        'median': sorted_lengths[len(lengths)//2]
                    }
                }
                
                # Track by type (subtype)
                if dataset_type not in initial_stats_by_type:
                    initial_stats_by_type[dataset_type] = {
                        'count': 0,
                        'lengths': []
                    }
                initial_stats_by_type[dataset_type]['count'] += len(samples)
                initial_stats_by_type[dataset_type]['lengths'].extend(lengths)
            all_samples.extend(samples)
    else:
        print(f"⚠️  Mol-Instructions directory not found: {mol_instructions_dir}")
    
    # Process AlphaFold structures
    print("\n" + "=" * 70)
    print("PROCESSING ALPHAFOLD STRUCTURES")
    print("=" * 70)
    
    if alphafold_dir.exists():
        # Process AlphaFold (you can limit the number with max_samples parameter)
        samples = process_alphafold_structures(alphafold_dir, max_samples=None, num_workers=num_workers)
        # Track initial stats for this subset and type
        if samples:
            subset_name = samples[0]['subset']
            dataset_type = samples[0]['type']
            lengths = [s['length'] for s in samples]
            sorted_lengths = sorted(lengths)
            
            # Track by subset
            initial_stats_by_subset[subset_name] = {
                'count': len(samples),
                'sequence_length_stats': {
                    'min': min(lengths),
                    'max': max(lengths),
                    'mean': round(sum(lengths)/len(lengths), 1),
                    'median': sorted_lengths[len(lengths)//2]
                }
            }
            
            # Track by type (subtype)
            if dataset_type not in initial_stats_by_type:
                initial_stats_by_type[dataset_type] = {
                    'count': 0,
                    'lengths': []
                }
            initial_stats_by_type[dataset_type]['count'] += len(samples)
            initial_stats_by_type[dataset_type]['lengths'].extend(lengths)
        all_samples.extend(samples)
    else:
        print(f"⚠️  AlphaFold directory not found: {alphafold_dir}")
    
    # Process PSPD AlphaFold structures
    print("\n" + "=" * 70)
    print("PROCESSING PSPD ALPHAFOLD STRUCTURES")
    print("=" * 70)
    
    if pspd_alphafold_dir.exists():
        samples = process_pspd_alphafold_structures(pspd_alphafold_dir, max_samples=None, num_workers=num_workers)
        # Track initial stats - PSPD AlphaFold may have multiple proteomes
        if samples:
            proteome_samples = defaultdict(list)
            for sample in samples:
                proteome_samples[sample['subset']].append(sample)
            
            for subset_name, subset_samples in proteome_samples.items():
                dataset_type = subset_samples[0]['type']
                lengths = [s['length'] for s in subset_samples]
                sorted_lengths = sorted(lengths)
                
                # Track by subset
                initial_stats_by_subset[subset_name] = {
                    'count': len(subset_samples),
                    'sequence_length_stats': {
                        'min': min(lengths),
                        'max': max(lengths),
                        'mean': round(sum(lengths)/len(lengths), 1),
                        'median': sorted_lengths[len(lengths)//2]
                    }
                }
                
                # Track by type (subtype)
                if dataset_type not in initial_stats_by_type:
                    initial_stats_by_type[dataset_type] = {
                        'count': 0,
                        'lengths': []
                    }
                initial_stats_by_type[dataset_type]['count'] += len(subset_samples)
                initial_stats_by_type[dataset_type]['lengths'].extend(lengths)
        all_samples.extend(samples)
    else:
        print(f"⚠️  PSPD AlphaFold directory not found: {pspd_alphafold_dir}")
    
    # Process PSPD RCSB structures
    print("\n" + "=" * 70)
    print("PROCESSING PSPD RCSB STRUCTURES")
    print("=" * 70)
    
    if pspd_rcsb_dir.exists():
        samples = process_pspd_rcsb_structures(pspd_rcsb_dir, max_samples=None, num_workers=num_workers)
        # Track initial stats for this subset and type
        if samples:
            subset_name = samples[0]['subset']
            dataset_type = samples[0]['type']
            lengths = [s['length'] for s in samples]
            sorted_lengths = sorted(lengths)
            
            # Track by subset
            initial_stats_by_subset[subset_name] = {
                'count': len(samples),
                'sequence_length_stats': {
                    'min': min(lengths),
                    'max': max(lengths),
                    'mean': round(sum(lengths)/len(lengths), 1),
                    'median': sorted_lengths[len(lengths)//2]
                }
            }
            
            # Track by type (subtype)
            if dataset_type not in initial_stats_by_type:
                initial_stats_by_type[dataset_type] = {
                    'count': 0,
                    'lengths': []
                }
            initial_stats_by_type[dataset_type]['count'] += len(samples)
            initial_stats_by_type[dataset_type]['lengths'].extend(lengths)
        all_samples.extend(samples)
    else:
        print(f"⚠️  PSPD RCSB directory not found: {pspd_rcsb_dir}")
    
    # Calculate final stats by type
    initial_stats_by_type_final = {}
    for dtype, data in initial_stats_by_type.items():
        lengths = data['lengths']
        sorted_lengths = sorted(lengths)
        initial_stats_by_type_final[dtype] = {
            'count': data['count'],
            'sequence_length_stats': {
                'min': min(lengths),
                'max': max(lengths),
                'mean': round(sum(lengths)/len(lengths), 1),
                'median': sorted_lengths[len(lengths)//2]
            }
        }
    
    # Print initial statistics before deduplication
    print("\n" + "=" * 70)
    print("INITIAL STATISTICS (BEFORE DEDUPLICATION)")
    print("=" * 70)
    print(f"Total samples: {len(all_samples)}")
    print("\nInitial stats by subset:")
    for subset_name, stats in sorted(initial_stats_by_subset.items()):
        print(f"  {subset_name}: {stats['count']} samples")
        print(f"    Length - Min: {stats['sequence_length_stats']['min']}, "
              f"Max: {stats['sequence_length_stats']['max']}, "
              f"Mean: {stats['sequence_length_stats']['mean']}, "
              f"Median: {stats['sequence_length_stats']['median']}")
    
    print("\nInitial stats by type (subtype):")
    for dtype, stats in sorted(initial_stats_by_type_final.items()):
        print(f"  {dtype}: {stats['count']} samples")
        print(f"    Length - Min: {stats['sequence_length_stats']['min']}, "
              f"Max: {stats['sequence_length_stats']['max']}, "
              f"Mean: {stats['sequence_length_stats']['mean']}, "
              f"Median: {stats['sequence_length_stats']['median']}")
    
    # Print formatted tables
    print("\n" + "=" * 70)
    print("INITIAL STATISTICS TABLES (BEFORE DEDUPLICATION)")
    print("=" * 70)
    
    # Table 1: By Subset
    print("\n📊 Statistics by Subset:")
    print("-" * 70)
    print(f"{'Subset':<50} {'Count':>15}")
    print("-" * 70)
    subset_total = 0
    for subset_name, stats in sorted(initial_stats_by_subset.items()):
        count = stats['count']
        subset_total += count
        print(f"{subset_name:<50} {count:>15,}")
    print("-" * 70)
    print(f"{'TOTAL':<50} {subset_total:>15,}")
    
    # Table 2: By Type (Subtype)
    print("\n📊 Statistics by Type (Subtype):")
    print("-" * 70)
    print(f"{'Type':<50} {'Count':>15}")
    print("-" * 70)
    type_total = 0
    for dtype, stats in sorted(initial_stats_by_type_final.items()):
        count = stats['count']
        type_total += count
        # Add description for each type
        type_desc = {
            'PFUD': 'Function',
            'PDD': 'Design',
            'PSAD': 'Analysis',
            'PSPD': 'Structure'
        }
        desc = type_desc.get(dtype, '')
        type_label = f"{dtype} ({desc})" if desc else dtype
        print(f"{type_label:<50} {count:>15,}")
    print("-" * 70)
    print(f"{'TOTAL':<50} {type_total:>15,}")
    
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
        },
        'initial_stats_by_subset': initial_stats_by_subset,
        'initial_stats_by_type': initial_stats_by_type_final
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
