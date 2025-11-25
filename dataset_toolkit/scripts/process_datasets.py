#!/usr/bin/env python3
"""
Process and prepare ProtTeX datasets for training
"""

import json
import os
from pathlib import Path
from Bio import SeqIO
from datetime import datetime
import random
import argparse

def parse_fasta(fasta_file, max_sequences=None):
    """
    Parse FASTA file and extract protein sequences
    
    Args:
        fasta_file: Path to FASTA file
        max_sequences: Maximum number of sequences to extract (None for all)
    
    Returns:
        List of dictionaries with id, sequence, description
    """
    proteins = []
    
    print(f"Reading FASTA file: {fasta_file}")
    
    for i, record in enumerate(SeqIO.parse(fasta_file, "fasta")):
        if max_sequences and i >= max_sequences:
            break
            
        protein = {
            "id": record.id,
            "sequence": str(record.seq),
            "description": record.description,
            "length": len(record.seq)
        }
        proteins.append(protein)
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1} sequences...")
    
    print(f"✓ Extracted {len(proteins)} proteins")
    return proteins

def create_function_understanding_samples(proteins, num_samples=100):
    """
    Create PFUD (Protein Function Understanding Dataset) samples
    """
    print("\n=== Creating PFUD Samples ===")
    
    samples = []
    
    # Function prediction templates
    templates = [
        {
            "question": "Based on the given protein sequence, predict its primary molecular function:",
            "answer_prefix": "The predicted molecular function is:"
        },
        {
            "question": "What is the subcellular localization of this protein?",
            "answer_prefix": "The protein is predicted to localize in:"
        },
        {
            "question": "Describe the biological process this protein is involved in:",
            "answer_prefix": "This protein participates in:"
        },
        {
            "question": "What domains or motifs are present in this protein?",
            "answer_prefix": "The protein contains the following domains:"
        }
    ]
    
    selected_proteins = random.sample(proteins, min(num_samples, len(proteins)))
    
    for protein in selected_proteins:
        template = random.choice(templates)
        
        sample = {
            "protein_id": protein["id"],
            "sequence": protein["sequence"],
            "question": template["question"],
            "answer": f"{template['answer_prefix']} [Requires annotation from database]",
            "task": "function_understanding"
        }
        samples.append(sample)
    
    print(f"✓ Created {len(samples)} PFUD samples")
    return samples

def create_structure_prediction_samples(proteins, num_samples=100):
    """
    Create PSPD (Protein Structure Prediction Dataset) samples
    """
    print("\n=== Creating PSPD Samples ===")
    
    samples = []
    
    selected_proteins = random.sample(proteins, min(num_samples, len(proteins)))
    
    for protein in selected_proteins:
        sample = {
            "protein_id": protein["id"],
            "sequence": protein["sequence"],
            "question": f"Given the protein sequence below, predict its 3D structure:\n<protein sequence>{protein['sequence']}</protein sequence>",
            "answer": "<protein structure>[Structure tokens would go here]</protein structure>",
            "task": "structure_prediction"
        }
        samples.append(sample)
    
    print(f"✓ Created {len(samples)} PSPD samples")
    return samples

def create_design_samples(num_samples=50):
    """
    Create PDD (Protein Design Dataset) samples
    """
    print("\n=== Creating PDD Samples ===")
    
    samples = []
    
    design_prompts = [
        {
            "requirements": "Design a protein with ATP binding capability and kinase activity for cytoplasm localization",
            "properties": ["ATP binding", "kinase activity", "cytoplasm localization"]
        },
        {
            "requirements": "Create a membrane protein with ion channel activity",
            "properties": ["transmembrane domain", "ion channel", "plasma membrane"]
        },
        {
            "requirements": "Design an enzyme with oxidoreductase activity for mitochondrial localization",
            "properties": ["oxidoreductase activity", "mitochondrial targeting", "cofactor binding"]
        },
        {
            "requirements": "Generate a DNA-binding protein with transcription factor activity",
            "properties": ["DNA binding", "transcription regulation", "nuclear localization"]
        }
    ]
    
    for i in range(num_samples):
        prompt = random.choice(design_prompts)
        
        sample = {
            "design_id": f"design_{i:04d}",
            "question": f"Synthesize a protein sequence that meets the following requirements:\n{prompt['requirements']}",
            "answer": "<protein sequence>[Designed sequence]</protein sequence>\n<protein structure>[Predicted structure]</protein structure>",
            "properties": prompt["properties"],
            "task": "protein_design"
        }
        samples.append(sample)
    
    print(f"✓ Created {len(samples)} PDD samples")
    return samples

def create_structure_analysis_samples(proteins, num_samples=50):
    """
    Create PSAD (Protein Structure Analysis Dataset) samples
    """
    print("\n=== Creating PSAD Samples ===")
    
    samples = []
    
    selected_proteins = random.sample(proteins, min(num_samples, len(proteins)))
    
    for protein in selected_proteins:
        sample = {
            "protein_id": protein["id"],
            "sequence": protein["sequence"],
            "question": f"Analyze the protein sequence and describe its subunit composition and predict the overall structure:\n<protein sequence>{protein['sequence']}</protein sequence>",
            "answer": "Based on the analysis, the protein's subunit structure is: [Analysis]. The predicted structure is: <protein structure>[Structure]</protein structure>",
            "task": "structure_analysis"
        }
        samples.append(sample)
    
    print(f"✓ Created {len(samples)} PSAD samples")
    return samples

def split_dataset(data, train_ratio=0.9, val_ratio=0.05, test_ratio=0.05):
    """
    Split dataset into train/val/test sets
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    random.shuffle(data)
    n = len(data)
    
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data

def main():
    parser = argparse.ArgumentParser(description="Process ProtTeX datasets")
    parser.add_argument("--input", type=str, default="data/swiss_prot/uniprot_sprot.fasta",
                       help="Input FASTA file")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                       help="Output directory")
    parser.add_argument("--max-proteins", type=int, default=1000,
                       help="Maximum number of proteins to process")
    parser.add_argument("--pfud-samples", type=int, default=100,
                       help="Number of PFUD samples")
    parser.add_argument("--pspd-samples", type=int, default=100,
                       help="Number of PSPD samples")
    parser.add_argument("--pdd-samples", type=int, default=50,
                       help="Number of PDD samples")
    parser.add_argument("--psad-samples", type=int, default=50,
                       help="Number of PSAD samples")
    
    args = parser.parse_args()
    
    print("="*60)
    print("ProtTeX Dataset Processing")
    print("="*60)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse proteins from FASTA
    if not Path(args.input).exists():
        print(f"✗ Input file not found: {args.input}")
        print("Please run prepare_dataset.py first to download the data")
        return
    
    proteins = parse_fasta(args.input, max_sequences=args.max_proteins)
    
    # Create datasets
    all_samples = []
    
    # PFUD - Function Understanding
    pfud_samples = create_function_understanding_samples(proteins, args.pfud_samples)
    all_samples.extend(pfud_samples)
    
    # PSPD - Structure Prediction
    pspd_samples = create_structure_prediction_samples(proteins, args.pspd_samples)
    all_samples.extend(pspd_samples)
    
    # PDD - Protein Design
    pdd_samples = create_design_samples(args.pdd_samples)
    all_samples.extend(pdd_samples)
    
    # PSAD - Structure Analysis
    psad_samples = create_structure_analysis_samples(proteins, args.psad_samples)
    all_samples.extend(psad_samples)
    
    # Split into train/val/test
    print("\n=== Splitting Dataset ===")
    train_data, val_data, test_data = split_dataset(all_samples)
    
    print(f"Train: {len(train_data)} samples ({len(train_data)/len(all_samples)*100:.1f}%)")
    print(f"Val: {len(val_data)} samples ({len(val_data)/len(all_samples)*100:.1f}%)")
    print(f"Test: {len(test_data)} samples ({len(test_data)/len(all_samples)*100:.1f}%)")
    
    # Save datasets
    print("\n=== Saving Datasets ===")
    
    train_file = output_dir / "train.json"
    val_file = output_dir / "val.json"
    test_file = output_dir / "test.json"
    
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"✓ Saved training data: {train_file}")
    
    with open(val_file, 'w') as f:
        json.dump(val_data, f, indent=2)
    print(f"✓ Saved validation data: {val_file}")
    
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"✓ Saved test data: {test_file}")
    
    # Create statistics
    stats = {
        "total_samples": len(all_samples),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
        "task_distribution": {
            "PFUD": len(pfud_samples),
            "PSPD": len(pspd_samples),
            "PDD": len(pdd_samples),
            "PSAD": len(psad_samples)
        },
        "created_at": datetime.now().isoformat()
    }
    
    stats_file = output_dir / "statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved statistics: {stats_file}")
    
    print("\n" + "="*60)
    print("✓ Dataset processing complete!")
    print("="*60)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print("\n📊 Summary:")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  PFUD (Function): {len(pfud_samples)}")
    print(f"  PSPD (Structure): {len(pspd_samples)}")
    print(f"  PDD (Design): {len(pdd_samples)}")
    print(f"  PSAD (Analysis): {len(psad_samples)}")

if __name__ == "__main__":
    main()












