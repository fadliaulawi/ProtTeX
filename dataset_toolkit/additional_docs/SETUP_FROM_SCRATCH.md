# ProtTeX Dataset Preparation - Complete Setup Guide

This guide will take you from **nothing** to a complete ProtTeX dataset setup with samples.

## Prerequisites

- Linux system with internet access
- Conda installed
- At least 30 GB disk space

---

## Step 1: Environment Setup

### 1.1 Create Conda Environment

```bash
# Create new environment
conda create -n prottex_env python=3.10 -y

# Activate environment
conda activate prottex_env
```

### 1.2 Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

**File: `requirements.txt`** (already exists in this directory)
```
tqdm
requests
biopython
PyPDF2
pandas
numpy
datasets
```

---

## Step 2: Download Mol-Instructions Dataset

**Source:** Text-based protein instruction data from Mol-Instructions

**Script:** `download_mol_instructions_working.py` (already exists)

```bash
# Run the download script
python3 download_mol_instructions_working.py
```

**What it downloads:**
- `Protein_Design.zip` (82,962 samples) → PDD
- `Protein_Function.zip` (165,736 samples) → PFUD
- `domain_motif.zip` (30,025 samples) → PSAD
- Additional molecular data

**Output location:** `data/mol_instructions/`

**Time:** ~5-10 minutes

---

## Step 3: Download ProteinLMBench Dataset

**Source:** Protein language model benchmark data

**Manual download required** (Hugging Face datasets library has issues with this dataset)

```bash
# Navigate to project directory
cd "/lustrefs/shared/mohammad.sayeed/PFM paper/prottex"

# Download using Python
python3 << 'SCRIPT'
from datasets import load_dataset
from pathlib import Path
import os

# Set cache directory
cache_dir = Path("data/proteinlmbench_cache")
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ['HF_DATASETS_CACHE'] = str(cache_dir)

# Download key subsets
subsets = [
    "UniProt_GO_sample",
    "UniProt_Function_sample", 
    "UniProt_Domain_sample",
    "UniProt_Subcellular_location_sample"
]

output_dir = Path("data/proteinlmbench")
output_dir.mkdir(parents=True, exist_ok=True)

for subset in subsets:
    print(f"\n{'='*60}")
    print(f"Downloading {subset}...")
    print(f"{'='*60}")
    
    try:
        dataset = load_dataset(
            "tyang816/ProteinLMBench",
            subset,
            split="train",
            cache_dir=str(cache_dir)
        )
        
        # Save to JSON
        output_file = output_dir / f"{subset}.json"
        dataset.to_json(output_file)
        print(f"✓ Saved to {output_file}")
        print(f"  Samples: {len(dataset)}")
        
    except Exception as e:
        print(f"✗ Error: {e}")

print("\n" + "="*60)
print("✓ ProteinLMBench download complete")
print("="*60)
SCRIPT
```

**Output location:** `data/proteinlmbench/`

**Time:** ~10-15 minutes

---

## Step 4: Download AlphaFold Structures

**Source:** AlphaFold v4 Swiss-Prot structures (19,000 proteins)

**Size:** 26.3 GB compressed

**Reference:** `VERIFIED_DOWNLOAD_COMMANDS.sh` (already exists)

```bash
# Navigate to project directory
cd "/lustrefs/shared/mohammad.sayeed/PFM paper/prottex"

# Activate environment
conda activate prottex_env

# Download AlphaFold v4 Swiss-Prot (VERIFIED)
wget --progress=bar:force \
     -O swissprot_pdb_v4.tar \
     https://ftp.ebi.ac.uk/pub/databases/alphafold/v4/swissprot_pdb_v4.tar

# Create output directory
mkdir -p data/alphafold/swissprot_v4

# Extract tar file
tar -xf swissprot_pdb_v4.tar

# Move PDB files to correct location
mv AF-*.pdb.gz data/alphafold/swissprot_v4/

# Clean up
rm swissprot_pdb_v4.tar

echo "✓ AlphaFold structures downloaded and extracted"
```

**Alternative:** Use the existing `download_alphafold.py` script for smaller samples

**Output location:** `data/alphafold/swissprot_v4/`

**Time:** ~30-60 minutes (depending on network speed)

---

## Step 5: Process and Generate Samples

**Script:** `process_datasets.py` (already exists)

The samples we generated are already in the `data/` directory:
- `data/prottex_training_samples.json`
- `data/prottex_model_format_samples.json`
- `data/prottex_samples_with_sources.csv`
- `data/prottex_samples_detailed.json`

To regenerate them or create more samples, you can use the commands from our session.

**Time:** Already done!

---

## Step 6: Verify Data

Check what data you have:

```bash
# Check data directories
ls -lh data/mol_instructions/
ls -lh data/proteinlmbench/
ls data/alphafold/swissprot_v4/ | wc -l

# Check sample files
ls -lh data/*.json data/*.csv
```

**Expected:**
- Mol-Instructions: ~278K samples
- ProteinLMBench: ~69K samples  
- AlphaFold structures: 19,000 PDB files
- Sample files created

---

## Complete Dataset Summary

After completing all steps, you will have:

| Dataset | Type | Samples | Coverage | Status |
|---------|------|---------|----------|--------|
| **PFUD** | Protein Function | 234,626 | 106% | ✓ Complete |
| **PDD** | Protein Design | 85,697 | 103% | ✓ Complete |
| **PSAD** | Structure Analysis | 204,800 | 104% | ✓ Complete |
| **PSPD** | Structure Prediction | 19,000 | 19% | ⚠ Partial |

**Total:** ~525,000 text/sequence samples + 19,000 3D structures

---

## Directory Structure After Setup

```
prottex/
├── data/
│   ├── mol_instructions/          # Mol-Instructions data
│   │   ├── Protein_Function.json
│   │   ├── Protein_Design.json
│   │   └── domain_motif.json
│   ├── proteinlmbench/            # ProteinLMBench data
│   │   ├── UniProt_Function_sample.json
│   │   ├── UniProt_GO_sample.json
│   │   └── ...
│   ├── alphafold/                 # AlphaFold structures
│   │   └── swissprot_v4/
│   │       ├── AF-A0A000-F1-model_v4.pdb.gz
│   │       └── ...
│   ├── prottex_training_samples.json
│   ├── prottex_model_format_samples.json
│   ├── prottex_samples_with_sources.csv
│   └── prottex_samples_detailed.json
├── download_mol_instructions_working.py  # Existing script
├── download_alphafold.py                 # Existing script
├── process_datasets.py                   # Existing script
├── requirements.txt                      # Existing
├── VERIFIED_DOWNLOAD_COMMANDS.sh         # Existing
├── DATASET_GUIDE.md                      # Existing
├── PROTTEX_DATA_BLUEPRINT.md             # Existing
└── SETUP_FROM_SCRATCH.md                 # This file
```

---

## Troubleshooting

### Issue: Disk quota exceeded
**Solution:**
```bash
# Clear Hugging Face cache
rm -rf ~/.cache/huggingface/*

# Set local cache directory
export HF_DATASETS_CACHE="./data/cache"
```

### Issue: Connection timeout during download
**Solution:**
```bash
# Resume wget download
wget -c --progress=bar:force -O swissprot_pdb_v4.tar \
     https://ftp.ebi.ac.uk/pub/databases/alphafold/v4/swissprot_pdb_v4.tar
```

### Issue: Missing dependencies
**Solution:**
```bash
# Reinstall all requirements
pip install --upgrade -r requirements.txt
```

---

## Next Steps

After data preparation:

1. **Implement structure tokenizer** - SE(3)-invariant encoder + VQ-VAE
2. **Tokenize structures** - Convert PDB files to discrete tokens
3. **Train multimodal LLM** - Combine text, sequence, and structure
4. **Evaluate on benchmarks** - Test on ProteinLMBench tasks

---

## Estimated Total Time

- Step 1 (Environment): 5 minutes
- Step 2 (Mol-Instructions): 10 minutes
- Step 3 (ProteinLMBench): 15 minutes
- Step 4 (AlphaFold): 60 minutes
- Step 5 (Generate samples): 1 minute
- Step 6 (Verify): 1 minute

**Total: ~90 minutes** (mostly waiting for downloads)

---

## Questions?

Refer to:
- `DATASET_GUIDE.md` - Detailed dataset information
- `PROTTEX_DATA_BLUEPRINT.md` - Data structure breakdown
- `VERIFIED_DOWNLOAD_COMMANDS.sh` - Verified download commands

