# ProtTeX Complete Data Blueprint

## 📋 EXACTLY WHAT PROTTEX USED (From Paper Section 2.3.1)

### Total Training Data: 3,707,426 samples across 4 datasets

---

## 🗂️ THE 4 DATASETS

### 1. PFUD - Protein Function Understanding Dataset
- **Samples:** 429,201
- **Token count:** 320.4M tokens
- **Sources:** 
  - ✅ Mol-Instructions (protein function, general function, catalytic activity)
  - ❌ ProteinLMBench (additional function QA pairs)
- **What it contains:** Predict molecular function, subcellular location, biological process, domains/motifs

### 2. PDD - Protein Design Dataset  
- **Samples:** 192,617
- **Token count:** 146.8M tokens
- **Source:** 
  - ✅ Mol-Instructions (protein_design.json)
- **What it contains:** Design proteins with specific properties

### 3. PSAD - Protein Structure Analysis Dataset
- **Samples:** 264,370
- **Token count:** 205.0M tokens
- **Sources:**
  - ✅ Mol-Instructions (domain_motif.json) - partial
  - ❌ ProteinLMBench (structure analysis QA pairs)
- **What it contains:** Analyze subunit composition, predict structure from sequence

### 4. PSPD - Protein Structure Prediction Dataset
- **Samples:** 2,821,238  
- **Token count:** 1,787.8M tokens
- **Source:**
  - ❌ Their custom database (3.36M proteins WITHOUT QA pairs)
- **What it contains:** Sequence → structure prediction for proteins without existing QA

---

## 🔍 THE UNDERLYING PROTEIN DATABASE (3.36M proteins)

### Required Sources:

#### 1. AlphaFold Protein Structure Database (AFDB) v4 - CLUSTERED
- **What:** Clustered at 50% sequence identity
- **When:** Released before July 25, 2022
- **Size:** ~100GB+ compressed
- **Status:** ✅ AVAILABLE

#### 2. Swiss-Prot (May 2022 release)
- **What:** Reviewed protein sequences from UniProt
- **When:** May 2022 release
- **Size:** ~270 MB
- **Status:** ✅ AVAILABLE (we have current version, need May 2022)

#### 3. RCSB PDB
- **What:** Experimental protein structures
- **When:** Released before July 25, 2022
- **Size:** Varies
- **Status:** ✅ AVAILABLE

---

## ✅ WHAT YOU CAN GET

### Available Now - Run These Commands:

```bash
# Already done:
# ✅ Mol-Instructions downloaded (495K samples)
# ✅ Swiss-Prot downloaded (current version, 573K proteins)
# ✅ Sample PDB structures (4 proteins)

# ==================================================================
# STEP 1: Download AlphaFold Database Clustered (Swiss-Prot subset)
# ==================================================================
# This gives you ~570K protein structures
# Size: ~20GB compressed

cd /lustrefs/shared/mohammad.sayeed/PFM\ paper/prottex

# Option A: Swiss-Prot subset (recommended to start)
wget https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/swissprot_pdb_v4.tar

# Extract
mkdir -p data/alphafold/swissprot
cd data/alphafold/swissprot
tar -xvf ../../../swissprot_pdb_v4.tar

cd /lustrefs/shared/mohammad.sayeed/PFM\ paper/prottex

# ==================================================================
# STEP 2: Download AlphaFold Clustered Database (Full - Optional)
# ==================================================================
# This is the CLUSTERED version at 50% identity
# Size: ~100GB compressed, several hundred GB uncompressed
# WARNING: This is LARGE and takes hours/days

# First, check available clustered versions:
curl -s https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/ | grep -i "cluster"

# Download clustered AF at 50% (if available):
# Note: You may need to check the FTP site for exact filename
# The paper mentions "clustered AlphaFold Protein Structure Database v4"

# ==================================================================  
# STEP 3: Download Swiss-Prot May 2022 Archive (Historical)
# ==================================================================
# Get the exact version they used

wget https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2022_02/knowledgebase/uniprot_sprot-only2022_02.tar.gz

mkdir -p data/swiss_prot_2022_may
tar -xzf uniprot_sprot-only2022_02.tar.gz -C data/swiss_prot_2022_may/

# ==================================================================
# STEP 4: Download PDB structures (filtered by date)
# ==================================================================
# Get PDB structures released before July 25, 2022

# This requires using PDB's rsync service with date filtering
# WARNING: Full PDB is ~1TB

# For specific entries, create a list and download:
# mkdir -p data/pdb_filtered
# Use PDB's batch download service at:
# https://www.rcsb.org/downloads

```

---

## ❌ WHAT YOU CANNOT GET

### 1. ProteinLMBench Dataset (~400K QA pairs)
- **Status:** NOT PUBLICLY AVAILABLE
- **Why:** 
  - Published in BIBM 2024 (recent)
  - No public data release yet
  - No GitHub repository with data
- **Impact:** Missing ~175K PFUD samples, ~219K PSAD samples
- **Workaround:** Use only Mol-Instructions (59% of PFUD, 17% of PSAD)

### 2. Exact AFDB v4 Clustered Version
- **Status:** MAY NOT BE AVAILABLE
- **Why:**
  - Paper doesn't specify exact clustering parameters
  - AFDB versions keep updating
  - "Clustered at 50%" - specific version unclear
- **Impact:** Cannot recreate exact PSPD dataset
- **Workaround:** Use current AFDB clustered or Swiss-Prot subset

### 3. ProtTeX Tokenizer
- **Status:** NOT RELEASED
- **Why:**
  - Code not open-sourced
  - Model weights not released
  - Tokenizer implementation proprietary
- **What it does:**
  - SE(3)-invariant encoder (modified AlphaFold EvoFormer)
  - Vector quantization with 512 codes
  - Converts structure to discrete tokens
- **Impact:** Cannot process structures into ProtTeX format
- **Workaround:** Implement your own tokenizer

### 4. Pre-processed ProtTeX Tokens
- **Status:** NOT AVAILABLE
- **Why:** Authors didn't release pre-tokenized data
- **Impact:** You need to implement tokenizer first
- **Workaround:** Use sequence-only or implement structure encoder

### 5. Exact 3.36M Protein Database
- **Status:** PARTIALLY AVAILABLE
- **Why:**
  - Exact selection criteria unclear
  - Specific date filtering needed (before July 25, 2022)
  - Custom curation and filtering applied
- **Impact:** Cannot match exact PSPD dataset
- **Workaround:** Use available subsets

---

## 📊 REALISTIC DATA COVERAGE

### What You Can Actually Get:

| Dataset | Paper | Available | Coverage | Source |
|---------|-------|-----------|----------|--------|
| **PDD** | 192,617 | 195,975 | 101% ✅ | Mol-Instructions |
| **PFUD** | 429,201 | 253,929 | 59% ⚠️ | Mol-Instructions (missing ProteinLMBench) |
| **PSAD** | 264,370 | 45,100 | 17% ⚠️ | Mol-Instructions (missing ProteinLMBench) |
| **PSPD** | 2,821,238 | ~570,000 | 20% ⚠️ | Swiss-Prot + AlphaFold |
| **TOTAL** | 3,707,426 | ~1,065,004 | 29% | |

### With AlphaFold Swiss-Prot:
- **Total samples:** ~1,065,004
- **Coverage:** 29% of paper's data
- **Good enough for:** Proof of concept, architecture validation
- **Not enough for:** Exact performance replication

---

## 🎯 RECOMMENDED ACTION PLAN

### Phase 1: Get What's Available NOW (Today)

```bash
# 1. Download AlphaFold Swiss-Prot structures (~20GB, 570K proteins)
cd /lustrefs/shared/mohammad.sayeed/PFM\ paper/prottex
wget https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/swissprot_pdb_v4.tar
mkdir -p data/alphafold/swissprot
tar -xvf swissprot_pdb_v4.tar -C data/alphafold/swissprot/

# 2. Verify what we have
python3 << 'EOF'
from pathlib import Path
af_dir = Path("data/alphafold/swissprot")
pdb_files = list(af_dir.glob("**/*.pdb")) + list(af_dir.glob("**/*.pdb.gz"))
print(f"AlphaFold structures: {len(pdb_files):,}")
EOF
```

### Phase 2: Process and Create PSPD (After Phase 1)

```bash
# Create structure prediction dataset from Swiss-Prot + AlphaFold
python process_structure_prediction.py
# This creates PSPD-like samples: sequence → structure
```

### Phase 3: Optional - Get More Data (If needed)

```bash
# Download full AlphaFold clustered (WARNING: 100GB+)
# Only do this if you need more than 570K proteins
wget https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/[clustered_file]
```

---

## 🚫 IMPOSSIBLE ITEMS - DO NOT WASTE TIME

1. ❌ ProteinLMBench dataset → Wait for authors to release or contact them
2. ❌ ProtTeX tokenizer code → Implement your own
3. ❌ Exact 3.36M protein selection → Use available subsets
4. ❌ Pre-tokenized structures → Process yourself after implementing tokenizer

---

## 💡 BOTTOM LINE

### You CAN replicate:
- ✅ 29% of training data (~1M samples)
- ✅ Core architecture (with implementation effort)
- ✅ Training approach
- ✅ Evaluation methodology

### You CANNOT replicate:
- ❌ Exact dataset (missing ProteinLMBench)
- ❌ Full 3.7M samples
- ❌ Exact model performance (less data = lower performance)

### This is enough to:
- ✅ Validate the approach works
- ✅ Publish research with proper attribution
- ✅ Build a working proof-of-concept
- ✅ Scale up later when more data available

---

## 📞 NEXT STEPS FOR MISSING DATA

### For ProteinLMBench:
Contact paper authors:
- Yuchen Shen (first author)
- Email from paper references: Not provided directly
- Try: Search for "Yuchen Shen protein" + affiliation from paper

### For ProtTeX Code:
Watch these:
- https://github.com/search?q=ProtTeX
- https://arxiv.org/abs/2503.08179 (check for code release updates)

### Join Community:
- Reddit: r/bioinformatics, r/MachineLearning
- Twitter: Search #ProtTeX, #ProteinLM
- Papers with Code: Check if code gets added

---

## ✅ START HERE - COMMANDS TO RUN NOW

```bash
# Navigate to project
cd "/lustrefs/shared/mohammad.sayeed/PFM paper/prottex"

# Activate environment  
conda activate prottex_env

# Download AlphaFold Swiss-Prot structures (THE BIG ONE)
# This takes ~1-2 hours depending on connection
wget https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/swissprot_pdb_v4.tar

# Create directory
mkdir -p data/alphafold/swissprot

# Extract (this takes time)
tar -xvf swissprot_pdb_v4.tar -C data/alphafold/swissprot/

# Verify
ls -lh data/alphafold/swissprot/ | head -20
```

After this completes, you'll have ~29% of their data and can start building the model.











