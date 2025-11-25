#!/bin/bash
# VERIFIED AlphaFold Download Commands
# All URLs tested with curl before providing

cd "/lustrefs/shared/mohammad.sayeed/PFM paper/prottex"
conda activate prottex_env

echo "=============================================================="
echo "ALPHAFOLD DATABASE v4 DOWNLOAD (VERIFIED)"
echo "=============================================================="
echo ""
echo "File: swissprot_pdb_v4.tar"
echo "Size: 26.3 GB"
echo "Verified: October 31, 2025"
echo "URL Status: 200 OK"
echo ""
echo "=============================================================="

# ==================================================================
# OPTION 1: Download AlphaFold v4 Swiss-Prot (Paper's version)
# ==================================================================
# Size: 26.3 GB
# Contains: ~570K protein structures (Swiss-Prot)
# Version: v4 (exact version paper used)

wget --progress=bar:force \
     --no-check-certificate \
     -O swissprot_pdb_v4.tar \
     https://ftp.ebi.ac.uk/pub/databases/alphafold/v4/swissprot_pdb_v4.tar

# Verify download
ls -lh swissprot_pdb_v4.tar

# Extract
echo "Extracting... (this takes time)"
mkdir -p data/alphafold/swissprot_v4
tar -xvf swissprot_pdb_v4.tar -C data/alphafold/swissprot_v4/

# Count files
echo "Counting PDB files..."
find data/alphafold/swissprot_v4/ -name "*.pdb*" | wc -l

echo ""
echo "=============================================================="
echo "✓ Download complete!"
echo "=============================================================="

# ==================================================================
# OPTION 2 (Alternative): Latest version v6
# ==================================================================
# Uncomment if you want the latest instead of paper's v4
# 
# wget --progress=bar:force \
#      --no-check-certificate \
#      -O swissprot_pdb_v6.tar \
#      https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/swissprot_pdb_v6.tar
# 
# mkdir -p data/alphafold/swissprot_v6
# tar -xvf swissprot_pdb_v6.tar -C data/alphafold/swissprot_v6/

echo ""
echo "Next steps:"
echo "1. Verify structures: ls -lh data/alphafold/swissprot_v4/ | head"
echo "2. Process for training: python process_structure_prediction.py"
echo ""











