# Clustered AlphaFold Database Download Script

This script downloads the **Clustered AlphaFold Database** structures as used by the ProtTeX paper.

## Background

ProtTeX uses the **Barrio-Hernandez et al. clustering** of AFDB v4:
- **Paper**: "Clustering predicted structures at the scale of the known protein universe"
- **Source**: 2.27M cluster representatives from AFDB v4 clustered at 50% sequence identity
- **Clustering method**: Foldseek with TM-score threshold

## What This Script Downloads

| Mode | Description | Size |
|------|-------------|------|
| `test` | 10 sample structures | ~1 MB |
| `full` | All 2.27M cluster representatives | ~500 GB |

## Usage

### Test Mode (Verify Setup)

```bash
python download_clustered_afdb.py --mode test --output_dir ./afdb_clustered_test
```

### Full Mode (All Structures)

```bash
# Using HTTP (slower, works everywhere)
python download_clustered_afdb.py --mode full --output_dir ./afdb_clustered_full

# Using gsutil (faster, requires Google Cloud SDK)
python download_clustered_afdb.py --mode full --output_dir ./afdb_clustered_full \
    --gsutil_path /path/to/gsutil
```

## Important Notes

### Version Difference
- ProtTeX used **AFDB v4**
- This script downloads **AFDB v6** (current version, v4 is archived)
- Structures are similar but not identical

### pLDDT Scores
pLDDT (confidence) scores are stored in the B-factor column of PDB files:

```bash
# Extract pLDDT scores from a PDB file
grep "^ATOM" structure.pdb | awk '{print $11}'
```

### TM-Score
TM-scores are NOT included in PDB files. To compute TM-scores between structures:

```bash
# Install TMalign
wget https://zhanggroup.org/TM-align/TMalign.gz
gunzip TMalign.gz && chmod +x TMalign

# Compare two structures
./TMalign structure1.pdb structure2.pdb
```

### Filtering (as done by ProtTeX)
The cluster representatives are **already filtered** - they were selected based on:
- Structural similarity (TM-score clustering)
- Sequence identity (50% threshold)

Additional filtering you can apply:
- pLDDT > 70 (high confidence residues)
- Average pLDDT > 80 (high confidence structures)

## Output Structure

```
afdb_clustered_full/
├── cluster_reps_v4.fasta.gz    # Cluster representative sequences
├── download_manifest.json       # Download metadata
└── structures/                  # Downloaded PDB files
    ├── AF-A0A1W2PQ64-F1.pdb
    ├── AF-Q9Y6K9-F1.pdb
    └── ...
```

## Requirements

```bash
pip install requests
```

For faster downloads:
```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
gcloud auth login
```

## Reference

- ProtTeX Paper: Structure-aware protein language model
- Barrio-Hernandez et al.: Clustering predicted structures at the scale of the known protein universe
- AlphaFold Database: https://alphafold.ebi.ac.uk/

