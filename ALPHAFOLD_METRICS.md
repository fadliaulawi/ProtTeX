# AlphaFold Metrics: pTM and pLDDT

## Overview

AlphaFold provides confidence metrics for each predicted structure:
- **pTM (predicted TM-score)**: Global structure confidence (domain arrangement)
- **pLDDT**: Per-residue local confidence

## API Documentation

Official AlphaFold API docs: https://www.alphafold.ebi.ac.uk/api-docs

> `globalMetricValue` - The predicted Template Modeling score (**pTM**). This metric assesses the confidence in the predicted global arrangement of protein domains. Ranges from 0-100.

## Fetch Metrics for a Single Protein

```bash
# Get ALL metrics (pTM, pLDDT fractions, URLs) for a UniProt ID
curl -o P00520_metrics.json "https://alphafold.ebi.ac.uk/api/prediction/P00520"

# View key metrics
cat P00520_metrics.json | python3 -c "import sys,json; d=json.load(sys.stdin)[0]; print(f'pTM: {d[\"globalMetricValue\"]}, Avg pLDDT breakdown: VeryHigh={d[\"fractionPlddtVeryHigh\"]:.1%}, Confident={d[\"fractionPlddtConfident\"]:.1%}, Low={d[\"fractionPlddtLow\"]:.1%}, VeryLow={d[\"fractionPlddtVeryLow\"]:.1%}')"
```

## Fetch Metrics for Multiple Proteins (Batch)

```bash
#!/bin/bash
# Save as: fetch_alphafold_metrics.sh
# Usage: ./fetch_alphafold_metrics.sh uniprot_ids.txt output_dir

INPUT_FILE=${1:-"uniprot_ids.txt"}
OUTPUT_DIR=${2:-"alphafold_metrics"}

mkdir -p "$OUTPUT_DIR"

while read -r UNIPROT_ID; do
    if [ -n "$UNIPROT_ID" ]; then
        echo "Fetching $UNIPROT_ID..."
        curl -s -o "$OUTPUT_DIR/${UNIPROT_ID}_metrics.json" \
            "https://alphafold.ebi.ac.uk/api/prediction/${UNIPROT_ID}"
    fi
done < "$INPUT_FILE"

echo "Done! Metrics saved to $OUTPUT_DIR/"
```

## Python Script for Batch Download with Summary

```python
#!/usr/bin/env python3
"""
Fetch AlphaFold metrics (pTM, pLDDT) for a list of UniProt IDs.
Usage: python fetch_alphafold_metrics.py uniprot_ids.txt output_dir
"""

import requests
import json
import sys
import csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_metrics(uniprot_id):
    """Fetch metrics for a single UniProt ID"""
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    try:
        response = requests.get(url, timeout=30)
        if response.ok:
            data = response.json()
            if data:
                entry = data[0]
                return {
                    'uniprot_id': uniprot_id,
                    'pTM': entry.get('globalMetricValue'),
                    'pLDDT_very_high': entry.get('fractionPlddtVeryHigh'),
                    'pLDDT_confident': entry.get('fractionPlddtConfident'),
                    'pLDDT_low': entry.get('fractionPlddtLow'),
                    'pLDDT_very_low': entry.get('fractionPlddtVeryLow'),
                    'sequence_length': entry.get('uniprotEnd'),
                    'pdb_url': entry.get('pdbUrl'),
                    'pae_url': entry.get('paeDocUrl'),
                    'plddt_url': entry.get('plddtDocUrl'),
                }
    except Exception as e:
        print(f"Error fetching {uniprot_id}: {e}")
    return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_alphafold_metrics.py uniprot_ids.txt [output_dir]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("alphafold_metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read UniProt IDs
    with open(input_file) as f:
        uniprot_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Fetching metrics for {len(uniprot_ids)} proteins...")
    
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_metrics, uid): uid for uid in uniprot_ids}
        for future in as_completed(futures):
            uid = futures[future]
            result = future.result()
            if result:
                results.append(result)
                # Save individual JSON
                with open(output_dir / f"{uid}_metrics.json", 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"✓ {uid}: pTM={result['pTM']:.2f}")
            else:
                print(f"✗ {uid}: Failed")
    
    # Save summary CSV
    if results:
        csv_file = output_dir / "metrics_summary.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSummary saved to {csv_file}")
        
        # Print statistics
        ptm_values = [r['pTM'] for r in results if r['pTM']]
        if ptm_values:
            print(f"\npTM Statistics:")
            print(f"  Mean: {sum(ptm_values)/len(ptm_values):.2f}")
            print(f"  Min:  {min(ptm_values):.2f}")
            print(f"  Max:  {max(ptm_values):.2f}")

if __name__ == "__main__":
    main()
```

## Available Files per Protein

| File | URL Pattern | Description |
|------|-------------|-------------|
| API (all metrics) | `https://alphafold.ebi.ac.uk/api/prediction/{UNIPROT_ID}` | JSON with pTM, pLDDT fractions, all URLs |
| PDB structure | `https://alphafold.ebi.ac.uk/files/AF-{UNIPROT_ID}-F1-model_v6.pdb` | 3D coordinates |
| PAE JSON | `https://alphafold.ebi.ac.uk/files/AF-{UNIPROT_ID}-F1-predicted_aligned_error_v6.json` | Predicted aligned error matrix |
| pLDDT JSON | `https://alphafold.ebi.ac.uk/files/AF-{UNIPROT_ID}-F1-confidence_v6.json` | Per-residue confidence |
| CIF structure | `https://alphafold.ebi.ac.uk/files/AF-{UNIPROT_ID}-F1-model_v6.cif` | mmCIF format |

## Metric Definitions

| Field in API | Metric | Description |
|--------------|--------|-------------|
| `globalMetricValue` | **pTM** | Global structure confidence (0-100) |
| `fractionPlddtVeryHigh` | pLDDT >90 | Fraction of residues with very high confidence |
| `fractionPlddtConfident` | pLDDT 70-90 | Fraction with confident prediction |
| `fractionPlddtLow` | pLDDT 50-70 | Fraction with low confidence |
| `fractionPlddtVeryLow` | pLDDT <50 | Fraction with very low confidence (likely disordered) |

## Example Output

```json
{
  "uniprot_id": "P00520",
  "pTM": 63.44,
  "pLDDT_very_high": 0.374,
  "pLDDT_confident": 0.092,
  "pLDDT_low": 0.044,
  "pLDDT_very_low": 0.491,
  "sequence_length": 1123,
  "pdb_url": "https://alphafold.ebi.ac.uk/files/AF-P00520-F1-model_v6.pdb",
  "pae_url": "https://alphafold.ebi.ac.uk/files/AF-P00520-F1-predicted_aligned_error_v6.json",
  "plddt_url": "https://alphafold.ebi.ac.uk/files/AF-P00520-F1-confidence_v6.json"
}
```

## Quality Filtering (as per ProtTeX paper)

The ProtTeX paper filters structures with TM-score > 0.9. For AlphaFold predictions:
- **pTM > 90**: Very high confidence, likely accurate global fold
- **pTM 70-90**: Confident prediction
- **pTM 50-70**: Lower confidence
- **pTM < 50**: Low confidence, may have incorrect fold

```python
# Filter high-quality structures
HIGH_QUALITY_PTM_THRESHOLD = 70  # Adjust as needed

high_quality = [r for r in results if r['pTM'] and r['pTM'] > HIGH_QUALITY_PTM_THRESHOLD]
print(f"High quality structures (pTM > {HIGH_QUALITY_PTM_THRESHOLD}): {len(high_quality)}")
```

