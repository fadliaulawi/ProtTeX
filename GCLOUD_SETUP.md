# Google Cloud SDK Setup for AlphaFold Bulk Download

The full AlphaFold Swiss-Prot dataset is hosted on Google Cloud. To download it efficiently, you need `gsutil`.

## Quick Setup (Recommended)

Install Google Cloud SDK to a custom directory (avoids home directory quota issues):

```bash
# Choose your install directory
INSTALL_DIR="/path/to/your/project"
cd $INSTALL_DIR

# Download and extract
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xf google-cloud-cli-linux-x86_64.tar.gz

# Install (non-interactive)
./google-cloud-sdk/install.sh --quiet --path-update=false

# Verify gsutil works
./google-cloud-sdk/bin/gsutil version
```

## Update download_pspd.py

After installation, update the `GSUTIL` path in `download_pspd.py`:

```python
# Line 15 in download_pspd.py
GSUTIL = "/path/to/your/project/google-cloud-sdk/bin/gsutil"
```

## Alternative: pip install (simpler but may have issues)

```bash
pip install gsutil
```

Note: The pip version may have authentication issues with some datasets.

## Verify Setup

```bash
# Test listing AlphaFold files
/path/to/google-cloud-sdk/bin/gsutil ls gs://public-datasets-deepmind-alphafold-v4/proteomes/ | head -5
```

## Without gsutil

If you can't install gsutil, the script will automatically fall back to HTTP downloads (slower, limited to ~500 structures).

## Usage

```bash
# Test mode (10 samples each)
python download_pspd.py --mode test

# Full download
python download_pspd.py --mode full
```

