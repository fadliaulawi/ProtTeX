# Protein Structure Tokenizer - Usage Guide

## Overview

The `ProteinStructureTokenizer` class provides a complete solution for tokenizing protein sequences into multimodal token streams combining:
- **Character-level amino acid tokens** (20 standard amino acids)
- **Structure tokens** (512 learned from ESM-2 embeddings via k-means)

## Quick Start

### Running the Demo

```bash
# Activate environment
conda activate esm_tokenizer

# Run demo (requires GPU)
cd /lustrefs/shared/mohammad.sayeed/Prot2Text/esmfold_tokenizer
python demo_tokenizer_class.py
```

### Using in Your Code

```python
from demo_tokenizer_class import ProteinStructureTokenizer

# Initialize
tokenizer = ProteinStructureTokenizer(
    codebook_path="data/structure_codebook_K512.pkl",
    esm_model_name="facebook/esm2_t33_650M_UR50D"
)

# Tokenize a sequence
sequence = "MKTAYIAKQRQISFVKSHFSRQLE"
result = tokenizer.encode(sequence)

print(f"Tokens: {result['tokens']}")
print(f"Length: {len(result['tokens'])}")

# Decode back to sequence
decoded = tokenizer.decode(result['tokens'])
print(f"Decoded: {decoded}")
```

## Token Space

The tokenizer uses a unified vocabulary of **536 tokens**:

| Range | Description | Count |
|-------|-------------|-------|
| 0-19 | Amino acid tokens (A-Y) | 20 |
| 20-531 | Structure tokens (k-means centroids) | 512 |
| 532 | PAD token | 1 |
| 533 | UNK token | 1 |
| 534 | BOS token (begin of sequence) | 1 |
| 535 | EOS token (end of sequence) | 1 |
| **Total** | | **536** |

## Token Stream Format

The tokenizer produces an **interleaved** token stream:

```
[BOS, aa_1, struct_1, aa_2, struct_2, aa_3, struct_3, ..., aa_N, struct_N, EOS]
```

### Example

For sequence "MKTA":

```
[534, 12, 245, 10, 387, 19, 102, 0, 421, 535]
 BOS  M  s_M   K  s_K   T  s_T  A  s_A  EOS
```

Where:
- `12, 10, 19, 0` = amino acid tokens for M, K, T, A
- `245, 387, 102, 421` = structure tokens (learned from ESM-2)

## API Reference

### `__init__(codebook_path, esm_model_name, device)`

Initialize the tokenizer.

**Parameters:**
- `codebook_path` (str): Path to trained codebook (.pkl file)
- `esm_model_name` (str): ESM-2 model from HuggingFace (default: "facebook/esm2_t33_650M_UR50D")
- `device` (str): 'cuda' or 'cpu' (auto-detected if None)

**Example:**
```python
tokenizer = ProteinStructureTokenizer(
    codebook_path="data/structure_codebook_K512.pkl"
)
```

---

### `encode(sequence, add_special_tokens, return_structure_tokens)`

Encode a protein sequence into tokens.

**Parameters:**
- `sequence` (str): Amino acid sequence (e.g., "MKTAYIAKQR")
- `add_special_tokens` (bool): Add BOS/EOS tokens (default: True)
- `return_structure_tokens` (bool): Return structure tokens separately (default: False)

**Returns:**
Dictionary with:
- `'tokens'`: Interleaved token list
- `'aa_tokens'`: Amino acid tokens only
- `'structure_tokens'`: Structure tokens (if requested)
- `'sequence'`: Original sequence
- `'length'`: Sequence length

**Example:**
```python
result = tokenizer.encode("MKTAYIAKQR", return_structure_tokens=True)
print(result['tokens'])  # [534, 12, 245, 10, 387, ...]
print(result['aa_tokens'])  # [12, 10, 19, 0, 24, ...]
print(result['structure_tokens'])  # [245, 387, 102, 421, ...]
```

---

### `decode(tokens, skip_special)`

Decode tokens back to amino acid sequence.

**Parameters:**
- `tokens` (List[int]): Token IDs
- `skip_special` (bool): Skip special tokens (default: True)

**Returns:**
- Amino acid sequence string

**Example:**
```python
tokens = [534, 12, 245, 10, 387, 535]
sequence = tokenizer.decode(tokens)  # "MK"
```

---

### `batch_encode(sequences, **kwargs)`

Encode multiple sequences.

**Example:**
```python
sequences = ["MKTAYIAKQR", "ACDEFGH", "ILVWYPQ"]
results = tokenizer.batch_encode(sequences)
for result in results:
    print(f"{result['sequence']}: {len(result['tokens'])} tokens")
```

---

### `get_token_name(token_id)`

Get human-readable name for a token ID.

**Example:**
```python
print(tokenizer.get_token_name(12))   # "AA:M"
print(tokenizer.get_token_name(245))  # "STRUCT:225"
print(tokenizer.get_token_name(534))  # "BOS"
```

---

## Integration with LLM Training

### 1. Create Embedding Layer

```python
import torch.nn as nn

vocab_size = tokenizer.vocab_size  # 536
embedding_dim = 768  # Match your LLM

embedding_layer = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim
)
```

### 2. Prepare Training Data

```python
# Tokenize protein sequence
protein_tokens = tokenizer.encode("MKTAYIAKQR")['tokens']

# Prepare input for LLM
input_ids = torch.tensor([protein_tokens])
attention_mask = torch.ones_like(input_ids)

# Forward pass
embeddings = embedding_layer(input_ids)
# Feed to LLM...
```

### 3. Batch Processing

```python
from torch.nn.utils.rnn import pad_sequence

sequences = ["MKTAYIAKQR", "ACDEFGH", "ILVWYPQ"]
results = tokenizer.batch_encode(sequences)

# Convert to tensors and pad
token_tensors = [torch.tensor(r['tokens']) for r in results]
input_ids = pad_sequence(
    token_tensors,
    batch_first=True,
    padding_value=tokenizer.PAD_TOKEN
)

# Create attention mask
attention_mask = (input_ids != tokenizer.PAD_TOKEN).long()
```

### 4. Training Loop Example

```python
from transformers import AutoModel, Trainer, TrainingArguments

# Your protein-to-text model
model = MyProteinLM(vocab_size=tokenizer.vocab_size)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    # ... other args
)

# Custom data collator
def collate_fn(examples):
    sequences = [ex['sequence'] for ex in examples]
    results = tokenizer.batch_encode(sequences)
    
    # Pad and create tensors
    input_ids = pad_sequence(
        [torch.tensor(r['tokens']) for r in results],
        batch_first=True,
        padding_value=tokenizer.PAD_TOKEN
    )
    attention_mask = (input_ids != tokenizer.PAD_TOKEN).long()
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': ...,  # Your target text tokens
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
)

trainer.train()
```

## Saving and Loading

### Save Tokenizer

```python
import json
import shutil

# Save codebook
shutil.copy(
    "data/structure_codebook_K512.pkl",
    "my_tokenizer/codebook.pkl"
)

# Save config
config = {
    "vocab_size": tokenizer.vocab_size,
    "n_clusters": tokenizer.n_clusters,
    "esm_model_name": "facebook/esm2_t33_650M_UR50D",
    "special_tokens": {
        "pad_token": tokenizer.PAD_TOKEN,
        "unk_token": tokenizer.UNK_TOKEN,
        "bos_token": tokenizer.BOS_TOKEN,
        "eos_token": tokenizer.EOS_TOKEN,
    }
}

with open("my_tokenizer/config.json", "w") as f:
    json.dump(config, f, indent=2)
```

### Load Tokenizer

```python
# Load from saved files
tokenizer = ProteinStructureTokenizer(
    codebook_path="my_tokenizer/codebook.pkl"
)
```

## Performance Notes

- **ESM-2 Inference**: ~0.1-0.5s per sequence (depends on length and GPU)
- **Memory Usage**: ~2GB for ESM-2 model + embeddings
- **Recommended Batch Size**: 4-8 sequences (for GPU with 16GB VRAM)

## Tips

1. **Cache embeddings** for training to avoid recomputing ESM-2 embeddings
2. **Pre-tokenize** your dataset and save as `.pt` files
3. **Use mixed precision** (fp16) for faster inference
4. **Normalize embeddings** before k-means lookup for better stability

## Examples

See `demo_tokenizer_class.py` for complete working examples including:
- Single sequence tokenization
- Batch processing
- Token distribution analysis
- Structure token statistics
- Decode verification





## Overview

The `ProteinStructureTokenizer` class provides a complete solution for tokenizing protein sequences into multimodal token streams combining:
- **Character-level amino acid tokens** (20 standard amino acids)
- **Structure tokens** (512 learned from ESM-2 embeddings via k-means)

## Quick Start

### Running the Demo

```bash
# Activate environment
conda activate esm_tokenizer

# Run demo (requires GPU)
cd /lustrefs/shared/mohammad.sayeed/Prot2Text/esmfold_tokenizer
python demo_tokenizer_class.py
```

### Using in Your Code

```python
from demo_tokenizer_class import ProteinStructureTokenizer

# Initialize
tokenizer = ProteinStructureTokenizer(
    codebook_path="data/structure_codebook_K512.pkl",
    esm_model_name="facebook/esm2_t33_650M_UR50D"
)

# Tokenize a sequence
sequence = "MKTAYIAKQRQISFVKSHFSRQLE"
result = tokenizer.encode(sequence)

print(f"Tokens: {result['tokens']}")
print(f"Length: {len(result['tokens'])}")

# Decode back to sequence
decoded = tokenizer.decode(result['tokens'])
print(f"Decoded: {decoded}")
```

## Token Space

The tokenizer uses a unified vocabulary of **536 tokens**:

| Range | Description | Count |
|-------|-------------|-------|
| 0-19 | Amino acid tokens (A-Y) | 20 |
| 20-531 | Structure tokens (k-means centroids) | 512 |
| 532 | PAD token | 1 |
| 533 | UNK token | 1 |
| 534 | BOS token (begin of sequence) | 1 |
| 535 | EOS token (end of sequence) | 1 |
| **Total** | | **536** |

## Token Stream Format

The tokenizer produces an **interleaved** token stream:

```
[BOS, aa_1, struct_1, aa_2, struct_2, aa_3, struct_3, ..., aa_N, struct_N, EOS]
```

### Example

For sequence "MKTA":

```
[534, 12, 245, 10, 387, 19, 102, 0, 421, 535]
 BOS  M  s_M   K  s_K   T  s_T  A  s_A  EOS
```

Where:
- `12, 10, 19, 0` = amino acid tokens for M, K, T, A
- `245, 387, 102, 421` = structure tokens (learned from ESM-2)

## API Reference

### `__init__(codebook_path, esm_model_name, device)`

Initialize the tokenizer.

**Parameters:**
- `codebook_path` (str): Path to trained codebook (.pkl file)
- `esm_model_name` (str): ESM-2 model from HuggingFace (default: "facebook/esm2_t33_650M_UR50D")
- `device` (str): 'cuda' or 'cpu' (auto-detected if None)

**Example:**
```python
tokenizer = ProteinStructureTokenizer(
    codebook_path="data/structure_codebook_K512.pkl"
)
```

---

### `encode(sequence, add_special_tokens, return_structure_tokens)`

Encode a protein sequence into tokens.

**Parameters:**
- `sequence` (str): Amino acid sequence (e.g., "MKTAYIAKQR")
- `add_special_tokens` (bool): Add BOS/EOS tokens (default: True)
- `return_structure_tokens` (bool): Return structure tokens separately (default: False)

**Returns:**
Dictionary with:
- `'tokens'`: Interleaved token list
- `'aa_tokens'`: Amino acid tokens only
- `'structure_tokens'`: Structure tokens (if requested)
- `'sequence'`: Original sequence
- `'length'`: Sequence length

**Example:**
```python
result = tokenizer.encode("MKTAYIAKQR", return_structure_tokens=True)
print(result['tokens'])  # [534, 12, 245, 10, 387, ...]
print(result['aa_tokens'])  # [12, 10, 19, 0, 24, ...]
print(result['structure_tokens'])  # [245, 387, 102, 421, ...]
```

---

### `decode(tokens, skip_special)`

Decode tokens back to amino acid sequence.

**Parameters:**
- `tokens` (List[int]): Token IDs
- `skip_special` (bool): Skip special tokens (default: True)

**Returns:**
- Amino acid sequence string

**Example:**
```python
tokens = [534, 12, 245, 10, 387, 535]
sequence = tokenizer.decode(tokens)  # "MK"
```

---

### `batch_encode(sequences, **kwargs)`

Encode multiple sequences.

**Example:**
```python
sequences = ["MKTAYIAKQR", "ACDEFGH", "ILVWYPQ"]
results = tokenizer.batch_encode(sequences)
for result in results:
    print(f"{result['sequence']}: {len(result['tokens'])} tokens")
```

---

### `get_token_name(token_id)`

Get human-readable name for a token ID.

**Example:**
```python
print(tokenizer.get_token_name(12))   # "AA:M"
print(tokenizer.get_token_name(245))  # "STRUCT:225"
print(tokenizer.get_token_name(534))  # "BOS"
```

---

## Integration with LLM Training

### 1. Create Embedding Layer

```python
import torch.nn as nn

vocab_size = tokenizer.vocab_size  # 536
embedding_dim = 768  # Match your LLM

embedding_layer = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim
)
```

### 2. Prepare Training Data

```python
# Tokenize protein sequence
protein_tokens = tokenizer.encode("MKTAYIAKQR")['tokens']

# Prepare input for LLM
input_ids = torch.tensor([protein_tokens])
attention_mask = torch.ones_like(input_ids)

# Forward pass
embeddings = embedding_layer(input_ids)
# Feed to LLM...
```

### 3. Batch Processing

```python
from torch.nn.utils.rnn import pad_sequence

sequences = ["MKTAYIAKQR", "ACDEFGH", "ILVWYPQ"]
results = tokenizer.batch_encode(sequences)

# Convert to tensors and pad
token_tensors = [torch.tensor(r['tokens']) for r in results]
input_ids = pad_sequence(
    token_tensors,
    batch_first=True,
    padding_value=tokenizer.PAD_TOKEN
)

# Create attention mask
attention_mask = (input_ids != tokenizer.PAD_TOKEN).long()
```

### 4. Training Loop Example

```python
from transformers import AutoModel, Trainer, TrainingArguments

# Your protein-to-text model
model = MyProteinLM(vocab_size=tokenizer.vocab_size)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    # ... other args
)

# Custom data collator
def collate_fn(examples):
    sequences = [ex['sequence'] for ex in examples]
    results = tokenizer.batch_encode(sequences)
    
    # Pad and create tensors
    input_ids = pad_sequence(
        [torch.tensor(r['tokens']) for r in results],
        batch_first=True,
        padding_value=tokenizer.PAD_TOKEN
    )
    attention_mask = (input_ids != tokenizer.PAD_TOKEN).long()
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': ...,  # Your target text tokens
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
)

trainer.train()
```

## Saving and Loading

### Save Tokenizer

```python
import json
import shutil

# Save codebook
shutil.copy(
    "data/structure_codebook_K512.pkl",
    "my_tokenizer/codebook.pkl"
)

# Save config
config = {
    "vocab_size": tokenizer.vocab_size,
    "n_clusters": tokenizer.n_clusters,
    "esm_model_name": "facebook/esm2_t33_650M_UR50D",
    "special_tokens": {
        "pad_token": tokenizer.PAD_TOKEN,
        "unk_token": tokenizer.UNK_TOKEN,
        "bos_token": tokenizer.BOS_TOKEN,
        "eos_token": tokenizer.EOS_TOKEN,
    }
}

with open("my_tokenizer/config.json", "w") as f:
    json.dump(config, f, indent=2)
```

### Load Tokenizer

```python
# Load from saved files
tokenizer = ProteinStructureTokenizer(
    codebook_path="my_tokenizer/codebook.pkl"
)
```

## Performance Notes

- **ESM-2 Inference**: ~0.1-0.5s per sequence (depends on length and GPU)
- **Memory Usage**: ~2GB for ESM-2 model + embeddings
- **Recommended Batch Size**: 4-8 sequences (for GPU with 16GB VRAM)

## Tips

1. **Cache embeddings** for training to avoid recomputing ESM-2 embeddings
2. **Pre-tokenize** your dataset and save as `.pt` files
3. **Use mixed precision** (fp16) for faster inference
4. **Normalize embeddings** before k-means lookup for better stability

## Examples

See `demo_tokenizer_class.py` for complete working examples including:
- Single sequence tokenization
- Batch processing
- Token distribution analysis
- Structure token statistics
- Decode verification




