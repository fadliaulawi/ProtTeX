#!/usr/bin/env python3
"""
Simple: Project & Query Llama for Protein Function
Tri-Modal Reasoning: Sequence → Embedding → Text → Structure Embedding → Function Prediction
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModelForCausalLM

LLAMA_HIDDEN_DIM = 4096


class ProteinProjectionHead(nn.Module):
    """Project protein embeddings to Llama space"""
    
    def __init__(self, input_dim: int, output_dim: int = LLAMA_HIDDEN_DIM, hidden_dim: int = 2048):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        proj = self.proj(x)
        return F.normalize(proj, p=2, dim=-1)


class StructureProjectionHead(nn.Module):
    """Project structure tokens to Llama space"""
    
    def __init__(self, vocab_size: int = 537, embedding_dim: int = 256, output_dim: int = LLAMA_HIDDEN_DIM, hidden_dim: int = 2048):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, token_ids):
        embedded = self.token_embedding(token_ids)
        pooled = embedded.mean(dim=1)
        proj = self.proj(pooled)
        return F.normalize(proj, p=2, dim=-1)


class TriModalAlignmentModel(nn.Module):
    """Tri-modal: Sequence + Structure ↔ Text"""
    
    def __init__(self, 
                 sequence_dim: int = 1280,
                 structure_vocab_size: int = 537,
                 text_dim: int = LLAMA_HIDDEN_DIM,
                 shared_dim: int = LLAMA_HIDDEN_DIM):
        super().__init__()
        
        self.sequence_proj = ProteinProjectionHead(sequence_dim, shared_dim)
        self.structure_proj = StructureProjectionHead(structure_vocab_size, embedding_dim=256, output_dim=shared_dim)
    
    def forward(self, sequence_emb, structure_tokens):
        seq_proj = self.sequence_proj(sequence_emb)
        struct_proj = self.structure_proj(structure_tokens)
        return seq_proj, struct_proj


class ProteinFunctionGenerator:
    """Query Llama for protein function using projected embeddings"""
    
    def __init__(self, 
                 alignment_checkpoint: str,
                 alignment_config: str,
                 llama_model: str = "meta-llama/Llama-3.1-8B",
                 device: str = "cuda"):
        """
        Args:
            alignment_checkpoint: Path to trained projection heads (.pt file)
            alignment_config: Path to alignment config (.json file)
            llama_model: HuggingFace Llama model ID
            device: 'cuda' or 'cpu'
        """
        self.device = device
        
        # Load alignment model
        print("📥 Loading trained projection heads...")
        with open(alignment_config) as f:
            config = json.load(f)
        
        self.alignment_model = TriModalAlignmentModel(
            sequence_dim=config['sequence_dim'],
            structure_vocab_size=config['structure_vocab'],
            text_dim=config['text_dim'],
            shared_dim=config['shared_dim']
        )
        
        checkpoint = torch.load(alignment_checkpoint, map_location=device)
        self.alignment_model.load_state_dict(checkpoint['model_state'])
        self.alignment_model.to(device)
        self.alignment_model.eval()
        print("✅ Loaded alignment model")
        
        # Load Llama
        print("📥 Loading Llama-3.1...")
        self.tokenizer = AutoTokenizer.from_pretrained(llama_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llama = AutoModelForCausalLM.from_pretrained(
            llama_model,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.llama.eval()
        print("✅ Loaded Llama-3.1")
    
    @torch.no_grad()
    def project_protein(self, sequence_embedding: np.ndarray, structure_tokens: np.ndarray):
        """
        Project protein embeddings to shared space.
        
        Args:
            sequence_embedding: [1280] or [1, 1280] ESM-2 embedding
            structure_tokens: [seq_len] or [1, seq_len] interleaved tokens
        
        Returns:
            seq_proj: [1, 4096] normalized
            struct_proj: [1, 4096] normalized
        """
        # Ensure batch dimension
        if sequence_embedding.ndim == 1:
            sequence_embedding = sequence_embedding[np.newaxis, :]
        if structure_tokens.ndim == 1:
            structure_tokens = structure_tokens[np.newaxis, :]
        
        # Convert to tensors
        seq_emb = torch.tensor(sequence_embedding, dtype=torch.float32).to(self.device)
        struct_tokens = torch.tensor(structure_tokens, dtype=torch.long).to(self.device)
        
        # Project
        seq_proj, struct_proj = self.alignment_model(seq_emb, struct_tokens)
        
        return seq_proj, struct_proj
    
    @torch.no_grad()
    def query_function(self, 
                      sequence_embedding: np.ndarray,
                      structure_tokens: np.ndarray,
                      protein_sequence: str,
                      protein_name: str = "protein",
                      max_new_tokens: int = 150,
                      temperature: float = 0.7):
        """
        Ask Llama to predict protein function given sequence, structure, and text.
        Tri-modal reasoning: Sequence → Sequence Embedding → Text → Structure Embedding → Decision
        
        Args:
            sequence_embedding: [1280] ESM-2 embedding
            structure_tokens: [seq_len] interleaved tokens (padded to 1024)
            protein_sequence: Raw amino acid sequence string (e.g., "MVHLTPEEKS...")
            protein_name: Name/ID of protein
            max_new_tokens: Max tokens to generate
            temperature: Generation temperature (higher = more creative)
        
        Returns:
            function_description: Generated text from Llama with tri-modal reasoning
        """
        # Project embeddings separately (NOT fused)
        seq_proj, struct_proj = self.project_protein(sequence_embedding, structure_tokens)
        seq_proj = seq_proj.unsqueeze(1)  # [1, 1, 4096] - virtual token
        struct_proj = struct_proj.unsqueeze(1)  # [1, 1, 4096] - virtual token
        
        # Build multi-part prompt with embeddings injected strategically
        # Part 1: Introduce sequence
        part1 = f"Protein: {protein_name}\nSequence: {protein_sequence}"
        inputs1 = self.tokenizer(part1, return_tensors="pt").to(self.device)
        emb1 = self.llama.get_input_embeddings()(inputs1['input_ids'])  # [1, len1, 4096]
        
        # Part 2: Text context (reasoning after sequence)
        part2 = "\n\nThis sequence typically encodes a protein involved in cellular processes. "
        inputs2 = self.tokenizer(part2, return_tensors="pt").to(self.device)
        emb2 = self.llama.get_input_embeddings()(inputs2['input_ids'])  # [1, len2, 4096]
        
        # Part 3: Structure-based reasoning
        part3 = "\n\nHowever, the structural analysis reveals that the 3D conformation has specific properties that constrain function. Given the sequence, structure, and biochemical properties, the most likely function is:"
        inputs3 = self.tokenizer(part3, return_tensors="pt").to(self.device)
        emb3 = self.llama.get_input_embeddings()(inputs3['input_ids'])  # [1, len3, 4096]
        
        # Construct combined embeddings: emb1 + seq_proj + emb2 + struct_proj + emb3
        combined_embeddings = torch.cat([
            emb1,           # Sequence text: [1, len1, 4096]
            seq_proj,       # Sequence embedding: [1, 1, 4096]
            emb2,           # Context text: [1, len2, 4096]
            struct_proj,    # Structure embedding: [1, 1, 4096]
            emb3            # Final reasoning text: [1, len3, 4096]
        ], dim=1)  # [1, len1+1+len2+1+len3, 4096]
        
        # Construct combined attention masks (all ones - everything is attended)
        mask1 = torch.ones(1, inputs1['input_ids'].shape[1], dtype=torch.long, device=self.device)
        seq_mask = torch.ones(1, 1, dtype=torch.long, device=self.device)  # seq embedding
        mask2 = torch.ones(1, inputs2['input_ids'].shape[1], dtype=torch.long, device=self.device)
        struct_mask = torch.ones(1, 1, dtype=torch.long, device=self.device)  # struct embedding
        mask3 = torch.ones(1, inputs3['input_ids'].shape[1], dtype=torch.long, device=self.device)
        
        combined_attention_mask = torch.cat([
            mask1, seq_mask, mask2, struct_mask, mask3
        ], dim=1)
        
        # Construct full prompt for later removal (text parts only)
        full_prompt = part1 + part2 + part3
        
        # Generate using embeddings directly
        print(f"\n🧬 Querying Llama for {protein_name}...")
        print(f"📝 Tri-modal reasoning: Sequence → Embedding → Text → Embedding → Reasoning")
        print(f"📊 Sequence length: {len(protein_sequence)} AA")
        print(f"🔗 Embeddings injected at 2 strategic points\n")
        
        outputs = self.llama.generate(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode and remove prompt
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Try to extract only the generated part
        if full_prompt in full_text:
            generated = full_text.split(full_prompt)[-1]
        else:
            # Fallback: try removing by length
            prompt_tokens = len(self.tokenizer.encode(full_prompt))
            generated_tokens = outputs[0][prompt_tokens:]
            generated = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated.strip()


def main():
    print("=" * 70)
    print("SIMPLE: PROJECT & QUERY LLAMA FOR PROTEIN FUNCTION")
    print("=" * 70)
    
    # Setup paths
    subset_name = "UniProt_Function"
    model_dir = Path('results/trimodal_alignment') / subset_name
    
    # Find best checkpoint
    checkpoint_files = sorted(model_dir.glob('best_model_global_epoch_*.pt'))
    if not checkpoint_files:
        print(f"❌ No checkpoint found in {model_dir}")
        print("\n💡 First run: python 07_train_clip_alignment.py UniProt_Function")
        return
    
    # Use the latest best checkpoint
    checkpoint_path = checkpoint_files[-1]
    config_path = model_dir / 'config.json'
    
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return
    
    print(f"\n✅ Using checkpoint: {checkpoint_path.name}")
    
    # Initialize generator
    generator = ProteinFunctionGenerator(
        alignment_checkpoint=str(checkpoint_path),
        alignment_config=str(config_path),
        device="cuda"
    )
    
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Predict Function from Tri-Modal Reasoning")
    print("=" * 70)
    
    # Create dummy embeddings and sequence for demo
    seq_embedding = np.random.randn(1280).astype(np.float32)  # ESM-2: 1280D
    struct_tokens = np.random.randint(0, 537, size=1024)  # Interleaved tokens: 1024 length
    
    # Example protein sequence (avg length ~350 AA)
    example_sequence = "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"
    
    function = generator.query_function(
        sequence_embedding=seq_embedding,
        structure_tokens=struct_tokens,
        protein_sequence=example_sequence,
        protein_name="Example_Protein_001",
        max_new_tokens=150
    )
    
    print(f"\n🔬 Generated Function Description:")
    print(f"{function}")
    
    print("\n" + "=" * 70)
    print("USAGE EXAMPLE")
    print("=" * 70)
    print("""
# Python code
from pathlib import Path
import numpy as np

# Initialize
generator = ProteinFunctionGenerator(
    alignment_checkpoint='results/trimodal_alignment/UniProt_Function/best_model_global_epoch_6_batch_5.pt',
    alignment_config='results/trimodal_alignment/UniProt_Function/config.json'
)

# Load your embeddings
seq_emb = np.load('my_sequence_embedding.npy')      # [1280]
struct_tokens = np.load('my_structure_tokens.npy')  # [1024]
protein_seq = "MVHLTPEEKSLVY..."  # Raw amino acid sequence

# Query Llama with tri-modal reasoning
function = generator.query_function(
    sequence_embedding=seq_emb,
    structure_tokens=struct_tokens,
    protein_sequence=protein_seq,
    protein_name="MyProtein_XYZ",
    max_new_tokens=150,
    temperature=0.7
)

print(f"Predicted Function: {function}")
    """)
    
    print("\n" + "=" * 70)
    print("TRI-MODAL REASONING ARCHITECTURE")
    print("=" * 70)
    print("""
INPUT:
  - Sequence (raw AA): ~350 chars
  - Sequence embedding (ESM-2): 1280-dim
  - Structure tokens: 1024-length
  
FLOW:
  1. Text: "Protein: {name}\\nSequence: {SEQUENCE}"
     ↓ [tokenize → embed]
  
  2. Virtual Token: Sequence Projection [4096-dim]
     ↓ [inject learned sequence representation]
  
  3. Text: "This sequence encodes a protein..."
     ↓ [tokenize → embed]
  
  4. Virtual Token: Structure Projection [4096-dim]
     ↓ [inject learned structure representation]
  
  5. Text: "Given sequence, structure, and biochemical
            properties, the most likely function is:"
     ↓ [tokenize → embed]
  
  6. Llama Generation
     ↓ [generate function description]

REASONING STRATEGY:
  ✅ Sequence visible as text (interpretable)
  ✅ Sequence embedding captures deep patterns (ESM-2)
  ✅ Text provides biological context
  ✅ Structure embedding provides 3D constraints (decision-maker)
  ✅ Structure comes last → final say on function
  ✅ Natural flow: Seq → Context → Structure → Function

TOTAL TOKENS: ~420 (well within Llama's 8K context)
    """)


if __name__ == '__main__':
    main()
