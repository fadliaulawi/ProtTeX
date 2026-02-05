"""Enhanced config with Cross-Attention + Gated Fusion for ESM2-Llama protein-to-text model."""

from transformers import EsmModel, LlamaForCausalLM, PretrainedConfig, PreTrainedModel, LlamaConfig, EsmConfig, Cache
from transformers import PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from typing import Dict, Optional, Union, Tuple, Literal, List
import torch
import torch.nn as nn
import pandas as pd
import random


# Import base classes from original config
from config_prot3text import (
    Prot2TextLightDataset,
    Prot2TextLightCollater,
    ModalityAdapterConfig,
    ModalityAdapter,
)


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention for enriching sequence and structure embeddings."""
    
    def __init__(self, hidden_dim: int = 4096, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Sequence attends to structure
        self.seq_to_struct_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Structure attends to sequence
        self.struct_to_seq_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norms for residual connections
        self.ln_seq = nn.LayerNorm(hidden_dim)
        self.ln_struct = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        seq_emb: torch.FloatTensor, 
        struct_emb: torch.FloatTensor,
        key_padding_mask: Optional[torch.BoolTensor] = None
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Args:
            seq_emb: (bsz, seq_len, hidden_dim)
            struct_emb: (bsz, seq_len, hidden_dim)
            key_padding_mask: (bsz, seq_len) - True for padding positions
            
        Returns:
            seq_enriched: (bsz, seq_len, hidden_dim)
            struct_enriched: (bsz, seq_len, hidden_dim)
        """
        # Sequence borrows from structure
        seq_attn_out, _ = self.seq_to_struct_attn(
            query=seq_emb,
            key=struct_emb,
            value=struct_emb,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        seq_enriched = self.ln_seq(seq_emb + self.dropout(seq_attn_out))
        
        # Structure borrows from sequence
        struct_attn_out, _ = self.struct_to_seq_attn(
            query=struct_emb,
            key=seq_emb,
            value=seq_emb,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        struct_enriched = self.ln_struct(struct_emb + self.dropout(struct_attn_out))
        
        return seq_enriched, struct_enriched


class GatedFusion(nn.Module):
    """Position-conditioned gated fusion with MLP-based alpha prediction."""
    
    def __init__(self, hidden_dim: int = 4096, gate_hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        # Simple 2-layer MLP for faster learning
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, seq_emb: torch.FloatTensor, struct_emb: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            seq_emb: (bsz, seq_len, hidden_dim)
            struct_emb: (bsz, seq_len, hidden_dim)
            
        Returns:
            fused: (bsz, seq_len, hidden_dim)
        """
        # Concatenate and predict per-position alpha
        combined = torch.cat([seq_emb, struct_emb], dim=-1)  # (bsz, seq_len, hidden_dim*2)
        alpha = self.gate_net(combined)  # (bsz, seq_len, 1)
        
        # Weighted fusion
        fused = alpha * seq_emb + (1 - alpha) * struct_emb
        return fused
    
    def get_alpha_stats(self, seq_emb: torch.FloatTensor, struct_emb: torch.FloatTensor) -> Dict[str, float]:
        """Get statistics of learned alpha values for logging."""
        with torch.no_grad():
            combined = torch.cat([seq_emb, struct_emb], dim=-1)
            alpha = self.gate_net(combined).squeeze(-1)  # (bsz, seq_len)
            return {
                'mean': alpha.mean().item(),
                'std': alpha.std().item(),
                'min': alpha.min().item(),
                'max': alpha.max().item()
            }

class Esm2LlamaInstructV2Config(PretrainedConfig):
    """Configuration for enhanced model with cross-attention + gated fusion."""
    model_type = "esm2llama_instruct_v2"

    def __init__(
        self, 
        esm_config: Optional[Union[EsmConfig, Dict]] = None, 
        adapter_config: Optional[Union[ModalityAdapterConfig, Dict]] = None,
        structure_adapter_config: Optional[Union[ModalityAdapterConfig, Dict]] = None,
        llama_config: Optional[Union[LlamaConfig, Dict]] = None, 
        placeholder_id: int = 128003,
        codebook_k: Optional[int] = None,
        fusion_num_heads: int = 8,
        fusion_dropout: float = 0.1,
        gate_hidden_dim: int = 128,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        if isinstance(esm_config, dict):
            self.esm_config = EsmConfig(**esm_config)
        else:
            self.esm_config = esm_config
            
        if isinstance(llama_config, dict):
            self.llama_config = LlamaConfig(**llama_config)
        else:
            self.llama_config = llama_config
            
        if isinstance(adapter_config, dict):
            self.adapter_config = ModalityAdapterConfig(**adapter_config)
        else:
            self.adapter_config = adapter_config
        
        if isinstance(structure_adapter_config, dict):
            self.structure_adapter_config = ModalityAdapterConfig(**structure_adapter_config)
        else:
            self.structure_adapter_config = structure_adapter_config
            
        self.placeholder_id = placeholder_id
        self.codebook_k = codebook_k
        self.fusion_num_heads = fusion_num_heads
        self.fusion_dropout = fusion_dropout
        self.gate_hidden_dim = gate_hidden_dim


class Esm2LlamaInstructV2ForCausalLM(PreTrainedModel):
    """Enhanced model with cross-attention + gated fusion."""
    config_class = Esm2LlamaInstructV2Config

    def __init__(
        self, 
        config: Optional[Esm2LlamaInstructV2Config] = None, 
        esm_encoder: Optional[EsmModel] = None, 
        adapter: Optional[ModalityAdapter] = None,
        esmfold_encoder = None,
        structure_adapter: Optional[ModalityAdapter] = None,
        cross_attention_fusion: Optional[CrossAttentionFusion] = None,
        gated_fusion: Optional[GatedFusion] = None,
        llama_decoder: Optional[LlamaForCausalLM] = None,
        codebook_centroids: Optional[torch.Tensor] = None,
        **kwargs
    ):
        if config is not None:
            super().__init__(config)
            self.esm_encoder = EsmModel(
                config.esm_config, 
                add_pooling_layer=False
            )
            self.adapter = ModalityAdapter(config.adapter_config)
            self.structure_adapter = ModalityAdapter(config.structure_adapter_config)
            self.cross_attention_fusion = CrossAttentionFusion(
                hidden_dim=config.llama_config.hidden_size,
                num_heads=config.fusion_num_heads,
                dropout=config.fusion_dropout
            )
            self.gated_fusion = GatedFusion(
                hidden_dim=config.llama_config.hidden_size,
                gate_hidden_dim=config.gate_hidden_dim
            )
            self.llama_decoder = LlamaForCausalLM(config.llama_config)
            self.esmfold_encoder = None
            self.codebook_centroids = None
        else: 
            config = Esm2LlamaInstructV2Config(
                esm_config=esm_encoder.config,
                adapter_config=adapter.config,
                structure_adapter_config=structure_adapter.config,
                llama_config=llama_decoder.config, 
                **kwargs
            ) 
            super().__init__(config)
            self.esm_encoder = esm_encoder
            self.adapter = adapter
            self.esmfold_encoder = esmfold_encoder
            self.structure_adapter = structure_adapter
            self.cross_attention_fusion = cross_attention_fusion
            self.gated_fusion = gated_fusion
            self.llama_decoder = llama_decoder
            self.codebook_centroids = codebook_centroids
            
    def prepare_decoder_inputs(
        self, 
        input_ids: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None, 
    ): 
        """Prepare decoder inputs by replacing placeholder tokens with encoder outputs."""

        batch_size, seq_len = input_ids.size()
        _, encoder_seq_len, _ = encoder_hidden_states.size()
        if attention_mask is None: 
            attention_mask = torch.ones(
                (batch_size, seq_len), 
                dtype=torch.long, 
                device=input_ids.device
            )
        if encoder_attention_mask is None: 
            encoder_attention_mask = torch.ones(
                (batch_size, encoder_seq_len), 
                dtype=torch.long, 
                device=encoder_hidden_states.device
            )
        inputs_embeds = self.llama_decoder.get_input_embeddings()(input_ids)
        placeholder_mask = input_ids == self.config.placeholder_id
        encoder_mask = encoder_attention_mask.bool()
        inputs_embeds[placeholder_mask] = encoder_hidden_states[encoder_mask]
        return inputs_embeds, attention_mask

    def forward(
        self, 
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        labels: Optional[torch.LongTensor] = None,
        protein_input_ids: Optional[torch.LongTensor] = None,
        protein_attention_mask: Optional[torch.LongTensor] = None,
        protein_position_ids: Optional[torch.LongTensor] = None, 
        protein_head_mask: Optional[torch.LongTensor] = None,
        protein_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_encoder_outputs: bool = False,
        return_adapter_outputs: bool = False, 
        return_decoder_inputs: bool = False,
        cache_position: Optional[torch.LongTensor] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]: 
        """Forward with cross-attention enrichment + gated fusion."""

        # === SEQUENCE PATH: ESM2 encoder ===
        seq_encoder_output = self.esm_encoder(
            input_ids=protein_input_ids,
            attention_mask=protein_attention_mask,
            position_ids=protein_position_ids,
            head_mask=protein_head_mask,
            inputs_embeds=protein_inputs_embeds,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        seq_hidden_states = seq_encoder_output.last_hidden_state
        encoder_attention_mask = protein_attention_mask
        if return_encoder_outputs:
            return seq_encoder_output
        
        # === STRUCTURE PATH: ESMFold encoder + K-means ===
        with torch.no_grad():
            struct_encoder_output = self.esmfold_encoder.esm(
                input_ids=protein_input_ids,
                attention_mask=protein_attention_mask,
                output_hidden_states=True
            )
            struct_hidden_states = struct_encoder_output.hidden_states[-1]
        
        # K-means inference
        if self.codebook_centroids is not None:
            bsz, seq_len, emb_dim = struct_hidden_states.shape
            struct_flat = struct_hidden_states.view(-1, emb_dim)
            centroids = self.codebook_centroids.to(struct_flat.device)
            distances = torch.cdist(struct_flat.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)
            cluster_ids = torch.argmin(distances, dim=1)
            struct_hidden_states = centroids[cluster_ids].view(bsz, seq_len, emb_dim)
        
        # === ADAPTER PROJECTION ===
        seq_projected = self.adapter(seq_hidden_states)  # (bsz, seq_len, 4096)
        struct_projected = self.structure_adapter(struct_hidden_states)  # (bsz, seq_len, 4096)
        
        if return_adapter_outputs:
            return seq_projected, struct_projected, encoder_attention_mask
        
        # === CROSS-ATTENTION ENRICHMENT ===
        # Create padding mask for attention (True = padding)
        key_padding_mask = ~encoder_attention_mask.bool() if encoder_attention_mask is not None else None
        seq_enriched, struct_enriched = self.cross_attention_fusion(
            seq_projected, struct_projected, key_padding_mask
        )
        
        # === GATED FUSION ===
        fused_embeddings = self.gated_fusion(seq_enriched, struct_enriched)
        
        # === DECODER INPUT PREPARATION ===
        inputs_embeds, attention_mask = self.prepare_decoder_inputs(
            input_ids=input_ids, 
            encoder_hidden_states=fused_embeddings, 
            attention_mask=attention_mask, 
            encoder_attention_mask=encoder_attention_mask, 
        )
        if return_decoder_inputs:
            return inputs_embeds, attention_mask
        
        # === LLAMA DECODER ===
        return self.llama_decoder.forward(
            input_ids=None,
            attention_mask=attention_mask, 
            position_ids=position_ids, 
            past_key_values=past_key_values, 
            inputs_embeds=inputs_embeds, 
            labels=labels, 
            use_cache=use_cache, 
            output_attentions=output_attentions, 
            return_dict=return_dict, 
            cache_position=cache_position
        )

    def generate(
        self,
        inputs: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        protein_input_ids: Optional[torch.LongTensor] = None,
        protein_attention_mask: Optional[torch.LongTensor] = None,
        protein_inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """Generate with enhanced fusion."""

        prompt_inputs_embeds, prompt_attention_mask = self(
            input_ids=inputs, 
            attention_mask=attention_mask,
            protein_input_ids=protein_input_ids,
            protein_attention_mask=protein_attention_mask,
            protein_inputs_embeds=protein_inputs_embeds,
            use_cache=False, 
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
            return_decoder_inputs=True
        )
        return self.llama_decoder.generate(
            inputs_embeds=prompt_inputs_embeds, 
            attention_mask=prompt_attention_mask, 
            **kwargs
        )

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.esm_encoder, "gradient_checkpointing_enable"):
            self.esm_encoder.gradient_checkpointing_enable()
        if hasattr(self.llama_decoder, "gradient_checkpointing_enable"):
            self.llama_decoder.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        if hasattr(self.esm_encoder, "gradient_checkpointing_disable"):
            self.esm_encoder.gradient_checkpointing_disable()
        if hasattr(self.llama_decoder, "gradient_checkpointing_disable"):
            self.llama_decoder.gradient_checkpointing_disable()


def setup(rank: int, world_size: int):
    """Initialize distributed training process group."""
    import os
    import torch.distributed as dist
    
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '9901')
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
