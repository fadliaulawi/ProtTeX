"""Config for Prot3Text v3: train from precomputed ESM + ESMFold embeddings (no encoders in training)."""

from transformers import LlamaForCausalLM, PretrainedConfig, PreTrainedModel, LlamaConfig, Cache
from transformers import PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from typing import Dict, Optional, Union, Tuple, List
import os
import bisect
import numpy as np
import torch
import torch.nn as nn
import random

from config_prot3text import ModalityAdapterConfig, ModalityAdapter
from config_prot3text_v2 import CrossAttentionFusion, GatedFusion


class PrecomputedEmbeddingDataset(torch.utils.data.Dataset):
    """Dataset: CSV rows + precomputed embeddings (index-aligned with CSV).
    Supports:
    - Chunked .npz: embeddings_dir/split_000.npz, split_001.npz, ... (load chunks on demand, cache 2).
    - Single .npz: one file with seq_flat, struct_flat, lengths.
    - Single .pt: list of dicts (legacy).
    """
    def __init__(self, csv_path: str, embeddings_dir: str, split_name: str):
        super().__init__()
        import glob
        import pandas as pd
        import numpy as np
        self.data = pd.read_csv(csv_path)
        self._split = split_name
        self._dir = embeddings_dir
        chunk_pattern = os.path.join(embeddings_dir, f"{split_name}_*.npz")
        self._chunk_paths = sorted(glob.glob(chunk_pattern))
        if self._chunk_paths:
            self._chunked = True
            self._chunk_cumsum = [0]
            for p in self._chunk_paths:
                with np.load(p, allow_pickle=False) as f:
                    self._chunk_cumsum.append(self._chunk_cumsum[-1] + len(f["lengths"]))
            self._cache = {}
            self._cache_order = []
            self._max_cache = 2
        else:
            self._chunked = False
            single_npz = os.path.join(embeddings_dir, f"{split_name}.npz")
            single_pt = os.path.join(embeddings_dir, f"{split_name}.pt")
            if os.path.exists(single_npz):
                self._arch = np.load(single_npz, allow_pickle=False)
                self._offsets = np.concatenate([[0], np.cumsum(self._arch["lengths"])])
                self._format = "npz"
            elif os.path.exists(single_pt):
                self.embeddings = torch.load(single_pt, map_location="cpu", weights_only=True)
                self._format = "pt"
            else:
                raise FileNotFoundError(f"No embeddings found in {embeddings_dir} for split '{split_name}'")

    def _load_chunk(self, chunk_id: int):
        if chunk_id in self._cache:
            return self._cache[chunk_id]
        while len(self._cache) >= self._max_cache and self._cache_order:
            evict = self._cache_order.pop(0)
            del self._cache[evict]
        with np.load(self._chunk_paths[chunk_id], allow_pickle=False) as f:
            data = (f["seq_flat"], f["struct_flat"], f["lengths"])
        self._cache[chunk_id] = data
        self._cache_order.append(chunk_id)
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        row = {c: self.data.iloc[idx][c] for c in self.data.columns}
        if self._chunked:
            chunk_id = bisect.bisect_right(self._chunk_cumsum, idx) - 1
            local_idx = idx - self._chunk_cumsum[chunk_id]
            seq_flat, struct_flat, lengths = self._load_chunk(chunk_id)
            lengths = np.asarray(lengths)
            start = int(np.sum(lengths[:local_idx]))
            end = int(start + lengths[local_idx])
            seq = torch.from_numpy(seq_flat[start:end].copy()).to(torch.bfloat16)
            struct = torch.from_numpy(struct_flat[start:end].copy()).to(torch.bfloat16)
            row["seq_emb"] = seq
            row["struct_emb"] = struct
            row["encoder_attention_mask"] = torch.ones(end - start, dtype=torch.long)
        elif self._format == "npz":
            start, end = int(self._offsets[idx]), int(self._offsets[idx + 1])
            seq = torch.from_numpy(self._arch["seq_flat"][start:end].copy()).to(torch.bfloat16)
            struct = torch.from_numpy(self._arch["struct_flat"][start:end].copy()).to(torch.bfloat16)
            row["seq_emb"] = seq
            row["struct_emb"] = struct
            row["encoder_attention_mask"] = torch.ones(end - start, dtype=torch.long)
        else:
            emb = self.embeddings[idx]
            row["seq_emb"] = emb["seq"]
            row["struct_emb"] = emb["struct"]
            row["encoder_attention_mask"] = emb["mask"]
        return row


class PrecomputedEmbeddingCollater:
    """Collate batch: pad seq/struct to max L, build prompt + labels (same as v2 but from precomputed emb)."""
    def __init__(
        self,
        description_tokenizer: PreTrainedTokenizer,
        mode: str = "train",
        include_text_fields: bool = True,
        name_dropout: float = 0.8,
        taxonomy_dropout: float = 0.8,
        max_description_length: Optional[int] = 512,
        max_encoder_length: Optional[int] = None,
        system_message: str = (
            "You are a scientific assistant specialized in protein function "
            "predictions. Given the sequence embeddings and other information "
            "of a protein, describe its function clearly and concisely in "
            "professional language. "
        ),
        placeholder_token: str = "<|reserved_special_token_1|>",
    ):
        self.description_tokenizer = description_tokenizer
        self.mode = mode
        self.include_text_fields = include_text_fields
        self.name_dropout = name_dropout
        self.taxonomy_dropout = taxonomy_dropout
        self.max_description_length = max_description_length
        self.max_encoder_length = max_encoder_length
        self.system_message = system_message
        self.placeholder_token = placeholder_token

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        fullnames = [item["Full Name"] for item in batch]
        taxons = [item["taxon"] for item in batch]
        descriptions = [item["function"] for item in batch]
        fullnames = [
            fn if isinstance(fn, str) and random.random() > self.name_dropout else "unknown"
            for fn in fullnames
        ]
        taxons = [
            t if isinstance(t, str) and random.random() > self.taxonomy_dropout else "unknown"
            for t in taxons
        ]

        # Pad seq_emb and struct_emb to same length in batch
        seq_list = [item["seq_emb"] for item in batch]
        struct_list = [item["struct_emb"] for item in batch]
        mask_list = [item["encoder_attention_mask"] for item in batch]
        lengths = [m.sum().item() for m in mask_list]
        if self.max_encoder_length is not None:
            cap = self.max_encoder_length
            seq_list = [s[:cap] for s in seq_list]
            struct_list = [st[:cap] for st in struct_list]
            mask_list = [m[:cap] for m in mask_list]
            lengths = [min(L, cap) for L in lengths]
        max_L = max(s.size(0) for s in seq_list)

        d_seq = seq_list[0].size(-1)
        d_struct = struct_list[0].size(-1)
        device = seq_list[0].device
        dtype_seq = seq_list[0].dtype
        dtype_struct = struct_list[0].dtype

        seq_padded = torch.zeros(len(batch), max_L, d_seq, dtype=dtype_seq, device=device)
        struct_padded = torch.zeros(len(batch), max_L, d_struct, dtype=dtype_struct, device=device)
        mask_padded = torch.zeros(len(batch), max_L, dtype=torch.long, device=device)
        for i in range(len(batch)):
            L = seq_list[i].size(0)
            seq_padded[i, :L] = seq_list[i]
            struct_padded[i, :L] = struct_list[i]
            mask_padded[i, :L] = mask_list[i]

        if self.include_text_fields:
            user_messages = [
                f"Protein name: {fn}; Taxon: {t}; " + "Sequence embeddings: " + self.placeholder_token * L
                for fn, t, L in zip(fullnames, taxons, lengths)
            ]
        else:
            user_messages = ["Sequence embeddings: " + self.placeholder_token * L for L in lengths]

        prompt_conversations = [
            [{"role": "system", "content": self.system_message}, {"role": "user", "content": msg}]
            for msg in user_messages
        ]

        self.description_tokenizer.padding_side = "left"
        tokenized_prompts = self.description_tokenizer.apply_chat_template(
            prompt_conversations,
            add_generation_prompt=True,
            tokenize=True,
            padding="longest",
            return_tensors="pt",
            return_dict=True,
        )
        prompt_input_ids = tokenized_prompts["input_ids"]
        prompt_attention_mask = tokenized_prompts["attention_mask"]

        self.description_tokenizer.padding_side = "right"
        tokenized_descriptions = self.description_tokenizer(
            [d + self.description_tokenizer.eos_token for d in descriptions],
            add_special_tokens=False,
            truncation=True,
            padding="longest",
            max_length=self.max_description_length,
            return_tensors="pt",
        )
        description_input_ids = tokenized_descriptions["input_ids"]
        description_attention_mask = tokenized_descriptions["attention_mask"]
        if description_input_ids.size(1) > self.max_description_length:
            description_input_ids = description_input_ids[:, : self.max_description_length]
            description_attention_mask = description_attention_mask[:, : self.max_description_length]

        labels = description_input_ids.clone()
        labels[description_attention_mask == 0] = -100

        if self.mode == "train":
            return {
                "input_ids": torch.cat([prompt_input_ids, description_input_ids], dim=1),
                "attention_mask": torch.cat([prompt_attention_mask, description_attention_mask], dim=1),
                "labels": torch.cat([
                    torch.full_like(prompt_input_ids, fill_value=-100),
                    labels,
                ], dim=1),
                "seq_emb": seq_padded,
                "struct_emb": struct_padded,
                "encoder_attention_mask": mask_padded,
            }
        else:
            return {
                "input_ids": prompt_input_ids,
                "attention_mask": prompt_attention_mask,
                "seq_emb": seq_padded,
                "struct_emb": struct_padded,
                "encoder_attention_mask": mask_padded,
                "description_input_ids": description_input_ids,
            }


class LlamaFromPrecomputedV3Config(PretrainedConfig):
    model_type = "llama_from_precomputed_v3"

    def __init__(
        self,
        adapter_config: Optional[Union[ModalityAdapterConfig, Dict]] = None,
        structure_adapter_config: Optional[Union[ModalityAdapterConfig, Dict]] = None,
        llama_config: Optional[Union[LlamaConfig, Dict]] = None,
        placeholder_id: int = 128003,
        fusion_num_heads: int = 8,
        fusion_dropout: float = 0.1,
        gate_hidden_dim: int = 256,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.adapter_config = adapter_config if isinstance(adapter_config, ModalityAdapterConfig) else ModalityAdapterConfig(**adapter_config)
        self.structure_adapter_config = structure_adapter_config if isinstance(structure_adapter_config, ModalityAdapterConfig) else ModalityAdapterConfig(**structure_adapter_config)
        self.llama_config = llama_config if isinstance(llama_config, LlamaConfig) else LlamaConfig(**llama_config)
        self.placeholder_id = placeholder_id
        self.fusion_num_heads = fusion_num_heads
        self.fusion_dropout = fusion_dropout
        self.gate_hidden_dim = gate_hidden_dim


class LlamaFromPrecomputedV3ForCausalLM(PreTrainedModel):
    """Adapter + fusion + LLaMA only; encoder inputs are precomputed (seq_emb, struct_emb, mask)."""
    config_class = LlamaFromPrecomputedV3Config

    def __init__(
        self,
        config: Optional[LlamaFromPrecomputedV3Config] = None,
        adapter: Optional[ModalityAdapter] = None,
        structure_adapter: Optional[ModalityAdapter] = None,
        cross_attention_fusion: Optional[CrossAttentionFusion] = None,
        gated_fusion: Optional[GatedFusion] = None,
        llama_decoder: Optional[LlamaForCausalLM] = None,
        **kwargs
    ):
        if config is not None:
            super().__init__(config)
            self.adapter = ModalityAdapter(config.adapter_config)
            self.structure_adapter = ModalityAdapter(config.structure_adapter_config)
            self.cross_attention_fusion = CrossAttentionFusion(
                hidden_dim=config.llama_config.hidden_size,
                num_heads=config.fusion_num_heads,
                dropout=config.fusion_dropout,
            )
            self.gated_fusion = GatedFusion(
                hidden_dim=config.llama_config.hidden_size,
                gate_hidden_dim=config.gate_hidden_dim,
                dropout=config.fusion_dropout,
            )
            self.llama_decoder = LlamaForCausalLM(config.llama_config)
        else:
            super().__init__(LlamaFromPrecomputedV3Config(
                adapter_config=adapter.config,
                structure_adapter_config=structure_adapter.config,
                llama_config=llama_decoder.config,
                fusion_num_heads=kwargs.get("fusion_num_heads", 8),
                fusion_dropout=kwargs.get("fusion_dropout", 0.1),
                gate_hidden_dim=kwargs.get("gate_hidden_dim", 256),
            ))
            self.adapter = adapter
            self.structure_adapter = structure_adapter
            self.cross_attention_fusion = cross_attention_fusion
            self.gated_fusion = gated_fusion
            self.llama_decoder = llama_decoder

    def prepare_decoder_inputs(
        self,
        input_ids: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
    ):
        batch_size, seq_len = input_ids.size()
        _, encoder_seq_len, _ = encoder_hidden_states.size()
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=input_ids.device)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                (batch_size, encoder_seq_len), dtype=torch.long, device=encoder_hidden_states.device
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
        seq_emb: Optional[torch.FloatTensor] = None,
        struct_emb: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_adapter_outputs: bool = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        seq_projected = self.adapter(seq_emb)
        struct_projected = self.structure_adapter(struct_emb)
        if return_adapter_outputs:
            return seq_projected, struct_projected, encoder_attention_mask
        key_padding_mask = ~encoder_attention_mask.bool() if encoder_attention_mask is not None else None
        seq_enriched, struct_enriched = self.cross_attention_fusion(
            seq_projected, struct_projected, key_padding_mask=key_padding_mask
        )
        fused_embeddings = self.gated_fusion(seq_enriched, struct_enriched)
        inputs_embeds, attention_mask = self.prepare_decoder_inputs(
            input_ids=input_ids,
            encoder_hidden_states=fused_embeddings,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )
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
            cache_position=cache_position,
        )


def setup(rank: int, world_size: int):
    import os
    import torch.distributed as dist
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "9901")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
