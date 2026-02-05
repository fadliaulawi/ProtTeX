"""Shared classes and utilities for ESM2-Llama protein-to-text model training and inference."""

from transformers import EsmModel, LlamaForCausalLM, PretrainedConfig, PreTrainedModel, LlamaConfig, EsmConfig, Cache
from transformers import PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from typing import Dict, Optional, Union, Tuple, Literal, List
import torch
import pandas as pd
import random


class Prot2TextLightDataset(torch.utils.data.Dataset): 
    """Dataset class loading directly from single CSV file."""
    def __init__(self, csv_path: str):
        super().__init__()
        self.data: pd.DataFrame = pd.read_csv(csv_path)

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        return {
            column_name: self.data.iloc[idx][column_name] 
            for column_name in self.data.columns
        }


class Prot2TextLightCollater: 
    """Collate function for batching protein-to-text data."""
    def __init__(
        self, 
        sequence_tokenizer: PreTrainedTokenizer,
        description_tokenizer: PreTrainedTokenizer,
        mode: Literal["train", "inference"] = "train", 
        include_text_fields: bool = True,
        name_dropout: float = 0.8, 
        taxonomy_dropout: float = 0.8,
        max_sequence_length: Optional[int] = 1021, 
        max_description_length: Optional[int] = 512, 
        system_message: str = (
            "You are a scientific assistant specialized in protein function "
            "predictions. Given the sequence embeddings and other information "
            "of a protein, describe its function clearly and concisely in "
            "professional language. "
        ), 
        placeholder_token: str = '<|reserved_special_token_1|>', 
    ):
        self.sequence_tokenizer = sequence_tokenizer
        self.description_tokenizer = description_tokenizer
        self.mode = mode

        self.include_text_fields = include_text_fields
        self.name_dropout = name_dropout
        self.taxonomy_dropout = taxonomy_dropout

        self.max_sequence_length = max_sequence_length
        self.max_description_length = max_description_length
        self.system_message = system_message
        self.placeholder_token = placeholder_token

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        # group data across batch
        accessions = [item["AlphaFoldDB"] for item in batch]
        fullnames = [item["Full Name"] for item in batch]
        taxons = [item["taxon"] for item in batch]
        sequences = [item["sequence"] for item in batch]
        descriptions = [item["function"] for item in batch]
        
        fullnames = [
            fullname 
            if isinstance(fullname, str) and random.random() > self.name_dropout 
            else "unknown" 
            for fullname in fullnames
        ]
        taxons = [
            taxon 
            if isinstance(taxon, str) and random.random() > self.taxonomy_dropout 
            else "unknown" 
            for taxon in taxons
        ]

        # if the sequence is originally longer than max_sequence_length, take a segment of that length randomly 
        for i in range(len(sequences)):
            if len(sequences[i]) > self.max_sequence_length:
                start = random.randint(0, len(sequences[i]) - self.max_sequence_length)
                sequences[i] = sequences[i][start:start + self.max_sequence_length]

        self.sequence_tokenizer.padding_side = "right"
        tokenized_sequences = self.sequence_tokenizer(
            sequences, 
            truncation=True, 
            padding="longest", 
            max_length=self.max_sequence_length + 2,  # including bos and eos tokens of esm tokenizer
            return_tensors="pt"
        )
        sequence_input_ids = tokenized_sequences["input_ids"]
        sequence_attention_mask = tokenized_sequences["attention_mask"]

        # apply chat template
        sequence_lens = sequence_attention_mask.sum(dim=1).tolist()

        if self.include_text_fields: 
            user_messages = [
                (
                    f"Protein name: {fullname}; Taxon: {taxon}; "
                    + "Sequence embeddings: " + self.placeholder_token * sequence_len
                )
                for fullname, taxon, sequence_len in zip(fullnames, taxons, sequence_lens)
            ]
        else: 
            user_messages = [
                "Sequence embeddings: " + self.placeholder_token * sequence_lens
                for sequence_lens in sequence_lens
            ]

        prompt_conversations = [
            [
                {"role": "system", "content": self.system_message}, 
                {"role": "user", "content": user_message}
            ]
            for user_message in user_messages
        ]

        # tokenize prompts
        self.description_tokenizer.padding_side = "left"
        tokenized_prompts = self.description_tokenizer.apply_chat_template(
            prompt_conversations, 
            add_generation_prompt=True, 
            tokenize=True, 
            padding="longest", 
            return_tensors="pt", 
            return_dict=True
        )
        prompt_input_ids = tokenized_prompts["input_ids"]
        prompt_attention_mask = tokenized_prompts["attention_mask"]

        # tokenize descriptions
        self.description_tokenizer.padding_side = "right"
        tokenized_descriptions = self.description_tokenizer(
            [description + self.description_tokenizer.eos_token for description in descriptions], 
            add_special_tokens=False,  # do not add bos token to the beginning
            truncation=True, 
            padding="longest", 
            max_length=self.max_description_length, 
            return_tensors="pt"
        )
        description_input_ids = tokenized_descriptions["input_ids"]
        description_attention_mask = tokenized_descriptions["attention_mask"]

        # truncate descriptions
        if description_input_ids.size(1) > self.max_description_length:
            description_input_ids = description_input_ids[:, :self.max_description_length]
            description_attention_mask = description_attention_mask[:, :self.max_description_length]

        # prepare labels
        labels = description_input_ids.clone()
        labels[description_attention_mask == 0] = -100

        # assemble
        if self.mode == "train": 
            return {
                "name": accessions,
                "protein_input_ids": sequence_input_ids, 
                "protein_attention_mask": sequence_attention_mask, 
                "input_ids": torch.cat([
                    prompt_input_ids, 
                    description_input_ids, 
                ], dim=1), 
                "attention_mask": torch.cat([
                    prompt_attention_mask, 
                    description_attention_mask, 
                ], dim=1),
                "labels": torch.cat([
                    torch.full_like(
                        prompt_input_ids, 
                        fill_value=-100, 
                    ), 
                    labels,
                ], dim=1), 
                "description_input_ids": description_input_ids,
                "description_attention_mask": description_attention_mask
            }

        elif self.mode == "inference":
            return {
                "name": accessions,
                "protein_input_ids": sequence_input_ids, 
                "protein_attention_mask": sequence_attention_mask, 
                "input_ids": prompt_input_ids, 
                "attention_mask": prompt_attention_mask, 
                "description_input_ids": description_input_ids, 
            }

        else: 
            raise ValueError(f"Invalid mode: {self.mode}")


class ModalityAdapterConfig(PretrainedConfig):
    """Configuration for the modality adapter that bridges ESM and LLaMA."""
    model_type = "modality_adapter"

    def __init__(
        self, 
        input_dim: int, 
        intermediate_dim: int,
        output_dim: int, 
        dropout_rate: float = 0.3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate


class ModalityAdapter(PreTrainedModel):
    """Adapter module that projects ESM embeddings to LLaMA embedding space."""
    config_class = ModalityAdapterConfig

    def __init__(self, config: ModalityAdapterConfig):
        super().__init__(config)
        self.config = config
        self.fc1 = torch.nn.Linear(config.input_dim, config.intermediate_dim)
        self.fc2 = torch.nn.Linear(config.intermediate_dim, config.output_dim)
        self.activation = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(p=config.dropout_rate)
        self.post_init()  # initialize weights and apply final processing

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        # input: (bsz, seq_len, input_dim)
        hidden_states = self.activation(self.fc1(hidden_states))
        hidden_states = self.dropout(hidden_states)
        # interm: (bsz, seq_len, interm_dim)
        hidden_states = self.activation(self.fc2(hidden_states))
        hidden_states = self.dropout(hidden_states)
        # ADD: Epsilon for numerical stability
        hidden_states = torch.nn.functional.normalize(hidden_states, p=2, dim=-1, eps=1e-6)
        return hidden_states  # (bsz, seq_len, output_dim)


class Esm2LlamaInstructConfig(PretrainedConfig):
    """Configuration for the combined ESM2-LLaMA instruction-following model."""
    model_type = "esm2llama_instruct"

    def __init__(
        self, 
        esm_config: Optional[Union[EsmConfig, Dict]] = None, 
        adapter_config: Optional[Union[ModalityAdapterConfig, Dict]] = None,
        llama_config: Optional[Union[LlamaConfig, Dict]] = None, 
        placeholder_id: int = 128003, 
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
            
        self.placeholder_id = placeholder_id


class Esm2LlamaInstructForCausalLM(PreTrainedModel):
    """Combined ESM2 encoder + adapter + LLaMA decoder for protein-to-text generation."""
    config_class = Esm2LlamaInstructConfig

    def __init__(
        self, 
        config: Optional[Esm2LlamaInstructConfig] = None, 
        esm_encoder: Optional[EsmModel] = None, 
        adapter: Optional[ModalityAdapter] = None,
        llama_decoder: Optional[LlamaForCausalLM] = None, 
        **kwargs
    ):
        if config is not None:
            super().__init__(config)
            self.esm_encoder = EsmModel(
                config.esm_config, 
                add_pooling_layer=False
            )
            self.adapter = ModalityAdapter(config.adapter_config)
            self.llama_decoder = LlamaForCausalLM(config.llama_config)
        else: 
            config = Esm2LlamaInstructConfig(
                esm_config=esm_encoder.config,
                adapter_config=adapter.config,
                llama_config=llama_decoder.config, 
                **kwargs  # override standalone attributes
            ) 
            super().__init__(config)
            self.esm_encoder = esm_encoder
            self.adapter = adapter
            self.llama_decoder = llama_decoder
            
    def prepare_decoder_inputs(
        self, 
        input_ids: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None, 
    ): 
        """Prepare decoder inputs by replacing placeholder tokens with encoder outputs."""

        # preparation
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
        # replacement
        placeholder_mask = input_ids == self.config.placeholder_id
        encoder_mask = encoder_attention_mask.bool()
        inputs_embeds[placeholder_mask] = encoder_hidden_states[encoder_mask]
        return inputs_embeds, attention_mask

    def forward(
        self, 
        # chat template text inputs
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        labels: Optional[torch.LongTensor] = None,
        # protein amino-acid sequence inputs
        protein_input_ids: Optional[torch.LongTensor] = None,
        protein_attention_mask: Optional[torch.LongTensor] = None,
        protein_position_ids: Optional[torch.LongTensor] = None, 
        protein_head_mask: Optional[torch.LongTensor] = None,
        protein_inputs_embeds: Optional[torch.FloatTensor] = None,
        # behavior control arguments
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_encoder_outputs: bool = False,
        return_adapter_outputs: bool = False, 
        return_decoder_inputs: bool = False,
        cache_position: Optional[torch.LongTensor] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]: 
        """Forward pass through ESM encoder, adapter, and LLaMA decoder."""

        # esm_encoder forward
        encoder_output = self.esm_encoder(
            input_ids=protein_input_ids,
            attention_mask=protein_attention_mask,
            position_ids=protein_position_ids,
            head_mask=protein_head_mask,
            inputs_embeds=protein_inputs_embeds,
            use_cache=False, # because config.esm_config.is_decoder=False
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        encoder_hidden_states = encoder_output[0]
        encoder_attention_mask = protein_attention_mask
        if return_encoder_outputs:
            return encoder_output
        # adapter forward
        adapter_output = self.adapter(encoder_hidden_states)
        if return_adapter_outputs:
            return adapter_output, encoder_attention_mask
        # decoder input preparation
        inputs_embeds, attention_mask = self.prepare_decoder_inputs(
            input_ids=input_ids, 
            encoder_hidden_states=adapter_output, 
            attention_mask=attention_mask, 
            encoder_attention_mask=encoder_attention_mask, 
        )
        if return_decoder_inputs:
            return inputs_embeds, attention_mask
        # llama_decoder forward
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
        """Generate text from protein embeddings."""

        # get decoder inputs
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
        # do generate on llama_decoder
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
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
