"""
Model Configuration for Unified LoRA Training Script
Maps model types to their specific configurations including:
- Model names
- Chat template formats
- Loading requirements
- Output directories
"""

from typing import Dict, Callable


class ModelConfig:
    """Configuration for a specific model type"""
    
    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool,
        prompt_builder: Callable,
        output_dir_suffix: str,
        wandb_project: str,
        checkpoint_prefix: str,
        hidden_dim: int = 4096
    ):
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.prompt_builder = prompt_builder
        self.output_dir_suffix = output_dir_suffix
        self.wandb_project = wandb_project
        self.checkpoint_prefix = checkpoint_prefix
        self.hidden_dim = hidden_dim


def build_llama_prompt(question: str, protein_seq: str, function_text: str) -> tuple:
    """
    Build Llama 3.1 chat template prompt parts.
    
    Returns:
        (part1, part2, part3, part4) - Four parts of the prompt
    """
    part1 = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}\n\nSequence: {protein_seq}\n"
    part2 = "\nStructure: "
    part3 = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    part4 = f"{function_text}<|eot_id|>"
    return part1, part2, part3, part4


def build_qwen_prompt(question: str, protein_seq: str, function_text: str) -> tuple:
    """
    Build Qwen chat template prompt parts.
    
    Returns:
        (part1, part2, part3, part4) - Four parts of the prompt
    """
    part1 = f"<|im_start|>user\n{question}\n\nSequence: {protein_seq}\n"
    part2 = "Structure:\n"
    part3 = "<|im_end|>\n<|im_start|>assistant\n"
    part4 = f"{function_text}<|im_end|>"
    return part1, part2, part3, part4


def build_deepseek_prompt(question: str, protein_seq: str, function_text: str) -> tuple:
    """
    Build DeepSeek chat template prompt parts.
    
    Returns:
        (part1, part2, part3, part4) - Four parts of the prompt
    """
    part1 = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{question}\n\nSequence: {protein_seq}\n"
    part2 = "Structure:\n"
    part3 = "<|end_header_id|><|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    part4 = f"{function_text}<|eot_id|>"
    return part1, part2, part3, part4


# Model configurations
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    'llama': ModelConfig(
        model_name='meta-llama/Llama-3.1-8B-Instruct',
        trust_remote_code=False,
        prompt_builder=build_llama_prompt,
        output_dir_suffix='llama',
        wandb_project='prottex-llama-lora',
        checkpoint_prefix='best_llama',
        hidden_dim=4096
    ),
    
    'qwen': ModelConfig(
        model_name='Qwen/Qwen2.5-7B-Instruct',
        trust_remote_code=True,
        prompt_builder=build_qwen_prompt,
        output_dir_suffix='qwen',
        wandb_project='prottex-qwen-lora',
        checkpoint_prefix='best_qwen',
        hidden_dim=3584  # Qwen 2.5 7B uses 3584-dim embeddings
    ),
    
    'qwen2.7': ModelConfig(
        model_name='Qwen/Qwen2.7-7B-Instruct',
        trust_remote_code=True,
        prompt_builder=build_qwen_prompt,
        output_dir_suffix='qwen2.7',
        wandb_project='prottex-qwen-lora',
        checkpoint_prefix='best_qwen',
        hidden_dim=3584  # Qwen 2.7 7B uses 3584-dim embeddings
    ),
    
    'deepseek-v2': ModelConfig(
        model_name='deepseek-ai/DeepSeek-V2-Chat',
        trust_remote_code=True,
        prompt_builder=build_deepseek_prompt,
        output_dir_suffix='deepseek',
        wandb_project='prottex-deepseek-lora',
        checkpoint_prefix='best_deepseek',
        hidden_dim=4096
    ),
    
    'deepseek-r1': ModelConfig(
        model_name='deepseek-ai/DeepSeek-R1',
        trust_remote_code=True,
        prompt_builder=build_deepseek_prompt,
        output_dir_suffix='deepseek',
        wandb_project='prottex-deepseek-lora',
        checkpoint_prefix='best_deepseek',
        hidden_dim=4096
    ),
    
    'deepseek-r1-distill': ModelConfig(
        model_name='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        trust_remote_code=True,
        prompt_builder=build_deepseek_prompt,
        output_dir_suffix='deepseek',
        wandb_project='prottex-deepseek-lora',
        checkpoint_prefix='best_deepseek',
        hidden_dim=4096
    ),
}


def get_model_config(model_type: str) -> ModelConfig:
    """
    Get configuration for a model type.
    
    Args:
        model_type: One of 'llama', 'qwen', 'qwen2.7', 'deepseek-v2', 'deepseek-r1', 'deepseek-r1-distill'
    
    Returns:
        ModelConfig object
    
    Raises:
        ValueError: If model_type is not supported
    """
    model_type_lower = model_type.lower()
    
    if model_type_lower not in MODEL_CONFIGS:
        available = ', '.join(MODEL_CONFIGS.keys())
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Available options: {available}"
        )
    
    return MODEL_CONFIGS[model_type_lower]


def list_available_models() -> list:
    """Return list of available model types"""
    return list(MODEL_CONFIGS.keys())

