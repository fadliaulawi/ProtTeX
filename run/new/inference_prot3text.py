import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from transformers import EsmModel, LlamaForCausalLM, PreTrainedTokenizer
from transformers import AutoTokenizer, EsmForProteinFolding

from torch.utils.data.distributed import DistributedSampler

from datetime import datetime
from typing import Any, Dict, List
from tqdm import tqdm
from torch.utils.data import DataLoader

import torch
import torch.distributed as dist
import json
import argparse
import pickle

# Import shared classes
from config_prot3text import (
    Prot2TextLightDataset,
    Prot2TextLightCollater,
    ModalityAdapterConfig,
    ModalityAdapter,
    FusionLayer,
    Esm2LlamaInstructConfig,
    Esm2LlamaInstructForCausalLM,
    setup
)

def inference_epoch(
    rank: int, 
    model: Esm2LlamaInstructForCausalLM,
    dataloader: DataLoader,
    llama_tokenizer: PreTrainedTokenizer,
    args: Dict[str, Any]
): 
    model.eval()
    local_names: List[str] = []
    local_predictions: List[str] = []
    local_labels: List[str] = []

    t = tqdm(iter(dataloader))
    for data_batch in t:
        with torch.no_grad(): 

            output = model.generate(
                inputs=data_batch["input_ids"].to(rank),
                attention_mask=data_batch["attention_mask"].to(rank),
                protein_input_ids=data_batch["protein_input_ids"].to(rank),
                protein_attention_mask=data_batch["protein_attention_mask"].to(rank),
                max_new_tokens=args["max_generation_length"],
                eos_token_id=128009, 
                pad_token_id=128002,
                return_dict_in_generate=False,
                num_beams=args["num_beams"],
                length_penalty=args["length_penalty"],
                temperature=args["temperature"], 
                do_sample=args["do_sample"],
                top_p=args["top_p"],
                top_k=args["top_k"]
            )

        local_names.extend(data_batch["name"])
        predicted_texts = llama_tokenizer.batch_decode(output.cpu(), skip_special_tokens=True)
        local_predictions.extend(predicted_texts)
        label_texts = llama_tokenizer.batch_decode(data_batch["description_input_ids"], skip_special_tokens=True)
        local_labels.extend(label_texts)
        t.set_postfix({
            "mode": "inference", 
            "batch_maxlen_gen": output.shape[1], 
            "device": f"rank:{rank}"
        })

    local_json_path = os.path.join(
        args["save_generation_dir"],
        f"generation_{args['save_generation_postfix_identifier']}_rank{rank}.json"
    )
    with open(local_json_path, "w") as file:
        json_dict = {
            name: {"true": label, "pred": prediction}
            for name, label, prediction in zip(local_names, local_labels, local_predictions)
        }
        json.dump(json_dict, file, indent=4)
        print(f"Saving {local_json_path}")

def inference_on_device(rank: int, world_size: int, args: Dict[str, Any]):

    setup(rank, world_size)

    # prepare dataset and dataloader
    esm_tokenizer = AutoTokenizer.from_pretrained(args["esm_path"])
    llama_tokenizer = AutoTokenizer.from_pretrained(
        args["llama_path"], 
        pad_token='<|reserved_special_token_0|>'
    )

    generate_dataset = Prot2TextLightDataset(
        csv_path=os.path.join(args["root_csv_dir"], f"{args['generate_split']}.csv"),
    )
    if args["debug_trim_generate_split"]:
        generate_dataset.data = generate_dataset.data[:args["debug_trim_generate_split"]]

    generate_sampler = DistributedSampler(
        generate_dataset, 
        rank=rank, 
        num_replicas=world_size, 
        shuffle=False
    )
    
    generate_collater = Prot2TextLightCollater(
        sequence_tokenizer=esm_tokenizer,
        description_tokenizer=llama_tokenizer,
        mode="inference",
        include_text_fields=args.get("include_text_fields", True),
        name_dropout=0.0,  # No dropout during inference
        taxonomy_dropout=0.0,  # No dropout during inference
    )
    
    generate_loader = DataLoader(
        generate_dataset,
        batch_size=args["batch_size_per_device"],
        sampler=generate_sampler,
        collate_fn=generate_collater,
        num_workers=2,
        pin_memory=True,
        shuffle=False,
        drop_last=False,  # Keep all samples for inference
    )

    # MODEL PREPARATION
    torch.cuda.set_device(rank)

    esm_encoder = EsmModel.from_pretrained(
        args["esm_path"], 
        add_pooling_layer=False,
        torch_dtype=torch.float16, 
        device_map="cpu"
    )
    
    # Load ESMFold for structure branch
    if rank == 0:
        print("Loading ESMFold encoder...")
    esmfold_encoder = EsmForProteinFolding.from_pretrained(
        args["esmfold_path"],
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    # Freeze ESMFold
    for param in esmfold_encoder.parameters():
        param.requires_grad = False
    if rank == 0:
        print(f"✅ ESMFold loaded and frozen (embedding dim: {esmfold_encoder.esm.config.hidden_size})")
    
    # Load K-means codebook
    codebook_path = os.path.join(args["codebook_dir"], f"structure_codebook_K{args['codebook_k']}.pkl")
    if rank == 0:
        print(f"Loading K-means codebook from {codebook_path}...")
    with open(codebook_path, 'rb') as f:
        codebook_data = pickle.load(f)
    codebook_centroids = torch.from_numpy(codebook_data['kmeans'].cluster_centers_).to(torch.float16)
    if rank == 0:
        print(f"✅ Codebook loaded: K={args['codebook_k']}, centroids shape={codebook_centroids.shape}")
    
    llama_decoder = LlamaForCausalLM.from_pretrained(
        args["llama_path"], 
        torch_dtype=torch.float16, 
        device_map="cpu"
    )

    adapter_config = ModalityAdapterConfig(
        input_dim=esm_encoder.config.hidden_size,
        intermediate_dim=2048,
        output_dim=llama_decoder.config.hidden_size,
    )

    adapter = ModalityAdapter(adapter_config)
    adapter.to(torch.float16)  # Match training dtype

    # Structure adapter (same architecture, different weights)
    structure_adapter_config = ModalityAdapterConfig(
        input_dim=esmfold_encoder.esm.config.hidden_size,
        intermediate_dim=2048,
        output_dim=llama_decoder.config.hidden_size,
    )
    structure_adapter = ModalityAdapter(structure_adapter_config)
    structure_adapter.to(torch.float16)
    
    # Fusion layer
    fusion_layer = FusionLayer(alpha_init=args.get("structure_alpha", 0.5))
    fusion_layer.to(torch.float16)

    model = Esm2LlamaInstructForCausalLM(
        esm_encoder=esm_encoder,
        adapter=adapter,
        esmfold_encoder=esmfold_encoder,
        structure_adapter=structure_adapter,
        fusion_layer=fusion_layer,
        llama_decoder=llama_decoder,
        codebook_centroids=codebook_centroids,
    )

    # Load trained checkpoint
    print(f"Loading adapter checkpoint from {args['load_adapter_checkpoint_dir']}")
    from peft import PeftModel
    model = PeftModel.from_pretrained(
        model,
        args["load_adapter_checkpoint_dir"],
        is_trainable=False
    )
    print("Checkpoint loaded successfully")

    model.print_trainable_parameters()    
    model = model.merge_and_unload()

    model = model.to(rank)
    print(f"Model loaded on rank{rank} (no DDP needed for inference)")

    inference_epoch(
        rank=rank,
        model=model,
        dataloader=generate_loader,
        llama_tokenizer=llama_tokenizer,
        args=args
    )

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Protein-to-text inference with ESM-LLAMA model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained adapter checkpoint directory (e.g., checkpoints_250126_143022/adapter_checkpoint_24)"
    )
    cmd_args = parser.parse_args()

    # Validate checkpoint exists
    if not os.path.exists(cmd_args.checkpoint):
        raise FileNotFoundError(
            f"ERROR: Adapter checkpoint directory not found: {cmd_args.checkpoint}\n"
            f"Please verify the path exists and contains the trained adapter weights."
        )

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["LOGURU_LEVEL"] = "INFO"
    
    world_size = torch.cuda.device_count()  # use up all available devices across nodes
    random_seed = 42
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    save_generation_dir = f"new/inference_{timestamp}"
    
    if not os.path.exists(save_generation_dir):
        os.makedirs(save_generation_dir)
    
    args = {
        "world_size": world_size,
        "random_seed": random_seed,
        "save_generation_dir": save_generation_dir,
        "save_generation_postfix_identifier": timestamp,

        "esm_path": "facebook/esm2_t36_3B_UR50D",
        "esmfold_path": "facebook/esmfold_v1",
        "llama_path": "meta-llama/Llama-3.1-8B-Instruct",
        "root_csv_dir": "data/csv/",
        "generate_split": "test",
        "codebook_dir": "data/codebooks",
        "codebook_k": 1024, # DEBUG

        "load_adapter_checkpoint_dir": cmd_args.checkpoint,
        "batch_size_per_device": 4,
        "max_generation_length": 512,
        "num_beams": 4,
        "length_penalty": 1.0,
        "temperature": 1.0,
        "do_sample": False,
        "top_p": 0.9,
        "top_k": 50,

        "include_text_fields": True,
        "debug_trim_generate_split": None,
    }

    torch.multiprocessing.spawn(
        inference_on_device, 
        args=(args["world_size"], args),
        nprocs=args["world_size"],
        join=True
    )

    # Combine results from all ranks    
    combined_results = {}
    for rank in range(args["world_size"]):
        rank_file = os.path.join(
            args["save_generation_dir"],
            f"generation_{args['save_generation_postfix_identifier']}_rank{rank}.json"
        )
        if os.path.exists(rank_file):
            with open(rank_file, 'r') as f:
                rank_data = json.load(f)
                combined_results.update(rank_data)
            print(f"✅ Loaded {len(rank_data)} samples from rank {rank}")
            os.remove(rank_file)
    
    combined_file = os.path.join(args["save_generation_dir"], "combined_predictions.json")
    with open(combined_file, 'w') as f:
        json.dump(combined_results, f, indent=4)
    
    print(f"\n✅ Combined {len(combined_results)} total predictions")
    print(f"📁 Saved to: {combined_file}")
    print("="*70)