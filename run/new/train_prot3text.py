import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from transformers import EsmModel, LlamaForCausalLM
from transformers import AutoTokenizer, EsmForProteinFolding

from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import StepLR

from typing import Dict, Any, Union
from peft import get_peft_model, LoraConfig
from datetime import datetime
from tqdm import tqdm
import pickle
import wandb

import torch
import torch.distributed as dist
import json

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


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=5, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, val_loss, epoch):
        score = -val_loss if self.mode == 'min' else val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            
        return self.early_stop

class MetricsLogger:
    """Logger for tracking training metrics."""
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.metrics_history = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rate': [],
            'grad_norm': [],
            'fusion_alpha': [],
            'epoch': []
        }
            
    def log_epoch(self, epoch, train_loss, eval_loss, lr, grad_norm, fusion_alpha=None):
        """Log metrics for current epoch."""
        self.metrics_history['epoch'].append(epoch)
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['eval_loss'].append(eval_loss)
        self.metrics_history['learning_rate'].append(lr)
        self.metrics_history['grad_norm'].append(grad_norm)
        if fusion_alpha is not None:
            self.metrics_history['fusion_alpha'].append(fusion_alpha)
        
        print(f"📊 Epoch {epoch} - Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}, LR: {lr:.2e}, Grad Norm: {grad_norm:.4f}", end="")
        if fusion_alpha is not None:
            print(f", Fusion Alpha: {fusion_alpha:.10f}")
        else:
            print()
        
        # Log to wandb
        log_dict = {
            'epoch': epoch,
            'train/loss': train_loss,
            'eval/loss': eval_loss,
            'train/learning_rate': lr,
            'train/grad_norm': grad_norm,
        }
        if fusion_alpha is not None:
            log_dict['model/fusion_alpha'] = fusion_alpha
        
        if wandb.run is not None:
            wandb.log(log_dict, step=epoch)
        
        # Save to JSON after each epoch
        self.save_json()
        
    def save_json(self):
        """Save metrics history to JSON file."""
        json_path = os.path.join(self.save_dir, 'metrics_history.json')
        with open(json_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

def train_epoch(
    rank: int,
    current_epoch: int,
    model: Union[DistributedDataParallel, FullyShardedDataParallel],
    dataloader: DataLoader,
    optimizer: Optimizer,
    args: Dict[str, Any]
):
    model.train()
    ddp_loss = torch.zeros(2).to(rank) # [0] for acc. loss and [1] for num. of seen batches
    ddp_gradnorm = torch.zeros(2).to(rank) # [0] for acc. gradnorm and [1] for num. of passed steps
    optimizer.zero_grad()  # erase accumulated gradients from last epoch

    t = tqdm(iter(dataloader))
    for batch_idx, data_batch in enumerate(t):

        loss = model(
            input_ids=data_batch["input_ids"].to(rank),
            attention_mask=data_batch["attention_mask"].to(rank),
            labels=data_batch["labels"].to(rank),
            protein_input_ids=data_batch["protein_input_ids"].to(rank),
            protein_attention_mask=data_batch["protein_attention_mask"].to(rank),
            use_cache=False,
            output_attentions=False, 
            output_hidden_states=False,
            return_dict=False,
        )[0]

        # rescale loss for consistency with different gradient accumulation steps
        loss = loss / args["gradient_accumulation_steps"]

        # summary current batch
        t.set_postfix({
            "mode": "train",
            "epoch": f"{current_epoch}/{args['num_epochs']}",
            "batch_loss": loss.item() * args["gradient_accumulation_steps"],
            "device": f"rank:{rank}"
        })
        ddp_loss[0] += loss.item() * args["gradient_accumulation_steps"]
        ddp_loss[1] += 1  # the loss is the weighted mean of the output of every batch

        loss.backward()  

        # update weights by loss if accumulation step is reached
        if (batch_idx + 1) % args["gradient_accumulation_steps"] == 0: 
            gradnorm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=(
                    float("inf") 
                    if args["gradient_clipping"] is None 
                    else args["gradient_clipping"]
                )
            )
            ddp_gradnorm[0] += gradnorm
            ddp_gradnorm[1] += 1

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    # summary current epoch
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(ddp_gradnorm, op=dist.ReduceOp.SUM)
    
    avg_loss = (ddp_loss[0] / ddp_loss[1]).item()
    avg_gradnorm = (ddp_gradnorm[0] / ddp_gradnorm[1]).item() if ddp_gradnorm[1] > 0 else 0.0
    
    total_epochs = args["resume_epoch"] + args["num_epochs"]
    if rank == 0:
        # Access the active fusion layer (PEFT wraps it in modules_to_save)
        fusion_layer = model.module.base_model.model.fusion_layer
        if hasattr(fusion_layer, 'modules_to_save'):
            fusion_layer = fusion_layer.modules_to_save.default
        fusion_alpha = fusion_layer.get_alpha()
        print(
            f"[epoch={current_epoch}/{total_epochs}, "
            f"train_loss={avg_loss:.4f}, "
            f"epoch_lr={optimizer.param_groups[0]['lr']:.2e}, "
            f"epoch_gradnorm={avg_gradnorm:.4f}, "
            f"fusion_alpha={fusion_alpha:.10f}]"
        )
        # NaN detection
        if avg_loss != avg_loss:
            raise ValueError(
                "NaN detected in the training loss of the epoch, training interrupted."
            )
    
    return avg_loss, avg_gradnorm


def eval_epoch(
        rank: int,
        current_epoch: int, 
        model: Union[DistributedDataParallel, FullyShardedDataParallel],
        dataloader: DataLoader,
        args: Dict[str, Any]
):

    model.eval()
    ddp_loss = torch.zeros(2).to(rank)  # [0] for acc. loss and [1] for num. of seen batches
    
    # Calculate total end epoch for display
    total_epochs = args["resume_epoch"] + args["num_epochs"]

    t = tqdm(iter(dataloader))
    for data_batch in t:
        with torch.no_grad():
            loss = model(
                input_ids=data_batch["input_ids"].to(rank),
                attention_mask=data_batch["attention_mask"].to(rank),
                labels=data_batch["labels"].to(rank),
                protein_input_ids=data_batch["protein_input_ids"].to(rank),
                protein_attention_mask=data_batch["protein_attention_mask"].to(rank),
                use_cache=False,
                output_attentions=False, 
                output_hidden_states=False,
                return_dict=False,
            )[0]

            t.set_postfix({
                "mode": "eval",
                "epoch": f"{current_epoch}/{total_epochs}",
                "batch_loss": loss.item(),
                "device": f"rank:{rank}"
            })
            ddp_loss[0] += loss.item()
            ddp_loss[1] += 1

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    avg_loss = (ddp_loss[0] / ddp_loss[1]).item()
    
    total_epochs = args["resume_epoch"] + args["num_epochs"]
    if rank == 0:
        print(
            f"[epoch={current_epoch}/{total_epochs}, "
            f"eval_loss={avg_loss:.4f}]"
        )
    
    return avg_loss

def train_on_device(rank: int, world_size: int, args: Dict):

    setup(rank, world_size)

    # Initialize wandb on rank 0
    if rank == 0:
        wandb_kwargs = {
            "project": args["wandb_project"],
            "config": args,
            "tags": ["dual-path", "seq+struct", f"K{args['codebook_k']}"],
        }
        if args.get("wandb_run_id"):
            wandb_kwargs["id"] = args["wandb_run_id"]
            wandb_kwargs["resume"] = "allow"
        else:
            wandb_kwargs["name"] = args["wandb_name"]
        wandb.init(**wandb_kwargs)
        print(f"✅ wandb initialized: {args['wandb_project']}/{args['wandb_name']}")
        if args.get("wandb_run_id"):
            print(f"   Resuming run ID: {args['wandb_run_id']}")

    # DATASET PREPARATION
    esm_tokenizer = AutoTokenizer.from_pretrained(args["esm_path"])
    llama_tokenizer = AutoTokenizer.from_pretrained(
        args["llama_path"], 
        pad_token='<|reserved_special_token_0|>'
    )

    train_dataset = Prot2TextLightDataset(
        csv_path=os.path.join(args["root_csv_dir"], f"{args['train_split']}.csv"),
    )
    if args["debug_trim_train_split"]:
        train_dataset.data = train_dataset.data[:args["debug_trim_train_split"]]

    train_sampler = DistributedSampler(
        train_dataset, 
        rank=rank, 
        num_replicas=world_size, 
        shuffle=True
    )

    train_collater = Prot2TextLightCollater(
        sequence_tokenizer=esm_tokenizer,
        description_tokenizer=llama_tokenizer,
        mode="train", 
        include_text_fields=args["include_text_fields"],
        name_dropout=args["name_dropout"],
        taxonomy_dropout=args["taxonomy_dropout"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args["batch_size_per_device"],
        sampler=train_sampler,
        collate_fn=train_collater,
        num_workers=4,  # parallel CPU cores used for data loading
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )
    print(f"Train dataset loaded on rank:{rank}")

    eval_dataset = Prot2TextLightDataset(
        csv_path=os.path.join(args["root_csv_dir"], f"{args['eval_split']}.csv"),
    )
    if args["debug_trim_eval_split"]:
        eval_dataset.data = eval_dataset.data[:args["debug_trim_eval_split"]]

    eval_sampler = DistributedSampler(
        eval_dataset, 
        rank=rank, 
        num_replicas=world_size, 
        shuffle=False
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args["batch_size_per_device"],
        sampler=eval_sampler,
        collate_fn=train_collater,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )
    print(f"Eval dataset loaded on rank:{rank}")

    # MODEL PREPARATION
    torch.cuda.set_device(rank)

    esm_encoder = EsmModel.from_pretrained(
        args["esm_path"], 
        add_pooling_layer=False,
        torch_dtype=torch.bfloat16, 
        device_map="cpu"
    )
    
    # Load ESMFold for structure branch
    if rank == 0:
        print("Loading ESMFold encoder...")
    esmfold_encoder = EsmForProteinFolding.from_pretrained(
        args["esmfold_path"],
        torch_dtype=torch.bfloat16,
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
    codebook_centroids = torch.from_numpy(codebook_data['kmeans'].cluster_centers_).to(torch.bfloat16)
    if rank == 0:
        print(f"✅ Codebook loaded: K={args['codebook_k']}, centroids shape={codebook_centroids.shape}")
    
    llama_decoder = LlamaForCausalLM.from_pretrained(
        args["llama_path"], 
        torch_dtype=torch.bfloat16, 
        device_map="cpu"
    )

    adapter_config = ModalityAdapterConfig(
        input_dim=esm_encoder.config.hidden_size,
        intermediate_dim=2048,
        output_dim=llama_decoder.config.hidden_size,
    )

    adapter = ModalityAdapter(adapter_config)
    adapter.to(torch.bfloat16)

    # Structure adapter (same architecture, different weights)
    structure_adapter_config = ModalityAdapterConfig(
        input_dim=esmfold_encoder.esm.config.hidden_size,
        intermediate_dim=2048,
        output_dim=llama_decoder.config.hidden_size,
    )
    structure_adapter = ModalityAdapter(structure_adapter_config)
    structure_adapter.to(torch.bfloat16)
    
    # Fusion layer
    fusion_layer = FusionLayer(alpha_init=args["structure_alpha"])
    fusion_layer.to(torch.bfloat16)

    model = Esm2LlamaInstructForCausalLM(
        esm_encoder=esm_encoder,
        adapter=adapter,
        esmfold_encoder=esmfold_encoder,
        structure_adapter=structure_adapter,
        fusion_layer=fusion_layer,
        llama_decoder=llama_decoder,
        codebook_centroids=codebook_centroids,
    )

    lora_rank = 32

    # Check if we're resuming from a checkpoint
    if args["resume_checkpoint_dir"] is not None and os.path.exists(args["resume_checkpoint_dir"]):
        if rank == 0:
            print(f"🔄 Resuming from checkpoint: {args['resume_checkpoint_dir']}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args["resume_checkpoint_dir"])
        if rank == 0:
            print(f"✅ Loaded PEFT adapter from {args['resume_checkpoint_dir']}")
    else:
        if rank == 0:
            print("Initializing LoRA adapter from scratch")
        lora_config = LoraConfig(
            r=lora_rank, 
            lora_alpha=lora_rank * 2, 
            lora_dropout=0.1,
            bias="none", 
            init_lora_weights=True, 
            target_modules=[
                "self_attn.q_proj", 
                "self_attn.k_proj", 
                "self_attn.v_proj", 
                "self_attn.o_proj", 
                "mlp.gate_proj", 
                "mlp.up_proj", 
                "mlp.down_proj"
            ],  # for llama_decoder 
            modules_to_save=[
                "adapter.fc1", "adapter.fc2",
                "structure_adapter.fc1", "structure_adapter.fc2",
                "fusion_layer"
            ] # train both adapters + fusion fully
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()
    model = model.to(rank)

    model = DistributedDataParallel(
        model,
        find_unused_parameters=True  # Required for PEFT modules_to_save (handles original_module copies)
    )
    print(f"DDP model loaded on rank:{rank}")

    # initialization of the optimizer after wrapping the model
    optimizer = Adam(model.parameters(), lr=args["learning_rate"])
    scheduler = StepLR(optimizer, step_size=1, gamma=args["scheduler_gamma"])
    
    # Load optimizer and scheduler states if resuming
    if args["resume_checkpoint_dir"] is not None and args["resume_epoch"] > 0:
        # Handle both relative and absolute paths
        checkpoint_dir = args["resume_checkpoint_dir"]
        if not os.path.isabs(checkpoint_dir):
            # If relative path, check if it needs 'new/' prefix
            if not os.path.exists(checkpoint_dir) and os.path.exists(os.path.join("new", checkpoint_dir)):
                checkpoint_dir = os.path.join("new", checkpoint_dir)
        
        optimizer_checkpoint_path = os.path.join(
            os.path.dirname(checkpoint_dir),
            f"optimizer_scheduler_checkpoint_{args['resume_epoch']}.pt"
        )
        if os.path.exists(optimizer_checkpoint_path):
            checkpoint = torch.load(optimizer_checkpoint_path, map_location=f'cuda:{rank}')
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if rank == 0:
                print(f"✅ Loaded optimizer/scheduler state from {optimizer_checkpoint_path}")
                print(f"   Current LR after loading: {optimizer.param_groups[0]['lr']:.2e}")
        else:
            if rank == 0:
                print(f"⚠️  Optimizer checkpoint not found: {optimizer_checkpoint_path}")
                print("    Starting with fresh optimizer/scheduler states")

    # Initialize logger and early stopping (only on rank 0)
    logger = None
    early_stopping = None
    
    if rank == 0:
        logger = MetricsLogger(
            save_dir=args["save_checkpoint_dir"]
        )
        early_stopping = EarlyStopping(
            patience=args.get("early_stopping_patience", 5),
            min_delta=args.get("early_stopping_min_delta", 0.001),
            mode='min'
        )

    # core loop of epochs
    start_epoch = args["resume_epoch"] + 1
    end_epoch = args["resume_epoch"] + args["num_epochs"]
    
    if rank == 0:
        if args["resume_epoch"] > 0:
            print(f"\n🚀 Resuming training from epoch {start_epoch} to {end_epoch}")
        else:
            print(f"\n🚀 Starting training from epoch {start_epoch} to {end_epoch}")
    
    for epoch_idx in range(start_epoch, end_epoch + 1):

        train_sampler.set_epoch(epoch=epoch_idx)
        train_loss, grad_norm = train_epoch(
            rank=rank,
            current_epoch=epoch_idx,
            model=model,    
            dataloader=train_loader,
            optimizer=optimizer,
            args=args
        )
        # Only step scheduler if not the first resumed epoch (scheduler already at correct state)
        if not (args["resume_epoch"] > 0 and epoch_idx == start_epoch):
            scheduler.step()
        dist.barrier()  # use a barrier to make sure training is done on all ranks
        
        eval_loss = eval_epoch(
            rank=rank,
            model=model,
            current_epoch=epoch_idx,
            dataloader=eval_loader,
            args=args
        )
        dist.barrier()
        
        # Log metrics and check early stopping (rank 0 only)
        if rank == 0:
            # Access the active fusion layer (PEFT wraps it in modules_to_save)
            fusion_layer = model.module.base_model.model.fusion_layer
            if hasattr(fusion_layer, 'modules_to_save'):
                fusion_layer = fusion_layer.modules_to_save.default
            fusion_alpha = fusion_layer.get_alpha()
            logger.log_epoch(
                epoch=epoch_idx,
                train_loss=train_loss,
                eval_loss=eval_loss,
                lr=optimizer.param_groups[0]['lr'],
                grad_norm=grad_norm,
                fusion_alpha=fusion_alpha
            )
            
            # Check early stopping
            if early_stopping(eval_loss, epoch_idx):
                print(f"\nEarly stopping triggered at epoch {epoch_idx}")
                print(f"Best epoch was {early_stopping.best_epoch} with loss {-early_stopping.best_score:.4f}")
                dist.barrier()
                break

        if (
            epoch_idx == 1 
            or epoch_idx == args["num_epochs"] 
            or epoch_idx % args["save_every_epochs"] == 0
        ):
            if rank == 0:
                adapter_checkpoint_dir = os.path.join(
                    args["save_checkpoint_dir"], 
                    f"adapter_checkpoint_{epoch_idx}"
                )
                model.module.save_pretrained(adapter_checkpoint_dir)
                print(f"Saving {adapter_checkpoint_dir}")

                optimizer_scheduler_checkpoint_path = os.path.join(
                    args["save_checkpoint_dir"], 
                    f"optimizer_scheduler_checkpoint_{epoch_idx}.pt"
                )
                torch.save(
                    {
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    }, 
                    optimizer_scheduler_checkpoint_path
                )
                print(f"Saving {optimizer_scheduler_checkpoint_path}")

            dist.barrier()
    
    # Finish wandb on rank 0
    if rank == 0:
        wandb.finish()
        print("✅ wandb run finished")

    dist.destroy_process_group()

### START ###
if __name__ == "__main__":

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["LOGURU_LEVEL"] = "INFO"

    world_size = torch.cuda.device_count()  # use up all visible GPUs
    random_seed = 42
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    save_checkpoint_dir = f"new/checkpoints_{timestamp}"

    if not os.path.exists(save_checkpoint_dir):
        os.mkdir(save_checkpoint_dir)

    args = {
        "world_size": world_size,
        "random_seed": random_seed,
        "save_checkpoint_dir": save_checkpoint_dir,

        "esm_path": "facebook/esm2_t36_3B_UR50D",
        "esmfold_path": "facebook/esmfold_v1",
        "llama_path": "meta-llama/Llama-3.1-8B-Instruct",
        "root_csv_dir": "data/csv/",
        "train_split": "train",
        "eval_split": "validation",
        "codebook_dir": "data/codebooks",
        "codebook_k": 1024,  #DEBUG
        "structure_alpha": 0.5, #DEBUG

        "num_epochs": 3, #DEBUG
        "save_every_epochs": 1, #DEBUG
        "batch_size_per_device": 4,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-4,
        "scheduler_gamma": 0.8,
        "gradient_clipping": 1.0,

        "include_text_fields": True,
        "debug_trim_train_split": None, #DEBUG
        "debug_trim_eval_split": None, #DEBUG
        "name_dropout": 0.8,
        "taxonomy_dropout": 0.8,

        "early_stopping_patience": 3,
        "early_stopping_min_delta": 0.001,
        
        "wandb_project": "prot3text",
        "wandb_name": f"train_prot3text_{timestamp}",
        "wandb_run_id": None,  #DEBUG
        
        "resume_checkpoint_dir": None,  #DEBUG
        "resume_epoch": 0,  #DEBUG
    }

    torch.multiprocessing.spawn(
        train_on_device, 
        args=(args["world_size"], args),
        nprocs=args["world_size"],
        join=True
    )
