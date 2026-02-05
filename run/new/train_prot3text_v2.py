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

# Import V2 components with enhanced fusion
from config_prot3text_v2 import (
    Prot2TextLightDataset,
    Prot2TextLightCollater,
    ModalityAdapterConfig,
    ModalityAdapter,
    CrossAttentionFusion,
    GatedFusion,
    Esm2LlamaInstructV2ForCausalLM,
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
    
    def state_dict(self):
        """Return state dictionary for checkpointing."""
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop,
            'best_epoch': self.best_epoch,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'mode': self.mode,
        }
    
    def load_state_dict(self, state_dict):
        """Load state from checkpoint."""
        self.counter = state_dict['counter']
        self.best_score = state_dict['best_score']
        self.early_stop = state_dict['early_stop']
        self.best_epoch = state_dict['best_epoch']
        self.patience = state_dict['patience']
        self.min_delta = state_dict['min_delta']
        self.mode = state_dict['mode']

class MetricsLogger:
    """Logger for tracking training metrics."""
    
    def __init__(self, save_dir, load_existing=True):
        self.save_dir = save_dir
        self.metrics_history = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rate': [],
            'grad_norm': [],
            'alpha_mean': [],
            'alpha_std': [],
            'epoch': []
        }
        
        # Try to load existing metrics history if resuming
        if load_existing:
            json_path = os.path.join(self.save_dir, 'metrics_history.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    self.metrics_history = json.load(f)
                print(f"✅ Loaded {len(self.metrics_history.get('epoch', []))} epochs of metrics history")
            
    def log_epoch(self, epoch, train_loss, eval_loss, lr, grad_norm, alpha_stats=None):
        """Log metrics for current epoch."""
        self.metrics_history['epoch'].append(epoch-1)
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['eval_loss'].append(eval_loss)
        self.metrics_history['learning_rate'].append(lr)
        self.metrics_history['grad_norm'].append(grad_norm)
        if alpha_stats is not None:
            self.metrics_history['alpha_mean'].append(alpha_stats['mean'])
            self.metrics_history['alpha_std'].append(alpha_stats['std'])
        
        print(f"📊 Epoch {epoch-1} - Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}, LR: {lr:.2e}, Grad Norm: {grad_norm:.4f}", end="")
        if alpha_stats is not None:
            print(f", Alpha: {alpha_stats['mean']:.6f}±{alpha_stats['std']:.6f} [{alpha_stats['min']:.6f}-{alpha_stats['max']:.6f}]")
        else:
            print()
        
        # Log to wandb
        log_dict = {
            'epoch': epoch-1,
            'train/loss': train_loss,
            'eval/loss': eval_loss,
            'train/learning_rate': lr,
            'train/grad_norm': grad_norm,
        }
        if alpha_stats is not None:
            log_dict.update({
                'model/alpha_mean': alpha_stats['mean'],
                'model/alpha_std': alpha_stats['std'],
                'model/alpha_min': alpha_stats['min'],
                'model/alpha_max': alpha_stats['max'],
            })
        
        # Only log to wandb if it's initialized (rank 0)
        if wandb.run is not None:
            print(f"✅ Logging metrics to wandb for epoch {epoch-1}")
            wandb.log(log_dict, step=epoch-1)
        else:
            print("⚠️  wandb not initialized, skipping logging")
        
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
    ddp_loss = torch.zeros(2).to(rank)
    ddp_gradnorm = torch.zeros(2).to(rank)
    optimizer.zero_grad()

    # Calculate total end epoch for display
    total_epochs = args["resume_epoch"] + args["num_epochs"]

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

        loss = loss / args["gradient_accumulation_steps"]

        t.set_postfix({
            "mode": "train",
            "epoch": f"{current_epoch}/{total_epochs}",
            "batch_loss": loss.item() * args["gradient_accumulation_steps"],
            "device": f"rank:{rank}"
        })
        ddp_loss[0] += loss.item() * args["gradient_accumulation_steps"]
        ddp_loss[1] += 1

        loss.backward()  

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

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(ddp_gradnorm, op=dist.ReduceOp.SUM)
    
    avg_loss = (ddp_loss[0] / ddp_loss[1]).item()
    avg_gradnorm = (ddp_gradnorm[0] / ddp_gradnorm[1]).item() if ddp_gradnorm[1] > 0 else 0.0
    
    total_epochs = args["resume_epoch"] + args["num_epochs"]
    if rank == 0:
        print(
            f"[epoch={current_epoch}/{total_epochs}, "
            f"train_loss={avg_loss:.4f}, "
            f"epoch_lr={optimizer.param_groups[0]['lr']:.2e}, "
            f"epoch_gradnorm={avg_gradnorm:.4f}]"
        )
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
    ddp_loss = torch.zeros(2).to(rank)
    
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

    # Auto-detect resume_epoch from checkpoint_dir if not set
    if args.get("resume_checkpoint_dir") is not None:
        checkpoint_dir = args["resume_checkpoint_dir"]
        # Extract epoch from directory name like "adapter_checkpoint_3" -> 3
        import re
        match = re.search(r'adapter_checkpoint_(\d+)', checkpoint_dir)
        if match:
            args["resume_epoch"] = int(match.group(1))
            if rank == 0:
                print(f"📌 Auto-detected resume_epoch={args['resume_epoch']} from {checkpoint_dir}")
        elif "resume_epoch" not in args or args["resume_epoch"] == 0:
            raise ValueError(
                f"Could not extract epoch number from resume_checkpoint_dir: {checkpoint_dir}\n"
                "Expected format: .../adapter_checkpoint_<epoch>"
            )
        
        # Load wandb_run_id from checkpoint BEFORE initializing wandb
        if args["resume_epoch"] > 0 and args.get("wandb_run_id") is None:
            # Handle both relative and absolute paths
            if not os.path.isabs(checkpoint_dir):
                if not os.path.exists(checkpoint_dir) and os.path.exists(os.path.join("new", checkpoint_dir)):
                    checkpoint_dir = os.path.join("new", checkpoint_dir)
            
            optimizer_checkpoint_path = os.path.join(
                os.path.dirname(checkpoint_dir),
                f"optimizer_scheduler_checkpoint_{args['resume_epoch']}.pt"
            )
            if os.path.exists(optimizer_checkpoint_path):
                checkpoint = torch.load(optimizer_checkpoint_path, map_location='cpu')
                if "wandb_run_id" in checkpoint:
                    args["wandb_run_id"] = checkpoint["wandb_run_id"]
                    if rank == 0:
                        print(f"📌 Loaded wandb_run_id from checkpoint: {args['wandb_run_id']}")
                del checkpoint  # Free memory
    else:
        if "resume_epoch" not in args:
            args["resume_epoch"] = 0

    # Initialize wandb on rank 0
    if rank == 0:
        wandb_kwargs = {
            "project": args["wandb_project"],
            "config": args,
            "tags": ["v2", "cross-attention", "gated-fusion", f"K{args['codebook_k']}"],
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
        num_workers=4,
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
    
    if rank == 0:
        print("Loading ESMFold encoder...")
    esmfold_encoder = EsmForProteinFolding.from_pretrained(
        args["esmfold_path"],
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )
    for param in esmfold_encoder.parameters():
        param.requires_grad = False
    if rank == 0:
        print(f"✅ ESMFold loaded and frozen (embedding dim: {esmfold_encoder.esm.config.hidden_size})")
    
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

    # Adapters
    adapter_config = ModalityAdapterConfig(
        input_dim=esm_encoder.config.hidden_size,
        intermediate_dim=2048,
        output_dim=llama_decoder.config.hidden_size,
    )
    adapter = ModalityAdapter(adapter_config)
    adapter.to(torch.bfloat16)

    structure_adapter_config = ModalityAdapterConfig(
        input_dim=esmfold_encoder.esm.config.hidden_size,
        intermediate_dim=2048,
        output_dim=llama_decoder.config.hidden_size,
    )
    structure_adapter = ModalityAdapter(structure_adapter_config)
    structure_adapter.to(torch.bfloat16)
    
    # Enhanced fusion components
    cross_attention_fusion = CrossAttentionFusion(
        hidden_dim=llama_decoder.config.hidden_size,
        num_heads=args["fusion_num_heads"],
        dropout=args["fusion_dropout"]
    )
    cross_attention_fusion.to(torch.bfloat16)
    
    gated_fusion = GatedFusion(
        hidden_dim=llama_decoder.config.hidden_size,
        gate_hidden_dim=args["gate_hidden_dim"],
        dropout=args["fusion_dropout"]
    )
    gated_fusion.to(torch.bfloat16)

    model = Esm2LlamaInstructV2ForCausalLM(
        esm_encoder=esm_encoder,
        adapter=adapter,
        esmfold_encoder=esmfold_encoder,
        structure_adapter=structure_adapter,
        cross_attention_fusion=cross_attention_fusion,
        gated_fusion=gated_fusion,
        llama_decoder=llama_decoder,
        codebook_centroids=codebook_centroids,
    )

    lora_rank = 32

    # Always initialize LoRA config first (even when resuming)
    lora_config = LoraConfig(
        r=lora_rank, 
        lora_alpha=lora_rank * 2, 
        lora_dropout=0.2,
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
        ],
        modules_to_save=[
            "adapter",
            "structure_adapter",
            "cross_attention_fusion",
            "gated_fusion"
        ]
    )
    model = get_peft_model(model, lora_config)
    
    # Load checkpoint weights if resuming
    if args["resume_checkpoint_dir"] is not None:
        if not os.path.exists(args["resume_checkpoint_dir"]):
            raise FileNotFoundError(f"resume_checkpoint_dir not found: {args['resume_checkpoint_dir']}")
        if rank == 0:
            print(f"🔄 Loading adapter weights from checkpoint: {args['resume_checkpoint_dir']}")
        
        # Load the adapter weights into the already-initialized PEFT model
        from peft import set_peft_model_state_dict
        import glob
        
        adapter_model_path = os.path.join(args["resume_checkpoint_dir"], "adapter_model.bin")
        if os.path.exists(adapter_model_path):
            adapter_state_dict = torch.load(adapter_model_path, map_location="cpu")
            set_peft_model_state_dict(model, adapter_state_dict)
            if rank == 0:
                print(f"✅ Loaded adapter weights from {adapter_model_path}")
        else:
            # Try safetensors format
            adapter_model_path = os.path.join(args["resume_checkpoint_dir"], "adapter_model.safetensors")
            if os.path.exists(adapter_model_path):
                from safetensors.torch import load_file
                adapter_state_dict = load_file(adapter_model_path)
                set_peft_model_state_dict(model, adapter_state_dict)
                if rank == 0:
                    print(f"✅ Loaded adapter weights from {adapter_model_path}")
            else:
                raise FileNotFoundError(f"No adapter weights found in {args['resume_checkpoint_dir']}")
    else:
        if rank == 0:
            print("✅ Initialized LoRA adapter from scratch")

    model.print_trainable_parameters()
    model = model.to(rank)

    model = DistributedDataParallel(
        model,
        find_unused_parameters=True  # Required for PEFT modules_to_save (handles original_module copies)
    )
    print(f"DDP model loaded on rank:{rank}")

    optimizer = Adam(model.parameters(), lr=args["learning_rate"])
    scheduler = StepLR(optimizer, step_size=1, gamma=args["scheduler_gamma"])
    
    # Load optimizer and scheduler states if resuming
    loaded_checkpoint = None
    if args["resume_epoch"] > 0:
        if args["resume_checkpoint_dir"] is None:
            raise ValueError("resume_epoch > 0 requires resume_checkpoint_dir to be set.")
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
            loaded_checkpoint = torch.load(optimizer_checkpoint_path, map_location=f'cuda:{rank}')
            optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(loaded_checkpoint["scheduler_state_dict"])
            
            if rank == 0:
                print(f"✅ Loaded optimizer/scheduler state from {optimizer_checkpoint_path}")
                print(f"   Resuming from epoch {loaded_checkpoint.get('epoch', args['resume_epoch'])}")
                print(f"   Current LR after loading: {optimizer.param_groups[0]['lr']:.2e}")
        else:
            if rank == 0:
                print(f"⚠️  Optimizer checkpoint not found: {optimizer_checkpoint_path}")
                print("    Starting with fresh optimizer/scheduler states")

    logger = None
    early_stopping = None
    
    if rank == 0:
        # MetricsLogger will load existing history if resuming
        logger = MetricsLogger(
            save_dir=args["save_checkpoint_dir"],
            load_existing=(args["resume_epoch"] > 0)
        )
        
        # Create early stopping and load state if resuming
        early_stopping = EarlyStopping(
            patience=args.get("early_stopping_patience", 5),
            min_delta=args.get("early_stopping_min_delta", 0.001),
            mode='min'
        )
        
        # Load early stopping state if resuming
        if args["resume_epoch"] > 0 and loaded_checkpoint is not None:
            if "early_stopping_state_dict" in loaded_checkpoint:
                early_stopping.load_state_dict(loaded_checkpoint["early_stopping_state_dict"])
                print(f"✅ Restored early stopping state (counter={early_stopping.counter}, best_epoch={early_stopping.best_epoch})")
            else:
                print("⚠️  No early stopping state found in checkpoint, starting fresh")

    # Training loop
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
        scheduler.step()
        dist.barrier()
        
        eval_loss = eval_epoch(
            rank=rank,
            model=model,
            current_epoch=epoch_idx,
            dataloader=eval_loader,
            args=args
        )
        dist.barrier()
        
        # Get alpha statistics from gated fusion
        if rank == 0:
            # Sample one batch to get alpha stats
            sample_batch = next(iter(eval_loader))
            with torch.no_grad():
                seq_proj, struct_proj, _ = model.module(
                    input_ids=sample_batch["input_ids"][:1].to(rank),
                    attention_mask=sample_batch["attention_mask"][:1].to(rank),
                    protein_input_ids=sample_batch["protein_input_ids"][:1].to(rank),
                    protein_attention_mask=sample_batch["protein_attention_mask"][:1].to(rank),
                    return_adapter_outputs=True
                )
                # Access the active gated_fusion (PEFT wraps it in modules_to_save)
                gated_fusion = model.module.base_model.model.gated_fusion
                if hasattr(gated_fusion, 'modules_to_save'):
                    gated_fusion = gated_fusion.modules_to_save.default
                alpha_stats = gated_fusion.get_alpha_stats(seq_proj, struct_proj)
            
            logger.log_epoch(
                epoch=epoch_idx,
                train_loss=train_loss,
                eval_loss=eval_loss,
                lr=optimizer.param_groups[0]['lr'],
                grad_norm=grad_norm,
                alpha_stats=alpha_stats
            )
        # Early stopping check - need to broadcast to all ranks
        should_stop = torch.tensor([0], dtype=torch.int, device=rank)
        if rank == 0:
            if early_stopping(eval_loss, epoch_idx):
                print(f"\n🛑 Early stopping triggered at epoch {epoch_idx}")
                print(f"Best epoch was {early_stopping.best_epoch} with loss {-early_stopping.best_score:.4f}")
                should_stop[0] = 1
        
        dist.broadcast(should_stop, src=0)
        dist.barrier()
        
        # Save checkpoint before breaking if early stopping triggered
        if should_stop[0] == 1:
            if rank == 0:
                print(f"💾 Saving final checkpoint at epoch {epoch_idx} before early stopping...")
                adapter_checkpoint_dir = os.path.join(
                    args["save_checkpoint_dir"], 
                    f"adapter_checkpoint_{epoch_idx}_final"
                )
                model.module.save_pretrained(adapter_checkpoint_dir)
                print(f"Saving {adapter_checkpoint_dir}")

                optimizer_scheduler_checkpoint_path = os.path.join(
                    args["save_checkpoint_dir"], 
                    f"optimizer_scheduler_checkpoint_{epoch_idx}_final.pt"
                )
                checkpoint_data = {
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch_idx,
                }
                if early_stopping is not None:
                    checkpoint_data["early_stopping_state_dict"] = early_stopping.state_dict()
                if wandb.run is not None:
                    checkpoint_data["wandb_run_id"] = wandb.run.id
                
                torch.save(checkpoint_data, optimizer_scheduler_checkpoint_path)
                print(f"Saving {optimizer_scheduler_checkpoint_path}")
            
            dist.barrier()
            break

        if (
            epoch_idx == 1 
            or epoch_idx == end_epoch 
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
                checkpoint_data = {
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch_idx,
                }
                if early_stopping is not None:
                    checkpoint_data["early_stopping_state_dict"] = early_stopping.state_dict()
                if wandb.run is not None:
                    checkpoint_data["wandb_run_id"] = wandb.run.id
                
                torch.save(checkpoint_data, optimizer_scheduler_checkpoint_path)
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

    world_size = torch.cuda.device_count()
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
        "codebook_k": 1024,
        
        # Enhanced fusion parameters
        "fusion_num_heads": 8,
        "fusion_dropout": 0.2,
        "gate_hidden_dim": 256,

        "num_epochs": 10, #DEBUG
        "save_every_epochs": 1, #DEBUG
        "batch_size_per_device": 4,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-4,
        "scheduler_gamma": 0.9,
        "gradient_clipping": 0.3,

        "include_text_fields": True,
        "debug_trim_train_split": None, #DEBUG
        "debug_trim_eval_split": None, #DEBUG
        "name_dropout": 0.8,
        "taxonomy_dropout": 0.8,

        "early_stopping_patience": 3,
        "early_stopping_min_delta": 0.005,
        
        "wandb_project": "prot3text",
        "wandb_name": f"train_prot3text_v2_{timestamp}",
        
        "resume_checkpoint_dir": "new/checkpoints_260204_190537/adapter_checkpoint_3",  #DEBUG
    }

    torch.multiprocessing.spawn(
        train_on_device, 
        args=(args["world_size"], args),
        nprocs=args["world_size"],
        join=True
    )
