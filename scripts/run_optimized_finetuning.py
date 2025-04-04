#!/usr/bin/env python3
"""
Optimized fine-tuning pipeline for expressive speech synthesis.

This script implements a robust fine-tuning pipeline with the following features:
- Efficient LoRA fine-tuning implementation
- Gradient accumulation and mixed precision
- Automatic checkpointing with versioning
- Periodic evaluation during training
- Early stopping based on validation metrics
- Automatic hyperparameter optimization (optional)
- Detailed logging and monitoring
- Resource utilization optimization

Usage:
    python run_optimized_finetuning.py --config configs/expressive_finetuning.yaml

Requirements:
    - PyTorch 2.0+
    - Transformers
    - Optimum (for LoRA)
    - Weights & Biases (for logging, optional)
    - Ray Tune (for hyperparameter optimization, optional)
"""

import os
import sys
import time
import json
import yaml
import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

# Optional imports for monitoring and hyperparameter tuning
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

# Import evaluation metrics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.evaluation_metrics import evaluate_sample, batch_evaluate, OverallMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("finetuning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("finetuning")

@dataclass
class TrainingState:
    """Class to track training state for checkpointing and resuming."""
    step: int = 0
    epoch: int = 0
    best_val_loss: float = float('inf')
    best_val_metrics: Optional[Dict[str, float]] = None
    early_stop_counter: int = 0
    completed: bool = False
    training_start_time: float = 0.0
    last_checkpoint_time: float = 0.0
    last_eval_time: float = 0.0
    last_log_time: float = 0.0
    train_losses: List[float] = None
    val_losses: List[float] = None
    learning_rates: List[float] = None
    gpu_memory_usage: List[float] = None
    
    def __post_init__(self):
        if self.train_losses is None:
            self.train_losses = []
        if self.val_losses is None:
            self.val_losses = []
        if self.learning_rates is None:
            self.learning_rates = []
        if self.gpu_memory_usage is None:
            self.gpu_memory_usage = []
        if self.best_val_metrics is None:
            self.best_val_metrics = {}
        if self.training_start_time == 0.0:
            self.training_start_time = time.time()
            self.last_checkpoint_time = time.time()
            self.last_eval_time = time.time()
            self.last_log_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert training state to dictionary for saving."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, state_dict: Dict[str, Any]) -> 'TrainingState':
        """Create training state from dictionary."""
        return cls(**state_dict)
    
    def save(self, filepath: str):
        """Save training state to file."""
        state_dict = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(state_dict, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'TrainingState':
        """Load training state from file."""
        with open(filepath, 'r') as f:
            state_dict = json.load(f)
        return cls.from_dict(state_dict)

class ExpressionDataset(Dataset):
    """Dataset for expressive speech fine-tuning."""
    
    def __init__(self, manifest_path: str, processor, max_length: int = 2048):
        """
        Initialize dataset from a manifest file.
        
        Args:
            manifest_path: Path to JSONL manifest file
            processor: Text processor for tokenization
            max_length: Maximum sequence length
        """
        self.samples = []
        self.max_length = max_length
        self.processor = processor
        
        # Load manifest
        with open(manifest_path, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                self.samples.append(sample)
        
        logger.info(f"Loaded {len(self.samples)} samples from {manifest_path}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        sample = self.samples[idx]
        
        # Load audio file and associated JSON file
        audio_path = sample['path']
        json_path = os.path.splitext(audio_path)[0] + ".json"
        
        # Load metadata
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        
        # Extract text and process it
        text = metadata.get('text', '')
        inputs = self.processor(text, return_tensors="pt", padding="max_length", 
                               max_length=self.max_length, truncation=True)
        
        # Create input dictionary
        input_dict = {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'audio_path': audio_path,
            'text': text,
            'emotion': metadata.get('emotion', 'neutral')
        }
        
        return input_dict

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def initialize_model_and_processor(config: Dict[str, Any]) -> Tuple[nn.Module, Any]:
    """
    Initialize the model and processor for fine-tuning.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of model and processor
    """
    from transformers import AutoProcessor, AutoModelForCausalLM
    
    # Load model and processor
    model_name = config['moshi_paths'].get('hf_repo_id', 'kyutai/moshiko-pytorch-bf16')
    
    logger.info(f"Loading model and processor from {model_name}")
    
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Apply LoRA if enabled
    if config['lora'].get('enable', False):
        from peft import LoraConfig, get_peft_model
        
        logger.info("Applying LoRA to the model")
        
        lora_config = LoraConfig(
            r=config['lora'].get('rank', 64),
            lora_alpha=config['lora'].get('alpha', 128),
            target_modules=config['lora'].get('target_modules', ["q_proj", "v_proj"]),
            lora_dropout=config['lora'].get('dropout', 0.1),
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    return model, processor

def create_optimizer_and_scheduler(
    model: nn.Module, 
    config: Dict[str, Any], 
    num_training_steps: int
) -> Tuple[optim.Optimizer, Any]:
    """
    Create optimizer and learning rate scheduler.
    
    Args:
        model: The model to optimize
        config: Configuration dictionary
        num_training_steps: Total number of training steps
        
    Returns:
        Tuple of optimizer and scheduler
    """
    from transformers import get_scheduler
    
    # Get optimizer parameters
    optimizer_name = config['optimizer'].get('name', 'adamw').lower()
    lr = config['optimizer'].get('lr', 5e-5)
    weight_decay = config['optimizer'].get('weight_decay', 0.01)
    
    # Get trainable parameters
    if config['lora'].get('enable', False):
        optimizer_params = [p for n, p in model.named_parameters() if p.requires_grad]
    else:
        optimizer_params = model.parameters()
    
    # Create optimizer
    if optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            optimizer_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(config['optimizer'].get('beta1', 0.9), 
                   config['optimizer'].get('beta2', 0.999)),
            eps=config['optimizer'].get('eps', 1e-8)
        )
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(
            optimizer_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(config['optimizer'].get('beta1', 0.9), 
                   config['optimizer'].get('beta2', 0.999)),
            eps=config['optimizer'].get('eps', 1e-8)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Create scheduler
    scheduler_name = config['scheduler'].get('name', 'cosine')
    warmup_steps = config['scheduler'].get('num_warmup_steps', 0)
    
    scheduler = get_scheduler(
        name=scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
    return optimizer, scheduler

def setup_wandb_logging(config: Dict[str, Any], resume: bool = False) -> Optional[str]:
    """
    Set up Weights & Biases logging if available and enabled.
    
    Args:
        config: Configuration dictionary
        resume: Whether to resume an existing run
        
    Returns:
        Run ID if logging is set up, None otherwise
    """
    if not WANDB_AVAILABLE or not config['logging'].get('wandb', False):
        return None
    
    # Get W&B configuration
    project = config['logging'].get('wandb_project', 'csm-expressive-finetuning')
    entity = config['logging'].get('wandb_entity', None)
    run_name = config['logging'].get('wandb_run_name', None)
    
    # Initialize W&B
    if resume and run_name:
        try:
            wandb.init(project=project, entity=entity, name=run_name, resume=True)
            logger.info(f"Resumed W&B logging for run: {run_name}")
            return wandb.run.id
        except Exception as e:
            logger.warning(f"Failed to resume W&B run, starting new one: {e}")
    
    # Start new run
    wandb.init(project=project, entity=entity, name=run_name, config=config)
    logger.info(f"Started W&B logging with run ID: {wandb.run.id}")
    
    return wandb.run.id

def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    eval_steps: int,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate the model on the validation set.
    
    Args:
        model: The model to evaluate
        val_loader: Validation data loader
        eval_steps: Number of steps to evaluate (0 for all)
        device: Device to run evaluation on
        
    Returns:
        Tuple of average loss and metrics dictionary
    """
    model.eval()
    total_loss = 0.0
    steps = 0
    
    # Prepare for audio evaluation
    test_samples = []
    
    with torch.no_grad():
        for batch in val_loader:
            if eval_steps > 0 and steps >= eval_steps:
                break
                
            # Move inputs to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Collect data for audio evaluation
            if steps < 5:  # Limit number of samples for audio evaluation
                for i in range(len(batch['audio_path'])):
                    test_samples.append({
                        'audio_path': batch['audio_path'][i],
                        'transcript': batch['text'][i],
                        'emotion': batch['emotion'][i]
                    })
            
            steps += 1
    
    # Calculate average loss
    avg_loss = total_loss / steps if steps > 0 else float('inf')
    
    # Evaluate audio samples if available
    audio_metrics = {}
    if test_samples:
        try:
            eval_results = batch_evaluate(test_samples)
            # Extract average metrics
            if 'average' in eval_results:
                audio_metrics = {
                    'expressivity': eval_results['average']['expressivity'],
                    'pronunciation': eval_results['average']['pronunciation'],
                    'mos': eval_results['average']['mos']
                }
        except Exception as e:
            logger.warning(f"Audio evaluation failed: {e}")
    
    # Combine metrics
    metrics = {
        'val_loss': avg_loss,
        **audio_metrics
    }
    
    model.train()
    return avg_loss, metrics

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    training_state: TrainingState,
    config: Dict[str, Any],
    checkpoint_dir: str,
    is_best: bool = False
) -> None:
    """
    Save model checkpoint with optimizer and scheduler states.
    
    Args:
        model: The model to save
        optimizer: The optimizer
        scheduler: The learning rate scheduler
        training_state: Training state object
        config: Configuration dictionary
        checkpoint_dir: Directory to save checkpoints
        is_best: Whether this is the best model so far
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model state
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{training_state.step:06d}")
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Save training state
    training_state.save(os.path.join(checkpoint_path, "training_state.json"))
    
    # Save model
    if config['lora'].get('enable', False):
        model.save_pretrained(checkpoint_path)
    else:
        torch.save(model.state_dict(), os.path.join(checkpoint_path, "model.pt"))
    
    # Save optimizer and scheduler
    torch.save({
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
    }, os.path.join(checkpoint_path, "optimizer.pt"))
    
    # Copy to best if needed
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best")
        if os.path.exists(best_path):
            shutil.rmtree(best_path)
        shutil.copytree(checkpoint_path, best_path)
    
    # Keep only the last k checkpoints
    keep_last_k = config['checkpoint'].get('keep_last_k', 3)
    if keep_last_k > 0:
        checkpoints = [
            d for d in os.listdir(checkpoint_dir) 
            if d.startswith("checkpoint_") and os.path.isdir(os.path.join(checkpoint_dir, d))
        ]
        checkpoints.sort(key=lambda x: int(x.split("_")[1]))
        
        if len(checkpoints) > keep_last_k:
            for old_checkpoint in checkpoints[:-keep_last_k]:
                old_path = os.path.join(checkpoint_dir, old_checkpoint)
                shutil.rmtree(old_path)
    
    logger.info(f"Saved checkpoint at step {training_state.step} to {checkpoint_path}")

def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    checkpoint_path: str,
    config: Dict[str, Any]
) -> TrainingState:
    """
    Load model checkpoint with optimizer and scheduler states.
    
    Args:
        model: The model to load into
        optimizer: The optimizer
        scheduler: The learning rate scheduler
        checkpoint_path: Path to the checkpoint directory
        config: Configuration dictionary
        
    Returns:
        Loaded training state
    """
    # Load training state
    training_state = TrainingState.load(os.path.join(checkpoint_path, "training_state.json"))
    
    # Load model
    if config['lora'].get('enable', False):
        from peft import PeftModel
        
        # Handle lora loading
        model = PeftModel.from_pretrained(
            model,
            checkpoint_path,
            is_trainable=True
        )
    else:
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, "model.pt")))
    
    # Load optimizer and scheduler
    opt_scheduler_dict = torch.load(os.path.join(checkpoint_path, "optimizer.pt"))
    optimizer.load_state_dict(opt_scheduler_dict['optimizer'])
    
    if scheduler and opt_scheduler_dict['scheduler']:
        scheduler.load_state_dict(opt_scheduler_dict['scheduler'])
    
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return training_state

def log_metrics(
    metrics: Dict[str, float],
    step: int,
    training_state: TrainingState,
    config: Dict[str, Any]
) -> None:
    """
    Log metrics to console and W&B if enabled.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Current training step
        training_state: Training state for tracking
        config: Configuration dictionary
    """
    # Log to console
    metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"Step {step} | {metrics_str}")
    
    # Track GPU memory usage if available
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
        
        training_state.gpu_memory_usage.append(memory_allocated)
        logger.info(f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        
        metrics['gpu_memory_allocated'] = memory_allocated
        metrics['gpu_memory_reserved'] = memory_reserved
    
    # Log to W&B if enabled
    if WANDB_AVAILABLE and config['logging'].get('wandb', False):
        wandb.log(metrics, step=step)

def train(
    config: Dict[str, Any],
    model: nn.Module,
    processor: Any,
    checkpoint_path: Optional[str] = None
) -> None:
    """
    Main training function.
    
    Args:
        config: Configuration dictionary
        model: The model to train
        processor: Text processor for tokenization
        checkpoint_path: Path to resume from checkpoint (optional)
    """
    # Initialize distributed training if needed
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_distributed = local_rank != -1
    
    if is_distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Setup W&B logging
    run_id = setup_wandb_logging(config, resume=checkpoint_path is not None)
    
    # Create output directory
    save_dir = config['training'].get('save_dir', 'checkpoints/expressive_finetuning')
    os.makedirs(save_dir, exist_ok=True)
    
    # Create consolidated dir for final model
    consolidated_dir = os.path.join(save_dir, "consolidated")
    os.makedirs(consolidated_dir, exist_ok=True)
    
    # Load datasets
    train_manifest = config['dataset'].get('manifest_path')
    val_manifest = config['dataset'].get('val_manifest_path')
    
    if not val_manifest and train_manifest:
        # If no validation manifest, use a portion of training data
        train_val_split = config['dataset'].get('train_val_split', 0.9)
        logger.info(f"No validation manifest provided, using {100 * (1 - train_val_split):.1f}% of training data for validation")
        
        # TODO: Implement train-val split from the same manifest
        pass
    
    # Create datasets
    train_dataset = ExpressionDataset(
        train_manifest, 
        processor, 
        max_length=config['dataset'].get('crop_length_tokens', 2048)
    )
    
    val_dataset = ExpressionDataset(
        val_manifest, 
        processor, 
        max_length=config['dataset'].get('crop_length_tokens', 2048)
    ) if val_manifest else None
    
    # Create data loaders
    batch_size = config['dataset'].get('batch_size', 8)
    
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset) if val_dataset else None
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    ) if val_dataset else None
    
    # Calculate total training steps
    max_steps = config['training'].get('max_steps', 1000)
    steps_per_epoch = len(train_loader)
    num_epochs = (max_steps + steps_per_epoch - 1) // steps_per_epoch
    
    logger.info(f"Training for {num_epochs} epochs ({max_steps} steps)")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, max_steps)
    
    # Initialize training state
    training_state = TrainingState()
    
    # Setup gradient scaler for mixed precision training
    use_amp = config['advanced'].get('amp', True) and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    
    # Enable gradient checkpointing if requested
    if config['advanced'].get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable()
    
    # Setup DDP if needed
    if is_distributed:
        find_unused_parameters = config['distributed'].get('find_unused_parameters', False)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, 
                   find_unused_parameters=find_unused_parameters)
    
    # Resume from checkpoint if provided
    if checkpoint_path:
        training_state = load_checkpoint(model, optimizer, scheduler, checkpoint_path, config)
    
    # Get configuration parameters
    save_every = config['training'].get('save_every', 100)
    eval_every = config['training'].get('eval_every', 50)
    log_every = config['training'].get('log_every', 10)
    eval_steps = config['training'].get('eval_steps', 50)
    patience = config['training'].get('patience', 10)
    gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    
    # Initialize best metrics
    if not training_state.best_val_metrics:
        training_state.best_val_metrics = {'val_loss': float('inf')}
    
    # Main training loop
    model.train()
    step = training_state.step
    
    if not is_distributed or local_rank == 0:
        logger.info("Starting training")
    
    try:
        for epoch in range(training_state.epoch, num_epochs):
            training_state.epoch = epoch
            
            if is_distributed:
                train_sampler.set_epoch(epoch)
            
            for batch_idx, batch in enumerate(train_loader):
                # Skip steps that were already processed
                if step < training_state.step:
                    step += 1
                    continue
                
                # Move inputs to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Forward and backward pass with optional mixed precision
                if use_amp:
                    with autocast():
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=input_ids
                        )
                        loss = outputs.loss / gradient_accumulation_steps
                    
                    # Scale loss and backward
                    scaler.scale(loss).backward()
                    
                    # Step if accumulated enough gradients
                    if (step + 1) % gradient_accumulation_steps == 0:
                        # Clip gradients
                        clip_grad_norm = config['optimizer'].get('clip_grad_norm', 1.0)
                        if clip_grad_norm > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), 
                                clip_grad_norm
                            )
                        
                        # Update parameters
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        
                        # Update learning rate
                        if scheduler:
                            scheduler.step()
                else:
                    # Standard precision training
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids
                    )
                    loss = outputs.loss / gradient_accumulation_steps
                    
                    # Backward pass
                    loss.backward()
                    
                    # Step if accumulated enough gradients
                    if (step + 1) % gradient_accumulation_steps == 0:
                        # Clip gradients
                        clip_grad_norm = config['optimizer'].get('clip_grad_norm', 1.0)
                        if clip_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), 
                                clip_grad_norm
                            )
                        
                        # Update parameters
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        # Update learning rate
                        if scheduler:
                            scheduler.step()
                
                # Track loss
                training_state.train_losses.append(loss.item() * gradient_accumulation_steps)
                
                # Track learning rate
                if scheduler:
                    current_lr = scheduler.get_last_lr()[0]
                    training_state.learning_rates.append(current_lr)
                
                # Logging
                if step % log_every == 0 and (not is_distributed or local_rank == 0):
                    # Get current learning rate
                    current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
                    
                    # Calculate metrics
                    metrics = {
                        'train_loss': loss.item() * gradient_accumulation_steps,
                        'learning_rate': current_lr,
                        'epoch': epoch,
                        'progress': step / max_steps
                    }
                    
                    log_metrics(metrics, step, training_state, config)
                    training_state.last_log_time = time.time()
                
                # Evaluation
                if val_loader and step % eval_every == 0 and (not is_distributed or local_rank == 0):
                    val_loss, val_metrics = evaluate_model(
                        model.module if is_distributed else model,
                        val_loader,
                        eval_steps,
                        device
                    )
                    
                    training_state.val_losses.append(val_loss)
                    
                    # Log validation metrics
                    log_metrics({**val_metrics, 'epoch': epoch}, step, training_state, config)
                    training_state.last_eval_time = time.time()
                    
                    # Check for best model
                    is_best = val_loss < training_state.best_val_loss
                    if is_best:
                        training_state.best_val_loss = val_loss
                        training_state.best_val_metrics = val_metrics
                        training_state.early_stop_counter = 0
                    else:
                        training_state.early_stop_counter += 1
                    
                    # Early stopping
                    if patience > 0 and training_state.early_stop_counter >= patience:
                        logger.info(f"Early stopping triggered after {training_state.early_stop_counter} evaluations without improvement")
                        training_state.completed = True
                        break
                
                # Save checkpoint
                if step % save_every == 0 and (not is_distributed or local_rank == 0):
                    is_best = val_loader is None  # If no validation, consider every save as best
                    save_checkpoint(
                        model.module if is_distributed else model,
                        optimizer,
                        scheduler,
                        training_state,
                        config,
                        save_dir,
                        is_best
                    )
                    training_state.last_checkpoint_time = time.time()
                
                # Update step
                step += 1
                training_state.step = step
                
                # Check if we've reached max steps
                if step >= max_steps:
                    training_state.completed = True
                    break
            
            # End epoch
            if training_state.completed:
                break
        
        # Final checkpoint and metrics
        if not is_distributed or local_rank == 0:
            # Final evaluation
            if val_loader:
                val_loss, val_metrics = evaluate_model(
                    model.module if is_distributed else model,
                    val_loader,
                    eval_steps,
                    device
                )
                training_state.val_losses.append(val_loss)
                log_metrics({**val_metrics, 'epoch': epoch, 'final': True}, step, training_state, config)
            
            # Save final checkpoint
            save_checkpoint(
                model.module if is_distributed else model,
                optimizer,
                scheduler,
                training_state,
                config,
                save_dir,
                is_best=True
            )
            
            # Save consolidated model (for inference)
            if config['lora'].get('enable', False):
                # For LoRA, save merged model for inference
                model_to_save = model.module if is_distributed else model
                model_to_save.save_pretrained(consolidated_dir)
                
                # Also save config for easy loading
                with open(os.path.join(consolidated_dir, "config.json"), 'w') as f:
                    json.dump(config, f, indent=2)
            else:
                # For full fine-tuning, save model state dict
                model_to_save = model.module if is_distributed else model
                torch.save(model_to_save.state_dict(), os.path.join(consolidated_dir, "model.pt"))
                
                # Save config
                with open(os.path.join(consolidated_dir, "config.json"), 'w') as f:
                    json.dump(config, f, indent=2)
            
            logger.info(f"Training completed in {(time.time() - training_state.training_start_time) / 3600:.2f} hours")
            logger.info(f"Best validation loss: {training_state.best_val_loss:.4f}")
            logger.info(f"Final model saved to {consolidated_dir}")
            
            # Mark training as completed
            training_state.completed = True
            training_state.save(os.path.join(save_dir, "final_training_state.json"))
            
            # Finish wandb run
            if WANDB_AVAILABLE and config['logging'].get('wandb', False):
                wandb.finish()
    
    except KeyboardInterrupt:
        logger.info("Training interrupted")
        
        if not is_distributed or local_rank == 0:
            # Save interrupted checkpoint
            save_checkpoint(
                model.module if is_distributed else model,
                optimizer,
                scheduler,
                training_state,
                config,
                save_dir,
                is_best=False
            )
            
            logger.info(f"Saved checkpoint at step {training_state.step}")
            
            # Finish wandb run
            if WANDB_AVAILABLE and config['logging'].get('wandb', False):
                wandb.finish()
    
    finally:
        # Clean up distributed training
        if is_distributed:
            dist.destroy_process_group()

def hyperparameter_optimization(config_path: str) -> None:
    """
    Run hyperparameter optimization using Ray Tune.
    
    Args:
        config_path: Path to the base configuration file
    """
    if not RAY_AVAILABLE:
        logger.error("Ray Tune is not installed. Cannot run hyperparameter optimization.")
        return
    
    # Load base configuration
    config = load_config(config_path)
    
    # Define search space
    search_space = {
        "optimizer.lr": tune.loguniform(1e-6, 1e-4),
        "lora.rank": tune.choice([64, 128, 256]),
        "lora.alpha": tune.choice([128, 256, 512]),
        "lora.dropout": tune.uniform(0.0, 0.2),
        "training.gradient_accumulation_steps": tune.choice([1, 2, 4]),
        "optimizer.weight_decay": tune.loguniform(0.001, 0.1),
    }
    
    # Define stopping criteria
    scheduler = ASHAScheduler(
        max_t=config['training'].get('max_steps', 1000),
        grace_period=100,
        reduction_factor=2
    )
    
    # Define reporting metrics
    reporter = CLIReporter(
        parameter_columns=list(search_space.keys()),
        metric_columns=["val_loss", "train_loss", "expressivity", "training_iteration"]
    )
    
    # Define training function for Ray Tune
    def train_tune(tune_config):
        # Update configuration with tune parameters
        for param, value in tune_config.items():
            if '.' in param:
                # Handle nested parameters
                parts = param.split('.')
                curr = config
                for part in parts[:-1]:
                    curr = curr[part]
                curr[parts[-1]] = value
            else:
                config[param] = value
        
        # Initialize model and processor
        model, processor = initialize_model_and_processor(config)
        
        # Set max_steps for faster tuning
        config['training']['max_steps'] = 500
        
        # Train with smaller evaluation frequency
        config['training']['eval_every'] = 50
        
        # Initialize training state
        training_state = TrainingState()
        
        # Train model and track metrics for tune
        train(config, model, processor)
        
        # Report final metrics
        tune.report(
            val_loss=training_state.best_val_loss,
            expressivity=training_state.best_val_metrics.get('expressivity', 0.0),
            train_loss=training_state.train_losses[-1] if training_state.train_losses else float('inf')
        )
    
    # Run hyperparameter search
    result = tune.run(
        train_tune,
        resources_per_trial={"cpu": 4, "gpu": 1},
        config=search_space,
        num_samples=20,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="expressive_finetuning_hpo",
        local_dir="./ray_results"
    )
    
    # Get best trial
    best_trial = result.get_best_trial("val_loss", "min", "last")
    logger.info(f"Best trial config: {best_trial.config}")
    logger.info(f"Best trial final validation loss: {best_trial.last_result['val_loss']}")
    logger.info(f"Best trial final expressivity score: {best_trial.last_result['expressivity']}")
    
    # Save best configuration
    best_config = config.copy()
    for param, value in best_trial.config.items():
        if '.' in param:
            # Handle nested parameters
            parts = param.split('.')
            curr = best_config
            for part in parts[:-1]:
                curr = curr[part]
            curr[parts[-1]] = value
        else:
            best_config[param] = value
    
    # Save optimized configuration
    optimized_config_path = config_path.replace('.yaml', '_optimized.yaml')
    with open(optimized_config_path, 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False)
    
    logger.info(f"Saved optimized configuration to {optimized_config_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized fine-tuning pipeline for expressive speech synthesis")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--optimize-hyperparams", action="store_true", help="Run hyperparameter optimization")
    
    args = parser.parse_args()
    
    if args.optimize_hyperparams:
        hyperparameter_optimization(args.config)
    else:
        # Load configuration
        config = load_config(args.config)
        
        # Initialize model and processor
        model, processor = initialize_model_and_processor(config)
        
        # Run training
        train(config, model, processor, args.resume) 