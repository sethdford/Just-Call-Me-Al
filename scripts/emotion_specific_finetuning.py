#!/usr/bin/env python3
"""
Emotion-specific fine-tuning with capability preservation.

This script implements fine-tuning for specific emotions while preserving
the model's original capabilities through continual learning techniques.
It uses techniques such as:
- Elastic Weight Consolidation (EWC)
- Knowledge Distillation
- Rehearsal with a buffer of original samples
- Adapter-based fine-tuning for specific emotions

Usage:
    python emotion_specific_finetuning.py --config configs/emotion_specific_finetuning.yaml

Requirements:
    - PyTorch 2.0+
    - Transformers
    - PEFT (for adapter-based fine-tuning)
    - A fine-tuned base model from the previous step
"""

import os
import sys
import argparse
import logging
import json
import yaml
import glob
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, random_split

# Import evaluation metrics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.evaluation_metrics import evaluate_sample, batch_evaluate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("emotion_finetuning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("emotion_finetuning")

@dataclass
class EmotionFineTuningConfig:
    """Configuration for emotion-specific fine-tuning."""
    # Base model
    base_model_path: str
    # Emotions to fine-tune
    emotions: List[str]
    # Continual learning parameters
    ewc_lambda: float = 5000.0  # EWC regularization strength
    distillation_alpha: float = 0.5  # Weight for distillation loss
    use_ewc: bool = True
    use_distillation: bool = True
    use_rehearsal: bool = True
    # Rehearsal parameters
    rehearsal_buffer_size: int = 1000
    replay_ratio: float = 0.3
    # Training parameters
    learning_rate: float = 2e-5
    batch_size: int = 8
    num_epochs: int = 5
    grad_accumulation_steps: int = 1
    # Adapter parameters
    use_adapters: bool = True
    adapter_r: int = 16
    # Output path
    output_dir: str = "emotion_models"
    # Data paths
    data_path: str = "data/combined"
    # Evaluation
    eval_steps: int = 100
    # Checkpointing
    save_every: int = 500
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EmotionFineTuningConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'EmotionFineTuningConfig':
        """Load config from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

class EmotionDataset(Dataset):
    """Dataset for emotion-specific speech samples."""
    
    def __init__(self, manifest_path: str, processor, emotion: Optional[str] = None):
        """
        Initialize dataset from a manifest file.
        
        Args:
            manifest_path: Path to JSONL manifest file
            processor: Text processor for tokenization
            emotion: Specific emotion to filter for (None for all)
        """
        self.samples = []
        self.processor = processor
        self.emotion = emotion
        
        # Load manifest
        with open(manifest_path, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                # Filter by emotion if specified
                if emotion is None or sample.get('emotion', '').lower() == emotion.lower():
                    self.samples.append(sample)
        
        if emotion:
            logger.info(f"Loaded {len(self.samples)} samples for emotion '{emotion}' from {manifest_path}")
        else:
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
                               max_length=2048, truncation=True)
        
        # Create input dictionary
        input_dict = {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'audio_path': audio_path,
            'text': text,
            'emotion': metadata.get('emotion', 'neutral')
        }
        
        return input_dict

class ElasticWeightConsolidation:
    """
    Elastic Weight Consolidation (EWC) for preventing catastrophic forgetting.
    
    This implements the EWC technique from Kirkpatrick et al. (2017):
    "Overcoming catastrophic forgetting in neural networks"
    """
    
    def __init__(self, model: nn.Module, dataset: Dataset, device: torch.device, ewc_lambda: float = 5000.0):
        """
        Initialize EWC.
        
        Args:
            model: The model to apply EWC to
            dataset: The dataset to compute Fisher information on
            device: Device to compute on
            ewc_lambda: Regularization strength
        """
        self.model = model
        self.device = device
        self.ewc_lambda = ewc_lambda
        
        # Store the parameter importance (Fisher information)
        self.fisher_information = {}
        # Store the original parameter values
        self.optpar = {}
        
        # Compute Fisher information and store original parameters
        self._compute_fisher_information(dataset)
        self._store_original_parameters()
    
    def _compute_fisher_information(self, dataset: Dataset) -> None:
        """
        Compute Fisher information (parameter importance).
        
        Args:
            dataset: Dataset to compute Fisher information on
        """
        logger.info("Computing Fisher information for EWC...")
        
        # Create a small subset for Fisher computation
        subset_size = min(100, len(dataset))
        indices = random.sample(range(len(dataset)), subset_size)
        subset = Subset(dataset, indices)
        dataloader = DataLoader(subset, batch_size=1, shuffle=True)
        
        # Initialize Fisher information
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher_information[name] = torch.zeros_like(param)
        
        # Compute Fisher information
        self.model.eval()
        for batch in dataloader:
            self.model.zero_grad()
            
            # Forward pass
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            loss = outputs.loss
            loss.backward()
            
            # Accumulate Fisher information
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher_information[name] += param.grad.detach() ** 2
        
        # Normalize Fisher information
        for name in self.fisher_information:
            self.fisher_information[name] /= subset_size
            
        logger.info("Fisher information computed")
    
    def _store_original_parameters(self) -> None:
        """Store original parameter values."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optpar[name] = param.data.clone()
    
    def compute_ewc_loss(self) -> torch.Tensor:
        """
        Compute EWC loss (regularization term).
        
        Returns:
            EWC loss tensor
        """
        ewc_loss = torch.tensor(0.0, device=self.device)
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.fisher_information:
                fisher = self.fisher_information[name]
                optpar = self.optpar[name]
                
                # Compute EWC loss: λ/2 * F * (θ - θ*)²
                ewc_loss += (fisher * (param - optpar) ** 2).sum()
        
        return self.ewc_lambda * ewc_loss / 2

class KnowledgeDistillation:
    """
    Knowledge Distillation for preserving original model capabilities.
    
    This implements the knowledge distillation technique from Hinton et al. (2015):
    "Distilling the Knowledge in a Neural Network"
    """
    
    def __init__(self, teacher_model: nn.Module, temperature: float = 2.0, alpha: float = 0.5):
        """
        Initialize Knowledge Distillation.
        
        Args:
            teacher_model: The original model to distill from
            temperature: Temperature for softening the teacher's predictions
            alpha: Weight for the distillation loss
        """
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        
        # Set teacher model to evaluation mode
        self.teacher_model.eval()
        
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def compute_distillation_loss(
        self, 
        student_logits: torch.Tensor, 
        teacher_logits: torch.Tensor,
        task_loss: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the knowledge distillation loss.
        
        Args:
            student_logits: Logits from the student model
            teacher_logits: Logits from the teacher model
            task_loss: Original task loss
            
        Returns:
            Combined loss with distillation
        """
        # Compute soft targets
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # Compute distillation loss
        distillation_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * \
                           (self.temperature ** 2)
        
        # Combine with task loss
        combined_loss = (1 - self.alpha) * task_loss + self.alpha * distillation_loss
        
        return combined_loss

class RehearsalBuffer:
    """
    Rehearsal buffer for storing and replaying samples from the original dataset.
    
    This implements the rehearsal technique to prevent catastrophic forgetting
    by periodically replaying samples from the original distribution.
    """
    
    def __init__(self, buffer_size: int = 1000):
        """
        Initialize Rehearsal Buffer.
        
        Args:
            buffer_size: Maximum number of samples to store
        """
        self.buffer_size = buffer_size
        self.buffer = []
        self.current_size = 0
    
    def add_examples(self, examples: List[Dict[str, torch.Tensor]]) -> None:
        """
        Add examples to the buffer.
        
        Args:
            examples: List of examples to add
        """
        for example in examples:
            if self.current_size < self.buffer_size:
                self.buffer.append(example)
                self.current_size += 1
            else:
                # Reservoir sampling: replace with probability buffer_size/t
                t = self.current_size + 1
                if random.randint(1, t) <= self.buffer_size:
                    # Replace a random item
                    idx = random.randint(0, self.buffer_size - 1)
                    self.buffer[idx] = example
                
                self.current_size += 1
    
    def get_batch(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """
        Get a batch of examples from the buffer.
        
        Args:
            batch_size: Number of examples to retrieve
            
        Returns:
            List of examples
        """
        if not self.buffer:
            return []
        
        # Sample a batch (with replacement if buffer is smaller than batch_size)
        if len(self.buffer) < batch_size:
            # Sample with replacement
            return random.choices(self.buffer, k=batch_size)
        else:
            # Sample without replacement
            return random.sample(self.buffer, batch_size)

class EmotionAdapter(nn.Module):
    """
    Adapter module for emotion-specific fine-tuning.
    
    This implements adapter modules as described in Houlsby et al. (2019):
    "Parameter-Efficient Transfer Learning for NLP"
    """
    
    def __init__(self, input_dim: int, reduction_factor: int = 16):
        """
        Initialize adapter module.
        
        Args:
            input_dim: Input dimension
            reduction_factor: Reduction factor for the bottleneck
        """
        super().__init__()
        self.adapter_down = nn.Linear(input_dim, input_dim // reduction_factor)
        self.adapter_up = nn.Linear(input_dim // reduction_factor, input_dim)
        self.act = nn.GELU()
        
        # Initialize adapter layers
        nn.init.normal_(self.adapter_down.weight, std=1e-3)
        nn.init.normal_(self.adapter_up.weight, std=1e-3)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the adapter.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with adapter residual connection
        """
        adapter_output = self.adapter_up(self.act(self.adapter_down(x)))
        return x + adapter_output

def add_emotion_adapter(model: nn.Module, emotion: str, adapter_r: int = 16) -> nn.Module:
    """
    Add emotion-specific adapters to transformer model.
    
    Args:
        model: Base model
        emotion: Emotion name for the adapter
        adapter_r: Reduction factor for adapters
        
    Returns:
        Model with adapters
    """
    # This is a simplified implementation focusing on key layers
    # A complete implementation would handle all layer types and model architectures
    
    from peft import get_peft_model, PeftConfig, LoraConfig
    
    logger.info(f"Adding adapters for emotion: {emotion}")
    
    # Use LoRA as the adapter mechanism
    lora_config = LoraConfig(
        r=adapter_r,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Add adapter to model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model

def finetune_emotion(
    config: EmotionFineTuningConfig,
    emotion: str,
    base_model: nn.Module,
    processor: Any
) -> nn.Module:
    """
    Fine-tune the model for a specific emotion while preserving original capabilities.
    
    Args:
        config: Fine-tuning configuration
        emotion: Target emotion for fine-tuning
        base_model: Base model to fine-tune
        processor: Text processor
        
    Returns:
        Fine-tuned model for the specific emotion
    """
    logger.info(f"Starting fine-tuning for emotion: {emotion}")
    
    # Set up device
    device = torch.device(config.device)
    
    # Create output directory
    emotion_output_dir = os.path.join(config.output_dir, emotion)
    os.makedirs(emotion_output_dir, exist_ok=True)
    
    # Create emotion-specific dataset
    emotion_manifest = os.path.join(config.data_path, "expressive_train.jsonl")
    emotion_dataset = EmotionDataset(emotion_manifest, processor, emotion=emotion)
    
    # Split into train/val
    val_size = min(100, int(len(emotion_dataset) * 0.1))
    train_size = len(emotion_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        emotion_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create a general dataset for rehearsal
    general_dataset = EmotionDataset(emotion_manifest, processor, emotion=None)
    
    # Initialize rehearsal buffer if enabled
    rehearsal_buffer = None
    if config.use_rehearsal:
        rehearsal_buffer = RehearsalBuffer(buffer_size=config.rehearsal_buffer_size)
        
        # Fill buffer with samples from general dataset
        buffer_loader = DataLoader(
            general_dataset, 
            batch_size=config.batch_size,
            shuffle=True
        )
        
        buffer_samples = []
        for batch in buffer_loader:
            for i in range(len(batch['input_ids'])):
                sample = {k: v[i] for k, v in batch.items() if k != 'audio_path'}
                sample['audio_path'] = batch['audio_path'][i]
                buffer_samples.append(sample)
                
                if len(buffer_samples) >= config.rehearsal_buffer_size:
                    break
            
            if len(buffer_samples) >= config.rehearsal_buffer_size:
                break
        
        rehearsal_buffer.add_examples(buffer_samples)
        logger.info(f"Filled rehearsal buffer with {rehearsal_buffer.current_size} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = None
    if val_size > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    # Clone the base model for fine-tuning
    model = type(base_model)(base_model.config)
    model.load_state_dict(base_model.state_dict())
    
    # Add adapters for emotion-specific tuning if enabled
    if config.use_adapters:
        model = add_emotion_adapter(model, emotion, config.adapter_r)
    
    # Move model to device
    model = model.to(device)
    
    # Set up EWC if enabled
    ewc = None
    if config.use_ewc:
        ewc = ElasticWeightConsolidation(model, general_dataset, device, config.ewc_lambda)
    
    # Set up knowledge distillation if enabled
    distillation = None
    if config.use_distillation:
        teacher_model = base_model.to(device)
        distillation = KnowledgeDistillation(
            teacher_model=teacher_model,
            alpha=config.distillation_alpha
        )
    
    # Set up optimizer
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate
    )
    
    # Track best validation loss
    best_val_loss = float('inf')
    
    # Training loop
    model.train()
    global_step = 0
    
    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Prepare inputs
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Mix in rehearsal samples if enabled
            if config.use_rehearsal and rehearsal_buffer and random.random() < config.replay_ratio:
                # Get a batch from the buffer
                buffer_batch = rehearsal_buffer.get_batch(config.batch_size)
                
                if buffer_batch:
                    # Create tensors from buffer samples
                    buffer_input_ids = torch.stack([sample['input_ids'] for sample in buffer_batch]).to(device)
                    buffer_attention_mask = torch.stack([sample['attention_mask'] for sample in buffer_batch]).to(device)
                    
                    # Combine with current batch (concatenate)
                    combined_size = min(len(input_ids) + len(buffer_input_ids), config.batch_size * 2)
                    
                    # Balance current and buffer samples
                    current_size = min(len(input_ids), combined_size // 2)
                    buffer_size = combined_size - current_size
                    
                    # Take subsets if needed
                    current_input_ids = input_ids[:current_size]
                    current_attention_mask = attention_mask[:current_size]
                    buffer_input_ids = buffer_input_ids[:buffer_size]
                    buffer_attention_mask = buffer_attention_mask[:buffer_size]
                    
                    # Combine
                    input_ids = torch.cat([current_input_ids, buffer_input_ids])
                    attention_mask = torch.cat([current_attention_mask, buffer_attention_mask])
            
            # Forward pass
            model.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            task_loss = outputs.loss
            total_loss = task_loss
            
            # Apply knowledge distillation if enabled
            if distillation:
                with torch.no_grad():
                    teacher_outputs = distillation.teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                
                total_loss = distillation.compute_distillation_loss(
                    student_logits=outputs.logits,
                    teacher_logits=teacher_outputs.logits,
                    task_loss=task_loss
                )
            
            # Apply EWC regularization if enabled
            if ewc:
                ewc_loss = ewc.compute_ewc_loss()
                total_loss += ewc_loss
            
            # Backward pass and optimization
            total_loss.backward()
            
            # Gradient accumulation
            if (global_step + 1) % config.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Update metrics
            epoch_loss += total_loss.item()
            num_batches += 1
            global_step += 1
            
            # Log progress
            if global_step % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{config.num_epochs}, Step {global_step}, Loss: {total_loss.item():.4f}")
            
            # Evaluate periodically
            if val_loader and global_step % config.eval_steps == 0:
                val_loss = evaluate_model(model, val_loader, device)
                logger.info(f"Validation loss: {val_loss:.4f}")
                
                # Save if best
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, epoch, global_step, best_val_loss, 
                                   os.path.join(emotion_output_dir, "best"))
                    logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
                
                model.train()
            
            # Regular checkpointing
            if global_step % config.save_every == 0:
                save_checkpoint(model, optimizer, epoch, global_step, best_val_loss, 
                               os.path.join(emotion_output_dir, f"checkpoint_{global_step}"))
        
        # End of epoch
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Evaluate at the end of each epoch
        if val_loader:
            val_loss = evaluate_model(model, val_loader, device)
            logger.info(f"End of epoch validation loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, global_step, best_val_loss, 
                               os.path.join(emotion_output_dir, "best"))
                logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
            
            model.train()
    
    # Final checkpoint
    save_checkpoint(model, optimizer, config.num_epochs-1, global_step, best_val_loss, 
                   os.path.join(emotion_output_dir, "final"))
    
    logger.info(f"Fine-tuning completed for emotion: {emotion}")
    
    # Load best model for return
    best_model_path = os.path.join(emotion_output_dir, "best")
    if os.path.exists(best_model_path):
        if config.use_adapters:
            # For adapter-based model, load adapter weights
            from peft import PeftModel
            model = PeftModel.from_pretrained(base_model, best_model_path)
        else:
            # For full fine-tuning, load state dict
            checkpoint = torch.load(os.path.join(best_model_path, "model.pt"))
            model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """
    Evaluate model on dataloader.
    
    Args:
        model: Model to evaluate
        dataloader: Validation dataloader
        device: Device to evaluate on
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            loss = outputs.loss
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    output_dir: str
) -> None:
    """
    Save a model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        global_step: Global training step
        best_val_loss: Best validation loss so far
        output_dir: Directory to save checkpoint
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # For adapter-based models, save adapter weights
    if hasattr(model, 'save_pretrained'):
        model.save_pretrained(output_dir)
    else:
        # For full fine-tuning, save state dict
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'global_step': global_step,
            'best_val_loss': best_val_loss
        }
        
        torch.save(checkpoint, os.path.join(output_dir, "model.pt"))
    
    # Save optimizer state separately
    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    
    # Save metadata
    metadata = {
        'epoch': epoch,
        'global_step': global_step,
        'best_val_loss': best_val_loss,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

def load_base_model(model_path: str, device: torch.device):
    """
    Load the base model for fine-tuning.
    
    Args:
        model_path: Path to the base model
        device: Device to load the model on
        
    Returns:
        Tuple of (model, processor)
    """
    logger.info(f"Loading base model from {model_path}")
    
    from transformers import AutoProcessor, AutoModelForCausalLM
    
    # Check if path contains a consolidated model with LoRA weights
    lora_path = os.path.join(model_path, "consolidated", "lora.safetensors")
    has_lora = os.path.exists(lora_path)
    
    # Check for base model path
    base_model_path = model_path
    if os.path.exists(os.path.join(model_path, "consolidated")):
        base_model_path = os.path.join(model_path, "consolidated")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(base_model_path)
    
    # Load model
    if has_lora:
        from peft import PeftModel
        
        # Load base model first
        base_model = AutoModelForCausalLM.from_pretrained(
            "kyutai/moshiko-pytorch-bf16",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Then load LoRA weights
        model = PeftModel.from_pretrained(
            base_model,
            lora_path
        )
    else:
        # Load full model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
    
    # Ensure model is on the correct device
    model = model.to(device)
    
    return model, processor

def main():
    parser = argparse.ArgumentParser(description="Emotion-specific fine-tuning with capability preservation")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    args = parser.parse_args()
    
    # Load configuration
    config = EmotionFineTuningConfig.from_yaml(args.config)
    
    # Set device
    device = torch.device(config.device)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load base model
    base_model, processor = load_base_model(config.base_model_path, device)
    
    # Fine-tune for each emotion
    emotion_models = {}
    for emotion in config.emotions:
        # Fine-tune for this emotion
        emotion_model = finetune_emotion(config, emotion, base_model, processor)
        emotion_models[emotion] = emotion_model
        
        logger.info(f"Completed fine-tuning for emotion: {emotion}")
    
    # Evaluate models on emotion-specific test sets
    logger.info("Evaluating emotion-specific models")
    
    test_manifest = os.path.join(config.data_path, "expressive_val.jsonl")
    for emotion, model in emotion_models.items():
        # Create emotion-specific test dataset
        emotion_test_dataset = EmotionDataset(test_manifest, processor, emotion=emotion)
        
        if len(emotion_test_dataset) == 0:
            logger.warning(f"No test samples for emotion: {emotion}")
            continue
        
        test_loader = DataLoader(
            emotion_test_dataset,
            batch_size=config.batch_size,
            shuffle=False
        )
        
        # Evaluate model
        test_loss = evaluate_model(model, test_loader, device)
        logger.info(f"Emotion: {emotion}, Test Loss: {test_loss:.4f}")
        
        # Evaluate expressivity
        try:
            eval_samples = []
            for i in range(min(10, len(emotion_test_dataset))):
                sample = emotion_test_dataset[i]
                eval_samples.append({
                    'audio_path': sample['audio_path'],
                    'transcript': sample['text'],
                    'emotion': sample['emotion']
                })
            
            if eval_samples:
                eval_results = batch_evaluate(eval_samples)
                
                # Save evaluation results
                eval_output_path = os.path.join(config.output_dir, f"{emotion}_evaluation.json")
                with open(eval_output_path, 'w') as f:
                    json.dump(eval_results, f, indent=2)
                
                logger.info(f"Saved evaluation results to {eval_output_path}")
                
                # Log average metrics
                if 'average' in eval_results:
                    avg = eval_results['average']
                    logger.info(f"Emotion: {emotion}, Expressivity: {avg.get('expressivity', 0):.4f}, "
                               f"Pronunciation: {avg.get('pronunciation', 0):.4f}, "
                               f"MOS: {avg.get('mos', 0):.4f}")
        except Exception as e:
            logger.error(f"Error evaluating expressivity for {emotion}: {e}")
    
    logger.info("Emotion-specific fine-tuning completed")

if __name__ == "__main__":
    main() 