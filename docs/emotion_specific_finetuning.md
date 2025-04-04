# Emotion-Specific Fine-tuning with Capability Preservation

This document provides a comprehensive guide to fine-tuning CSM models for specific emotions while preserving their general speech capabilities.

## Table of Contents

- [Emotion-Specific Fine-tuning with Capability Preservation](#emotion-specific-fine-tuning-with-capability-preservation)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Continual Learning Techniques](#continual-learning-techniques)
    - [1. Elastic Weight Consolidation (EWC)](#1-elastic-weight-consolidation-ewc)
    - [2. Knowledge Distillation](#2-knowledge-distillation)
    - [3. Rehearsal](#3-rehearsal)
  - [Adapter-Based Fine-tuning](#adapter-based-fine-tuning)
    - [LoRA (Low-Rank Adaptation)](#lora-low-rank-adaptation)
  - [Emotion-Specific Models](#emotion-specific-models)
  - [Usage Instructions](#usage-instructions)
    - [Basic Usage](#basic-usage)
    - [Using an Emotion-Specific Model](#using-an-emotion-specific-model)
  - [Advanced Configuration](#advanced-configuration)
    - [Adding Custom Emotions](#adding-custom-emotions)
  - [Evaluation](#evaluation)
  - [Best Practices](#best-practices)
  - [Troubleshooting](#troubleshooting)
    - [Common Issues](#common-issues)
    - [Memory Requirements](#memory-requirements)

## Overview

Emotion-specific fine-tuning allows the CSM model to express particular emotions (happiness, sadness, anger, etc.) more effectively while maintaining its core speech capabilities. This is achieved through a combination of continual learning techniques and parameter-efficient fine-tuning.

The key challenge in emotion-specific fine-tuning is avoiding "catastrophic forgetting," where the model becomes specialized for a specific emotion but loses its general capabilities. Our approach addresses this challenge through multiple complementary techniques.

## Continual Learning Techniques

We employ several continual learning techniques to preserve the model's original capabilities:

### 1. Elastic Weight Consolidation (EWC)

EWC prevents catastrophic forgetting by identifying which parameters in the network are important for previously learned tasks and penalizing changes to these parameters.

- **How it works**: EWC computes the Fisher Information Matrix on general speech samples to determine which parameters are most important for maintaining general speech capabilities.
- **Implementation**: During fine-tuning, a regularization term is added to the loss function that penalizes changes to important parameters.
- **Key parameter**: `ewc_lambda` controls the strength of this regularization.

### 2. Knowledge Distillation

Knowledge distillation uses the original model as a "teacher" to guide the fine-tuning process of the "student" model.

- **How it works**: The original pre-trained model produces outputs that are used as soft targets for the emotion-specific model.
- **Implementation**: The loss function combines standard cross-entropy loss with KL divergence between student and teacher outputs.
- **Key parameters**: `distillation_alpha` determines the weight given to the distillation loss, and `temperature` controls how soft the teacher's outputs are.

### 3. Rehearsal

Rehearsal involves periodically replaying samples from the original distribution during fine-tuning.

- **How it works**: A buffer of general speech samples is maintained and mixed with emotion-specific samples during training.
- **Implementation**: Each batch during training contains a mix of emotion-specific samples and general samples.
- **Key parameters**: `rehearsal_buffer_size` determines how many samples to keep in the buffer, and `replay_ratio` controls how frequently they're used.

## Adapter-Based Fine-tuning

Rather than fine-tuning all parameters in the model, we employ parameter-efficient techniques using adapters:

### LoRA (Low-Rank Adaptation)

LoRA adds small, trainable low-rank matrices to existing weights, allowing efficient adaptation without changing most parameters.

- **How it works**: For key weight matrices in the model, LoRA adds a low-rank decomposition (A×B) that captures emotion-specific adaptations.
- **Implementation**: We apply LoRA to attention modules and feed-forward networks in the transformer.
- **Key parameter**: `adapter_r` controls the rank of the adaptation matrices (higher means more expressive but less efficient).

## Emotion-Specific Models

Our approach creates separate specialized models for each emotion:

1. **Base Model**: The original CSM model fine-tuned for general expressivity
2. **Emotion Models**: Individual models fine-tuned for specific emotions:
   - Happy
   - Sad
   - Angry
   - Fearful
   - Surprised
   - Neutral (with more controlled expression)

Each emotion-specific model builds upon the base expressive model but specializes in a particular emotional style.

## Usage Instructions

### Basic Usage

1. **Prepare Environment**:
   ```bash
   pip install -r requirements-expressivity-finetuning.txt
   ```

2. **Run the Automated Script**:
   ```bash
   ./scripts/run_emotion_finetuning.sh
   ```

3. **Fine-tune for a Specific Emotion**:
   ```bash
   ./scripts/run_emotion_finetuning.sh --emotion happy
   ```

### Using an Emotion-Specific Model

```python
from transformers import AutoProcessor
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "kyutai/moshiko-pytorch-bf16",
    torch_dtype=torch.bfloat16
)

# Load emotion-specific adapter
emotion_model = PeftModel.from_pretrained(
    base_model,
    "emotion_models/happy/best"
)

# Process text
processor = AutoProcessor.from_pretrained("kyutai/moshiko-pytorch-bf16")
inputs = processor("Say this with emotion: Hello, world!", return_tensors="pt")

# Generate speech with emotion
outputs = emotion_model.generate(**inputs)
```

## Advanced Configuration

You can customize the fine-tuning process by modifying `configs/emotion_specific_finetuning.yaml`:

```yaml
# Key parameters for tuning
ewc_lambda: 5000.0          # EWC regularization strength
distillation_alpha: 0.5     # Weight for distillation loss
adapter_r: 16               # LoRA rank
learning_rate: 1.0e-5       # Learning rate
num_epochs: 3               # Number of epochs
```

### Adding Custom Emotions

To fine-tune for additional emotions:

1. Add the emotion to the `emotions` list in the config file
2. Ensure you have sufficient training samples for that emotion
3. Run the fine-tuning script

## Evaluation

We provide tools to evaluate emotion-specific models:

```bash
python scripts/evaluation_metrics.py \
    --eval-dir evaluation/test_samples/happy \
    --emotion happy \
    --output evaluation_results/happy_results.json \
    --batch
```

The evaluation examines:
- **Expressivity**: How well the emotion is conveyed
- **Pronunciation**: Whether speech clarity is maintained
- **MOS (Mean Opinion Score)**: Overall quality estimation

## Best Practices

1. **Dataset Balance**: Ensure your emotion-specific datasets have sufficient variety and quality.
2. **EWC Strength**: If the model loses general capabilities, increase `ewc_lambda`; if it doesn't express emotions well enough, decrease it.
3. **Adapter Size**: Larger adapters (`adapter_r`) can capture more emotional nuance but require more data to train effectively.
4. **Combined Techniques**: The best results come from using all techniques (EWC, distillation, rehearsal, and adapters) together.
5. **Evaluation**: Always compare with the base model to ensure no degradation of core capabilities.

## Troubleshooting

### Common Issues

1. **Weak Emotional Expression**:
   - Decrease `ewc_lambda` to allow more parameter changes
   - Increase `adapter_r` for more expressive power
   - Ensure the emotion-specific dataset has clear emotional characteristics

2. **Loss of General Capabilities**:
   - Increase `ewc_lambda` to more strongly preserve original parameters
   - Increase `distillation_alpha` to make the teacher model more influential
   - Increase `rehearsal_buffer_size` and `replay_ratio` for more frequent rehearsal

3. **Training Instability**:
   - Decrease the learning rate
   - Increase gradient accumulation steps
   - Check for dataset inconsistencies

### Memory Requirements

- Each emotion model requires approximately 3-4GB of VRAM during training (with adapters)
- Full fine-tuning (without adapters) requires 12-16GB or more of VRAM
- Inference requires about 2-3GB of VRAM per emotion model 