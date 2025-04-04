# CSM Expressivity Fine-tuning Guide

This guide explains how to fine-tune the CSM model to enhance its expressivity and emotional speech capabilities. By following this process, you'll be able to create a model that can express a wide range of emotions and speaking styles.

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Datasets](#datasets)
4. [Fine-tuning Process](#fine-tuning-process)
5. [Configuration](#configuration)
6. [Automated Workflow](#automated-workflow)
7. [Using the Fine-tuned Model](#using-the-fine-tuned-model)
8. [Troubleshooting](#troubleshooting)

## Overview

The CSM model can be fine-tuned to enhance its expressivity using LoRA (Low-Rank Adaptation), a technique that allows efficient fine-tuning without modifying all model parameters. This guide focuses on fine-tuning the model on datasets of expressive speech to improve its ability to convey different emotions and speaking styles.

## System Requirements

- **Hardware**:
  - At least one GPU with 16GB+ VRAM
  - 32GB+ system RAM
  - 100GB+ disk space (for datasets and model checkpoints)

- **Software**:
  - Python 3.10+
  - PyTorch 2.0+
  - Necessary dependencies for CSM (see main project README)

## Datasets

The fine-tuning process uses emotional speech datasets that contain recordings with a variety of emotions and speaking styles. We recommend the following datasets:

### Recommended Datasets

1. **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
   - 24 professional actors (12 female, 12 male)
   - Emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised
   - [Download from Zenodo](https://zenodo.org/record/1188976)

2. **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset)
   - 7,442 clips from 91 actors of diverse backgrounds
   - Emotions: anger, disgust, fear, happiness, neutral, sadness
   - [GitHub Repository](https://github.com/CheyneyComputerScience/CREMA-D)

3. **ESD** (Emotional Speech Dataset)
   - 350 parallel utterances by 10 English and 10 Mandarin speakers
   - Emotions: neutral, happy, angry, sad, surprised
   - [GitHub Repository](https://github.com/HLTSingapore/Emotional-Speech-Data)

4. **MSP-PODCAST**
   - Natural emotional speech from podcast recordings
   - [Request access](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html)

We provide scripts to automatically prepare RAVDESS and CREMA-D datasets (`scripts/prepare_ravdess_dataset.py` and `scripts/prepare_cremad_dataset.py`).

### Dataset Format

The datasets need to be prepared in a specific format for fine-tuning:

- Audio files should be 24kHz stereo WAV files
- Each audio file should have an associated JSON file with transcription and emotion information
- A JSONL manifest file lists all samples with their paths and metadata

Our preparation scripts handle this conversion automatically.

## Fine-tuning Process

The fine-tuning process consists of these main steps:

1. **Dataset Preparation**: Convert datasets to the required format
2. **Dataset Merging**: Combine and balance multiple datasets
3. **Configuration**: Set up fine-tuning parameters
4. **Training**: Run the fine-tuning process using LoRA
5. **Evaluation**: Assess the fine-tuned model's expressivity

## Configuration

The fine-tuning process is controlled by a YAML configuration file. A sample configuration is provided in `configs/expressive_finetuning.yaml`.

Key configuration parameters:

### Model Configuration
```yaml
moshi_paths:
  hf_repo_id: "kyutai/moshiko-pytorch-bf16"  # Base model to fine-tune
```

### Dataset Configuration
```yaml
dataset:
  manifest_path: "data/combined/expressive_train.jsonl"
  val_manifest_path: "data/combined/expressive_val.jsonl"
  duration_sec: 80  # Duration in seconds of each training sample
  batch_size: 8  # Batch size for training
```

### LoRA Configuration
```yaml
lora:
  enable: true
  rank: 128  # Rank of the low-rank matrices
  alpha: 256  # Alpha scaling factor
  target_modules:  # Modules to apply LoRA to
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  scaling: 2.0  # LoRA scaling factor
```

### Training Parameters
```yaml
training:
  max_steps: 3000  # Maximum number of training steps
  save_every: 500  # Save checkpoint every N steps
  eval_every: 100  # Evaluate every N steps
```

### Optimizer Parameters
```yaml
optimizer:
  name: "adamw"
  lr: 4.0e-6  # Learning rate
  weight_decay: 0.01
```

## Automated Workflow

For your convenience, we provide a shell script to automate the entire fine-tuning process: `scripts/run_expressive_finetuning.sh`.

### Using the Automated Script

1. Make the script executable:
   ```bash
   chmod +x scripts/run_expressive_finetuning.sh
   ```

2. Run the script:
   ```bash
   ./scripts/run_expressive_finetuning.sh
   ```

### Script Options

- `--gpus N`: Number of GPUs to use for training (default: 1)
- `--skip-datasets`: Skip dataset preparation and use existing data
- `--resume-from-checkpoint PATH`: Resume training from a checkpoint

Example:
```bash
./scripts/run_expressive_finetuning.sh --gpus 2 --resume-from-checkpoint checkpoints/expressive_finetuning/checkpoint_000500
```

## Using the Fine-tuned Model

After fine-tuning, you can use the model with the following steps:

1. **Server Mode**:
   ```bash
   python -m moshi.server \
     --lora-weight=checkpoints/expressive_finetuning/consolidated/lora.safetensors \
     --config-path=checkpoints/expressive_finetuning/consolidated/config.json
   ```

2. **Access the Web UI**:
   Open http://localhost:8998 in your browser to interact with the fine-tuned model.

3. **Using Emotions**:
   - The web UI doesn't have explicit controls for emotions yet
   - Use prompt wording to suggest emotions, such as "Say this happily: ..." or "Respond in an angry tone: ..."

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   - Reduce `batch_size` in the configuration
   - Reduce `duration_sec` in the configuration
   - Enable gradient checkpointing with `advanced.gradient_checkpointing: true`

2. **Poor Expressivity**:
   - Ensure your datasets have good emotional range
   - Try increasing the LoRA `rank` parameter (e.g., to 256)
   - Train for more steps

3. **Training Instability**:
   - Reduce the learning rate (`optimizer.lr`)
   - Increase warmup steps (`scheduler.num_warmup_steps`)

4. **Dataset Issues**:
   - Check if the dataset JSONL manifest is correctly formatted
   - Ensure audio files are properly converted to 24kHz stereo WAV
   - Verify that JSON annotation files contain correct emotion labels

### Getting Help

If you encounter issues not covered here, please:
1. Check the logs in the `logs/` directory
2. Refer to the main CSM documentation
3. Open an issue on the project's GitHub repository with detailed information about your problem

## References

1. RAVDESS: Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
2. CREMA-D: Cao H, Cooper DG, Keutmann MK, Gur RC, Nenkova A, Verma R (2014) CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset
3. LoRA: Hu, E. J., et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models" 