#!/bin/bash
# Script to automate the entire expressivity fine-tuning process
# Usage: ./run_expressive_finetuning.sh [--gpus N] [--skip-datasets] [--resume-from-checkpoint PATH]

set -e  # Exit on error

# Default parameters
NUM_GPUS=1
DATA_DIR="data"
CONFIG_FILE="configs/expressive_finetuning.yaml"
SKIP_DATASETS=false
RESUME_CHECKPOINT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --gpus)
            NUM_GPUS="$2"
            shift
            shift
            ;;
        --skip-datasets)
            SKIP_DATASETS=true
            shift
            ;;
        --resume-from-checkpoint)
            RESUME_CHECKPOINT="$2"
            shift
            shift
            ;;
        --help)
            echo "Usage: ./run_expressive_finetuning.sh [--gpus N] [--skip-datasets] [--resume-from-checkpoint PATH]"
            echo ""
            echo "Options:"
            echo "  --gpus N                      Number of GPUs to use for training (default: 1)"
            echo "  --skip-datasets               Skip dataset preparation and use existing data"
            echo "  --resume-from-checkpoint PATH Resume training from a checkpoint"
            echo "  --help                        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $key"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "= CSM Expressivity Fine-tuning Pipeline ="
echo "=========================================="
echo "GPUs: $NUM_GPUS"
echo "Data directory: $DATA_DIR"
echo "Configuration file: $CONFIG_FILE"
echo "Skip datasets: $SKIP_DATASETS"
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "Resuming from checkpoint: $RESUME_CHECKPOINT"
fi
echo "=========================================="

# Create necessary directories
mkdir -p $DATA_DIR
mkdir -p logs
mkdir -p checkpoints/expressive_finetuning

# Step 1: Prepare datasets (if not skipped)
if [ "$SKIP_DATASETS" = false ]; then
    echo "\n[Step 1/5] Preparing RAVDESS dataset..."
    python scripts/prepare_ravdess_dataset.py --output_dir $DATA_DIR/ravdess | tee logs/ravdess_preparation.log
    
    echo "\n[Step 2/5] Preparing CREMA-D dataset..."
    python scripts/prepare_cremad_dataset.py --output_dir $DATA_DIR/cremad | tee logs/cremad_preparation.log
    
    echo "\n[Step 3/5] Merging datasets and balancing emotions..."
    python scripts/merge_expressive_datasets.py \
        --input_manifests $DATA_DIR/ravdess/ravdess_dataset.jsonl $DATA_DIR/cremad/cremad_dataset.jsonl \
        --output_dir $DATA_DIR/combined \
        --max_samples_per_emotion 1000 | tee logs/dataset_merge.log
else
    echo "\n[Steps 1-3/5] Skipping dataset preparation as requested."
    if [ ! -f "$DATA_DIR/combined/expressive_train.jsonl" ]; then
        echo "Error: Training dataset not found at $DATA_DIR/combined/expressive_train.jsonl"
        echo "Please run without --skip-datasets first, or ensure the datasets exist."
        exit 1
    fi
fi

# Step 4: Verify configuration file
echo "\n[Step 4/5] Verifying configuration file..."
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found at $CONFIG_FILE"
    exit 1
fi

# Update configuration with correct paths if needed
sed -i.bak "s|manifest_path:.*|manifest_path: \"$DATA_DIR/combined/expressive_train.jsonl\"|" $CONFIG_FILE
sed -i.bak "s|val_manifest_path:.*|val_manifest_path: \"$DATA_DIR/combined/expressive_val.jsonl\"|" $CONFIG_FILE
rm $CONFIG_FILE.bak

# Step 5: Run fine-tuning
echo "\n[Step 5/5] Running fine-tuning..."

# Construct training command
TRAIN_CMD="torchrun --nproc-per-node $NUM_GPUS -m train $CONFIG_FILE"

# Add checkpoint resumption if specified
if [ -n "$RESUME_CHECKPOINT" ]; then
    TRAIN_CMD="$TRAIN_CMD --checkpoint.path=$RESUME_CHECKPOINT"
fi

# Run the training command
echo "Executing: $TRAIN_CMD"
$TRAIN_CMD | tee logs/expressive_finetuning.log

echo "\n=========================================="
echo "= Fine-tuning completed successfully! ="
echo "=========================================="
echo "Check the logs directory for detailed logs."
echo "Checkpoints are saved in: checkpoints/expressive_finetuning"
echo ""
echo "To use the fine-tuned model, run:"
echo "python -m moshi.server --lora-weight=checkpoints/expressive_finetuning/consolidated/lora.safetensors --config-path=checkpoints/expressive_finetuning/consolidated/config.json"
echo "==========================================" 