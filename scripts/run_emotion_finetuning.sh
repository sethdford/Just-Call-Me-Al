#!/bin/bash
# Script to automate emotion-specific fine-tuning with capability preservation
# Usage: ./run_emotion_finetuning.sh [--gpu N] [--skip-datasets] [--emotion EMOTION]

set -e  # Exit on error

# Default parameters
GPU_ID=0
CONFIG_FILE="configs/emotion_specific_finetuning.yaml"
SKIP_DATASETS=false
SINGLE_EMOTION=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --gpu)
            GPU_ID="$2"
            shift
            shift
            ;;
        --skip-datasets)
            SKIP_DATASETS=true
            shift
            ;;
        --emotion)
            SINGLE_EMOTION="$2"
            shift
            shift
            ;;
        --help)
            echo "Usage: ./run_emotion_finetuning.sh [--gpu N] [--skip-datasets] [--emotion EMOTION]"
            echo ""
            echo "Options:"
            echo "  --gpu N                 GPU ID to use (default: 0)"
            echo "  --skip-datasets         Skip dataset preparation steps"
            echo "  --emotion EMOTION       Fine-tune for a specific emotion only"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $key"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

echo "=========================================================="
echo "= CSM Emotion-Specific Fine-tuning with Cap Preservation ="
echo "=========================================================="
echo "GPU ID: $GPU_ID"
echo "Configuration file: $CONFIG_FILE"
echo "Skip datasets: $SKIP_DATASETS"
if [ -n "$SINGLE_EMOTION" ]; then
    echo "Fine-tuning for specific emotion: $SINGLE_EMOTION"
fi
echo "=========================================================="

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Create necessary directories
mkdir -p data
mkdir -p logs
mkdir -p emotion_models
mkdir -p evaluation_results

# 1. Prepare datasets (if not skipped)
if [ "$SKIP_DATASETS" = false ]; then
    echo "\n[Step 1/4] Checking for expressive datasets..."
    
    if [ ! -f "data/combined/expressive_train.jsonl" ]; then
        echo "Expressive datasets not found. Running dataset preparation first..."
        ./scripts/run_expressive_finetuning.sh --skip-datasets=false
    else
        echo "Expressive datasets found. Proceeding with fine-tuning."
    fi
    
    # Update config file to use the correct dataset path if needed
    sed -i.bak "s|data_path:.*|data_path: \"data/combined\"|" $CONFIG_FILE
    rm $CONFIG_FILE.bak
else
    echo "\n[Step 1/4] Skipping dataset preparation as requested."
    if [ ! -f "data/combined/expressive_train.jsonl" ]; then
        echo "Warning: Expressive datasets not found at data/combined/expressive_train.jsonl"
        echo "The fine-tuning might fail if datasets don't exist."
    fi
fi

# 2. Check if base model exists
echo "\n[Step 2/4] Checking for base expressive model..."
if [ ! -d "checkpoints/expressive_finetuning/consolidated" ]; then
    echo "Base expressive model not found at checkpoints/expressive_finetuning/consolidated"
    echo "Running expressive fine-tuning first to create base model..."
    ./scripts/run_expressive_finetuning.sh --gpus 1
else
    echo "Base expressive model found. Proceeding with emotion-specific fine-tuning."
fi

# Update config to use the base model path
sed -i.bak "s|base_model_path:.*|base_model_path: \"checkpoints/expressive_finetuning\"|" $CONFIG_FILE
rm $CONFIG_FILE.bak

# 3. Run emotion-specific fine-tuning
echo "\n[Step 3/4] Running emotion-specific fine-tuning..."

# If a specific emotion is provided, modify the config
if [ -n "$SINGLE_EMOTION" ]; then
    # Create a temporary config with only the specified emotion
    TMP_CONFIG="${CONFIG_FILE%.yaml}_tmp.yaml"
    cp $CONFIG_FILE $TMP_CONFIG
    
    # Replace emotions list with single emotion
    sed -i.bak "/^emotions:/,/^[a-z]/ s/^  - .*//g" $TMP_CONFIG
    sed -i.bak "/^emotions:/a\\  - \"$SINGLE_EMOTION\"" $TMP_CONFIG
    rm $TMP_CONFIG.bak
    
    # Use the temporary config
    CONFIG_FILE=$TMP_CONFIG
    echo "Created temporary config for emotion: $SINGLE_EMOTION"
fi

# Run the fine-tuning script
echo "Starting fine-tuning with config: $CONFIG_FILE"
python scripts/emotion_specific_finetuning.py --config $CONFIG_FILE 2>&1 | tee logs/emotion_finetuning.log

# Remove temporary config if created
if [ -n "$SINGLE_EMOTION" ] && [ -f "$TMP_CONFIG" ]; then
    rm $TMP_CONFIG
fi

# 4. Evaluate the emotion-specific models
echo "\n[Step 4/4] Evaluating emotion-specific models..."

# Extract emotions from the config
EMOTIONS=$(grep -A 10 "^emotions:" $CONFIG_FILE | grep -E "^\s*-" | sed 's/.*"\(.*\)".*/\1/')

# If a specific emotion was provided, only evaluate that one
if [ -n "$SINGLE_EMOTION" ]; then
    EMOTIONS="$SINGLE_EMOTION"
fi

# For each emotion, evaluate the model
for emotion in $EMOTIONS; do
    echo "Evaluating model for emotion: $emotion"
    
    # Check if the model exists
    if [ ! -d "emotion_models/$emotion/best" ]; then
        echo "Model for $emotion not found. Skipping evaluation."
        continue
    fi
    
    # Evaluate on test set
    python scripts/evaluation_metrics.py \
        --eval-dir evaluation/test_samples/$emotion \
        --emotion $emotion \
        --output evaluation_results/${emotion}_results.json \
        --batch 2>&1 | tee logs/${emotion}_evaluation.log
    
    # Generate visualization
    python scripts/visualization_tools.py \
        --eval-results evaluation_results/${emotion}_results.json \
        --output-dir evaluation_results/visualizations/$emotion
done

# Create a combined dashboard
python scripts/visualization_tools.py \
    --dashboard \
    --log-dir emotion_models \
    --eval-dir evaluation_results \
    --output-dir evaluation_results/dashboard

echo "\n=========================================================="
echo "= Emotion-specific fine-tuning completed! ="
echo "=========================================================="
echo "Check the logs directory for detailed logs."
echo "Emotion-specific models are saved in: emotion_models/<emotion>"
echo "Evaluation results are in: evaluation_results"
echo "Dashboard is available at: evaluation_results/dashboard/dashboard.html"
echo "=========================================================="

# Make the script executable
chmod +x scripts/run_emotion_finetuning.sh 