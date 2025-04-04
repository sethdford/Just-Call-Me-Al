#!/bin/bash
set -e

# Create directories if they don't exist
mkdir -p models/moshi
mkdir -p models/mimi

# Define model URLs - update these URLs with the actual sources when available
MOSHI_LM_URL="https://huggingface.co/kyutai-labs/moshi/resolve/main/language_model.safetensors"
TOKENIZER_URL="https://huggingface.co/kyutai-labs/moshi/resolve/main/tokenizer.model"
MIMI_MODEL_URL="https://huggingface.co/kyutai-labs/moshi/resolve/main/mimi/model.safetensors"

# Download the model files
echo "Downloading language model..."
if [ ! -f "models/moshi/language_model.safetensors" ]; then
    curl -L $MOSHI_LM_URL -o models/moshi/language_model.safetensors
else
    echo "Language model already exists, skipping download."
fi

echo "Downloading tokenizer..."
if [ ! -f "models/moshi/tokenizer.model" ]; then
    curl -L $TOKENIZER_URL -o models/moshi/tokenizer.model
else
    echo "Tokenizer already exists, skipping download."
fi

echo "Downloading Mimi model..."
if [ ! -f "models/mimi/model.safetensors" ]; then
    curl -L $MIMI_MODEL_URL -o models/mimi/model.safetensors
else
    echo "Mimi model already exists, skipping download."
fi

echo "Model downloads complete." 