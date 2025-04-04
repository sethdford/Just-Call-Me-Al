#!/bin/bash
set -e

# Create test audio directory
mkdir -p test_data/audio

# Download a sample WAV file from Common Voice dataset or another source
# Replace this URL with an actual sample audio file when available
SAMPLE_AUDIO_URL="https://example.com/test_audio.wav"

echo "Downloading sample audio file for testing..."
if [ ! -f "test_data/audio/test_audio.wav" ]; then
    curl -L $SAMPLE_AUDIO_URL -o test_data/audio/test_audio.wav
else
    echo "Sample audio file already exists, skipping download."
fi

echo "Sample audio download complete."
echo "You can test STT with: cargo run --bin stt_test -- test_data/audio/test_audio.wav" 