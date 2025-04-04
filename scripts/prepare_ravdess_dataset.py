#!/usr/bin/env python3
"""
Script to download, preprocess, and prepare the RAVDESS dataset for fine-tuning.

RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) is a dataset
of emotional speech recordings from 24 professional actors (12 female, 12 male)
speaking lexically-matched sentences with different emotions.

This script:
1. Downloads the RAVDESS dataset
2. Converts the audio to the required format
3. Creates transcription JSON files
4. Generates the dataset manifest
5. Organizes the files in the expected structure for fine-tuning

Usage:
    python prepare_ravdess_dataset.py --output_dir data/ravdess

References:
    - https://zenodo.org/record/1188976
    - Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of 
      Emotional Speech and Song (RAVDESS)
"""

import os
import json
import argparse
import shutil
from pathlib import Path
import urllib.request
import zipfile
import subprocess
import random
import tqdm
import soundfile as sf
import librosa
import numpy as np

# Define the audio sentences in RAVDESS
RAVDESS_SENTENCES = {
    "statement_1": "Kids are talking by the door",
    "statement_2": "Dogs are sitting by the door"
}

# Define the emotion mapping
EMOTION_MAP = {
    "01": {"emotion": "neutral", "style": "normal"},
    "02": {"emotion": "calm", "style": "normal"},
    "03": {"emotion": "happy", "style": "normal"},
    "04": {"emotion": "sad", "style": "normal"},
    "05": {"emotion": "angry", "style": "normal"},
    "06": {"emotion": "fearful", "style": "normal"},
    "07": {"emotion": "disgust", "style": "normal"},
    "08": {"emotion": "surprised", "style": "normal"}
}

# Define the intensity mapping
INTENSITY_MAP = {
    "01": "normal",
    "02": "strong"
}

# Define the statement mapping
STATEMENT_MAP = {
    "01": "statement_1",
    "02": "statement_2"
}

# Define the RAVDESS URL
RAVDESS_URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"

def download_ravdess(output_dir):
    """Download the RAVDESS dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    zip_path = os.path.join(output_dir, "ravdess.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading RAVDESS dataset to {zip_path}...")
        urllib.request.urlretrieve(RAVDESS_URL, zip_path)
    
    extract_dir = os.path.join(output_dir, "raw")
    if not os.path.exists(extract_dir):
        print(f"Extracting dataset to {extract_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    
    return extract_dir

def get_audio_files(extract_dir):
    """Get all audio files in the dataset."""
    audio_files = []
    for root, _, files in os.walk(extract_dir):
        for file in files:
            if file.endswith(".wav"):
                audio_files.append(os.path.join(root, file))
    return audio_files

def create_stereo_audio(audio_path, output_path):
    """Convert mono audio to stereo with the same content in both channels."""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=24000, mono=True)
        
        # Create stereo audio (duplicate mono audio to both channels)
        stereo_audio = np.vstack((audio, audio)).T
        
        # Save as 16-bit WAV
        sf.write(output_path, stereo_audio, sr, subtype='PCM_16')
        
        # Calculate duration
        duration = len(audio) / sr
        
        return duration
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def parse_filename(filename):
    """Parse the RAVDESS filename to extract metadata."""
    # Format: modality-vocal_channel-emotion-emotional_intensity-statement-repetition-actor.wav
    parts = os.path.basename(filename).split('.')[0].split('-')
    
    emotion_code = parts[2]
    intensity_code = parts[3]
    statement_code = parts[4]
    actor_id = parts[6]
    
    emotion_data = EMOTION_MAP.get(emotion_code, {"emotion": "unknown", "style": "normal"})
    intensity = INTENSITY_MAP.get(intensity_code, "normal")
    statement_key = STATEMENT_MAP.get(statement_code, "unknown")
    text = RAVDESS_SENTENCES.get(statement_key, "Unknown sentence")
    
    return {
        "text": text,
        "emotion": emotion_data["emotion"],
        "style": emotion_data["style"],
        "intensity": intensity,
        "actor_id": actor_id,
        "is_male": int(actor_id) % 2 == 1  # Odd numbers are male
    }

def create_json_annotation(audio_path, metadata, duration):
    """Create a JSON annotation file for the audio."""
    # Basic annotation structure
    annotation = {
        "text": metadata["text"],
        "emotion": metadata["emotion"],
        "style": metadata["style"],
        "intensity": metadata["intensity"],
        "actor_id": metadata["actor_id"],
        "gender": "male" if metadata["is_male"] else "female",
        "duration": duration,
        "segments": [
            {
                "start": 0,
                "end": duration,
                "text": metadata["text"],
                "speaker": "model" if metadata["is_male"] else "user"
            }
        ]
    }
    
    # Save JSON file
    json_path = os.path.splitext(audio_path)[0] + ".json"
    with open(json_path, 'w') as f:
        json.dump(annotation, f, indent=2)
    
    return json_path

def process_dataset(extract_dir, output_dir):
    """Process the RAVDESS dataset and prepare it for fine-tuning."""
    # Create output directories
    stereo_dir = os.path.join(output_dir, "data_stereo")
    os.makedirs(stereo_dir, exist_ok=True)
    
    # Get all audio files
    audio_files = get_audio_files(extract_dir)
    print(f"Found {len(audio_files)} audio files in the dataset.")
    
    # Process each audio file
    dataset_entries = []
    for audio_path in tqdm.tqdm(audio_files, desc="Processing audio files"):
        try:
            # Parse filename to get metadata
            metadata = parse_filename(audio_path)
            
            # Create an output filename
            filename = os.path.basename(audio_path)
            output_path = os.path.join(stereo_dir, filename)
            
            # Convert to stereo and get duration
            duration = create_stereo_audio(audio_path, output_path)
            if duration is None:
                continue
            
            # Create JSON annotation
            json_path = create_json_annotation(output_path, metadata, duration)
            
            # Add to dataset entries
            rel_path = os.path.relpath(output_path, output_dir)
            dataset_entries.append({
                "path": rel_path,
                "duration": duration,
                "emotion": metadata["emotion"],
                "style": metadata["style"]
            })
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue
    
    # Create dataset manifest
    manifest_path = os.path.join(output_dir, "ravdess_dataset.jsonl")
    with open(manifest_path, 'w') as f:
        for entry in dataset_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Created dataset manifest at {manifest_path} with {len(dataset_entries)} entries.")
    
    # Create train/val split
    train_path = os.path.join(output_dir, "ravdess_train.jsonl")
    val_path = os.path.join(output_dir, "ravdess_val.jsonl")
    
    # Shuffle and split (80% train, 20% val)
    random.seed(42)  # For reproducibility
    random.shuffle(dataset_entries)
    split_idx = int(len(dataset_entries) * 0.8)
    train_entries = dataset_entries[:split_idx]
    val_entries = dataset_entries[split_idx:]
    
    # Write train and val manifests
    with open(train_path, 'w') as f:
        for entry in train_entries:
            f.write(json.dumps(entry) + '\n')
    
    with open(val_path, 'w') as f:
        for entry in val_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Created train manifest with {len(train_entries)} entries.")
    print(f"Created validation manifest with {len(val_entries)} entries.")
    
    # Create dataset stats
    emotion_counts = {}
    style_counts = {}
    for entry in dataset_entries:
        emotion = entry["emotion"]
        style = entry["style"]
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        style_counts[style] = style_counts.get(style, 0) + 1
    
    print("\nDataset Statistics:")
    print("Emotion distribution:")
    for emotion, count in emotion_counts.items():
        print(f"  - {emotion}: {count} samples")
    
    print("Style distribution:")
    for style, count in style_counts.items():
        print(f"  - {style}: {count} samples")
    
    return manifest_path, train_path, val_path

def main():
    parser = argparse.ArgumentParser(description="Download and prepare the RAVDESS dataset for fine-tuning")
    parser.add_argument("--output_dir", type=str, default="data/ravdess", 
                        help="Output directory for the prepared dataset")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download dataset
    extract_dir = download_ravdess(args.output_dir)
    
    # Process dataset
    manifest_path, train_path, val_path = process_dataset(extract_dir, args.output_dir)
    
    print("\nDataset preparation complete!")
    print(f"  - Full dataset: {manifest_path}")
    print(f"  - Training set: {train_path}")
    print(f"  - Validation set: {val_path}")
    print("\nYou can now use these manifests for fine-tuning:")
    print("  e.g., torchrun --nproc-per-node 1 -m train example/moshi_7B.yaml --dataset.manifest_path", train_path)

if __name__ == "__main__":
    main() 