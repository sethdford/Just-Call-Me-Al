#!/usr/bin/env python3
"""
Script to download, preprocess, and prepare the CREMA-D dataset for fine-tuning.

CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset) contains 7,442 clips
from 91 actors of diverse ethnic backgrounds expressing different emotions.

This script:
1. Downloads the CREMA-D dataset
2. Converts the audio to the required format
3. Creates transcription JSON files
4. Generates the dataset manifest
5. Organizes the files in the expected structure for fine-tuning

Usage:
    python prepare_cremad_dataset.py --output_dir data/cremad

References:
    - https://github.com/CheyneyComputerScience/CREMA-D
    - Cao H, Cooper DG, Keutmann MK, Gur RC, Nenkova A, Verma R (2014) 
      CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset
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
import re

# Define the emotion mapping
EMOTION_MAP = {
    "ANG": {"emotion": "angry", "style": "normal"},
    "DIS": {"emotion": "disgust", "style": "normal"},
    "FEA": {"emotion": "fearful", "style": "normal"},
    "HAP": {"emotion": "happy", "style": "normal"},
    "NEU": {"emotion": "neutral", "style": "normal"},
    "SAD": {"emotion": "sad", "style": "normal"}
}

# Define the intensity mapping
INTENSITY_MAP = {
    "LO": "low",
    "MD": "medium",
    "HI": "high",
    "XX": "unspecified"
}

# Define the sentence mapping
CREMAD_SENTENCES = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I'll think about it some more",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes"
}

# Define the CREMA-D URL
CREMAD_URL = "https://github.com/CheyneyComputerScience/CREMA-D/archive/refs/heads/master.zip"

def download_cremad(output_dir):
    """Download the CREMA-D dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    zip_path = os.path.join(output_dir, "cremad.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading CREMA-D dataset to {zip_path}...")
        urllib.request.urlretrieve(CREMAD_URL, zip_path)
    
    extract_dir = os.path.join(output_dir, "raw")
    if not os.path.exists(extract_dir):
        print(f"Extracting dataset to {extract_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    
    return os.path.join(extract_dir, "CREMA-D-master")

def get_audio_files(extract_dir):
    """Get all audio files in the dataset."""
    audio_dir = os.path.join(extract_dir, "AudioWAV")
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Audio directory not found at {audio_dir}")
    
    audio_files = []
    for file in os.listdir(audio_dir):
        if file.endswith(".wav"):
            audio_files.append(os.path.join(audio_dir, file))
    
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

def load_actor_info(extract_dir):
    """Load actor information from the dataset."""
    actor_info_path = os.path.join(extract_dir, "ActorInfo", "ActorInfo.csv")
    if not os.path.exists(actor_info_path):
        print(f"Actor info file not found at {actor_info_path}")
        return {}
    
    actor_info = {}
    with open(actor_info_path, 'r') as f:
        # Skip header
        f.readline()
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                actor_id = parts[0]
                gender = parts[2]
                actor_info[actor_id] = {
                    "gender": gender.lower(),
                    "is_male": gender.lower() == "male"
                }
    
    return actor_info

def parse_filename(filename, actor_info):
    """Parse the CREMA-D filename to extract metadata."""
    # Format: ActorID_SentenceID_EmotionLevel.wav
    # Example: 1076_MTI_SAD_XX.wav
    
    basename = os.path.basename(filename)
    match = re.match(r"(\d+)_([A-Z]+)_([A-Z]+)_([A-Z]+)\.wav", basename)
    
    if not match:
        raise ValueError(f"Invalid filename format: {basename}")
    
    actor_id, sentence_id, emotion_code, intensity_code = match.groups()
    
    # Get emotion data
    emotion_data = EMOTION_MAP.get(emotion_code, {"emotion": "unknown", "style": "normal"})
    
    # Get intensity
    intensity = INTENSITY_MAP.get(intensity_code, "unspecified")
    
    # Get sentence text
    text = CREMAD_SENTENCES.get(sentence_id, "Unknown sentence")
    
    # Get actor info
    actor_data = actor_info.get(actor_id, {"gender": "unknown", "is_male": False})
    
    return {
        "text": text,
        "emotion": emotion_data["emotion"],
        "style": emotion_data["style"],
        "intensity": intensity,
        "actor_id": actor_id,
        "is_male": actor_data["is_male"],
        "gender": actor_data["gender"]
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
        "gender": metadata["gender"],
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
    """Process the CREMA-D dataset and prepare it for fine-tuning."""
    # Create output directories
    stereo_dir = os.path.join(output_dir, "data_stereo")
    os.makedirs(stereo_dir, exist_ok=True)
    
    # Load actor information
    actor_info = load_actor_info(extract_dir)
    
    # Get all audio files
    audio_files = get_audio_files(extract_dir)
    print(f"Found {len(audio_files)} audio files in the dataset.")
    
    # Process each audio file
    dataset_entries = []
    for audio_path in tqdm.tqdm(audio_files, desc="Processing audio files"):
        try:
            # Parse filename to get metadata
            metadata = parse_filename(audio_path, actor_info)
            
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
    manifest_path = os.path.join(output_dir, "cremad_dataset.jsonl")
    with open(manifest_path, 'w') as f:
        for entry in dataset_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Created dataset manifest at {manifest_path} with {len(dataset_entries)} entries.")
    
    # Create train/val split
    train_path = os.path.join(output_dir, "cremad_train.jsonl")
    val_path = os.path.join(output_dir, "cremad_val.jsonl")
    
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
    parser = argparse.ArgumentParser(description="Download and prepare the CREMA-D dataset for fine-tuning")
    parser.add_argument("--output_dir", type=str, default="data/cremad", 
                        help="Output directory for the prepared dataset")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download dataset
    extract_dir = download_cremad(args.output_dir)
    
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