#!/usr/bin/env python3
"""
Script to merge multiple expressive speech datasets into a single balanced dataset.

This script takes multiple JSONL manifests as input and creates a merged manifest
with stratified sampling to ensure a balanced distribution of emotions and styles.
It also generates statistics about the merged dataset.

Usage:
    python merge_expressive_datasets.py \
        --input_manifests data/ravdess/ravdess_dataset.jsonl data/cremad/cremad_dataset.jsonl \
        --output_dir data/combined \
        --max_samples_per_emotion 1000 \
        --max_samples_per_style 1000

"""

import os
import json
import argparse
import random
import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from pathlib import Path

def read_manifest(manifest_path):
    """Read a JSONL manifest file and return the entries."""
    entries = []
    with open(manifest_path, 'r') as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    return entries

def merge_manifests(manifest_paths, max_samples_per_emotion=None, max_samples_per_style=None):
    """Merge multiple JSONL manifests and return the merged entries."""
    all_entries = []
    
    # Read all manifests
    for manifest_path in manifest_paths:
        entries = read_manifest(manifest_path)
        # Add source information to each entry
        for entry in entries:
            entry["source"] = os.path.basename(manifest_path).split('_')[0]
        all_entries.extend(entries)
    
    print(f"Read {len(all_entries)} entries from {len(manifest_paths)} manifests.")
    
    # Get all unique emotions and styles
    emotions = set(entry["emotion"] for entry in all_entries if "emotion" in entry)
    styles = set(entry["style"] for entry in all_entries if "style" in entry)
    
    print(f"Found {len(emotions)} emotions: {', '.join(emotions)}")
    print(f"Found {len(styles)} styles: {', '.join(styles)}")
    
    # Group entries by emotion and style
    entries_by_emotion = defaultdict(list)
    entries_by_style = defaultdict(list)
    
    for entry in all_entries:
        if "emotion" in entry:
            entries_by_emotion[entry["emotion"]].append(entry)
        if "style" in entry:
            entries_by_style[entry["style"]].append(entry)
    
    # Print statistics
    print("\nEntries per emotion:")
    for emotion, entries in sorted(entries_by_emotion.items()):
        print(f"  - {emotion}: {len(entries)} entries")
    
    print("\nEntries per style:")
    for style, entries in sorted(entries_by_style.items()):
        print(f"  - {style}: {len(entries)} entries")
    
    # Perform balanced sampling if requested
    balanced_entries = []
    
    if max_samples_per_emotion is not None:
        print(f"\nBalancing emotions (max {max_samples_per_emotion} samples per emotion)...")
        for emotion, entries in entries_by_emotion.items():
            if len(entries) > max_samples_per_emotion:
                # Shuffle entries for random sampling
                random.shuffle(entries)
                entries = entries[:max_samples_per_emotion]
            balanced_entries.extend(entries)
    
    if max_samples_per_style is not None and max_samples_per_emotion is None:
        print(f"\nBalancing styles (max {max_samples_per_style} samples per style)...")
        for style, entries in entries_by_style.items():
            if len(entries) > max_samples_per_style:
                # Shuffle entries for random sampling
                random.shuffle(entries)
                entries = entries[:max_samples_per_style]
            balanced_entries.extend(entries)
    
    # If neither balancing was requested, use all entries
    if max_samples_per_emotion is None and max_samples_per_style is None:
        balanced_entries = all_entries
    
    # Remove duplicates (based on path)
    seen_paths = set()
    unique_entries = []
    
    for entry in balanced_entries:
        if entry["path"] not in seen_paths:
            seen_paths.add(entry["path"])
            unique_entries.append(entry)
    
    print(f"\nFinal dataset: {len(unique_entries)} unique entries")
    
    return unique_entries

def generate_statistics(entries, output_dir):
    """Generate statistics about the merged dataset."""
    # Create a DataFrame for easier analysis
    df = pd.DataFrame(entries)
    
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Emotion distribution
    if "emotion" in df.columns:
        emotion_counts = df["emotion"].value_counts()
        
        plt.figure(figsize=(10, 6))
        emotion_counts.plot(kind="bar")
        plt.title("Distribution of Emotions")
        plt.xlabel("Emotion")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "emotion_distribution.png"))
        
        print("\nEmotion distribution:")
        for emotion, count in emotion_counts.items():
            print(f"  - {emotion}: {count} samples")
    
    # Style distribution
    if "style" in df.columns:
        style_counts = df["style"].value_counts()
        
        plt.figure(figsize=(10, 6))
        style_counts.plot(kind="bar")
        plt.title("Distribution of Styles")
        plt.xlabel("Style")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "style_distribution.png"))
        
        print("\nStyle distribution:")
        for style, count in style_counts.items():
            print(f"  - {style}: {count} samples")
    
    # Source distribution
    if "source" in df.columns:
        source_counts = df["source"].value_counts()
        
        plt.figure(figsize=(10, 6))
        source_counts.plot(kind="bar")
        plt.title("Distribution of Sources")
        plt.xlabel("Source")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "source_distribution.png"))
        
        print("\nSource distribution:")
        for source, count in source_counts.items():
            print(f"  - {source}: {count} samples")
    
    # Duration statistics
    if "duration" in df.columns:
        duration_stats = df["duration"].describe()
        
        plt.figure(figsize=(10, 6))
        df["duration"].plot(kind="hist", bins=50)
        plt.title("Distribution of Sample Durations")
        plt.xlabel("Duration (seconds)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "duration_distribution.png"))
        
        print("\nDuration statistics:")
        print(f"  - Mean: {duration_stats['mean']:.2f} seconds")
        print(f"  - Median: {duration_stats['50%']:.2f} seconds")
        print(f"  - Min: {duration_stats['min']:.2f} seconds")
        print(f"  - Max: {duration_stats['max']:.2f} seconds")
        print(f"  - Total: {df['duration'].sum() / 60:.2f} minutes")
    
    # Emotion-style cross-distribution
    if "emotion" in df.columns and "style" in df.columns:
        cross_table = pd.crosstab(df["emotion"], df["style"])
        
        plt.figure(figsize=(12, 8))
        cross_table.plot(kind="bar", stacked=True)
        plt.title("Emotion-Style Distribution")
        plt.xlabel("Emotion")
        plt.ylabel("Count")
        plt.legend(title="Style")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "emotion_style_distribution.png"))
        
        print("\nEmotion-Style distribution:")
        print(cross_table)
    
    # Save statistics to CSV
    stats_file = os.path.join(output_dir, "dataset_statistics.csv")
    with open(stats_file, 'w') as f:
        f.write("Category,Value,Count\n")
        
        if "emotion" in df.columns:
            for emotion, count in emotion_counts.items():
                f.write(f"emotion,{emotion},{count}\n")
        
        if "style" in df.columns:
            for style, count in style_counts.items():
                f.write(f"style,{style},{count}\n")
        
        if "source" in df.columns:
            for source, count in source_counts.items():
                f.write(f"source,{source},{count}\n")
    
    print(f"\nStatistics saved to {output_dir}")

def create_train_val_split(entries, output_dir, train_ratio=0.8):
    """Split the entries into training and validation sets."""
    # Shuffle entries
    random.seed(42)  # For reproducibility
    random.shuffle(entries)
    
    # Split
    split_idx = int(len(entries) * train_ratio)
    train_entries = entries[:split_idx]
    val_entries = entries[split_idx:]
    
    # Create JSONL files
    full_path = os.path.join(output_dir, "expressive_dataset.jsonl")
    train_path = os.path.join(output_dir, "expressive_train.jsonl")
    val_path = os.path.join(output_dir, "expressive_val.jsonl")
    
    # Write full dataset
    with open(full_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    
    # Write train set
    with open(train_path, 'w') as f:
        for entry in train_entries:
            f.write(json.dumps(entry) + '\n')
    
    # Write validation set
    with open(val_path, 'w') as f:
        for entry in val_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"\nDataset split:")
    print(f"  - Full dataset: {len(entries)} samples ({full_path})")
    print(f"  - Training set: {len(train_entries)} samples ({train_path})")
    print(f"  - Validation set: {len(val_entries)} samples ({val_path})")
    
    return full_path, train_path, val_path

def main():
    parser = argparse.ArgumentParser(description="Merge multiple expressive speech datasets")
    parser.add_argument("--input_manifests", type=str, nargs='+', required=True,
                        help="Input JSONL manifest files")
    parser.add_argument("--output_dir", type=str, default="data/combined",
                        help="Output directory for the merged dataset")
    parser.add_argument("--max_samples_per_emotion", type=int, default=None,
                        help="Maximum number of samples per emotion (for balancing)")
    parser.add_argument("--max_samples_per_style", type=int, default=None,
                        help="Maximum number of samples per style (for balancing)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Ratio of training samples (default: 0.8)")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Merge manifests
    merged_entries = merge_manifests(
        args.input_manifests,
        max_samples_per_emotion=args.max_samples_per_emotion,
        max_samples_per_style=args.max_samples_per_style
    )
    
    # Generate statistics
    generate_statistics(merged_entries, args.output_dir)
    
    # Create train/val split
    full_path, train_path, val_path = create_train_val_split(
        merged_entries,
        args.output_dir,
        train_ratio=args.train_ratio
    )
    
    print("\nDataset merge complete!")
    print(f"You can now use these manifests for fine-tuning:")
    print(f"  e.g., torchrun --nproc-per-node 1 -m train example/moshi_7B.yaml --dataset.manifest_path {train_path}")

if __name__ == "__main__":
    main() 