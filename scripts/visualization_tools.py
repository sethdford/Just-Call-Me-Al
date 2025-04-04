#!/usr/bin/env python3
"""
Visualization tools for monitoring expressive speech fine-tuning.

This module provides tools for visualizing and analyzing the fine-tuning
process and results, including:
- Training progress plots (loss, learning rate, etc.)
- Expressivity metrics visualization
- Comparison of models before and after fine-tuning
- Audio feature visualization

Usage:
    python visualization_tools.py --log-dir checkpoints/expressive_finetuning

Requirements:
    - matplotlib
    - pandas
    - seaborn
    - librosa
    - numpy
"""

import os
import argparse
import json
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display

# Set up plot style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

def load_training_state(log_dir: str) -> Dict[str, Any]:
    """
    Load training state from the checkpoint directory.
    
    Args:
        log_dir: Path to the checkpoint directory
        
    Returns:
        Dictionary with training state data
    """
    # Try to load final training state first
    final_state_path = os.path.join(log_dir, "final_training_state.json")
    if os.path.exists(final_state_path):
        with open(final_state_path, 'r') as f:
            return json.load(f)
    
    # If not available, look for the latest checkpoint
    checkpoints = sorted(glob.glob(os.path.join(log_dir, "checkpoint_*")))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {log_dir}")
    
    latest_checkpoint = checkpoints[-1]
    training_state_path = os.path.join(latest_checkpoint, "training_state.json")
    
    with open(training_state_path, 'r') as f:
        return json.load(f)

def plot_training_progress(training_state: Dict[str, Any], output_dir: str) -> None:
    """
    Plot training progress metrics from training state.
    
    Args:
        training_state: Training state dictionary
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert lists to arrays
    train_losses = np.array(training_state['train_losses'])
    step_numbers = np.arange(1, len(train_losses) + 1)
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # Plot 1: Training Loss
    axs[0].plot(step_numbers, train_losses, label='Training Loss', color='blue')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training Loss vs. Steps')
    axs[0].grid(True)
    
    # Add validation loss if available
    if 'val_losses' in training_state and training_state['val_losses']:
        val_losses = np.array(training_state['val_losses'])
        # We need to map validation steps to actual steps
        if 'eval_every' in training_state:
            eval_every = training_state['eval_every']
        else:
            # Estimate eval_every from the data
            total_steps = training_state['step']
            total_vals = len(val_losses)
            eval_every = total_steps // total_vals if total_vals > 0 else 50
            
        val_step_numbers = np.arange(eval_every, eval_every * (len(val_losses) + 1), eval_every)
        if len(val_step_numbers) > len(val_losses):
            val_step_numbers = val_step_numbers[:len(val_losses)]
        elif len(val_step_numbers) < len(val_losses):
            # Extend with extrapolated steps
            last_step = val_step_numbers[-1]
            additional_steps = np.arange(last_step + eval_every, 
                                        last_step + eval_every * (len(val_losses) - len(val_step_numbers) + 1), 
                                        eval_every)
            val_step_numbers = np.concatenate([val_step_numbers, additional_steps])
            
        axs[0].plot(val_step_numbers, val_losses, label='Validation Loss', color='red')
        axs[0].legend()
    
    # Plot 2: Learning Rate
    if 'learning_rates' in training_state and training_state['learning_rates']:
        learning_rates = np.array(training_state['learning_rates'])
        lr_steps = np.arange(1, len(learning_rates) + 1)
        axs[1].plot(lr_steps, learning_rates, label='Learning Rate', color='green')
        axs[1].set_ylabel('Learning Rate')
        axs[1].set_title('Learning Rate Schedule')
        axs[1].grid(True)
    
    # Plot 3: GPU Memory Usage (if available)
    if 'gpu_memory_usage' in training_state and training_state['gpu_memory_usage']:
        gpu_memory = np.array(training_state['gpu_memory_usage'])
        memory_steps = np.arange(1, len(gpu_memory) + 1)
        axs[2].plot(memory_steps, gpu_memory, label='GPU Memory (GB)', color='purple')
        axs[2].set_ylabel('Memory (GB)')
        axs[2].set_xlabel('Training Steps')
        axs[2].set_title('GPU Memory Usage')
        axs[2].grid(True)
    else:
        # If no GPU memory info, plot something else useful
        if 'best_val_metrics' in training_state and training_state['best_val_metrics']:
            axs[2].text(0.5, 0.5, f"Best Validation Metrics:\n{json.dumps(training_state['best_val_metrics'], indent=2)}", 
                      horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes)
        else:
            fig.delaxes(axs[2])
            plt.tight_layout()
    
    # Add overall title
    completed_status = "Completed" if training_state.get('completed', False) else "In Progress"
    total_steps = training_state['step']
    plt.suptitle(f'Training Progress ({completed_status}, {total_steps} steps)', fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(output_dir, 'training_progress.png'), dpi=300)
    plt.close(fig)
    
    # Create additional plots if we have validation metrics
    if ('best_val_metrics' in training_state and 
        training_state['best_val_metrics'] and 
        'expressivity' in training_state['best_val_metrics']):
        
        # Bar chart of best metrics
        metrics = training_state['best_val_metrics'].copy()
        if 'val_loss' in metrics:
            metrics.pop('val_loss')  # Already plotted above
            
        if metrics:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(list(metrics.keys()), list(metrics.values()), color=sns.color_palette("husl", len(metrics)))
            plt.title('Best Validation Metrics')
            plt.ylabel('Score')
            plt.ylim(0, 1.0 if max(metrics.values()) <= 1.0 else 5.0)  # Adjust for MOS scale if needed
            
            # Add value labels on the bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
                
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'validation_metrics.png'), dpi=300)
            plt.close()

def plot_audio_features(audio_path: str, output_path: str, emotion: Optional[str] = None) -> None:
    """
    Plot audio features for a speech sample.
    
    Args:
        audio_path: Path to audio file
        output_path: Path to save the output plot
        emotion: Emotion label (optional)
    """
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    
    # Plot 1: Waveform
    librosa.display.waveshow(y, sr=sr, ax=axs[0])
    axs[0].set_title('Waveform')
    axs[0].set_ylabel('Amplitude')
    
    # Plot 2: Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=axs[1])
    axs[1].set_title('Log-frequency Spectrogram')
    fig.colorbar(img, ax=axs[1], format='%+2.0f dB')
    
    # Plot 3: Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, y_axis='mel', x_axis='time', sr=sr, ax=axs[2])
    axs[2].set_title('Mel Spectrogram')
    fig.colorbar(img, ax=axs[2], format='%+2.0f dB')
    
    # Plot 4: Pitch (fundamental frequency)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), 
                                                fmax=librosa.note_to_hz('C7'), sr=sr)
    times = librosa.times_like(f0, sr=sr)
    axs[3].plot(times, f0, label='f0', color='blue')
    axs[3].set_yscale('log')
    axs[3].set_title('Pitch Contour (F0)')
    axs[3].set_ylabel('Frequency (Hz)')
    axs[3].set_xlabel('Time (s)')
    axs[3].grid(True)
    
    # Add emotion information if provided
    if emotion:
        plt.suptitle(f'Audio Features for {os.path.basename(audio_path)} - {emotion}', fontsize=16)
    else:
        plt.suptitle(f'Audio Features for {os.path.basename(audio_path)}', fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_emotion_comparison(
    evaluation_results: Dict[str, Any], 
    output_dir: str
) -> None:
    """
    Plot comparison of expressivity metrics across emotions.
    
    Args:
        evaluation_results: Dictionary with evaluation results
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract samples and their emotions
    samples = []
    for key, value in evaluation_results.items():
        if key.startswith('sample_') and isinstance(value, dict):
            # Try to extract emotion from the sample
            if 'emotion' in value:
                emotion = value['emotion']
                expressivity = value.get('expressivity', {}).get('overall', 0.0)
                emotion_match = value.get('expressivity', {}).get('emotion_match', 0.0)
                pronunciation = value.get('pronunciation', {}).get('overall', 0.0)
                mos = value.get('mos', 0.0)
                
                samples.append({
                    'sample': key,
                    'emotion': emotion,
                    'expressivity': expressivity,
                    'emotion_match': emotion_match,
                    'pronunciation': pronunciation,
                    'mos': mos
                })
    
    if not samples:
        print("No samples with emotion data found in evaluation results")
        return
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(samples)
    
    # Group by emotion and calculate means
    emotion_means = df.groupby('emotion').mean().reset_index()
    
    # Plot 1: Expressivity by emotion
    plt.figure(figsize=(12, 8))
    sns.barplot(data=emotion_means, x='emotion', y='expressivity', palette='viridis')
    plt.title('Overall Expressivity by Emotion')
    plt.ylim(0, 1.0)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'expressivity_by_emotion.png'), dpi=300)
    plt.close()
    
    # Plot 2: Emotion match by emotion (how well model expresses the intended emotion)
    plt.figure(figsize=(12, 8))
    sns.barplot(data=emotion_means, x='emotion', y='emotion_match', palette='viridis')
    plt.title('Emotion Match Score by Emotion')
    plt.ylim(0, 1.0)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'emotion_match_by_emotion.png'), dpi=300)
    plt.close()
    
    # Plot 3: All metrics by emotion
    metrics = ['expressivity', 'emotion_match', 'pronunciation', 'mos']
    
    # Normalize MOS to 0-1 scale for comparison
    if 'mos' in emotion_means:
        emotion_means['mos_normalized'] = emotion_means['mos'] / 5.0
        metrics = ['expressivity', 'emotion_match', 'pronunciation', 'mos_normalized']
    
    # Convert to long format for grouped bar chart
    emotion_means_long = pd.melt(
        emotion_means, 
        id_vars=['emotion'], 
        value_vars=metrics,
        var_name='Metric', 
        value_name='Score'
    )
    
    # Create custom labels for the legend
    metric_labels = {
        'expressivity': 'Overall Expressivity',
        'emotion_match': 'Emotion Match',
        'pronunciation': 'Pronunciation Quality',
        'mos_normalized': 'MOS (normalized)',
        'mos': 'MOS'
    }
    
    emotion_means_long['Metric'] = emotion_means_long['Metric'].map(lambda x: metric_labels.get(x, x))
    
    plt.figure(figsize=(14, 8))
    sns.barplot(data=emotion_means_long, x='emotion', y='Score', hue='Metric', palette='tab10')
    plt.title('Quality Metrics by Emotion')
    plt.ylim(0, 1.0)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_by_emotion.png'), dpi=300)
    plt.close()
    
    # Plot 4: Radar chart for each emotion
    if len(emotion_means) > 0:
        # Create radar chart
        fig = plt.figure(figsize=(15, 12))
        
        # Calculate number of rows and columns for subplots
        num_emotions = len(emotion_means)
        ncols = min(3, num_emotions)
        nrows = (num_emotions + ncols - 1) // ncols
        
        for i, (_, row) in enumerate(emotion_means.iterrows()):
            emotion = row['emotion']
            
            # Create radar chart for this emotion
            ax = fig.add_subplot(nrows, ncols, i+1, polar=True)
            
            # Number of metrics
            N = len(metrics)
            
            # Get values
            values = [row[metric] for metric in metrics]
            if 'mos' in metrics:
                # Scale MOS from 0-5 to 0-1
                mos_idx = metrics.index('mos')
                values[mos_idx] = values[mos_idx] / 5.0
            
            # Repeat the first value to close the polygon
            values += [values[0]]
            
            # Calculate angles for each metric
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += [angles[0]]  # Close the polygon
            
            # Plot data
            ax.plot(angles, values, linewidth=2, linestyle='solid')
            ax.fill(angles, values, alpha=0.25)
            
            # Set labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([metric_labels.get(metric, metric) for metric in metrics])
            
            # Set y limits
            ax.set_ylim(0, 1)
            
            # Add emotion as title
            ax.set_title(emotion.capitalize())
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'emotion_radar_charts.png'), dpi=300)
        plt.close(fig)

def compare_before_after(
    before_results: Dict[str, Any],
    after_results: Dict[str, Any],
    output_dir: str
) -> None:
    """
    Compare evaluation results before and after fine-tuning.
    
    Args:
        before_results: Evaluation results before fine-tuning
        after_results: Evaluation results after fine-tuning
        output_dir: Directory to save the comparison plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract average metrics
    if 'average' not in before_results or 'average' not in after_results:
        print("Average metrics not found in evaluation results")
        return
    
    before_avg = before_results['average']
    after_avg = after_results['average']
    
    # Combine metrics for comparison
    metrics = []
    for metric in ['expressivity', 'pronunciation', 'mos']:
        if metric in before_avg and metric in after_avg:
            metrics.append({
                'metric': metric,
                'before': before_avg[metric],
                'after': after_avg[metric]
            })
    
    if not metrics:
        print("No common metrics found for comparison")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(metrics)
    
    # Create grouped bar chart
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(df))
    width = 0.35
    
    plt.bar(x - width/2, df['before'], width, label='Before Fine-tuning', color='skyblue')
    plt.bar(x + width/2, df['after'], width, label='After Fine-tuning', color='lightcoral')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Performance Comparison Before vs After Fine-tuning')
    plt.xticks(x, df['metric'])
    plt.ylim(0, max(df['before'].max(), df['after'].max()) * 1.1)  # Add some headroom
    plt.legend()
    plt.grid(axis='y')
    
    # Add value labels
    for i, (before, after) in enumerate(zip(df['before'], df['after'])):
        plt.text(i - width/2, before + 0.01, f'{before:.3f}', 
                ha='center', va='bottom', color='black')
        plt.text(i + width/2, after + 0.01, f'{after:.3f}', 
                ha='center', va='bottom', color='black')
        
        # Add percentage improvement
        if before > 0:
            improvement = (after - before) / before * 100
            plt.text(i, min(before, after) / 2, f"{improvement:+.1f}%", 
                    ha='center', va='center', color='black',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'before_after_comparison.png'), dpi=300)
    plt.close()
    
    # Calculate and display relative improvements
    improvements = []
    for metric in metrics:
        before = metric['before']
        after = metric['after']
        if before > 0:
            rel_improvement = (after - before) / before * 100
            abs_improvement = after - before
            improvements.append({
                'metric': metric['metric'],
                'before': before,
                'after': after,
                'absolute_improvement': abs_improvement,
                'relative_improvement': rel_improvement
            })
    
    if improvements:
        df_imp = pd.DataFrame(improvements)
        
        # Plot relative improvements
        plt.figure(figsize=(10, 6))
        bars = plt.bar(df_imp['metric'], df_imp['relative_improvement'], color='lightgreen')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.xlabel('Metrics')
        plt.ylabel('Relative Improvement (%)')
        plt.title('Relative Improvement After Fine-tuning')
        plt.grid(axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'relative_improvements.png'), dpi=300)
        plt.close()

def visualize_evaluation_results(results_path: str, output_dir: str) -> None:
    """
    Visualize evaluation results from a JSON file.
    
    Args:
        results_path: Path to evaluation results JSON file
        output_dir: Directory to save visualization outputs
    """
    # Load evaluation results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot emotion comparison if possible
    plot_emotion_comparison(results, output_dir)
    
    # If average metrics are available, create summary plots
    if 'average' in results:
        avg = results['average']
        
        plt.figure(figsize=(10, 6))
        metrics = {k: v for k, v in avg.items() if k != 'sample_count'}
        
        bars = plt.bar(list(metrics.keys()), list(metrics.values()), color=sns.color_palette("husl", len(metrics)))
        plt.title('Average Evaluation Metrics')
        plt.ylabel('Score')
        plt.ylim(0, max(list(metrics.values())) * 1.1)  # Add some headroom
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
        
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'average_metrics.png'), dpi=300)
        plt.close()

def create_dashboard(log_dir: str, eval_dir: str, output_dir: str) -> None:
    """
    Create a comprehensive dashboard with all visualizations.
    
    Args:
        log_dir: Path to training logs directory
        eval_dir: Path to evaluation results directory
        output_dir: Directory to save the dashboard
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find training state
    try:
        training_state = load_training_state(log_dir)
        plot_training_progress(training_state, output_dir)
        print(f"Generated training progress plots in {output_dir}")
    except Exception as e:
        print(f"Error plotting training progress: {e}")
    
    # Find evaluation results
    eval_files = glob.glob(os.path.join(eval_dir, '*.json'))
    for eval_file in eval_files:
        try:
            # Create a subdirectory based on the filename
            file_basename = os.path.splitext(os.path.basename(eval_file))[0]
            eval_output_dir = os.path.join(output_dir, file_basename)
            os.makedirs(eval_output_dir, exist_ok=True)
            
            visualize_evaluation_results(eval_file, eval_output_dir)
            print(f"Generated evaluation visualizations for {file_basename} in {eval_output_dir}")
        except Exception as e:
            print(f"Error visualizing evaluation results from {eval_file}: {e}")
    
    # Check for before/after comparisons
    if len(eval_files) >= 2:
        # Try to find before and after files
        before_file = next((f for f in eval_files if 'before' in f.lower()), None)
        after_file = next((f for f in eval_files if 'after' in f.lower()), None)
        
        if before_file and after_file:
            try:
                with open(before_file, 'r') as f:
                    before_results = json.load(f)
                with open(after_file, 'r') as f:
                    after_results = json.load(f)
                
                compare_before_after(before_results, after_results, 
                                   os.path.join(output_dir, 'comparison'))
                print(f"Generated before/after comparison in {os.path.join(output_dir, 'comparison')}")
            except Exception as e:
                print(f"Error comparing before/after results: {e}")
    
    # Create HTML dashboard (optional)
    try:
        create_html_dashboard(output_dir)
        print(f"Generated HTML dashboard at {os.path.join(output_dir, 'dashboard.html')}")
    except Exception as e:
        print(f"Error creating HTML dashboard: {e}")

def create_html_dashboard(output_dir: str) -> None:
    """
    Create an HTML dashboard from generated visualizations.
    
    Args:
        output_dir: Directory with visualization outputs
    """
    # Find all generated images
    image_files = []
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                # Get relative path from output_dir
                rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                image_files.append(rel_path)
    
    if not image_files:
        print("No images found for HTML dashboard")
        return
    
    # Group images by directory
    images_by_dir = {}
    for img_path in image_files:
        directory = os.path.dirname(img_path)
        if directory == '':
            directory = 'main'
        
        if directory not in images_by_dir:
            images_by_dir[directory] = []
        
        images_by_dir[directory].append(img_path)
    
    # Create HTML content
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Expressive Fine-tuning Dashboard</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            h1, h2 {
                color: #333;
            }
            .section {
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
                padding: 20px;
            }
            .image-container {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                justify-content: center;
            }
            .image-card {
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                overflow: hidden;
                max-width: 100%;
            }
            .image-card img {
                max-width: 100%;
                height: auto;
                display: block;
            }
            .image-caption {
                padding: 10px;
                text-align: center;
                font-size: 14px;
                color: #666;
            }
            .navbar {
                position: sticky;
                top: 0;
                background-color: #333;
                padding: 10px;
                border-radius: 8px;
                margin-bottom: 20px;
            }
            .navbar a {
                color: white;
                text-decoration: none;
                margin-right: 15px;
                font-weight: bold;
            }
            .navbar a:hover {
                text-decoration: underline;
            }
            footer {
                text-align: center;
                margin-top: 30px;
                padding: 20px;
                color: #666;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <h1>Expressive Fine-tuning Dashboard</h1>
        
        <div class="navbar">
            <a href="#main">Main</a>
    """
    
    # Add navigation links
    for directory in images_by_dir.keys():
        if directory != 'main':
            html_content += f'<a href="#{directory}">{directory.replace("_", " ").title()}</a>\n'
    
    html_content += """
        </div>
        
        <div class="section">
            <p>This dashboard provides visualizations of the expressive speech fine-tuning process and results.</p>
            <p>Generated on: """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
        </div>
    """
    
    # Add sections for each directory
    for directory, images in images_by_dir.items():
        section_title = directory.replace("_", " ").title()
        html_content += f'''
        <div class="section" id="{directory}">
            <h2>{section_title}</h2>
            <div class="image-container">
        '''
        
        for img_path in images:
            img_name = os.path.basename(img_path)
            img_title = img_name.replace(".png", "").replace("_", " ").title()
            
            html_content += f'''
                <div class="image-card">
                    <img src="{img_path}" alt="{img_title}">
                    <div class="image-caption">{img_title}</div>
                </div>
            '''
            
        html_content += '''
            </div>
        </div>
        '''
    
    # Add footer
    html_content += """
        <footer>
            <p>CSM Expressivity Fine-tuning Dashboard</p>
        </footer>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(os.path.join(output_dir, 'dashboard.html'), 'w') as f:
        f.write(html_content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualization tools for expressive fine-tuning")
    parser.add_argument("--log-dir", type=str, help="Path to training logs directory")
    parser.add_argument("--eval-results", type=str, help="Path to evaluation results JSON file")
    parser.add_argument("--eval-dir", type=str, help="Path to directory with evaluation results")
    parser.add_argument("--audio", type=str, help="Path to audio file for feature visualization")
    parser.add_argument("--emotion", type=str, help="Emotion label for audio visualization")
    parser.add_argument("--output-dir", type=str, default="visualizations", 
                       help="Directory to save visualizations")
    parser.add_argument("--dashboard", action="store_true", 
                       help="Create comprehensive dashboard from log-dir and eval-dir")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.dashboard and args.log_dir and args.eval_dir:
        # Create comprehensive dashboard
        create_dashboard(args.log_dir, args.eval_dir, args.output_dir)
    elif args.log_dir:
        # Plot training progress
        try:
            training_state = load_training_state(args.log_dir)
            plot_training_progress(training_state, args.output_dir)
            print(f"Generated training progress plots in {args.output_dir}")
        except Exception as e:
            print(f"Error plotting training progress: {e}")
    elif args.eval_results:
        # Visualize evaluation results
        try:
            visualize_evaluation_results(args.eval_results, args.output_dir)
            print(f"Generated evaluation visualizations in {args.output_dir}")
        except Exception as e:
            print(f"Error visualizing evaluation results: {e}")
    elif args.audio:
        # Visualize audio features
        try:
            output_path = os.path.join(args.output_dir, 
                                     os.path.splitext(os.path.basename(args.audio))[0] + "_features.png")
            plot_audio_features(args.audio, output_path, args.emotion)
            print(f"Generated audio feature visualization at {output_path}")
        except Exception as e:
            print(f"Error visualizing audio features: {e}")
    else:
        parser.print_help() 