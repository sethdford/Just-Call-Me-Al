import csv
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np # For dummy audio creation
from scipy.io.wavfile import write # For dummy audio creation

# Ensure the metrics directory is in the Python path
project_root = Path(__file__).parent.parent
metrics_path = project_root / 'evaluation' / 'metrics'
sys.path.insert(0, str(project_root))

from evaluation.metrics.calculate_wer import calculate_wer
from evaluation.metrics.evaluate_homographs import evaluate_homographs
from evaluation.metrics.evaluate_pronunciation import evaluate_pronunciation
from evaluation.metrics.evaluate_prosody import evaluate_prosody

# --- Constants ---
TEST_SENTENCES_CSV = project_root / 'evaluation' / 'test_sentences.csv'
OUTPUT_AUDIO_DIR = project_root / 'evaluation' / 'output_audio'
OUTPUT_RESULTS_FILE = project_root / 'evaluation' / 'evaluation_results.json'

# --- Ensure Output Directory Exists ---
OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# --- Function to Create Dummy Audio (Helper) ---
def create_dummy_audio(file_path: Path, sr=22050, duration=2, freq=440):
    """Creates a simple sine wave audio file if it doesn't exist."""
    if not file_path.exists():
        print(f"Creating dummy audio file at {file_path}...")
        t = np.linspace(0., duration, int(sr * duration))
        amplitude = np.iinfo(np.int16).max * 0.3 # Reduced amplitude
        data = amplitude * np.sin(2. * np.pi * freq * t)
        try:
            write(file_path, sr, data.astype(np.int16))
            print(f"Dummy audio file created.")
        except Exception as e:
            print(f"Error creating dummy audio file {file_path}: {e}")
            return False
    return True

def read_test_sentences(csv_path: Path) -> List[Dict[str, Any]]:
    """Reads test sentences from the specified CSV file."""
    sentences = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                print(f"Warning: CSV file '{csv_path}' appears empty.")
                return []
            print(f"CSV Header: {header}")
            for i, row in enumerate(reader):
                if len(row) < 5:
                    print(f"Warning: Skipping malformed row {i+1} in {csv_path}: {row}")
                    continue
                sentences.append({
                    'id': row[0].strip(),
                    'text': row[1].strip(),
                    'type': row[2].strip(),
                    'emotion': row[3].strip(),
                    'style': row[4].strip(),
                })
    except FileNotFoundError:
        print(f"Error: Test sentences file not found at {csv_path}")
        return []
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return []
    return sentences

def generate_audio_and_transcribe(sentence_data: Dict[str, Any]) -> Tuple[str, Path | None]:
    """Placeholder for generating audio and getting ASR transcript.

    Returns:
        Tuple[str, Path | None]: The ASR transcript and the path to the audio file (or None if failed).
    """
    print(f"--- Simulating Audio Generation/Transcription for: {sentence_data['id']} ---")
    reference_text = sentence_data['text']
    audio_file_path = OUTPUT_AUDIO_DIR / f"{sentence_data['id']}.wav"

    # --- Placeholder Audio Generation ---
    # In reality, call Rust binary here. For now, create a dummy file.
    if not create_dummy_audio(audio_file_path):
        print(f"Failed to create dummy audio for {sentence_data['id']}. Skipping audio metrics.")
        audio_file_path = None
    # ----------------------------------

    # --- Placeholder ASR Simulation ---
    transcript = reference_text.lower().replace('.', '').replace('?', '').replace('!', '').replace(',', '')
    # Simulate errors
    if sentence_data['id'] == 'sent_011':
        transcript = transcript.replace("took the lead", "took the led")
    if sentence_data['id'] == 'sent_012':
        transcript = transcript.replace("out the refuse", "out the refyoos")
    # -------------------------------------

    print(f"Simulated ASR Transcript: {transcript}")
    print(f"Audio File Path: {audio_file_path}")
    return transcript, audio_file_path

def main():
    """Main function to run the evaluation pipeline."""
    print("Starting Evaluation Pipeline...")

    test_sentences = read_test_sentences(TEST_SENTENCES_CSV)
    if not test_sentences:
        print("No test sentences found. Exiting.")
        return
    print(f"Read {len(test_sentences)} sentences from {TEST_SENTENCES_CSV}")

    all_results = []
    total_wer = 0.0
    evaluated_count = 0

    for sentence in test_sentences:
        print(f"\nProcessing Sentence ID: {sentence['id']} - '{sentence['text']}'")

        # Generate Audio & Get ASR Transcript (Placeholder)
        asr_transcript, audio_path = generate_audio_and_transcribe(sentence)

        sentence_results = {
            'id': sentence['id'],
            'text': sentence['text'],
            'type': sentence['type'],
            'emotion': sentence['emotion'],
            'style': sentence['style']
        }

        # Calculate WER
        wer_score = calculate_wer(sentence['text'], asr_transcript)
        sentence_results['wer'] = wer_score
        total_wer += wer_score
        evaluated_count += 1
        print(f"WER: {wer_score:.4f}")

        # Evaluate Homographs if applicable
        if sentence['type'] == 'homograph':
            homograph_results = evaluate_homographs(sentence, asr_transcript)
            sentence_results['homograph_evaluation'] = homograph_results
            print(f"Homograph Results: {homograph_results}")

        # Calculate Audio-based Metrics if audio was generated
        if audio_path:
            # Pronunciation Evaluation
            pronunciation_results = evaluate_pronunciation(sentence, audio_path)
            sentence_results['pronunciation_evaluation'] = pronunciation_results
            print(f"Pronunciation Results: {pronunciation_results}")

            # Prosody Evaluation
            prosody_results = evaluate_prosody(sentence, audio_path)
            sentence_results['prosody_evaluation'] = prosody_results
            print(f"Prosody Results: {prosody_results}")
        else:
            print("Skipping audio-based metrics due to generation failure.")

        all_results.append(sentence_results)

    print("\n--- Evaluation Summary ---")
    if evaluated_count > 0:
        average_wer = total_wer / evaluated_count
        print(f"Average WER across {evaluated_count} sentences: {average_wer:.4f}")
    else:
        print("No sentences were evaluated.")

    # TODO: Add summary stats for pronunciation and prosody
    # TODO: Save detailed results to OUTPUT_RESULTS_FILE (e.g., as JSON)
    print(f"Detailed results (first few): {all_results[:2]}") # Show less detail for brevity
    print("\nEvaluation Pipeline Complete.")

if __name__ == "__main__":
    main()
