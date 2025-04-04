# Expressive Speech Datasets for Fine-tuning

This document outlines the approach for preparing high-quality expressive speech datasets for fine-tuning the CSM model to enhance its expressivity capabilities.

## 1. Dataset Selection Criteria

When selecting datasets for expressivity fine-tuning, we prioritize the following characteristics:

- **Emotional Diversity**: Datasets should cover a wide range of emotions (happiness, sadness, anger, fear, surprise, etc.)
- **Prosodic Variety**: The speech should exhibit varied prosodic features (pitch, intensity, rhythm, speed)
- **High Audio Quality**: Clean recordings with minimal background noise and consistent volume levels
- **Speaker Diversity**: Multiple speakers with different vocal characteristics (where possible)
- **Well-Annotated**: Accurate transcriptions with emotion/style labels
- **Sufficient Size**: Enough samples per emotion category (minimum 30 minutes per emotion)

## 2. Recommended Datasets

Based on our research, we recommend the following publicly available datasets:

### 2.1. RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Content**: 24 professional actors (12 female, 12 male) speaking lexically-matched sentences with different emotions
- **Emotions**: Neutral, calm, happy, sad, angry, fearful, disgust, surprised
- **Advantages**: High-quality recordings, consistent content across emotions
- **License**: Creative Commons BY-NC-SA 4.0
- **URL**: https://zenodo.org/record/1188976

### 2.2. CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)
- **Content**: 7,442 clips from 91 actors of diverse ethnic backgrounds
- **Emotions**: Anger, disgust, fear, happiness, neutral, sadness
- **Advantages**: Diverse speaker demographics, multiple emotional intensities
- **License**: Open access for research
- **URL**: https://github.com/CheyneyComputerScience/CREMA-D

### 2.3. ESD (Emotional Speech Dataset)
- **Content**: 350 parallel utterances by 10 native English speakers and 10 native Mandarin speakers
- **Emotions**: Neutral, happy, angry, sad, surprised
- **Advantages**: Bilingual, sentence-level emotion annotations
- **License**: Academic research only
- **URL**: https://github.com/HLTSingapore/Emotional-Speech-Data

### 2.4. MSP-PODCAST
- **Content**: Natural emotional speech from podcast recordings
- **Emotions**: Anger, happiness, sadness, neutral, and others
- **Advantages**: Naturally occurring emotions in conversational speech
- **License**: Application required
- **URL**: https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html

## 3. Dataset Preparation Process

The following steps should be taken to prepare the datasets for fine-tuning:

### 3.1. Format Conversion
Convert audio to consistent format:
- Sample rate: 24kHz (to match Moshi's requirements)
- Format: 16-bit WAV
- Channels: Stereo (with left channel for model output and right channel for input)

### 3.2. Data Preprocessing
- Apply normalization to ensure consistent volume
- Remove silence at beginning and end of recordings
- Filter out samples with poor recording quality
- Apply light noise reduction if necessary

### 3.3. Transcription and Annotation
Create JSON files with:
- Accurate transcriptions of speech content
- Emotion labels
- Speaking style annotations
- Any additional metadata (speaker ID, intensity, etc.)

### 3.4. Dataset Structure
Organize the dataset in the following structure:
```
data/
├── expressive_dataset.jsonl
└── data_stereo/
    ├── sample1.json
    ├── sample1.wav
    ├── sample2.json
    ├── sample2.wav
    └── ...
```

Where `expressive_dataset.jsonl` contains entries like:
```json
{"path": "data_stereo/sample1.wav", "duration": 3.75, "emotion": "happy", "style": "excited"}
{"path": "data_stereo/sample2.wav", "duration": 2.92, "emotion": "sad", "style": "whispering"}
```

### 3.5. Data Augmentation
Consider applying these augmentation techniques to increase dataset diversity:
- Pitch shifting (minor variations to simulate different speakers)
- Time stretching (for speed/rhythm variations within reasonable bounds)
- Adding minimal environmental noise (to improve robustness)

## 4. Kyutai DailyTalk Dataset Integration

As mentioned in the Moshi-finetune repository, the Kyutai DailyTalk dataset can be used as a good starting point. We should:
- Download this dataset from Hugging Face
- Analyze its emotional content and speaking styles
- Use it as a baseline and supplement with other expressive datasets

## 5. Custom Dataset Recording (Optional)

If the existing datasets don't provide sufficient coverage for our expressivity needs, we should consider recording custom samples focusing on:
- Underrepresented emotions or speaking styles
- Domain-specific content
- Consistent voice characteristics with our base model

## 6. Dataset Balancing

Ensure proper balancing across:
- Different emotions (equal representation)
- Speaking styles (normal, whispered, shouted, etc.)
- Sentence types (declarative, interrogative, exclamatory)
- Sentence lengths (short, medium, long utterances)

## 7. Validation Process

Before finalizing the dataset:
- Create a validation split (10-20% of data)
- Verify emotion annotations with human evaluators
- Test audio quality with objective metrics
- Ensure proper format for the fine-tuning pipeline

## 8. Implementation Plan

1. Download and preprocess selected datasets
2. Convert to required format and create annotation files
3. Merge datasets and ensure balanced representation
4. Split into training and validation sets
5. Create JSONL manifest file in the required format
6. Verify compatibility with the Moshi-finetune pipeline
7. Document dataset statistics and characteristics

## 9. References

1. Moshi-finetune repository: https://github.com/kyutai-labs/moshi-finetune
2. Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
3. Cao H, Cooper DG, Keutmann MK, Gur RC, Nenkova A, Verma R (2014) CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset
4. Zhou K, Sisman B, Liu R, Li H (2021) Emotional Speech Dataset for Speech Synthesis
5. Lotfian R, Busso C (2017) Building Naturalistic Emotionally Balanced Speech Corpus by Retrieving Emotional Speech from Existing Podcast Recordings 