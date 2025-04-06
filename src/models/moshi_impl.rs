use std::pin::Pin;
use thiserror::Error;
use tokenizers::Tokenizer;
use candle_core::{Device, Tensor, DType};
use tracing::{info, debug};
use tokio::sync::mpsc;
use futures::stream::{Stream, StreamExt};

// Moshi models
use moshi::mimi::Mimi;
use moshi::lm::LmModel;
use moshi::asr::State as AsrState;

// Project imports
use crate::models::{AudioOutput, AudioChunk};
use crate::models::prosody::ProsodyControl;

// Types for the internal implementation
#[derive(Debug, Clone)]
pub struct WordTimings {
    pub text: String,
    pub start_time: f64,
    pub stop_time: f64,
}

#[derive(Debug, Clone)]
pub struct OutputResult {
    pub word: WordTimings,
    pub confidence: f32,
}

// Error type for the implementation
#[derive(Error, Debug)]
pub enum SpeechModelError {
    #[error("Model initialization error: {0}")]
    Initialization(String),
    
    #[error("Model processing error: {0}")]
    Processing(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
}

// Inner implementation that does the actual work
pub struct MoshiSpeechModelImpl {
    // Remove model_path, replace with actual models
    // model_path: PathBuf, 
    mimi: Mimi, // Add Mimi model
    lm: LmModel,  // Add Language Model for ASR
    tokenizer: Tokenizer,
    device: Device,
}

impl MoshiSpeechModelImpl {
    pub fn new(
        // Add paths for Mimi and ASR LM
        mimi_model_path: &str,
        tokenizer_path: &str,
        model_type: &str,
        device: Device,
    ) -> Result<Self, SpeechModelError> {
        info!("Initializing MoshiSpeechModelImpl with model type: {}", model_type);
        
        // Verify all model paths are valid
        let mimi_dir = std::path::Path::new(mimi_model_path);
        if !mimi_dir.exists() {
            return Err(SpeechModelError::Initialization(
                format!("Model directory does not exist: {}", mimi_model_path)
            ));
        }
        
        // Expand path to model.safetensors if mimi_model_path is a directory
        let mimi_model_file = if mimi_dir.is_dir() {
            mimi_dir.join("model.safetensors")
        } else {
            mimi_dir.to_path_buf()
        };
        
        if !mimi_model_file.exists() {
            return Err(SpeechModelError::Initialization(
                format!("Mimi model file not found at: {}", mimi_model_file.display())
            ));
        }
        
        // Tokenizer path verification
        let tokenizer_path_obj = std::path::Path::new(tokenizer_path);
        if !tokenizer_path_obj.exists() {
            return Err(SpeechModelError::Initialization(
                format!("Tokenizer file not found at: {}", tokenizer_path)
            ));
        }
        
        // Load the tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| SpeechModelError::Tokenizer(
                format!("Failed to load tokenizer from {}: {}", tokenizer_path, e)
            ))?;

        // Look for ASR LM file in the same directory as the model
        let asr_lm_path = mimi_dir.join("asr_lm.safetensors");
        if !asr_lm_path.exists() {
            return Err(SpeechModelError::Initialization(
                format!("ASR LM file not found at: {}", asr_lm_path.display())
            ));
        }

        // Load Mimi model
        info!("Loading Mimi model from: {}", mimi_model_file.display());
        let mimi = moshi::mimi::load(
            mimi_model_file.to_str().unwrap_or(mimi_model_path), 
            None, 
            &device
        ).map_err(|e| SpeechModelError::Initialization(
            format!("Failed to load Mimi model: {}", e)
        ))?;

        // Load ASR LM
        info!("Loading Moshi ASR LM from: {}", asr_lm_path.display());
        let lm = moshi::lm::load_asr(
            asr_lm_path.to_str().unwrap_or(""), 
            candle_core::DType::F32, 
            &device
        ).map_err(|e| SpeechModelError::Initialization(
            format!("Failed to load Moshi ASR LM: {}", e)
        ))?;
        
        // Create the implementation
        Ok(Self {
            // Remove model_path
            mimi, 
            lm,
            tokenizer,
            device,
        })
    }
    
    // Process audio method - this would contain the actual speech processing logic
    pub async fn process_audio(&self, pcm_data: &[f32], sample_rate: u32) 
        -> Result<(Vec<OutputResult>, Vec<f32>), SpeechModelError> {
        
        // This is just a placeholder - in a real implementation, this would perform
        // audio processing using the ML model
        // TODO: Implement ASR logic using self.mimi and self.lm
        
        debug!("Processing audio: {} samples at {}Hz", pcm_data.len(), sample_rate);
        
        // Return a dummy result for now
        let dummy_word = WordTimings {
            text: "hello".to_string(),
            start_time: 0.0,
            stop_time: 1.0,
        };
        
        let dummy_result = OutputResult {
            word: dummy_word,
            confidence: 0.95,
        };
        
        Ok((vec![dummy_result], vec![0.0; 10])) // Feature vector
    }
    
    // Implement transcribe_stream method here
    // Asynchronously transcribe a stream of audio chunks
    pub async fn transcribe_stream(
        &self,
        mut audio_stream: Pin<Box<dyn Stream<Item = AudioChunk> + Send>>,
        results_sender: mpsc::Sender<String>,
    ) -> Result<(), SpeechModelError> {
        info!("Starting ASR transcription stream...");

        let _sample_rate: u32 = 16000; // TODO: Get from config or stream

        // Initialize ASR state with correct parameters
        // ASR state delay (in tokens), audio tokenizer (Mimi), language model
        let asr_delay_tokens = 5; // Example value, adjust based on requirements
        let mut asr_state = AsrState::new(asr_delay_tokens, self.mimi.clone(), self.lm.clone())
            .map_err(|e| SpeechModelError::Initialization(format!("Failed to create ASR state: {}", e)))?;

        while let Some(chunk) = audio_stream.next().await {
            debug!("Processing audio chunk of size: {}", chunk.samples.len());

            // Convert i16 samples to f32
            let pcm_f32: Vec<f32> = chunk.samples.iter()
                .map(|&s| s as f32 / 32768.0) // Convert i16 to f32 in [-1.0, 1.0]
                .collect();

            // Convert Vec<f32> to candle_core::Tensor
            let pcm_tensor = Tensor::from_vec(pcm_f32, (1, chunk.samples.len()), &self.device)
                 .map_err(|e| SpeechModelError::Processing(format!("Failed to create tensor from audio chunk: {}", e)))?;

            // Ensure tensor is f32
            let pcm_tensor = if pcm_tensor.dtype() != DType::F32 {
                 pcm_tensor.to_dtype(DType::F32)
                    .map_err(|e| SpeechModelError::Processing(format!("Failed to cast tensor to f32: {}", e)))?
            } else {
                 pcm_tensor
            };

            // step_pcm takes a callback that processes audio at a specific sample rate
            let segments = asr_state.step_pcm(pcm_tensor, |_sr, _audio| {
                // This callback is called by step_pcm with sample rate and audio tensor
                // We don't need to do anything special here for basic ASR
                Ok(())
            })
            .map_err(|e| SpeechModelError::Processing(format!("ASR step_pcm failed: {}", e)))?;

            // Process all segments
            for segment in segments.iter() {
                // Use two-argument decode
                let text = self.tokenizer.decode(&segment.tokens, true)
                    .map_err(|e| SpeechModelError::Tokenizer(format!("Failed to decode tokens: {}", e)))?;

                info!("ASR Segment: {}", text);
                
                // We need to await the send operation
                if results_sender.send(text).await.is_err() {
                    info!("ASR results channel closed by receiver.");
                    return Ok(());
                }
            }
        }

        // Log completion
        info!("Audio stream ended. ASR transcription stream finished.");
        Ok(())
    }
    
    // Synthesize method - placeholder for speech synthesis
    pub async fn synthesize(
        &mut self,
        text: &str,
        _prosody: Option<ProsodyControl>,
        _style_preset: Option<String>,
    ) -> Result<AudioOutput, SpeechModelError> {
        info!("Synthesizing text: {}", text);
        
        // This would contain the actual speech synthesis logic.
        // For now, return a dummy audio output
        
        let dummy_samples = vec![0i16; 1000]; // 1000 samples of silence
        
        Ok(AudioOutput {
            samples: dummy_samples,
            sample_rate: 24000,
        })
    }
    
    // Streaming synthesis method - placeholder
    pub async fn synthesize_streaming(
        &mut self,
        text: &str,
        _prosody: Option<ProsodyControl>,
        _style_preset: Option<String>,
        chunk_callback: Box<dyn Fn(AudioChunk) -> bool + Send>,
    ) -> Result<(), SpeechModelError> {
        info!("Streaming synthesis for text: {}", text);
        
        // For now, just produce a few chunks of silence
        let chunk_size = 480; // 20ms at 24kHz
        let num_chunks = 10;
        
        for i in 0..num_chunks {
            let samples = vec![0i16; chunk_size];
            let is_final = i == num_chunks - 1;
            
            let chunk = AudioChunk {
                samples,
                is_final,
            };
            
            // Call the callback with the chunk
            let continue_synthesis = chunk_callback(chunk);
            if !continue_synthesis {
                info!("Synthesis canceled by callback");
                break;
            }
        }
        
        Ok(())
    }
} 