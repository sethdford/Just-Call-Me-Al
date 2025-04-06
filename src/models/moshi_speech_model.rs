// Import required dependencies 
use anyhow::Result;
use std::sync::Arc;
use tracing::warn;
use tokio::sync::Mutex as TokioMutex;
use thiserror::Error;

use crate::models::{AudioOutput, CSMModel, ModelError, Device, CsmModelConfig};
use crate::models::prosody::ProsodyControl;
use crate::models::moshi_impl; // Import our implementation module
use async_trait::async_trait;

// Import Candle device type for mapping
use candle_core::Device as CandleDevice;
use tch::Device as TchDevice; // Add this import

use crate::audio::AudioProcessing;
use crate::context::ConversationHistory; // Add this import

// Re-export error type that matches the implementation
#[derive(Error, Debug)]
pub enum SpeechModelError { 
    #[error("Feature not enabled: {0}")]
    FeatureNotEnabled(String),

    #[error("Implementation error: {0}")] 
    Implementation(#[from] moshi_impl::SpeechModelError),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Configuration error: {0}")] 
    ConfigError(String),

    #[error("Initialization error: {0}")] 
    Initialization(String),

    #[error("I/O error: {0}")] 
    Io(#[from] std::io::Error),

    #[error("Candle error: {0}")] 
    Candle(#[from] candle_core::Error),
}

// Map facade error to the common ModelError
impl From<SpeechModelError> for ModelError {
    fn from(err: SpeechModelError) -> Self {
        match err {
            SpeechModelError::FeatureNotEnabled(s) => ModelError::FeatureNotEnabled(s),
            SpeechModelError::Implementation(e) => ModelError::ModelSpecificError(format!("Moshi Impl Error: {}", e)),
            SpeechModelError::InvalidInput(s) => ModelError::InvalidInput(s),
            SpeechModelError::Tokenizer(s) => ModelError::TokenizationError(s),
            SpeechModelError::ConfigError(s) => ModelError::ConfigError(s),
            SpeechModelError::Initialization(s) => ModelError::InitializationError(s), 
            SpeechModelError::Io(e) => ModelError::Io(e),
            SpeechModelError::Candle(e) => ModelError::Candle(e),
        }
    }
}

// Public API struct
pub struct MoshiSpeechModel {
    inner: Arc<TokioMutex<moshi_impl::MoshiSpeechModelImpl>>,
    device: Device,
    sample_rate: u32,
    model_name: String,
    audio_processor: Arc<TokioMutex<dyn AudioProcessing + Send + Sync>>,
}

// Public types needed for API
#[derive(Debug, Clone)]
pub struct RecognizedWord {
    pub text: String,
    pub start_time: f64,
    pub stop_time: f64,
}

#[derive(Debug, Clone)]
pub struct MoshiOutputResult {
    pub words: Vec<RecognizedWord>,
}

// Helper function to map models::Device to candle_core::Device
// This is duplicated from models/mod.rs - consider moving to models/device.rs
fn map_to_candle_device(device: &Device) -> Result<CandleDevice, ModelError> {
    match device {
        Device::Cpu => Ok(CandleDevice::Cpu),
        Device::Cuda(idx) => { // Match with index
            if candle_core::utils::cuda_is_available() {
                candle_core::Device::cuda_if_available(*idx)
                    .map_err(|e| ModelError::DeviceError(format!("Failed to create CUDA device (idx {}): {}", idx, e)))
            } else {
                warn!("CUDA requested but not available in Candle, falling back to CPU");
                Ok(CandleDevice::Cpu)
            }
        },
        Device::Mps => { // Match Mps variant
            #[cfg(feature = "metal")]
            {
                if candle_core::utils::metal_is_available() {
                    // Use new_metal to attempt creation
                    match candle_core::Device::new_metal(0) {
                        Ok(d) => Ok(d),
                        Err(e) => {
                            warn!("Failed to create Metal device: {}, falling back to CPU", e);
                            Ok(CandleDevice::Cpu)
                        }
                    }
                } else {
                    warn!("Metal requested but not available in Candle, falling back to CPU");
                    Ok(CandleDevice::Cpu)
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                warn!("Metal requested but feature not enabled in Candle, falling back to CPU");
                Ok(CandleDevice::Cpu)
            }
        },
        Device::Vulkan => {
             warn!("Vulkan requested for Candle mapping, falling back to CPU.");
            Ok(candle_core::Device::Cpu)
        }
    }
}

// Add this helper function definition EARLIER
#[inline]
fn map_to_tch_device(device: &Device) -> Result<TchDevice, ModelError> {
    match device {
        Device::Cpu => Ok(TchDevice::Cpu),
        Device::Cuda(ordinal) => Ok(TchDevice::Cuda(*ordinal)),
        Device::Mps => {
            warn!("Moshi model (tch backend) does not support MPS, falling back to CPU.");
            Ok(TchDevice::Cpu)
        },
        Device::Vulkan => {
            warn!("Moshi model (tch backend) does not support Vulkan, falling back to CPU.");
            Ok(TchDevice::Cpu)
        },
    }
}

impl MoshiSpeechModel {
    pub fn new(config: &CsmModelConfig, device: Device) -> Result<Box<dyn CSMModel + Send>, ModelError> {
        // Convert our device to candle device
        let candle_device = map_to_candle_device(&device)?;
        
        // Convert our device to tch device
        let tch_device = map_to_tch_device(&device)?;

        // Get model directory and file paths from config
        let model_dir = if let Some(ref paths) = config.moshi_model_paths {
            paths.model_dir.clone()
        } else if let Some(ref dir) = config.model_dir {
            dir.clone()
        } else {
            return Err(ModelError::ConfigError("Missing model directory path in config".to_string()));
        };
        
        // Get tokenizer path from config
        let tokenizer_path = if let Some(ref paths) = config.moshi_model_paths {
            paths.tokenizer_path.clone()
        } else if let Some(ref path) = config.tokenizer_path {
            path.clone()
        } else {
            // Default to model_dir/tokenizer.json if not explicitly specified
            format!("{}/tokenizer.json", model_dir)
        };
        
        // Get model type
        let model_type = config.model_type.as_deref().unwrap_or("moshi");

        // Initialize the MoshiSpeechModelImpl
        let moshi_model = moshi_impl::MoshiSpeechModelImpl::new(
            &model_dir,
            &tokenizer_path,
            model_type,
            candle_device,
        ).map_err(|e| ModelError::InitializationError(format!("Failed to initialize Moshi Speech Model: {}", e)))?;
        
        // Create a fake audio processor to satisfy the interface
        let mimi_config = crate::audio::codec::MimiEncoderConfig {
            sample_rate: 24000.0,
            input_channels: 1,
            dimension: 128,
            hidden_size: 512,
            num_hidden_layers: 4,
            ..Default::default()
        };
        
        let rvq_config = crate::rvq::RVQConfig {
            num_codebooks: 8,
            codebook_size: 1024,
            vector_dim: 128,
            device: tch_device.clone(), // Assign the converted tch::Device
            learning_rate: 0.0,
            normalize: false,
        };
        
        // Create audio processor without loading weights
        let audio_processor = crate::audio::AudioProcessor::new(
            mimi_config,
            rvq_config,
            None,
            None,
            tch_device // Use the converted tch_device
        )
        .map_err(|e| ModelError::InitializationError(format!("Failed to create audio processor: {}", e)))?;
        
        Ok(Box::new(Self {
            inner: Arc::new(TokioMutex::new(moshi_model)),
            device: device.clone(),
            sample_rate: 24000,
            model_name: config.model_type.clone().unwrap_or_else(|| "moshi".to_string()),
            audio_processor: Arc::new(TokioMutex::new(audio_processor)),
        }))
    }
    
    // Process audio function - public API
    pub async fn process_audio(&self, pcm_data: &[f32], sample_rate: u32) -> Result<MoshiOutputResult, SpeechModelError> {
        let inner_guard = self.inner.lock().await;
        
        let (outputs, _) = inner_guard.process_audio(pcm_data, sample_rate).await?;
        
        // Convert internal types to public API types
        let words = outputs.into_iter()
            .map(|out| RecognizedWord {
                text: out.word.text,
                start_time: out.word.start_time,
                stop_time: out.word.stop_time,
            })
            .collect();
            
        Ok(MoshiOutputResult { words })
    }
    
    /// Returns true if the MoshiSpeechModel feature is enabled
    #[inline]
    pub fn is_enabled(&self) -> bool {
        true // Always enabled now
    }
}

// Add CSMModel implementation for MoshiSpeechModel
#[async_trait]
impl CSMModel for MoshiSpeechModel {
    // Comment out methods not in the trait
    /*
    fn device(&self) -> Device { // Assuming Device is crate::models::Device
        self.device.clone()
    }
    */

    /// Predict RVQ tokens from text and context.
    async fn predict_rvq_tokens(
        &self,
        text: &str,
        conversation_history: Option<&ConversationHistory>,
        temperature: Option<f32>,
    ) -> Result<Vec<Vec<i64>>, ModelError> {
        warn!("MoshiSpeechModel does not support predict_rvq_tokens.");
        Err(ModelError::NotImplemented)
    }

    // Implement synthesize_codes (match trait signature)
    async fn synthesize_codes(
        &self, 
    ) -> Result<AudioOutput, ModelError> {
        warn!("MoshiSpeechModel::synthesize_codes is not implemented.");
        Err(ModelError::NotImplemented)
    }

    // Implement synthesize_codes_streaming (match trait signature)
    async fn synthesize_codes_streaming(
        &self,
    ) -> Result<(), ModelError> {
        warn!("MoshiSpeechModel::synthesize_codes_streaming is not implemented.");
        Err(ModelError::NotImplemented)
    }

    async fn synthesize(
        &self,
        text: &str,
        conversation_history: Option<&ConversationHistory>,
        temperature: Option<f32>,
        top_k: Option<i64>,
        seed: Option<i64>,
    ) -> Result<AudioOutput, ModelError> {
        let mut inner_guard = self.inner.lock().await;
        
        // Call the inner implementation's synthesize method directly
        // Note: The inner synthesize likely doesn't use history, temp, top_k, seed yet.
        // We pass dummy values (None) for prosody/style for now.
        inner_guard.synthesize(text, None, None).await
            .map_err(|e| ModelError::ProcessError(format!("Moshi synthesis error: {}", e)))
    }

    async fn synthesize_streaming(
        &self,
        text: &str,
        prosody: Option<ProsodyControl>,
        style_preset: Option<String>,
        chunk_tx: tokio::sync::mpsc::Sender<Result<Vec<u8>, ModelError>>,
    ) -> Result<(), ModelError> {
        let inner_guard = self.inner.lock().await;
        
        // Call the inner implementation's synthesize_streaming method
        // Note: The inner synthesize_streaming currently expects std::sync::mpsc.
        // We need to bridge tokio::sync::mpsc to std::sync::mpsc or refactor the inner impl.
        // For now, this will cause a type error inside the inner call.
        warn!("Moshi synthesize_streaming needs bridging between tokio and std mpsc.");
        // Placeholder: return error until bridging is implemented
        let _ = chunk_tx; // Use the parameter to avoid unused warning
        Err(ModelError::NotImplemented)
        // inner_guard.synthesize_streaming(text, prosody, style_preset, chunk_tx).await
        //      .map_err(|e| ModelError::ProcessError(format!("Moshi streaming synthesis error: {}", e)))
    }

    // Comment out methods not in the trait
    /*
    async fn synthesize_with_history(
        &self,
        _text: &str,
        _history: &[String],
        _prosody: Option<ProsodyControl>,
        _style_preset: Option<String>,
    ) -> Result<AudioOutput, ModelError> {
        warn!("MoshiSpeechModel does not support synthesize_with_history.");
        Err(ModelError::NotImplemented)
    }
    */
    
    /*
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    */

    /*
    fn model_name(&self) -> &str {
        &self.model_name
    }
    */

    /*
    fn process_audio(&self, _audio: &[f32]) -> Result<Vec<i64>, ModelError> {
        warn!("MoshiSpeechModel::process_audio (trait method) is not implemented.");
        Err(ModelError::NotImplemented)
    }
    */

    // Keep get_config
    fn get_config(&self) -> Result<CsmModelConfig, ModelError> {
        // Placeholder: Return a default or dummy config
        Ok(CsmModelConfig::default()) // Return Ok() with a value
    }

    // Update get_processor implementation to use AudioProcessing
    fn get_processor(&self) -> Result<Arc<TokioMutex<dyn AudioProcessing + Send + Sync>>, ModelError> {
        // No need to wrap again, self.audio_processor is already the correct type
        Ok(self.audio_processor.clone())
    }
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    
    use crate::models::config::MoshiModelPaths;
    use tracing_test::traced_test;
    
    #[tokio::test]
    #[traced_test]
    async fn test_moshi_speech_model_creation() {
        // Use tempfile for temporary directories and files
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let model_dir = temp_dir.path().join("models").join("mimi");
        std::fs::create_dir_all(&model_dir).expect("Failed to create model dir");

        // Create dummy files expected by the model loader
        let tokenizer_path = model_dir.join("tokenizer.json");
        let asr_lm_path = model_dir.join("asr_lm.safetensors");
        let model_path = model_dir.join("model.safetensors"); // The internal model it tries to load
        std::fs::File::create(&tokenizer_path).expect("Failed to create dummy tokenizer file");
        std::fs::File::create(&asr_lm_path).expect("Failed to create dummy ASR LM file");
        std::fs::File::create(&model_path).expect("Failed to create dummy model file");

        // Define config using the temporary paths
        let config = CsmModelConfig {
            model_type: Some("moshi".to_string()),
            device_type: Some("cpu".to_string()),
            moshi_model_paths: Some(MoshiModelPaths {
                model_dir: model_dir.to_str().unwrap().to_string(),
                tokenizer_path: tokenizer_path.to_str().unwrap().to_string(),
                asr_lm_path: asr_lm_path.to_str().unwrap().to_string(),
            }),
            ..Default::default()
        };
        
        // Test with CPU device
        let device = Device::Cpu;
        
        // This should now likely fail with LoadError due to invalid dummy files, or still InitializationError
        let result = MoshiSpeechModel::new(&config, device);
        
        // The result should be an error
        assert!(result.is_err());
        match result {
            // Update expected error to InitializationError
            Err(ModelError::InitializationError(_)) => (), // Expected error type
            Err(e) => panic!("Expected InitializationError but got: {:?}", e),
            Ok(_) => panic!("Expected error but got Ok"),
        }
    }
} 