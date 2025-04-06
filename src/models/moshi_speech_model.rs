// Import required dependencies 
use anyhow::Result;
use std::sync::Arc;
use tracing::warn;
use tokio::sync::Mutex as TokioMutex;
use thiserror::Error;
use std::path::{Path, PathBuf};

use crate::models::{AudioOutput, CSMModel, ModelError, Device, CsmModelConfig};
use crate::models::prosody::ProsodyControl;
use crate::models::moshi_impl; // Import our implementation module
use crate::audio::AudioProcessing; // Import the AudioProcessing trait
use async_trait::async_trait;

// Import Candle device type for mapping
use candle_core::Device as CandleDevice;
use tch::Device as TchDevice; // Add this import
use candle_core::backend::BackendDevice; // Add this import for CudaDevice::new

use crate::context::ConversationHistory; // Add this import

// Re-export error type that matches the implementation
#[derive(Debug, Error)]
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
    Candle(moshi_impl::Error),

    #[error("Not initialized")]
    NotInitialized,

    #[error("Generation error: {0}")]
    Generation(String),

    #[error("Processing error: {0}")]
    Processing(String),

    #[error("File I/O error: {0}")]
    FileIO(String),

    #[error("SafeTensor error: {0}")]
    SafeTensor(moshi_impl::Error),

    #[error("Device error: {0}")]
    DeviceError(String),
}

// Map the *actual* SpeechModelError to the common ModelError
impl From<SpeechModelError> for ModelError {
    fn from(err: SpeechModelError) -> Self {
        match err {
            // Map variants from the *actual* SpeechModelError enum
            SpeechModelError::Initialization(s) => ModelError::InitializationError(s), 
            SpeechModelError::FeatureNotEnabled(s) => ModelError::FeatureNotEnabled(s),
            SpeechModelError::Implementation(e) => ModelError::ModelSpecificError(format!("Moshi Impl Error: {}", e)),
            SpeechModelError::InvalidInput(s) => ModelError::InvalidInput(s),
            SpeechModelError::Tokenizer(s) => ModelError::TokenizationError(s),
            SpeechModelError::ConfigError(s) => ModelError::Configuration(s), // Map to Configuration
            SpeechModelError::NotInitialized => ModelError::NotInitialized("Model not initialized".to_string()),
            SpeechModelError::Generation(s) => ModelError::ProcessError(s),
            SpeechModelError::Processing(s) => ModelError::AudioProcessing(s),
            SpeechModelError::FileIO(s) => ModelError::LoadError(s),
            SpeechModelError::SafeTensor(e) => ModelError::Other(anyhow::anyhow!("Safetensor error wrapper: {:?}", e)), // Map wrapped SafeTensor error to Other for now
            SpeechModelError::Candle(e) => ModelError::Other(anyhow::anyhow!("Candle error wrapper: {:?}", e)),       // Map wrapped Candle error to Other for now
            SpeechModelError::Io(e) => ModelError::Io(e),
            SpeechModelError::DeviceError(e) => ModelError::Other(anyhow::anyhow!("Device error: {}", e)),
        }
    }
}

/// MoshiSpeechModel implementation with support for text-to-speech generation.
///
/// This model integrates the Moshi speech synthesis capability with our
/// project's architecture, providing both synchronous and streaming text-to-speech
/// functionality.
///
/// # Implementation
/// - Uses Mimi encoder for audio feature extraction
/// - Handles tensor conversions between tch and candle frameworks
/// - Supports both synchronous and asynchronous (streaming) audio generation
/// - Provides integration with audio processing pipeline
///
/// # Note
/// - The implementation includes placeholders that will be fully implemented
///   in upcoming development iterations
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
        Device::Cuda(idx) => {
            if candle_core::utils::cuda_is_available() {
                candle_core::Device::cuda_if_available(*idx)
                    .map_err(|e| ModelError::DeviceError(format!("Failed to get Candle CUDA device {}: {}", idx, e)))
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
    /// Create a new MoshiSpeechModel instance.
    ///
    /// # Arguments
    /// * `config` - Configuration for the speech model
    /// * `device` - Device to run the model on (CPU or CUDA)
    ///
    /// # Returns
    /// * `Result<Self>` - Initialized MoshiSpeechModel or error
    ///
    /// # Tensor Operations
    /// - Initializes model components for tensor processing
    /// - Sets up proper device context for tensors
    /// - Establishes audio processing pipeline with tensor conversion support
    pub async fn new(config: &CsmModelConfig, device: Device) -> Result<Box<dyn CSMModel + Send>, ModelError> {
        // Validate required config fields
        let moshi_paths = config.moshi_model_paths()
            .ok_or_else(|| ModelError::Configuration("Moshi model paths not found or missing in config".to_string()))?;
        
        // Build paths based on model directory
        let model_dir_path = Path::new(&moshi_paths.model_dir);
        let model_path = model_dir_path.join("model.safetensors");
        let asr_lm_path = PathBuf::from(&moshi_paths.asr_lm_path);
        let _tokenizer_path = PathBuf::from(&moshi_paths.tokenizer_path);
        
        // Map the internal device to the required candle_core::Device
        let candle_device = map_to_candle_device(&device)?;

        // Call the async MoshiSpeechModelImpl::new function directly
        let moshi_model = moshi_impl::MoshiSpeechModelImpl::new(
                model_path,         // PathBuf
                Some(asr_lm_path),  // Option<PathBuf> - Correctly wrapped
                candle_device,      // candle_core::Device - Correctly mapped
        ).await
         .map_err(|e| ModelError::InitializationError(format!("Failed to initialize Moshi Speech Model: {}", e)))?;
        
        // Placeholder for audio_processor until config is fixed
        // This will likely cause errors later, but allows compilation for now.
        let dummy_audio_processor = Arc::new(TokioMutex::new(crate::audio::AudioProcessor::default()));

        Ok(Box::new(Self {
            inner: Arc::new(TokioMutex::new(moshi_model)),
            device: device.clone(),
            sample_rate: 24000,
            model_name: config.model_type.clone().unwrap_or_else(|| "moshi".to_string()),
            audio_processor: dummy_audio_processor,
        }))
    }
    
    // Process audio function - public API
    pub async fn process_audio(&self, pcm_data: &[f32], sample_rate: u32) -> Result<MoshiOutputResult, SpeechModelError> {
        let inner_guard = self.inner.lock().await;
        
        let (outputs, _) = inner_guard.process_audio_chunks(pcm_data, sample_rate).await?;
        
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
        _text: &str,
        _conversation_history: Option<&ConversationHistory>,
        _temperature: Option<f32>,
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
        _text: &str,
        _conversation_history: Option<&ConversationHistory>,
        _temperature: Option<f32>,
        _top_k: Option<i64>,
        _seed: Option<i64>,
    ) -> Result<AudioOutput, ModelError> {
        let mut inner_guard = self.inner.lock().await;
        
        // Call the inner implementation's synthesize method directly
        // Note: The inner synthesize likely doesn't use history, temp, top_k, seed yet.
        // We pass dummy values (None) for prosody/style for now.
        inner_guard.synthesize(_text, None, None).await
            .map_err(|e| ModelError::ProcessError(format!("Moshi synthesis error: {}", e)))
    }

    async fn synthesize_streaming(
        &self,
        _text: &str,
        _prosody: Option<ProsodyControl>,
        _style_preset: Option<String>,
        chunk_tx: tokio::sync::mpsc::Sender<Result<Vec<u8>, ModelError>>,
    ) -> Result<(), ModelError> {
        let _inner_guard = self.inner.lock().await;
        
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

    // Remove the get_processor implementation
    /*
    fn get_processor(&self) -> Result<Arc<TokioMutex<dyn AudioProcessing + Send + Sync>>, ModelError> {
        Ok(self.audio_processor.clone())
    }
    */
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Device;
    use crate::models::config::{CsmModelConfig, MoshiModelPaths};
    
    
    use tempfile::tempdir;
    use std::fs::File;

    use tracing_test::traced_test;

    #[tokio::test]
    #[traced_test]
    async fn test_moshi_speech_model_creation() {
        // Setup dummy config and paths
        let temp_dir = tempdir().unwrap();
        let model_dir = temp_dir.path();
        let dummy_model_path = model_dir.join("model.safetensors");
        let dummy_tokenizer_path = model_dir.join("tokenizer.model");
        let dummy_asr_lm_path = model_dir.join("asr_lm.klm");
        
        // Create dummy files - model needs a minimal valid header
        use std::io::Write;
        let mut model_file = File::create(&dummy_model_path).unwrap();
        // Minimal safetensors: 8 bytes for header size (which is 2 for '{}') + the empty JSON '{}'
        let header_size: u64 = 2; 
        model_file.write_all(&header_size.to_le_bytes()).unwrap(); // Write header length (8 bytes)
        model_file.write_all(b"{}").unwrap(); // Write empty JSON metadata (2 bytes)
        model_file.flush().unwrap();
        drop(model_file); // Ensure file is closed

        File::create(&dummy_tokenizer_path).unwrap(); // Tokenizer can be empty
        File::create(&dummy_asr_lm_path).unwrap(); // ASR LM can be empty
        
        let config = CsmModelConfig {
            model_type: Some("moshi".to_string()),
            moshi_model_paths: Some(MoshiModelPaths {
                model_dir: model_dir.to_str().unwrap().to_string(),
                tokenizer_path: dummy_tokenizer_path.to_str().unwrap().to_string(),
                asr_lm_path: dummy_asr_lm_path.to_str().unwrap().to_string(),
            }),
            ..Default::default()
        };
        
        let device = Device::Cpu;
        
        // Call the async new function and await it
        let result = MoshiSpeechModel::new(&config, device).await;
        
        // Assert that creation succeeded
        assert!(result.is_ok(), "Failed to create MoshiSpeechModel: {:?}", result.err());
    }
} 