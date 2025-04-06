use anyhow::{Result, anyhow};
use thiserror::Error;
use std::path::Path;
use std::sync::Arc;
use tracing::warn;
use candle_core;
use tokio::sync::Mutex as TokioMutex;
use async_trait::async_trait;
use crate::context::ConversationHistory;

// Import AudioProcessing trait (new name) instead of AudioProcessor
use crate::audio::AudioProcessing;

// --- Define needed modules ---
pub mod config;
pub mod csm;
pub mod prosody;
pub mod moshi_speech_model;
pub mod device;
pub mod moshi_impl;

// Test module
mod csm_tests;

// --- Re-exports ---
pub use config::CsmModelConfig;
pub use csm::CSMImpl;
pub use prosody::{ProsodyControl, EmotionalTone, ProsodyGenerator, ProsodyIntegration};
pub use moshi_speech_model::MoshiSpeechModel;
pub use device::Device;

// --- Error Type ---
#[derive(Debug, Error)]
pub enum ModelError {
    #[error("Method not implemented")]
    NotImplemented,
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Invalid output: {0}")]
    InvalidOutput(String),
    #[error("Load error: {0}")]
    LoadError(String),
    #[error("Process error: {0}")]
    ProcessError(String),
    #[error("Device error: {0}")]
    DeviceError(String),
    #[error("Channel send error: {0}")]
    ChannelSendError(String),
    #[error("Tensor operation failed: {0}")]
    TensorError(String),
    #[error("Configuration error: {0}")]
    ConfigError(String),
    #[error("Initialization error: {0}")]
    InitializationError(String),
    #[error("Tokenization error: {0}")]
    TokenizationError(String),
    #[error("Audio processing error: {0}")]
    AudioProcessingError(String),
    #[error("Prosody control error: {0}")]
    ProsodyControlError(String),
    #[error("Feature not enabled: {0}")]
    FeatureNotEnabled(String),
    #[error("Underlying model (Tch) error: {0}")]
    ModelSpecificError(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Safetensor error: {0}")]
    Safetensor(#[from] safetensors::SafeTensorError),
    #[error("LibTorch error: {0}")]
    Tch(#[from] tch::TchError),
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
    #[error("Feature extraction error: {0}")]
    FeatureExtraction(String),
    #[error("Inference error: {0}")]
    Inference(String),
    #[error("Token processing error: {0}")]
    TokenProcessing(String),
    #[error("Streaming error: {0}")]
    StreamingError(String),
    #[error("File I/O error: {0}")]
    FileIOError(String),
}

// --- CSMModel Trait ---
#[async_trait]
pub trait CSMModel: Send + Sync {
    /// Get the model configuration.
    fn get_config(&self) -> Result<CsmModelConfig, ModelError>;

    /// Get the audio processor associated with the model.
    fn get_processor(&self) -> Result<Arc<TokioMutex<dyn AudioProcessing + Send + Sync>>, ModelError>;

    /// Predict RVQ tokens from text and context.
    async fn predict_rvq_tokens(
        &self,
        text: &str,
        conversation_history: Option<&ConversationHistory>,
        temperature: Option<f32>,
    ) -> Result<Vec<Vec<i64>>, ModelError>;

    /// Synthesize audio samples from RVQ tokens.
    async fn synthesize_codes(
        &self,
    ) -> Result<AudioOutput, ModelError>;

    /// Synthesize audio samples from RVQ tokens, streaming the output.
    async fn synthesize_codes_streaming(
        &self,
    ) -> Result<(), ModelError>;

    /// Synthesize audio end-to-end from text.
    async fn synthesize(
        &self,
        text: &str,
        conversation_history: Option<&ConversationHistory>,
        temperature: Option<f32>,
        top_k: Option<i64>,
        seed: Option<i64>,
    ) -> Result<AudioOutput, ModelError>;

    /// Synthesize audio end-to-end from text, streaming the output.
    async fn synthesize_streaming(
        &self,
        text: &str,
        prosody: Option<ProsodyControl>,
        style_preset: Option<String>,
        audio_chunk_tx: tokio::sync::mpsc::Sender<Result<Vec<u8>, ModelError>>,
    ) -> Result<(), ModelError>;

    /* Commented out incompatible function
    /// Initiates streaming audio synthesis from text.
    ///
    // ... existing code ...
    */

    // TODO: Add a function to detect features/capabilities (e.g., supports_streaming)
}

// --- Simplified Device Mapping Logic ---

/// Map from our device abstraction to tch::Device
// Comment out unused function
/*
#[inline]
fn map_to_tch_device(device: Device) -> Result<tch::Device, ModelError> {
    match device {
        Device::Cpu => Ok(tch::Device::Cpu),
        Device::Cuda(ordinal) => Ok(tch::Device::Cuda(ordinal)),
        Device::Mps => {
            // tch-rs doesn't support MPS, fall back to CPU
            warn!("MPS not supported by tch-rs, falling back to CPU");
            Ok(tch::Device::Cpu)
        },
        Device::Vulkan => {
            warn!("Vulkan not supported by tch-rs, falling back to CPU");
            Ok(tch::Device::Cpu)
        },
    }
}
*/

/// Map from our device abstraction to candle_core::Device
#[inline]
fn map_to_candle_device(device: &Device) -> Result<candle_core::Device, ModelError> {
    match device {
        Device::Cpu => Ok(candle_core::Device::Cpu),
        Device::Cuda(idx) => {
            if candle_core::utils::cuda_is_available() {
                candle_core::Device::cuda_if_available(*idx)
                    .map_err(|e| ModelError::DeviceError(format!("Failed to create CUDA device (idx {}): {}", idx, e)))
            } else {
                warn!("CUDA requested but not available in Candle, falling back to CPU");
                Ok(candle_core::Device::Cpu)
            }
        },
        Device::Mps => {
            #[cfg(feature = "metal")]
            {
                if candle_core::utils::metal_is_available() {
                    match candle_core::Device::new_metal(0) {
                        Ok(device) => return Ok(device),
                        Err(e) => {
                            warn!("Failed to create Metal device: {}, falling back to CPU", e);
                        }
                    }
                } else {
                    warn!("Metal requested but not available in Candle, falling back to CPU");
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                warn!("Metal requested but feature not enabled in Candle, falling back to CPU");
            }
            Ok(candle_core::Device::Cpu)
        },
        Device::Vulkan => {
            warn!("Vulkan requested for Candle mapping, falling back to CPU.");
            Ok(candle_core::Device::Cpu)
        },
    }
}

// --- Factory Function ---
pub fn load_csm(
    model_path: &str,
    _config_path: &str,
    _vocoder_path: &str,
    _vocoder_config_path: &str,
    device: Device,
) -> Result<Box<dyn CSMModel + Send + Sync>> {
    use std::path::Path;
    
    let model_dir = Path::new(model_path);

    CSMImpl::new(model_dir, device)
        .map(|model| Box::new(model) as Box<dyn CSMModel + Send + Sync>)
        .map_err(|e| anyhow!("Failed to load CSMImpl: {}", e))
}

// Define placeholder types for AudioOutput and AudioChunk
#[derive(Debug, Clone)]
pub struct AudioOutput {
    pub samples: Vec<i16>, 
    pub sample_rate: u32,
}

#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub samples: Vec<i16>,
    pub is_final: bool,
}

// Audio processing types 
pub enum AudioCodesType {
    Feature,
    Linguistic,
}

// Audio token types
pub enum TokenType {
    Text,
    Prosody,
    Acoustic,
    Semantic,
    Unsupervised,  
}

// Factory function to create model - update to handle Moshi
pub fn create_model(config_path: &Path) -> Result<Arc<TokioMutex<Box<dyn CSMModel + Send>>>, ModelError> {
    let config = config::CsmModelConfig::from_file(config_path)
        .map_err(|e| ModelError::ConfigError(format!("Failed to load config: {}", e)))?;

    // Determine initial device based on config, defaulting to CPU
    let device_str = config.device_type.as_deref().unwrap_or("cpu");
    
    // Create device using our standardized Device enum
    let device = match device_str {
        "cpu" => Device::Cpu,
        "cuda" => Device::cuda_if_available(),
        "mps" => Device::mps_if_available(),
        "vulkan" => Device::Vulkan, // No fallback here since it's already handled in map_to_X_device funcs
        _ => {
            warn!("Unsupported device type '{}' in config, defaulting to CPU.", device_str);
            Device::Cpu
        }
    };

    let model_type = config.model_type.as_deref().unwrap_or("csm"); // Default to csm

    let model: Box<dyn CSMModel + Send> = match model_type {
        "csm" => {
            let model_paths = config.csm_model_paths()
                .ok_or_else(|| ModelError::ConfigError("CSM model paths not found or incorrect type in config".to_string()))?;
            let model_dir = Path::new(&model_paths.model_dir);
            Box::new(CSMImpl::new(model_dir, device.clone())?)
        }
        "moshi" => {
            moshi_speech_model::MoshiSpeechModel::new(&config, device.clone())?
        }
        _ => {
            return Err(ModelError::ConfigError(format!("Unsupported model type: {}", model_type)));
        }
    };

    Ok(Arc::new(TokioMutex::new(model)))
}

// Feature detection wrapper for the Moshi model
pub fn detect_moshi_features() -> bool {
    true
}

pub mod patch;