use anyhow::{Result, anyhow};
use async_trait::async_trait;
use tch::{Device, TchError};
use std::path::Path;
use tokio::sync::mpsc;
// Note: Removed unused imports like Serialize, Deserialize, fmt, Arc, TokioMutex for now
// Add them back if they become necessary later

// --- Define needed modules ---
// Only declare modules that have corresponding files.
pub mod config;
pub mod csm;
pub mod moshi_speech_model;

// Remove declarations for non-existent files:
// pub mod attention;
// pub mod backbone;
// pub mod decoder;
// pub mod embeddings;
// pub mod vocoder;

// --- Re-exports ---
// Re-export items needed outside this module
pub use config::CsmModelConfig; // Example
// pub use csm::RustCsmModel; // Removed duplicate/unnecessary export
pub use csm::CSMImpl;      // Keep CSMImpl as the main export
// Remove the example SafeTensor export if it's not defined
// pub use tensor::SafeTensor; // Example
pub use moshi_speech_model::MoshiSpeechModel;

// --- Error Type ---
#[derive(Debug, thiserror::Error)]
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
    AudioError(String),

    // --- From implementations ---
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Safetensor error: {0}")]
    Safetensor(#[from] safetensors::SafeTensorError),
    #[error("LibTorch error: {0}")]
    Tch(#[from] TchError),
    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
}

// --- CSMModel Trait ---
#[async_trait]
pub trait CSMModel: Send + Sync {
    /// Synthesize audio fully, returning the complete i16 vector.
    async fn synthesize(
        &self,
        text: &str,
        temperature: Option<f64>,
        top_k: Option<i64>,
        seed: Option<u64>,
    ) -> Result<Vec<i16>, ModelError>;

    /// Synthesize multi-codebook tokens and stream them via an MPSC channel.
    async fn synthesize_streaming(
        &self,
        text: &str,
        temperature: Option<f64>,
        top_k: Option<i64>,
        seed: Option<u64>,
        audio_token_tx: mpsc::Sender<Vec<(i64, Vec<i64>)>>,
    ) -> Result<(), ModelError>;
}

// --- Factory Function ---
// REMOVED create_csm function

pub fn load_csm(
    model_dir: &Path,
    device: Device,
) -> Result<Box<dyn CSMModel + Send + Sync>> {
    // Use the CSMImpl constructor which handles wrapping
    CSMImpl::new(model_dir, device)
        .map(|impl_model| Box::new(impl_model) as Box<dyn CSMModel + Send + Sync>)
        .map_err(|e| anyhow!("Failed to load CSMImpl: {}", e))
}

// Ensure all previously defined structs/enums/traits like Model, DummyModel,
// ProsodyParams, ConversationContext, etc., are removed unless they are
// actually used by the CSMImpl or exported modules.