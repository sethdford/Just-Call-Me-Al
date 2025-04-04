use tch::Device;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use crate::models::ModelError;
use serde::Deserialize;

// Helper function for serde default
fn default_device() -> Device {
    Device::Cpu
}

// Based on Python ModelArgs and llama3_2_1B function
#[derive(Debug, Deserialize, Clone)]
pub struct CsmModelConfig {
    // Architecture Flavors (informational for now)
    pub backbone_flavor: String, // e.g., "llama-1B"
    pub decoder_flavor: String,  // e.g., "llama-100M" or "llama-1B"

    // Vocabulary Sizes
    pub text_vocab_size: i64,     // Llama vocab size: 128_256
    pub semantic_audio_vocab_size: i64, // NEW: Size for the single semantic codebook
    pub acoustic_vocab_size: i64, // RENAMED: Size for each acoustic codebook (was audio_vocab_size)
    pub num_acoustic_codebooks: i64, // RENAMED: Number of acoustic codebooks (was audio_num_codebooks)

    // Llama Backbone Params (from llama3_2_1B)
    pub backbone_num_layers: i64,    // 16
    pub backbone_num_heads: i64,     // 32 (Attention Heads)
    pub backbone_num_kv_heads: i64,  // 8  (Key/Value Heads for GQA)
    pub backbone_embed_dim: i64,     // 2048
    pub backbone_intermediate_dim: i64, // 8192
    pub backbone_norm_eps: f64,      // 1e-5
    pub backbone_rope_base: f64,     // 500_000.0

    // Decoder Params (Assuming 100M for now, adjust if needed)
    // Note: Decoder embed_dim might need adjustment based on projection layer
    pub decoder_num_layers: i64,    // 4
    pub decoder_num_heads: i64,     // 8
    pub decoder_num_kv_heads: i64,  // 2
    pub decoder_embed_dim: i64,     // 2048 (This is the internal dim, input comes from projection)
    pub decoder_intermediate_dim: i64, // 8192
    pub decoder_norm_eps: f64,      // 1e-5
    pub decoder_rope_base: f64,     // 500_000.0

    // Shared Params
    pub max_seq_len: i64,       // 2048 (Max sequence length for transformers)
    pub attn_dropout: f64,      // 0.0
    // pub scale_factor: f64,   // 32.0 (Seems specific to torchtune impl, maybe not needed directly)

    // LLM Integration Params
    #[serde(default)] // Use default if not present in config.json
    pub llm_embedding_dim: Option<i64>, // Optional dimension for incoming LLM embeddings

    // Synthesis Params (add defaults as needed)
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default = "default_top_k")]
    pub top_k: i64,

    // Runtime Params (can be overridden)
    #[serde(skip)]
    #[serde(default = "default_device")]
    pub device: Device,

    // Add other fields from config.json as needed
    pub vocab_size: i64, // Ensure this is loaded if needed
    pub num_codebooks: i64, // Ensure this is loaded if needed (used in RustCsmModel::synthesize_streaming_internal)
}

// Helper functions for serde defaults
fn default_temperature() -> f64 { 0.7 }
fn default_top_k() -> i64 { 50 }

impl CsmModelConfig {
    // Function to load from config.json
    pub fn from_file(path: &Path) -> Result<Self, ModelError> {
        let file = File::open(path).map_err(ModelError::Io)?;
        let reader = BufReader::new(file);
        // Restore original deserialization logic
        let config: Self = serde_json::from_reader(reader)
             .map_err(|e| ModelError::ConfigError(format!("Failed to parse config.json: {}", e)))?;
        // Note: Device field will be default (Cpu) after this, must be set externally if needed.
        Ok(config)
    }
}

// Placeholder for RVQ config if needed later
#[derive(Clone, Debug)]
pub struct RVQConfig {
   // ... RVQ parameters ...
}

// Re-define SynthesizerParams here and make it public
#[derive(Debug, Clone, Default, Deserialize)] // Added Deserialize for potential future use
pub struct SynthesizerParams {
    pub temperature: Option<f64>,
    pub guidance_scale: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<i64>,
    pub seed: Option<u64>,
    pub max_output_len: Option<usize>,
    pub eos_token_id: Option<i64>,
    pub prompt_tokens: Option<Vec<i64>>, // For prompt-based generation
    pub conditioning_embedding: Option<Vec<f32>>, // Direct embedding (might not be needed here)
    pub emotion: Option<String>, // Control token: Emotion
    pub style: Option<String>, // Control token: Style
}
