use tch::Device;
use std::path::Path;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use anyhow::{Result, anyhow};

// Helper function for serde default
fn default_device() -> Device {
    Device::Cpu
}

// Define MoshiModelPaths struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoshiModelPaths {
    pub model_dir: String,
    pub tokenizer_path: String,
    pub asr_lm_path: String, // Add path for ASR LM
}

// Define CsmModelPaths struct with necessary fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsmModelPaths {
    pub model_dir: String,
    // Add other relevant paths for CSM models here
}

// Define ModelTypeConfig enum to handle different model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelTypeConfig {
    Csm(CsmModelPaths),
    Moshi(MoshiModelPaths),
}

// Based on Python ModelArgs and llama3_2_1B function
#[derive(Debug, Deserialize, Clone, Serialize)]
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

    // --- NEW: Pre-trained LLM Backbone Loading Params ---
    #[serde(default)] // Defaults to false
    pub load_pretrained_llm_backbone: bool,

    #[serde(default)]
    pub pretrained_llm_backbone_path: Option<PathBuf>,

    #[serde(default)] // e.g., "llama", "mistral" - for potential weight name mapping
    pub pretrained_llm_backbone_type: Option<String>,

    // Configuration for LLM backbone loading
    /// Flag to indicate whether to load a pretrained LLM as the backbone
    #[serde(default)]
    pub llm_backbone_type: Option<String>,
    
    /// Path to safetensors file for pretrained LLM
    #[serde(default)]
    pub llm_safetensors_path: Option<String>,
    
    /// Embedding dimension for LLM context
    #[serde(default)]
    pub llm_embedding_dim: Option<i64>,

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
    #[serde(default)]
    pub vocab_size: i64, // Ensure this is loaded if needed
    
    #[serde(default)]
    pub num_codebooks: i64, // Ensure this is loaded if needed (used in RustCsmModel::synthesize_streaming_internal)

    /// Directory containing model files
    pub model_dir: Option<String>,
    
    /// Type of model to load ("csm" or "moshi")
    #[serde(default = "default_model_type")]
    pub model_type: Option<String>,
    
    /// Device type to use ("cpu", "cuda", or "mps")
    #[serde(default = "default_device_type")]
    pub device_type: Option<String>,
    
    /// Sample rate for audio processing
    #[serde(default = "default_sample_rate")]
    pub sample_rate: Option<u32>,
    
    /// Moshi model path (for Moshi model type)
    pub moshi_model_path: Option<String>,
    
    /// Tokenizer path (for Moshi model type)
    pub tokenizer_path: Option<String>,
    
    /// Mimi model path (for Moshi model type)
    pub mimi_model_path: Option<String>,
    
    /// Voice preset to use
    pub voice_preset: Option<String>,
    
    /// Streaming chunk size
    #[serde(default = "default_chunk_size")]
    pub chunk_size: Option<usize>,
    
    /// Experimental features
    #[serde(default)]
    pub experimental: bool,

    /// Whether to enable prosody generation features
    #[serde(default)]
    pub enable_prosody: Option<bool>,

    // Configuration for model paths
    /// Moshi model paths (for Moshi model type)
    #[serde(default)]
    pub moshi_model_paths: Option<MoshiModelPaths>,

    // Configuration for the Llama model used for text processing
    pub text_encoder: TextEncoderConfig,
    // Configuration for the acoustic model (SoundStorm-like)
    pub acoustic_model: AcousticModelConfig, 
    // Configuration for the RVQ (Residual Vector Quantization) used by the vocoder
    pub rvq_config: RVQConfig, // Use the specific RVQConfig type
}

// ADDED Placeholder structs with necessary derives
#[derive(Debug, Default, Deserialize, Clone, Serialize)]
pub struct TextEncoderConfig {
    // Add fields as needed based on actual usage or definition
}

#[derive(Debug, Default, Deserialize, Clone, Serialize)]
pub struct AcousticModelConfig {
    // Add fields as needed based on actual usage or definition
}

// Helper functions for serde defaults
fn default_temperature() -> f64 { 0.7 }
fn default_top_k() -> i64 { 50 }

fn default_model_type() -> Option<String> {
    Some("csm".to_string())
}

fn default_device_type() -> Option<String> {
    Some("cpu".to_string())
}

fn default_sample_rate() -> Option<u32> {
    Some(24000)
}

fn default_chunk_size() -> Option<usize> {
    Some(1024)
}

impl Default for CsmModelConfig {
    fn default() -> Self {
        Self {
            model_dir: None,
            model_type: default_model_type(),
            device_type: default_device_type(),
            sample_rate: default_sample_rate(),
            moshi_model_path: None,
            tokenizer_path: None,
            mimi_model_path: None,
            voice_preset: None,
            chunk_size: default_chunk_size(),
            experimental: false,
            enable_prosody: Some(false),
            backbone_flavor: String::new(),
            decoder_flavor: String::new(),
            text_vocab_size: 0,
            semantic_audio_vocab_size: 0,
            acoustic_vocab_size: 0,
            num_acoustic_codebooks: 0,
            backbone_num_layers: 0,
            backbone_num_heads: 0,
            backbone_num_kv_heads: 0,
            backbone_embed_dim: 0,
            backbone_intermediate_dim: 0,
            backbone_norm_eps: 0.0,
            backbone_rope_base: 0.0,
            decoder_num_layers: 0,
            decoder_num_heads: 0,
            decoder_num_kv_heads: 0,
            decoder_embed_dim: 0,
            decoder_intermediate_dim: 0,
            decoder_norm_eps: 0.0,
            decoder_rope_base: 0.0,
            max_seq_len: 0,
            attn_dropout: 0.0,
            load_pretrained_llm_backbone: false,
            pretrained_llm_backbone_path: None,
            pretrained_llm_backbone_type: None,
            llm_backbone_type: None,
            llm_safetensors_path: None,
            llm_embedding_dim: None,
            temperature: 0.7,
            top_k: 50,
            device: Device::Cpu,
            vocab_size: 0,
            num_codebooks: 0,
            moshi_model_paths: None,
            text_encoder: TextEncoderConfig::default(),
            acoustic_model: AcousticModelConfig::default(),
            rvq_config: RVQConfig::default(),
        }
    }
}

impl CsmModelConfig {
    // Function to load from config.json or .toml
    pub fn from_file(path: &Path) -> Result<Self> {
        let config_str = std::fs::read_to_string(path)
            .map_err(|e| anyhow!("Failed to read config file: {}", e))?;
        
        // Try parsing as TOML first
        let result = toml::from_str(&config_str);
        if let Ok(config) = result {
            return Ok(config);
        }
        
        // If TOML parsing fails, try JSON as fallback
        let config: Self = serde_json::from_str(&config_str)
            .map_err(|e| anyhow!("Failed to parse config file: {}", e))?;
        
        Ok(config)
    }

    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let config_str = if path.extension().map_or("", |e| e.to_str().unwrap_or("")) == "toml" {
            toml::to_string_pretty(self)
                .map_err(|e| anyhow!("Failed to serialize config as TOML: {}", e))?
        } else {
            serde_json::to_string_pretty(self)
                .map_err(|e| anyhow!("Failed to serialize config as JSON: {}", e))?
        };
        
        std::fs::write(path, config_str)
            .map_err(|e| anyhow!("Failed to write config file: {}", e))?;
        
        Ok(())
    }

    // Add method to retrieve Moshi model paths
    pub fn moshi_model_paths(&self) -> Option<MoshiModelPaths> {
        // First check if we have explicit moshi_model_paths structure
        if let Some(paths) = &self.moshi_model_paths {
            return Some(paths.clone());
        }
        
        // Otherwise, try to construct from individual fields
        if self.model_type.as_deref() == Some("moshi") {
            if let (Some(model_path), Some(tokenizer_path)) = (&self.moshi_model_path, &self.tokenizer_path) {
                return Some(MoshiModelPaths {
                    model_dir: model_path.clone(),
                    tokenizer_path: tokenizer_path.clone(),
                    asr_lm_path: String::new(), // Placeholder for ASR LM path
                });
            }
        }
        
        None
    }

    // Method to retrieve CSM model paths
    pub fn csm_model_paths(&self) -> Option<CsmModelPaths> {
        if self.model_type.as_deref() == Some("csm") {
            Some(CsmModelPaths {
                model_dir: self.model_dir.clone().unwrap_or_default(),
                // Add other necessary paths for CSM model if they exist in config
            })
        } else {
            None
        }
    }
}

pub fn find_config_file() -> Option<PathBuf> {
    // Look in common locations
    let paths = vec![
        "config.toml",
        "config.json",
        "models/config.toml",
        "models/config.json",
        "../models/config.toml",
        "../models/config.json",
    ];
    
    for path in paths {
        let path_buf = PathBuf::from(path);
        if path_buf.exists() {
            return Some(path_buf);
        }
    }
    
    None
}

// Placeholder for RVQ config if needed later
// ADDED necessary derives
#[derive(Clone, Debug, Default, Deserialize, Serialize)] 
pub struct RVQConfig {
   // ... RVQ parameters ...
   // Placeholder fields to satisfy Default/Deserialize/Serialize
    #[serde(default)]
    pub num_codebooks: usize,
    #[serde(default)]
    pub vector_dim: usize,
    #[serde(default)]
    pub codebook_size: usize,
    #[serde(default)]
    pub normalize: bool,
    #[serde(default)]
    pub learning_rate: f64,
    // Add the device field if it should be part of the config
    // #[serde(skip)] // Or skip serialization if managed elsewhere
    // pub device: Device,
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
