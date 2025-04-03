// src/audio/codec.rs
// Mimi Neural Audio Codec implementation (Encoder part)
// Architecture adapted from https://github.com/kyutai-labs/moshi (SEANet + Transformer)

// Import the full nn module
use tch::{nn, Device, Tensor, Kind, TchError};
use anyhow::{Result};
use std::path::Path;
use crate::models::ModelError; 
use tracing::{info, trace, span, Level, warn, error};
use std::fs::File;
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::HashMap;

// Use Module and ModuleT traits directly if needed
use tch::nn::{Module, LayerNorm, LayerNormConfig, Linear, LinearConfig, Conv1D};

// --- Configuration --- 

// Based on config.json and typical audio model params
#[derive(Debug, Clone)]
pub struct MimiEncoderConfig {
    pub input_channels: i64,    // Usually 1 for mono audio
    pub dimension: i64,         // Target feature dimension (output of encoder) - Should match transformer hidden_size
    pub hidden_size: i64,       // Transformer hidden size
    pub hidden_channels: i64,   // Base number of channels in initial conv layers (used by downsampler)
    pub activation: String,     // Activation function type ("relu", "gelu", etc.) for MLP
    pub kernel_size: i64,       // Initial conv kernel size (unused if input goes direct to downsample)
    pub causal: bool,           // Whether convolutions/attention are causal
    pub compress: i64,          // Downsampling factor for the initial conv
    pub num_hidden_layers: usize, // Transformer layers
    pub num_attention_heads: i64, // Transformer heads
    pub head_dim: i64,          // Dimension per head
    pub intermediate_size: i64, // MLP intermediate size
    pub norm_eps: f64,          // LayerNorm epsilon
    pub rope_theta: f64,        // RoPE theta
}

// Example Default Configuration (Values from models/mimi/config.json)
impl Default for MimiEncoderConfig {
    fn default() -> Self {
        Self {
            input_channels: 1, // From audio_channels
            dimension: 512, // Match hidden_size
            hidden_size: 512, // From hidden_size
            hidden_channels: 512, // Match hidden_size for downsampler input/output
            activation: "gelu".to_string(), // From hidden_act
            kernel_size: 7, // From kernel_size (potentially unused)
            causal: true, // From use_causal_conv (Assuming encoder is causal too)
            compress: 2, // From compress
            num_hidden_layers: 8, // From num_hidden_layers
            num_attention_heads: 8, // From num_attention_heads
            head_dim: 64, // From head_dim
            intermediate_size: 2048, // From intermediate_size
            norm_eps: 1e-5, // From norm_eps
            rope_theta: 10000.0, // From rope_theta
        }
    }
}

// --- Rotary Embedding (Copied from vocoder.rs) ---
#[derive(Debug)]
struct RotaryEmbedding {
    dim: i64,
    base: f64,
    inv_freq: Tensor,
}

impl Clone for RotaryEmbedding {
    fn clone(&self) -> Self {
        Self {
            dim: self.dim,
            base: self.base,
            inv_freq: self.inv_freq.shallow_clone(),
        }
    }
}

impl RotaryEmbedding {
    // Note: max_seq_len is not used in this implementation, remove if not needed elsewhere
    fn new(dim: i64, _max_seq_len: i64, base: f64, device: Device) -> Result<Self, TchError> {
        let theta: Vec<_> = (0..dim / 2)
            .map(|i| 1.0 / base.powf(2.0 * i as f64 / dim as f64))
            .collect();
        let theta = Tensor::from_slice(&theta).to_kind(Kind::Float).to_device(device);
        Ok(Self {
            dim,
            base,
            inv_freq: theta,
        })
    }

    // Apply RoPE to query and key tensors
    fn forward(&self, xq: &Tensor, xk: &Tensor, start_pos: usize) -> Result<(Tensor, Tensor), TchError> {
        let (_b_sz_q, _n_head_q, seq_len_q, _head_dim_q) = xq.size4()?;
        let seq_len = seq_len_q;
        let device = self.inv_freq.device();

        let t = Tensor::arange_start(start_pos as i64, (start_pos as i64) + seq_len, (Kind::Float, device));
        let freqs = t.outer(&self.inv_freq);
        let freqs_cis = Tensor::polar(&Tensor::ones_like(&freqs), &freqs); 

        let xq_r = xq.reshape(&[-1, seq_len, self.dim / 2, 2]);
        let xk_r = xk.reshape(&[-1, seq_len, self.dim / 2, 2]);
        let xq_c = xq_r.view_as_complex();
        let xk_c = xk_r.view_as_complex();

        let freqs_cis = freqs_cis.unsqueeze(0); 

        let xq_out_c = xq_c.f_mul(&freqs_cis)?;
        let xk_out_c = xk_c.f_mul(&freqs_cis)?;

        let xq_out_r = xq_out_c.view_as_real();
        let xk_out_r = xk_out_c.view_as_real();

        let xq_out = xq_out_r.reshape_as(xq);
        let xk_out = xk_out_r.reshape_as(xk);

        Ok((xq_out, xk_out))
    }
}

// --- Attention Block (Simplified from vocoder.rs) ---
#[derive(Debug)]
struct MimiAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    n_head: i64,
    head_dim: i64,
    rotary_emb: RotaryEmbedding, // Add RoPE
}

impl MimiAttention {
    fn new(vs: &nn::Path, config: &MimiEncoderConfig, device: Device) -> Result<Self, ModelError> { 
        let embed_dim = config.hidden_size;
        let head_dim = config.head_dim;
        let n_head = config.num_attention_heads;
        let linear_cfg = LinearConfig { bias: false, ..Default::default() };
        let q_proj = nn::linear(vs / "q_proj", embed_dim, n_head * head_dim, linear_cfg);
        let k_proj = nn::linear(vs / "k_proj", embed_dim, n_head * head_dim, linear_cfg);
        let v_proj = nn::linear(vs / "v_proj", embed_dim, n_head * head_dim, linear_cfg);
        let o_proj = nn::linear(vs / "o_proj", n_head * head_dim, embed_dim, linear_cfg);
        // Assuming a large max_seq_len for RoPE initialization
        let rotary_emb = RotaryEmbedding::new(head_dim, 8192, config.rope_theta, device)
            .map_err(ModelError::Tch)?;
        Ok(Self { q_proj, k_proj, v_proj, o_proj, n_head, head_dim, rotary_emb })
    }

    // Forward with start_pos for potential streaming later
    fn forward(&self, xs: &Tensor, start_pos: usize) -> Result<Tensor, ModelError> {
        let (b_sz, seq_len, _hidden_dim) = xs.size3()
            .map_err(|e| ModelError::TensorError(format!("Attention input must be 3D: {}", e)))?;

        let q = self.q_proj.forward(xs);
        let k = self.k_proj.forward(xs);
        let v = self.v_proj.forward(xs);

        let q = q.contiguous().view((b_sz, seq_len, self.n_head, self.head_dim)).transpose(1, 2);
        let k = k.contiguous().view((b_sz, seq_len, self.n_head, self.head_dim)).transpose(1, 2);
        let v = v.contiguous().view((b_sz, seq_len, self.n_head, self.head_dim)).transpose(1, 2);

        let (q, k) = self.rotary_emb.forward(&q, &k, start_pos)?;

        // Scaled Dot-Product Attention
        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = q.matmul(&k.transpose(-2, -1)) / scale;

        // TODO: Add causal mask if config.causal is true
        // let mask = if config.causal { ... } else { None };

        let attn_probs = attn_weights.softmax(-1, Kind::Float);
        let output = attn_probs.matmul(&v);

        let output = output.transpose(1, 2)
                         .contiguous()
                         .view((b_sz, seq_len, -1)); 
        Ok(self.o_proj.forward(&output))
    }
}

// --- MLP Block (Simplified from vocoder.rs) ---
#[derive(Debug)]
struct MimiMLP {
    fc1: Linear,
    fc2: Linear,
    activation: fn(&Tensor) -> Tensor,
}

impl MimiMLP {
     fn new(vs: &nn::Path, config: &MimiEncoderConfig) -> Result<Self, ModelError> {
         let embed_dim = config.hidden_size;
         let hidden_dim = config.intermediate_size;
         let linear_cfg = LinearConfig { bias: false, ..Default::default() };
         let fc1 = nn::linear(vs / "fc1", embed_dim, hidden_dim, linear_cfg);
         let fc2 = nn::linear(vs / "fc2", hidden_dim, embed_dim, linear_cfg);
         let activation = match config.activation.as_str() {
            "gelu" => |t: &Tensor| t.gelu("none"),
            _ => { 
                warn!("Unsupported MLP activation '{}', falling back to ReLU.", config.activation);
                Tensor::relu 
            },
         };
         Ok(Self { fc1, fc2, activation })
     }
}

impl Module for MimiMLP {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let hidden = self.fc1.forward(xs);
        self.fc2.forward(&(self.activation)(&hidden))
    }
}

// --- Transformer Block (Simplified from vocoder.rs) ---
#[derive(Debug)]
struct MimiTransformerBlock {
    input_layernorm: LayerNorm,
    self_attn: MimiAttention,
    post_attention_layernorm: LayerNorm,
    mlp: MimiMLP,
}

impl MimiTransformerBlock {
     fn new(vs: &nn::Path, config: &MimiEncoderConfig, device: Device) -> Result<Self, ModelError> {
          let embed_dim = config.hidden_size;
          let ln_eps = config.norm_eps;
          let ln_cfg = LayerNormConfig { eps: ln_eps, ..Default::default() };
          let input_layernorm = nn::layer_norm(vs / "input_layernorm", vec![embed_dim], ln_cfg);
          let self_attn = MimiAttention::new(&(vs / "self_attn"), config, device)?;
          let post_attention_layernorm = nn::layer_norm(vs / "post_attention_layernorm", vec![embed_dim], ln_cfg);
          let mlp = MimiMLP::new(&(vs / "mlp"), config)?;
          Ok(Self { input_layernorm, self_attn, post_attention_layernorm, mlp })
     }

     // Forward with start_pos for potential streaming
     fn forward(&self, xs: &Tensor, start_pos: usize) -> Result<Tensor, ModelError> {
          let residual = xs;
          let hidden_states = self.input_layernorm.forward(xs);
          let attn_output = self.self_attn.forward(&hidden_states, start_pos)?;
          let hidden_states = attn_output + residual;

          let residual = &hidden_states;
          let hidden_states_norm = self.post_attention_layernorm.forward(&hidden_states);
          let mlp_output = self.mlp.forward(&hidden_states_norm);
          Ok(mlp_output + residual)
     }
}

// --- Mimi Encoder (Refactored) --- 

#[derive(Debug)]
pub struct MimiEncoder {
    pub config: MimiEncoderConfig,
    downsample: Conv1D,
    layers: Vec<MimiTransformerBlock>, // Changed from EncoderLayer
    final_norm: LayerNorm,             // Changed from final_conv
    vs: nn::VarStore,
}

// Helper function for manual weight loading (updated mapping)
fn load_encoder_weights_manual(
    vs: &mut nn::VarStore,
    safetensors_path: &Path,
) -> Result<(), ModelError> {
    info!(
        "Loading Mimi ENCODER weights MANUALLY from: \"{}\"",
        safetensors_path.display()
    );

    let file = File::open(safetensors_path)
        .map_err(|e| ModelError::Io(e))?; 
    let buffer = unsafe { Mmap::map(&file).map_err(|e| ModelError::Io(e))? };
    let safe_tensors = SafeTensors::deserialize(&buffer)
        .map_err(|e| ModelError::Safetensor(e))?;
    
    info!(
        "Successfully deserialized SafeTensors file with {} tensors.",
        safe_tensors.names().len()
    );

    // Load tensors from file into a map
    let mut st_tensors = HashMap::new();
    for tensor_name in safe_tensors.names() {
        // Only load weights relevant to the encoder structure
        if tensor_name.starts_with("downsample.") || tensor_name.starts_with("encoder_transformer.") {
            match safe_tensors.tensor(tensor_name) {
                Ok(tensor_view) => {
                    let shape: Vec<i64> = tensor_view.shape().iter().map(|&d| d as i64).collect();
                    let kind = match tensor_view.dtype() {
                        safetensors::Dtype::F32 => Kind::Float,
                        // Add other dtypes as needed
                        _ => {
                            warn!("Unsupported dtype for tensor {}: {:?}", tensor_name, tensor_view.dtype());
                            continue;
                        }
                    };
                    let data = tensor_view.data();
                    let tch_tensor = Tensor::from_data_size(data, &shape, kind)
                        .to_device(vs.device());
                    st_tensors.insert(tensor_name.to_string(), tch_tensor);
                },
                Err(e) => warn!("Failed to get tensor '{}' from SafeTensors: {}", tensor_name, e),
            }
        }
    }
    info!("[Manual Encoder Load] Loaded {} relevant tensors into temporary map.", st_tensors.len());

    let mut copied_count = 0;
    let mut missing_key_count = 0;
    let mut shape_mismatch_count = 0;
    let vs_vars = vs.variables();
    let total_vs_vars = vs_vars.len();

    info!(
        "[Manual Encoder Load] Attempting to copy weights into {} VarStore variables...",
        total_vs_vars
    );

    for (vs_name, mut vs_tensor) in vs_vars {
        // --- UPDATED MAPPING LOGIC --- 
        // Convert VarStore path (e.g., "encoder_transformer/layers/0/input_layernorm/weight")
        // to SafeTensors key (e.g., "encoder_transformer.layers.0.input_layernorm.weight")
        let st_key = vs_name.replace('/', ".");
        // ----------------------------------

        match st_tensors.get(&st_key) {
            Some(st_tensor) => {
                if vs_tensor.size() == st_tensor.size() {
                    tch::no_grad(|| {
                        vs_tensor.copy_(st_tensor);
                    });
                    copied_count += 1;
                } else {
                    warn!(
                        "[Manual Encoder Load] Shape mismatch for '{}': VarStore={:?}, File={:?}. Skipping.",
                        vs_name, vs_tensor.size(), st_tensor.size()
                    );
                    shape_mismatch_count += 1;
                }
            }
            None => {
                warn!(
                    "[Manual Encoder Load] Variable '{}' (mapped to '{}') in VS, but key not found in file.",
                    vs_name, st_key
                );
                missing_key_count += 1;
            }
        }
    }

    let missing_in_vs = st_tensors.len() as i32 - copied_count - shape_mismatch_count;
    info!(
        "[Manual Encoder Load] Finished. Total VS Vars: {}, Copied: {}, Shape Mismatch: {}, Missing in File: {}, Missing in VS: {}",
        total_vs_vars, copied_count, shape_mismatch_count, missing_key_count, missing_in_vs.max(0)
    );
    
    if copied_count == 0 && total_vs_vars > 0 {
        warn!("[Manual Encoder Load] No weights were copied. Check paths and file content.");
        return Err(ModelError::LoadError("No weights were loaded for MimiEncoder.".to_string()));
    } else if shape_mismatch_count > 0 || missing_key_count > 0 {
         warn!("[Manual Encoder Load] Some weights might be missing or mismatched. Review warnings.");
    }

    Ok(())
}

impl MimiEncoder {
    // Take VarStore by value instead of reference
    pub fn new(device: Device) -> Result<Self, ModelError> {
        let vs = nn::VarStore::new(device);
        let config = MimiEncoderConfig::default(); 
        let vs_path = vs.root(); 
        
        // Initial Downsampling Convolution
        // Path should match "downsample.conv" from safetensors file
        let downsample_path = vs_path.clone() / "downsample" / "conv";
        let downsample_kernel_size = 4; // From previous attempt, seems reasonable
        let ds_padding = (downsample_kernel_size - config.compress) / 2; // Use config.compress as stride
        let ds_cfg = nn::ConvConfig {
            padding: ds_padding, 
            stride: config.compress, 
            bias: false, // Match safetensors key (no bias for downsample.conv)
            ..Default::default()
        };
        // Input channels = config.input_channels (1), Output channels = config.hidden_size (512)
        let downsample = nn::conv1d(&downsample_path, config.input_channels, config.hidden_size, downsample_kernel_size, ds_cfg);

        // Encoder Transformer Layers
        let transformer_path = vs_path.clone() / "encoder_transformer";
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer_vs = transformer_path.clone() / "layers" / i;
            let block = MimiTransformerBlock::new(&layer_vs, &config, device)?;
            layers.push(block);
        }

        // Final LayerNorm
        // Path should match "encoder_transformer.norm"
        let norm_path = transformer_path.clone() / "norm";
        let ln_cfg = LayerNormConfig { eps: config.norm_eps, ..Default::default() };
        let final_norm = nn::layer_norm(&norm_path, vec![config.hidden_size], ln_cfg);

        Ok(Self {
            config,
            downsample,
            layers,
            final_norm,
            vs, 
        })
    }

    pub fn load_weights(&mut self, path: &Path) -> Result<(), ModelError> {
        self.vs.unfreeze();
        info!("Attempting to load MimiEncoder weights MANUALLY from: {:?}", path);
        // Call the updated manual loading function
        let result = load_encoder_weights_manual(&mut self.vs, path);
        self.vs.freeze();
        match &result {
            Ok(_) => info!("Manual MimiEncoder weight loading completed successfully."),
            Err(e) => error!("Manual MimiEncoder weight loading failed: {}", e),
        }
        result 
    }

    // The forward pass takes raw audio tensor and outputs feature tensor
    pub fn forward(&self, audio_input: &Tensor) -> Result<Tensor, ModelError> {
        let _span = span!(Level::TRACE, "MimiEncoder::forward").entered();
        trace!("Input shape: {:?}", audio_input.size());

        // Declare input tensor variable
        let input: Tensor;

        // Process input shape, handle errors, and assign to input variable
        if audio_input.dim() == 2 {
            input = audio_input.unsqueeze(0) // Add batch dim -> [1, T]
                             .unsqueeze(1); // Add channel dim -> [1, 1, T]
        } else if audio_input.dim() == 1 {
            input = audio_input.unsqueeze(0).unsqueeze(0); // Add batch & channel -> [1, 1, T]
        } else if audio_input.dim() == 3 && audio_input.size()[1] == 1 {
            // Already [B, 1, T]
            input = audio_input.shallow_clone();
        } else if audio_input.dim() == 3 && audio_input.size()[1] != 1 {
            // Return error directly for wrong channel count
            return Err(ModelError::InvalidInput(format!("Expected mono audio (1 channel), got {} channels. Shape: {:?}", audio_input.size()[1], audio_input.size())));
        } else {
            // Return error directly for other invalid dimensions
            return Err(ModelError::InvalidInput(format!("Expected 1D, 2D or 3D (mono) audio tensor, got {}D. Shape: {:?}", audio_input.dim(), audio_input.size())));
        }
        // Input variable is now guaranteed to be assigned if no error was returned
        trace!("Processed Input shape [B, C, T]: {:?}", input.size());

        // Initial Downsampling
        // Input: [B, 1, T], Output: [B, H, T_down] (H=hidden_size)
        let mut hidden = self.downsample.forward(&input);
        trace!("After downsample shape [B, C, T]: {:?}", hidden.size());

        // Transpose for Transformer: [B, C, T_down] -> [B, T_down, C]
        hidden = hidden.transpose(1, 2);
        trace!("Transposed for transformer shape [B, T, C]: {:?}", hidden.size());

        // Transformer Layers
        let current_pos = 0; // For RoPE, assuming non-streaming for now
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, current_pos)?;
            trace!("After transformer layer {} shape [B, T, C]: {:?}", i, hidden.size());
             // If streaming, update current_pos += seq_len;
        }

        // Final Norm
        // Input: [B, T_down, C], Output: [B, T_down, C]
        hidden = self.final_norm.forward(&hidden);
        trace!("After final_norm shape [B, T, C]: {:?}", hidden.size());

        // Output feature_tensor shape for RVQ: [batch, sequence_length, feature_dim]
        Ok(hidden) 
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Tensor, Kind};

    #[test]
    fn test_mimi_encoder_forward_shape() -> Result<(), ModelError> {
        let device = Device::Cpu;
        let encoder = MimiEncoder::new(device)?; // Uses default config
        
        // Example input: Batch=1, Channels=1, Length=16000 samples
        let input_tensor = Tensor::randn(&[1, 1, 16000], (Kind::Float, device));
        
        let features = encoder.forward(&input_tensor)?;

        // Get expected output dimensions from config
        let expected_dim = encoder.config.hidden_size; // Output dim is transformer hidden size
        // Calculate expected sequence length after downsampling
        let expected_len = (16000.0 / encoder.config.compress as f64).ceil() as i64;

        // Expected shape for RVQ: [Batch, SeqLen, FeatureDim]
        let expected_shape = &[1, expected_len, expected_dim]; 
        let actual_shape = features.size();

        assert_eq!(actual_shape, expected_shape, 
                   "Unexpected output shape. Expected {:?}, Got {:?}", 
                   expected_shape, actual_shape);
        
        Ok(())
    }
}
