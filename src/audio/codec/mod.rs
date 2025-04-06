//! Audio codec module
//!
//! Contains the audio encoding/decoding functionality and compatibility layers.

// Remove the path attribute and module definition, code is now inline
// #[path = "../codec.rs"]
// mod codec_impl;

// pub use self::codec_impl::*;

// --- Start of content moved from src/audio/codec.rs --- 

// src/audio/codec.rs
// Mimi Neural Audio Codec implementation (Encoder part)
// Architecture adapted from https://github.com/kyutai-labs/moshi (SEANet + Transformer)

// Import the full nn module
use tch::{nn, Device, Tensor, Kind, TchError};
use anyhow::Result;
use std::path::Path;
use crate::models::ModelError; 
use tracing::{info, trace, span, Level, warn};
use std::fs::File;
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::HashMap;

// Use Module and ModuleT traits directly if needed
use tch::nn::{Module, LayerNorm, LayerNormConfig, Linear, LinearConfig};

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
    // --- New Fields for Frame Rate --- 
    pub frame_length_ms: f32,   // Frame length in milliseconds
    pub hop_length_ms: f32,     // Hop length in milliseconds
    pub sample_rate: f64,        // Sample rate of the audio
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
            // Default frame/hop lengths (e.g., 25ms frame, 10ms hop)
            frame_length_ms: 25.0,
            hop_length_ms: 10.0,
            sample_rate: 24000.0, // Assuming a default sample rate
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

/// State to be maintained across streaming calls for MimiEncoder
#[derive(Debug, Clone, Default)]
pub struct MimiTransformerLayerState {
    // Assuming fields like key, value for caching might be here later
    // Placeholder for now if it wasn't fully defined
    _placeholder: Option<()>, 
}

#[derive(Debug, Clone, Default)]
pub struct MimiEncoderState {
    pub start_pos: usize, 
    pub audio_buffer: Vec<f32>, 
    pub layer_states: Vec<MimiTransformerLayerState>, // Now this type is in scope
}

#[derive(Debug)]
pub struct MimiEncoder {
    pub config: MimiEncoderConfig,
    input_conv: tch::nn::Conv1D,
    downsample: tch::nn::Conv1D,
    layers: Vec<MimiTransformerBlock>,
    final_norm: nn::LayerNorm,
    vs: nn::VarStore,
    device: Device,
}

impl MimiEncoder {
    pub fn new(config: MimiEncoderConfig, device: Device) -> Result<Self> {
        let vs = nn::VarStore::new(device);
        let vs_path = vs.root();

        // ADDED: Initial Input Convolution (1 -> hidden_size)
        let input_conv_path = vs_path.clone() / "input_conv"; 
        let input_kernel_size = 7; 
        let input_padding = (input_kernel_size - 1) / 2; 
        let input_conv_cfg = nn::ConvConfig { padding: input_padding, stride: 1, bias: true, ..Default::default() };
        let input_conv = nn::conv1d(&input_conv_path, config.input_channels, config.hidden_size, input_kernel_size, input_conv_cfg);

        // Downsample Layer
        let downsample_path = vs_path.clone() / "downsample" / "conv";
        let downsample_kernel_size = 4; 
        let ds_padding = (downsample_kernel_size - config.compress) / 2;
        let ds_cfg = nn::ConvConfig { padding: ds_padding, stride: config.compress, bias: false, ..Default::default() };
        let downsample = nn::conv1d(&downsample_path, config.hidden_size, config.hidden_size, downsample_kernel_size, ds_cfg);

        // Encoder Transformer Layers
        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let layer_path = vs_path.clone() / "encoder_transformer" / i.to_string();
            layers.push(MimiTransformerBlock::new(&layer_path, &config, device)?); // Pass device
        }

        // Final Layer Norm
        let norm_path = vs_path.clone() / "encoder_transformer" / "norm";
        let norm_cfg = nn::LayerNormConfig { eps: config.norm_eps, ..Default::default() };
        let final_norm = nn::layer_norm(&norm_path, vec![config.hidden_size as i64], norm_cfg);

        Ok(Self {
            config,
            input_conv,
            downsample,
            layers,     
            final_norm, 
            vs,
            device,
        })
    }

    // --- Weight Loading Logic --- 
    pub fn load_weights(&mut self, path: &Path) -> Result<()> {
        self.vs = nn::VarStore::new(self.device); // Recreate VarStore to ensure clean state
        let vs_path = self.vs.root();
        let config = self.config.clone(); 

        // Reload structure
        let downsample_path = vs_path.clone() / "downsample" / "conv";
        let downsample_kernel_size = 4; 
        let ds_padding = (downsample_kernel_size - config.compress) / 2;
        let ds_cfg = nn::ConvConfig { padding: ds_padding, stride: config.compress, bias: false, ..Default::default() };
        self.downsample = nn::conv1d(&downsample_path, config.hidden_size, config.hidden_size, downsample_kernel_size, ds_cfg);

        let input_conv_path = vs_path.clone() / "input_conv"; 
        let input_kernel_size = 7; 
        let input_padding = (input_kernel_size - 1) / 2; 
        let input_conv_cfg = nn::ConvConfig { padding: input_padding, stride: 1, bias: true, ..Default::default() };
        self.input_conv = nn::conv1d(&input_conv_path, config.input_channels, config.hidden_size, input_kernel_size, input_conv_cfg);

        self.layers.clear();
        for i in 0..config.num_hidden_layers {
            let layer_path = vs_path.clone() / "encoder_transformer" / i.to_string();
            self.layers.push(MimiTransformerBlock::new(&layer_path, &config, self.device)?);
        }

        let norm_path = vs_path.clone() / "encoder_transformer" / "norm";
        let norm_cfg = nn::LayerNormConfig { eps: config.norm_eps, ..Default::default() };
        self.final_norm = nn::layer_norm(&norm_path, vec![config.hidden_size as i64], norm_cfg);

        // Call the standalone loading function
        load_encoder_weights_manual(&mut self.vs, path, "MimiEncoder")
    }

    pub fn forward(&self, input_tensor: &Tensor, state: Option<MimiEncoderState>) 
        -> Result<(Tensor, MimiEncoderState)> 
    {
        span!(Level::DEBUG, "MimiEncoder::forward").in_scope(|| {
            trace!("Input audio shape: {:?}", input_tensor.size());
            let mut current_state = state.unwrap_or_default();
            trace!("Input state: start_pos={}, buffer_len={}", current_state.start_pos, current_state.audio_buffer.len());

            let mut combined_audio = current_state.audio_buffer.clone();
            let squeezed_xs = input_tensor.squeeze(); 
            let current_chunk_vec: Vec<f32> = squeezed_xs.try_into()
                .map_err(|e| anyhow::anyhow!("Failed to convert input tensor chunk to Vec<f32>: {}", e))?;
            combined_audio.extend_from_slice(&current_chunk_vec);
            trace!("Combined audio buffer length: {}", combined_audio.len());

            let frame_length_samples = (self.config.frame_length_ms / 1000.0 * self.config.sample_rate as f32) as usize;
            let hop_length_samples = (self.config.hop_length_ms / 1000.0 * self.config.sample_rate as f32) as usize;

            if combined_audio.len() < frame_length_samples {
                trace!("Combined audio length ({}) less than frame length ({}), buffering.", combined_audio.len(), frame_length_samples);
                current_state.audio_buffer = combined_audio; // Keep buffered audio
                return Ok((Tensor::zeros(&[input_tensor.size()[0], self.config.hidden_size as i64, 0], (input_tensor.kind(), input_tensor.device())), current_state));
            }

            let num_frames = (combined_audio.len() - frame_length_samples) / hop_length_samples + 1;
            let processable_len = (num_frames - 1) * hop_length_samples + frame_length_samples;
            trace!("Processing {} frames, total length: {}", num_frames, processable_len);

            let processable_tensor_vec = combined_audio[..processable_len].to_vec();
            let processable_tensor = Tensor::from_slice(&processable_tensor_vec).to(self.device);

            // Update buffer for next call
            let remaining_buffer = combined_audio[num_frames * hop_length_samples..].to_vec();
            let mut updated_state = current_state.clone(); // Clone state to update it
            updated_state.audio_buffer = remaining_buffer;
            trace!("Remaining buffer length: {}", updated_state.audio_buffer.len());

            // Ensure input is 3D: (batch, channels, sequence_length)
            let input = if processable_tensor.dim() == 1 {
                processable_tensor.unsqueeze(0).unsqueeze(0) // [T] -> [1, 1, T]
            } else if processable_tensor.dim() == 2 {
                processable_tensor.unsqueeze(1) // [B, T] -> [B, 1, T]
            } else {
                processable_tensor.shallow_clone() // Assume [B, C, T]
            };
            // Use anyhow::anyhow! for the error message
            let (b, c, t) = input.size3().map_err(|e| anyhow::anyhow!("Input must be 3D: {}", e))?;
            trace!("Input tensor shape: [{}, {}, {}]", b, c, t);

            if t < frame_length_samples as i64 {
                 warn!("Input tensor length ({}) is shorter than frame length ({}), cannot create frames. Returning empty features and updated state.", t, frame_length_samples);
                 return Ok((Tensor::zeros(&[b, self.config.hidden_size as i64, 0], (input.kind(), input.device())), updated_state));
            }

            let frames = input.unfold(2, frame_length_samples as i64, hop_length_samples as i64);
            let (b, c, nf, frame_len) = frames.size4().map_err(|e| anyhow::anyhow!("Unfold failed: {}", e))?;
            trace!("Frames shape after unfold: [{}, {}, {}, {}]", b, c, nf, frame_len);

            let conv_input = frames.permute(&[0, 2, 1, 3])
                                  .contiguous()
                                  .view([b * nf, c, frame_len]);
            trace!("Shape before input_conv: {:?}", conv_input.size());

            // --- Apply Layers --- 
            // 1. Initial input convolution (1 -> hidden_size)
            let projected_input = self.input_conv.forward(&conv_input);
            trace!("Shape after input_conv: {:?}", projected_input.size());

            // 2. Downsampling convolution (hidden_size -> hidden_size)
            let mut hidden_states = self.downsample.forward(&projected_input);
            trace!("Shape after downsample: {:?}", hidden_states.size());

            // Reshape and transpose 
            let ds_frame_len = hidden_states.size()[2]; 
            hidden_states = hidden_states.view([b, nf, self.config.hidden_size as i64, ds_frame_len])
                                         .permute(&[0, 1, 3, 2]); // Shape: [B, NF, DS_FrameLen, D]
            trace!("Shape before pooling: {:?}", hidden_states.size());
                                         
            // Apply average pooling across the DS_FrameLen dimension (dim 2)
            // Keepdim=false removes the dimension, resulting in [B, NF, D]
            hidden_states = hidden_states.mean_dim(Some(&[2i64][..]), false, None); 
            trace!("Shape after pooling (for Transformer): {:?}", hidden_states.size());

            // --- Transformer Layers --- 
            let input_start_pos = current_state.start_pos;

            for (i, layer) in self.layers.iter().enumerate() {
                let layer_span = span!(Level::TRACE, "MimiTransformerBlock", layer = i);
                let _enter = layer_span.enter();
                hidden_states = layer.forward(&hidden_states, input_start_pos)?;
                trace!("Layer {} output shape: {:?}", i, hidden_states.size());
            }
            updated_state.start_pos += (nf * ds_frame_len) as usize; 
            trace!("Updated OUTGOING state start_pos to: {}", updated_state.start_pos);

            // --- Final Norm --- 
            hidden_states = self.final_norm.forward(&hidden_states);
            trace!("Final norm output shape: {:?}", hidden_states.size());

            // Return hidden_states directly (should be [B, SeqLen, Dim])
            Ok((hidden_states, updated_state))
        })
    }

    // Update encode to match the forward signature (potentially remove later)
    pub fn encode(&self, samples: &Tensor, state: Option<MimiEncoderState>) -> Result<(Tensor, MimiEncoderState)> {
        self.forward(samples, state).map_err(anyhow::Error::from)
    }
}

// Standalone weight loading function (Keep OUTSIDE impl block)
fn load_encoder_weights_manual(vs: &mut nn::VarStore, model_path: &Path, model_name: &str) -> Result<()> {
    info!("Manually loading weights for {} from {}", model_name, model_path.display());

    // Open, memory-map, and deserialize the file
    let file = File::open(model_path)
        .map_err(|e| anyhow::anyhow!("Failed to open safetensors file \"{}\": {}", model_path.display(), e))?;
    let buffer = unsafe {
        Mmap::map(&file)
            .map_err(|e| anyhow::anyhow!("Failed to memory map safetensors file \"{}\": {}", model_path.display(), e))?
    };
    let file = SafeTensors::deserialize(&buffer)
        .map_err(|e| anyhow::anyhow!("Failed to deserialize safetensors file \"{}\": {}", model_path.display(), e))?;
    
    // Convert the Vec of tensors to a HashMap for efficient lookup
    let st_tensors_vec = file.tensors();
    let st_tensors: HashMap<String, _> = st_tensors_vec.into_iter().collect();

    let vs_vars = vs.variables();
    let mut copied_count = 0;
    let mut shape_mismatch_count = 0;
    let mut missing_in_file_count = 0;

    // Iterate over variables from the mutable borrow
    for (vs_name, mut vs_tensor) in vs_vars { 
        // --- Refined Key Mapping --- 
        let base_key = vs_name.replace('/', "."); // Replace all slashes first
        let st_key = 
            if let Some(rest) = base_key.strip_prefix("encoder_transformer.") {
                // Check if the part after "encoder_transformer." starts with a number and a dot
                if let Some(first_dot_idx) = rest.find('.') {
                    let potential_num = &rest[..first_dot_idx];
                    if potential_num.chars().all(char::is_numeric) {
                        // It matches! Insert .layers. 
                        format!("encoder_transformer.layers.{}", rest)
                    } else {
                        base_key.clone() // Doesn't match num. pattern, use base_key
                    }
                } else {
                    base_key.clone() // No dot after prefix, use base_key (e.g., encoder_transformer.norm)
                }
            } else {
                 base_key.clone() // Doesn't start with encoder_transformer., use base_key (e.g., input_conv, downsample)
            };
        // --- End Refined Key Mapping --- 

        // Use HashMap lookup
        match st_tensors.get(&st_key) {
            Some(st_view) => { 
                let st_shape = st_view.shape();
                // Use the original Vec<i64> from vs_tensor.size()
                let vs_shape_i64 = vs_tensor.size();
                let vs_shape_usize: Vec<usize> = vs_shape_i64.iter().map(|&d| d as usize).collect();
                
                if st_shape == vs_shape_usize.as_slice() { 
                    // --- Correct Tensor Conversion --- 
                    let kind = vs_tensor.kind(); // Get kind from the target tensor
                    let data_slice = st_view.data();
                    // Use from_data_size with the i64 shape
                    let st_tensor = Tensor::from_data_size(data_slice, &vs_shape_i64, kind)
                                        .to_device(vs.device());
                    // --- End Correction ---
                    tch::no_grad(|| {
                        vs_tensor.copy_(&st_tensor);
                    });
                    copied_count += 1;
                } else {
                    trace!("Shape mismatch for key '{}' (VS: {:?}, File: {:?})", st_key, vs_shape_i64, st_shape);
                    shape_mismatch_count += 1;
                }
            }
            None => {
                trace!("Key '{}' not found in SafeTensors file.", st_key);
                missing_in_file_count += 1;
            }
        }
    }

    let missing_in_vs_count = st_tensors.len() - copied_count - shape_mismatch_count;
    
    if copied_count == 0 {
        warn!("No weights were copied for {}. Check paths and file content.", model_name);
        Err(anyhow::anyhow!("Load error: No weights were loaded for {}.", model_name))
    } else if shape_mismatch_count > 0 || missing_in_file_count > 0 || missing_in_vs_count > 0 {
        warn!("Some weights might be missing or mismatched for {}. Copied: {}, Shape Mismatch: {}, Missing in File: {}, Missing in VS: {}", 
            model_name, copied_count, shape_mismatch_count, missing_in_file_count, missing_in_vs_count);
        Ok(()) 
    } else {
        info!("Successfully loaded {} weights for {}.", copied_count, model_name);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Tensor, Kind};

    #[test]
    fn test_mimi_encoder_forward_shape() -> Result<(), ModelError> {
        let device = Device::Cpu;
        let encoder = MimiEncoder::new(MimiEncoderConfig::default(), device)?;
        
        let input_tensor = Tensor::randn(&[1, 1, 16000], (Kind::Float, device));
        
        // Call forward without state for basic shape test
        let (features, _state) = encoder.forward(&input_tensor, None)?;

        // Expected dimension from config
        let expected_dim = encoder.config.hidden_size;
        
        // Calculate frames - this should match what the encoder does internally
        let sample_rate = encoder.config.sample_rate;
        let frame_length_samples = (encoder.config.frame_length_ms / 1000.0 * sample_rate as f32) as i64;
        let hop_length_samples = (encoder.config.hop_length_ms / 1000.0 * sample_rate as f32) as i64;
        let input_len = 16000;
        let num_frames = (input_len - frame_length_samples) / hop_length_samples + 1;
        
        // The actual shape will be [batch, num_frames, features]
        // The encoder does average pooling over the downsampled frame dimension
        let expected_shape = &[1, num_frames, expected_dim]; 
        let actual_shape = features.size();

        assert_eq!(actual_shape, expected_shape, 
                   "Unexpected output shape. Expected {:?}, Got {:?}. InputLen={}, FrameLen={}, HopLen={}, NumFrames={}, FrameLenDown={}", 
                   expected_shape, actual_shape, input_len, frame_length_samples, hop_length_samples, num_frames, 0);
        
        Ok(())
    }

    #[test]
    fn test_mimi_encoder_configurable_frame_rate() -> Result<(), ModelError> {
        let device = Device::Cpu;
        let config1 = MimiEncoderConfig::default();
        let encoder1 = MimiEncoder::new(config1, device)?;
        let config2 = MimiEncoderConfig {
            frame_length_ms: 50.0,
            hop_length_ms: 20.0,
            ..Default::default()
        };
        let encoder2 = MimiEncoder::new(config2, device)?;
        let input_len = 24000;
        let input_tensor = Tensor::randn(&[1, 1, input_len], (Kind::Float, device));

        // Calculate expected output shapes similar to test_mimi_encoder_forward_shape
        let sample_rate1 = encoder1.config.sample_rate;
        let frame_len1 = (encoder1.config.frame_length_ms / 1000.0 * sample_rate1 as f32) as i64;
        let hop_len1 = (encoder1.config.hop_length_ms / 1000.0 * sample_rate1 as f32) as i64;
        let num_frames1 = (input_len - frame_len1) / hop_len1 + 1;
        let expected_shape1 = &[1, num_frames1, encoder1.config.hidden_size];

        let sample_rate2 = encoder2.config.sample_rate;
        let frame_len2 = (encoder2.config.frame_length_ms / 1000.0 * sample_rate2 as f32) as i64;
        let hop_len2 = (encoder2.config.hop_length_ms / 1000.0 * sample_rate2 as f32) as i64;
        let num_frames2 = (input_len - frame_len2) / hop_len2 + 1;
        let expected_shape2 = &[1, num_frames2, encoder2.config.hidden_size];

        // Call forward without state
        let (features1, _state1) = encoder1.forward(&input_tensor, None)?;
        let (features2, _state2) = encoder2.forward(&input_tensor, None)?;
        
        assert_eq!(features1.size(), expected_shape1, "Shape mismatch for config 1");
        assert_eq!(features2.size(), expected_shape2, "Shape mismatch for config 2");
        assert_ne!(features1.size()[1], features2.size()[1], "Expected different sequence lengths for different frame rates");

        Ok(())
    }

    // New test for streaming state
    #[test]
    fn test_mimi_encoder_streaming_state() -> Result<(), ModelError> {
        let device = Device::Cpu;
        let config = MimiEncoderConfig::default();
        let encoder = MimiEncoder::new(config.clone(), device)?;
        
        // Input split into two chunks
        let chunk1_len = 8000;
        let chunk2_len = 8000;
        let input_chunk1 = Tensor::randn(&[1, 1, chunk1_len], (Kind::Float, device));
        let input_chunk2 = Tensor::randn(&[1, 1, chunk2_len], (Kind::Float, device));
        let full_input = Tensor::cat(&[&input_chunk1, &input_chunk2], 2); // Concat along sequence dim

        // Process first chunk
        let (_features1, state1) = encoder.forward(&input_chunk1, None)?;
        
        // Process second chunk with state from first
        let (_features2, state2) = encoder.forward(&input_chunk2, Some(state1.clone()))?;

        // Process full input at once (for comparison)
        let (_features_full, state_full) = encoder.forward(&full_input, None)?;

        // Verify the start position is monotonically increasing
        assert!(state2.start_pos >= state1.start_pos, 
                "State start_pos should increase or stay the same after processing the second chunk");
        
        // Check that the total length processed is as expected
        assert_eq!(state_full.start_pos, state2.start_pos, 
                  "Final state positions should match between chunked and full processing");

        // Print relevant values for debugging and future test refinement
        println!("Chunk1 start_pos: {}", state1.start_pos);
        println!("Chunk2 start_pos: {}", state2.start_pos);
        println!("Full start_pos: {}", state_full.start_pos);

        Ok(())
    }
}

// --- End of content moved from src/audio/codec.rs --- 

pub mod compat;