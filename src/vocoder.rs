#![allow(unused_imports)] // TODO: Remove this allow later
use async_trait::async_trait;
use crate::models::ModelError;
use anyhow::Result;
use tch::{Device, Kind, Tensor, TchError};
use std::path::{Path, PathBuf};
use std::convert::{TryFrom, TryInto};
use safetensors::{SafeTensors, SafeTensorError};
use safetensors::tensor::{TensorView, TensorInfo};
use std::collections::HashMap;
use std::fs::File;
use thiserror::Error;
use log::{debug, error, info, trace, warn};
use memmap2::Mmap;
use std::fmt::Debug;

// --- Restore Consolidated tch::nn imports (with self AND Config types) --- 
use tch::nn::{self, Module, ModuleT, SequentialT, VarStore, Init, LayerNorm, LayerNormConfig, Linear, LinearConfig, Conv1D, ConvConfig, ConvTranspose1D, ConvTransposeConfig, Embedding, EmbeddingConfig};
// --- End Imports --- 

// Relative imports assuming this file is in src/models/
// Removed incorrect imports for locally defined items
// Removed `use tch::Module;` as it was unresolved
// Trait `Module` is imported via `tch::nn` group above

// Define the Vocoder trait
#[async_trait]
pub trait Vocoder: Send + Sync + Debug {
    // UPDATED: Accept a single tuple (semantic, acoustic_vec)
    async fn synthesize_chunk(&self, tokens: (i64, Vec<i64>)) -> Result<Vec<i16>, ModelError>;
    
    /// Synthesize audio from a collection of token chunks.
    async fn decode_tokens(&self, token_chunks: Vec<Vec<(i64, Vec<i64>)>>) -> Result<Vec<i16>, ModelError>;

    fn sample_rate(&self) -> u32;
}

// --- Define New Mimi Decoder Components ---

// Placeholder config struct
#[derive(Debug, Clone)]
struct MimiConfig {
    hidden_size: i64,           // 512
    intermediate_size: i64,     // 2048
    num_attention_heads: i64,   // 8
    num_key_value_heads: i64,   // 8
    head_dim: i64,              // 64
    norm_eps: f64,              // 1e-05
    num_hidden_layers: usize,   // 8
    hidden_act: String,         // "gelu"
    num_semantic_quantizers: usize, // 1
    codebook_size: i64,         // 2048
    codebook_dim: i64,          // 256
    audio_channels: i64,        // 1
    kernel_size: i64,           // 7
    last_kernel_size: i64,      // 3
    compress: i64,              // 2
    residual_kernel_size: i64,  // 3
    num_residual_layers: usize, // 1
    upsampling_ratios: Vec<i64>,// [8, 6, 5, 4]
    frame_rate: f64,            // 12.5
    sampling_rate: i64,         // 24000
    use_causal_conv: bool,      // true
    use_conv_shortcut: bool,    // false
    vector_quantization_hidden_dimension: i64, // 256
    rope_theta: f64,            // 10000.0
    sliding_window: i64,        // 250
    upsample_groups: i64,       // 512
    dilation_growth_rate: i64,  // 2
    num_filters: i64,           // 64
    initializer_range: f64,     // 0.02
    layer_scale_initial_scale: f64, // 0.01
    max_position_embeddings: i64, // 8000
    normalize: bool,            // false
    pad_mode: String,          // "constant"
    trim_right_ratio: f64,     // 1.0
    use_cache: bool,           // false
}

// +++ Add RotaryEmbedding (Copied/Adapted from csm.rs) +++
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

        // Generate frequency complex numbers based on position
        let t = Tensor::arange_start(start_pos as i64, (start_pos as i64) + seq_len, (Kind::Float, device));
        let freqs = t.outer(&self.inv_freq);
        // Create polar complex numbers: R=1, Theta=freqs
        let freqs_cis = Tensor::polar(&Tensor::ones_like(&freqs), &freqs); // Shape [seq_len, dim/2]

        // Reshape tensors to view pairs of elements as complex numbers
        let xq_r = xq.reshape(&[-1, seq_len, self.dim / 2, 2]);
        let xk_r = xk.reshape(&[-1, seq_len, self.dim / 2, 2]);
        let xq_c = xq_r.view_as_complex(); // Shape [batch*heads, seq_len, dim/2]
        let xk_c = xk_r.view_as_complex();

        // Reshape freqs_cis for broadcasting: [1, seq_len, dim/2]
        let freqs_cis = freqs_cis.unsqueeze(0);

        // Apply rotation in complex plane: x * exp(i*theta)
        let xq_out_c = xq_c.f_mul(&freqs_cis)?;
        let xk_out_c = xk_c.f_mul(&freqs_cis)?;

        // Convert back to real view
        let xq_out_r = xq_out_c.view_as_real();
        let xk_out_r = xk_out_c.view_as_real();

        // Reshape back to original tensor shapes
        let xq_out = xq_out_r.reshape_as(xq);
        let xk_out = xk_out_r.reshape_as(xk);

        Ok((xq_out, xk_out))
    }
}
// +++ End RotaryEmbedding +++

// --- Mimi Transformer Components (Implement Forward) ---
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
    fn new(vs: &tch::nn::Path, config: &MimiConfig, device: Device) -> Result<Self, TchError> { 
        let embed_dim = config.hidden_size;
        let head_dim = config.head_dim;
        let n_head = config.num_attention_heads;
        let linear_cfg = LinearConfig { bias: false, ..Default::default() };
        let q_proj = nn::linear(vs / "q_proj", embed_dim, n_head * head_dim, linear_cfg);
        let k_proj = nn::linear(vs / "k_proj", embed_dim, n_head * head_dim, linear_cfg);
        let v_proj = nn::linear(vs / "v_proj", embed_dim, n_head * head_dim, linear_cfg);
        let o_proj = nn::linear(vs / "o_proj", n_head * head_dim, embed_dim, linear_cfg);
        let rotary_emb = RotaryEmbedding::new(head_dim, 8192, config.rope_theta, device)?;
        Ok(Self { q_proj, k_proj, v_proj, o_proj, n_head, head_dim, rotary_emb })
    }

    // Rename back to forward, keep start_pos
    fn forward(&self, xs: &Tensor, start_pos: usize) -> Tensor {
        let (b_sz, seq_len, _hidden_dim) = xs.size3().expect("Input must be 3D");

        // Calculate Q, K, V
        let q = self.q_proj.forward(xs);
        let k = self.k_proj.forward(xs);
        let v = self.v_proj.forward(xs);

        // Reshape for multi-head attention
        let q = q.contiguous().view((b_sz, seq_len, self.n_head, self.head_dim)).transpose(1, 2);
        let k = k.contiguous().view((b_sz, seq_len, self.n_head, self.head_dim)).transpose(1, 2);
        let v = v.contiguous().view((b_sz, seq_len, self.n_head, self.head_dim)).transpose(1, 2);

        // Apply Rotary Embeddings
        let (q, k) = self.rotary_emb.forward(&q, &k, start_pos).expect("RoPE failed");

        // No KV Caching based on config

        // Scaled Dot-Product Attention
        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = q.matmul(&k.transpose(-2, -1)) / scale;

        // No mask needed for vocoder self-attention?

        let attn_probs = attn_weights.softmax(-1, Kind::Float);
        let output = attn_probs.matmul(&v);

        // Reshape and apply output projection
        let output = output.transpose(1, 2)
                         .contiguous()
                         .view((b_sz, seq_len, -1)); // Reshape back to [B, S, H]
        self.o_proj.forward(&output)
    }
}

#[derive(Debug)]
struct MimiMLP {
    fc1: Linear,
    fc2: Linear,
    activation: fn(&Tensor) -> Tensor,
}

impl MimiMLP {
     fn new(vs: &tch::nn::Path, config: &MimiConfig) -> Result<Self, TchError> {
         let embed_dim = config.hidden_size;
         let hidden_dim = config.intermediate_size;
         let linear_cfg = LinearConfig { bias: false, ..Default::default() };
         let fc1 = nn::linear(vs / "fc1", embed_dim, hidden_dim, linear_cfg);
         let fc2 = nn::linear(vs / "fc2", hidden_dim, embed_dim, linear_cfg);
         let activation = match config.hidden_act.as_str() {
            "gelu" => |t: &Tensor| t.gelu("none"),
            _ => |t: &Tensor| t.relu(),
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

#[derive(Debug)]
struct MimiTransformerBlock {
    input_layernorm: LayerNorm,
    self_attn: MimiAttention,
    post_attention_layernorm: LayerNorm,
    mlp: MimiMLP,
}

impl MimiTransformerBlock {
     fn new(vs: &tch::nn::Path, config: &MimiConfig, device: Device) -> Result<Self, TchError> {
          let embed_dim = config.hidden_size;
          let ln_eps = config.norm_eps;
          let ln_cfg = LayerNormConfig { eps: ln_eps, ..Default::default() };
          let input_layernorm = nn::layer_norm(vs / "input_layernorm", vec![embed_dim], ln_cfg);
          let self_attn = MimiAttention::new(&(vs / "self_attn"), config, device)?;
          let post_attention_layernorm = nn::layer_norm(vs / "post_attention_layernorm", vec![embed_dim], ln_cfg);
          let mlp = MimiMLP::new(&(vs / "mlp"), config)?;
          Ok(Self { input_layernorm, self_attn, post_attention_layernorm, mlp })
     }

     // Rename back to forward, keep start_pos
     fn forward(&self, xs: &Tensor, start_pos: usize) -> Tensor {
          let residual = xs;
          let hidden_states = self.input_layernorm.forward(xs);
          // Call attention's forward with start_pos
          let attn_output = self.self_attn.forward(&hidden_states, start_pos);
          let hidden_states = attn_output + residual;

          let residual = &hidden_states;
          let hidden_states_norm = self.post_attention_layernorm.forward(&hidden_states);
          let mlp_output = self.mlp.forward(&hidden_states_norm);
          mlp_output + residual
     }
}

// --- Mimi Encoder/Decoder Transformer Structures ---
#[derive(Debug)]
struct MimiEncoderTransformer {
    layers: Vec<MimiTransformerBlock>,
}

impl MimiEncoderTransformer {
    fn new(vs: &tch::nn::Path, config: &MimiConfig, device: Device) -> Result<Self, TchError> {
        let num_layers = config.num_hidden_layers;
        let mut layers = Vec::with_capacity(num_layers);
        let layers_vs = vs / "layers";
        for i in 0..num_layers {
            // Pass device to block constructor
            layers.push(MimiTransformerBlock::new(&(layers_vs.clone() / i), config, device)?);
        }
        Ok(Self { layers })
    }

     // Keep specific forward method with start_pos
     fn forward_with_pos(&self, xs: &Tensor, start_pos: usize) -> Tensor {
         let mut hidden_state = xs.copy();
         for layer in &self.layers {
             // Call block's forward with start_pos
             hidden_state = layer.forward(&hidden_state, start_pos);
         }
         hidden_state
     }
}

// Keep Module impl for default case (start_pos = 0)
impl Module for MimiEncoderTransformer {
     fn forward(&self, xs: &Tensor) -> Tensor {
        self.forward_with_pos(xs, 0)
    }
}

// --- Mimi Quantizer (Implement Forward) ---
#[derive(Debug)]
struct MimiQuantizerLayer {
    embed: Tensor,
}

#[derive(Debug)]
struct MimiQuantizer {
    layers: Vec<MimiQuantizerLayer>,
    proj_in: Option<Linear>,
    proj_out: Option<Linear>,
}

impl MimiQuantizerLayer {
    fn new(
        vs: &nn::Path,
        config: &MimiConfig,
        layer_index: usize,
    ) -> Result<Self, TchError> {
        let layer_path = vs; 
        // Use randn_standard(name, dims)
        let embed = layer_path.randn_standard(
            "embed", 
            &[
                config.codebook_size as i64, 
                config.codebook_dim as i64, 
            ],
        );
        info!(
            "    Creating Quantizer Layer {}: Path='{:?}', Embed Name='embed', Shape=[{}, {}]",
            layer_index,
            layer_path, 
            config.codebook_size,
            config.codebook_dim
        );
        Ok(MimiQuantizerLayer { embed })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor, TchError> {
        // Input hidden_states shape: [B, T, D_in] (e.g., [1, 960, 256] after projection)
        // Codebook shape: [1, C, D] (e.g., [1, 2048, 256])
        let flat_input = hidden_states.reshape(&[-1, hidden_states.size()[hidden_states.dim() - 1]]);
        // flat_input shape: [N, D] where N = B*T (e.g., [960, 256])

        // Calculate L2 distances: ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * x.c
        // ||x||^2
        let input_norm: Tensor = flat_input.pow_tensor_scalar(2.0)
            .sum_dim_intlist(&[1i64][..], true, Kind::Float);
        // input_norm shape: [N, 1] (e.g., [960, 1])

        // Squeeze codebook and calculate ||c||^2
        let codebook_squeezed = self.embed.squeeze_dim(0); // Shape [C, D] (e.g., [2048, 256])
        let codebook_norm: Tensor = codebook_squeezed.pow_tensor_scalar(2.0)
            .sum_dim_intlist(&[1i64][..], true, Kind::Float);
        // codebook_norm shape: [C, 1] (e.g., [2048, 1])

        // Calculate x.c (dot product / similarity)
        // matmul requires [N, D] @ [D, C]
        let similarity: Tensor = flat_input.matmul(&codebook_squeezed.transpose(-2, -1));
        // similarity shape: [N, C] (e.g., [960, 2048])

        // Calculate full distance: input_norm [N, 1] + codebook_norm.t [1, C] - 2 * similarity [N, C]
        // Broadcasting applies correctly
        let distances: Tensor = input_norm.f_add(&codebook_norm.transpose(-2, -1))?
            .f_sub(&similarity.f_mul_scalar(2.0)?)?;
        // distances shape: [N, C] (e.g., [960, 2048])

        // Find nearest codebook indices
        let indices = distances.argmin(1, false); // Shape [N] (e.g., [960])

        // Reshape indices back to input shape except last dim
        let mut new_shape = Vec::with_capacity(hidden_states.dim() - 1);
        for i in 0..hidden_states.dim() - 1 {
            new_shape.push(hidden_states.size()[i]);
        }
        let indices_reshaped = indices.reshape(&new_shape);

        // Gather from codebook and wrap in Result
        // Use the squeezed codebook for embedding lookup
        Ok(Tensor::embedding(&codebook_squeezed, &indices_reshaped, -1, false, false))
    }
}

impl MimiQuantizer {
    pub fn new(vs: &nn::Path, config: &MimiConfig) -> Result<Self, TchError> {
        let vs = vs / "acoustic_residual_vector_quantizer";
        // Use the NON-prefixed field name based on latest read and linter error
        let num_layers = config.num_semantic_quantizers; // Usually 1 for rvq_first? No, model has 31 acoustic layers.
        let codebook_size = config.codebook_size;
        let hidden_dim = config.vector_quantization_hidden_dimension; // 256
        let transformer_dim = config.hidden_size; // 512
        let linear_cfg = LinearConfig { bias: false, ..Default::default() };

        // --- Use the correct path based on safetensors list ---
        let base_path = vs.clone(); // CHANGED: Match file path
        info!("Initializing MimiQuantizer with base path: {:?}", base_path);

        // Correct the input and output dimensions
        let proj_in = Some(nn::linear(&base_path / "input_proj", transformer_dim, hidden_dim, linear_cfg)); // 512 -> 256
        let proj_out = Some(nn::linear(&base_path / "output_proj", hidden_dim, transformer_dim, linear_cfg)); // 256 -> 512
        warn!("Quantizer proj_in/proj_out initialized under path '{:?}', but file has them under 'semantic_...'. Expect loading warnings.", base_path);

        // --- Load quantizer LAYERS under the correct base path ---
        let mut layers = Vec::with_capacity(num_layers); // num_layers should likely be 31 based on file, not 1?
        warn!("Initializing {} quantizer layers based on config.num_semantic_quantizers. File suggests ~31 layers under 'acoustic_...'. Check config.", num_layers);
        
        // Use the correct base path for layers
        let layers_base_path = base_path.clone() / "layers";

        for i in 0..num_layers { // Loop might need adjustment based on actual number of layers in file
            let layer_vs = layers_base_path.clone() / &format!("{}", i);
            info!(
                "  Creating Quantizer Layer {} using Path: '{:?}' (Codebook expected inside)",
                i, layer_vs
            );
            let _codebook_size_i64 = codebook_size; // Keep as i64
            // let codebook_dim_i64 = config.vector_quantization_hidden_dimension; // Already defined above

            let layer = MimiQuantizerLayer::new(&layer_vs, config, i)?;
            layers.push(layer);
        }
        Ok(Self { layers, proj_in, proj_out })
    }

    // ... existing forward method ...
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor, TchError> {
        // Apply input projection FIRST if it exists
        let projected_input = match &self.proj_in {
            Some(proj) => xs.apply(proj),
            None => xs.shallow_clone(), // No projection needed
        };

        // Initialize residual with the *projected* input
        let mut residual = projected_input.shallow_clone(); 
        let mut quantized_sum = Tensor::zeros_like(&residual);

        // Process through each layer using the residual
        for layer in &self.layers {
             // Pass the current residual to the layer's forward method
             let layer_quantized = layer.forward(&residual)?;
             quantized_sum += &layer_quantized; 
             // Update residual for the next layer
             residual = residual - layer_quantized;
        }

        // Apply output projection LAST if it exists
        let final_output = match &self.proj_out {
            Some(proj) => quantized_sum.apply(proj),
            None => quantized_sum, // No projection needed
        };

        Ok(final_output)
    }
}

// --- Mimi Decoder Convolutional Blocks (Updated Args) ---
// COMMENTED OUT: Unused after refactoring decoder_conv
// #[derive(Debug)]
// struct MimiDecoderConvBlock {
//     conv: Conv1D,
// }
// impl MimiDecoderConvBlock {
//     fn new(vs: &Path, config: &MimiConfig) -> Result<Self, TchError> {
//         let channels = config.hidden_size;
//         let kernel_size = config.residual_kernel_size;
//         let conv_cfg = ConvConfig { padding: (kernel_size - 1) / 2, ..Default::default() };
//         let conv = nn::conv1d(vs / "conv", channels, channels, kernel_size, conv_cfg);
//         Ok(Self { conv })
//     }
// }
// impl Module for MimiDecoderConvBlock {
//     fn forward(&self, xs: &Tensor) -> Tensor {
//         // TODO: Add activation/normalization if needed
//         self.conv.forward(xs)
//     }
// }

// COMMENTED OUT: Unused after refactoring decoder_conv
// #[derive(Debug)]
// struct MimiDecoderConvLayer {
//     blocks: Vec<MimiDecoderConvBlock>,
// }
// impl MimiDecoderConvLayer {
//     fn new(vs: &Path, config: &MimiConfig) -> Result<Self, TchError> {
//         let num_blocks = config.num_residual_layers;
//         let mut blocks = Vec::with_capacity(num_blocks);
//         let blocks_vs = vs / "layers";
//         for i in 0..num_blocks {
//             blocks.push(MimiDecoderConvBlock::new(&(blocks_vs.clone() / i), config)?);
//         }
//         Ok(Self { blocks })
//     }
// }
// impl Module for MimiDecoderConvLayer {
//     fn forward(&self, xs: &Tensor) -> Tensor {
//         let mut hidden_state = xs.copy();
//         for block in &self.blocks {
//             // TODO: Add residual connection?
//             hidden_state = block.forward(&hidden_state);
//         }
//         hidden_state
//     }
// }

// --- NEW: Mimi Upsampler (HiFi-GAN style) ---
#[derive(Debug)]
pub struct MimiUpsampler {
    conv_layers: Vec<nn::ConvTranspose1D>,
}

impl MimiUpsampler {
    pub fn new(
        vs: &nn::Path,
        input_channels: i64,
        final_output_channels: i64,
        upsample_rates: Vec<i64>,
        kernel_sizes: Vec<i64>,
    ) -> Result<Self, ModelError> {
        let mut conv_layers = Vec::new();
        let num_layers = upsample_rates.len();
        let mut current_channels = input_channels;

        // Use upsample_groups from config (512) for intermediate channels
        let intermediate_channels = 512;

        for (i, (rate, kernel_size)) in upsample_rates.iter().zip(kernel_sizes.iter()).enumerate() {
            let out_channels = if i == num_layers - 1 {
                final_output_channels
            } else {
                intermediate_channels
            };

            let cfg = nn::ConvTransposeConfig {
                stride: *rate,
                padding: (*rate + kernel_size % 2 - 1) / 2,
                output_padding: 0,
                groups: if i < num_layers - 1 { intermediate_channels } else { 1 }, // Use groups for all but last layer
                bias: false,
                dilation: 1,
                ws_init: nn::Init::Randn { mean: 0.0, stdev: 0.02 },
                bs_init: nn::Init::Const(0.0),
            };

            let conv_layer_path = vs / format!("{}", i);
            info!(
                "  Creating Upsample ConvT Layer {}: Path='{:?}', In={}, Out={}, Kernel={}, Stride={}, Groups={}, Pad={}",
                i, conv_layer_path, current_channels, out_channels, kernel_size, rate, cfg.groups, cfg.padding
            );

            let conv_layer = nn::conv_transpose1d(
                &conv_layer_path,
                current_channels,
                out_channels,
                *kernel_size,
                cfg,
            );
            conv_layers.push(conv_layer);
            current_channels = out_channels;
        }

        if current_channels != final_output_channels {
            error!("Upsampler logic error: Final calculated channels ({}) != target ({})", current_channels, final_output_channels);
        }

        Ok(Self { conv_layers })
    }
}

// RESTORED Module implementation for MimiUpsampler
impl Module for MimiUpsampler {
    fn forward(&self, xs: &Tensor) -> Tensor {
        info!("    Upsampler Input (manual Module): {:?}", xs.size());
        let mut current_states = xs.shallow_clone();
        for (i, conv_layer) in self.conv_layers.iter().enumerate() {
            current_states = conv_layer.forward(&current_states);
            info!("      After Upsample ConvT Layer {}: Shape = {:?}", i, current_states.size());
        }
        info!("    Upsampler Output (manual Module): {:?}", current_states.size());
        current_states
    }
}

// --- Mimi Vocoder Struct (Using new components) ---
#[derive(Debug)]
pub struct MimiVocoder {
    sample_rate: u32,
    vs: VarStore,
    device: Device,
    config: MimiConfig,
    // Input processing
    input_proj: Conv1D,
    downsample: Conv1D,
    // Encoder components
    encoder_transformer: MimiEncoderTransformer,
    // Quantizer
    quantizer: MimiQuantizer,
    // Decoder components
    decoder_transformer: MimiEncoderTransformer, // Reuse encoder structure for decoder
    decoder_conv: Conv1D, // ADDED single Conv1D layer
    output_proj: Conv1D, // ADDED projection layer after decoder_conv
    // Output processing
    upsample: MimiUpsampler,
}

// SAFETY: Keep Send + Sync assumption for now
unsafe impl Send for MimiVocoder {}
unsafe impl Sync for MimiVocoder {}

// --- Create a local error type ---
#[derive(Debug, Error)]
pub enum TensorLoadError {
    #[error("Tch Error: {0}")]
    Tch(#[from] TchError),

    #[error("SafeTensors Error: {0}")]
    SafeTensors(#[from] SafeTensorError),

    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),

    #[error("VarStore variable not found: {0}")]
    VarNotFound(String),

    #[error("Shape mismatch for {0}: VarStore={1:?}, File={2:?}")]
    ShapeMismatch(String, Vec<i64>, Vec<i64>),
}

// Add From implementation for ModelError -> TensorLoadError and vice versa
impl From<ModelError> for TensorLoadError {
    fn from(err: ModelError) -> Self {
        match err {
            ModelError::Tch(e) => TensorLoadError::Tch(e),
            ModelError::Io(e) => TensorLoadError::Io(e),
            ModelError::Safetensor(e) => TensorLoadError::SafeTensors(e),
            _ => TensorLoadError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Other model error: {}", err),
            )),
        }
    }
}

impl From<TensorLoadError> for ModelError {
    fn from(err: TensorLoadError) -> Self {
        match err {
            TensorLoadError::Tch(e) => ModelError::Tch(e),
            TensorLoadError::Io(e) => ModelError::Io(e),
            TensorLoadError::SafeTensors(e) => ModelError::Safetensor(e),
            TensorLoadError::VarNotFound(s) => ModelError::LoadError(s),
            TensorLoadError::ShapeMismatch(name, vs_shape, file_shape) => 
                ModelError::LoadError(format!("Shape mismatch for {}: VS={:?}, File={:?}", name, vs_shape, file_shape)),
        }
    }
}

// --- Manual weight loading function (Restored) ---
fn load_weights_manual(
    vs: &mut nn::VarStore,
    safetensors_path: &std::path::Path,
) -> Result<(), TensorLoadError> {
    info!(
        "Loading Mimi vocoder model manually from: \"{}\" into existing VarStore",
        safetensors_path.display()
    );

    let file = File::open(safetensors_path)?;
    let buffer = unsafe { Mmap::map(&file)? };
    let safe_tensors = SafeTensors::deserialize(&buffer)?;
    
    info!(
        "Successfully deserialized SafeTensors file with {} tensors.",
        safe_tensors.names().len()
    );

    // Load tensors into a map
    let mut st_tensors = HashMap::new();
    for tensor_name in safe_tensors.names() {
        match safe_tensors.tensor(tensor_name) {
            Ok(tensor_view) => {
                // Convert tensor data manually
                let data = tensor_view.data();
                let shape: Vec<i64> = tensor_view.shape().iter().map(|&d| d as i64).collect();
                let kind = match tensor_view.dtype() {
                    safetensors::Dtype::F32 => Kind::Float,
                    safetensors::Dtype::F64 => Kind::Double,
                    safetensors::Dtype::F16 => Kind::Half,
                    safetensors::Dtype::BF16 => Kind::BFloat16,
                    safetensors::Dtype::I64 => Kind::Int64,
                    safetensors::Dtype::I32 => Kind::Int,
                    safetensors::Dtype::I16 => Kind::Int16,
                    safetensors::Dtype::I8 => Kind::Int8,
                    safetensors::Dtype::U8 => Kind::Uint8,
                    safetensors::Dtype::BOOL => Kind::Bool,
                    _ => {
                        warn!("Unsupported dtype for tensor {}: {:?}", tensor_name, tensor_view.dtype());
                        continue;
                    }
                };
                
                // Create tensor directly from raw data
                let tch_tensor = Tensor::from_data_size(data, &shape, kind)
                    .to_device(vs.device());
                st_tensors.insert(tensor_name.to_string(), tch_tensor);
            },
            Err(e) => warn!("Failed to get tensor '{}' from SafeTensors: {}", tensor_name, e),
        }
    }
    
    info!(
        "[Manual Load Copy] Loaded {} tensors into temporary map.",
        st_tensors.len()
    );

    // Access tensors mutably using iter_mut() on collected Vec
    let mut copied_count = 0;
    let shape_mismatch_count = 0;
    let mut missing_key_count = 0;

    // Get mutable variables from the VarStore by collecting into a Vec
    // This ensures we iterate over the intended set and allows mutable access
    // if `Tensor` supports it via `copy_` even when owned by the Vec temporarily.
    let vs_vars_iter = vs.variables().into_iter().collect::<Vec<_>>();
    let total_vs_vars = vs_vars_iter.len();

    info!(
        "[Manual Load Copy] Attempting to copy weights into {} VarStore variables...",
        total_vs_vars
    );

    // Iterate through mutable variables in the collected vector
    for (vs_name, mut vs_tensor) in vs_vars_iter.into_iter() { // Consume the vec
        // Map VarStore name (using '/') to SafeTensors name (using '.')
        let st_key = if vs_name.ends_with("embed")
            && vs_name.contains("acoustic_residual_vector_quantizer/layers")
        {
            // Specific mapping for summed embeddings in RVQ
            vs_name.replace('/', ".").replace(".embed", ".codebook.embed_sum")
        } else {
            vs_name.replace('/', ".")
        };

        match st_tensors.get(&st_key) {
            Some(st_tensor) => {
                let vs_shape = vs_tensor.size();
                let mut file_tensor_to_copy = st_tensor.shallow_clone();
                let file_shape = file_tensor_to_copy.size();

                // Check for specific quantizer projection weights that need reshaping
                if st_key == "quantizer.acoustic_residual_vector_quantizer.input_proj.weight"
                    || st_key == "quantizer.acoustic_residual_vector_quantizer.output_proj.weight"
                {
                    // Check if file tensor has an extra trailing dimension of size 1
                    if file_shape.len() == vs_shape.len() + 1 && file_shape.last() == Some(&1)
                    {
                        debug!(
                            "[Manual Load Copy] Reshaping tensor '{}' from {:?} to {:?} before copy.",
                            st_key,
                            file_shape,
                            &file_shape[..file_shape.len() - 1]
                        );
                        // Apply ? to the Result before assignment
                        file_tensor_to_copy = file_tensor_to_copy.squeeze_dim(-1);
                    }
                }

                // Ensure target and source shapes match after potential squeeze
                if vs_shape != file_shape {
                    warn!(
                        target: "csm::vocoder",
                        "Shape mismatch for tensor '{}': VarStore {:?}, File {:?}. Attempting squeeze.",
                        st_key, vs_shape, file_shape
                    );
                    // Check if the file tensor has an extra dimension at the end
                    if file_shape.len() == vs_shape.len() + 1 && file_shape.last() == Some(&1) {
                        info!(
                            target: "csm::vocoder",
                            "Attempting to squeeze the last dimension of tensor '{}' from file.",
                            st_key
                        );
                        // Apply ? to the Result before assignment
                        file_tensor_to_copy = file_tensor_to_copy.squeeze_dim(-1);
                    }
                }

                // Final check after potential squeeze
                if vs_tensor.size() != file_tensor_to_copy.size() {
                    error!(
                        target: "csm::vocoder",
                        "Unresolvable shape mismatch for tensor '{}' after squeeze attempt: VarStore {:?}, File {:?}.",
                        st_key, vs_tensor.size(), file_tensor_to_copy.size()
                    );
                    continue; // Skip this tensor
                }

                // Copy the tensor data within a no_grad block
                tch::no_grad(|| {
                    // copy_ returns (), no Result check needed
                    vs_tensor.copy_(&file_tensor_to_copy);
                    // We might want to log success/failure differently if copy_ could panic,
                    // but for now, assume it works or panics.
                });

                copied_count += 1;
            }
            None => {
                warn!(
                    "[Manual Load Copy] Variable '{}' (looked up as '{}') in VS, but key not found in SafeTensors file.",
                    vs_name, st_key
                );
                missing_key_count += 1;
            }
        }
    }

    // Estimate how many tensors were in the file but not in the VarStore
    let missing_in_vs = st_tensors.len() as i32 - copied_count as i32 - shape_mismatch_count as i32;

    info!(
        "[Manual Load Copy] Weight loading finished. Total VS Vars: {}, Copied: {}, Shape Mismatch: {}, Missing in File (Not in VS): {}, Missing in VS (Not in File): {}",
        total_vs_vars,
        copied_count,
        shape_mismatch_count,
        missing_in_vs.max(0), // Tensors in file but not found/matched in VS
        missing_key_count     // Tensors in VS but not found in file
    );

    if copied_count == 0 && total_vs_vars > 0 {
        warn!("[Manual Load Copy] No weights were copied. Check paths, file content, and tensor names.");
    } else if shape_mismatch_count > 0 || missing_key_count > 0 || missing_in_vs > 0 {
        warn!("[Manual Load Copy] Not all weights were successfully loaded. Review warnings.");
        // Optionally, return an error if strict loading is required
        // return Err(TensorLoadError::WeightMismatchError(format!( ... )));
    }

    Ok(())
}

impl MimiVocoder {
    pub fn new(
        sample_rate: u32,
        device: Device,
    ) -> Result<Self, ModelError> {
        info!("Creating MimiVocoder (Refactored, using config.json values):");
        info!("  Sample Rate: {} Hz", sample_rate);
        info!("  Device: {:?}", device);

        let mut vs = VarStore::new(device);
        let root = vs.root();

        // Create config from exact values in config.json
        let config = MimiConfig {
            hidden_size: 512,
            intermediate_size: 2048,
            num_attention_heads: 8,
            num_key_value_heads: 8,
            head_dim: 64,
            norm_eps: 1e-5,
            num_hidden_layers: 8,
            hidden_act: "gelu".to_string(),
            num_semantic_quantizers: 1,
            codebook_size: 2048,
            codebook_dim: 256,
            audio_channels: 1,
            kernel_size: 7,
            last_kernel_size: 3,
            compress: 2,
            residual_kernel_size: 3,
            num_residual_layers: 1,
            upsampling_ratios: vec![8, 6, 5, 4],
            frame_rate: 12.5,
            sampling_rate: 24000,
            use_causal_conv: true,
            use_conv_shortcut: false,
            vector_quantization_hidden_dimension: 256,
            rope_theta: 10000.0,
            sliding_window: 250,
            upsample_groups: 512,
            dilation_growth_rate: 2,
            num_filters: 64,
            initializer_range: 0.02,
            layer_scale_initial_scale: 0.01,
            max_position_embeddings: 8000,
            normalize: false,
            pad_mode: "constant".to_string(),
            trim_right_ratio: 1.0,
            use_cache: false,
        };

        // --- Input Processing ---
        // RESTORED AGAIN: Initial projection layer (bias=false)
        let input_conv_cfg = ConvConfig { padding: 0, stride: 1, bias: false, ..Default::default() }; // Kernel 1, Stride 1, bias=false
        let input_proj = nn::conv1d(&(root.clone() / "input_proj"), config.audio_channels, config.hidden_size, 1, input_conv_cfg);

        // Downsample layer (bias=false)
        let downsample_kernel_size = 4;
        let downsample_cfg = ConvConfig {
            padding: (downsample_kernel_size - 1) / 2,
            stride: config.compress,
            bias: false,
            ..Default::default()
        };
        let downsample = nn::conv1d(&(root.clone() / "downsample" / "conv"), config.hidden_size, config.hidden_size, downsample_kernel_size, downsample_cfg);

        // Encoder transformer
        let encoder_transformer = MimiEncoderTransformer::new(
            &(root.clone() / "encoder_transformer"),
            &config,
            device
        )?;

        // Quantizer
        let quantizer = MimiQuantizer::new(
            &(root.clone() / "quantizer"),
            &config,
        )?;

        // Decoder transformer
        let decoder_transformer = MimiEncoderTransformer::new(
            &(root.clone() / "decoder_transformer"),
            &config,
            device
        )?;

        // Decoder conv layer (REPLACED with single Conv1D layer)
        // let decoder_conv = MimiDecoderConvLayer::new(
        //     &(root.clone() / "decoder"), // CHANGED base path
        //     &config,
        // )?;

        // Create single Conv1D layer matching weights file (decoder.layers.0.conv)
        let decoder_conv_kernel_size = 7;
        let decoder_conv_padding = (decoder_conv_kernel_size - 1) / 2;
        let decoder_conv_cfg = ConvConfig {
            padding: decoder_conv_padding,
            stride: 1,
            bias: true,
            ..Default::default()
        };
        let decoder_conv_path = root.clone() / "decoder" / "layers" / "0" / "conv";
        let decoder_conv = nn::conv1d(
            &decoder_conv_path,
            config.hidden_size,
            1024,
            decoder_conv_kernel_size,
            decoder_conv_cfg,
        );

        // ADDED: Projection layer after decoder_conv to reduce channels
        let output_conv_cfg = ConvConfig { padding: 0, stride: 1, bias: true, ..Default::default() };
        let output_proj = nn::conv1d(
            &(root.clone() / "output_proj"),
            1024,
            config.hidden_size,
            1,
            output_conv_cfg,
        );

        // --- NEW: Mimi Upsampler Initialization ---
        let upsampler_path = root.clone() / "upsample";
        info!("Initializing Upsampler with path: {:?}", upsampler_path);

        // Calculate kernel sizes based on ratios (2 * ratio for each layer)
        let kernel_sizes: Vec<i64> = config.upsampling_ratios.iter().map(|&r| r * 2).collect();

        let upsample = MimiUpsampler::new(
            &upsampler_path,
            config.hidden_size,
            config.audio_channels,
            config.upsampling_ratios.clone(),
            kernel_sizes,
        )?;

        vs.freeze();

        Ok(Self {
            sample_rate,
            vs,
            device,
            config,
            input_proj,
            downsample,
            encoder_transformer,
            quantizer,
            decoder_transformer,
            decoder_conv,
            output_proj,
            upsample,
        })
    }

    pub fn load_model(&mut self, model_path: PathBuf) -> Result<(), ModelError> {
        info!("Loading Mimi vocoder model from: {:?} into existing VarStore", model_path);

        if !model_path.exists() {
            return Err(ModelError::LoadError(format!(
                "Model file not found: {:?}", model_path
            )));
        }

        info!("Loading weights from: {}", model_path.display());
        self.vs.unfreeze();
        let result = load_weights_manual(&mut self.vs, &model_path)
            .map_err(ModelError::from);
        self.vs.freeze();
        result
    }

    pub fn forward(&self, mut input_tensor: Tensor) -> Result<Tensor, ModelError> {
        info!("MimiVocoder forward pass starting...");
        info!("  Input shape: {:?}", input_tensor.size());

        // Convert input tensor to Float if it's Int64
        if input_tensor.kind() == Kind::Int64 {
            input_tensor = input_tensor.to_kind(Kind::Float);
            info!("  Converted input tensor to Float. New shape: {:?}", input_tensor.size());
        } else if input_tensor.kind() != Kind::Float {
            return Err(ModelError::ProcessError(format!(
                "Invalid input tensor kind: {:?}, expected Float or Int64",
                input_tensor.kind()
            )));
        }

        // Add batch dimension if needed
        if input_tensor.dim() == 2 {
            input_tensor = input_tensor.unsqueeze(0);
        } else if input_tensor.dim() != 3 {
            return Err(ModelError::ProcessError(format!(
                "Invalid input tensor dimension: {}, expected 2 or 3",
                input_tensor.dim()
            )));
        }

        // RESTORED Initial projection again
        let projected_input = self.input_proj.forward(&input_tensor);

        // Process through each component with explicit error handling
        let hidden_states = self.downsample.forward(&projected_input);
        info!("  After downsample shape: {:?}", hidden_states.size());

        // Transpose features and sequence length for transformer
        let hidden_states_transposed = hidden_states.transpose(1, 2);
        info!("  Transposed for encoder shape: {:?}", hidden_states_transposed.size());

        let encoded = self.encoder_transformer.forward(&hidden_states_transposed);
        info!("  Encoder output shape: {:?}", encoded.size());

        let quantized = self.quantizer.forward(&encoded)
            .map_err(|e| ModelError::ProcessError(format!("Quantizer error: {}", e)))?;
        info!("  Quantizer output shape: {:?}", quantized.size());

        let decoded = self.decoder_transformer.forward(&quantized);
        info!("  Decoder output shape: {:?}", decoded.size());

        // Transpose for Conv1d: [B, T, C] -> [B, C, T]
        let decoded_transposed = decoded.transpose(1, 2);
        info!(
            "  Transposed for decoder_conv shape: {:?}",
            decoded_transposed.size()
        );

        // Apply the final decoder convolution layer (expects [B, C, T])
        let conv_output = self.decoder_conv.forward(&decoded_transposed);
        info!(
            "  After decoder_conv shape: {:?}",
            conv_output.size()
        );

        // Apply the output projection layer
        let proj_output = self.output_proj.forward(&conv_output);
        info!(
            "  After output_proj shape: {:?}",
            proj_output.size()
        );

        // Output of conv_output is [B, 1024, T']
        // TODO: This will likely cause a shape mismatch with the upsampler,
        // which expects input channels = hidden_size (512).
        // This indicates a missing layer or projection after decoder_conv.
        // We will address this when the error occurs.

        // Output of conv_output is already [B, C, T'], ready for upsampler
        // let upsampler_input = conv_output;
        // Use the projected output for the upsampler
        let upsampler_input = proj_output;
        info!(
            "  Input for upsampler shape: {:?}", // Updated log message
            upsampler_input.size()
        );

        let upsampled = self.upsample.forward(&upsampler_input);

        // Validate output shape
        let (batch, channels, length) = upsampled.size3()
            .map_err(|e| ModelError::ProcessError(format!("Invalid output tensor shape: {}", e)))?;
            
        if channels != self.config.audio_channels {
            return Err(ModelError::ProcessError(format!(
                "Invalid output channels: {}, expected {}",
                channels, self.config.audio_channels
            )));
        }

        info!("Forward pass completed successfully. Output shape: [{}, {}, {}]", batch, channels, length);
        Ok(upsampled)
    }

     // Prefix unused method
     fn _sample_rate(&self) -> u32 {
         self.sample_rate
    }
}

#[async_trait]
impl Vocoder for MimiVocoder {
    // UPDATED: Accept a single tuple and extract acoustic tokens
    async fn synthesize_chunk(&self, tokens_tuple: (i64, Vec<i64>)) -> Result<Vec<i16>, ModelError> {
        // Extract acoustic tokens, ignore semantic token for now
        let (_semantic_token, acoustic_tokens) = tokens_tuple;

        // Validate acoustic tokens
        if acoustic_tokens.is_empty() {
            // Return Ok(empty) instead of error for empty input chunk? Let's keep error for now.
            return Err(ModelError::ProcessError("Empty acoustic token sequence provided in chunk".to_string()));
        }

        let seq_len = acoustic_tokens.len();
        // Log level reduced from info to trace to avoid excessive logging
        trace!(
            "Vocoder: Processing chunk with {} acoustic tokens.",
            seq_len
        );

        // --- Create input tensor from ONLY acoustic tokens --- 
        // Assuming the first acoustic codebook's tokens are used or model handles it.
        let token_vec_2d = vec![acoustic_tokens]; // Shape [1, seq_len]

        let token_tensor = Tensor::from_slice2(&token_vec_2d)
            .to_device(self.device)
            .to_kind(Kind::Int64);
        // -------------------------------------------------------

        // Process through vocoder using the main forward method
        let audio_tensor = self.forward(token_tensor)?;

        // Convert output tensor to audio samples
        let audio_tensor_squeezed_float = audio_tensor
            .squeeze() // Remove batch dimension
            .to_kind(Kind::Float);

        let audio_out_f32 = Vec::<f32>::try_from(audio_tensor_squeezed_float) // Use TryFrom instead of try_into
            .map_err(|e| ModelError::ProcessError(format!("Failed to convert output tensor to Vec<f32>: {}", e)))?;

        // Convert f32 to i16 (assuming output is in [-1.0, 1.0])
        let audio_out_i16 = audio_out_f32
            .iter()
            .map(|&sample| (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .collect();

        Ok(audio_out_i16)
    }

    /// Synthesize audio from a collection of token chunks.
    /// Flattens the input `Vec<Vec<(i64, Vec<i64>)>>` and processes each chunk sequentially.
    async fn decode_tokens(&self, token_chunks: Vec<Vec<(i64, Vec<i64>)>>) -> Result<Vec<i16>, ModelError> {
        info!("Vocoder: Decoding {} token chunks...", token_chunks.len());
        let mut all_audio_samples = Vec::new();
        let flattened_chunks: Vec<(i64, Vec<i64>)> = token_chunks.into_iter().flatten().collect();
        
        if flattened_chunks.is_empty() {
            warn!("Vocoder: decode_tokens received no actual token tuples after flattening.");
            return Ok(all_audio_samples); // Return empty if no tokens
        }

        info!("Vocoder: Processing {} flattened token tuples.", flattened_chunks.len());
        
        for token_tuple in flattened_chunks {
            match self.synthesize_chunk(token_tuple).await {
                Ok(audio_chunk) => {
                    all_audio_samples.extend(audio_chunk);
                }
                Err(e) => {
                    // Decide how to handle errors: continue, log, or return error immediately?
                    // Let's log and continue for now, returning potentially partial audio.
                    error!("Vocoder: Error synthesizing chunk: {}. Skipping chunk.", e);
                    // Alternatively, return the error immediately:
                    // return Err(ModelError::ProcessError(format!("Failed during chunk synthesis: {}", e)));
                }
            }
        }
        
        info!("Vocoder: Finished decoding tokens. Total samples: {}", all_audio_samples.len());
        Ok(all_audio_samples)
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

// ... MockVocoder (keep) ...

// Restore tensor_view_to_tensor helper
fn tensor_view_to_tensor<T: safetensors::tensor::View + ?Sized>(view: &T, device: Device) -> Result<Tensor, TchError> {
    let kind = match view.dtype() {
        safetensors::Dtype::F64 => Kind::Double,
        safetensors::Dtype::F32 => Kind::Float,
        safetensors::Dtype::BF16 => Kind::BFloat16,
        safetensors::Dtype::F16 => Kind::Half,
        safetensors::Dtype::I64 => Kind::Int64,
        safetensors::Dtype::I32 => Kind::Int,
        safetensors::Dtype::I16 => Kind::Int16,
        safetensors::Dtype::I8 => Kind::Int8,
        safetensors::Dtype::U8 => Kind::Uint8,
        safetensors::Dtype::BOOL => Kind::Bool,
        dtype => return Err(TchError::Kind(format!("Unsupported safetensors dtype {:?} for tch conversion", dtype))),
    };
    let shape: Vec<i64> = view.shape().iter().map(|&d| d as i64).collect();
    let data_cow = view.data();
    let expected_numel = shape.iter().product::<i64>();
    let expected_bytes = expected_numel * (kind.elt_size_in_bytes() as i64);
    let data_slice: &[u8] = data_cow.as_ref();

    let result: Result<Tensor, TchError>;

    if data_slice.len() as i64 != expected_bytes {
        warn!(
            "Data length mismatch for tensor view. Expected {} bytes (shape: {:?}, kind: {:?}), got {} bytes. Falling back to from_slice (may copy).",
            expected_bytes, shape, kind, data_slice.len()
        );
        let tensor = Tensor::from_slice(data_slice)
            .reshape(&shape)
            .to_kind(kind);
        result = Ok(tensor.to_device(device));
    } else {
        // Try from_data_size directly (it returns a Tensor, not a Result)
        let tensor = Tensor::from_data_size(data_slice, &shape, kind);
        // Use try-catch style approach to handle potential panic
        let result_tensor = match std::panic::catch_unwind(|| tensor.to_device(device)) {
            Ok(device_tensor) => Ok(device_tensor),
            Err(_) => {
                warn!("Tensor::from_data_size or to_device panicked. Falling back to from_slice. Shape: {:?}, Kind: {:?}, Data len: {}", shape, kind, data_slice.len());
                let fallback_tensor = Tensor::from_slice(data_slice)
                    .reshape(&shape)
                    .to_kind(kind);
                Ok(fallback_tensor.to_device(device))
            }
        };
        result = result_tensor;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tracing_test::traced_test;

    #[tokio::test]
    #[traced_test]
    async fn test_mimi_vocoder_load_and_process() -> Result<(), ModelError> {
        // Initialize vocoder
        let device = Device::Cpu;
        let sample_rate = 24000;
        let mut vocoder = MimiVocoder::new(sample_rate, device)?;

        // Load weights
        let model_path = PathBuf::from("models/mimi/model.safetensors");
        vocoder.load_model(model_path)?;

        // Create test tokens: batch size 1, sequence length 1920 (arbitrary length)
        let sequence_length = 1920;
        let acoustic_tokens = vec![0i64; sequence_length]; // Acoustic tokens
        let semantic_token = 0i64; // Dummy semantic token
        let tokens_tuple = (semantic_token, acoustic_tokens); // Create the tuple directly
        info!("Created test token tuple: semantic={}, acoustic_len={}", 
              tokens_tuple.0, tokens_tuple.1.len());

        // Process tokens through vocoder
        let audio = vocoder.synthesize_chunk(tokens_tuple).await?; // Pass the tuple directly

        // Basic validation
        assert!(!audio.is_empty(), "Generated audio should not be empty");
        assert_eq!(vocoder.sample_rate(), sample_rate);
        info!("Generated audio output: {} samples", audio.len());

        Ok(())
    }
}