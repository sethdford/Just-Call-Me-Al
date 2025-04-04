use std::path::Path;
use std::sync::Arc;
use std::collections::HashMap;
use std::fs::File;

// External crates
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use memmap2::MmapOptions;
use serde::Deserialize;
use tracing::{debug, error, info, warn};

// tch related
use tch::{Device, Tensor, Kind, TchError, nn};
use tch::nn::Module;

// tokio related
use tokio::sync::{mpsc, Mutex as TokioMutex};

// Internal imports
use crate::models::{CSMModel, ModelError};
use crate::models::config::CsmModelConfig;
use crate::vocoder::{Vocoder, MimiVocoder};
use crate::llm_integration::{LlmProcessor, ContextEmbedding, LlmConfig, create_llm_service};
use crate::context::ConversationHistory;
use crate::tokenization::{Tokenizer, LlamaTokenizer};
use safetensors::{SafeTensors, tensor::TensorView};

const EOS_TOKEN_ID: i64 = 2; // Assuming standard EOS token ID for SentencePiece/Llama

fn sample_topk(logits: &Tensor, temperature: f64, top_k: i64) -> Result<Tensor, TchError> {
    let _guard = tch::no_grad_guard(); // Ensure no gradients are computed

    // --- Input Validation ---
    if !logits.isfinite().all().int64_value(&[]) == 1 {
        error!("sample_topk: Input logits contain NaN/Inf. Shape: {:?}", logits.size());
        // Potentially return error or handle differently
        return Err(TchError::Kind("Input logits are not finite.".to_string()));
    }

    if temperature <= 0.0 {
        error!("sample_topk: Temperature must be positive, got {}", temperature);
        return Err(TchError::Kind("Temperature must be positive.".to_string()));
    }

    if top_k <= 0 {
        error!("sample_topk: top_k must be positive, got {}", top_k);
        return Err(TchError::Kind("top_k must be positive.".to_string()));
    }
    // -----------------------

    // Scale logits by temperature
    let scaled_logits = logits / temperature;
    
    // Check for NaN/Inf after temperature scaling but before Top-K
    // Note: Dividing by a small temperature can cause large values or Inf
    if !scaled_logits.isfinite().all().int64_value(&[]) == 1 {
        warn!(
            "sample_topk: Non-finite logits AFTER temperature scaling (Temp: {}). Shape: {:?}. Top-K might resolve this.", 
            temperature, scaled_logits.size()
        );
        // If Inf exists, Top-K should select it. If NaN, it might cause issues downstream.
    }

    // --- Clamp scaled_logits BEFORE topk to prevent Inf/NaN selection --- 
    let clamped_scaled_logits = scaled_logits.clamp(-f32::MAX as f64, 80.0);
    // --------------------------------------------------------------------

    // Top-k selection
    // Ensure top_k isn't larger than the available vocabulary size
    let vocab_size = clamped_scaled_logits.size().last().copied().unwrap_or(0); // Use clamped logits size
    let actual_top_k = top_k.min(vocab_size);
    if actual_top_k <= 0 {
        return Err(TchError::Kind("Cannot select top-k=0.".to_string()));
    }

    // Use clamped logits for topk
    let (top_p, _top_i) = clamped_scaled_logits.topk(actual_top_k, -1, true, true); 

    // Compute softmax probabilities with numerical stability
    let max_logits = top_p.max_dim(-1, true).0; // Find max along the last dimension
    let shifted = top_p - &max_logits; // Subtract max for stability

    // Clamp shifted logits before exponentiation to prevent overflow to Inf
    // A value like 80.0 is safe as exp(80) is large but representable
    let clamped_shifted = shifted.clamp(-f32::MAX as f64, 80.0);

    let exp_logits = clamped_shifted.exp(); // Use clamped values
    // Specify dimension type for sum_dim_intlist using slice notation
    let sum_exp = exp_logits.sum_dim_intlist(&[-1i64][..], true, Kind::Float);

    // Clone sum_exp for potential logging in the error case
    let sum_exp_for_logging = sum_exp.shallow_clone();

    // --- Log before division --- 
    let exp_logits_min: f64 = exp_logits.min().double_value(&[]);
    let exp_logits_max: f64 = exp_logits.max().double_value(&[]);
    let exp_logits_finite = exp_logits.isfinite().all().int64_value(&[]) == 1;
    let sum_exp_val: f64 = sum_exp.double_value(&[]); 
    let sum_exp_finite = sum_exp.isfinite().all().int64_value(&[]) == 1;
    info!(
        "sample_topk: BEFORE division - ExpLogits(Min: {:.6e}, Max: {:.6e}, Finite: {}), SumExp(Val: {:.6e}, Finite: {})",
        exp_logits_min, exp_logits_max, exp_logits_finite, sum_exp_val, sum_exp_finite
    );
    // --------------------------

    // Calculate probabilities
    let probs = exp_logits / (sum_exp + 1e-9); // Add epsilon for stability if sum_exp is near zero

    // --- Replace NaN with 0.0 BEFORE clamping --- 
    let probs = probs.nan_to_num(0.0, f64::INFINITY, f64::NEG_INFINITY); // Remove ?
    // -------------------------------------------

    // Check for finite probabilities immediately after calculation
    if !probs.isfinite().all().int64_value(&[]) == 1 {
        error!(
            "sample_topk: Non-finite probabilities AFTER softmax calculation (before clamp/renorm). Sum_Exp: {:?}",
            // Use the cloned tensor for logging
            sum_exp_for_logging.squeeze().size() 
        );
        // Extract scalar values using .double_value()
        let shifted_min: f64 = shifted.min().double_value(&[]);
        let shifted_max: f64 = shifted.max().double_value(&[]);
        error!(
            " -> Shifted min: {:.6e}, max: {:.6e}", 
            shifted_min, shifted_max
        );
        return Err(TchError::Kind("Probabilities became non-finite during softmax.".to_string()));
    }

    // Clamp probabilities to avoid small negative values from floating point errors
    // Although theoretically probs should be [0, 1], clamp for robustness.
    let probs = probs.clamp(0.0, 1.0);

    // Renormalize probabilities to ensure they sum to 1, especially after clamping
    let sum_probs = probs.sum_dim_intlist(&[-1i64][..], true, Kind::Float);
    let clamped_sum_probs = sum_probs.clamp_min(1e-9); // <-- Restore clamping
    
    // Log the sum before division
    let sum_val_before_div: f64 = clamped_sum_probs.double_value(&[]); // <-- Log the clamped value
    info!("sample_topk: Sum of probabilities before final division (clamped): {:.6e}", sum_val_before_div);

    // Perform the final renormalization using the clamped sum
    let probs = probs / clamped_sum_probs; // <-- Use clamped value for division

    // Final check and logging *AFTER* renormalization, *BEFORE* multinomial
    let final_check_finite = probs.isfinite().all().int64_value(&[]) == 1;
    let final_min_prob: f64 = probs.min().double_value(&[]);
    let final_max_prob: f64 = probs.max().double_value(&[]);
    let final_sum_prob: f64 = probs.sum(Kind::Float).double_value(&[]);
    info!( 
        "sample_topk: FINAL check before multinomial - Shape: {:?}, Min: {:.6e}, Max: {:.6e}, Sum: {:.6e}, IsFinite: {}",
        probs.size(), final_min_prob, final_max_prob, final_sum_prob, final_check_finite
    );

    // Check explicitly again if finite before calling multinomial
    if !final_check_finite {
        error!("sample_topk: Probabilities became non-finite AFTER final renormalization. Aborting before multinomial.");
        return Err(TchError::Kind("Probabilities non-finite after final renormalization.".to_string()));
    }

    // Additional safeguards before multinomial
    // 1. Clamp again to ensure strictly positive values (avoid zeros that can cause issues)
    let probs = probs.clamp(1e-10, 1.0);
    
    // 2. Re-normalize to ensure sum is exactly 1.0
    let sum_probs = probs.sum_dim_intlist(&[-1i64][..], true, Kind::Float);
    let probs = probs / sum_probs;
    
    // 3. Final validation - verify shape, values and sum to catch any issues
    let final_min_prob: f64 = probs.min().double_value(&[]);
    let final_max_prob: f64 = probs.max().double_value(&[]);
    let final_sum_prob: f64 = probs.sum(Kind::Float).double_value(&[]);
    let final_is_finite = probs.isfinite().all().int64_value(&[]) == 1;
    
    info!(
        "sample_topk: FINAL validation before multinomial - Shape: {:?}, Min: {:.6e}, Max: {:.6e}, Sum: {:.6e}, IsFinite: {}",
        probs.size(), final_min_prob, final_max_prob, final_sum_prob, final_is_finite
    );
    
    // 4. Explicit check for invalid values to provide a clearer error message
    if !final_is_finite || final_min_prob < 0.0 || final_sum_prob == 0.0 {
        error!("sample_topk: Invalid probability tensor before multinomial: min={:.6e}, max={:.6e}, sum={:.6e}, is_finite={}",
               final_min_prob, final_max_prob, final_sum_prob, final_is_finite);
        return Err(TchError::Kind("Invalid probability tensor for multinomial sampling".to_string()));
    }

    // Convert to float kind and sample
    let probs_float = probs.to_kind(Kind::Float);
    
    // NOTE: Apparently multinomial actually returns a Tensor directly, not a Result<Tensor, TchError> 
    // as our original code assumed (despite the method actually failing in some cases with NaN values).
    // We need to wrap it in Ok() to match the function's return type.
    Ok(probs_float.multinomial(1, false))
}

fn create_causal_mask(seq_len: i64, device: Device) -> Result<Tensor, TchError> {
    let mask = Tensor::ones(&[seq_len, seq_len], (Kind::Bool, device))
        .tril(0); // Lower triangular matrix
    // Result shape: [seq_len, seq_len]
    Ok(mask)
}

// Define ConversationContext (Placeholder - replace with actual definition if it exists elsewhere)
#[derive(Debug, Clone, Default)]
pub struct ConversationContext {
    messages: Vec<String>, // Example field
}

impl ConversationContext {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add_message(&mut self, message: String) {
        self.messages.push(message);
    }
    pub fn clear(&mut self) {
        self.messages.clear();
    }
}

// --- Cache Struct Definitions ---

#[derive(Default, Debug)]
struct AttentionCache {
    kv: Option<(Tensor, Tensor)>,
}

// Manual Clone implementation for AttentionCache
impl Clone for AttentionCache {
    fn clone(&self) -> Self {
        Self {
            kv: self.kv.as_ref().map(|(k, v)| (k.copy(), v.copy()))
        }
    }
}

#[derive(Default, Debug, Clone)]
struct TransformerCache {
    layer_caches: Vec<AttentionCache>,
}

impl TransformerCache {
    fn new(num_layers: usize) -> Self {
        Self { layer_caches: vec![AttentionCache::default(); num_layers] }
    }
    fn reset(&mut self) {
        for cache in &mut self.layer_caches {
            cache.kv = None;
        }
    }
}

// This holds the state for a *single* ongoing synthesis stream
#[derive(Debug)]
pub struct StreamingState {
    backbone_cache: TransformerCache,
    decoder_cache: TransformerCache,
}

impl StreamingState {
    pub fn new(config: &CsmModelConfig) -> Self {
        info!("Creating new StreamingState for Backbone ({} layers) and Decoder ({} layers)", config.backbone_num_layers, config.decoder_num_layers);
        Self {
            backbone_cache: TransformerCache::new(config.backbone_num_layers as usize),
            decoder_cache: TransformerCache::new(config.decoder_num_layers as usize),
        }
    }
    pub fn reset(&mut self) {
        warn!("Resetting StreamingState caches.");
        self.backbone_cache.reset();
        self.decoder_cache.reset();
    }
}

// --- End Cache Struct Definitions ---

// --- Concrete Implementation Wrapper ---
#[derive(Debug, Clone)] // CSMImpl now holds Arc<TokioMutex<RustCsmModel>> which is Clone
pub struct CSMImpl {
    rust_model: Arc<TokioMutex<RustCsmModel>>, // Use Arc<Mutex<...>>
}

impl CSMImpl {
    // Original new function (kept for compatibility or internal use if needed)
    pub fn new(model_dir: &Path, device: Device) -> Result<Self, ModelError> {
        info!("Initializing CSMImpl Wrapper (basic) for model dir: {:?} and device: {:?}", model_dir, device);
        // Create default LLM service if none supplied
        let default_llm_config = LlmConfig::default();
        let default_llm_processor = create_llm_service(default_llm_config)
            .map_err(|e| ModelError::Other(anyhow!("Failed to create default LLM service: {}", e)))?;
        let rust_model_instance = RustCsmModel::new(model_dir, device, None, default_llm_processor)?;

        // Wrap in Mutex then Arc
        let model_mutex = TokioMutex::new(rust_model_instance);
        let model_arc = Arc::new(model_mutex);

        Ok(Self { rust_model: model_arc })
    }
    
    // New function to accept an LLM processor
    pub fn new_with_processor(
        model_dir: &Path, 
        device: Device, 
        llm_processor: Arc<dyn LlmProcessor>
    ) -> Result<Self, ModelError> {
        info!("Initializing CSMImpl Wrapper with LLM Processor for model dir: {:?} and device: {:?}", model_dir, device);
        let rust_model_instance = RustCsmModel::new(model_dir, device, None, llm_processor)?;

        // Wrap in Mutex then Arc
        let model_mutex = TokioMutex::new(rust_model_instance);
        let model_arc = Arc::new(model_mutex);

        Ok(Self { rust_model: model_arc })
    }
}

// --- CSMModel Trait Implementation for CSMImpl ---
#[async_trait]
impl CSMModel for CSMImpl {
    // Correct signature: Match the trait (remove conversation_history)
    async fn synthesize(
        &self,
        text: &str,
        // conversation_history: Option<&ConversationHistory>, // REMOVED
        temperature: Option<f64>,
        top_k: Option<i64>,
        seed: Option<u64>,
    ) -> Result<Vec<i16>, ModelError> {
        let mut model_guard = self.rust_model.lock().await; // Acquire lock
        // Call the internal method - must be updated to match
        // For now, assume internal method is also updated (or call a different one)
        // We need to decide how non-streaming synthesis should handle history.
        // Option 1: Error if history is implicitly needed but not provided.
        // Option 2: Call internal method without history (None).
        // Let's go with Option 2 for now, but mark it for review.
        warn!("Non-streaming synthesize called; conversation history is ignored in this path.");
        model_guard.synthesize(text, None, temperature, top_k, seed).await // Pass None for history
    }

    // Correct signature: Match the trait (remove conversation_history)
    #[allow(clippy::future_not_send)]
    async fn synthesize_streaming(
        &self,
        text: &str,
        // conversation_history: Option<&ConversationHistory>, // REMOVED
        temperature: Option<f64>,
        top_k: Option<i64>,
        seed: Option<u64>,
        audio_token_tx: mpsc::Sender<Vec<(i64, Vec<i64>)>>, // Match trait signature
    ) -> Result<(), ModelError> {
        let mut model_guard = self.rust_model.lock().await; // Acquire lock
        // Call the internal method - this one needs history, where does it come from?
        // The trait doesn't provide it. This indicates a design mismatch.
        // For now, let's pass None, but this needs resolution.
        // TODO: Resolve how streaming synthesis gets conversation history via trait.
        warn!("Streaming synthesize called via trait; conversation history is currently unavailable in this path.");
        model_guard.synthesize_streaming(text, None, temperature, top_k, seed, audio_token_tx).await // Pass None for history
    }
}

// --- Utility functions for loading weights ---

fn tensor_view_to_tensor(view: &TensorView<'_>, device: Device) -> Result<Tensor, TchError> {
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
    let data = view.data();
    let shape: Vec<i64> = view.shape().iter().map(|&d| d as i64).collect();

    // Create tensor first, then move to device.
    // Use from_data_size for efficiency, requires matching data length.
    let tensor = Tensor::from_data_size(data, &shape, kind);
    Ok(tensor.to_device(device))
}

fn load_weights_safe(vs: &mut nn::VarStore, tensors: &SafeTensors, device: Device) -> Result<(), TchError> {
    info!("Loading weights safely from SafeTensors...");
    let mut loaded_tensors_map: HashMap<String, Tensor> = HashMap::new();

    for name in tensors.names() {
        match tensors.tensor(name) {
             Ok(tensor_view) => {
        match tensor_view_to_tensor(&tensor_view, device) {
            Ok(tensor) => { 
                loaded_tensors_map.insert(name.to_string(), tensor);
                     },
                     Err(e) => warn!("Failed to convert tensor view '{}' to tch::Tensor: {}", name, e),
            }
             }
             Err(e) => warn!("Failed to get tensor view '{}' from SafeTensors: {}", name, e),
        }
    }
    info!("Loaded {} tensors into memory map.", loaded_tensors_map.len());

    let mut loaded_count = 0;
    let mut skipped_count = 0;
    let mut missing_count = 0;
    let variables_in_vs = vs.variables(); 
    info!("Attempting to copy weights into {} VarStore variables...", variables_in_vs.len());

    for (var_path, mut var) in variables_in_vs {
        let safetensors_key = var_path.replace('/', ".");
        if let Some(loaded_tensor) = loaded_tensors_map.get(&safetensors_key) {
            if var.size() == loaded_tensor.size() {
                tch::no_grad(|| { var.copy_(loaded_tensor); });
                loaded_count += 1;
            } else {
                warn!("Shape mismatch for '{}': VarStore={:?}, SafeTensor={:?}. Skipping.",
                       safetensors_key, var.size(), loaded_tensor.size());
                skipped_count += 1;
            }
        } else {
            warn!("Variable '{}' defined in VarStore, but no corresponding key '{}' found in SafeTensors file.",
                   var_path, safetensors_key);
            missing_count += 1;
        }
    }

    info!("Weight loading finished. Copied: {}, Shape Mismatch (Skipped): {}, Missing in File: {}",
            loaded_count, skipped_count, missing_count);

    if loaded_count == 0 && !loaded_tensors_map.is_empty() {
        return Err(TchError::FileFormat("Failed to load any weights.".to_string()));
    }

    Ok(())
}

// --- Start Layer Definitions (Restored) ---

#[derive(Debug)]
struct RmsNorm {
    eps: f64,
}

impl RmsNorm {
    fn new(_vs: nn::Path, _dim: i64, eps: f64) -> Result<Self, TchError> {
        Ok(Self { eps })
    }

    fn forward(&self, xs: &Tensor, gamma: Option<&Tensor>, beta: Option<&Tensor>) -> Result<Tensor, TchError> {
        let variance = xs.to_kind(Kind::Float).pow_tensor_scalar(2.0).mean_dim(&[-1i64][..], true, Kind::Float);
        let hidden_states_norm = xs.to_kind(Kind::Float) * (&variance + self.eps).rsqrt();
        
        // Apply external gamma and beta if provided
        let result = match (gamma, beta) {
            (Some(g), Some(b)) => {
                // Apply gamma and beta with proper error handling
                let scaled = hidden_states_norm.f_mul(g)?;
                scaled.f_add(b)?
            },
            (Some(g), None) => {
                // Just apply gamma
                hidden_states_norm.f_mul(g)?
            },
            (None, Some(b)) => {
                // Just apply beta
                hidden_states_norm.f_add(b)?
            },
            (None, None) => {
                // Use normalized states as-is
                hidden_states_norm
            }
        };
        
        // Cast back to original input type
        Ok(result)
    }
}

#[derive(Debug)]
struct RotaryEmbedding {
    dim: i64,
    base: f64,
    inv_freq: Tensor,
    device: Device,
    max_seq_len_cached: i64,
    cos_cached: Option<Tensor>,
    sin_cached: Option<Tensor>,
}

impl Clone for RotaryEmbedding {
    fn clone(&self) -> Self {
        Self {
            dim: self.dim,
            base: self.base,
            inv_freq: self.inv_freq.shallow_clone(),
            device: self.device, // Assuming Device is copyable or cloneable
            max_seq_len_cached: self.max_seq_len_cached,
            cos_cached: self.cos_cached.as_ref().map(|t| t.shallow_clone()),
            sin_cached: self.sin_cached.as_ref().map(|t| t.shallow_clone()),
        }
    }
}

impl RotaryEmbedding {
    fn new(
        dim: i64,
        max_position_embeddings: i64,
        base: f64,
        device: Device,
    ) -> Result<Self, TchError> {
        let inv_freq = (0..(dim + 1) / 2)
            .map(|i| 1f64 / base.powf(i as f64 * 2.0 / dim as f64))
            .collect::<Vec<_>>();
        let inv_freq = Tensor::from_slice(&inv_freq).to_kind(Kind::Float).to_device(device);
        // Ensure inv_freq has no grad if it does by default
        let inv_freq = inv_freq.set_requires_grad(false); // Removed ?
        Ok(Self {
            dim,
            base,
            inv_freq,
            device, // Store device
            // Initialize cached fields
            max_seq_len_cached: max_position_embeddings,
            cos_cached: None,
            sin_cached: None,
        })
    }

    fn cache_sincos(&mut self, seq_len: i64) -> Result<(), TchError> {
        if self.cos_cached.is_some() && seq_len <= self.max_seq_len_cached {
            return Ok(());
        }

        let t = Tensor::arange(seq_len, (Kind::Float, self.inv_freq.device()));
        let freqs = t.outer(&self.inv_freq);
        let emb = Tensor::cat(&[&freqs, &freqs], -1);
        self.cos_cached = Some(emb.cos());
        self.sin_cached = Some(emb.sin());
        self.max_seq_len_cached = seq_len; // Update cached length
        Ok(())
    }

    fn forward(&mut self, x: &Tensor, seq_len_offset: usize) -> Result<(Tensor, Tensor), TchError> {
        let (_b_sz, _num_heads, seq_len, _head_dim) = x.size4()?;
        let current_seq_len = seq_len_offset + seq_len as usize;

        // Ensure cache is populated up to current_seq_len
        self.cache_sincos(current_seq_len as i64)?;

        let cos = self.cos_cached.as_ref().unwrap().narrow(0, seq_len_offset as i64, seq_len);
        let sin = self.sin_cached.as_ref().unwrap().narrow(0, seq_len_offset as i64, seq_len);

        // Adjust shapes for broadcasting
        let cos = cos.unsqueeze(0).unsqueeze(0); // [1, 1, seq_len, head_dim]
        let sin = sin.unsqueeze(0).unsqueeze(0); // [1, 1, seq_len, head_dim]

        Ok((cos, sin))
    }
}

fn apply_rotary_pos_emb(q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor), TchError> {
    let q_embed = (q * cos) + rotate_half(q)? * sin;
    let k_embed = (k * cos) + rotate_half(k)? * sin;
    Ok((q_embed, k_embed))
}

fn rotate_half(x: &Tensor) -> Result<Tensor, TchError> {
    // Get the last dimension size
    let size = x.size();
    let last_dim = size.last().unwrap_or(&0);
    
    // Slice along the last dimension (-1)
    let x1 = x.slice(-1, 0, size[size.len() - 1] / 2, 1);
    let x2 = x.slice(-1, size[size.len() - 1] / 2, size[size.len() - 1], 1);
    
    // Concatenate along the last dimension
    Ok(Tensor::cat(&[-x2, x1], -1))
}

#[derive(Debug)]
struct Attention {
    wq: nn::Linear,
    wk: nn::Linear,
    wv: nn::Linear,
    wo: nn::Linear,
    n_head: i64,
    head_dim: i64,
    rotary_emb: RotaryEmbedding,
    #[allow(dead_code)] // May not be used if scale isn't applied
    scale: f64,
}

impl Attention {
    fn new(vs: nn::Path, config: &CsmModelConfig, rotary_emb: RotaryEmbedding, is_backbone: bool) -> Result<Self, TchError> {
        let (embed_dim, n_head, head_dim) = if is_backbone {
            (config.backbone_embed_dim, config.backbone_num_heads, config.backbone_embed_dim / config.backbone_num_heads)
        } else {
            (config.decoder_embed_dim, config.decoder_num_heads, config.decoder_embed_dim / config.decoder_num_heads)
        };

        let wq = nn::linear(&vs / "q_proj", embed_dim, n_head * head_dim, Default::default());
        let wk = nn::linear(&vs / "k_proj", embed_dim, n_head * head_dim, Default::default());
        let wv = nn::linear(&vs / "v_proj", embed_dim, n_head * head_dim, Default::default());
        let wo = nn::linear(&vs / "o_proj", n_head * head_dim, embed_dim, Default::default());
        
        let scale = 1.0 / (head_dim as f64).sqrt();

        Ok(Self { wq, wk, wv, wo, n_head, head_dim, rotary_emb, scale })
    }

    fn forward(&mut self, xs: &Tensor, start_pos: usize, mask: Option<&Tensor>, cache: &mut AttentionCache) -> Result<Tensor, TchError> {
        let (b_sz, seq_len, _embed_dim) = xs.size3()?;

        // Project input to Q, K, V
        let q = self.wq.forward(xs); 
        let k = self.wk.forward(xs); 
        let v = self.wv.forward(xs);
        
        // Reshape and transpose in one step using view
        let q = q.view([b_sz, seq_len, self.n_head, self.head_dim]).transpose(1, 2); // [B, N, S, H]
        let k = k.view([b_sz, seq_len, self.n_head, self.head_dim]).transpose(1, 2); // [B, N, S, H]
        let v = v.view([b_sz, seq_len, self.n_head, self.head_dim]).transpose(1, 2); // [B, N, S, H]

        // Apply rotary embeddings
        let (cos, sin) = self.rotary_emb.forward(&q, start_pos)?;
        let (q, k) = apply_rotary_pos_emb(&q, &k, &cos, &sin)?;

        // Update cache
        let (k, v) = match &cache.kv {
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &k], 2); // Concat along seq_len dimension
                let v = Tensor::cat(&[prev_v, &v], 2);
                (k, v)
            }
            None => (k, v),
        };
        cache.kv = Some((k.copy(), v.copy())); // Update cache AFTER concatenation
        
        let k = cache.kv.as_ref().unwrap().0.shallow_clone();
        let v = cache.kv.as_ref().unwrap().1.shallow_clone();

        // Compute attention scores with proper dimensions
        let att_scores = q.matmul(&k.transpose(-2, -1)) / (self.head_dim as f64).sqrt();

        // Apply mask if provided
        let att_scores = match mask {
            Some(m) => {
                // Get the sequence length of K/V from the last dimension of attention scores
                let (_b, _n, _sq, s_kv) = att_scores.size4()?;
                
                // Adjust mask to match attention scores shape
                // First unsqueeze to add batch and head dimensions if needed
                let mask_adjusted = if m.dim() == 2 {
                    m.unsqueeze(0).unsqueeze(0) // Add batch and head dims
                } else if m.dim() == 3 {
                    m.unsqueeze(1) // Add head dim only
                } else {
                    m.shallow_clone() // Already 4D
                };
                
                // Now select the relevant time steps
                let mask_adjusted = mask_adjusted.narrow(2, start_pos as i64, seq_len)
                                               .narrow(3, 0, s_kv);
                
                // Apply mask
                att_scores.masked_fill(&mask_adjusted.logical_not(), f64::NEG_INFINITY)
            }
            None => att_scores
        };

        // Apply softmax and compute weighted sum
        let probs = att_scores.softmax(-1, Kind::Float);
        let output = probs.matmul(&v); // [B, N, S, H]

        // Reshape back to [batch_size, seq_len, embed_dim]
        let output = output.transpose(1, 2).contiguous().view([b_sz, seq_len, -1]); // [B, S, N*H]
        
        // Final projection
        Ok(self.wo.forward(&output))
    }
}

#[derive(Debug)]
struct FeedForward {
    w1: nn::Linear,
    w2: nn::Linear,
    w3: nn::Linear,
}

impl FeedForward {
    fn new(vs: nn::Path, config: &CsmModelConfig, is_backbone: bool) -> Result<Self, TchError> {
        let (embed_dim, hidden_dim) = if is_backbone {
            (config.backbone_embed_dim, config.backbone_intermediate_dim)
        } else {
            (config.decoder_embed_dim, config.decoder_intermediate_dim)
        };
        let w1 = nn::linear(&vs / "w1", embed_dim, hidden_dim, Default::default());
        let w2 = nn::linear(&vs / "w2", hidden_dim, embed_dim, Default::default());
        let w3 = nn::linear(&vs / "w3", embed_dim, hidden_dim, Default::default());
        Ok(Self { w1, w2, w3 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor, TchError> {
        let swish = xs.apply(&self.w1).silu();
        let linear = xs.apply(&self.w3);
        let combined = swish * linear;
        Ok(combined.apply(&self.w2))
    }
}

#[derive(Debug)]
struct TransformerBlock {
    attn: Attention,
    ffn: FeedForward,
    attn_norm: RmsNorm,
    ffn_norm: RmsNorm,
    attn_gamma_proj: nn::Linear,
    attn_beta_proj: nn::Linear,
    ffn_gamma_proj: nn::Linear,
    ffn_beta_proj: nn::Linear,
}

impl TransformerBlock {
    fn new(vs: nn::Path, config: &CsmModelConfig, rotary_emb: RotaryEmbedding, is_backbone: bool) -> Result<Self, TchError> {
        let (embed_dim, norm_eps) = if is_backbone {
            (config.backbone_embed_dim, config.backbone_norm_eps)
        } else {
            (config.decoder_embed_dim, config.decoder_norm_eps)
        };
        let attn = Attention::new(vs.clone() / "attn", config, rotary_emb, is_backbone)?;
        let ffn = FeedForward::new(vs.clone() / "ffn", config, is_backbone)?;
        let attn_norm = RmsNorm::new(vs.clone() / "attn_norm", embed_dim, norm_eps)?;
        let ffn_norm = RmsNorm::new(vs.clone() / "ffn_norm", embed_dim, norm_eps)?;

        let conditioning_dim = embed_dim;

        let attn_gamma_proj = nn::Linear { 
            ws: vs.var("attn_gamma_proj_weight", &[embed_dim, conditioning_dim], nn::Init::Const(0.0)),
            bs: Some(vs.var("attn_gamma_proj_bias", &[embed_dim], nn::Init::Const(1.0)))
        };
        let attn_beta_proj = nn::Linear { 
            ws: vs.var("attn_beta_proj_weight", &[embed_dim, conditioning_dim], nn::Init::Const(0.0)),
            bs: Some(vs.var("attn_beta_proj_bias", &[embed_dim], nn::Init::Const(0.0)))
        };
        let ffn_gamma_proj = nn::Linear { 
            ws: vs.var("ffn_gamma_proj_weight", &[embed_dim, conditioning_dim], nn::Init::Const(0.0)),
            bs: Some(vs.var("ffn_gamma_proj_bias", &[embed_dim], nn::Init::Const(1.0)))
        };
        let ffn_beta_proj = nn::Linear { 
            ws: vs.var("ffn_beta_proj_weight", &[embed_dim, conditioning_dim], nn::Init::Const(0.0)),
            bs: Some(vs.var("ffn_beta_proj_bias", &[embed_dim], nn::Init::Const(0.0)))
        };

        Ok(Self { 
            attn, 
            ffn, 
            attn_norm, 
            ffn_norm,
            attn_gamma_proj,
            attn_beta_proj,
            ffn_gamma_proj,
            ffn_beta_proj,
        })
    }

    fn forward(
        &mut self, 
        xs: &Tensor, 
        start_pos: usize, 
        mask: Option<&Tensor>, 
        cache: &mut AttentionCache,
        conditioning_embedding: Option<&Tensor>
    ) -> Result<Tensor, TchError> {
        // Residual connection start
        let residual = xs;
        
        // --- Apply Conditioning (Additive Bias Example) --- 
        let mut h = xs.copy(); // Copy to modify
        if let Some(cond_emb) = conditioning_embedding {
             // Ensure embedding matches sequence dimension or broadcast
             // Assuming cond_emb is [B, 1, D] or [B, S, D] and xs is [B, S, D]
             // Simple addition might require broadcasting cond_emb from [B, 1, D] to [B, S, D]
             // Placeholder: Directly add if dimensions allow, otherwise log warning.
             // This needs refinement based on actual embedding preparation.
             let size = h.size();
             let last_dim = size.last().unwrap_or(&0);
             // Cast to match types for comparison
             if cond_emb.dim() as i64 == *last_dim {
                 h = &h + cond_emb;
                 debug!("Added conditioning embedding directly.");
             } else if cond_emb.dim() as i64 == 1 { // Check if broadcast is possible
                 h = &h + cond_emb;
                 debug!("Added conditioning embedding via broadcasting.");
             } else {
                 warn!("Conditioning embedding seq_len ({}) doesn't match input seq_len ({}) and is not 1. Skipping addition.", cond_emb.dim(), h.dim());
             }
         }
        // --------------------------------------------------
        
        // Apply attention norm with conditioning
        let attn_norm_gamma = self.attn_gamma_proj.forward(&h);
        let attn_norm_beta = self.attn_beta_proj.forward(&h);
        let h_norm = self.attn_norm.forward(&h, Some(&attn_norm_gamma), Some(&attn_norm_beta))?;
        
        // Self-attention
        let attn_output = self.attn.forward(&h_norm, start_pos, mask, cache)?;
        
        // First residual connection
        h = residual + attn_output;
        
        // Feed-forward part
        let ffn_input = &h; // Input for FFN norm
        let ffn_norm_gamma = self.ffn_gamma_proj.forward(ffn_input);
        let ffn_norm_beta = self.ffn_beta_proj.forward(ffn_input);
        let h_norm = self.ffn_norm.forward(ffn_input, Some(&ffn_norm_gamma), Some(&ffn_norm_beta))?;
        let ffn_output = self.ffn.forward(&h_norm)?;
        
        // Second residual connection
        let output = &h + ffn_output;
        
        Ok(output)
    }
}

#[derive(Debug)]
struct LlamaTransformer {
    layers: Vec<TransformerBlock>,
    norm: RmsNorm,
    final_gamma_proj: nn::Linear,
    final_beta_proj: nn::Linear,
}

impl LlamaTransformer {
    fn new(vs: &nn::Path, config: &CsmModelConfig, is_backbone: bool) -> Result<Self, TchError> {
        let (num_layers, embed_dim, head_dim, norm_eps, rope_base) = if is_backbone {
            (
                config.backbone_num_layers,
                config.backbone_embed_dim,
                config.backbone_embed_dim / config.backbone_num_heads,
                config.backbone_norm_eps,
                config.backbone_rope_base
            )
        } else {
            (
                config.decoder_num_layers,
                config.decoder_embed_dim,
                config.decoder_embed_dim / config.decoder_num_heads,
                config.decoder_norm_eps,
                config.decoder_rope_base
            )
        };

        let rotary_emb = RotaryEmbedding::new(head_dim, config.max_seq_len, rope_base, vs.device())?;

        let mut layers = Vec::with_capacity(num_layers as usize);
        let layers_vs = vs / "layers";
        for i in 0..num_layers {
            let layer = TransformerBlock::new(layers_vs.clone() / i, config, rotary_emb.clone(), is_backbone)?;
            layers.push(layer);
        }
        let norm = RmsNorm::new(vs / "norm", embed_dim, norm_eps)?;

        let conditioning_dim = embed_dim;
        let proj_config = Default::default();
        let final_gamma_proj = nn::linear(vs.clone() / "final_gamma_proj", conditioning_dim, embed_dim, proj_config);
        let final_beta_proj = nn::linear(vs.clone() / "final_beta_proj", conditioning_dim, embed_dim, proj_config);

        Ok(Self { layers, norm, final_gamma_proj, final_beta_proj })
    }

    fn forward(
        &mut self, 
        xs: &Tensor, 
        start_pos: usize, 
        mask: Option<&Tensor>, 
        cache: &mut TransformerCache,
        conditioning_embedding: Option<&Tensor> // Parameter already exists
    ) -> Result<Tensor, TchError> {
        let mut h = xs.copy(); // Copy input to allow mutation
        
        if cache.layer_caches.len() != self.layers.len() {
             return Err(TchError::Shape(format!(
                 "Cache layers ({}) do not match transformer layers ({})",
                 cache.layer_caches.len(),
                 self.layers.len()
             )));
         }

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let layer_cache = &mut cache.layer_caches[i];
            
            // --- Pass conditioning_embedding down to the block --- 
            h = layer.forward(&h, start_pos, mask, layer_cache, conditioning_embedding)?;
            // ----------------------------------------------------
        }
        
        // Final normalization is handled outside this transformer block
        Ok(h)
    }
}

// Define the Backbone component
#[derive(Debug)]
struct CsmBackbone {
    text_embeddings: nn::Embedding,
    // TODO: Add audio embeddings if multimodal
    transformer: LlamaTransformer,
    // TODO: Add semantic head projection
    // semantic_head: nn::Linear,
    // ADDED: Projection layer for context embeddings
    context_projection: Option<nn::Linear>,
}

impl CsmBackbone {
    fn new(vs: &nn::Path, config: &CsmModelConfig) -> Result<Self, TchError> {
        // Fix 1: Remove '?' as nn::embedding likely doesn't return Result anymore
        // Fix 2: Replace .pp() with /
        let text_embeddings = nn::embedding(
            vs / "text_embeddings", // Changed from vs.pp(...)
            config.vocab_size as i64,
            config.backbone_embed_dim, // Changed from config.backbone_dim
            Default::default()
        ); // Removed '?'

        // Fix 2: Replace .pp() with /
        // Fix 1: Remove '?' as LlamaTransformer::new might not return Result or handles errors internally
        let transformer = LlamaTransformer::new(&(vs / "transformer"), config, true)?; // Keep '?' for now, assuming constructor might fail

        // Initialize context projection layer if embedding dim is configured and differs
        let context_projection = if let Some(llm_embed_dim) = config.llm_embedding_dim {
             // Fix 3: Use backbone_embed_dim
             if llm_embed_dim != config.backbone_embed_dim {
                 info!("Creating projection layer for context embedding: {} -> {}", llm_embed_dim, config.backbone_embed_dim);
                 // Fix 1 & 2: Remove '?' and use /
                 Some(nn::linear(vs / "context_projection", llm_embed_dim, config.backbone_embed_dim, Default::default())) // Removed '?'
             } else {
                 info!("Context embedding dimension matches backbone dimension. No projection needed.");
                 None
             }
         } else {
             info!("LLM embedding dimension not configured. No projection layer created.");
             None
         };

        Ok(Self {
            text_embeddings,
            transformer,
            context_projection,
        })
    }

    fn forward(
        &mut self,
        text_tokens: &Tensor,
        start_pos: usize,
        _mask: Option<&Tensor>, // Marked unused for now, recalculate below
        cache: &mut TransformerCache,
        config: &CsmModelConfig,
        context_embedding: Option<&ContextEmbedding> // ADDED context embedding param
    ) -> Result<(Tensor, Tensor), TchError> // Return tuple (hidden_states, semantic_logits)
    {
        let (batch_size, seq_len) = text_tokens.size2()?;
        let device = text_tokens.device(); // Fix 5: Get device from input tensor

        if start_pos > 0 && cache.layer_caches[0].kv.is_none() {
             warn!("Using start_pos > 0 but cache is empty. This might be inefficient or incorrect.");
        }

        // 1. Get text embeddings
        // Fix 1: Remove '?' from forward call
        let txt_embeddings = self.text_embeddings.forward(text_tokens);
        let mut h = txt_embeddings;

        // --- Project and Prepare Context Embedding --- 
        let conditioning_tensor = if let Some(embedding) = context_embedding {
            let projected_embedding = if let Some(proj_layer) = &self.context_projection {
                 debug!("Projecting context embedding...");
                 proj_layer.forward(&embedding.tensor)
             } else {
                 // Check if dimensions match if no projection layer exists
                 let size = h.size();
                 let last_dim = size.last().unwrap_or(&0);
                 // Cast to match types for comparison
                 if embedding.dim() as i64 == *last_dim {
                     embedding.tensor.copy() // Copy if dims match and no projection needed
                 } else {
                     warn!(
                         "Context embedding dim ({}) doesn't match hidden dim ({}) and no projection layer exists. Skipping backbone conditioning.", 
                         embedding.dim(), last_dim
                     );
                     return Err(TchError::Kind("Context embedding dimensions don't match and no projection layer exists".to_string()));
                 }
             };
             
             // Ensure the projected embedding can be broadcast/added (e.g., shape [B, 1, D])
             projected_embedding.unsqueeze(0).unsqueeze(0) // Assuming original embedding is [D], make it [1, 1, D]
             // Note: Adjust unsqueezing based on actual embedding tensor shape from LLM

        } else {
             return Err(TchError::Kind("No context embedding provided".to_string()));
        };
        // ---------------------------------------------
        
        // 2. Create Causal Mask if needed
        // Fix 6: Adjust mask creation logic
        let mask_tensor; // Declare mask tensor variable
        let mask = if seq_len <= 1 {
             None
         } else {
             match create_causal_mask(seq_len, device) { // Fix 5: Use device obtained earlier
                 Ok(m) => {
                     mask_tensor = m; // Assign to the declared variable
                     Some(&mask_tensor) // Return a reference
                 }
                 Err(e) => {
                     error!("Failed to create causal mask: {}", e);
                     None
                 }
             }
         };

        // 3. Pass through transformer layers
        // Note: Still potentially ambiguous 'forward' call - Fix 7 pending
        h = self.transformer.forward(
            &h, 
            start_pos, 
            mask, 
            cache,
            Some(conditioning_tensor.as_ref()) // Wrap in Some() to create Option<&Tensor>
        )?;

        // 4. Final normalization
        // Fix 1: Remove '?' from forward calls
        let final_norm_gamma = self.transformer.final_gamma_proj.forward(&h);
        let final_norm_beta = self.transformer.final_beta_proj.forward(&h);
        let backbone_hidden_states = self.transformer.norm.forward(&h, Some(&final_norm_gamma), Some(&final_norm_beta))?; // Keep '?' for norm

        let vocab_size = config.vocab_size as i64;
        // Fix 5: Use device obtained earlier
        let semantic_logits = Tensor::zeros(&[batch_size, seq_len, vocab_size], (Kind::Float, device));

        Ok((backbone_hidden_states, semantic_logits))
    }
}

// Define the Decoder component
#[derive(Debug)]
struct CsmDecoder {
    config: CsmModelConfig,
    audio_embeddings: nn::Embedding,
    transformer: LlamaTransformer,
    projection: nn::Linear,
    codebook_heads: Vec<nn::Linear>,
}

impl CsmDecoder {
    fn new(vs: &nn::Path, config: &CsmModelConfig) -> Result<Self, TchError> {
        // Fix 1: Remove '?'
        let audio_embeddings = nn::embedding(
            vs / "audio_embeddings",
            config.acoustic_vocab_size * config.num_acoustic_codebooks, // Assuming these are i64
            config.decoder_embed_dim,
            Default::default()
        ); // Removed '?'

        let transformer = LlamaTransformer::new(&(vs / "decoder"), config, false)?; // Keep '?'

        // Fix 1: Remove '?'
        let projection = nn::linear(
            vs / "projection",
            config.backbone_embed_dim, // Assuming this is correct
            config.decoder_embed_dim,
            Default::default()
        ); // Removed '?'

        // Fix 8: Convert i64 to usize for capacity
        let num_codebooks_usize = config.num_acoustic_codebooks.try_into().map_err(|e| TchError::Convert(format!("Invalid num_acoustic_codebooks: {}", e)))?;
        let mut codebook_heads = Vec::with_capacity(num_codebooks_usize);

        for i in 0..config.num_acoustic_codebooks {
            let head_name = format!("acoustic_head_{}", i);
            // Fix 1: Remove '?'
            let head = nn::linear(
                vs / &head_name,
                config.decoder_embed_dim,
                config.acoustic_vocab_size, // Assuming i64
                Default::default()
            ); // Removed '?'
            codebook_heads.push(head);
        }

        Ok(Self {
            config: config.clone(),
            audio_embeddings,
            transformer,
            projection,
            codebook_heads,
        })
    }

    // Re-add embed_audio method
    fn embed_audio(&self, codebook_idx: i64, tokens: &Tensor) -> Result<Tensor, TchError> {
        let acoustic_vocab_size = self.config.acoustic_vocab_size;
        let offset = codebook_idx * acoustic_vocab_size;
        let offset_tokens = tokens + offset;
        // Ensure tensor is on the correct device
        Ok(self.audio_embeddings.forward(&offset_tokens.to_device(self.audio_embeddings.ws.device())))
    }

    // Corrected forward signature to accept _conditioning_embedding
    fn forward(
        &mut self,
        backbone_output: &Tensor,
        prev_acoustic_tokens_embedded: &Tensor,
        start_pos: usize,
        mask: Option<&Tensor>,
        cache: &mut TransformerCache,
        conditioning_embedding: Option<&Tensor> // Pass embedding TENSOR if available
    ) -> Result<Vec<Tensor>, TchError> // Returns Vec of acoustic logits
    {
        // Combine backbone output with previous acoustic embeddings
        // Fix 1: Remove '?' from tensor addition (likely panics on error)
        let decoder_input = backbone_output + prev_acoustic_tokens_embedded;

        // Pass through decoder transformer layers
        // Note: Still potentially ambiguous 'forward' call - Fix 7 pending
        let transformer_output = self.transformer.forward(
            &decoder_input,
            start_pos,
            mask,
            cache,
            conditioning_embedding // Pass conditioning embedding down
        )?;

        // Project to acoustic logits
        // Fix 1: Remove '?' from forward call
        let projected_output = self.projection.forward(&transformer_output);

        // Fix 8 requires capacity check already done in `new`
        let mut acoustic_logits = Vec::with_capacity(self.codebook_heads.len());
        for head in &self.codebook_heads {
            // Fix 1: Remove '?' from forward call
            acoustic_logits.push(head.forward(&projected_output));
        }

        // Now we can safely return the tokens without any further await points
        Ok(acoustic_logits)
    }
}

// Remove Debug derive and implement it manually to skip llm_processor
// #[derive(Debug)]
pub struct RustCsmModel {
    config: CsmModelConfig,
    backbone: CsmBackbone,
    decoder: CsmDecoder,
    tokenizer: Box<dyn Tokenizer>,
    device: Device,
    vocoder: Box<dyn Vocoder>, // Assuming Vocoder trait
    llm_processor: Arc<dyn LlmProcessor>, // Add LLM processor
}

// Manually implement Debug to exclude llm_processor
impl std::fmt::Debug for RustCsmModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RustCsmModel")
         .field("config", &self.config)
         .field("backbone", &self.backbone) // Assuming CsmBackbone implements Debug
         .field("decoder", &self.decoder)   // Assuming CsmDecoder implements Debug
         .field("tokenizer", &"<Tokenizer>") // Placeholder for tokenizer
         .field("device", &self.device)
         .field("vocoder", &"<Vocoder>")     // Placeholder for vocoder
         .field("llm_processor", &"<LlmProcessor>") // Indicate presence without debugging
         .finish()
    }
}

// SAFETY: Mark Send + Sync as layers hold Tensors
unsafe impl Send for RustCsmModel {}
unsafe impl Sync for RustCsmModel {}

impl RustCsmModel {
    pub fn new(
        model_dir: &Path,
        device: Device,
        _seed: Option<u64>,
        llm_processor: Arc<dyn LlmProcessor>, // Add llm_processor parameter
    ) -> Result<Self, ModelError> {
        let config_path = model_dir.join("config.json");
        let mut config = CsmModelConfig::from_file(&config_path)?;
        config.device = device;
        
        let weights_path = model_dir.join("model.safetensors");
        let tokenizer_path = model_dir.join("tokenizer.json");

        if !weights_path.exists() {
            return Err(ModelError::LoadError(format!("Weights file not found: {:?}", weights_path)));
        }
        if !tokenizer_path.exists() {
            return Err(ModelError::LoadError(format!("Tokenizer file not found: {:?}", tokenizer_path)));
        }

        let mut vs = nn::VarStore::new(device);
        let p = &vs.root();

        // --- Instantiate components using VarStore ---
        let backbone = CsmBackbone::new(p, &config)?;
        let decoder = CsmDecoder::new(p, &config)?;
        // ------------------------------------------

        // --- Load weights from SafeTensors ---
        let file = File::open(&weights_path)
            .map_err(|e| ModelError::LoadError(format!("Failed to open weights file {:?}: {}", weights_path, e)))?;
        let mmap = unsafe {
            MmapOptions::new().map(&file)
                .map_err(|e| ModelError::LoadError(format!("Failed to memory map weights file {:?}: {}", weights_path, e)))?
        };
        let tensors = SafeTensors::deserialize(&mmap)
             .map_err(|e| ModelError::LoadError(format!("Failed to deserialize safetensors from {:?}: {}", weights_path, e)))?;

        match load_weights_safe(&mut vs, &tensors, device) {
             Ok(_) => info!("Successfully attempted weights load."),
             Err(e) => warn!("Weight loading error: {}", e),
        };
        // -------------------------------------

        // Use Box<dyn Tokenizer> with LlamaTokenizer created from config
        let tokenizer_config = crate::tokenization::TokenizerConfig {
            vocab_path: tokenizer_path.to_string_lossy().to_string(),
            max_length: 2048,
            add_special_tokens: true,
            cache_size: 10000,
        };
        
        let tokenizer: Box<dyn Tokenizer> = Box::new(
            LlamaTokenizer::new(tokenizer_config)
                .map_err(|e| ModelError::TokenizationError(format!("Failed to load tokenizer: {}", e)))?
        );

        // Create and initialize the vocoder
        let mut vocoder_impl = MimiVocoder::new(24000, device)?;
        let mimi_path = model_dir.parent().unwrap_or(model_dir).join("mimi/model.safetensors");
        vocoder_impl.load_model(mimi_path)?;
        let vocoder = Box::new(vocoder_impl);

        Ok(Self {
            config,
            backbone,
            decoder,
            tokenizer,
            device,
            vocoder,
            llm_processor, // Store the passed LLM processor
        })
    }

    pub async fn synthesize_streaming_internal(
        &mut self,
        text: &str,
        conversation_history: Option<&ConversationHistory>,
        temperature: Option<f64>,
        top_k: Option<i64>,
        seed: Option<u64>,
        audio_token_tx: mpsc::Sender<Vec<(i64, Vec<i64>)>>,
    ) -> Result<(), ModelError> {
        let _guard = tch::no_grad_guard();
        let temperature = temperature.unwrap_or(self.config.temperature);
        let top_k = top_k.unwrap_or(self.config.top_k);

        if let Some(seed) = seed {
            tch::manual_seed(seed as i64);
        }

        // --- Generate Context Embedding --- 
        let context_embedding = if let Some(history) = conversation_history {
            match self.llm_processor.generate_embeddings(history) {
                Ok(embedding) => {
                    info!("Generated context embedding (dim: {}) for synthesis.", embedding.dim());
                    Some(embedding) // Store the whole embedding struct
                },
                Err(e) => {
                    warn!("Failed to generate context embedding: {}. Proceeding without.", e);
                    None
                }
            }
        } else {
            None
        };
        // ----------------------------------
        
        // Tokenize input with proper error handling
        let tokens = self.tokenizer.encode(text, true)
            .map_err(|e| ModelError::Other(anyhow!("Tokenization failed: {}", e)))?; // Wrap error
        
        // Convert tokens to tensors
        let token_ids = tokens.iter().map(|&t| t as i64).collect::<Vec<_>>();
        let text_tokens = Tensor::from_slice(&token_ids).to(self.device).unsqueeze(0);

        let mut streaming_state = StreamingState::new(&self.config);
        streaming_state.reset(); // Ensure caches are clear
        let mut all_acoustic_tokens_step: Vec<(i64, Vec<i64>)> = Vec::new();
        let mut acoustic_start_token = Tensor::zeros(&[1, 1], (Kind::Int64, self.device)); // Start token [B, T]

        let start_time = std::time::Instant::now();
        let mut token_count = 0;
        
        // --- Backbone Processing (Once) --- 
        debug!("Running backbone forward pass...");
        let backbone_result = self.backbone.forward(
            &text_tokens, 
            0, // start_pos for backbone is always 0
            None, // No mask needed for initial backbone processing?
            &mut streaming_state.backbone_cache,
            &self.config,
            context_embedding.as_ref() // Pass the ContextEmbedding Option
         );
         
        let (backbone_hidden_states, _semantic_logits) = match backbone_result {
            Ok(result) => result,
            Err(e) => {
                error!("Backbone forward pass failed: {}", e);
                return Err(ModelError::Tch(e));
            }
        };
        debug!("Backbone forward pass completed.");
        // ----------------------------------

        // --- Decoder Loop --- 
        for step in 0..self.config.max_seq_len {
            token_count += 1;
            let start_pos = if step == 0 { 0 } else { streaming_state.decoder_cache.layer_caches[0].kv.as_ref().map_or(0, |(k, _)| k.size()[2] as usize) };
            
            // Embed the previously generated acoustic token for this step
            // Use the re-added embed_audio method
            let prev_acoustic_embedded = self.decoder.embed_audio(0, &acoustic_start_token) 
                 .map_err(ModelError::Tch)?;
                 
            // Decoder forward pass
            let decoder_result = self.decoder.forward(
                &backbone_hidden_states, // Use the processed backbone output
                &prev_acoustic_embedded, 
                start_pos,
                None, // Masking for decoder?
                &mut streaming_state.decoder_cache,
                context_embedding.as_ref().map(|e| &e.tensor) // Pass embedding TENSOR if available
            );

            let acoustic_logits = match decoder_result {
                Ok(logits) => logits,
                Err(e) => {
                    error!("Decoder forward pass failed at step {}: {}", step, e);
                    // Attempt to send what we have before erroring
                    let tokens_to_send = all_acoustic_tokens_step.clone();
                    if !tokens_to_send.is_empty() {
                        // All tensor ops done, now safe to await
                        if audio_token_tx.send(tokens_to_send).await.is_err() {
                            warn!("Receiver dropped before completing partial send on error.");
                        }
                    }
                    return Err(ModelError::Tch(e));
                }
            };
            
            // Sample next acoustic tokens
            let mut next_acoustic_tokens_step = Vec::with_capacity(self.config.num_codebooks as usize);
            
            // Sample from logits for each codebook
            for (_codebook_idx, logits) in acoustic_logits.iter().enumerate() {
                let squeezed_logits = logits.squeeze_dim(1);
                let token = match sample_topk(&squeezed_logits, temperature, top_k) {
                    Ok(t) => t.squeeze().int64_value(&[]),
                    Err(e) => return Err(ModelError::Tch(e)),
                };
                next_acoustic_tokens_step.push(token);
            }
            
            // Prepare the next input token (only the 0th codebook? Needs clarification based on model design)
            // TODO: Verify this logic - should subsequent inputs use the previously generated token?
            if next_acoustic_tokens_step.is_empty() {
                return Err(ModelError::Other(anyhow!("No tokens generated from acoustic logits")));
            }
            
            acoustic_start_token = Tensor::from_slice(&[next_acoustic_tokens_step[0]])
                 .to(self.device)
                 .unsqueeze(0); // Shape [1, 1]

            // Store the full set of acoustic tokens for this step
            // Use step as the frame index (assuming 1 frame per step)
            all_acoustic_tokens_step.push((step as i64, next_acoustic_tokens_step));
            
            // --- Check for EOS token --- 
            // Need to define how EOS is represented in acoustic tokens. 
            // Placeholder: Assume a special value or pattern signifies EOS.
            // This logic needs to be adapted based on the actual model's EOS mechanism.
            // if next_acoustic_tokens_step[0] == EOS_TOKEN_ID { // Example: check 0th codebook
            //     info!("EOS token generated at step {}. Stopping generation.", step);
            //     break;
            // }
            // ------------------------- 

            // Send tokens periodically (e.g., every N steps or T milliseconds)
            // This needs tuning based on desired latency and chunk size.
            const SEND_INTERVAL_STEPS: usize = 50; // Example: send every 50 steps
            if !all_acoustic_tokens_step.is_empty() && all_acoustic_tokens_step.len() % SEND_INTERVAL_STEPS == 0 {
                debug!("Sending batch of {} audio token frames...", all_acoustic_tokens_step.len());
                
                // Get copy of tokens, drop tensor references, then await
                let tokens_to_send = all_acoustic_tokens_step.clone();
                drop(acoustic_logits);
                
                // Now safe to await
                if audio_token_tx.send(tokens_to_send).await.is_err() {
                    warn!("Receiver dropped. Stopping generation.");
                    return Ok(()); // Exit gracefully if the receiver is gone
                }
                all_acoustic_tokens_step.clear(); // Clear buffer after sending
            }
        }
        
        // Send any remaining tokens
        if !all_acoustic_tokens_step.is_empty() {
            debug!("Sending final batch of {} audio token frames...", all_acoustic_tokens_step.len());
            
            // Safe to await here as we're done with tensors
            let tokens_to_send = all_acoustic_tokens_step.clone();
            if audio_token_tx.send(tokens_to_send).await.is_err() {
                warn!("Receiver dropped before final send.");
            }
        }
        
        let duration = start_time.elapsed();
        info!(
            "Generated {} audio tokens in {:.3}s ({:.2} tokens/s)",
            token_count,
            duration.as_secs_f32(),
            token_count as f32 / duration.as_secs_f32()
        );

        Ok(())
    }

    // Make sure synthesize properly handles the revised token format of (i64, Vec<i64>)
    pub async fn synthesize(
        &mut self,
        text: &str,
        conversation_history: Option<&ConversationHistory>,
        temperature: Option<f64>,
        top_k: Option<i64>,
        seed: Option<u64>,
    ) -> Result<Vec<i16>, ModelError> {
        // First, call synthesize_streaming to get all tokens
        let (tx, mut rx) = mpsc::channel::<Vec<(i64, Vec<i64>)>>(1024); // Buffer size
        
        // Run the synthesize_streaming method directly (no spawning)
        let synthesis_result = self.synthesize_streaming(
            text, 
            conversation_history, 
            temperature, 
            top_k, 
            seed, 
            tx
        ).await;
        
        // Check if synthesis succeeded
        synthesis_result?;
        
        // Collect all audio tokens from the channel
        let mut all_acoustic_tokens: Vec<(i64, Vec<i64>)> = Vec::new();
        while let Ok(Some(tokens)) = rx.try_recv().map_err(|_| ()).map(Some) {
            all_acoustic_tokens.extend(tokens);
        }
        
        if all_acoustic_tokens.is_empty() {
            warn!("No audio tokens were generated");
            return Ok(Vec::new());
        }
            
        // Now that synthesis is complete, decode the tokens
        let audio_output = self.vocoder.decode(&all_acoustic_tokens)
            .map_err(|e| {
                error!("Vocoder failed to decode audio tokens: {}", e);
                e
            })?;
            
        info!("Synthesized audio: {} tokens -> {} samples", 
            all_acoustic_tokens.len(),
            audio_output.len()
        );
        
        Ok(audio_output)
    }

    // This is the internal streaming synthesize method, KEEP conversation_history here
    pub async fn synthesize_streaming(
        &mut self,
        text: &str,
        conversation_history: Option<&ConversationHistory>,
        temperature: Option<f64>,
        top_k: Option<i64>,
        seed: Option<u64>,
        audio_token_tx: mpsc::Sender<Vec<(i64, Vec<i64>)>>,
    ) -> Result<(), ModelError> {
        // Directly call the internal streaming logic, passing history
        self.synthesize_streaming_internal(text, conversation_history, temperature, top_k, seed, audio_token_tx).await
    }

    // Removed the problematic clone_for_async method
}