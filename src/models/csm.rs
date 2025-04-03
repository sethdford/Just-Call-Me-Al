use std::path::Path;
use tch::{Device, Tensor, Kind, TchError, nn};
use crate::models::{CSMModel, ModelError};
use crate::models::config::{CsmModelConfig, SynthesizerParams};
use tokenizers::Tokenizer;
use async_trait::async_trait;
use safetensors::SafeTensors;
use safetensors::tensor::TensorView;
use std::sync::Arc;
use tch::nn::Module;
use std::collections::HashMap;
use tokio::sync::mpsc;
use tracing::{info, warn, error};
use std::fs::File;
use memmap2::MmapOptions;
use tokio::sync::Mutex as TokioMutex;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use crate::vocoder::Vocoder; // Import the Vocoder trait if needed for decode
use crate::vocoder::MimiVocoder; // Add MimiVocoder import

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
    pub fn new(model_dir: &Path, device: Device) -> Result<Self, ModelError> {
        info!("Initializing CSMImpl Wrapper for model dir: {:?} and device: {:?}", model_dir, device);

        let rust_model_instance = RustCsmModel::new(model_dir, device, None)?;

        // Wrap in Mutex then Arc
        let rust_model_arc = Arc::new(TokioMutex::new(rust_model_instance));

        Ok(Self {
            rust_model: rust_model_arc,
        })
    }
}

// --- CSMModel Trait Implementation for CSMImpl ---
#[async_trait]
impl CSMModel for CSMImpl {
    // Update return type to i16
    async fn synthesize(
        &self,
        text: &str,
        temperature: Option<f64>,
        top_k: Option<i64>,
        seed: Option<u64>,
    ) -> Result<Vec<i16>, ModelError> {
        info!("Synthesizing non-streamed audio for: {}", text);
        let (audio_token_tx, mut audio_token_rx) = mpsc::channel(100);

        let model_clone = self.rust_model.clone();
        let text_owned = text.to_string();
        // Clone individual params
        let temp_clone = temperature;
        let topk_clone = top_k;
        let seed_clone = seed;

        let synthesis_handle = tokio::spawn(async move {
            let mut model_guard = model_clone.lock().await;
            // Pass individual params to internal method
            model_guard.synthesize_streaming_internal(
                &text_owned,
                temp_clone,
                topk_clone,
                seed_clone,
                audio_token_tx,
            ).await
        });

        let mut all_token_chunks: Vec<Vec<(i64, Vec<i64>)>> = Vec::new();
        while let Some(tokens) = audio_token_rx.recv().await {
             if !tokens.is_empty() {
                 all_token_chunks.push(tokens);
             }
        }

        match synthesis_handle.await {
            Ok(Ok(())) => info!("Synthesis task completed successfully."),
            Ok(Err(e)) => return Err(e),
            Err(join_err) => return Err(ModelError::ProcessError(format!("Task join error: {}", join_err))),
        }

        if all_token_chunks.is_empty() { return Ok(Vec::new()); }
        info!("Collected {} RVQ token chunks.", all_token_chunks.len());

        info!("Starting vocoding...");
        let model_guard = self.rust_model.lock().await;
        let audio_samples = model_guard.vocoder.decode_tokens(all_token_chunks).await?;
        info!("Vocoding complete. Generated {} audio samples.", audio_samples.len());
        Ok(audio_samples)
    }

    // Revert signature to use individual params
    async fn synthesize_streaming(
        &self,
        text: &str,
        temperature: Option<f64>,
        top_k: Option<i64>,
        seed: Option<u64>,
        audio_token_tx: mpsc::Sender<Vec<(i64, Vec<i64>)>>, // Correct channel type
    ) -> Result<(), ModelError> {
        info!("Calling internal streaming synthesis for: {}", text);
        let mut model_guard = self.rust_model.lock().await;
        // Delegate directly, passing individual params
        model_guard.synthesize_streaming_internal(
            text,
            temperature,
            top_k,
            seed,
            audio_token_tx,
        ).await
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
    let last_dim = match x.size().last() {
        Some(&dim) => dim,
        None => return Err(TchError::Kind("Tensor has no dimensions".to_string())),
    };
    let half_dim = last_dim / 2;
    
    // Slice along the last dimension (-1)
    let x1 = x.slice(-1, 0, half_dim, 1);
    let x2 = x.slice(-1, half_dim, last_dim, 1);
    
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
        info!("  TransformerBlock input shape: {:?}", xs.size());

        let residual = xs;
        // Remove unused variable
        // let normed_xs_attn = xs;

        let (gamma_attn, beta_attn) = match conditioning_embedding {
            Some(cond_emb) => {
                let gamma = self.attn_gamma_proj.forward(cond_emb);
                let beta = self.attn_beta_proj.forward(cond_emb);
                (Some(gamma), Some(beta))
            }
            None => (None, None),
        };
        
        let normed_xs_attn = self.attn_norm.forward(xs, gamma_attn.as_ref(), beta_attn.as_ref())?;

        info!("    Normed for Attn shape: {:?}", normed_xs_attn.size());
        let attn_output = self.attn.forward(&normed_xs_attn, start_pos, mask, cache)?;
        info!("    Attn output shape: {:?}", attn_output.size());
        let h = residual.f_add(&attn_output)?;
        info!("    After Attn Add shape: {:?}", h.size());

        let residual_ffn = &h;
        
        let (ffn_gamma, ffn_beta) = match conditioning_embedding {
            Some(cond_emb) => {
                let gamma = self.ffn_gamma_proj.forward(cond_emb);
                let beta = self.ffn_beta_proj.forward(cond_emb);
                (Some(gamma), Some(beta))
            }
            None => (None, None),
        };

        let normed_h_ffn = self.ffn_norm.forward(&h, ffn_gamma.as_ref(), ffn_beta.as_ref())?; 

        info!("    Normed for FFN shape: {:?}", normed_h_ffn.size());
        let ffn_output = self.ffn.forward(&normed_h_ffn)?;
        info!("    FFN output shape: {:?}", ffn_output.size());
        let final_output = residual_ffn.f_add(&ffn_output)?;
        info!("  TransformerBlock output shape: {:?}", final_output.size());
        Ok(final_output)
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
        conditioning_embedding: Option<&Tensor>
    ) -> Result<Tensor, TchError> {
        info!("Shape entering Llama::forward: {:?}", xs.size());
        let mut current_xs = xs.copy();
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
             let layer_cache = cache.layer_caches.get_mut(layer_idx)
                 .ok_or(TchError::FileFormat(format!("Layer cache index {} out of bounds", layer_idx)))?;
             current_xs = layer.forward(&current_xs, start_pos, mask, layer_cache, conditioning_embedding)?;
        }

        let (gamma_final, beta_final) = match conditioning_embedding {
            Some(cond_emb) => {
                let gamma = self.final_gamma_proj.forward(cond_emb);
                let beta = self.final_beta_proj.forward(cond_emb);
                (gamma.unsqueeze(1), beta.unsqueeze(1))
            }
            None => {
                return Err(TchError::Kind("Conditioning embedding required for final norm but not provided".into()));
            }
        };
        
        self.norm.forward(&current_xs, Some(&gamma_final), Some(&beta_final))
    }
}

// Define the Backbone component
#[derive(Debug)]
struct CsmBackbone {
    text_embeddings: nn::Embedding,
    transformer: LlamaTransformer,
}

impl CsmBackbone {
    fn new(vs: &nn::Path, config: &CsmModelConfig) -> Result<Self, TchError> {
        // Explicitly set the vocabulary size to include the new control tokens
        let effective_vocab_size = 128261; // Original size + 5 control tokens
        info!("Initializing CsmBackbone text embeddings with effective vocab size: {}", effective_vocab_size);

        let text_embeddings = nn::embedding(
            vs / "text_embeddings",
            effective_vocab_size, // Use the new size here
            config.backbone_embed_dim,
            Default::default()
        );
        let transformer = LlamaTransformer::new(&(vs / "backbone"), config, true)?;
        Ok(Self {
            text_embeddings,
            transformer,
        })
    }

    // Corrected forward method signature to accept config
    fn forward(
        &mut self, 
        text_tokens: &Tensor, 
        start_pos: usize, 
        mask: Option<&Tensor>, 
        cache: &mut TransformerCache,
        config: &CsmModelConfig // ADDED config parameter
    ) -> Result<(Tensor, Tensor), TchError> // Return tuple (hidden_states, semantic_logits)
    {
        // --- Calculate Conditioning Embedding --- 
        // Identify control token IDs (assuming they are >= 128256 based on vocab size)
        let control_token_mask = text_tokens.ge(128256i64);
        
        let conditioning_embedding = if control_token_mask.any().int64_value(&[]) == 1 {
            // Get embeddings for control tokens only
            // We need to select the embeddings corresponding to the control tokens.
            // This is tricky with indexing. A simpler approach is to average *all* token embeddings
            // where the mask is true, or just use the embedding of the *last* control token.
            
            // Strategy: Use the embedding of the LAST control token in the sequence.
            let (batch_size, _seq_len) = text_tokens.size2()?;
            let mut last_control_token_embeddings = Vec::with_capacity(batch_size as usize);
            let mut found_any_control = false;

            for b in 0..batch_size {
                let batch_tokens = text_tokens.select(0, b);
                let batch_mask = control_token_mask.select(0, b);
                let control_indices = batch_mask.nonzero().squeeze_dim(-1);

                if control_indices.numel() > 0 {
                    let last_control_index = control_indices.select(0, -1); // Get the last index
                    let last_control_token_id = batch_tokens.index_select(0, &last_control_index);
                    let emb = self.text_embeddings.forward(&last_control_token_id);
                    last_control_token_embeddings.push(emb);
                    found_any_control = true;
        } else {
                    // Use the passed config parameter here, NOT self.config
                    let default_emb = Tensor::zeros(&[1, config.backbone_embed_dim], (Kind::Float, config.device));
                    last_control_token_embeddings.push(default_emb);
                }
            }
            
            if found_any_control {
                // Stack embeddings from each batch item: Vec<[1, D]> -> [B, D]
                Some(Tensor::cat(&last_control_token_embeddings, 0))
            } else {
                // No control tokens found in any batch item
                None
            }
        } else {
            // No control tokens in the input
            None
        };
        // ------------------------------------------
        
        let initial_embeddings = self.text_embeddings.forward(text_tokens);
        
        // Pass conditioning embedding to the transformer and handle Result explicitly
        let transformer_result = self.transformer.forward(
            &initial_embeddings, 
            start_pos, 
            mask, 
            cache,
            conditioning_embedding.as_ref() // Pass Option<&Tensor>
        );

        let hidden_states = match transformer_result {
            Ok(hs) => hs,
            Err(e) => return Err(e), // Propagate error
        };
        
        let semantic_logits = hidden_states.copy(); // Or apply a projection head if needed

        Ok((hidden_states, semantic_logits)) 
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
        let audio_embeddings = nn::embedding(
            vs / "audio_embeddings",
            config.acoustic_vocab_size * config.num_acoustic_codebooks,
            config.decoder_embed_dim,
            Default::default()
        );
        let transformer = LlamaTransformer::new(&(vs / "decoder"), config, false)?;
        let projection = nn::linear(
            vs / "projection",
            config.backbone_embed_dim,
            config.decoder_embed_dim,
            Default::default()
        );

        let mut codebook_heads = Vec::with_capacity(config.num_acoustic_codebooks as usize);
        for i in 0..config.num_acoustic_codebooks {
            let head_name = format!("acoustic_head_{}", i);
            let head = nn::linear(
                vs / &head_name,
                config.decoder_embed_dim,
                config.acoustic_vocab_size,
                Default::default()
            );
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

    // Corrected forward signature to accept _conditioning_embedding
    fn forward(
        &mut self, 
        backbone_output: &Tensor, 
        prev_acoustic_tokens_embedded: &Tensor, 
        start_pos: usize,
        mask: Option<&Tensor>, 
        cache: &mut TransformerCache,
        _conditioning_embedding: Option<&Tensor> // Add unused 5th arg 
    ) -> Result<Vec<Tensor>, TchError> // Returns Vec of acoustic logits
    {
        let projected_backbone = self.projection.forward(backbone_output);
        let decoder_input = projected_backbone.f_add(prev_acoustic_tokens_embedded)?;
        
        // Pass None for conditioning_embedding to the decoder's transformer
        let hidden_states = self.transformer.forward(
            &decoder_input, 
            start_pos, 
            mask, 
            cache,
            None // Decoder is not conditioned for now
        )?;

        // Calculate logits ONLY for acoustic heads
        let mut acoustic_logits = Vec::with_capacity(self.config.num_acoustic_codebooks as usize);
        for head in &self.codebook_heads { // Iterate over acoustic heads
            acoustic_logits.push(head.forward(&hidden_states));
        }
        Ok(acoustic_logits)
    }

    // embed_audio and embed_audio_timestep handle acoustic tokens and remain correct
    fn embed_audio(&self, codebook_idx: i64, tokens: &Tensor) -> Result<Tensor, TchError> {
        let acoustic_vocab_size = self.config.acoustic_vocab_size;
        let offset = codebook_idx * acoustic_vocab_size;
        let offset_tokens = tokens + offset;
        Ok(self.audio_embeddings.forward(&offset_tokens.to_device(self.config.device)))
    }

    fn embed_audio_timestep(&self, audio_tokens_step: &[i64]) -> Result<Tensor, ModelError> {
        if audio_tokens_step.len() != self.config.num_acoustic_codebooks as usize {
            return Err(ModelError::InvalidInput(format!(
                "Expected {} acoustic tokens, got {}",
                self.config.num_acoustic_codebooks,
                audio_tokens_step.len()
            )));
        }
        let mut summed_embedding: Option<Tensor> = None;
        for (codebook_idx, &token_id) in audio_tokens_step.iter().enumerate() {
            let token_tensor = Tensor::from_slice(&[token_id]).to_kind(Kind::Int64).to_device(self.config.device);
            let embedding = self.embed_audio(codebook_idx as i64, &token_tensor)?;
            if let Some(sum) = summed_embedding {
                summed_embedding = Some(sum.f_add(&embedding)?);
            } else {
                summed_embedding = Some(embedding);
            }
        }
        summed_embedding.ok_or_else(|| ModelError::ProcessError("Failed to sum acoustic audio embeddings".to_string()))
    }
}

#[derive(Debug)]
pub struct RustCsmModel {
    config: CsmModelConfig,
    backbone: CsmBackbone,
    decoder: CsmDecoder,
    tokenizer: Tokenizer,
    device: Device,
    vocoder: Box<dyn Vocoder>, // Assuming Vocoder trait
}

// SAFETY: Mark Send + Sync as layers hold Tensors
unsafe impl Send for RustCsmModel {}
unsafe impl Sync for RustCsmModel {}

impl RustCsmModel {
    pub fn new(
        model_dir: &Path,
        device: Device,
        _seed: Option<u64>,
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

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| ModelError::TokenizationError(format!("Failed to load tokenizer: {}", e)))?;

        // Create and initialize the vocoder
        let mut vocoder_impl = MimiVocoder::new(24000, device)?;
        vocoder_impl.load_model(model_dir.join("mimi/model.safetensors"))?;
        let vocoder = Box::new(vocoder_impl);

        Ok(Self {
            config,
            backbone,
            decoder,
            tokenizer,
            device,
            vocoder,
        })
    }

    async fn synthesize_streaming_internal(
        &mut self,
        text: &str,
        temperature: Option<f64>,
        top_k: Option<i64>,
        seed: Option<u64>,
        audio_token_tx: mpsc::Sender<Vec<(i64, Vec<i64>)>>,
    ) -> Result<(), ModelError> {
        let mut state = StreamingState::new(&self.config);
        let temperature = temperature.unwrap_or(0.7);
        let top_k = top_k.unwrap_or(50);
        let num_acoustic_codebooks = self.config.num_acoustic_codebooks;

        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| ModelError::TokenizationError(format!("Encode failed: {}", e)))?;
        let prompt_tokens = encoding.get_ids().iter().map(|&id| id as i64).collect::<Vec<i64>>();
        info!(input_text = %text, token_ids = ?prompt_tokens, "Encoded input text to token IDs");
        let prompt_tensor = Tensor::from_slice(&prompt_tokens).view((1, -1)).to_device(self.device);
        let (_batch_size, prompt_len) = prompt_tensor.size2()?;

        state.reset();
        warn!("StreamingState reset (start of streaming).");

        let prompt_mask = create_causal_mask(prompt_len, self.device)?;
        let (backbone_hidden_states, prompt_semantic_logits) = self.backbone.forward(
            &prompt_tensor,
            0,
            Some(&prompt_mask),
            &mut state.backbone_cache,
            &self.config
        )?;

        let max_len = self.config.max_seq_len;
        let max_generation_steps = max_len.saturating_sub(prompt_len);
        info!("Starting generation loop. Max steps: {}", max_generation_steps);

        let mut prev_step_acoustic_embedding = Tensor::zeros(&[1, self.config.decoder_embed_dim], (Kind::Float, self.device));
        let last_backbone_state = backbone_hidden_states.select(1, -1).unsqueeze(1);

        let last_semantic_logits = prompt_semantic_logits.select(1, -1);
        let initial_semantic_token_tensor = sample_topk(&last_semantic_logits, temperature, top_k)?;
        let current_semantic_token = initial_semantic_token_tensor.int64_value(&[0, 0]);

        for audio_step_counter in 0..max_generation_steps {
            let decoder_start_pos = prompt_len as usize + audio_step_counter as usize;

            let acoustic_logits_step: Vec<Tensor> = self.decoder.forward(
                &last_backbone_state,
                &prev_step_acoustic_embedding.unsqueeze(1),
                decoder_start_pos,
                None,
                &mut state.decoder_cache,
                None
            )?;

            let expected_num_acoustic_logits = num_acoustic_codebooks;
            if acoustic_logits_step.len() != expected_num_acoustic_logits as usize {
                return Err(ModelError::ProcessError(format!(
                    "Decoder returned {} acoustic logits, expected {}",
                    acoustic_logits_step.len(), expected_num_acoustic_logits
                )));
            }
            let mut current_step_acoustic_tokens = Vec::with_capacity(num_acoustic_codebooks as usize);
            for acoustic_idx in 0..num_acoustic_codebooks as usize {
                let logits = &acoustic_logits_step[acoustic_idx].squeeze_dims(&[1]);
                let next_token_tensor = sample_topk(logits, temperature, top_k)?;
                let next_token = next_token_tensor.int64_value(&[0, 0]);
                current_step_acoustic_tokens.push(next_token);
            }

            let eos_generated = current_semantic_token == EOS_TOKEN_ID;

            let tokens_to_send = vec![(current_semantic_token, current_step_acoustic_tokens.clone())];
            if audio_token_tx.send(tokens_to_send).await.is_err() {
                 error!("Failed to send audio tokens chunk (semantic, acoustic). Channel closed?");
                 break;
            }

            if eos_generated {
                info!("EOS semantic token ({}) detected at audio step {}, stopping.", current_semantic_token, audio_step_counter);
                break;
            }

            prev_step_acoustic_embedding = self.decoder.embed_audio_timestep(&current_step_acoustic_tokens)?;
        }

        info!("Token generation loop finished.");
        Ok(())
    }

    pub async fn synthesize(
        &mut self,
        text: &str,
        temperature: Option<f64>,
        top_k: Option<i64>,
        seed: Option<u64>,
    ) -> Result<Vec<i16>, ModelError> {
        info!("RustCsmModel: Inherent synthesize calling streaming...");
        let (tx, mut rx) = mpsc::channel(100);
        let text_clone = text.to_string();
        
        let stream_result = self.synthesize_streaming_internal(
             &text_clone, temperature, top_k, seed, tx,
         ).await;
        
        if let Err(e) = stream_result {
             error!("Underlying streaming token synthesis failed: {}", e);
             return Err(e);
        }
        let mut all_token_tuples: Vec<(i64, Vec<i64>)> = Vec::new();
        while let Some(chunk) = rx.recv().await {
            all_token_tuples.extend(chunk);
        }
        info!("Collected {} token tuples (semantic, acoustic_vec).", all_token_tuples.len());
        if all_token_tuples.is_empty() { warn!("No token data produced by streaming."); }
        warn!("Synthesize finished, returning empty audio (vocoder needed). Actual token collection requires vocoder integration.");
        Ok(Vec::new())
    }

    async fn synthesize_streaming(
        &mut self,
        text: &str,
        temperature: Option<f64>,
        top_k: Option<i64>,
        seed: Option<u64>,
        audio_token_tx: mpsc::Sender<Vec<(i64, Vec<i64>)>>,
    ) -> Result<(), ModelError> {
         self.synthesize_streaming_internal(text, temperature, top_k, seed, audio_token_tx).await
    }
}

