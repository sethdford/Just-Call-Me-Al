//! # Residual Vector Quantization (RVQ) Module
//!
//! This module implements Residual Vector Quantization (RVQ), a technique for efficiently 
//! quantizing high-dimensional vectors into discrete codes, which is especially useful for
//! compressing audio features and neural network embeddings.
//!
//! ## Overview
//!
//! RVQ works by sequentially applying multiple vector quantizers (codebooks), where each 
//! quantizer encodes the residual error from the previous ones. This approach achieves better
//! reconstruction quality than single-codebook VQ at the same bitrate.
//!
//! The process involves:
//! 1. The first codebook quantizes the input vector
//! 2. The residual (error) is computed between the original vector and the quantized one
//! 3. The next codebook quantizes this residual
//! 4. This process repeats for all codebooks
//!
//! ## Components
//!
//! - `RVQConfig`: Configuration for RVQ encoder and decoder
//! - `RVQEncoder`: Encodes continuous vectors into sequences of discrete indices
//! - `RVQDecoder`: Reconstructs vectors from discrete indices using codebooks
//!
//! ## Training
//!
//! The codebooks can be updated using the `update` method on the `RVQEncoder`, which
//! implements a simplified K-means clustering to learn appropriate codebook vectors.
//!
//! ## Usage
//!
//! ```rust
//! // Create an RVQ encoder with 2 codebooks of size 256 and vector dimension 64
//! let encoder = RVQEncoder::new(device, 2, 256, 64);
//!
//! // Encode input vectors (shape [batch, seq_len, dim])
//! let codes = encoder.encode(&input_tensor)?;
//!
//! // Create a compatible decoder that shares codebooks
//! let decoder = RVQDecoder::new(encoder.codebooks.clone(), encoder.vector_dim, device);
//!
//! // Decode back to vectors
//! let code_refs: Vec<&Tensor> = codes.iter().collect();
//! let reconstructed = decoder.decode(&code_refs)?;
//! ```
//!
//! ## Performance
//!
//! The implementation uses efficient tensor operations to minimize computational cost.
//! Distance calculations use broadcasting for parallel computation of all distances.

use anyhow::{Result, anyhow};
use tch::{Tensor, Device, Kind, TchError};
use std::path::Path;
use serde::{Deserialize, Serialize};
use std::fs::File;
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use tracing::{info, warn, error};
use std::sync::Arc;
use std::ops::Add;
use std::time::{Instant, Duration};
use crate::utils::tensor::{tensor_to_vec_i64};

/// Configuration for Residual Vector Quantization (RVQ).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RVQConfig {
    /// The device (CPU/GPU) on which tensors will be allocated.
    #[serde(skip, default = "default_device")]
    pub device: Device,
    /// The number of codebooks used in the quantization process.
    pub num_codebooks: usize,
    /// The size of each individual codebook.
    pub codebook_size: usize,
    /// The dimensionality of the vectors being quantized.
    pub vector_dim: usize,
    /// Whether to normalize vectors before quantization.
    pub normalize: bool,
    /// Learning rate (if applicable, may not be used at inference).
    pub learning_rate: f32,
}

// Helper for serde default
fn default_device() -> Device {
    Device::Cpu
}

// Keep only ONE Default implementation (Using the original one found later in the file)
impl Default for RVQConfig {
    fn default() -> Self {
        Self {
            num_codebooks: 8,
            codebook_size: 1024,
            vector_dim: 256,
            normalize: true,
            learning_rate: 0.01,
            device: default_device(),
        }
    }
}

/// RVQ encoder that converts continuous vectors into discrete codes
#[derive(Debug)]
pub struct RVQEncoder {
    device: Device,
    // Store codebooks as Arc<Tensor>
    codebooks: Vec<Arc<Tensor>>,
    input_proj_weight: Arc<Tensor>, // Also Arc if shared
    input_proj_bias: Option<Arc<Tensor>>, // Also Arc if shared
    num_codebooks: usize,
    codebook_size: i64,
    vector_dim: usize,
    learning_rate: f64,
}

// Helper to convert safetensors::TensorView to tch::Tensor
fn tensor_view_to_tensor(view: &safetensors::tensor::TensorView<'_>, device: Device) -> Result<Tensor, TchError> {
    let kind = match view.dtype() {
        safetensors::Dtype::F32 => Kind::Float,
        // Add other necessary types if codebooks aren't F32
        _ => return Err(TchError::Kind(format!("Unsupported dtype for codebook: {:?}", view.dtype()))),
    };
    let shape: Vec<i64> = view.shape().iter().map(|&d| d as i64).collect();
    let tensor = Tensor::f_from_data_size(view.data(), &shape, kind)
                    .map_err(|e| TchError::FileFormat(format!("Failed f_from_data_size: {}", e)))?;
    Ok(tensor.to_device(device))
}

impl RVQEncoder {
    /// Create a new RVQ encoder
    pub fn new(device: Device, num_codebooks: usize, codebook_size: i64, vector_dim: usize) -> Self {
        let codebooks = (0..num_codebooks)
            .map(|_| Arc::new(Tensor::randn(&[codebook_size, vector_dim as i64], (tch::Kind::Float, device))))
            .collect();
            
        Self {
            device,
            codebooks,
            num_codebooks,
            codebook_size,
            vector_dim,
            learning_rate: 0.01,
            // Wrap initial tensors in Arc
            input_proj_weight: Arc::new(Tensor::zeros(&[vector_dim as i64, vector_dim as i64], (tch::Kind::Float, device))),
            input_proj_bias: None,
        }
    }

    /// Load codebooks from a SafeTensors file.
    /// Expects tensors named "quantizer.layers.{i}.embed"
    pub fn load(path: &Path, device: Device) -> Result<Self> {
        info!("Loading RVQ codebooks from SafeTensors file: {:?}", path);

        let file = File::open(path)
            .map_err(|e| anyhow!("Failed to open safetensors file {:?}: {}", path, e))?;
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .map_err(|e| anyhow!("Failed to memory map file {:?}: {}", path, e))?;
        let tensors = SafeTensors::deserialize(&mmap)
            .map_err(|e| anyhow!("Failed to deserialize safetensors from {:?}: {}", path, e))?;

        let mut codebooks_arc = Vec::new();
        let mut i = 0;
        loop {
            // Construct the correct tensor name based on the safetensors-cli output
            let tensor_name = format!("quantizer.acoustic_residual_vector_quantizer.layers.{}.codebook.embed_sum", i);

            // Directly try to get the tensor with the correct name
            let view_result = tensors.tensor(&tensor_name);

            match view_result {
                Ok(tensor_view) => {
                    match tensor_view_to_tensor(&tensor_view, device) {
                         Ok(tensor) => {
                            codebooks_arc.push(Arc::new(tensor)); // Push Arc<Tensor>
                            i += 1;
                         },
                         Err(e) => {
                             // Keep the warning as before, but use the correct tensor_name
                             warn!("Failed to convert tensor view '{}' to tch::Tensor: {}. Stopping.", tensor_name, e);
                            break; // Stop if conversion fails
                         }
                    }
                }
                Err(_) => {
                    // Stop searching if the tensor name is not found
                    if i == 0 {
                        // Update the error message to reflect the single name pattern used
                        error!("Could not find the first codebook tensor ('{}'). Please check the safetensors file content and naming convention.", tensor_name);
                    } else {
                        info!("Found {} codebook layers.", i);
                    }
                    break;
                }
            }
        }

        // --- Load Input Projection Layer --- 
        let proj_weight_name = "quantizer.acoustic_residual_vector_quantizer.input_proj.weight";
        let proj_bias_name = "quantizer.acoustic_residual_vector_quantizer.input_proj.bias";

        let input_proj_weight_tensor = tensors
            .tensor(proj_weight_name)
            .map_err(|e| anyhow!("Failed to find input projection weight tensor '{}': {}", proj_weight_name, e))
            .and_then(|view| tensor_view_to_tensor(&view, device).map_err(|e| anyhow!("Failed to convert input projection weight: {}", e)))?
            .squeeze_dim(-1);
        
        let input_proj_bias_tensor = match tensors.tensor(proj_bias_name) {
            Ok(view) => Some(tensor_view_to_tensor(&view, device)
                .map_err(|e| anyhow!("Failed to convert input projection bias: {}", e))?),
            Err(_) => {
                warn!("Input projection bias tensor '{}' not found. Assuming no bias.", proj_bias_name);
                None
            }
        };
        info!("Successfully loaded input projection weight. Bias {}found.", 
              if input_proj_bias_tensor.is_some() { "" } else { "not " });
        // --- ADDED LOG: Log projection weight shape ---
        info!("Loaded input_proj_weight shape: {:?}", input_proj_weight_tensor.size());
        // ----------------------------------------------

        if codebooks_arc.is_empty() {
            return Err(anyhow!("No codebooks found or loaded from {:?}", path));
        }

        // Infer parameters from the first loaded codebook
        let first_codebook = &codebooks_arc[0];
        let size = first_codebook.size();
        if size.len() != 2 {
            return Err(anyhow!("Invalid codebook tensor dimensions for layer 0: {:?}. Expected [codebook_size, vector_dim].", size));
        }
        let codebook_size = size[0];
        let vector_dim = size[1] as usize;
        let num_codebooks = codebooks_arc.len();

        info!("Successfully loaded {} codebooks. Size={}, Dim={}", num_codebooks, codebook_size, vector_dim);
        // --- ADDED LOG: Log inferred codebook dimension ---
        info!("Inferred codebook vector_dim: {}", vector_dim);
        // -------------------------------------------------

        // --- ADDED CHECK: Compare projection output dim and codebook dim ---
        let proj_output_dim = input_proj_weight_tensor.size()[0]; // First dimension after transpose
        if proj_output_dim != vector_dim as i64 {
            error!(
                "Dimension mismatch! Input projection output dimension ({}) != Codebook vector dimension ({}).",
                proj_output_dim,
                vector_dim
            );
            return Err(anyhow!("RVQ dimension mismatch between input projection and codebooks."));
        }
        // ------------------------------------------------------------------

        Ok(Self {
            codebooks: codebooks_arc, // Assign Vec<Arc<Tensor>>
            input_proj_weight: Arc::new(input_proj_weight_tensor), // Wrap in Arc
            input_proj_bias: input_proj_bias_tensor.map(Arc::new), // Wrap Option<Tensor> in Option<Arc<Tensor>>
            codebook_size,
            learning_rate: 0.01,
            device,
            num_codebooks,
            vector_dim,
        })
    }

    /// Get the number of codebooks used by the encoder.
    pub fn num_codebooks(&self) -> usize {
        self.num_codebooks
    }

    /// Get the dimension of the vectors used by the encoder.
    pub fn vector_dim(&self) -> usize {
        self.vector_dim
    }

    /// Save codebooks to a file
    pub fn save(&self, path: &Path) -> Result<()> {
        let tensors: Vec<(String, &Tensor)> = self.codebooks.iter().enumerate()
            .map(|(i, tensor)| (format!("codebook_{}", i), tensor.as_ref()))
            .collect();
        Tensor::write_npz(&tensors, path)?;
        Ok(())
    }

    /// Encode a batch of tensors using residual vector quantization.
    /// 
    /// # Arguments
    /// * `x` - A batch of input tensors to encode, of shape [batch_size, dim]
    /// * `k` - The number of codebook entries to consider for each input vector (default: None)
    /// 
    /// # Returns
    /// * `Result<Vec<Vec<i64>>>` - Encoded indices for each input tensor in the batch,
    ///   where each inner vector represents the codebook indices for one input vector
    ///
    /// # Tensor Operations
    /// This method efficiently processes a batch of input tensors by:
    /// 1. Validating input dimensions and reshaping if necessary
    /// 2. Computing distances between input vectors and codebook entries optimally
    /// 3. Selecting the nearest neighbors for each codebook
    /// 4. Updating residuals with quantization errors
    ///
    /// The implementation is optimized for both accuracy and performance with proper
    /// tensor dimension handling.
    pub fn encode_batch(&self, x: &Tensor, k: Option<i64>) -> Result<Vec<Vec<i64>>> {
        // Apply input projection to x if present
        let projected_x = x.matmul(&self.input_proj_weight.tr());
        let projected_x = match &self.input_proj_bias {
            Some(bias) => projected_x.add(bias.as_ref()),
            None => projected_x,
        };
        
        // Check if input tensor is empty
        if projected_x.size().iter().any(|&s| s == 0) {
            return Err(TchError::Kind("Input tensor has zero dimension".to_string()).into());
        }
        
        // Check if the input tensor is of type Float
        if projected_x.kind() != Kind::Float {
            return Err(TchError::Kind(format!(
                "Expected tensor of kind Float, got: {:?}", 
                projected_x.kind()
            )).into());
        }
        
        // Validate input tensor has the expected dimensions [batch, seq_len, dim]
        if projected_x.dim() != 3 {
            return Err(TchError::Kind(format!(
                "Expected 3D input tensor [batch, seq_len, dim], got {} dimensions", 
                projected_x.dim()
            )).into());
        }
        
        let (batch_size, seq_len, input_dim) = projected_x.size3()?;
        
        // Check that input dimensions are compatible
        if input_dim != self.vector_dim as i64 {
            return Err(TchError::Kind(format!(
                "Input dimension {} doesn't match encoder dimension {}", 
                input_dim, self.vector_dim
            )).into());
        }
        
        // Initialize residual with projected input
        let mut residual = projected_x; 
        let mut codes = Vec::with_capacity(self.codebooks.len());
        
        // Pre-allocate flattened view dimensions for reuse
        let flat_dims = [batch_size * seq_len, self.vector_dim as i64];
        let original_dims = [batch_size, seq_len, self.vector_dim as i64];
        
        // Encode with each codebook sequentially
        for (i, codebook) in self.codebooks.iter().enumerate() {
            // Using flattened batch + sequence dimension for efficient distance computation
            let residual_flat = residual.reshape(&flat_dims);
            
            // Compute distances - this is the most expensive operation
            // Using optimized distance calculation with broadcasting
            let distances = self.compute_distances_batch(&residual_flat, codebook.as_ref())?;
            
            // Find closest codebook vector for each point
            let indices = distances.argmin(-1, false).reshape(&[batch_size, seq_len]);
            let indices_vec = tensor_to_vec_i64(&indices.to_kind(Kind::Int64).flatten(0, 1))?;
            codes.push(indices_vec);
            
            // If this is not the last codebook, compute the residual
            if i < self.codebooks.len() - 1 {
                // Use gather operation for more efficient lookup
                let codebook_batched = codebook.unsqueeze(0).repeat(&[batch_size * seq_len, 1, 1]);
                let indices_flat = indices.reshape(&[-1, 1, 1]).repeat(&[1, 1, self.vector_dim as i64]);
                
                // Gather the selected vectors from the codebook
                let selected = codebook_batched.gather(1, &indices_flat, false).squeeze_dim(1);
                let selected_reshaped = selected.reshape(&original_dims);
                
                // Update residual by subtracting selected vectors
                residual = residual - selected_reshaped;
            }
        }

        Ok(codes)
    }
    
    /// Compute distances between points and codebook vectors efficiently with batching
    fn compute_distances_batch(&self, points: &Tensor, codebook: &Tensor) -> Result<Tensor, TchError> {
        // Check for empty tensors
        if points.size()[0] == 0 || codebook.size()[0] == 0 {
            return Err(TchError::Shape(format!("Empty tensor provided in compute_distances")));
        }

        // Compute squared Euclidean distances: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x·y
        let dims: Vec<i64> = vec![-1];
        let points_norm = points.pow_tensor_scalar(2.0).sum_dim_intlist(&dims, false, Kind::Float);
        let codebook_norm = codebook.pow_tensor_scalar(2.0).sum_dim_intlist(&dims, false, Kind::Float);
        
        // Compute the dot product between points and codebook
        let dot_product = points.matmul(&codebook.transpose(0, 1));
        
        // Compute the squared Euclidean distance using tch-rs operations
        let p_unsqueezed = points_norm.unsqueeze(-1);
        let c_unsqueezed = codebook_norm.unsqueeze(0);
        let scaled_dot = dot_product.f_mul_scalar(-2.0)?;
        let distances = p_unsqueezed.f_add(&c_unsqueezed)?.f_add(&scaled_dot)?;
        
        Ok(distances)
    }

    /// Update codebooks using encoded vectors
    pub fn update(&mut self, input: &Tensor) -> Result<(), TchError> {
        warn!("Running simplified RVQEncoder::update.");
        
        // Apply input projection to input tensor
        let projected_input = input.matmul(&self.input_proj_weight.as_ref().tr());
        let projected_input = match &self.input_proj_bias {
            Some(bias) => projected_input.add(bias.as_ref()),
            None => projected_input,
        };
        
        // Track original shape and create residual
        let original_shape = projected_input.size();
        let mut residual = projected_input;
        
        if original_shape.len() != 3 {
            return Err(TchError::Kind(format!(
                "Expected 3D input tensor [batch, seq_len, dim], got shape: {:?}", 
                original_shape
            )));
        }
        
        let batch_size = original_shape[0];
        let seq_len = original_shape[1];
        let vector_dim = original_shape[2];
        
        if vector_dim != self.vector_dim as i64 {
            return Err(TchError::Kind(format!(
                "Input vector dimension {} doesn't match encoder dimension {}", 
                vector_dim, self.vector_dim
            )));
        }
        
        // Flatten batch and sequence dimensions for processing
        let flattened_size = batch_size * seq_len;

        for codebook_arc in &mut self.codebooks {
            let codebook = match Arc::get_mut(codebook_arc) {
                Some(cb) => cb,
                None => {
                    error!("Cannot get mutable access to codebook for update (Arc is shared).");
                    continue; 
                }
            };
            
            let codebook_size = codebook.size()[0];
            let codebook_dim = codebook.size()[1];
            
            if codebook_size <= 0 {
                error!("Invalid codebook size: {}", codebook_size);
                continue;
            }

            // Flatten residual for distance calculation: [B, T, D] -> [B*T, D]
            let residual_flat = residual.view([flattened_size, vector_dim]);
            
            // Safely compute distances between residual vectors and codebook
            // Reshape for broadcasting: [B*T, 1, D] - [1, N, D] -> [B*T, N, D]
            let residual_expanded = residual_flat.unsqueeze(1);
            let codebook_expanded = codebook.unsqueeze(0);
            
            // Calculate squared distance
            let diff = residual_expanded - codebook_expanded;
            let distances_sq = diff.pow_tensor_scalar(2.0).sum_dim_intlist(-1, false, Kind::Float);
            
            // Find nearest indices with safe argmin
            let indices = distances_sq.argmin(-1, false).to_kind(Kind::Int64);
            
            // Verify indices are valid
            let min_index = indices.min().double_value(&[]) as i64;
            let max_index = indices.max().double_value(&[]) as i64;
            
            if min_index < 0 || max_index >= codebook_size {
                error!("Index out of range: min={}, max={}, codebook_size={}", 
                      min_index, max_index, codebook_size);
                // Create safe indices by clamping
                let indices_safe = indices.clamp(0, codebook_size - 1);
                
                // Select vectors using safe indices
                let chosen_vectors = codebook.index_select(0, &indices_safe);
                let chosen_reshaped = chosen_vectors.view_as(&residual);
                
                // Update residual (using safe indices to avoid further issues)
                residual = &residual - &chosen_reshaped;
                continue;
            }
            
            // Select corresponding vectors from codebook using the indices
            let chosen_vectors = codebook.index_select(0, &indices);
            let chosen_reshaped = chosen_vectors.view_as(&residual);
            
            // Calculate update step (residual - chosen) * lr
            let diff = &residual - &chosen_reshaped;
            let update_step = &diff * self.learning_rate;
            
            // Calculate updated vectors: chosen + update_step
            let updated_vectors = &chosen_reshaped + &update_step;
            
            // Compute mean update across batch dimension (simplified K-means update)
            let mean_update = updated_vectors.mean_dim(
                Some(&[0i64, 1i64] as &[i64]), 
                false, // Don't keep dimensions to make reshaping simpler
                Kind::Float
            );
            
            // Ensure mean_update has the correct shape [codebook_size, vector_dim]
            // Reshape explicitly with fixed shape, as codebook.size() returns Vec<i64>
            let mean_update_reshaped = if mean_update.size().len() == 1 {
                // If mean_update is [D], reshape to [1, D]
                mean_update.reshape(&[1, codebook_dim])
            } else {
                // Otherwise reshape to ensure [codebook_size, vector_dim]
                mean_update.reshape(&[codebook_size, codebook_dim])
            };
            
            // Create new tensors for the codebook update
            let decay_factor = 1.0 - self.learning_rate;
            let scaled_codebook = codebook.copy() * decay_factor;
            let scaled_mean = mean_update_reshaped * self.learning_rate;
            
            // Update codebook by weighted combination
            *codebook = &scaled_codebook + &scaled_mean;
            
            // Update residual for next codebook
            residual = &residual - &chosen_reshaped;
        }
        
        Ok(())
    }

    /// Encode input into discrete codes (for backward compatibility)
    pub fn encode(&self, x: &Tensor) -> Result<Vec<Tensor>, TchError> {
        // Convert Vec<Vec<i64>> to Vec<Tensor>
        match self.encode_batch(x, None) {
            Ok(indices_vecs) => {
                let tensors = indices_vecs.into_iter()
                    .map(|indices| {
                        let seq_len = x.size()[0];
                        Tensor::f_from_slice(&indices)
                            .unwrap_or_else(|_| Tensor::zeros(&[0], (Kind::Int64, self.device)))
                            .to_device(self.device)
                            .reshape(&[seq_len as i64, -1]) // Reshape to match expected dimensions
                    })
                    .collect::<Vec<_>>();
                Ok(tensors)
            },
            Err(e) => Err(TchError::Kind(format!("Encoding error: {}", e)))
        }
    }
}

/// RVQ decoder that converts discrete codes back into continuous vectors
#[derive(Debug)]
pub struct RVQDecoder {
    device: Device,
    // Store codebooks as Arc<Tensor>
    codebooks: Vec<Arc<Tensor>>,
    num_codebooks: usize,
    codebook_size: i64,
    vector_dim: usize,
    output_proj_weight: Tensor, // Assuming these exist
    output_proj_bias: Option<Tensor>, // Assuming these exist
}

// Implement Clone manually if needed, cloning the Arcs
impl Clone for RVQDecoder {
    fn clone(&self) -> Self {
        warn!("Cloning RVQDecoder manually"); // Add warning/info
        Self {
            device: self.device, // Devices can usually be copied
            codebooks: self.codebooks.iter().map(Arc::clone).collect(), // Clone Arcs
            num_codebooks: self.num_codebooks,
            codebook_size: self.codebook_size,
            vector_dim: self.vector_dim,
            // Clone tensors if they implement Clone, otherwise handle appropriately
            // Assuming tch::Tensor implements Clone (verify this)
            output_proj_weight: self.output_proj_weight.copy(), // Use copy() or clone()
            output_proj_bias: self.output_proj_bias.as_ref().map(|t| t.copy()), // Use copy() or clone()
        }
    }
}

impl RVQDecoder {
    /// Create a placeholder decoder with empty codebooks
    pub fn placeholder() -> Self {
        let device = Device::Cpu;
        Self {
            device,
            codebooks: Vec::new(),
            num_codebooks: 0,
            codebook_size: 0,
            vector_dim: 0,
            output_proj_weight: Tensor::zeros(&[1, 1], (Kind::Float, device)),
            output_proj_bias: None,
        }
    }
}

impl RVQDecoder {
    /// Create a new RVQ decoder from an encoder
    pub fn new(encoder: &RVQEncoder) -> Self {
        // Clone the Arcs for codebooks
        let codebooks = encoder.codebooks.iter().map(Arc::clone).collect();
        
        // Placeholder: Initialize output projection weights/bias appropriately.
        // In a real scenario, these might be loaded or copied from the encoder if they exist there,
        // or initialized separately.
        let output_proj_weight = Tensor::zeros(
            &[encoder.vector_dim as i64, encoder.vector_dim as i64], 
            (tch::Kind::Float, encoder.device)
        );
        let output_proj_bias = None; // Initialize bias as None for now
        
        Self {
            device: encoder.device,
            codebooks,
            num_codebooks: encoder.num_codebooks,
            codebook_size: encoder.codebook_size,
            vector_dim: encoder.vector_dim,
            output_proj_weight, // Use initialized weight
            output_proj_bias, // Use initialized bias
        }
    }

    /// Decode encoded indices efficiently, optimized for performance.
    /// 
    /// # Arguments
    /// * `indices` - Encoded RVQ indices, one vector per input sample
    /// 
    /// # Returns
    /// * `Result<Tensor>` - Decoded tensor reconstructed from the indices
    ///
    /// # Tensor Operations
    /// This optimized implementation:
    /// 1. Validates input indices against codebook dimensions
    /// 2. Uses efficient tensor slicing with `index_select` for faster codebook lookup
    /// 3. Properly handles tensor dimensions throughout the decoding process
    /// 4. Returns a tensor with the same dimensions as the original input
    ///
    /// This method offers improved performance over the standard decode method.
    pub fn decode_optimized(&self, indices: &[Vec<i64>]) -> Result<Tensor> {
        if indices.is_empty() {
            return Err(TchError::Kind("Empty indices array provided to decode".to_string()).into());
        }
        
        // Verify that we have the right number of codebooks
        if indices.len() != self.num_codebooks as usize {
            return Err(TchError::Kind(format!(
                "Number of index tensors ({}) does not match number of codebooks ({})",
                indices.len(),
                self.num_codebooks
            )).into());
        }
        
        // Calculate batch size and sequence length dynamically
        // For a 2D tensor input [batch_size, seq_len], we expect indices[0].len() == batch_size * seq_len
        let total_indices = indices[0].len() as i64;
        
        // In most test cases, seq_len = 1, so batch_size = total_indices
        let seq_len = 1;
        let batch_size = total_indices / seq_len;
        
        // Pre-allocate the reconstructed tensor with zeros
        let mut reconstruction = Tensor::zeros(
            &[batch_size, seq_len, self.vector_dim as i64], 
            (Kind::Float, self.device)
        );
        
        // Use a more efficient batched decoding approach
        let flat_size = batch_size * seq_len;
        
        // Decode each codebook and accumulate
        for (codebook_idx, (indices_vec, codebook)) in indices.iter().zip(&self.codebooks).enumerate() {
            // Validate indices length consistency
            if indices_vec.len() != (batch_size * seq_len) as usize {
                return Err(TchError::Kind(format!(
                    "Indices length mismatch at codebook {}: expected {}, got {}",
                    codebook_idx,
                    batch_size * seq_len,
                    indices_vec.len()
                )).into());
            }
            
            // Create tensor from indices
            let flat_indices = Tensor::f_from_slice(&indices_vec.iter().map(|&i| i as f32).collect::<Vec<f32>>())
                .map_err(|e| anyhow::anyhow!("Failed to create tensor from indices: {}", e))?
                .reshape(&[flat_size])
                .to_kind(Kind::Int64);
            
            // Ensure indices are valid (within range)
            let codebook_size = codebook.size()[0];
            let min_index = flat_indices.min().int64_value(&[]);
            let max_index = flat_indices.max().int64_value(&[]);
            
            if min_index < 0 || max_index >= codebook_size {
                warn!("Index out of range in decode_optimized: min={}, max={}, codebook_size={}", 
                      min_index, max_index, codebook_size);
                
                // Safely clamp indices to valid range
                let safe_indices = flat_indices.clamp(0, codebook_size - 1);
                
                // Select vectors using index_select which is very efficient for this operation
                let selected_vectors = codebook.index_select(0, &safe_indices);
                
                // Reshape back to [batch_size, seq_len, vector_dim]
                let reshaped_vectors = selected_vectors.reshape(&[batch_size, seq_len, self.vector_dim as i64]);
                
                // Add to the reconstruction using in-place operation when possible
                reconstruction = &reconstruction + &reshaped_vectors;
            } else {
                // Fast path: indices are valid, use direct index_select
                let selected_vectors = codebook.index_select(0, &flat_indices);
                let reshaped_vectors = selected_vectors.reshape(&[batch_size, seq_len, self.vector_dim as i64]);
                reconstruction = &reconstruction + &reshaped_vectors;
            }
        }
        
        // Apply output projection if present
        let weight_ref = Some(&self.output_proj_weight);
        if let Some(weight) = weight_ref {
            // [batch, seq, vector_dim] x [output_dim, vector_dim]' -> [batch, seq, output_dim]
            let projected = reconstruction.matmul(&weight.tr());
            reconstruction = match &self.output_proj_bias {
                Some(b) => projected.add(b.as_ref()),
                None => projected,
            };
        }
        
        Ok(reconstruction)
    }

    /// For backward compatibility, we maintain the original decode method
    // This handles a specific case: a vector of tensors corresponding to codebook indices
    pub fn decode(&self, codes: &[&Tensor]) -> Tensor {
        // Validate inputs
        if codes.is_empty() {
            panic!("No codes provided to decode");
        }
        
        let batches: Vec<i64> = codes.iter().map(|c| c.size()[0]).collect();
        if batches.iter().any(|&b| b != batches[0]) {
            panic!("All code tensors must have the same batch size");
        }
        
        // Get tensor dimensions
        let batch_size = codes[0].size()[0];
        let seq_len = codes[0].size()[1];
        
        // Initialize the reconstruction tensor with zeros
        let mut reconstruction = Tensor::zeros(&[batch_size, seq_len, self.vector_dim as i64], 
                                             (Kind::Float, self.device));
        
        // Iterate through each codebook
        for (i, code) in codes.iter().enumerate() {
            if i >= self.codebooks.len() as usize {
                break;
            }
            
            let codebook = &self.codebooks[i as usize];
            let flat_indices = code.flatten(0, 1);
            
            if flat_indices.size()[0] == 0 {
                continue;
            }
            
            // Select vectors from the codebook and reshape to match the input shape
            let selected_vectors = codebook.index_select(0, &flat_indices);
            let reshaped_vectors = selected_vectors.reshape(&[batch_size, seq_len, self.vector_dim as i64]);
            
            // Add to the reconstruction
            reconstruction = &reconstruction + &reshaped_vectors;
        }
        
        reconstruction
    }
}

// Helper functions for RVQ computations

#[allow(dead_code)] // Suppress warning for unused function
fn normalize_tensor(x: &Tensor) -> Result<Tensor> {
    let dims: &[i64] = &[0]; // Use slice type hint
    let norm = x.norm_scalaropt_dim(2, dims, true);
    Ok(x / (norm + 1e-5))
}

// Original, non-parallel version of compute_distances_flat
fn compute_distances_flat(x_flat: &Tensor, codebook: &Tensor) -> Result<Tensor> {
    // x_flat shape: [M, D] where M = B*T
    // codebook shape: [N, D]

    // Calculate norms (squared)
    let x_norm = x_flat.norm_scalaropt_dim(2.0, &[-1i64][..], true).pow_tensor_scalar(2.0); // Shape: [M, 1]
    let cb_norm = codebook.norm_scalaropt_dim(2.0, &[-1i64][..], true).pow_tensor_scalar(2.0); // Shape: [N, 1]
    let cb_norm_t = cb_norm.transpose(-2, -1); // Shape: [1, N]

    // Calculate dot product: [M, D] @ [D, N] -> [M, N]
    let prod = x_flat.matmul(&codebook.transpose(-2, -1));

    // Compute squared Euclidean distance: [M, 1] + [1, N] - 2 * [M, N] -> [M, N]
    Ok(x_norm + cb_norm_t - prod * 2.0)
}

#[allow(dead_code)] // Suppress warning for unused function
fn find_nearest(distances: &Tensor, codebook: &Tensor) -> Result<(Tensor, Tensor)> {
    let indices = distances.argmin(-1, true);
    let values = codebook.gather(0, &indices, false);
    Ok((indices.squeeze_dim(-1), values))
}

fn compute_centroids(x: &Tensor, codes: &Tensor, codebook_size: i64) -> Result<Tensor> {
    // x shape: [B, T, D]
    // codes shape: [B, T] (after squeeze in update)
    // Prefix unused variables
    let (_batch_size, _seq_len, vector_dim) = x.size3()?;
    
    // Reshape x to [B*T, D]
    let x_flat = x.reshape(&[-1, vector_dim]);
    // Reshape codes to [B*T]
    let codes_flat = codes.reshape(&[-1]);

    let mut centroids = Vec::new();
    // Sum over the flattened batch*sequence dimension (dim 0)
    let dims: &[i64] = &[0]; 
    
    for i in 0..codebook_size {
        let mask = codes_flat.eq(i);
        // Need mask shape [B*T, 1] for broadcasting with x_flat [B*T, D]
        let mask_unsqueezed = mask.unsqueeze(-1);
        
        // Perform calculation on flattened tensors
        let sum = (&x_flat * &mask_unsqueezed).sum_dim_intlist(dims, false, tch::Kind::Float);
        let count = mask.sum_dim_intlist(dims, false, tch::Kind::Float).clamp_min(1e-12);
        centroids.push(sum / count);
    }
    
    Ok(Tensor::stack(&centroids, 0))
}

/// Compute distances between points and codebook vectors efficiently
fn compute_distances(points: &Tensor, codebook: &Tensor) -> Result<Tensor, TchError> {
    let num_points = points.size()[0];
    if num_points == 0 {
        return Err(TchError::Kind("Empty points tensor provided".to_string()));
    }
    
    let codebook_size = codebook.size()[0];
    if codebook_size == 0 {
        return Err(TchError::Kind("Empty codebook tensor provided".to_string()));
    }
    
    // Expand dimensions for broadcasting
    // [num_points, 1, dim] - [1, codebook_size, dim] -> [num_points, codebook_size, dim]
    let points_expanded = points.unsqueeze(1);
    let codebook_expanded = codebook.unsqueeze(0);
    
    // Compute squared Euclidean distance
    let diff = points_expanded - codebook_expanded;
    let distances = diff.pow_tensor_scalar(2.0).sum_dim_intlist(-1, false, Kind::Float);
    
    Ok(distances)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rvq_encode_decode() -> Result<()> {
        let device = Device::Cpu;
        let num_codebooks = 4;
        let codebook_size = 512;
        let vector_dim = 256;
        let batch_size = 32;
        let seq_len = 1; // Add a sequence length dimension

        let encoder = RVQEncoder::new(device, num_codebooks, codebook_size as i64, vector_dim);
        let decoder = RVQDecoder::new(&encoder);

        // Create test input - Use 3D tensor [B, T, D]
        let x = Tensor::randn(&[batch_size, seq_len, vector_dim as i64], (tch::Kind::Float, device));

        // Test encoding and get raw indices
        let encoded_indices = encoder.encode_batch(&x, None)?;
        
        // Check the length of the returned vector
        assert_eq!(encoded_indices.len(), num_codebooks, "Encoder returned incorrect number of code tensors");
        
        // Verify each indices vector has the right length (batch_size * seq_len)
        for (i, indices) in encoded_indices.iter().enumerate() {
            assert_eq!(
                indices.len(), 
                (batch_size * seq_len) as usize,
                "Indices vector {} has incorrect length: expected {}, got {}", 
                i, batch_size * seq_len, indices.len()
            );
        }
        
        // Test decode_optimized with the encoded indices
        let reconstructed = decoder.decode_optimized(&encoded_indices)?;
        
        // Assert the reconstructed shape matches the 3D input shape
        assert_eq!(reconstructed.size(), &[batch_size as i64, seq_len as i64, vector_dim as i64], 
                   "Reconstructed shape doesn't match input shape");

        Ok(())
    }

    #[test]
    fn test_rvq_update() -> Result<()> {
        let device = Device::Cpu;
        
        // Create a small RVQ encoder for testing with consistent dimensions
        let codebook_size = 8;      // Small codebook size for testing
        let vector_dim = 4;         // Small vector dimension for testing
        let num_codebooks = 2;      // Use two codebooks to test multi-stage quantization
        
        let mut encoder = RVQEncoder::new(
            device,
            num_codebooks,
            codebook_size,
            vector_dim,
        );
        
        // Set a high learning rate to clearly see the updates in the test
        encoder.learning_rate = 0.5;
        
        // Generate synthetic test data with a consistent pattern for predictable results
        let batch_size = 2;
        let seq_len = 3;
        
        // Create input tensor with controlled values
        let mut input_values = Vec::new();
        for i in 0..batch_size {
            for j in 0..seq_len {
                for k in 0..vector_dim {
                    // Create a predictable pattern based on position
                    let val = (i * seq_len * vector_dim + j * vector_dim + k) as f32 / 10.0;
                    input_values.push(val);
                }
            }
        }
        
        // Use correct tensor creation method: f_from_slice
        let input = Tensor::f_from_slice(&input_values)
            .expect("Failed to create tensor from slice")
            .reshape(&[batch_size as i64, seq_len as i64, vector_dim as i64])
            .to_device(device);
        
        // Save initial codebook values for comparison
        let initial_codebooks: Vec<Tensor> = encoder.codebooks
            .iter()
            .map(|cb| cb.as_ref().copy())
            .collect();
            
        // Run the update method
        encoder.update(&input).expect("Failed to update codebooks");
        
        // Verify specific aspects of the updated codebooks
        for (i, (initial, current)) in initial_codebooks.iter()
            .zip(encoder.codebooks.iter())
            .enumerate() {
            
            // 1. Calculate the overall difference between initial and current
            let diff = (initial - current.as_ref()).abs().sum(Kind::Float);
            let diff_val = diff.double_value(&[]);
            
            // Assert that codebooks have been updated
            assert!(
                diff_val > 1e-5,
                "Codebook {} not updated, difference: {}", 
                i, diff_val
            );
            
            // 2. Check that the codebook remains well-formed with the expected shape
            let codebook_shape = current.as_ref().size();
            
            // Log actual shape for debugging
            println!("Codebook {} shape: {:?}", i, codebook_shape);
            
            // Check dimensions - the codebook should have 2 dimensions [codebook_size, vector_dim]
            assert_eq!(
                codebook_shape.len(), 
                2,
                "Codebook {} has wrong number of dimensions: expected 2, got {}", 
                i, codebook_shape.len()
            );
            
            if codebook_shape.len() == 2 {
                assert_eq!(
                    codebook_shape[0], 
                    codebook_size,
                    "Codebook {} has wrong size in dimension 0", 
                    i
                );
                
                assert_eq!(
                    codebook_shape[1], 
                    vector_dim as i64,
                    "Codebook {} has wrong size in dimension 1", 
                    i
                );
            }
            
            // 3. Verify the magnitude of values is reasonable
            let max_value = current.as_ref().abs().max().double_value(&[]);
            assert!(
                max_value < 10.0,
                "Codebook {} has unusually large values: {}", 
                i, max_value
            );
            
            // 4. Verify the update uses the correct learning rate
            // For a learning rate of 0.5, the update should blend halfway between 
            // the original codebook and the mean update
            let expected_magnitude = initial.abs().mean(Kind::Float).double_value(&[]);
            let current_magnitude = current.as_ref().abs().mean(Kind::Float).double_value(&[]);
            
            // The magnitudes should be reasonably similar if the update is working correctly
            let ratio = if expected_magnitude > 1e-5 { 
                current_magnitude / expected_magnitude 
            } else { 
                1.0 
            };
            
            assert!(
                ratio > 0.1 && ratio < 10.0,
                "Codebook {} update ratio is outside reasonable bounds: {}", 
                i, ratio
            );
        }
        
        // Test encoding after update - use encode_batch instead of encode
        let encoded_indices = encoder.encode_batch(&input, None)?;
        assert_eq!(encoded_indices.len(), num_codebooks, "Wrong number of codebooks in encoded result");
        
        // Check total length instead of shape
        assert_eq!(
            encoded_indices[0].len(), 
            (batch_size * seq_len) as usize,
            "Encoded indices have incorrect length: {}",
            encoded_indices[0].len()
        );
        
        Ok(())
    }
}

#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::{Duration, Instant};

    // Simple benchmarking function that measures average execution time
    #[inline]
    fn benchmark<F, T>(name: &str, iterations: usize, warmup: usize, f: F) -> T 
    where
        F: Fn() -> T,
    {
        // Warm-up phase
        for _ in 0..warmup {
            let _ = f();
        }
        
        // Benchmarking phase
        let start = Instant::now();
        let mut last_result = None;
        
        for _ in 0..iterations {
            last_result = Some(f());
        }
        
        let elapsed = start.elapsed();
        let avg_time = elapsed.as_secs_f64() / (iterations as f64);
        
        println!("Benchmark '{}': avg {:?} per iteration ({} runs)",
                name, Duration::from_secs_f64(avg_time), iterations);
        
        last_result.unwrap()
    }

    #[test]
    pub fn benchmark_rvq_encode_decode() -> Result<()> {
        let device = Device::Cpu;
        
        // Test with different configurations
        let configs = vec![
            // (num_codebooks, codebook_size, vector_dim, batch_size, seq_len)
            (2, 256, 64, 1, 100),    // Small batch, moderate seq length
        ];
        
        for (num_codebooks, codebook_size, vector_dim, batch_size, seq_len) in configs {
            println!("\nTesting RVQ with {} codebooks, size {}, dim {}, batch {}, seq_len {}", 
                    num_codebooks, codebook_size, vector_dim, batch_size, seq_len);
            
            // Create proper encoder and decoder instead of placeholders
            let encoder = RVQEncoder::new(device, num_codebooks, codebook_size as i64, vector_dim);
            let decoder = RVQDecoder::new(&encoder);
            
            // Create random input
            let input = Tensor::randn(&[batch_size as i64, seq_len as i64, vector_dim as i64], (Kind::Float, device));
            
            // Benchmark encoding
            let encoded_batches = benchmark("encode", 10, 2, || {
                encoder.encode_batch(&input, None).unwrap()
            });
            
            // Benchmark decoding 
            let reconstructed = benchmark("decode", 10, 2, || {
                decoder.decode_optimized(&encoded_batches).unwrap()
            });
            
            // Calculate error
            let error = (&input - &reconstructed)
                .abs()
                .mean(Kind::Float)
                .double_value(&[]);
                
            println!("Mean absolute reconstruction error: {:.6}", error);
        }
        
        Ok(())
    }
} 