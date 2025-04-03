use anyhow::{Result, anyhow};
use tch::{Tensor, Device, Kind, TchError};
use std::path::Path;
use serde::{Deserialize, Serialize};
use std::fs::File;
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use tracing::{info, warn, error};

/// Configuration for RVQ encoder/decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RVQConfig {
    /// Number of codebooks
    pub num_codebooks: usize,
    /// Size of each codebook
    pub codebook_size: usize,
    /// Dimension of each vector
    pub vector_dim: usize,
    /// Whether to normalize input vectors
    pub normalize: bool,
    /// Learning rate for codebook updates
    pub learning_rate: f32,
    /// Device to run computations on
    #[serde(skip, default = "default_device")]
    pub device: Device,
}

fn default_device() -> Device {
    Device::Cpu
}

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
    codebooks: Vec<Tensor>,
    input_proj_weight: Tensor,
    input_proj_bias: Option<Tensor>,
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
            .map(|_| Tensor::randn(&[codebook_size, vector_dim as i64], (tch::Kind::Float, device)))
            .collect();
            
        Self {
            device,
            codebooks,
            num_codebooks,
            codebook_size,
            vector_dim,
            learning_rate: 0.01,
            input_proj_weight: Tensor::zeros(&[vector_dim as i64, vector_dim as i64], (tch::Kind::Float, device)),
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

        let mut codebooks = Vec::new();
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
                            codebooks.push(tensor);
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

        let input_proj_weight = tensors
            .tensor(proj_weight_name)
            .map_err(|e| anyhow!("Failed to find input projection weight tensor '{}': {}", proj_weight_name, e))
            .and_then(|view| tensor_view_to_tensor(&view, device).map_err(|e| anyhow!("Failed to convert input projection weight: {}", e)))?
            .squeeze_dim(-1);
        
        let input_proj_bias = match tensors.tensor(proj_bias_name) {
            Ok(view) => Some(tensor_view_to_tensor(&view, device)
                .map_err(|e| anyhow!("Failed to convert input projection bias: {}", e))?),
            Err(_) => {
                warn!("Input projection bias tensor '{}' not found. Assuming no bias.", proj_bias_name);
                None
            }
        };
        info!("Successfully loaded input projection weight. Bias {}found.", 
              if input_proj_bias.is_some() { "" } else { "not " });
        // --- ADDED LOG: Log projection weight shape ---
        info!("Loaded input_proj_weight shape: {:?}", input_proj_weight.size());
        // ----------------------------------------------

        if codebooks.is_empty() {
            return Err(anyhow!("No codebooks found or loaded from {:?}", path));
        }

        // Infer parameters from the first loaded codebook
        let first_codebook = &codebooks[0];
        let size = first_codebook.size();
        if size.len() != 2 {
            return Err(anyhow!("Invalid codebook tensor dimensions for layer 0: {:?}. Expected [codebook_size, vector_dim].", size));
        }
        let codebook_size = size[0];
        let vector_dim = size[1] as usize;
        let num_codebooks = codebooks.len();

        info!("Successfully loaded {} codebooks. Size={}, Dim={}", num_codebooks, codebook_size, vector_dim);
        // --- ADDED LOG: Log inferred codebook dimension ---
        info!("Inferred codebook vector_dim: {}", vector_dim);
        // -------------------------------------------------

        // --- ADDED CHECK: Compare projection output dim and codebook dim ---
        let proj_output_dim = input_proj_weight.size()[0]; // First dimension after transpose
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
            codebooks,
            input_proj_weight,
            input_proj_bias,
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
            .map(|(i, tensor)| (format!("codebook_{}", i), tensor))
            .collect();
        Tensor::write_npz(&tensors, path)?;
        Ok(())
    }

    /// Encode a batch of vectors into discrete codes
    pub fn encode(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        // Ensure input tensor is on the correct device
        let x = x.to_device(self.device);
        
        // --- Apply Input Projection --- 
        let transposed_weight = self.input_proj_weight.transpose(-2, -1).contiguous();
        let mut projected_x = x.matmul(&transposed_weight);

        // Add bias conditionally if it exists
        if let Some(bias) = &self.input_proj_bias {
            projected_x = projected_x + bias;
        }

        // Get original shapes
        let (batch_size, seq_len, _vector_dim) = projected_x.size3()?;
        
        let mut residual = projected_x; 
        let mut codes = Vec::with_capacity(self.num_codebooks);

        for codebook in &self.codebooks {
            let codebook = codebook.to_device(self.device);

            // Flatten input for distance calculation: [B, T, D] -> [B*T, D]
            let residual_flat = residual.view([-1, self.vector_dim as i64]);

            // Compute distances: input [B*T, D], codebook [N, D] -> output [B*T, N]
            let distances_flat = compute_distances_flat(&residual_flat, &codebook)?;
            
            // Find nearest indices: [B*T, N] -> [B*T]
            let indices_flat = distances_flat.argmin(1, false); // Argmin over codebook dim (N)
            
            // Get quantized vectors: Lookup [B*T] indices in [N, D] codebook -> [B*T, D]
            let quantized_vectors_flat = Tensor::embedding(&codebook, &indices_flat, -1, false, false);

            // Reshape quantized vectors back: [B*T, D] -> [B, T, D]
            let quantized_vectors = quantized_vectors_flat.view([batch_size, seq_len, -1]);

            // Update residual: [B, T, D] - [B, T, D]
            residual = residual - quantized_vectors;

            // Reshape indices and store: [B*T] -> [B, T]
            let indices = indices_flat.view([batch_size, seq_len]);
            codes.push(indices);
        }

        Ok(codes)
    }

    /// Update codebooks using encoded vectors
    pub fn update(&mut self, x: &Tensor, codes: &Tensor) -> Result<()> {
        let mut residual = x.shallow_clone();
        
        for (i, codebook) in self.codebooks.iter_mut().enumerate() {
            let current_codes = codes.select(0, i as i64);
            let centroids = compute_centroids(&residual, &current_codes, self.codebook_size as i64)?;
            *codebook = &*codebook * (1.0 - self.learning_rate) + &centroids * self.learning_rate;
            let selected = codebook.index_select(0, &current_codes.squeeze()); // Squeeze indices to 1D
            residual = &residual - &selected.unsqueeze(1); // Unsqueeze seq_len dim for subtraction
        }
        
        Ok(())
    }
}

/// RVQ decoder that converts discrete codes back into continuous vectors
pub struct RVQDecoder {
    device: Device,
    codebooks: Vec<Tensor>,
}

impl RVQDecoder {
    /// Create a new RVQ decoder
    pub fn new(encoder: &RVQEncoder) -> Self {
        Self {
            device: encoder.device,
            codebooks: encoder.codebooks.iter().map(|t| t.shallow_clone()).collect(),
        }
    }

    /// Decode discrete codes into continuous vectors
    pub fn decode(&self, codes: &[Tensor]) -> Result<Tensor> {
        if codes.is_empty() {
            return Err(anyhow!("Cannot decode empty code list"));
        }
        if codes.len() != self.codebooks.len() {
            return Err(anyhow!(
                "Number of codes ({}) does not match number of codebooks ({})",
                codes.len(), self.codebooks.len()
            ));
        }

        // Ensure all codes and codebooks are on the same device
        let target_device = self.device;
        let codes: Vec<Tensor> = codes.iter().map(|c| c.to_device(target_device)).collect();
        let codebooks: Vec<Tensor> = self.codebooks.iter().map(|cb| cb.to_device(target_device)).collect();

        let mut reconstructed_vector: Option<Tensor> = None;

        // Iterate through each codebook layer
        for (indices, codebook) in codes.iter().zip(codebooks.iter()) {
            // Dequantize: Look up the vectors corresponding to the indices
            let vectors = codebook.index_select(0, &indices.squeeze()); // Squeeze indices to 1D
            // Unsqueeze the seq_len dimension before checking shape and adding
            let vectors = vectors.unsqueeze(1); 

            // Add the vectors from this layer to the total reconstructed vector
            if let Some(ref mut recon) = reconstructed_vector {
                *recon = recon.f_add(&vectors)?;
            } else {
                // Initialize with the first layer's vectors
                reconstructed_vector = Some(vectors);
            }
        }

        reconstructed_vector.ok_or_else(|| anyhow!("Failed to reconstruct vector, possibly empty codes?"))
    }
}

// Helper functions for RVQ computations

#[allow(dead_code)] // Suppress warning for unused function
fn normalize_tensor(x: &Tensor) -> Result<Tensor> {
    let dims = vec![1i64];
    let norm = x.norm_scalaropt_dim(2, &dims[..], true);
    Ok(x / (norm + 1e-5))
}

// Flattened version of compute_distances
fn compute_distances_flat(x_flat: &Tensor, codebook: &Tensor) -> Result<Tensor> {
    // x_flat shape: [M, D] where M = B*T
    // codebook shape: [N, D]

    // Calculate norms (squared)
    let x_norm = x_flat.norm_scalaropt_dim(2.0, &[-1i64], true).pow_tensor_scalar(2.0); // Shape: [M, 1]
    let cb_norm = codebook.norm_scalaropt_dim(2.0, &[-1i64], true).pow_tensor_scalar(2.0); // Shape: [N, 1]
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

        let encoder = RVQEncoder::new(device, num_codebooks, codebook_size, vector_dim);
        let decoder = RVQDecoder::new(&encoder);

        // Create test input - Use 3D tensor [B, T, D]
        let seq_len = 1; // Add a sequence length dimension
        let x = Tensor::randn(&[batch_size, seq_len, vector_dim as i64], (tch::Kind::Float, device));

        // Test encoding
        let codes: Vec<Tensor> = encoder.encode(&x)?;
        // Check the length of the returned vector
        assert_eq!(codes.len(), num_codebooks, "Encoder returned incorrect number of code tensors");
        // Check the shape of the first code tensor (indices)
        assert_eq!(codes[0].size(), &[batch_size as i64, seq_len as i64], "Incorrect shape for code tensor");

        // Test decoding using the codes vector
        let reconstructed = decoder.decode(&codes)?;
        // Assert the reconstructed shape matches the 3D input shape
        assert_eq!(reconstructed.size(), &[batch_size as i64, seq_len as i64, vector_dim as i64]);

        Ok(())
    }

    #[test]
    fn test_rvq_update() -> Result<()> {
        let config = RVQConfig {
            num_codebooks: 2,
            codebook_size: 8,
            vector_dim: 4,
            learning_rate: 0.1,
            ..Default::default()
        };

        let mut encoder = RVQEncoder::new(config.device, config.num_codebooks, config.codebook_size as i64, config.vector_dim);
        
        // Create 3D test input [B, T, D]
        let batch_size = 16;
        let seq_len = 1;
        let x = Tensor::randn(&[batch_size, seq_len, config.vector_dim as i64], (tch::Kind::Float, config.device));
        
        // Encode and update
        let codes: Vec<Tensor> = encoder.encode(&x)?;
        encoder.update(&x, &Tensor::stack(&codes, 0))?; // Stack the Vec<Tensor> directly

        Ok(())
    }
} 