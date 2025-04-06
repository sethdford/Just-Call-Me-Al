use anyhow::{Result, anyhow};
use tch::{Tensor, Device, Kind, TchError};
use crate::utils::tensor::tensor_to_vec_f32;
use tracing::{info, debug, trace, warn, span, Level};
use crate::rvq::{RVQEncoder, RVQDecoder, RVQConfig};
use crate::audio::codec::{MimiEncoder, MimiEncoderConfig, MimiEncoderState};
use std::path::Path;
use thiserror::Error;

// Declare the codec and streaming modules
pub mod codec;
pub mod streaming;

// Audio processor state for stateful processing
pub struct AudioProcessorState {
    pub mimi_encoder_state: Option<MimiEncoderState>,
    // Add other state components as needed
}

// Define AudioProcessorError
#[derive(Debug, Error)]
pub enum AudioProcessorError {
    #[error("Processing error: {0}")]
    ProcessingError(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Initialization error: {0}")]
    InitializationError(String),
    
    #[error("Device error: {0}")]
    DeviceError(String),
    
    #[error("Internal error: {0}")]
    InternalError(#[from] anyhow::Error),
    
    #[error("Tensor error: {0}")]
    TensorError(#[from] TchError)
}

// Default sample rate constant
const DEFAULT_SAMPLE_RATE: i64 = 24000;

// Make the struct public
pub struct AudioProcessor {
    target_sample_rate: u32,
    target_channels: u16,
    _max_batch_size: usize, // Prefixed unused field
    device: Device,
    mimi_encoder: MimiEncoder,
    rvq_encoder: RVQEncoder,
    _rvq_decoder: RVQDecoder, // Prefixed unused field
}

// SAFETY: This is marked Send + Sync because we intend for it to be used
// within an Arc<Mutex<...>>. All internal tch operations (which are not Sync)
// MUST happen only while the mutex is held. This requires careful usage
// patterns and potentially refactoring to ensure safety across await points.
// Consider using spawn_blocking for long-running tch operations.
unsafe impl Send for AudioProcessor {}
unsafe impl Sync for AudioProcessor {}

impl AudioProcessor {
    pub fn new(
        mimi_config: MimiEncoderConfig,
        rvq_config: RVQConfig,
        mimi_weights_path: Option<&Path>,
        rvq_weights_path: Option<&Path>,
        device: Device,
    ) -> Result<Self> {
        info!(
            "Initializing AudioProcessor with Device={:?}",
            device
        );
        info!("Mimi Config: {:?}", mimi_config);
        info!("RVQ Config: {:?}", rvq_config);

        let mut mimi_encoder = MimiEncoder::new(mimi_config.clone(), device)
            .map_err(|e| anyhow!("Failed to create MimiEncoder: {}", e))?;
        if let Some(path) = mimi_weights_path {
            info!("Loading Mimi weights from: {}", path.display());
            mimi_encoder.load_weights(path)
                .map_err(|e| anyhow!("Failed to load Mimi weights: {}", e))?;
        } else {
            warn!("Mimi weights path not provided, using uninitialized encoder.");
        }

        let encoder = if let Some(path) = rvq_weights_path {
             info!("Loading RVQ Encoder weights from: {}", path.display());
             RVQEncoder::load(path, device)? 
        } else {
             warn!("RVQ Encoder weights path not provided, using UNINITIALIZED encoder.");
             RVQEncoder::new(
                 device, 
                 rvq_config.num_codebooks, 
                 rvq_config.codebook_size as i64, 
                 rvq_config.vector_dim
             )
        };
        
        let decoder = RVQDecoder::new(&encoder);

        Ok(Self {
            target_sample_rate: mimi_config.sample_rate as u32,
            target_channels: mimi_config.input_channels as u16,
            _max_batch_size: 0,
            device,
            mimi_encoder,
            rvq_encoder: encoder,
            _rvq_decoder: decoder,
        })
    }

    pub fn get_sample_rate(&self) -> u32 {
        self.target_sample_rate
    }

    pub fn get_channels(&self) -> u16 {
        self.target_channels
    }

    pub fn process_audio(&self, audio: &[f32]) -> Result<Tensor> {
        // Validate input
        if audio.is_empty() {
            return Err(anyhow!("Empty audio input"));
        }

        // Convert to tensor and normalize
        let tensor = Tensor::f_from_slice(audio)
            .map_err(|e| anyhow!("Failed to create tensor: {}", e))?
            .to_device(self.device)
            .view([1, -1]);
            
        let normalized = self.normalize_audio(&tensor)?;
        
        // Resample if needed
        if (self.target_sample_rate as i64) != DEFAULT_SAMPLE_RATE {
            debug!("Resampling from {} Hz to {} Hz", self.target_sample_rate, DEFAULT_SAMPLE_RATE);
            self.resample(&normalized, DEFAULT_SAMPLE_RATE)
                .map_err(|e| anyhow!("Failed to resample audio: {}", e))
        } else {
            Ok(normalized)
        }
    }

    pub fn normalize_audio(&self, tensor: &Tensor) -> Result<Tensor> {
        let max_abs = tensor.abs().max();
        let max_val = max_abs.double_value(&[]);
        if max_val > 0.0 {
            tensor.f_div_scalar(max_val)
                .map_err(|e| anyhow!("Failed to normalize audio: {}", e))
        } else {
            Ok(tensor.shallow_clone())
        }
    }

    pub fn resample(&self, tensor: &Tensor, new_sample_rate: i64) -> Result<Tensor> {
        let old_len = tensor.size()[0];
        let new_len = (old_len as f64 * new_sample_rate as f64 / self.target_sample_rate as f64) as i64;
        
        // Create time indices using arange
        let _x = Tensor::arange(old_len, (Kind::Float, tensor.device()));
        let new_x = Tensor::arange(new_len, (Kind::Float, tensor.device()));
        
        // Calculate interpolation weights
        let scale = (old_len - 1) as f64 / (new_len - 1) as f64;
        let x_interp = new_x.f_mul_scalar(scale)
            .map_err(|e| anyhow!("Failed to calculate interpolation: {}", e))?;
        let x_floor = x_interp.f_floor()
            .map_err(|e| anyhow!("Failed to calculate floor: {}", e))?;
        let x_ceil = x_floor.f_add_scalar(1.0)
            .map_err(|e| anyhow!("Failed to calculate ceiling: {}", e))?;
        let weights = x_interp.f_sub(&x_floor)
            .map_err(|e| anyhow!("Failed to calculate weights: {}", e))?;
        
        // Get values at floor and ceil indices
        let values_floor = tensor.f_index_select(0, &x_floor.to_kind(Kind::Int64))
            .map_err(|e| anyhow!("Failed to select floor values: {}", e))?;
        let values_ceil = tensor.f_index_select(0, &x_ceil.to_kind(Kind::Int64))
            .map_err(|e| anyhow!("Failed to select ceil values: {}", e))?;
        
        // Linear interpolation
        let interp = values_floor
            .f_mul(&weights.f_neg()?.f_add_scalar(1.0)?)?
            .f_add(&values_ceil.f_mul(&weights)?)?;
            
        Ok(interp)
    }

    pub fn resample_tensor(&self, audio: &Tensor, from_rate: i32, to_rate: i32) -> Result<Tensor> {
        let ratio = to_rate as f64 / from_rate as f64;
        let new_len = (audio.size()[0] as f64 * ratio).round() as i64;
        
        let x = Tensor::arange(new_len, (Kind::Float, audio.device()));
        let x = x.f_div_scalar(ratio)?;
        
        let x0 = x.f_floor()?;
        let x1 = x.f_ceil()?;
        let w1 = x.f_sub(&x0)?;
        let w0 = Tensor::f_ones_like(&w1)?.f_sub(&w1)?;
        
        let x0 = x0.to_kind(Kind::Int64);
        let x1 = x1.to_kind(Kind::Int64);
        
        let y0 = audio.f_index_select(0, &x0)?;
        let y1 = audio.f_index_select(0, &x1)?;
        
        Ok(y0.f_mul(&w0.f_unsqueeze(-1)?)? + y1.f_mul(&w1.f_unsqueeze(-1)?)?)
    }

    pub fn to_tensor(&self, samples: &[f32], device: Device) -> Result<Tensor> {
        let tensor = Tensor::f_from_slice(samples)?.to_device(device);
        Ok(tensor)
    }

    pub fn from_tensor(&self, tensor: &Tensor) -> Result<Vec<f32>> {
        tensor_to_vec_f32(tensor).map_err(|e| anyhow!("Failed to convert tensor to vec: {}", e))
    }

    // Tokenize audio into tokens using MimiEncoder + RVQEncoder
    pub fn tokenize(
        &self, 
        input_tensor: &Tensor,
        state: Option<AudioProcessorState>
    ) -> Result<(Vec<Tensor>, AudioProcessorState)> {
        let _span = span!(Level::DEBUG, "AudioProcessor::tokenize").entered();
        trace!("Input tensor shape: {:?}", input_tensor.size());
        if input_tensor.kind() != Kind::Float {
            return Err(anyhow!("Input tensor must be of kind Float, got {:?}", input_tensor.kind()));
        }

        // --- 1. Pass audio through Mimi Encoder ---
        // Ensure input tensor has correct shape for MimiEncoder [B, C, T] or [B, T] -> [B, 1, T]
        let prepared_input = if input_tensor.dim() == 2 { // Assuming [B, T]
            input_tensor.unsqueeze(1) // Add channel dim -> [B, 1, T]
        } else if input_tensor.dim() == 3 { // Assuming [B, C, T]
            // TODO: Add check/handling if C != mimi_encoder.config.input_channels
            input_tensor.copy()
        } else {
            return Err(anyhow!("Input tensor must have 2 ([B, T]) or 3 ([B, C, T]) dimensions, got {:?}", input_tensor.size()));
        };
        trace!("Prepared input shape for Mimi: {:?}", prepared_input.size());

        // Get initial Mimi state if none provided, and flatten the Option<Option<T>>
        let initial_mimi_state = state.map(|s| s.mimi_encoder_state).flatten();
        
        let (mimi_embeddings, new_mimi_state) = self.mimi_encoder.forward(&prepared_input, initial_mimi_state)
            .map_err(|e| anyhow!("MimiEncoder forward pass failed: {}", e))?;
        trace!("Mimi output embeddings shape: {:?}", mimi_embeddings.size());

        // --- 2. Pass Mimi embeddings through RVQ Encoder ---
        let rvq_codes = self.rvq_encoder.encode(&mimi_embeddings)
            .map_err(|e| anyhow!("RVQEncoder encode failed: {}", e))?;
        trace!("RVQ output codes count: {}", rvq_codes.len());
        if !rvq_codes.is_empty() {
            trace!("First RVQ code shape: {:?}", rvq_codes[0].size());
        }

        // --- 3. Construct new state ---
        let new_state = AudioProcessorState {
            mimi_encoder_state: Some(new_mimi_state),
            // rvq_encoder_state: None, // Add RVQ state if it becomes stateful
        };

        // Return RVQ codes and the new state
        Ok((rvq_codes, new_state))
    }

    // Detokenize needs rethinking for streaming state - placeholder for now
    pub fn detokenize(&self, tokens: &Tensor) -> Result<Vec<f32>> {
        span!(Level::TRACE, "AudioProcessor::detokenize").in_scope(|| {
            trace!("Input tokens shape: {:?}", tokens.size());
            warn!("Detokenization not fully implemented yet, especially for streaming.");
            Ok(vec![]) 
        })
    }

    // Write audio to a WAV file
    pub fn write_wav(&self, filename: &str, samples: &[f32]) -> Result<()> {
        let spec = hound::WavSpec {
            channels: self.get_channels() as u16,
            sample_rate: self.get_sample_rate() as u32,
            bits_per_sample: 16, // Assume 16-bit for now, adjust if needed
            sample_format: hound::SampleFormat::Int, // Assume Int for now
        };

        let mut writer = hound::WavWriter::create(filename, spec)?;

        let scale = (1 << (spec.bits_per_sample - 1)) as f32;
        for sample in samples {
            let int_sample = (sample * scale).max(std::i16::MIN as f32).min(std::i16::MAX as f32) as i16;
            writer.write_sample(int_sample)?;
        }

        writer.finalize()?;
        Ok(())
    }

    // Read audio from a WAV file
    pub fn read_wav(&self, filename: &str) -> Result<(Vec<f32>, u32)> {
        let mut reader = hound::WavReader::open(filename)?;
        let spec = reader.spec();
        
        let samples = if spec.sample_format == hound::SampleFormat::Float {
            reader.samples::<f32>()
                  .collect::<std::result::Result<Vec<f32>, _>>()?
        } else {
            // Convert integer samples to float
            let scale = 1.0 / (1 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>()
                  .map(|s| s.map(|s| s as f32 * scale))
                  .collect::<std::result::Result<Vec<f32>, _>>()?
        };

        Ok((samples, spec.sample_rate))
    }

    pub fn bytes_to_samples(&self, bytes: &[u8]) -> Result<Vec<f32>> {
        // Assuming 16-bit PCM little-endian based on typical WAV usage
        let bit_depth = 16;
        let bytes_per_sample = (bit_depth / 8) as usize;
        if bytes.len() % bytes_per_sample != 0 {
            return Err(anyhow!("Byte slice length ({}) is not a multiple of bytes per sample ({})", bytes.len(), bytes_per_sample));
        }
        let num_samples = bytes.len() / bytes_per_sample;
        let mut samples = Vec::with_capacity(num_samples);
        
        for chunk in bytes.chunks_exact(bytes_per_sample) {
            if bit_depth == 16 {
                let sample_i16 = i16::from_le_bytes([chunk[0], chunk[1]]);
                samples.push(sample_i16 as f32 / 32768.0);
            } else {
                // TODO: Handle other bit depths if necessary
                return Err(anyhow!("Unsupported bit depth for bytes_to_samples: {}", bit_depth));
            }
        }
        Ok(samples)
    }

    pub fn samples_to_bytes(&self, samples: &[f32]) -> Result<Vec<u8>> {
        // Assuming 16-bit PCM little-endian
        let bit_depth = 16;
        let bytes_per_sample = (bit_depth / 8) as usize;
        let mut bytes = Vec::with_capacity(samples.len() * bytes_per_sample);
        
        for &sample in samples {
            if bit_depth == 16 {
                // Clamp sample to [-1.0, 1.0] before converting
                let clamped_sample = sample.max(-1.0).min(1.0);
                let int_sample = (clamped_sample * 32767.0) as i16;
                bytes.extend_from_slice(&int_sample.to_le_bytes());
            } else {
                 // TODO: Handle other bit depths if necessary
                return Err(anyhow!("Unsupported bit depth for samples_to_bytes: {}", bit_depth));
            }
        }
        Ok(bytes)
    }

    pub fn samples_to_frames(&self, num_samples: usize) -> usize {
        let sr = self.mimi_encoder.config.sample_rate;
        let frame_ms = self.mimi_encoder.config.frame_length_ms;
        let hop_ms = self.mimi_encoder.config.hop_length_ms;
        if sr <= 0.0 || frame_ms <= 0.0 || hop_ms <= 0.0 {
            warn!("Invalid sample rate ({}), frame length ({}), or hop length ({}) in Mimi config.", sr, frame_ms, hop_ms);
            return 0;
        }
        
        // Calculate frame and hop lengths in samples
        let frame_samples_f64 = sr * frame_ms as f64 / 1000.0;
        let hop_samples_f64 = sr * hop_ms as f64 / 1000.0;
        
        // Round to nearest usize, ensuring hop_samples is at least 1
        let frame_samples = frame_samples_f64.round() as usize;
        let hop_samples = (hop_samples_f64.round() as usize).max(1);
        
        // Debug prints
        println!(
            "samples_to_frames: num_samples={}, frame_samples={}, hop_samples={}",
            num_samples, frame_samples, hop_samples
        );

        if num_samples < frame_samples {
            println!("samples_to_frames: num_samples < frame_samples, returning 0 frames");
            return 0; 
        }

        // Use the same formula as MimiEncoder::forward
        let num_frames = (num_samples - frame_samples) / hop_samples + 1;
        println!("samples_to_frames: Calculated num_frames = {}", num_frames);
        num_frames
    }
}

pub struct Utterance {
    pub speaker: String,
    pub text: String,
    pub timestamp: f64,
    pub audio: Tensor,
}

impl Utterance {
    pub fn new(speaker: String, text: String, timestamp: f64, audio: Tensor) -> Self {
        Self {
            speaker,
            text,
            timestamp,
            audio,
        }
    }
}

// Comment out unresolved module for now
// pub mod player;

pub trait ChunkedAudioProcessing {
    fn process_chunk(&mut self, chunk: &[f32]) -> Result<()>;
}

pub fn resample_audio(tensor: &Tensor, new_len: i64) -> Result<Tensor> {
    let old_len = tensor.size()[0];
    if old_len == new_len {
        return Ok(tensor.shallow_clone());
    }

    let _x = Tensor::arange(old_len, (Kind::Float, tensor.device()));
    let step = (old_len as f64 - 1.0) / (new_len as f64 - 1.0);
    let new_x = Tensor::arange(new_len, (Kind::Float, tensor.device()))
        .f_mul_scalar(step)?;

    let x0 = new_x.floor();
    let x1 = new_x.ceil();
    let w1 = new_x - &x0;
    let w0 = Tensor::f_ones(&[new_len], (Kind::Float, tensor.device()))?.f_sub(&w1)?;

    let y0 = tensor.index(&[Some(x0.to_kind(Kind::Int64))]);
    let y1 = tensor.index(&[Some(x1.to_kind(Kind::Int64))]);

    Ok(y0.f_mul(&w0)?.f_add(&y1.f_mul(&w1)?)?)
}

#[cfg(test)]
mod tests {
    use super::*; 
    // Removed direct codec import, rely on super::*
    // use crate::audio::codec::{AudioProcessor, MimiEncoderConfig};
    use crate::rvq::RVQConfig; 
    use anyhow::Result;
    use tch::{Tensor, Kind, Device, IndexOp}; // Added IndexOp

    // Helper to create dummy audio data (Tensor directly)
    fn generate_dummy_audio(duration_secs: f32, sample_rate: i64, _channels: i64) -> Tensor {
        let num_samples = (duration_secs * sample_rate as f32) as i64;
        let shape = vec![1, num_samples]; 
        Tensor::randn(&shape, (Kind::Float, Device::Cpu))
    }

    #[test]
    fn test_audio_processor_integration() -> Result<()> {
        println!("Starting test_audio_processor_integration...");
        let device = Device::Cpu;
        
        // Set up RVQ config to match the model's codebook_dim from config.json
        let rvq_config = RVQConfig {
            vector_dim: 512, // Match Mimi's hidden_size from config.json
            num_codebooks: 8, // Default is fine
            codebook_size: 2048, // Match from config.json
            ..Default::default()
        };
        
        // Use config that matches the model's config.json values
        let mimi_config = MimiEncoderConfig {
            input_channels: 1, // Match audio_channels from config.json
            dimension: 512, // Must match hidden_size for proper tensor dimensions
            hidden_size: 512, // Match hidden_size from config.json
            hidden_channels: 512, // Match hidden_size
            sample_rate: 24000.0, // Match sampling_rate from config.json
            causal: true, // Match use_causal_conv from config.json
            kernel_size: 7, // Match kernel_size from config.json
            compress: 2, // Match compress from config.json
            num_hidden_layers: 8, // Match num_hidden_layers from config.json
            num_attention_heads: 8, // Match num_attention_heads from config.json
            head_dim: 64, // Match head_dim from config.json
            intermediate_size: 2048, // Match intermediate_size from config.json
            norm_eps: 1e-5, // Match norm_eps from config.json
            rope_theta: 10000.0, // Match rope_theta from config.json
            // These are still estimates, but should be close based on the frame_rate
            frame_length_ms: 80.0, // Based on frame_rate 12.5Hz
            hop_length_ms: 80.0,
            // Use other defaults
            ..Default::default()
        };

        // --- Add path for Mimi weights ---
        let mimi_weights_path = std::path::Path::new("models/mimi/model.safetensors");
        if !mimi_weights_path.exists() {
            warn!("Mimi weights file not found at {:?}, test might not load weights.", mimi_weights_path);
            // Or return Ok(()) to skip test: return Ok(()); 
            // Or panic: panic!("Mimi weights file not found");
        }

        println!("Creating AudioProcessor with config matching model.safetensors...");
        // Initialize processor using the new signature, providing the Mimi path
        let processor = AudioProcessor::new(
            mimi_config.clone(), 
            rvq_config,
            Some(mimi_weights_path), // Provide the path
            None, // rvq_weights_path (still None for now)
            device,
        )?;
        // --- End modification ---

        // Print dimensions for debugging
        println!("Mimi config dimension: {}, hidden_size: {}", mimi_config.dimension, mimi_config.hidden_size);
        println!("RVQ config vector_dim: {}", processor.rvq_encoder.vector_dim());

        let sample_rate = processor.get_sample_rate() as i64; // Get SR from processor
        let audio_duration = 0.2;
        // Generate audio with 1 channel, as specified in mimi_config
        let input_audio = generate_dummy_audio(audio_duration, sample_rate, 1); 
        let expected_samples = (audio_duration * sample_rate as f32) as usize;
        // Use the new samples_to_frames method
        let expected_frames = processor.samples_to_frames(expected_samples); 

        println!("Input audio shape: {:?}", input_audio.size());
        println!("Expected samples: {}, Calculated expected frames: {}", expected_samples, expected_frames);

        // Tokenize the audio tensor
        let (codes, _state) = processor.tokenize(&input_audio, None)?; 

        println!("Tokenization successful.");
        println!("Number of code tensors: {}", codes.len());

        // Use processor.rvq_encoder.num_codebooks()
        assert_eq!(codes.len(), processor.rvq_encoder.num_codebooks(), "Number of code tensors should match number of RVQ codebooks");

        if !codes.is_empty() {
            let first_code_tensor_size = codes[0].size();
             println!("First code tensor shape: {:?}", first_code_tensor_size);
             // Expected shape: [Batch=1, NumFrames]
             assert_eq!(first_code_tensor_size.len(), 2, "Code tensor should have 2 dimensions [B, T]");
             assert_eq!(first_code_tensor_size[0], 1, "Batch size should be 1");
             // Use the calculated expected_frames in the assertion
             assert_eq!(first_code_tensor_size[1], expected_frames as i64, 
                        "Number of frames mismatch (Output: {}, Expected: {})", 
                        first_code_tensor_size[1], expected_frames);
        }

        Ok(())
    }

    #[test]
    fn test_audio_processor_tokenize_streaming() -> Result<()> {
        println!("Starting test_audio_processor_tokenize_streaming...");
        let device = Device::Cpu;

        // Use RVQ config that matches the Mimi encoder output dimension
        let rvq_config = RVQConfig {
            vector_dim: 512, // Match Mimi's hidden_size
            num_codebooks: 8, // Default
            codebook_size: 2048, // Match from config.json
            ..Default::default()
        };

        // Use config that matches the test_audio_processor_integration test
        let mimi_config = MimiEncoderConfig {
            input_channels: 1, // Match audio_channels from config.json
            dimension: 512, // Must match hidden_size for proper tensor dimensions
            hidden_size: 512, // Match hidden_size from config.json
            hidden_channels: 512, // Match hidden_size
            sample_rate: 24000.0, // Match sampling_rate from config.json
            // For the streaming test, we'll use different frame/hop settings to test chunking
            frame_length_ms: 25.0, // Smaller frame size for more frames per chunk
            hop_length_ms: 10.0,   // Smaller hop for overlap
            // Use other defaults
            ..Default::default()
        };

        // Initialize processor using new signature (without loading weights for speed)
        let processor = AudioProcessor::new( // Removed unused mut
            mimi_config.clone(), 
            rvq_config,
            None, // mimi_weights_path - no weights for faster testing
            None, // rvq_weights_path
            device,
        )?;

        let sample_rate = processor.get_sample_rate() as i64;
        let total_duration = 0.5;
        let total_samples = (total_duration * sample_rate as f32) as i64;
        // Generate audio with 1 channel
        let full_audio = generate_dummy_audio(total_duration, sample_rate, 1); 
        println!("Full audio shape: {:?}", full_audio.size()); // Should be [1, N]

        // Split audio tensor into chunks
        let num_chunks = 3;
        let chunk_samples = total_samples / num_chunks;
        let mut chunks = Vec::new();
        for i in 0..num_chunks {
            let start = i * chunk_samples;
            let end = if i == num_chunks - 1 { total_samples } else { (i + 1) * chunk_samples };
            // Select along the time dimension (dim 1 for shape [1, T])
            // Use IndexOp trait via .i()
            let chunk = full_audio.i((.., start..end)); //Requires `use tch::IndexOp;`
            chunks.push(chunk);
            println!("Chunk {} shape: {:?}", i, chunks.last().unwrap().size());
        }

        let mut all_codes: Vec<Vec<Tensor>> = Vec::new();
        let mut processor_state: Option<AudioProcessorState> = None; // Use the correct state type

        // Process chunks sequentially
        for (i, chunk) in chunks.iter().enumerate() {
            println!("Processing chunk {}...", i);
            // Pass the tensor chunk to tokenize
            let (codes, new_state) = processor.tokenize(chunk, processor_state)?; 
            println!("Chunk {} produced {} code tensors.", i, codes.len());
            if !codes.is_empty() {
                println!("First code tensor shape for chunk {}: {:?}", i, codes[0].size());
            }

            all_codes.push(codes);
            processor_state = Some(new_state); // Update state for the next iteration
        }

        // --- Structural Checks ---
        assert_eq!(all_codes.len(), num_chunks as usize, "Should have results for each chunk");

        // Use the new samples_to_frames method
        let expected_total_frames = processor.samples_to_frames(total_samples as usize); 
        
        // Fix tensor cloning: Use map/collect instead of vec! macro
        let num_codebooks = processor.rvq_encoder.num_codebooks();
        let mut concatenated_codes: Vec<Tensor> = (0..num_codebooks)
            .map(|_| Tensor::zeros(&[1, 0], (Kind::Int64, device))) // Assuming codes are Int64
            .collect();

        for chunk_codes in all_codes {
             // Use processor.rvq_encoder.num_codebooks()
             assert_eq!(chunk_codes.len(), processor.rvq_encoder.num_codebooks(), "Each chunk should produce the correct number of codebooks");
             for (i, code_tensor) in chunk_codes.iter().enumerate() {
                 // Concatenate along the time dimension (dim 1)
                 concatenated_codes[i] = Tensor::cat(&[&concatenated_codes[i], code_tensor], 1);
             }
        }

        // --- Result Checks ---
        println!("Concatenated codes shapes:");
        let mut total_frames = 0;
        for (i, tensor) in concatenated_codes.iter().enumerate() {
            println!("  Codebook {}: {:?}", i, tensor.size());
            if i == 0 {
                total_frames = tensor.size()[1];
            }
        }

        println!("Total frames calculated from codes: {}", total_frames);
        println!("Expected total frames (calculated): {}", expected_total_frames);
        
        assert_eq!(total_frames, expected_total_frames as i64, 
                   "Total frames mismatch: have {}, expected {}", 
                   total_frames, expected_total_frames);

        Ok(())
    }

    // Remove unused helper function if test weights aren't loaded
    // #[allow(dead_code)] 
    // fn get_test_weights_path() -> PathBuf { ... }

} // end mod tests

// --- REMOVED Re-exports --- (No longer needed as imports at top + pub struct suffice)
// pub use self::codec::{MimiEncoderConfig, MimiEncoderState}; 
// pub use self::AudioProcessor; 

// Remove extraneous module declarations
// mod processor;
// mod streaming;

// Remove erroneous pub use for non-existent module
// pub use processor::AudioProcessor;
// Correct the pub use for the streaming module
pub use streaming::AudioStream;

// --- Define AudioProcessor Trait ---
#[async_trait::async_trait]
pub trait AudioProcessing: Send + Sync {
    async fn process_chunk(&mut self, samples: &[f32], sample_rate: u32)
        -> Result<AudioProcessorState, AudioProcessorError>;

    fn sample_rate(&self) -> u32;
}

// --- Implement AudioProcessing for AudioProcessor ---
#[async_trait::async_trait]
impl AudioProcessing for AudioProcessor {
    async fn process_chunk(&mut self, samples: &[f32], sample_rate: u32)
        -> Result<AudioProcessorState, AudioProcessorError> 
    {
        let _span = span!(Level::DEBUG, "AudioProcessor::process_chunk (trait impl)").entered();
        trace!("Processing chunk of {} samples at {} Hz", samples.len(), sample_rate);

        // Basic validation
        if samples.is_empty() {
            return Err(AudioProcessorError::InvalidInput("Input samples slice is empty".to_string()));
        }
        if sample_rate == 0 {
            return Err(AudioProcessorError::InvalidInput("Input sample_rate is zero".to_string()));
        }

        // Convert samples to tensor
        let input_tensor = self.to_tensor(samples, self.device)
            .map_err(AudioProcessorError::InternalError)?;

        let resampled_tensor = if sample_rate != self.target_sample_rate {
            debug!("Resampling chunk from {} Hz to {} Hz", sample_rate, self.target_sample_rate);
            // Fix error mapping - the Result from resample_tensor returns anyhow::Error not TchError
            self.resample_tensor(&input_tensor, sample_rate as i32, self.target_sample_rate as i32)
                .map_err(|e| AudioProcessorError::InternalError(e))? 
        } else {
            input_tensor
        };

        let batch_tensor = resampled_tensor.unsqueeze(0);

        let (rvq_codes, new_state) = self.tokenize(&batch_tensor, None)
            .map_err(AudioProcessorError::InternalError)?; 
        
        debug!("process_chunk produced {} RVQ code tensors (unused here).", rvq_codes.len());

        Ok(new_state)
    }

    fn sample_rate(&self) -> u32 {
        self.target_sample_rate
    }
}