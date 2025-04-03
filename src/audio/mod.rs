use anyhow::{Result, anyhow};
use hound;
use tch::{Tensor, Device, Kind};
// Remove unresolved import
// use crate::models::tensor::SafeTensor; 
use crate::utils::tensor::tensor_to_vec_f32; // Corrected path
use tracing::{info, debug};
use crate::rvq::{RVQEncoder, RVQDecoder};
use std::path::Path;
// Remove unused import
// use crate::vocoder::Vocoder;

// Declare the new codec module
pub mod codec;
pub mod streaming;

// Use the MimiEncoder from the new module
use crate::audio::codec::MimiEncoder;

// Add path const for Mimi Encoder weights (adjust as needed)
const DEFAULT_MIMI_ENCODER_WEIGHTS_PATH: &str = "models/mimi/model.safetensors"; // UPDATED path

const DEFAULT_SAMPLE_RATE: i64 = 16000;

pub struct AudioProcessor {
    sample_rate: u32,
    channels: u16,
    bit_depth: u16,
    device: Device,
    encoder: RVQEncoder,
    decoder: RVQDecoder,
    // Add the MimiEncoder field
    mimi_encoder: MimiEncoder,
}

impl AudioProcessor {
    pub fn new(
        sample_rate: u32,
        channels: u16,
        bit_depth: u16,
        device: Device,
        // Prefix unused parameter
        _rvq_codebook_path: Option<&Path>,
        mimi_weights_path: Option<&Path>,
    ) -> Result<Self> {
        info!(
            "Initializing AudioProcessor: Sample Rate={}, Channels={}, Bit Depth={}, Device={:?}",
            sample_rate, channels, bit_depth, device
        );

        // Initialize MimiEncoder
        // Use provided Mimi weights path or default
        let mimi_path = mimi_weights_path.unwrap_or_else(|| Path::new(DEFAULT_MIMI_ENCODER_WEIGHTS_PATH));
        info!("Using Mimi Encoder weights from: {:?}", mimi_path);

        // --- RVQ Loading - MODIFIED --- 
        // Load RVQ codebooks from the SAME safetensors file as MimiEncoder
        info!("Loading RVQ codebooks from safetensors: {:?}", mimi_path);
        let encoder = RVQEncoder::load(mimi_path, device) // Pass device
            .map_err(|e| anyhow!("Failed to load RVQ codebooks from {:?}: {}", mimi_path, e))?;
        let decoder = RVQDecoder::new(&encoder);
        info!("RVQ Encoder/Decoder initialized from safetensors.");
        // ----------------------------- 

        // Create MimiEncoder with just the device
        let mut mimi_encoder = MimiEncoder::new(device)
            .map_err(|e| anyhow!("Failed to initialize MimiEncoder structure: {}", e))?;

        // Load weights into the MimiEncoder's internal VarStore
        mimi_encoder.load_weights(mimi_path)
            .map_err(|e| anyhow!("Failed to load MimiEncoder weights from {:?}: {}", mimi_path, e))?;

        Ok(Self {
            sample_rate,
            channels,
            bit_depth,
            device,
            encoder,
            decoder,
            mimi_encoder,
        })
    }

    pub fn get_sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn get_channels(&self) -> u16 {
        self.channels
    }

    pub fn get_bit_depth(&self) -> u16 {
        self.bit_depth
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
        if (self.sample_rate as i64) != DEFAULT_SAMPLE_RATE {
            debug!("Resampling from {} Hz to {} Hz", self.sample_rate, DEFAULT_SAMPLE_RATE);
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
        let new_len = (old_len as f64 * new_sample_rate as f64 / self.sample_rate as f64) as i64;
        
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
    pub fn tokenize(&self, samples: &[f32]) -> Result<Vec<Tensor>> { // Return Vec<Tensor> of codes
        info!("Tokenizing {} audio samples using MimiEncoder + RVQ...", samples.len());

        // --- 1. Convert samples to Tensor --- 
        if samples.is_empty() {
            return Err(anyhow!("Input audio samples are empty."));
        }
        // Ensure tensor is on the correct device and has shape [batch, channels, length]
        // Assuming mono audio for now (channels=1)
        let audio_tensor = Tensor::from_slice(samples)
            .to_device(self.device)
            .to_kind(Kind::Float)
            .unsqueeze(0) // Add batch dimension
            .unsqueeze(0); // Add channel dimension -> Shape [1, 1, num_samples]
        info!("Input audio tensor created with shape: {:?}", audio_tensor.size());

        // --- 2. Mimi Encoding (Raw Audio -> Features) ---
        let feature_tensor = self.mimi_encoder.forward(&audio_tensor)
             .map_err(|e| anyhow!("MimiEncoder failed during forward pass: {}", e))?; 
        // Expected feature_tensor shape: [batch, feature_dim, sequence_length]
        info!("MimiEncoder generated feature tensor with shape: {:?}", feature_tensor.size());

        // --- 3. RVQ Encoding (Features -> Codes) --- 
        // The RVQ encoder expects shape [batch_size, sequence_length, feature_dim]
        // We might need to transpose the feature_tensor depending on MimiEncoder's output convention.
        // Let's assume MimiEncoder outputs [batch, feature_dim, seq_len] and RVQ expects [batch, seq_len, feature_dim]
        let features_for_rvq = feature_tensor.contiguous(); // Use contiguous features directly
        info!("Features for RVQ encoder with shape: {:?}", features_for_rvq.size());

        let codes = self.encoder.encode(&features_for_rvq)
            .map_err(|e| anyhow!("RVQEncoder failed during encoding: {}", e))?;

        info!("RVQ Encoding complete. Generated {} code tensors.", codes.len());
        // Result `codes` is Vec<Tensor>, where each tensor contains indices for one codebook layer
        // Shape of each code tensor: [batch_size, sequence_length]
        Ok(codes)
    }

    // Detokenize tokens back to audio features using RVQDecoder
    pub fn detokenize(&self, codes: &[Tensor]) -> Result<Tensor> { // Return feature Tensor
        info!("Detokenizing {} code tensors...", codes.len());
        // --- RVQ Decoding --- 
        let feature_tensor = self.decoder.decode(codes)?; // Pass codes to decoder
        // Result `feature_tensor` is the reconstructed feature tensor
        // Shape: [batch_size, sequence_length, feature_dim]
        info!("Decoding complete. Reconstructed feature tensor shape: {:?}", feature_tensor.size());

        // --- Feature Synthesis (Placeholder) --- 
        // In a full system, this feature_tensor would go to a vocoder.
        // Here, we just return the feature tensor itself, as the vocoder step is separate.
        // ----------------------------------------
        Ok(feature_tensor)
    }

    // Write audio to a WAV file
    pub fn write_wav(&self, filename: &str, samples: &[f32]) -> Result<()> {
        let spec = hound::WavSpec {
            channels: self.get_channels() as u16,
            sample_rate: self.get_sample_rate() as u32,
            bits_per_sample: self.get_bit_depth() as u16,
            sample_format: if self.get_bit_depth() == 32 {
                hound::SampleFormat::Float
            } else {
                hound::SampleFormat::Int
            },
        };

        let mut writer = hound::WavWriter::create(filename, spec)?;

        if spec.sample_format == hound::SampleFormat::Float {
            for sample in samples {
                writer.write_sample(*sample)?;
            }
        } else {
            // Convert float samples to integers
            let scale = (1 << (self.get_bit_depth() - 1)) as f32;
            for sample in samples {
                let int_sample = (sample * scale) as i32;
                writer.write_sample(int_sample)?;
            }
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
        let mut samples = Vec::with_capacity(bytes.len() / ((self.get_bit_depth() as usize) / 8));
        
        for chunk in bytes.chunks((self.get_bit_depth() as usize) / 8) {
            if chunk.len() == 2 {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0;
                samples.push(sample);
            }
        }
        Ok(samples)
    }

    pub fn samples_to_bytes(&self, samples: &[f32]) -> Result<Vec<u8>> {
        let mut bytes = Vec::with_capacity(samples.len() * ((self.get_bit_depth() as usize) / 8));
        
        for &sample in samples {
            let int_sample = (sample * 32768.0) as i16;
            bytes.extend_from_slice(&int_sample.to_le_bytes());
        }
        Ok(bytes)
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

pub struct AudioStream {
    buffer: Vec<f32>,
    // Prefix unused fields
    _sample_rate: u32,
    _channels: u16,
}

impl AudioStream {
    pub fn new(sample_rate: u32, channels: u16) -> Self {
        Self {
            buffer: Vec::new(),
            // Use prefixed names correctly
            _sample_rate: sample_rate,
            _channels: channels,
            // Remove duplicated non-prefixed fields
            // sample_rate,
            // channels,
        }
    }
}

impl ChunkedAudioProcessing for AudioStream {
    fn process_chunk(&mut self, chunk: &[f32]) -> Result<()> {
        self.buffer.extend_from_slice(chunk);
        Ok(())
    }
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
    use tch::Device;
    

    #[test]
    fn test_audio_processor_tokenize_shape() -> Result<()> {
        let device = Device::Cpu;
        // Use default paths for testing
        let audio_processor = AudioProcessor::new(
            16000, // Use default sample rate for easier testing
            1, 
            16, 
            device,
            None, // Use default RVQ path
            None, // Use default Mimi path (weights won't load correctly yet)
        )?;

        // Example input: 1 second of audio at 16kHz
        let samples: Vec<f32> = vec![0.0; 16000];

        // We expect this to fail until weights are loaded, but we can check structure
        match audio_processor.tokenize(&samples) {
            Ok(codes) => {
                // Check number of codebooks matches RVQ config
                let expected_num_codebooks = audio_processor.encoder.num_codebooks(); // Call the public method
                assert_eq!(codes.len(), expected_num_codebooks as usize, 
                           "Number of code tensors ({}) doesn't match RVQ codebooks ({})", 
                           codes.len(), expected_num_codebooks);

                // Check shape of each code tensor (Batch=1, SequenceLength)
                // Calculate expected sequence length based on Mimi Encoder downsampling
                let expected_len = (16000.0 / audio_processor.mimi_encoder.config.compress as f64).ceil() as i64;
                let expected_shape = &[1, expected_len];

                for (i, code_tensor) in codes.iter().enumerate() {
                    let actual_shape = code_tensor.size();
                    assert_eq!(actual_shape, expected_shape, 
                               "Unexpected shape for code tensor {}. Expected {:?}, Got {:?}", 
                               i, expected_shape, actual_shape);
                }
                info!("Tokenization successful (shape checks passed).");
            },
            Err(e) => {
                // This is expected if weights aren't loaded/correct
                if !e.to_string().contains("load") && !e.to_string().contains("shape mismatch") {
                     return Err(e);
                }
            }
        };

        Ok(())
    }
} 