use anyhow::Result;

// Function that demonstrates PyTorch functionality via tch
#[cfg(feature = "enable_tch")]
fn run_tch_demo() -> Result<()> {
    use tch::{Device, Kind, Tensor};
    
    println!("\n==== PyTorch 2.6 Demo (tch 0.19) ====");
    
    // Check CUDA availability and device
    let has_cuda = tch::Cuda::is_available();
    println!("CUDA available: {}", has_cuda);
    let device = Device::cuda_if_available();
    println!("Using device: {:?}", device);

    // Check autocast status (if CUDA is available)
    if has_cuda {
        // TODO: Verify correct function for checking autocast in tch 0.19 / PyTorch 2.6
        // println!("Autocast enabled: {}", tch::autocast::is_autocast_enabled(DeviceType::Cuda)); 
        println!("Autocast check skipped (needs verification)."); // Commented out for now
    } else {
        println!("Autocast status check skipped (CUDA not available).");
    }
    
    // Create a tensor
    let tensor = Tensor::zeros(&[3, 4], (Kind::Float, device));
    println!("Created tensor with shape: {:?}", tensor.size());
    
    // Simple operation
    let filled = Tensor::ones_like(&tensor);
    let result = tensor + filled;
    println!("Result of tensor addition: {:?}", result.size());
    
    println!("PyTorch demo completed successfully\n");
    Ok(())
}

#[cfg(not(feature = "enable_tch"))]
fn run_tch_demo() -> Result<()> {
    println!("\n==== PyTorch Demo Disabled ====");
    println!("Build with --features=enable_tch to enable\n");
    Ok(())
}

// Function that demonstrates Moshi functionality via Candle
#[cfg(feature = "enable_moshi")]
fn run_moshi_demo() -> Result<()> {
    use candle_core::{Device, Tensor, DType};
    
    println!("\n==== Moshi Demo (via Candle) ====");
    
    // Use CPU for simplicity
    let device = Device::Cpu;
    println!("Using device: {:?}", device);
    
    // Check if the model file exists
    let model_path = "models/mimi/model.safetensors";
    if !Path::new(model_path).exists() {
        println!("⚠️ Model file not found at: {}", model_path);
        println!("This demo will continue without loading weights.");
        println!("For a full test, download model weights to this location.");
    } else {
        println!("Found model at: {}", model_path);
    }
    
    // Create a dummy audio input (1 second of audio at 16kHz)
    // Avoid rand-dependent code paths by using deterministic data
    let sample_count = 16000; // 1 second of 16kHz audio
    let mut audio_data = vec![0.0f32; sample_count];
    
    // Fill with simple pattern
    for i in 0..sample_count {
        // Simple sine wave
        audio_data[i] = (i as f32 / 100.0).sin() * 0.5;
    }
    
    // Create tensor from our deterministic data
    let audio_input = Tensor::from_vec(
        audio_data, 
        &[1, sample_count as i64], 
        &device
    ).context("Failed to create audio tensor")?;
    
    println!("Created audio input tensor with shape: {:?}", audio_input.shape());
    
    // Time the operation
    let start = Instant::now();
    
    // Create a dummy output embedding tensor to simulate output
    let embedding_dim = 256;
    let frames = sample_count / 320; // Assuming 20ms frames at 16kHz
    
    println!("Simulating audio processing...");
    println!("Input audio: {} samples at 16kHz", sample_count);
    println!("Output representation would have shape: [1, {}, {}]", frames, embedding_dim);
    
    let duration = start.elapsed();
    println!("Processing completed in {:?}", duration);
    
    println!("Moshi demo completed successfully\n");
    Ok(())
}

#[cfg(not(feature = "enable_moshi"))]
fn run_moshi_demo() -> Result<()> {
    println!("\n==== Moshi Demo Disabled ====");
    println!("Build with --features=enable_moshi to enable\n");
    Ok(())
}

fn main() -> Result<()> {
    println!("Starting Hybrid ML Backend Demo");
    
    // Run PyTorch demo via tch
    if let Err(e) = run_tch_demo() {
        eprintln!("Error in PyTorch demo: {}", e);
    }
    
    // Run Moshi demo via Candle
    if let Err(e) = run_moshi_demo() {
        eprintln!("Error in Moshi demo: {}", e);
    }
    
    println!("Hybrid demo completed");
    Ok(())
} 