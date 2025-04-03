use anyhow::{Context, Result, anyhow};
use clap::Parser;
use csm::models::CSMModel;
use csm::models::CSMImpl;
use csm::models::config::CsmModelConfig;
use hound::{WavSpec, SampleFormat, WavWriter};
use std::path::PathBuf;
use std::time::Instant;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;
use tch::Device;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Text to synthesize.
    #[arg(long)]
    text: String,

    /// Path to the output WAV file.
    #[arg(long, short)]
    output: PathBuf,

    /// Path to the main CSM model weights (.safetensors).
    #[arg(long, default_value = "models/csm_model.safetensors")]
    model_path: PathBuf,

    /// Path to the main CSM model configuration (config.json).
    #[arg(long, default_value = "models/csm_config.json")]
    config_path: PathBuf,

    /// Path to the MiMi vocoder configuration (config.json) for sample rate.
    #[arg(long, default_value = "models/vocoder_config.json")]
    vocoder_config_path: PathBuf,

    /// Device to use (e.g., cpu, cuda, mps). Defaults to auto-detect.
    #[arg(long)]
    device: Option<String>,

    /// Seed for random number generation.
    #[arg(long)]
    seed: Option<u64>,

    /// Sampling temperature.
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let args = Args::parse();

    info!("Starting audio generation for text: {}", args.text);
    info!("Output file: {}", args.output.display());

    // --- Setup ---
    let device = match args.device.as_deref() {
        Some("cpu") => Device::Cpu,
        Some("cuda") | Some("gpu") if tch::utils::has_cuda() => Device::cuda_if_available(),
        Some("mps") if tch::utils::has_mps() => Device::Mps,
        None => { // Auto-detect: CUDA -> MPS -> CPU
            if tch::utils::has_cuda() {
                Device::cuda_if_available()
            } else if tch::utils::has_mps() {
                Device::Mps
            } else {
                Device::Cpu
            }
        }
        Some(other) => {
            // Handle cases where CUDA/MPS is requested but not available
            if (other == "cuda" || other == "gpu") && !tch::utils::has_cuda() {
                return Err(anyhow!("CUDA device requested ('{}') but not available.", other));
            } else if other == "mps" && !tch::utils::has_mps() {
                return Err(anyhow!("MPS device requested ('{}') but not available.", other));
            } else {
                return Err(anyhow!("Unsupported or unavailable device specified: {}", other));
            }
        }
    };
    
    let seed = args.seed;
    info!("Using device: {:?}", device);
    info!("Using seed: {:?}", seed);

    // --- Load Models and Configs ---
    info!("Loading CSM config from: {}", args.config_path.display());
    let _config = CsmModelConfig::from_file(&args.config_path)
        .map_err(|e| anyhow!("Failed to load config from {:?}: {}", args.config_path, e))?;

    info!("Loading CSM model from: {}", args.model_path.display());
    let model = CSMImpl::new(&args.model_path, device)
        .context("Failed to load CSM model")?;
    info!("CSM model loaded successfully.");

    info!("Loading configuration for sample rate...");
    let sample_rate = 24000;

    // --- Synthesize ---
    info!("Starting synthesis...");
    let start_time = Instant::now();

    let audio_samples_i16 = model.synthesize(
        &args.text,
        Some(args.temperature as f64),
        None,
        seed,
    ).await?;

    let duration = start_time.elapsed();
    info!("Synthesis completed in {:?}", duration);
    info!("Generated {} audio samples.", audio_samples_i16.len());

    if audio_samples_i16.is_empty() {
        anyhow::bail!("Synthesis resulted in empty audio output.");
    }

    // --- Write WAV File ---
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    info!("Writing WAV file to {} with spec: {:?}", args.output.display(), spec);
    let mut writer = WavWriter::create(&args.output, spec)
        .context(format!("Failed to create WAV writer for {}", args.output.display()))?;

    for sample_i16 in audio_samples_i16 {
        writer.write_sample(sample_i16)?;
    }

    writer.finalize().context("Failed to finalize WAV writer")?;
    info!("Successfully wrote WAV file: {}", args.output.display());

    Ok(())
} 