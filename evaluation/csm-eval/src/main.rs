use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json;
use std::path::{Path, PathBuf};
use async_std::fs;
use async_std::io::{self, WriteExt};
use std::sync::Arc;
use chrono::Instant;
use std::time::Duration;
use hound::{WavSpec, SampleFormat, WavWriter, Error as HoundError};
use anyhow::{Result, Context, anyhow};
use edit_distance::edit_distance;
use futures::stream::{self, StreamExt, TryStreamExt}; // For parallel processing
use rubato::{Resampler, SincFixedIn, SincInterpolationType, SincInterpolationParameters, WindowFunction}; // For resampling
use rand::Rng; // For placeholder MOS
use rustfft::{FftPlanner, num_complex::Complex};

// Import necessary items from the main `csm` crate
use csm::models::{CSMImpl, MoshiSpeechModel, Vocoder, SynthesisInput, SynthesisParams, MimiVocoder};
use csm::utils::device::get_device;

// Conditionally import ASR crate
#[cfg(feature = "asr")]
use whisper_rs::{WhisperContext, FullParams, SamplingStrategy};

// Placeholder struct for MOS predictor
struct MosPredictor;
impl MosPredictor {
    async fn load(path: &Path) -> Result<Self> {
        println!("    (Placeholder) Loading MOS predictor from {:?}...", path);
        // In a real scenario, load model weights here
        Ok(MosPredictor)
    }
    async fn predict(&self, _audio_data: &[f32], _sample_rate: u32) -> Result<f64> {
        println!("    (Placeholder) Predicting MOS score...");
        // TODO: Implement actual MOS prediction logic using the loaded model
        // For now, return a dummy score with some randomness
        let score = 4.0 + (rand::thread_rng().gen::<f64>() * 0.6 - 0.3); // Dummy score 3.7-4.3
        Ok(score)
    }
}

// --- Data Structures ---

/// Represents a single sample in the evaluation dataset
#[derive(Debug, Clone)]
struct EvaluationSample {
    id: String, // Unique identifier (e.g., filename without extension)
    audio_path: PathBuf,
    transcript: String,
    // Optional: Add speaker ID, metadata, etc. if needed
}

/// Represents a dataset definition in the config file
#[derive(Deserialize, Debug, Clone)]
struct DatasetConfigEntry {
    name: String,
    base_path: String,
    // Mutually exclusive ways to specify files:
    manifest_file: Option<String>, // Path relative to base_path
    files: Option<Vec<AudioTranscriptPair>>,
}

/// Represents a pair of audio and transcript file paths
#[derive(Deserialize, Debug, Clone)]
struct AudioTranscriptPair {
    audio: String,      // Path relative to base_path
    transcript: String, // Path relative to base_path
}

/// Represents the top-level structure of the dataset config file
#[derive(Deserialize, Debug, Clone)]
struct DatasetConfig {
    datasets: Vec<DatasetConfigEntry>,
}

/// Represents the expected structure of a line in a JSONL manifest file
#[derive(Deserialize, Debug, Clone)]
struct ManifestSample {
    audio_filepath: String,
    text: String,
    // Optional: Add duration, speaker_id, etc. if present in the manifest
    // duration: Option<f64>,
    // speaker_id: Option<String>,
}

/// Represents the vocoder configuration in the model config file
#[derive(Deserialize, Debug, Clone)]
struct VocoderConfig {
    model_type: String, // e.g., "Mimi"
    weights_path: String,
    // Add other vocoder-specific configs if needed
}

/// Refined Synthesis Params Config
#[derive(Deserialize, Debug, Clone, Default)]
struct SynthesisParamsConfig {
    /// Controls randomness (lower -> more deterministic). Example: 0.7
    temperature: Option<f64>,
    /// Nucleus sampling probability. Example: 0.9
    top_p: Option<f64>,
    /// Number of candidates for top-k sampling. Example: 50
    top_k: Option<i64>,
    /// Repetition penalty. Example: 1.1
    repetition_penalty: Option<f32>,
    /// Maximum number of new tokens to generate.
    max_new_tokens: Option<usize>,
    /// Seed for reproducibility.
    seed: Option<u64>,
    // Add other params as needed from the actual SynthesisParams struct
}

/// Represents the main model configuration in the model config file
#[derive(Deserialize, Debug, Clone)]
struct ModelConfig {
    model_type: String, // e.g., "MoshiSpeechModel" or "CSMImpl"
    weights_path: Option<String>, // Optional for composite models like CSMImpl
    config_path: Option<String>, // Optional for composite models like CSMImpl
    device: Option<String>, // "cpu" or "cuda"
    vocoder: Option<VocoderConfig>, // Optional, depends on model_type
    sample_rate: Option<u32>, // Add configurable sample rate
    asr_model_path: Option<String>, // Path to ASR model (e.g., Whisper ggml)
    mos_model_path: Option<String>, // Path to MOS prediction model (optional)
    synthesis_params: Option<SynthesisParamsConfig>, // Optional synthesis params
    // Add other model-specific configs if needed
}

/// Command-line arguments for the CSM Evaluation Pipeline
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the evaluation dataset configuration file (e.g., datasets.yaml)
    #[arg(short, long)]
    dataset_config: String,

    /// Path to the model configuration file or directory
    #[arg(short, long)]
    model_config: String,

    /// Directory to store evaluation results
    #[arg(short, long)]
    output_dir: String,

    /// Optional: Run only specific evaluation metrics (e.g., "wer,mos")
    #[arg(long)]
    metrics: Option<String>,

    /// Optional: Number of samples to evaluate from the dataset
    #[arg(long)]
    limit: Option<usize>,
    
    /// Optional: Number of samples to process in parallel
    #[arg(short, long, default_value_t = 4)]
    concurrency: usize,
}

// Structure to hold all loaded models and configs
struct LoadedCtx {
    speech_model: Arc<MoshiSpeechModel>,
    vocoder: Option<Arc<dyn Vocoder + Send + Sync>>,
    asr_context: Option<Arc<WhisperContext>>,
    mos_predictor: Option<Arc<MosPredictor>>,
    sample_rate: u32,
    synthesis_params: SynthesisParams,
}

// Updated SampleResult with SNR
#[derive(Debug, Clone, Serialize)]
struct SampleResult {
    id: String,
    transcript: String,
    synthesis_time_ms: u128,
    audio_length_ms: u64,
    rtf: f64,
    wer: Option<f64>,
    mos_score: Option<f64>,
    snr_db: Option<f64>, // Signal-to-Noise Ratio (placeholder)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("Starting CSM Evaluation Pipeline...");
    println!("  Dataset Config: {}", args.dataset_config);
    println!("  Model Config: {}", args.model_config);
    println!("  Output Directory: {}", args.output_dir);
    if let Some(metrics) = &args.metrics {
        println!("  Metrics: {}", metrics);
    }
    if let Some(limit) = args.limit {
        println!("  Limit: {}", limit);
    }
    println!("  Concurrency: {}", args.concurrency);

    // Conditionally print ASR status
    #[cfg(feature = "asr")]
    println!("  ASR (WER calculation) Enabled.");
    #[cfg(not(feature = "asr"))]
    println!("  ASR (WER calculation) Disabled. Build with --features asr to enable.");

    // 1. Load Dataset
    let samples = load_dataset(&args.dataset_config, args.limit).await
        .context("Failed to load dataset")?;

    // 2. Load Models & Context
    let ctx = load_evaluation_context(&args.model_config).await
        .context("Failed to load models and context")?;

    // Create output directory if it doesn't exist
    fs::create_dir_all(&args.output_dir).await
        .with_context(|| format!("Failed to create output directory: {}", args.output_dir))?;

    // 3. Run Evaluation Loop (now returns Vec<Result<SampleResult>>)
    let results_stream = run_evaluation_loop(&args, samples, ctx);
    let results: Vec<SampleResult> = results_stream
        .filter_map(|res| async move {
            match res {
                Ok(sample_res) => Some(sample_res),
                Err(e) => {
                    eprintln!("Error processing sample: {:?}. Skipping.", e);
                    None
                }
            }
        })
        .collect()
        .await;

    // 4. Generate Report
    generate_report(&args.output_dir, &results).await
        .context("Failed to generate report")?;

    println!("\nEvaluation Pipeline Finished.");
    println!("Results saved in {}", args.output_dir);

    Ok(())
}

async fn load_dataset(config_path: &str, limit: Option<usize>) -> Result<Vec<EvaluationSample>> {
    println!("\nLoading dataset from {}... (Limit: {:?})", config_path, limit);

    let yaml_content = fs::read_to_string(config_path).await
        .with_context(|| format!("Failed to read dataset config: {}", config_path))?;
    let config: DatasetConfig = serde_yaml::from_str(&yaml_content)
        .with_context(|| format!("Failed to parse YAML from dataset config: {}", config_path))?;

    let mut all_samples: Vec<EvaluationSample> = Vec::new();

    for dataset_entry in config.datasets {
        println!("  Processing dataset: {}", dataset_entry.name);
        let base_path = Path::new(&dataset_entry.base_path);

        if let Some(manifest_path_rel) = &dataset_entry.manifest_file {
            let manifest_path = base_path.join(manifest_path_rel);
            println!("    Loading from manifest: {:?}", manifest_path);
            
            let manifest_content = fs::read_to_string(&manifest_path).await
                .with_context(|| format!("Failed to read manifest file: {:?}", manifest_path))?;
            for line in manifest_content.lines() {
                if line.trim().is_empty() { continue; }
                
                match serde_json::from_str::<ManifestSample>(line) {
                    Ok(sample_info) => {
                        let audio_path_rel = Path::new(&sample_info.audio_filepath);
                        let audio_path = if audio_path_rel.is_absolute() {
                            audio_path_rel.to_path_buf()
                        } else {
                            base_path.join(audio_path_rel)
                        };
                        
                        let transcript = sample_info.text.trim().to_string();
                        let id = audio_path.file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or_else(|| sample_info.audio_filepath.as_str())
                            .to_string();
                            
                        all_samples.push(EvaluationSample { id, audio_path, transcript });
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to parse manifest line: '{}'. Error: {}. Skipping.", line, e);
                    }
                }
            }
        } else if let Some(files) = &dataset_entry.files {
            println!("    Loading from explicitly listed files...");
            for pair in files {
                let audio_path = base_path.join(&pair.audio);
                let transcript_path = base_path.join(&pair.transcript);

                let transcript = match fs::read_to_string(&transcript_path).await {
                    Ok(text) => text.trim().to_string(),
                    Err(e) => {
                        eprintln!("Warning: Failed to read transcript {:?}: {}. Skipping sample.", transcript_path, e);
                        continue;
                    }
                };

                let id = audio_path.file_stem().unwrap_or_default().to_string_lossy().to_string();
                all_samples.push(EvaluationSample { id, audio_path, transcript });
            }
        } else {
            eprintln!("Warning: Dataset '{}' must specify either 'manifest_file' or 'files'. Skipping.", dataset_entry.name);
        }
    }

    println!("  Total samples loaded initially: {}", all_samples.len());

    // Apply limit if provided
    if let Some(l) = limit {
        all_samples.truncate(l);
        println!("  Applied limit, samples to evaluate: {}", all_samples.len());
    }

    Ok(all_samples)
}

async fn load_evaluation_context(config_path: &str) -> Result<LoadedCtx> {
    println!("\nLoading evaluation context from {}...", config_path);

    let yaml_content = fs::read_to_string(config_path).await
        .with_context(|| format!("Failed to read model config: {}", config_path))?;
    let config: ModelConfig = serde_yaml::from_str(&yaml_content)
        .with_context(|| format!("Failed to parse YAML from model config: {}", config_path))?;

    let device_str = config.device.as_deref().unwrap_or("cpu");
    let device = get_device(device_str)?;
    println!("  Using device: {:?}", device);

    let config_dir = Path::new(config_path).parent().unwrap_or_else(|| Path::new("."));

    let sample_rate = config.sample_rate.unwrap_or(24000);
    println!("  Using sample rate: {}", sample_rate);

    // --- Load Speech Model ---
    let speech_model = if config.model_type == "MoshiSpeechModel" {
        let weights_path = config.weights_path
            .ok_or_else(|| anyhow!("weights_path required"))?;
        let config_file_path = config.config_path
            .ok_or_else(|| anyhow!("config_path required"))?;
            
        let full_weights_path = config_dir.join(weights_path);
        let full_config_path = config_dir.join(config_file_path);

        println!("  Loading MoshiSpeechModel...");
        println!("    Weights: {:?}", full_weights_path);
        println!("    Config: {:?}", full_config_path);

        Arc::new(MoshiSpeechModel::load(
            &full_weights_path,
            &full_config_path,
            device,
        ).await.context("Failed to load MoshiSpeechModel")?)
    } else {
        return Err(anyhow!("Unsupported model_type: {}", config.model_type));
    };

    // --- Load Vocoder --- 
    let vocoder: Option<Arc<dyn Vocoder + Send + Sync>> = if let Some(voc_config) = &config.vocoder {
        println!("  Loading Vocoder ({})...", voc_config.model_type);
        let full_voc_weights_path = config_dir.join(&voc_config.weights_path);
        println!("    Weights: {:?}", full_voc_weights_path);

        if voc_config.model_type == "Mimi" {
            let loaded_voc = MimiVocoder::load(&full_voc_weights_path, device).await
                .with_context(|| format!("Failed to load MimiVocoder from {:?}", full_voc_weights_path))?;
            Some(Arc::new(loaded_voc))
        } else {
            eprintln!("Warning: Unsupported vocoder_type: {}. Vocoder not loaded.", voc_config.model_type);
            None
        }
    } else {
        println!("  No vocoder specified in config.");
        None
    };

    // --- Load ASR Model (Conditional) ---
    let mut asr_context_opt = None;
    #[cfg(feature = "asr")]
    {
        if let Some(asr_model_path_str) = &config.asr_model_path {
            let asr_model_path = config_dir.join(asr_model_path_str);
            println!("  Loading ASR model from: {:?}", asr_model_path);
            if !asr_model_path.exists() {
                eprintln!("Warning: ASR model path not found: {:?}. WER will not be calculated.", asr_model_path);
            } else {
                // Assuming WhisperContext::new takes the model path
                let context = WhisperContext::new(asr_model_path.to_str().unwrap())
                    .with_context(|| format!("Failed to load Whisper model from {:?}", asr_model_path))?;
                asr_context_opt = Some(Arc::new(context));
                println!("  ASR model loaded successfully.");
            }
        } else {
            println!("  ASR model path not specified in config. WER will not be calculated.");
        }
    }

    // --- Load MOS Model --- 
    let mut mos_predictor_opt = None;
    if let Some(mos_model_path_str) = &config.mos_model_path {
        let mos_model_path = config_dir.join(mos_model_path_str);
        println!("  Loading MOS predictor model from: {:?}", mos_model_path);
        if !mos_model_path.exists() {
             eprintln!("Warning: MOS model path not found: {:?}. MOS scores will not be calculated.", mos_model_path);
        } else {
            match MosPredictor::load(&mos_model_path).await {
                Ok(predictor) => {
                    mos_predictor_opt = Some(Arc::new(predictor));
                    println!("  MOS predictor loaded successfully.");
                }
                Err(e) => {
                     eprintln!("Warning: Failed to load MOS predictor from {:?}: {:?}. MOS scores will not be calculated.", mos_model_path, e);
                }
            }
        }
    } else {
        println!("  MOS predictor path not specified. MOS scores will not be calculated.");
    }

    // --- Configure Synthesis Params ---
    let default_params = SynthesisParams::default();
    let synthesis_params = config.synthesis_params.map_or(default_params.clone(), |p_config| {
        SynthesisParams {
            temperature: p_config.temperature.unwrap_or(default_params.temperature),
            top_p: p_config.top_p.unwrap_or(default_params.top_p),
            top_k: p_config.top_k.unwrap_or(default_params.top_k),
            repetition_penalty: p_config.repetition_penalty.unwrap_or(default_params.repetition_penalty),
            max_new_tokens: p_config.max_new_tokens.unwrap_or(default_params.max_new_tokens),
            seed: p_config.seed.unwrap_or(default_params.seed),
            // Ensure all fields from SynthesisParams are covered or defaulted
            ..default_params // Use spread for any remaining default fields
        }
    });
    println!("  Using Synthesis Params: {:?}", synthesis_params);

    Ok(LoadedCtx {
        speech_model,
        vocoder,
        asr_context: asr_context_opt,
        mos_predictor: mos_predictor_opt,
        sample_rate,
        synthesis_params,
    })
}

// Updated to process sample and calculate SNR
async fn process_sample(
    sample: EvaluationSample,
    ctx: Arc<LoadedCtx>,
    audio_output_dir: PathBuf,
) -> Result<SampleResult> {
    let start_time = Instant::now();
    let input = SynthesisInput { text: sample.transcript.clone() };
    let params = ctx.synthesis_params.clone();

    let synthesis_result = ctx.speech_model.synthesize_audio(input, params).await;
    let synthesis_duration = start_time.elapsed();

    match synthesis_result {
        Ok(audio_data) => {
            let audio_length_samples = audio_data.len();
            let audio_length_ms = (audio_length_samples as u64 * 1000) / ctx.sample_rate as u64;
            let synthesis_time_ms = synthesis_duration.as_millis();
            let rtf = if synthesis_time_ms > 0 {
                (audio_length_ms as f64 / 1000.0) / (synthesis_time_ms as f64 / 1000.0)
            } else { f64::INFINITY };

            // --- Save Audio ---
            let audio_filename = format!("{}.wav", sample.id);
            let audio_save_path = audio_output_dir.join(&audio_filename);
            save_wav(&audio_save_path, &audio_data, ctx.sample_rate)
                .with_context(|| format!("Failed to save WAV file: {:?}", audio_save_path))?;
            println!("    Saved audio to: {:?}", audio_save_path);

            // --- Run ASR & Calculate WER (Conditional) ---
            let mut wer = None;
            #[cfg(feature = "asr")]
            {
                if let Some(asr_ctx) = &ctx.asr_context {
                     println!("    Running ASR for sample: {}", sample.id);
                    match run_asr(asr_ctx.clone(), &audio_data, ctx.sample_rate).await {
                        Ok(recognized_text) => {
                            wer = Some(calculate_wer(&sample.transcript, &recognized_text));
                            println!("      WER: {:.4}", wer.unwrap_or(f64::NAN));
                        }
                        Err(e) => eprintln!("    ASR failed for sample {}: {:?}. Skipping WER.", sample.id, e),
                    }
                }
            }
            
            // --- Predict MOS Score --- 
            let mut mos_score = None;
            if let Some(predictor) = &ctx.mos_predictor {
                 println!("    Predicting MOS for sample: {}", sample.id);
                match predictor.predict(&audio_data, ctx.sample_rate).await {
                    Ok(score) => {
                        mos_score = Some(score);
                        println!("      Predicted MOS: {:.3}", score);
                    }
                    Err(e) => eprintln!("    MOS prediction failed for sample {}: {:?}. Skipping MOS.", sample.id, e),
                }
            }
            
            // --- Calculate SNR (Placeholder) ---
            let snr_db = calculate_snr_db(&audio_data, ctx.sample_rate);
            if let Some(snr) = snr_db {
                 println!("    Estimated SNR: {:.2} dB", snr);
            }

            Ok(SampleResult {
                id: sample.id.clone(),
                transcript: sample.transcript.clone(),
                synthesis_time_ms,
                audio_length_ms,
                rtf,
                wer,
                mos_score,
                snr_db, // Add SNR
            })
        }
        Err(e) => Err(anyhow!(e)).context(format!("Synthesis failed for sample {}", sample.id)),
    }
}

// Updated to use futures::stream for parallelism
fn run_evaluation_loop(
    args: &Args,
    samples: Vec<EvaluationSample>,
    ctx: LoadedCtx,
) -> impl StreamExt<Item = Result<SampleResult>> { // Return a stream
    println!(
        "\nRunning evaluation loop for {} samples (Concurrency: {})...",
        samples.len(),
        args.concurrency
    );

    let shared_ctx = Arc::new(ctx);
    let audio_output_dir = Path::new(&args.output_dir).join("audio");

    stream::iter(samples)
        .map(move |sample| {
            let ctx_clone = shared_ctx.clone();
            let audio_dir_clone = audio_output_dir.clone();
            // Wrap the async block in tokio::spawn if significant CPU work occurs outside await points
            // For now, map directly to the async function
            async move {
                println!("  Scheduling sample: {}", sample.id);
                let result = process_sample(sample, ctx_clone, audio_dir_clone).await;
                // Log result immediately after processing, before buffer/collect
                 match &result {
                    Ok(sr) => println!("  Finished sample: {} (WER: {:?})", sr.id, sr.wer),
                    Err(e) => println!("  Failed sample processing: {:?}", e),
                 };
                 result
            }
        })
        .buffer_unordered(args.concurrency) // Process samples in parallel
}

async fn generate_report(
    output_dir: &str,
    results: &[SampleResult],
) -> Result<()> {
    println!("\nGenerating report in {}...", output_dir);

    if results.is_empty() {
        println!("  No results to generate report from.");
        return Ok(());
    }

    // --- Calculate Aggregated Metrics --- 
    let total_samples = results.len();
    let total_synthesis_time_ms: u128 = results.iter().map(|r| r.synthesis_time_ms).sum();
    let total_audio_length_ms: u64 = results.iter().map(|r| r.audio_length_ms).sum();
    
    let avg_synthesis_time_ms = total_synthesis_time_ms as f64 / total_samples as f64;
    let overall_rtf = if total_synthesis_time_ms > 0 {
         (total_audio_length_ms as f64 / 1000.0) / (total_synthesis_time_ms as f64 / 1000.0)
    } else { f64::INFINITY };
    
    let valid_wers: Vec<f64> = results.iter().filter_map(|r| r.wer).filter(|w| !w.is_nan()).collect();
    let avg_wer = if !valid_wers.is_empty() { Some(valid_wers.iter().sum::<f64>() / valid_wers.len() as f64) } else { None };

    let valid_mos: Vec<f64> = results.iter().filter_map(|r| r.mos_score).collect();
    let avg_mos = if !valid_mos.is_empty() { Some(valid_mos.iter().sum::<f64>() / valid_mos.len() as f64) } else { None };

    let valid_snrs: Vec<f64> = results.iter().filter_map(|r| r.snr_db).collect();
    let avg_snr = if !valid_snrs.is_empty() { Some(valid_snrs.iter().sum::<f64>() / valid_snrs.len() as f64) } else { None };

    // --- Generate Report Content ---
    let mut report = String::new();
    report.push_str("# Evaluation Report\n\n");
    report.push_str(&format!("Total Samples Evaluated: {}\n", total_samples));
    report.push_str(&format!("Average Synthesis Time: {:.2} ms\n", avg_synthesis_time_ms));
    report.push_str(&format!("Overall Real-Time Factor (RTF): {:.3}\n", overall_rtf));
    if let Some(wer) = avg_wer {
        report.push_str(&format!("Average WER: {:.4}\n", wer));
    }
    if let Some(mos) = avg_mos {
        report.push_str(&format!("Average Predicted MOS: {:.3}\n", mos));
    }
    if let Some(snr) = avg_snr {
        report.push_str(&format!("Average Estimated SNR: {:.2} dB\n", snr));
    }

    report.push_str("\n## Per-Sample Results\n\n");
    report.push_str("See results.json for detailed per-sample metrics.\n");

    // --- Save Report Files --- 
    let report_path = Path::new(output_dir).join("evaluation_report.md");
    let mut report_file = fs::File::create(&report_path).await?;
    report_file.write_all(report.as_bytes()).await?;
    println!("  Markdown report saved.");

    let results_path = Path::new(output_dir).join("results.json");
    let json_results = serde_json::to_string_pretty(results)?;
    let mut json_file = fs::File::create(&results_path).await?;
    json_file.write_all(json_results.as_bytes()).await?;
    println!("  JSON results saved.");

    Ok(())
}

// --- Helper Function to Save WAV --- 
fn save_wav(path: &Path, audio: &[f32], sample_rate: u32) -> Result<(), HoundError> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(path, spec)?;
    for sample in audio {
        writer.write_sample(*sample)?;
    }
    writer.finalize()
}

// --- Updated ASR Function (Conditional) ---
#[cfg(feature = "asr")]
async fn run_asr(ctx: Arc<WhisperContext>, audio_data: &[f32], sample_rate: u32) -> Result<String> {
    let target_sr = 16000u32;
    let resampled_audio = if sample_rate == target_sr {
        // No resampling needed
        audio_data.to_vec() 
    } else {
        println!("    Resampling audio from {}Hz to {}Hz for ASR...", sample_rate, target_sr);
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };
        // Using SincFixedIn for potentially better quality
        let mut resampler = SincFixedIn::<f32>::new(
            target_sr as f64 / sample_rate as f64,
            2.0, // Adjust max_resample_ratio if needed
            params,
            audio_data.len(),
            1, // Number of channels
        )?;

        let waves_in = vec![audio_data.to_vec()]; // Wrap in Vec<Vec<f32>>
        let waves_out = resampler.process(&waves_in, None)?;
        waves_out.into_iter().next().unwrap_or_default() // Get the first channel's data
    };

    let mut state = ctx.create_state().context("Failed to create ASR state")?;
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    state.full(params, &resampled_audio).context("Failed to run full transcription")?;

    let num_segments = state.full_n_segments().context("Failed to get number of segments")?;
    let mut recognized_text = String::new();
    for i in 0..num_segments {
        if let Ok(segment) = state.full_get_segment_text(i) {
            recognized_text.push_str(&segment);
        } else {
            eprintln!("Warning: Failed to get segment {}", i);
        }
    }

    Ok(recognized_text.trim().to_string())
}

// Provide a stub for when the 'asr' feature is not enabled
#[cfg(not(feature = "asr"))]
async fn run_asr(_ctx: Option<()>, _audio_data: &[f32], _sample_rate: u32) -> Result<String> {
    // Return empty string or specific indicator when ASR is disabled
    Ok("[ASR_DISABLED]".to_string())
}

// --- WER Calculation --- 
fn calculate_wer(reference: &str, hypothesis: &str) -> f64 {
    if hypothesis == "[ASR_DISABLED]" {
        return f64::NAN; // Not a Number indicates WER couldn't be calculated
    }
    let ref_words: Vec<&str> = reference.trim().split_whitespace().filter(|s| !s.is_empty()).collect();
    let hyp_words: Vec<&str> = hypothesis.trim().split_whitespace().filter(|s| !s.is_empty()).collect();

    if ref_words.is_empty() {
        return if hyp_words.is_empty() { 0.0 } else { 1.0 }; // Handle empty reference
    }
    if hyp_words.is_empty() {
        return 1.0; // All words deleted
    }

    // Use word-level edit distance for standard WER
    let distance = edit_distance::edit_distance(&ref_words, &hyp_words);
    distance as f64 / ref_words.len() as f64
}

// --- Placeholder MOS Prediction Function --- 
async fn predict_mos(predictor: Arc<MosPredictor>, audio_data: &[f32], sample_rate: u32) -> Result<f64> {
    // Call the predictor's method
    predictor.predict(audio_data, sample_rate).await
}

// --- SNR Calculation (Basic Placeholder) --- 
// This is a very basic estimation assuming silence/noise at start/end
fn calculate_snr_db(audio_data: &[f32], sample_rate: u32) -> Option<f64> {
    if audio_data.is_empty() { return None; }

    // Parameters for noise estimation (e.g., first/last 100ms)
    let noise_duration_ms = 100;
    let noise_samples = (sample_rate as f64 * (noise_duration_ms as f64 / 1000.0)) as usize;
    let min_len_for_noise = noise_samples * 2 + 100; // Need enough samples for start/end noise + some signal

    if audio_data.len() < min_len_for_noise { return None; }

    // Estimate noise power from start and end segments
    let start_noise = &audio_data[0..noise_samples];
    let end_noise = &audio_data[audio_data.len() - noise_samples..];
    let noise_power = (start_noise.iter().chain(end_noise.iter())
        .map(|&s| (s as f64).powi(2))
        .sum::<f64>()) / (noise_samples * 2) as f64;

    // Estimate signal power from the whole clip
    let signal_power = audio_data.iter()
        .map(|&s| (s as f64).powi(2))
        .sum::<f64>() / audio_data.len() as f64;

    // Ensure signal power is greater than noise power to avoid log(<=0)
    if signal_power <= noise_power || noise_power <= 1e-10 { 
        // If signal is weaker than noise or noise is near zero, SNR is effectively 0 or undefined
        // Return 0 dB as a reasonable floor, or None if preferred
        return Some(0.0); 
    }

    // SNR = Signal Power / Noise Power
    // SNR (dB) = 10 * log10(SNR)
    let snr = signal_power / noise_power;
    Some(10.0 * snr.log10())
}

// --- Test Module Skeleton ---
#[cfg(test)]
mod tests {
    use super::*; // Import items from outer scope
    use tempfile::tempdir; // For creating temporary directories
    use std::fs::File;
    use std::io::Write;
    use async_std::task;

    // Helper function to create a dummy YAML config file
    fn create_dummy_dataset_config(dir: &Path, content: &str) -> PathBuf {
        let config_path = dir.join("datasets.yaml");
        let mut file = File::create(&config_path).expect("Failed to create dummy config file");
        writeln!(file, "{}", content).expect("Failed to write to dummy config file");
        config_path
    }

    // Helper function to create dummy audio/transcript files
    fn create_dummy_sample_files(dir: &Path, base_name: &str, transcript_content: &str) -> (PathBuf, PathBuf) {
        let audio_path = dir.join(format!("{}.wav", base_name));
        let transcript_path = dir.join(format!("{}.txt", base_name));
        
        // Create empty audio file (content doesn't matter for loading test)
        File::create(&audio_path).expect("Failed to create dummy audio file");
        
        let mut file = File::create(&transcript_path).expect("Failed to create dummy transcript file");
        writeln!(file, "{}", transcript_content).expect("Failed to write to dummy transcript file");
        
        (audio_path, transcript_path)
    }

    #[test]
    fn test_calculate_wer_basic() {
        let ref_text = "hello world";
        let hyp_text = "hello there world";
        // Expected distance = 1 (insertion), words = 2 -> WER = 0.5
        assert!((calculate_wer(ref_text, hyp_text) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_calculate_wer_empty() {
        assert!((calculate_wer("", "") - 0.0).abs() < 1e-6);
        assert!((calculate_wer("a b", "") - 1.0).abs() < 1e-6); // Deletion
        assert!((calculate_wer("", "a b") - 1.0).abs() < 1e-6); // Insertion
    }

    #[test]
    fn test_calculate_wer_disabled_asr() {
        assert!(calculate_wer("test", "[ASR_DISABLED]").is_nan());
    }

    #[tokio::test]
    async fn test_load_dataset_explicit_files() {
        let dir = tempdir().unwrap();
        let base_path = dir.path().join("data");
        async_std::fs::create_dir_all(&base_path).await.unwrap();

        create_dummy_sample_files(&base_path, "sample1", "this is the first transcript");
        create_dummy_sample_files(&base_path, "sample2", "this is the second");

        let config_content = format!(r#"
        datasets:
          - name: "test_dataset"
            base_path: "{}"
            files:
              - audio: "sample1.wav"
                transcript: "sample1.txt"
              - audio: "sample2.wav"
                transcript: "sample2.txt"
        "#, base_path.to_str().unwrap().replace('\',"/")); // Ensure forward slashes for YAML
        
        let config_path = create_dummy_dataset_config(dir.path(), &config_content);

        let samples = load_dataset(config_path.to_str().unwrap(), None).await.unwrap();

        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].id, "sample1");
        assert_eq!(samples[0].transcript, "this is the first transcript");
        assert!(samples[0].audio_path.ends_with("sample1.wav"));
        assert_eq!(samples[1].id, "sample2");
        assert_eq!(samples[1].transcript, "this is the second");
        assert!(samples[1].audio_path.ends_with("sample2.wav"));
    }
    
    #[tokio::test]
    async fn test_load_dataset_manifest() {
        let dir = tempdir().unwrap();
        let base_path = dir.path().join("data");
        async_std::fs::create_dir_all(&base_path).await.unwrap();

        let manifest_path = dir.path().join("manifest.jsonl");
        let mut manifest_file = File::create(&manifest_path).unwrap();
        writeln!(manifest_file, "{{\"audio_filepath\": \"rel/path/audio1.wav\", \"text\": \"transcript one\"}}").unwrap();
        writeln!(manifest_file, "{{\"audio_filepath\": \"rel/path/audio2.wav\", \"text\": \"transcript two\"}}").unwrap();

        // Create dummy audio files referenced by manifest
        let audio_dir = base_path.join("rel/path");
        async_std::fs::create_dir_all(&audio_dir).await.unwrap();
        File::create(audio_dir.join("audio1.wav")).unwrap();
        File::create(audio_dir.join("audio2.wav")).unwrap();

        let config_content = format!(r#"
        datasets:
          - name: "manifest_test"
            base_path: "{}"
            manifest_file: "../manifest.jsonl" # Relative to base_path
        "#, base_path.to_str().unwrap().replace('\', "/"));
        
        let config_path = create_dummy_dataset_config(dir.path(), &config_content);
        
        let samples = load_dataset(config_path.to_str().unwrap(), None).await.unwrap();

        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].id, "audio1");
        assert_eq!(samples[0].transcript, "transcript one");
        assert!(samples[0].audio_path.ends_with("audio1.wav"));
         assert_eq!(samples[1].id, "audio2");
        assert_eq!(samples[1].transcript, "transcript two");
        assert!(samples[1].audio_path.ends_with("audio2.wav"));
    }
    
    #[tokio::test]
    async fn test_load_dataset_limit() {
        let dir = tempdir().unwrap();
        let base_path = dir.path().join("data");
        async_std::fs::create_dir_all(&base_path).await.unwrap();

        create_dummy_sample_files(&base_path, "s1", "t1");
        create_dummy_sample_files(&base_path, "s2", "t2");
        create_dummy_sample_files(&base_path, "s3", "t3");

        let config_content = format!(r#"
        datasets:
          - name: "limit_test"
            base_path: "{}"
            files:
              - audio: "s1.wav"
                transcript: "s1.txt"
              - audio: "s2.wav"
                transcript: "s2.txt"
              - audio: "s3.wav"
                transcript: "s3.txt"
        "#, base_path.to_str().unwrap().replace('\',"/")); 
        let config_path = create_dummy_dataset_config(dir.path(), &config_content);

        let samples = load_dataset(config_path.to_str().unwrap(), Some(2)).await.unwrap();
        assert_eq!(samples.len(), 2);
    }

    #[tokio::test]
    async fn test_generate_report() {
        let dir = tempdir().unwrap();
        let output_dir = dir.path();

        let results = vec![
            SampleResult {
                id: "sample1".to_string(),
                transcript: "ref one".to_string(),
                synthesis_time_ms: 150,
                audio_length_ms: 3000,
                rtf: 20.0,
                wer: Some(0.1),
                mos_score: Some(4.1),
                snr_db: Some(25.5),
            },
             SampleResult {
                id: "sample2".to_string(),
                transcript: "ref two".to_string(),
                synthesis_time_ms: 250,
                audio_length_ms: 4000,
                rtf: 16.0,
                wer: Some(0.2),
                mos_score: Some(3.9),
                snr_db: Some(24.5),
            },
        ];

        generate_report(output_dir.to_str().unwrap(), &results).await.unwrap();

        let report_path = output_dir.join("evaluation_report.md");
        let json_path = output_dir.join("results.json");

        assert!(report_path.exists());
        assert!(json_path.exists());

        let report_content = async_std::fs::read_to_string(report_path).await.unwrap();
        assert!(report_content.contains("Total Samples Evaluated: 2"));
        assert!(report_content.contains("Average Synthesis Time: 200.00 ms")); // (150+250)/2
        assert!(report_content.contains("Overall Real-Time Factor (RTF): 17.500")); // (3+4)/(0.15+0.25)
        assert!(report_content.contains("Average WER: 0.1500")); // (0.1+0.2)/2
        assert!(report_content.contains("Average Predicted MOS: 4.000")); // (4.1+3.9)/2
        assert!(report_content.contains("Average Estimated SNR: 25.00 dB")); // (25.5+24.5)/2

        let json_content = async_std::fs::read_to_string(json_path).await.unwrap();
        let parsed_results: Vec<SampleResult> = serde_json::from_str(&json_content).unwrap();
        assert_eq!(parsed_results.len(), 2);
        assert_eq!(parsed_results[0].id, "sample1");
    }

    #[test]
    fn test_calculate_snr_db_basic() {
        // Create a signal with noise at the beginning and end
        let sample_rate = 16000;
        let noise_samples = (sample_rate as f64 * 0.1) as usize; // 100ms noise
        let signal_samples = sample_rate * 1; // 1 second signal
        let total_samples = noise_samples * 2 + signal_samples;

        let mut audio = vec![0.0f32; total_samples];
        let noise_amplitude = 0.01;
        let signal_amplitude = 0.5;

        // Add noise
        for i in 0..noise_samples {
            audio[i] = (rand::thread_rng().gen::<f32>() - 0.5) * 2.0 * noise_amplitude;
            audio[total_samples - 1 - i] = (rand::thread_rng().gen::<f32>() - 0.5) * 2.0 * noise_amplitude;
        }
        // Add signal (simple sine wave)
        let freq = 440.0;
        for i in noise_samples..(noise_samples + signal_samples) {
             let t = (i - noise_samples) as f32 / sample_rate as f32;
             audio[i] = signal_amplitude * (2.0 * std::f32::consts::PI * freq * t).sin();
        }
        
        let snr = calculate_snr_db(&audio, sample_rate);
        assert!(snr.is_some());
        // Expect a positive SNR, likely high given the amplitudes
        // Power is amplitude squared, signal power ~0.5^2/2 = 0.125, noise power ~ (0.01^2)/3 (uniform noise approx)
        // SNR ~ 0.125 / (0.0001/3) ~ 3750. SNR_dB = 10*log10(3750) ~ 35 dB
        let snr_val = snr.unwrap();
        println!("Calculated SNR: {:.2} dB", snr_val);
        assert!(snr_val > 25.0 && snr_val < 45.0); 
    }

     #[test]
    fn test_calculate_snr_db_short_signal() {
        let audio = vec![0.1f32; 100]; // Too short for noise estimation
        assert!(calculate_snr_db(&audio, 16000).is_none());
    }

     #[test]
    fn test_calculate_snr_db_all_silence() {
        let audio = vec![0.0f32; 16000]; // All silence
        let snr = calculate_snr_db(&audio, 16000);
         assert!(snr.is_some());
         // Expect 0 dB or potentially negative if noise estimate slightly > 0 due to float precision
         assert!(snr.unwrap() <= 0.0); 
    }
}
