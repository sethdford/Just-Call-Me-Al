//! LLM Monitoring and Benchmarking Utility
//!
//! This command-line tool provides functionality for monitoring and benchmarking
//! LLM performance in the CSM system, including comprehensive metrics reporting.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use anyhow::{Result, anyhow, Context as AnyhowContext};
use clap::Parser;
use tracing::info;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

use csm::llm_integration::evaluation::LlmBenchmark;

use csm::llm_integration::{
    create_llm_service, create_optimized_llm_service,
    LlmConfig, LlmType, LlmProcessor,
};
use csm::context::{ConversationHistory, ConversationTurn, Speaker};

/// CLI Arguments
#[derive(Parser, Debug)]
#[clap(
    name = "llm-monitor",
    about = "Monitor and benchmark LLM performance with comprehensive metrics",
    version
)]
struct Args {
    /// Directory containing LLM model files
    #[clap(long, short = 'd')]
    model_dir: Option<PathBuf>,
    
    /// LLM type to use (llama, mistral, local, mock)
    #[clap(long, default_value = "mock")]
    llm_type: String,
    
    /// Number of iterations for benchmarking
    #[clap(long, default_value = "10")]
    iterations: usize,
    
    /// Use GPU acceleration if available
    #[clap(long)]
    gpu: bool,
    
    /// Enable optimized LLM processor
    #[clap(long)]
    optimized: bool,
    
    /// Run benchmark tests
    #[clap(long)]
    benchmark: bool,
    
    /// Monitor metrics continuously
    #[clap(long)]
    monitor: bool,
    
    /// Export metrics to files
    #[clap(long)]
    export: bool,
    
    /// Directory for exporting metrics
    #[clap(long, default_value = "metrics")]
    export_dir: String,
    
    /// Monitoring duration in seconds (0 = indefinite)
    #[clap(long, default_value = "300")]
    duration: u64,
    
    /// Sample interval in seconds
    #[clap(long, default_value = "5")]
    interval: u64,
}

/// Initialize tracing
fn init_tracing() -> Result<()> {
    let subscriber = tracing_subscriber::FmtSubscriber::builder()
        .with_max_level(tracing::Level::INFO)
        .finish();
    
    tracing::subscriber::set_global_default(subscriber)
        .context("Failed to set tracing subscriber")?;
    
    Ok(())
}

/// Parse LLM type from string
fn parse_llm_type(llm_type: &str) -> Result<LlmType> {
    match llm_type.to_lowercase().as_str() {
        "llama" => Ok(LlmType::Llama),
        "mistral" => Ok(LlmType::Mistral),
        "local" => Ok(LlmType::Local),
        "mock" => Ok(LlmType::Mock),
        _ => Err(anyhow!("Unsupported LLM type: {}", llm_type)),
    }
}

/// Create LLM configuration based on arguments
fn create_llm_config(args: &Args) -> Result<LlmConfig> {
    let llm_type = parse_llm_type(&args.llm_type)?;
    
    let model_path = if llm_type != LlmType::Mock {
        match &args.model_dir {
            Some(dir) => {
                // Default model filename based on type
                let filename = match llm_type {
                    LlmType::Llama => "llama-model.safetensors",
                    LlmType::Mistral => "mistral-model.safetensors",
                    LlmType::Local => "local-model.safetensors",
                    LlmType::Mock => "mock-model.safetensors", // Not actually used
                };
                
                Some(dir.join(filename))
            },
            None if llm_type != LlmType::Mock => {
                return Err(anyhow!("Model directory is required for {} LLM type", args.llm_type));
            },
            None => None,
        }
    } else {
        None
    };
    
    let config = LlmConfig {
        llm_type,
        model_path,
        _model_id: None,
        _api_key: None,
        embedding_dim: 768,
        use_gpu: args.gpu,
        max_context_window: 4096,
        _temperature: 0.7,
        _parameters: std::collections::HashMap::new(),
    };
    
    Ok(config)
}

/// Create sample conversation history for testing
fn create_sample_history() -> ConversationHistory {
    let mut history = ConversationHistory::new(Some(10));
    
    history.add_turn(ConversationTurn::new(
        Speaker::User,
        "Hello, how are you today?".to_string()
    ));
    
    history.add_turn(ConversationTurn::new(
        Speaker::Assistant,
        "I'm doing well, thank you for asking! How can I assist you today?".to_string()
    ));
    
    history.add_turn(ConversationTurn::new(
        Speaker::User,
        "I'd like to know more about voice synthesis technologies.".to_string()
    ));
    
    history.add_turn(ConversationTurn::new(
        Speaker::Assistant,
        "Mock response".to_string(),
    ));
    
    history
}

/// Run benchmarking tests
async fn run_benchmark(args: &Args) -> Result<()> {
    info!("Running LLM benchmarks with {} iterations", args.iterations);
    
    // Create LLM configuration
    let llm_config = create_llm_config(&args)?;
    
    // Create LLM service
    let llm: Arc<dyn LlmProcessor> = if args.optimized {
        info!("Using optimized LLM processor");
        create_optimized_llm_service(llm_config)?
    } else {
        info!("Using standard LLM processor");
        create_llm_service(llm_config)?
    };
    
    // Create benchmark
    let benchmark = LlmBenchmark::new(llm, args.iterations);
    
    // Run benchmark
    info!("Starting benchmark...");
    let results = benchmark.run()?;
    
    // Display results
    println!("\n{}", results.format());
    
    // Write results to file if export is enabled
    if args.export {
        // Create export directory if it doesn't exist
        tokio::fs::create_dir_all(&args.export_dir).await
            .context("Failed to create export directory")?;
        
        // Write results to file
        let path = format!("{}/benchmark_results.txt", args.export_dir);
        let mut file = File::create(&path).await
            .context("Failed to create benchmark results file")?;
        
        file.write_all(results.format().as_bytes()).await
            .context("Failed to write benchmark results to file")?;
        
        info!("Benchmark results written to {}", path);
    }
    
    info!("Benchmark finished.");
    
    Ok(())
}

/// Run continuous monitoring
async fn run_monitoring(args: &Args) -> Result<()> {
    info!("Starting LLM performance monitoring");
    
    // Create LLM configuration
    let llm_config = create_llm_config(&args)?;
    
    // Create LLM service
    let llm = create_optimized_llm_service(llm_config)?;
    
    info!("LLM service initialized.");

    // Create sample conversation history
    let history = create_sample_history();
    
    // Monitoring duration
    let duration = if args.duration == 0 {
        None // Run indefinitely
    } else {
        Some(Duration::from_secs(args.duration))
    };
    
    // Start time
    let start_time = std::time::Instant::now();
    
    // Interval
    let interval_duration = Duration::from_secs(args.interval);
    
    // Run monitoring loop
    info!("Monitoring started. Press Ctrl+C to stop.");
    info!("Collecting metrics every {} seconds", args.interval);
    
    let mut interval = tokio::time::interval(interval_duration);
    
    loop {
        interval.tick().await;
        
        // Check if monitoring duration has elapsed
        if let Some(duration) = duration {
            if start_time.elapsed() >= duration {
                info!("Monitoring duration ({} seconds) elapsed", args.duration);
                break;
            }
        }
        
        // Simulate some LLM operations
        let _ = llm.generate_embeddings(&history);
        let _ = llm.process_text("What is voice synthesis?");
        let _ = llm.generate_response(&history);
        
        // Get and print metrics report
        // let report = llm.get_metrics_registry().get_report().await;
        // println!("\n{}", report.format());
    }
    
    info!("Monitoring completed");
    
    Ok(())
}

/// Main entry point
#[tokio::main]
async fn main() -> Result<()> {
    // Parse arguments
    let args = Args::parse();
    
    // Initialize tracing
    init_tracing()?;
    
    // Run benchmark if requested
    if args.benchmark {
        run_benchmark(&args).await?;
    }
    
    // Run monitoring if requested
    if args.monitor {
        run_monitoring(&args).await?;
    }
    
    // If neither benchmark nor monitor is requested, print help
    if !args.benchmark && !args.monitor {
        println!("No action specified. Use --benchmark or --monitor to perform operations.");
        println!("Run with --help for more information.");
    }
    
    Ok(())
} 