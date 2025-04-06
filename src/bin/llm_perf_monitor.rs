//! LLM Performance Monitoring Tool
//!
//! This utility helps measure and analyze the performance of LLM operations
//! with different optimization techniques.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use anyhow::Result;
use clap::Parser;
use tracing::{info, warn};
use csm::llm_integration::{
    create_llm_service, create_optimized_llm_service,
    LlmConfig, LlmType, LlmProcessor, OptimizedLlm
};
use csm::context::ConversationHistory;

/// CLI Arguments
#[derive(Parser, Debug)]
#[clap(
    name = "llm-perf-monitor",
    about = "Monitor and analyze LLM performance with various optimizations",
    version
)]
struct Args {
    /// Directory containing LLM model files
    #[clap(long, short = 'd')]
    model_dir: PathBuf,
    
    /// LLM type to use (llama, mistral, local, mock)
    #[clap(long, default_value = "mock")]
    llm_type: String,
    
    /// Number of iterations for benchmarking
    #[clap(long, default_value = "10")]
    iterations: usize,
    
    /// Use GPU acceleration if available
    #[clap(long)]
    gpu: bool,
    
    /// Enable batching optimization
    #[clap(long)]
    batching: bool,
    
    /// Enable KV cache optimization
    #[clap(long)]
    kv_cache: bool,
    
    /// Use optimized processor
    #[clap(long)]
    optimized: bool,
}

/// Initialize tracing
fn init_tracing() {
    let subscriber = tracing_subscriber::FmtSubscriber::builder()
        .with_max_level(tracing::Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set tracing subscriber");
}

/// Create a sample conversation history for benchmarking
fn create_sample_history() -> ConversationHistory {
    let mut history = ConversationHistory::new(None);
    history.add_turn(csm::context::ConversationTurn::new(
        csm::context::Speaker::User,
        "Hello, how are you today?".to_string()
    ));
    history.add_turn(csm::context::ConversationTurn::new(
        csm::context::Speaker::Assistant,
        "I'm doing well, thank you for asking! How can I help you today?".to_string()
    ));
    history.add_turn(csm::context::ConversationTurn::new(
        csm::context::Speaker::User,
        "Can you tell me about the weather forecast for tomorrow?".to_string()
    ));
    history.add_turn(csm::context::ConversationTurn::new(
        csm::context::Speaker::Assistant,
        "Mock response".to_string(),
    ));
    history
}

/// Run performance test with standard LLM processor
fn run_standard_test(
    llm: Arc<dyn LlmProcessor>,
    history: &ConversationHistory,
    iterations: usize
) -> Result<(f64, f64, f64)> {
    let mut embedding_times = Vec::with_capacity(iterations);
    let mut text_times = Vec::with_capacity(iterations);
    let mut response_times = Vec::with_capacity(iterations);
    
    info!("Running standard LLM processor test with {} iterations", iterations);
    
    for i in 0..iterations {
        info!("Iteration {}/{}", i+1, iterations);
        
        // Test embedding generation
        let start = Instant::now();
        let _embedding = llm.generate_embeddings(history)?;
        let embedding_time = start.elapsed().as_millis() as f64;
        embedding_times.push(embedding_time);
        
        // Test text processing
        let start = Instant::now();
        let _result = llm.process_text("What is the meaning of life?")?;
        let text_time = start.elapsed().as_millis() as f64;
        text_times.push(text_time);
        
        // Test response generation
        let start = Instant::now();
        let _response = llm.generate_response(history)?;
        let response_time = start.elapsed().as_millis() as f64;
        response_times.push(response_time);
    }
    
    // Calculate averages
    let avg_embedding_time = embedding_times.iter().sum::<f64>() / iterations as f64;
    let avg_text_time = text_times.iter().sum::<f64>() / iterations as f64;
    let avg_response_time = response_times.iter().sum::<f64>() / iterations as f64;
    
    Ok((avg_embedding_time, avg_text_time, avg_response_time))
}

/// Run performance test with optimized LLM processor
fn run_optimized_test(
    llm: Arc<OptimizedLlm>,
    history: &ConversationHistory,
    iterations: usize
) -> Result<(f64, f64, f64)> {
    let mut embedding_times = Vec::with_capacity(iterations);
    let mut text_times = Vec::with_capacity(iterations);
    let mut response_times = Vec::with_capacity(iterations);
    
    info!("Running optimized LLM processor test with {} iterations", iterations);
    
    for i in 0..iterations {
        info!("Iteration {}/{}", i+1, iterations);
        
        // Test embedding generation
        let start = Instant::now();
        let _embedding = llm.generate_embeddings(history)?;
        let embedding_time = start.elapsed().as_millis() as f64;
        embedding_times.push(embedding_time);
        
        // Test text processing
        let start = Instant::now();
        let _result = llm.process_text("What is the meaning of life?")?;
        let text_time = start.elapsed().as_millis() as f64;
        text_times.push(text_time);
        
        // Test response generation
        let start = Instant::now();
        let _response = llm.generate_response(history)?;
        let response_time = start.elapsed().as_millis() as f64;
        response_times.push(response_time);
    }
    
    // Calculate averages
    let avg_embedding_time = embedding_times.iter().sum::<f64>() / iterations as f64;
    let avg_text_time = text_times.iter().sum::<f64>() / iterations as f64;
    let avg_response_time = response_times.iter().sum::<f64>() / iterations as f64;
    
    // Get the internal stats if available
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let stats = llm.get_timing_stats().await;
        info!("Cache hit rate: {:.1}%", stats.cache_hit_rate());
        info!("Internal timing: {}", stats.format());
    });
    
    Ok((avg_embedding_time, avg_text_time, avg_response_time))
}

/// The main function
fn main() -> Result<()> {
    // Initialize tracing
    init_tracing();
    
    // Parse arguments
    let args = Args::parse();
    info!("Starting LLM performance monitor");
    
    // Determine device based on arguments and availability
    let device = if args.gpu && tch::Cuda::is_available() {
        info!("Using CUDA GPU");
        tch::Device::Cuda(0)
    } else if args.gpu {
        warn!("GPU requested but CUDA is not available; falling back to CPU");
        tch::Device::Cpu
    } else {
        info!("Using CPU");
        tch::Device::Cpu
    };
    
    // Create LLM configuration
    let llm_type = match args.llm_type.to_lowercase().as_str() {
        "llama" => LlmType::Llama,
        "mistral" => LlmType::Mistral,
        "local" => LlmType::Local,
        _ => LlmType::Mock,
    };
    
    let llm_config = LlmConfig {
        llm_type,
        model_path: Some(args.model_dir.clone()),
        embedding_dim: if llm_type == LlmType::Mistral { 1024 } else { 768 },
        use_gpu: device != tch::Device::Cpu,
        ..Default::default()
    };
    
    // Create sample conversation history for benchmarking
    let history = create_sample_history();
    
    // Run tests
    if args.optimized {
        info!("Using optimized LLM processor");
        let llm = create_optimized_llm_service(llm_config)?;
        
        let (avg_embedding_time, avg_text_time, avg_response_time) = 
            run_optimized_test(llm, &history, args.iterations)?;
            
        info!("Performance Results (Optimized):");
        info!("Average embedding generation time: {:.2} ms", avg_embedding_time);
        info!("Average text processing time: {:.2} ms", avg_text_time);
        info!("Average response generation time: {:.2} ms", avg_response_time);
    } else {
        info!("Using standard LLM processor");
        let llm = create_llm_service(llm_config)?;
        
        let (avg_embedding_time, avg_text_time, avg_response_time) = 
            run_standard_test(llm, &history, args.iterations)?;
            
        info!("Performance Results (Standard):");
        info!("Average embedding generation time: {:.2} ms", avg_embedding_time);
        info!("Average text processing time: {:.2} ms", avg_text_time);
        info!("Average response generation time: {:.2} ms", avg_response_time);
    }
    
    Ok(())
} 