#![allow(dead_code)] // Allow dead code for this evaluation/benchmarking module

//! LLM Evaluation and Monitoring System
//!
//! This module provides tools for evaluating and monitoring the LLM integration,
//! including quality metrics, performance monitoring, and diagnostics.

use std::sync::Arc;
use std::collections::HashMap;
use std::time::Instant;

use tokio::sync::RwLock as TokioRwLock;
use tracing::{info, debug};
use anyhow::{Result, Context};

use crate::context::ConversationHistory;
use super::{LlmProcessor, ContextEmbedding};

/// Metric types supported by the evaluation system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricType {
    /// Latency in milliseconds
    Latency,
    /// Memory usage in bytes
    MemoryUsage,
    /// Cache hit rate as percentage
    CacheHitRate,
    /// Success rate as percentage
    SuccessRate,
    /// Quality score (0-100)
    QualityScore,
    /// Custom metric
    Custom(u32),
}

/// Represents a single metric reading
#[derive(Debug, Clone)]
pub struct MetricReading {
    /// Type of metric
    pub metric_type: MetricType,
    /// Value of the metric
    pub value: f64,
    /// Timestamp of the reading
    pub timestamp: std::time::SystemTime,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

impl MetricReading {
    /// Create a new metric reading
    pub fn new(metric_type: MetricType, value: f64) -> Self {
        Self {
            metric_type,
            value,
            timestamp: std::time::SystemTime::now(),
            metadata: HashMap::new(),
        }
    }
    
    /// Create a new metric reading with metadata
    pub fn with_metadata(metric_type: MetricType, value: f64, metadata: HashMap<String, String>) -> Self {
        Self {
            metric_type,
            value,
            timestamp: std::time::SystemTime::now(),
            metadata,
        }
    }
}

/// Stores a time series of metric readings
#[derive(Debug, Clone)]
pub struct MetricsTimeSeries {
    /// The metric readings
    readings: Vec<MetricReading>,
    /// Maximum number of readings to store
    max_readings: usize,
}

impl MetricsTimeSeries {
    /// Create a new metrics time series
    pub fn new(max_readings: usize) -> Self {
        Self {
            readings: Vec::with_capacity(max_readings),
            max_readings,
        }
    }
    
    /// Add a new reading to the time series
    pub fn add_reading(&mut self, reading: MetricReading) {
        if self.readings.len() >= self.max_readings {
            self.readings.remove(0); // Remove oldest reading
        }
        self.readings.push(reading);
    }
    
    /// Get the average value of the metric
    pub fn average(&self) -> Option<f64> {
        if self.readings.is_empty() {
            return None;
        }
        
        let sum: f64 = self.readings.iter().map(|r| r.value).sum();
        Some(sum / self.readings.len() as f64)
    }
    
    /// Get the maximum value of the metric
    pub fn max(&self) -> Option<f64> {
        self.readings.iter().map(|r| r.value).fold(None, |max, val| {
            match max {
                None => Some(val),
                Some(max_val) => Some(max_val.max(val)),
            }
        })
    }
    
    /// Get the minimum value of the metric
    pub fn min(&self) -> Option<f64> {
        self.readings.iter().map(|r| r.value).fold(None, |min, val| {
            match min {
                None => Some(val),
                Some(min_val) => Some(min_val.min(val)),
            }
        })
    }
    
    /// Get the 95th percentile value
    pub fn percentile_95(&self) -> Option<f64> {
        if self.readings.is_empty() {
            return None;
        }
        
        let mut values: Vec<f64> = self.readings.iter().map(|r| r.value).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let idx = (values.len() as f64 * 0.95).floor() as usize;
        if idx >= values.len() {
            values.last().copied()
        } else {
            Some(values[idx])
        }
    }
    
    /// Get all readings
    pub fn get_readings(&self) -> &[MetricReading] {
        &self.readings
    }
}

/// Central registry for collecting and analyzing metrics
#[derive(Debug)]
pub struct MetricsRegistry {
    /// Metrics time series by metric type
    metrics: TokioRwLock<HashMap<MetricType, MetricsTimeSeries>>,
    /// Configuration
    config: MetricsConfig,
}

/// Configuration for the metrics registry
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    /// Maximum number of readings per metric
    pub max_readings_per_metric: usize,
    /// Whether to export metrics to files
    pub export_to_file: bool,
    /// Directory to export metrics to
    pub export_directory: Option<String>,
    /// Log metrics to tracing
    pub log_metrics: bool,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            max_readings_per_metric: 1000,
            export_to_file: false,
            export_directory: None,
            log_metrics: true,
        }
    }
}

impl MetricsRegistry {
    /// Create a new metrics registry with the given configuration
    pub fn new(config: MetricsConfig) -> Self {
        Self {
            metrics: TokioRwLock::new(HashMap::new()),
            config,
        }
    }
    
    /// Record a metric reading
    pub async fn record_metric(&self, reading: MetricReading) {
        let mut metrics = self.metrics.write().await;
        
        // Get or create the time series for this metric
        let time_series = metrics
            .entry(reading.metric_type)
            .or_insert_with(|| MetricsTimeSeries::new(self.config.max_readings_per_metric));
        
        // Add the reading
        time_series.add_reading(reading.clone());
        
        // Log the metric if enabled
        if self.config.log_metrics {
            let metric_name = match reading.metric_type {
                MetricType::Latency => "Latency",
                MetricType::MemoryUsage => "MemoryUsage",
                MetricType::CacheHitRate => "CacheHitRate",
                MetricType::SuccessRate => "SuccessRate",
                MetricType::QualityScore => "QualityScore",
                MetricType::Custom(_id) => return,
            };
            
            debug!("{}: {:.2} (metadata: {:?})", metric_name, reading.value, reading.metadata);
        }
    }
    
    /// Export metrics to a file
    pub async fn export_metrics(&self) -> Result<(), anyhow::Error> {
        // Return early if export is disabled
        if !self.config.export_to_file || self.config.export_directory.is_none() {
            return Ok(());
        }
        
        // Create export directory if it doesn't exist
        let export_dir = self.config.export_directory.as_ref().unwrap();
        std::fs::create_dir_all(export_dir)
            .context(format!("Failed to create export directory: {}", export_dir))?;
        
        // Export each metric to a separate file
        let metrics = self.metrics.read().await;
        for (metric_type, time_series) in metrics.iter() {
            let metric_name = match metric_type {
                MetricType::Latency => "latency",
                MetricType::MemoryUsage => "memory_usage",
                MetricType::CacheHitRate => "cache_hit_rate",
                MetricType::SuccessRate => "success_rate",
                MetricType::QualityScore => "quality_score",
                MetricType::Custom(_id) => continue,
            };
            
            let file_path = format!("{}/{}.csv", export_dir, metric_name);
            let mut file = std::fs::File::create(&file_path)
                .context(format!("Failed to create file: {}", file_path))?;
            
            // Write CSV header
            use std::io::Write;
            writeln!(file, "timestamp,value,metadata")
                .with_context(|| format!("Failed to write header to file: {}", file_path))?;
            
            // Write readings
            for reading in time_series.get_readings() {
                let metadata_str = if reading.metadata.is_empty() {
                    "{}".to_string()
                } else {
                    serde_json::to_string(&reading.metadata)
                        .unwrap_or_else(|_| "{}".to_string())
                };
                
                let timestamp = reading.timestamp
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                
                writeln!(file, "{},{:.2},{}", timestamp, reading.value, metadata_str)
                    .with_context(|| format!("Failed to write reading to file: {}", file_path))?;
            }
        }
        
        Ok(())
    }
    
    /// Get metrics report
    pub async fn get_report(&self) -> MetricsReport {
        let metrics = self.metrics.read().await;
        let mut report = MetricsReport::new();
        
        for (metric_type, time_series) in metrics.iter() {
            let avg = time_series.average().unwrap_or(0.0);
            let max = time_series.max().unwrap_or(0.0);
            let min = time_series.min().unwrap_or(0.0);
            let p95 = time_series.percentile_95().unwrap_or(0.0);
            
            let stat = MetricStatistics {
                metric_type: *metric_type,
                count: time_series.get_readings().len(),
                avg,
                max,
                min,
                p95,
            };
            
            report.metrics.push(stat);
        }
        
        report
    }
}

/// Statistics for a metric
#[derive(Debug, Clone)]
pub struct MetricStatistics {
    /// Type of metric
    pub metric_type: MetricType,
    /// Number of readings
    pub count: usize,
    /// Average value
    pub avg: f64,
    /// Maximum value
    pub max: f64,
    /// Minimum value
    pub min: f64,
    /// 95th percentile value
    pub p95: f64,
}

/// Report of all metrics
#[derive(Debug, Clone)]
pub struct MetricsReport {
    /// List of metric statistics
    pub metrics: Vec<MetricStatistics>,
    /// Generated at
    pub generated_at: std::time::SystemTime,
}

impl MetricsReport {
    /// Create a new metrics report
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
            generated_at: std::time::SystemTime::now(),
        }
    }
    
    /// Format the report as a string
    pub fn format(&self) -> String {
        let mut result = String::from("=== Metrics Report ===\n");
        result.push_str(&format!("Generated at: {:?}\n\n", self.generated_at));
        
        for stat in &self.metrics {
            let metric_name = match stat.metric_type {
                MetricType::Latency => "Latency (ms)",
                MetricType::MemoryUsage => "Memory Usage (bytes)",
                MetricType::CacheHitRate => "Cache Hit Rate (%)",
                MetricType::SuccessRate => "Success Rate (%)",
                MetricType::QualityScore => "Quality Score (0-100)",
                MetricType::Custom(_id) => continue,
            };
            
            result.push_str(&format!("-- {} --\n", metric_name));
            result.push_str(&format!("Count: {}\n", stat.count));
            result.push_str(&format!("Avg: {:.2}\n", stat.avg));
            result.push_str(&format!("Min: {:.2}\n", stat.min));
            result.push_str(&format!("Max: {:.2}\n", stat.max));
            result.push_str(&format!("P95: {:.2}\n\n", stat.p95));
        }
        
        result
    }
}

/// Instrumented LLM processor that collects metrics
pub struct InstrumentedLlmProcessor {
    /// The underlying LLM processor
    inner: Arc<dyn LlmProcessor>,
    /// Metrics registry
    metrics: Arc<MetricsRegistry>,
}

impl InstrumentedLlmProcessor {
    /// Create a new instrumented LLM processor
    pub fn new(inner: Arc<dyn LlmProcessor>, metrics: Arc<MetricsRegistry>) -> Self {
        Self { inner, metrics }
    }
    
    /// Get the metrics registry
    pub fn get_metrics_registry(&self) -> Arc<MetricsRegistry> {
        self.metrics.clone()
    }
}

/// Implementation of LlmProcessor for InstrumentedLlmProcessor
impl LlmProcessor for InstrumentedLlmProcessor {
    fn generate_embeddings(&self, context: &ConversationHistory) -> Result<ContextEmbedding, anyhow::Error> {
        let start = Instant::now();
        let result = self.inner.generate_embeddings(context);
        let elapsed = start.elapsed();
        
        // Record latency metric
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            let mut metadata = HashMap::new();
            metadata.insert("operation".to_string(), "generate_embeddings".to_string());
            metadata.insert("context_turns".to_string(), context.get_turns().len().to_string());
            
            self.metrics.record_metric(MetricReading::with_metadata(
                MetricType::Latency,
                elapsed.as_millis() as f64,
                metadata,
            )).await;
            
            if let Ok(embedding) = &result {
                let mut metadata = HashMap::new();
                metadata.insert("operation".to_string(), "generate_embeddings".to_string());
                metadata.insert("embedding_dim".to_string(), embedding.dim().to_string());
                
                self.metrics.record_metric(MetricReading::with_metadata(
                    MetricType::SuccessRate,
                    100.0, // Success = 100%
                    metadata,
                )).await;
            } else {
                let mut metadata = HashMap::new();
                metadata.insert("operation".to_string(), "generate_embeddings".to_string());
                metadata.insert("error".to_string(), "failed".to_string());
                
                self.metrics.record_metric(MetricReading::with_metadata(
                    MetricType::SuccessRate,
                    0.0, // Failure = 0%
                    metadata,
                )).await;
            }
        });
        
        result
    }
    
    fn process_text(&self, text: &str) -> Result<String, anyhow::Error> {
        let start = Instant::now();
        let result = self.inner.process_text(text);
        let elapsed = start.elapsed();
        
        // Record latency metric
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            let mut metadata = HashMap::new();
            metadata.insert("operation".to_string(), "process_text".to_string());
            metadata.insert("text_length".to_string(), text.len().to_string());
            
            self.metrics.record_metric(MetricReading::with_metadata(
                MetricType::Latency,
                elapsed.as_millis() as f64,
                metadata,
            )).await;
            
            let success_value = if result.is_ok() { 100.0 } else { 0.0 };
            let mut metadata = HashMap::new();
            metadata.insert("operation".to_string(), "process_text".to_string());
            
            self.metrics.record_metric(MetricReading::with_metadata(
                MetricType::SuccessRate,
                success_value,
                metadata,
            )).await;
        });
        
        result
    }
    
    fn generate_response(&self, context: &ConversationHistory) -> Result<String, anyhow::Error> {
        let start = Instant::now();
        let result = self.inner.generate_response(context);
        let elapsed = start.elapsed();
        
        // Record latency metric
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            let mut metadata = HashMap::new();
            metadata.insert("operation".to_string(), "generate_response".to_string());
            metadata.insert("context_turns".to_string(), context.get_turns().len().to_string());
            
            self.metrics.record_metric(MetricReading::with_metadata(
                MetricType::Latency,
                elapsed.as_millis() as f64,
                metadata,
            )).await;
            
            let success_value = if result.is_ok() { 100.0 } else { 0.0 };
            let mut metadata = HashMap::new();
            metadata.insert("operation".to_string(), "generate_response".to_string());
            
            self.metrics.record_metric(MetricReading::with_metadata(
                MetricType::SuccessRate,
                success_value,
                metadata,
            )).await;
        });
        
        result
    }
    
    fn _as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Create an instrumented LLM processor
pub fn create_instrumented_llm(
    llm: Arc<dyn LlmProcessor>,
    config: Option<MetricsConfig>,
) -> Arc<InstrumentedLlmProcessor> {
    let metrics_config = config.unwrap_or_default();
    let metrics = Arc::new(MetricsRegistry::new(metrics_config));
    Arc::new(InstrumentedLlmProcessor::new(llm, metrics))
}

/// Benchmark utility for LLM processors
pub struct LlmBenchmark {
    /// The LLM processor to benchmark
    llm: Arc<dyn LlmProcessor>,
    /// Sample conversation history for benchmarking
    sample_history: ConversationHistory,
    /// Number of iterations for each test
    iterations: usize,
}

impl LlmBenchmark {
    /// Create a new LLM benchmark
    pub fn new(llm: Arc<dyn LlmProcessor>, iterations: usize) -> Self {
        // Create a sample conversation history
        let mut history = ConversationHistory::new(None);
        history.add_turn(crate::context::ConversationTurn::new(
            crate::context::Speaker::User,
            "Hello, how are you today?".to_string()
        ));
        history.add_turn(crate::context::ConversationTurn::new(
            crate::context::Speaker::Assistant,
            "I'm doing well, thank you for asking! How can I help you today?".to_string()
        ));
        history.add_turn(crate::context::ConversationTurn::new(
            crate::context::Speaker::User,
            "Can you tell me about the weather forecast for tomorrow?".to_string()
        ));
        
        Self {
            llm,
            sample_history: history,
            iterations,
        }
    }
    
    /// Run the benchmark
    pub fn run(&self) -> Result<BenchmarkResults, anyhow::Error> {
        info!("Running LLM benchmark with {} iterations", self.iterations);
        
        let mut results = BenchmarkResults::new();
        
        // Benchmark embedding generation
        let mut embedding_times = Vec::with_capacity(self.iterations);
        for i in 0..self.iterations {
            debug!("Embedding generation, iteration {}/{}", i+1, self.iterations);
            
            let start = Instant::now();
            let _embedding = self.llm.generate_embeddings(&self.sample_history)?;
            let elapsed = start.elapsed().as_millis() as f64;
            
            embedding_times.push(elapsed);
        }
        
        // Benchmark text processing
        let mut text_times = Vec::with_capacity(self.iterations);
        for i in 0..self.iterations {
            debug!("Text processing, iteration {}/{}", i+1, self.iterations);
            
            let start = Instant::now();
            let _result = self.llm.process_text("What is the meaning of life?")?;
            let elapsed = start.elapsed().as_millis() as f64;
            
            text_times.push(elapsed);
        }
        
        // Benchmark response generation
        let mut response_times = Vec::with_capacity(self.iterations);
        for i in 0..self.iterations {
            debug!("Response generation, iteration {}/{}", i+1, self.iterations);
            
            let start = Instant::now();
            let _response = self.llm.generate_response(&self.sample_history)?;
            let elapsed = start.elapsed().as_millis() as f64;
            
            response_times.push(elapsed);
        }
        
        // Populate results
        results.embedding_avg = embedding_times.iter().sum::<f64>() / self.iterations as f64;
        results.embedding_min = *embedding_times.iter().min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(&0.0);
        results.embedding_max = *embedding_times.iter().max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(&0.0);
        
        results.text_avg = text_times.iter().sum::<f64>() / self.iterations as f64;
        results.text_min = *text_times.iter().min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(&0.0);
        results.text_max = *text_times.iter().max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(&0.0);
        
        results.response_avg = response_times.iter().sum::<f64>() / self.iterations as f64;
        results.response_min = *response_times.iter().min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(&0.0);
        results.response_max = *response_times.iter().max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(&0.0);
        
        Ok(results)
    }
}

/// Results of LLM benchmark
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Average embedding generation time (ms)
    pub embedding_avg: f64,
    /// Minimum embedding generation time (ms)
    pub embedding_min: f64,
    /// Maximum embedding generation time (ms)
    pub embedding_max: f64,
    
    /// Average text processing time (ms)
    pub text_avg: f64,
    /// Minimum text processing time (ms)
    pub text_min: f64,
    /// Maximum text processing time (ms)
    pub text_max: f64,
    
    /// Average response generation time (ms)
    pub response_avg: f64,
    /// Minimum response generation time (ms)
    pub response_min: f64,
    /// Maximum response generation time (ms)
    pub response_max: f64,
}

impl BenchmarkResults {
    /// Create a new benchmark results
    pub fn new() -> Self {
        Self {
            embedding_avg: 0.0,
            embedding_min: 0.0,
            embedding_max: 0.0,
            text_avg: 0.0,
            text_min: 0.0,
            text_max: 0.0,
            response_avg: 0.0,
            response_min: 0.0,
            response_max: 0.0,
        }
    }
    
    /// Format the results as a string
    pub fn format(&self) -> String {
        let mut result = String::from("=== LLM Benchmark Results ===\n\n");
        
        result.push_str("-- Embedding Generation Time (ms) --\n");
        result.push_str(&format!("Avg: {:.2}\n", self.embedding_avg));
        result.push_str(&format!("Min: {:.2}\n", self.embedding_min));
        result.push_str(&format!("Max: {:.2}\n\n", self.embedding_max));
        
        result.push_str("-- Text Processing Time (ms) --\n");
        result.push_str(&format!("Avg: {:.2}\n", self.text_avg));
        result.push_str(&format!("Min: {:.2}\n", self.text_min));
        result.push_str(&format!("Max: {:.2}\n\n", self.text_max));
        
        result.push_str("-- Response Generation Time (ms) --\n");
        result.push_str(&format!("Avg: {:.2}\n", self.response_avg));
        result.push_str(&format!("Min: {:.2}\n", self.response_min));
        result.push_str(&format!("Max: {:.2}\n", self.response_max));
        
        result
    }
} 