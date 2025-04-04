use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use anyhow::{Result, anyhow};

use crate::llm_integration::{
    LlmProcessor, LlmConfig, LlmType, create_llm_service,
    MetricType, MetricReading, MetricsRegistry, MetricsConfig,
    InstrumentedLlmProcessor, create_instrumented_llm
};
use crate::context::{ConversationHistory, ConversationTurn, Speaker};

// Mock LLM processor for testing
struct TestLlmProcessor;

impl LlmProcessor for TestLlmProcessor {
    fn generate_embeddings(&self, _context: &ConversationHistory) -> Result<crate::llm_integration::ContextEmbedding> {
        // Simulate small processing delay
        std::thread::sleep(Duration::from_millis(50));
        
        // Create a mock embedding
        let tensor = tch::Tensor::randn(&[1, 768], tch::kind::FLOAT_CPU);
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "test_processor".to_string());
        
        Ok(crate::llm_integration::ContextEmbedding::new(tensor, Some(metadata)))
    }
    
    fn process_text(&self, text: &str) -> Result<String> {
        // Simulate processing delay
        std::thread::sleep(Duration::from_millis(100));
        
        Ok(format!("Processed: {}", text))
    }
    
    fn generate_response(&self, context: &ConversationHistory) -> Result<String> {
        // Simulate processing delay
        std::thread::sleep(Duration::from_millis(150));
        
        // Get last user message
        let last_turn = context.get_turns().iter()
            .filter(|turn| matches!(turn.speaker, Speaker::User))
            .last();
        
        match last_turn {
            Some(turn) => Ok(format!("Response to: {}", turn.text)),
            None => Ok("Default response".to_string()),
        }
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[tokio::test]
async fn test_metrics_registry() -> Result<()> {
    // Create metrics registry
    let config = MetricsConfig {
        max_readings_per_metric: 10,
        export_to_file: false,
        export_directory: None,
        log_metrics: false,
    };
    
    let registry = Arc::new(MetricsRegistry::new(config));
    
    // Record some metrics
    registry.record_metric(MetricReading::new(MetricType::Latency, 100.0)).await;
    registry.record_metric(MetricReading::new(MetricType::Latency, 150.0)).await;
    registry.record_metric(MetricReading::new(MetricType::Latency, 120.0)).await;
    
    registry.record_metric(MetricReading::new(MetricType::SuccessRate, 100.0)).await;
    registry.record_metric(MetricReading::new(MetricType::SuccessRate, 0.0)).await;
    registry.record_metric(MetricReading::new(MetricType::SuccessRate, 100.0)).await;
    
    // Get metrics report
    let report = registry.get_report().await;
    
    // Check metrics
    for stat in &report.metrics {
        match stat.metric_type {
            MetricType::Latency => {
                assert_eq!(stat.count, 3);
                assert!((stat.avg - 123.33).abs() < 0.1); // Average of 100, 150, 120
                assert!((stat.min - 100.0).abs() < 0.1);
                assert!((stat.max - 150.0).abs() < 0.1);
            },
            MetricType::SuccessRate => {
                assert_eq!(stat.count, 3);
                assert!((stat.avg - 66.67).abs() < 0.1); // Average of 100, 0, 100
                assert!((stat.min - 0.0).abs() < 0.1);
                assert!((stat.max - 100.0).abs() < 0.1);
            },
            _ => {},
        }
    }
    
    Ok(())
}

#[test]
fn test_instrumented_processor() -> Result<()> {
    // Create test LLM processor
    let llm = Arc::new(TestLlmProcessor);
    
    // Create instrumented processor
    let instrumented = create_instrumented_llm(llm, None);
    
    // Create sample conversation history
    let mut history = ConversationHistory::new(None);
    history.add_turn(ConversationTurn::new(
        Speaker::User,
        "Hello, how are you?".to_string()
    ));
    
    // Test operations
    let embedding = instrumented.generate_embeddings(&history)?;
    assert_eq!(embedding.dim(), 768);
    
    let processed = instrumented.process_text("Test text")?;
    assert_eq!(processed, "Processed: Test text");
    
    let response = instrumented.generate_response(&history)?;
    assert_eq!(response, "Response to: Hello, how are you?");
    
    // Get metrics registry
    let registry = instrumented.get_metrics_registry();
    
    // Get metrics report
    let rt = tokio::runtime::Runtime::new()?;
    let report = rt.block_on(async {
        registry.get_report().await
    });
    
    // Verify metrics were recorded
    assert!(!report.metrics.is_empty());
    
    // Check for latency metrics
    let has_latency = report.metrics.iter()
        .any(|stat| stat.metric_type == MetricType::Latency);
    
    assert!(has_latency, "Expected latency metrics to be recorded");
    
    // Check for success rate metrics
    let has_success_rate = report.metrics.iter()
        .any(|stat| stat.metric_type == MetricType::SuccessRate);
    
    assert!(has_success_rate, "Expected success rate metrics to be recorded");
    
    Ok(())
}

#[test]
fn test_time_series_metrics() -> Result<()> {
    // Create a metrics time series
    let mut time_series = crate::llm_integration::evaluation::MetricsTimeSeries::new(5);
    
    // Add some readings
    time_series.add_reading(MetricReading::new(MetricType::Latency, 100.0));
    time_series.add_reading(MetricReading::new(MetricType::Latency, 200.0));
    time_series.add_reading(MetricReading::new(MetricType::Latency, 300.0));
    
    // Check statistics
    assert_eq!(time_series.average().unwrap(), 200.0);
    assert_eq!(time_series.min().unwrap(), 100.0);
    assert_eq!(time_series.max().unwrap(), 300.0);
    
    // Add more readings to test overflow
    time_series.add_reading(MetricReading::new(MetricType::Latency, 400.0));
    time_series.add_reading(MetricReading::new(MetricType::Latency, 500.0));
    time_series.add_reading(MetricReading::new(MetricType::Latency, 600.0));
    
    // The oldest reading (100.0) should be removed
    assert_eq!(time_series.get_readings().len(), 5);
    assert!(time_series.min().unwrap() > 100.0);
    
    // Check 95th percentile
    let p95 = time_series.percentile_95().unwrap();
    assert!(p95 >= 500.0, "95th percentile should be at least 500.0, got {}", p95);
    
    Ok(())
}

#[tokio::test]
async fn test_mock_llm_service_with_metrics() -> Result<()> {
    // Create mock LLM config
    let config = LlmConfig {
        llm_type: LlmType::Mock,
        ..Default::default()
    };
    
    // Create LLM service
    let llm = create_llm_service(config)?;
    
    // Create metrics config
    let metrics_config = MetricsConfig {
        max_readings_per_metric: 100,
        export_to_file: false,
        export_directory: None,
        log_metrics: false,
    };
    
    // Create instrumented processor
    let instrumented = create_instrumented_llm(llm, Some(metrics_config));
    
    // Create sample conversation history
    let mut history = ConversationHistory::new(None);
    history.add_turn(ConversationTurn::new(
        Speaker::User,
        "What is speech synthesis?".to_string()
    ));
    
    // Run operations multiple times
    for _ in 0..5 {
        let _ = instrumented.generate_embeddings(&history)?;
        let _ = instrumented.process_text("Test query")?;
        let _ = instrumented.generate_response(&history)?;
    }
    
    // Get metrics registry
    let registry = instrumented.get_metrics_registry();
    
    // Get metrics report
    let report = registry.get_report().await;
    
    // Verify metrics were recorded for all operations
    let latency_metrics = report.metrics.iter()
        .filter(|stat| stat.metric_type == MetricType::Latency)
        .next()
        .expect("Expected latency metrics");
    
    assert!(latency_metrics.count >= 15, "Expected at least 15 latency readings (5 iterations * 3 operations)");
    
    // Format and check the report
    let report_str = report.format();
    assert!(report_str.contains("Latency"));
    assert!(report_str.contains("Success Rate"));
    
    Ok(())
} 