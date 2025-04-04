#[cfg(test)]
mod tests {
    use crate::models::{CSMImpl, ModelError, CSMModel};
    
    use crate::llm_integration::{create_service as create_llm_service, LlmConfig, LlmType};
    use std::path::PathBuf;
    
    use tokio::sync::mpsc;
    use tokio::time::{timeout, Duration};
    use tch::Device;
    use anyhow::anyhow;

    // Helper to find the project root (adjust based on your test setup)
    fn find_project_root() -> PathBuf {
        // Simple approach: Assume tests run from workspace root
        // More robust: Use env vars or search upwards for Cargo.toml
        PathBuf::from("./")
    }

    // Helper to initialize the model for testing
    async fn setup_test_model() -> Result<CSMImpl, ModelError> {
        let project_root = find_project_root();
        let model_dir = project_root.join("models/csm-1b"); // Adjust if needed
        let device = Device::Cpu; // Use CPU for testing

        // Use Mock LLM for this test
        let llm_config = LlmConfig {
            llm_type: LlmType::Mock,
            ..Default::default()
        };
        let llm_processor = create_llm_service(llm_config)
            .map_err(|e| ModelError::Other(anyhow!("Test setup: Failed to create mock LLM: {}", e)))?;

        CSMImpl::new_with_processor(&model_dir, device, llm_processor)
    }

    #[tokio::test]
    async fn test_synthesize_streaming_with_history() -> Result<(), Box<dyn std::error::Error>> {
        let model: CSMImpl = setup_test_model().await?;

        let text = "This is a test sentence.";
        let temperature = Some(0.7);
        let top_k = Some(50);
        let seed = Some(1234);

        // --- Test 1: No History --- 
        let (tx1, mut rx1) = mpsc::channel(100);
        
        // Use the CSMModel trait method through the CSMImpl
        model.synthesize_streaming(
            &text, 
            temperature, 
            top_k, 
            seed, 
            tx1
        ).await?;

        let mut tokens_no_history = Vec::new();
        loop {
            match timeout(Duration::from_secs(10), rx1.recv()).await {
                Ok(Some(batch)) => tokens_no_history.extend(batch),
                Ok(None) => break, // Channel closed
                Err(_) => panic!("Timeout waiting for tokens (no history)"),
            }
        }
        
        println!("Tokens (No History): {} frames", tokens_no_history.len());
        assert!(!tokens_no_history.is_empty(), "Should generate tokens without history");

        Ok(())
    }
} 