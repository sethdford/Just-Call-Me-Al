#[cfg(test)]
mod tests {
    use crate::models::{ModelError, CSMModel, AudioOutput, config::CsmModelConfig};
    use crate::context::{ConversationHistory, ConversationTurn, Speaker};
    use crate::models::prosody::ProsodyControl;
    use crate::audio::AudioProcessing;
    
    use tokio::sync::{mpsc, Mutex as TokioMutex};
    use tokio::time::{timeout, Duration};
    use async_trait::async_trait;
    use tracing::{info, warn};
    use std::sync::Arc;

    // Mock CSM model implementation that doesn't require real model weights
    #[derive(Clone)]
    struct MockCSMModel;

    impl MockCSMModel {
        fn new() -> Self {
            MockCSMModel
        }
    }

    #[async_trait]
    impl CSMModel for MockCSMModel {
        fn get_config(&self) -> Result<CsmModelConfig, ModelError> {
            Ok(Default::default())
        }

        fn get_processor(&self) -> Result<Arc<TokioMutex<dyn AudioProcessing + Send + Sync>>, ModelError> {
            Err(ModelError::NotImplemented)
        }

        async fn predict_rvq_tokens(
            &self,
            _text: &str,
            _conversation_history: Option<&ConversationHistory>,
            _temperature: Option<f32>,
        ) -> Result<Vec<Vec<i64>>, ModelError> {
            warn!("MockCSMModel::predict_rvq_tokens called (not implemented).");
            // Return an empty vector of tokens to satisfy the trait
            Ok(vec![vec![0i64; 5]])
        }

        async fn synthesize(
            &self,
            text: &str,
            _conversation_history: Option<&ConversationHistory>,
            _temperature: Option<f32>,
            _top_k: Option<i64>,
            _seed: Option<i64>,
        ) -> Result<AudioOutput, ModelError> {
            info!("MockCSMModel synthesize called with text: {:?}", text);
            // Return empty audio output
            Ok(AudioOutput {
                samples: vec![0i16; 1000],
                sample_rate: 24000,
            })
        }

        async fn synthesize_streaming(
            &self,
            text: &str,
            _prosody: Option<ProsodyControl>,
            _style_preset: Option<String>,
            chunk_tx: mpsc::Sender<Result<Vec<u8>, ModelError>>,
        ) -> Result<(), ModelError> {
            info!("MockCSMModel synthesize_streaming called with text: {:?} (using sender)", text);

            // Send mock byte data
            for i in 0..5 {
                let is_final = i == 4;
                // Simulate sending byte chunks (e.g., empty for simplicity)
                let audio_bytes = if is_final { Vec::new() } else { vec![0u8; 1024] }; 

                if chunk_tx.send(Ok(audio_bytes)).await.is_err() {
                    warn!("MockCSMModel: Chunk receiver dropped.");
                    break; // Stop if channel is closed
                }
            }

            Ok(())
        }

        async fn synthesize_codes(
            &self,
        ) -> Result<AudioOutput, ModelError> {
            warn!("MockCSMModel::synthesize_codes is not implemented.");
            Err(ModelError::NotImplemented)
        }

        async fn synthesize_codes_streaming(
            &self,
        ) -> Result<(), ModelError> {
            warn!("MockCSMModel::synthesize_codes_streaming is not implemented.");
            Err(ModelError::NotImplemented)
        }
    }

    #[tokio::test]
    async fn test_synthesize_streaming_with_history() -> Result<(), Box<dyn std::error::Error>> {
        let model = MockCSMModel::new();

        let text = "This is a test sentence.";

        // --- Test 1: No History --- 
        let (tx1, mut rx1) = mpsc::channel(100);
        
        // Use the CSMModel trait method
        model.synthesize_streaming(
            &text, 
            None, // prosody
            None, // style_preset
            tx1   // Pass the sender directly
        ).await.map_err(|e| Box::<dyn std::error::Error>::from(e))?;

        let mut chunks_no_history = Vec::new();
        let mut total_bytes = 0;
        loop {
            match timeout(Duration::from_secs(2), rx1.recv()).await {
                Ok(Some(Ok(bytes))) => {
                    total_bytes += bytes.len();
                    if bytes.is_empty() {
                        // Empty Vec<u8> indicates final chunk
                        chunks_no_history.push(bytes);
                        break;
                    } else {
                        chunks_no_history.push(bytes);
                    }
                },
                Ok(Some(Err(e))) => return Err(Box::<dyn std::error::Error>::from(e)),
                Ok(None) => break,
                Err(_) => panic!("Timeout waiting for chunks (no history)"),
            }
        }
        
        info!("Chunks (No History): {} chunks, {} bytes received", chunks_no_history.len(), total_bytes);
        assert!(!chunks_no_history.is_empty(), "Should generate chunks without history");
        assert!(chunks_no_history.last().map_or(false, |c| c.is_empty()), "Last chunk should be empty (final)");

        // --- Test 2: With History ---
        let (tx2, mut rx2) = mpsc::channel(100);
        
        // Create some history
        let mut history = ConversationHistory::new(None);
        history.add_turn(ConversationTurn::new(Speaker::User, "Hello".to_string()));
        history.add_turn(ConversationTurn::new(Speaker::Assistant, "Hi there!".to_string()));
        
        // Use the CSMModel trait method through model
        model.synthesize_streaming(
            &text,
            None, // prosody
            None, // style_preset
            tx2   // Pass the sender directly
        ).await.map_err(|e| Box::<dyn std::error::Error>::from(e))?;

        let mut chunks_with_history = Vec::new();
        total_bytes = 0;
        loop {
            match timeout(Duration::from_secs(2), rx2.recv()).await {
                Ok(Some(Ok(bytes))) => {
                    total_bytes += bytes.len();
                    if bytes.is_empty() {
                        // Empty Vec<u8> indicates final chunk
                        chunks_with_history.push(bytes);
                        break;
                    } else {
                        chunks_with_history.push(bytes);
                    }
                },
                Ok(Some(Err(e))) => return Err(Box::<dyn std::error::Error>::from(e)),
                Ok(None) => break,
                Err(_) => panic!("Timeout waiting for chunks (with history)"),
            }
        }
        
        info!("Chunks (With History): {} chunks, {} bytes received", chunks_with_history.len(), total_bytes);
        assert!(!chunks_with_history.is_empty(), "Should generate chunks with history");
        assert!(chunks_with_history.last().map_or(false, |c| c.is_empty()), "Last chunk should be empty (final with history)");

        Ok(())
    }
} 