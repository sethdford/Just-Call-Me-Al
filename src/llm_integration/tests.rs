#[cfg(test)]
mod tests {
    
    use std::collections::HashMap;
    use std::time::Duration;
    use crate::context::{ConversationHistory, ConversationTurn, Speaker};
    use crate::llm_integration::{
        context_embeddings::{ContextEmbedding, ContextEmbeddingGenerator, ContextEmbeddingConfig},
        prompt_templates::{PromptTemplate, PromptTemplateRegistry, PromptType},
        llm_service::{MockLlmService, LlmConfig},
        LlmProcessor,
    };
    use tch::{Tensor, Device};
    use anyhow::Result;

    #[test]
    fn test_context_embedding_creation() {
        let options = (tch::Kind::Float, Device::Cpu);
        let tensor = Tensor::rand(&[768], options);
        let mut metadata = HashMap::new();
        metadata.insert("test".to_string(), "value".to_string());
        
        let embedding = ContextEmbedding::new(tensor, Some(metadata));
        
        assert_eq!(embedding.dim(), 768);
        assert_eq!(embedding.metadata.get("test").unwrap(), "value");
        assert!(!embedding.is_stale(Duration::from_secs(10)));
    }
    
    #[test]
    fn test_embedding_staleness() {
        let options = (tch::Kind::Float, Device::Cpu);
        let tensor = Tensor::rand(&[768], options);
        let embedding = ContextEmbedding::new(tensor, None);
        
        // Should not be stale immediately
        assert!(!embedding.is_stale(Duration::from_secs(1)));
        
        // Wait for a bit
        std::thread::sleep(Duration::from_millis(50));
        
        // Should be stale after a very short timeout
        assert!(embedding.is_stale(Duration::from_millis(10)));
        
        // But not with a longer timeout
        assert!(!embedding.is_stale(Duration::from_secs(1)));
    }

    #[test]
    fn test_context_embedding_generator() {
        let config = ContextEmbeddingConfig {
            embedding_dim: 512,
            max_age: Duration::from_secs(30),
            max_cache_size: 10,
            enable_caching: true,
        };
        
        let generator = ContextEmbeddingGenerator::new(config);
        let mut history = ConversationHistory::new(None);
        history.add_turn(ConversationTurn::new(Speaker::User, "Hello, how are you?".to_string()));
        history.add_turn(ConversationTurn::new(Speaker::Model, "I'm doing well, thanks!".to_string()));
        
        // Create a simple embedding generator function
        let embedding_fn = |_: &str| -> Result<Tensor, anyhow::Error> {
            let options = (tch::Kind::Float, Device::Cpu);
            let tensor = Tensor::rand(&[512], options);
            Ok(tensor)
        };
        
        // Generate an embedding
        let embedding = generator.generate(&history, &embedding_fn).unwrap();
        
        // Verify properties
        assert_eq!(embedding.dim(), 512);
        assert!(embedding.metadata.contains_key("turns"));
        assert_eq!(embedding.metadata.get("turns").unwrap(), "2");
        
        // Check cache stats
        let stats = generator.cache_stats();
        assert_eq!(stats.total, 1);
        assert_eq!(stats.stale, 0);
        
        // Test cache hit (generate again with the same history)
        let cached_embedding = generator.generate(&history, &embedding_fn).unwrap();
        assert_eq!(cached_embedding.dim(), 512);
        
        // Cache size should still be 1
        let stats = generator.cache_stats();
        assert_eq!(stats.total, 1);
        
        // Test cache clear
        generator.clear_cache();
        let stats = generator.cache_stats();
        assert_eq!(stats.total, 0);
    }

    #[test]
    fn test_prompt_template() {
        // Create a simple template
        let template = PromptTemplate::new(
            PromptType::ContextEmbedding,
            String::from("Hello, {name}! Here is the conversation: {conversation_history}"),
            Some(String::from("You are a helpful assistant.")),
            vec![]
        );
        
        // Format with values
        let mut values = HashMap::new();
        values.insert("name".to_string(), "world".to_string());
        values.insert("conversation_history".to_string(), "User: Hi\nModel: Hello".to_string());
        
        let formatted = template.format(&values).unwrap();
        
        // Check formatting
        assert!(formatted.contains("You are a helpful assistant."));
        assert!(formatted.contains("Hello, world!"));
        assert!(formatted.contains("User: Hi\nModel: Hello"));
    }
    
    #[test]
    fn test_prompt_template_registry() {
        let registry = PromptTemplateRegistry::new();
        
        // Get a template
        let template = registry.get(PromptType::ContextEmbedding).unwrap();
        assert_eq!(template.template_type, PromptType::ContextEmbedding);
        
        // Get with default
        let template = registry.get_or_default(PromptType::ProsodyControl);
        assert_eq!(template.template_type, PromptType::ProsodyControl);
    }

    #[test]
    fn test_mock_llm_service() {
        let config = LlmConfig::default();
        let mock_service = MockLlmService::new(config);
        
        // Create a simple conversation history
        let mut history = ConversationHistory::new(None);
        history.add_turn(ConversationTurn::new(Speaker::User, "Hello, how are you?".to_string()));
        
        // Generate embeddings
        let embedding = mock_service.generate_embeddings(&history).unwrap();
        assert_eq!(embedding.dim(), 768); // Default dim in LlmConfig
        assert_eq!(embedding.metadata.get("source").unwrap(), "mock");
        
        // Process text
        let response = mock_service.process_text("hello").unwrap();
        assert!(response.contains("Hello"));
        
        // Generate response
        let response = mock_service.generate_response(&history).unwrap();
        assert!(response.contains("Hello"));
    }
} 