//! LLM Service Implementation
//!
//! This module provides implementations of the LLM service interface
//! for different types of Large Language Models.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use anyhow::{Result, Context as AnyhowContext};
use tch::{Tensor, Device};
use tokio::sync::Mutex as TokioMutex;

use super::{LlmProcessor, ContextEmbedding};
use super::context_embeddings::{ContextEmbeddingGenerator, ContextEmbeddingConfig};
use super::prompt_templates::{PromptTemplateRegistry, PromptType};
use crate::context::ConversationHistory;

/// Supported LLM types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmType {
    /// Llama model family (e.g., Llama 2, Llama 3)
    Llama,
    /// Mistral model family
    Mistral,
    /// Local language model file
    Local,
    /// Mock LLM for testing
    Mock,
}

/// Configuration for an LLM service
#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// Type of LLM to use
    pub llm_type: LlmType,
    /// Path to the model weights file (if using a local model)
    pub model_path: Option<PathBuf>,
    /// Model identifier for API-based models
    pub model_id: Option<String>,
    /// API key for cloud-based LLMs
    pub api_key: Option<String>,
    /// Embedding dimension for context embeddings
    pub embedding_dim: i64,
    /// Use GPU acceleration if available
    pub use_gpu: bool,
    /// Maximum context window size
    pub max_context_window: usize,
    /// Temperature for text generation
    pub temperature: f64,
    /// Other model-specific parameters
    pub parameters: std::collections::HashMap<String, String>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            llm_type: LlmType::Mock,
            model_path: None,
            model_id: None,
            api_key: None,
            embedding_dim: 768,
            use_gpu: true,
            max_context_window: 4096,
            temperature: 0.7,
            parameters: std::collections::HashMap::new(),
        }
    }
}

/// Create an LLM service based on the provided configuration
pub fn create_service(config: LlmConfig) -> Result<Arc<dyn LlmProcessor>> {
    match config.llm_type {
        LlmType::Llama => {
            let service = LlamaService::new(config.clone())?;
            Ok(Arc::new(service))
        }
        LlmType::Mistral => {
            let service = MistralService::new(config.clone())?;
            Ok(Arc::new(service))
        }
        LlmType::Local => {
            let config_clone = config.clone();
            if let Some(path) = &config_clone.model_path {
                let service = LocalModelService::new(config, path)?;
                Ok(Arc::new(service))
            } else {
                anyhow::bail!("Model path is required for local LLM service")
            }
        }
        LlmType::Mock => {
            let service = MockLlmService::new(config.clone());
            Ok(Arc::new(service))
        }
    }
}

/// Implementation of LLM service using Llama models
pub struct LlamaService {
    config: LlmConfig,
    embedding_generator: ContextEmbeddingGenerator,
    template_registry: PromptTemplateRegistry,
    // Using a mutex to ensure thread safety for the model
    model: TokioMutex<tch::CModule>,
    device: Device,
}

impl LlamaService {
    pub fn new(config: LlmConfig) -> Result<Self> {
        let device = if config.use_gpu && tch::Cuda::is_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };
        
        let model_path = config.model_path
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model path is required for Llama service"))?;
        
        let model = tch::CModule::load_on_device(model_path, device)
            .with_context(|| format!("Failed to load Llama model from {:?}", model_path))?;
        
        let embedding_config = ContextEmbeddingConfig {
            embedding_dim: config.embedding_dim,
            ..Default::default()
        };
        
        Ok(Self {
            config,
            embedding_generator: ContextEmbeddingGenerator::new(embedding_config),
            template_registry: PromptTemplateRegistry::new(),
            model: TokioMutex::new(model),
            device,
        })
    }
    
    /// Helper method to generate embeddings from raw text
    async fn generate_embedding_tensor(&self, text: &str) -> Result<Tensor> {
        let model = self.model.lock().await;
        
        // Create input tensor with the text
        let input = tch::IValue::String(text.to_string());
        
        // Call the model's embedding function
        let output = model.forward_is(&[input])
            .with_context(|| "Failed to generate embeddings with Llama model")?;
        
        // Extract tensor from output
        match output {
            tch::IValue::Tensor(tensor) => Ok(tensor),
            _ => anyhow::bail!("Expected tensor output from model, got {:?}", output),
        }
    }
}

#[async_trait::async_trait]
impl LlmProcessor for LlamaService {
    fn generate_embeddings(&self, context: &ConversationHistory) -> Result<ContextEmbedding> {
        // Get the embedding prompt template
        let template = self.template_registry.get_or_default(PromptType::ContextEmbedding);
        
        // Use a tokio runtime to handle the async operation
        let rt = tokio::runtime::Runtime::new()?;
        
        // Generate an embedding from the context using the template
        self.embedding_generator.generate(context, &|_prompt| {
            let prompt_str = template.format_with_history(context, self.config.max_context_window)?;
            rt.block_on(async {
                self.generate_embedding_tensor(&prompt_str).await
            })
        })
    }
    
    fn process_text(&self, text: &str) -> Result<String> {
        // Implement using tokio runtime if needed, simplified for now
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            self.process_text_async(text).await
        })
    }
    
    fn generate_response(&self, context: &ConversationHistory) -> Result<String> {
        // Implement using tokio runtime
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            self.generate_response_async(context).await
        })
    }
}

impl LlamaService {
    // Async implementations for the LLM processor methods
    async fn process_text_async(&self, text: &str) -> Result<String> {
        let model = self.model.lock().await;
        
        // Create input tensor with the text
        let input = tch::IValue::String(text.to_string());
        
        // Call the model to generate text
        let output = model.forward_is(&[input])
            .with_context(|| "Failed to process text with Llama model")?;
        
        // Extract string from output
        match output {
            tch::IValue::String(text) => Ok(text),
            _ => anyhow::bail!("Expected string output from model, got {:?}", output),
        }
    }
    
    async fn generate_response_async(&self, context: &ConversationHistory) -> Result<String> {
        // Format the entire conversation history into a prompt
        let formatted_history = context.format_for_prompt(self.config.max_context_window);
        
        // Process the formatted history to generate a response
        self.process_text_async(&formatted_history).await
    }
}

/// Implementation of LLM service using Mistral models
pub struct MistralService {
    config: LlmConfig,
    embedding_generator: ContextEmbeddingGenerator,
    template_registry: PromptTemplateRegistry,
    // Similar structure to LlamaService, but with Mistral-specific implementation
    model: TokioMutex<tch::CModule>,
    device: Device,
}

impl MistralService {
    pub fn new(config: LlmConfig) -> Result<Self> {
        let device = if config.use_gpu && tch::Cuda::is_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };
        
        let model_path = config.model_path
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model path is required for Mistral service"))?;
        
        let model = tch::CModule::load_on_device(model_path, device)
            .with_context(|| format!("Failed to load Mistral model from {:?}", model_path))?;
        
        let embedding_config = ContextEmbeddingConfig {
            embedding_dim: config.embedding_dim,
            ..Default::default()
        };
        
        Ok(Self {
            config,
            embedding_generator: ContextEmbeddingGenerator::new(embedding_config),
            template_registry: PromptTemplateRegistry::new(),
            model: TokioMutex::new(model),
            device,
        })
    }
    
    /// Helper method to generate embeddings from raw text
    async fn generate_embedding_tensor(&self, text: &str) -> Result<Tensor> {
        let model = self.model.lock().await;
        
        // Create input tensor with the text
        let input = tch::IValue::String(text.to_string());
        
        // Call the model's embedding function (implementation may differ from Llama)
        let output = model.forward_is(&[input])
            .with_context(|| "Failed to generate embeddings with Mistral model")?;
        
        // Extract tensor from output
        match output {
            tch::IValue::Tensor(tensor) => Ok(tensor),
            _ => anyhow::bail!("Expected tensor output from model, got {:?}", output),
        }
    }
}

#[async_trait::async_trait]
impl LlmProcessor for MistralService {
    fn generate_embeddings(&self, context: &ConversationHistory) -> Result<ContextEmbedding> {
        // Implementation similar to LlamaService with Mistral-specific adjustments
        let template = self.template_registry.get_or_default(PromptType::ContextEmbedding);
        let rt = tokio::runtime::Runtime::new()?;
        
        self.embedding_generator.generate(context, &|_prompt| {
            let prompt_str = template.format_with_history(context, self.config.max_context_window)?;
            rt.block_on(async {
                self.generate_embedding_tensor(&prompt_str).await
            })
        })
    }
    
    fn process_text(&self, text: &str) -> Result<String> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            self.process_text_async(text).await
        })
    }
    
    fn generate_response(&self, context: &ConversationHistory) -> Result<String> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            self.generate_response_async(context).await
        })
    }
}

impl MistralService {
    // Async implementations for the LLM processor methods
    async fn process_text_async(&self, text: &str) -> Result<String> {
        let model = self.model.lock().await;
        
        // Create input tensor with the text
        let input = tch::IValue::String(text.to_string());
        
        // Call the model to generate text
        let output = model.forward_is(&[input])
            .with_context(|| "Failed to process text with Mistral model")?;
        
        // Extract string from output
        match output {
            tch::IValue::String(text) => Ok(text),
            _ => anyhow::bail!("Expected string output from model, got {:?}", output),
        }
    }
    
    async fn generate_response_async(&self, context: &ConversationHistory) -> Result<String> {
        // Format the entire conversation history into a prompt
        let formatted_history = context.format_for_prompt(self.config.max_context_window);
        
        // Process the formatted history to generate a response
        self.process_text_async(&formatted_history).await
    }
}

/// Implementation of LLM service using a local model file
pub struct LocalModelService {
    config: LlmConfig,
    embedding_generator: ContextEmbeddingGenerator,
    template_registry: PromptTemplateRegistry,
    model: TokioMutex<tch::CModule>,
    device: Device,
}

impl LocalModelService {
    pub fn new(config: LlmConfig, model_path: &Path) -> Result<Self> {
        let device = if config.use_gpu && tch::Cuda::is_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };
        
        let model = tch::CModule::load_on_device(model_path, device)
            .with_context(|| format!("Failed to load local model from {:?}", model_path))?;
        
        let embedding_config = ContextEmbeddingConfig {
            embedding_dim: config.embedding_dim,
            ..Default::default()
        };
        
        Ok(Self {
            config,
            embedding_generator: ContextEmbeddingGenerator::new(embedding_config),
            template_registry: PromptTemplateRegistry::new(),
            model: TokioMutex::new(model),
            device,
        })
    }
    
    /// Helper method to generate embeddings from raw text
    async fn generate_embedding_tensor(&self, text: &str) -> Result<Tensor> {
        let model = self.model.lock().await;
        
        // Create input tensor with the text
        let input = tch::IValue::String(text.to_string());
        
        // Call the model's embedding function
        let output = model.forward_is(&[input])
            .with_context(|| "Failed to generate embeddings with local model")?;
        
        // Extract tensor from output
        match output {
            tch::IValue::Tensor(tensor) => Ok(tensor),
            _ => anyhow::bail!("Expected tensor output from model, got {:?}", output),
        }
    }
}

#[async_trait::async_trait]
impl LlmProcessor for LocalModelService {
    fn generate_embeddings(&self, context: &ConversationHistory) -> Result<ContextEmbedding> {
        // Implementation similar to LlamaService
        let template = self.template_registry.get_or_default(PromptType::ContextEmbedding);
        let rt = tokio::runtime::Runtime::new()?;
        
        self.embedding_generator.generate(context, &|_prompt| {
            let prompt_str = template.format_with_history(context, self.config.max_context_window)?;
            rt.block_on(async {
                self.generate_embedding_tensor(&prompt_str).await
            })
        })
    }
    
    fn process_text(&self, text: &str) -> Result<String> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            self.process_text_async(text).await
        })
    }
    
    fn generate_response(&self, context: &ConversationHistory) -> Result<String> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            self.generate_response_async(context).await
        })
    }
}

impl LocalModelService {
    // Async implementations for the LLM processor methods
    async fn process_text_async(&self, text: &str) -> Result<String> {
        let model = self.model.lock().await;
        
        // Create input tensor with the text
        let input = tch::IValue::String(text.to_string());
        
        // Call the model to generate text
        let output = model.forward_is(&[input])
            .with_context(|| "Failed to process text with local model")?;
        
        // Extract string from output
        match output {
            tch::IValue::String(text) => Ok(text),
            _ => anyhow::bail!("Expected string output from model, got {:?}", output),
        }
    }
    
    async fn generate_response_async(&self, context: &ConversationHistory) -> Result<String> {
        // Format the entire conversation history into a prompt
        let formatted_history = context.format_for_prompt(self.config.max_context_window);
        
        // Process the formatted history to generate a response
        self.process_text_async(&formatted_history).await
    }
}

/// Mock implementation of LLM service for testing
pub struct MockLlmService {
    config: LlmConfig,
    embedding_generator: ContextEmbeddingGenerator,
    template_registry: PromptTemplateRegistry,
}

impl MockLlmService {
    pub fn new(config: LlmConfig) -> Self {
        let embedding_config = ContextEmbeddingConfig {
            embedding_dim: config.embedding_dim,
            ..Default::default()
        };
        
        Self {
            config,
            embedding_generator: ContextEmbeddingGenerator::new(embedding_config),
            template_registry: PromptTemplateRegistry::new(),
        }
    }
}

#[async_trait::async_trait]
impl LlmProcessor for MockLlmService {
    fn generate_embeddings(&self, context: &ConversationHistory) -> Result<ContextEmbedding> {
        // Generate a mock embedding for testing
        let device = Device::Cpu;
        
        // Create a random tensor with the configured embedding dimension
        let options = (tch::Kind::Float, device);
        let tensor = Tensor::rand(&[self.config.embedding_dim], options);
        
        // Create metadata for the embedding
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("source".to_string(), "mock".to_string());
        metadata.insert("turns".to_string(), format!("{}", context.get_turns().len()));
        
        Ok(ContextEmbedding::new(tensor, Some(metadata)))
    }
    
    fn process_text(&self, text: &str) -> Result<String> {
        // Return a mock response based on the input
        if text.contains("hello") || text.contains("hi") {
            Ok("Hello! How can I help you today?".to_string())
        } else if text.contains("help") {
            Ok("I'm here to help. What do you need assistance with?".to_string())
        } else {
            Ok(format!("You said: {}. How can I help with that?", text))
        }
    }
    
    fn generate_response(&self, context: &ConversationHistory) -> Result<String> {
        // For mock service, we'll just provide canned responses based on the last user turn
        if let Some(last_turn) = context.get_turns().last() {
            if last_turn.speaker == crate::context::Speaker::User {
                return self.process_text(&last_turn.text);
            }
        }
        
        Ok("I'm not sure what to say. Can you please tell me more?".to_string())
    }
} 