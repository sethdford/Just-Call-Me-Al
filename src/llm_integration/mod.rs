//! LLM Integration Module
//! 
//! This module provides functionality for integrating Large Language Models (LLMs)
//! with the CSM system to enhance contextual understanding and generate
//! more appropriate speech characteristics.

use std::sync::Arc;
use anyhow::Result;
use tch::Tensor;

mod context_embeddings;
mod llm_service;
mod prompt_templates;
#[cfg(test)]
mod tests;

pub use context_embeddings::{ContextEmbedding, ContextEmbeddingGenerator};
pub use llm_service::{LlmService, LlmConfig, LlmType};
pub use prompt_templates::PromptTemplate;

/// Unified interface for LLM operations in the CSM system
pub trait LlmProcessor: Send + Sync {
    /// Generate contextual embeddings from conversation history
    fn generate_embeddings(&self, context: &crate::context::ConversationHistory) -> Result<ContextEmbedding>;
    
    /// Process raw text with the LLM
    fn process_text(&self, text: &str) -> Result<String>;
    
    /// Generate a response based on conversation history
    fn generate_response(&self, context: &crate::context::ConversationHistory) -> Result<String>;
}

/// Factory function to create an appropriate LLM service based on configuration
pub fn create_llm_service(config: LlmConfig) -> Result<Arc<dyn LlmProcessor>> {
    llm_service::create_service(config)
} 