//! LLM Integration Module
//!
//! Provides traits and implementations for interacting with various LLMs
//! for tasks like response generation and contextual embedding.

use anyhow::Result;
use std::sync::Arc;
use std::any::Any;

// Modules
mod context_embeddings;
mod llm_service;
mod prompt_templates;
mod optimization;

// Re-exports
pub use context_embeddings::{ContextEmbedding, ContextEmbeddingConfig, ContextEmbeddingGenerator, CacheStats};
pub use prompt_templates::{PromptTemplate, PromptTemplateRegistry, PromptType};
pub use optimization::{OptimizedLlm, OptimizedInference, TimingStats, ComputationCache, create_optimized_llm, optimized_tensor_op};

/// Unified interface for LLM operations in the CSM system
pub trait LlmProcessor: Send + Sync {
    /// Generate contextual embeddings from conversation history
    fn generate_embeddings(&self, context: &crate::context::ConversationHistory) -> Result<ContextEmbedding>;
    
    /// Process raw text with the LLM
    fn process_text(&self, text: &str) -> Result<String>;
    
    /// Generate a response based on conversation history
    fn generate_response(&self, context: &crate::context::ConversationHistory) -> Result<String>;
    
    /// Convert to Any for downcasting
    fn as_any(&self) -> &dyn Any;
}

// Export the trait and necessary concrete types/configs
pub use llm_service::{LlmConfig, LlmType, create_service, MockLlmService, LlamaService, MistralService, LocalModelService};

// Export test module if needed for external tests
#[cfg(test)]
pub mod tests;

/// Factory function to create an appropriate LLM service based on configuration
pub fn create_llm_service(config: LlmConfig) -> Result<Arc<dyn LlmProcessor>> {
    llm_service::create_service(config)
}

/// Factory function to create an optimized LLM service based on configuration
pub fn create_optimized_llm_service(config: LlmConfig) -> Result<Arc<OptimizedLlm>> {
    let llm = create_llm_service(config)?;
    Ok(create_optimized_llm(llm))
}

/// Convert from context::ConversationHistory to our local ConversationHistory
pub fn create_llm_history(local_history: &crate::context::ConversationHistory) -> crate::context::ConversationHistory {
    let mut llm_history = crate::context::ConversationHistory::new(None);
    
    // Copy each turn using the public get_turns() method
    for turn in local_history.get_turns() {
        let speaker = match turn.speaker {
            crate::context::Speaker::User => crate::context::Speaker::User,
            crate::context::Speaker::Model => crate::context::Speaker::Model,
            crate::context::Speaker::System => crate::context::Speaker::System,
        };
        
        llm_history.add_turn(crate::context::ConversationTurn::new(
            speaker,
            turn.text.clone()
        ));
    }
    
    llm_history
} 