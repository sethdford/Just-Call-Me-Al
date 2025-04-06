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
pub mod evaluation;

// Re-exports
pub use context_embeddings::ContextEmbedding;
pub use optimization::{OptimizedLlm, create_optimized_llm};
pub use evaluation::{
    MetricsConfig,
    create_instrumented_llm
};

/// Unified interface for LLM operations in the CSM system
pub trait LlmProcessor: Send + Sync {
    /// Generate contextual embeddings from conversation history
    fn generate_embeddings(&self, context: &crate::context::ConversationHistory) -> Result<ContextEmbedding>;
    
    /// Process raw text with the LLM
    fn process_text(&self, text: &str) -> Result<String>;
    
    /// Generate a response based on conversation history
    fn generate_response(&self, context: &crate::context::ConversationHistory) -> Result<String>;
    
    /// Return self as Any to allow downcasting
    fn _as_any(&self) -> &dyn Any;
}

// Export the trait and necessary concrete types/configs
pub use llm_service::{LlmConfig, LlmType};

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

/// Creates an LLM service, potentially wrapped with monitoring capabilities
/// This function handles the logic of choosing the correct LLM implementation and wrapping it
pub fn _create_monitored_llm_service(config: LlmConfig, metrics_config: Option<MetricsConfig>) -> Result<Arc<dyn LlmProcessor>> {
    // Create the base LLM service
    let base_llm = create_llm_service(config)?;
    Ok(create_instrumented_llm(base_llm, metrics_config))
}

/// Convert from context::ConversationHistory to our local ConversationHistory
pub fn create_llm_history(local_history: &crate::context::ConversationHistory) -> crate::context::ConversationHistory {
    let mut llm_history = crate::context::ConversationHistory::new(None);
    
    // Copy each turn using the public get_turns() method
    for turn in local_history.get_turns() {
        // Clone speaker since it doesn't implement Copy
        let speaker = map_speaker(turn.speaker.clone());
        
        llm_history.add_turn(crate::context::ConversationTurn::new(
            speaker,
            turn.text.clone()
        ));
    }
    
    llm_history
}

fn map_speaker(speaker: crate::context::Speaker) -> crate::context::Speaker {
    match speaker {
        crate::context::Speaker::User => crate::context::Speaker::User,
        crate::context::Speaker::Assistant => crate::context::Speaker::Assistant,
        crate::context::Speaker::_System => crate::context::Speaker::_System,
    }
} 