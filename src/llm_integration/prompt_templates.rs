//! Prompt Template Management
//!
//! This module defines templates for different LLM prompting scenarios,
//! such as embedding generation, contextual understanding, and prosody control.

use std::collections::HashMap;
use anyhow::{Result, bail};

use crate::context::{ConversationHistory, Speaker};

/// Available types of prompt templates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PromptType {
    /// For generating embeddings from conversation history
    ContextEmbedding,
    /// For determining emotional context
    EmotionDetection,
    /// For context-aware prosody instruction
    ProsodyControl,
    /// For summarizing conversation history
    Summarization,
    /// For extracting key entities and concepts
    EntityExtraction,
}

/// A template for formatting prompts for the LLM
#[derive(Debug, Clone)]
pub struct PromptTemplate {
    /// The type of prompt template
    pub template_type: PromptType,
    /// The template string with placeholders
    pub template: String,
    /// The system instruction to prepend
    pub system_instruction: Option<String>,
    /// The example prompts and expected outputs
    pub examples: Vec<(String, String)>,
}

impl PromptTemplate {
    /// Create a new prompt template
    pub fn new(
        template_type: PromptType,
        template: String,
        system_instruction: Option<String>,
        examples: Vec<(String, String)>,
    ) -> Self {
        Self {
            template_type,
            template,
            system_instruction,
            examples,
        }
    }
    
    /// Format the template with values from the given map
    pub fn format(&self, values: &HashMap<String, String>) -> Result<String> {
        let mut result = self.template.clone();
        
        // Replace placeholders in template
        for (key, value) in values {
            let placeholder = format!("{{{}}}", key);
            result = result.replace(&placeholder, value);
        }
        
        // Check if there are any remaining placeholders
        if result.contains('{') && result.contains('}') {
            bail!("Not all placeholders were replaced in the template");
        }
        
        // If system instruction is present, prepend it
        if let Some(sys_instruction) = &self.system_instruction {
            result = format!("{}\n\n{}", sys_instruction, result);
        }
        
        // If examples are present, add them
        if !self.examples.is_empty() {
            let examples_text = self.examples
                .iter()
                .map(|(input, output)| format!("Example Input:\n{}\n\nExample Output:\n{}", input, output))
                .collect::<Vec<_>>()
                .join("\n\n");
            
            result = format!("{}\n\nExamples:\n{}", result, examples_text);
        }
        
        Ok(result)
    }
    
    /// Format specifically for conversation history
    pub fn format_with_history(&self, history: &ConversationHistory, max_chars: usize) -> Result<String> {
        let history_text = history.format_for_prompt(max_chars);
        
        let mut values = HashMap::new();
        values.insert("conversation_history".to_string(), history_text);
        values.insert("turn_count".to_string(), history.get_turns().len().to_string());
        
        self.format(&values)
    }
}

/// Get a default template for the given prompt type
pub fn get_default_template(template_type: PromptType) -> PromptTemplate {
    match template_type {
        PromptType::ContextEmbedding => {
            PromptTemplate::new(
                template_type,
                String::from("Given the following conversation history, create a dense vector representation that captures the semantic meaning and emotional context of the conversation. Focus on key topics, entities, and the emotional tone.\n\nConversation History:\n{conversation_history}\n\nNow, generate a dense vector representation of this conversation."),
                Some(String::from("You are a helpful assistant that generates vector representations of conversations.")),
                vec![],
            )
        },
        PromptType::EmotionDetection => {
            PromptTemplate::new(
                template_type,
                String::from("Analyze the emotional context of the following conversation. Identify the overall emotional tone, any changes in emotion, and the current emotional state at the end of the conversation.\n\nConversation History:\n{conversation_history}\n\nEmotional Analysis:"),
                Some(String::from("You are an emotional intelligence expert that can detect emotions from text.")),
                vec![
                    (
                        String::from("User: I can't believe they rejected my application.\nModel: I understand that's frustrating. What will you do next?\nUser: I don't know. Maybe I'll try somewhere else, but I'm feeling pretty discouraged."),
                        String::from("The conversation begins with the user expressing disappointment and frustration. The emotion intensifies to discouragement by the end. The overall tone is negative with feelings of rejection and uncertainty. The current emotional state is discouraged and somewhat defeated.")
                    ),
                ],
            )
        },
        PromptType::ProsodyControl => {
            PromptTemplate::new(
                template_type,
                String::from("Based on the following conversation, generate instructions for controlling speech prosody (pitch, rate, emphasis, pauses) to convey appropriate emotion and meaning.\n\nConversation History:\n{conversation_history}\n\nProsody Instructions:"),
                Some(String::from("You are an expert in speech synthesis who can provide detailed instructions for natural-sounding speech prosody.")),
                vec![],
            )
        },
        PromptType::Summarization => {
            PromptTemplate::new(
                template_type,
                String::from("Summarize the following conversation in a concise way that captures the most important points. This summary will be used to provide context for future turns in the conversation.\n\nConversation History:\n{conversation_history}\n\nSummary:"),
                Some(String::from("You are a helpful assistant that provides concise and accurate summaries of conversations.")),
                vec![],
            )
        },
        PromptType::EntityExtraction => {
            PromptTemplate::new(
                template_type,
                String::from("Extract the key entities, concepts, and relationships mentioned in the following conversation. These will be used to maintain context in a long conversation.\n\nConversation History:\n{conversation_history}\n\nEntities and Concepts:"),
                Some(String::from("You are a precise entity recognition system that extracts structured information from text.")),
                vec![],
            )
        },
    }
}

/// A registry of prompt templates that can be used to retrieve and manage templates
#[derive(Debug, Clone)]
pub struct PromptTemplateRegistry {
    templates: HashMap<PromptType, PromptTemplate>,
}

impl PromptTemplateRegistry {
    /// Create a new registry with default templates
    pub fn new() -> Self {
        let mut registry = Self {
            templates: HashMap::new(),
        };
        
        // Add all default templates
        registry.register(get_default_template(PromptType::ContextEmbedding));
        registry.register(get_default_template(PromptType::EmotionDetection));
        registry.register(get_default_template(PromptType::ProsodyControl));
        registry.register(get_default_template(PromptType::Summarization));
        registry.register(get_default_template(PromptType::EntityExtraction));
        
        registry
    }
    
    /// Register a new template or replace an existing one
    pub fn register(&mut self, template: PromptTemplate) {
        self.templates.insert(template.template_type, template);
    }
    
    /// Get a template by type, returns None if not found
    pub fn get(&self, template_type: PromptType) -> Option<&PromptTemplate> {
        self.templates.get(&template_type)
    }
    
    /// Get a template by type, or return a default if not found
    pub fn get_or_default(&self, template_type: PromptType) -> PromptTemplate {
        self.templates
            .get(&template_type)
            .cloned()
            .unwrap_or_else(|| get_default_template(template_type))
    }
}

impl Default for PromptTemplateRegistry {
    fn default() -> Self {
        Self::new()
    }
} 