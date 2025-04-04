//! Context Embedding Generation
//!
//! This module handles the generation of fixed-dimension embeddings from
//! conversation history using LLM models.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use anyhow::{Result, Context as AnyhowContext};
use tch::Tensor;

use crate::context::{ConversationHistory, Speaker};

/// A fixed-dimension embedding representing the context of a conversation
#[derive(Debug, Clone)]
pub struct ContextEmbedding {
    /// The tensor containing the embedding values
    pub tensor: Tensor,
    /// The timestamp when this embedding was generated
    pub generated_at: Instant,
    /// Metadata about the embedding
    pub metadata: HashMap<String, String>,
}

impl ContextEmbedding {
    /// Create a new context embedding
    pub fn new(tensor: Tensor, metadata: Option<HashMap<String, String>>) -> Self {
        Self {
            tensor,
            generated_at: Instant::now(),
            metadata: metadata.unwrap_or_default(),
        }
    }
    
    /// Check if the embedding is older than the specified duration
    pub fn is_stale(&self, max_age: Duration) -> bool {
        self.generated_at.elapsed() > max_age
    }
    
    /// Get the dimensionality of the embedding
    pub fn dim(&self) -> i64 {
        self.tensor.size1().unwrap_or(0)
    }
}

/// Configuration for the context embedding generator
#[derive(Debug, Clone)]
pub struct ContextEmbeddingConfig {
    /// Dimension of the output embeddings
    pub embedding_dim: i64,
    /// Maximum age before an embedding is considered stale
    pub max_age: Duration,
    /// Maximum number of embeddings to cache
    pub max_cache_size: usize,
    /// Whether to enable caching
    pub enable_caching: bool,
}

impl Default for ContextEmbeddingConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 768,
            max_age: Duration::from_secs(60), // 1 minute
            max_cache_size: 100,
            enable_caching: true,
        }
    }
}

/// A generator for context embeddings with caching
#[derive(Debug)]
pub struct ContextEmbeddingGenerator {
    config: ContextEmbeddingConfig,
    cache: Mutex<HashMap<String, ContextEmbedding>>,
}

impl ContextEmbeddingGenerator {
    /// Create a new context embedding generator with the given configuration
    pub fn new(config: ContextEmbeddingConfig) -> Self {
        Self {
            config,
            cache: Mutex::new(HashMap::new()),
        }
    }
    
    /// Generate a context embedding for the given conversation history
    /// 
    /// This method will check the cache first if caching is enabled.
    /// If the cached embedding is stale or not found, a new embedding will be generated.
    pub fn generate(&self, 
                   history: &ConversationHistory, 
                   generator: &dyn Fn(&str) -> Result<Tensor>) -> Result<ContextEmbedding> {
        // Convert history to a string for cache key and embedding generation
        let prompt = history.format_for_prompt(4000);
        let cache_key = compute_cache_key(&prompt);
        
        // Check cache if enabled
        if self.config.enable_caching {
            let cache = self.cache.lock().unwrap();
            if let Some(embedding) = cache.get(&cache_key) {
                if !embedding.is_stale(self.config.max_age) {
                    return Ok(embedding.clone());
                }
            }
        }
        
        // Generate new embedding
        let tensor = generator(&prompt)
            .with_context(|| "Failed to generate embedding from conversation history")?;
        
        // Create metadata for embedding
        let mut metadata = HashMap::new();
        metadata.insert("turns".to_string(), format!("{}", history.get_turns().len()));
        metadata.insert("embedding_dim".to_string(), format!("{}", self.config.embedding_dim));
        
        // Create embedding
        let embedding = ContextEmbedding::new(tensor, Some(metadata));
        
        // Update cache if enabled
        if self.config.enable_caching {
            let mut cache = self.cache.lock().unwrap();
            // If cache is full, remove oldest entries
            if cache.len() >= self.config.max_cache_size {
                prune_cache(&mut cache, self.config.max_cache_size / 2);
            }
            cache.insert(cache_key, embedding.clone());
        }
        
        Ok(embedding)
    }
    
    /// Clear the embedding cache
    pub fn clear_cache(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();
    }
    
    /// Get the current cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        let cache = self.cache.lock().unwrap();
        let total = cache.len();
        let stale = cache.values().filter(|e| e.is_stale(self.config.max_age)).count();
        CacheStats { total, stale }
    }
}

/// Statistics about the embedding cache
#[derive(Debug, Clone, Copy)]
pub struct CacheStats {
    /// Total number of embeddings in the cache
    pub total: usize,
    /// Number of stale embeddings in the cache
    pub stale: usize,
}

/// Prune the cache to the target size by removing the oldest entries
fn prune_cache(cache: &mut HashMap<String, ContextEmbedding>, target_size: usize) {
    // If cache is already at or below target size, return early
    if cache.len() <= target_size {
        return;
    }
    
    // Sort cache entries by age (oldest first)
    let mut entries: Vec<(String, Instant)> = cache
        .iter()
        .map(|(k, v)| (k.clone(), v.generated_at))
        .collect();
    
    entries.sort_by_key(|(_, time)| *time);
    
    // Remove oldest entries until we reach target size
    let to_remove = cache.len() - target_size;
    for (key, _) in entries.into_iter().take(to_remove) {
        cache.remove(&key);
    }
}

/// Compute a cache key for the given prompt
fn compute_cache_key(prompt: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    prompt.hash(&mut hasher);
    format!("{:x}", hasher.finish())
} 