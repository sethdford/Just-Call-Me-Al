use std::collections::HashMap;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::path::Path;
use crate::tokenization::{Token, Tokenizer, TokenCache};
use super::vocab::{VocabLoader, SpecialTokens};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BPEVocab {
    /// Token to ID mapping
    pub token_to_id: HashMap<String, usize>,
    /// ID to token mapping
    pub id_to_token: HashMap<usize, String>,
    /// BPE merges with their scores, key is space-separated string pair
    pub merges: HashMap<String, (usize, f32)>,
}

impl BPEVocab {
    pub fn new() -> Self {
        Self {
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
            merges: HashMap::new(),
        }
    }

    /// Load vocabulary from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        let vocab: BPEVocab = serde_json::from_str(json)?;
        Ok(vocab)
    }

    /// Get the size of the vocabulary
    pub fn len(&self) -> usize {
        self.token_to_id.len()
    }

    /// Add a token to the vocabulary
    pub fn add_token(&mut self, token: &str, id: usize, _score: f32) {
        if !self.token_to_id.contains_key(token) {
            self.id_to_token.insert(id, token.to_string());
            self.token_to_id.insert(token.to_string(), id);
        }
    }

    /// Add a BPE merge rule (now expects string key)
    pub fn add_merge(&mut self, pair_str: String, id: usize, score: f32) {
        self.merges.insert(pair_str, (id, score));
    }
}

/// Implements the BPE tokenization algorithm
pub struct BPETokenizer {
    vocab: BPEVocab,
}

impl BPETokenizer {
    pub fn new(vocab: BPEVocab) -> Self {
        Self { vocab }
    }

    pub fn get_token(&self, id: usize) -> Option<&String> {
        self.vocab.id_to_token.get(&id)
    }

    pub fn get_token_id(&self, token: &str) -> Option<usize> {
        self.vocab.token_to_id.get(token).copied()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Find the best BPE merge in a sequence of tokens
    fn find_best_merge(&self, tokens: &[String]) -> Option<(usize, (String, String))> {
        let mut best_score = f32::NEG_INFINITY;
        let mut best_pair_tuple: Option<(String, String)> = None;
        let mut best_idx = 0;

        for i in 0..tokens.len() - 1 {
            let pair_str = format!("{} {}", tokens[i], tokens[i + 1]); // Construct string key
            if let Some(&(_, score)) = self.vocab.merges.get(&pair_str) {
                if score > best_score {
                    best_score = score;
                    best_pair_tuple = Some((tokens[i].clone(), tokens[i + 1].clone())); // Still store tuple for return
                    best_idx = i;
                }
            }
        }

        best_pair_tuple.map(|pair| (best_idx, pair))
    }

    /// Tokenize text using BPE
    pub fn tokenize(&self, text: &str) -> Result<Vec<Token>> {
        // Start with characters as initial tokens
        let mut current_tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();

        // Apply BPE merges iteratively until no more merges can be found in a pass
        loop {
            if let Some((idx, (first, second))) = self.find_best_merge(&current_tokens) {
                // Merge the best pair found in this pass
                let merged = format!("{}{}", first, second);
                current_tokens[idx] = merged;
                current_tokens.remove(idx + 1);
            } else {
                // No more merges possible in this pass, break the loop
                break;
            }
             // If only one token remains, we are done merging
             if current_tokens.len() == 1 {
                break;
             }
        }

        println!("Final tokens before conversion: {:?}", current_tokens);

        // Convert final token sequence to Token structs
        let mut result = Vec::with_capacity(current_tokens.len());
        for token_str in current_tokens {
            if let Some(&id) = self.vocab.token_to_id.get(&token_str) {
                result.push(Token {
                    text: token_str,
                    id,
                    score: 1.0, // TODO: Use actual score if available from vocab
                });
            } else {
                // Handle unknown tokens - this should ideally not happen if vocab loading is correct
                 // Including adding base characters
                return Err(anyhow!("Unknown token encountered after BPE: {}", token_str));
            }
        }

        Ok(result)
    }
}

/// Configuration for the Llama tokenizer
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    /// Path to the vocabulary file
    pub vocab_path: String,
    /// Maximum sequence length
    pub max_length: usize,
    /// Whether to add special tokens
    pub add_special_tokens: bool,
    /// Cache size for tokenization results
    pub cache_size: usize,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            vocab_path: "vocab.json".to_string(),
            max_length: 2048,
            add_special_tokens: true,
            cache_size: 10000,
        }
    }
}

/// Implementation of the Llama tokenizer
pub struct LlamaTokenizer {
    config: TokenizerConfig,
    bpe: BPETokenizer,
    special_tokens: SpecialTokens,
    cache: TokenCache,
}

impl LlamaTokenizer {
    pub fn new(config: TokenizerConfig) -> Result<Self> {
        let vocab_loader = VocabLoader::new(None);
        let vocab = vocab_loader.load_vocab(Path::new(&config.vocab_path))?;
        let bpe = BPETokenizer::new(vocab);
        let special_tokens = SpecialTokens::default();
        let cache = TokenCache::new(config.cache_size);

        Ok(Self {
            config,
            bpe,
            special_tokens,
            cache,
        })
    }

    fn tokenize_with_cache(&self, text: &str) -> Result<Vec<Token>> {
        if let Some(cached) = self.cache.get(text) {
            return Ok(cached);
        }

        let tokens = self.bpe.tokenize(text)?;
        self.cache.insert(text.to_string(), tokens.clone());
        Ok(tokens)
    }

    pub fn vocab_size(&self) -> usize {
        self.bpe.vocab_size()
    }
}

impl Tokenizer for LlamaTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<usize>> {
        let mut ids = Vec::new();
        
        // Add BOS token if requested
        if add_special_tokens {
            ids.push(self.bos_token_id());
        }
        
        // Tokenize the text
        let tokens = self.tokenize_with_cache(text)?;
        ids.extend(tokens.iter().map(|t| t.id));
        
        // Add EOS token if requested
        if add_special_tokens {
            ids.push(self.eos_token_id());
        }
        
        // Truncate if necessary
        if ids.len() > self.config.max_length {
            ids.truncate(self.config.max_length);
        }
        
        Ok(ids)
    }
    
    fn decode(&self, ids: &[usize], skip_special_tokens: bool) -> Result<String> {
        let special_ids = if skip_special_tokens {
            vec![
                self.pad_token_id(),
                self.bos_token_id(),
                self.eos_token_id(),
            ]
        } else {
            vec![]
        };

        let mut text = String::new();
        for &id in ids {
            if skip_special_tokens && special_ids.contains(&id) {
                continue;
            }
            
            let token = self.bpe.get_token(id)
                .ok_or_else(|| anyhow!("Unknown token ID: {}", id))?;
            text.push_str(token);
        }
        
        Ok(text)
    }
    
    fn vocab_size(&self) -> usize {
        self.bpe.vocab_size()
    }
    
    fn pad_token_id(&self) -> usize {
        self.bpe.get_token_id(&self.special_tokens.pad_token).unwrap_or(2)
    }
    
    fn unk_token_id(&self) -> usize {
        self.bpe.get_token_id(&self.special_tokens.unk_token).unwrap_or(3)
    }
    
    fn bos_token_id(&self) -> usize {
        self.bpe.get_token_id(&self.special_tokens.bos_token).unwrap_or(0)
    }
    
    fn eos_token_id(&self) -> usize {
        self.bpe.get_token_id(&self.special_tokens.eos_token).unwrap_or(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_vocab() -> BPEVocab {
        let mut vocab = BPEVocab::new();
        
        // Add some basic tokens
        vocab.add_token("a", 1, 1.0);
        vocab.add_token("b", 2, 1.0);
        vocab.add_token("ab", 3, 2.0);
        
        // Add merge rules
        vocab.add_merge("a b".to_string(), 3, 2.0);
        
        vocab
    }

    #[test]
    fn test_bpe_tokenization() {
        let vocab = create_test_vocab();
        let tokenizer = BPETokenizer::new(vocab);
        
        // Test basic merge
        let tokens = tokenizer.tokenize("ab").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].id, 3);
        assert_eq!(tokens[0].text, "ab");
        
        // Test no merge possible
        let tokens = tokenizer.tokenize("a").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].id, 1);
        assert_eq!(tokens[0].text, "a");
    }

    #[test]
    fn test_vocab_operations() {
        let mut vocab = BPEVocab::new();
        
        // Test adding tokens
        vocab.add_token("test", 1, 1.0);
        assert_eq!(vocab.len(), 1);
        assert_eq!(vocab.token_to_id.get("test"), Some(&1));
        assert_eq!(vocab.id_to_token.get(&1), Some(&"test".to_string()));
        
        // Test adding merges
        vocab.add_merge("te st".to_string(), 2, 1.5);
        assert!(vocab.merges.contains_key(&("te st".to_string())));
    }
} 