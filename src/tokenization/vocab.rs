use std::path::Path;
use std::fs;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use crate::tokenization::bpe::BPEVocab;
use log;

/// Special tokens used by the Llama tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialTokens {
    pub bos_token: String,
    pub eos_token: String,
    pub pad_token: String,
    pub unk_token: String,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            bos_token: "<s>".to_string(),
            eos_token: "</s>".to_string(),
            pad_token: "<pad>".to_string(),
            unk_token: "<unk>".to_string(),
        }
    }
}

/// Vocabulary loader that handles loading and preprocessing of the vocabulary
pub struct VocabLoader {
    special_tokens: SpecialTokens,
}

impl VocabLoader {
    pub fn new(special_tokens: Option<SpecialTokens>) -> Self {
        Self {
            special_tokens: special_tokens.unwrap_or_default(),
        }
    }

    /// Load vocabulary from a JSON file
    pub fn load_vocab(&self, vocab_path: &Path) -> Result<BPEVocab> {
        let json = fs::read_to_string(vocab_path)
            .map_err(|e| anyhow!("Failed to read vocabulary file: {}", e))?;

        let mut vocab = BPEVocab::from_json(&json)?;

        // Add special tokens
        self.add_special_tokens(&mut vocab)?;

        // --- Add derived tokens from merges --- 
        let mut next_id = vocab.len(); // Start assigning IDs after existing + special tokens
        let merges_to_process: Vec<(String, (usize, f32))> = vocab.merges.clone().into_iter().collect();
        
        // Collect all characters involved in tokens and merges
        let mut chars_to_add = std::collections::HashSet::new();
        for token in vocab.token_to_id.keys() {
            for char in token.chars() { chars_to_add.insert(char); }
        }
        for merge_key in vocab.merges.keys() {
            for char in merge_key.chars().filter(|c| *c != ' ') { chars_to_add.insert(char); }
        }

        for (merge_key, (existing_id, score)) in merges_to_process {
            // Derive the merged token string (remove space)
            let merged_token_string: String = merge_key.chars().filter(|c| *c != ' ').collect();

            if !vocab.token_to_id.contains_key(&merged_token_string) {
                // Check if the ID assigned in the merge file is already used.
                // If the ID is higher than current vocab size, it might be intended as the new token's ID.
                // Otherwise, assign a new ID.
                let final_id = if existing_id >= next_id {
                    // Potentially use the ID from the merge file if it seems intended
                    existing_id 
                } else {
                    // Assign the next available ID
                     let id_to_assign = next_id;
                     next_id += 1;
                     id_to_assign
                };

                // Ensure the final_id is not already taken before adding
                if !vocab.id_to_token.contains_key(&final_id) {
                    vocab.add_token(&merged_token_string, final_id, score);
                    // We might need to update the merge entry itself if we assigned a new ID
                    // Although, the current BPE logic seems to use the merge score, not the ID
                } else {
                    // Handle ID conflict - potentially log a warning or error
                    // For now, let's just skip adding if ID is taken, assuming vocab is consistent
                     log::warn!("ID {} for derived token '{}' from merge '{}' is already taken. Skipping.", final_id, merged_token_string, merge_key);
                }
            }
        }
        // ---------------------------------------

        // --- Add base character tokens --- 
        for char_to_add in chars_to_add {
            let char_str = char_to_add.to_string();
            if !vocab.token_to_id.contains_key(&char_str) {
                vocab.add_token(&char_str, next_id, 0.0); // Assign score 0.0 or lowest?
                next_id += 1;
            }
        }
        // ----------------------------------

        Ok(vocab)
    }

    /// Add special tokens to the vocabulary
    fn add_special_tokens(&self, vocab: &mut BPEVocab) -> Result<()> {
        let special_tokens = [
            (&self.special_tokens.bos_token, 0),
            (&self.special_tokens.eos_token, 1),
            (&self.special_tokens.pad_token, 2),
            (&self.special_tokens.unk_token, 3),
        ];

        for (token, id) in special_tokens.iter() {
            vocab.add_token(token, *id, 0.0); // Special tokens have score 0.0
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_special_tokens() {
        let special_tokens = SpecialTokens::default();
        assert_eq!(special_tokens.bos_token, "<s>");
        assert_eq!(special_tokens.eos_token, "</s>");
        assert_eq!(special_tokens.pad_token, "<pad>");
        assert_eq!(special_tokens.unk_token, "<unk>");
    }

    #[test]
    fn test_vocab_loading() -> Result<()> {
        // Create a temporary directory
        let dir = tempdir()?;
        let vocab_path = dir.path().join("vocab.json");

        // Create a test vocabulary file
        let test_vocab = r#"{
            "token_to_id": {},
            "id_to_token": {},
            "merges": {}
        }"#;
        let mut file = File::create(&vocab_path)?;
        file.write_all(test_vocab.as_bytes())?;

        // Load the vocabulary
        let loader = VocabLoader::new(None);
        let vocab = loader.load_vocab(&vocab_path)?;

        // Check if special tokens were added
        assert!(vocab.token_to_id.contains_key("<s>"));
        assert!(vocab.token_to_id.contains_key("</s>"));
        assert!(vocab.token_to_id.contains_key("<pad>"));
        assert!(vocab.token_to_id.contains_key("<unk>"));

        Ok(())
    }

    #[test]
    fn test_custom_special_tokens() {
        let custom_tokens = SpecialTokens {
            bos_token: "<BOS>".to_string(),
            eos_token: "<EOS>".to_string(),
            pad_token: "<PAD>".to_string(),
            unk_token: "<UNK>".to_string(),
        };

        let loader = VocabLoader::new(Some(custom_tokens.clone()));
        let mut vocab = BPEVocab::new();
        loader.add_special_tokens(&mut vocab).unwrap();

        assert_eq!(vocab.token_to_id.get(&custom_tokens.bos_token), Some(&0));
        assert_eq!(vocab.token_to_id.get(&custom_tokens.eos_token), Some(&1));
        assert_eq!(vocab.token_to_id.get(&custom_tokens.pad_token), Some(&2));
        assert_eq!(vocab.token_to_id.get(&custom_tokens.unk_token), Some(&3));
    }
} 