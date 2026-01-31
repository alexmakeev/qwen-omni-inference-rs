//! Tokenizer wrapper for Qwen3 model.
//!
//! Wraps the HuggingFace tokenizers-rs library to provide text encoding/decoding
//! for the Qwen3-0.6B model (supports 119 languages including Chinese, Japanese, emoji).

use tokenizers::Tokenizer as HfTokenizer;

use crate::error::{LludaError, Result};

/// Tokenizer for Qwen3 model.
///
/// Wraps HuggingFace tokenizers-rs with BOS/EOS token management.
/// Qwen3-0.6B uses:
/// - BOS token ID: 151643
/// - EOS token IDs: [151645, 151643]
/// - Vocabulary size: 151936
#[derive(Debug)]
pub struct Tokenizer {
    tokenizer: HfTokenizer,
    bos_token_id: u32,
    eos_token_ids: Vec<u32>,
}

impl Tokenizer {
    /// Load tokenizer from file.
    ///
    /// # Arguments
    /// * `tokenizer_path` - Path to tokenizer.json file
    /// * `bos_token_id` - Beginning-of-sequence token ID
    /// * `eos_token_ids` - End-of-sequence token IDs
    ///
    /// # Example
    /// ```no_run
    /// use lluda_inference::tokenizer::Tokenizer;
    ///
    /// let tokenizer = Tokenizer::from_file(
    ///     "models/Qwen3-0.6B/tokenizer.json",
    ///     151643,
    ///     vec![151645, 151643],
    /// ).unwrap();
    /// ```
    pub fn from_file(
        tokenizer_path: impl AsRef<std::path::Path>,
        bos_token_id: u32,
        eos_token_ids: Vec<u32>,
    ) -> Result<Self> {
        let tokenizer = HfTokenizer::from_file(tokenizer_path)
            .map_err(|e| LludaError::Tokenizer(e.to_string()))?;

        Ok(Self {
            tokenizer,
            bos_token_id,
            eos_token_ids,
        })
    }

    /// Encode text to token IDs.
    ///
    /// # Arguments
    /// * `text` - Text to encode
    /// * `add_special_tokens` - Whether to add BOS/EOS tokens
    ///
    /// # Returns
    /// Vector of token IDs
    ///
    /// # Example
    /// ```no_run
    /// # use lluda_inference::tokenizer::Tokenizer;
    /// # let tokenizer = Tokenizer::from_file("models/Qwen3-0.6B/tokenizer.json", 151643, vec![151645, 151643]).unwrap();
    /// let ids = tokenizer.encode("Hello, world!", true).unwrap();
    /// assert!(!ids.is_empty());
    /// ```
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| LludaError::Tokenizer(e.to_string()))?;

        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs to text.
    ///
    /// # Arguments
    /// * `ids` - Token IDs to decode
    /// * `skip_special_tokens` - Whether to skip BOS/EOS tokens in output
    ///
    /// # Returns
    /// Decoded text string
    ///
    /// # Example
    /// ```no_run
    /// # use lluda_inference::tokenizer::Tokenizer;
    /// # let tokenizer = Tokenizer::from_file("models/Qwen3-0.6B/tokenizer.json", 151643, vec![151645, 151643]).unwrap();
    /// let text = tokenizer.decode(&[151643, 9906, 151645], false).unwrap();
    /// assert!(!text.is_empty());
    /// ```
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let text = self
            .tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| LludaError::Tokenizer(e.to_string()))?;

        Ok(text)
    }

    /// Get vocabulary size.
    ///
    /// For Qwen3-0.6B, this should be 151936.
    ///
    /// # Example
    /// ```no_run
    /// # use lluda_inference::tokenizer::Tokenizer;
    /// # let tokenizer = Tokenizer::from_file("models/Qwen3-0.6B/tokenizer.json", 151643, vec![151645, 151643]).unwrap();
    /// assert_eq!(tokenizer.vocab_size(), 151936);
    /// ```
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// Get BOS (beginning-of-sequence) token ID.
    ///
    /// For Qwen3-0.6B, this is 151643.
    pub fn bos_token_id(&self) -> u32 {
        self.bos_token_id
    }

    /// Get EOS (end-of-sequence) token IDs.
    ///
    /// For Qwen3-0.6B, this is [151645, 151643].
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// Helper to load tokenizer (skips test if file not present)
    fn load_test_tokenizer() -> Option<Tokenizer> {
        let tokenizer_path = Path::new("models/Qwen3-0.6B/tokenizer.json");
        if !tokenizer_path.exists() {
            eprintln!(
                "Skipping tokenizer test: file not found at {}",
                tokenizer_path.display()
            );
            return None;
        }

        Some(
            Tokenizer::from_file(tokenizer_path, 151643, vec![151645, 151643])
                .expect("Failed to load tokenizer"),
        )
    }

    #[test]
    fn test_load_tokenizer() {
        let tokenizer = match load_test_tokenizer() {
            Some(t) => t,
            None => return,
        };

        assert_eq!(tokenizer.vocab_size(), 151936);
        assert_eq!(tokenizer.bos_token_id(), 151643);
        assert_eq!(tokenizer.eos_token_ids(), &[151645, 151643]);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let tokenizer = match load_test_tokenizer() {
            Some(t) => t,
            None => return,
        };

        let text = "Hello, world!";
        let ids = tokenizer.encode(text, false).unwrap();
        assert!(!ids.is_empty());

        let decoded = tokenizer.decode(&ids, false).unwrap();
        // Should preserve text (modulo whitespace/normalization)
        assert!(
            decoded.contains("Hello") && decoded.contains("world"),
            "Decoded text '{}' doesn't contain 'Hello' and 'world'",
            decoded
        );
    }

    #[test]
    fn test_encode_with_special_tokens() {
        let tokenizer = match load_test_tokenizer() {
            Some(t) => t,
            None => return,
        };

        let text = "Test";
        let ids_without_special = tokenizer.encode(text, false).unwrap();
        let ids_with_special = tokenizer.encode(text, true).unwrap();

        // With special tokens should be longer (BOS added)
        assert!(
            ids_with_special.len() >= ids_without_special.len(),
            "Expected special tokens to be added"
        );
    }

    #[test]
    fn test_decode_with_skip_special_tokens() {
        let tokenizer = match load_test_tokenizer() {
            Some(t) => t,
            None => return,
        };

        let text = "Hello";
        let ids = tokenizer.encode(text, true).unwrap();

        let decoded_with_special = tokenizer.decode(&ids, false).unwrap();
        let decoded_without_special = tokenizer.decode(&ids, true).unwrap();

        // Skipping special tokens should give cleaner output
        assert!(
            decoded_without_special.len() <= decoded_with_special.len(),
            "Expected skip_special_tokens=true to produce cleaner output"
        );
    }

    #[test]
    fn test_unicode_handling() {
        let tokenizer = match load_test_tokenizer() {
            Some(t) => t,
            None => return,
        };

        // Test Chinese (Qwen supports 119 languages)
        let chinese = "你好世界";
        let ids = tokenizer.encode(chinese, false).unwrap();
        let decoded = tokenizer.decode(&ids, false).unwrap();
        assert!(
            decoded.contains("你好") && decoded.contains("世界"),
            "Failed to roundtrip Chinese text: '{}'",
            decoded
        );

        // Test emoji
        let emoji = "Hello 👋 World 🌍";
        let ids = tokenizer.encode(emoji, false).unwrap();
        let decoded = tokenizer.decode(&ids, false).unwrap();
        assert!(
            decoded.contains("Hello") && decoded.contains("World"),
            "Failed to roundtrip emoji text: '{}'",
            decoded
        );

        // Test Japanese
        let japanese = "こんにちは";
        let ids = tokenizer.encode(japanese, false).unwrap();
        let decoded = tokenizer.decode(&ids, false).unwrap();
        assert!(
            decoded.contains("こんにちは"),
            "Failed to roundtrip Japanese text: '{}'",
            decoded
        );
    }

    #[test]
    fn test_empty_string() {
        let tokenizer = match load_test_tokenizer() {
            Some(t) => t,
            None => return,
        };

        let ids = tokenizer.encode("", false).unwrap();
        // Empty string may produce empty or minimal token list
        assert!(
            ids.is_empty() || ids.len() <= 1,
            "Expected empty or single token for empty string, got {} tokens",
            ids.len()
        );
    }

    #[test]
    fn test_error_handling_invalid_file() {
        let result = Tokenizer::from_file(
            "/nonexistent/path/tokenizer.json",
            151643,
            vec![151645, 151643],
        );
        assert!(result.is_err());

        match result.unwrap_err() {
            LludaError::Tokenizer(msg) => {
                assert!(!msg.is_empty(), "Expected error message");
            }
            other => panic!("Expected Tokenizer error, got: {:?}", other),
        }
    }

    #[test]
    fn test_vocab_size() {
        let tokenizer = match load_test_tokenizer() {
            Some(t) => t,
            None => return,
        };

        let vocab_size = tokenizer.vocab_size();
        assert_eq!(
            vocab_size, 151936,
            "Expected Qwen3-0.6B vocab size of 151936, got {}",
            vocab_size
        );
    }

    #[test]
    fn test_bos_eos_tokens() {
        let tokenizer = match load_test_tokenizer() {
            Some(t) => t,
            None => return,
        };

        assert_eq!(tokenizer.bos_token_id(), 151643);
        assert_eq!(tokenizer.eos_token_ids(), &[151645, 151643]);

        // Verify EOS token IDs are returned as slice
        let eos_slice = tokenizer.eos_token_ids();
        assert_eq!(eos_slice.len(), 2);
        assert_eq!(eos_slice[0], 151645);
        assert_eq!(eos_slice[1], 151643);
    }

    #[test]
    fn test_long_text_encoding() {
        let tokenizer = match load_test_tokenizer() {
            Some(t) => t,
            None => return,
        };

        // Test with longer text
        let long_text = "The quick brown fox jumps over the lazy dog. \
                         This is a longer text to test tokenization. \
                         It should handle multiple sentences without issues.";
        let ids = tokenizer.encode(long_text, false).unwrap();
        assert!(
            ids.len() > 10,
            "Expected long text to produce many tokens, got {}",
            ids.len()
        );

        let decoded = tokenizer.decode(&ids, false).unwrap();
        // Should preserve key words
        assert!(decoded.contains("quick") && decoded.contains("fox"));
    }

    #[test]
    fn test_special_characters() {
        let tokenizer = match load_test_tokenizer() {
            Some(t) => t,
            None => return,
        };

        // Test various special characters
        let text = "Hello! How are you? I'm fine, thanks. @user #hashtag";
        let ids = tokenizer.encode(text, false).unwrap();
        let decoded = tokenizer.decode(&ids, false).unwrap();

        assert!(
            decoded.contains("Hello") && decoded.contains("thanks"),
            "Failed to handle special characters: '{}'",
            decoded
        );
    }
}
