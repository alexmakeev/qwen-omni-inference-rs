//! Model configuration parsing for Qwen3 models.
//!
//! Loads and parses the HuggingFace `config.json` file to extract
//! model hyperparameters needed for architecture construction.

use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Configuration for Qwen3 models.
///
/// Loaded from HuggingFace `config.json` files.
/// Contains all hyperparameters needed to construct the model architecture.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Qwen3Config {
    /// Hidden dimension size (embedding dimension).
    /// For Qwen3-0.6B: 1024
    pub hidden_size: usize,

    /// MLP intermediate dimension size.
    /// For Qwen3-0.6B: 3072 (3x hidden_size)
    pub intermediate_size: usize,

    /// Number of transformer decoder layers.
    /// For Qwen3-0.6B: 28
    pub num_hidden_layers: usize,

    /// Number of query attention heads.
    /// For Qwen3-0.6B: 16
    pub num_attention_heads: usize,

    /// Number of key/value attention heads (for GQA).
    /// For Qwen3-0.6B: 8
    pub num_key_value_heads: usize,

    /// Dimension of each attention head.
    /// For Qwen3-0.6B: 128
    pub head_dim: usize,

    /// Vocabulary size.
    /// For Qwen3-0.6B: 151936
    pub vocab_size: usize,

    /// Maximum sequence length for positional embeddings.
    /// For Qwen3-0.6B: 40960
    pub max_position_embeddings: usize,

    /// RoPE theta base frequency.
    /// For Qwen3-0.6B: 1000000.0
    pub rope_theta: f64,

    /// RMSNorm epsilon for numerical stability.
    /// For Qwen3-0.6B: 1e-6
    pub rms_norm_eps: f64,

    /// Hidden layer activation function.
    /// For Qwen3-0.6B: "silu"
    pub hidden_act: String,

    /// Whether attention layers have bias terms.
    /// For Qwen3-0.6B: false
    pub attention_bias: bool,

    /// Attention dropout probability.
    /// For Qwen3-0.6B: 0.0 (disabled)
    #[serde(default)]
    pub attention_dropout: f64,

    /// BOS (beginning of sequence) token ID.
    /// For Qwen3-0.6B: 151643
    pub bos_token_id: u32,

    /// EOS (end of sequence) token ID.
    /// Can be a single token or multiple tokens.
    /// For Qwen3-0.6B: 151645
    pub eos_token_id: EosTokenId,

    /// Whether word embeddings are tied with LM head.
    /// For Qwen3-0.6B: true (embedding weight is reused as LM head)
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

/// EOS token ID configuration.
///
/// Supports both single token and multiple token formats.
/// HuggingFace configs may specify EOS as either:
/// - A single integer: `"eos_token_id": 151645`
/// - An array: `"eos_token_id": [151645, 151643]`
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum EosTokenId {
    /// Single EOS token ID.
    Single(u32),
    /// Multiple EOS token IDs (any can trigger end of generation).
    Multiple(Vec<u32>),
}

impl Qwen3Config {
    /// Load configuration from a JSON file.
    ///
    /// # Arguments
    /// * `path` - Path to the `config.json` file
    ///
    /// # Returns
    /// Parsed configuration or error if file cannot be read or JSON is invalid.
    ///
    /// # Example
    /// ```no_run
    /// use lluda_inference::config::Qwen3Config;
    ///
    /// let config = Qwen3Config::from_file("models/Qwen3-0.6B/config.json")?;
    /// assert_eq!(config.hidden_size, 1024);
    /// # Ok::<(), lluda_inference::error::LludaError>(())
    /// ```
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: Qwen3Config = serde_json::from_str(&contents)?;
        Ok(config)
    }

    /// Get number of KV groups for Grouped Query Attention.
    ///
    /// # Returns
    /// Number of query heads per KV head (num_attention_heads / num_key_value_heads).
    /// For Qwen3-0.6B: 2 (16 / 8)
    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_qwen3_06b_config() {
        let config = Qwen3Config::from_file("../models/Qwen3-0.6B/config.json").unwrap();

        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.intermediate_size, 3072);
        assert_eq!(config.num_hidden_layers, 28);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.max_position_embeddings, 40960);
        assert_eq!(config.rope_theta, 1000000.0);
        assert_eq!(config.rms_norm_eps, 1e-6);
        assert_eq!(config.hidden_act, "silu");
        assert!(!config.attention_bias);
        assert_eq!(config.attention_dropout, 0.0);
        assert_eq!(config.bos_token_id, 151643);
        assert!(config.tie_word_embeddings);
    }

    #[test]
    fn test_num_kv_groups() {
        let config = Qwen3Config::from_file("../models/Qwen3-0.6B/config.json").unwrap();
        assert_eq!(config.num_kv_groups(), 2); // 16 / 8
    }

    #[test]
    fn test_eos_token_id_single() {
        let json = r#"{
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "attention_bias": false,
            "bos_token_id": 151643,
            "eos_token_id": 151645
        }"#;

        let config: Qwen3Config = serde_json::from_str(json).unwrap();
        match config.eos_token_id {
            EosTokenId::Single(id) => assert_eq!(id, 151645),
            EosTokenId::Multiple(_) => panic!("Expected Single variant"),
        }
    }

    #[test]
    fn test_eos_token_id_multiple() {
        let json = r#"{
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "attention_bias": false,
            "bos_token_id": 151643,
            "eos_token_id": [151645, 151643]
        }"#;

        let config: Qwen3Config = serde_json::from_str(json).unwrap();
        match config.eos_token_id {
            EosTokenId::Multiple(ids) => {
                assert_eq!(ids.len(), 2);
                assert_eq!(ids[0], 151645);
                assert_eq!(ids[1], 151643);
            }
            EosTokenId::Single(_) => panic!("Expected Multiple variant"),
        }
    }

    #[test]
    fn test_default_fields() {
        // Test that optional fields get default values when missing
        let json = r#"{
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "attention_bias": false,
            "bos_token_id": 151643,
            "eos_token_id": 151645
        }"#;

        let config: Qwen3Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.attention_dropout, 0.0);
        assert!(!config.tie_word_embeddings); // default is false when not specified
    }

    #[test]
    fn test_invalid_json() {
        let json = r#"{ "invalid": json }"#;
        let result = serde_json::from_str::<Qwen3Config>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_required_field() {
        // Missing hidden_size field
        let json = r#"{
            "intermediate_size": 3072,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "attention_bias": false,
            "bos_token_id": 151643,
            "eos_token_id": 151645
        }"#;

        let result = serde_json::from_str::<Qwen3Config>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_file_not_found() {
        let result = Qwen3Config::from_file("/nonexistent/path/config.json");
        assert!(result.is_err());
        match result.unwrap_err() {
            crate::error::LludaError::Io(_) => (),
            other => panic!("Expected IO error, got: {:?}", other),
        }
    }

    #[test]
    fn test_config_roundtrip() {
        // Test serialization and deserialization
        let config = Qwen3Config::from_file("../models/Qwen3-0.6B/config.json").unwrap();

        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: Qwen3Config = serde_json::from_str(&serialized).unwrap();

        assert_eq!(config.hidden_size, deserialized.hidden_size);
        assert_eq!(config.num_hidden_layers, deserialized.num_hidden_layers);
        assert_eq!(config.vocab_size, deserialized.vocab_size);
    }
}
