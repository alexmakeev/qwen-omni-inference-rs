//! Model configuration parsing for Qwen3 models.
//!
//! Loads and parses the HuggingFace `config.json` file to extract
//! model hyperparameters needed for architecture construction.
//!
//! Supports both flat Qwen3 configs and nested Qwen2.5-Omni configs.
//! Auto-detection is based on the presence of `thinker_config` key.

use serde::{Deserialize, Serialize};

use crate::error::Result;

// ──────────────────────────────────────────────────────────────────────────────
// Private helper structs for Qwen2.5-Omni nested config deserialization.
// These are used only in `from_file()` and not exposed publicly.
// ──────────────────────────────────────────────────────────────────────────────

/// Flat text model parameters nested inside `thinker_config.text_config`.
#[derive(Debug, Deserialize)]
struct OmniTextConfigRaw {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    rope_theta: f64,
    rms_norm_eps: f64,
    hidden_act: String,
    #[serde(default)]
    attention_dropout: f64,
    #[serde(default)]
    tie_word_embeddings: bool,
}

/// Top-level `thinker_config` object in an Omni config.
#[derive(Debug, Deserialize)]
struct OmniThinkerConfigRaw {
    text_config: OmniTextConfigRaw,
    bos_token_id: u32,
    eos_token_id: EosTokenId,
}

/// Root structure for Qwen2.5-Omni `config.json`.
#[derive(Debug, Deserialize)]
struct OmniConfigRaw {
    thinker_config: OmniThinkerConfigRaw,
}

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

    /// Tensor name prefix for weight lookup.
    /// Empty string for flat Qwen3 models, "thinker." for Qwen2.5-Omni.
    #[serde(default)]
    pub tensor_prefix: String,

    /// Whether attention layers have per-head Q/K RMSNorm.
    /// True for Qwen3 flat models, false for Qwen2.5-Omni.
    #[serde(default = "default_true")]
    pub has_qk_norm: bool,
}

fn default_true() -> bool {
    true
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
    /// Automatically detects whether the file is a flat Qwen3 config or a
    /// nested Qwen2.5-Omni config (detected by the presence of a top-level
    /// `thinker_config` key). In the Omni case the text model parameters are
    /// extracted from `thinker_config.text_config` and the token IDs from
    /// `thinker_config` itself.
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

        // Use serde_json::Value for format detection before full deserialization.
        let value: serde_json::Value = serde_json::from_str(&contents)?;

        if value.get("thinker_config").is_some() {
            // Qwen2.5-Omni nested format: extract text params from thinker_config.
            let omni: OmniConfigRaw = serde_json::from_value(value)?;
            let tc = omni.thinker_config;
            let txt = tc.text_config;
            let head_dim = txt.hidden_size / txt.num_attention_heads;
            Ok(Qwen3Config {
                hidden_size: txt.hidden_size,
                intermediate_size: txt.intermediate_size,
                num_hidden_layers: txt.num_hidden_layers,
                num_attention_heads: txt.num_attention_heads,
                num_key_value_heads: txt.num_key_value_heads,
                head_dim,
                vocab_size: txt.vocab_size,
                max_position_embeddings: txt.max_position_embeddings,
                rope_theta: txt.rope_theta,
                rms_norm_eps: txt.rms_norm_eps,
                hidden_act: txt.hidden_act,
                attention_bias: false,
                attention_dropout: txt.attention_dropout,
                bos_token_id: tc.bos_token_id,
                eos_token_id: tc.eos_token_id,
                tie_word_embeddings: txt.tie_word_embeddings,
                tensor_prefix: "thinker.".to_string(),
                has_qk_norm: false,
            })
        } else {
            // Flat Qwen3 format.
            let config: Qwen3Config = serde_json::from_value(value)?;
            Ok(config)
        }
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
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "tie_word_embeddings": true
        }"#;

        let config: Qwen3Config = serde_json::from_str(json).unwrap();

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

        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: Qwen3Config = serde_json::from_str(&serialized).unwrap();

        assert_eq!(config.hidden_size, deserialized.hidden_size);
        assert_eq!(config.num_hidden_layers, deserialized.num_hidden_layers);
        assert_eq!(config.vocab_size, deserialized.vocab_size);
    }

    /// Unit test for Omni nested config using inline JSON written to a temp file.
    /// Verifies correct extraction of text model parameters from thinker_config
    /// through the public `from_file()` entry point.
    #[test]
    fn test_omni_config_inline() {
        // Minimal Omni-style config with nested thinker_config.text_config structure.
        let json = r#"{
            "model_type": "qwen2_5_omni",
            "thinker_config": {
                "bos_token_id": 151644,
                "eos_token_id": 151645,
                "text_config": {
                    "hidden_size": 2048,
                    "intermediate_size": 11008,
                    "num_hidden_layers": 36,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 2,
                    "vocab_size": 151936,
                    "max_position_embeddings": 32768,
                    "rope_theta": 1000000.0,
                    "rms_norm_eps": 1e-6,
                    "hidden_act": "silu",
                    "attention_dropout": 0.0,
                    "tie_word_embeddings": false
                }
            }
        }"#;

        // Write to a temp file so we exercise the full from_file() path.
        let tmp = std::env::temp_dir().join("lluda_test_omni_config.json");
        std::fs::write(&tmp, json).expect("failed to write temp config");

        let config = Qwen3Config::from_file(&tmp).expect("failed to parse inline Omni config");

        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_hidden_layers, 36);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_key_value_heads, 2);
        assert_eq!(config.head_dim, 128); // computed: 2048 / 16
        assert!(!config.attention_bias);
        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.rope_theta, 1000000.0);
        assert_eq!(config.bos_token_id, 151644);
        match &config.eos_token_id {
            EosTokenId::Single(id) => assert_eq!(*id, 151645),
            EosTokenId::Multiple(_) => panic!("Expected Single eos_token_id"),
        }

        // Clean up temp file.
        let _ = std::fs::remove_file(&tmp);
    }

    /// Integration test: load real Qwen2.5-Omni-3B config via `from_file()`.
    /// Skipped gracefully when the model directory is not present.
    #[test]
    fn test_load_omni_config() {
        let path = "/home/alexmak/lluda/models/Qwen2.5-Omni-3B/config.json";
        if !std::path::Path::new(path).exists() {
            eprintln!("Skipping test_load_omni_config: model file not found at {path}");
            return;
        }

        let config = Qwen3Config::from_file(path).expect("failed to parse Omni config");

        // Values derived from actual config.json inspection.
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_hidden_layers, 36);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_key_value_heads, 2);
        assert_eq!(config.head_dim, 128); // computed: 2048 / 16
        assert!(!config.attention_bias);
        assert_eq!(config.bos_token_id, 151644);
        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.rope_theta, 1000000.0);
    }
}
