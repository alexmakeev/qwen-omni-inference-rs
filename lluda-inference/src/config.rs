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

// ── Qwen2.5-Omni configuration ────────────────────────────────────────────────

/// Audio encoder configuration (Whisper-style).
///
/// Corresponds to `thinker_config.audio_config` in the Qwen2.5-Omni `config.json`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OmniAudioConfig {
    /// Hidden dimension of the audio encoder.
    /// For Qwen2.5-Omni-3B: 1280
    pub d_model: usize,

    /// Number of encoder transformer layers.
    /// For Qwen2.5-Omni-3B: 32
    pub encoder_layers: usize,

    /// Number of encoder attention heads.
    /// For Qwen2.5-Omni-3B: 20
    pub encoder_attention_heads: usize,

    /// Feed-forward network dimension in the encoder.
    /// For Qwen2.5-Omni-3B: 5120
    pub encoder_ffn_dim: usize,

    /// Output projection dimension (to text model hidden size).
    /// For Qwen2.5-Omni-3B: 2048
    pub output_dim: usize,

    /// Number of mel frequency bins for audio preprocessing.
    /// For Qwen2.5-Omni-3B: 128
    pub num_mel_bins: usize,

    /// LayerNorm epsilon for numerical stability.
    /// Defaults to 1e-5 when not present in the config file.
    #[serde(default = "default_audio_eps")]
    pub layer_norm_eps: f64,

    /// Maximum number of source positions for sinusoidal positional embedding.
    /// For Qwen2.5-Omni-3B: 1500
    /// Defaults to 1500 when not present in the config file.
    #[serde(default = "default_max_source_positions")]
    pub max_source_positions: usize,
}

fn default_audio_eps() -> f64 {
    1e-5
}

fn default_max_source_positions() -> usize {
    1500
}

/// Talker (TTS decoder) configuration.
///
/// Corresponds to `talker_config` in the Qwen2.5-Omni `config.json`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OmniTalkerConfig {
    /// Hidden dimension of the talker model.
    /// For Qwen2.5-Omni-3B: 896
    pub hidden_size: usize,

    /// MLP intermediate dimension size.
    /// For Qwen2.5-Omni-3B: 4864
    pub intermediate_size: usize,

    /// Number of transformer decoder layers.
    /// For Qwen2.5-Omni-3B: 24
    pub num_hidden_layers: usize,

    /// Number of query attention heads.
    /// For Qwen2.5-Omni-3B: 14
    pub num_attention_heads: usize,

    /// Number of key/value attention heads (for GQA).
    /// For Qwen2.5-Omni-3B: 2
    pub num_key_value_heads: usize,

    /// Dimension of each attention head.
    /// For Qwen2.5-Omni-3B: 64
    pub head_dim: usize,

    /// Vocabulary size (codec tokens).
    /// For Qwen2.5-Omni-3B: 8448
    pub vocab_size: usize,

    /// Audio embedding dimension matching the thinker output.
    /// For Qwen2.5-Omni-3B: 2048
    pub embedding_size: usize,

    /// RMSNorm epsilon for numerical stability.
    /// For Qwen2.5-Omni-3B: 1e-6
    pub rms_norm_eps: f64,

    /// RoPE theta base frequency.
    /// For Qwen2.5-Omni-3B: 1000000.0
    pub rope_theta: f64,

    /// Maximum sequence length for positional embeddings.
    /// For Qwen2.5-Omni-3B: 32768
    pub max_position_embeddings: usize,

    /// Whether attention layers have bias terms.
    /// For Qwen2.5-Omni-3B: absent in config (defaults to false)
    #[serde(default)]
    pub attention_bias: bool,

    /// Start token ID for TTS codec output.
    /// For Qwen2.5-Omni-3B: 8293
    #[serde(default = "default_tts_codec_start", rename = "tts_codec_start_token_id")]
    pub tts_codec_start: u32,

    /// End token ID for TTS codec output.
    /// For Qwen2.5-Omni-3B: 8294
    #[serde(default = "default_tts_codec_end", rename = "tts_codec_end_token_id")]
    pub tts_codec_end: u32,

    /// Pad token ID for TTS codec output.
    /// For Qwen2.5-Omni-3B: 8292
    #[serde(default = "default_tts_codec_pad", rename = "tts_codec_pad_token_id")]
    pub tts_codec_pad: u32,
}

fn default_tts_codec_start() -> u32 {
    8293
}
fn default_tts_codec_end() -> u32 {
    8294
}
fn default_tts_codec_pad() -> u32 {
    8292
}

/// Text model configuration within the thinker.
///
/// Corresponds to `thinker_config.text_config` in the Qwen2.5-Omni `config.json`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OmniTextConfig {
    /// Hidden dimension size (embedding dimension).
    /// For Qwen2.5-Omni-3B: 2048
    pub hidden_size: usize,

    /// MLP intermediate dimension size.
    /// For Qwen2.5-Omni-3B: 11008
    pub intermediate_size: usize,

    /// Number of transformer decoder layers.
    /// For Qwen2.5-Omni-3B: 36
    pub num_hidden_layers: usize,

    /// Number of query attention heads.
    /// For Qwen2.5-Omni-3B: 16
    pub num_attention_heads: usize,

    /// Number of key/value attention heads (for GQA).
    /// For Qwen2.5-Omni-3B: 2
    pub num_key_value_heads: usize,

    /// Dimension of each attention head.
    /// Defaults to 128 when not present in the config file.
    #[serde(default = "default_omni_text_head_dim")]
    pub head_dim: usize,

    /// Vocabulary size.
    /// For Qwen2.5-Omni-3B: 151936
    pub vocab_size: usize,

    /// RMSNorm epsilon for numerical stability.
    /// For Qwen2.5-Omni-3B: 1e-6
    pub rms_norm_eps: f64,

    /// RoPE theta base frequency.
    /// For Qwen2.5-Omni-3B: 1000000.0
    pub rope_theta: f64,

    /// Maximum sequence length for positional embeddings.
    /// For Qwen2.5-Omni-3B: 32768
    pub max_position_embeddings: usize,

    /// Whether attention layers have bias terms.
    /// For Qwen2.5-Omni-3B: absent in config (defaults to false)
    #[serde(default)]
    pub attention_bias: bool,

    /// Whether word embeddings are tied with LM head.
    /// For Qwen2.5-Omni-3B: false
    pub tie_word_embeddings: bool,
}

fn default_omni_text_head_dim() -> usize {
    128
}

/// Thinker (main multimodal model) configuration wrapper.
///
/// Corresponds to `thinker_config` in the Qwen2.5-Omni `config.json`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OmniThinkerConfig {
    /// Language model backbone configuration.
    pub text_config: OmniTextConfig,

    /// Whisper-style audio encoder configuration.
    pub audio_config: OmniAudioConfig,

    /// Token ID used to represent audio spans in the text sequence.
    /// Defaults to 151646 when not present in the config file.
    #[serde(default = "default_audio_token_index")]
    pub audio_token_index: u32,
}

fn default_audio_token_index() -> u32 {
    151646
}

/// Top-level Qwen2.5-Omni model configuration.
///
/// Loaded from HuggingFace `config.json` for Qwen2.5-Omni models.
/// The config is structured as nested sub-configs for each component.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OmniConfig {
    /// Thinker (main language + audio understanding) sub-config.
    pub thinker_config: OmniThinkerConfig,

    /// Talker (TTS decoder) sub-config.
    pub talker_config: OmniTalkerConfig,

    /// Whether audio output generation is enabled.
    /// For Qwen2.5-Omni-3B: true
    #[serde(default)]
    pub enable_audio_output: bool,

    /// Whether the talker component is active.
    /// For Qwen2.5-Omni-3B: true
    #[serde(default)]
    pub enable_talker: bool,
}

impl OmniConfig {
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
    /// use lluda_inference::config::OmniConfig;
    ///
    /// let config = OmniConfig::from_file("models/Qwen2.5-Omni-3B/config.json")?;
    /// assert_eq!(config.thinker_config.text_config.hidden_size, 2048);
    /// # Ok::<(), lluda_inference::error::LludaError>(())
    /// ```
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: OmniConfig = serde_json::from_str(&contents)?;
        Ok(config)
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

    #[test]
    fn test_omni_config_from_file() {
        let path = "/home/alexmak/lluda/models/Qwen2.5-Omni-3B/config.json";
        if !std::path::Path::new(path).exists() {
            return; // skip if model not available
        }
        let config = OmniConfig::from_file(path).unwrap();
        assert_eq!(config.thinker_config.text_config.hidden_size, 2048);
        assert_eq!(config.thinker_config.text_config.num_hidden_layers, 36);
        assert_eq!(config.thinker_config.audio_config.d_model, 1280);
        assert_eq!(config.thinker_config.audio_config.encoder_layers, 32);
        assert_eq!(config.thinker_config.audio_config.encoder_attention_heads, 20);
        assert_eq!(config.talker_config.hidden_size, 896);
        assert_eq!(config.talker_config.num_hidden_layers, 24);
        assert_eq!(config.talker_config.vocab_size, 8448);
    }
}
