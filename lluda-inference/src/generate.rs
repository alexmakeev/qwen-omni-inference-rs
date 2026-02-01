//! Autoregressive text generation for Qwen3 models.
//!
//! Implements the generation loop for autoregressive language models:
//! 1. Encode input prompt to token IDs
//! 2. Forward pass through model (prefill)
//! 3. Sample next token from logits
//! 4. Iteratively generate tokens until EOS or max_new_tokens
//!
//! Supports various sampling strategies:
//! - Greedy decoding (argmax)
//! - Temperature sampling
//! - Top-k sampling
//! - Top-p (nucleus) sampling
//!
//! # Example
//!
//! ```ignore
//! use lluda_inference::generate::{GenerationConfig, generate};
//! use lluda_inference::tokenizer::Tokenizer;
//! // Assume model is loaded...
//!
//! let config = GenerationConfig {
//!     max_new_tokens: 20,
//!     temperature: 0.0, // Greedy
//!     top_k: 0,
//!     top_p: 1.0,
//!     repetition_penalty: 1.0,
//! };
//!
//! // let output = generate(&mut model, &tokenizer, "Hello, world!", &config).unwrap();
//! ```

use crate::error::{LludaError, Result};
use crate::tensor::Tensor;

/// Configuration for text generation.
///
/// Controls sampling behavior and generation length.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: usize,

    /// Sampling temperature.
    /// - 0.0: Greedy decoding (argmax)
    /// - < 1.0: Sharper distribution (more deterministic)
    /// - > 1.0: Flatter distribution (more random)
    pub temperature: f32,

    /// Top-k sampling: only sample from top k tokens.
    /// - 0: Disabled (use all tokens)
    /// - > 0: Only consider top k highest probability tokens
    pub top_k: usize,

    /// Top-p (nucleus) sampling: sample from smallest set of tokens with cumulative probability >= p.
    /// - 1.0: Disabled (use all tokens)
    /// - < 1.0: Only consider tokens in the top-p cumulative probability mass
    pub top_p: f32,

    /// Repetition penalty.
    /// - 1.0: Disabled (no penalty)
    /// - > 1.0: Penalize repeated tokens (divide logits by this value)
    pub repetition_penalty: f32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 100,
            temperature: 0.0, // Greedy by default
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
        }
    }
}

impl GenerationConfig {
    /// Create a new generation config with validation.
    ///
    /// # Arguments
    ///
    /// * `max_new_tokens` - Maximum tokens to generate
    /// * `temperature` - Sampling temperature (>= 0.0)
    /// * `top_k` - Top-k sampling (0 to disable)
    /// * `top_p` - Top-p sampling (0.0 to 1.0)
    /// * `repetition_penalty` - Repetition penalty (>= 1.0)
    ///
    /// # Returns
    ///
    /// Validated GenerationConfig.
    ///
    /// # Errors
    ///
    /// Returns error if parameters are out of valid range.
    pub fn new(
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
    ) -> Result<Self> {
        if temperature < 0.0 {
            return Err(LludaError::Msg(format!(
                "temperature must be >= 0.0, got {}",
                temperature
            )));
        }
        if !(0.0..=1.0).contains(&top_p) {
            return Err(LludaError::Msg(format!(
                "top_p must be in [0.0, 1.0], got {}",
                top_p
            )));
        }
        if repetition_penalty < 1.0 {
            return Err(LludaError::Msg(format!(
                "repetition_penalty must be >= 1.0, got {}",
                repetition_penalty
            )));
        }

        Ok(Self {
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
        })
    }
}

/// Sample the next token using greedy decoding (argmax).
///
/// # Arguments
///
/// * `logits` - Logits tensor of shape [batch, 1, vocab_size] or [1, vocab_size]
///
/// # Returns
///
/// Token ID of the highest probability token.
///
/// # Example
///
/// ```rust
/// use lluda_inference::generate::sample_greedy;
/// use lluda_inference::tensor::Tensor;
///
/// // Logits for 5 tokens (token 2 has highest value)
/// let logits = Tensor::new(vec![1.0, 2.0, 5.0, 1.0, 0.5], vec![1, 5]).unwrap();
/// let token = sample_greedy(&logits).unwrap();
/// assert_eq!(token, 2);
/// ```
pub fn sample_greedy(logits: &Tensor) -> Result<u32> {
    // Logits shape: [B, 1, vocab_size] or [1, vocab_size]
    // We want to take argmax over last dimension

    if logits.ndim() < 2 {
        return Err(LludaError::Msg(format!(
            "Expected logits with at least 2 dimensions, got shape {:?}",
            logits.shape()
        )));
    }

    // For multi-dimensional logits, narrow to last token position
    // Shape: [B, L, vocab_size] -> take last L position -> [B, vocab_size]
    let logits_last = if logits.shape()[logits.ndim() - 2] > 1 {
        logits.narrow(logits.ndim() - 2, logits.shape()[logits.ndim() - 2] - 1, 1)?
    } else {
        logits.clone()
    };

    // Argmax over last dimension (vocab_size)
    logits_last.argmax_last_dim()
}

/// Sample the next token with temperature scaling.
///
/// # Arguments
///
/// * `logits` - Logits tensor of shape [batch, 1, vocab_size]
/// * `temperature` - Temperature parameter (> 0)
///
/// # Returns
///
/// Sampled token ID.
///
/// # Note
///
/// Phase 0 implements deterministic sampling (argmax after temperature scaling).
/// True probabilistic sampling will be added in later phases.
pub fn sample_temperature(logits: &Tensor, temperature: f32) -> Result<u32> {
    if temperature <= 0.0 {
        return Err(LludaError::Msg(format!(
            "temperature must be > 0, got {}",
            temperature
        )));
    }

    // For temperature = 1.0, no scaling needed
    if (temperature - 1.0).abs() < 1e-6 {
        return sample_greedy(logits);
    }

    // Scale logits by temperature
    let scaled_logits = logits.mul_scalar(1.0 / temperature)?;

    // For Phase 0: deterministic sampling (argmax)
    // TODO: Phase 1 will add true probabilistic sampling via softmax + random sampling
    sample_greedy(&scaled_logits)
}

/// Sample using top-k filtering.
///
/// Only considers the top k highest probability tokens.
///
/// # Arguments
///
/// * `logits` - Logits tensor of shape [batch, 1, vocab_size]
/// * `k` - Number of top tokens to consider
/// * `temperature` - Temperature for sampling
///
/// # Returns
///
/// Sampled token ID.
///
/// # Note
///
/// Phase 0 implementation: filters to top-k then applies greedy sampling.
/// True probabilistic sampling will be added in later phases.
pub fn sample_top_k(logits: &Tensor, k: usize, temperature: f32) -> Result<u32> {
    if k == 0 {
        return sample_temperature(logits, temperature);
    }

    // Get logits data
    let logits_data = logits.to_vec_f32();
    let vocab_size = logits.shape()[logits.ndim() - 1];

    // Get last token's logits
    let start = logits_data.len() - vocab_size;
    let token_logits = &logits_data[start..];

    // Create (index, value) pairs and sort by value descending
    let mut indexed_logits: Vec<(usize, f32)> = token_logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Take top k
    let _k_actual = k.min(vocab_size);
    let top_k_idx = indexed_logits[0].0;

    // For Phase 0: return the highest logit index (greedy within top-k)
    // TODO: Phase 1 will add true probabilistic sampling from top-k distribution
    Ok(top_k_idx as u32)
}

/// Sample using top-p (nucleus) filtering.
///
/// Only considers the smallest set of tokens whose cumulative probability >= p.
///
/// # Arguments
///
/// * `logits` - Logits tensor of shape [batch, 1, vocab_size]
/// * `p` - Cumulative probability threshold (0.0 to 1.0)
/// * `temperature` - Temperature for sampling
///
/// # Returns
///
/// Sampled token ID.
///
/// # Note
///
/// Phase 0 implementation: applies nucleus filtering then greedy sampling.
/// True probabilistic sampling will be added in later phases.
pub fn sample_top_p(logits: &Tensor, p: f32, temperature: f32) -> Result<u32> {
    if (p - 1.0).abs() < 1e-6 {
        return sample_temperature(logits, temperature);
    }

    // Get logits data
    let logits_data = logits.to_vec_f32();
    let vocab_size = logits.shape()[logits.ndim() - 1];

    // Get last token's logits
    let start = logits_data.len() - vocab_size;
    let token_logits = &logits_data[start..];

    // Apply temperature scaling
    let scaled_logits: Vec<f32> = if (temperature - 1.0).abs() > 1e-6 {
        token_logits.iter().map(|&x| x / temperature).collect()
    } else {
        token_logits.to_vec()
    };

    // Compute softmax to get probabilities
    let max_logit = scaled_logits
        .iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0);

    let exp_sum: f32 = scaled_logits.iter().map(|&x| (x - max_logit).exp()).sum();

    let probs: Vec<f32> = scaled_logits
        .iter()
        .map(|&x| (x - max_logit).exp() / exp_sum)
        .collect();

    // Create (index, prob) pairs and sort by prob descending
    let mut indexed_probs: Vec<(usize, f32)> =
        probs.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Find smallest set with cumulative prob >= p
    let mut cumulative = 0.0;
    for (_, prob) in indexed_probs.iter() {
        cumulative += prob;
        if cumulative >= p {
            break;
        }
    }

    // For Phase 0: return the highest probability token in nucleus (greedy)
    // TODO: Phase 1 will add true probabilistic sampling from nucleus distribution
    Ok(indexed_probs[0].0 as u32)
}

/// Apply repetition penalty to logits.
///
/// Divides positive logits and multiplies negative logits by the penalty factor.
/// This prevents the bug where dividing negative logits makes them less negative
/// (increasing their probability).
///
/// # Arguments
///
/// * `logits` - Mutable logits tensor to modify
/// * `generated_tokens` - List of previously generated token IDs
/// * `penalty` - Penalty factor (> 1.0 to discourage repetition)
///
/// # Note
///
/// Modifies logits in-place by converting to F32, applying penalty, and reconstructing.
///
/// # Algorithm
///
/// For each generated token's logit:
/// - If logit > 0: divide by penalty (reduce positive score)
/// - If logit < 0: multiply by penalty (make more negative)
/// - If logit = 0: unchanged
pub fn apply_repetition_penalty(
    logits: &Tensor,
    generated_tokens: &[u32],
    penalty: f32,
) -> Result<Tensor> {
    if (penalty - 1.0).abs() < 1e-6 || generated_tokens.is_empty() {
        return Ok(logits.clone());
    }

    let mut logits_data = logits.to_vec_f32();
    let vocab_size = logits.shape()[logits.ndim() - 1];
    let start = logits_data.len() - vocab_size;

    // Apply penalty to tokens that have been generated
    for &token_id in generated_tokens {
        let idx = start + token_id as usize;
        if idx < logits_data.len() {
            let logit = logits_data[idx];
            // Correct handling for negative logits
            if logit > 0.0 {
                logits_data[idx] = logit / penalty;
            } else if logit < 0.0 {
                logits_data[idx] = logit * penalty;
            }
            // If logit == 0.0, leave unchanged
        }
    }

    Tensor::new(logits_data, logits.shape().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_new_tokens, 100);
        assert_eq!(config.temperature, 0.0);
        assert_eq!(config.top_k, 0);
        assert_eq!(config.top_p, 1.0);
        assert_eq!(config.repetition_penalty, 1.0);
    }

    #[test]
    fn test_generation_config_validation() {
        // Valid config
        let config = GenerationConfig::new(50, 0.8, 10, 0.9, 1.2).unwrap();
        assert_eq!(config.max_new_tokens, 50);
        assert_eq!(config.temperature, 0.8);

        // Invalid temperature
        let result = GenerationConfig::new(50, -0.1, 10, 0.9, 1.2);
        assert!(result.is_err());

        // Invalid top_p
        let result = GenerationConfig::new(50, 0.8, 10, 1.5, 1.2);
        assert!(result.is_err());

        // Invalid repetition_penalty
        let result = GenerationConfig::new(50, 0.8, 10, 0.9, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_sample_greedy() {
        // Simple 2D case: [1, 5]
        let logits = Tensor::new(vec![1.0, 2.0, 5.0, 1.0, 0.5], vec![1, 5]).unwrap();
        let token = sample_greedy(&logits).unwrap();
        assert_eq!(token, 2, "Should select token with highest logit");
    }

    #[test]
    fn test_sample_greedy_3d() {
        // 3D case: [1, 2, 5] - should take last position
        let logits = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 1.0, 0.5, // First position
                2.0, 1.0, 0.5, 4.0, 1.0, // Second position (should use this)
            ],
            vec![1, 2, 5],
        )
        .unwrap();
        let token = sample_greedy(&logits).unwrap();
        assert_eq!(
            token, 3,
            "Should select token 3 from last position (highest logit 4.0)"
        );
    }

    #[test]
    fn test_sample_temperature_zero() {
        // Temperature 0 should be same as greedy
        let logits = Tensor::new(vec![1.0, 3.0, 2.0, 0.5], vec![1, 4]).unwrap();

        // Very small temperature (effectively greedy)
        let token = sample_temperature(&logits, 0.01).unwrap();
        assert_eq!(token, 1);
    }

    #[test]
    fn test_sample_temperature_one() {
        // Temperature 1.0 should not change distribution (but still greedy in Phase 0)
        let logits = Tensor::new(vec![1.0, 3.0, 2.0, 0.5], vec![1, 4]).unwrap();
        let token = sample_temperature(&logits, 1.0).unwrap();
        assert_eq!(token, 1, "Temperature 1.0 should still be greedy in Phase 0");
    }

    #[test]
    fn test_sample_top_k() {
        let logits = Tensor::new(vec![1.0, 5.0, 2.0, 4.0, 3.0], vec![1, 5]).unwrap();

        // Top-k=3 should consider only top 3 tokens: indices 1 (5.0), 3 (4.0), 4 (3.0)
        // Should return index 1 (highest)
        let token = sample_top_k(&logits, 3, 1.0).unwrap();
        assert_eq!(token, 1, "Should select highest token in top-k");
    }

    #[test]
    fn test_sample_top_k_zero() {
        // top_k=0 should be same as greedy
        let logits = Tensor::new(vec![1.0, 5.0, 2.0, 4.0, 3.0], vec![1, 5]).unwrap();
        let token = sample_top_k(&logits, 0, 1.0).unwrap();
        assert_eq!(token, sample_greedy(&logits).unwrap());
    }

    #[test]
    fn test_sample_top_p() {
        let logits = Tensor::new(vec![1.0, 5.0, 2.0, 4.0, 3.0], vec![1, 5]).unwrap();

        // Top-p should filter tokens and return highest
        let token = sample_top_p(&logits, 0.9, 1.0).unwrap();
        assert_eq!(
            token, 1,
            "Should select highest probability token in nucleus"
        );
    }

    #[test]
    fn test_sample_top_p_one() {
        // top_p=1.0 should be same as greedy
        let logits = Tensor::new(vec![1.0, 5.0, 2.0, 4.0, 3.0], vec![1, 5]).unwrap();
        let token = sample_top_p(&logits, 1.0, 1.0).unwrap();
        assert_eq!(token, sample_greedy(&logits).unwrap());
    }

    #[test]
    fn test_apply_repetition_penalty() {
        let logits = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]).unwrap();

        // Penalize tokens 1 and 3
        let generated = vec![1, 3];
        let penalized = apply_repetition_penalty(&logits, &generated, 2.0).unwrap();

        let original = logits.to_vec_f32();
        let modified = penalized.to_vec_f32();

        // Tokens 1 and 3 should be divided by 2.0
        assert_eq!(modified[0], original[0]); // Token 0: unchanged
        assert!((modified[1] - original[1] / 2.0).abs() < 1e-5); // Token 1: penalized
        assert_eq!(modified[2], original[2]); // Token 2: unchanged
        assert!((modified[3] - original[3] / 2.0).abs() < 1e-5); // Token 3: penalized
        assert_eq!(modified[4], original[4]); // Token 4: unchanged
    }

    #[test]
    fn test_apply_repetition_penalty_no_penalty() {
        let logits = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]).unwrap();

        // Penalty = 1.0 should not change logits
        let penalized = apply_repetition_penalty(&logits, &[1, 3], 1.0).unwrap();

        let original = logits.to_vec_f32();
        let modified = penalized.to_vec_f32();

        for (i, (&o, &m)) in original.iter().zip(modified.iter()).enumerate() {
            assert!(
                (o - m).abs() < 1e-5,
                "Token {} should be unchanged with penalty=1.0",
                i
            );
        }
    }

    #[test]
    fn test_apply_repetition_penalty_empty_tokens() {
        let logits = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();

        // Empty generated tokens should not change logits
        let penalized = apply_repetition_penalty(&logits, &[], 2.0).unwrap();

        let original = logits.to_vec_f32();
        let modified = penalized.to_vec_f32();

        for (i, (&o, &m)) in original.iter().zip(modified.iter()).enumerate() {
            assert!(
                (o - m).abs() < 1e-5,
                "Token {} should be unchanged with empty generated list",
                i
            );
        }
    }
}

/// Generate text autoregressively.
///
/// Implements the core generation loop:
/// 1. Encode prompt with tokenizer
/// 2. Run prefill forward pass (all prompt tokens at once)
/// 3. Sample next token from logits
/// 4. Iteratively generate: forward(new_token) -> sample -> append
/// 5. Stop on EOS token or max_new_tokens reached
/// 6. Decode final token sequence to text
///
/// # Arguments
///
/// * `model` - Mutable reference to Qwen3ForCausalLM model
/// * `tokenizer` - Tokenizer for encoding/decoding
/// * `prompt` - Input text prompt
/// * `config` - Generation configuration (temperature, top-k, etc.)
///
/// # Returns
///
/// Generated text string (including the original prompt).
///
/// # Errors
///
/// Returns error if:
/// - Tokenization fails
/// - Model forward pass fails
/// - Sampling fails
/// - Decoding fails
///
/// # Example
///
/// ```no_run
/// use lluda_inference::config::Qwen3Config;
/// use lluda_inference::loader::ModelWeights;
/// use lluda_inference::model::Qwen3ForCausalLM;
/// use lluda_inference::tokenizer::Tokenizer;
/// use lluda_inference::generate::{GenerationConfig, generate};
///
/// let config = Qwen3Config::from_file("models/Qwen3-0.6B/config.json")?;
/// let weights = ModelWeights::from_safetensors("models/Qwen3-0.6B/model.safetensors")?;
/// let mut model = Qwen3ForCausalLM::load(&config, &weights)?;
///
/// let tokenizer = Tokenizer::from_file(
///     "models/Qwen3-0.6B/tokenizer.json",
///     151643,
///     vec![151645, 151643],
/// )?;
///
/// let gen_config = GenerationConfig::default();
/// let output = generate(&mut model, &tokenizer, "Hello, world!", &gen_config)?;
/// println!("Generated: {}", output);
/// # Ok::<(), lluda_inference::error::LludaError>(())
/// ```
pub fn generate(
    model: &mut crate::model::Qwen3ForCausalLM,
    tokenizer: &crate::tokenizer::Tokenizer,
    prompt: &str,
    config: &GenerationConfig,
) -> Result<String> {
    // Clear KV cache to start fresh generation
    model.clear_kv_cache();

    // 1. Encode prompt to token IDs (with BOS token)
    let mut input_ids = tokenizer.encode(prompt, true)?;

    if input_ids.is_empty() {
        return Err(LludaError::Msg("Empty input after tokenization".into()));
    }

    // 2. Prefill: forward pass with all prompt tokens
    let prefill_len = input_ids.len();
    let logits = model.forward(&input_ids, 0)?;

    // 3. Sample first new token
    let mut generated_tokens = Vec::new();
    let next_token = sample_with_config(&logits, &generated_tokens, config)?;
    generated_tokens.push(next_token);
    input_ids.push(next_token);

    // Check if we hit EOS immediately
    if is_eos_token(next_token, tokenizer) {
        return tokenizer.decode(&input_ids, true);
    }

    // 4. Generation loop: iteratively generate tokens
    for _ in 1..config.max_new_tokens {
        // Forward pass with just the new token (KV cache handles context)
        let offset = prefill_len + generated_tokens.len() - 1;
        let logits = model.forward(&[next_token], offset)?;

        // Sample next token
        let next_token = sample_with_config(&logits, &generated_tokens, config)?;
        generated_tokens.push(next_token);
        input_ids.push(next_token);

        // Check for EOS token
        if is_eos_token(next_token, tokenizer) {
            break;
        }
    }

    // 5. Decode final sequence to text
    tokenizer.decode(&input_ids, true)
}

/// Sample next token using generation config.
///
/// Applies repetition penalty and dispatches to appropriate sampling strategy.
fn sample_with_config(
    logits: &Tensor,
    generated_tokens: &[u32],
    config: &GenerationConfig,
) -> Result<u32> {
    // Apply repetition penalty if enabled
    let logits = apply_repetition_penalty(logits, generated_tokens, config.repetition_penalty)?;

    // Dispatch to sampling strategy
    if config.temperature == 0.0 || (config.temperature - 1.0).abs() < 1e-6 && config.top_k == 0 && (config.top_p - 1.0).abs() < 1e-6 {
        // Greedy decoding
        sample_greedy(&logits)
    } else if config.top_k > 0 {
        // Top-k sampling
        sample_top_k(&logits, config.top_k, config.temperature)
    } else if (config.top_p - 1.0).abs() >= 1e-6 {
        // Top-p (nucleus) sampling
        sample_top_p(&logits, config.top_p, config.temperature)
    } else {
        // Temperature sampling
        sample_temperature(&logits, config.temperature)
    }
}

/// Check if token is an EOS token.
fn is_eos_token(token: u32, tokenizer: &crate::tokenizer::Tokenizer) -> bool {
    tokenizer.eos_token_ids().contains(&token)
}

#[cfg(test)]
mod generate_tests {
    use super::*;
    use crate::config::Qwen3Config;
    use crate::loader::ModelWeights;
    use crate::model::Qwen3ForCausalLM;
    use crate::tokenizer::Tokenizer;
    use std::path::PathBuf;

    /// Helper to check if model files exist
    fn get_model_path() -> Option<PathBuf> {
        let path = PathBuf::from("models/Qwen3-0.6B");
        if path.exists() {
            Some(path)
        } else {
            None
        }
    }

    #[test]
    fn test_generate_basic() {
        let Some(model_dir) = get_model_path() else {
            eprintln!("Skipping: model files not found");
            return;
        };

        let config = Qwen3Config::from_file(model_dir.join("config.json")).unwrap();
        let weights = ModelWeights::from_safetensors(model_dir.join("model.safetensors")).unwrap();
        let mut model = Qwen3ForCausalLM::load(&config, &weights).unwrap();

        let tokenizer = Tokenizer::from_file(
            model_dir.join("tokenizer.json"),
            151643,
            vec![151645, 151643],
        )
        .unwrap();

        let gen_config = GenerationConfig {
            max_new_tokens: 5,
            temperature: 0.0, // Greedy
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
        };

        let output = generate(&mut model, &tokenizer, "Hello", &gen_config).unwrap();

        // Output should contain input prompt
        assert!(!output.is_empty());
        eprintln!("Generated: {}", output);
    }

    #[test]
    fn test_generate_with_repetition_penalty() {
        let Some(model_dir) = get_model_path() else {
            eprintln!("Skipping: model files not found");
            return;
        };

        let config = Qwen3Config::from_file(model_dir.join("config.json")).unwrap();
        let weights = ModelWeights::from_safetensors(model_dir.join("model.safetensors")).unwrap();
        let mut model = Qwen3ForCausalLM::load(&config, &weights).unwrap();

        let tokenizer = Tokenizer::from_file(
            model_dir.join("tokenizer.json"),
            151643,
            vec![151645, 151643],
        )
        .unwrap();

        let gen_config = GenerationConfig {
            max_new_tokens: 10,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.5,
        };

        let output = generate(&mut model, &tokenizer, "The", &gen_config).unwrap();
        assert!(!output.is_empty());
        eprintln!("Generated with repetition penalty: {}", output);
    }

    #[test]
    fn test_generate_empty_prompt_fails() {
        let Some(model_dir) = get_model_path() else {
            eprintln!("Skipping: model files not found");
            return;
        };

        let config = Qwen3Config::from_file(model_dir.join("config.json")).unwrap();
        let weights = ModelWeights::from_safetensors(model_dir.join("model.safetensors")).unwrap();
        let mut model = Qwen3ForCausalLM::load(&config, &weights).unwrap();

        let tokenizer = Tokenizer::from_file(
            model_dir.join("tokenizer.json"),
            151643,
            vec![151645, 151643],
        )
        .unwrap();

        let gen_config = GenerationConfig::default();

        // Empty prompt may result in empty input_ids after tokenization
        // Depending on tokenizer behavior, this may or may not fail
        // But generate() should handle it gracefully
        let result = generate(&mut model, &tokenizer, "", &gen_config);

        // Either succeeds with minimal output or fails with clear error
        if let Err(e) = result {
            eprintln!("Empty prompt error (expected): {}", e);
        }
    }

    #[test]
    fn test_sample_with_config_greedy() {
        let logits = Tensor::new(vec![1.0, 5.0, 2.0, 4.0], vec![1, 4]).unwrap();
        let config = GenerationConfig {
            max_new_tokens: 10,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
        };

        let token = sample_with_config(&logits, &[], &config).unwrap();
        assert_eq!(token, 1, "Greedy should select highest logit");
    }

    #[test]
    fn test_is_eos_token() {
        let tokenizer_path = PathBuf::from("models/Qwen3-0.6B/tokenizer.json");
        if !tokenizer_path.exists() {
            eprintln!("Skipping: tokenizer not found");
            return;
        }

        let tokenizer = Tokenizer::from_file(
            tokenizer_path,
            151643,
            vec![151645, 151643],
        )
        .unwrap();

        // EOS tokens
        assert!(is_eos_token(151645, &tokenizer));
        assert!(is_eos_token(151643, &tokenizer));

        // Non-EOS token
        assert!(!is_eos_token(1234, &tokenizer));
    }

    #[test]
    fn test_apply_repetition_penalty_negative_logits() {
        // Test the fix for W3: negative logits should be multiplied, not divided
        let logits = Tensor::new(vec![-2.0, 3.0, -1.0, 4.0], vec![1, 4]).unwrap();
        let generated = vec![0, 2]; // Penalize tokens 0 and 2

        let penalized = apply_repetition_penalty(&logits, &generated, 2.0).unwrap();
        let result = penalized.to_vec_f32();

        // Token 0: -2.0 -> -2.0 * 2.0 = -4.0 (more negative)
        assert!((result[0] - (-4.0)).abs() < 1e-5, "Negative logit should be multiplied");

        // Token 1: 3.0 -> unchanged (not in generated)
        assert!((result[1] - 3.0).abs() < 1e-5, "Non-penalized token unchanged");

        // Token 2: -1.0 -> -1.0 * 2.0 = -2.0 (more negative)
        assert!((result[2] - (-2.0)).abs() < 1e-5, "Negative logit should be multiplied");

        // Token 3: 4.0 -> unchanged (not in generated)
        assert!((result[3] - 4.0).abs() < 1e-5, "Non-penalized token unchanged");
    }

    #[test]
    fn test_apply_repetition_penalty_positive_logits() {
        let logits = Tensor::new(vec![2.0, 3.0, 1.0, 4.0], vec![1, 4]).unwrap();
        let generated = vec![0, 2]; // Penalize tokens 0 and 2

        let penalized = apply_repetition_penalty(&logits, &generated, 2.0).unwrap();
        let result = penalized.to_vec_f32();

        // Token 0: 2.0 -> 2.0 / 2.0 = 1.0 (divided)
        assert!((result[0] - 1.0).abs() < 1e-5, "Positive logit should be divided");

        // Token 2: 1.0 -> 1.0 / 2.0 = 0.5 (divided)
        assert!((result[2] - 0.5).abs() < 1e-5, "Positive logit should be divided");
    }
}
