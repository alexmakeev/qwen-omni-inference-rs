//! Full Qwen3 model assembly for text generation.
//!
//! Combines all model components (embedding, transformer layers, normalization)
//! into complete Qwen3Model and Qwen3ForCausalLM structures.
//!
//! # Architecture
//!
//! ```text
//! input_ids [B, L]
//!   ↓
//! embed_tokens: [vocab_size, hidden_size]
//!   ↓
//! embeddings [B, L, hidden_size]
//!   ↓
//! layers[0..27]: DecoderLayer
//!   ↓
//! final norm: RmsNorm
//!   ↓
//! hidden_states [B, L, hidden_size]
//!   ↓
//! lm_head (tied to embed_tokens.weight): [hidden_size, vocab_size]
//!   ↓
//! logits [B, L, vocab_size]
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use lluda_inference::config::Qwen3Config;
//! use lluda_inference::loader::ModelWeights;
//! use lluda_inference::model::Qwen3ForCausalLM;
//!
//! // Load config and weights
//! let config = Qwen3Config::from_file("models/Qwen3-0.6B/config.json")?;
//! let weights = ModelWeights::from_safetensors("models/Qwen3-0.6B/model.safetensors")?;
//!
//! // Create model
//! let mut model = Qwen3ForCausalLM::load(&config, &weights)?;
//!
//! // Generate logits for input tokens
//! let input_ids = vec![151643, 1234, 5678]; // BOS + tokens
//! let logits = model.forward(&input_ids, 0)?;
//! assert_eq!(logits.shape(), &[1, 1, 151936]); // [B=1, L=1 (last token), vocab_size]
//!
//! # Ok::<(), lluda_inference::error::LludaError>(())
//! ```

use std::sync::Arc;

use crate::attention::{Attention, KvCache, Linear};
use crate::causal_mask::causal_mask;
use crate::config::Qwen3Config;
use crate::embedding::Embedding;
use crate::error::{LludaError, Result};
use crate::loader::ModelWeights;
use crate::mlp::MLP;
use crate::rms_norm::RmsNorm;
use crate::rope::RotaryEmbedding;
use crate::tensor::{DType, Tensor};
use crate::transformer::DecoderLayer;

/// Core Qwen3 model without the language modeling head.
///
/// Contains token embeddings, transformer layers, and final normalization.
#[derive(Debug, Clone)]
pub struct Qwen3Model {
    /// Token embedding layer
    embed_tokens: Embedding,
    /// Transformer decoder layers (28 for Qwen3-0.6B)
    layers: Vec<DecoderLayer>,
    /// Final RMSNorm before LM head
    norm: RmsNorm,
    /// KV caches (one per layer)
    kv_caches: Vec<KvCache>,
    /// Model configuration (stored for potential future use)
    #[allow(dead_code)]
    config: Qwen3Config,
}

/// Qwen3 model with causal language modeling head.
///
/// For Qwen3, the LM head weight is tied to the embedding weight
/// (tie_word_embeddings=true in config).
#[derive(Debug, Clone)]
pub struct Qwen3ForCausalLM {
    /// Base Qwen3 model
    model: Qwen3Model,
    /// LM head weight (tied to embed_tokens.weight)
    #[allow(dead_code)]
    lm_head_weight: Tensor,
    /// Pre-transposed LM head weight for efficient matmul: [hidden_size, vocab_size]
    lm_head_weight_transposed: Tensor,
}

impl Qwen3Model {
    /// Load Qwen3Model from config and weights.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration
    /// * `weights` - Loaded model weights from SafeTensors
    ///
    /// # Returns
    ///
    /// Initialized Qwen3Model with all layers loaded.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Required weights are missing
    /// - Weight shapes don't match config
    /// - Layer construction fails
    fn load(config: &Qwen3Config, weights: &ModelWeights) -> Result<Self> {
        // Load embedding
        let embed_weight = weights
            .get("model.embed_tokens.weight")
            .ok_or_else(|| LludaError::Msg("Missing embed_tokens.weight".into()))?
            .clone();
        let embed_tokens = Embedding::new(embed_weight)?;

        // Create shared RoPE embeddings
        let rotary = Arc::new(RotaryEmbedding::new(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )?);

        // Load all decoder layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            let layer = load_decoder_layer(config, weights, layer_idx, Arc::clone(&rotary))?;
            layers.push(layer);
        }

        // Load final norm
        let norm_weight = weights
            .get("model.norm.weight")
            .ok_or_else(|| LludaError::Msg("Missing model.norm.weight".into()))?
            .clone();
        let norm = RmsNorm::new(norm_weight, config.rms_norm_eps)?;

        // Initialize KV caches (one per layer)
        let kv_caches = vec![KvCache::new(); config.num_hidden_layers];

        Ok(Qwen3Model {
            embed_tokens,
            layers,
            norm,
            kv_caches,
            config: config.clone(),
        })
    }

    /// Forward pass through the model.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Token IDs to process, shape [B, L] (flattened)
    /// * `batch_size` - Batch size
    /// * `seq_len` - Sequence length
    /// * `offset` - Position offset for KV cache (length of cached sequence)
    ///
    /// # Returns
    ///
    /// Hidden states after all layers and final norm, shape [B, L, hidden_size].
    ///
    /// # Errors
    ///
    /// Returns error if shapes are incompatible or operations fail.
    fn forward(
        &mut self,
        input_ids: &[u32],
        batch_size: usize,
        seq_len: usize,
        offset: usize,
    ) -> Result<Tensor> {
        // 1. Embed tokens: [B, L] -> [B, L, hidden_size]
        let mut hidden_states = self.embed_tokens.forward(input_ids, &[batch_size, seq_len])?;

        // 2. Generate causal mask (if seq_len > 1)
        let mask = causal_mask(batch_size, seq_len, offset, DType::F32)?;

        // 3. Pass through all transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(
                &hidden_states,
                mask.as_ref(),
                &mut self.kv_caches[layer_idx],
                offset,
            )?;
        }

        // 4. Final normalization
        self.norm.forward(&hidden_states)
    }

    /// Forward pass with intermediate layer outputs exposed (for validation).
    ///
    /// Returns (embedding_output, layer_outputs, final_norm_output).
    #[allow(dead_code)]
    pub(crate) fn forward_with_intermediates(
        &mut self,
        input_ids: &[u32],
        batch_size: usize,
        seq_len: usize,
        offset: usize,
    ) -> Result<(Tensor, Vec<Tensor>, Tensor)> {
        // 1. Embed tokens: [B, L] -> [B, L, hidden_size]
        let mut hidden_states = self.embed_tokens.forward(input_ids, &[batch_size, seq_len])?;
        let embedding_output = hidden_states.clone();

        // 2. Generate causal mask (if seq_len > 1)
        let mask = causal_mask(batch_size, seq_len, offset, DType::F32)?;

        // 3. Pass through all transformer layers, saving each output
        let mut layer_outputs = Vec::with_capacity(self.layers.len());
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(
                &hidden_states,
                mask.as_ref(),
                &mut self.kv_caches[layer_idx],
                offset,
            )?;
            layer_outputs.push(hidden_states.clone());
        }

        // 4. Final normalization
        let final_norm_output = self.norm.forward(&hidden_states)?;

        Ok((embedding_output, layer_outputs, final_norm_output))
    }

    /// Clear all KV caches.
    ///
    /// Call this at the start of a new conversation or generation.
    fn clear_kv_cache(&mut self) {
        for cache in &mut self.kv_caches {
            cache.reset();
        }
    }
}

impl Qwen3ForCausalLM {
    /// Load Qwen3ForCausalLM from config and weights.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration
    /// * `weights` - Loaded model weights from SafeTensors
    ///
    /// # Returns
    ///
    /// Initialized model ready for text generation.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Model loading fails
    /// - tie_word_embeddings is false (not supported yet)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use lluda_inference::config::Qwen3Config;
    /// use lluda_inference::loader::ModelWeights;
    /// use lluda_inference::model::Qwen3ForCausalLM;
    ///
    /// let config = Qwen3Config::from_file("models/Qwen3-0.6B/config.json")?;
    /// let weights = ModelWeights::from_safetensors("models/Qwen3-0.6B/model.safetensors")?;
    /// let model = Qwen3ForCausalLM::load(&config, &weights)?;
    /// # Ok::<(), lluda_inference::error::LludaError>(())
    /// ```
    pub fn load(config: &Qwen3Config, weights: &ModelWeights) -> Result<Self> {
        // Load base model
        let model = Qwen3Model::load(config, weights)?;

        // Get LM head weight
        // For Qwen3, tie_word_embeddings=true means we reuse embed_tokens.weight
        let lm_head_weight = if config.tie_word_embeddings {
            model.embed_tokens.weight().clone()
        } else {
            // If not tied, load separate lm_head.weight
            weights
                .get("lm_head.weight")
                .ok_or_else(|| {
                    LludaError::Msg(
                        "tie_word_embeddings=false requires lm_head.weight (not found)".into(),
                    )
                })?
                .clone()
        };

        // Pre-transpose LM head weight: [vocab_size, hidden_size] -> [hidden_size, vocab_size]
        // This is expensive but we only do it once during model loading
        let lm_head_weight_transposed = lm_head_weight.transpose_dims(0, 1)?;

        Ok(Qwen3ForCausalLM {
            model,
            lm_head_weight,
            lm_head_weight_transposed,
        })
    }

    /// Forward pass: token IDs -> logits for next token.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Token IDs to process [B*L elements]
    /// * `offset` - Current position in sequence (for KV cache)
    ///
    /// # Returns
    ///
    /// Logits for last token only, shape [B, 1, vocab_size].
    /// During generation, we only need logits for the last position to predict next token.
    ///
    /// # Errors
    ///
    /// Returns error if operations fail or shapes are incompatible.
    ///
    /// # Algorithm
    ///
    /// 1. Embed tokens
    /// 2. Pass through all transformer layers with KV cache
    /// 3. Apply final normalization
    /// 4. Select last token: hidden_states[:, -1:, :]
    /// 5. Project to vocabulary: logits = hidden @ lm_head_weight.T
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use lluda_inference::config::Qwen3Config;
    /// use lluda_inference::loader::ModelWeights;
    /// use lluda_inference::model::Qwen3ForCausalLM;
    ///
    /// let config = Qwen3Config::from_file("models/Qwen3-0.6B/config.json")?;
    /// let weights = ModelWeights::from_safetensors("models/Qwen3-0.6B/model.safetensors")?;
    /// let mut model = Qwen3ForCausalLM::load(&config, &weights)?;
    ///
    /// // Forward pass with 4 tokens
    /// let input_ids = vec![151643, 1234, 5678, 9012];
    /// let logits = model.forward(&input_ids, 0)?;
    /// assert_eq!(logits.shape(), &[1, 1, 151936]);
    /// # Ok::<(), lluda_inference::error::LludaError>(())
    /// ```
    pub fn forward(&mut self, input_ids: &[u32], offset: usize) -> Result<Tensor> {
        // Validate input
        if input_ids.is_empty() {
            return Err(LludaError::Msg("input_ids cannot be empty".into()));
        }

        let batch_size = 1; // Currently only support batch_size=1
        let seq_len = input_ids.len();

        // Forward through base model
        let hidden_states = self.model.forward(input_ids, batch_size, seq_len, offset)?;

        // Select last token: [B, L, hidden_size] -> [B, 1, hidden_size]
        let last_hidden = hidden_states.narrow(1, seq_len - 1, 1)?;

        // Project to vocabulary: [B, 1, hidden_size] @ [hidden_size, vocab_size]
        // Reshape last_hidden to 2D: [B*1, hidden_size] = [1, hidden_size]
        let last_hidden_2d = last_hidden.reshape(&[batch_size, last_hidden.shape()[2]])?;

        // lm_head_weight_transposed is [hidden_size, vocab_size]
        // Compute: [1, hidden_size] @ [hidden_size, vocab_size] -> [1, vocab_size]
        let logits_2d = last_hidden_2d.matmul(&self.lm_head_weight_transposed)?;

        // Reshape back to [B, 1, vocab_size]
        let logits = logits_2d.reshape(&[batch_size, 1, logits_2d.shape()[1]])?;

        Ok(logits)
    }

    /// Clear all KV caches.
    ///
    /// Call this at the start of a new conversation or generation.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use lluda_inference::config::Qwen3Config;
    /// use lluda_inference::loader::ModelWeights;
    /// use lluda_inference::model::Qwen3ForCausalLM;
    ///
    /// let config = Qwen3Config::from_file("models/Qwen3-0.6B/config.json")?;
    /// let weights = ModelWeights::from_safetensors("models/Qwen3-0.6B/model.safetensors")?;
    /// let mut model = Qwen3ForCausalLM::load(&config, &weights)?;
    ///
    /// // Generate some tokens...
    /// // let _ = model.forward(&[151643, 1234], 0)?;
    ///
    /// // Start new conversation
    /// model.clear_kv_cache();
    /// # Ok::<(), lluda_inference::error::LludaError>(())
    /// ```
    pub fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }

    /// Forward pass with intermediate layer outputs (for validation/debugging).
    ///
    /// Returns (embedding_output, layer_outputs, final_norm_output).
    ///
    /// # Note
    ///
    /// This method is primarily for validation and debugging. It clones intermediate
    /// tensors which adds memory overhead.
    #[doc(hidden)]
    pub fn forward_with_intermediates(
        &mut self,
        input_ids: &[u32],
    ) -> Result<(Tensor, Vec<Tensor>, Tensor)> {
        let batch_size = 1;
        let seq_len = input_ids.len();
        let offset = 0;

        self.model.forward_with_intermediates(input_ids, batch_size, seq_len, offset)
    }

    /// Get reference to LM head weight (for validation).
    #[doc(hidden)]
    pub fn lm_head_weight_transposed(&self) -> &Tensor {
        &self.lm_head_weight_transposed
    }
}

/// Load a single decoder layer from weights.
///
/// # Arguments
///
/// * `config` - Model configuration
/// * `weights` - All model weights
/// * `layer_idx` - Index of layer to load (0..27 for Qwen3-0.6B)
/// * `rotary` - Shared RoPE embeddings
///
/// # Returns
///
/// Initialized DecoderLayer with all weights loaded.
fn load_decoder_layer(
    config: &Qwen3Config,
    weights: &ModelWeights,
    layer_idx: usize,
    rotary: Arc<RotaryEmbedding>,
) -> Result<DecoderLayer> {
    let prefix = format!("model.layers.{}", layer_idx);

    // Load attention projections
    let q_proj_weight = weights
        .get(&format!("{}.self_attn.q_proj.weight", prefix))
        .ok_or_else(|| LludaError::Msg(format!("Missing {}.self_attn.q_proj.weight", prefix)))?
        .clone();
    let k_proj_weight = weights
        .get(&format!("{}.self_attn.k_proj.weight", prefix))
        .ok_or_else(|| LludaError::Msg(format!("Missing {}.self_attn.k_proj.weight", prefix)))?
        .clone();
    let v_proj_weight = weights
        .get(&format!("{}.self_attn.v_proj.weight", prefix))
        .ok_or_else(|| LludaError::Msg(format!("Missing {}.self_attn.v_proj.weight", prefix)))?
        .clone();
    let o_proj_weight = weights
        .get(&format!("{}.self_attn.o_proj.weight", prefix))
        .ok_or_else(|| LludaError::Msg(format!("Missing {}.self_attn.o_proj.weight", prefix)))?
        .clone();

    let q_proj = Linear::new(q_proj_weight)?;
    let k_proj = Linear::new(k_proj_weight)?;
    let v_proj = Linear::new(v_proj_weight)?;
    let o_proj = Linear::new(o_proj_weight)?;

    // Load Q/K norms
    let q_norm_weight = weights
        .get(&format!("{}.self_attn.q_norm.weight", prefix))
        .ok_or_else(|| LludaError::Msg(format!("Missing {}.self_attn.q_norm.weight", prefix)))?
        .clone();
    let k_norm_weight = weights
        .get(&format!("{}.self_attn.k_norm.weight", prefix))
        .ok_or_else(|| LludaError::Msg(format!("Missing {}.self_attn.k_norm.weight", prefix)))?
        .clone();

    let q_norm = RmsNorm::new(q_norm_weight, config.rms_norm_eps)?;
    let k_norm = RmsNorm::new(k_norm_weight, config.rms_norm_eps)?;

    // Create attention layer
    let self_attn = Attention::new(
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm,
        k_norm,
        rotary,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
    )?;

    // Load MLP projections
    let gate_proj_weight = weights
        .get(&format!("{}.mlp.gate_proj.weight", prefix))
        .ok_or_else(|| LludaError::Msg(format!("Missing {}.mlp.gate_proj.weight", prefix)))?
        .clone();
    let up_proj_weight = weights
        .get(&format!("{}.mlp.up_proj.weight", prefix))
        .ok_or_else(|| LludaError::Msg(format!("Missing {}.mlp.up_proj.weight", prefix)))?
        .clone();
    let down_proj_weight = weights
        .get(&format!("{}.mlp.down_proj.weight", prefix))
        .ok_or_else(|| LludaError::Msg(format!("Missing {}.mlp.down_proj.weight", prefix)))?
        .clone();

    let mlp = MLP::new(gate_proj_weight, up_proj_weight, down_proj_weight)?;

    // Load layer norms
    let input_layernorm_weight = weights
        .get(&format!("{}.input_layernorm.weight", prefix))
        .ok_or_else(|| LludaError::Msg(format!("Missing {}.input_layernorm.weight", prefix)))?
        .clone();
    let post_attention_layernorm_weight = weights
        .get(&format!("{}.post_attention_layernorm.weight", prefix))
        .ok_or_else(|| {
            LludaError::Msg(format!(
                "Missing {}.post_attention_layernorm.weight",
                prefix
            ))
        })?
        .clone();

    let input_layernorm = RmsNorm::new(input_layernorm_weight, config.rms_norm_eps)?;
    let post_attention_layernorm =
        RmsNorm::new(post_attention_layernorm_weight, config.rms_norm_eps)?;

    Ok(DecoderLayer::new(
        self_attn,
        mlp,
        input_layernorm,
        post_attention_layernorm,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Helper to get model path, skipping tests if not available
    fn get_model_path() -> Option<PathBuf> {
        let path = PathBuf::from("models/Qwen3-0.6B");
        if path.exists() {
            Some(path)
        } else {
            eprintln!("Skipping: model files not found at {}", path.display());
            None
        }
    }

    #[test]
    fn test_load_qwen3_model() {
        let Some(model_dir) = get_model_path() else {
            return;
        };

        let config_path = model_dir.join("config.json");
        let weights_path = model_dir.join("model.safetensors");

        let config = Qwen3Config::from_file(&config_path).unwrap();
        let weights = ModelWeights::from_safetensors(&weights_path).unwrap();

        let model = Qwen3ForCausalLM::load(&config, &weights).unwrap();

        // Verify model structure
        assert_eq!(model.model.layers.len(), 28);
        assert_eq!(model.model.kv_caches.len(), 28);
    }

    #[test]
    fn test_forward_single_token() {
        let Some(model_dir) = get_model_path() else {
            return;
        };

        let config_path = model_dir.join("config.json");
        let weights_path = model_dir.join("model.safetensors");

        let config = Qwen3Config::from_file(&config_path).unwrap();
        let weights = ModelWeights::from_safetensors(&weights_path).unwrap();

        let mut model = Qwen3ForCausalLM::load(&config, &weights).unwrap();

        // Forward pass with single token
        let input_ids = vec![151643]; // BOS token
        let logits = model.forward(&input_ids, 0).unwrap();

        // Verify output shape: [B=1, L=1, vocab_size=151936]
        assert_eq!(logits.shape(), &[1, 1, 151936]);

        // Verify logits are finite
        let logits_data = logits.to_vec_f32();
        assert!(
            logits_data.iter().all(|&x| x.is_finite()),
            "Logits contain non-finite values"
        );
    }

    #[test]
    fn test_forward_multiple_tokens() {
        let Some(model_dir) = get_model_path() else {
            return;
        };

        let config_path = model_dir.join("config.json");
        let weights_path = model_dir.join("model.safetensors");

        let config = Qwen3Config::from_file(&config_path).unwrap();
        let weights = ModelWeights::from_safetensors(&weights_path).unwrap();

        let mut model = Qwen3ForCausalLM::load(&config, &weights).unwrap();

        // Forward pass with multiple tokens
        let input_ids = vec![151643, 1234, 5678, 9012]; // BOS + 3 tokens
        let logits = model.forward(&input_ids, 0).unwrap();

        // Output is last token only
        assert_eq!(logits.shape(), &[1, 1, 151936]);

        let logits_data = logits.to_vec_f32();
        assert!(logits_data.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_sequential_generation_with_kv_cache() {
        let Some(model_dir) = get_model_path() else {
            return;
        };

        let config_path = model_dir.join("config.json");
        let weights_path = model_dir.join("model.safetensors");

        let config = Qwen3Config::from_file(&config_path).unwrap();
        let weights = ModelWeights::from_safetensors(&weights_path).unwrap();

        let mut model = Qwen3ForCausalLM::load(&config, &weights).unwrap();

        // Step 1: Process initial prompt (4 tokens)
        let prompt = vec![151643, 1234, 5678, 9012];
        let logits1 = model.forward(&prompt, 0).unwrap();
        assert_eq!(logits1.shape(), &[1, 1, 151936]);

        // Verify KV cache has 4 positions
        assert_eq!(model.model.kv_caches[0].seq_len(), 4);

        // Step 2: Generate next token (offset=4)
        let next_token = vec![1111];
        let logits2 = model.forward(&next_token, 4).unwrap();
        assert_eq!(logits2.shape(), &[1, 1, 151936]);

        // Verify KV cache has 5 positions now
        assert_eq!(model.model.kv_caches[0].seq_len(), 5);

        // Step 3: Generate another token (offset=5)
        let next_token2 = vec![2222];
        let logits3 = model.forward(&next_token2, 5).unwrap();
        assert_eq!(logits3.shape(), &[1, 1, 151936]);

        // Verify KV cache has 6 positions
        assert_eq!(model.model.kv_caches[0].seq_len(), 6);
    }

    #[test]
    fn test_clear_kv_cache() {
        let Some(model_dir) = get_model_path() else {
            return;
        };

        let config_path = model_dir.join("config.json");
        let weights_path = model_dir.join("model.safetensors");

        let config = Qwen3Config::from_file(&config_path).unwrap();
        let weights = ModelWeights::from_safetensors(&weights_path).unwrap();

        let mut model = Qwen3ForCausalLM::load(&config, &weights).unwrap();

        // Forward pass
        let input_ids = vec![151643, 1234];
        let _ = model.forward(&input_ids, 0).unwrap();
        assert_eq!(model.model.kv_caches[0].seq_len(), 2);

        // Clear cache
        model.clear_kv_cache();
        assert_eq!(model.model.kv_caches[0].seq_len(), 0);

        // New forward pass should work
        let input_ids2 = vec![151643, 5678];
        let logits = model.forward(&input_ids2, 0).unwrap();
        assert_eq!(logits.shape(), &[1, 1, 151936]);
        assert_eq!(model.model.kv_caches[0].seq_len(), 2);
    }

    #[test]
    fn test_tied_embeddings() {
        let Some(model_dir) = get_model_path() else {
            return;
        };

        let config_path = model_dir.join("config.json");
        let weights_path = model_dir.join("model.safetensors");

        let config = Qwen3Config::from_file(&config_path).unwrap();
        let weights = ModelWeights::from_safetensors(&weights_path).unwrap();

        // Qwen3-0.6B has tie_word_embeddings=true
        assert!(config.tie_word_embeddings);

        let model = Qwen3ForCausalLM::load(&config, &weights).unwrap();

        // Verify lm_head_weight IS the embedding weight
        let embed_weight = model.model.embed_tokens.weight();
        let lm_head_weight = &model.lm_head_weight;

        assert_eq!(embed_weight.shape(), lm_head_weight.shape());
        assert_eq!(embed_weight.dtype(), lm_head_weight.dtype());

        // Verify same underlying data (compare first few values)
        let embed_data = embed_weight.to_vec_f32();
        let lm_head_data = lm_head_weight.to_vec_f32();

        assert_eq!(embed_data.len(), lm_head_data.len());
        for i in 0..100.min(embed_data.len()) {
            assert!(
                (embed_data[i] - lm_head_data[i]).abs() < 1e-6,
                "Embedding and LM head weights should be identical (tied)"
            );
        }
    }

    #[test]
    fn test_all_layers_loaded() {
        let Some(model_dir) = get_model_path() else {
            return;
        };

        let config_path = model_dir.join("config.json");
        let weights_path = model_dir.join("model.safetensors");

        let config = Qwen3Config::from_file(&config_path).unwrap();
        let weights = ModelWeights::from_safetensors(&weights_path).unwrap();

        let model = Qwen3ForCausalLM::load(&config, &weights).unwrap();

        // Qwen3-0.6B has 28 layers
        assert_eq!(model.model.layers.len(), 28);

        // Each layer should have initialized KV cache
        for (i, cache) in model.model.kv_caches.iter().enumerate() {
            assert_eq!(cache.seq_len(), 0, "Layer {} cache should be empty initially", i);
        }
    }

    #[test]
    fn test_model_config_stored() {
        let Some(model_dir) = get_model_path() else {
            return;
        };

        let config_path = model_dir.join("config.json");
        let weights_path = model_dir.join("model.safetensors");

        let config = Qwen3Config::from_file(&config_path).unwrap();
        let weights = ModelWeights::from_safetensors(&weights_path).unwrap();

        let model = Qwen3ForCausalLM::load(&config, &weights).unwrap();

        // Verify config is stored correctly
        assert_eq!(model.model.config.hidden_size, 1024);
        assert_eq!(model.model.config.num_hidden_layers, 28);
        assert_eq!(model.model.config.vocab_size, 151936);
    }

    #[test]
    fn test_logits_shape_matches_vocab_size() {
        let Some(model_dir) = get_model_path() else {
            return;
        };

        let config_path = model_dir.join("config.json");
        let weights_path = model_dir.join("model.safetensors");

        let config = Qwen3Config::from_file(&config_path).unwrap();
        let weights = ModelWeights::from_safetensors(&weights_path).unwrap();

        let mut model = Qwen3ForCausalLM::load(&config, &weights).unwrap();

        let input_ids = vec![151643];
        let logits = model.forward(&input_ids, 0).unwrap();

        // Last dimension should match vocab_size
        assert_eq!(
            logits.shape()[2],
            config.vocab_size,
            "Logits vocab dimension should match config"
        );
    }

    #[test]
    fn test_forward_produces_different_logits_for_different_inputs() {
        let Some(model_dir) = get_model_path() else {
            return;
        };

        let config_path = model_dir.join("config.json");
        let weights_path = model_dir.join("model.safetensors");

        let config = Qwen3Config::from_file(&config_path).unwrap();
        let weights = ModelWeights::from_safetensors(&weights_path).unwrap();

        let mut model = Qwen3ForCausalLM::load(&config, &weights).unwrap();

        // Two different inputs should produce different logits
        let input1 = vec![151643, 1234];
        let logits1 = model.forward(&input1, 0).unwrap();
        model.clear_kv_cache();

        let input2 = vec![151643, 5678];
        let logits2 = model.forward(&input2, 0).unwrap();

        let data1 = logits1.to_vec_f32();
        let data2 = logits2.to_vec_f32();

        // Count how many logits are different
        let differences = data1
            .iter()
            .zip(data2.iter())
            .filter(|(&a, &b)| (a - b).abs() > 1e-4)
            .count();

        assert!(
            differences > 0,
            "Different inputs should produce different logits"
        );
    }

    #[test]
    fn test_forward_empty_input_ids_fails() {
        let Some(model_dir) = get_model_path() else {
            return;
        };

        let config_path = model_dir.join("config.json");
        let weights_path = model_dir.join("model.safetensors");

        let config = Qwen3Config::from_file(&config_path).unwrap();
        let weights = ModelWeights::from_safetensors(&weights_path).unwrap();

        let mut model = Qwen3ForCausalLM::load(&config, &weights).unwrap();

        // Empty input_ids should fail with clear error
        let result = model.forward(&[], 0);

        assert!(result.is_err(), "Empty input_ids should return error");

        match result.unwrap_err() {
            LludaError::Msg(msg) => {
                assert!(
                    msg.contains("empty"),
                    "Error message should mention 'empty', got: {}",
                    msg
                );
            }
            other => panic!("Expected Msg error, got: {:?}", other),
        }
    }
}
