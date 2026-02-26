//! Talker (TTS decoder) for Qwen2.5-Omni.
//!
//! The Talker is a transformer decoder that converts thinker hidden states
//! and codec token embeddings into codec token logits for speech synthesis.
//!
//! # Architecture
//!
//! ```text
//! input_ids (codec tokens)   thinker_hidden
//!       ↓                          ↓
//! embed_tokens [8448, 2048]         |
//!       ↓ (token embeds)            |
//!       +──────────────────────────┘  (element-wise add)
//!       ↓ [seq, 2048]
//! thinker_to_talker_proj [896, 2048] + bias
//!       ↓ [seq, 896]
//! OmniDecoderLayer × 24
//!       ↓ [seq, 896]
//! final RmsNorm
//!       ↓ [seq, 896]
//! codec_head.T  [896, 8448]
//!       ↓ [seq, 8448]
//! logits
//! ```
//!
//! The talker operates autoregressively: at each step it receives
//! the last generated codec token as `input_ids` and the corresponding
//! thinker hidden state slice as `thinker_hidden`.
//!
//! # Dimensions (Qwen2.5-Omni-3B)
//!
//! | Component            | Shape           |
//! |----------------------|-----------------|
//! | embed_tokens         | [8448, 2048]    |
//! | thinker_to_talker    | [896, 2048] + b |
//! | num_layers           | 24              |
//! | hidden_size (talker) | 896             |
//! | vocab_size (codec)   | 8448            |
//! | head_dim             | 64              |
//! | num_heads            | 14              |
//! | num_kv_heads         | 2               |

use std::sync::Arc;

use crate::attention::KvCache;
use crate::causal_mask::causal_mask;
use crate::config::OmniTalkerConfig;
use crate::embedding::Embedding;
use crate::error::{LludaError, Result};
use crate::mlp::MLP;
use crate::omni_attention::{LinearNoBias, LinearWithBias, OmniAttention, OmniDecoderLayer};
use crate::rms_norm::RmsNorm;
use crate::rope::RotaryEmbedding;
use crate::tensor::{DType, Tensor};

/// Talker decoder for Qwen2.5-Omni TTS.
///
/// Converts thinker hidden states + codec token embeddings into
/// codec vocabulary logits for speech token generation.
#[derive(Debug, Clone)]
pub struct Talker {
    /// Codec token embedding in thinker's dimension space [vocab_size, embedding_size]
    embed_tokens: Embedding,
    /// Projects thinker dim to talker dim: [hidden_size, embedding_size] + bias
    thinker_to_talker_proj: LinearWithBias,
    /// Transformer decoder layers (24 for Qwen2.5-Omni-3B)
    layers: Vec<OmniDecoderLayer>,
    /// Final RMSNorm
    norm: RmsNorm,
    /// LM head weight for codec tokens [vocab_size, hidden_size]
    /// Stored pre-transposed as [hidden_size, vocab_size] for efficient matmul.
    codec_head_transposed: Tensor,
    /// KV caches (one per layer)
    kv_caches: Vec<KvCache>,
    /// Model configuration
    #[allow(dead_code)]
    config: OmniTalkerConfig,
}

impl Talker {
    /// Load Talker from config and a weight lookup function.
    ///
    /// # Arguments
    ///
    /// * `config` - Talker configuration
    /// * `get_tensor` - Function that looks up a tensor by name
    ///
    /// # Returns
    ///
    /// Initialized Talker ready for codec token generation.
    ///
    /// # Errors
    ///
    /// Returns error if any required weight is missing or shapes are invalid.
    ///
    /// # Weight names
    ///
    /// ```text
    /// talker.model.embed_tokens.weight
    /// talker.thinker_to_talker_proj.weight
    /// talker.thinker_to_talker_proj.bias
    /// talker.model.layers.{i}.self_attn.q_proj.weight
    /// talker.model.layers.{i}.self_attn.q_proj.bias
    /// talker.model.layers.{i}.self_attn.k_proj.weight
    /// talker.model.layers.{i}.self_attn.k_proj.bias
    /// talker.model.layers.{i}.self_attn.v_proj.weight
    /// talker.model.layers.{i}.self_attn.v_proj.bias
    /// talker.model.layers.{i}.self_attn.o_proj.weight
    /// talker.model.layers.{i}.mlp.gate_proj.weight
    /// talker.model.layers.{i}.mlp.up_proj.weight
    /// talker.model.layers.{i}.mlp.down_proj.weight
    /// talker.model.layers.{i}.input_layernorm.weight
    /// talker.model.layers.{i}.post_attention_layernorm.weight
    /// talker.model.norm.weight
    /// talker.codec_head.weight
    /// ```
    pub fn load(
        config: &OmniTalkerConfig,
        get_tensor: impl Fn(&str) -> Option<Tensor>,
    ) -> Result<Self> {
        // Load embedding
        let embed_weight = get_tensor("talker.model.embed_tokens.weight")
            .ok_or_else(|| LludaError::Msg("Missing talker.model.embed_tokens.weight".into()))?;
        let embed_tokens = Embedding::new(embed_weight)?;

        // Load thinker_to_talker projection
        let proj_weight = get_tensor("talker.thinker_to_talker_proj.weight")
            .ok_or_else(|| LludaError::Msg("Missing talker.thinker_to_talker_proj.weight".into()))?;
        let proj_bias = get_tensor("talker.thinker_to_talker_proj.bias")
            .ok_or_else(|| LludaError::Msg("Missing talker.thinker_to_talker_proj.bias".into()))?;
        let thinker_to_talker_proj = LinearWithBias::new(proj_weight, proj_bias)?;

        // Create shared RoPE for talker
        let rotary = Arc::new(RotaryEmbedding::new(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )?);

        // Load all decoder layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            let layer = load_talker_layer(config, layer_idx, Arc::clone(&rotary), &get_tensor)?;
            layers.push(layer);
        }

        // Load final norm
        let norm_weight = get_tensor("talker.model.norm.weight")
            .ok_or_else(|| LludaError::Msg("Missing talker.model.norm.weight".into()))?;
        let norm = RmsNorm::new(norm_weight, config.rms_norm_eps)?;

        // Load codec head weight [vocab_size, hidden_size]
        // Pre-transpose to [hidden_size, vocab_size] for efficient matmul
        let codec_head_weight = get_tensor("talker.codec_head.weight")
            .ok_or_else(|| LludaError::Msg("Missing talker.codec_head.weight".into()))?;
        let codec_head_transposed = codec_head_weight.transpose_dims(0, 1)?;

        // Initialize KV caches (one per layer)
        let kv_caches = vec![KvCache::new(); config.num_hidden_layers];

        Ok(Talker {
            embed_tokens,
            thinker_to_talker_proj,
            layers,
            norm,
            codec_head_transposed,
            kv_caches,
            config: config.clone(),
        })
    }

    /// Forward pass: codec token IDs + thinker context -> codec logits.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Codec token IDs to embed, shape [seq]
    /// * `thinker_hidden` - Thinker hidden states for context fusion, shape [B, seq, embedding_size]
    /// * `offset` - Position offset for KV cache (length of cached sequence)
    ///
    /// # Returns
    ///
    /// Logits over the codec vocabulary, shape [B, seq, vocab_size].
    ///
    /// # Errors
    ///
    /// Returns error if input shapes are incompatible or operations fail.
    ///
    /// # Algorithm
    ///
    /// 1. Embed codec tokens into thinker's embedding space: [seq] -> [B, seq, embedding_size]
    /// 2. Add thinker hidden states (element-wise): [B, seq, embedding_size]
    /// 3. Project from embedding_size to talker hidden_size: [B, seq, hidden_size]
    /// 4. Pass through transformer layers with KV cache
    /// 5. Apply final RMSNorm
    /// 6. Compute logits: hidden @ codec_head.T -> [B, seq, vocab_size]
    pub fn forward(
        &mut self,
        input_ids: &[u32],
        thinker_hidden: &Tensor,
        offset: usize,
    ) -> Result<Tensor> {
        if input_ids.is_empty() {
            return Err(LludaError::Msg("input_ids cannot be empty".into()));
        }

        let thinker_shape = thinker_hidden.shape();
        if thinker_shape.len() < 2 {
            return Err(LludaError::ShapeMismatch {
                expected: vec![0, 0, 0],
                got: thinker_shape.to_vec(),
            });
        }

        // Infer batch and seq from thinker_hidden
        // thinker_hidden: [B, seq, embedding_size] or [seq, embedding_size]
        let (batch, seq_len) = if thinker_shape.len() == 3 {
            (thinker_shape[0], thinker_shape[1])
        } else {
            // Treat 2D as [seq, embedding_size] with batch=1
            (1, thinker_shape[0])
        };

        if seq_len != input_ids.len() {
            return Err(LludaError::ShapeMismatch {
                expected: vec![seq_len],
                got: vec![input_ids.len()],
            });
        }

        // 1. Embed codec tokens in thinker's space: [B*seq] -> [B, seq, embedding_size]
        let token_embeds = self.embed_tokens.forward(input_ids, &[batch, seq_len])?;
        // token_embeds: [B, seq, embedding_size]

        // 2. Add thinker hidden states
        // Both are [B, seq, embedding_size]
        let thinker_3d = if thinker_shape.len() == 2 {
            thinker_hidden.reshape(&[1, thinker_shape[0], thinker_shape[1]])?
        } else {
            thinker_hidden.clone()
        };
        let combined = token_embeds.add(&thinker_3d)?; // [B, seq, embedding_size]

        // 3. Project from embedding_size to hidden_size
        let mut hidden_states = self.thinker_to_talker_proj.forward(&combined)?; // [B, seq, hidden_size]

        // 4. Generate causal mask (if seq_len > 1)
        let mask = causal_mask(batch, seq_len, offset, DType::F32)?;

        // 5. Pass through all transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(
                &hidden_states,
                mask.as_ref(),
                &mut self.kv_caches[layer_idx],
                offset,
            )?;
        }

        // 6. Final normalization
        hidden_states = self.norm.forward(&hidden_states)?;

        // 7. Compute codec logits
        // hidden_states: [B, seq, hidden_size]
        // codec_head_transposed: [hidden_size, vocab_size]
        // logits: [B, seq, vocab_size]
        let hs_shape = hidden_states.shape();
        let batch_seq = hs_shape[0..hs_shape.len() - 1].iter().product::<usize>();
        let hidden_dim = hs_shape[hs_shape.len() - 1];

        let hs_2d = hidden_states.reshape(&[batch_seq, hidden_dim])?;
        let logits_2d = hs_2d.matmul(&self.codec_head_transposed)?;

        let vocab_size = logits_2d.shape()[1];
        let logits = logits_2d.reshape(&[batch, seq_len, vocab_size])?;

        Ok(logits)
    }

    /// Clear all KV caches.
    ///
    /// Call this at the start of a new TTS generation.
    pub fn clear_kv_cache(&mut self) {
        for cache in &mut self.kv_caches {
            cache.reset();
        }
    }
}

/// Load a single talker decoder layer from weights.
fn load_talker_layer(
    config: &OmniTalkerConfig,
    layer_idx: usize,
    rotary: Arc<RotaryEmbedding>,
    get_tensor: &impl Fn(&str) -> Option<Tensor>,
) -> Result<OmniDecoderLayer> {
    let prefix = format!("talker.model.layers.{}", layer_idx);

    // Load attention projections with bias
    let q_weight = get_tensor(&format!("{}.self_attn.q_proj.weight", prefix))
        .ok_or_else(|| LludaError::Msg(format!("Missing {}.self_attn.q_proj.weight", prefix)))?;
    let q_bias = get_tensor(&format!("{}.self_attn.q_proj.bias", prefix))
        .ok_or_else(|| LludaError::Msg(format!("Missing {}.self_attn.q_proj.bias", prefix)))?;
    let q_proj = LinearWithBias::new(q_weight, q_bias)?;

    let k_weight = get_tensor(&format!("{}.self_attn.k_proj.weight", prefix))
        .ok_or_else(|| LludaError::Msg(format!("Missing {}.self_attn.k_proj.weight", prefix)))?;
    let k_bias = get_tensor(&format!("{}.self_attn.k_proj.bias", prefix))
        .ok_or_else(|| LludaError::Msg(format!("Missing {}.self_attn.k_proj.bias", prefix)))?;
    let k_proj = LinearWithBias::new(k_weight, k_bias)?;

    let v_weight = get_tensor(&format!("{}.self_attn.v_proj.weight", prefix))
        .ok_or_else(|| LludaError::Msg(format!("Missing {}.self_attn.v_proj.weight", prefix)))?;
    let v_bias = get_tensor(&format!("{}.self_attn.v_proj.bias", prefix))
        .ok_or_else(|| LludaError::Msg(format!("Missing {}.self_attn.v_proj.bias", prefix)))?;
    let v_proj = LinearWithBias::new(v_weight, v_bias)?;

    // Load output projection (no bias)
    let o_weight = get_tensor(&format!("{}.self_attn.o_proj.weight", prefix))
        .ok_or_else(|| LludaError::Msg(format!("Missing {}.self_attn.o_proj.weight", prefix)))?;
    let o_proj = LinearNoBias::new(o_weight)?;

    let self_attn = OmniAttention::new(
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        rotary,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
    )?;

    // Load MLP projections
    let gate_proj_weight = get_tensor(&format!("{}.mlp.gate_proj.weight", prefix))
        .ok_or_else(|| LludaError::Msg(format!("Missing {}.mlp.gate_proj.weight", prefix)))?;
    let up_proj_weight = get_tensor(&format!("{}.mlp.up_proj.weight", prefix))
        .ok_or_else(|| LludaError::Msg(format!("Missing {}.mlp.up_proj.weight", prefix)))?;
    let down_proj_weight = get_tensor(&format!("{}.mlp.down_proj.weight", prefix))
        .ok_or_else(|| LludaError::Msg(format!("Missing {}.mlp.down_proj.weight", prefix)))?;
    let mlp = MLP::new(gate_proj_weight, up_proj_weight, down_proj_weight)?;

    // Load layer norms
    let input_layernorm_weight = get_tensor(&format!("{}.input_layernorm.weight", prefix))
        .ok_or_else(|| LludaError::Msg(format!("Missing {}.input_layernorm.weight", prefix)))?;
    let post_attention_layernorm_weight =
        get_tensor(&format!("{}.post_attention_layernorm.weight", prefix))
            .ok_or_else(|| {
                LludaError::Msg(format!(
                    "Missing {}.post_attention_layernorm.weight",
                    prefix
                ))
            })?;

    let input_layernorm = RmsNorm::new(input_layernorm_weight, config.rms_norm_eps)?;
    let post_attention_layernorm =
        RmsNorm::new(post_attention_layernorm_weight, config.rms_norm_eps)?;

    Ok(OmniDecoderLayer::new(
        self_attn,
        mlp,
        input_layernorm,
        post_attention_layernorm,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::OmniTalkerConfig;

    /// Build a minimal OmniTalkerConfig for testing.
    fn test_config() -> OmniTalkerConfig {
        OmniTalkerConfig {
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            vocab_size: 32,       // small codec vocab for testing
            embedding_size: 8,    // thinker dim (embedding_size)
            rms_norm_eps: 1e-6,
            rope_theta: 1000000.0,
            max_position_embeddings: 64,
            attention_bias: false,
            tts_codec_start: 28,
            tts_codec_end: 29,
            tts_codec_pad: 30,
        }
    }

    /// Create a Talker filled with small non-zero weights for testing.
    fn make_test_talker(cfg: &OmniTalkerConfig) -> Talker {
        let h = cfg.hidden_size;
        let e = cfg.embedding_size;
        let v = cfg.vocab_size;
        let n_heads = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let im = cfg.intermediate_size;
        let n_layers = cfg.num_hidden_layers;

        let get_tensor = |name: &str| -> Option<Tensor> {
            if name == "talker.model.embed_tokens.weight" {
                return Some(Tensor::new(vec![0.01f32; v * e], vec![v, e]).unwrap());
            }
            if name == "talker.thinker_to_talker_proj.weight" {
                return Some(Tensor::new(vec![0.01f32; h * e], vec![h, e]).unwrap());
            }
            if name == "talker.thinker_to_talker_proj.bias" {
                return Some(Tensor::new(vec![0.0f32; h], vec![h]).unwrap());
            }
            if name == "talker.model.norm.weight" {
                return Some(Tensor::new(vec![1.0f32; h], vec![h]).unwrap());
            }
            if name == "talker.codec_head.weight" {
                return Some(Tensor::new(vec![0.01f32; v * h], vec![v, h]).unwrap());
            }

            for i in 0..n_layers {
                let pfx = format!("talker.model.layers.{}", i);
                if name == format!("{}.self_attn.q_proj.weight", pfx) {
                    return Some(Tensor::new(vec![0.01f32; n_heads * hd * h], vec![n_heads * hd, h]).unwrap());
                }
                if name == format!("{}.self_attn.q_proj.bias", pfx) {
                    return Some(Tensor::new(vec![0.0f32; n_heads * hd], vec![n_heads * hd]).unwrap());
                }
                if name == format!("{}.self_attn.k_proj.weight", pfx) {
                    return Some(Tensor::new(vec![0.01f32; nkv * hd * h], vec![nkv * hd, h]).unwrap());
                }
                if name == format!("{}.self_attn.k_proj.bias", pfx) {
                    return Some(Tensor::new(vec![0.0f32; nkv * hd], vec![nkv * hd]).unwrap());
                }
                if name == format!("{}.self_attn.v_proj.weight", pfx) {
                    return Some(Tensor::new(vec![0.01f32; nkv * hd * h], vec![nkv * hd, h]).unwrap());
                }
                if name == format!("{}.self_attn.v_proj.bias", pfx) {
                    return Some(Tensor::new(vec![0.0f32; nkv * hd], vec![nkv * hd]).unwrap());
                }
                if name == format!("{}.self_attn.o_proj.weight", pfx) {
                    return Some(Tensor::new(vec![0.01f32; h * n_heads * hd], vec![h, n_heads * hd]).unwrap());
                }
                if name == format!("{}.mlp.gate_proj.weight", pfx) {
                    return Some(Tensor::new(vec![0.01f32; im * h], vec![im, h]).unwrap());
                }
                if name == format!("{}.mlp.up_proj.weight", pfx) {
                    return Some(Tensor::new(vec![0.01f32; im * h], vec![im, h]).unwrap());
                }
                if name == format!("{}.mlp.down_proj.weight", pfx) {
                    return Some(Tensor::new(vec![0.01f32; h * im], vec![h, im]).unwrap());
                }
                if name == format!("{}.input_layernorm.weight", pfx) {
                    return Some(Tensor::new(vec![1.0f32; h], vec![h]).unwrap());
                }
                if name == format!("{}.post_attention_layernorm.weight", pfx) {
                    return Some(Tensor::new(vec![1.0f32; h], vec![h]).unwrap());
                }
            }
            None
        };

        Talker::load(cfg, get_tensor).unwrap()
    }

    #[test]
    fn test_talker_load_shape() {
        let cfg = test_config();
        let talker = make_test_talker(&cfg);

        // Verify structural integrity: correct number of layers and KV caches
        assert_eq!(talker.layers.len(), cfg.num_hidden_layers);
        assert_eq!(talker.kv_caches.len(), cfg.num_hidden_layers);

        // KV caches should be empty initially
        for cache in &talker.kv_caches {
            assert_eq!(cache.seq_len(), 0);
        }
    }

    #[test]
    fn test_talker_forward_shape() {
        let cfg = test_config();
        let mut talker = make_test_talker(&cfg);

        let seq_len = 3;
        let input_ids: Vec<u32> = vec![0, 1, 2];

        // thinker_hidden: [1, seq, embedding_size]
        let thinker_hidden = Tensor::new(
            vec![0.1f32; 1 * seq_len * cfg.embedding_size],
            vec![1, seq_len, cfg.embedding_size],
        )
        .unwrap();

        let logits = talker.forward(&input_ids, &thinker_hidden, 0).unwrap();

        // Expected shape: [1, seq, vocab_size]
        assert_eq!(logits.shape(), &[1, seq_len, cfg.vocab_size]);

        let data = logits.to_vec_f32();
        assert!(data.iter().all(|&v| v.is_finite()), "Logits contain non-finite values");
    }

    #[test]
    fn test_talker_forward_single_token() {
        let cfg = test_config();
        let mut talker = make_test_talker(&cfg);

        // Prefill: process 3 tokens
        let input_ids_prefill: Vec<u32> = vec![0, 1, 2];
        let thinker_prefill = Tensor::new(
            vec![0.1f32; 1 * 3 * cfg.embedding_size],
            vec![1, 3, cfg.embedding_size],
        )
        .unwrap();
        let logits_prefill = talker.forward(&input_ids_prefill, &thinker_prefill, 0).unwrap();
        assert_eq!(logits_prefill.shape(), &[1, 3, cfg.vocab_size]);

        // Decode: single token at offset=3
        let input_ids_decode: Vec<u32> = vec![3];
        let thinker_decode =
            Tensor::new(vec![0.2f32; 1 * 1 * cfg.embedding_size], vec![1, 1, cfg.embedding_size])
                .unwrap();
        let logits_decode = talker.forward(&input_ids_decode, &thinker_decode, 3).unwrap();
        assert_eq!(logits_decode.shape(), &[1, 1, cfg.vocab_size]);

        let data = logits_decode.to_vec_f32();
        assert!(data.iter().all(|&v| v.is_finite()), "Decode logits contain non-finite values");
    }

    #[test]
    fn test_talker_clear_kv_cache() {
        let cfg = test_config();
        let mut talker = make_test_talker(&cfg);

        let input_ids: Vec<u32> = vec![0, 1];
        let thinker_hidden = Tensor::new(
            vec![0.1f32; 1 * 2 * cfg.embedding_size],
            vec![1, 2, cfg.embedding_size],
        )
        .unwrap();

        // Forward pass fills caches
        let _ = talker.forward(&input_ids, &thinker_hidden, 0).unwrap();
        assert_eq!(talker.kv_caches[0].seq_len(), 2);

        // Clear and verify
        talker.clear_kv_cache();
        for cache in &talker.kv_caches {
            assert_eq!(cache.seq_len(), 0);
        }
    }

    #[test]
    fn test_talker_load_missing_weight_fails() {
        let cfg = test_config();

        // get_tensor that always returns None
        let result = Talker::load(&cfg, |_name| None);
        assert!(result.is_err(), "Loading with no weights should fail");
    }
}
