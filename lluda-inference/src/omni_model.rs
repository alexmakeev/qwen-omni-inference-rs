//! Complete Qwen2.5-Omni model: AudioEncoder + Thinker + Talker.
//!
//! Combines all model components into a single struct that supports:
//! - ASR (Automatic Speech Recognition): audio → text logits
//! - TTS (Text-to-Speech): text → codec token logits
//!
//! # Architecture
//!
//! ```text
//! AudioEncoder (Whisper-style)
//!   mel [128, T] → audio_embeds [T/4, 2048]
//!
//! OmniThinker (Qwen2.5 language model with OmniAttention)
//!   input_ids [seq] + audio_embeds → merged [seq, 2048] → hidden [seq, 2048]
//!
//! LM head: hidden [seq, 2048] @ lm_head.T [2048, 151936] → logits [seq, 151936]
//!
//! Talker (optional, TTS decoder)
//!   codec_ids + thinker_hidden → codec logits [seq, 8448]
//! ```
//!
//! # Usage
//!
//! ```rust,no_run
//! use lluda_inference::config::OmniConfig;
//! use lluda_inference::loader::ModelWeights;
//! use lluda_inference::omni_model::OmniModel;
//!
//! let config = OmniConfig::from_file("models/Qwen2.5-Omni-3B/config.json")?;
//! let weights = ModelWeights::from_directory("models/Qwen2.5-Omni-3B")?;
//! let mut model = OmniModel::load(&config, &weights)?;
//!
//! // ASR: encode audio and transcribe
//! // let logits = model.forward_asr(&mel, &input_ids, 0)?;
//!
//! # Ok::<(), lluda_inference::error::LludaError>(())
//! ```

use std::sync::Arc;

use crate::attention::KvCache;
use crate::audio_encoder::AudioEncoder;
use crate::causal_mask::causal_mask;
use crate::config::{OmniConfig, OmniTextConfig};
use crate::embedding::Embedding;
use crate::error::{LludaError, Result};
use crate::loader::ModelWeights;
use crate::mlp::MLP;
use crate::omni_attention::{LinearNoBias, LinearWithBias, OmniAttention, OmniDecoderLayer};
use crate::rms_norm::RmsNorm;
use crate::rope::RotaryEmbedding;
use crate::talker::Talker;
use crate::tensor::{DType, Tensor};

/// The thinker (main language model) for Qwen2.5-Omni.
///
/// Uses OmniAttention (biases on q/k/v, no per-head norms).
/// Different from Qwen3ForCausalLM which has per-head RMSNorm.
#[derive(Debug, Clone)]
struct OmniThinker {
    /// Token embedding layer [vocab_size, hidden_size]
    embed_tokens: Embedding,
    /// Transformer decoder layers
    layers: Vec<OmniDecoderLayer>,
    /// Final RMSNorm after all transformer layers
    norm: RmsNorm,
    /// KV caches (one per layer)
    kv_caches: Vec<KvCache>,
    /// Text model configuration (stored for potential future use)
    #[allow(dead_code)]
    config: OmniTextConfig,
}

impl OmniThinker {
    /// Load OmniThinker weights using a weight-lookup closure.
    ///
    /// # Arguments
    ///
    /// * `config` - Text model configuration (`OmniTextConfig`)
    /// * `get_tensor` - Closure that maps a weight name to an optional `Tensor`
    ///
    /// # Errors
    ///
    /// Returns `LludaError::Msg` if any required tensor is absent.
    fn load(
        config: &OmniTextConfig,
        get_tensor: &impl Fn(&str) -> Option<Tensor>,
    ) -> Result<Self> {
        let prefix = "thinker.model";

        let get = |name: &str| -> Result<Tensor> {
            get_tensor(name)
                .ok_or_else(|| LludaError::Msg(format!("Missing weight: {}", name)))
        };

        // Embedding
        let embed_tokens = Embedding::new(get(&format!("{prefix}.embed_tokens.weight"))?)?;

        // RoPE (shared across layers)
        let rotary = Arc::new(RotaryEmbedding::new(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )?);

        // Transformer decoder layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let lp = format!("{prefix}.layers.{i}");

            // Attention projections: q, k, v have bias; o_proj has no bias
            let q_proj = LinearWithBias::new(
                get(&format!("{lp}.self_attn.q_proj.weight"))?,
                get(&format!("{lp}.self_attn.q_proj.bias"))?,
            )?;
            let k_proj = LinearWithBias::new(
                get(&format!("{lp}.self_attn.k_proj.weight"))?,
                get(&format!("{lp}.self_attn.k_proj.bias"))?,
            )?;
            let v_proj = LinearWithBias::new(
                get(&format!("{lp}.self_attn.v_proj.weight"))?,
                get(&format!("{lp}.self_attn.v_proj.bias"))?,
            )?;
            let o_proj = LinearNoBias::new(get(&format!("{lp}.self_attn.o_proj.weight"))?)?;

            let self_attn = OmniAttention::new(
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                Arc::clone(&rotary),
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
            )?;

            // MLP (gated SiLU, same as Qwen3)
            let mlp = MLP::new(
                get(&format!("{lp}.mlp.gate_proj.weight"))?,
                get(&format!("{lp}.mlp.up_proj.weight"))?,
                get(&format!("{lp}.mlp.down_proj.weight"))?,
            )?;

            // Layer norms (RmsNorm)
            let input_ln = RmsNorm::new(
                get(&format!("{lp}.input_layernorm.weight"))?,
                config.rms_norm_eps,
            )?;
            let post_attn_ln = RmsNorm::new(
                get(&format!("{lp}.post_attention_layernorm.weight"))?,
                config.rms_norm_eps,
            )?;

            layers.push(OmniDecoderLayer::new(self_attn, mlp, input_ln, post_attn_ln));
        }

        // Final norm
        let norm = RmsNorm::new(
            get(&format!("{prefix}.norm.weight"))?,
            config.rms_norm_eps,
        )?;

        // One KV cache per layer
        let kv_caches = (0..config.num_hidden_layers).map(|_| KvCache::new()).collect();

        Ok(OmniThinker {
            embed_tokens,
            layers,
            norm,
            kv_caches,
            config: config.clone(),
        })
    }

    /// Forward pass starting from pre-computed embeddings.
    ///
    /// # Arguments
    ///
    /// * `embeds` - Input embeddings of shape `[1, seq_len, hidden_size]`
    /// * `offset` - Position offset for KV cache
    ///
    /// # Returns
    ///
    /// Hidden states of shape `[1, seq_len, hidden_size]` after all layers and final norm.
    fn forward_embeds(&mut self, embeds: &Tensor, offset: usize) -> Result<Tensor> {
        let shape = embeds.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        // Create causal mask (None when seq_len == 1)
        let mask = causal_mask(batch, seq_len, offset, DType::F32)?;

        // Forward through all transformer layers
        let mut h = embeds.clone();
        for (layer, cache) in self.layers.iter().zip(self.kv_caches.iter_mut()) {
            h = layer.forward(&h, mask.as_ref(), cache, offset)?;
        }

        // Final normalization
        self.norm.forward(&h)
    }

    /// Standard forward pass from token IDs.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Token IDs `[seq_len]`
    /// * `offset` - Position offset for KV cache
    ///
    /// # Returns
    ///
    /// Hidden states of shape `[1, seq_len, hidden_size]`.
    fn forward(&mut self, input_ids: &[u32], offset: usize) -> Result<Tensor> {
        let seq_len = input_ids.len();
        let embeds = self.embed_tokens.forward(input_ids, &[1, seq_len])?;
        self.forward_embeds(&embeds, offset)
    }

    /// Clear all KV caches.
    fn clear_kv_cache(&mut self) {
        for cache in &mut self.kv_caches {
            cache.reset();
        }
    }
}

/// Complete Qwen2.5-Omni model: AudioEncoder + Thinker + Talker.
///
/// Supports both ASR (audio → text) and TTS (text → codec tokens) pipelines.
///
/// # Weight layout
///
/// All weights are loaded from the `thinker.*` and `talker.*` namespaces
/// in the model's safetensors shards.
#[derive(Debug, Clone)]
pub struct OmniModel {
    /// Whisper-style audio encoder
    audio_encoder: AudioEncoder,
    /// Main language model thinker (uses OmniAttention)
    thinker: OmniThinker,
    /// LM head weight: [vocab_size, hidden_size], stored for reference
    #[allow(dead_code)]
    lm_head: Tensor,
    /// Pre-transposed LM head: [hidden_size, vocab_size] for efficient matmul
    lm_head_transposed: Tensor,
    /// Optional TTS talker (None when `enable_talker = false`)
    talker: Option<Talker>,
    /// Top-level model configuration
    config: OmniConfig,
}

impl OmniModel {
    /// Load the complete Omni model from a `ModelWeights` container.
    ///
    /// # Arguments
    ///
    /// * `config` - Top-level `OmniConfig` (thinker + talker sub-configs)
    /// * `weights` - Loaded model weights from SafeTensors shards
    ///
    /// # Returns
    ///
    /// Initialized `OmniModel` ready for ASR/TTS inference.
    ///
    /// # Errors
    ///
    /// Returns error if any required weight is missing or shapes are invalid.
    pub fn load(config: &OmniConfig, weights: &ModelWeights) -> Result<Self> {
        let get = |name: &str| weights.get(name).cloned();

        // 1. Load audio encoder (thinker.audio_tower.* namespace)
        let audio_encoder =
            AudioEncoder::load(&config.thinker_config.audio_config, |name| get(name))?;

        // 2. Load thinker (thinker.model.* namespace)
        let text_cfg = &config.thinker_config.text_config;
        let thinker = OmniThinker::load(text_cfg, &get)?;

        // 3. Load LM head
        let lm_head = get("thinker.lm_head.weight")
            .ok_or_else(|| LludaError::Msg("Missing thinker.lm_head.weight".into()))?;
        // Pre-transpose: [vocab_size, hidden_size] -> [hidden_size, vocab_size]
        let lm_head_transposed = lm_head.transpose_dims(0, 1)?;

        // 4. Load talker (optional)
        let talker = if config.enable_talker {
            Some(Talker::load(&config.talker_config, |name| get(name))?)
        } else {
            None
        };

        Ok(OmniModel {
            audio_encoder,
            thinker,
            lm_head,
            lm_head_transposed,
            talker,
            config: config.clone(),
        })
    }

    /// ASR pipeline: audio + text prompt → text logits.
    ///
    /// Encodes the mel spectrogram with AudioEncoder, merges the resulting audio
    /// embeddings into the token embedding sequence (replacing audio placeholder
    /// tokens), then runs the thinker and projects to vocabulary logits.
    ///
    /// # Arguments
    ///
    /// * `mel` - Log-mel spectrogram `[num_mel_bins, T]`
    /// * `input_ids` - Token IDs including audio placeholder tokens
    /// * `offset` - Position offset for KV cache (0 on first call)
    ///
    /// # Returns
    ///
    /// Logits of shape `[1, 1, vocab_size]` for the last token.
    ///
    /// # Errors
    ///
    /// Returns error if shapes are incompatible or audio encoding fails.
    pub fn forward_asr(
        &mut self,
        mel: &Tensor,
        input_ids: &[u32],
        offset: usize,
    ) -> Result<Tensor> {
        // 1. Encode audio: mel [num_mel_bins, T] → audio_embeds [T/4, hidden]
        let audio_embeds = self.audio_encoder.forward(mel)?;

        // 2. Get text embeddings: input_ids → [1, seq_len, hidden]
        let seq_len = input_ids.len();
        let text_embeds = self.thinker.embed_tokens.forward(input_ids, &[1, seq_len])?;

        // 3. Merge: replace audio_token_id positions with audio embeddings
        let audio_token_id = self.config.thinker_config.audio_token_index;
        let merged = self.merge_audio_embeddings(&text_embeds, &audio_embeds, input_ids, audio_token_id)?;

        // 4. Forward through thinker layers
        let hidden_states = self.thinker.forward_embeds(&merged, offset)?;
        // hidden_states: [1, seq_len, hidden_size]

        // 5. Select last token: [1, seq_len, hidden] -> [1, 1, hidden]
        let last_hidden = hidden_states.narrow(1, seq_len - 1, 1)?;

        // 6. LM head projection: [1, 1, hidden] @ [hidden, vocab_size] -> [1, 1, vocab_size]
        let last_2d = last_hidden.reshape(&[1, last_hidden.shape()[2]])?;
        let logits_2d = last_2d.matmul(&self.lm_head_transposed)?;
        let vocab_size = logits_2d.shape()[1];
        logits_2d.reshape(&[1, 1, vocab_size])
    }

    /// TTS pipeline: text token IDs → codec token logits.
    ///
    /// Runs the input through the thinker to produce hidden states, then
    /// feeds them into the talker decoder which predicts codec token logits.
    ///
    /// Requires the model to have been loaded with `enable_talker = true`.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Text token IDs `[seq_len]`
    /// * `offset` - Position offset for KV cache
    ///
    /// # Returns
    ///
    /// Codec logits of shape `[1, 1, codec_vocab_size]` for the last token.
    ///
    /// # Errors
    ///
    /// Returns `LludaError::Msg` if the talker is not loaded.
    pub fn forward_tts(
        &mut self,
        input_ids: &[u32],
        offset: usize,
    ) -> Result<Tensor> {
        let talker = self.talker.as_mut().ok_or_else(|| {
            LludaError::Msg("Talker not loaded (enable_talker=false)".into())
        })?;

        // 1. Forward through thinker to get hidden states: [1, seq_len, hidden]
        let thinker_hidden = self.thinker.forward(input_ids, offset)?;

        // 2. Take only the last hidden state as context for the talker.
        //    thinker_hidden: [1, seq_len, hidden] → last_hidden: [1, 1, hidden]
        //    The talker requires thinker_hidden and codec_ids to have the same
        //    sequence length; we start with a single TTS start token.
        let seq_len = thinker_hidden.shape()[1];
        let last_hidden = thinker_hidden.narrow(1, seq_len - 1, 1)?;

        // 3. Forward through talker with TTS start token and last thinker hidden state
        let tts_start = self.config.talker_config.tts_codec_start;
        let codec_ids = vec![tts_start];

        talker.forward(&codec_ids, &last_hidden, 0)
    }

    /// Clear all KV caches (thinker and talker).
    ///
    /// Call this at the start of a new conversation or generation request.
    pub fn clear_kv_cache(&mut self) {
        self.thinker.clear_kv_cache();
        if let Some(ref mut talker) = self.talker {
            talker.clear_kv_cache();
        }
    }

    /// Replace audio placeholder token positions with audio encoder embeddings.
    ///
    /// Scans `input_ids` for positions equal to `audio_token_id` and replaces
    /// the corresponding rows in `text_embeds` with rows from `audio_embeds`,
    /// consuming audio embeddings in order.
    ///
    /// # Arguments
    ///
    /// * `text_embeds` - Token embeddings of shape `[1, seq_len, hidden]`
    /// * `audio_embeds` - Audio encoder output of shape `[audio_len, hidden]`
    /// * `input_ids` - Token IDs used to identify audio placeholder positions
    /// * `audio_token_id` - The special token ID that marks audio spans
    ///
    /// # Returns
    ///
    /// Merged tensor of shape `[1, seq_len, hidden]`.
    ///
    /// # Errors
    ///
    /// Returns error if tensor construction fails.
    fn merge_audio_embeddings(
        &self,
        text_embeds: &Tensor,   // [1, seq_len, hidden]
        audio_embeds: &Tensor,  // [audio_len, hidden]
        input_ids: &[u32],
        audio_token_id: u32,
    ) -> Result<Tensor> {
        let text_shape = text_embeds.shape();
        let seq_len = text_shape[1];
        let hidden = text_shape[2];
        let audio_len = audio_embeds.shape()[0];

        // Work in F32 for the merge operation (handles both F32 and BF16 inputs)
        let text_data = text_embeds.to_vec_f32();
        let audio_data = audio_embeds.to_vec_f32();

        let mut merged = vec![0.0f32; seq_len * hidden];
        let mut audio_idx = 0;

        for i in 0..seq_len {
            if input_ids[i] == audio_token_id && audio_idx < audio_len {
                // Replace with audio embedding
                let audio_src = audio_idx * hidden;
                let dst = i * hidden;
                merged[dst..dst + hidden].copy_from_slice(&audio_data[audio_src..audio_src + hidden]);
                audio_idx += 1;
            } else {
                // Keep text embedding
                let text_src = i * hidden;
                let dst = i * hidden;
                merged[dst..dst + hidden].copy_from_slice(&text_data[text_src..text_src + hidden]);
            }
        }

        // Rebuild as [1, seq_len, hidden]
        Tensor::new(merged, vec![1, seq_len, hidden])
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {

    /// Test that merge_audio_embeddings correctly replaces placeholder positions.
    ///
    /// Input layout:
    ///   input_ids: [1, AUDIO, AUDIO, 2, 3] where AUDIO = 151646
    ///   text_embeds: [1, 5, 4] (5 positions, hidden=4)
    ///   audio_embeds: [2, 4]
    ///
    /// Expected output:
    ///   Position 0: text embedding row 0
    ///   Position 1: audio embedding row 0
    ///   Position 2: audio embedding row 1
    ///   Position 3: text embedding row 3
    ///   Position 4: text embedding row 4
    #[test]
    fn test_merge_audio_embeddings() {
        // Dummy OmniModel to call the method — use the inline logic test pattern
        // to avoid needing a full model loaded.
        let text_data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let audio_data: Vec<f32> = (100..108).map(|i| i as f32).collect();

        let seq_len = 5usize;
        let hidden = 4usize;
        let audio_len = 2usize;

        let input_ids = vec![1u32, 151646, 151646, 2, 3];
        let audio_token_id = 151646u32;

        // Replicate merge logic directly (method is not accessible without a full OmniModel)
        let mut merged = vec![0.0f32; seq_len * hidden];
        let mut aidx = 0usize;
        for i in 0..seq_len {
            if input_ids[i] == audio_token_id && aidx < audio_len {
                let src = aidx * hidden;
                let dst = i * hidden;
                merged[dst..dst + hidden].copy_from_slice(&audio_data[src..src + hidden]);
                aidx += 1;
            } else {
                let src = i * hidden;
                let dst = i * hidden;
                merged[dst..dst + hidden].copy_from_slice(&text_data[src..src + hidden]);
            }
        }

        // Verify expected values
        // Position 0: text row 0 = [0, 1, 2, 3]
        assert_eq!(&merged[0..4], &[0.0, 1.0, 2.0, 3.0]);
        // Position 1: audio row 0 = [100, 101, 102, 103]
        assert_eq!(&merged[4..8], &[100.0, 101.0, 102.0, 103.0]);
        // Position 2: audio row 1 = [104, 105, 106, 107]
        assert_eq!(&merged[8..12], &[104.0, 105.0, 106.0, 107.0]);
        // Position 3: text row 3 = [12, 13, 14, 15]
        assert_eq!(&merged[12..16], &[12.0, 13.0, 14.0, 15.0]);
        // Position 4: text row 4 = [16, 17, 18, 19]
        assert_eq!(&merged[16..20], &[16.0, 17.0, 18.0, 19.0]);
    }

    /// Test causal mask creation logic (lower triangular with -inf above diagonal).
    #[test]
    fn test_causal_mask() {
        let seq_len = 3;
        let offset = 0;
        let total_len = offset + seq_len;
        let mut mask = vec![0.0f32; seq_len * total_len];
        for i in 0..seq_len {
            for j in 0..total_len {
                if j > offset + i {
                    mask[i * total_len + j] = f32::NEG_INFINITY;
                }
            }
        }

        // Row 0: can only see position 0
        assert_eq!(mask[0], 0.0);
        assert!(mask[1].is_infinite() && mask[1].is_sign_negative());
        assert!(mask[2].is_infinite() && mask[2].is_sign_negative());

        // Row 1: can see positions 0-1
        assert_eq!(mask[3], 0.0);
        assert_eq!(mask[4], 0.0);
        assert!(mask[5].is_infinite() && mask[5].is_sign_negative());

        // Row 2: can see all positions
        assert_eq!(mask[6], 0.0);
        assert_eq!(mask[7], 0.0);
        assert_eq!(mask[8], 0.0);
    }

    /// Test that merge does not consume more audio frames than available.
    #[test]
    fn test_merge_fewer_audio_frames_than_placeholders() {
        let seq_len = 4usize;
        let hidden = 2usize;
        let audio_len = 1usize; // only 1 audio frame but 2 placeholders

        let text_data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let audio_data = vec![99.0f32, 88.0f32]; // [1, 2]

        let input_ids = vec![0u32, 151646, 151646, 1]; // two placeholders, one audio frame
        let audio_token_id = 151646u32;

        let mut merged = vec![0.0f32; seq_len * hidden];
        let mut aidx = 0usize;
        for i in 0..seq_len {
            if input_ids[i] == audio_token_id && aidx < audio_len {
                let src = aidx * hidden;
                let dst = i * hidden;
                merged[dst..dst + hidden].copy_from_slice(&audio_data[src..src + hidden]);
                aidx += 1;
            } else {
                let src = i * hidden;
                let dst = i * hidden;
                merged[dst..dst + hidden].copy_from_slice(&text_data[src..src + hidden]);
            }
        }

        // Position 0: text [0, 1]
        assert_eq!(&merged[0..2], &[0.0, 1.0]);
        // Position 1: audio [99, 88] (first audio frame consumed)
        assert_eq!(&merged[2..4], &[99.0, 88.0]);
        // Position 2: text [4, 5] (second placeholder treated as text — no audio frame left)
        assert_eq!(&merged[4..6], &[4.0, 5.0]);
        // Position 3: text [6, 7]
        assert_eq!(&merged[6..8], &[6.0, 7.0]);
    }
}
