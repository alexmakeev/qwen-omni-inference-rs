//! Whisper-style audio encoder for Qwen2.5-Omni.
//!
//! Encodes a log-mel spectrogram into a sequence of audio embeddings that can be
//! merged into the text token sequence of the thinker (language model).
//!
//! # Architecture
//!
//! ```text
//! Input: mel spectrogram [num_mel_bins, T]  (e.g. [128, 3000] for 30 s audio)
//!
//! 1. Conv stem:
//!    conv1(mel) → GELU        # [d_model, T]      (stride=1, padding=1)
//!    conv2(h)   → GELU        # [d_model, T/2]    (stride=2, padding=1)
//!
//! 2. Transpose:
//!    h = h.transpose()        # [T/2, d_model]
//!
//! 3. Add positional embedding:
//!    h = h + pos_embed[:T/2]  # [T/2, d_model]    (sinusoidal, stored as [1500, d_model])
//!
//! 4. Transformer encoder layers (32 × AudioEncoderLayer):
//!    h = layer.forward(h)     # [T/2, d_model]
//!
//! 5. Average pooling (stride 2):
//!    h = avg_pool_1d(h, 2)    # [T/4, d_model]
//!
//! 6. Final layer norm:
//!    h = ln_post(h)           # [T/4, d_model]
//!
//! 7. Projection to thinker hidden dimension:
//!    h = proj(h)              # [T/4, output_dim]  (2048 for Qwen2.5-Omni-3B)
//! ```
//!
//! # Example shape trace (short clip)
//!
//! mel [128, 186] → conv1 [1280, 186] → conv2 [1280, 93] → transpose [93, 1280]
//!   → +pos → 32 layers [93, 1280] → avg_pool [46, 1280] → ln_post [46, 1280]
//!   → proj [46, 2048]
//!
//! # Weight naming
//!
//! All weights are loaded from the `thinker.audio_tower.*` namespace in the
//! model's safetensors file.

use crate::audio_attention::{AudioAttention, Linear, LinearBias};
use crate::audio_encoder_layer::AudioEncoderLayer;
use crate::audio_mlp::AudioMlp;
use crate::config::OmniAudioConfig;
use crate::conv1d::Conv1d;
use crate::error::{LludaError, Result};
use crate::layer_norm::LayerNorm;
use crate::tensor::Tensor;

/// Whisper-style audio encoder for Qwen2.5-Omni.
///
/// Converts a log-mel spectrogram into audio embeddings compatible with the
/// thinker (language model) hidden dimension.
///
/// # Weight sources
///
/// Loaded from the `thinker.audio_tower.*` namespace:
/// - `conv1`, `conv2` — convolutional stem weights
/// - `layers.{i}.*` — 32 transformer encoder layer weights
/// - `ln_post.*` — final layer normalization
/// - `proj.*` — linear projection to thinker dimension
///
/// NOTE: `positional_embedding` is NOT loaded from safetensors.
/// It is a non-persistent PyTorch buffer and is computed deterministically via
/// `sinusoidal_position_embedding` at load time.
#[derive(Debug, Clone)]
pub struct AudioEncoder {
    /// First conv: [d_model, num_mel_bins, 3], stride=1, padding=1.
    conv1: Conv1d,
    /// Second conv: [d_model, d_model, 3], stride=2, padding=1.
    conv2: Conv1d,
    /// Sinusoidal positional embeddings [max_source_positions, d_model].
    /// NOT quantized — always stored as F32.
    positional_embedding: Tensor,
    /// Transformer encoder layers.
    layers: Vec<AudioEncoderLayer>,
    /// Final layer normalization applied after all transformer layers.
    ln_post: LayerNorm,
    /// Linear projection (with bias) from d_model to output_dim.
    proj: LinearBias,
}

impl AudioEncoder {
    /// Load AudioEncoder weights using a weight-lookup closure.
    ///
    /// # Arguments
    ///
    /// * `config` - Audio encoder configuration (`OmniAudioConfig`)
    /// * `get_tensor` - Closure that maps a weight name to an optional `Tensor`.
    ///   Returns `None` for missing weights; the loader will surface an error.
    ///
    /// # Returns
    ///
    /// Initialized `AudioEncoder` with all weights loaded.
    ///
    /// # Errors
    ///
    /// Returns `LludaError::Msg` listing the first missing weight name if any
    /// required tensor is absent, or propagates shape validation errors from
    /// the component constructors.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use lluda_inference::audio_encoder::AudioEncoder;
    /// use lluda_inference::config::OmniAudioConfig;
    ///
    /// # fn example(config: &OmniAudioConfig, weights: &std::collections::HashMap<String, lluda_inference::tensor::Tensor>) -> lluda_inference::error::Result<()> {
    /// let encoder = AudioEncoder::load(config, |name| weights.get(name).cloned())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn load(
        config: &OmniAudioConfig,
        get_tensor: impl Fn(&str) -> Option<Tensor>,
    ) -> Result<Self> {
        let prefix = "thinker.audio_tower";

        // Helper: retrieve a required tensor or fail with a descriptive error.
        let get = |name: &str| -> Result<Tensor> {
            get_tensor(name)
                .ok_or_else(|| LludaError::Msg(format!("Missing weight: {}", name)))
        };

        // ── Convolutional stem ────────────────────────────────────────────────
        // conv1: [d_model, num_mel_bins, 3], stride=1, padding=1
        let conv1 = Conv1d::new(
            get(&format!("{prefix}.conv1.weight"))?,
            get(&format!("{prefix}.conv1.bias"))?,
            1, // stride
            1, // padding
        )?;

        // conv2: [d_model, d_model, 3], stride=2, padding=1
        let conv2 = Conv1d::new(
            get(&format!("{prefix}.conv2.weight"))?,
            get(&format!("{prefix}.conv2.bias"))?,
            2, // stride
            1, // padding
        )?;

        // ── Positional embeddings ─────────────────────────────────────────────
        // Shape: [max_source_positions, d_model] (1500, 1280 for Qwen2.5-Omni-3B)
        // Not saved in safetensors (non-persistent buffer); computed deterministically.
        let positional_embedding = Self::sinusoidal_position_embedding(
            config.max_source_positions,
            config.d_model,
        )?;

        // ── Transformer encoder layers ────────────────────────────────────────
        let mut layers = Vec::with_capacity(config.encoder_layers);
        for i in 0..config.encoder_layers {
            let lp = format!("{prefix}.layers.{i}");

            // Self-attention projections
            let q_proj = LinearBias::new(
                get(&format!("{lp}.self_attn.q_proj.weight"))?,
                get(&format!("{lp}.self_attn.q_proj.bias"))?,
            )?;
            // k_proj has NO bias (Whisper convention)
            let k_proj = Linear::new(get(&format!("{lp}.self_attn.k_proj.weight"))?)?;
            let v_proj = LinearBias::new(
                get(&format!("{lp}.self_attn.v_proj.weight"))?,
                get(&format!("{lp}.self_attn.v_proj.bias"))?,
            )?;
            let out_proj = LinearBias::new(
                get(&format!("{lp}.self_attn.out_proj.weight"))?,
                get(&format!("{lp}.self_attn.out_proj.bias"))?,
            )?;

            let head_dim = config.d_model / config.encoder_attention_heads;
            let self_attn = AudioAttention::new(
                q_proj,
                k_proj,
                v_proj,
                out_proj,
                config.encoder_attention_heads,
                head_dim,
            )?;

            let self_attn_layer_norm = LayerNorm::new(
                get(&format!("{lp}.self_attn_layer_norm.weight"))?,
                get(&format!("{lp}.self_attn_layer_norm.bias"))?,
                config.layer_norm_eps,
            )?;

            // Feed-forward MLP
            let fc1 = LinearBias::new(
                get(&format!("{lp}.fc1.weight"))?,
                get(&format!("{lp}.fc1.bias"))?,
            )?;
            let fc2 = LinearBias::new(
                get(&format!("{lp}.fc2.weight"))?,
                get(&format!("{lp}.fc2.bias"))?,
            )?;
            let mlp = AudioMlp::new(fc1, fc2);

            let final_layer_norm = LayerNorm::new(
                get(&format!("{lp}.final_layer_norm.weight"))?,
                get(&format!("{lp}.final_layer_norm.bias"))?,
                config.layer_norm_eps,
            )?;

            layers.push(AudioEncoderLayer::new(
                self_attn,
                self_attn_layer_norm,
                mlp,
                final_layer_norm,
            ));
        }

        // ── Post-transformer normalization and projection ─────────────────────
        let ln_post = LayerNorm::new(
            get(&format!("{prefix}.ln_post.weight"))?,
            get(&format!("{prefix}.ln_post.bias"))?,
            config.layer_norm_eps,
        )?;

        // proj: [output_dim, d_model] + bias [output_dim]
        let proj = LinearBias::new(
            get(&format!("{prefix}.proj.weight"))?,
            get(&format!("{prefix}.proj.bias"))?,
        )?;

        Ok(Self {
            conv1,
            conv2,
            positional_embedding,
            layers,
            ln_post,
            proj,
        })
    }

    /// Encode a log-mel spectrogram into audio embeddings.
    ///
    /// # Arguments
    ///
    /// * `mel` - Log-mel spectrogram of shape `[num_mel_bins, T]`.
    ///   For 30 s audio at 50 frames/s: `[128, 3000]`.
    ///
    /// # Returns
    ///
    /// Audio embeddings of shape `[T/4, output_dim]`.
    ///
    /// # Errors
    ///
    /// Returns error if shape operations fail (e.g., sequence length too long for
    /// positional embedding table) or any sub-layer reports a shape mismatch.
    ///
    /// # Shape trace
    ///
    /// ```text
    /// mel [128, T]
    ///   conv1 + gelu → [d_model, T]
    ///   conv2 + gelu → [d_model, T/2]
    ///   transpose    → [T/2, d_model]
    ///   + pos_embed  → [T/2, d_model]
    ///   32 × layer   → [T/2, d_model]
    ///   avg_pool(2)  → [T/4, d_model]
    ///   ln_post      → [T/4, d_model]
    ///   proj         → [T/4, output_dim]
    /// ```
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        // 1. Convolutional stem with GELU activations
        //    mel: [num_mel_bins, T]  →  h: [d_model, T]
        let h = self.conv1.forward(mel)?;
        let h = h.gelu()?;

        //    h: [d_model, T]  →  [d_model, T/2]
        let h = self.conv2.forward(&h)?;
        let h = h.gelu()?;

        // 2. Transpose so sequence length is the leading dimension
        //    [d_model, T/2]  →  [T/2, d_model]
        let h = h.transpose()?;

        // 3. Add sinusoidal positional embeddings (truncated to actual length)
        let seq_len = h.shape()[0];
        let pos = self.positional_embedding.narrow(0, 0, seq_len)?;
        let h = h.add(&pos)?;

        // 4. Transformer encoder layers (bidirectional, no causal mask)
        let mut h = h;
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }

        // 5. Average pooling along the sequence axis (stride 2)
        //    [T/2, d_model]  →  [T/4, d_model]
        h = self.avg_pool_1d(&h, 2)?;

        // 6. Final layer normalization
        h = self.ln_post.forward(&h)?;

        // 7. Project to thinker hidden dimension
        //    [T/4, d_model]  →  [T/4, output_dim]
        self.proj.forward(&h)
    }

    /// Run the encoder step-by-step and return all intermediate tensors.
    ///
    /// Used for diagnostic tests to pinpoint where Rust diverges from Python.
    ///
    /// # Returns
    ///
    /// `Ok((after_conv_stem, after_pos_embed, layer_outputs))` where:
    /// - `after_conv_stem`  — output after conv1+gelu+conv2+gelu+transpose, shape `[T/2, d_model]`
    /// - `after_pos_embed`  — after adding positional embedding, shape `[T/2, d_model]`
    /// - `layer_outputs`    — Vec of per-layer hidden states, each `[T/2, d_model]`
    pub fn forward_diagnostic(
        &self,
        mel: &Tensor,
    ) -> Result<(Tensor, Tensor, Vec<Tensor>)> {
        // Conv stem
        let h = self.conv1.forward(mel)?;
        let h = h.gelu()?;
        let h = self.conv2.forward(&h)?;
        let h = h.gelu()?;
        let h = h.transpose()?;
        let after_conv_stem = h.clone();

        // Positional embedding
        let seq_len = h.shape()[0];
        let pos = self.positional_embedding.narrow(0, 0, seq_len)?;
        let h = h.add(&pos)?;
        let after_pos_embed = h.clone();

        // Transformer layers
        let mut h = h;
        let mut layer_outputs = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            h = layer.forward(&h)?;
            layer_outputs.push(h.clone());
        }

        Ok((after_conv_stem, after_pos_embed, layer_outputs))
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Generate sinusoidal positional embedding matching Whisper/Qwen2.5-Omni.
    ///
    /// The formula is:
    /// ```text
    ///   inv_timescales[i] = exp(-ln(10000) * i / (channels/2 - 1))
    ///   result[pos, i]              = sin(pos * inv_timescales[i])
    ///   result[pos, channels/2 + i] = cos(pos * inv_timescales[i])
    /// ```
    ///
    /// This matches the non-persistent buffer produced by Whisper and Qwen2.5-Omni
    /// and is therefore not stored in safetensors checkpoints.
    ///
    /// # Arguments
    ///
    /// * `length`   - Number of positions (rows). Typically 1500.
    /// * `channels` - Embedding dimension (columns). Must be even. Typically 1280.
    ///
    /// # Returns
    ///
    /// Tensor of shape `[length, channels]` filled with F32 sinusoidal values.
    pub(crate) fn sinusoidal_position_embedding(length: usize, channels: usize) -> Result<Tensor> {
        let half = channels / 2;
        let log_timescale_increment = (10000.0_f64).ln() / (half - 1) as f64;

        let mut data = vec![0.0f32; length * channels];

        for pos in 0..length {
            for i in 0..half {
                let inv_timescale = (-log_timescale_increment * i as f64).exp() as f32;
                let angle = pos as f32 * inv_timescale;
                data[pos * channels + i] = angle.sin();
                data[pos * channels + half + i] = angle.cos();
            }
        }

        Tensor::new(data, vec![length, channels])
    }

    /// 1-D average pooling over the first dimension with the given stride.
    ///
    /// Pools consecutive groups of `stride` frames by averaging.
    /// For `stride=2` this halves the sequence length.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape `[seq_len, d_model]`
    /// * `stride` - Number of frames to average together (must be ≥ 1)
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[seq_len / stride, d_model]`.
    ///
    /// # Note
    ///
    /// Trailing frames that do not fill a complete window are discarded
    /// (floor division). This matches PyTorch's `nn.AvgPool1d` default behaviour.
    fn avg_pool_1d(&self, x: &Tensor, stride: usize) -> Result<Tensor> {
        if x.ndim() != 2 {
            return Err(LludaError::Msg(format!(
                "avg_pool_1d expects 2D input, got {}D",
                x.ndim()
            )));
        }
        // x: [seq_len, d_model]
        let seq_len = x.shape()[0];
        let d_model = x.shape()[1];
        let out_len = seq_len / stride;

        let data = x.to_vec_f32();
        let mut output = vec![0.0f32; out_len * d_model];

        for i in 0..out_len {
            for j in 0..d_model {
                let mut sum = 0.0f32;
                for s in 0..stride {
                    let idx = (i * stride + s) * d_model + j;
                    sum += data[idx];
                }
                output[i * d_model + j] = sum / stride as f32;
            }
        }

        Tensor::new(output, vec![out_len, d_model])
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn mk_w(rows: usize, cols: usize) -> Tensor {
        Tensor::new(vec![0.0f32; rows * cols], vec![rows, cols]).unwrap()
    }

    fn mk_b(n: usize) -> Tensor {
        Tensor::new(vec![0.0f32; n], vec![n]).unwrap()
    }

    fn mk_w3(a: usize, b: usize, c: usize) -> Tensor {
        Tensor::new(vec![0.0f32; a * b * c], vec![a, b, c]).unwrap()
    }

    /// Build a minimal weight map for a tiny AudioEncoder.
    ///
    /// Dimensions:
    ///   num_mel_bins = 4
    ///   d_model      = 8
    ///   num_heads    = 2   (head_dim = 4)
    ///   ffn_dim      = 16
    ///   output_dim   = 6
    ///   num_layers   = 1
    ///
    /// Note: positional_embedding is NOT in the weight map — it is computed
    /// deterministically via sinusoidal_position_embedding.
    fn make_weight_map(
        num_mel_bins: usize,
        d_model: usize,
        ffn_dim: usize,
        output_dim: usize,
        num_layers: usize,
    ) -> HashMap<String, Tensor> {
        let mut map: HashMap<String, Tensor> = HashMap::new();
        let p = "thinker.audio_tower";

        // Convolutional stem
        map.insert(format!("{p}.conv1.weight"), mk_w3(d_model, num_mel_bins, 3));
        map.insert(format!("{p}.conv1.bias"), mk_b(d_model));
        map.insert(format!("{p}.conv2.weight"), mk_w3(d_model, d_model, 3));
        map.insert(format!("{p}.conv2.bias"), mk_b(d_model));

        // Transformer layers
        for i in 0..num_layers {
            let lp = format!("{p}.layers.{i}");
            map.insert(format!("{lp}.self_attn.q_proj.weight"), mk_w(d_model, d_model));
            map.insert(format!("{lp}.self_attn.q_proj.bias"), mk_b(d_model));
            map.insert(format!("{lp}.self_attn.k_proj.weight"), mk_w(d_model, d_model));
            map.insert(format!("{lp}.self_attn.v_proj.weight"), mk_w(d_model, d_model));
            map.insert(format!("{lp}.self_attn.v_proj.bias"), mk_b(d_model));
            map.insert(format!("{lp}.self_attn.out_proj.weight"), mk_w(d_model, d_model));
            map.insert(format!("{lp}.self_attn.out_proj.bias"), mk_b(d_model));
            map.insert(format!("{lp}.self_attn_layer_norm.weight"), mk_b(d_model));
            map.insert(format!("{lp}.self_attn_layer_norm.bias"), mk_b(d_model));
            map.insert(format!("{lp}.fc1.weight"), mk_w(ffn_dim, d_model));
            map.insert(format!("{lp}.fc1.bias"), mk_b(ffn_dim));
            map.insert(format!("{lp}.fc2.weight"), mk_w(d_model, ffn_dim));
            map.insert(format!("{lp}.fc2.bias"), mk_b(d_model));
            map.insert(format!("{lp}.final_layer_norm.weight"), mk_b(d_model));
            map.insert(format!("{lp}.final_layer_norm.bias"), mk_b(d_model));
        }

        // Post-norm and projection
        map.insert(format!("{p}.ln_post.weight"), mk_b(d_model));
        map.insert(format!("{p}.ln_post.bias"), mk_b(d_model));
        map.insert(format!("{p}.proj.weight"), mk_w(output_dim, d_model));
        map.insert(format!("{p}.proj.bias"), mk_b(output_dim));

        map
    }

    // ── Test: sinusoidal_position_embedding ──────────────────────────────────

    /// Verify shape and known boundary values for sinusoidal positional embedding.
    ///
    /// Mathematical anchors:
    ///   pos=0, i=0        → sin(0) = 0.0
    ///   pos=0, i=channels/2 → cos(0) = 1.0
    ///   pos=1, i=0        → sin(1 * exp(0)) = sin(1.0)
    #[test]
    fn test_sinusoidal_position_embedding() {
        let pe = AudioEncoder::sinusoidal_position_embedding(1500, 1280).unwrap();
        assert_eq!(pe.shape(), &[1500, 1280]);

        let data = pe.to_vec_f32();
        // Position 0, first element: sin(0) = 0
        assert!((data[0] - 0.0).abs() < 1e-6, "pe[0,0] should be sin(0)=0, got {}", data[0]);
        // Position 0, element at channels/2: cos(0) = 1
        assert!(
            (data[640] - 1.0).abs() < 1e-6,
            "pe[0,640] should be cos(0)=1, got {}",
            data[640]
        );
        // Position 1, first element: sin(1 * exp(0)) = sin(1.0)
        let expected = 1.0_f32.sin();
        assert!(
            (data[1280] - expected).abs() < 1e-4,
            "pe[1,0] should be sin(1.0)={}, got {}",
            expected,
            data[1280]
        );
    }

    /// Verify that the reference .npy matches our generated embedding if available.
    #[test]
    fn test_sinusoidal_embedding_matches_reference() {
        let ref_path = "/home/alexmak/lluda/reference_data/omni_3b/positional_embedding.npy";
        if !std::path::Path::new(ref_path).exists() {
            eprintln!("Skipping: reference not found at {}", ref_path);
            return;
        }

        // Read the .npy file manually: header + raw f32 data.
        // Standard npy format: magic "\x93NUMPY", version byte × 2, header_len LE u16, header, data.
        let raw = std::fs::read(ref_path).expect("Failed to read reference npy");
        // Parse npy header to get data offset
        assert_eq!(&raw[0..6], b"\x93NUMPY", "Not a valid .npy file");
        let header_len = u16::from_le_bytes([raw[8], raw[9]]) as usize;
        let data_offset = 10 + header_len;
        let float_bytes = &raw[data_offset..];
        let ref_data: Vec<f32> = float_bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        let pe = AudioEncoder::sinusoidal_position_embedding(1500, 1280).unwrap();
        let our_data = pe.to_vec_f32();

        assert_eq!(our_data.len(), ref_data.len(), "Length mismatch vs reference");

        let max_diff: f32 = our_data
            .iter()
            .zip(ref_data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!("Positional embedding max diff from Python: {:.6e}", max_diff);
        assert!(
            max_diff < 1e-3,
            "Positional embedding diverges from reference: max_diff={}",
            max_diff
        );
    }

    // ── Test: avg_pool_1d ────────────────────────────────────────────────────

    /// avg_pool_1d with stride=2 on a known input.
    ///
    /// Input [4, 2]:
    ///   row 0: [1, 2]
    ///   row 1: [3, 4]
    ///   row 2: [5, 6]
    ///   row 3: [7, 8]
    ///
    /// Expected output [2, 2]:
    ///   row 0: [(1+3)/2, (2+4)/2] = [2, 3]
    ///   row 1: [(5+7)/2, (6+8)/2] = [6, 7]
    #[test]
    fn test_avg_pool_1d() {
        // Build a dummy encoder to call avg_pool_1d (private method via the struct)
        let num_mel_bins = 4;
        let d_model = 8;
        let ffn_dim = 16;
        let output_dim = 6;
        let num_layers = 1;

        let weights = make_weight_map(num_mel_bins, d_model, ffn_dim, output_dim, num_layers);

        let config = OmniAudioConfig {
            d_model,
            encoder_layers: num_layers,
            encoder_attention_heads: 2,
            encoder_ffn_dim: ffn_dim,
            output_dim,
            num_mel_bins,
            layer_norm_eps: 1e-5,
            max_source_positions: 32,
        };

        let encoder =
            AudioEncoder::load(&config, |name| weights.get(name).cloned()).unwrap();

        // Known-value input [4, 2]
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![4, 2],
        )
        .unwrap();

        let out = encoder.avg_pool_1d(&x, 2).unwrap();
        assert_eq!(out.shape(), &[2, 2], "avg_pool output shape");

        let data = out.to_vec_f32();
        let tol = 1e-5f32;
        assert!((data[0] - 2.0).abs() < tol, "pool[0,0] expected 2.0, got {}", data[0]);
        assert!((data[1] - 3.0).abs() < tol, "pool[0,1] expected 3.0, got {}", data[1]);
        assert!((data[2] - 6.0).abs() < tol, "pool[1,0] expected 6.0, got {}", data[2]);
        assert!((data[3] - 7.0).abs() < tol, "pool[1,1] expected 7.0, got {}", data[3]);
    }

    // ── Test: output shape ────────────────────────────────────────────────────

    /// Forward pass through the full encoder — verify output shape chain.
    ///
    /// Tiny configuration:
    ///   num_mel_bins = 4, d_model = 8, num_heads = 2, head_dim = 4
    ///   ffn_dim = 16, output_dim = 6, num_layers = 1, max_source_positions = 32
    ///
    /// Input mel: [4, 16]
    ///   conv1 (stride=1, pad=1): [8, 16]
    ///   conv2 (stride=2, pad=1): [8, 8]
    ///   transpose:               [8, 8]
    ///   + pos (truncated to 8):  [8, 8]
    ///   1 × layer:               [8, 8]
    ///   avg_pool(stride=2):      [4, 8]
    ///   ln_post:                 [4, 8]
    ///   proj:                    [4, 6]
    #[test]
    fn test_audio_encoder_shape() {
        let num_mel_bins = 4;
        let d_model = 8;
        let ffn_dim = 16;
        let output_dim = 6;
        let num_layers = 1;

        let weights = make_weight_map(num_mel_bins, d_model, ffn_dim, output_dim, num_layers);

        let config = OmniAudioConfig {
            d_model,
            encoder_layers: num_layers,
            encoder_attention_heads: 2,
            encoder_ffn_dim: ffn_dim,
            output_dim,
            num_mel_bins,
            layer_norm_eps: 1e-5,
            max_source_positions: 32,
        };

        let encoder =
            AudioEncoder::load(&config, |name| weights.get(name).cloned()).unwrap();

        // Input mel spectrogram: [num_mel_bins=4, T=16]
        let mel = Tensor::new(vec![0.5f32; num_mel_bins * 16], vec![num_mel_bins, 16]).unwrap();

        let out = encoder.forward(&mel).unwrap();

        // Expected shape: [T/4, output_dim] = [4, 6]
        assert_eq!(out.shape(), &[4, output_dim], "AudioEncoder output shape mismatch");

        // Output must be finite
        let data = out.to_vec_f32();
        assert!(
            data.iter().all(|&v| v.is_finite()),
            "AudioEncoder output contains non-finite values"
        );
    }

    // ── Test: missing weight returns error ────────────────────────────────────

    /// Loading with an empty weight map must return an error (missing weights).
    #[test]
    fn test_audio_encoder_missing_weight_error() {
        let config = OmniAudioConfig {
            d_model: 8,
            encoder_layers: 1,
            encoder_attention_heads: 2,
            encoder_ffn_dim: 16,
            output_dim: 6,
            num_mel_bins: 4,
            layer_norm_eps: 1e-5,
            max_source_positions: 32,
        };

        let result = AudioEncoder::load(&config, |_name| None);
        assert!(result.is_err(), "Empty weight map should produce an error");
    }
}
