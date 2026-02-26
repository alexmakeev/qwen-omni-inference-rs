//! Single encoder layer for the audio transformer.
//!
//! `AudioEncoderLayer` implements one block of the Whisper-style audio encoder.
//! Each block applies pre-normalization before both the attention and MLP sub-layers,
//! with residual connections after each:
//!
//! ```text
//! h   = x + AudioAttention(LayerNorm(x))
//! out = h + AudioMlp(LayerNorm(h))
//! ```
//!
//! # Components
//!
//! | Component            | Type           | Role                                   |
//! |----------------------|----------------|----------------------------------------|
//! | `self_attn`          | AudioAttention | Bidirectional MHA (no causal mask)     |
//! | `self_attn_layer_norm` | LayerNorm    | Pre-norm before attention              |
//! | `mlp`                | AudioMlp       | GELU two-layer feed-forward            |
//! | `final_layer_norm`   | LayerNorm      | Pre-norm before MLP                    |
//!
//! # Input shape
//!
//! The audio encoder operates on 2D tensors: `[seq_len, d_model]`.
//! For Qwen-Audio: `seq_len` is typically 1500, `d_model` is 1280.

use crate::audio_attention::AudioAttention;
use crate::audio_mlp::AudioMlp;
use crate::error::Result;
use crate::layer_norm::LayerNorm;
use crate::tensor::Tensor;

/// Single encoder layer for the audio transformer.
///
/// Architecture: `LayerNorm → AudioAttention → residual → LayerNorm → AudioMlp → residual`
#[derive(Debug, Clone)]
pub struct AudioEncoderLayer {
    /// Bidirectional multi-head self-attention.
    self_attn: AudioAttention,
    /// Layer normalization applied before self-attention.
    self_attn_layer_norm: LayerNorm,
    /// GELU two-layer feed-forward network.
    mlp: AudioMlp,
    /// Layer normalization applied before MLP.
    final_layer_norm: LayerNorm,
}

impl AudioEncoderLayer {
    /// Create a new audio encoder layer.
    ///
    /// # Arguments
    ///
    /// * `self_attn` - Bidirectional multi-head attention layer
    /// * `self_attn_layer_norm` - LayerNorm applied before attention
    /// * `mlp` - GELU feed-forward MLP
    /// * `final_layer_norm` - LayerNorm applied before MLP
    ///
    /// # Returns
    ///
    /// New `AudioEncoderLayer` instance.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::audio_encoder_layer::AudioEncoderLayer;
    /// use lluda_inference::audio_attention::{AudioAttention, Linear, LinearBias};
    /// use lluda_inference::audio_mlp::AudioMlp;
    /// use lluda_inference::layer_norm::LayerNorm;
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let d_model = 8;
    /// let num_heads = 2;
    /// let head_dim = 4;
    /// let intermediate = 16;
    ///
    /// let mk_w = |r: usize, c: usize| Tensor::new(vec![0.0; r * c], vec![r, c]).unwrap();
    /// let mk_b = |n: usize| Tensor::new(vec![0.0; n], vec![n]).unwrap();
    ///
    /// let attn = AudioAttention::new(
    ///     LinearBias::new(mk_w(d_model, d_model), mk_b(d_model)).unwrap(),
    ///     Linear::new(mk_w(d_model, d_model)).unwrap(),
    ///     LinearBias::new(mk_w(d_model, d_model), mk_b(d_model)).unwrap(),
    ///     LinearBias::new(mk_w(d_model, d_model), mk_b(d_model)).unwrap(),
    ///     num_heads,
    ///     head_dim,
    /// ).unwrap();
    ///
    /// let norm1 = LayerNorm::new(mk_b(d_model), mk_b(d_model), 1e-5).unwrap();
    ///
    /// let fc1 = LinearBias::new(mk_w(intermediate, d_model), mk_b(intermediate)).unwrap();
    /// let fc2 = LinearBias::new(mk_w(d_model, intermediate), mk_b(d_model)).unwrap();
    /// let mlp = AudioMlp::new(fc1, fc2);
    ///
    /// let norm2 = LayerNorm::new(mk_b(d_model), mk_b(d_model), 1e-5).unwrap();
    ///
    /// let layer = AudioEncoderLayer::new(attn, norm1, mlp, norm2);
    /// ```
    pub fn new(
        self_attn: AudioAttention,
        self_attn_layer_norm: LayerNorm,
        mlp: AudioMlp,
        final_layer_norm: LayerNorm,
    ) -> Self {
        Self {
            self_attn,
            self_attn_layer_norm,
            mlp,
            final_layer_norm,
        }
    }

    /// Forward pass through one audio encoder layer.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape `[seq_len, d_model]`
    ///
    /// # Returns
    ///
    /// Output tensor of the same shape `[seq_len, d_model]`.
    ///
    /// # Errors
    ///
    /// Returns error if input shape is incompatible with any sub-layer.
    ///
    /// # Process
    ///
    /// 1. Pre-norm attention block:
    ///    `h = x + self_attn(self_attn_layer_norm(x))`
    /// 2. Pre-norm MLP block:
    ///    `out = h + mlp(final_layer_norm(h))`
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Attention block with residual
        let residual = x.clone();
        let h = self.self_attn_layer_norm.forward(x)?;
        let h = self.self_attn.forward(&h)?;
        let x = residual.add(&h)?;

        // MLP block with residual
        let residual = x.clone();
        let h = self.final_layer_norm.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        residual.add(&h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_attention::{AudioAttention, Linear, LinearBias};

    /// Build a minimal AudioEncoderLayer for testing.
    ///
    /// Uses zero weights for all projections so the attention output is
    /// zero (the output bias is also zero). The LayerNorm weights are set
    /// to 1.0 (scale) and 0.0 (bias) so normalization is transparent for
    /// the residual-check test.
    fn make_layer(d_model: usize, num_heads: usize, head_dim: usize) -> AudioEncoderLayer {
        let mk_w = |r: usize, c: usize, v: f32| {
            Tensor::new(vec![v; r * c], vec![r, c]).unwrap()
        };
        let mk_b = |n: usize, v: f32| Tensor::new(vec![v; n], vec![n]).unwrap();

        let intermediate = d_model * 4;

        let q_proj = LinearBias::new(mk_w(d_model, d_model, 0.0), mk_b(d_model, 0.0)).unwrap();
        let k_proj = Linear::new(mk_w(d_model, d_model, 0.0)).unwrap();
        let v_proj = LinearBias::new(mk_w(d_model, d_model, 0.0), mk_b(d_model, 0.0)).unwrap();
        let out_proj = LinearBias::new(mk_w(d_model, d_model, 0.0), mk_b(d_model, 0.0)).unwrap();

        let attn = AudioAttention::new(q_proj, k_proj, v_proj, out_proj, num_heads, head_dim)
            .unwrap();

        let norm1 = LayerNorm::new(mk_b(d_model, 1.0), mk_b(d_model, 0.0), 1e-5).unwrap();

        let fc1 = LinearBias::new(mk_w(intermediate, d_model, 0.0), mk_b(intermediate, 0.0))
            .unwrap();
        let fc2 = LinearBias::new(mk_w(d_model, intermediate, 0.0), mk_b(d_model, 0.0)).unwrap();
        let mlp = AudioMlp::new(fc1, fc2);

        let norm2 = LayerNorm::new(mk_b(d_model, 1.0), mk_b(d_model, 0.0), 1e-5).unwrap();

        AudioEncoderLayer::new(attn, norm1, mlp, norm2)
    }

    /// Output shape must equal input shape [seq_len, d_model].
    #[test]
    fn test_audio_encoder_layer_shape() {
        let d_model = 8;
        let num_heads = 2;
        let head_dim = 4;
        let seq_len = 5;

        let layer = make_layer(d_model, num_heads, head_dim);

        let x = Tensor::new(vec![1.0; seq_len * d_model], vec![seq_len, d_model]).unwrap();
        let out = layer.forward(&x).unwrap();

        assert_eq!(out.shape(), &[seq_len, d_model]);
    }

    /// With zero attention weights and zero MLP weights (and zero biases),
    /// the residual connections pass the input through unchanged.
    /// The attention output is zero, so x + 0 = x.
    /// The MLP output is also zero, so x + 0 = x.
    /// Therefore output equals input (up to floating-point tolerance).
    #[test]
    fn test_audio_encoder_layer_residual() {
        let d_model = 8;
        let num_heads = 2;
        let head_dim = 4;
        let seq_len = 3;

        let layer = make_layer(d_model, num_heads, head_dim);

        // Use distinct non-zero input so that identity pass-through is meaningful
        let input_data: Vec<f32> = (0..seq_len * d_model)
            .map(|i| (i as f32) * 0.1 + 0.5)
            .collect();
        let x = Tensor::new(input_data.clone(), vec![seq_len, d_model]).unwrap();
        let out = layer.forward(&x).unwrap();

        assert_eq!(out.shape(), &[seq_len, d_model]);

        let out_data = out.to_vec_f32();

        // With zero attention + zero MLP, residual connections mean output == input
        for (i, (&expected, &actual)) in input_data.iter().zip(out_data.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-4,
                "residual check failed at index {}: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }

    /// Output must not contain NaN or Inf.
    #[test]
    fn test_audio_encoder_layer_finite_output() {
        let d_model = 8;
        let num_heads = 2;
        let head_dim = 4;
        let seq_len = 4;

        let layer = make_layer(d_model, num_heads, head_dim);

        let data: Vec<f32> = (0..seq_len * d_model)
            .map(|i| (i as f32) * 0.1 - 1.6)
            .collect();
        let x = Tensor::new(data, vec![seq_len, d_model]).unwrap();
        let out = layer.forward(&x).unwrap();

        let out_data = out.to_vec_f32();
        assert!(
            out_data.iter().all(|&v| v.is_finite()),
            "output must not contain NaN or Inf"
        );
    }
}
