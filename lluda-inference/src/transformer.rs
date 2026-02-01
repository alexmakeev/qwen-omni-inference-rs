//! Transformer decoder layer for Qwen3 models.
//!
//! Implements a single transformer block combining:
//! - Self-attention with pre-normalization
//! - Feed-forward MLP with pre-normalization
//! - Residual connections
//!
//! # Architecture (Pre-norm)
//!
//! ```text
//! h = x + attention(layer_norm(x))
//! out = h + mlp(layer_norm(h))
//! ```
//!
//! Pre-normalization (RMSNorm before attention/MLP) is used instead of
//! post-normalization for better training stability.

use crate::attention::{Attention, KvCache};
use crate::error::Result;
use crate::mlp::MLP;
use crate::rms_norm::RmsNorm;
use crate::tensor::Tensor;

/// Transformer decoder layer.
///
/// Combines self-attention and feed-forward network with residual connections
/// and pre-normalization (RMSNorm).
#[derive(Debug, Clone)]
pub struct DecoderLayer {
    /// Self-attention layer
    self_attn: Attention,
    /// Feed-forward MLP layer
    mlp: MLP,
    /// RMSNorm before self-attention
    input_layernorm: RmsNorm,
    /// RMSNorm before MLP
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    /// Create a new decoder layer.
    ///
    /// # Arguments
    ///
    /// * `self_attn` - Self-attention layer
    /// * `mlp` - Feed-forward MLP layer
    /// * `input_layernorm` - RMSNorm applied before attention
    /// * `post_attention_layernorm` - RMSNorm applied before MLP
    ///
    /// # Returns
    ///
    /// New DecoderLayer instance.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::sync::Arc;
    /// use lluda_inference::transformer::DecoderLayer;
    /// use lluda_inference::attention::{Attention, Linear};
    /// use lluda_inference::mlp::MLP;
    /// use lluda_inference::rms_norm::RmsNorm;
    /// use lluda_inference::rope::RotaryEmbedding;
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let hidden_size = 64;
    /// let intermediate_size = 192;
    /// let num_heads = 4;
    /// let num_kv_heads = 2;
    /// let head_dim = 16;
    ///
    /// // Create attention layer components
    /// let q_proj = Linear::new(
    ///     Tensor::new(vec![0.1; num_heads * head_dim * hidden_size], vec![num_heads * head_dim, hidden_size]).unwrap()
    /// ).unwrap();
    /// let k_proj = Linear::new(
    ///     Tensor::new(vec![0.1; num_kv_heads * head_dim * hidden_size], vec![num_kv_heads * head_dim, hidden_size]).unwrap()
    /// ).unwrap();
    /// let v_proj = Linear::new(
    ///     Tensor::new(vec![0.1; num_kv_heads * head_dim * hidden_size], vec![num_kv_heads * head_dim, hidden_size]).unwrap()
    /// ).unwrap();
    /// let o_proj = Linear::new(
    ///     Tensor::new(vec![0.1; hidden_size * num_heads * head_dim], vec![hidden_size, num_heads * head_dim]).unwrap()
    /// ).unwrap();
    /// let q_norm = RmsNorm::new(Tensor::new(vec![1.0; head_dim], vec![head_dim]).unwrap(), 1e-6).unwrap();
    /// let k_norm = RmsNorm::new(Tensor::new(vec![1.0; head_dim], vec![head_dim]).unwrap(), 1e-6).unwrap();
    /// let rotary = Arc::new(RotaryEmbedding::new(head_dim, 100, 10000.0).unwrap());
    ///
    /// let self_attn = Attention::new(q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, rotary, num_heads, num_kv_heads, head_dim).unwrap();
    ///
    /// // Create MLP components
    /// let gate_proj = Tensor::new(vec![0.1; intermediate_size * hidden_size], vec![intermediate_size, hidden_size]).unwrap();
    /// let up_proj = Tensor::new(vec![0.1; intermediate_size * hidden_size], vec![intermediate_size, hidden_size]).unwrap();
    /// let down_proj = Tensor::new(vec![0.1; hidden_size * intermediate_size], vec![hidden_size, intermediate_size]).unwrap();
    /// let mlp = MLP::new(gate_proj, up_proj, down_proj).unwrap();
    ///
    /// // Create normalization layers
    /// let input_layernorm = RmsNorm::new(Tensor::new(vec![1.0; hidden_size], vec![hidden_size]).unwrap(), 1e-6).unwrap();
    /// let post_attention_layernorm = RmsNorm::new(Tensor::new(vec![1.0; hidden_size], vec![hidden_size]).unwrap(), 1e-6).unwrap();
    ///
    /// let layer = DecoderLayer::new(self_attn, mlp, input_layernorm, post_attention_layernorm);
    /// ```
    pub fn new(
        self_attn: Attention,
        mlp: MLP,
        input_layernorm: RmsNorm,
        post_attention_layernorm: RmsNorm,
    ) -> Self {
        DecoderLayer {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        }
    }

    /// Forward pass through the decoder layer.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape `[B, L, hidden_size]`
    /// * `mask` - Optional attention mask of shape `[B, 1, L, L+offset]`
    /// * `kv_cache` - Mutable KV cache (will be updated)
    /// * `offset` - Position offset (length of cached sequence)
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[B, L, hidden_size]` (same as input).
    ///
    /// # Errors
    ///
    /// Returns error if shapes are incompatible or operations fail.
    ///
    /// # Algorithm (Pre-norm)
    ///
    /// 1. Apply RMSNorm to input: `x_norm = input_layernorm(x)`
    /// 2. Compute attention: `attn_out = self_attn(x_norm, mask, kv_cache, offset)`
    /// 3. First residual connection: `h = x + attn_out`
    /// 4. Apply RMSNorm: `h_norm = post_attention_layernorm(h)`
    /// 5. Compute MLP: `mlp_out = mlp(h_norm)`
    /// 6. Second residual connection: `out = h + mlp_out`
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::sync::Arc;
    /// use lluda_inference::transformer::DecoderLayer;
    /// use lluda_inference::attention::{Attention, Linear, KvCache};
    /// use lluda_inference::mlp::MLP;
    /// use lluda_inference::rms_norm::RmsNorm;
    /// use lluda_inference::rope::RotaryEmbedding;
    /// use lluda_inference::tensor::Tensor;
    ///
    /// // ... (create layer as in previous example)
    /// # let hidden_size = 64;
    /// # let intermediate_size = 192;
    /// # let num_heads = 4;
    /// # let num_kv_heads = 2;
    /// # let head_dim = 16;
    /// # let q_proj = Linear::new(Tensor::new(vec![0.1; num_heads * head_dim * hidden_size], vec![num_heads * head_dim, hidden_size]).unwrap()).unwrap();
    /// # let k_proj = Linear::new(Tensor::new(vec![0.1; num_kv_heads * head_dim * hidden_size], vec![num_kv_heads * head_dim, hidden_size]).unwrap()).unwrap();
    /// # let v_proj = Linear::new(Tensor::new(vec![0.1; num_kv_heads * head_dim * hidden_size], vec![num_kv_heads * head_dim, hidden_size]).unwrap()).unwrap();
    /// # let o_proj = Linear::new(Tensor::new(vec![0.1; hidden_size * num_heads * head_dim], vec![hidden_size, num_heads * head_dim]).unwrap()).unwrap();
    /// # let q_norm = RmsNorm::new(Tensor::new(vec![1.0; head_dim], vec![head_dim]).unwrap(), 1e-6).unwrap();
    /// # let k_norm = RmsNorm::new(Tensor::new(vec![1.0; head_dim], vec![head_dim]).unwrap(), 1e-6).unwrap();
    /// # let rotary = Arc::new(RotaryEmbedding::new(head_dim, 100, 10000.0).unwrap());
    /// # let self_attn = Attention::new(q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, rotary, num_heads, num_kv_heads, head_dim).unwrap();
    /// # let gate_proj = Tensor::new(vec![0.1; intermediate_size * hidden_size], vec![intermediate_size, hidden_size]).unwrap();
    /// # let up_proj = Tensor::new(vec![0.1; intermediate_size * hidden_size], vec![intermediate_size, hidden_size]).unwrap();
    /// # let down_proj = Tensor::new(vec![0.1; hidden_size * intermediate_size], vec![hidden_size, intermediate_size]).unwrap();
    /// # let mlp = MLP::new(gate_proj, up_proj, down_proj).unwrap();
    /// # let input_layernorm = RmsNorm::new(Tensor::new(vec![1.0; hidden_size], vec![hidden_size]).unwrap(), 1e-6).unwrap();
    /// # let post_attention_layernorm = RmsNorm::new(Tensor::new(vec![1.0; hidden_size], vec![hidden_size]).unwrap(), 1e-6).unwrap();
    /// # let layer = DecoderLayer::new(self_attn, mlp, input_layernorm, post_attention_layernorm);
    ///
    /// let x = Tensor::new(vec![0.5; 1 * 4 * 64], vec![1, 4, 64]).unwrap();
    /// let mut cache = KvCache::new();
    /// let output = layer.forward(&x, None, &mut cache, 0).unwrap();
    /// assert_eq!(output.shape(), &[1, 4, 64]);
    /// ```
    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        kv_cache: &mut KvCache,
        offset: usize,
    ) -> Result<Tensor> {
        // 1. Pre-norm: apply RMSNorm before attention
        let x_norm = self.input_layernorm.forward(x)?;

        // 2. Self-attention
        let attn_out = self.self_attn.forward(&x_norm, mask, kv_cache, offset)?;

        // 3. First residual connection
        let h = x.add(&attn_out)?;

        // 4. Pre-norm: apply RMSNorm before MLP
        let h_norm = self.post_attention_layernorm.forward(&h)?;

        // 5. MLP
        let mlp_out = self.mlp.forward(&h_norm)?;

        // 6. Second residual connection
        h.add(&mlp_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::Linear;
    use crate::rope::RotaryEmbedding;
    use std::sync::Arc;

    fn create_test_decoder_layer(
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> DecoderLayer {
        // Create attention layer
        let q_proj = Linear::new(
            Tensor::new(
                vec![0.1; num_heads * head_dim * hidden_size],
                vec![num_heads * head_dim, hidden_size],
            )
            .unwrap(),
        )
        .unwrap();
        let k_proj = Linear::new(
            Tensor::new(
                vec![0.1; num_kv_heads * head_dim * hidden_size],
                vec![num_kv_heads * head_dim, hidden_size],
            )
            .unwrap(),
        )
        .unwrap();
        let v_proj = Linear::new(
            Tensor::new(
                vec![0.1; num_kv_heads * head_dim * hidden_size],
                vec![num_kv_heads * head_dim, hidden_size],
            )
            .unwrap(),
        )
        .unwrap();
        let o_proj = Linear::new(
            Tensor::new(
                vec![0.1; hidden_size * num_heads * head_dim],
                vec![hidden_size, num_heads * head_dim],
            )
            .unwrap(),
        )
        .unwrap();
        let q_norm = RmsNorm::new(
            Tensor::new(vec![1.0; head_dim], vec![head_dim]).unwrap(),
            1e-6,
        )
        .unwrap();
        let k_norm = RmsNorm::new(
            Tensor::new(vec![1.0; head_dim], vec![head_dim]).unwrap(),
            1e-6,
        )
        .unwrap();
        let rotary = Arc::new(RotaryEmbedding::new(head_dim, 100, 10000.0).unwrap());

        let self_attn = Attention::new(
            q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, rotary, num_heads, num_kv_heads,
            head_dim,
        ).unwrap();

        // Create MLP layer
        let gate_proj = Tensor::new(
            vec![0.1; intermediate_size * hidden_size],
            vec![intermediate_size, hidden_size],
        )
        .unwrap();
        let up_proj = Tensor::new(
            vec![0.1; intermediate_size * hidden_size],
            vec![intermediate_size, hidden_size],
        )
        .unwrap();
        let down_proj = Tensor::new(
            vec![0.1; hidden_size * intermediate_size],
            vec![hidden_size, intermediate_size],
        )
        .unwrap();
        let mlp = MLP::new(gate_proj, up_proj, down_proj).unwrap();

        // Create normalization layers
        let input_layernorm = RmsNorm::new(
            Tensor::new(vec![1.0; hidden_size], vec![hidden_size]).unwrap(),
            1e-6,
        )
        .unwrap();
        let post_attention_layernorm = RmsNorm::new(
            Tensor::new(vec![1.0; hidden_size], vec![hidden_size]).unwrap(),
            1e-6,
        )
        .unwrap();

        DecoderLayer::new(self_attn, mlp, input_layernorm, post_attention_layernorm)
    }

    #[test]
    fn test_decoder_layer_shape_invariant() {
        // Test that output shape matches input shape
        let hidden_size = 64;
        let intermediate_size = 192;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 16;

        let layer = create_test_decoder_layer(
            hidden_size,
            intermediate_size,
            num_heads,
            num_kv_heads,
            head_dim,
        );

        let batch = 1;
        let seq_len = 4;

        let x = Tensor::new(
            vec![0.5; batch * seq_len * hidden_size],
            vec![batch, seq_len, hidden_size],
        )
        .unwrap();

        let mut cache = KvCache::new();
        let output = layer.forward(&x, None, &mut cache, 0).unwrap();

        // Output shape should match input shape
        assert_eq!(output.shape(), &[batch, seq_len, hidden_size]);
    }

    #[test]
    fn test_decoder_layer_residual_connections() {
        // Test that residual connections preserve information
        let hidden_size = 64;
        let intermediate_size = 192;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 16;

        let layer = create_test_decoder_layer(
            hidden_size,
            intermediate_size,
            num_heads,
            num_kv_heads,
            head_dim,
        );

        let x = Tensor::new(vec![1.0; 4 * 64], vec![1, 4, 64]).unwrap();
        let mut cache = KvCache::new();
        let output = layer.forward(&x, None, &mut cache, 0).unwrap();

        // Output should be different from input (due to attention and MLP)
        let output_data = output.to_vec_f32();

        // But output should be finite and non-zero (residual connections preserve signal)
        assert!(output_data.iter().all(|&x| x.is_finite()));
        assert!(output_data.iter().any(|&x| x != 0.0));

        // Output should contain the input signal (residual connection adds input)
        // Since we have two residual connections (x + attn, h + mlp), the output
        // will be influenced by the input
        let output_mean: f32 = output_data.iter().sum::<f32>() / output_data.len() as f32;
        assert!(output_mean.abs() > 0.1, "Residual connections should preserve signal strength");
    }

    #[test]
    fn test_decoder_layer_with_causal_mask() {
        let hidden_size = 64;
        let intermediate_size = 192;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 16;

        let layer = create_test_decoder_layer(
            hidden_size,
            intermediate_size,
            num_heads,
            num_kv_heads,
            head_dim,
        );

        let batch = 1;
        let seq_len = 4;

        let x = Tensor::new(
            vec![0.5; batch * seq_len * hidden_size],
            vec![batch, seq_len, hidden_size],
        )
        .unwrap();

        // Create causal mask
        let mut mask_data = vec![0.0f32; batch * seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i {
                    mask_data[i * seq_len + j] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::new(mask_data, vec![batch, 1, seq_len, seq_len]).unwrap();

        let mut cache = KvCache::new();
        let output = layer.forward(&x, Some(&mask), &mut cache, 0).unwrap();

        assert_eq!(output.shape(), &[batch, seq_len, hidden_size]);

        let output_data = output.to_vec_f32();
        assert!(
            output_data.iter().all(|&x| x.is_finite()),
            "Output contains non-finite values"
        );
    }

    #[test]
    fn test_decoder_layer_with_kv_cache() {
        // Test autoregressive generation with KV cache
        let hidden_size = 64;
        let intermediate_size = 192;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 16;

        let layer = create_test_decoder_layer(
            hidden_size,
            intermediate_size,
            num_heads,
            num_kv_heads,
            head_dim,
        );

        let mut cache = KvCache::new();

        // Step 1: Process initial prompt
        let x1 = Tensor::new(vec![0.5; 4 * 64], vec![1, 4, 64]).unwrap();
        let output1 = layer.forward(&x1, None, &mut cache, 0).unwrap();
        assert_eq!(output1.shape(), &[1, 4, 64]);
        assert_eq!(cache.seq_len(), 4);

        // Step 2: Generate next token
        let x2 = Tensor::new(vec![0.6; 64], vec![1, 1, 64]).unwrap();
        let output2 = layer.forward(&x2, None, &mut cache, 4).unwrap();
        assert_eq!(output2.shape(), &[1, 1, 64]);
        assert_eq!(cache.seq_len(), 5);

        // Step 3: Generate another token
        let x3 = Tensor::new(vec![0.7; 64], vec![1, 1, 64]).unwrap();
        let output3 = layer.forward(&x3, None, &mut cache, 5).unwrap();
        assert_eq!(output3.shape(), &[1, 1, 64]);
        assert_eq!(cache.seq_len(), 6);

        // Verify all outputs are finite
        for output in [output1, output2, output3] {
            let data = output.to_vec_f32();
            assert!(
                data.iter().all(|&x| x.is_finite()),
                "Output contains non-finite values"
            );
        }
    }

    #[test]
    fn test_decoder_layer_pre_norm() {
        // Test that normalization is applied BEFORE attention and MLP (pre-norm)
        // This is tested indirectly by verifying output is finite and reasonable
        let hidden_size = 64;
        let intermediate_size = 192;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 16;

        let layer = create_test_decoder_layer(
            hidden_size,
            intermediate_size,
            num_heads,
            num_kv_heads,
            head_dim,
        );

        // Use large input values to test normalization effectiveness
        let x = Tensor::new(vec![100.0; 4 * 64], vec![1, 4, 64]).unwrap();
        let mut cache = KvCache::new();
        let output = layer.forward(&x, None, &mut cache, 0).unwrap();

        let output_data = output.to_vec_f32();
        assert!(output_data.iter().all(|&x| x.is_finite()));

        // Pre-norm should keep values in reasonable range despite large input
        let max_val = output_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        assert!(max_val < 1000.0, "Pre-norm should prevent value explosion");
    }

    #[test]
    fn test_decoder_layer_batched() {
        // Test with batched input [B=2, L=3, D=64]
        let hidden_size = 64;
        let intermediate_size = 192;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 16;

        let layer = create_test_decoder_layer(
            hidden_size,
            intermediate_size,
            num_heads,
            num_kv_heads,
            head_dim,
        );

        let batch = 2;
        let seq_len = 3;

        let x = Tensor::new(
            vec![0.5; batch * seq_len * hidden_size],
            vec![batch, seq_len, hidden_size],
        )
        .unwrap();

        let mut cache = KvCache::new();
        let output = layer.forward(&x, None, &mut cache, 0).unwrap();

        assert_eq!(output.shape(), &[batch, seq_len, hidden_size]);

        let output_data = output.to_vec_f32();
        assert!(output_data.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_decoder_layer_output_is_not_input() {
        // Verify that the layer actually transforms the input (not identity)
        let hidden_size = 64;
        let intermediate_size = 192;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 16;

        let layer = create_test_decoder_layer(
            hidden_size,
            intermediate_size,
            num_heads,
            num_kv_heads,
            head_dim,
        );

        let x = Tensor::new(vec![1.0; 4 * 64], vec![1, 4, 64]).unwrap();
        let mut cache = KvCache::new();
        let output = layer.forward(&x, None, &mut cache, 0).unwrap();

        let input_data = x.to_vec_f32();
        let output_data = output.to_vec_f32();

        // Output should be different from input
        let differences: usize = input_data
            .iter()
            .zip(output_data.iter())
            .filter(|(&a, &b)| (a - b).abs() > 1e-6)
            .count();

        assert!(
            differences > 0,
            "Layer should transform input, not return it unchanged"
        );
    }
}
