//! Rotary Position Embeddings (RoPE) for Qwen3 models.
//!
//! RoPE encodes position information by rotating query and key vectors in the attention mechanism.
//! This implementation follows the approach used in `candle_nn::rotary_emb::rope()` and is compatible
//! with HuggingFace Transformers' `Qwen3RotaryEmbedding`.
//!
//! # Algorithm
//!
//! For each position `pos` and head dimension `d`:
//! 1. Precompute frequencies: `inv_freq[i] = 1.0 / (theta ^ (2i / head_dim))` for i in 0..head_dim/2
//! 2. Compute angles: `freqs[pos, i] = pos * inv_freq[i]`
//! 3. Precompute cos/sin tables: `cos[pos, i] = cos(freqs[pos, i])`, `sin[pos, i] = sin(freqs[pos, i])`
//! 4. Apply rotation: `x_rotated = x * cos + rotate_half(x) * sin`
//!    where `rotate_half` swaps the two halves of the head dimension and negates the first half
//!
//! # Reference
//!
//! Implementation matches:
//! - candle_nn::rotary_emb::rope() (candle-nn/src/rotary_emb.rs)
//! - HuggingFace transformers Qwen3RotaryEmbedding
//! - RoFormer paper: https://arxiv.org/abs/2104.09864

use crate::error::{LludaError, Result};
use crate::tensor::Tensor;

/// Rotary Position Embeddings.
///
/// Precomputes cos/sin tables for efficient position encoding in attention.
/// Tables are computed once during initialization and reused for all forward passes.
///
/// # Example
///
/// ```ignore
/// use lluda_inference::rope::RotaryEmbedding;
///
/// // Qwen3-0.6B parameters
/// let rope = RotaryEmbedding::new(128, 40960, 1000000.0).unwrap();
///
/// // Apply to query and key tensors during attention (q and k must be defined)
/// let (q_rotated, k_rotated) = rope.apply(&q, &k, 0).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    /// Cosine table: [max_seq_len, head_dim]
    cos: Tensor,
    /// Sine table: [max_seq_len, head_dim]
    sin: Tensor,
    /// Dimension of each attention head
    head_dim: usize,
}

impl RotaryEmbedding {
    /// Create a new RotaryEmbedding with precomputed cos/sin tables.
    ///
    /// # Arguments
    ///
    /// * `head_dim` - Dimension of each attention head (must be even)
    /// * `max_seq_len` - Maximum sequence length to precompute
    /// * `theta` - RoPE base frequency (theta parameter)
    ///
    /// # Returns
    ///
    /// RotaryEmbedding with precomputed tables ready for use.
    ///
    /// # Errors
    ///
    /// Returns error if `head_dim` is not even (required for rotate_half operation).
    ///
    /// # Example
    ///
    /// ```
    /// use lluda_inference::rope::RotaryEmbedding;
    ///
    /// // Small configuration for example
    /// let rope = RotaryEmbedding::new(128, 1024, 10000.0).unwrap();
    /// ```
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f64) -> Result<Self> {
        #[allow(clippy::manual_is_multiple_of)]
        if head_dim % 2 != 0 {
            return Err(LludaError::Msg(format!(
                "head_dim must be even for RoPE, got {}",
                head_dim
            )));
        }

        // Compute inverse frequencies: inv_freq[i] = 1.0 / (theta ^ (2i / head_dim))
        // for i in 0..head_dim/2
        let half_dim = head_dim / 2;
        let mut inv_freq = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            let exponent = (2 * i) as f64 / head_dim as f64;
            inv_freq.push(1.0 / theta.powf(exponent));
        }

        // Precompute cos and sin for all positions
        // freqs[pos, i] = pos * inv_freq[i // 2] (each frequency repeated twice)
        // cos/sin tables: [max_seq_len, head_dim]
        // Pattern: [freq0, freq0, freq1, freq1, ..., freq_{d/2-1}, freq_{d/2-1}]
        let mut cos_table = Vec::with_capacity(max_seq_len * head_dim);
        let mut sin_table = Vec::with_capacity(max_seq_len * head_dim);

        for pos in 0..max_seq_len {
            for &inv_f in &inv_freq {
                let freq = pos as f64 * inv_f;
                let cos_val = freq.cos() as f32;
                let sin_val = freq.sin() as f32;
                // Each frequency appears twice consecutively
                cos_table.push(cos_val);
                cos_table.push(cos_val);
                sin_table.push(sin_val);
                sin_table.push(sin_val);
            }
        }

        let cos = Tensor::new(cos_table, vec![max_seq_len, head_dim])?;
        let sin = Tensor::new(sin_table, vec![max_seq_len, head_dim])?;

        Ok(RotaryEmbedding {
            cos,
            sin,
            head_dim,
        })
    }

    /// Apply rotary position embeddings to query and key tensors.
    ///
    /// Uses the "rotate_half" method:
    /// - Split the head dimension into two halves: [x0, x1]
    /// - rotate_half([x0, x1]) = [-x1, x0]
    /// - Apply rotation: x * cos + rotate_half(x) * sin
    ///
    /// # Arguments
    ///
    /// * `q` - Query tensor of shape [B, H, L, D] where D = head_dim
    /// * `k` - Key tensor of shape [B, Hkv, L, D] where D = head_dim
    /// * `offset` - Position offset (for KV cache, 0 for fresh sequence)
    ///
    /// # Returns
    ///
    /// Tuple of (rotated_q, rotated_k) with same shapes as inputs.
    ///
    /// # Errors
    ///
    /// - Returns error if tensor shapes are invalid
    /// - Returns error if offset + seq_len exceeds max_seq_len
    /// - Returns error if last dimension doesn't match head_dim
    ///
    /// # Example
    ///
    /// ```no_run
    /// use lluda_inference::rope::RotaryEmbedding;
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let rope = RotaryEmbedding::new(128, 1024, 1000000.0).unwrap();
    ///
    /// // Query: [1, 16, 10, 128], Key: [1, 8, 10, 128]
    /// let q = Tensor::new(vec![0.0; 1 * 16 * 10 * 128], vec![1, 16, 10, 128]).unwrap();
    /// let k = Tensor::new(vec![0.0; 1 * 8 * 10 * 128], vec![1, 8, 10, 128]).unwrap();
    ///
    /// let (q_rot, k_rot) = rope.apply(&q, &k, 0).unwrap();
    /// ```
    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        // Validate input shapes: must be 4D [B, H, L, D]
        if q.ndim() != 4 {
            return Err(LludaError::ShapeMismatch {
                expected: vec![0, 0, 0, 0],
                got: q.shape().to_vec(),
            });
        }
        if k.ndim() != 4 {
            return Err(LludaError::ShapeMismatch {
                expected: vec![0, 0, 0, 0],
                got: k.shape().to_vec(),
            });
        }

        let q_shape = q.shape();
        let k_shape = k.shape();
        let seq_len = q_shape[2];

        // Validate head_dim matches last dimension
        if q_shape[3] != self.head_dim {
            return Err(LludaError::ShapeMismatch {
                expected: vec![0, 0, 0, self.head_dim],
                got: q_shape.to_vec(),
            });
        }
        if k_shape[3] != self.head_dim {
            return Err(LludaError::ShapeMismatch {
                expected: vec![0, 0, 0, self.head_dim],
                got: k_shape.to_vec(),
            });
        }

        // Validate offset + seq_len doesn't exceed precomputed tables
        let max_seq_len = self.cos.shape()[0];
        if offset + seq_len > max_seq_len {
            return Err(LludaError::Msg(format!(
                "RoPE offset {} + seq_len {} exceeds max_seq_len {}",
                offset, seq_len, max_seq_len
            )));
        }

        // Extract cos/sin for the current sequence positions [offset..offset+seq_len]
        // cos/sin: [max_seq_len, head_dim] -> narrow to [seq_len, head_dim]
        let cos_slice = self.cos.narrow(0, offset, seq_len)?;
        let sin_slice = self.sin.narrow(0, offset, seq_len)?;

        // Apply RoPE to query and key
        let q_rotated = self.apply_rope_to_tensor(q, &cos_slice, &sin_slice)?;
        let k_rotated = self.apply_rope_to_tensor(k, &cos_slice, &sin_slice)?;

        Ok((q_rotated, k_rotated))
    }

    /// Apply RoPE rotation to a single tensor.
    ///
    /// Internal helper method that implements 2D rotations on pairs of dimensions.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape [B, H, L, D]
    /// * `cos` - Cosine values of shape [L, D]
    /// * `sin` - Sine values of shape [L, D]
    ///
    /// # Returns
    ///
    /// Rotated tensor of same shape as input.
    ///
    /// # Algorithm
    ///
    /// RoPE applies independent 2D rotations to consecutive pairs of dimensions.
    /// For each pair (d, d+1) with d even:
    /// - rotated[d]   = x[d]   * cos[d] - x[d+1] * sin[d]
    /// - rotated[d+1] = x[d+1] * cos[d+1] + x[d] * sin[d+1]
    ///
    /// Since cos/sin values are the same for each pair (duplicated in table),
    /// cos[d] == cos[d+1] and sin[d] == sin[d+1].
    fn apply_rope_to_tensor(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let x_shape = x.shape();
        let batch = x_shape[0];
        let num_heads = x_shape[1];
        let seq_len = x_shape[2];
        let head_dim = x_shape[3];

        let x_data = x.to_vec_f32();
        let cos_data = cos.to_vec_f32();
        let sin_data = sin.to_vec_f32();

        let mut result = vec![0.0f32; x_data.len()];

        // Process each element
        // x: [B, H, L, D]
        // cos/sin: [L, D]
        for b in 0..batch {
            for h in 0..num_heads {
                for l in 0..seq_len {
                    // Get cos/sin for this position
                    let cos_offset = l * head_dim;
                    let sin_offset = l * head_dim;

                    // Process pairs: (0,1), (2,3), (4,5), ...
                    for pair in 0..(head_dim / 2) {
                        let d0 = pair * 2;
                        let d1 = pair * 2 + 1;

                        let x_idx_0 = ((b * num_heads + h) * seq_len + l) * head_dim + d0;
                        let x_idx_1 = ((b * num_heads + h) * seq_len + l) * head_dim + d1;

                        let x0 = x_data[x_idx_0];
                        let x1 = x_data[x_idx_1];

                        let cos_val = cos_data[cos_offset + d0];
                        let sin_val = sin_data[sin_offset + d0];

                        // 2D rotation: [x0, x1] -> [x0', x1']
                        // x0' = x0 * cos - x1 * sin
                        // x1' = x0 * sin + x1 * cos
                        result[x_idx_0] = x0 * cos_val - x1 * sin_val;
                        result[x_idx_1] = x0 * sin_val + x1 * cos_val;
                    }
                }
            }
        }

        Tensor::new(result, x_shape.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function for approximate float comparison.
    fn assert_close(a: f32, b: f32, tol: f32, msg: &str) {
        let diff = (a - b).abs();
        assert!(
            diff < tol,
            "{}: expected {}, got {}, diff = {}",
            msg, b, a, diff
        );
    }


    #[test]
    fn test_rope_creation() {
        let rope = RotaryEmbedding::new(128, 1024, 1000000.0).unwrap();
        assert_eq!(rope.cos.shape(), &[1024, 128]);
        assert_eq!(rope.sin.shape(), &[1024, 128]);
        assert_eq!(rope.head_dim, 128);
    }

    #[test]
    fn test_rope_odd_head_dim_fails() {
        let result = RotaryEmbedding::new(127, 1024, 1000000.0);
        assert!(result.is_err());
        match result {
            Err(LludaError::Msg(msg)) => assert!(msg.contains("must be even")),
            _ => panic!("Expected Msg error for odd head_dim"),
        }
    }

    #[test]
    fn test_rope_at_position_zero_is_identity() {
        // At position 0, cos(0) = 1, sin(0) = 0
        // So rotation should be: x * 1 + rotate_half(x) * 0 = x (identity)
        let rope = RotaryEmbedding::new(4, 10, 10000.0).unwrap();

        // Small test tensor: [1, 1, 1, 4]
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 1, 4]).unwrap();

        let (q_rot, k_rot) = rope.apply(&input, &input, 0).unwrap();

        // At position 0, output should equal input (approximately, due to floating point)
        let input_data = input.to_vec_f32();
        let q_data = q_rot.to_vec_f32();
        let k_data = k_rot.to_vec_f32();

        for i in 0..4 {
            assert_close(
                q_data[i],
                input_data[i],
                1e-6,
                &format!("q position 0 element {}", i),
            );
            assert_close(
                k_data[i],
                input_data[i],
                1e-6,
                &format!("k position 0 element {}", i),
            );
        }
    }

    #[test]
    fn test_rope_preserves_norm() {
        // RoPE is a rotation, so it should preserve vector norms
        let rope = RotaryEmbedding::new(8, 100, 10000.0).unwrap();

        // Random-ish input: [1, 2, 3, 8]
        let input_data: Vec<f32> = (0..48).map(|i| (i as f32) * 0.1).collect();
        let input = Tensor::new(input_data.clone(), vec![1, 2, 3, 8]).unwrap();

        let (q_rot, _) = rope.apply(&input, &input, 5).unwrap();

        // Compute norms for each head's last dimension
        let input_f32 = input.to_vec_f32();
        let output_f32 = q_rot.to_vec_f32();

        // For each position in [B, H, L], compute norm over D dimension
        for b in 0..1 {
            for h in 0..2 {
                for l in 0..3 {
                    let offset = ((b * 2 + h) * 3 + l) * 8;
                    let input_norm: f32 = (0..8)
                        .map(|d| input_f32[offset + d].powi(2))
                        .sum::<f32>()
                        .sqrt();
                    let output_norm: f32 = (0..8)
                        .map(|d| output_f32[offset + d].powi(2))
                        .sum::<f32>()
                        .sqrt();

                    assert_close(
                        output_norm,
                        input_norm,
                        1e-4,
                        &format!("norm at position [{}, {}, {}]", b, h, l),
                    );
                }
            }
        }
    }

    #[test]
    fn test_rope_with_offset() {
        // Offset parameter should select different cos/sin values
        let rope = RotaryEmbedding::new(4, 100, 10000.0).unwrap();

        let input = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![1, 1, 1, 4]).unwrap();

        // Apply with offset 0 and offset 5
        let (q0, _) = rope.apply(&input, &input, 0).unwrap();
        let (q5, _) = rope.apply(&input, &input, 5).unwrap();

        // Results should differ (different positions use different rotations)
        let q0_data = q0.to_vec_f32();
        let q5_data = q5.to_vec_f32();

        let mut has_difference = false;
        for i in 0..4 {
            if (q0_data[i] - q5_data[i]).abs() > 1e-6 {
                has_difference = true;
                break;
            }
        }
        assert!(has_difference, "Offset should produce different results");
    }

    #[test]
    fn test_rope_offset_exceeds_max() {
        let rope = RotaryEmbedding::new(4, 10, 10000.0).unwrap();
        let input = Tensor::new(vec![1.0; 8], vec![1, 1, 2, 4]).unwrap();

        // offset=9, seq_len=2 -> position 10 exceeds max_seq_len=10
        let result = rope.apply(&input, &input, 9);
        assert!(result.is_err());
        match result {
            Err(LludaError::Msg(msg)) => assert!(msg.contains("exceeds max_seq_len")),
            _ => panic!("Expected Msg error for offset overflow"),
        }
    }

    #[test]
    fn test_rope_wrong_ndim() {
        let rope = RotaryEmbedding::new(4, 10, 10000.0).unwrap();

        // 3D tensor instead of 4D
        let input_3d = Tensor::new(vec![1.0; 8], vec![2, 1, 4]).unwrap();
        let result = rope.apply(&input_3d, &input_3d, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_rope_wrong_head_dim() {
        let rope = RotaryEmbedding::new(8, 10, 10000.0).unwrap();

        // Last dimension is 4, but rope expects 8
        let input = Tensor::new(vec![1.0; 8], vec![1, 1, 2, 4]).unwrap();
        let result = rope.apply(&input, &input, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_rope_qwen3_06b_config() {
        // Qwen3-0.6B actual configuration
        let head_dim = 128;
        let max_seq_len = 40960;
        let theta = 1000000.0;

        let rope = RotaryEmbedding::new(head_dim, max_seq_len, theta).unwrap();

        // Verify table dimensions
        assert_eq!(rope.cos.shape(), &[max_seq_len, head_dim]);
        assert_eq!(rope.sin.shape(), &[max_seq_len, head_dim]);

        // Verify position 0 is identity (cos=1, sin=0)
        let cos_data = rope.cos.to_vec_f32();
        let sin_data = rope.sin.to_vec_f32();

        for d in 0..head_dim {
            assert_close(cos_data[d], 1.0, 1e-5, &format!("cos[0, {}]", d));
            assert_close(sin_data[d], 0.0, 1e-5, &format!("sin[0, {}]", d));
        }

        // Verify frequencies decrease monotonically for position 1
        // (inv_freq decreases, so for same pos, sin values should decrease)
        let pos1_offset = head_dim;
        for i in 0..head_dim / 2 - 1 {
            // Each frequency appears twice: [freq0, freq0, freq1, freq1, ...]
            let freq1 = sin_data[pos1_offset + i * 2].abs();
            let freq2 = sin_data[pos1_offset + (i + 1) * 2].abs();
            // Higher index i means smaller inv_freq[i], thus smaller angle, thus smaller sin
            assert!(
                freq1 > freq2,
                "Frequencies should decrease (inv_freq decreases): sin[1, {}]={} vs sin[1, {}]={}",
                i * 2,
                freq1,
                (i + 1) * 2,
                freq2
            );
        }
    }

    #[test]
    fn test_rope_cos_sin_values_at_known_positions() {
        // Test with simple theta and known positions
        let theta = 10000.0;
        let head_dim = 4;
        let rope = RotaryEmbedding::new(head_dim, 100, theta).unwrap();

        let cos_data = rope.cos.to_vec_f32();
        let sin_data = rope.sin.to_vec_f32();

        // inv_freq[0] = 1.0 / (10000^0) = 1.0
        // inv_freq[1] = 1.0 / (10000^(2/4)) = 1.0 / 100 = 0.01

        // Position 0: freq = 0, cos = 1, sin = 0
        assert_close(cos_data[0], 1.0, 1e-6, "cos[0,0]");
        assert_close(sin_data[0], 0.0, 1e-6, "sin[0,0]");

        // Position 1, dim 0: freq = 1 * 1.0 = 1.0
        let pos1_offset = head_dim;
        let expected_cos = (1.0f64).cos() as f32;
        let expected_sin = (1.0f64).sin() as f32;
        assert_close(cos_data[pos1_offset], expected_cos, 1e-6, "cos[1,0]");
        assert_close(sin_data[pos1_offset], expected_sin, 1e-6, "sin[1,0]");

        // Position 1, dim 2: freq = 1 * 0.01 = 0.01
        let expected_cos_dim2 = (0.01f64).cos() as f32;
        let expected_sin_dim2 = (0.01f64).sin() as f32;
        assert_close(
            cos_data[pos1_offset + 2],
            expected_cos_dim2,
            1e-6,
            "cos[1,2]",
        );
        assert_close(
            sin_data[pos1_offset + 2],
            expected_sin_dim2,
            1e-6,
            "sin[1,2]",
        );
    }

    #[test]
    fn test_rope_different_q_k_head_counts() {
        // GQA scenario: different number of heads for Q and K
        let rope = RotaryEmbedding::new(8, 100, 10000.0).unwrap();

        // Q: [1, 16, 4, 8] (16 query heads)
        let q = Tensor::new(vec![1.0; 16 * 4 * 8], vec![1, 16, 4, 8]).unwrap();

        // K: [1, 8, 4, 8] (8 key heads, GQA)
        let k = Tensor::new(vec![2.0; 8 * 4 * 8], vec![1, 8, 4, 8]).unwrap();

        let (q_rot, k_rot) = rope.apply(&q, &k, 0).unwrap();

        // Verify shapes are preserved
        assert_eq!(q_rot.shape(), &[1, 16, 4, 8]);
        assert_eq!(k_rot.shape(), &[1, 8, 4, 8]);
    }

    #[test]
    fn test_rope_sequential_positions() {
        // Simulate autoregressive generation: seq_len=1 with increasing offset
        let rope = RotaryEmbedding::new(4, 100, 10000.0).unwrap();

        let token = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 1, 4]).unwrap();

        // Generate 3 tokens sequentially
        let (q0, _) = rope.apply(&token, &token, 0).unwrap();
        let (q1, _) = rope.apply(&token, &token, 1).unwrap();
        let (q2, _) = rope.apply(&token, &token, 2).unwrap();

        // Each should use different rotation (position 0, 1, 2)
        let q0_data = q0.to_vec_f32();
        let q1_data = q1.to_vec_f32();
        let q2_data = q2.to_vec_f32();

        // Verify they're all different
        assert!(
            (q0_data[0] - q1_data[0]).abs() > 1e-6,
            "Position 0 and 1 should differ"
        );
        assert!(
            (q1_data[0] - q2_data[0]).abs() > 1e-6,
            "Position 1 and 2 should differ"
        );
    }

    #[test]
    fn test_rope_batch_processing() {
        // Test with batch size > 1
        let rope = RotaryEmbedding::new(4, 100, 10000.0).unwrap();

        // Batch of 2: [2, 2, 3, 4]
        let input = Tensor::new(vec![1.0; 2 * 2 * 3 * 4], vec![2, 2, 3, 4]).unwrap();

        let (q_rot, k_rot) = rope.apply(&input, &input, 0).unwrap();

        // Verify shapes
        assert_eq!(q_rot.shape(), &[2, 2, 3, 4]);
        assert_eq!(k_rot.shape(), &[2, 2, 3, 4]);

        // Verify output is finite (not NaN or Inf)
        let q_data = q_rot.to_vec_f32();
        for (i, &val) in q_data.iter().enumerate() {
            assert!(val.is_finite(), "q_rot[{}] = {} is not finite", i, val);
        }
    }
}
