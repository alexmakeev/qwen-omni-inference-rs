//! SiLU Gated MLP for Qwen3 models.
//!
//! Implements the feed-forward network used in each transformer layer.
//! Uses SiLU (Swish) activation with gating mechanism.
//!
//! # Architecture (Qwen3-0.6B)
//!
//! - Input: [hidden_size] = [1024]
//! - Intermediate: [intermediate_size] = [3072]
//! - Output: [hidden_size] = [1024]
//!
//! # Algorithm
//!
//! ```text
//! MLP(x) = down_proj(silu(gate_proj(x)) * up_proj(x))
//! ```
//!
//! where:
//! - `gate_proj` projects input to intermediate dimension
//! - `silu` is the SiLU activation: `x * sigmoid(x) = x / (1 + exp(-x))`
//! - `up_proj` projects input to intermediate dimension (parallel path)
//! - Element-wise multiplication combines the two paths
//! - `down_proj` projects back to hidden dimension

use crate::attention::Linear;
use crate::error::Result;
use crate::tensor::Tensor;

/// SiLU gated MLP layer.
///
/// Implements the feed-forward network with gating mechanism used in Qwen3 models.
#[derive(Debug, Clone)]
pub struct MLP {
    /// Gate projection [intermediate_size, hidden_size]
    gate_proj: Linear,
    /// Up projection [intermediate_size, hidden_size]
    up_proj: Linear,
    /// Down projection [hidden_size, intermediate_size]
    down_proj: Linear,
}

impl MLP {
    /// Create a new MLP from weight tensors.
    ///
    /// # Arguments
    ///
    /// * `gate_proj` - Gate projection weight [intermediate_size, hidden_size]
    /// * `up_proj` - Up projection weight [intermediate_size, hidden_size]
    /// * `down_proj` - Down projection weight [hidden_size, intermediate_size]
    ///
    /// # Returns
    ///
    /// New MLP instance or error if any weight tensor has invalid shape.
    ///
    /// # Errors
    ///
    /// Returns `ShapeMismatch` if:
    /// - Any weight is not 2D
    /// - gate_proj and up_proj have different shapes (must match)
    /// - down_proj input dimension doesn't match gate_proj/up_proj output dimension
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::mlp::MLP;
    /// use lluda_inference::tensor::Tensor;
    ///
    /// // Create MLP: 4 -> 8 -> 4
    /// let gate_weight = Tensor::new(vec![0.1; 32], vec![8, 4]).unwrap();
    /// let up_weight = Tensor::new(vec![0.2; 32], vec![8, 4]).unwrap();
    /// let down_weight = Tensor::new(vec![0.3; 32], vec![4, 8]).unwrap();
    ///
    /// let mlp = MLP::new(gate_weight, up_weight, down_weight).unwrap();
    /// ```
    pub fn new(gate_proj: Tensor, up_proj: Tensor, down_proj: Tensor) -> Result<Self> {
        // Validate all weights are 2D (Linear::new does this)
        let gate_linear = Linear::new(gate_proj)?;
        let up_linear = Linear::new(up_proj)?;
        let down_linear = Linear::new(down_proj)?;

        // Validate gate_proj and up_proj have same shape
        // Both are [intermediate_size, hidden_size]
        let gate_shape = gate_linear.weight.shape();
        let up_shape = up_linear.weight.shape();

        if gate_shape != up_shape {
            return Err(crate::error::LludaError::ShapeMismatch {
                expected: gate_shape.to_vec(),
                got: up_shape.to_vec(),
            });
        }

        // Validate down_proj input matches gate/up output
        // down_proj is [hidden_size, intermediate_size]
        // gate/up output is intermediate_size (dim 0)
        let intermediate_size = gate_shape[0];
        let down_shape = down_linear.weight.shape();

        if down_shape[1] != intermediate_size {
            return Err(crate::error::LludaError::ShapeMismatch {
                expected: vec![down_shape[0], intermediate_size],
                got: down_shape.to_vec(),
            });
        }

        Ok(MLP {
            gate_proj: gate_linear,
            up_proj: up_linear,
            down_proj: down_linear,
        })
    }

    /// Forward pass through MLP.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape `[..., hidden_size]`
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[..., hidden_size]` (same as input).
    ///
    /// # Errors
    ///
    /// Returns error if shapes are incompatible or operations fail.
    ///
    /// # Algorithm
    ///
    /// 1. Gate path: `gate = gate_proj(x)` → apply SiLU activation
    /// 2. Up path: `up = up_proj(x)` → no activation
    /// 3. Combine: `gate * up` (element-wise multiplication)
    /// 4. Project down: `down_proj(gate * up)`
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::mlp::MLP;
    /// use lluda_inference::tensor::Tensor;
    ///
    /// # let gate_weight = Tensor::new(vec![0.1; 32], vec![8, 4]).unwrap();
    /// # let up_weight = Tensor::new(vec![0.2; 32], vec![8, 4]).unwrap();
    /// # let down_weight = Tensor::new(vec![0.3; 32], vec![4, 8]).unwrap();
    /// # let mlp = MLP::new(gate_weight, up_weight, down_weight).unwrap();
    /// #
    /// // Input: batch_size=2, seq_len=3, hidden_size=4
    /// let x = Tensor::new(vec![1.0; 24], vec![2, 3, 4]).unwrap();
    /// let output = mlp.forward(&x).unwrap();
    ///
    /// assert_eq!(output.shape(), &[2, 3, 4]);
    /// ```
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 1. Gate path with SiLU activation
        let gate = self.gate_proj.forward(x)?;
        let gate_activated = gate.silu()?;

        // 2. Up path (no activation)
        let up = self.up_proj.forward(x)?;

        // 3. Element-wise multiplication (gating)
        let gated = gate_activated.mul(&up)?;

        // 4. Down projection
        self.down_proj.forward(&gated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bf16::BF16;
    use crate::tensor::DType;

    #[test]
    fn mlp_constructor_validates_2d_weights() {
        // gate_proj must be 2D
        let gate = Tensor::new(vec![1.0; 8], vec![8]).unwrap(); // 1D - invalid
        let up = Tensor::new(vec![1.0; 32], vec![8, 4]).unwrap();
        let down = Tensor::new(vec![1.0; 32], vec![4, 8]).unwrap();

        let result = MLP::new(gate, up, down);
        assert!(result.is_err(), "Should reject 1D gate_proj");
    }

    #[test]
    fn mlp_constructor_validates_matching_shapes() {
        // gate_proj and up_proj must have same shape
        let gate = Tensor::new(vec![1.0; 32], vec![8, 4]).unwrap(); // [8, 4]
        let up = Tensor::new(vec![1.0; 40], vec![10, 4]).unwrap(); // [10, 4] - mismatch
        let down = Tensor::new(vec![1.0; 32], vec![4, 8]).unwrap();

        let result = MLP::new(gate, up, down);
        assert!(
            result.is_err(),
            "Should reject mismatched gate_proj and up_proj shapes"
        );
    }

    #[test]
    fn mlp_constructor_validates_down_proj_compatibility() {
        // down_proj input must match gate/up output
        let gate = Tensor::new(vec![1.0; 32], vec![8, 4]).unwrap(); // Output: 8
        let up = Tensor::new(vec![1.0; 32], vec![8, 4]).unwrap(); // Output: 8
        let down = Tensor::new(vec![1.0; 24], vec![4, 6]).unwrap(); // Input: 6 - mismatch!

        let result = MLP::new(gate, up, down);
        assert!(
            result.is_err(),
            "Should reject incompatible down_proj dimensions"
        );
    }

    #[test]
    fn mlp_constructor_accepts_valid_weights() {
        let gate = Tensor::new(vec![1.0; 32], vec![8, 4]).unwrap();
        let up = Tensor::new(vec![1.0; 32], vec![8, 4]).unwrap();
        let down = Tensor::new(vec![1.0; 32], vec![4, 8]).unwrap();

        let mlp = MLP::new(gate, up, down);
        assert!(mlp.is_ok(), "Should accept valid weight tensors");
    }

    #[test]
    fn test_mlp_shape_invariant() {
        // Test that output shape matches input shape
        let gate = Tensor::new(vec![0.1; 32], vec![8, 4]).unwrap();
        let up = Tensor::new(vec![0.1; 32], vec![8, 4]).unwrap();
        let down = Tensor::new(vec![0.1; 32], vec![4, 8]).unwrap();
        let mlp = MLP::new(gate, up, down).unwrap();

        // Test with 1D batch
        let x = Tensor::new(vec![0.5; 4], vec![1, 4]).unwrap();
        let output = mlp.forward(&x).unwrap();
        assert_eq!(output.shape(), &[1, 4]);

        // Test with 2D batch [B, L, hidden_size]
        let x = Tensor::new(vec![0.5; 2 * 3 * 4], vec![2, 3, 4]).unwrap();
        let output = mlp.forward(&x).unwrap();
        assert_eq!(output.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_mlp_silu_applied_to_gate_only() {
        // Verify that SiLU is applied to gate_proj output, not up_proj
        // Use identity-like weights for simplicity
        // gate_proj: [[1, 0], [0, 1]]
        let gate = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        // up_proj: [[1, 0], [0, 1]] (identity)
        let up = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        // down_proj: [[1, 0], [0, 1]] (identity)
        let down = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();

        let mlp = MLP::new(gate, up, down).unwrap();

        // Input: [1, 1]
        let x = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
        let output = mlp.forward(&x).unwrap();

        // Expected:
        // gate = [1, 1]
        // gate_activated = silu([1, 1]) ≈ [0.7311, 0.7311] (silu(1) ≈ 0.7311)
        // up = [1, 1]
        // gated = [0.7311, 0.7311] * [1, 1] = [0.7311, 0.7311]
        // output = [0.7311, 0.7311]
        let data = output.to_vec_f32();
        assert_eq!(data.len(), 2);

        // silu(1) = 1 / (1 + exp(-1)) ≈ 0.7311
        let expected_silu_1 = 1.0 / (1.0 + (-1.0f32).exp());
        for &val in &data {
            assert!(
                (val - expected_silu_1).abs() < 1e-4,
                "Expected ~{}, got {}",
                expected_silu_1,
                val
            );
        }
    }

    #[test]
    fn test_mlp_output_is_finite() {
        // Ensure output doesn't contain NaN or Inf
        let gate = Tensor::new(vec![0.1; 32], vec![8, 4]).unwrap();
        let up = Tensor::new(vec![0.1; 32], vec![8, 4]).unwrap();
        let down = Tensor::new(vec![0.1; 32], vec![4, 8]).unwrap();

        let mlp = MLP::new(gate, up, down).unwrap();

        let x = Tensor::new(vec![0.5; 2 * 4], vec![2, 4]).unwrap();
        let output = mlp.forward(&x).unwrap();

        let data = output.to_vec_f32();
        assert!(
            data.iter().all(|&x| x.is_finite()),
            "Output contains non-finite values"
        );
    }

    #[test]
    fn test_mlp_element_wise_gating() {
        // Verify that gating is element-wise multiplication, not addition
        // gate_proj: multiply by 2: [[2, 0], [0, 2]]
        let gate = Tensor::new(vec![2.0, 0.0, 0.0, 2.0], vec![2, 2]).unwrap();
        // up_proj: multiply by 3: [[3, 0], [0, 3]]
        let up = Tensor::new(vec![3.0, 0.0, 0.0, 3.0], vec![2, 2]).unwrap();
        // down_proj: identity
        let down = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();

        let mlp = MLP::new(gate, up, down).unwrap();

        // Input: [1, 1]
        let x = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
        let output = mlp.forward(&x).unwrap();

        // gate = [2, 2]
        // gate_activated = silu([2, 2])
        // up = [3, 3]
        // gated = silu([2, 2]) * [3, 3]
        // If it were addition: silu([2, 2]) + [3, 3], result would be different

        let data = output.to_vec_f32();
        let silu_2 = 2.0 / (1.0 + (-2.0f32).exp());
        let expected = silu_2 * 3.0;

        for &val in &data {
            // Should be multiplication result, not addition
            assert!(
                (val - expected).abs() < 1e-4,
                "Expected multiplication result ~{}, got {}",
                expected,
                val
            );

            // Verify it's NOT addition
            let wrong_expected = silu_2 + 3.0;
            assert!(
                (val - wrong_expected).abs() > 0.1,
                "Output should NOT be addition result"
            );
        }
    }

    #[test]
    fn test_mlp_batched_3d_input() {
        // Test with realistic shape [B=2, L=3, hidden_size=4]
        let gate = Tensor::new(vec![0.1; 32], vec![8, 4]).unwrap();
        let up = Tensor::new(vec![0.1; 32], vec![8, 4]).unwrap();
        let down = Tensor::new(vec![0.1; 32], vec![4, 8]).unwrap();

        let mlp = MLP::new(gate, up, down).unwrap();

        let x = Tensor::new(vec![0.5; 2 * 3 * 4], vec![2, 3, 4]).unwrap();
        let output = mlp.forward(&x).unwrap();

        assert_eq!(output.shape(), &[2, 3, 4]);

        let data = output.to_vec_f32();
        assert!(data.iter().all(|&x| x.is_finite()));
        assert_eq!(data.len(), 2 * 3 * 4);
    }

    #[test]
    fn test_mlp_zero_input() {
        // Test with zero input
        let gate = Tensor::new(vec![0.1; 32], vec![8, 4]).unwrap();
        let up = Tensor::new(vec![0.1; 32], vec![8, 4]).unwrap();
        let down = Tensor::new(vec![0.1; 32], vec![4, 8]).unwrap();

        let mlp = MLP::new(gate, up, down).unwrap();

        let x = Tensor::new(vec![0.0; 4], vec![1, 4]).unwrap();
        let output = mlp.forward(&x).unwrap();

        assert_eq!(output.shape(), &[1, 4]);

        let data = output.to_vec_f32();
        // silu(0) = 0, so gate_activated will be all zeros
        // 0 * up_proj(0) = 0
        // Result should be all zeros or very small values
        assert!(data.iter().all(|&x| x.abs() < 1e-3));
    }

    #[test]
    fn mlp_forward_rejects_wrong_input_dim() {
        let gate = Tensor::new(vec![1.0; 32], vec![8, 4]).unwrap();
        let up = Tensor::new(vec![1.0; 32], vec![8, 4]).unwrap();
        let down = Tensor::new(vec![1.0; 32], vec![4, 8]).unwrap();
        let mlp = MLP::new(gate, up, down).unwrap();

        // Wrong input dimension (should be 4, not 5)
        let x = Tensor::new(vec![1.0; 10], vec![2, 5]).unwrap();
        let result = mlp.forward(&x);

        assert!(result.is_err(), "Should reject wrong input dimension");
    }

    #[test]
    fn mlp_forward_works_with_bf16_weights() {
        // Create MLP with BF16 weights
        let gate_bf16: Vec<BF16> = [0.1f32; 32].iter().map(|&x| BF16::from(x)).collect();
        let up_bf16: Vec<BF16> = [0.2f32; 32].iter().map(|&x| BF16::from(x)).collect();
        let down_bf16: Vec<BF16> = [0.3f32; 32].iter().map(|&x| BF16::from(x)).collect();

        let gate = Tensor::from_bf16(gate_bf16, vec![8, 4]).unwrap();
        let up = Tensor::from_bf16(up_bf16, vec![8, 4]).unwrap();
        let down = Tensor::from_bf16(down_bf16, vec![4, 8]).unwrap();

        let mlp = MLP::new(gate, up, down).unwrap();

        let x = Tensor::new(vec![1.0; 12], vec![3, 4]).unwrap();
        let output = mlp.forward(&x).unwrap();

        // Should work with BF16 weights (auto-convert to F32 for compute)
        assert_eq!(output.shape(), &[3, 4]);
        assert!(output.to_vec_f32().iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn mlp_forward_batch_dimension() {
        // Test with various batch shapes: [B, L, D]
        let gate = Tensor::new(vec![0.1; 32], vec![8, 4]).unwrap();
        let up = Tensor::new(vec![0.2; 32], vec![8, 4]).unwrap();
        let down = Tensor::new(vec![0.3; 32], vec![4, 8]).unwrap();
        let mlp = MLP::new(gate, up, down).unwrap();

        // 1D: [4]
        let x1 = Tensor::new(vec![1.0; 4], vec![4]).unwrap();
        let out1 = mlp.forward(&x1).unwrap();
        assert_eq!(out1.shape(), &[4]);

        // 2D: [3, 4]
        let x2 = Tensor::new(vec![1.0; 12], vec![3, 4]).unwrap();
        let out2 = mlp.forward(&x2).unwrap();
        assert_eq!(out2.shape(), &[3, 4]);

        // 3D: [2, 3, 4] (typical: batch, seq_len, hidden)
        let x3 = Tensor::new(vec![1.0; 24], vec![2, 3, 4]).unwrap();
        let out3 = mlp.forward(&x3).unwrap();
        assert_eq!(out3.shape(), &[2, 3, 4]);

        // 4D: [2, 2, 3, 4]
        let x4 = Tensor::new(vec![1.0; 48], vec![2, 2, 3, 4]).unwrap();
        let out4 = mlp.forward(&x4).unwrap();
        assert_eq!(out4.shape(), &[2, 2, 3, 4]);
    }

    #[test]
    fn mlp_forward_no_nan_or_inf() {
        let gate = Tensor::new(vec![0.5; 32], vec![8, 4]).unwrap();
        let up = Tensor::new(vec![0.5; 32], vec![8, 4]).unwrap();
        let down = Tensor::new(vec![0.5; 32], vec![4, 8]).unwrap();
        let mlp = MLP::new(gate, up, down).unwrap();

        // Random-ish input values
        let x_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let x = Tensor::new(x_data, vec![3, 4]).unwrap();

        let output = mlp.forward(&x).unwrap();
        let out_data = output.to_vec_f32();

        assert!(
            out_data.iter().all(|&v| !v.is_nan()),
            "Output should not contain NaN"
        );
        assert!(
            out_data.iter().all(|&v| !v.is_infinite()),
            "Output should not contain Inf"
        );
    }

    #[test]
    fn mlp_qwen3_06b_dimensions() {
        // Test with actual Qwen3-0.6B dimensions
        // hidden_size = 1024, intermediate_size = 3072
        let gate = Tensor::zeros(&[3072, 1024], DType::F32).unwrap();
        let up = Tensor::zeros(&[3072, 1024], DType::F32).unwrap();
        let down = Tensor::zeros(&[1024, 3072], DType::F32).unwrap();

        let mlp = MLP::new(gate, up, down).unwrap();

        // Input: [batch=2, seq_len=10, hidden_size=1024]
        let x = Tensor::zeros(&[2, 10, 1024], DType::F32).unwrap();
        let output = mlp.forward(&x).unwrap();

        assert_eq!(
            output.shape(),
            &[2, 10, 1024],
            "Output should preserve batch and sequence dimensions"
        );
    }

    #[test]
    fn test_mlp_negative_inputs_silu_behavior() {
        // Test SiLU activation behavior with negative values
        // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        // For negative x, SiLU is small but non-zero (approaches 0 as x -> -inf)

        // Use identity-like weights for predictability
        let gate = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let up = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let down = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();

        let mlp = MLP::new(gate, up, down).unwrap();

        // Test values covering negative, zero, and positive range
        let test_values = vec![-5.0, -1.0, 0.0, 1.0, 5.0];

        for &val in &test_values {
            let x = Tensor::new(vec![val, val], vec![1, 2]).unwrap();
            let output = mlp.forward(&x).unwrap();
            let output_data = output.to_vec_f32();

            // Verify output is finite
            assert!(
                output_data.iter().all(|&x| x.is_finite()),
                "Output should be finite for input {}",
                val
            );

            // Compute expected SiLU value
            let expected_silu = val / (1.0 + (-val).exp());

            // Since we use identity weights: output = silu(val) * val
            let expected_output = expected_silu * val;

            // For negative values, SiLU output is also negative but magnitude is reduced
            // SiLU(-5) ≈ -0.0337, so output = -0.0337 * -5 ≈ 0.1683 (positive!)
            // The gating multiplies silu(gate) * up, so both negative -> positive result
            if val < 0.0 {
                // SiLU dampens the magnitude but the sign depends on the multiplication
                // For negative input with identity weights: silu(neg) * neg = small_neg * neg = small_pos
                assert!(
                    output_data[0].abs() < val.abs(),
                    "SiLU should dampen magnitude: input={}, output={}",
                    val,
                    output_data[0]
                );
            }

            // Verify approximate match with expected value
            assert!(
                (output_data[0] - expected_output).abs() < 1e-4,
                "Expected ~{} for input {}, got {}",
                expected_output,
                val,
                output_data[0]
            );
        }
    }
}
