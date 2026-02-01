//! RMS Layer Normalization.
//!
//! Root Mean Square Layer Normalization normalizes activations by their RMS value
//! and applies a learned scale. This is used throughout Qwen3 models for both
//! layer normalization and per-head query/key normalization.
//!
//! # Algorithm
//!
//! ```text
//! RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * weight
//! ```
//!
//! where:
//! - `mean(x^2)` is computed over the last dimension
//! - `eps` prevents division by zero
//! - `weight` is a learned per-dimension scale parameter
//!
//! # Usage in Qwen3-0.6B
//!
//! 1. **Layer normalization**: Applied to hidden states before attention and MLP.
//!    - Weight shape: `[1024]` (hidden_size)
//!    - Input shape: `[B, L, 1024]`
//!
//! 2. **Per-head Q/K normalization**: Applied to query and key projections after
//!    splitting into heads.
//!    - Weight shape: `[128]` (head_dim)
//!    - Input shape: `[B*H, L, 128]` (flattened batch and heads)
//!
//! # Reference
//!
//! Zhang & Sennrich (2019): "Root Mean Square Layer Normalization"
//! https://arxiv.org/abs/1910.07467

use crate::error::{LludaError, Result};
use crate::tensor::Tensor;

/// RMS Layer Normalization.
///
/// Normalizes input by root mean square along the last dimension,
/// then applies learned scale.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    /// Learned scale parameter, shape `[dim]`.
    weight: Tensor,
    /// Epsilon for numerical stability (prevents division by zero).
    eps: f64,
}

impl RmsNorm {
    /// Create a new RMSNorm layer.
    ///
    /// # Arguments
    ///
    /// * `weight` - Learned scale parameter tensor of shape `[dim]`
    /// * `eps` - Epsilon for numerical stability (typically 1e-6)
    ///
    /// # Returns
    ///
    /// New RMSNorm instance.
    ///
    /// # Errors
    ///
    /// Returns error if weight is not a 1D tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::rms_norm::RmsNorm;
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let weight = Tensor::new(vec![1.0; 1024], vec![1024]).unwrap();
    /// let norm = RmsNorm::new(weight, 1e-6).unwrap();
    /// ```
    pub fn new(weight: Tensor, eps: f64) -> Result<Self> {
        if weight.ndim() != 1 {
            return Err(LludaError::ShapeMismatch {
                expected: vec![0], // Expect 1D
                got: weight.shape().to_vec(),
            });
        }
        Ok(RmsNorm { weight, eps })
    }

    /// Apply RMS normalization to input tensor.
    ///
    /// # Algorithm
    ///
    /// For input `x` with shape `[..., dim]`:
    /// 1. Compute RMS: `rms = sqrt(mean(x^2) + eps)`
    /// 2. Normalize: `x_norm = x / rms`
    /// 3. Scale: `output = x_norm * weight`
    ///
    /// The normalization is applied independently over the last dimension.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape `[..., dim]` where `dim` matches weight dimension
    ///
    /// # Returns
    ///
    /// Normalized tensor of same shape as input.
    ///
    /// # Errors
    ///
    /// Returns `ShapeMismatch` if:
    /// - Input has fewer than 1 dimension
    /// - Last dimension of input doesn't match weight dimension
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::rms_norm::RmsNorm;
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let weight = Tensor::new(vec![1.0; 4], vec![4]).unwrap();
    /// let norm = RmsNorm::new(weight, 1e-6).unwrap();
    ///
    /// let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
    /// let output = norm.forward(&x).unwrap();
    /// assert_eq!(output.shape(), &[1, 4]);
    /// ```
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Validate input shape
        if x.ndim() < 1 {
            return Err(LludaError::ShapeMismatch {
                expected: vec![1],
                got: vec![x.ndim()],
            });
        }

        let input_shape = x.shape();
        let last_dim = input_shape[input_shape.len() - 1];
        let weight_dim = self.weight.shape()[0];

        if last_dim != weight_dim {
            return Err(LludaError::ShapeMismatch {
                expected: vec![weight_dim],
                got: vec![last_dim],
            });
        }

        // Get input data as F32 for compute
        let data = x.to_vec_f32();
        let weight_data = self.weight.to_vec_f32();

        // Compute total number of elements and number of vectors to normalize
        let numel = data.len();
        let num_vectors = numel / last_dim;

        let mut result = vec![0.0f32; numel];

        // Process each vector independently
        for vec_idx in 0..num_vectors {
            let start = vec_idx * last_dim;
            let end = start + last_dim;
            let vec_data = &data[start..end];

            // Compute inverse RMS: 1.0 / sqrt(mean(x^2) + eps)
            let sum_sq: f32 = vec_data.iter().map(|&x| x * x).sum();
            let mean_sq = sum_sq / last_dim as f32;
            let inv_rms = 1.0 / (mean_sq + self.eps as f32).sqrt();

            // Normalize and scale: x * inv_rms * weight
            for i in 0..last_dim {
                result[start + i] = vec_data[i] * inv_rms * weight_data[i];
            }
        }

        Tensor::new(result, input_shape.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32, tol: f32, msg: &str) {
        assert!(
            (a - b).abs() < tol,
            "{}: {} vs {}, diff = {}",
            msg,
            a,
            b,
            (a - b).abs()
        );
    }

    #[test]
    fn test_rms_norm_uniform_input() {
        // Input: [1, 1, 1], weight: [1, 1, 1], eps: 1e-6
        // Expected: normalized to unit RMS, then scaled by 1
        let weight = Tensor::new(vec![1.0, 1.0, 1.0], vec![3]).unwrap();
        let norm = RmsNorm::new(weight, 1e-6).unwrap();

        let x = Tensor::new(vec![1.0, 1.0, 1.0], vec![1, 3]).unwrap();
        let output = norm.forward(&x).unwrap();

        assert_eq!(output.shape(), &[1, 3]);
        let data = output.to_vec_f32();

        // RMS of [1, 1, 1] is 1.0
        // Normalized: [1/1, 1/1, 1/1] = [1, 1, 1]
        // Scaled by weight [1, 1, 1]: [1, 1, 1]
        assert_close(data[0], 1.0, 1e-5, "element 0");
        assert_close(data[1], 1.0, 1e-5, "element 1");
        assert_close(data[2], 1.0, 1e-5, "element 2");
    }

    #[test]
    fn test_rms_norm_non_uniform_input() {
        // Input: [2, 0, 0], weight: [1, 1, 1], eps: 1e-6
        let weight = Tensor::new(vec![1.0, 1.0, 1.0], vec![3]).unwrap();
        let norm = RmsNorm::new(weight, 1e-6).unwrap();

        let x = Tensor::new(vec![2.0, 0.0, 0.0], vec![1, 3]).unwrap();
        let output = norm.forward(&x).unwrap();

        assert_eq!(output.shape(), &[1, 3]);
        let data = output.to_vec_f32();

        // RMS = sqrt((4 + 0 + 0) / 3) = sqrt(4/3) ≈ 1.1547
        // Normalized: [2/1.1547, 0, 0] ≈ [1.732, 0, 0]
        let rms = (4.0f32 / 3.0).sqrt();
        assert_close(data[0], 2.0 / rms, 1e-5, "element 0");
        assert_close(data[1], 0.0, 1e-5, "element 1");
        assert_close(data[2], 0.0, 1e-5, "element 2");
    }

    #[test]
    fn test_rms_norm_with_weight() {
        // Test that weight scaling is applied correctly
        let weight = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]).unwrap();
        let norm = RmsNorm::new(weight, 1e-6).unwrap();

        let x = Tensor::new(vec![1.0, 1.0, 1.0], vec![1, 3]).unwrap();
        let output = norm.forward(&x).unwrap();

        assert_eq!(output.shape(), &[1, 3]);
        let data = output.to_vec_f32();

        // RMS of [1, 1, 1] is 1.0
        // Normalized: [1, 1, 1]
        // Scaled by weight: [1*2, 1*3, 1*4] = [2, 3, 4]
        assert_close(data[0], 2.0, 1e-5, "element 0");
        assert_close(data[1], 3.0, 1e-5, "element 1");
        assert_close(data[2], 4.0, 1e-5, "element 2");
    }

    #[test]
    fn test_rms_norm_batched() {
        // Test with batch dimension [2, 3]
        let weight = Tensor::new(vec![1.0, 1.0, 1.0], vec![3]).unwrap();
        let norm = RmsNorm::new(weight, 1e-6).unwrap();

        // Two vectors: [1, 2, 3] and [4, 5, 6]
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        )
        .unwrap();
        let output = norm.forward(&x).unwrap();

        assert_eq!(output.shape(), &[2, 3]);
        let data = output.to_vec_f32();

        // First vector [1, 2, 3]:
        // RMS = sqrt((1 + 4 + 9) / 3) = sqrt(14/3) ≈ 2.1602
        let rms1 = (14.0f32 / 3.0).sqrt();
        assert_close(data[0], 1.0 / rms1, 1e-5, "batch 0, element 0");
        assert_close(data[1], 2.0 / rms1, 1e-5, "batch 0, element 1");
        assert_close(data[2], 3.0 / rms1, 1e-5, "batch 0, element 2");

        // Second vector [4, 5, 6]:
        // RMS = sqrt((16 + 25 + 36) / 3) = sqrt(77/3) ≈ 5.0664
        let rms2 = (77.0f32 / 3.0).sqrt();
        assert_close(data[3], 4.0 / rms2, 1e-5, "batch 1, element 0");
        assert_close(data[4], 5.0 / rms2, 1e-5, "batch 1, element 1");
        assert_close(data[5], 6.0 / rms2, 1e-5, "batch 1, element 2");
    }

    #[test]
    fn test_rms_norm_3d_input() {
        // Test with 3D input [B, L, D] = [1, 2, 4]
        let weight = Tensor::new(vec![1.0; 4], vec![4]).unwrap();
        let norm = RmsNorm::new(weight, 1e-6).unwrap();

        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![1, 2, 4],
        )
        .unwrap();
        let output = norm.forward(&x).unwrap();

        assert_eq!(output.shape(), &[1, 2, 4]);
        // Verify normalization is applied per last dimension
        let data = output.to_vec_f32();

        // First vector [1, 2, 3, 4]:
        // RMS = sqrt((1 + 4 + 9 + 16) / 4) = sqrt(30/4) = sqrt(7.5) ≈ 2.7386
        let rms1 = (30.0f32 / 4.0).sqrt();
        assert_close(data[0], 1.0 / rms1, 1e-5, "vec 0, elem 0");
        assert_close(data[1], 2.0 / rms1, 1e-5, "vec 0, elem 1");
        assert_close(data[2], 3.0 / rms1, 1e-5, "vec 0, elem 2");
        assert_close(data[3], 4.0 / rms1, 1e-5, "vec 0, elem 3");

        // Second vector [5, 6, 7, 8]:
        // RMS = sqrt((25 + 36 + 49 + 64) / 4) = sqrt(174/4) = sqrt(43.5) ≈ 6.5955
        let rms2 = (174.0f32 / 4.0).sqrt();
        assert_close(data[4], 5.0 / rms2, 1e-5, "vec 1, elem 0");
        assert_close(data[5], 6.0 / rms2, 1e-5, "vec 1, elem 1");
        assert_close(data[6], 7.0 / rms2, 1e-5, "vec 1, elem 2");
        assert_close(data[7], 8.0 / rms2, 1e-5, "vec 1, elem 3");
    }

    #[test]
    fn test_rms_norm_eps_prevents_zero_division() {
        // Test that eps prevents division by zero for zero input
        let weight = Tensor::new(vec![1.0, 1.0, 1.0], vec![3]).unwrap();
        let norm = RmsNorm::new(weight, 1e-6).unwrap();

        let x = Tensor::new(vec![0.0, 0.0, 0.0], vec![1, 3]).unwrap();
        let output = norm.forward(&x).unwrap();

        assert_eq!(output.shape(), &[1, 3]);
        let data = output.to_vec_f32();

        // RMS = sqrt(0 + eps) = sqrt(1e-6) ≈ 1e-3
        // Normalized: [0/1e-3, 0/1e-3, 0/1e-3] = [0, 0, 0]
        assert_close(data[0], 0.0, 1e-5, "element 0");
        assert_close(data[1], 0.0, 1e-5, "element 1");
        assert_close(data[2], 0.0, 1e-5, "element 2");
    }

    #[test]
    fn test_rms_norm_shape_mismatch() {
        // Test error when last dimension doesn't match weight dimension
        let weight = Tensor::new(vec![1.0, 1.0, 1.0], vec![3]).unwrap();
        let norm = RmsNorm::new(weight, 1e-6).unwrap();

        // Input has last dim = 4, but weight has dim = 3
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let result = norm.forward(&x);

        assert!(result.is_err());
        match result.unwrap_err() {
            LludaError::ShapeMismatch { expected, got } => {
                assert_eq!(expected, vec![3]);
                assert_eq!(got, vec![4]);
            }
            other => panic!("Expected ShapeMismatch, got: {:?}", other),
        }
    }

    #[test]
    fn test_rms_norm_empty_dimension_error() {
        // Test error when input has 0 dimensions
        let weight = Tensor::new(vec![1.0], vec![1]).unwrap();
        let norm = RmsNorm::new(weight, 1e-6).unwrap();

        // Scalar (0D tensor) should error
        let x = Tensor::new(vec![1.0], vec![]).unwrap();
        let result = norm.forward(&x);

        assert!(result.is_err());
        match result.unwrap_err() {
            LludaError::ShapeMismatch { .. } => (),
            other => panic!("Expected ShapeMismatch, got: {:?}", other),
        }
    }

    #[test]
    fn test_rms_norm_1d_input() {
        // Test with 1D input (edge case: single vector)
        let weight = Tensor::new(vec![2.0, 3.0], vec![2]).unwrap();
        let norm = RmsNorm::new(weight, 1e-6).unwrap();

        let x = Tensor::new(vec![1.0, 1.0], vec![2]).unwrap();
        let output = norm.forward(&x).unwrap();

        assert_eq!(output.shape(), &[2]);
        let data = output.to_vec_f32();

        // RMS of [1, 1] is 1.0
        // Normalized: [1, 1]
        // Scaled by weight [2, 3]: [2, 3]
        assert_close(data[0], 2.0, 1e-5, "element 0");
        assert_close(data[1], 3.0, 1e-5, "element 1");
    }

    #[test]
    fn test_rms_norm_large_values() {
        // Test numerical stability with large values
        let weight = Tensor::new(vec![1.0; 3], vec![3]).unwrap();
        let norm = RmsNorm::new(weight, 1e-6).unwrap();

        let x = Tensor::new(vec![1000.0, 2000.0, 3000.0], vec![1, 3]).unwrap();
        let output = norm.forward(&x).unwrap();

        assert_eq!(output.shape(), &[1, 3]);
        let data = output.to_vec_f32();

        // RMS = sqrt((1e6 + 4e6 + 9e6) / 3) = sqrt(14e6/3) ≈ 2160.25
        let rms = ((1000.0f32 * 1000.0 + 2000.0 * 2000.0 + 3000.0 * 3000.0) / 3.0).sqrt();
        assert_close(data[0], 1000.0 / rms, 1e-3, "element 0");
        assert_close(data[1], 2000.0 / rms, 1e-3, "element 1");
        assert_close(data[2], 3000.0 / rms, 1e-3, "element 2");

        // Verify output is finite (no overflow/NaN)
        assert!(data.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_rms_norm_negative_values() {
        // Test that negative values are handled correctly
        let weight = Tensor::new(vec![1.0; 3], vec![3]).unwrap();
        let norm = RmsNorm::new(weight, 1e-6).unwrap();

        let x = Tensor::new(vec![-1.0, 2.0, -3.0], vec![1, 3]).unwrap();
        let output = norm.forward(&x).unwrap();

        assert_eq!(output.shape(), &[1, 3]);
        let data = output.to_vec_f32();

        // RMS = sqrt((1 + 4 + 9) / 3) = sqrt(14/3) ≈ 2.1602
        let rms = (14.0f32 / 3.0).sqrt();
        assert_close(data[0], -1.0 / rms, 1e-5, "element 0");
        assert_close(data[1], 2.0 / rms, 1e-5, "element 1");
        assert_close(data[2], -3.0 / rms, 1e-5, "element 2");
    }

    #[test]
    fn test_rms_norm_qwen3_layer_norm_shape() {
        // Test with shape typical for Qwen3-0.6B layer normalization
        // Weight: [1024], Input: [B=1, L=4, hidden_size=1024]
        let weight = Tensor::new(vec![1.0; 1024], vec![1024]).unwrap();
        let norm = RmsNorm::new(weight, 1e-6).unwrap();

        // Create random-like input (not all ones to test actual normalization)
        let mut input_data = vec![0.0f32; 4 * 1024];
        for (i, val) in input_data.iter_mut().enumerate() {
            *val = (i % 100) as f32 / 100.0; // Values in [0, 0.99]
        }
        let x = Tensor::new(input_data, vec![1, 4, 1024]).unwrap();

        let output = norm.forward(&x).unwrap();
        assert_eq!(output.shape(), &[1, 4, 1024]);

        // Verify output is finite and within reasonable range
        let data = output.to_vec_f32();
        assert!(data.iter().all(|&x| x.is_finite()));
        assert!(data.iter().all(|&x| x.abs() < 100.0)); // Normalized values should be bounded
    }

    #[test]
    fn test_rms_norm_qwen3_head_norm_shape() {
        // Test with shape typical for Qwen3-0.6B per-head Q/K normalization
        // Weight: [128], Input: [B*H=1*16, L=4, head_dim=128]
        let weight = Tensor::new(vec![1.0; 128], vec![128]).unwrap();
        let norm = RmsNorm::new(weight, 1e-6).unwrap();

        let mut input_data = vec![0.0f32; 16 * 4 * 128];
        for (i, val) in input_data.iter_mut().enumerate() {
            *val = (i % 50) as f32 / 50.0;
        }
        let x = Tensor::new(input_data, vec![16, 4, 128]).unwrap();

        let output = norm.forward(&x).unwrap();
        assert_eq!(output.shape(), &[16, 4, 128]);

        // Verify output is finite
        let data = output.to_vec_f32();
        assert!(data.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_rms_norm_non_1d_weight_fails() {
        // Weight must be 1D - should fail at construction time
        let weight_2d = Tensor::new(vec![1.0; 12], vec![3, 4]).unwrap();
        let result = RmsNorm::new(weight_2d, 1e-6);
        assert!(result.is_err());
        match result.unwrap_err() {
            LludaError::ShapeMismatch { expected, got } => {
                assert_eq!(expected, vec![0]);
                assert_eq!(got, vec![3, 4]);
            }
            other => panic!("Expected ShapeMismatch, got: {:?}", other),
        }
    }

    #[test]
    fn test_rms_norm_0d_weight_fails() {
        // 0D (scalar) weight should fail
        let weight_0d = Tensor::new(vec![1.0], vec![]).unwrap();
        let result = RmsNorm::new(weight_0d, 1e-6);
        assert!(result.is_err());
    }
}
