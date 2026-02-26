//! Layer Normalization.
//!
//! Standard Layer Normalization normalizes activations by subtracting the mean
//! and dividing by the standard deviation, then applies learned scale and bias.
//! This is used in audio encoders (e.g. Whisper-style) and other transformer
//! variants that require full mean-variance normalization.
//!
//! # Algorithm
//!
//! ```text
//! LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
//! ```
//!
//! where:
//! - `mean(x)` and `var(x)` are computed over the last dimension
//! - `eps` prevents division by zero
//! - `weight` is a learned per-dimension scale parameter
//! - `bias` is a learned per-dimension shift parameter
//!
//! # Usage
//!
//! Used in audio encoders where full mean-variance normalization is required,
//! unlike RMSNorm which only normalizes by RMS without centering.
//!
//! - Weight shape: `[hidden_size]`
//! - Bias shape: `[hidden_size]`
//! - Input shape: `[..., hidden_size]`
//!
//! # Reference
//!
//! Ba et al. (2016): "Layer Normalization"
//! https://arxiv.org/abs/1607.06450

#![warn(missing_docs)]

use crate::bf16::BF16;
use crate::error::{LludaError, Result};
use crate::tensor::{DType, Tensor};

/// Standard Layer Normalization.
///
/// Normalizes input by mean and variance along the last dimension,
/// then applies learned scale and bias.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    /// Learned scale parameter, shape `[dim]`.
    weight: Tensor,
    /// Learned bias parameter, shape `[dim]`.
    bias: Tensor,
    /// Epsilon for numerical stability (prevents division by zero).
    eps: f64,
}

impl LayerNorm {
    /// Create a new LayerNorm layer.
    ///
    /// # Arguments
    ///
    /// * `weight` - Learned scale parameter tensor of shape `[dim]`
    /// * `bias` - Learned bias parameter tensor of shape `[dim]`
    /// * `eps` - Epsilon for numerical stability (typically 1e-5)
    ///
    /// # Returns
    ///
    /// New LayerNorm instance.
    ///
    /// # Errors
    ///
    /// Returns `ShapeMismatch` if:
    /// - Weight is not a 1D tensor
    /// - Bias is not a 1D tensor
    /// - Weight and bias have different sizes
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::layer_norm::LayerNorm;
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let weight = Tensor::new(vec![1.0; 512], vec![512]).unwrap();
    /// let bias = Tensor::new(vec![0.0; 512], vec![512]).unwrap();
    /// let norm = LayerNorm::new(weight, bias, 1e-5).unwrap();
    /// ```
    pub fn new(weight: Tensor, bias: Tensor, eps: f64) -> Result<Self> {
        if weight.ndim() != 1 {
            return Err(LludaError::ShapeMismatch {
                expected: vec![0], // Expect 1D
                got: weight.shape().to_vec(),
            });
        }
        if bias.ndim() != 1 {
            return Err(LludaError::ShapeMismatch {
                expected: vec![0], // Expect 1D
                got: bias.shape().to_vec(),
            });
        }
        let weight_dim = weight.shape()[0];
        let bias_dim = bias.shape()[0];
        if weight_dim != bias_dim {
            return Err(LludaError::ShapeMismatch {
                expected: vec![weight_dim],
                got: vec![bias_dim],
            });
        }
        Ok(LayerNorm { weight, bias, eps })
    }

    /// Apply Layer Normalization to input tensor.
    ///
    /// # Algorithm
    ///
    /// For input `x` with shape `[..., dim]`:
    /// 1. Compute mean: `mean = sum(x) / dim`
    /// 2. Compute variance: `var = sum((x - mean)^2) / dim`
    /// 3. Normalize: `x_norm = (x - mean) / sqrt(var + eps)`
    /// 4. Scale and shift: `output = x_norm * weight + bias`
    ///
    /// Computation is performed in F32 for numerical stability.
    /// The output is returned in the same dtype as the input.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape `[..., dim]` where `dim` matches weight dimension
    ///
    /// # Returns
    ///
    /// Normalized tensor of same shape and dtype as input.
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
    /// use lluda_inference::layer_norm::LayerNorm;
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let weight = Tensor::new(vec![1.0; 4], vec![4]).unwrap();
    /// let bias = Tensor::new(vec![0.0; 4], vec![4]).unwrap();
    /// let norm = LayerNorm::new(weight, bias, 1e-5).unwrap();
    ///
    /// let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
    /// let output = norm.forward(&x).unwrap();
    /// assert_eq!(output.shape(), &[1, 4]);
    /// ```
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Validate input has at least 1 dimension
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

        let input_dtype = x.dtype();

        // Convert input to F32 for computation
        let data = x.to_vec_f32();
        let weight_data = self.weight.to_vec_f32();
        let bias_data = self.bias.to_vec_f32();

        let numel = data.len();
        let num_vectors = numel / last_dim;
        let eps = self.eps as f32;

        let mut result_f32 = Vec::with_capacity(numel);

        // Process each vector along the last dimension independently
        for vec_idx in 0..num_vectors {
            let start = vec_idx * last_dim;
            let end = start + last_dim;
            let vec_data = &data[start..end];

            // Step 1: compute mean
            let mean: f32 = vec_data.iter().sum::<f32>() / last_dim as f32;

            // Step 2: compute variance (biased, dividing by N not N-1)
            let var: f32 = vec_data
                .iter()
                .map(|&v| {
                    let diff = v - mean;
                    diff * diff
                })
                .sum::<f32>()
                / last_dim as f32;

            // Step 3: compute inverse standard deviation
            let inv_std = 1.0 / (var + eps).sqrt();

            // Step 4: normalize, scale, and shift
            for i in 0..last_dim {
                let normalized = (vec_data[i] - mean) * inv_std;
                let output_val = normalized * weight_data[i] + bias_data[i];
                result_f32.push(output_val);
            }
        }

        // Return in same dtype as input
        match input_dtype {
            DType::F32 => Tensor::new(result_f32, input_shape.to_vec()),
            DType::BF16 => {
                let bf16_data: Vec<BF16> =
                    result_f32.iter().map(|&v| BF16::from(v)).collect();
                Tensor::from_bf16(bf16_data, input_shape.to_vec())
            }
        }
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

    // F32 tolerance for exact computation
    const F32_TOL: f32 = 1e-5;
    // BF16 precision tolerance — roughly 1/256 precision
    const BF16_TOL: f32 = 2e-2;

    #[test]
    fn test_layer_norm_basic() {
        // Simple 2D input with known values.
        // Input: [[1, 2, 3, 4]], weight: [1, 1, 1, 1], bias: [0, 0, 0, 0]
        // mean = (1 + 2 + 3 + 4) / 4 = 2.5
        // var = ((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2) / 4
        //     = (2.25 + 0.25 + 0.25 + 2.25) / 4 = 5.0 / 4 = 1.25
        // std = sqrt(1.25 + 1e-5) ≈ 1.118034
        // normalized = [(1-2.5), (2-2.5), (3-2.5), (4-2.5)] / std
        //            ≈ [-1.3416, -0.4472, 0.4472, 1.3416]
        let weight = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![4]).unwrap();
        let bias = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![4]).unwrap();
        let norm = LayerNorm::new(weight, bias, 1e-5).unwrap();

        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let output = norm.forward(&x).unwrap();

        assert_eq!(output.shape(), &[1, 4]);
        let data = output.to_vec_f32();

        let mean = 2.5f32;
        let var = 1.25f32;
        let std_val = (var + 1e-5f32).sqrt();

        assert_close(data[0], (1.0 - mean) / std_val, F32_TOL, "element 0");
        assert_close(data[1], (2.0 - mean) / std_val, F32_TOL, "element 1");
        assert_close(data[2], (3.0 - mean) / std_val, F32_TOL, "element 2");
        assert_close(data[3], (4.0 - mean) / std_val, F32_TOL, "element 3");
    }

    #[test]
    fn test_layer_norm_3d() {
        // Batch input [2, 3, 4]: verify normalization is applied per last-dim slice.
        let weight = Tensor::new(vec![1.0; 4], vec![4]).unwrap();
        let bias = Tensor::new(vec![0.0; 4], vec![4]).unwrap();
        let norm = LayerNorm::new(weight, bias, 1e-5).unwrap();

        // Two batches, three sequences of length 4
        let data: Vec<f32> = (1..=24).map(|i| i as f32).collect();
        let x = Tensor::new(data.clone(), vec![2, 3, 4]).unwrap();
        let output = norm.forward(&x).unwrap();

        assert_eq!(output.shape(), &[2, 3, 4]);

        let out = output.to_vec_f32();

        // Verify each of the 6 vectors independently
        for vec_idx in 0..6usize {
            let slice = &data[vec_idx * 4..(vec_idx + 1) * 4];
            let mean: f32 = slice.iter().sum::<f32>() / 4.0;
            let var: f32 = slice.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
            let inv_std = 1.0 / (var + 1e-5f32).sqrt();

            for i in 0..4 {
                let expected = (slice[i] - mean) * inv_std;
                let actual = out[vec_idx * 4 + i];
                assert_close(actual, expected, F32_TOL, &format!("vec {vec_idx}, elem {i}"));
            }
        }
    }

    #[test]
    fn test_layer_norm_shape_mismatch() {
        // weight and bias of different sizes should fail at construction.
        let weight = Tensor::new(vec![1.0, 1.0, 1.0], vec![3]).unwrap();
        let bias = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![4]).unwrap();
        let result = LayerNorm::new(weight, bias, 1e-5);

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
    fn test_layer_norm_matches_pytorch() {
        // Verify results against precomputed PyTorch values.
        //
        // Python reference:
        //   import torch
        //   import torch.nn as nn
        //   x = torch.tensor([[1.0, 2.0, 3.0]])
        //   ln = nn.LayerNorm(3, eps=1e-5)
        //   ln.weight.data = torch.tensor([2.0, 1.0, 0.5])
        //   ln.bias.data   = torch.tensor([0.1, 0.2, 0.3])
        //   print(ln(x))
        //   # mean = 2.0, var = 2/3, std = sqrt(2/3 + 1e-5)
        //   # norm = [−1/std, 0/std, 1/std]  (centered around mean=2)
        //   # out  = norm * weight + bias

        let weight = Tensor::new(vec![2.0, 1.0, 0.5], vec![3]).unwrap();
        let bias = Tensor::new(vec![0.1, 0.2, 0.3], vec![3]).unwrap();
        let norm = LayerNorm::new(weight, bias, 1e-5).unwrap();

        let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let output = norm.forward(&x).unwrap();

        assert_eq!(output.shape(), &[1, 3]);
        let data = output.to_vec_f32();

        // Compute expected by hand (matches PyTorch LayerNorm)
        let mean = 2.0f32;
        let var = 2.0f32 / 3.0;
        let inv_std = 1.0 / (var + 1e-5f32).sqrt();

        let expected = [
            (1.0 - mean) * inv_std * 2.0 + 0.1,
            (2.0 - mean) * inv_std * 1.0 + 0.2,
            (3.0 - mean) * inv_std * 0.5 + 0.3,
        ];

        assert_close(data[0], expected[0], F32_TOL, "PyTorch element 0");
        assert_close(data[1], expected[1], F32_TOL, "PyTorch element 1");
        assert_close(data[2], expected[2], F32_TOL, "PyTorch element 2");
    }

    #[test]
    fn test_layer_norm_with_bias() {
        // Verify that bias shifts the output as expected.
        // weight = [1, 1, 1], bias = [1, 2, 3]
        // Normalized output (weight=1) plus bias.
        let weight = Tensor::new(vec![1.0, 1.0, 1.0], vec![3]).unwrap();
        let bias = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let norm = LayerNorm::new(weight, bias, 1e-5).unwrap();

        // All equal input → normalized values are all 0.0
        let x = Tensor::new(vec![5.0, 5.0, 5.0], vec![1, 3]).unwrap();
        let output = norm.forward(&x).unwrap();

        assert_eq!(output.shape(), &[1, 3]);
        let data = output.to_vec_f32();

        // mean = 5, var = 0 → normalized = [0, 0, 0] → output = [0+1, 0+2, 0+3]
        assert_close(data[0], 1.0, F32_TOL, "element 0 (bias only)");
        assert_close(data[1], 2.0, F32_TOL, "element 1 (bias only)");
        assert_close(data[2], 3.0, F32_TOL, "element 2 (bias only)");
    }

    #[test]
    fn test_layer_norm_bf16_input() {
        // BF16 input should produce BF16 output with the same normalization.
        let weight = Tensor::new(vec![1.0; 4], vec![4]).unwrap();
        let bias = Tensor::new(vec![0.0; 4], vec![4]).unwrap();
        let norm = LayerNorm::new(weight, bias, 1e-5).unwrap();

        let bf16_data: Vec<BF16> = vec![1.0f32, 2.0, 3.0, 4.0]
            .into_iter()
            .map(BF16::from)
            .collect();
        let x = Tensor::from_bf16(bf16_data, vec![1, 4]).unwrap();
        let output = norm.forward(&x).unwrap();

        // Output dtype must match input dtype
        assert_eq!(output.dtype(), DType::BF16);
        assert_eq!(output.shape(), &[1, 4]);

        // Values should be close to F32 version (within BF16 precision)
        let out_data = output.to_vec_f32();
        let mean = 2.5f32;
        let var = 1.25f32;
        let inv_std = 1.0 / (var + 1e-5f32).sqrt();

        assert_close(out_data[0], (1.0 - mean) * inv_std, BF16_TOL, "bf16 elem 0");
        assert_close(out_data[1], (2.0 - mean) * inv_std, BF16_TOL, "bf16 elem 1");
        assert_close(out_data[2], (3.0 - mean) * inv_std, BF16_TOL, "bf16 elem 2");
        assert_close(out_data[3], (4.0 - mean) * inv_std, BF16_TOL, "bf16 elem 3");
    }

    #[test]
    fn test_layer_norm_f32_input_returns_f32() {
        // F32 input must produce F32 output.
        let weight = Tensor::new(vec![1.0; 4], vec![4]).unwrap();
        let bias = Tensor::new(vec![0.0; 4], vec![4]).unwrap();
        let norm = LayerNorm::new(weight, bias, 1e-5).unwrap();

        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let output = norm.forward(&x).unwrap();

        assert_eq!(output.dtype(), DType::F32);
    }

    #[test]
    fn test_layer_norm_input_dim_mismatch() {
        // Last dim of input != weight dim should fail.
        let weight = Tensor::new(vec![1.0; 3], vec![3]).unwrap();
        let bias = Tensor::new(vec![0.0; 3], vec![3]).unwrap();
        let norm = LayerNorm::new(weight, bias, 1e-5).unwrap();

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
    fn test_layer_norm_non_1d_weight_fails() {
        // 2D weight must be rejected at construction.
        let weight_2d = Tensor::new(vec![1.0; 12], vec![3, 4]).unwrap();
        let bias = Tensor::new(vec![0.0; 3], vec![3]).unwrap();
        let result = LayerNorm::new(weight_2d, bias, 1e-5);
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
    fn test_layer_norm_non_1d_bias_fails() {
        // 2D bias must be rejected at construction.
        let weight = Tensor::new(vec![1.0; 3], vec![3]).unwrap();
        let bias_2d = Tensor::new(vec![0.0; 12], vec![3, 4]).unwrap();
        let result = LayerNorm::new(weight, bias_2d, 1e-5);
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
    fn test_layer_norm_eps_prevents_zero_division() {
        // All equal input → variance = 0; eps must prevent division by zero.
        let weight = Tensor::new(vec![1.0; 4], vec![4]).unwrap();
        let bias = Tensor::new(vec![0.0; 4], vec![4]).unwrap();
        let norm = LayerNorm::new(weight, bias, 1e-5).unwrap();

        let x = Tensor::new(vec![3.0, 3.0, 3.0, 3.0], vec![1, 4]).unwrap();
        let output = norm.forward(&x).unwrap();

        let data = output.to_vec_f32();
        // All values identical → normalized output is 0 (just eps noise)
        assert!(data.iter().all(|&v| v.is_finite()), "output must be finite");
        assert!(data.iter().all(|&v| v.abs() < 1e-3), "output near zero for uniform input");
    }

    #[test]
    fn test_layer_norm_4d_input() {
        // Verify support for 4D input [B, C, H, W] pattern.
        let dim = 3usize;
        let weight = Tensor::new(vec![1.0; dim], vec![dim]).unwrap();
        let bias = Tensor::new(vec![0.0; dim], vec![dim]).unwrap();
        let norm = LayerNorm::new(weight, bias, 1e-5).unwrap();

        let numel = 2 * 2 * 2 * dim;
        let data: Vec<f32> = (1..=numel as u32).map(|i| i as f32).collect();
        let x = Tensor::new(data.clone(), vec![2, 2, 2, dim]).unwrap();
        let output = norm.forward(&x).unwrap();

        assert_eq!(output.shape(), &[2, 2, 2, dim]);

        // Verify output is finite
        let out = output.to_vec_f32();
        assert!(out.iter().all(|&v| v.is_finite()));
    }
}
