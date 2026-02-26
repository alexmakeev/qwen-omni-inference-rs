//! Feed-forward MLP for the audio encoder.
//!
//! `AudioMlp` is a simple two-layer MLP with GELU activation used in
//! each audio encoder layer. Unlike the text MLP (`mlp.rs`), there is
//! no gating — just a linear expand, GELU, and linear contract:
//!
//! ```text
//! AudioMlp(x) = fc2(gelu(fc1(x)))
//! ```
//!
//! # Dimensions (Qwen-Audio)
//!
//! | Layer | Input dim | Output dim |
//! |-------|-----------|------------|
//! | fc1   | 1280      | 5120       |
//! | fc2   | 5120      | 1280       |
//!
//! Both layers include bias terms.

use crate::audio_attention::LinearBias;
use crate::error::Result;
use crate::tensor::Tensor;

/// Feed-forward MLP for the audio encoder (GELU activation, no gating).
///
/// Computes `fc2(gelu(fc1(x)))` where both `fc1` and `fc2` are
/// linear projections with bias.
#[derive(Debug, Clone)]
pub struct AudioMlp {
    /// First projection: expands from d_model to intermediate_size.
    fc1: LinearBias,
    /// Second projection: contracts from intermediate_size to d_model.
    fc2: LinearBias,
}

impl AudioMlp {
    /// Create a new AudioMlp layer.
    ///
    /// # Arguments
    ///
    /// * `fc1` - First linear layer (expand). Weight shape: [intermediate_size, d_model]
    /// * `fc2` - Second linear layer (contract). Weight shape: [d_model, intermediate_size]
    ///
    /// # Returns
    ///
    /// New `AudioMlp` instance.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::audio_mlp::AudioMlp;
    /// use lluda_inference::audio_attention::LinearBias;
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let d_model = 4;
    /// let intermediate = 8;
    ///
    /// let fc1_w = Tensor::new(vec![0.0; intermediate * d_model], vec![intermediate, d_model]).unwrap();
    /// let fc1_b = Tensor::new(vec![0.0; intermediate], vec![intermediate]).unwrap();
    /// let fc1 = LinearBias::new(fc1_w, fc1_b).unwrap();
    ///
    /// let fc2_w = Tensor::new(vec![0.0; d_model * intermediate], vec![d_model, intermediate]).unwrap();
    /// let fc2_b = Tensor::new(vec![0.0; d_model], vec![d_model]).unwrap();
    /// let fc2 = LinearBias::new(fc2_w, fc2_b).unwrap();
    ///
    /// let mlp = AudioMlp::new(fc1, fc2);
    /// ```
    pub fn new(fc1: LinearBias, fc2: LinearBias) -> Self {
        Self { fc1, fc2 }
    }

    /// Forward pass: `fc2(gelu(fc1(x)))`.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape `[..., d_model]`
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[..., d_model]` (same leading dimensions and final dimension).
    ///
    /// # Errors
    ///
    /// Returns error if shapes are incompatible or tensor operations fail.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::audio_mlp::AudioMlp;
    /// use lluda_inference::audio_attention::LinearBias;
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let d_model = 4;
    /// let intermediate = 8;
    ///
    /// let fc1_w = Tensor::new(vec![0.1; intermediate * d_model], vec![intermediate, d_model]).unwrap();
    /// let fc1_b = Tensor::new(vec![0.0; intermediate], vec![intermediate]).unwrap();
    /// let fc1 = LinearBias::new(fc1_w, fc1_b).unwrap();
    ///
    /// let fc2_w = Tensor::new(vec![0.1; d_model * intermediate], vec![d_model, intermediate]).unwrap();
    /// let fc2_b = Tensor::new(vec![0.0; d_model], vec![d_model]).unwrap();
    /// let fc2 = LinearBias::new(fc2_w, fc2_b).unwrap();
    ///
    /// let mlp = AudioMlp::new(fc1, fc2);
    ///
    /// let x = Tensor::new(vec![1.0; 3 * d_model], vec![3, d_model]).unwrap();
    /// let out = mlp.forward(&x).unwrap();
    /// assert_eq!(out.shape(), &[3, d_model]);
    /// ```
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.fc1.forward(x)?;
        let h = h.gelu()?;
        self.fc2.forward(&h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a LinearBias with constant weight and bias.
    fn make_linear_bias(out: usize, in_: usize, w_val: f32, b_val: f32) -> LinearBias {
        let weight = Tensor::new(vec![w_val; out * in_], vec![out, in_]).unwrap();
        let bias = Tensor::new(vec![b_val; out], vec![out]).unwrap();
        LinearBias::new(weight, bias).unwrap()
    }

    /// Verify that output shape equals input shape [seq, d_model] -> [seq, d_model].
    #[test]
    fn test_audio_mlp_forward() {
        let d_model = 4;
        let intermediate = 8;
        let seq_len = 3;

        let fc1 = make_linear_bias(intermediate, d_model, 0.1, 0.0);
        let fc2 = make_linear_bias(d_model, intermediate, 0.1, 0.0);
        let mlp = AudioMlp::new(fc1, fc2);

        let x = Tensor::new(vec![1.0; seq_len * d_model], vec![seq_len, d_model]).unwrap();
        let out = mlp.forward(&x).unwrap();

        assert_eq!(out.shape(), &[seq_len, d_model]);

        // Values must be finite
        let data = out.to_vec_f32();
        assert!(data.iter().all(|&v| v.is_finite()), "output must be finite");
    }

    /// With zero input, fc1 output = 0 + bias; gelu(bias) if bias=0 gives 0;
    /// fc2(0) = 0 + fc2.bias. When both biases are zero, output is all zeros.
    #[test]
    fn test_audio_mlp_zero_input() {
        let d_model = 4;
        let intermediate = 8;

        // Both biases zero — gelu(0) = 0, so output = fc2_bias = 0
        let fc1 = make_linear_bias(intermediate, d_model, 0.5, 0.0);
        let fc2 = make_linear_bias(d_model, intermediate, 0.5, 0.0);
        let mlp = AudioMlp::new(fc1, fc2);

        let x = Tensor::new(vec![0.0; d_model], vec![1, d_model]).unwrap();
        let out = mlp.forward(&x).unwrap();

        assert_eq!(out.shape(), &[1, d_model]);
        let data = out.to_vec_f32();
        // gelu(0) = 0 => fc2(0-vector) = 0-vector (no bias)
        assert!(
            data.iter().all(|&v| v.abs() < 1e-5),
            "zero input with zero bias should give zero output"
        );
    }

    /// Verify output is finite for various input values.
    #[test]
    fn test_audio_mlp_finite_output() {
        let d_model = 4;
        let intermediate = 8;

        let fc1 = make_linear_bias(intermediate, d_model, 0.1, 0.1);
        let fc2 = make_linear_bias(d_model, intermediate, 0.1, 0.1);
        let mlp = AudioMlp::new(fc1, fc2);

        let data: Vec<f32> = (0..d_model * 5).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let x = Tensor::new(data, vec![5, d_model]).unwrap();
        let out = mlp.forward(&x).unwrap();

        assert_eq!(out.shape(), &[5, d_model]);
        assert!(out.to_vec_f32().iter().all(|&v| v.is_finite()));
    }
}
