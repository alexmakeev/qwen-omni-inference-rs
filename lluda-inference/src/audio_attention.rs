//! Multi-head attention for the audio encoder (Whisper-style).
//!
//! Implements bidirectional full multi-head attention used in the audio encoder
//! of Qwen-Audio and similar Whisper-based models. Unlike the text GQA attention
//! (`attention.rs`), the audio encoder uses:
//!
//! - Full MHA: all 20 heads share no KV (no GQA)
//! - Bidirectional: no causal mask (encoder sees the whole sequence)
//! - Sinusoidal position encoding: applied before the transformer, not in attention
//! - Bias terms: q, v, and out projections have bias; k projection does not
//! - No per-head norms: no RMSNorm on Q or K heads
//! - No KV cache: the encoder processes the full audio sequence once
//!
//! # Architecture
//!
//! For Qwen-Audio encoder:
//! - Hidden dimension: 1280
//! - Attention heads: 20
//! - Head dimension: 64 (1280 / 20)
//!
//! # Computation
//!
//! ```text
//! q = x @ q_weight.T + q_bias       # [seq, 1280]
//! k = x @ k_weight.T                # [seq, 1280] — no bias
//! v = x @ v_weight.T + v_bias       # [seq, 1280]
//!
//! q, k, v -> reshape to [20, seq, 64], transpose(0, 1)
//!
//! scale = 1 / sqrt(64)
//! attn = softmax(q @ k.T * scale, dim=-1) @ v  # [20, seq, 64]
//!
//! output = merge_heads(attn)                    # [seq, 1280]
//! output = output @ out_weight.T + out_bias
//! ```
//!
//! # References
//!
//! - Whisper: https://arxiv.org/abs/2212.04356
//! - Qwen-Audio: https://arxiv.org/abs/2309.10780

use crate::error::{LludaError, Result};
use crate::tensor::Tensor;

/// Linear projection with a bias term.
///
/// Computes `x @ weight.T + bias` where weight is stored in
/// [out_features, in_features] format (matching PyTorch convention).
///
/// Used for q, v, and out projections in the audio encoder, and also
/// by `AudioMlp`.
#[derive(Debug, Clone)]
pub struct LinearBias {
    /// Weight matrix [out_features, in_features]
    pub(crate) weight: Tensor,
    /// Bias vector [out_features]
    pub(crate) bias: Tensor,
}

impl LinearBias {
    /// Create a new linear-with-bias layer.
    ///
    /// # Arguments
    ///
    /// * `weight` - Weight tensor of shape [out_features, in_features]
    /// * `bias` - Bias tensor of shape [out_features]
    ///
    /// # Returns
    ///
    /// New `LinearBias` instance.
    ///
    /// # Errors
    ///
    /// Returns `ShapeMismatch` if:
    /// - `weight` is not 2D
    /// - `bias` is not 1D
    /// - `bias.len()` does not match `weight.shape()[0]` (out_features)
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::audio_attention::LinearBias;
    /// use lluda_inference::tensor::Tensor;
    ///
    /// // 4 inputs -> 8 outputs
    /// let weight = Tensor::new(vec![0.0; 32], vec![8, 4]).unwrap();
    /// let bias = Tensor::new(vec![0.0; 8], vec![8]).unwrap();
    /// let layer = LinearBias::new(weight, bias).unwrap();
    /// ```
    pub fn new(weight: Tensor, bias: Tensor) -> Result<Self> {
        if weight.ndim() != 2 {
            return Err(LludaError::ShapeMismatch {
                expected: vec![0, 0], // Expect 2D
                got: weight.shape().to_vec(),
            });
        }
        if bias.ndim() != 1 {
            return Err(LludaError::ShapeMismatch {
                expected: vec![0], // Expect 1D
                got: bias.shape().to_vec(),
            });
        }
        let out_features = weight.shape()[0];
        let bias_len = bias.shape()[0];
        if bias_len != out_features {
            return Err(LludaError::ShapeMismatch {
                expected: vec![out_features],
                got: vec![bias_len],
            });
        }
        Ok(LinearBias { weight, bias })
    }

    /// Forward pass: compute `x @ weight.T + bias`.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape [..., in_features]
    ///
    /// # Returns
    ///
    /// Output tensor of shape [..., out_features]
    ///
    /// # Errors
    ///
    /// Returns `ShapeMismatch` if the last dimension of `x` doesn't match
    /// `weight`'s `in_features`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::audio_attention::LinearBias;
    /// use lluda_inference::tensor::Tensor;
    ///
    /// // Identity-like: weight = I (2x2), bias = [1, 1]
    /// let weight = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
    /// let bias = Tensor::new(vec![1.0, 1.0], vec![2]).unwrap();
    /// let layer = LinearBias::new(weight, bias).unwrap();
    ///
    /// let x = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
    /// let y = layer.forward(&x).unwrap();
    ///
    /// // x @ I + bias = [2+1, 3+1] = [3, 4]
    /// assert_eq!(y.shape(), &[1, 2]);
    /// let data = y.to_vec_f32();
    /// assert!((data[0] - 3.0).abs() < 1e-5);
    /// assert!((data[1] - 4.0).abs() < 1e-5);
    /// ```
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let input_shape = x.shape();
        let weight_shape = self.weight.shape();

        let out_features = weight_shape[0];
        let in_features = weight_shape[1];

        // Validate last dimension matches in_features
        if input_shape[input_shape.len() - 1] != in_features {
            return Err(LludaError::ShapeMismatch {
                expected: vec![in_features],
                got: vec![input_shape[input_shape.len() - 1]],
            });
        }

        // Flatten all leading dimensions: [..., in] -> [batch, in]
        let batch_size: usize = input_shape[..input_shape.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch_size, in_features])?;

        // Transpose weight: [out, in] -> [in, out]
        let weight_t = self.weight.transpose()?;

        // Compute: [batch, in] @ [in, out] -> [batch, out]
        let output_2d = x_2d.matmul(&weight_t)?;

        // Add bias: [out] broadcasts to [batch, out]
        let output_biased = output_2d.add(&self.bias)?;

        // Reshape back to original leading dims + out_features
        let mut output_shape = input_shape[..input_shape.len() - 1].to_vec();
        output_shape.push(out_features);

        output_biased.reshape(&output_shape)
    }
}

/// Linear projection without a bias term.
///
/// Computes `x @ weight.T` where weight is stored in [out_features, in_features]
/// format (matching PyTorch convention).
///
/// Used for the key projection in the audio encoder, which has no bias
/// (`Whisper`-style: only k has no bias).
///
/// This type is self-contained and does not reuse the `Linear` from `attention.rs`
/// so that all audio encoder modules remain independent.
#[derive(Debug, Clone)]
pub struct Linear {
    /// Weight matrix [out_features, in_features]
    pub(crate) weight: Tensor,
}

impl Linear {
    /// Create a new linear layer from a weight tensor.
    ///
    /// # Arguments
    ///
    /// * `weight` - Weight tensor of shape [out_features, in_features]
    ///
    /// # Returns
    ///
    /// New `Linear` instance.
    ///
    /// # Errors
    ///
    /// Returns `ShapeMismatch` if `weight` is not 2D.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::audio_attention::Linear;
    /// use lluda_inference::tensor::Tensor;
    ///
    /// // 4 inputs -> 8 outputs (no bias)
    /// let weight = Tensor::new(vec![0.0; 32], vec![8, 4]).unwrap();
    /// let layer = Linear::new(weight).unwrap();
    /// ```
    pub fn new(weight: Tensor) -> Result<Self> {
        if weight.ndim() != 2 {
            return Err(LludaError::ShapeMismatch {
                expected: vec![0, 0], // Expect 2D
                got: weight.shape().to_vec(),
            });
        }
        Ok(Linear { weight })
    }

    /// Forward pass: compute `x @ weight.T`.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape [..., in_features]
    ///
    /// # Returns
    ///
    /// Output tensor of shape [..., out_features]
    ///
    /// # Errors
    ///
    /// Returns `ShapeMismatch` if the last dimension of `x` doesn't match
    /// `weight`'s `in_features`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::audio_attention::Linear;
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let weight = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let layer = Linear::new(weight).unwrap();
    ///
    /// let x = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
    /// let y = layer.forward(&x).unwrap();
    ///
    /// // x @ [[1,2],[3,4]].T = x @ [[1,3],[2,4]] = [[1*1+2*2, 1*3+2*4]] = [[5, 11]]
    /// assert_eq!(y.shape(), &[1, 2]);
    /// ```
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let input_shape = x.shape();
        let weight_shape = self.weight.shape();

        let out_features = weight_shape[0];
        let in_features = weight_shape[1];

        // Validate last dimension matches in_features
        if input_shape[input_shape.len() - 1] != in_features {
            return Err(LludaError::ShapeMismatch {
                expected: vec![in_features],
                got: vec![input_shape[input_shape.len() - 1]],
            });
        }

        // Flatten all leading dimensions: [..., in] -> [batch, in]
        let batch_size: usize = input_shape[..input_shape.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch_size, in_features])?;

        // Transpose weight: [out, in] -> [in, out]
        let weight_t = self.weight.transpose()?;

        // Compute: [batch, in] @ [in, out] -> [batch, out]
        let output_2d = x_2d.matmul(&weight_t)?;

        // Reshape back to original leading dims + out_features
        let mut output_shape = input_shape[..input_shape.len() - 1].to_vec();
        output_shape.push(out_features);

        output_2d.reshape(&output_shape)
    }
}

/// Multi-head attention for the audio encoder (Whisper-style).
///
/// Bidirectional full MHA — no causal mask, no GQA, no per-head norms, no KV cache.
/// Position information is provided via sinusoidal embeddings applied before this layer.
///
/// # Projection bias conventions (Whisper)
///
/// | Projection | Has bias |
/// |------------|----------|
/// | q_proj     | yes      |
/// | k_proj     | **no**   |
/// | v_proj     | yes      |
/// | out_proj   | yes      |
///
/// # Shape flow
///
/// ```text
/// x:    [seq_len, d_model]
///
/// q = q_proj(x)   -> [seq_len, num_heads * head_dim]
/// k = k_proj(x)   -> [seq_len, num_heads * head_dim]
/// v = v_proj(x)   -> [seq_len, num_heads * head_dim]
///
/// reshape + transpose -> [num_heads, seq_len, head_dim] for each
///
/// scores = softmax(q @ k.T / sqrt(head_dim), dim=2)  # [num_heads, seq_len, seq_len]
/// ctx    = scores @ v                                 # [num_heads, seq_len, head_dim]
///
/// merge  -> [seq_len, d_model]
/// output = out_proj(merge)                            # [seq_len, d_model]
/// ```
#[derive(Debug, Clone)]
pub struct AudioAttention {
    /// Query projection with bias [d_model, d_model]
    q_proj: LinearBias,
    /// Key projection without bias [d_model, d_model]
    k_proj: Linear,
    /// Value projection with bias [d_model, d_model]
    v_proj: LinearBias,
    /// Output projection with bias [d_model, d_model]
    out_proj: LinearBias,
    /// Number of attention heads (20 for Qwen-Audio encoder)
    num_heads: usize,
    /// Dimension per head; equals d_model / num_heads (64 for Qwen-Audio)
    head_dim: usize,
}

impl AudioAttention {
    /// Create a new audio attention layer.
    ///
    /// # Arguments
    ///
    /// * `q_proj` - Query projection (with bias)
    /// * `k_proj` - Key projection (without bias)
    /// * `v_proj` - Value projection (with bias)
    /// * `out_proj` - Output projection (with bias)
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension of each head
    ///
    /// # Returns
    ///
    /// New `AudioAttention` instance.
    ///
    /// # Errors
    ///
    /// Returns `Model` error if `head_dim * num_heads` does not equal `d_model`
    /// (inferred from `q_proj.weight.shape()[0]`).
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::audio_attention::{AudioAttention, Linear, LinearBias};
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let d_model = 1280;
    /// let num_heads = 20;
    /// let head_dim = 64;
    ///
    /// let mk_w = |rows: usize, cols: usize| {
    ///     Tensor::new(vec![0.0; rows * cols], vec![rows, cols]).unwrap()
    /// };
    /// let mk_b = |n: usize| Tensor::new(vec![0.0; n], vec![n]).unwrap();
    ///
    /// let q_proj = LinearBias::new(mk_w(d_model, d_model), mk_b(d_model)).unwrap();
    /// let k_proj = Linear::new(mk_w(d_model, d_model)).unwrap();
    /// let v_proj = LinearBias::new(mk_w(d_model, d_model), mk_b(d_model)).unwrap();
    /// let out_proj = LinearBias::new(mk_w(d_model, d_model), mk_b(d_model)).unwrap();
    ///
    /// let attn = AudioAttention::new(q_proj, k_proj, v_proj, out_proj, num_heads, head_dim).unwrap();
    /// ```
    pub fn new(
        q_proj: LinearBias,
        k_proj: Linear,
        v_proj: LinearBias,
        out_proj: LinearBias,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        // Validate that num_heads * head_dim == d_model
        let d_model = q_proj.weight.shape()[0];
        if num_heads * head_dim != d_model {
            return Err(LludaError::Model(format!(
                "AudioAttention: num_heads ({}) * head_dim ({}) = {} does not match \
                 d_model ({}) from q_proj weight shape",
                num_heads,
                head_dim,
                num_heads * head_dim,
                d_model,
            )));
        }

        Ok(AudioAttention {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
        })
    }

    /// Forward pass through the audio attention layer.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape [seq_len, d_model]
    ///
    /// # Returns
    ///
    /// Output tensor of shape [seq_len, d_model], same shape as input.
    ///
    /// # Errors
    ///
    /// Returns error if input shape is incompatible or any tensor operation fails.
    ///
    /// # Process
    ///
    /// 1. Project input to Q (with bias), K (no bias), V (with bias)
    /// 2. Reshape each to [num_heads, seq_len, head_dim]
    /// 3. Compute scaled dot-product attention (no mask — bidirectional)
    /// 4. Merge heads back to [seq_len, d_model]
    /// 5. Apply output projection (with bias)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let shape = x.shape();

        if shape.len() != 2 {
            return Err(LludaError::ShapeMismatch {
                expected: vec![0, self.num_heads * self.head_dim], // 2D [seq, d_model]
                got: shape.to_vec(),
            });
        }

        let seq_len = shape[0];
        let d_model = self.num_heads * self.head_dim;

        // 1. Project to Q, K, V
        //    q: [seq_len, d_model] (bias added)
        //    k: [seq_len, d_model] (no bias)
        //    v: [seq_len, d_model] (bias added)
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // 2. Reshape to multi-head format
        //    [seq_len, d_model] -> [seq_len, num_heads, head_dim] -> [num_heads, seq_len, head_dim]
        let q = q
            .reshape(&[seq_len, self.num_heads, self.head_dim])?
            .transpose_dims(0, 1)?;
        let k = k
            .reshape(&[seq_len, self.num_heads, self.head_dim])?
            .transpose_dims(0, 1)?;
        let v = v
            .reshape(&[seq_len, self.num_heads, self.head_dim])?
            .transpose_dims(0, 1)?;

        // 3. Scaled dot-product attention (no causal mask — bidirectional encoder)
        //
        //    k.T: [num_heads, head_dim, seq_len]
        //    scores = q @ k.T -> [num_heads, seq_len, seq_len]
        //    Softmax over last dim (the key positions)
        //    ctx = scores @ v -> [num_heads, seq_len, head_dim]

        let scale = 1.0 / (self.head_dim as f32).sqrt();

        // Transpose last two dims of k: [num_heads, seq_len, head_dim] -> [num_heads, head_dim, seq_len]
        let k_t = k.transpose_dims(1, 2)?;

        // [num_heads, seq_len, seq_len]
        let scores = q.matmul(&k_t)?;
        let scores = scores.mul_scalar(scale)?;

        // Softmax over the key dimension (dim 2, last dim)
        let attn_weights = scores.softmax(2)?;

        // Weighted sum over values: [num_heads, seq_len, head_dim]
        let ctx = attn_weights.matmul(&v)?;

        // 4. Merge heads
        //    [num_heads, seq_len, head_dim] -> [seq_len, num_heads, head_dim] -> [seq_len, d_model]
        let ctx = ctx.transpose_dims(0, 1)?;
        let ctx = ctx.reshape(&[seq_len, d_model])?;

        // 5. Output projection (with bias)
        self.out_proj.forward(&ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper utilities
    // -----------------------------------------------------------------------

    /// Build a [rows x cols] weight tensor filled with a constant.
    fn make_weight(rows: usize, cols: usize, val: f32) -> Tensor {
        Tensor::new(vec![val; rows * cols], vec![rows, cols]).unwrap()
    }

    /// Build a bias vector of length `n` filled with `val`.
    fn make_bias(n: usize, val: f32) -> Tensor {
        Tensor::new(vec![val; n], vec![n]).unwrap()
    }

    /// Tolerance for float comparisons (F32 output from matmul chains).
    fn assert_close(a: f32, b: f32, tol: f32, label: &str) {
        assert!(
            (a - b).abs() < tol,
            "{}: expected {}, got {}, diff = {}",
            label,
            b,
            a,
            (a - b).abs()
        );
    }

    const TOL: f32 = 1e-4;

    // -----------------------------------------------------------------------
    // LinearBias tests
    // -----------------------------------------------------------------------

    /// Verify that LinearBias correctly applies weight matmul and bias addition.
    #[test]
    fn test_linear_bias_basic() {
        // weight = [[1, 0], [0, 1]] (identity), bias = [10, 20]
        // forward([3, 4]) => [3+10, 4+20] = [13, 24]
        let weight = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let bias = Tensor::new(vec![10.0, 20.0], vec![2]).unwrap();
        let layer = LinearBias::new(weight, bias).unwrap();

        let x = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let y = layer.forward(&x).unwrap();

        assert_eq!(y.shape(), &[1, 2]);
        let data = y.to_vec_f32();
        assert_close(data[0], 13.0, TOL, "bias output[0]");
        assert_close(data[1], 24.0, TOL, "bias output[1]");
    }

    /// Verify that LinearBias broadcast works for multi-row inputs.
    #[test]
    fn test_linear_bias_batch() {
        // weight = [[1, 0], [0, 1]], bias = [1, 2]
        // Two rows: [1,0] -> [2, 2], [0,1] -> [1, 3]
        let weight = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let bias = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
        let layer = LinearBias::new(weight, bias).unwrap();

        let x = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let y = layer.forward(&x).unwrap();

        assert_eq!(y.shape(), &[2, 2]);
        let data = y.to_vec_f32();
        assert_close(data[0], 2.0, TOL, "row0 col0");
        assert_close(data[1], 2.0, TOL, "row0 col1");
        assert_close(data[2], 1.0, TOL, "row1 col0");
        assert_close(data[3], 3.0, TOL, "row1 col1");
    }

    // -----------------------------------------------------------------------
    // Linear (no bias) tests
    // -----------------------------------------------------------------------

    /// Verify that Linear without bias computes the correct matmul.
    #[test]
    fn test_linear_no_bias_basic() {
        // weight = [[2, 0], [0, 3]] (diagonal scaling), no bias
        // forward([1, 1]) = [2, 3]
        let weight = Tensor::new(vec![2.0, 0.0, 0.0, 3.0], vec![2, 2]).unwrap();
        let layer = Linear::new(weight).unwrap();

        let x = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
        let y = layer.forward(&x).unwrap();

        assert_eq!(y.shape(), &[1, 2]);
        let data = y.to_vec_f32();
        assert_close(data[0], 2.0, TOL, "no-bias output[0]");
        assert_close(data[1], 3.0, TOL, "no-bias output[1]");
    }

    // -----------------------------------------------------------------------
    // Validation / error path tests
    // -----------------------------------------------------------------------

    /// LinearBias must reject a non-2D weight.
    #[test]
    fn test_linear_bias_validation_weight_not_2d() {
        let weight_3d = Tensor::new(vec![0.0; 8], vec![2, 2, 2]).unwrap();
        let bias = Tensor::new(vec![0.0; 2], vec![2]).unwrap();
        let result = LinearBias::new(weight_3d, bias);
        assert!(result.is_err());
        match result.unwrap_err() {
            LludaError::ShapeMismatch { .. } => {}
            other => panic!("Expected ShapeMismatch, got: {:?}", other),
        }
    }

    /// LinearBias must reject a non-1D bias.
    #[test]
    fn test_linear_bias_validation_bias_not_1d() {
        let weight = Tensor::new(vec![0.0; 4], vec![2, 2]).unwrap();
        let bias_2d = Tensor::new(vec![0.0; 4], vec![2, 2]).unwrap();
        let result = LinearBias::new(weight, bias_2d);
        assert!(result.is_err());
        match result.unwrap_err() {
            LludaError::ShapeMismatch { .. } => {}
            other => panic!("Expected ShapeMismatch, got: {:?}", other),
        }
    }

    /// LinearBias must reject bias length that doesn't match weight out_features.
    #[test]
    fn test_linear_bias_validation_bias_len_mismatch() {
        let weight = Tensor::new(vec![0.0; 6], vec![3, 2]).unwrap(); // out=3, in=2
        let bias_wrong = Tensor::new(vec![0.0; 2], vec![2]).unwrap(); // length 2, should be 3
        let result = LinearBias::new(weight, bias_wrong);
        assert!(result.is_err());
        match result.unwrap_err() {
            LludaError::ShapeMismatch { expected, got } => {
                assert_eq!(expected, vec![3]);
                assert_eq!(got, vec![2]);
            }
            other => panic!("Expected ShapeMismatch, got: {:?}", other),
        }
    }

    /// Linear must reject a non-2D weight.
    #[test]
    fn test_linear_validation_weight_not_2d() {
        let weight_1d = Tensor::new(vec![0.0; 4], vec![4]).unwrap();
        let result = Linear::new(weight_1d);
        assert!(result.is_err());
        match result.unwrap_err() {
            LludaError::ShapeMismatch { .. } => {}
            other => panic!("Expected ShapeMismatch, got: {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // AudioAttention tests
    // -----------------------------------------------------------------------

    /// Build a minimal AudioAttention with zero weights and zero biases.
    /// The output with zero weights is always zero — useful for shape checks.
    fn make_zero_attention(d_model: usize, num_heads: usize, head_dim: usize) -> AudioAttention {
        let mk_lb = |r: usize, c: usize| {
            LinearBias::new(make_weight(r, c, 0.0), make_bias(r, 0.0)).unwrap()
        };
        let mk_l = |r: usize, c: usize| Linear::new(make_weight(r, c, 0.0)).unwrap();

        AudioAttention::new(
            mk_lb(d_model, d_model),
            mk_l(d_model, d_model),
            mk_lb(d_model, d_model),
            mk_lb(d_model, d_model),
            num_heads,
            head_dim,
        )
        .unwrap()
    }

    /// Output shape must match input shape [seq_len, d_model].
    #[test]
    fn test_audio_attention_shape() {
        let d_model = 1280;
        let num_heads = 20;
        let head_dim = 64;
        let seq_len = 8;

        let attn = make_zero_attention(d_model, num_heads, head_dim);

        let x = Tensor::new(vec![0.0; seq_len * d_model], vec![seq_len, d_model]).unwrap();
        let out = attn.forward(&x).unwrap();

        assert_eq!(out.shape(), &[seq_len, d_model]);
    }

    /// Smaller dimensions for faster test: shape invariance check.
    #[test]
    fn test_audio_attention_shape_small() {
        // d_model=8, num_heads=2, head_dim=4, seq_len=3
        let d_model = 8;
        let num_heads = 2;
        let head_dim = 4;
        let seq_len = 3;

        let attn = make_zero_attention(d_model, num_heads, head_dim);

        let x = Tensor::new(vec![1.0; seq_len * d_model], vec![seq_len, d_model]).unwrap();
        let out = attn.forward(&x).unwrap();

        assert_eq!(out.shape(), &[seq_len, d_model]);
    }

    /// Same input must produce the same output (determinism check).
    #[test]
    fn test_audio_attention_deterministic() {
        let d_model = 8;
        let num_heads = 2;
        let head_dim = 4;
        let seq_len = 4;

        // Use non-trivial weights so output is not all zeros
        let mk_lb = |r: usize, c: usize, v: f32| {
            LinearBias::new(make_weight(r, c, v), make_bias(r, 0.1)).unwrap()
        };
        let mk_l = |r: usize, c: usize, v: f32| Linear::new(make_weight(r, c, v)).unwrap();

        let attn = AudioAttention::new(
            mk_lb(d_model, d_model, 0.01),
            mk_l(d_model, d_model, 0.01),
            mk_lb(d_model, d_model, 0.01),
            mk_lb(d_model, d_model, 0.01),
            num_heads,
            head_dim,
        )
        .unwrap();

        let input_data: Vec<f32> = (0..seq_len * d_model)
            .map(|i| (i as f32) * 0.01)
            .collect();
        let x = Tensor::new(input_data.clone(), vec![seq_len, d_model]).unwrap();

        let out1 = attn.forward(&x).unwrap();
        let out2 = attn.forward(&x).unwrap();

        let d1 = out1.to_vec_f32();
        let d2 = out2.to_vec_f32();

        for (i, (&a, &b)) in d1.iter().zip(d2.iter()).enumerate() {
            assert_eq!(
                a, b,
                "determinism violation at index {}: {} != {}",
                i, a, b
            );
        }
    }

    /// Bidirectional attention: verify no causal masking by confirming that
    /// a token at position 0 can attend to tokens at later positions.
    ///
    /// We construct a scenario where the output at position 0 would differ
    /// depending on whether position 1 is visible (bidirectional) or masked.
    /// With zero weights for projections and a constant v after out_proj, the
    /// simplest check is that the attention weights are uniform (each position
    /// attends equally to all positions), which follows from bidirectional softmax
    /// over a flat score matrix.
    #[test]
    fn test_audio_attention_bidirectional() {
        // With all-zero Q and K weights, all scores = 0 before scale,
        // so softmax gives uniform attention (1/seq_len for each position).
        // This is only possible if no causal mask is applied — otherwise some
        // positions would be masked to -inf, breaking the uniformity.
        //
        // We use distinct V values and verify the output is a uniform average.
        let d_model = 4;
        let num_heads = 1;
        let head_dim = 4;
        let seq_len = 3;

        // Q, K weight = 0  => all attention scores = 0 => softmax = uniform 1/3
        // V weight = identity (scale = 0.1 so values remain distinct)
        // out_proj = identity with zero bias

        // V projection: identity-like weight [4, 4], no bias on out; bias on v
        let v_weight = Tensor::new(
            vec![
                1.0, 0.0, 0.0, 0.0, // row 0
                0.0, 1.0, 0.0, 0.0, // row 1
                0.0, 0.0, 1.0, 0.0, // row 2
                0.0, 0.0, 0.0, 1.0, // row 3
            ],
            vec![4, 4],
        )
        .unwrap();
        let v_proj =
            LinearBias::new(v_weight, Tensor::new(vec![0.0; 4], vec![4]).unwrap()).unwrap();

        let out_weight = Tensor::new(
            vec![
                1.0, 0.0, 0.0, 0.0, // row 0
                0.0, 1.0, 0.0, 0.0, // row 1
                0.0, 0.0, 1.0, 0.0, // row 2
                0.0, 0.0, 0.0, 1.0, // row 3
            ],
            vec![4, 4],
        )
        .unwrap();
        let out_proj =
            LinearBias::new(out_weight, Tensor::new(vec![0.0; 4], vec![4]).unwrap()).unwrap();

        let q_proj = LinearBias::new(
            make_weight(d_model, d_model, 0.0),
            make_bias(d_model, 0.0),
        )
        .unwrap();
        let k_proj = Linear::new(make_weight(d_model, d_model, 0.0)).unwrap();

        let attn = AudioAttention::new(q_proj, k_proj, v_proj, out_proj, num_heads, head_dim)
            .unwrap();

        // Three distinct input rows: [1,0,0,0], [0,1,0,0], [0,0,1,0]
        // With identity V and zero Q/K, each output row = uniform avg of all V rows
        let x = Tensor::new(
            vec![
                1.0, 0.0, 0.0, 0.0, // pos 0
                0.0, 1.0, 0.0, 0.0, // pos 1
                0.0, 0.0, 1.0, 0.0, // pos 2
            ],
            vec![seq_len, d_model],
        )
        .unwrap();

        let out = attn.forward(&x).unwrap();
        assert_eq!(out.shape(), &[seq_len, d_model]);

        let data = out.to_vec_f32();

        // Expected: uniform average of v rows [1,0,0,0], [0,1,0,0], [0,0,1,0]
        // avg = [1/3, 1/3, 1/3, 0]
        let expected_row = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0];
        let tol = 1e-5;

        for pos in 0..seq_len {
            for col in 0..d_model {
                assert_close(
                    data[pos * d_model + col],
                    expected_row[col],
                    tol,
                    &format!("bidirectional: pos={} col={}", pos, col),
                );
            }
        }
    }

    /// AudioAttention must reject num_heads * head_dim != d_model.
    #[test]
    fn test_audio_attention_invalid_head_config() {
        let d_model = 8;
        let mk_lb = |r: usize, c: usize| {
            LinearBias::new(make_weight(r, c, 0.0), make_bias(r, 0.0)).unwrap()
        };
        let mk_l = |r: usize, c: usize| Linear::new(make_weight(r, c, 0.0)).unwrap();

        // num_heads=3, head_dim=4 => 12 != 8: must fail
        let result = AudioAttention::new(
            mk_lb(d_model, d_model),
            mk_l(d_model, d_model),
            mk_lb(d_model, d_model),
            mk_lb(d_model, d_model),
            3,  // num_heads
            4,  // head_dim; 3*4=12 != 8
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            LludaError::Model(_) => {}
            other => panic!("Expected Model error, got: {:?}", other),
        }
    }
}
