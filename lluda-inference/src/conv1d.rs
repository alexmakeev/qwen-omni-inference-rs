//! 1-D Convolutional layer.
//!
//! A standard Conv1d layer with configurable stride and zero-padding.
//! Used in audio encoder convolutional stems, e.g. the two-layer stem
//! in Qwen-Audio that downsamples log-mel spectrogram features before
//! feeding them into the transformer:
//!
//! - `conv1`: weight `[1280, 128, 3]`, bias `[1280]`, stride=1, padding=1
//! - `conv2`: weight `[1280, 1280, 3]`, bias `[1280]`, stride=2, padding=1
//!
//! # Algorithm
//!
//! ```text
//! output[c_out][t] = bias[c_out]
//!   + sum_{c_in} sum_{k} weight[c_out][c_in][k]
//!                        * input[c_in][t * stride + k - padding]
//! ```
//!
//! Input positions that fall outside the sequence boundary are treated as 0
//! (zero-padding).  The output length follows the standard formula:
//!
//! ```text
//! out_len = (seq_len + 2 * padding - kernel_size) / stride + 1
//! ```
//!
//! # Usage
//!
//! Weights are always stored and computed in F32 (not quantized).
//! Both 2-D inputs `[C_in, T]` and 3-D batched inputs `[B, C_in, T]`
//! are accepted; the output shape mirrors the input rank.
//!
//! # Reference
//!
//! LeCun et al. (1989): "Backpropagation Applied to Handwritten Zip Code Recognition"
//! (original convolutional layer reference)

use crate::error::{LludaError, Result};
use crate::tensor::Tensor;

/// 1-D Convolutional layer.
///
/// Applies a 1-D convolution over an input sequence.
/// Both 2-D `[C_in, T]` and 3-D `[B, C_in, T]` inputs are supported.
#[derive(Debug, Clone)]
pub struct Conv1d {
    /// Convolution kernel, shape `[out_channels, in_channels, kernel_size]`.
    weight: Tensor,
    /// Per-output-channel bias, shape `[out_channels]`.
    bias: Tensor,
    /// Step size between successive application positions.
    stride: usize,
    /// Number of zeros appended on each side of the time axis.
    padding: usize,
}

impl Conv1d {
    /// Create a new Conv1d layer.
    ///
    /// # Arguments
    ///
    /// * `weight` - Kernel tensor of shape `[out_channels, in_channels, kernel_size]`
    /// * `bias`   - Bias tensor of shape `[out_channels]`
    /// * `stride` - Step size between kernel application positions (≥ 1)
    /// * `padding` - Number of zeros prepended/appended on each side of the input
    ///
    /// # Returns
    ///
    /// New Conv1d instance.
    ///
    /// # Errors
    ///
    /// Returns `ShapeMismatch` if:
    /// - `weight` is not a 3-D tensor
    /// - `bias` is not a 1-D tensor
    /// - `bias` length does not equal `weight.shape()[0]` (out_channels)
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::conv1d::Conv1d;
    /// use lluda_inference::tensor::Tensor;
    ///
    /// // kernel: 8 output channels, 4 input channels, kernel size 3
    /// let weight = Tensor::new(vec![0.0; 8 * 4 * 3], vec![8, 4, 3]).unwrap();
    /// let bias   = Tensor::new(vec![0.0; 8], vec![8]).unwrap();
    /// let conv   = Conv1d::new(weight, bias, 1, 1).unwrap();
    /// ```
    pub fn new(weight: Tensor, bias: Tensor, stride: usize, padding: usize) -> Result<Self> {
        // Weight must be 3-D: [out_channels, in_channels, kernel_size]
        if weight.ndim() != 3 {
            return Err(LludaError::ShapeMismatch {
                expected: vec![0, 0, 0], // Indicate 3-D expectation
                got: weight.shape().to_vec(),
            });
        }

        // Bias must be 1-D: [out_channels]
        if bias.ndim() != 1 {
            return Err(LludaError::ShapeMismatch {
                expected: vec![0], // Indicate 1-D expectation
                got: bias.shape().to_vec(),
            });
        }

        // Bias length must equal out_channels
        let out_channels = weight.shape()[0];
        let bias_len = bias.shape()[0];
        if bias_len != out_channels {
            return Err(LludaError::ShapeMismatch {
                expected: vec![out_channels],
                got: vec![bias_len],
            });
        }

        Ok(Conv1d { weight, bias, stride, padding })
    }

    /// Apply the 1-D convolution to an input tensor.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape `[C_in, T]` or `[B, C_in, T]`
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[C_out, T_out]` or `[B, C_out, T_out]` where
    /// `T_out = (T + 2 * padding - kernel_size) / stride + 1`.
    ///
    /// # Errors
    ///
    /// Returns `ShapeMismatch` if:
    /// - Input is not 2-D or 3-D
    /// - In-channels dimension of input does not match `weight.shape()[1]`
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::conv1d::Conv1d;
    /// use lluda_inference::tensor::Tensor;
    ///
    /// // Identity-like kernel: [0, 1, 0] passes the input through unchanged.
    /// let weight = Tensor::new(
    ///     vec![0.0, 1.0, 0.0,  // out_channel 0 ← in_channel 0
    ///          0.0, 0.0, 0.0,  // out_channel 0 ← in_channel 1 (zero)
    ///          0.0, 0.0, 0.0,  // out_channel 1 ← in_channel 0 (zero)
    ///          0.0, 1.0, 0.0], // out_channel 1 ← in_channel 1
    ///     vec![2, 2, 3],
    /// ).unwrap();
    /// let bias  = Tensor::new(vec![0.0; 2], vec![2]).unwrap();
    /// let conv  = Conv1d::new(weight, bias, 1, 1).unwrap();
    ///
    /// let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    /// let y = conv.forward(&x).unwrap();
    /// assert_eq!(y.shape(), &[2, 3]);
    /// ```
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match x.ndim() {
            2 => self.forward_2d(x),
            3 => self.forward_3d(x),
            other => Err(LludaError::ShapeMismatch {
                expected: vec![0, 0],   // 2-D or 3-D expected
                got: vec![other],
            }),
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Convolution for a single sample `[C_in, T]` → `[C_out, T_out]`.
    fn forward_2d(&self, x: &Tensor) -> Result<Tensor> {
        let out_channels = self.weight.shape()[0];
        let in_channels  = self.weight.shape()[1];
        let kernel_size  = self.weight.shape()[2];

        let x_in_channels = x.shape()[0];
        let seq_len       = x.shape()[1];

        // Validate in-channels
        if x_in_channels != in_channels {
            return Err(LludaError::ShapeMismatch {
                expected: vec![in_channels],
                got: vec![x_in_channels],
            });
        }

        let out_len = (seq_len + 2 * self.padding - kernel_size) / self.stride + 1;

        let x_data = x.to_vec_f32();
        let w_data = self.weight.to_vec_f32();
        let b_data = self.bias.to_vec_f32();

        let mut output = vec![0.0f32; out_channels * out_len];

        for co in 0..out_channels {
            for t in 0..out_len {
                let mut sum = b_data[co];
                for ci in 0..in_channels {
                    for k in 0..kernel_size {
                        let input_t = (t * self.stride + k) as isize - self.padding as isize;
                        if input_t >= 0 && (input_t as usize) < seq_len {
                            sum += w_data[co * in_channels * kernel_size + ci * kernel_size + k]
                                * x_data[ci * seq_len + input_t as usize];
                        }
                    }
                }
                output[co * out_len + t] = sum;
            }
        }

        Tensor::new(output, vec![out_channels, out_len])
    }

    /// Convolution for a batched input `[B, C_in, T]` → `[B, C_out, T_out]`.
    fn forward_3d(&self, x: &Tensor) -> Result<Tensor> {
        let batch_size    = x.shape()[0];
        let out_channels  = self.weight.shape()[0];
        let in_channels   = self.weight.shape()[1];
        let kernel_size   = self.weight.shape()[2];

        let x_in_channels = x.shape()[1];
        let seq_len       = x.shape()[2];

        // Validate in-channels
        if x_in_channels != in_channels {
            return Err(LludaError::ShapeMismatch {
                expected: vec![in_channels],
                got: vec![x_in_channels],
            });
        }

        let out_len = (seq_len + 2 * self.padding - kernel_size) / self.stride + 1;

        let x_data = x.to_vec_f32();
        let w_data = self.weight.to_vec_f32();
        let b_data = self.bias.to_vec_f32();

        // Stride for one sample in the input: C_in * T
        let x_sample_stride = in_channels * seq_len;
        // Stride for one sample in the output: C_out * T_out
        let out_sample_stride = out_channels * out_len;

        let mut output = vec![0.0f32; batch_size * out_channels * out_len];

        for b in 0..batch_size {
            let x_base  = b * x_sample_stride;
            let out_base = b * out_sample_stride;

            for co in 0..out_channels {
                for t in 0..out_len {
                    let mut sum = b_data[co];
                    for ci in 0..in_channels {
                        for k in 0..kernel_size {
                            let input_t =
                                (t * self.stride + k) as isize - self.padding as isize;
                            if input_t >= 0 && (input_t as usize) < seq_len {
                                sum += w_data
                                    [co * in_channels * kernel_size + ci * kernel_size + k]
                                    * x_data[x_base + ci * seq_len + input_t as usize];
                            }
                        }
                    }
                    output[out_base + co * out_len + t] = sum;
                }
            }
        }

        Tensor::new(output, vec![batch_size, out_channels, out_len])
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

    const F32_TOL: f32 = 1e-5;

    // -----------------------------------------------------------------------
    // test_conv1d_basic
    // -----------------------------------------------------------------------
    #[test]
    fn test_conv1d_basic() {
        // Input: [2, 4], kernel_size=3, stride=1, padding=1
        // out_len = (4 + 2*1 - 3) / 1 + 1 = 4
        //
        // weight shape: [1, 2, 3]  (1 out_channel, 2 in_channels, kernel 3)
        // weight[0, 0, :] = [1, 0, 0]  (only centre tap of channel 0 — but left)
        // weight[0, 1, :] = [0, 1, 0]  (only centre tap of channel 1)
        // bias[0] = 0.5
        //
        // input channel 0: [1, 2, 3, 4]
        // input channel 1: [5, 6, 7, 8]
        //
        // For t=0, padding=1, stride=1:
        //   k=0 → input_t = 0*1+0-1 = -1  → 0 (pad)
        //   k=1 → input_t = 0*1+1-1 =  0  → w[0,0,1]*x[0,0] + w[0,1,1]*x[1,0]
        //        = 0*1 + 1*5 = 5
        //   k=2 → input_t = 0*1+2-1 =  1  → w[0,0,2]*x[0,1] + w[0,1,2]*x[1,1]
        //        = 0*2 + 0*6 = 0
        // out[0] = 0.5 + 5 = 5.5
        let weight = Tensor::new(
            vec![
                1.0f32, 0.0, 0.0, // out_ch 0, in_ch 0
                0.0, 1.0, 0.0,    // out_ch 0, in_ch 1
            ],
            vec![1, 2, 3],
        )
        .unwrap();
        let bias = Tensor::new(vec![0.5], vec![1]).unwrap();
        let conv = Conv1d::new(weight, bias, 1, 1).unwrap();

        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0,   // channel 0
                 5.0, 6.0, 7.0, 8.0],  // channel 1
            vec![2, 4],
        )
        .unwrap();
        let y = conv.forward(&x).unwrap();

        assert_eq!(y.shape(), &[1, 4]);
        let data = y.to_vec_f32();

        // t=0: w[0,0,0]*pad + w[0,0,1]*x[0,0] + w[0,0,2]*x[0,1]
        //    + w[0,1,0]*pad + w[0,1,1]*x[1,0] + w[0,1,2]*x[1,1] + bias
        // = 1*0 + 0*1 + 0*2 + 0*0 + 1*5 + 0*6 + 0.5 = 5.5
        assert_close(data[0], 5.5, F32_TOL, "t=0");
        // t=1: w[0,0,0]*x[0,0] + w[0,0,1]*x[0,1] + w[0,0,2]*x[0,2]
        //    + w[0,1,0]*x[1,0] + w[0,1,1]*x[1,1] + w[0,1,2]*x[1,2] + bias
        // = 1*1 + 0*2 + 0*3 + 0*5 + 1*6 + 0*7 + 0.5 = 7.5
        assert_close(data[1], 7.5, F32_TOL, "t=1");
        // t=2: 1*2 + 0*3 + 0*4 + 0*6 + 1*7 + 0*8 + 0.5 = 9.5
        assert_close(data[2], 9.5, F32_TOL, "t=2");
        // t=3: 1*3 + 0*4 + 0*pad + 0*7 + 1*8 + 0*pad + 0.5 = 11.5
        assert_close(data[3], 11.5, F32_TOL, "t=3");
    }

    // -----------------------------------------------------------------------
    // test_conv1d_stride2
    // -----------------------------------------------------------------------
    #[test]
    fn test_conv1d_stride2() {
        // Input: [1, 8], stride=2, padding=1, kernel_size=3
        // out_len = (8 + 2*1 - 3) / 2 + 1 = 7/2 + 1 = 4
        let weight = Tensor::new(vec![1.0f32, 0.0, 0.0], vec![1, 1, 3]).unwrap();
        let bias   = Tensor::new(vec![0.0], vec![1]).unwrap();
        let conv   = Conv1d::new(weight, bias, 2, 1).unwrap();

        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![1, 8],
        )
        .unwrap();
        let y = conv.forward(&x).unwrap();

        assert_eq!(y.shape(), &[1, 4], "output length must equal 4 for stride=2");
    }

    // -----------------------------------------------------------------------
    // test_conv1d_shape_validation
    // -----------------------------------------------------------------------
    #[test]
    fn test_conv1d_shape_validation() {
        // 2-D weight must be rejected (need 3-D)
        let weight_2d = Tensor::new(vec![1.0; 6], vec![2, 3]).unwrap();
        let bias      = Tensor::new(vec![0.0; 2], vec![2]).unwrap();
        let result    = Conv1d::new(weight_2d, bias, 1, 0);
        assert!(result.is_err(), "2-D weight must be rejected");
        match result.unwrap_err() {
            LludaError::ShapeMismatch { got, .. } => {
                assert_eq!(got, vec![2, 3]);
            }
            other => panic!("Expected ShapeMismatch, got: {:?}", other),
        }

        // 2-D bias must be rejected (need 1-D)
        let weight_ok = Tensor::new(vec![0.0; 2 * 3 * 3], vec![2, 3, 3]).unwrap();
        let bias_2d   = Tensor::new(vec![0.0; 6], vec![2, 3]).unwrap();
        let result    = Conv1d::new(weight_ok, bias_2d, 1, 0);
        assert!(result.is_err(), "2-D bias must be rejected");

        // Mismatched out_channels: weight has 2, bias has 3
        let weight_oc2 = Tensor::new(vec![0.0; 2 * 3 * 3], vec![2, 3, 3]).unwrap();
        let bias_oc3   = Tensor::new(vec![0.0; 3], vec![3]).unwrap();
        let result     = Conv1d::new(weight_oc2, bias_oc3, 1, 0);
        assert!(result.is_err(), "bias/weight out_channel mismatch must be rejected");
        match result.unwrap_err() {
            LludaError::ShapeMismatch { expected, got } => {
                assert_eq!(expected, vec![2]);
                assert_eq!(got, vec![3]);
            }
            other => panic!("Expected ShapeMismatch, got: {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // test_conv1d_identity_kernel
    // -----------------------------------------------------------------------
    #[test]
    fn test_conv1d_identity_kernel() {
        // Kernel [0, 1, 0] with stride=1, padding=1 is the identity operation:
        // output[c][t] = input[c][t]  for all c, t.
        //
        // weight shape: [2, 2, 3]
        // weight[co, ci, :] = [0, 1, 0] if co==ci, else [0, 0, 0]
        let weight = Tensor::new(
            vec![
                0.0, 1.0, 0.0,  // co=0, ci=0  ← identity tap
                0.0, 0.0, 0.0,  // co=0, ci=1
                0.0, 0.0, 0.0,  // co=1, ci=0
                0.0, 1.0, 0.0,  // co=1, ci=1  ← identity tap
            ],
            vec![2, 2, 3],
        )
        .unwrap();
        let bias = Tensor::new(vec![0.0, 0.0], vec![2]).unwrap();
        let conv = Conv1d::new(weight, bias, 1, 1).unwrap();

        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0,
                 5.0, 6.0, 7.0, 8.0],
            vec![2, 4],
        )
        .unwrap();
        let y = conv.forward(&x).unwrap();

        assert_eq!(y.shape(), &[2, 4]);
        let data = y.to_vec_f32();

        // Channel 0 must match input channel 0
        assert_close(data[0], 1.0, F32_TOL, "ch0, t=0");
        assert_close(data[1], 2.0, F32_TOL, "ch0, t=1");
        assert_close(data[2], 3.0, F32_TOL, "ch0, t=2");
        assert_close(data[3], 4.0, F32_TOL, "ch0, t=3");

        // Channel 1 must match input channel 1
        assert_close(data[4], 5.0, F32_TOL, "ch1, t=0");
        assert_close(data[5], 6.0, F32_TOL, "ch1, t=1");
        assert_close(data[6], 7.0, F32_TOL, "ch1, t=2");
        assert_close(data[7], 8.0, F32_TOL, "ch1, t=3");
    }

    // -----------------------------------------------------------------------
    // test_conv1d_output_length
    // -----------------------------------------------------------------------
    #[test]
    fn test_conv1d_output_length() {
        // Verify the formula: out_len = (seq + 2*pad - kernel) / stride + 1
        // for several (seq, kernel, stride, padding) combinations.

        let cases: &[(usize, usize, usize, usize, usize)] = &[
            // (seq, kernel, stride, padding, expected_out)
            (8, 3, 1, 1, 8),   // common: same-padding with stride 1
            (8, 3, 2, 1, 4),   // stride 2 halves the sequence
            (7, 3, 1, 0, 5),   // no padding, reduces length by (kernel-1)
            (10, 5, 2, 2, 5),  // larger kernel with stride 2
            (100, 3, 1, 1, 100), // long sequence, same padding
        ];

        for &(seq, kernel, stride, pad, expected_out) in cases {
            let out_channels = 1usize;
            let in_channels  = 1usize;

            let weight = Tensor::new(
                vec![0.0f32; out_channels * in_channels * kernel],
                vec![out_channels, in_channels, kernel],
            )
            .unwrap();
            let bias = Tensor::new(vec![0.0f32; out_channels], vec![out_channels]).unwrap();
            let conv = Conv1d::new(weight, bias, stride, pad).unwrap();

            let x = Tensor::new(vec![1.0f32; in_channels * seq], vec![in_channels, seq]).unwrap();
            let y = conv.forward(&x).unwrap();

            assert_eq!(
                y.shape()[1],
                expected_out,
                "seq={seq}, kernel={kernel}, stride={stride}, pad={pad}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // test_conv1d_3d_batch
    // -----------------------------------------------------------------------
    #[test]
    fn test_conv1d_3d_batch() {
        // Verify that a batched 3-D input [B=2, C_in=1, T=4] is processed
        // independently per batch element and produces [B=2, C_out=1, T_out=4].
        //
        // Use the identity kernel [0, 1, 0] so the expected output equals input.
        let weight = Tensor::new(vec![0.0, 1.0, 0.0], vec![1, 1, 3]).unwrap();
        let bias   = Tensor::new(vec![0.0], vec![1]).unwrap();
        let conv   = Conv1d::new(weight, bias, 1, 1).unwrap();

        // batch 0: [10, 20, 30, 40], batch 1: [1, 2, 3, 4]
        let x = Tensor::new(
            vec![10.0, 20.0, 30.0, 40.0,  // batch 0, channel 0
                  1.0,  2.0,  3.0,  4.0], // batch 1, channel 0
            vec![2, 1, 4],
        )
        .unwrap();
        let y = conv.forward(&x).unwrap();

        assert_eq!(y.shape(), &[2, 1, 4]);
        let data = y.to_vec_f32();

        // Batch 0
        assert_close(data[0], 10.0, F32_TOL, "b0, t=0");
        assert_close(data[1], 20.0, F32_TOL, "b0, t=1");
        assert_close(data[2], 30.0, F32_TOL, "b0, t=2");
        assert_close(data[3], 40.0, F32_TOL, "b0, t=3");

        // Batch 1
        assert_close(data[4],  1.0, F32_TOL, "b1, t=0");
        assert_close(data[5],  2.0, F32_TOL, "b1, t=1");
        assert_close(data[6],  3.0, F32_TOL, "b1, t=2");
        assert_close(data[7],  4.0, F32_TOL, "b1, t=3");
    }
}
