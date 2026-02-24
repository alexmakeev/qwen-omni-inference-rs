//! Q8_0 quantization for efficient weight storage and inference.
//!
//! # Overview
//!
//! Q8_0 is a block quantization format where each block of 32 f32 values
//! is represented as:
//! - One FP16 scale factor (IEEE 754 half-precision, 2 bytes)
//! - 32 signed 8-bit integer quantized values (32 bytes)
//!
//! Total: 34 bytes per block vs 128 bytes for f32 — a 3.76x compression ratio.
//!
//! # Design principles
//!
//! - **No dequantization during inference.** The `block_dot_f32` method computes
//!   the dot product directly from quantized values without materializing f32 weights.
//! - **FP16 scale, not BF16.** GGML uses IEEE 754 FP16 for scale factors.
//!   BF16 has different bit layout (8 exponent + 7 mantissa vs FP16's 5 + 10).
//! - **Trait-based.** `QuantBlock` is monomorphized at compile time — zero runtime
//!   dispatch cost, enabling the compiler to vectorize inner loops.
//!
//! # Example
//!
//! ```rust
//! use lluda_inference::quant::{Q8Block, QuantBlock, quantize_f32_to_q8, matmul_f32_x_quant};
//!
//! let weights: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
//! let blocks = quantize_f32_to_q8(&weights);
//! assert_eq!(blocks.len(), 2); // 64 values / 32 per block
//! ```

// ─────────────────────────────────────────────────────────────────────────────
// QuantBlock trait
// ─────────────────────────────────────────────────────────────────────────────

/// Trait for quantized block types. Monomorphized at compile time — zero runtime dispatch cost.
///
/// Implement this trait for each quantization format (Q8_0, Q4_0, etc.).
/// All implementations share the same block size of 32 values to align with GGML.
pub trait QuantBlock: Sized + Send + Sync + 'static {
    /// Number of values per block (32 for Q8_0 and Q4_0).
    const BLOCK_VALUES: usize;

    /// Bytes per serialized block (34 for Q8_0, 18 for Q4_0).
    const BLOCK_BYTES: usize;

    /// Human-readable name for this quantization format.
    const DTYPE_NAME: &'static str;

    /// Quantize a block of f32 values to this format.
    fn quantize(values: &[f32; 32]) -> Self;

    /// Get the FP16 scale as f32.
    fn scale(&self) -> f32;

    /// Fused dot product: `sum(quants[i] * activations[i]) * scale`.
    ///
    /// This is the PRIMARY compute operation during inference.
    /// No intermediate f32 weight array is materialized — fused into a single loop.
    fn block_dot_f32(&self, activations: &[f32; 32]) -> f32;

    /// Dequantize to f32.
    ///
    /// **ONLY for testing and debugging, NOT for inference.**
    /// Inference uses `block_dot_f32` to avoid materializing full f32 weight tensors.
    fn dequantize(&self) -> [f32; 32];
}

// ─────────────────────────────────────────────────────────────────────────────
// Q8Block struct
// ─────────────────────────────────────────────────────────────────────────────

/// A single Q8_0 quantized block: 32 signed 8-bit integers with one FP16 scale.
///
/// Layout (34 bytes total, naturally aligned — no padding):
/// - Bytes 0–1: FP16 scale factor (IEEE 754 half-precision)
/// - Bytes 2–33: 32 signed 8-bit quantized values
///
/// The quantization formula is:
/// ```text
/// scale = max(|values[i]|) / 127.0
/// quants[i] = round(values[i] / scale), clamped to [-128, 127]
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Q8Block {
    /// FP16 scale factor (IEEE 754 half-precision, NOT BF16).
    pub scale_bits: u16,
    /// Quantized values, signed 8-bit integers.
    pub quants: [i8; 32],
}

// ─────────────────────────────────────────────────────────────────────────────
// FP16 helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Convert IEEE 754 half-precision float (FP16) to f32.
///
/// FP16 format (16 bits total):
/// - Bit 15: sign
/// - Bits 14–10: exponent (5 bits, bias = 15)
/// - Bits 9–0: mantissa (10 bits)
///
/// **Note:** This is IEEE 754 FP16, not BF16. BF16 has 8 exponent bits
/// and 7 mantissa bits. FP16 has 5 exponent bits and 10 mantissa bits.
pub fn fp16_to_f32(bits: u16) -> f32 {
    let sign: u32 = ((bits >> 15) & 1) as u32;
    let exp: u32 = ((bits >> 10) & 0x1F) as u32;
    let mantissa: u32 = (bits & 0x3FF) as u32;

    if exp == 0 {
        // Denormal (subnormal) or zero.
        // value = (-1)^sign * 2^(-14) * (mantissa / 1024)
        if mantissa == 0 {
            // Positive or negative zero.
            let f32_bits: u32 = sign << 31;
            f32::from_bits(f32_bits)
        } else {
            // Denormal FP16: value = (-1)^sign * mantissa * 2^(-14) / 1024
            // = (-1)^sign * mantissa * 2^(-24)
            // Normalize by finding the leading 1 bit in mantissa.
            let mut m = mantissa;
            let mut e: i32 = -14; // FP16 denormal exponent is 2^(-14)
            // Shift mantissa left until bit 10 is set (the implicit leading 1 position).
            while m & (1 << 10) == 0 {
                m <<= 1;
                e -= 1;
            }
            // Remove the implicit leading 1.
            m &= 0x3FF;
            // Convert to f32 biased exponent.
            let f32_exp = ((e + 127) as u32) & 0xFF;
            let f32_bits: u32 = (sign << 31) | (f32_exp << 23) | (m << 13);
            f32::from_bits(f32_bits)
        }
    } else if exp == 31 {
        // Infinity or NaN.
        if mantissa == 0 {
            // Infinity: map to f32 infinity (exponent all-ones, mantissa zero).
            let f32_bits: u32 = (sign << 31) | (0xFF << 23);
            f32::from_bits(f32_bits)
        } else {
            // NaN: map to f32 NaN, preserving mantissa bits (shifted left by 13).
            let f32_bits: u32 = (sign << 31) | (0xFF << 23) | (mantissa << 13);
            f32::from_bits(f32_bits)
        }
    } else {
        // Normal number.
        // value = (-1)^sign * 2^(exp - 15) * (1 + mantissa / 1024)
        // In f32: exponent bias is 127, so new_exp = exp - 15 + 127 = exp + 112
        let f32_exp: u32 = exp + 112;
        let f32_mantissa: u32 = mantissa << 13; // Extend 10-bit mantissa to 23 bits.
        let f32_bits: u32 = (sign << 31) | (f32_exp << 23) | f32_mantissa;
        f32::from_bits(f32_bits)
    }
}

/// Convert f32 to IEEE 754 half-precision float (FP16).
///
/// Handles all special cases: ±0, ±infinity, NaN, overflow to infinity,
/// underflow to denormal or zero. Uses round-to-nearest-even for normal values.
///
/// **Note:** This is IEEE 754 FP16, not BF16.
pub fn f32_to_fp16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign: u32 = (bits >> 31) & 1;
    let f32_exp: u32 = (bits >> 23) & 0xFF;
    let f32_mantissa: u32 = bits & 0x007F_FFFF;

    if f32_exp == 0xFF {
        // f32 infinity or NaN.
        if f32_mantissa == 0 {
            // Infinity.
            ((sign << 15) | (0x1F << 10)) as u16
        } else {
            // NaN: preserve top mantissa bit (quiet NaN signal), shift down.
            // Ensure result is still NaN (non-zero mantissa).
            let fp16_mantissa = (f32_mantissa >> 13) | 1; // force non-zero
            ((sign << 15) | (0x1F << 10) | fp16_mantissa) as u16
        }
    } else if f32_exp == 0 && f32_mantissa == 0 {
        // Zero.
        (sign << 15) as u16
    } else {
        // Normal or denormal f32.
        // FP16 exponent = f32_exp - 127 + 15 = f32_exp - 112
        // FP16 can represent exponents -14..15 (biased: 1..30).
        // f32_exp range for normal fp16: 113..142 (biased fp16 exponent 1..30)

        let new_exp = f32_exp as i32 - 112; // unbiased + fp16 bias = f32_exp - 127 + 15

        if new_exp >= 31 {
            // Overflow: result is infinity.
            ((sign << 15) | (0x1F << 10)) as u16
        } else if new_exp <= 0 {
            // Underflow: result is denormal FP16 or zero.
            if new_exp < -10 {
                // Too small even for denormal FP16.
                (sign << 15) as u16
            } else {
                // Denormal FP16: implicit leading 1 becomes explicit.
                // mantissa = (1 << 23 | f32_mantissa) >> (1 - new_exp + 13)
                // = (implicit_1_mantissa) >> (14 - new_exp)
                let shift = (1 - new_exp + 13) as u32; // shift = 14 - new_exp
                let implicit_mantissa = (1 << 23) | f32_mantissa;
                // Round to nearest even.
                let half = 1u32 << (shift - 1);
                let round_bit = implicit_mantissa & half;
                let sticky = implicit_mantissa & (half - 1);
                let truncated = implicit_mantissa >> shift;
                let fp16_mantissa = if round_bit != 0 && (sticky != 0 || (truncated & 1) != 0) {
                    truncated + 1
                } else {
                    truncated
                };
                ((sign << 15) | fp16_mantissa) as u16
            }
        } else {
            // Normal FP16.
            // Round mantissa from 23 bits to 10 bits (drop 13 bits).
            let round_bit = (f32_mantissa >> 12) & 1;
            let sticky = f32_mantissa & 0xFFF;
            let truncated = f32_mantissa >> 13;
            let fp16_mantissa = if round_bit != 0 && (sticky != 0 || (truncated & 1) != 0) {
                truncated + 1
            } else {
                truncated
            };
            // Check if rounding overflowed the mantissa (carry into exponent).
            if fp16_mantissa >= (1 << 10) {
                let new_exp_after_round = new_exp + 1;
                if new_exp_after_round >= 31 {
                    // Overflow to infinity.
                    ((sign << 15) | (0x1F << 10)) as u16
                } else {
                    ((sign << 15) | ((new_exp_after_round as u32) << 10)) as u16
                }
            } else {
                ((sign << 15) | ((new_exp as u32) << 10) | fp16_mantissa) as u16
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// QuantBlock implementation for Q8Block
// ─────────────────────────────────────────────────────────────────────────────

impl QuantBlock for Q8Block {
    const BLOCK_VALUES: usize = 32;
    const BLOCK_BYTES: usize = 34;
    const DTYPE_NAME: &'static str = "Q8_0";

    /// Quantize 32 f32 values into a Q8_0 block.
    ///
    /// The scale is derived from the absolute maximum value in the block:
    /// `scale = max(|values[i]|) / 127.0`
    fn quantize(values: &[f32; 32]) -> Self {
        let abs_max = values
            .iter()
            .fold(0.0f32, |acc, &v| acc.max(v.abs()));

        if abs_max == 0.0 {
            return Q8Block {
                scale_bits: 0,
                quants: [0i8; 32],
            };
        }

        let scale = abs_max / 127.0;
        let inv_scale = 1.0 / scale;

        let mut quants = [0i8; 32];
        for (i, &v) in values.iter().enumerate() {
            let q = (v * inv_scale).round() as i32;
            quants[i] = q.clamp(-128, 127) as i8;
        }

        Q8Block {
            scale_bits: f32_to_fp16(scale),
            quants,
        }
    }

    /// Return the FP16 scale as f32.
    fn scale(&self) -> f32 {
        fp16_to_f32(self.scale_bits)
    }

    /// Fused dot product without intermediate dequantization.
    ///
    /// Computes `sum(quants[i] * activations[i]) * scale` directly.
    /// No temporary f32 weight array is created — the compiler can vectorize this loop.
    fn block_dot_f32(&self, activations: &[f32; 32]) -> f32 {
        let mut acc = 0.0f32;
        for i in 0..32 {
            acc += (self.quants[i] as f32) * activations[i];
        }
        acc * self.scale()
    }

    /// Dequantize all 32 values to f32.
    ///
    /// **For testing only.** During inference, use `block_dot_f32` instead.
    fn dequantize(&self) -> [f32; 32] {
        let s = self.scale();
        let mut out = [0.0f32; 32];
        for i in 0..32 {
            out[i] = (self.quants[i] as f32) * s;
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch quantization helper
// ─────────────────────────────────────────────────────────────────────────────

/// Quantize a flat f32 slice to Q8 blocks.
///
/// If the slice length is not a multiple of 32, the last block is zero-padded.
///
/// # Example
///
/// ```rust
/// use lluda_inference::quant::quantize_f32_to_q8;
///
/// let values: Vec<f32> = (0..48).map(|i| i as f32 * 0.01).collect();
/// let blocks = quantize_f32_to_q8(&values);
/// assert_eq!(blocks.len(), 2); // ceil(48 / 32) = 2 blocks
/// ```
pub fn quantize_f32_to_q8(values: &[f32]) -> Vec<Q8Block> {
    let num_blocks = (values.len() + 31) / 32;
    let mut blocks = Vec::with_capacity(num_blocks);

    for b in 0..num_blocks {
        let start = b * 32;
        let end = (start + 32).min(values.len());
        let mut block_vals = [0.0f32; 32];
        block_vals[..end - start].copy_from_slice(&values[start..end]);
        // Remaining positions (if any) stay as 0.0 — the zero-padding.
        blocks.push(Q8Block::quantize(&block_vals));
    }

    blocks
}

// ─────────────────────────────────────────────────────────────────────────────
// Generic matmul
// ─────────────────────────────────────────────────────────────────────────────

/// Fused matmul: `x[M, K](F32) @ W[N, K/32 blocks](Quant) → out[M, N](F32)`.
///
/// `W` is stored pre-transposed as N rows of `K/32` quantized blocks each
/// (layout `[N, K]`). For each output element `out[i, j]`:
/// ```text
/// W row j has K/32 blocks
/// for each block b:
///     out[i, j] += block_dot_f32(x[i, b*32 .. b*32+32])
/// ```
///
/// # Arguments
///
/// - `x`: Row-major activations, shape `[M, K]`.
/// - `w`: Weight blocks, `N * (K/32)` total, N rows of `K/32` blocks each.
/// - `m`: Number of rows in `x`.
/// - `k`: Shared dimension. Must be a multiple of `B::BLOCK_VALUES` (32).
/// - `n`: Number of output columns (number of rows in the weight matrix).
///
/// # Returns
///
/// Row-major output, shape `[M, N]`.
///
/// # Panics
///
/// Panics if `k` is not a multiple of `B::BLOCK_VALUES`.
pub fn matmul_f32_x_quant<B: QuantBlock>(
    x: &[f32],
    w: &[B],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    assert!(
        k % B::BLOCK_VALUES == 0,
        "k ({}) must be a multiple of BLOCK_VALUES ({})",
        k,
        B::BLOCK_VALUES
    );

    let blocks_per_row = k / B::BLOCK_VALUES;
    let mut out = vec![0.0f32; m * n];

    for i in 0..m {
        let x_row = &x[i * k..(i + 1) * k];
        for j in 0..n {
            let w_blocks = &w[j * blocks_per_row..(j + 1) * blocks_per_row];
            let mut sum = 0.0f32;
            for (b_idx, block) in w_blocks.iter().enumerate() {
                let x_slice: &[f32; 32] = x_row[b_idx * 32..(b_idx + 1) * 32]
                    .try_into()
                    .unwrap();
                sum += block.block_dot_f32(x_slice);
            }
            out[i * n + j] = sum;
        }
    }

    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: relative error between two finite f32 values.
    fn rel_error(a: f32, b: f32) -> f32 {
        if a == 0.0 && b == 0.0 {
            return 0.0;
        }
        let denom = a.abs().max(b.abs());
        (a - b).abs() / denom
    }

    #[test]
    fn test_fp16_roundtrip() {
        // Normal FP16 values that roundtrip within 0.1% relative error.
        // Note: 5.96e-8 is right at the FP16 minimum denormal boundary where precision
        // is exactly 1 ULP, causing inherent >0.1% error. Use values clearly within
        // representable ranges.
        let values = [0.0f32, 1.0, -1.0, 0.5, 65504.0, 3.14];
        // Minimum positive FP16 normal: ~6.104e-5. A denormal safely in range:
        let denormal_test = fp16_to_f32(1u16); // The actual minimum FP16 denormal bit pattern.

        for &v in &values {
            let bits = f32_to_fp16(v);
            let recovered = fp16_to_f32(bits);

            if v == 0.0 {
                assert_eq!(recovered, 0.0,
                    "FP16 roundtrip for 0.0: expected 0.0, got {}", recovered);
            } else {
                let err = rel_error(v, recovered);
                assert!(
                    err < 0.001,
                    "FP16 roundtrip for {}: recovered {}, relative error {} > 0.1%",
                    v, recovered, err
                );
            }
        }

        // Roundtrip the minimum FP16 denormal (bit pattern 0x0001) — exact by construction.
        let bits_back = f32_to_fp16(denormal_test);
        let recovered = fp16_to_f32(bits_back);
        assert_eq!(
            recovered, denormal_test,
            "Minimum FP16 denormal roundtrip: expected {}, got {}",
            denormal_test, recovered
        );
    }

    #[test]
    fn test_fp16_special_values() {
        // Zero.
        let zero_bits = f32_to_fp16(0.0f32);
        let zero_recovered = fp16_to_f32(zero_bits);
        assert_eq!(zero_recovered, 0.0f32,
            "FP16 zero roundtrip failed: got {}", zero_recovered);

        // Positive infinity.
        let inf_bits = f32_to_fp16(f32::INFINITY);
        let inf_recovered = fp16_to_f32(inf_bits);
        assert!(inf_recovered.is_infinite() && inf_recovered > 0.0,
            "FP16 +inf roundtrip failed: got {}", inf_recovered);

        // Negative infinity.
        let neg_inf_bits = f32_to_fp16(f32::NEG_INFINITY);
        let neg_inf_recovered = fp16_to_f32(neg_inf_bits);
        assert!(neg_inf_recovered.is_infinite() && neg_inf_recovered < 0.0,
            "FP16 -inf roundtrip failed: got {}", neg_inf_recovered);

        // NaN.
        let nan_bits = f32_to_fp16(f32::NAN);
        let nan_recovered = fp16_to_f32(nan_bits);
        assert!(nan_recovered.is_nan(),
            "FP16 NaN roundtrip failed: got {}", nan_recovered);
    }

    #[test]
    fn test_q8_block_quantize_basic() {
        // Values in [-1, 1] range — typical neural network weights.
        let mut values = [0.0f32; 32];
        for (i, v) in values.iter_mut().enumerate() {
            *v = (i as f32 / 32.0) * 2.0 - 1.0; // linearly spaced in [-1, 1)
        }

        let block = Q8Block::quantize(&values);

        // Scale should be positive and reasonable.
        let scale = block.scale();
        assert!(scale > 0.0, "Scale should be positive, got {}", scale);
        assert!(scale < 2.0, "Scale should be < 2.0, got {}", scale);

        // Dequantized values should be close to originals.
        let deq = block.dequantize();
        let mut max_err = 0.0f32;
        for (i, (&orig, &rec)) in values.iter().zip(deq.iter()).enumerate() {
            let err = (orig - rec).abs();
            if err > max_err {
                max_err = err;
            }
            assert!(
                err < 0.02,
                "Index {}: original {}, recovered {}, error {} > 0.02",
                i, orig, rec, err
            );
        }
        // Max error < 0.02 (Q8 has ~0.008 step at scale ~1/127).
        assert!(max_err < 0.02,
            "Max dequantization error {} exceeds 0.02", max_err);
    }

    #[test]
    fn test_q8_block_zero() {
        let values = [0.0f32; 32];
        let block = Q8Block::quantize(&values);

        assert_eq!(block.scale_bits, 0, "Scale bits should be 0 for all-zero input");
        assert_eq!(block.scale(), 0.0f32, "Scale should be 0.0 for all-zero input");
        for (i, &q) in block.quants.iter().enumerate() {
            assert_eq!(q, 0, "Quant[{}] should be 0 for all-zero input", i);
        }
    }

    #[test]
    fn test_q8_block_roundtrip_quality() {
        // Deterministic pattern: sin(i) values in [-1, 1].
        let mut values = [0.0f32; 32];
        for (i, v) in values.iter_mut().enumerate() {
            *v = (i as f32).sin();
        }

        let block = Q8Block::quantize(&values);
        let deq = block.dequantize();

        // Compute mean squared error.
        let mse: f32 = values
            .iter()
            .zip(deq.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f32>()
            / 32.0;

        assert!(
            mse < 1e-4,
            "MSE {} exceeds 1e-4 for sin(i) roundtrip",
            mse
        );
    }

    #[test]
    fn test_q8_block_dot_f32() {
        // Create a block from known values and compute dot product.
        let mut weights = [0.0f32; 32];
        let mut activations = [0.0f32; 32];
        for i in 0..32 {
            weights[i] = (i as f32 + 1.0) / 32.0; // 1/32, 2/32, ..., 32/32
            activations[i] = 1.0 / (i as f32 + 1.0); // 1, 1/2, 1/3, ...
        }

        let block = Q8Block::quantize(&weights);

        // Reference dot product using full f32.
        let deq = block.dequantize();
        let ref_dot: f32 = deq.iter().zip(activations.iter()).map(|(&w, &a)| w * a).sum();

        let fused_dot = block.block_dot_f32(&activations);

        // Should match within 0.1% relative error.
        let err = rel_error(ref_dot, fused_dot);
        assert!(
            err < 0.001,
            "block_dot_f32 {} vs reference {}, relative error {} > 0.1%",
            fused_dot, ref_dot, err
        );
    }

    #[test]
    fn test_quantize_f32_to_q8_padding() {
        // 48 values = 1.5 blocks → should produce 2 blocks, last 16 quants zero-padded.
        let values: Vec<f32> = (0..48).map(|i| i as f32 * 0.01 + 0.1).collect();
        let blocks = quantize_f32_to_q8(&values);

        assert_eq!(blocks.len(), 2,
            "48 values should produce 2 blocks, got {}", blocks.len());

        // Last 16 quants of the second block should be zero (zero-padding).
        for i in 16..32 {
            assert_eq!(
                blocks[1].quants[i], 0,
                "Second block quants[{}] should be 0 (padding), got {}",
                i, blocks[1].quants[i]
            );
        }
    }

    #[test]
    fn test_matmul_f32_x_q8_identity() {
        // Build a 32x32 "identity-like" weight matrix.
        // W[i, j] = 1.0 if i == j, else 0.0.
        // With K=32 and N=32, each weight row has exactly 1 block.
        let n = 32usize;
        let k = 32usize;

        let mut w_f32 = vec![0.0f32; n * k];
        for i in 0..n {
            w_f32[i * k + i] = 1.0;
        }

        let w_blocks = quantize_f32_to_q8(&w_f32);

        let x: Vec<f32> = (0..k).map(|i| i as f32 + 1.0).collect();
        let out = matmul_f32_x_quant::<Q8Block>(&x, &w_blocks, 1, k, n);

        assert_eq!(out.len(), n);

        // Output should match input within tolerance (Q8 quantization error).
        for i in 0..n {
            let expected = x[i];
            let actual = out[i];
            let err = (expected - actual).abs();
            assert!(
                err < 0.1,
                "Output[{}]: expected {}, got {}, absolute error {} > 0.1",
                i, expected, actual, err
            );
        }
    }

    #[test]
    fn test_matmul_f32_x_q8_simple() {
        // M=2, K=64, N=3.
        let m = 2usize;
        let k = 64usize;
        let n = 3usize;

        // Deterministic weights and activations.
        let w_f32: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32 * 0.7 + 1.0) % 2.0) - 1.0)
            .collect();
        let x: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32 * 0.3 + 0.5) % 1.5) - 0.75)
            .collect();

        // Quantize weights.
        let w_blocks = quantize_f32_to_q8(&w_f32);

        // Compute output with quantized matmul.
        let out_quant = matmul_f32_x_quant::<Q8Block>(&x, &w_blocks, m, k, n);

        // Reference: dequantize weights and compute exact f32 matmul.
        let mut w_deq = vec![0.0f32; n * k];
        for (b_idx, block) in w_blocks.iter().enumerate() {
            let deq = block.dequantize();
            let start = b_idx * 32;
            w_deq[start..start + 32].copy_from_slice(&deq);
        }
        let mut ref_out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += x[i * k + l] * w_deq[j * k + l];
                }
                ref_out[i * n + j] = sum;
            }
        }

        // Compare with 0.5% relative error tolerance.
        for idx in 0..m * n {
            let err = rel_error(ref_out[idx], out_quant[idx]);
            assert!(
                err < 0.005,
                "Output[{}]: ref {}, quant {}, relative error {} > 0.5%",
                idx, ref_out[idx], out_quant[idx], err
            );
        }
    }

    #[test]
    fn test_fp16_denormal_values() {
        // FP16 denormal: exponent=0, mantissa=m → value = m * 2^(-24)
        // Minimum denormal: mantissa=1 → 2^(-24) ≈ 5.96e-8
        let min_denorm = fp16_to_f32(0x0001);
        assert!((min_denorm - 5.960464e-8).abs() < 1e-12, "min denormal: {}", min_denorm);

        // Mantissa=512 → 512 * 2^(-24) = 2^(-15) ≈ 3.0517578e-5
        let mid_denorm = fp16_to_f32(0x0200);
        let expected = 512.0 * f32::powi(2.0, -24);
        assert!((mid_denorm - expected).abs() / expected < 1e-6,
                "mid denormal: got {}, expected {}", mid_denorm, expected);

        // Mantissa=1023 (max denormal) → 1023 * 2^(-24) ≈ 6.097555e-5
        let max_denorm = fp16_to_f32(0x03FF);
        let expected_max = 1023.0 * f32::powi(2.0, -24);
        assert!((max_denorm - expected_max).abs() / expected_max < 1e-6,
                "max denormal: got {}, expected {}", max_denorm, expected_max);

        // Negative denormal
        let neg_denorm = fp16_to_f32(0x8200); // sign=1, exp=0, mantissa=512
        assert!((neg_denorm + expected).abs() / expected < 1e-6,
                "neg denormal: got {}, expected {}", neg_denorm, -expected);

        // Roundtrip: f32 → fp16 → f32 for denormal values
        let small_val = 3.0e-5_f32;
        let fp16_bits = f32_to_fp16(small_val);
        let roundtrip = fp16_to_f32(fp16_bits);
        assert!((roundtrip - small_val).abs() / small_val < 0.05,
                "denormal roundtrip: {} → bits {:04x} → {}", small_val, fp16_bits, roundtrip);
    }

    #[test]
    fn test_matmul_f32_x_q8_larger() {
        // M=4, K=128, N=8.
        let m = 4usize;
        let k = 128usize;
        let n = 8usize;

        let w_f32: Vec<f32> = (0..n * k)
            .map(|i| {
                let phase = i as f32 * 0.13;
                phase.sin() * 0.5
            })
            .collect();
        let x: Vec<f32> = (0..m * k)
            .map(|i| {
                let phase = i as f32 * 0.17 + 0.5;
                phase.cos() * 0.8
            })
            .collect();

        let w_blocks = quantize_f32_to_q8(&w_f32);
        let out_quant = matmul_f32_x_quant::<Q8Block>(&x, &w_blocks, m, k, n);

        // Reference f32 matmul using dequantized weights.
        let mut w_deq = vec![0.0f32; n * k];
        for (b_idx, block) in w_blocks.iter().enumerate() {
            let deq = block.dequantize();
            let start = b_idx * 32;
            if start + 32 <= w_deq.len() {
                w_deq[start..start + 32].copy_from_slice(&deq);
            }
        }
        let mut ref_out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += x[i * k + l] * w_deq[j * k + l];
                }
                ref_out[i * n + j] = sum;
            }
        }

        // Should be close — use 1% tolerance for the larger matrix.
        for idx in 0..m * n {
            let err = rel_error(ref_out[idx], out_quant[idx]);
            assert!(
                err < 0.01,
                "Output[{}]: ref {}, quant {}, relative error {} > 1%",
                idx, ref_out[idx], out_quant[idx], err
            );
        }
    }
}
