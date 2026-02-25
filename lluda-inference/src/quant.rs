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

    /// Dequantize block to f32 values.
    /// Used for dtype conversion (Q8→F32) and testing.
    /// Inference should use `block_dot_f32` for fused computation.
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
// Q4Block struct
// ─────────────────────────────────────────────────────────────────────────────

/// Q4_0 quantized block: 32 values in 18 bytes.
/// Format: 2-byte FP16 scale + 16 bytes of packed 4-bit unsigned nibbles.
/// Each nibble stores round(value/scale) + 8, so effective signed range is [-8, 7].
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Q4Block {
    /// FP16 scale factor (IEEE 754 half-precision)
    pub scale_bits: u16,
    /// Packed nibbles: each byte holds 2 values
    /// byte[i] low nibble (bits 0-3) = quant[2*i]
    /// byte[i] high nibble (bits 4-7) = quant[2*i+1]
    pub nibs: [u8; 16],
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

    /// Dequantize block to f32 values.
    /// Used for dtype conversion (Q8→F32) and testing.
    /// Inference should use `block_dot_f32` for fused computation.
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
// QuantBlock implementation for Q4Block
// ─────────────────────────────────────────────────────────────────────────────

impl QuantBlock for Q4Block {
    const BLOCK_VALUES: usize = 32;
    const BLOCK_BYTES: usize = 18;
    const DTYPE_NAME: &'static str = "Q4_0";

    /// Quantize 32 f32 values into a Q4_0 block.
    ///
    /// The scale is derived from the absolute maximum value in the block:
    /// `scale = max(|values[i]|) / 8.0`
    ///
    /// Values are offset by 8 before packing so they fit in [0, 15]:
    /// `nibble = round(value / scale) + 8`, clamped to [0, 15]
    fn quantize(values: &[f32; 32]) -> Self {
        let abs_max = values
            .iter()
            .fold(0.0f32, |acc, &v| acc.max(v.abs()));

        if abs_max == 0.0 {
            // All zeros: nibbles = 0x88 (both nibbles = 8 = zero offset), scale = 0.
            return Q4Block {
                scale_bits: 0,
                nibs: [0x88u8; 16],
            };
        }

        let scale = abs_max / 8.0;
        let inv_scale = 1.0 / scale;

        let mut nibs = [0u8; 16];
        for i in 0..16 {
            let q0 = ((values[2 * i] * inv_scale).round() as i32 + 8).clamp(0, 15) as u8;
            let q1 = ((values[2 * i + 1] * inv_scale).round() as i32 + 8).clamp(0, 15) as u8;
            nibs[i] = q0 | (q1 << 4);
        }

        Q4Block {
            scale_bits: f32_to_fp16(scale),
            nibs,
        }
    }

    /// Return the FP16 scale as f32.
    fn scale(&self) -> f32 {
        fp16_to_f32(self.scale_bits)
    }

    /// Fused dot product without intermediate dequantization.
    ///
    /// Computes `sum((nibble[i] - 8) * activations[i]) * scale` directly.
    /// No temporary f32 weight array is created — the compiler can vectorize this loop.
    fn block_dot_f32(&self, activations: &[f32; 32]) -> f32 {
        let s = self.scale();
        let mut acc = 0.0f32;
        for i in 0..16 {
            let byte = self.nibs[i];
            let q0 = (byte & 0x0F) as i32 - 8; // signed: [-8, 7]
            let q1 = ((byte >> 4) & 0x0F) as i32 - 8;
            acc += q0 as f32 * activations[2 * i];
            acc += q1 as f32 * activations[2 * i + 1];
        }
        acc * s
    }

    /// Dequantize block to f32 values.
    /// Used for dtype conversion (Q4→F32) and testing.
    /// Inference should use `block_dot_f32` for fused computation.
    fn dequantize(&self) -> [f32; 32] {
        let s = self.scale();
        let mut out = [0.0f32; 32];
        for i in 0..16 {
            let byte = self.nibs[i];
            let q0 = (byte & 0x0F) as i32 - 8;
            let q1 = ((byte >> 4) & 0x0F) as i32 - 8;
            out[2 * i] = q0 as f32 * s;
            out[2 * i + 1] = q1 as f32 * s;
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

/// Quantize a flat f32 slice to Q4_0 blocks.
///
/// If the slice length is not a multiple of 32, the last block is zero-padded.
///
/// # Example
///
/// ```rust
/// use lluda_inference::quant::quantize_f32_to_q4;
///
/// let values: Vec<f32> = (0..48).map(|i| i as f32 * 0.01).collect();
/// let blocks = quantize_f32_to_q4(&values);
/// assert_eq!(blocks.len(), 2); // ceil(48 / 32) = 2 blocks
/// ```
pub fn quantize_f32_to_q4(values: &[f32]) -> Vec<Q4Block> {
    let block_size = Q4Block::BLOCK_VALUES;
    let n_blocks = (values.len() + block_size - 1) / block_size;
    let mut blocks = Vec::with_capacity(n_blocks);
    for b in 0..n_blocks {
        let start = b * block_size;
        let end = (start + block_size).min(values.len());
        let mut block_vals = [0.0f32; 32];
        block_vals[..end - start].copy_from_slice(&values[start..end]);
        // Remaining positions (if any) stay as 0.0 — the zero-padding.
        blocks.push(Q4Block::quantize(&block_vals));
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

    // ─────────────────────────────────────────────────────────────────────────
    // Integration tests: Q8_0 pipeline end-to-end
    // ─────────────────────────────────────────────────────────────────────────

    /// Helper: compute cosine similarity between two slices.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "cosine_similarity: length mismatch");
        let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    /// Helper: compute reference f32 matmul given dequantized weights.
    /// Weight layout: [out, in], input layout: [batch, in].
    fn ref_matmul_f32(x: &[f32], w: &[f32], batch: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; batch * n];
        for i in 0..batch {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += x[i * k + l] * w[j * k + l];
                }
                out[i * n + j] = sum;
            }
        }
        out
    }

    // ── Test 1: Tensor F32 → Q8_0 → F32 roundtrip ──────────────────────────

    #[test]
    fn test_tensor_f32_to_q8_roundtrip() {
        use crate::tensor::{DType, Tensor};

        // 64 deterministic f32 values shaped [2, 32].
        let data: Vec<f32> = (0..64).map(|i| (i as f32 * 0.05).sin()).collect();
        let original = data.clone();

        let t = Tensor::new(data, vec![2, 32]).unwrap();
        assert_eq!(t.dtype(), DType::F32, "initial dtype should be F32");
        assert_eq!(t.shape(), &[2, 32], "initial shape should be [2, 32]");

        // Convert F32 -> Q8_0.
        let t_q8 = t.to_dtype(DType::Q8_0).unwrap();
        assert_eq!(t_q8.dtype(), DType::Q8_0,
            "after to_dtype(Q8_0) dtype should be Q8_0, got {:?}", t_q8.dtype());
        assert_eq!(t_q8.shape(), &[2, 32],
            "shape after Q8_0 conversion should be [2, 32], got {:?}", t_q8.shape());

        // Convert Q8_0 -> F32.
        let t_f32 = t_q8.to_dtype(DType::F32).unwrap();
        assert_eq!(t_f32.dtype(), DType::F32,
            "after to_dtype(F32) dtype should be F32, got {:?}", t_f32.dtype());
        assert_eq!(t_f32.shape(), &[2, 32],
            "shape after F32 recovery should be [2, 32], got {:?}", t_f32.shape());

        // Compute MSE between original and roundtrip values.
        let recovered = t_f32.to_vec_f32();
        assert_eq!(recovered.len(), 64, "recovered vector should have 64 elements");
        let mse: f32 = original.iter().zip(recovered.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f32>()
            / 64.0;
        assert!(
            mse < 1e-3,
            "MSE {} between original and Q8 roundtrip exceeds 1e-3",
            mse
        );
    }

    // ── Test 2: Tensor from Q8 blocks → to_vec_f32 ──────────────────────────

    #[test]
    fn test_tensor_q8_to_vec_f32() {
        use crate::tensor::Tensor;

        // Known values: 32 values in [−1, 1], one block.
        let mut values = [0.0f32; 32];
        for (i, v) in values.iter_mut().enumerate() {
            *v = (i as f32 * 0.2).cos();
        }

        let block = Q8Block::quantize(&values);
        let expected_deq = block.dequantize();

        // Wrap in a Tensor via from_q8_blocks.
        let t = Tensor::from_q8_blocks(vec![block], 32, vec![32]).unwrap();

        let result = t.to_vec_f32();
        assert_eq!(result.len(), 32,
            "to_vec_f32 should return 32 elements, got {}", result.len());

        // Each element should match the dequantized block value.
        for i in 0..32 {
            let err = (result[i] - expected_deq[i]).abs();
            assert!(
                err < 1e-6,
                "to_vec_f32[{}]: expected {}, got {}, diff {}",
                i, expected_deq[i], result[i], err
            );
        }
    }

    // ── Test 3: Tensor into_q8_blocks ────────────────────────────────────────

    #[test]
    fn test_tensor_into_q8_blocks() {
        use crate::tensor::Tensor;

        // Two blocks (64 values).
        let blocks_in: Vec<Q8Block> = (0..2usize)
            .map(|b| {
                let mut vals = [0.0f32; 32];
                for (i, v) in vals.iter_mut().enumerate() {
                    *v = ((b * 32 + i) as f32 * 0.1).sin();
                }
                Q8Block::quantize(&vals)
            })
            .collect();

        let t = Tensor::from_q8_blocks(blocks_in.clone(), 64, vec![2, 32]).unwrap();

        let blocks_out = t.into_q8_blocks().unwrap();
        assert_eq!(
            blocks_out.len(), 2,
            "into_q8_blocks should return 2 blocks, got {}", blocks_out.len()
        );

        // Block scale bits should match exactly (we put them in, should get them back).
        for b in 0..2 {
            assert_eq!(
                blocks_out[b].scale_bits, blocks_in[b].scale_bits,
                "Block {} scale_bits mismatch: expected {}, got {}",
                b, blocks_in[b].scale_bits, blocks_out[b].scale_bits
            );
            for q in 0..32 {
                assert_eq!(
                    blocks_out[b].quants[q], blocks_in[b].quants[q],
                    "Block {} quants[{}] mismatch: expected {}, got {}",
                    b, q, blocks_in[b].quants[q], blocks_out[b].quants[q]
                );
            }
        }
    }

    // ── Test 4: Q8Linear forward — identity approximation ──────────────────

    #[test]
    fn test_q8linear_forward_identity() {
        use crate::attention::Q8Linear;
        use crate::tensor::Tensor;

        // 32x32 identity matrix: W[i, j] = 1.0 if i == j else 0.0.
        // Each row is 32 values (1 block). There are 32 rows.
        let mut w_f32 = vec![0.0f32; 32 * 32];
        for i in 0..32 {
            w_f32[i * 32 + i] = 1.0;
        }

        // Quantize: 32 rows × 1 block each = 32 blocks total.
        let blocks = quantize_f32_to_q8(&w_f32);
        assert_eq!(blocks.len(), 32,
            "identity matrix should produce 32 blocks, got {}", blocks.len());

        let q8lin = Q8Linear::from_blocks(blocks, 32, 32).unwrap();

        // Input: [1, 32] tensor with known values.
        let input_data: Vec<f32> = (0..32).map(|i| i as f32 * 0.1 + 0.5).collect();
        let x = Tensor::new(input_data.clone(), vec![1, 32]).unwrap();

        let out = q8lin.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 32],
            "output shape should be [1, 32], got {:?}", out.shape());

        // For an identity matrix, output ≈ input (within Q8 tolerance).
        let out_vec = out.to_vec_f32();
        for i in 0..32 {
            let err = (out_vec[i] - input_data[i]).abs();
            // Q8 tolerance: scale ≈ 1/127 ≈ 0.008 per step.
            // For identity diagonal entry 1.0, quant error ≈ 0.008.
            // For input values up to ~3.6, absolute output error ≈ input * quant_error ≈ 0.03.
            assert!(
                err < 0.1,
                "identity forward[{}]: expected {}, got {}, error {} > 0.1",
                i, input_data[i], out_vec[i], err
            );
        }
    }

    // ── Test 5: Q8Linear forward vs F32 Linear (same weights) ───────────────

    #[test]
    fn test_q8linear_forward_vs_linear() {
        use crate::attention::{Linear, Q8Linear};
        use crate::tensor::Tensor;

        let out_features = 64usize;
        let in_features = 32usize;

        // Deterministic weight matrix with sin/cos pattern.
        let w_f32: Vec<f32> = (0..out_features * in_features)
            .map(|i| {
                let t = i as f32 * 0.07;
                if i % 2 == 0 { t.sin() } else { t.cos() }
            })
            .collect();

        // F32 Linear.
        let w_tensor_f32 = Tensor::new(w_f32.clone(), vec![out_features, in_features]).unwrap();
        let linear_f32 = Linear::new(w_tensor_f32).unwrap();

        // Q8Linear from quantized blocks.
        let blocks = quantize_f32_to_q8(&w_f32);
        let q8lin = Q8Linear::from_blocks(blocks, out_features, in_features).unwrap();

        // Input: [1, 32] tensor.
        let x_data: Vec<f32> = (0..in_features)
            .map(|i| (i as f32 * 0.15 + 0.3).sin())
            .collect();
        let x = Tensor::new(x_data.clone(), vec![1, in_features]).unwrap();

        // Forward through both.
        let out_f32 = linear_f32.forward(&x).unwrap().to_vec_f32();
        let out_q8 = q8lin.forward(&x).unwrap().to_vec_f32();

        assert_eq!(out_f32.len(), out_features);
        assert_eq!(out_q8.len(), out_features);

        // Cosine similarity should be very high.
        let cos_sim = cosine_similarity(&out_f32, &out_q8);
        assert!(
            cos_sim > 0.999,
            "cosine similarity between F32 and Q8 output: {} < 0.999",
            cos_sim
        );

        // Max relative error should be within Q8_0 quantization tolerance.
        // Q8_0 introduces ~1/127 ≈ 0.8% error per weight element; accumulated
        // over a dot product of 32 values this can reach ~2% on individual outputs.
        // We use 2% as a practical upper bound for this pattern.
        let mut max_rel_err = 0.0f32;
        for i in 0..out_features {
            let err = rel_error(out_f32[i], out_q8[i]);
            if err > max_rel_err {
                max_rel_err = err;
            }
        }
        assert!(
            max_rel_err < 0.02,
            "max relative error between F32 and Q8 output: {} > 2%",
            max_rel_err
        );
    }

    // ── Test 6: AnyLinear dispatch — F32 and Q8 produce same results ─────────

    #[test]
    fn test_anylinear_dispatch() {
        use crate::attention::{AnyLinear, Linear, Q8Linear};
        use crate::tensor::Tensor;

        let out_features = 32usize;
        let in_features = 64usize;

        // Deterministic weights.
        let w_f32: Vec<f32> = (0..out_features * in_features)
            .map(|i| ((i as f32 * 0.09).sin()) * 0.5)
            .collect();

        // AnyLinear::F32
        let w_tensor = Tensor::new(w_f32.clone(), vec![out_features, in_features]).unwrap();
        let any_f32 = AnyLinear::F32(Linear::new(w_tensor).unwrap());

        // AnyLinear::Q8 — same weights, quantized.
        let blocks = quantize_f32_to_q8(&w_f32);
        let q8lin = Q8Linear::from_blocks(blocks, out_features, in_features).unwrap();
        let any_q8 = AnyLinear::Q8(q8lin);

        // Input: [1, 64] tensor.
        let x_data: Vec<f32> = (0..in_features)
            .map(|i| (i as f32 * 0.11 + 0.1).cos() * 0.8)
            .collect();
        let x = Tensor::new(x_data, vec![1, in_features]).unwrap();

        let out_f32 = any_f32.forward(&x).unwrap().to_vec_f32();
        let out_q8 = any_q8.forward(&x).unwrap().to_vec_f32();

        assert_eq!(out_f32.len(), out_features,
            "F32 output length should be {}, got {}", out_features, out_f32.len());
        assert_eq!(out_q8.len(), out_features,
            "Q8 output length should be {}, got {}", out_features, out_q8.len());

        // Both enum arms should produce close results.
        let cos_sim = cosine_similarity(&out_f32, &out_q8);
        assert!(
            cos_sim > 0.999,
            "AnyLinear dispatch: cosine similarity {} < 0.999 — F32 and Q8 outputs diverged",
            cos_sim
        );
    }

    // ── Test 7: Q8Linear batched forward ─────────────────────────────────────

    #[test]
    fn test_q8linear_batched() {
        use crate::attention::Q8Linear;
        use crate::tensor::Tensor;

        let out_features = 32usize;
        let in_features = 64usize;

        let w_f32: Vec<f32> = (0..out_features * in_features)
            .map(|i| (i as f32 * 0.05).sin() * 0.3)
            .collect();
        let blocks = quantize_f32_to_q8(&w_f32);
        let q8lin = Q8Linear::from_blocks(blocks, out_features, in_features).unwrap();

        // 2D batch: [4, 64] → [4, 32].
        let x_2d_data: Vec<f32> = (0..4 * in_features)
            .map(|i| (i as f32 * 0.03 + 0.2).cos())
            .collect();
        let x_2d = Tensor::new(x_2d_data, vec![4, in_features]).unwrap();
        let out_2d = q8lin.forward(&x_2d).unwrap();
        assert_eq!(
            out_2d.shape(), &[4, 32],
            "2D batched output shape should be [4, 32], got {:?}", out_2d.shape()
        );

        // 3D batch: [2, 3, 64] → [2, 3, 32].
        let x_3d_data: Vec<f32> = (0..2 * 3 * in_features)
            .map(|i| (i as f32 * 0.04 + 0.1).sin())
            .collect();
        let x_3d = Tensor::new(x_3d_data, vec![2, 3, in_features]).unwrap();
        let out_3d = q8lin.forward(&x_3d).unwrap();
        assert_eq!(
            out_3d.shape(), &[2, 3, 32],
            "3D batched output shape should be [2, 3, 32], got {:?}", out_3d.shape()
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Q4_0 unit tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_q4_block_quantize_basic() {
        // Quantize known values and verify dequantized values are reasonable.
        // Q4 has only 16 levels so tolerance is wider than Q8.
        let mut values = [0.0f32; 32];
        // Set first 4 known values; rest zero.
        values[0] = -1.0;
        values[1] = 0.5;
        values[2] = 0.0;
        values[3] = 1.0;

        let block = Q4Block::quantize(&values);

        // Scale should be positive (abs_max = 1.0, scale = 1.0/8.0 = 0.125).
        let scale = block.scale();
        assert!(scale > 0.0, "Scale should be positive, got {}", scale);

        let deq = block.dequantize();

        // -1.0: nibble = round(-1.0/0.125) + 8 = round(-8) + 8 = 0 → deq = (0-8)*0.125 = -1.0
        let err0 = (deq[0] - (-1.0f32)).abs();
        assert!(err0 < 0.2, "deq[0]={} expected ~-1.0, error {}", deq[0], err0);

        // 0.5: nibble = round(0.5/0.125) + 8 = 4 + 8 = 12 → deq = (12-8)*0.125 = 0.5
        let err1 = (deq[1] - 0.5f32).abs();
        assert!(err1 < 0.2, "deq[1]={} expected ~0.5, error {}", deq[1], err1);

        // 0.0: nibble = 8 → deq = 0.0
        let err2 = deq[2].abs();
        assert!(err2 < 0.2, "deq[2]={} expected ~0.0, error {}", deq[2], err2);

        // 1.0: nibble = round(1.0/0.125) + 8 = 8 + 8 = 16 → clamped to 15 → deq = 7*0.125 = 0.875
        // (max positive is 7 due to asymmetry)
        let err3 = (deq[3] - 1.0f32).abs();
        assert!(err3 < 0.2, "deq[3]={} expected ~1.0, error {}", deq[3], err3);
    }

    #[test]
    fn test_q4_block_zero() {
        let values = [0.0f32; 32];
        let block = Q4Block::quantize(&values);

        assert_eq!(block.scale_bits, 0, "Scale bits should be 0 for all-zero input");
        assert_eq!(block.scale(), 0.0f32, "Scale should be 0.0 for all-zero input");

        // All nibbles should be 0x88: low nibble = 8, high nibble = 8 (zero offset).
        for (i, &nib) in block.nibs.iter().enumerate() {
            assert_eq!(
                nib, 0x88,
                "nibs[{}] should be 0x88 for all-zero input, got 0x{:02X}",
                i, nib
            );
        }
    }

    #[test]
    fn test_q4_block_roundtrip_quality() {
        // sin(i) pattern: values in [-1, 1].
        let mut values = [0.0f32; 32];
        for (i, v) in values.iter_mut().enumerate() {
            *v = (i as f32).sin();
        }

        let block = Q4Block::quantize(&values);
        let deq = block.dequantize();

        // Compute mean squared error.
        let mse: f32 = values
            .iter()
            .zip(deq.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f32>()
            / 32.0;

        // Q4 is much coarser than Q8 but MSE should still be reasonable for [-1, 1] values.
        assert!(
            mse < 0.01,
            "Q4 MSE {} exceeds 0.01 for sin(i) roundtrip",
            mse
        );
    }

    #[test]
    fn test_q4_block_dot_f32() {
        // Create a block from known values and verify fused dot matches dequantize+dot.
        let mut weights = [0.0f32; 32];
        let mut activations = [0.0f32; 32];
        for i in 0..32 {
            weights[i] = (i as f32 + 1.0) / 32.0; // 1/32, 2/32, ..., 1.0
            activations[i] = 1.0 / (i as f32 + 1.0); // 1, 1/2, 1/3, ...
        }

        let block = Q4Block::quantize(&weights);

        // Reference dot product using dequantized weights.
        let deq = block.dequantize();
        let ref_dot: f32 = deq.iter().zip(activations.iter()).map(|(&w, &a)| w * a).sum();

        let fused_dot = block.block_dot_f32(&activations);

        // Q4 is coarser — use wider tolerance (~5%).
        let err = rel_error(ref_dot, fused_dot);
        assert!(
            err < 0.001,
            "Q4 block_dot_f32 {} vs reference {}, relative error {} > 0.1%",
            fused_dot, ref_dot, err
        );
    }

    #[test]
    fn test_q4_nibble_packing() {
        // Manually verify nibble packing for a known block.
        // Use values where we can predict the nibble content.
        // abs_max = 0.5, scale = 0.5/8 = 0.0625
        // values[0] = -0.5 → nibble = round(-0.5/0.0625) + 8 = -8 + 8 = 0 → low nibble = 0
        // values[1] =  0.5 → nibble = round(0.5/0.0625)  + 8 =  8 + 8 = 16 clamped to 15 → high nibble = 0xF
        // values[2] =  0.0 → nibble = 8 → low nibble = 8
        // values[3] = -0.25 → nibble = round(-0.25/0.0625) + 8 = -4 + 8 = 4 → high nibble = 4
        let mut values = [0.0f32; 32];
        values[0] = -0.5;
        values[1] = 0.5;
        values[2] = 0.0;
        values[3] = -0.25;

        let block = Q4Block::quantize(&values);

        // Verify nibs[0]: low nibble = q0, high nibble = q1
        let nib0_lo = block.nibs[0] & 0x0F;
        let nib0_hi = (block.nibs[0] >> 4) & 0x0F;

        // q0 for values[0] = -0.5: nibble = 0
        assert_eq!(nib0_lo, 0, "nibs[0] low nibble should be 0, got {}", nib0_lo);
        // q1 for values[1] = 0.5: nibble = 15 (clamped from 16)
        assert_eq!(nib0_hi, 15, "nibs[0] high nibble should be 15, got {}", nib0_hi);

        // Verify nibs[1]: low nibble = q2, high nibble = q3
        let nib1_lo = block.nibs[1] & 0x0F;
        let nib1_hi = (block.nibs[1] >> 4) & 0x0F;

        // q2 for values[2] = 0.0: nibble = 8
        assert_eq!(nib1_lo, 8, "nibs[1] low nibble should be 8, got {}", nib1_lo);
        // q3 for values[3] = -0.25: nibble = round(-0.25/0.0625) + 8 = -4 + 8 = 4
        assert_eq!(nib1_hi, 4, "nibs[1] high nibble should be 4, got {}", nib1_hi);
    }

    #[test]
    fn test_quantize_f32_to_q4_padding() {
        // 48 values → 2 blocks, last 16 values of second block padded with 0.
        let values: Vec<f32> = (0..48).map(|i| i as f32 * 0.01 + 0.1).collect();
        let blocks = quantize_f32_to_q4(&values);

        assert_eq!(blocks.len(), 2,
            "48 values should produce 2 blocks, got {}", blocks.len());

        // Dequantize the second block and verify the last 16 values are ~0 (zero-padded input).
        let deq1 = blocks[1].dequantize();
        for i in 16..32 {
            // Padded positions were 0.0 in input, so dequantized should be 0.0 (nibble = 8 → (8-8)*s = 0).
            assert_eq!(
                deq1[i], 0.0f32,
                "Second block deq[{}] should be 0.0 (padding), got {}",
                i, deq1[i]
            );
        }
    }

    #[test]
    fn test_matmul_f32_x_q4_simple() {
        // 2×64 activations @ 3×64 weight matrix → [2, 3] output.
        let m = 2usize;
        let k = 64usize;
        let n = 3usize;

        let w_f32: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32 * 0.7 + 1.0) % 2.0) - 1.0)
            .collect();
        let x: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32 * 0.3 + 0.5) % 1.5) - 0.75)
            .collect();

        // Quantize weights to Q4.
        let w_blocks = quantize_f32_to_q4(&w_f32);

        // Compute output with quantized matmul.
        let out_quant = matmul_f32_x_quant::<Q4Block>(&x, &w_blocks, m, k, n);

        // Reference: dequantize Q4 weights and compute exact f32 matmul.
        let mut w_deq = vec![0.0f32; n * k];
        for (b_idx, block) in w_blocks.iter().enumerate() {
            let deq = block.dequantize();
            let start = b_idx * 32;
            w_deq[start..start + 32].copy_from_slice(&deq);
        }
        let out_ref = ref_matmul_f32(&x, &w_deq, m, k, n);

        // Cosine similarity should be > 0.995 for Q4.
        let cos_sim = cosine_similarity(&out_ref, &out_quant);
        assert!(
            cos_sim > 0.995,
            "Q4 matmul cosine similarity {} < 0.995",
            cos_sim
        );
    }

    #[test]
    fn test_q4_vs_q8_quality() {
        // Quantize same data to both Q4 and Q8, verify Q8 is more accurate.
        let mut values = [0.0f32; 32];
        for (i, v) in values.iter_mut().enumerate() {
            *v = (i as f32 * 0.2).sin();
        }

        let q4_block = Q4Block::quantize(&values);
        let q8_block = Q8Block::quantize(&values);

        let q4_deq = q4_block.dequantize();
        let q8_deq = q8_block.dequantize();

        let q4_mse: f32 = values
            .iter()
            .zip(q4_deq.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f32>()
            / 32.0;

        let q8_mse: f32 = values
            .iter()
            .zip(q8_deq.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f32>()
            / 32.0;

        // Q4 should have higher MSE than Q8 (coarser quantization).
        assert!(
            q4_mse > q8_mse,
            "Q4 MSE {} should be > Q8 MSE {} (Q4 is coarser)",
            q4_mse, q8_mse
        );

        // Q4 MSE should still be reasonable.
        assert!(
            q4_mse < 0.01,
            "Q4 MSE {} exceeds 0.01 (too lossy)",
            q4_mse
        );

        // Cosine similarity comparisons.
        let q4_cos = cosine_similarity(&values, &q4_deq);
        let q8_cos = cosine_similarity(&values, &q8_deq);

        // Q4 cosine should be lower than Q8 cosine.
        assert!(
            q4_cos < q8_cos,
            "Q4 cosine {} should be < Q8 cosine {} (Q4 is coarser)",
            q4_cos, q8_cos
        );

        // But Q4 cosine should still be high.
        assert!(
            q4_cos > 0.99,
            "Q4 cosine similarity {} < 0.99 (too lossy)",
            q4_cos
        );
    }

    // ── Test 8: Q8 matmul accumulation accuracy over many blocks ─────────────

    #[test]
    fn test_q8_matmul_accumulation_accuracy() {
        // Weight [128, 256]: 128 output neurons, 256 input features.
        // Each row has 256/32 = 8 blocks. Total: 128 × 8 = 1024 blocks.
        let out_features = 128usize;
        let in_features = 256usize;

        let w_f32: Vec<f32> = (0..out_features * in_features)
            .map(|i| {
                let t = i as f32 * 0.031;
                t.sin() * 0.4
            })
            .collect();

        // Quantize the weights.
        let w_blocks = quantize_f32_to_q8(&w_f32);
        assert_eq!(
            w_blocks.len(), out_features * (in_features / 32),
            "expected {} blocks, got {}",
            out_features * (in_features / 32), w_blocks.len()
        );

        // Input: [1, 256].
        let x: Vec<f32> = (0..in_features)
            .map(|i| (i as f32 * 0.023 + 0.5).cos() * 0.6)
            .collect();

        // Q8 matmul result.
        let out_q8 = matmul_f32_x_quant::<Q8Block>(&x, &w_blocks, 1, in_features, out_features);

        // Reference: dequantize all blocks and do f32 matmul.
        let mut w_deq = vec![0.0f32; out_features * in_features];
        for (b_idx, block) in w_blocks.iter().enumerate() {
            let deq = block.dequantize();
            let start = b_idx * 32;
            let end = (start + 32).min(w_deq.len());
            w_deq[start..end].copy_from_slice(&deq[..end - start]);
        }
        let out_ref = ref_matmul_f32(&x, &w_deq, 1, in_features, out_features);

        assert_eq!(out_q8.len(), out_features);
        assert_eq!(out_ref.len(), out_features);

        // Cosine similarity over output vectors.
        let cos_sim = cosine_similarity(&out_ref, &out_q8);
        assert!(
            cos_sim > 0.999,
            "accumulation accuracy: cosine similarity {} < 0.999",
            cos_sim
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Q4_0 integration tests
    // ─────────────────────────────────────────────────────────────────────────

    // ── Test: F32 tensor roundtrip through Q4_0 ───────────────────────────────

    #[test]
    fn test_tensor_f32_to_q4_roundtrip() {
        use crate::tensor::{DType, Tensor};

        // [2, 32] F32 tensor — 64 values total (2 full Q4_0 blocks).
        let data: Vec<f32> = (0..64)
            .map(|i| (i as f32 * 0.17).sin() * 0.8)
            .collect();
        let original = Tensor::new(data.clone(), vec![2, 32]).unwrap();

        // Convert F32 -> Q4_0 -> F32.
        let q4_tensor = original.to_dtype(DType::Q4_0).unwrap();
        assert_eq!(q4_tensor.dtype(), DType::Q4_0,
            "After to_dtype(Q4_0), dtype should be Q4_0");
        assert_eq!(q4_tensor.shape(), &[2, 32],
            "Shape should be preserved after quantization");

        let recovered = q4_tensor.to_dtype(DType::F32).unwrap();
        assert_eq!(recovered.dtype(), DType::F32,
            "After to_dtype(F32), dtype should be F32");

        let recovered_data = recovered.to_vec_f32();
        assert_eq!(recovered_data.len(), data.len(),
            "Recovered tensor should have same number of elements");

        // Compute MSE — Q4 is 4-bit so tolerance is wider than Q8.
        let mse: f64 = data.iter().zip(recovered_data.iter())
            .map(|(&a, &b)| { let d = a as f64 - b as f64; d * d })
            .sum::<f64>()
            / data.len() as f64;

        assert!(
            mse < 0.05,
            "F32->Q4_0->F32 roundtrip MSE {:.4e} exceeds 0.05 (Q4 wider tolerance)",
            mse
        );

        // Cosine similarity should still be high.
        let cos = cosine_similarity(&data, &recovered_data);
        assert!(
            cos > 0.99,
            "F32->Q4_0->F32 roundtrip cosine {:.6} < 0.99",
            cos
        );
    }

    // ── Test: Q4Linear forward vs F32 Linear ─────────────────────────────────

    #[test]
    fn test_q4linear_forward_vs_linear() {
        use crate::attention::{Linear, Q4Linear};
        use crate::tensor::Tensor;

        let out_features = 64usize;
        let in_features = 32usize;

        // Deterministic weights [64, 32].
        let w_f32: Vec<f32> = (0..out_features * in_features)
            .map(|i| (i as f32 * 0.13).sin() * 0.6)
            .collect();

        // F32 reference linear layer.
        let w_tensor_f32 = Tensor::new(w_f32.clone(), vec![out_features, in_features]).unwrap();
        let linear_f32 = Linear::new(w_tensor_f32).unwrap();

        // Q4Linear from quantized blocks.
        let q4_blocks = quantize_f32_to_q4(&w_f32);
        let q4lin = Q4Linear::from_blocks(q4_blocks, out_features, in_features).unwrap();

        // Input: [1, 32].
        let x_data: Vec<f32> = (0..in_features)
            .map(|i| (i as f32 * 0.11 + 0.1).cos() * 0.9)
            .collect();
        let x = Tensor::new(x_data, vec![1, in_features]).unwrap();

        let out_f32 = linear_f32.forward(&x).unwrap().to_vec_f32();
        let out_q4 = q4lin.forward(&x).unwrap().to_vec_f32();

        assert_eq!(out_f32.len(), out_features,
            "F32 output length should be {}, got {}", out_features, out_f32.len());
        assert_eq!(out_q4.len(), out_features,
            "Q4 output length should be {}, got {}", out_features, out_q4.len());

        // Q4 has less precision than Q8, so threshold is 0.99 instead of 0.999.
        let cos = cosine_similarity(&out_f32, &out_q4);
        assert!(
            cos > 0.99,
            "Q4Linear vs F32 Linear cosine {:.6} < 0.99 — Q4 output diverged too much",
            cos
        );
    }

    // ── Test: Q4Linear batched forward (2D and 3D input) ─────────────────────

    #[test]
    fn test_q4linear_batched() {
        use crate::attention::Q4Linear;
        use crate::tensor::Tensor;

        let out_features = 32usize;
        let in_features = 64usize;

        let w_f32: Vec<f32> = (0..out_features * in_features)
            .map(|i| (i as f32 * 0.05).sin() * 0.4)
            .collect();
        let q4_blocks = quantize_f32_to_q4(&w_f32);
        let q4lin = Q4Linear::from_blocks(q4_blocks, out_features, in_features).unwrap();

        // 2D batch: [4, 64] → [4, 32].
        let x_2d_data: Vec<f32> = (0..4 * in_features)
            .map(|i| (i as f32 * 0.03 + 0.2).cos())
            .collect();
        let x_2d = Tensor::new(x_2d_data, vec![4, in_features]).unwrap();
        let out_2d = q4lin.forward(&x_2d).unwrap();
        assert_eq!(
            out_2d.shape(), &[4, out_features],
            "2D batched Q4Linear output shape should be [4, {}], got {:?}",
            out_features, out_2d.shape()
        );

        // 3D batch: [2, 3, 64] → [2, 3, 32].
        let x_3d_data: Vec<f32> = (0..2 * 3 * in_features)
            .map(|i| (i as f32 * 0.04 + 0.1).sin())
            .collect();
        let x_3d = Tensor::new(x_3d_data, vec![2, 3, in_features]).unwrap();
        let out_3d = q4lin.forward(&x_3d).unwrap();
        assert_eq!(
            out_3d.shape(), &[2, 3, out_features],
            "3D batched Q4Linear output shape should be [2, 3, {}], got {:?}",
            out_features, out_3d.shape()
        );
    }

    // ── Test: AnyLinear::Q4 dispatch ─────────────────────────────────────────

    #[test]
    fn test_anylinear_q4_dispatch() {
        use crate::attention::{AnyLinear, Q4Linear};
        use crate::tensor::Tensor;

        let out_features = 32usize;
        let in_features = 64usize;

        // Build Q4Linear and wrap in AnyLinear::Q4.
        let w_f32: Vec<f32> = (0..out_features * in_features)
            .map(|i| (i as f32 * 0.09).sin() * 0.5)
            .collect();
        let q4_blocks = quantize_f32_to_q4(&w_f32);
        let q4lin = Q4Linear::from_blocks(q4_blocks, out_features, in_features).unwrap();
        let any_q4 = AnyLinear::Q4(q4lin);

        // Input: [1, 64].
        let x_data: Vec<f32> = (0..in_features)
            .map(|i| (i as f32 * 0.11 + 0.1).cos() * 0.8)
            .collect();
        let x = Tensor::new(x_data, vec![1, in_features]).unwrap();

        // forward() should succeed and return [1, 32] tensor.
        let out = any_q4.forward(&x).unwrap();
        let out_shape = out.shape().to_vec();
        assert_eq!(
            out_shape, vec![1, out_features],
            "AnyLinear::Q4 forward should produce [1, {}], got {:?}",
            out_features, out_shape
        );

        // Output should contain finite values (not NaN/Inf).
        let out_data = out.to_vec_f32();
        for (i, &v) in out_data.iter().enumerate() {
            assert!(
                v.is_finite(),
                "AnyLinear::Q4 output[{}] is non-finite: {}",
                i, v
            );
        }
    }
}
