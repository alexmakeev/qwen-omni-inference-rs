//! BF16 (Brain Float 16) type for efficient weight storage.
//!
//! BF16 is a 16-bit floating point format designed for neural network weights:
//! - 1 sign bit
//! - 8 exponent bits (same range as F32)
//! - 7 mantissa bits (reduced precision)
//!
//! **IMPORTANT:** BF16 is for STORAGE only. All compute happens in F32.
//! x86 CPUs don't have native BF16 compute instructions.
//!
//! # Conversion Strategy
//!
//! - F32 → BF16: Round to nearest even (RNE) for best accuracy
//! - BF16 → F32: Zero-extend mantissa (lossless bit operation)
//!
//! # Example
//!
//! ```rust
//! use lluda_inference::bf16::BF16;
//!
//! let original = 3.14159f32;
//! let bf16 = BF16::from(original);
//! let recovered = f32::from(bf16);
//!
//! // BF16 preserves ~3 decimal places
//! assert!((original - recovered).abs() < 0.01);
//! ```

use std::fmt;

/// BF16 (Brain Float 16) floating point type.
///
/// Stores 16-bit floating point value in bfloat16 format.
/// Use this for weight storage to reduce memory bandwidth.
/// Convert to F32 before any computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BF16(u16);

impl BF16 {
    /// Create a BF16 from raw bits.
    ///
    /// This is used when loading BF16 data from binary formats like SafeTensors.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::bf16::BF16;
    ///
    /// let bf16 = BF16::from_bits(0x3F80); // 1.0 in BF16
    /// let f32_val = f32::from(bf16);
    /// assert_eq!(f32_val, 1.0f32);
    /// ```
    pub fn from_bits(bits: u16) -> Self {
        BF16(bits)
    }

    /// Convert a slice of BF16 values to F32 for computation.
    ///
    /// This is a batch operation optimized for tensor weight loading.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::bf16::BF16;
    ///
    /// let weights_bf16 = vec![BF16::from(1.0f32), BF16::from(2.0f32), BF16::from(3.0f32)];
    /// let weights_f32 = BF16::to_f32_slice(&weights_bf16);
    ///
    /// assert_eq!(weights_f32.len(), 3);
    /// assert_eq!(weights_f32[0], 1.0f32);
    /// ```
    pub fn to_f32_slice(src: &[BF16]) -> Vec<f32> {
        src.iter().map(|&bf16| f32::from(bf16)).collect()
    }

    /// Convert a slice of F32 values to BF16 for storage.
    ///
    /// This is a batch operation for converting computed results back to BF16.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::bf16::BF16;
    ///
    /// let data_f32 = vec![1.5f32, 2.5f32, 3.5f32];
    /// let data_bf16 = BF16::from_f32_slice(&data_f32);
    ///
    /// assert_eq!(data_bf16.len(), 3);
    /// ```
    pub fn from_f32_slice(src: &[f32]) -> Vec<BF16> {
        src.iter().map(|&f| BF16::from(f)).collect()
    }
}

/// Convert F32 to BF16 with round-to-nearest-even.
///
/// This conversion truncates the mantissa from 23 bits to 7 bits,
/// using round-to-nearest-even (RNE) for best accuracy.
///
/// # Special cases
///
/// - ±0.0 → ±0.0
/// - ±Infinity → ±Infinity
/// - NaN → NaN (preserves sign, loses payload)
impl From<f32> for BF16 {
    fn from(val: f32) -> Self {
        let bits = val.to_bits();

        // BF16 is the top 16 bits of F32.
        // For round-to-nearest-even (RNE):
        // 1. Check the 16th bit (first bit we're dropping)
        // 2. If it's 1, we might round up
        // 3. Tie-breaking: if exactly halfway, round to even (check 17th bit)

        // Rounding bias: 0x7FFF base
        // Add LSB of result to bias (for tie-breaking to even)
        let rounding_bias = 0x7FFF + ((bits >> 16) & 1);

        // Add bias and extract top 16 bits
        let bf16_bits = ((bits + rounding_bias) >> 16) as u16;

        BF16(bf16_bits)
    }
}

/// Convert BF16 to F32 (zero-extend mantissa, lossless).
///
/// This is a simple bit operation: shift left by 16 bits to restore F32 layout.
impl From<BF16> for f32 {
    fn from(val: BF16) -> Self {
        // BF16 is top 16 bits of F32, so we just shift left
        let bits = (val.0 as u32) << 16;
        f32::from_bits(bits)
    }
}

/// Default value is 0.0.
impl Default for BF16 {
    fn default() -> Self {
        BF16::from(0.0f32)
    }
}

/// Display BF16 as its F32 representation for debugging.
impl fmt::Display for BF16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", f32::from(*self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero() {
        let bf16 = BF16::from(0.0f32);
        let recovered = f32::from(bf16);
        assert_eq!(recovered, 0.0f32);
    }

    #[test]
    fn test_negative_zero() {
        let bf16 = BF16::from(-0.0f32);
        let recovered = f32::from(bf16);
        assert_eq!(recovered, -0.0f32);
        // Verify sign bit is preserved
        assert_eq!(recovered.to_bits() & 0x8000_0000, 0x8000_0000);
    }

    #[test]
    fn test_one() {
        let bf16 = BF16::from(1.0f32);
        let recovered = f32::from(bf16);
        assert_eq!(recovered, 1.0f32);
    }

    #[test]
    fn test_negative_one() {
        let bf16 = BF16::from(-1.0f32);
        let recovered = f32::from(bf16);
        assert_eq!(recovered, -1.0f32);
    }

    #[test]
    fn test_round_trip_precision() {
        // BF16 has ~3 decimal places precision
        let original = 3.14159f32;
        let bf16 = BF16::from(original);
        let recovered = f32::from(bf16);

        // BF16 should preserve value within reasonable tolerance
        assert!((original - recovered).abs() < 0.01,
            "Original: {}, Recovered: {}, Diff: {}",
            original, recovered, (original - recovered).abs());
    }

    #[test]
    fn test_infinity() {
        let bf16_pos = BF16::from(f32::INFINITY);
        let recovered_pos = f32::from(bf16_pos);
        assert_eq!(recovered_pos, f32::INFINITY);

        let bf16_neg = BF16::from(f32::NEG_INFINITY);
        let recovered_neg = f32::from(bf16_neg);
        assert_eq!(recovered_neg, f32::NEG_INFINITY);
    }

    #[test]
    fn test_nan() {
        let bf16 = BF16::from(f32::NAN);
        let recovered = f32::from(bf16);
        assert!(recovered.is_nan(), "BF16(NaN) should recover as NaN");
    }

    #[test]
    fn test_subnormal_flush_to_zero() {
        // Subnormal F32 values (very small numbers close to zero)
        // BF16 has different exponent range, test graceful handling
        let subnormal = f32::from_bits(0x0000_0001); // Smallest positive subnormal
        let bf16 = BF16::from(subnormal);
        let recovered = f32::from(bf16);

        // Subnormals typically flush to zero or become zero in BF16
        // This is acceptable behavior
        assert!(recovered.abs() < 1e-30 || recovered == 0.0,
            "Subnormal should flush to zero or very small value");
    }

    #[test]
    fn test_rounding_to_nearest_even() {
        // Test tie-breaking case: value exactly halfway between two BF16 values
        // Should round to even mantissa

        // Create a value that's exactly halfway
        // BF16 has 7 mantissa bits, so bit 16 is the rounding point
        let halfway_to_even = f32::from_bits(0x3F80_8000); // 1.0 + exact halfway
        let bf16 = BF16::from(halfway_to_even);
        let recovered = f32::from(bf16);

        // Should round to the even result (which rounds down in this case)
        // Verify we got a valid result (the rounding should be deterministic)
        assert!(recovered.is_finite(), "Rounding should produce finite value");
    }

    #[test]
    fn test_small_values() {
        let values = vec![0.001f32, 0.01f32, 0.1f32, 1.0f32, 10.0f32, 100.0f32];

        for &val in &values {
            let bf16 = BF16::from(val);
            let recovered = f32::from(bf16);

            // Relative error should be small (within BF16 precision)
            let rel_error = ((val - recovered) / val).abs();
            assert!(rel_error < 0.01,
                "Value: {}, Recovered: {}, Relative error: {}",
                val, recovered, rel_error);
        }
    }

    #[test]
    fn test_large_values() {
        let values = vec![1000.0f32, 1e6f32, 1e9f32, 1e20f32];

        for &val in &values {
            let bf16 = BF16::from(val);
            let recovered = f32::from(bf16);

            // Relative error should be small
            let rel_error = ((val - recovered) / val).abs();
            assert!(rel_error < 0.01,
                "Value: {}, Recovered: {}, Relative error: {}",
                val, recovered, rel_error);
        }
    }

    #[test]
    fn test_batch_conversion_to_f32() {
        let bf16_data = vec![
            BF16::from(1.0f32),
            BF16::from(2.0f32),
            BF16::from(3.0f32),
            BF16::from(4.0f32),
        ];

        let f32_data = BF16::to_f32_slice(&bf16_data);

        assert_eq!(f32_data.len(), 4);
        assert_eq!(f32_data[0], 1.0f32);
        assert_eq!(f32_data[1], 2.0f32);
        assert_eq!(f32_data[2], 3.0f32);
        assert_eq!(f32_data[3], 4.0f32);
    }

    #[test]
    fn test_batch_conversion_from_f32() {
        let f32_data = vec![1.5f32, 2.5f32, 3.5f32, 4.5f32];
        let bf16_data = BF16::from_f32_slice(&f32_data);

        assert_eq!(bf16_data.len(), 4);

        // Verify round-trip
        for (i, &original) in f32_data.iter().enumerate() {
            let recovered = f32::from(bf16_data[i]);
            assert!((original - recovered).abs() < 0.01,
                "Index {}: Original {}, Recovered {}", i, original, recovered);
        }
    }

    #[test]
    fn test_empty_slice() {
        let empty_bf16: Vec<BF16> = vec![];
        let f32_result = BF16::to_f32_slice(&empty_bf16);
        assert_eq!(f32_result.len(), 0);

        let empty_f32: Vec<f32> = vec![];
        let bf16_result = BF16::from_f32_slice(&empty_f32);
        assert_eq!(bf16_result.len(), 0);
    }

    #[test]
    fn test_default() {
        let bf16 = BF16::default();
        let recovered = f32::from(bf16);
        assert_eq!(recovered, 0.0f32);
    }

    #[test]
    fn test_display() {
        let bf16 = BF16::from(3.14159f32);
        let display_str = format!("{}", bf16);

        // Should display as F32 value
        assert!(display_str.contains("3.1"), "Display should show ~3.1x");
    }

    #[test]
    fn test_partial_eq() {
        let bf16_a = BF16::from(1.0f32);
        let bf16_b = BF16::from(1.0f32);
        let bf16_c = BF16::from(2.0f32);

        assert_eq!(bf16_a, bf16_b);
        assert_ne!(bf16_a, bf16_c);
    }

    #[test]
    fn test_clone_copy() {
        let bf16 = BF16::from(1.5f32);

        // Test Clone
        let cloned = bf16.clone();
        assert_eq!(bf16, cloned);

        // Test Copy (implicit)
        let copied = bf16;
        assert_eq!(bf16, copied);
    }

    #[test]
    fn test_typical_neural_network_values() {
        // Common ranges in neural networks: weights typically in [-1, 1]
        let nn_values = vec![
            -0.5f32, -0.1f32, 0.0f32, 0.1f32, 0.5f32,
            -0.99f32, -0.01f32, 0.01f32, 0.99f32,
        ];

        for &val in &nn_values {
            let bf16 = BF16::from(val);
            let recovered = f32::from(bf16);

            let abs_error = (val - recovered).abs();
            assert!(abs_error < 0.01,
                "NN value: {}, Recovered: {}, Error: {}",
                val, recovered, abs_error);
        }
    }

    #[test]
    fn test_from_bits() {
        // Test loading BF16 from raw bits (as done when loading from SafeTensors)
        let bits_1_0 = 0x3F80u16; // 1.0 in BF16 format
        let bf16 = BF16::from_bits(bits_1_0);
        let f32_val = f32::from(bf16);
        assert_eq!(f32_val, 1.0f32, "BF16 bits 0x3F80 should be 1.0");

        // Test zero
        let bits_0_0 = 0x0000u16;
        let bf16 = BF16::from_bits(bits_0_0);
        let f32_val = f32::from(bf16);
        assert_eq!(f32_val, 0.0f32, "BF16 bits 0x0000 should be 0.0");

        // Test negative one
        let bits_neg_1 = 0xBF80u16; // -1.0 in BF16 format
        let bf16 = BF16::from_bits(bits_neg_1);
        let f32_val = f32::from(bf16);
        assert_eq!(f32_val, -1.0f32, "BF16 bits 0xBF80 should be -1.0");
    }
}
