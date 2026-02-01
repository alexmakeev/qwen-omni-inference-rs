//! Causal attention mask generation for autoregressive models.
//!
//! Provides functions to create causal masks that prevent attention to future tokens
//! during transformer forward passes.
//!
//! # Causal Masking
//!
//! In autoregressive language models, each token can only attend to itself and previous
//! tokens, not future ones. This is enforced by adding a causal mask to attention scores
//! before applying softmax.
//!
//! The mask contains:
//! - `0.0` for positions that can be attended to
//! - `-inf` for positions that should be masked out
//!
//! When added to attention scores, `-inf` positions become 0 after softmax.
//!
//! # Example
//!
//! ```rust
//! use lluda_inference::causal_mask::causal_mask;
//! use lluda_inference::tensor::DType;
//!
//! // Generate mask for prompt processing (seq_len > 1)
//! let mask = causal_mask(1, 4, 0, DType::F32).unwrap();
//! assert!(mask.is_some());
//! let mask = mask.unwrap();
//! assert_eq!(mask.shape(), &[1, 1, 4, 4]);
//!
//! // During generation (seq_len == 1), no mask needed
//! let mask = causal_mask(1, 1, 5, DType::F32).unwrap();
//! assert!(mask.is_none());
//! ```
//!
//! # References
//!
//! - Attention is All You Need: https://arxiv.org/abs/1706.03762
//! - Qwen3 attention implementation in candle

use crate::error::Result;
use crate::tensor::{DType, Tensor};

/// Generate a causal attention mask for autoregressive generation.
///
/// Creates a mask that prevents attention to future tokens. The mask is added
/// to attention scores before softmax, with `-inf` values masking out future positions.
///
/// # Arguments
///
/// * `batch_size` - Number of sequences in the batch
/// * `seq_len` - Length of the current sequence (query length)
/// * `offset` - Number of tokens already cached (past key/value length)
/// * `dtype` - Data type for the mask tensor
///
/// # Returns
///
/// - `Ok(Some(mask))` - Causal mask of shape `[batch_size, 1, seq_len, seq_len + offset]`
/// - `Ok(None)` - When `seq_len == 1` (single-token generation, no masking needed)
///
/// # Mask Layout
///
/// For prompt processing (`seq_len > 1`, `offset = 0`):
/// ```text
/// Position: 0    1    2    3
///        0  0.0  -inf -inf -inf
///        1  0.0  0.0  -inf -inf
///        2  0.0  0.0  0.0  -inf
///        3  0.0  0.0  0.0  0.0
/// ```
///
/// For generation with cache (`seq_len = 1`, `offset = 3`):
/// - Returns `None` (all cached tokens are visible to the new token)
///
/// For continuation with cache (`seq_len = 2`, `offset = 2`):
/// ```text
/// Past (cached):  0    1  | New:  2    3
///             2   0.0  0.0 |       0.0  -inf
///             3   0.0  0.0 |       0.0  0.0
/// ```
///
/// # Optimization
///
/// During single-token generation (`seq_len == 1`), the new token can attend to all
/// previous tokens (from KV cache), so no masking is needed. This function returns
/// `None` to avoid allocating unnecessary mask tensors.
///
/// # Example
///
/// ```rust
/// use lluda_inference::causal_mask::causal_mask;
/// use lluda_inference::tensor::DType;
///
/// // Prompt processing: 4 tokens, no cache
/// let mask = causal_mask(1, 4, 0, DType::F32).unwrap().unwrap();
/// assert_eq!(mask.shape(), &[1, 1, 4, 4]);
///
/// // Verify lower triangular structure
/// let data = mask.to_vec_f32();
/// // Position [0, 0, 0, 1] should be -inf (future token)
/// assert!(data[1].is_infinite() && data[1].is_sign_negative());
/// // Position [0, 0, 1, 0] should be 0.0 (past token)
/// assert_eq!(data[4], 0.0);
/// ```
pub fn causal_mask(
    batch_size: usize,
    seq_len: usize,
    offset: usize,
    dtype: DType,
) -> Result<Option<Tensor>> {
    // Validate input parameters
    if batch_size == 0 {
        return Err(crate::error::LludaError::Msg(
            "batch_size must be greater than 0".to_string()
        ));
    }
    if seq_len == 0 {
        return Err(crate::error::LludaError::Msg(
            "seq_len must be greater than 0".to_string()
        ));
    }

    // Optimization: No mask needed for single-token generation
    // The new token can attend to all cached tokens (all positions are valid)
    if seq_len == 1 {
        return Ok(None);
    }

    let total_len = seq_len + offset;

    // Create mask: [batch_size, 1, seq_len, total_len]
    let numel = batch_size * seq_len * total_len;
    let mut mask_data = vec![0.0f32; numel];

    // Fill in -inf for positions where j > i + offset
    // This creates a lower-triangular pattern shifted by offset
    for b in 0..batch_size {
        for i in 0..seq_len {
            for j in 0..total_len {
                // Current token at absolute position (i + offset)
                // Can attend to tokens at positions 0..=(i + offset)
                // Cannot attend to tokens at positions (i + offset + 1)..total_len
                if j > i + offset {
                    let idx = (b * seq_len + i) * total_len + j;
                    mask_data[idx] = f32::NEG_INFINITY;
                }
            }
        }
    }

    // Convert to appropriate dtype
    let mask = match dtype {
        DType::F32 => Tensor::new(mask_data, vec![batch_size, 1, seq_len, total_len])?,
        DType::BF16 => {
            // Convert F32 to BF16
            let bf16_data: Vec<_> = mask_data
                .iter()
                .map(|&x| crate::bf16::BF16::from(x))
                .collect();
            Tensor::from_bf16(bf16_data, vec![batch_size, 1, seq_len, total_len])?
        }
    };

    Ok(Some(mask))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_mask_single_token_returns_none() {
        // Single-token generation should return None (no mask needed)
        let result = causal_mask(1, 1, 0, DType::F32).unwrap();
        assert!(result.is_none(), "Expected None for seq_len=1, offset=0");

        let result = causal_mask(1, 1, 5, DType::F32).unwrap();
        assert!(result.is_none(), "Expected None for seq_len=1, offset=5");

        let result = causal_mask(2, 1, 10, DType::F32).unwrap();
        assert!(
            result.is_none(),
            "Expected None for seq_len=1 with batch_size=2"
        );
    }

    #[test]
    fn test_causal_mask_prompt_no_offset() {
        // 4-token prompt, no cache
        let mask = causal_mask(1, 4, 0, DType::F32).unwrap().unwrap();
        assert_eq!(
            mask.shape(),
            &[1, 1, 4, 4],
            "Mask shape should be [B, 1, L, L]"
        );

        let data = mask.to_vec_f32();

        // Verify lower triangular structure
        // Row 0: only position 0 visible
        assert_eq!(data[0], 0.0, "Position [0,0] should be 0.0");
        assert!(
            data[1].is_infinite() && data[1].is_sign_negative(),
            "Position [0,1] should be -inf"
        );
        assert!(
            data[2].is_infinite() && data[2].is_sign_negative(),
            "Position [0,2] should be -inf"
        );
        assert!(
            data[3].is_infinite() && data[3].is_sign_negative(),
            "Position [0,3] should be -inf"
        );

        // Row 1: positions 0,1 visible
        assert_eq!(data[4], 0.0, "Position [1,0] should be 0.0");
        assert_eq!(data[5], 0.0, "Position [1,1] should be 0.0");
        assert!(
            data[6].is_infinite() && data[6].is_sign_negative(),
            "Position [1,2] should be -inf"
        );
        assert!(
            data[7].is_infinite() && data[7].is_sign_negative(),
            "Position [1,3] should be -inf"
        );

        // Row 3: all positions visible
        assert_eq!(data[12], 0.0, "Position [3,0] should be 0.0");
        assert_eq!(data[13], 0.0, "Position [3,1] should be 0.0");
        assert_eq!(data[14], 0.0, "Position [3,2] should be 0.0");
        assert_eq!(data[15], 0.0, "Position [3,3] should be 0.0");
    }

    #[test]
    fn test_causal_mask_with_offset() {
        // 3 new tokens, 2 cached tokens
        let mask = causal_mask(1, 3, 2, DType::F32).unwrap().unwrap();
        assert_eq!(
            mask.shape(),
            &[1, 1, 3, 5],
            "Mask shape should be [1, 1, 3, 5] (seq_len=3, total=5)"
        );

        let data = mask.to_vec_f32();

        // Row 0 (absolute position 2): can attend to positions 0,1,2
        assert_eq!(data[0], 0.0, "Position [0,0] (cached) should be 0.0");
        assert_eq!(data[1], 0.0, "Position [0,1] (cached) should be 0.0");
        assert_eq!(data[2], 0.0, "Position [0,2] (current) should be 0.0");
        assert!(
            data[3].is_infinite() && data[3].is_sign_negative(),
            "Position [0,3] (future) should be -inf"
        );
        assert!(
            data[4].is_infinite() && data[4].is_sign_negative(),
            "Position [0,4] (future) should be -inf"
        );

        // Row 1 (absolute position 3): can attend to positions 0,1,2,3
        assert_eq!(data[5], 0.0, "Position [1,0] should be 0.0");
        assert_eq!(data[6], 0.0, "Position [1,1] should be 0.0");
        assert_eq!(data[7], 0.0, "Position [1,2] should be 0.0");
        assert_eq!(data[8], 0.0, "Position [1,3] should be 0.0");
        assert!(
            data[9].is_infinite() && data[9].is_sign_negative(),
            "Position [1,4] should be -inf"
        );

        // Row 2 (absolute position 4): can attend to all positions 0,1,2,3,4
        for i in 0..5 {
            assert_eq!(
                data[10 + i],
                0.0,
                "Position [2,{}] should be 0.0 (all visible)",
                i
            );
        }
    }

    #[test]
    fn test_causal_mask_batch_size() {
        // Test with batch_size > 1
        let mask = causal_mask(2, 3, 0, DType::F32).unwrap().unwrap();
        assert_eq!(
            mask.shape(),
            &[2, 1, 3, 3],
            "Mask shape should be [2, 1, 3, 3]"
        );

        let data = mask.to_vec_f32();

        // Each batch should have identical mask patterns
        let first_batch = &data[0..9];
        let second_batch = &data[9..18];

        for i in 0..9 {
            if first_batch[i].is_finite() {
                assert_eq!(
                    first_batch[i], second_batch[i],
                    "Batch patterns should be identical"
                );
            } else {
                assert!(
                    second_batch[i].is_infinite() && second_batch[i].is_sign_negative(),
                    "Both batches should have -inf at same positions"
                );
            }
        }
    }

    #[test]
    fn test_causal_mask_bf16_dtype() {
        // Test BF16 mask generation
        let mask = causal_mask(1, 3, 0, DType::BF16).unwrap().unwrap();
        assert_eq!(mask.dtype(), DType::BF16, "Mask dtype should be BF16");
        assert_eq!(mask.shape(), &[1, 1, 3, 3]);

        // Convert to F32 to verify values
        let data = mask.to_vec_f32();

        // Verify lower triangular pattern preserved
        assert_eq!(data[0], 0.0);
        assert!(data[1].is_infinite() && data[1].is_sign_negative());
        assert_eq!(data[4], 0.0);
    }

    #[test]
    fn test_causal_mask_zero_offset() {
        // Explicit test: offset=0 means no cache, standard causal mask
        let mask = causal_mask(1, 4, 0, DType::F32).unwrap().unwrap();
        let data = mask.to_vec_f32();

        // Diagonal and below should be 0.0, above diagonal should be -inf
        for i in 0..4 {
            for j in 0..4 {
                let idx = i * 4 + j;
                if j > i {
                    assert!(
                        data[idx].is_infinite() && data[idx].is_sign_negative(),
                        "Position [{},{}] should be -inf",
                        i,
                        j
                    );
                } else {
                    assert_eq!(data[idx], 0.0, "Position [{},{}] should be 0.0", i, j);
                }
            }
        }
    }

    #[test]
    fn test_causal_mask_large_offset() {
        // Test with large cache (simulating long generation sequence)
        let mask = causal_mask(1, 2, 100, DType::F32).unwrap().unwrap();
        assert_eq!(
            mask.shape(),
            &[1, 1, 2, 102],
            "Mask should span cache + new tokens"
        );

        let data = mask.to_vec_f32();

        // First new token (absolute position 100): can attend to all 101 positions (0..100)
        for (j, value) in data.iter().enumerate().take(101) {
            assert_eq!(
                *value,
                0.0,
                "First new token at position {} should attend to all cached positions", j
            );
        }
        assert!(
            data[101].is_infinite() && data[101].is_sign_negative(),
            "First new token cannot attend to second new token"
        );

        // Second new token (absolute position 101): can attend to all positions
        for j in 0..102 {
            assert_eq!(
                data[102 + j],
                0.0,
                "Second new token should attend to all positions"
            );
        }
    }

    #[test]
    fn test_causal_mask_edge_case_seq_len_2() {
        // Minimal non-trivial case: 2 tokens
        let mask = causal_mask(1, 2, 0, DType::F32).unwrap().unwrap();
        assert_eq!(mask.shape(), &[1, 1, 2, 2]);

        let data = mask.to_vec_f32();

        // Token 0: can only see itself
        assert_eq!(data[0], 0.0);
        assert!(data[1].is_infinite() && data[1].is_sign_negative());

        // Token 1: can see both tokens
        assert_eq!(data[2], 0.0);
        assert_eq!(data[3], 0.0);
    }
}
