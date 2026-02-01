//! Token embedding layer.
//!
//! Maps token IDs to dense embedding vectors.
//! For Qwen3-0.6B: [vocab_size=151936, hidden_size=1024]

use crate::error::{LludaError, Result};
use crate::tensor::Tensor;

/// Token embedding layer.
///
/// Performs lookup from a weight matrix: token ID -> embedding vector.
/// Weight shape: [vocab_size, hidden_size]
///
/// # Example
///
/// ```rust
/// use lluda_inference::embedding::Embedding;
/// use lluda_inference::tensor::Tensor;
///
/// // Create small embedding: 10 tokens, 4 dims
/// let weight = Tensor::new(
///     (0..40).map(|x| x as f32).collect(),
///     vec![10, 4]
/// ).unwrap();
/// let emb = Embedding::new(weight).unwrap();
///
/// // Lookup single token
/// let result = emb.forward(&[2], &[1]).unwrap();
/// assert_eq!(result.shape(), &[1, 4]);
/// ```
#[derive(Debug, Clone)]
pub struct Embedding {
    weight: Tensor,
}

impl Embedding {
    /// Create embedding layer from weight tensor.
    ///
    /// # Arguments
    ///
    /// * `weight` - Embedding matrix [vocab_size, hidden_size]
    ///
    /// # Returns
    ///
    /// New Embedding instance.
    ///
    /// # Errors
    ///
    /// Returns error if weight is not a 2D tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::embedding::Embedding;
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let weight = Tensor::new(vec![1.0; 100], vec![10, 10]).unwrap();
    /// let emb = Embedding::new(weight).unwrap();
    /// ```
    pub fn new(weight: Tensor) -> Result<Self> {
        if weight.ndim() != 2 {
            return Err(LludaError::ShapeMismatch {
                expected: vec![0, 0], // Expect 2D
                got: weight.shape().to_vec(),
            });
        }
        Ok(Embedding { weight })
    }

    /// Lookup embeddings for token indices.
    ///
    /// # Arguments
    ///
    /// * `indices` - Token IDs to lookup
    /// * `batch_shape` - Shape of the batch (e.g., [B] or [B, L])
    ///
    /// # Returns
    ///
    /// Tensor of shape `[batch_shape..., hidden_size]`
    /// All computation in F32 regardless of weight dtype.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Any index is out of range (>= vocab_size)
    /// - batch_shape product doesn't match indices length
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::embedding::Embedding;
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let weight = Tensor::new(
    ///     vec![1.0, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0],
    ///     vec![2, 4]
    /// ).unwrap();
    /// let emb = Embedding::new(weight);
    ///
    /// // Lookup batch: 2 sequences of 3 tokens each
    /// let indices = vec![0, 1, 0, 1, 0, 1];
    /// let result = emb.forward(&indices, &[2, 3]).unwrap();
    /// assert_eq!(result.shape(), &[2, 3, 4]);
    /// ```
    pub fn forward(&self, indices: &[u32], batch_shape: &[usize]) -> Result<Tensor> {
        // Validate batch_shape matches indices length
        let expected_len: usize = batch_shape.iter().product();
        if indices.len() != expected_len {
            return Err(LludaError::ShapeMismatch {
                expected: vec![expected_len],
                got: vec![indices.len()],
            });
        }

        // Get weight properties
        let weight_shape = self.weight.shape();
        if weight_shape.len() != 2 {
            return Err(LludaError::ShapeMismatch {
                expected: vec![0, 0], // Expect 2D
                got: weight_shape.to_vec(),
            });
        }

        let vocab_size = weight_shape[0];
        let hidden_size = weight_shape[1];

        // Validate all indices are in range
        for &idx in indices {
            if idx as usize >= vocab_size {
                return Err(LludaError::Msg(format!(
                    "Index {} out of range for vocab_size {}",
                    idx, vocab_size
                )));
            }
        }

        // Convert weight to F32 for lookup
        let weight_data = self.weight.to_vec_f32();

        // Perform embedding lookup
        let mut result = Vec::with_capacity(indices.len() * hidden_size);
        for &idx in indices {
            let row_start = idx as usize * hidden_size;
            let row_end = row_start + hidden_size;
            result.extend_from_slice(&weight_data[row_start..row_end]);
        }

        // Construct output shape: batch_shape + [hidden_size]
        let mut output_shape = batch_shape.to_vec();
        output_shape.push(hidden_size);

        Tensor::new(result, output_shape)
    }

    /// Get reference to embedding weight.
    ///
    /// Used for tied embeddings (tie_word_embeddings=true in Qwen3).
    /// The same weight matrix is used as LM head.
    ///
    /// # Returns
    ///
    /// Reference to the embedding weight tensor.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bf16::BF16;
    use crate::tensor::DType;

    #[test]
    fn test_embedding_single_token() {
        // Create small embedding: 10 tokens, 4 dims
        let weight = Tensor::new((0..40).map(|x| x as f32).collect(), vec![10, 4]).unwrap();
        let emb = Embedding::new(weight).unwrap();

        // Lookup token 2 (should get row 2: [8, 9, 10, 11])
        let result = emb.forward(&[2], &[1]).unwrap();

        assert_eq!(result.shape(), &[1, 4]);
        let data = result.to_vec_f32();
        assert_eq!(data, vec![8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn test_embedding_batch() {
        // Create 4-token vocab, 3 dims
        let weight = Tensor::new(
            vec![
                1.0, 2.0, 3.0, // token 0
                4.0, 5.0, 6.0, // token 1
                7.0, 8.0, 9.0, // token 2
                10.0, 11.0, 12.0, // token 3
            ],
            vec![4, 3],
        )
        .unwrap();
        let emb = Embedding::new(weight).unwrap();

        // Lookup [0, 1, 2] as batch of 3
        let result = emb.forward(&[0, 1, 2], &[3]).unwrap();

        assert_eq!(result.shape(), &[3, 3]);
        let data = result.to_vec_f32();
        assert_eq!(
            data,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        );
    }

    #[test]
    fn test_embedding_2d_batch() {
        // Simulate batch of sequences: [B=2, L=3]
        let weight = Tensor::new((0..20).map(|x| x as f32).collect(), vec![5, 4]).unwrap();
        let emb = Embedding::new(weight).unwrap();

        // 6 tokens total (2 sequences × 3 tokens)
        let indices = vec![0, 1, 2, 3, 4, 0];
        let result = emb.forward(&indices, &[2, 3]).unwrap();

        assert_eq!(result.shape(), &[2, 3, 4]);
        assert_eq!(result.numel(), 2 * 3 * 4);
    }

    #[test]
    fn test_embedding_out_of_range() {
        let weight = Tensor::new(vec![1.0; 20], vec![5, 4]).unwrap();
        let emb = Embedding::new(weight).unwrap();

        // Index 5 is out of range for vocab_size=5
        let result = emb.forward(&[5], &[1]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("out of range"));
    }

    #[test]
    fn test_embedding_shape_mismatch() {
        let weight = Tensor::new(vec![1.0; 20], vec![5, 4]).unwrap();
        let emb = Embedding::new(weight).unwrap();

        // 3 indices but batch_shape says 2
        let result = emb.forward(&[0, 1, 2], &[2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_embedding_bf16_weight() {
        // BF16 weight should be converted to F32 automatically
        let bf16_data: Vec<BF16> = (0..12).map(|x| BF16::from(x as f32)).collect();
        let weight = Tensor::from_bf16(bf16_data, vec![3, 4]).unwrap();
        let emb = Embedding::new(weight).unwrap();

        let result = emb.forward(&[1], &[1]).unwrap();

        assert_eq!(result.shape(), &[1, 4]);
        assert_eq!(result.dtype(), DType::F32);
        let data = result.to_vec_f32();
        // Row 1: [4, 5, 6, 7] (with BF16 precision loss)
        for (i, &val) in data.iter().enumerate() {
            let expected = (4 + i) as f32;
            assert!((val - expected).abs() < 0.01, "Expected {}, got {}", expected, val);
        }
    }

    #[test]
    fn test_embedding_weight_getter() {
        let weight = Tensor::new(vec![1.0; 20], vec![5, 4]).unwrap();
        let emb = Embedding::new(weight.clone()).unwrap();

        let retrieved = emb.weight();
        assert_eq!(retrieved.shape(), weight.shape());
        assert_eq!(retrieved.dtype(), weight.dtype());
    }

    #[test]
    fn test_embedding_zeros() {
        // All zeros should produce all zeros
        let weight = Tensor::zeros(&[3, 4], DType::F32).unwrap();
        let emb = Embedding::new(weight).unwrap();

        let result = emb.forward(&[0, 1, 2], &[3]).unwrap();
        let data = result.to_vec_f32();
        assert!(data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_embedding_same_token_multiple_times() {
        let weight = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        ).unwrap();
        let emb = Embedding::new(weight).unwrap();

        // Lookup same token multiple times
        let result = emb.forward(&[1, 1, 1], &[3]).unwrap();

        assert_eq!(result.shape(), &[3, 3]);
        let data = result.to_vec_f32();
        // Each lookup should return [4.0, 5.0, 6.0]
        assert_eq!(
            data,
            vec![4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn test_embedding_empty_indices() {
        let weight = Tensor::new(vec![1.0; 12], vec![3, 4]).unwrap();
        let emb = Embedding::new(weight).unwrap();

        let result = emb.forward(&[], &[0]).unwrap();
        assert_eq!(result.shape(), &[0, 4]);
        assert_eq!(result.numel(), 0);
    }

    #[test]
    fn test_embedding_non_2d_weight() {
        // Weight must be 2D - should fail at construction time
        let weight = Tensor::new(vec![1.0; 24], vec![2, 3, 4]).unwrap();
        let result = Embedding::new(weight);
        assert!(result.is_err());
        match result.unwrap_err() {
            LludaError::ShapeMismatch { expected, got } => {
                assert_eq!(expected, vec![0, 0]);
                assert_eq!(got, vec![2, 3, 4]);
            }
            other => panic!("Expected ShapeMismatch, got: {:?}", other),
        }
    }

    #[test]
    fn test_embedding_1d_weight_fails() {
        // 1D weight should fail
        let weight = Tensor::new(vec![1.0; 10], vec![10]).unwrap();
        let result = Embedding::new(weight);
        assert!(result.is_err());
    }
}
