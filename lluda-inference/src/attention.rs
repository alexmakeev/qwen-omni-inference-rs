//! Grouped Query Attention (GQA) for Qwen3 models.
//!
//! Implements multi-head attention with grouped key-value heads (GQA),
//! per-head query/key normalization (RMSNorm), and rotary position embeddings (RoPE).
//!
//! # Architecture
//!
//! For Qwen3-0.6B:
//! - Query heads: 16 (num_attention_heads)
//! - KV heads: 8 (num_key_value_heads)
//! - Head dimension: 128
//! - Each KV head is shared by 2 query heads (num_kv_groups = 2)
//!
//! # References
//!
//! - Grouped Query Attention: https://arxiv.org/abs/2305.13245
//! - RoFormer (RoPE): https://arxiv.org/abs/2104.09864

use std::sync::Arc;

use crate::error::Result;
use crate::rms_norm::RmsNorm;
use crate::rope::RotaryEmbedding;
use crate::tensor::Tensor;

/// Simple linear transformation layer (weight matrix multiplication).
///
/// Computes `x @ weight.T` where weight is stored in [out_features, in_features] format
/// (matching PyTorch convention).
///
/// Qwen3 does not use bias terms in attention projections (`attention_bias: false`).
#[derive(Debug, Clone)]
pub struct Linear {
    /// Weight matrix [out_features, in_features]
    weight: Tensor,
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
    /// New Linear instance.
    ///
    /// # Errors
    ///
    /// Returns error if weight is not a 2D tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::attention::Linear;
    /// use lluda_inference::tensor::Tensor;
    ///
    /// // Linear layer: 4 inputs -> 8 outputs
    /// let weight = Tensor::new(vec![0.0; 32], vec![8, 4]).unwrap();
    /// let linear = Linear::new(weight).unwrap();
    /// ```
    pub fn new(weight: Tensor) -> Result<Self> {
        if weight.ndim() != 2 {
            return Err(crate::error::LludaError::ShapeMismatch {
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
    /// Returns `ShapeMismatch` if the last dimension of `x` doesn't match weight's in_features.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::attention::Linear;
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let weight = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let linear = Linear::new(weight);
    ///
    /// let x = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
    /// let y = linear.forward(&x).unwrap();
    ///
    /// assert_eq!(y.shape(), &[1, 2]);
    /// ```
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Weight is [out_features, in_features]
        // Input x can be [..., in_features]
        // Output should be [..., out_features]

        let input_shape = x.shape();
        let weight_shape = self.weight.shape();

        let out_features = weight_shape[0];
        let in_features = weight_shape[1];

        // Validate last dimension matches
        if input_shape[input_shape.len() - 1] != in_features {
            return Err(crate::error::LludaError::ShapeMismatch {
                expected: vec![in_features],
                got: vec![input_shape[input_shape.len() - 1]],
            });
        }

        // Flatten all leading dimensions into batch
        // [..., in_features] -> [batch, in_features]
        let batch_size: usize = input_shape[..input_shape.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch_size, in_features])?;

        // Transpose weight: [out_features, in_features] -> [in_features, out_features]
        let weight_t = self.weight.transpose()?;

        // Compute: [batch, in_features] @ [in_features, out_features] -> [batch, out_features]
        let output_2d = x_2d.matmul(&weight_t)?;

        // Reshape back to original shape with last dim = out_features
        let mut output_shape = input_shape[..input_shape.len() - 1].to_vec();
        output_shape.push(out_features);

        output_2d.reshape(&output_shape)
    }
}

/// Grouped Query Attention with per-head Q/K normalization.
///
/// Key features:
/// - Grouped query attention: Multiple query heads share KV heads
/// - Per-head RMSNorm on queries and keys (unique to Qwen3)
/// - Rotary position embeddings (RoPE)
/// - KV cache support for autoregressive generation
///
/// # Architecture (Qwen3-0.6B)
///
/// - Query heads: 16
/// - KV heads: 8
/// - Head dimension: 128
/// - KV groups: 2 (each KV head serves 2 query heads)
#[derive(Debug, Clone)]
pub struct Attention {
    /// Query projection [hidden_size, num_heads * head_dim]
    q_proj: Linear,
    /// Key projection [hidden_size, num_kv_heads * head_dim]
    k_proj: Linear,
    /// Value projection [hidden_size, num_kv_heads * head_dim]
    v_proj: Linear,
    /// Output projection [num_heads * head_dim, hidden_size]
    o_proj: Linear,
    /// Query normalization [head_dim]
    q_norm: RmsNorm,
    /// Key normalization [head_dim]
    k_norm: RmsNorm,
    /// Rotary position embeddings
    rotary: Arc<RotaryEmbedding>,
    /// Number of query heads
    num_heads: usize,
    /// Number of key/value heads
    num_kv_heads: usize,
    /// Number of query heads per KV head
    num_kv_groups: usize,
    /// Dimension of each head
    head_dim: usize,
}

impl Attention {
    /// Create a new attention layer.
    ///
    /// # Arguments
    ///
    /// * `q_proj` - Query projection layer
    /// * `k_proj` - Key projection layer
    /// * `v_proj` - Value projection layer
    /// * `o_proj` - Output projection layer
    /// * `q_norm` - Query normalization layer
    /// * `k_norm` - Key normalization layer
    /// * `rotary` - Rotary position embeddings
    /// * `num_heads` - Number of query attention heads
    /// * `num_kv_heads` - Number of key/value heads
    /// * `head_dim` - Dimension of each attention head
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::sync::Arc;
    /// use lluda_inference::attention::{Attention, Linear};
    /// use lluda_inference::rms_norm::RmsNorm;
    /// use lluda_inference::rope::RotaryEmbedding;
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let hidden_size = 64;
    /// let num_heads = 4;
    /// let num_kv_heads = 2;
    /// let head_dim = 16;
    ///
    /// let q_proj = Linear::new(Tensor::new(vec![0.0; num_heads * head_dim * hidden_size], vec![num_heads * head_dim, hidden_size]).unwrap());
    /// let k_proj = Linear::new(Tensor::new(vec![0.0; num_kv_heads * head_dim * hidden_size], vec![num_kv_heads * head_dim, hidden_size]).unwrap());
    /// let v_proj = Linear::new(Tensor::new(vec![0.0; num_kv_heads * head_dim * hidden_size], vec![num_kv_heads * head_dim, hidden_size]).unwrap());
    /// let o_proj = Linear::new(Tensor::new(vec![0.0; hidden_size * num_heads * head_dim], vec![hidden_size, num_heads * head_dim]).unwrap());
    /// let q_norm = RmsNorm::new(Tensor::new(vec![1.0; head_dim], vec![head_dim]).unwrap(), 1e-6);
    /// let k_norm = RmsNorm::new(Tensor::new(vec![1.0; head_dim], vec![head_dim]).unwrap(), 1e-6);
    /// let rotary = Arc::new(RotaryEmbedding::new(head_dim, 100, 10000.0).unwrap());
    ///
    /// let attn = Attention::new(q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, rotary, num_heads, num_kv_heads, head_dim);
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        q_norm: RmsNorm,
        k_norm: RmsNorm,
        rotary: Arc<RotaryEmbedding>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        #[allow(clippy::manual_is_multiple_of)]
        {
            assert!(
                num_heads % num_kv_heads == 0,
                "num_heads ({}) must be divisible by num_kv_heads ({})",
                num_heads,
                num_kv_heads
            );
        }

        // Validate projection weight shapes
        let q_shape = q_proj.weight.shape();
        let k_shape = k_proj.weight.shape();
        let v_shape = v_proj.weight.shape();
        let o_shape = o_proj.weight.shape();

        assert_eq!(
            q_shape[0],
            num_heads * head_dim,
            "q_proj output features ({}) must equal num_heads * head_dim ({})",
            q_shape[0],
            num_heads * head_dim
        );
        assert_eq!(
            k_shape[0],
            num_kv_heads * head_dim,
            "k_proj output features ({}) must equal num_kv_heads * head_dim ({})",
            k_shape[0],
            num_kv_heads * head_dim
        );
        assert_eq!(
            v_shape[0],
            num_kv_heads * head_dim,
            "v_proj output features ({}) must equal num_kv_heads * head_dim ({})",
            v_shape[0],
            num_kv_heads * head_dim
        );
        assert_eq!(
            o_shape[1],
            num_heads * head_dim,
            "o_proj input features ({}) must equal num_heads * head_dim ({})",
            o_shape[1],
            num_heads * head_dim
        );

        let num_kv_groups = num_heads / num_kv_heads;

        Attention {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rotary,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
        }
    }

    /// Forward pass through attention layer.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape [B, L, hidden_size]
    /// * `mask` - Optional attention mask of shape [B, 1, L, L+offset]
    /// * `kv_cache` - Mutable KV cache (will be updated with new k, v)
    /// * `offset` - Position offset (length of cached sequence)
    ///
    /// # Returns
    ///
    /// Output tensor of shape [B, L, hidden_size]
    ///
    /// # Errors
    ///
    /// Returns error if shapes are incompatible or operations fail.
    ///
    /// # Process
    ///
    /// 1. Project to Q, K, V
    /// 2. Reshape to multi-head format
    /// 3. Apply per-head RMSNorm to Q and K
    /// 4. Apply RoPE to Q and K
    /// 5. Update KV cache
    /// 6. Repeat KV heads for GQA
    /// 7. Compute attention scores
    /// 8. Apply mask and softmax
    /// 9. Compute attention output
    /// 10. Reshape and project output
    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        kv_cache: &mut KvCache,
        offset: usize,
    ) -> Result<Tensor> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        // 1. Project to Q, K, V
        let q = self.q_proj.forward(x)?; // [B, L, num_heads * head_dim]
        let k = self.k_proj.forward(x)?; // [B, L, num_kv_heads * head_dim]
        let v = self.v_proj.forward(x)?; // [B, L, num_kv_heads * head_dim]

        // 2. Reshape to multi-head format
        // Q: [B, L, num_heads * head_dim] -> [B, num_heads, L, head_dim]
        let q = q
            .reshape(&[batch, seq_len, self.num_heads, self.head_dim])?
            .transpose_dims(1, 2)?;

        // K, V: [B, L, num_kv_heads * head_dim] -> [B, num_kv_heads, L, head_dim]
        let k = k
            .reshape(&[batch, seq_len, self.num_kv_heads, self.head_dim])?
            .transpose_dims(1, 2)?;
        let v = v
            .reshape(&[batch, seq_len, self.num_kv_heads, self.head_dim])?
            .transpose_dims(1, 2)?;

        // 3. Per-head RMSNorm on Q and K
        // Flatten [B, H, L, D] -> [B*H*L, D] for per-head norm
        let q_flat = q.reshape(&[batch * self.num_heads * seq_len, self.head_dim])?;
        let k_flat = k.reshape(&[batch * self.num_kv_heads * seq_len, self.head_dim])?;

        let q_normed = self.q_norm.forward(&q_flat)?;
        let k_normed = self.k_norm.forward(&k_flat)?;

        // Reshape back to [B, H, L, D]
        let q = q_normed.reshape(&[batch, self.num_heads, seq_len, self.head_dim])?;
        let k = k_normed.reshape(&[batch, self.num_kv_heads, seq_len, self.head_dim])?;

        // 4. Apply RoPE
        let (q, k) = self.rotary.apply(&q, &k, offset)?;

        // 5. Update KV cache and get full K, V
        let (k, v) = kv_cache.append(&k, &v)?;

        // 6. Repeat KV heads for GQA
        // [B, num_kv_heads, L, D] -> [B, num_heads, L, D]
        let k = repeat_kv(&k, self.num_kv_groups)?;
        let v = repeat_kv(&v, self.num_kv_groups)?;

        // 7. Compute attention scores: Q @ K.T / sqrt(head_dim)
        // Q: [B, num_heads, L_q, D]
        // K: [B, num_heads, L_kv, D] -> transpose -> [B, num_heads, D, L_kv]
        let k_t = k.transpose_dims(2, 3)?; // [B, H, L_kv, D] -> [B, H, D, L_kv]
        let scores = q.matmul(&k_t)?; // [B, H, L_q, L_kv]

        // Scale by sqrt(head_dim)
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let scores = scores.mul_scalar(scale)?;

        // 8. Apply mask (if provided) and softmax
        let scores = if let Some(mask) = mask {
            scores.add(mask)?
        } else {
            scores
        };

        let attn_weights = scores.softmax(3)?; // Softmax over last dim (L_kv)

        // 9. Compute attention output: attn_weights @ V
        // attn_weights: [B, H, L_q, L_kv]
        // V: [B, H, L_kv, D]
        // output: [B, H, L_q, D]
        let attn_output = attn_weights.matmul(&v)?;

        // 10. Reshape and project output
        // [B, H, L, D] -> [B, L, H, D] -> [B, L, H*D]
        let attn_output = attn_output.transpose_dims(1, 2)?; // [B, L, H, D]
        let attn_output = attn_output.reshape(&[batch, seq_len, self.num_heads * self.head_dim])?;

        // Final output projection
        self.o_proj.forward(&attn_output)
    }
}

/// Repeat KV heads for Grouped Query Attention.
///
/// Expands KV tensors from num_kv_heads to num_heads by repeating each KV head
/// num_kv_groups times.
///
/// # Arguments
///
/// * `x` - Input tensor of shape [B, num_kv_heads, L, D]
/// * `num_groups` - Number of times to repeat each KV head
///
/// # Returns
///
/// Tensor of shape [B, num_kv_heads * num_groups, L, D]
///
/// # Example
///
/// For Qwen3-0.6B:
/// - Input: [B, 8, L, 128] (8 KV heads)
/// - Groups: 2
/// - Output: [B, 16, L, 128] (16 query heads)
fn repeat_kv(x: &Tensor, num_groups: usize) -> Result<Tensor> {
    if num_groups == 1 {
        // No repetition needed
        return Ok(x.clone());
    }

    let shape = x.shape();
    let batch = shape[0];
    let num_kv_heads = shape[1];
    let seq_len = shape[2];
    let head_dim = shape[3];

    let data = x.to_vec_f32();
    let mut result = Vec::with_capacity(data.len() * num_groups);

    // Repeat each KV head num_groups times
    for b in 0..batch {
        for kv_head in 0..num_kv_heads {
            for _group in 0..num_groups {
                for pos in 0..seq_len {
                    for d in 0..head_dim {
                        let idx = ((b * num_kv_heads + kv_head) * seq_len + pos) * head_dim + d;
                        result.push(data[idx]);
                    }
                }
            }
        }
    }

    Tensor::new(result, vec![batch, num_kv_heads * num_groups, seq_len, head_dim])
}

/// KV cache for autoregressive generation.
///
/// Stores key and value tensors from previous tokens to avoid recomputation
/// during sequential generation.
#[derive(Debug, Clone)]
pub struct KvCache {
    /// Cached keys [B, num_kv_heads, seq_so_far, head_dim]
    k: Option<Tensor>,
    /// Cached values [B, num_kv_heads, seq_so_far, head_dim]
    v: Option<Tensor>,
}

impl KvCache {
    /// Create a new empty KV cache.
    pub fn new() -> Self {
        KvCache { k: None, v: None }
    }

    /// Append new key and value tensors to the cache.
    ///
    /// # Arguments
    ///
    /// * `k` - New key tensor [B, num_kv_heads, L, D]
    /// * `v` - New value tensor [B, num_kv_heads, L, D]
    ///
    /// # Returns
    ///
    /// Tuple of (full_k, full_v) tensors including all cached tokens.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::attention::KvCache;
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let mut cache = KvCache::new();
    ///
    /// // First call: cache is empty, returns input as-is
    /// let k1 = Tensor::new(vec![1.0; 32], vec![1, 2, 4, 4]).unwrap();
    /// let v1 = Tensor::new(vec![2.0; 32], vec![1, 2, 4, 4]).unwrap();
    /// let (k_full, v_full) = cache.append(&k1, &v1).unwrap();
    /// assert_eq!(k_full.shape(), &[1, 2, 4, 4]);
    ///
    /// // Second call: concatenates with cached values
    /// let k2 = Tensor::new(vec![3.0; 8], vec![1, 2, 1, 4]).unwrap();
    /// let v2 = Tensor::new(vec![4.0; 8], vec![1, 2, 1, 4]).unwrap();
    /// let (k_full, v_full) = cache.append(&k2, &v2).unwrap();
    /// assert_eq!(k_full.shape(), &[1, 2, 5, 4]); // seq_len = 4 + 1 = 5
    /// ```
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let k_new = match &self.k {
            None => {
                // First append: just store
                k.clone()
            }
            Some(k_cached) => {
                // Concatenate along sequence dimension (dim=2)
                Tensor::cat(&[k_cached, k], 2)?
            }
        };

        let v_new = match &self.v {
            None => {
                // First append: just store
                v.clone()
            }
            Some(v_cached) => {
                // Concatenate along sequence dimension (dim=2)
                Tensor::cat(&[v_cached, v], 2)?
            }
        };

        // Update cache
        self.k = Some(k_new.clone());
        self.v = Some(v_new.clone());

        Ok((k_new, v_new))
    }

    /// Reset the cache (clear all stored key-value pairs).
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::attention::KvCache;
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let mut cache = KvCache::new();
    /// let k = Tensor::new(vec![1.0; 32], vec![1, 2, 4, 4]).unwrap();
    /// let v = Tensor::new(vec![2.0; 32], vec![1, 2, 4, 4]).unwrap();
    ///
    /// cache.append(&k, &v).unwrap();
    /// assert_eq!(cache.seq_len(), 4);
    ///
    /// cache.reset();
    /// assert_eq!(cache.seq_len(), 0);
    /// ```
    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }

    /// Get the current sequence length stored in the cache.
    ///
    /// # Returns
    ///
    /// Number of tokens currently cached (0 if empty).
    pub fn seq_len(&self) -> usize {
        self.k.as_ref().map_or(0, |k| k.shape()[2])
    }
}

impl Default for KvCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward() {
        // Weight matrix [2, 3]: maps 3 inputs to 2 outputs
        let weight = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let linear = Linear::new(weight).unwrap();

        // Input [1, 3]
        let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let y = linear.forward(&x).unwrap();

        assert_eq!(y.shape(), &[1, 2]);

        // Verify output values
        // x @ weight.T = [1,2,3] @ [[1,4], [2,5], [3,6]]
        // = [1*1+2*2+3*3, 1*4+2*5+3*6] = [14, 32]
        let data = y.to_vec_f32();
        assert_eq!(data.len(), 2);
        assert!((data[0] - 14.0).abs() < 1e-5);
        assert!((data[1] - 32.0).abs() < 1e-5);
    }

    #[test]
    fn test_linear_non_2d_weight_fails() {
        // Weight must be 2D - should fail at construction time
        let weight_3d = Tensor::new(vec![1.0; 24], vec![2, 3, 4]).unwrap();
        let result = Linear::new(weight_3d);
        assert!(result.is_err());
        match result.unwrap_err() {
            crate::error::LludaError::ShapeMismatch { expected, got } => {
                assert_eq!(expected, vec![0, 0]);
                assert_eq!(got, vec![2, 3, 4]);
            }
            other => panic!("Expected ShapeMismatch, got: {:?}", other),
        }
    }

    #[test]
    fn test_linear_1d_weight_fails() {
        // 1D weight should fail
        let weight_1d = Tensor::new(vec![1.0; 10], vec![10]).unwrap();
        let result = Linear::new(weight_1d);
        assert!(result.is_err());
    }

    #[test]
    fn test_repeat_kv_no_groups() {
        let x = Tensor::new(vec![1.0; 32], vec![1, 2, 4, 4]).unwrap();
        let result = repeat_kv(&x, 1).unwrap();

        assert_eq!(result.shape(), x.shape());
    }

    #[test]
    fn test_repeat_kv_with_groups() {
        // [B=1, num_kv_heads=2, L=3, D=4]
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let x = Tensor::new(data, vec![1, 2, 3, 4]).unwrap();

        let result = repeat_kv(&x, 2).unwrap();

        // Should expand to [1, 4, 3, 4] (2 heads * 2 groups = 4 heads)
        assert_eq!(result.shape(), &[1, 4, 3, 4]);

        let result_data = result.to_vec_f32();

        // First group should repeat head 0
        for i in 0..12 {
            assert_eq!(result_data[i], i as f32);
        }
        // Second group should also be head 0
        for i in 0..12 {
            assert_eq!(result_data[12 + i], i as f32);
        }
        // Third group should be head 1
        for i in 0..12 {
            assert_eq!(result_data[24 + i], (12 + i) as f32);
        }
        // Fourth group should also be head 1
        for i in 0..12 {
            assert_eq!(result_data[36 + i], (12 + i) as f32);
        }
    }

    #[test]
    fn test_kv_cache_first_append() {
        let mut cache = KvCache::new();

        let k = Tensor::new(vec![1.0; 32], vec![1, 2, 4, 4]).unwrap();
        let v = Tensor::new(vec![2.0; 32], vec![1, 2, 4, 4]).unwrap();

        let (k_full, v_full) = cache.append(&k, &v).unwrap();

        assert_eq!(k_full.shape(), &[1, 2, 4, 4]);
        assert_eq!(v_full.shape(), &[1, 2, 4, 4]);
        assert_eq!(cache.seq_len(), 4);
    }

    #[test]
    fn test_kv_cache_multiple_appends() {
        let mut cache = KvCache::new();

        // First append: 4 tokens
        let k1 = Tensor::new(vec![1.0; 32], vec![1, 2, 4, 4]).unwrap();
        let v1 = Tensor::new(vec![2.0; 32], vec![1, 2, 4, 4]).unwrap();
        cache.append(&k1, &v1).unwrap();

        // Second append: 1 token
        let k2 = Tensor::new(vec![3.0; 8], vec![1, 2, 1, 4]).unwrap();
        let v2 = Tensor::new(vec![4.0; 8], vec![1, 2, 1, 4]).unwrap();
        let (k_full, v_full) = cache.append(&k2, &v2).unwrap();

        // Should have 5 tokens total
        assert_eq!(k_full.shape(), &[1, 2, 5, 4]);
        assert_eq!(v_full.shape(), &[1, 2, 5, 4]);
        assert_eq!(cache.seq_len(), 5);
    }

    #[test]
    fn test_kv_cache_reset() {
        let mut cache = KvCache::new();

        let k = Tensor::new(vec![1.0; 32], vec![1, 2, 4, 4]).unwrap();
        let v = Tensor::new(vec![2.0; 32], vec![1, 2, 4, 4]).unwrap();
        cache.append(&k, &v).unwrap();

        assert_eq!(cache.seq_len(), 4);

        cache.reset();

        assert_eq!(cache.seq_len(), 0);
    }

    #[test]
    fn test_attention_shape_invariant() {
        let batch = 1;
        let seq_len = 4;
        let hidden_size = 64;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 16;

        // Create attention layer
        let q_proj = Linear::new(
            Tensor::new(
                vec![0.1; num_heads * head_dim * hidden_size],
                vec![num_heads * head_dim, hidden_size],
            )
            .unwrap(),
        ).unwrap();
        let k_proj = Linear::new(
            Tensor::new(
                vec![0.1; num_kv_heads * head_dim * hidden_size],
                vec![num_kv_heads * head_dim, hidden_size],
            )
            .unwrap(),
        ).unwrap();
        let v_proj = Linear::new(
            Tensor::new(
                vec![0.1; num_kv_heads * head_dim * hidden_size],
                vec![num_kv_heads * head_dim, hidden_size],
            )
            .unwrap(),
        ).unwrap();
        let o_proj = Linear::new(
            Tensor::new(
                vec![0.1; hidden_size * num_heads * head_dim],
                vec![hidden_size, num_heads * head_dim],
            )
            .unwrap(),
        ).unwrap();
        let q_norm = RmsNorm::new(Tensor::new(vec![1.0; head_dim], vec![head_dim]).unwrap(), 1e-6).unwrap();
        let k_norm = RmsNorm::new(Tensor::new(vec![1.0; head_dim], vec![head_dim]).unwrap(), 1e-6).unwrap();
        let rotary = Arc::new(RotaryEmbedding::new(head_dim, 100, 10000.0).unwrap());

        let attn = Attention::new(
            q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, rotary, num_heads, num_kv_heads,
            head_dim,
        );

        // Input tensor
        let x = Tensor::new(vec![0.5; batch * seq_len * hidden_size], vec![batch, seq_len, hidden_size]).unwrap();

        // Forward pass without mask
        let mut cache = KvCache::new();
        let output = attn.forward(&x, None, &mut cache, 0).unwrap();

        // Output shape should match input shape
        assert_eq!(output.shape(), &[batch, seq_len, hidden_size]);
    }

    #[test]
    fn test_attention_with_causal_mask() {
        let batch = 1;
        let seq_len = 4;
        let hidden_size = 64;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 16;

        // Create attention layer
        let q_proj = Linear::new(
            Tensor::new(
                vec![0.1; num_heads * head_dim * hidden_size],
                vec![num_heads * head_dim, hidden_size],
            )
            .unwrap(),
        ).unwrap();
        let k_proj = Linear::new(
            Tensor::new(
                vec![0.1; num_kv_heads * head_dim * hidden_size],
                vec![num_kv_heads * head_dim, hidden_size],
            )
            .unwrap(),
        ).unwrap();
        let v_proj = Linear::new(
            Tensor::new(
                vec![0.1; num_kv_heads * head_dim * hidden_size],
                vec![num_kv_heads * head_dim, hidden_size],
            )
            .unwrap(),
        ).unwrap();
        let o_proj = Linear::new(
            Tensor::new(
                vec![0.1; hidden_size * num_heads * head_dim],
                vec![hidden_size, num_heads * head_dim],
            )
            .unwrap(),
        ).unwrap();
        let q_norm = RmsNorm::new(Tensor::new(vec![1.0; head_dim], vec![head_dim]).unwrap(), 1e-6).unwrap();
        let k_norm = RmsNorm::new(Tensor::new(vec![1.0; head_dim], vec![head_dim]).unwrap(), 1e-6).unwrap();
        let rotary = Arc::new(RotaryEmbedding::new(head_dim, 100, 10000.0).unwrap());

        let attn = Attention::new(
            q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, rotary, num_heads, num_kv_heads,
            head_dim,
        );

        // Input tensor
        let x = Tensor::new(
            vec![0.5; batch * seq_len * hidden_size],
            vec![batch, seq_len, hidden_size],
        )
        .unwrap();

        // Create causal mask: [B, 1, L, L]
        // Upper triangle should be -inf (masked out)
        let mut mask_data = vec![0.0f32; batch * 1 * seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i {
                    // Mask future positions
                    mask_data[i * seq_len + j] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::new(mask_data, vec![batch, 1, seq_len, seq_len]).unwrap();

        // Forward pass with causal mask
        let mut cache = KvCache::new();
        let output = attn.forward(&x, Some(&mask), &mut cache, 0).unwrap();

        // Output shape should match input shape
        assert_eq!(output.shape(), &[batch, seq_len, hidden_size]);

        // Verify output is finite (no NaN or Inf despite mask)
        let output_data = output.to_vec_f32();
        assert!(
            output_data.iter().all(|&x| x.is_finite()),
            "Output contains non-finite values"
        );
    }

    #[test]
    fn test_attention_with_kv_cache_multi_step() {
        // Test autoregressive generation with KV cache
        let batch = 1;
        let hidden_size = 64;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 16;

        // Create attention layer
        let q_proj = Linear::new(
            Tensor::new(
                vec![0.1; num_heads * head_dim * hidden_size],
                vec![num_heads * head_dim, hidden_size],
            )
            .unwrap(),
        ).unwrap();
        let k_proj = Linear::new(
            Tensor::new(
                vec![0.1; num_kv_heads * head_dim * hidden_size],
                vec![num_kv_heads * head_dim, hidden_size],
            )
            .unwrap(),
        ).unwrap();
        let v_proj = Linear::new(
            Tensor::new(
                vec![0.1; num_kv_heads * head_dim * hidden_size],
                vec![num_kv_heads * head_dim, hidden_size],
            )
            .unwrap(),
        ).unwrap();
        let o_proj = Linear::new(
            Tensor::new(
                vec![0.1; hidden_size * num_heads * head_dim],
                vec![hidden_size, num_heads * head_dim],
            )
            .unwrap(),
        ).unwrap();
        let q_norm = RmsNorm::new(Tensor::new(vec![1.0; head_dim], vec![head_dim]).unwrap(), 1e-6).unwrap();
        let k_norm = RmsNorm::new(Tensor::new(vec![1.0; head_dim], vec![head_dim]).unwrap(), 1e-6).unwrap();
        let rotary = Arc::new(RotaryEmbedding::new(head_dim, 100, 10000.0).unwrap());

        let attn = Attention::new(
            q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, rotary, num_heads, num_kv_heads,
            head_dim,
        );

        let mut cache = KvCache::new();

        // Step 1: Process initial prompt (seq_len=4)
        let x1 = Tensor::new(
            vec![0.5; batch * 4 * hidden_size],
            vec![batch, 4, hidden_size],
        )
        .unwrap();
        let output1 = attn.forward(&x1, None, &mut cache, 0).unwrap();
        assert_eq!(output1.shape(), &[batch, 4, hidden_size]);
        assert_eq!(cache.seq_len(), 4);

        // Step 2: Generate next token (seq_len=1)
        let x2 = Tensor::new(
            vec![0.6; batch * 1 * hidden_size],
            vec![batch, 1, hidden_size],
        )
        .unwrap();
        let output2 = attn.forward(&x2, None, &mut cache, 4).unwrap();
        assert_eq!(output2.shape(), &[batch, 1, hidden_size]);
        assert_eq!(cache.seq_len(), 5);

        // Step 3: Generate another token (seq_len=1)
        let x3 = Tensor::new(
            vec![0.7; batch * 1 * hidden_size],
            vec![batch, 1, hidden_size],
        )
        .unwrap();
        let output3 = attn.forward(&x3, None, &mut cache, 5).unwrap();
        assert_eq!(output3.shape(), &[batch, 1, hidden_size]);
        assert_eq!(cache.seq_len(), 6);

        // Verify all outputs are finite
        for output in [output1, output2, output3] {
            let data = output.to_vec_f32();
            assert!(
                data.iter().all(|&x| x.is_finite()),
                "Output contains non-finite values"
            );
        }
    }

    #[test]
    #[should_panic(expected = "num_heads (5) must be divisible by num_kv_heads (2)")]
    fn test_attention_invalid_head_configuration() {
        // num_heads=5, num_kv_heads=2 should panic (not divisible)
        let hidden_size = 64;
        let num_heads = 5;
        let num_kv_heads = 2;
        let head_dim = 16;

        let q_proj = Linear::new(
            Tensor::new(
                vec![0.1; num_heads * head_dim * hidden_size],
                vec![num_heads * head_dim, hidden_size],
            )
            .unwrap(),
        ).unwrap();
        let k_proj = Linear::new(
            Tensor::new(
                vec![0.1; num_kv_heads * head_dim * hidden_size],
                vec![num_kv_heads * head_dim, hidden_size],
            )
            .unwrap(),
        ).unwrap();
        let v_proj = Linear::new(
            Tensor::new(
                vec![0.1; num_kv_heads * head_dim * hidden_size],
                vec![num_kv_heads * head_dim, hidden_size],
            )
            .unwrap(),
        ).unwrap();
        let o_proj = Linear::new(
            Tensor::new(
                vec![0.1; hidden_size * num_heads * head_dim],
                vec![hidden_size, num_heads * head_dim],
            )
            .unwrap(),
        ).unwrap();
        let q_norm = RmsNorm::new(Tensor::new(vec![1.0; head_dim], vec![head_dim]).unwrap(), 1e-6).unwrap();
        let k_norm = RmsNorm::new(Tensor::new(vec![1.0; head_dim], vec![head_dim]).unwrap(), 1e-6).unwrap();
        let rotary = Arc::new(RotaryEmbedding::new(head_dim, 100, 10000.0).unwrap());

        // This should panic
        Attention::new(
            q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, rotary, num_heads, num_kv_heads,
            head_dim,
        );
    }
}
