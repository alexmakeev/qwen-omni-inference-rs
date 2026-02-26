//! Grouped Query Attention for Qwen2.5-Omni (thinker and talker).
//!
//! Differs from Qwen3 Attention in two ways:
//! - Bias on q, k, v projections (no bias on o_proj)
//! - No per-head RMSNorm on queries and keys
//!
//! Used by both the thinker (text part of Qwen2.5-Omni) and the talker (TTS decoder).
//!
//! # Architecture
//!
//! For Qwen2.5-Omni-3B talker:
//! - Query heads: 14 (num_attention_heads)
//! - KV heads: 2 (num_key_value_heads)
//! - Head dimension: 64
//! - Each KV head is shared by 7 query heads (num_kv_groups = 7)
//!
//! # References
//!
//! - Grouped Query Attention: https://arxiv.org/abs/2305.13245
//! - RoFormer (RoPE): https://arxiv.org/abs/2104.09864

use std::sync::Arc;

use crate::attention::KvCache;
use crate::error::{LludaError, Result};
use crate::mlp::MLP;
use crate::rms_norm::RmsNorm;
use crate::rope::RotaryEmbedding;
use crate::tensor::Tensor;

/// Linear projection with bias.
///
/// Computes `x @ weight.T + bias`.
/// Used for q, k, v projections in Qwen2.5-Omni attention.
#[derive(Debug, Clone)]
pub struct LinearWithBias {
    /// Weight matrix [out_features, in_features]
    pub(crate) weight: Tensor,
    /// Bias vector [out_features]
    pub(crate) bias: Tensor,
}

impl LinearWithBias {
    /// Create a new linear layer with bias.
    ///
    /// # Arguments
    ///
    /// * `weight` - Weight tensor of shape [out_features, in_features]
    /// * `bias` - Bias tensor of shape [out_features]
    ///
    /// # Errors
    ///
    /// Returns error if weight is not 2D or bias is not 1D, or shapes are inconsistent.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::omni_attention::LinearWithBias;
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let weight = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let bias = Tensor::new(vec![0.1, 0.2], vec![2]).unwrap();
    /// let linear = LinearWithBias::new(weight, bias).unwrap();
    /// ```
    pub fn new(weight: Tensor, bias: Tensor) -> Result<Self> {
        if weight.ndim() != 2 {
            return Err(LludaError::ShapeMismatch {
                expected: vec![0, 0],
                got: weight.shape().to_vec(),
            });
        }
        if bias.ndim() != 1 {
            return Err(LludaError::ShapeMismatch {
                expected: vec![0],
                got: bias.shape().to_vec(),
            });
        }
        let out_features = weight.shape()[0];
        if bias.shape()[0] != out_features {
            return Err(LludaError::ShapeMismatch {
                expected: vec![out_features],
                got: bias.shape().to_vec(),
            });
        }
        Ok(LinearWithBias { weight, bias })
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
    /// Returns `ShapeMismatch` if the last dimension of `x` doesn't match weight's in_features.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::omni_attention::LinearWithBias;
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let weight = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
    /// let bias = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
    /// let linear = LinearWithBias::new(weight, bias).unwrap();
    ///
    /// let x = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
    /// let y = linear.forward(&x).unwrap();
    /// // y = x @ I + bias = [1, 2] + [1, 2] = [2, 4]
    /// assert_eq!(y.shape(), &[1, 2]);
    /// ```
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let input_shape = x.shape();
        let weight_shape = self.weight.shape();

        let out_features = weight_shape[0];
        let in_features = weight_shape[1];

        if input_shape[input_shape.len() - 1] != in_features {
            return Err(LludaError::ShapeMismatch {
                expected: vec![in_features],
                got: vec![input_shape[input_shape.len() - 1]],
            });
        }

        // Flatten all leading dimensions into batch
        let batch_size: usize = input_shape[..input_shape.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch_size, in_features])?;

        // Transpose weight: [out_features, in_features] -> [in_features, out_features]
        let weight_t = self.weight.transpose()?;

        // Compute: [batch, in_features] @ [in_features, out_features] -> [batch, out_features]
        let output_2d = x_2d.matmul(&weight_t)?;

        // Add bias: broadcast [out_features] over batch dimension
        // bias shape: [out_features], output_2d shape: [batch, out_features]
        let bias_2d = self.bias.reshape(&[1, out_features])?;
        let output_with_bias = output_2d.add(&bias_2d)?;

        // Reshape back to original leading dims + out_features
        let mut output_shape = input_shape[..input_shape.len() - 1].to_vec();
        output_shape.push(out_features);

        output_with_bias.reshape(&output_shape)
    }
}

/// Linear projection without bias.
///
/// Computes `x @ weight.T`.
/// Used for o_proj in Qwen2.5-Omni attention (no bias on output projection).
#[derive(Debug, Clone)]
pub struct LinearNoBias {
    /// Weight matrix [out_features, in_features]
    pub(crate) weight: Tensor,
}

impl LinearNoBias {
    /// Create a new linear layer without bias.
    ///
    /// # Arguments
    ///
    /// * `weight` - Weight tensor of shape [out_features, in_features]
    ///
    /// # Errors
    ///
    /// Returns error if weight is not a 2D tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::omni_attention::LinearNoBias;
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let weight = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
    /// let linear = LinearNoBias::new(weight).unwrap();
    /// ```
    pub fn new(weight: Tensor) -> Result<Self> {
        if weight.ndim() != 2 {
            return Err(LludaError::ShapeMismatch {
                expected: vec![0, 0],
                got: weight.shape().to_vec(),
            });
        }
        Ok(LinearNoBias { weight })
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
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let input_shape = x.shape();
        let weight_shape = self.weight.shape();

        let out_features = weight_shape[0];
        let in_features = weight_shape[1];

        if input_shape[input_shape.len() - 1] != in_features {
            return Err(LludaError::ShapeMismatch {
                expected: vec![in_features],
                got: vec![input_shape[input_shape.len() - 1]],
            });
        }

        // Flatten all leading dimensions into batch
        let batch_size: usize = input_shape[..input_shape.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch_size, in_features])?;

        // Transpose weight: [out_features, in_features] -> [in_features, out_features]
        let weight_t = self.weight.transpose()?;

        // Compute: [batch, in_features] @ [in_features, out_features] -> [batch, out_features]
        let output_2d = x_2d.matmul(&weight_t)?;

        // Reshape back to original leading dims + out_features
        let mut output_shape = input_shape[..input_shape.len() - 1].to_vec();
        output_shape.push(out_features);

        output_2d.reshape(&output_shape)
    }
}

/// GQA attention for Qwen2.5-Omni (thinker and talker).
///
/// Differences from Qwen3 Attention:
/// - Bias on q, k, v projections (no bias on o_proj)
/// - No per-head RMSNorm on queries and keys
#[derive(Debug, Clone)]
pub struct OmniAttention {
    /// Query projection [num_heads * head_dim, hidden_size] + bias
    q_proj: LinearWithBias,
    /// Key projection [num_kv_heads * head_dim, hidden_size] + bias
    k_proj: LinearWithBias,
    /// Value projection [num_kv_heads * head_dim, hidden_size] + bias
    v_proj: LinearWithBias,
    /// Output projection [hidden_size, num_heads * head_dim] (no bias)
    o_proj: LinearNoBias,
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

impl OmniAttention {
    /// Create a new OmniAttention layer.
    ///
    /// # Arguments
    ///
    /// * `q_proj` - Query projection with bias
    /// * `k_proj` - Key projection with bias
    /// * `v_proj` - Value projection with bias
    /// * `o_proj` - Output projection without bias
    /// * `rotary` - Shared rotary position embeddings
    /// * `num_heads` - Number of query attention heads
    /// * `num_kv_heads` - Number of key/value heads
    /// * `head_dim` - Dimension of each attention head
    ///
    /// # Errors
    ///
    /// Returns error if num_heads is not divisible by num_kv_heads or weight shapes mismatch.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::sync::Arc;
    /// use lluda_inference::omni_attention::{OmniAttention, LinearWithBias, LinearNoBias};
    /// use lluda_inference::rope::RotaryEmbedding;
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let hidden_size = 32;
    /// let num_heads = 4;
    /// let num_kv_heads = 2;
    /// let head_dim = 8;
    ///
    /// let q_w = Tensor::new(vec![0.0; num_heads * head_dim * hidden_size], vec![num_heads * head_dim, hidden_size]).unwrap();
    /// let q_b = Tensor::new(vec![0.0; num_heads * head_dim], vec![num_heads * head_dim]).unwrap();
    /// let q_proj = LinearWithBias::new(q_w, q_b).unwrap();
    ///
    /// let k_w = Tensor::new(vec![0.0; num_kv_heads * head_dim * hidden_size], vec![num_kv_heads * head_dim, hidden_size]).unwrap();
    /// let k_b = Tensor::new(vec![0.0; num_kv_heads * head_dim], vec![num_kv_heads * head_dim]).unwrap();
    /// let k_proj = LinearWithBias::new(k_w, k_b).unwrap();
    ///
    /// let v_w = Tensor::new(vec![0.0; num_kv_heads * head_dim * hidden_size], vec![num_kv_heads * head_dim, hidden_size]).unwrap();
    /// let v_b = Tensor::new(vec![0.0; num_kv_heads * head_dim], vec![num_kv_heads * head_dim]).unwrap();
    /// let v_proj = LinearWithBias::new(v_w, v_b).unwrap();
    ///
    /// let o_w = Tensor::new(vec![0.0; hidden_size * num_heads * head_dim], vec![hidden_size, num_heads * head_dim]).unwrap();
    /// let o_proj = LinearNoBias::new(o_w).unwrap();
    ///
    /// let rotary = Arc::new(RotaryEmbedding::new(head_dim, 100, 10000.0).unwrap());
    ///
    /// let attn = OmniAttention::new(q_proj, k_proj, v_proj, o_proj, rotary, num_heads, num_kv_heads, head_dim).unwrap();
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        q_proj: LinearWithBias,
        k_proj: LinearWithBias,
        v_proj: LinearWithBias,
        o_proj: LinearNoBias,
        rotary: Arc<RotaryEmbedding>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        // Validate GQA: num_heads must be divisible by num_kv_heads
        #[allow(clippy::manual_is_multiple_of)]
        if num_heads % num_kv_heads != 0 {
            return Err(LludaError::Model(format!(
                "num_heads ({}) must be divisible by num_kv_heads ({})",
                num_heads, num_kv_heads
            )));
        }

        // Validate projection weight shapes
        let q_shape = q_proj.weight.shape();
        let k_shape = k_proj.weight.shape();
        let v_shape = v_proj.weight.shape();
        let o_shape = o_proj.weight.shape();

        if q_shape[0] != num_heads * head_dim {
            return Err(LludaError::ShapeMismatch {
                expected: vec![num_heads * head_dim, q_shape[1]],
                got: q_shape.to_vec(),
            });
        }
        if k_shape[0] != num_kv_heads * head_dim {
            return Err(LludaError::ShapeMismatch {
                expected: vec![num_kv_heads * head_dim, k_shape[1]],
                got: k_shape.to_vec(),
            });
        }
        if v_shape[0] != num_kv_heads * head_dim {
            return Err(LludaError::ShapeMismatch {
                expected: vec![num_kv_heads * head_dim, v_shape[1]],
                got: v_shape.to_vec(),
            });
        }
        if o_shape[1] != num_heads * head_dim {
            return Err(LludaError::ShapeMismatch {
                expected: vec![o_shape[0], num_heads * head_dim],
                got: o_shape.to_vec(),
            });
        }

        let num_kv_groups = num_heads / num_kv_heads;

        Ok(OmniAttention {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
        })
    }

    /// Forward pass through OmniAttention.
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
    /// 1. Project to Q, K, V (with bias)
    /// 2. Reshape to multi-head format
    /// 3. Apply RoPE to Q and K (no per-head norms)
    /// 4. Update KV cache
    /// 5. Repeat KV heads for GQA
    /// 6. Compute attention scores
    /// 7. Apply mask and softmax
    /// 8. Compute attention output
    /// 9. Reshape and project output (no bias)
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

        // 1. Project to Q, K, V (with bias)
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

        // 3. Apply RoPE (no per-head norms — the key difference from Qwen3 Attention)
        let (q, k) = self.rotary.apply(&q, &k, offset)?;

        // 4. Update KV cache and get full K, V
        let (k, v) = kv_cache.append(&k, &v)?;

        // 5. Repeat KV heads for GQA
        // [B, num_kv_heads, L, D] -> [B, num_heads, L, D]
        let k = repeat_kv(&k, self.num_kv_groups)?;
        let v = repeat_kv(&v, self.num_kv_groups)?;

        // 6. Compute attention scores: Q @ K.T / sqrt(head_dim)
        // Q: [B, num_heads, L_q, D]
        // K: [B, num_heads, L_kv, D] -> transpose -> [B, num_heads, D, L_kv]
        let k_t = k.transpose_dims(2, 3)?; // [B, H, L_kv, D] -> [B, H, D, L_kv]
        let scores = q.matmul(&k_t)?; // [B, H, L_q, L_kv]

        // Scale by sqrt(head_dim)
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let scores = scores.mul_scalar(scale)?;

        // 7. Apply mask (if provided) and softmax
        let scores = if let Some(mask) = mask {
            scores.add(mask)?
        } else {
            scores
        };

        let attn_weights = scores.softmax(3)?; // Softmax over last dim (L_kv)

        // 8. Compute attention output: attn_weights @ V
        // attn_weights: [B, H, L_q, L_kv]
        // V: [B, H, L_kv, D]
        // output: [B, H, L_q, D]
        let attn_output = attn_weights.matmul(&v)?;

        // 9. Reshape and project output (no bias)
        // [B, H, L, D] -> [B, L, H, D] -> [B, L, H*D]
        let attn_output = attn_output.transpose_dims(1, 2)?; // [B, L, H, D]
        let attn_output =
            attn_output.reshape(&[batch, seq_len, self.num_heads * self.head_dim])?;

        // Final output projection (no bias)
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
fn repeat_kv(x: &Tensor, num_groups: usize) -> Result<Tensor> {
    if num_groups == 1 {
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
                        let idx =
                            ((b * num_kv_heads + kv_head) * seq_len + pos) * head_dim + d;
                        result.push(data[idx]);
                    }
                }
            }
        }
    }

    Tensor::new(result, vec![batch, num_kv_heads * num_groups, seq_len, head_dim])
}

/// Transformer decoder layer for Qwen2.5-Omni (thinker and talker).
///
/// Identical in structure to the Qwen3 DecoderLayer, but uses OmniAttention
/// instead of Qwen3 Attention (no per-head norms, bias on q/k/v).
///
/// # Algorithm (Pre-norm)
///
/// ```text
/// h = x + attention(layer_norm(x))
/// out = h + mlp(layer_norm(h))
/// ```
#[derive(Debug, Clone)]
pub struct OmniDecoderLayer {
    /// Self-attention layer (OmniAttention, no per-head norms, bias on q/k/v)
    self_attn: OmniAttention,
    /// Feed-forward MLP layer
    mlp: MLP,
    /// RMSNorm before self-attention
    input_layernorm: RmsNorm,
    /// RMSNorm before MLP
    post_attention_layernorm: RmsNorm,
}

impl OmniDecoderLayer {
    /// Create a new OmniDecoderLayer.
    ///
    /// # Arguments
    ///
    /// * `self_attn` - OmniAttention layer
    /// * `mlp` - Feed-forward MLP layer
    /// * `input_layernorm` - RMSNorm applied before attention
    /// * `post_attention_layernorm` - RMSNorm applied before MLP
    ///
    /// # Returns
    ///
    /// New OmniDecoderLayer instance.
    pub fn new(
        self_attn: OmniAttention,
        mlp: MLP,
        input_layernorm: RmsNorm,
        post_attention_layernorm: RmsNorm,
    ) -> Self {
        OmniDecoderLayer {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        }
    }

    /// Forward pass through the OmniDecoderLayer.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape [B, L, hidden_size]
    /// * `mask` - Optional attention mask of shape [B, 1, L, L+offset]
    /// * `kv_cache` - Mutable KV cache (will be updated)
    /// * `offset` - Position offset (length of cached sequence)
    ///
    /// # Returns
    ///
    /// Output tensor of shape [B, L, hidden_size] (same as input).
    ///
    /// # Errors
    ///
    /// Returns error if shapes are incompatible or operations fail.
    ///
    /// # Algorithm (Pre-norm)
    ///
    /// 1. Apply RMSNorm to input: `x_norm = input_layernorm(x)`
    /// 2. Compute attention: `attn_out = self_attn(x_norm, mask, kv_cache, offset)`
    /// 3. First residual connection: `h = x + attn_out`
    /// 4. Apply RMSNorm: `h_norm = post_attention_layernorm(h)`
    /// 5. Compute MLP: `mlp_out = mlp(h_norm)`
    /// 6. Second residual connection: `out = h + mlp_out`
    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        kv_cache: &mut KvCache,
        offset: usize,
    ) -> Result<Tensor> {
        // 1. Pre-norm: apply RMSNorm before attention
        let x_norm = self.input_layernorm.forward(x)?;

        // 2. Self-attention
        let attn_out = self.self_attn.forward(&x_norm, mask, kv_cache, offset)?;

        // 3. First residual connection
        let h = x.add(&attn_out)?;

        // 4. Pre-norm: apply RMSNorm before MLP
        let h_norm = self.post_attention_layernorm.forward(&h)?;

        // 5. MLP
        let mlp_out = self.mlp.forward(&h_norm)?;

        // 6. Second residual connection
        h.add(&mlp_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rms_norm::RmsNorm;
    use std::sync::Arc;

    /// Helper to create a small OmniAttention for testing.
    fn make_omni_attention(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> OmniAttention {
        let q_w = Tensor::new(
            vec![0.01; num_heads * head_dim * hidden_size],
            vec![num_heads * head_dim, hidden_size],
        )
        .unwrap();
        let q_b = Tensor::new(vec![0.0; num_heads * head_dim], vec![num_heads * head_dim])
            .unwrap();
        let q_proj = LinearWithBias::new(q_w, q_b).unwrap();

        let k_w = Tensor::new(
            vec![0.01; num_kv_heads * head_dim * hidden_size],
            vec![num_kv_heads * head_dim, hidden_size],
        )
        .unwrap();
        let k_b =
            Tensor::new(vec![0.0; num_kv_heads * head_dim], vec![num_kv_heads * head_dim])
                .unwrap();
        let k_proj = LinearWithBias::new(k_w, k_b).unwrap();

        let v_w = Tensor::new(
            vec![0.01; num_kv_heads * head_dim * hidden_size],
            vec![num_kv_heads * head_dim, hidden_size],
        )
        .unwrap();
        let v_b =
            Tensor::new(vec![0.0; num_kv_heads * head_dim], vec![num_kv_heads * head_dim])
                .unwrap();
        let v_proj = LinearWithBias::new(v_w, v_b).unwrap();

        let o_w = Tensor::new(
            vec![0.01; hidden_size * num_heads * head_dim],
            vec![hidden_size, num_heads * head_dim],
        )
        .unwrap();
        let o_proj = LinearNoBias::new(o_w).unwrap();

        let rotary =
            Arc::new(RotaryEmbedding::new(head_dim, 100, 1000000.0).unwrap());

        OmniAttention::new(
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary,
            num_heads,
            num_kv_heads,
            head_dim,
        )
        .unwrap()
    }

    #[test]
    fn test_linear_with_bias() {
        // Identity weight + constant bias
        let weight = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let bias = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
        let linear = LinearWithBias::new(weight, bias).unwrap();

        let x = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let y = linear.forward(&x).unwrap();

        assert_eq!(y.shape(), &[1, 2]);
        let data = y.to_vec_f32();
        // y[0] = 3 * 1 + 4 * 0 + 1 = 4
        // y[1] = 3 * 0 + 4 * 1 + 2 = 6
        assert!((data[0] - 4.0).abs() < 1e-5, "expected 4.0, got {}", data[0]);
        assert!((data[1] - 6.0).abs() < 1e-5, "expected 6.0, got {}", data[1]);
    }

    #[test]
    fn test_linear_with_bias_batched() {
        // Verify batch dimension is handled correctly
        let weight = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let bias = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
        let linear = LinearWithBias::new(weight, bias).unwrap();

        // [3, 2] batch
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
        let y = linear.forward(&x).unwrap();

        assert_eq!(y.shape(), &[3, 2]);
        let data = y.to_vec_f32();
        // Row 0: [1+1, 2+2] = [2, 4]
        assert!((data[0] - 2.0).abs() < 1e-5);
        assert!((data[1] - 4.0).abs() < 1e-5);
        // Row 1: [3+1, 4+2] = [4, 6]
        assert!((data[2] - 4.0).abs() < 1e-5);
        assert!((data[3] - 6.0).abs() < 1e-5);
        // Row 2: [5+1, 6+2] = [6, 8]
        assert!((data[4] - 6.0).abs() < 1e-5);
        assert!((data[5] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_omni_attention_shape() {
        let hidden_size = 32;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;

        let attn = make_omni_attention(hidden_size, num_heads, num_kv_heads, head_dim);

        let batch = 1;
        let seq_len = 4;

        let x =
            Tensor::new(vec![0.1f32; batch * seq_len * hidden_size], vec![batch, seq_len, hidden_size])
                .unwrap();
        let mut cache = KvCache::new();

        let output = attn.forward(&x, None, &mut cache, 0).unwrap();

        // Output shape should match input shape
        assert_eq!(output.shape(), &[batch, seq_len, hidden_size]);

        // Output should be finite
        let data = output.to_vec_f32();
        assert!(data.iter().all(|&v| v.is_finite()), "Output contains non-finite values");
    }

    #[test]
    fn test_omni_attention_with_kv_cache() {
        let hidden_size = 32;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;

        let attn = make_omni_attention(hidden_size, num_heads, num_kv_heads, head_dim);
        let mut cache = KvCache::new();

        // Step 1: prefill
        let x1 = Tensor::new(vec![0.1f32; 4 * hidden_size], vec![1, 4, hidden_size]).unwrap();
        let out1 = attn.forward(&x1, None, &mut cache, 0).unwrap();
        assert_eq!(out1.shape(), &[1, 4, hidden_size]);
        assert_eq!(cache.seq_len(), 4);

        // Step 2: single token decode
        let x2 = Tensor::new(vec![0.2f32; hidden_size], vec![1, 1, hidden_size]).unwrap();
        let out2 = attn.forward(&x2, None, &mut cache, 4).unwrap();
        assert_eq!(out2.shape(), &[1, 1, hidden_size]);
        assert_eq!(cache.seq_len(), 5);
    }

    #[test]
    fn test_omni_decoder_layer_shape() {
        let hidden_size = 32;
        let intermediate_size = 64;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;

        let attn = make_omni_attention(hidden_size, num_heads, num_kv_heads, head_dim);

        let gate_proj =
            Tensor::new(vec![0.01f32; intermediate_size * hidden_size], vec![intermediate_size, hidden_size])
                .unwrap();
        let up_proj =
            Tensor::new(vec![0.01f32; intermediate_size * hidden_size], vec![intermediate_size, hidden_size])
                .unwrap();
        let down_proj =
            Tensor::new(vec![0.01f32; hidden_size * intermediate_size], vec![hidden_size, intermediate_size])
                .unwrap();
        let mlp = MLP::new(gate_proj, up_proj, down_proj).unwrap();

        let input_layernorm =
            RmsNorm::new(Tensor::new(vec![1.0f32; hidden_size], vec![hidden_size]).unwrap(), 1e-6)
                .unwrap();
        let post_attention_layernorm =
            RmsNorm::new(Tensor::new(vec![1.0f32; hidden_size], vec![hidden_size]).unwrap(), 1e-6)
                .unwrap();

        let layer = OmniDecoderLayer::new(attn, mlp, input_layernorm, post_attention_layernorm);

        let batch = 1;
        let seq_len = 5;
        let x = Tensor::new(vec![0.1f32; batch * seq_len * hidden_size], vec![batch, seq_len, hidden_size])
            .unwrap();
        let mut cache = KvCache::new();

        let output = layer.forward(&x, None, &mut cache, 0).unwrap();

        assert_eq!(output.shape(), &[batch, seq_len, hidden_size]);

        let data = output.to_vec_f32();
        assert!(data.iter().all(|&v| v.is_finite()), "Layer output contains non-finite values");
    }

    #[test]
    fn test_omni_decoder_layer_residual() {
        // Residual connections should preserve signal; output should be non-trivially different from input
        let hidden_size = 32;
        let intermediate_size = 64;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;

        let attn = make_omni_attention(hidden_size, num_heads, num_kv_heads, head_dim);

        let gate_proj =
            Tensor::new(vec![0.01f32; intermediate_size * hidden_size], vec![intermediate_size, hidden_size])
                .unwrap();
        let up_proj =
            Tensor::new(vec![0.01f32; intermediate_size * hidden_size], vec![intermediate_size, hidden_size])
                .unwrap();
        let down_proj =
            Tensor::new(vec![0.01f32; hidden_size * intermediate_size], vec![hidden_size, intermediate_size])
                .unwrap();
        let mlp = MLP::new(gate_proj, up_proj, down_proj).unwrap();

        let input_layernorm =
            RmsNorm::new(Tensor::new(vec![1.0f32; hidden_size], vec![hidden_size]).unwrap(), 1e-6)
                .unwrap();
        let post_attention_layernorm =
            RmsNorm::new(Tensor::new(vec![1.0f32; hidden_size], vec![hidden_size]).unwrap(), 1e-6)
                .unwrap();

        let layer = OmniDecoderLayer::new(attn, mlp, input_layernorm, post_attention_layernorm);

        let x = Tensor::new(vec![1.0f32; 1 * 3 * hidden_size], vec![1, 3, hidden_size]).unwrap();
        let mut cache = KvCache::new();
        let output = layer.forward(&x, None, &mut cache, 0).unwrap();

        let output_data = output.to_vec_f32();
        // Output should be finite and non-zero
        assert!(output_data.iter().all(|&v| v.is_finite()));
        // Mean should reflect residual signal
        let mean: f32 = output_data.iter().sum::<f32>() / output_data.len() as f32;
        assert!(mean.abs() > 0.01, "Residual connections should preserve signal, mean={}", mean);
    }
}
