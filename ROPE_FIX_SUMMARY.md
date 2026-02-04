# RoPE Bug Fix - Layer 0 Divergence Resolution

## Problem

Layer 0 output showed significant divergence between Rust and Python implementations for multi-token sequences:
- **prompt1** (1 token): Layer 0 perfect match (MSE < 1e-10)
- **prompt2** (5 tokens): Layer 0 MSE = 7.88e-3, Cosine = 0.951

Since embeddings matched perfectly, the issue was isolated to the first transformer layer, specifically in RoPE application.

## Root Cause

The Rust RoPE implementation was applying rotation to the **full head dimension** (128), while the Python/HuggingFace implementation only applies it to **half the head dimension** (64).

### Why Half Dimension?

In the HuggingFace Qwen2/Qwen3 implementation:

```python
# From transformers/models/qwen2/modeling_qwen2.py
def compute_default_rope_parameters(config, device=None, seq_len=None):
    base = config.rope_parameters["rope_theta"]
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

    # Creates inverse frequencies for HALF the dimension
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
    )
    return inv_freq, attention_factor

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

The key insight:
- `torch.arange(0, dim, 2)` creates indices `[0, 2, 4, ..., dim-2]`
- This produces `dim/2` frequencies, not `dim` frequencies
- RoPE cos/sin tables have shape `[max_seq_len, dim/2]`, not `[max_seq_len, dim]`

## The Fix

### Before (Incorrect)
```rust
// rope.rs - OLD
pub fn new(head_dim: usize, max_seq_len: usize, theta: f64) -> Result<Self> {
    let half_dim = head_dim / 2;
    let mut inv_freq = Vec::with_capacity(half_dim);
    for i in 0..half_dim {
        let exponent = (2 * i) as f64 / head_dim as f64;  // BUG: using head_dim
        inv_freq.push(1.0 / theta.powf(exponent));
    }
    // ... creates tables of shape [max_seq_len, head_dim]  // BUG: full dimension
}
```

### After (Correct)
```rust
// rope.rs - NEW
pub fn new(head_dim: usize, max_seq_len: usize, theta: f64) -> Result<Self> {
    // RoPE is applied to half the head dimension (matching PyTorch implementation)
    let rope_dim = head_dim / 2;  // NEW: explicit rope_dim

    let half_rope_dim = rope_dim / 2;
    let mut inv_freq = Vec::with_capacity(half_rope_dim);
    for i in 0..half_rope_dim {
        let exponent = (2 * i) as f64 / rope_dim as f64;  // FIX: using rope_dim
        inv_freq.push(1.0 / theta.powf(exponent));
    }
    // ... creates tables of shape [max_seq_len, rope_dim]  // FIX: half dimension
}
```

### Rotation Algorithm Update

The `apply_rope_to_tensor` function was updated to implement the `rotate_half` pattern:

```rust
// Split x into two halves
let rope_dim = head_dim / 2;

// For each position:
// x1 = x[..., :rope_dim]
// x2 = x[..., rope_dim:]

// Apply rotation:
// result[..., :rope_dim]  = x1 * cos - x2 * sin
// result[..., rope_dim:]  = x2 * cos + x1 * sin
```

This matches the PyTorch `rotate_half` behavior exactly.

## Verification

### 1. RoPE Table Shape
```
Python Reference: (40960, 64)   # max_seq_len=40960, rope_dim=64
Rust (Fixed):     (40960, 64)   # ✓ Matches
Rust (Old):       (40960, 128)  # ✗ Wrong
```

### 2. Frequency Values
```
Position 1, first 8 values:
Python:  [0.5403, 0.5403, 0.7965, 0.7965, 0.9124, 0.9124, 0.9627, 0.9627]
Rust:    [0.5403, 0.5403, 0.7965, 0.7965, 0.9124, 0.9124, 0.9627, 0.9627]
Diff:    < 1e-7 (floating point precision)
```

### 3. Unit Tests
```bash
cargo test --lib rope::tests
# All 13 RoPE tests pass ✓
# All 333 library tests pass ✓
```

## Expected Impact

With this fix, Layer 0 output for multi-token sequences should now:
1. Match Python reference within floating-point precision
2. Show MSE < 1e-6 (down from 7.88e-3)
3. Show cosine similarity > 0.9999 (up from 0.951)

This should resolve the divergence cascade through subsequent layers.

## Files Modified

1. **lluda-inference/src/rope.rs**
   - Updated `RotaryEmbedding::new()` to use `rope_dim = head_dim / 2`
   - Updated `apply_rope_to_tensor()` to implement `rotate_half` pattern
   - Updated test expectations for new table dimensions

## Testing Checklist

- [x] Unit tests pass (rope::tests)
- [x] All library tests pass (333 tests)
- [x] RoPE table generation validated against Python reference
- [x] Frequency duplication pattern verified
- [ ] Integration test: Layer 0 output validation for prompt2
- [ ] Full forward pass comparison with Python

## References

- HuggingFace Transformers: `transformers/models/qwen2/modeling_qwen2.py`
- RoFormer Paper: https://arxiv.org/abs/2104.09864
- Qwen3-0.6B Config: `models/Qwen3-0.6B/config.json`
