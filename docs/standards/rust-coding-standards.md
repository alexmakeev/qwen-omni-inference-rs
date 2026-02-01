# Rust Coding Standards — lluda-inference

**Date:** 2026-01-31
**Scope:** All code in `lluda-inference/` crate

---

## 1. Error Handling

### Result Pattern (No panics in library code)

```rust
// GOOD: Return Result
pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
    if x.shape().ndim() < 2 {
        return Err(LludaError::ShapeMismatch {
            expected: vec![0, 0], // at least 2D
            got: x.shape().dims().to_vec(),
        });
    }
    // ...
}

// BAD: Panic
pub fn forward(&self, x: &Tensor) -> Tensor {
    assert!(x.shape().ndim() >= 2);  // NO! Returns nothing useful on error
    // ...
}
```

### Rules:
- Library code (`src/`): NEVER panic. All errors via `Result<T, LludaError>`.
- Examples (`examples/`): May use `.unwrap()` or `anyhow::Result`.
- Tests (`tests/`): May use `.unwrap()` for brevity.
- Use `?` operator for error propagation. Never `.unwrap()` in library code.
- Add context to errors: prefer specific error variants over generic `Msg`.

### Error creation:
```rust
// GOOD: Specific variant
return Err(LludaError::ShapeMismatch { expected, got });

// OK: When no specific variant fits
return Err(LludaError::Msg(format!("RoPE offset {} exceeds max_seq_len {}", offset, max)));

// BAD: Losing context
.map_err(|_| LludaError::Msg("something failed".into()))?;

// GOOD: Preserving context
.map_err(|e| LludaError::Msg(format!("SafeTensors parse: {e}")))?;
```

---

## 2. Naming Conventions

### Modules and Files
- Snake_case for files: `rms_norm.rs`, `kv_cache.rs`, `causal_mask.rs`
- One primary type per file (file name = type name in snake_case)

### Types
- PascalCase: `Tensor`, `RmsNorm`, `RotaryEmbedding`, `KvCache`
- Abbreviations: treat as words: `Gqa` not `GQA`, `Mlp` not `MLP`
  - Exception: well-known abbreviations in type position: `MLP`, `GQA` are acceptable

### Functions and Methods
- snake_case: `forward()`, `matmul()`, `softmax_last_dim()`
- Getters: no `get_` prefix: `shape()` not `get_shape()`, `dtype()` not `get_dtype()`
- Constructors: `new()` for primary, `from_file()`, `load()` for I/O-based

### Constants
- SCREAMING_SNAKE_CASE: `NEG_INFINITY`, `DEFAULT_EPS`

### Mathematical operations
- Follow the common ML naming: `matmul`, `softmax`, `silu`, `rope`
- Match PyTorch names where possible for familiarity

---

## 3. Module Structure

### File organization:
```rust
// 1. Imports (grouped: std, external crates, internal modules)
use std::path::Path;

use half::bf16;
use memmap2::Mmap;

use crate::error::{LludaError, Result};
use crate::tensor::Tensor;

// 2. Type definitions (structs, enums)
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

// 3. Implementation blocks
impl RmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self { ... }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> { ... }
}

// 4. Private helper functions (at bottom)
fn compute_rms(x: &[f32]) -> f32 { ... }
```

### Public API:
- Minimize public surface. Only expose what examples/tests need.
- Use `pub(crate)` for internal-but-cross-module items.
- `lib.rs` re-exports the public API with explicit `pub use`.

### Module visibility:
```rust
// lib.rs
pub mod error;
pub mod tensor;
pub mod model;
pub mod tokenizer;
pub mod generate;
pub mod sampling;

// Internal modules NOT re-exported
mod weights;  // Used only by model loading
```

---

## 4. Tensor Operations Style

### Compute always in F32:
```rust
// GOOD: Convert to F32, compute, store result as F32
let data = self.to_f32_vec()?;
let result: Vec<f32> = data.iter().map(|&x| x * x).collect();
Tensor::from_f32_vec(result, self.shape().dims())

// BAD: Try to compute in BF16
// (BF16 has no hardware compute on x86, and manual BF16 math is error-prone)
```

### DType Preservation Rule

**Rule:** Tensor operations that change shape/layout must preserve original dtype.

**Operations affected:** reshape, squeeze, unsqueeze, flatten, permute, transpose

**Rationale:**
- BF16 is for storage (memory efficiency)
- Silently converting BF16 → F32 doubles memory usage
- Model weights should stay BF16 until compute time

**Implementation pattern:**
```rust
pub fn reshape(&self, new_shape: &[usize]) -> Result<Tensor> {
    // Validate shape...

    // Preserve dtype (CORRECT)
    match &self.data {
        TensorData::F32(data) => Tensor::new(data.clone(), new_shape.to_vec()),
        TensorData::BF16(data) => Tensor::from_bf16(data.clone(), new_shape.to_vec()),
    }
}

// WRONG: Always converts to F32
// let data = self.to_vec_f32();
// Tensor::new(data, new_shape.to_vec())
```

**Exception:** Compute operations (matmul, add, mul, etc.) always return F32.

### Broadcasting Rules

**Rule:** Binary ops (add, mul, sub, div) must support NumPy-style broadcasting.

**Not sufficient:** Scalar-only broadcast (`[3,4] + [1]`)

**Required:** Full broadcasting:
- Trailing dimension: `[B, L, D] + [D]` → broadcast [D] to match
- Middle dimension: `[B, 1, L] + [B, H, L]` → broadcast dim 1
- Multi-dim: `[1, 3, 4] + [2, 1, 4]` → broadcast dims 0 and 1

**Algorithm:**
```rust
// 1. Align shapes from right
// 2. Each pair of dims must: equal OR one is 1 OR one is missing (treated as 1)
// 3. Output shape = max of each aligned dim pair

fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let max_ndim = a.len().max(b.len());
    let mut result = vec![1; max_ndim];

    for i in 0..max_ndim {
        let a_dim = a.get(a.len().saturating_sub(max_ndim - i)).copied().unwrap_or(1);
        let b_dim = b.get(b.len().saturating_sub(max_ndim - i)).copied().unwrap_or(1);

        result[i] = if a_dim == b_dim {
            a_dim
        } else if a_dim == 1 {
            b_dim
        } else if b_dim == 1 {
            a_dim
        } else {
            return Err(ShapeMismatch { ... });
        };
    }

    Ok(result)
}
```

**Why critical:** Many model operations need broadcasting:
- RMSNorm: `x * weight` where x=[B,L,D], weight=[D]
- Attention mask: `scores + mask` where mask has 1-dims

### Multi-dimensional Operation Support

**Rule:** Operations must support dimensions beyond 2D when spec requires it.

**Examples:**

**matmul:** Must support 2D, 3D, AND 4D batched:
- 2D: `[M, K] @ [K, N]` → `[M, N]`
- 3D: `[B, M, K] @ [B, K, N]` → `[B, M, N]`
- 4D: `[B, H, M, K] @ [B, H, K, N]` → `[B, H, M, N]` (for multi-head attention)

**transpose:** Must support arbitrary dimension pairs:
- `transpose(0, 1)` — swap first two dims
- `transpose(1, 2)` — swap middle dims (common in attention)
- `transpose(2, 3)` — swap last two dims (for key transposition)

**Not acceptable:** "Only 2D supported" when model needs 3D/4D.

**Implementation tip:** Start with general N-D implementation, optimize 2D special case if needed.

### Config Parsing Rule

**Rule:** Always parse fields directly from config.json, never compute derived values.

**Rationale:**
- Computed values may have wrong semantic meaning for other models
- Breaks abstraction: config should be source of truth
- Hidden assumptions about relationships between fields

**Example violation:**
```rust
// BAD: Computing head_dim from other fields
impl ModelConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_key_value_heads  // WRONG!
    }
}
```

**Why wrong:**
- For Qwen3-0.6B: `hidden_size=1024`, `num_key_value_heads=2`, giving `head_dim=512`
- But actual `head_dim=128` (from `num_attention_heads=8`)
- Computed value is semantically incorrect (KV heads vs Q heads)
- Works accidentally for some models, breaks for others

**Correct pattern:**
```rust
// GOOD: Parse explicit field from config
#[derive(Deserialize)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,  // Parse from JSON
}

impl ModelConfig {
    // Getter returns parsed value
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}
```

**Guidelines:**
1. Add explicit field to config struct
2. Parse from JSON (add to `config.json` if missing)
3. Provide getter if needed for API consistency
4. Never derive one config value from others

### Specification Completeness for Operations

**Rule:** Before implementing task, verify ALL operations from spec are included.

**Process:**
1. Read task spec in `docs/architecture/phase0-implementation-plan.md`
2. Create checklist of ALL operations mentioned
3. Implement each operation with tests
4. Cross-check against spec before marking complete

**Why critical:** Missing operations block downstream tasks.

**Example failure pattern (T05 element-wise operations):**
- Spec required: add, mul, div, sub, neg, recip, exp, sum, max, mean, add_scalar, mul_scalar, silu
- Initially implemented: add, mul, div, neg, recip, add_scalar, mul_scalar (7 of 13)
- Missing: exp, sum, max, mean, sub, silu
- Impact: T11-T19 blocked because `silu()` needed for MLP, `exp()` for softmax

**Verification checklist format:**
```markdown
## T05 Element-wise Operations Checklist

From spec section "T05: Element-wise Operations":

- [x] add(other: &Tensor)
- [x] mul(other: &Tensor)
- [x] div(other: &Tensor)
- [x] sub(other: &Tensor)
- [x] neg()
- [x] recip()
- [x] exp()
- [x] sum()
- [x] max()
- [x] mean()
- [x] add_scalar(scalar: f32)
- [x] mul_scalar(scalar: f32)
- [x] silu()
```

**Another example (T06 view operations):**
- Spec required: reshape, squeeze, unsqueeze, transpose, permute, flatten, select, narrow, slice
- Common miss: `narrow()`, `select()` (often overlooked as "similar to slice")
- Each is distinct: select extracts one index, narrow extracts range, slice is generic

**Pattern:** Create checklist BEFORE coding, implement all operations together in one pass.

### DType Preservation in View Operations (Extended)

**Rule:** ALL view/shape operations must preserve original dtype, not just reshape.

**Applies to:** reshape, squeeze, unsqueeze, flatten, permute, transpose, transpose_dims, select, narrow, slice

**Common violation:** `transpose_dims()` converting BF16 → F32

**Why wrong:**
- BF16 is for storage efficiency (half memory)
- View operations only change shape/layout, not data
- Silently converting to F32 doubles memory usage
- Model weights should stay BF16 until compute time

**Correct implementation pattern:**
```rust
pub fn transpose_dims(&self, dim0: usize, dim1: usize) -> Result<Tensor> {
    // Validate dimensions...

    // CORRECT: Preserve dtype by matching on TensorData
    match &self.data {
        TensorData::F32(data) => {
            // Permute F32 data, return F32 tensor
            Tensor::new(permuted_data, new_shape)
        }
        TensorData::BF16(data) => {
            // Permute BF16 data, return BF16 tensor
            Tensor::from_bf16(permuted_data, new_shape)
        }
    }
}

// WRONG: Always converts to F32
pub fn transpose_dims(&self, dim0: usize, dim1: usize) -> Result<Tensor> {
    let data = self.to_f32_vec()?;  // Loses BF16!
    // ... permute ...
    Tensor::new(permuted_data, new_shape)
}
```

**Testing requirement:** Each view operation must have explicit BF16 preservation test:
```rust
#[test]
fn transpose_dims_preserves_bf16() {
    let bf16_data: Vec<bf16> = vec![bf16::from_f32(1.0), bf16::from_f32(2.0),
                                     bf16::from_f32(3.0), bf16::from_f32(4.0)];
    let tensor = Tensor::from_bf16(bf16_data, vec![2, 2]).unwrap();

    let result = tensor.transpose_dims(0, 1).unwrap();

    // Verify dtype preserved
    assert!(matches!(result.data, TensorData::BF16(_)),
            "transpose_dims converted BF16 to F32");
}
```

**Checklist for new view operations:**
1. Implement logic for both F32 and BF16 branches
2. Match on `TensorData`, preserve variant in result
3. Add BF16 preservation test
4. Verify no `.to_f32_vec()` calls in implementation

## Constructor Validation Rule

**Rule:** All constructors with invariants MUST validate inputs eagerly, return `Result<Self>`.

**Why:** Fail-fast principle. Invalid state catches at initialization, not in forward() call.

**Common violations:**
- `Embedding`: accepts invalid vocab_size/embedding_dim
- `RmsNorm`: accepts invalid weight shape
- `Linear`: accepts mismatched input_dim/output_dim

**Correct pattern:**
```rust
impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let weight_shape = weight.shape().dims();
        if weight_shape.len() != 2 {
            return Err(LludaError::ShapeMismatch {
                expected: vec![0, 0],  // 2D required
                got: weight_shape.to_vec(),
            });
        }

        if let Some(ref b) = bias {
            let bias_shape = b.shape().dims();
            if bias_shape != &[weight_shape[0]] {
                return Err(LludaError::ShapeMismatch {
                    expected: vec![weight_shape[0]],
                    got: bias_shape.to_vec(),
                });
            }
        }

        Ok(Self { weight, bias })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // No shape validation needed here — invariants guaranteed
        x.matmul(&self.weight)?
            .add(&self.bias.as_ref().unwrap())
    }
}
```

**Benefits:**
- Errors occur at construction time, not during forward()
- Stack traces point to initialization, not buried in computation
- Invariants are guaranteed for all method calls

## GQA Configuration Validation

**Rule:** Group Query Attention requires `num_heads % num_kv_heads == 0`. Assert in constructor.

**Why:** Silent truncation of head groups leads to incorrect attention computation.

**Common violation:**
```rust
// BAD: num_heads=5, num_kv_heads=3
// Silently truncates to group_size=1, losing intended GQA benefits
impl AttentionConfig {
    pub fn group_size(&self) -> usize {
        self.num_heads / self.num_kv_heads  // Integer division truncates!
    }
}
```

**Correct pattern:**
```rust
impl AttentionConfig {
    pub fn new(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        // Validate GQA invariant
        if num_heads % num_kv_heads != 0 {
            return Err(LludaError::Msg(format!(
                "GQA requires num_heads ({}) divisible by num_kv_heads ({})",
                num_heads, num_kv_heads
            )));
        }

        Ok(Self {
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq_len,
        })
    }

    pub fn group_size(&self) -> usize {
        // Safe to use integer division — invariant guaranteed
        self.num_heads / self.num_kv_heads
    }
}
```

**Testing requirement:**
```rust
#[test]
fn gqa_rejects_invalid_head_counts() {
    let result = AttentionConfig::new(5, 3, 128, 2048);
    assert!(result.is_err(), "GQA should reject num_heads=5, num_kv_heads=3");
}
```

## Critical Path Test Coverage

**Rule:** All critical paths (masking, cache, generation) must have integration tests.

**Critical paths in inference:**
1. **Attention with mask** — causal mask, mask application to scores
2. **KV cache accumulation** — cache grows over generation steps
3. **Token generation loop** — sampling, cache updates, next token prediction

**Common violations:**
- Causal mask implementation untested with actual attention
- KV cache shape/indexing never checked in full forward pass
- Generation loop tested only in isolation, not end-to-end

**Test structure:**
```rust
#[test]
fn attention_with_causal_mask_generation_step() {
    // Test that causal mask correctly zeros future tokens during generation
    let (q, k, v) = attention_inputs_for_generation();

    let mask = create_causal_mask(seq_len, past_seq_len).unwrap();
    let masked_scores = scores.add(&mask).unwrap();  // Apply mask to scores

    // Verify future tokens are masked out (softmax → near-zero attention)
    let attn_weights = masked_scores.softmax_last_dim().unwrap();
    for j in (seq_pos + 1)..seq_len {
        assert!(attn_weights.index([batch, head, seq_pos, j]) < 1e-6);
    }
}

#[test]
fn kv_cache_accumulation_over_steps() {
    // Simulate generation: tokens generated one at a time
    let mut cache = KvCache::new(batch_size, max_seq_len, num_kv_heads, head_dim).unwrap();

    for step in 0..5 {
        let token_embedding = model.embed(token).unwrap();

        // Forward with cache
        let (output, new_cache) = attn_forward_with_cache(&token_embedding, &cache).unwrap();
        cache = new_cache;

        // Verify cache size grows correctly
        assert_eq!(cache.seq_len(), step + 1);
        assert_eq!(cache.keys().shape().dims()[1], step + 1);
        assert_eq!(cache.values().shape().dims()[1], step + 1);
    }
}

#[test]
fn full_generation_loop_end_to_end() {
    let model = load_test_model().unwrap();
    let mut tokens = vec![BOS_TOKEN];

    for _ in 0..10 {
        let logits = model.forward(&tokens).unwrap();
        let next_token = logits.argmax_last_dim().unwrap();
        tokens.push(next_token);

        if next_token == EOS_TOKEN {
            break;
        }
    }

    // Verify: generation completed, output is coherent
    assert!(tokens.len() > 1);
    assert!(tokens.contains(&EOS_TOKEN));
}
```

## Stable Rust API Compliance

**Rule:** Never use nightly-only APIs without feature gate. Prefer stable equivalents.

**Why:** Ensures code works on stable Rust (MSRV requirement).

**Common violation:**
```rust
// BAD: is_multiple_of() only on nightly (stabilized Rust 1.93.0+)
if x.is_multiple_of(2) {
    // ...
}
```

**Stable equivalent:**
```rust
// GOOD: Works on stable Rust 1.56+
if x % 2 == 0 {
    // even
}

// GOOD: More explicit
if x % divisor == 0 {
    // Multiple of divisor
}
```

**Testing for nightly API usage:**
```bash
# Find nightly-only method calls
cargo build --all-targets 2>&1 | grep "stabilize"
```

**Approved nightly features (if needed):**
Only use with `#![feature(...)]` at crate level AND documented reason:
```rust
#![feature(const_fn_floating_point_arithmetic)]  // Needed for compile-time constants in T09
```

**Checklist for new code:**
1. All method calls, types, macros must exist on MSRV (Minimum Supported Rust Version)
2. Test compilation on stable: `rustup default stable && cargo build`
3. If nightly required, document in code AND update MSRV in Cargo.toml

## No Panics in Library Code (Strict Enforcement)

**Rule:** Library code (src/) MUST NEVER use panic!, assert!, unwrap(), expect() or any panic-inducing operation. ALL errors via Result<T, LludaError>.

**Rationale:**
- Libraries should never crash the program — let caller decide how to handle errors
- User-provided configs/weights should produce recoverable errors, not panics
- Found violation: Attention::new used assert! for GQA divisibility and shape checks
- This would panic on malformed model files instead of returning proper error

**Pattern - WRONG (panics):**
```rust
pub fn new(num_heads: usize, num_kv_heads: usize) -> Self {
    assert!(num_heads % num_kv_heads == 0, "divisibility check");
    // ...
}
```

**Pattern - CORRECT (returns Result):**
```rust
pub fn new(num_heads: usize, num_kv_heads: usize) -> Result<Self> {
    if num_heads % num_kv_heads != 0 {
        return Err(LludaError::Model(format!(
            "GQA requires num_heads ({}) divisible by num_kv_heads ({})",
            num_heads, num_kv_heads
        )));
    }
    Ok(Self { ... })
}
```

**Exceptions:**
- Binary/example code (examples/, tests/) MAY use unwrap() for simplicity
- Test code MAY use assert! for test assertions
- Internal private helper functions MAY panic IF documented and caller handles

**Verification:**
- Search codebase for: panic!, assert!, unwrap(), expect()
- Ensure all occurrences are in test/example code only
- Code review MUST catch any library panics

### Shape assertions at function entry:
```rust
pub fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {
    let a_dims = self.shape().dims();
    let b_dims = rhs.shape().dims();

    // Validate shapes FIRST, before any computation
    if a_dims.last() != b_dims.get(b_dims.len().saturating_sub(2)).copied().as_ref() {
        return Err(LludaError::ShapeMismatch {
            expected: vec![/* ... */],
            got: vec![/* ... */],
        });
    }
    // ... compute
}
```

### Builder pattern for complex tensors:
```rust
// Not needed for Phase 0. Keep it simple: constructors and methods.
```

---

## 5. Documentation

### Every public item must have a doc comment:
```rust
/// RMS Layer Normalization.
///
/// Computes: `x * rsqrt(mean(x^2) + eps) * weight`
///
/// # Arguments
/// * `x` - Input tensor of shape `[..., dim]`
///
/// # Returns
/// Normalized tensor of same shape as input.
///
/// # Reference
/// Qwen3-0.6B uses this for:
/// - Layer norm: weight `[1024]`, applied to `[B, L, 1024]`
/// - Per-head q/k norm: weight `[128]`, applied to `[B*H, L, 128]`
pub fn forward(&self, x: &Tensor) -> Result<Tensor> { ... }
```

### Document non-obvious decisions:
```rust
// The causal mask returns None when seq_len == 1 because during generation,
// the single new token can attend to all past tokens (via KV cache).
// This optimization avoids allocating a 1xN mask that's all zeros.
if seq_len == 1 {
    return Ok(None);
}
```

### Reference existing implementations:
```rust
// Implementation matches candle qwen3.rs lines 56-63 (Qwen3RotaryEmbedding::apply)
// and Python transformers Qwen3RotaryEmbedding.forward()
```

---

## 6. Testing

### Test file organization:
```
tests/
  tensor_ops.rs          -- Tensor math operations
  model_components.rs    -- Individual model layers
  integration.rs         -- Full model tests (may require model files)
```

### Test naming:
```rust
#[test]
fn matmul_2d_basic() { ... }

#[test]
fn matmul_4d_batched_attention_shape() { ... }

#[test]
fn rms_norm_matches_pytorch() { ... }

#[test]
fn attention_with_kv_cache_sequential() { ... }
```

### Test structure (Arrange-Act-Assert):
```rust
#[test]
fn softmax_sums_to_one() {
    // Arrange
    let input = Tensor::from_f32_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();

    // Act
    let output = input.softmax_last_dim().unwrap();

    // Assert
    let data = output.to_f32_vec().unwrap();
    let sum: f32 = data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "softmax sum = {sum}, expected 1.0");
}
```

### Test categories:
1. **Unit tests** (inline `#[cfg(test)] mod tests`): For private functions
2. **Integration tests** (`tests/`): For public API
3. **Model tests** (guarded): Skip if model files not present
```rust
#[test]
fn load_qwen3_06b() {
    let model_path = Path::new("models/Qwen3-0.6B");
    if !model_path.exists() {
        eprintln!("Skipping: model files not found at {}", model_path.display());
        return;
    }
    // ... test with real model
}
```

### Test Module Structure

**Rule:** ALL tests must be inside `#[cfg(test)] mod tests { ... }`

**Common mistake:** Closing `}` of test module too early, leaving test functions outside.

**Verification:**
1. Find the line with `#[cfg(test)] mod tests {`
2. Count opening `{` and closing `}` braces to find matching pair
3. Ensure ALL test functions are between those braces
4. Last line of tests should be `}` closing the mod tests

**Example (CORRECT):**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foo() { ... }

    #[test]
    fn test_bar() { ... }

    // ALL tests here
} // <-- This closes mod tests (should be at end of file)
```

**Example (WRONG):**
```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_foo() { ... }
} // <-- TOO EARLY!

#[test]  // <-- This test is OUTSIDE #[cfg(test)], will compile into production!
fn test_bar() { ... }
```

**Why critical:** Tests outside `#[cfg(test)]` compile into production binary, increasing binary size and potentially exposing test-only code paths.

### Test isolation:
- All tests run offline (no network)
- No shared mutable state between tests
- Tests that need model files are skippable

### Approximate comparisons:
```rust
// For floating point comparisons
fn assert_close(a: f32, b: f32, atol: f32, msg: &str) {
    assert!((a - b).abs() < atol, "{msg}: {a} vs {b}, diff = {}", (a - b).abs());
}

fn assert_tensors_close(a: &Tensor, b: &Tensor, atol: f32) {
    let a_data = a.to_f32_vec().unwrap();
    let b_data = b.to_f32_vec().unwrap();
    assert_eq!(a_data.len(), b_data.len(), "tensor sizes differ");
    for (i, (x, y)) in a_data.iter().zip(b_data.iter()).enumerate() {
        assert!((x - y).abs() < atol,
            "element {i}: {x} vs {y}, diff = {}", (x - y).abs());
    }
}
```

---

## 7. Specification Completeness

**Rule:** Before marking a task complete, verify ALL spec requirements implemented.

**Process:**
1. Read task spec in `docs/architecture/phase0-implementation-plan.md`
2. List ALL required operations/features
3. Implement each one with tests
4. Cross-check: did I miss anything?

**Common mistake:** Implementing "most" of the spec, missing 30-40% of required ops.

**Example from T05 (element-wise ops):**
- Implemented: add, mul, div, neg, recip, add_scalar, mul_scalar
- Missed initially: exp, sum, max, mean, sub, silu
- **Lesson:** Review spec line-by-line, not just summary

**Verification:** Use table format to track:
| Operation | Spec Section | Status | Tests |
|-----------|--------------|--------|-------|
| add() | T05 | ✅ | test_add_* |
| exp() | T05 | ✅ | test_exp_* |
| ... | ... | ... | ... |

---

## 8. Code Quality

### Clippy Compliance

**Rule:** Zero Clippy warnings before task completion.

**Run:** `cargo clippy -- -D warnings` (treat warnings as errors)

**Common issues:**
- `clippy::op_ref`: Use `== [1]` instead of `== &[1]`
- `clippy::doc_overindented`: Fix doc comment indentation
- `clippy::unnecessary_wraps`: Return T instead of Result<T> if never errors

**Fix before:** Marking task complete, committing code, passing to next task.

---

## 9. Performance Annotations

Phase 0 prioritizes correctness over performance. Mark optimization opportunities:

```rust
// PERF: Naive O(n^3) matmul. Replace with BLAS or tiled implementation in Phase 3.
for i in 0..m {
    for j in 0..n {
        for k in 0..p {
            result[i * n + j] += a[i * p + k] * b[k * n + j];
        }
    }
}

// PERF: Allocates new Vec for each operation. Consider in-place ops for Phase 3.
let result: Vec<f32> = data.iter().map(|&x| f32::exp(x)).collect();
```

---

## 10. Safety

### No unsafe in Phase 0:
```rust
// GOOD: Use bytemuck for safe type punning
let f32_slice: &[f32] = bytemuck::cast_slice(raw_bytes);

// BAD: Manual unsafe transmute
let f32_slice: &[f32] = unsafe { std::slice::from_raw_parts(ptr, len) };
```

**Exception:** If performance requires it later, `unsafe` blocks must:
1. Have a `// SAFETY:` comment explaining the invariant
2. Be wrapped in a safe function with documented preconditions
3. Be reviewed

### Memory-mapped files:
```rust
// memmap2::Mmap is safe to use. The OS handles access.
// Wrap in Arc for shared ownership across tensors.
let mmap = Arc::new(unsafe { Mmap::map(&file)? });
// SAFETY: File is read-only, opened with read permissions.
// Memory-mapped access is safe as long as file is not modified externally.
```

---

## 11. Journaling

### When to journal (write to `docs/journal/phase0/`):
- After completing each task
- When discovering unexpected behavior
- When making a design decision not in the plan
- When performance is notably better/worse than expected

### Format:
```markdown
# T{XX}: {Task Name}

## Date
YYYY-MM-DD

## Summary
{What was implemented, in 2-3 sentences}

## Decisions
- {Decision}: {Rationale}

## Findings
- {Finding}: {Detail, implication for Omni}

## Metrics
| Metric | Value |
|--------|-------|
| Lines of code | {N} |
| Test count | {N} |
| {Other} | {Value} |

## Next
{What this enables}
```

### Ledger updates:
After each task, append to `logs/ledger.md`:
```markdown
## YYYY-MM-DD HH:MM
Done: T{XX} {task name} — {one-line summary}
Next: T{YY} {next task}
```

---

## 12. Commit Standards

### Commit when:
- A task is complete and its tests pass
- Never commit broken code

### Commit message format:
```
phase0: T{XX} {component} — {what was done}

{Optional detail about key decisions or findings}
```

Examples:
```
phase0: T03 tensor core — implement Tensor struct with F32/BF16 storage
phase0: T13 rope — rotary position embeddings matching candle reference
phase0: T14 attention — GQA with per-head q/k norm and KV cache
```

### Commit between tasks:
Each completed task = one commit. This preserves working versions and makes bisection possible if validation fails.
