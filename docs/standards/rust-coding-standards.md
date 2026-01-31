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
