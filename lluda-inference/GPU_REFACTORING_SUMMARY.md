# GPU Strict Mode Refactoring Summary

## Overview

Successfully implemented strict GPU-or-fail policy throughout lluda-inference codebase. Removed all silent CPU fallbacks when GPU feature is enabled.

## Core Principle

**"Если вдруг что-то не поддерживается железом, мы сразу падаем."**

When `--features gpu` is enabled:
- Operations MUST use GPU or return clear error
- NO silent fallback to CPU
- NO conditional "try GPU then CPU" patterns
- Fail fast with descriptive error messages

## Changes Made

### 1. GPU_DESIGN_PRINCIPLES.md

Created comprehensive design document explaining:
- Rationale for no-fallback policy
- Error message guidelines
- Code patterns (correct vs wrong)
- Testing requirements
- GPU context initialization strategy

### 2. src/tensor.rs - Matmul Function Refactoring

**Before** (lines 778-933):
- Conditional `is_gemm_candidate()` check
- `try_gemm_gpu()` with silent fallback on error
- Falls through to CPU code after GPU failure
- Same pattern for GEMV

**After**:
```rust
pub fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "gpu")]
    {
        return self.matmul_gpu(rhs);  // GPU-only, no fallback
    }

    #[cfg(not(feature = "gpu"))]
    {
        return self.matmul_cpu(rhs);  // CPU-only
    }
}
```

**GPU implementation** (`matmul_gpu`):
- Checks 2D×2D → `matmul_gemm_gpu()`
- Checks 2D×1D → `matmul_gemv_gpu()`
- Everything else → Error with clear message

**CPU implementation** (`matmul_cpu`):
- Supports all shapes (2D, 3D, 4D)
- Isolated from GPU code via `#[cfg(not(feature = "gpu"))]`

### 3. Removed Fallback Methods

**Deleted** (237 lines total):
- `is_gemm_candidate()` - conditional GPU suitability check
- `try_gemm_gpu()` - GPU with fallback on error
- `is_gemv_candidate()` - conditional GPU suitability check
- `try_gemv_gpu()` - GPU with fallback on error

**Added** (strict GPU-only):
- `matmul_gemm_gpu()` - GPU GEMM or error
- `matmul_gemv_gpu()` - GPU GEMV or error

### 4. Strict GPU Methods

**matmul_gemm_gpu()** (2D×2D matrix multiplication):

```rust
fn matmul_gemm_gpu(&self, rhs: &Tensor) -> Result<Tensor> {
    // 1. Validate shapes (2D×2D)
    if lhs_shape.len() != 2 || rhs_shape.len() != 2 {
        return Err(LludaError::Msg(
            "GPU GEMM requires 2D×2D shapes, got {}D×{}D"
        ));
    }

    // 2. Validate dimensions match
    if k1 != k2 {
        return Err(LludaError::Msg(
            "GPU GEMM shape mismatch: {}×{} @ {}×{} (K dimensions must match)"
        ));
    }

    // 3. Strict dtype check - BF16 required
    if self.dtype() != DType::BF16 || rhs.dtype() != DType::BF16 {
        return Err(LludaError::Msg(
            "GPU GEMM requires BF16 tensors, got {:?}×{:?}. \
             Convert tensors to BF16 before GPU operations."
        ));
    }

    // 4. Execute on GPU - no fallback
    let ctx = gpu::get_context()?;
    gpu::gemm::gemm_forward(ctx, self, rhs)
}
```

**matmul_gemv_gpu()** (2D×1D matrix-vector multiplication):

Similar structure with 2D×1D shape requirements.

**Key improvements**:
- No size thresholds (removed M*K*N > 8192 check)
- No conditional logic
- Clear error messages with resolution steps
- Profiling output simplified

### 5. src/gpu/mod.rs - Context Initialization

**Before**:
```rust
Err(e) => {
    eprintln!("[GPU] GPU init failed: {}, falling back to CPU", e);
    None  // Silent fallback
}
```

**After**:
```rust
Err(e) => {
    panic!(
        "GPU initialization failed: {}.\n\
         Cannot run in GPU mode. Remove --features gpu to use CPU.",
        e
    );
}
```

**Rationale**:
- Binary fails immediately at startup if GPU unavailable
- No wasted time before first GPU operation error
- Clear message tells user exactly how to fix (remove `--features gpu`)
- No silent degradation to CPU mode

### 6. Test Updates

Updated test comments to reflect new architecture:
- Old: `Tensor::matmul() -> is_gemm_candidate -> try_gemm_gpu -> gemm_forward`
- New: `Tensor::matmul() -> matmul_gpu() -> matmul_gemm_gpu() -> gemm_forward`

### 7. Example: test_strict_gpu.rs

Created example demonstrating strict GPU behavior:

```bash
cargo run --features gpu --example test_strict_gpu
```

Tests:
1. F32 tensors → Clear dtype error
2. BF16 tensors → Success
3. 3D tensors → Unsupported shape error

## Error Messages

### Example 1: F32 tensors in GPU mode

```
Error: GPU GEMM requires BF16 tensors, got F32×F32. Convert tensors to BF16 before GPU operations.
```

### Example 2: Unsupported shape

```
Error: GPU mode: unsupported matmul shapes 3D×3D. Only 2D×2D (GEMM) and 2D×1D (GEMV) are supported on GPU. Use CPU mode (remove --features gpu) for batched operations (3D, 4D).
```

### Example 3: GPU init failure

```
thread 'main' panicked at src/gpu/mod.rs:169:17:
GPU initialization failed: Failed to request GPU adapter: ().
Cannot run in GPU mode. Remove --features gpu to use CPU.
```

## Verification

### Compilation

```bash
# GPU mode - compiles successfully
cargo check --features gpu
# Warnings: matmul_2d, matmul_3d, matmul_4d unused (expected - CPU helpers not used in GPU mode)

# CPU mode - compiles successfully
cargo check
# No warnings
```

### Expected Behavior

**GPU mode** (`--features gpu`):
```bash
cargo run --features gpu --example benchmark_inference
# Expected: Clear error about F32 tensors
# Message: "GPU GEMM requires BF16 tensors, got F32×F32..."
```

**CPU mode** (default):
```bash
cargo run --example benchmark_inference
# Expected: Runs successfully using CPU
# No GPU code compiled at all
```

### Profiling Mode

```bash
PROFILE=1 cargo run --features gpu --example test_strict_gpu
```

Output shows:
```
[GPU] matmul dispatch: 2×2 @ 2×2 (2D×2D GEMM)
[GPU]   dtype: BF16×BF16
[GPU]   Executing GEMM: 2×2 @ 2×2
[GPU]   GEMM execution: 0.45ms
```

No "falling back to CPU" messages - because there is no fallback!

## Benefits

1. **Predictable**: GPU code always runs on GPU, never silently switches to CPU
2. **Debuggable**: All issues surface immediately as errors with clear messages
3. **Testable**: Can verify GPU is actually being used (no hidden CPU fallback)
4. **Observable**: Profiling shows actual execution path, no silent decisions
5. **Maintainable**: Simpler code, no conditional logic, clear separation of GPU/CPU paths
6. **Performance**: No wasted time running CPU fallback when user expects GPU

## Code Statistics

- **Removed**: 237 lines (4 fallback methods + conditional logic)
- **Added**: ~200 lines (2 strict methods + 2 mode-specific dispatch functions)
- **Net change**: -37 lines of code
- **Complexity**: Significantly reduced (no conditional fallback paths)

## Migration for Users

### Before

```rust
// Silent behavior - might use GPU or CPU, user doesn't know
let result = tensor_a.matmul(&tensor_b)?;
```

### After - CPU Mode (default)

```rust
// Explicit CPU mode - clear and predictable
cargo build  // or cargo build --no-default-features
let result = tensor_a.matmul(&tensor_b)?;  // Always uses CPU
```

### After - GPU Mode

```rust
cargo build --features gpu

// Convert to BF16 before GPU operations
let a_bf16 = tensor_a.to_dtype(DType::BF16)?;
let b_bf16 = tensor_b.to_dtype(DType::BF16)?;

// GPU mode - uses GPU or fails with clear error
let result = a_bf16.matmul(&b_bf16)?;
```

If error occurs:
```
Error: GPU GEMM requires BF16 tensors, got F32×F32. Convert tensors to BF16 before GPU operations.
```

User knows exactly what to fix.

## Future Work

As GPU implementations expand:

1. Add GPU support for 3D/4D batched matmul
2. Add GPU support for other operations (conv2d, etc.)
3. Each new GPU operation follows same pattern:
   - Strict dtype/shape validation
   - Clear error messages
   - No CPU fallback
   - CPU mode via `#[cfg(not(feature = "gpu"))]`

## Files Changed

1. `/home/alexii/lluda/lluda-inference/GPU_DESIGN_PRINCIPLES.md` - New file
2. `/home/alexii/lluda/lluda-inference/src/tensor.rs` - Major refactoring
3. `/home/alexii/lluda/lluda-inference/src/gpu/mod.rs` - Strict initialization
4. `/home/alexii/lluda/lluda-inference/examples/test_strict_gpu.rs` - New example
5. `/home/alexii/lluda/lluda-inference/GPU_REFACTORING_SUMMARY.md` - This file

## Backup

Original tensor.rs backed up to: `/home/alexii/lluda/lluda-inference/src/tensor.rs.backup`

Can restore with: `cp src/tensor.rs.backup src/tensor.rs`
