# GPU Strict Mode - Quick Reference

## TL;DR

**GPU mode = GPU only. No fallbacks. Fail fast with clear errors.**

## Compilation Modes

```bash
# CPU mode (default) - no GPU code compiled
cargo build
cargo run --example benchmark_inference

# GPU mode - GPU required, fails if unavailable
cargo build --features gpu
cargo run --features gpu --example benchmark_inference
```

## Error Messages You'll See

### 1. Wrong dtype (most common)

```
Error: GPU GEMM requires BF16 tensors, got F32×F32.
Convert tensors to BF16 before GPU operations.
```

**Fix**:
```rust
let a_bf16 = tensor_a.to_dtype(DType::BF16)?;
let b_bf16 = tensor_b.to_dtype(DType::BF16)?;
let result = a_bf16.matmul(&b_bf16)?;
```

### 2. Unsupported shape

```
Error: GPU mode: unsupported matmul shapes 3D×3D.
Only 2D×2D (GEMM) and 2D×1D (GEMV) are supported on GPU.
Use CPU mode (remove --features gpu) for batched operations (3D, 4D).
```

**Fix**: Use CPU mode or wait for GPU implementation
```bash
cargo build  # Remove --features gpu
```

### 3. GPU not available

```
thread 'main' panicked at src/gpu/mod.rs:169:17:
GPU initialization failed: Failed to request GPU adapter: ().
Cannot run in GPU mode. Remove --features gpu to use CPU.
```

**Fix**: Build without GPU feature
```bash
cargo build  # Remove --features gpu
```

## Supported Operations (GPU Mode)

### Matmul

✅ **2D×2D (GEMM)** - BF16 tensors
```rust
// [M, K] @ [K, N] -> [M, N]
let a = Tensor::from_bf16(data_a, vec![128, 1024])?;
let b = Tensor::from_bf16(data_b, vec![1024, 256])?;
let c = a.matmul(&b)?;  // GPU GEMM
```

✅ **2D×1D (GEMV)** - BF16 tensors
```rust
// [M, K] @ [K] -> [M]
let a = Tensor::from_bf16(data_a, vec![128, 1024])?;
let b = Tensor::from_bf16(data_b, vec![1024])?;
let c = a.matmul(&b)?;  // GPU GEMV
```

❌ **3D×3D, 4D×4D** - Not supported
```rust
// Use CPU mode for batched operations
cargo build  # Remove --features gpu
```

❌ **F32 tensors** - Not supported
```rust
// Convert to BF16 first
let a_bf16 = a_f32.to_dtype(DType::BF16)?;
```

## Profiling

```bash
# See exactly what GPU is doing
PROFILE=1 cargo run --features gpu --example test_strict_gpu
```

Output:
```
[GPU] matmul dispatch: 2×2 @ 2×2 (2D×2D GEMM)
[GPU]   dtype: BF16×BF16
[GPU]   Executing GEMM: 2×2 @ 2×2
[GPU]   GEMM execution: 0.45ms
```

No "falling back to CPU" - because there is NO fallback!

## Testing

```bash
# Test GPU strict mode
cargo run --features gpu --example test_strict_gpu

# Run GPU integration tests
cargo test --features gpu --lib test_tensor_matmul_gemm_integration
```

## Common Workflows

### Development/Testing (flexible)
```bash
cargo build              # CPU mode, works everywhere
cargo test               # All tests with CPU
```

### Production (GPU required)
```bash
cargo build --features gpu --release
cargo run --features gpu --release --example inference
```

### Debugging GPU Issues
```bash
PROFILE=1 cargo run --features gpu --example test_strict_gpu
# Check error messages
# Verify tensors are BF16
# Verify shapes are 2D×2D or 2D×1D
```

## Migration Checklist

If you get GPU errors after refactoring:

1. ✅ Check dtype: `tensor.dtype()` should be `DType::BF16`
   - If F32: `tensor.to_dtype(DType::BF16)?`

2. ✅ Check shapes: `tensor.shape()` should be 2D for GEMM/GEMV
   - 2D×2D → GEMM
   - 2D×1D → GEMV
   - 3D/4D → Use CPU mode

3. ✅ Check GPU available: Does `cargo run --features gpu` work?
   - No GPU? → Use CPU mode: `cargo build`

4. ✅ Check error message: Read it carefully
   - All errors now include resolution steps
   - Follow instructions in error message

## Design Principle

See `GPU_DESIGN_PRINCIPLES.md` for full rationale.

**Summary**: "Если вдруг что-то не поддерживается железом, мы сразу падаем."

- GPU mode means GPU ONLY
- No silent fallbacks
- No conditional logic
- Clear errors with fix instructions
- Debuggable, testable, predictable
