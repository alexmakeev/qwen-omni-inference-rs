# GPU Design Principles - lluda-inference

## Core Principle: NO FALLBACKS

**Rule**: When GPU feature is enabled (`--features gpu`), operations MUST use GPU or FAIL with clear error.

### Rationale

Silent fallbacks mask bugs and performance issues:
- User thinks GPU is working, but CPU is actually running 100%
- Impossible to verify GPU is actually being used
- Cannot debug what we cannot observe
- Performance regressions go undetected
- "если она уже переключена, то либо она работает, либо не работает"
- "мы неясно что проверять будем" - silent fallbacks make validation impossible

### Implementation Rules

1. **GPU mode** (`--features gpu`): All operations use GPU or return error
2. **CPU mode** (default): All operations use CPU (no GPU code compiled)
3. **No hybrid**: No automatic fallback from GPU to CPU
4. **Fail fast**: If operation cannot run on GPU, return `Err` immediately with clear message

### Error Messages

Every GPU rejection must provide:
- Clear indication of what failed
- Why it failed (dtype mismatch, size issue, unsupported operation, etc.)
- What user needs to fix (convert to BF16, change settings, use CPU mode, etc.)

Examples:
```
Error: GPU GEMM requires BF16 tensors, got F32×F32. Convert tensors to BF16 before GPU operations.

Error: GPU mode: unsupported matmul shapes 3D×3D (only 2D×2D and 2D×1D supported). Use CPU mode for batched operations.

Error: GPU initialization failed: no suitable adapter found. Remove --features gpu to use CPU mode.
```

### Testing

- GPU tests must fail if GPU not available or operation rejected
- No `#[ignore]` or conditional test skipping based on GPU availability
- Test should error with clear message, not skip silently
- Profiling mode (`PROFILE=1`) logs EVERY operation dispatch decision

### Code Patterns

✅ **CORRECT** - Fail fast with clear error:
```rust
#[cfg(feature = "gpu")]
pub fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {
    // GPU mode - must use GPU or error

    // Check 2D×2D GEMM
    if self.ndim() == 2 && rhs.ndim() == 2 {
        return self.matmul_gemm_gpu(rhs);  // No fallback, propagate error
    }

    // Check 2D×1D GEMV
    if self.ndim() == 2 && rhs.ndim() == 1 {
        return self.matmul_gemv_gpu(rhs);  // No fallback, propagate error
    }

    // Unsupported in GPU mode
    Err(LludaError::Msg(format!(
        "GPU mode: unsupported matmul shapes {}D×{}D (only 2D×2D and 2D×1D supported)",
        self.ndim(), rhs.ndim()
    )))
}

fn matmul_gemm_gpu(&self, rhs: &Tensor) -> Result<Tensor> {
    // Strict dtype check
    if self.dtype() != DType::BF16 || rhs.dtype() != DType::BF16 {
        return Err(LludaError::Msg(format!(
            "GPU GEMM requires BF16 tensors, got {:?}×{:?}. \
             Convert tensors to BF16 before GPU operations.",
            self.dtype(), rhs.dtype()
        )));
    }

    // Validate shapes
    let (m, k1) = (self.shape[0], self.shape[1]);
    let (k2, n) = (rhs.shape[0], rhs.shape[1]);
    if k1 != k2 {
        return Err(LludaError::Msg(format!(
            "GPU GEMM shape mismatch: {}×{} @ {}×{} (K dimensions must match)",
            m, k1, k2, n
        )));
    }

    // Execute on GPU - no fallback
    let ctx = crate::gpu::get_context()?;
    crate::gpu::gemm::gemm_forward(ctx, self, rhs)
}
```

❌ **WRONG** - Silent fallback pattern:
```rust
#[cfg(feature = "gpu")]
{
    if let Ok(result) = self.try_gemm_gpu(rhs) {
        return Ok(result);  // Silent fallback to CPU if fails
    }
    // Falls through to CPU - MASKS GPU ISSUES!
}

// CPU implementation executes, user thinks GPU is working
self.matmul_cpu(rhs)
```

❌ **WRONG** - Conditional try with size threshold fallback:
```rust
if self.is_gemm_candidate(rhs) {
    // Returns false for F32 or small matrices
    // Silently falls back to CPU without error
    match self.try_gemm_gpu(rhs) {
        Ok(result) => return Ok(result),
        Err(_) => {} // Swallow error, fallback to CPU
    }
}
```

### Size Thresholds

Current approach: **No size thresholds in GPU mode**

Rationale:
- Small matrices still test GPU code path
- Thresholds create another silent fallback point
- User should control dispatch strategy explicitly
- If GPU overhead is too high, user should use CPU mode

Alternative (if absolutely needed):
```rust
// Error instead of fallback
if m * k * n < 8192 {
    return Err(LludaError::Msg(format!(
        "GPU GEMM: matrix too small (M×K×N = {}), use CPU mode or increase batch size",
        m * k * n
    )));
}
```

Recommendation: Remove size thresholds entirely in GPU mode.

### GPU Context Initialization

**Strict mode**: Panic on init failure (cannot run in GPU mode without GPU)

```rust
pub fn get_context() -> Result<&'static GpuContext> {
    GPU_CONTEXT.get_or_init(|| {
        match init() {
            Ok(ctx) => {
                eprintln!("GPU initialized: {:?}", ctx.adapter_info().name);
                Some(ctx)
            }
            Err(e) => {
                // FAIL FAST - cannot run in GPU mode
                panic!(
                    "GPU initialization failed: {}.\n\
                     Cannot run in GPU mode. Remove --features gpu to use CPU.",
                    e
                );
            }
        }
    })
    .as_ref()
    .ok_or_else(|| LludaError::Msg("GPU not available".into()))
}
```

This ensures:
- Binary fails immediately at startup if GPU unavailable
- No silent fallback that wastes time before first error
- Clear error message tells user how to fix (remove `--features gpu`)

### Profiling and Observability

In `PROFILE=1` mode, log every dispatch decision:

```rust
let profiling = std::env::var("PROFILE").is_ok();
if profiling {
    eprintln!("[GPU] matmul dispatch: {}×{} @ {}×{}", ...);
    eprintln!("[GPU]   dtype: {:?}×{:?}", self.dtype(), rhs.dtype());
    eprintln!("[GPU]   decision: executing GEMM on GPU");
}
```

NEVER log "falling back to CPU" - because there is no fallback!

### Migration Path

For operations not yet implemented on GPU:

1. Document unsupported operations clearly
2. Return explicit error in GPU mode
3. Implement CPU fallback ONLY in CPU mode (`#[cfg(not(feature = "gpu"))]`)
4. User can choose: use CPU mode or wait for GPU implementation

Example:
```rust
#[cfg(feature = "gpu")]
pub fn conv2d(&self, kernel: &Tensor) -> Result<Tensor> {
    Err(LludaError::Msg(
        "GPU mode: conv2d not yet implemented. Use CPU mode (remove --features gpu)".into()
    ))
}

#[cfg(not(feature = "gpu"))]
pub fn conv2d(&self, kernel: &Tensor) -> Result<Tensor> {
    // CPU implementation
    self.conv2d_cpu(kernel)
}
```

### Summary

**GPU or fail** principle ensures:
- Predictable behavior: GPU code always runs on GPU
- Debuggable: All issues surface immediately as errors
- Testable: Can verify GPU is actually being used
- Observable: No hidden fallbacks, clear error messages
- Maintainable: Simpler code, no conditional logic

"Если вдруг что-то не поддерживается железом, мы сразу падаем. То есть в режиме GPU, если мы хотим на GPU запуститься, а мы должны запускаться на GPU, а не на CPU. Без всякого фуллбэка на соседний вариант."
