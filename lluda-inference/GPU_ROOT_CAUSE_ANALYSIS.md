# GPU Performance Root Cause Analysis

## Executive Summary

**Problem:** GPU mode is SLOWER than CPU (0.20 tok/s vs 0.21 tok/s)

**Root Cause:** GPU is NOT being used at all. All operations fall back to CPU.

**Reason:** GPU only implements GEMV (matrix × vector), but model uses GEMM (matrix × matrix).

## Detailed Findings

### Current GPU Implementation

The GPU acceleration (`--features gpu`) only implements:
- **GEMV**: Matrix × Vector multiplication
- **Required shape**: 2D × 1D → 1D
- **Example**: `[1024, 512] × [512]` → `[1024]`

### Actual Model Operations

All matmul operations in Qwen3-0.6B are:
- **GEMM**: Matrix × Matrix multiplication
- **Actual shapes**: 2D × 2D → 2D or 4D × 4D → 4D
- **Examples**:
  - Linear layers: `[5, 1024] × [1024, 2048]` → `[5, 2048]`
  - Attention: `[1, 16, 5, 128] × [1, 16, 128, 5]` → `[1, 16, 5, 5]`

### Operation Breakdown (Per Layer)

From profiling output, each transformer layer performs:

**Linear Projections (Matrix-Matrix):**
1. Q projection: `[batch*seq, 1024] × [1024, 1024]`
2. K projection: `[batch*seq, 1024] × [1024, 1024]`
3. V projection: `[batch*seq, 1024] × [1024, 2048]`
4. O projection: `[batch*seq, 2048] × [2048, 1024]`
5. FFN up1: `[batch*seq, 1024] × [1024, 3072]`
6. FFN up2: `[batch*seq, 1024] × [1024, 3072]`
7. FFN down: `[batch*seq, 3072] × [3072, 1024]`

**Attention (4D Batched):**
1. Q @ K^T: `[1, 16, seq, 128] × [1, 16, 128, seq]`
2. attn @ V: `[1, 16, seq, seq] × [1, 16, seq, 128]`

**None of these match 2D×1D GEMV pattern!**

### Why Even Single Token is 2D×2D

During autoregressive decoding (batch=1, seq=1):
- Input shape: `[1, 1024]` (2D tensor)
- NOT `[1024]` (1D tensor)
- Weight shape: `[1024, 2048]` (2D matrix)
- Operation: `[1, 1024] × [1024, 2048]` → `[1, 2048]`
- **Still 2D×2D, not 2D×1D!**

The model maintains batch dimension throughout, even when batch=1.

### Profiling Evidence

Sample output from profiling benchmark:
```
[GPU] GEMV rejected: shape mismatch (need 2D×1D, got 2D×2D): [5, 1024] × [1024, 2048]
[GPU] GEMV rejected: shape mismatch (need 2D×1D, got 2D×2D): [5, 1024] × [1024, 1024]
[GPU] GEMV rejected: shape mismatch (need 2D×1D, got 4D×4D): [1, 16, 5, 128] × [1, 16, 128, 5]
[GPU] GEMV rejected: shape mismatch (need 2D×1D, got 2D×2D): [5, 2048] × [2048, 1024]
[GPU] GEMV rejected: shape mismatch (need 2D×1D, got 2D×2D): [5, 1024] × [1024, 3072]
```

**Result**: 100% of matmul operations rejected, 0% GPU utilization.

### Performance Impact

Current situation:
- CPU path: All operations
- GPU path: None (0% utilization)
- Overhead: GPU initialization, checking, rejection
- Result: GPU mode ≈ CPU mode (both use CPU compute)

## Solutions

### Option 1: Implement General GEMM (Recommended)

**What to implement:**
```rust
// General matrix-matrix multiplication
fn gemm_forward(
    ctx: &GpuContext,
    matrix_a: &Tensor,  // [M, K]
    matrix_b: &Tensor,  // [K, N]
) -> Result<Tensor> {  // [M, N]
    // GPU kernel for GEMM
}
```

**Impact:**
- Covers all linear layer operations (7 per layer × 28 layers = 196 ops)
- Expected speedup: 5-10x on GPU vs CPU for these operations
- ~80% of total compute time

**Complexity:** Medium
- Need GEMM shader (more complex than GEMV)
- Workgroup tiling for efficiency
- Shared memory optimization

### Option 2: Implement Batched 4D GEMM

**What to implement:**
```rust
// Batched matrix multiplication for attention
fn batched_gemm_4d(
    ctx: &GpuContext,
    tensor_a: &Tensor,  // [B, H, M, K]
    tensor_b: &Tensor,  // [B, H, K, N]
) -> Result<Tensor> {  // [B, H, M, N]
    // GPU kernel for batched GEMM
}
```

**Impact:**
- Covers attention operations (2 per layer × 28 layers = 56 ops)
- Expected speedup: 3-8x on GPU vs CPU
- ~20% of total compute time

**Complexity:** Medium-High
- More complex than 2D GEMM
- Need to handle batch/head dimensions
- Memory layout critical

### Option 3: Hybrid Approach (Best)

Implement both:
1. **2D GEMM** for linear layers
2. **4D Batched GEMM** for attention
3. Keep existing **GEMV** for potential future use

**Impact:**
- Complete GPU coverage
- Expected overall speedup: 5-10x total
- Optimal for all scenarios

## Implementation Roadmap

### Phase 1: Basic 2D GEMM (High Priority)
1. Implement naive GEMM shader
2. Add to matmul dispatch logic
3. Test with linear layers
4. Benchmark vs CPU

### Phase 2: Optimized 2D GEMM
1. Add tiling (8×8 or 16×16 workgroups)
2. Shared memory optimization
3. Coalesced memory access
4. Tune for RADV

### Phase 3: Batched 4D GEMM
1. Implement batched GEMM shader
2. Handle multi-head attention layout
3. Optimize for attention patterns
4. Benchmark vs CPU

### Phase 4: Production Readiness
1. Comprehensive testing
2. Error handling
3. Fallback strategies
4. Performance validation

## Testing Strategy

### Verification Tests

1. **Correctness**: Compare GPU vs CPU outputs (tolerance: 1e-3 for BF16)
2. **Performance**: Benchmark individual operations
3. **Integration**: Full model inference validation
4. **Stress**: Large batch sizes, long sequences

### Benchmark Targets

Based on CPU baseline:
- Linear layers: 5-10x faster on GPU
- Attention: 3-8x faster on GPU
- Overall: 5-10x faster tokens/sec

## Files Modified for Profiling

1. `examples/benchmark_inference_profile.rs` - Profiling benchmark
2. `src/tensor.rs` - Added logging to matmul operations
3. `src/gpu/gemv.rs` - Detailed GPU operation timing
4. `src/gpu/mod.rs` - Enhanced GPU initialization logging
5. `PROFILING_ANALYSIS.md` - Detailed analysis document
6. `GPU_ROOT_CAUSE_ANALYSIS.md` - This summary

## Key Takeaways

1. **GPU was never used** - all operations fell back to CPU
2. **GEMV is insufficient** - need GEMM for transformer models
3. **Architecture mismatch** - GPU designed for vector ops, model uses matrix ops
4. **Easy fix** - implement GEMM kernel, huge performance gains expected
5. **Profiling works** - new infrastructure can verify GPU usage going forward

## Next Steps

1. Review GEMM implementation options (handwritten WGSL vs. library)
2. Decide on workgroup size and tiling strategy
3. Implement basic 2D GEMM kernel
4. Test and validate correctness
5. Benchmark and optimize
6. Expand to 4D batched operations

## References

- Profiling output: (see stderr logs above)
- GEMV implementation: `src/gpu/gemv.rs`
- Matmul dispatch: `src/tensor.rs:778`
- Model architecture: `src/model.rs`, `src/attention.rs`, `src/mlp.rs`
