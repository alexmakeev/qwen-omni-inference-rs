# Phase 1: GPU Acceleration - AMD Strix Halo

**Status**: 🟡 In Progress (90% complete)
**Started**: 2026-02-05
**Goal**: GPU-accelerated inference on AMD RADV GFX1151

## Completed ✅

### T23-T26: GPU Infrastructure
- ✅ wgpu/Vulkan setup (GpuContext, device init)
- ✅ GPU buffer abstraction (CPU↔GPU transfers)
- ✅ GEMV shader (2D×1D matrix-vector)
- ✅ GEMV integration into Tensor::matmul()

### T27: GEMM Shader (2D×2D)
- ✅ Tiled GEMM shader (16×16 tiles, shared memory)
- ✅ GemmPipeline with compute dispatch
- ✅ Integration into Tensor::matmul()
- ✅ Tests passing: 5×1024 @ 1024×128 validated

### Architecture Fixes
- ✅ **Removed ALL GPU fallbacks** (strict GPU-or-fail policy)
- ✅ **BF16 native pipeline** (removed all BF16→F32 conversions)
- ✅ GPU Design Principles documented

### Validation on Luda (AMD RADV GFX1151)
- ✅ GPU initialization working
- ✅ GEMM dispatch: BF16×BF16 ✅
- ✅ GPU operations executing:
  - First GEMM: 15.68ms
  - Second GEMM: 2.39ms
  - Third GEMM: 1.62ms (cached!)

## Current Issue 🔴

**4D Batched Matmul Not Supported**

Attention operations use 4D tensors:
```
Q @ K^T: [1, 16, 5, 128] × [1, 16, 128, 5]  ← 4D×4D unsupported
attn @ V: [1, 16, 5, 5] × [1, 16, 5, 128]   ← 4D×4D unsupported
```

Error:
```
GPU mode: unsupported matmul shapes 4D×4D. 
Only 2D×2D (GEMM) and 2D×1D (GEMV) are supported on GPU.
```

## Next Task 🎯

**T28: Add 4D Batched Matmul Support**

### Solution: Reshape 4D→2D (Quick Win)

Instead of implementing full 4D shader, reshape to 2D:

```rust
// Input: [batch, heads, seq, dim]
// Reshape: [batch*heads, seq, dim]
// Use existing 2D GEMM!
// Reshape back: [batch, heads, seq, dim]

Example:
[1, 16, 5, 128] → [16, 5, 128]  // merge batch*heads
[16, 5, 128] @ [16, 128, 5]     // 2D GEMM works!
[16, 5, 5] → [1, 16, 5, 5]      // reshape back
```

### Implementation Plan

1. **Add reshape helper in src/tensor.rs:**
   - `reshape_4d_to_2d()`: merge batch*heads dimension
   - `reshape_2d_to_4d()`: restore original shape

2. **Update matmul_gpu() for 4D case:**
   ```rust
   if self.ndim() == 4 && rhs.ndim() == 4 {
       return self.matmul_4d_as_2d(rhs);  // reshape + 2D GEMM
   }
   ```

3. **Test on attention operations:**
   - Validate Q@K^T and attn@V
   - Compare with CPU reference

4. **Run full benchmark:**
   - Complete 50-token generation
   - Measure tokens/sec on Luda
   - Compare CPU vs GPU speedup

### Expected Result

- Full inference working on GPU
- 5-10x speedup over CPU
- All operations in BF16
- No 4D shader needed (use 2D GEMM)

## Git Commits

1. `17f720a` - T23: wgpu infrastructure
2. `62f4d4a` - T24: GPU buffers
3. `1edf3aa` - T25: GEMV shader
4. `14130ac` - T26: GEMV integration
5. `95c68ce` - Profiling & root cause
6. `7083d91` - T27: GEMM shader
7. `65936fd` - Remove GPU fallbacks
8. `3494cc7` - BF16 native pipeline

## Files to Review

- `src/gpu/shaders/gemm.wgsl` - GEMM shader
- `src/gpu/gemm.rs` - GEMM pipeline
- `src/tensor.rs` - matmul_gpu() dispatch
- `GPU_DESIGN_PRINCIPLES.md` - No fallbacks policy
- `PROFILING_ANALYSIS.md` - Root cause analysis

## Resuming Work

When returning to this task:
1. Implement reshape_4d_to_2d helper
2. Add 4D case to matmul_gpu()
3. Test with attention operations
4. Run full benchmark on Luda
5. Measure final speedup

**Time estimate**: 1-2 hours to complete
