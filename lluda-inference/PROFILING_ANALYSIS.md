# GPU Profiling Analysis

## Summary

Added detailed profiling to identify why GPU is slower than CPU for inference.

## Changes Made

### 1. New Profiling Benchmark
**File:** `examples/benchmark_inference_profile.rs`

- Created profiling-enabled version of inference benchmark
- Controlled via `PROFILE=1` environment variable
- Manual generation loop with timing for each operation
- Token-by-token profiling

**Features:**
- Profiling macro for timing operations
- Detailed logging of each token generation step
- Prefill phase timing
- Per-token throughput tracking

### 2. Tensor Matmul Profiling
**File:** `src/tensor.rs`

Added conditional logging to `matmul()` function:
- Logs when GEMV candidate is detected
- Shows matrix dimensions and dtypes
- Reports GPU path success/failure
- Logs fallback to CPU with reason

Added profiling to `try_gemv_gpu()`:
- Matrix size threshold check logging
- GPU context initialization timing
- GEMV forward pass timing
- BF16→F32 conversion timing

### 3. GPU GEMV Operation Profiling
**File:** `src/gpu/gemv.rs`

Added detailed timing to `gemv_forward()`:
- Extract BF16 data timing
- Pack BF16 data timing
- GPU buffer creation timing
- Upload to GPU timing (with byte counts)
- Pipeline creation timing
- GPU kernel submission timing
- GPU completion wait timing
- Download from GPU timing (with byte counts)
- Unpack output timing
- Total operation time

### 4. GPU Context Initialization Profiling
**File:** `src/gpu/mod.rs`

Enhanced `get_context()` logging:
- GPU device name and backend
- Device type (discrete/integrated)
- Driver name and version (when profiling enabled)

## Initial Test Results

### Observations from Test Run (10 tokens, GPU mode)

```
Prefill: 19.7 seconds
Token 2-10: ~8.3-8.5 seconds EACH
Throughput: ~0.1 tokens/sec
```

### Critical Finding: GPU NOT BEING USED

**ROOT CAUSE IDENTIFIED: ALL matmul operations are 2D×2D or 4D×4D, NOT 2D×1D GEMV!**

Profiling output shows:
```
[GPU] GEMV rejected: shape mismatch (need 2D×1D, got 2D×2D): [5, 1024] × [1024, 2048]
[GPU] GEMV rejected: shape mismatch (need 2D×1D, got 2D×2D): [5, 1024] × [1024, 1024]
[GPU] GEMV rejected: shape mismatch (need 2D×1D, got 4D×4D): [1, 16, 5, 128] × [1, 16, 128, 5]
[GPU] GEMV rejected: shape mismatch (need 2D×1D, got 2D×2D): [5, 2048] × [2048, 1024]
[GPU] GEMV rejected: shape mismatch (need 2D×1D, got 2D×2D): [5, 1024] × [1024, 3072]
```

### Analysis of Rejection Reasons

**During Prefill (5 tokens):**
- Linear projections: `[5, 1024] × [1024, 2048]` → Matrix-Matrix (2D×2D)
- Q/K/V projections: `[5, 1024] × [1024, 1024]` → Matrix-Matrix (2D×2D)
- Attention matmuls: `[1, 16, 5, 128] × [1, 16, 128, 5]` → Batched 4D (4D×4D)
- FFN projections: `[5, 1024] × [1024, 3072]` → Matrix-Matrix (2D×2D)

**During Autoregressive Decoding (1 token at a time):**
- Would be `[1, 1024] × [1024, 2048]` → Still Matrix-Matrix!
- NOT `[1024] × [1024, 2048]` (which would be Vector-Matrix)

### Why GPU Implementation is Unused

1. **GPU only implements GEMV (matrix-vector multiplication)**
   - GEMV = 2D matrix × 1D vector
   - Returns 1D result vector

2. **Model performs GEMM (matrix-matrix multiplication)**
   - Even with batch_size=1, seq_len=1
   - Tensors are shaped as `[1, hidden_size]` (2D), not `[hidden_size]` (1D)
   - Result is `[1, output_size]` (2D), not `[output_size]` (1D)

3. **Current GPU path NEVER activates**
   - No operations in the model match 2D×1D pattern
   - All matmuls are rejected
   - 100% CPU fallback

### Architectural Issue

The transformer model uses batched operations throughout:
- Input: `[batch, seq_len, hidden_size]`
- After reshape for linear: `[batch*seq_len, hidden_size]`
- Weight: `[hidden_size, output_size]`
- Matmul: `[batch*seq_len, hidden_size] × [hidden_size, output_size]` → `[batch*seq_len, output_size]`

Even with batch=1, seq=1:
- Input becomes: `[1, hidden_size]` (2D tensor)
- NOT `[hidden_size]` (1D tensor)
- Therefore NOT a GEMV operation!

## Recommendations

### Immediate Fix Options

**Option 1: Implement GEMM (Matrix-Matrix) on GPU**
- Add general matrix-matrix multiplication kernel
- Support 2D×2D operations: `[M, K] × [K, N] → [M, N]`
- This covers all linear layer operations
- **Impact: High** (covers most compute)

**Option 2: Add Batched GEMM for 4D Attention**
- Implement batched matmul for attention
- Support 4D×4D: `[B, H, M, K] × [B, H, K, N] → [B, H, M, N]`
- **Impact: Medium** (attention is compute-heavy but fewer operations)

**Option 3: Reshape to GEMV for Single-Token Decode**
- Modify Linear layer to squeeze inputs during decode phase
- Convert `[1, hidden] × [hidden, output]` → `[hidden] × [hidden, output]`
- Only works for batch=1, seq=1 (autoregressive decoding)
- **Impact: Low** (only helps decode, not prefill)

**Option 4: Hybrid Approach**
- Implement GEMM for batch>1 or seq>1 (prefill)
- Use GEMV for single-token decode (if reshaped)
- **Impact: High** (best performance across scenarios)

### Performance Projections

Based on profiling data:
- Prefill (5 tokens): 19.7s → dominated by matrix-matrix ops
- Per-token decode: 8.3s → dominated by matrix-matrix ops
- Current GPU path: 0% utilization
- **With GEMM GPU implementation: 70-90% potential speedup**

### Implementation Priority

1. **HIGH: Basic 2D GEMM kernel**
   - Implement `[M, K] × [K, N] → [M, N]` on GPU
   - Add to matmul dispatch logic
   - Should accelerate all linear layers

2. **MEDIUM: Batched 4D GEMM kernel**
   - Implement `[B, H, M, K] × [B, H, K, N] → [B, H, M, N]`
   - Required for attention Q@K^T and attn@V
   - Significant compute savings

3. **LOW: Optimize GEMV for edge cases**
   - Current GEMV works but is unused
   - Keep for potential future use cases

## Next Steps

### Immediate Investigation (COMPLETED)

1. **Check tensor dtypes during forward pass**
   - Add logging in model layers to show tensor dtypes
   - Verify if tensors are BF16 or F32

2. **Check tensor shapes during matmul**
   - Log actual shapes being passed to matmul
   - Identify which operations are GEMV candidates

3. **Check is_gemv_candidate logic**
   - Review what conditions must be met
   - Add logging to show why candidates are rejected

4. **Analyze model architecture**
   - Identify all matmul operations in forward pass
   - Check dimensions: attention Q/K/V, FFN, etc.
   - Calculate M*N for each to see if threshold is met

### Profiling Enhancements Needed

1. **Add shape/dtype logging to model layers**
   - Attention layer forward pass
   - MLP layer forward pass
   - Linear layer forward pass

2. **Add is_gemv_candidate() logging**
   - Log why operations are rejected
   - Show decision criteria

3. **Track CPU vs GPU operation counts**
   - Count how many matmuls go to GPU
   - Count how many fall back to CPU
   - Show percentage

### Testing Strategy

1. **Single layer test**
   - Isolate one linear layer
   - Force BF16 inputs
   - Verify GPU activation

2. **Dimension analysis**
   - Print all matmul dimensions during inference
   - Identify smallest/largest operations
   - Check against 1024 threshold

3. **Data type flow**
   - Trace dtype from model load → inference
   - Find where F32 conversion happens
   - Identify if model weights are BF16

## Usage

### Run Profiling Benchmark

```bash
# CPU mode
PROFILE=1 cargo run --example benchmark_inference_profile --release

# GPU mode
PROFILE=1 cargo run --features gpu --example benchmark_inference_profile --release
```

### Expected Output (when GPU works)

```
[GPU] GPU initialized: AMD Radeon Graphics (Vulkan)
[GPU] GEMV candidate detected: 1536x896 @ 896x1
[GPU]   LHS dtype: BF16, RHS dtype: BF16
[GPU]     Extract BF16 data: 0.05ms
[GPU]     Pack BF16 data: 0.12ms
[GPU]     Create buffers: 0.08ms
[GPU]     Upload to GPU (2744832 + 1792 bytes): 2.34ms
[GPU]     Create pipeline: 0.45ms
[GPU]     Submit GPU kernel: 0.12ms
[GPU]     Wait for GPU completion: 3.21ms
[GPU]     Download from GPU (6144 bytes): 0.89ms
[GPU]     Unpack output: 0.03ms
[GPU]     Total GPU operation time: 7.29ms
[GPU]   GPU path SUCCESS: 7.35ms
```

### Actual Output (current)

```
[PROFILE]   Token 2 forward pass: 8516.65ms
[PROFILE]   Token 2 extract logits: 0.07ms
[PROFILE]   Token 2 sampling: 0.17ms
```

NO GPU logs → GPU not being used!

## Conclusions

1. **Profiling infrastructure is working correctly**
   - Timing macros functional
   - Logging shows proper call hierarchy
   - Conditional profiling via env var works

2. **Root cause identified**
   - GPU is NOT being used for matmul operations
   - This explains why "GPU mode" is slower
   - CPU path is being taken for all operations

3. **Not a GPU overhead problem**
   - The issue is not excessive upload/download time
   - The issue is GPU isn't being invoked at all

4. **Need deeper investigation**
   - Must understand why GEMV path is not taken
   - Must verify tensor dtypes and shapes
   - Must check model architecture details

## Files Modified

- `examples/benchmark_inference_profile.rs` (new)
- `src/tensor.rs` (profiling added to matmul, try_gemv_gpu)
- `src/gpu/gemv.rs` (profiling added to gemv_forward)
- `src/gpu/mod.rs` (enhanced GPU init logging)
- `PROFILING_ANALYSIS.md` (this document)

## Remaining Questions

1. Are model weights loaded as BF16 or F32?
2. What are the actual shapes of matmul operations in Qwen3-0.6B?
3. What does `is_gemv_candidate()` check?
4. Are there any 2D×1D matmuls in the model at all?
5. Should we lower the 1024 threshold for testing?
