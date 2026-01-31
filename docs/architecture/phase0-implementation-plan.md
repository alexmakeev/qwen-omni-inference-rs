# Phase 0: Qwen3-0.6B Infrastructure — Implementation Plan

**Date:** 2026-01-31
**Status:** Pending Approval
**Parent:** arch-2026-01-31-qwen3-bf16-first.md (Option C modified: NO candle-core)

## Overview

Build a pure Rust reference implementation of Qwen3-0.6B inference from scratch.
No candle-core, no candle-nn, no external ML framework dependencies.
Every tensor operation implemented manually for full control over future quantization and GPU migration.

**Model:** Qwen3-0.6B (Qwen3ForCausalLM, 1.4GB BF16)
**Validation:** Compare logits token-by-token against HuggingFace Transformers (Python)
**Deliverable:** Text generation matching Python output exactly

---

## Architecture: Module Map

```
lluda-inference/
  Cargo.toml
  src/
    lib.rs                    -- Public API, re-exports
    error.rs                  -- LludaError, Result<T> type alias

    # Tensor infrastructure (NO candle-core)
    tensor/
      mod.rs                  -- Tensor struct, DType, Shape
      storage.rs              -- TensorStorage: Vec<u8> + dtype dispatch
      ops.rs                  -- Core ops: matmul, add, mul, softmax, etc.
      bf16.rs                 -- BF16 <-> F32 conversion, bf16 type
      view.rs                 -- Tensor views, reshape, transpose, narrow, contiguous

    # Model components
    model/
      mod.rs                  -- Qwen3Model, Qwen3ForCausalLM
      config.rs               -- ModelConfig (parsed from HF config.json)
      embedding.rs            -- Token embedding lookup
      rms_norm.rs             -- RMSNorm (pre-norm)
      rope.rs                 -- Rotary Position Embeddings
      attention.rs            -- Grouped Query Attention (GQA) with q_norm/k_norm
      mlp.rs                  -- SiLU gated MLP (gate_proj, up_proj, down_proj)
      transformer.rs          -- DecoderLayer = attention + mlp + residuals
      kv_cache.rs             -- KV cache for autoregressive generation
      causal_mask.rs          -- Causal attention mask generation

    # Weight loading
    weights/
      mod.rs                  -- WeightLoader trait
      safetensors.rs          -- SafeTensors mmap loader
      tensor_map.rs           -- HF tensor name -> internal name mapping

    # Tokenizer
    tokenizer.rs              -- HF tokenizer-rs wrapper

    # Generation
    generate.rs               -- Autoregressive generation loop
    sampling.rs               -- Greedy, temperature, top-k, top-p

  # Examples
  examples/
    generate.rs               -- Text generation CLI
    validate.rs               -- Compare against Python reference data

  # Tests
  tests/
    tensor_ops.rs             -- Unit tests for tensor operations
    model_components.rs       -- Unit tests for model components
    integration.rs            -- Full model forward pass tests
```

**Dependencies (Cargo.toml):**
```toml
[dependencies]
safetensors = "0.4"           # SafeTensors file format
memmap2 = "0.9"               # Memory-mapped file access
tokenizers = { version = "0.21", default-features = false, features = ["onig"] }
half = { version = "2.4", features = ["num-traits", "bytemuck"] }   # BF16/F16 types
bytemuck = { version = "1", features = ["derive"] }                 # Zero-copy type punning
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"               # Error derive macro

[dev-dependencies]
ndarray = "0.16"              # For loading Python .npz reference data
ndarray-npy = "0.9"           # NumPy file loading
approx = "0.5"                # Approximate float comparison
```

---

## Task Breakdown

### Dependency Graph

```
                    T01 (error.rs)
                    T02 (bf16)
                         │
                    T03 (tensor core)
                    ╱    │    ╲
              T04       T05       T06
           (matmul)  (elementwise) (view ops)
                ╲       │       ╱
                    T07 (softmax + advanced ops)
                         │
          ┌──────┬───────┼───────┬──────────┐
         T08    T09     T10     T11        T12
      (config) (safetensors) (tokenizer) (embed) (rms_norm)
          │      │              │          │        │
          └──────┴──────────────┴──────────┴────────┘
                         │
                  T13 (rope)
                  T14 (attention)
                  T15 (mlp)
                         │
                  T16 (transformer block)
                  T17 (kv_cache)
                  T18 (causal_mask)
                         │
                  T19 (full model assembly)
                         │
               ┌─────────┴─────────┐
              T20                  T21
         (generation loop)   (Python reference extraction)
               │                   │
              T22 (validation: Rust vs Python)
```

**Parallelism opportunities:**
- T01 + T02: independent, run first
- T04 + T05 + T06: independent after T03
- T08 + T09 + T10 + T11 + T12: independent after T07
- T13 + T14 + T15: partially parallel (T14 depends on T13)
- T20 + T21: fully parallel

---

### T01: Error Types

**Module:** `src/error.rs`
**Complexity:** Simple
**Depends on:** Nothing

**Specification:**
```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LludaError {
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },

    #[error("DType mismatch: expected {expected}, got {got}")]
    DTypeMismatch { expected: String, got: String },

    #[error("Dimension out of range: {dim} for tensor with {ndim} dimensions")]
    DimOutOfRange { dim: usize, ndim: usize },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("SafeTensors error: {0}")]
    SafeTensors(String),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("{0}")]
    Msg(String),
}

pub type Result<T> = std::result::Result<T, LludaError>;
```

**Unit tests:**
- Error Display formatting
- Error conversion from std::io::Error, serde_json::Error

**Acceptance criteria:**
- All error variants constructible
- `?` operator works with std::io and serde_json errors
- Error messages are human-readable

---

### T02: BF16 Support

**Module:** `src/tensor/bf16.rs`
**Complexity:** Simple
**Depends on:** T01

**Specification:**
Use `half::bf16` type. Provide conversion functions:
```rust
/// Convert BF16 slice to F32 slice (for compute)
pub fn bf16_to_f32(src: &[bf16]) -> Vec<f32>;

/// Convert F32 slice to BF16 slice (for storage)
pub fn f32_to_bf16(src: &[f32]) -> Vec<bf16>;

/// Convert single BF16 to F32 (inline, zero-cost via bitshift)
#[inline(always)]
pub fn bf16_to_f32_single(v: bf16) -> f32;
```

**Key insight:** BF16 has no hardware compute on x86. All computation happens in F32. BF16 is storage-only (halves memory bandwidth for weight loading). The `half` crate handles the bit manipulation correctly.

**Unit tests:**
- Round-trip: f32 -> bf16 -> f32 preserves value within BF16 precision
- Known values: 1.0, -1.0, 0.0, 0.5, NaN, Inf
- Large batch conversion correctness
- BF16 range limits (max ~3.39e38, min subnormal ~9.18e-41)

**Acceptance criteria:**
- Conversions match Python `torch.bfloat16` behavior exactly
- Performance: bulk conversion at memory bandwidth (not compute-bound)

---

### T03: Tensor Core

**Module:** `src/tensor/mod.rs`, `src/tensor/storage.rs`
**Complexity:** Hard
**Depends on:** T01, T02

**Specification:**
```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DType {
    F32,
    BF16,
}

#[derive(Debug, Clone)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    pub fn numel(&self) -> usize;      // Total number of elements
    pub fn ndim(&self) -> usize;       // Number of dimensions
    pub fn strides(&self) -> Vec<usize>; // Row-major strides
}

/// Core tensor type. Stores data in a flat buffer with shape metadata.
/// All compute happens in F32. BF16 is converted on-read.
#[derive(Debug, Clone)]
pub struct Tensor {
    storage: TensorStorage,    // Owned or borrowed data
    shape: Shape,
    strides: Vec<usize>,       // For non-contiguous views
    offset: usize,             // Byte offset into storage
    dtype: DType,
}

/// Storage backends
pub enum TensorStorage {
    Owned(Vec<u8>),            // Owned heap allocation
    Mmap(Arc<memmap2::Mmap>),  // Memory-mapped (for weights)
}
```

**Key methods on Tensor:**
```rust
impl Tensor {
    // Construction
    pub fn zeros(shape: &[usize], dtype: DType) -> Result<Self>;
    pub fn from_f32_vec(data: Vec<f32>, shape: &[usize]) -> Result<Self>;
    pub fn from_bf16_slice(data: &[bf16], shape: &[usize]) -> Result<Self>;

    // Properties
    pub fn shape(&self) -> &Shape;
    pub fn dtype(&self) -> DType;
    pub fn is_contiguous(&self) -> bool;

    // Data access (always returns F32 for compute)
    pub fn to_f32_vec(&self) -> Result<Vec<f32>>;

    // Type conversion
    pub fn to_dtype(&self, dtype: DType) -> Result<Self>;

    // Contiguous copy (resolves strides)
    pub fn contiguous(&self) -> Result<Self>;
}
```

**Unit tests:**
- Create tensor from Vec<f32>, verify shape/strides/data
- Create tensor from BF16, read back as F32
- zeros() for all dtypes
- Shape::numel for various dims (scalar, 1D, 2D, 3D, 4D)
- Contiguity detection for fresh tensors (should be contiguous)
- Clone preserves data independently

**Acceptance criteria:**
- Tensor correctly stores F32 and BF16 data
- to_f32_vec() always returns correct F32 values regardless of internal dtype
- Memory layout is row-major (C-contiguous) by default
- Shape validation: reject zero-dimensional shapes, mismatched numel

---

### T04: Matrix Multiplication

**Module:** `src/tensor/ops.rs` (matmul section)
**Complexity:** Hard
**Depends on:** T03

**Specification:**
```rust
impl Tensor {
    /// General matrix multiplication: C = A @ B
    /// Supports: [M, K] @ [K, N] -> [M, N]
    /// Supports: [B, M, K] @ [B, K, N] -> [B, M, N] (batched)
    /// Supports: [B, H, M, K] @ [B, H, K, N] -> [B, H, M, N] (4D batched)
    pub fn matmul(&self, rhs: &Tensor) -> Result<Tensor>;
}
```

**Implementation strategy:**
- Convert both operands to F32 before compute
- Naive triple-loop for correctness first
- Mark with `// TODO: BLAS/optimized kernel` for Phase 3
- BF16 weights: load as F32, multiply, store result as F32 (activations stay F32)

**Important:** The naive implementation is sufficient for Phase 0. Qwen3-0.6B has only 0.6B params. On a modern CPU, even naive matmul runs the full model in seconds. Performance optimization is Phase 3.

**Unit tests:**
- 2x2 matmul: known result
- 3x4 @ 4x5 = 3x5: verify against hand-calculated values
- Batched: [2, 3, 4] @ [2, 4, 5] = [2, 3, 5]
- 4D: [1, 16, 10, 64] @ [1, 16, 64, 10] = [1, 16, 10, 10] (attention score shape)
- Identity: A @ I = A
- Zero matrix: A @ 0 = 0
- BF16 input: matmul with BF16 tensors, verify F32 result matches F32-only matmul
- Shape mismatch: error on incompatible dimensions

**Acceptance criteria:**
- Correct results for all supported shapes
- BF16 inputs produce same results as F32 (within BF16 precision)
- Clear error messages for shape mismatches
- Batched dimensions broadcast correctly

**References:**
- NumPy matmul broadcasting rules
- existing qwen3.rs line 225: `q.matmul(&k.transpose(2, 3)?.contiguous()?)`

---

### T05: Element-wise Operations

**Module:** `src/tensor/ops.rs` (element-wise section)
**Complexity:** Medium
**Depends on:** T03

**Specification:**
```rust
impl Tensor {
    // Arithmetic (element-wise, with broadcasting)
    pub fn add(&self, rhs: &Tensor) -> Result<Tensor>;
    pub fn mul(&self, rhs: &Tensor) -> Result<Tensor>;
    pub fn sub(&self, rhs: &Tensor) -> Result<Tensor>;

    // Scalar ops
    pub fn mul_scalar(&self, s: f32) -> Result<Tensor>;
    pub fn add_scalar(&self, s: f32) -> Result<Tensor>;

    // Activation functions
    pub fn silu(&self) -> Result<Tensor>;   // x * sigmoid(x) = x / (1 + exp(-x))
    pub fn exp(&self) -> Result<Tensor>;

    // Reduction
    pub fn sum(&self, dim: usize) -> Result<Tensor>;
    pub fn max(&self, dim: usize) -> Result<Tensor>;
    pub fn mean(&self, dim: usize) -> Result<Tensor>;
}
```

**Broadcasting rules:** Follow NumPy broadcasting. Dimensions are aligned from the right. Size-1 dimensions broadcast to match the other tensor.

**Unit tests:**
- add/mul/sub: same shape, verify element-wise
- Broadcasting: [3, 4] + [1, 4] = [3, 4] (broadcast dim 0)
- Broadcasting: [3, 4] + [4] = [3, 4] (broadcast missing dim)
- Scalar ops: mul_scalar, add_scalar
- SiLU: known values (silu(0) = 0, silu(1) ~= 0.7311, silu(-1) ~= -0.2689)
- sum/max/mean along specific dimensions, verify shape and values
- Shape mismatch: non-broadcastable shapes produce error

**Acceptance criteria:**
- Broadcasting matches NumPy behavior
- SiLU matches `torch.nn.functional.silu` within 1e-6

---

### T06: View Operations

**Module:** `src/tensor/view.rs`
**Complexity:** Medium
**Depends on:** T03

**Specification:**
```rust
impl Tensor {
    /// Reshape without copying data (if contiguous)
    pub fn reshape(&self, shape: &[usize]) -> Result<Tensor>;

    /// Transpose two dimensions (returns a view, changes strides)
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Tensor>;

    /// Select a sub-range along a dimension
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Tensor>;

    /// Flatten dimensions [start..=end] into one
    pub fn flatten(&self, start_dim: usize, end_dim: usize) -> Result<Tensor>;

    /// Concatenate tensors along a dimension
    pub fn cat(tensors: &[&Tensor], dim: usize) -> Result<Tensor>;

    /// Broadcast-add (for attention mask)
    pub fn broadcast_add(&self, rhs: &Tensor) -> Result<Tensor>;

    /// Embedding lookup: self is [vocab, dim], indices is [B, L] -> [B, L, dim]
    pub fn embedding(&self, indices: &Tensor) -> Result<Tensor>;
}
```

**Critical for attention:**
- `reshape((b, l, num_heads, head_dim))` — split heads
- `transpose(1, 2)` — (B, L, H, D) -> (B, H, L, D)
- `narrow(1, l-1, 1)` — select last token logits
- `flatten(0, 2)` — for per-head RMSNorm: (B, H, L, D) -> (B*H*L, D)
- `cat` — for KV cache concatenation

**Unit tests:**
- reshape: [12] -> [3, 4], verify data order
- reshape: [2, 3, 4] -> [6, 4], verify
- transpose: [2, 3] -> [3, 2], verify values at each position
- transpose: [B, H, L, D] -> [B, H, D, L] (attention K transpose)
- narrow: select middle rows
- flatten: [2, 3, 4] flatten(0,1) -> [6, 4]
- cat: along dim 0 and dim 2 (KV cache append)
- embedding: lookup known indices, verify correct rows returned
- reshape non-contiguous tensor: should auto-contiguous or error

**Acceptance criteria:**
- reshape is zero-copy when input is contiguous
- transpose creates a view (no data copy), changes strides
- narrow creates a view (no data copy)
- contiguous() after transpose copies data into new contiguous layout
- All operations match PyTorch behavior

---

### T07: Softmax and Advanced Ops

**Module:** `src/tensor/ops.rs` (softmax section)
**Complexity:** Medium
**Depends on:** T04, T05, T06

**Specification:**
```rust
impl Tensor {
    /// Numerically stable softmax along last dimension
    /// softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    pub fn softmax_last_dim(&self) -> Result<Tensor>;
}
```

**Implementation:** Must be numerically stable (subtract max before exp). This is critical for attention scores which can have values of -inf (from causal mask).

**Unit tests:**
- softmax([1, 2, 3]): verify sum = 1.0, monotonically increasing
- softmax with -inf: masked positions get probability 0.0
- softmax([0, 0, 0]): uniform distribution (1/3 each)
- Large values: softmax([1000, 1001, 1002]) should not overflow
- 4D tensor softmax: [1, 16, 10, 10] (attention shape), verify last dim sums to 1

**Acceptance criteria:**
- Output sums to 1.0 along last dim (within 1e-6)
- Handles -inf correctly (masked attention)
- No NaN or overflow for large/small inputs
- Matches `torch.nn.functional.softmax(x, dim=-1)` within 1e-6

---

### T08: Model Config

**Module:** `src/model/config.rs`
**Complexity:** Simple
**Depends on:** T01

**Specification:**
```rust
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,            // 151936
    pub hidden_size: usize,           // 1024
    pub intermediate_size: usize,     // 3072
    pub num_hidden_layers: usize,     // 28
    pub num_attention_heads: usize,   // 16
    pub num_key_value_heads: usize,   // 8
    pub head_dim: usize,              // 128
    pub max_position_embeddings: usize, // 40960
    pub rope_theta: f64,              // 1000000.0
    pub rms_norm_eps: f64,            // 1e-6
    pub hidden_act: String,           // "silu"
    pub attention_bias: bool,         // false
    pub tie_word_embeddings: bool,    // true
    pub bos_token_id: u32,            // 151643
    pub eos_token_id: u32,            // 151645
}

impl ModelConfig {
    /// Load from HF config.json
    pub fn from_file(path: &Path) -> Result<Self>;

    /// Derived: number of KV groups for GQA
    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}
```

**Unit tests:**
- Parse actual models/Qwen3-0.6B/config.json
- Verify all fields match expected values
- num_kv_groups() = 16 / 8 = 2
- Missing optional fields: graceful defaults
- Invalid JSON: proper error

**Acceptance criteria:**
- Successfully parses the real Qwen3-0.6B config.json
- All numerical values correct
- Serde deserialization handles snake_case field names

---

### T09: SafeTensors Loading

**Module:** `src/weights/safetensors.rs`, `src/weights/tensor_map.rs`
**Complexity:** Medium
**Depends on:** T03

**Specification:**
```rust
pub struct SafeTensorsLoader {
    mmap: Arc<memmap2::Mmap>,
    metadata: SafeTensorsMetadata,
}

impl SafeTensorsLoader {
    /// Open a SafeTensors file with memory mapping
    pub fn open(path: &Path) -> Result<Self>;

    /// List all tensor names in the file
    pub fn tensor_names(&self) -> Vec<&str>;

    /// Load a single tensor by name as our Tensor type
    pub fn load_tensor(&self, name: &str) -> Result<Tensor>;

    /// Load all tensors matching a prefix
    pub fn load_prefix(&self, prefix: &str) -> Result<HashMap<String, Tensor>>;
}
```

**Tensor name mapping for Qwen3-0.6B:**
```
model.embed_tokens.weight                          -> [151936, 1024] BF16
model.layers.{i}.self_attn.q_proj.weight           -> [2048, 1024] BF16  (16 heads * 128 dim)
model.layers.{i}.self_attn.k_proj.weight           -> [1024, 1024] BF16  (8 kv_heads * 128 dim)
model.layers.{i}.self_attn.v_proj.weight           -> [1024, 1024] BF16
model.layers.{i}.self_attn.o_proj.weight           -> [1024, 2048] BF16
model.layers.{i}.self_attn.q_norm.weight           -> [128] BF16
model.layers.{i}.self_attn.k_norm.weight           -> [128] BF16
model.layers.{i}.mlp.gate_proj.weight              -> [3072, 1024] BF16
model.layers.{i}.mlp.up_proj.weight                -> [3072, 1024] BF16
model.layers.{i}.mlp.down_proj.weight              -> [1024, 3072] BF16
model.layers.{i}.input_layernorm.weight            -> [1024] BF16
model.layers.{i}.post_attention_layernorm.weight   -> [1024] BF16
model.norm.weight                                  -> [1024] BF16
```

**Note:** `tie_word_embeddings: true` means there is NO `lm_head.weight`. The embedding weight is reused as the LM head.

**Unit tests:**
- Open models/Qwen3-0.6B/model.safetensors (skip if not present)
- List all tensor names, verify expected count
- Load embed_tokens.weight: shape [151936, 1024], dtype BF16
- Load layer 0 q_proj: shape [2048, 1024]
- Load q_norm: shape [128]
- Verify mmap: file stays open, no full copy to RAM
- Non-existent tensor name: proper error

**Acceptance criteria:**
- All tensors loadable from Qwen3-0.6B model.safetensors
- Shapes match expected values from config
- Memory-mapped: loading 1.4GB file doesn't allocate 1.4GB heap
- BF16 data readable as Tensor with DType::BF16

**References:**
- safetensors crate docs: https://docs.rs/safetensors
- HuggingFace SafeTensors format: https://huggingface.co/docs/safetensors

---

### T10: Tokenizer Integration

**Module:** `src/tokenizer.rs`
**Complexity:** Simple
**Depends on:** T01

**Specification:**
```rust
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    bos_token_id: u32,
    eos_token_id: u32,
}

impl Tokenizer {
    /// Load HF tokenizer.json
    pub fn from_file(path: &Path) -> Result<Self>;

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<u32>>;

    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32]) -> Result<String>;

    /// Apply chat template (Qwen3 format)
    pub fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String>;

    pub fn bos_token_id(&self) -> u32;
    pub fn eos_token_id(&self) -> u32;
}

pub struct ChatMessage {
    pub role: String,    // "system", "user", "assistant"
    pub content: String,
}
```

**Chat template for Qwen3:**
```
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
```

**Unit tests:**
- Load models/Qwen3-0.6B/tokenizer.json
- Encode "Hello": verify non-empty token list
- Decode(Encode("Hello world")) = "Hello world" (round-trip)
- BOS/EOS token IDs match config (151643, 151645)
- Chat template produces expected format
- Empty string: produces empty or single-token output
- Unicode: Chinese characters, emoji

**Acceptance criteria:**
- Token IDs match Python `transformers.AutoTokenizer`
- Chat template matches HuggingFace format
- No panics on any valid UTF-8 input

---

### T11: Token Embedding

**Module:** `src/model/embedding.rs`
**Complexity:** Simple
**Depends on:** T03, T09

**Specification:**
```rust
pub struct Embedding {
    weight: Tensor,  // [vocab_size, hidden_size]
}

impl Embedding {
    /// Create from loaded weight tensor
    pub fn new(weight: Tensor) -> Self;

    /// Lookup: indices [B, L] -> embeddings [B, L, hidden_size]
    /// Compute in F32 regardless of weight dtype
    pub fn forward(&self, indices: &[u32], batch_shape: &[usize]) -> Result<Tensor>;

    /// For tie_word_embeddings: return reference to weight
    pub fn weight(&self) -> &Tensor;
}
```

**Unit tests:**
- Create embedding with small weight [10, 4]
- Lookup single index: verify correct row returned
- Lookup batch [2, 3]: verify shape [2, 3, 4] and correct values
- Out of range index: error
- BF16 weight: result should be F32

**Acceptance criteria:**
- Lookup values match `torch.nn.Embedding` output
- BF16 -> F32 conversion automatic and correct

---

### T12: RMSNorm

**Module:** `src/model/rms_norm.rs`
**Complexity:** Simple
**Depends on:** T03, T05

**Specification:**
```rust
pub struct RmsNorm {
    weight: Tensor,   // [hidden_size] or [head_dim]
    eps: f64,
}

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self;

    /// RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * weight
    /// Input: [..., dim], weight: [dim]
    /// Output: same shape as input
    pub fn forward(&self, x: &Tensor) -> Result<Tensor>;
}
```

**Two uses in Qwen3-0.6B:**
1. Layer norm: weight shape [1024], applied to [B, L, 1024]
2. Per-head q/k norm: weight shape [128], applied to [B*H, L, 128] (after flatten)

**Unit tests:**
- RMSNorm([1, 1, 1], weight=[1,1,1], eps=1e-6): should be ~[1, 1, 1]
- RMSNorm([2, 0, 0], weight=[1,1,1], eps=1e-6): verify normalization
- Compare against `torch.nn.RMSNorm` for random inputs
- Different shapes: [B, L, D] and [B*H*L, D]
- eps effect: very small values don't produce NaN

**Acceptance criteria:**
- Matches PyTorch `torch.nn.RMSNorm` within 1e-5
- Handles zero vectors (eps prevents division by zero)
- Works for both 3D and 2D inputs

---

### T13: Rotary Position Embeddings (RoPE)

**Module:** `src/model/rope.rs`
**Complexity:** Medium
**Depends on:** T03, T05, T06

**Specification:**
```rust
pub struct RotaryEmbedding {
    cos: Tensor,  // [max_seq_len, head_dim]
    sin: Tensor,  // [max_seq_len, head_dim]
}

impl RotaryEmbedding {
    /// Precompute cos/sin tables
    /// inv_freq[i] = 1.0 / (theta ^ (2i / head_dim))  for i in 0..head_dim/2
    /// freqs[pos, i] = pos * inv_freq[i]
    /// cos/sin tables: [max_seq_len, head_dim/2]
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f64) -> Result<Self>;

    /// Apply RoPE to q and k tensors
    /// Input shapes: q [B, H, L, D], k [B, Hkv, L, D]
    /// Uses the "rotate_half" method:
    ///   q_rot = q * cos + rotate_half(q) * sin
    /// where rotate_half splits D into two halves and negates the first
    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)>;
}
```

**Implementation detail (from candle_nn::rotary_emb::rope):**
The `rope` function in candle uses the interleaved approach:
- Split q into [q0, q1] along last dim (first half, second half)
- rotate_half(q) = [-q1, q0]
- q_rotated = q * cos + rotate_half(q) * sin

**CRITICAL:** The reference `qwen3.rs` uses `candle_nn::rotary_emb::rope()`. The `thinker.rs` uses a different manual implementation. Phase 0 MUST match the `candle_nn::rotary_emb::rope()` behavior since that's what the validated Qwen3 dense model uses.

**Unit tests:**
- RoPE at position 0: cos(0)=1, sin(0)=0, so output = input (identity)
- RoPE preserves tensor norms (rotation is norm-preserving)
- Compare against Python HuggingFace RoPE implementation for known inputs
- Offset parameter: position 5 with offset=3 should use position 8's cos/sin
- head_dim=128, theta=1000000.0 (actual Qwen3-0.6B config)
- Verify cos/sin table values at specific positions against Python

**Acceptance criteria:**
- Matches `candle_nn::rotary_emb::rope()` output exactly (within 1e-5)
- Matches Python HuggingFace `Qwen3RotaryEmbedding` output within 1e-5
- cos/sin tables precomputed only once (not per forward pass)

**References:**
- RoFormer paper: https://arxiv.org/abs/2104.09864
- candle rotary_emb source: old/candle-16b/candle-nn/src/rotary_emb.rs
- qwen3.rs lines 30-64: Qwen3RotaryEmbedding implementation

---

### T14: Grouped Query Attention (GQA)

**Module:** `src/model/attention.rs`
**Complexity:** Hard
**Depends on:** T04, T06, T07, T12, T13

**Specification:**
```rust
pub struct Attention {
    q_proj: Linear,     // [hidden_size, num_heads * head_dim]
    k_proj: Linear,     // [hidden_size, num_kv_heads * head_dim]
    v_proj: Linear,     // [hidden_size, num_kv_heads * head_dim]
    o_proj: Linear,     // [num_heads * head_dim, hidden_size]
    q_norm: RmsNorm,    // [head_dim]
    k_norm: RmsNorm,    // [head_dim]
    rotary: Arc<RotaryEmbedding>,
    num_heads: usize,       // 16
    num_kv_heads: usize,    // 8
    num_kv_groups: usize,   // 2
    head_dim: usize,        // 128
}

/// Simple linear layer (weight @ input, no bias for Qwen3)
pub struct Linear {
    weight: Tensor,  // [out_features, in_features]
}

impl Linear {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor>;
    // Computes: x @ weight.T  (note: weight stored transposed as in PyTorch)
}

impl Attention {
    pub fn forward(
        &self,
        x: &Tensor,                    // [B, L, hidden_size]
        mask: Option<&Tensor>,         // [B, 1, L, L+offset]
        kv_cache: &mut KvCache,
        offset: usize,
    ) -> Result<Tensor>;               // [B, L, hidden_size]
}
```

**Forward pass (from qwen3.rs reference):**
1. Project: q = q_proj(x), k = k_proj(x), v = v_proj(x)
2. Reshape: q -> [B, L, H, D] -> [B, H, L, D]; k,v -> [B, L, Hkv, D] -> [B, Hkv, L, D]
3. Per-head RMSNorm: flatten to [B*H, L, D], apply q_norm/k_norm, reshape back
4. RoPE: apply rotary embeddings to q and k
5. KV cache: append k,v to cache; get full k,v
6. GQA repeat: repeat k,v from Hkv to H heads (repeat_kv)
7. Attention: scores = (q @ k^T) / sqrt(head_dim); add mask; softmax; scores @ v
8. Output: transpose, reshape to [B, L, hidden_size], apply o_proj

**GQA repeat_kv:**
```rust
/// Repeat KV heads: [B, Hkv, L, D] -> [B, H, L, D]
/// For Qwen3-0.6B: Hkv=8, H=16, groups=2
/// Each KV head is used by 2 query heads
fn repeat_kv(x: &Tensor, groups: usize) -> Result<Tensor>;
```

**Unit tests:**
- Linear layer: verify x @ w.T for known small matrices
- Full attention forward with random small tensors (B=1, L=4, H=2, Hkv=1, D=4)
- GQA: verify repeat_kv produces correct expanded tensor
- Per-head RMSNorm: verify norm is applied per-head, not globally
- Attention scores: verify causal masking (upper triangle = -inf -> 0 after softmax)
- Shape invariant: input [B, L, D] -> output [B, L, D]
- With KV cache: sequential decoding (L=1 at each step)

**Acceptance criteria:**
- Output matches candle qwen3.rs Qwen3Attention.forward() within 1e-4
- KV cache grows correctly with each step
- Per-head q_norm/k_norm applied correctly (this is unique to Qwen3!)
- GQA correctly maps 8 KV heads to 16 query heads

**References:**
- qwen3.rs lines 93-241: Full Qwen3Attention implementation
- GQA paper: https://arxiv.org/abs/2305.13245

---

### T15: SiLU Gated MLP

**Module:** `src/model/mlp.rs`
**Complexity:** Simple
**Depends on:** T04, T05

**Specification:**
```rust
pub struct MLP {
    gate_proj: Linear,  // [hidden_size, intermediate_size] = [1024, 3072]
    up_proj: Linear,    // [hidden_size, intermediate_size] = [1024, 3072]
    down_proj: Linear,  // [intermediate_size, hidden_size] = [3072, 1024]
}

impl MLP {
    /// forward(x) = down_proj(silu(gate_proj(x)) * up_proj(x))
    pub fn forward(&self, x: &Tensor) -> Result<Tensor>;
}
```

**Unit tests:**
- Small MLP (4 -> 8 -> 4): verify output shape
- Compare against PyTorch Qwen3MLP for random inputs
- Verify SiLU activation is applied to gate_proj output only (not up_proj)
- Shape invariant: input [..., hidden_size] -> output [..., hidden_size]

**Acceptance criteria:**
- Matches candle qwen3.rs Qwen3MLP within 1e-5
- Element-wise gating: gate * up, not gate + up

---

### T16: Transformer Block

**Module:** `src/model/transformer.rs`
**Complexity:** Medium
**Depends on:** T12, T14, T15

**Specification:**
```rust
pub struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    /// Pre-norm transformer block:
    /// h = x + attention(layer_norm(x))
    /// out = h + mlp(layer_norm(h))
    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        kv_cache: &mut KvCache,
        offset: usize,
    ) -> Result<Tensor>;
}
```

**Unit tests:**
- Single layer forward: verify output shape equals input shape
- Residual connection: output = input + something (not just something)
- Pre-norm: norm is applied before attention/MLP, not after
- Gradient-free: no NaN/Inf in output for random inputs

**Acceptance criteria:**
- Matches candle qwen3.rs DecoderLayer within 1e-4
- Residual connections preserve information

---

### T17: KV Cache

**Module:** `src/model/kv_cache.rs`
**Complexity:** Medium
**Depends on:** T03, T06

**Specification:**
```rust
pub struct KvCache {
    k: Option<Tensor>,  // [B, Hkv, seq_so_far, D]
    v: Option<Tensor>,  // [B, Hkv, seq_so_far, D]
}

impl KvCache {
    pub fn new() -> Self;

    /// Append new k, v and return full cached tensors
    /// First call: just stores k, v
    /// Subsequent calls: concatenates along seq dimension (dim=2)
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)>;

    /// Reset cache (for new conversation)
    pub fn reset(&mut self);

    /// Current sequence length in cache
    pub fn seq_len(&self) -> usize;
}
```

**Unit tests:**
- First append: cache stores [B, H, 1, D], returns same
- Second append: cache becomes [B, H, 2, D]
- After 10 appends: cache is [B, H, 10, D]
- Reset: cache is empty again
- Verify values are preserved (not overwritten)

**Acceptance criteria:**
- Concatenation along correct dimension (seq_len, dim=2)
- No data corruption on repeated appends
- Memory grows linearly with sequence length

---

### T18: Causal Attention Mask

**Module:** `src/model/causal_mask.rs`
**Complexity:** Simple
**Depends on:** T03

**Specification:**
```rust
/// Generate causal attention mask
/// For prompt (L > 1): lower-triangular mask, -inf above diagonal
/// For generation (L = 1): no mask needed (all past tokens visible)
///
/// Shape: [B, 1, L, L + offset]
/// Values: 0.0 (attend) or -inf (mask)
pub fn causal_mask(
    batch_size: usize,
    seq_len: usize,
    offset: usize,   // KV cache length (past tokens)
    dtype: DType,
) -> Result<Option<Tensor>>;
```

**Optimization:** When seq_len == 1 (generation), return None (no mask needed).

**Unit tests:**
- seq_len=4, offset=0: 4x4 lower triangular
- seq_len=1, offset=5: returns None
- seq_len=3, offset=2: 3x5 mask, first 2 columns all visible
- Verify -inf values in masked positions

**Acceptance criteria:**
- Matches candle qwen3.rs causal_mask() behavior
- Returns None for single-token generation (optimization)

---

### T19: Full Model Assembly

**Module:** `src/model/mod.rs`
**Complexity:** Hard
**Depends on:** T08, T09, T11, T16, T17, T18

**Specification:**
```rust
pub struct Qwen3Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    kv_caches: Vec<KvCache>,   // One per layer
}

pub struct Qwen3ForCausalLM {
    model: Qwen3Model,
    lm_head_weight: Tensor,    // Tied to embed_tokens.weight
}

impl Qwen3ForCausalLM {
    /// Load from config + SafeTensors
    pub fn load(config: &ModelConfig, weights: &SafeTensorsLoader) -> Result<Self>;

    /// Forward pass: token IDs -> logits for next token
    /// input: [B, L] token IDs
    /// offset: current position in sequence (for KV cache)
    /// Returns: [B, 1, vocab_size] logits (last token only)
    pub fn forward(&mut self, input: &[u32], offset: usize) -> Result<Tensor>;

    /// Reset all KV caches
    pub fn clear_kv_cache(&mut self);
}
```

**Load process:**
1. Parse config.json
2. Open SafeTensors file
3. Load embed_tokens.weight
4. For each layer (0..27): load all attention + MLP + norm weights
5. Load model.norm.weight
6. Set lm_head = embed_tokens.weight (tie_word_embeddings)

**Forward pass:**
1. Embed tokens: [B, L] -> [B, L, 1024]
2. Generate causal mask (if L > 1)
3. For each layer: h = layer.forward(h, mask, kv_cache[i], offset)
4. Final norm: h = rms_norm(h)
5. Last token only: h = h.narrow(1, L-1, 1) -> [B, 1, 1024]
6. LM head: logits = h @ embed_weight.T -> [B, 1, 151936]
7. Return logits

**Unit tests:**
- Load Qwen3-0.6B model (skip if files not present)
- Forward pass with single token: verify output shape [1, 1, 151936]
- Forward pass with prompt (4 tokens): verify shape
- Two sequential forward passes (simulating generation): verify KV cache grows
- Verify tie_word_embeddings: lm_head weight IS embed_tokens weight

**Acceptance criteria:**
- Model loads from disk in < 5 seconds (mmap)
- Forward pass produces valid logits (finite, reasonable range)
- Sequential generation works (KV cache accumulates)
- Memory usage: ~1.5GB (model weights) + minimal for activations

**References:**
- qwen3.rs lines 283-389: Model and ModelForCausalLM

---

### T20: Generation Loop

**Module:** `src/generate.rs`, `src/sampling.rs`
**Complexity:** Medium
**Depends on:** T10, T19

**Specification:**
```rust
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,       // 0.0 = greedy
    pub top_k: usize,           // 0 = disabled
    pub top_p: f32,             // 1.0 = disabled
    pub repetition_penalty: f32, // 1.0 = disabled
}

pub fn generate(
    model: &mut Qwen3ForCausalLM,
    tokenizer: &Tokenizer,
    prompt: &str,
    config: &GenerationConfig,
) -> Result<String>;

// Sampling
pub fn sample_greedy(logits: &Tensor) -> Result<u32>;
pub fn sample_temperature(logits: &Tensor, temp: f32) -> Result<u32>;
pub fn sample_top_k(logits: &Tensor, k: usize, temp: f32) -> Result<u32>;
pub fn sample_top_p(logits: &Tensor, p: f32, temp: f32) -> Result<u32>;
```

**Generation loop:**
1. Encode prompt with tokenizer
2. Forward pass with full prompt (prefill)
3. Sample next token from logits
4. Loop: forward(new_token, offset) -> sample -> append
5. Stop on EOS token or max_new_tokens

**Unit tests:**
- Greedy sampling: argmax of logits
- Temperature=0 equivalent to greedy
- Top-k: only top-k logits considered
- Top-p (nucleus): cumulative probability threshold
- Stop on EOS token
- Max tokens limit works
- Output is valid UTF-8 text

**Acceptance criteria:**
- Greedy decoding produces deterministic output
- EOS detection works correctly
- Token-by-token generation with KV cache reuse

---

### T21: Python Reference Data Extraction

**Module:** N/A (Python script)
**Complexity:** Medium
**Depends on:** Nothing (runs independently)

**Specification:**
Create `scripts/extract_reference.py`:
```python
"""
Extract reference tensors from Qwen3-0.6B using HuggingFace Transformers.
Saves intermediate activations for validation of Rust implementation.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load model and tokenizer
# 2. Run forward pass with test prompt
# 3. Extract and save:
#    - Token IDs (input)
#    - Embedding output [B, L, 1024]
#    - Per-layer hidden states (after each transformer block) [28 x [B, L, 1024]]
#    - Attention weights per layer (optional, large) [28 x [B, H, L, L]]
#    - RoPE cos/sin tables [max_seq, head_dim/2]
#    - Final norm output [B, L, 1024]
#    - Logits [B, L, vocab_size]  (full, not just last token)
#    - Generated token sequence (greedy, 20 tokens)
# 4. Save all to reference_data/ directory as .npy files
```

**Test prompts (deterministic):**
1. `"Hello"` — single token, simplest case
2. `"The capital of France is"` — multi-token, factual
3. Chat template: `[{"role": "user", "content": "What is 2+2?"}]`

**Extracted data format:**
```
reference_data/
  prompt1/
    input_ids.npy          # [1, L]
    embedding_output.npy   # [1, L, 1024]
    layer_00_output.npy    # [1, L, 1024]
    layer_01_output.npy    # ...
    ...
    layer_27_output.npy
    final_norm_output.npy  # [1, L, 1024]
    logits.npy             # [1, L, 151936]
    generated_ids.npy      # [1, L+20]
    rope_cos.npy           # [max_seq, 64]
    rope_sin.npy           # [max_seq, 64]
  prompt2/
    ...
```

**Acceptance criteria:**
- Script runs offline with local model files
- All .npy files produced and loadable
- Logits are full precision (float32)
- Generation is greedy (deterministic, reproducible)

---

### T22: Validation (Rust vs Python)

**Module:** `examples/validate.rs`, `tests/integration.rs`
**Complexity:** Medium
**Depends on:** T19, T20, T21

**Specification:**
```rust
/// Load Python reference .npy files and compare against Rust forward pass.
/// Reports per-layer MSE and cosine similarity.
fn validate_against_reference(
    model: &mut Qwen3ForCausalLM,
    reference_dir: &Path,
) -> Result<ValidationReport>;

struct ValidationReport {
    embedding_mse: f64,
    embedding_cosine: f64,
    per_layer: Vec<LayerValidation>,
    final_norm_mse: f64,
    logits_mse: f64,
    logits_cosine: f64,
    generated_tokens_match: bool,
}

struct LayerValidation {
    layer_idx: usize,
    mse: f64,
    cosine_similarity: f64,
    max_abs_diff: f64,
}
```

**Validation metrics:**
- **MSE < 1e-5**: Acceptable for BF16 precision
- **Cosine similarity > 0.999**: Confirms same direction
- **Max absolute diff < 1e-3**: No catastrophic outliers
- **Generated tokens match**: Greedy output identical

**Validation strategy:**
1. First validate embedding layer alone (simplest)
2. Then validate layer-by-layer (cumulative error check)
3. Then validate final logits
4. Finally validate greedy generation output

**Unit tests:**
- MSE calculation correctness
- Cosine similarity calculation correctness
- Report formatting

**Acceptance criteria:**
- Per-layer MSE < 1e-5 for first 10 layers
- Per-layer MSE < 1e-4 for all 28 layers (accumulated error)
- Final logits MSE < 1e-3
- Greedy generated tokens match Python for first 20 tokens
- If validation fails: report identifies which layer diverges

---

## Coding Standards

See companion document: `docs/standards/rust-coding-standards.md`

---

## Integration Strategy

### Component Assembly Order

1. **Tensor layer** (T01-T07): Self-contained, testable with pure math
2. **Config + Loading** (T08-T10): Verified against real files
3. **Model components** (T11-T18): Each tested in isolation with synthetic weights
4. **Full model** (T19): Integration of all components, tested with real weights
5. **Generation** (T20): End-to-end text output
6. **Validation** (T21-T22): Proof of correctness

### Integration Tests

```rust
// tests/integration.rs

#[test]
fn test_single_token_forward() {
    // Load model, forward single token, verify logits shape
}

#[test]
fn test_greedy_generation() {
    // Load model, generate 10 tokens, verify non-empty text
}

#[test]
fn test_kv_cache_consistency() {
    // Generate tokens one-by-one vs all-at-once, verify same logits
}

#[test]
fn test_vs_python_reference() {
    // Load reference .npy, compare layer-by-layer
}
```

### Final Validation Checklist

- [ ] Model loads from SafeTensors in < 5 seconds
- [ ] Single forward pass produces valid logits
- [ ] Greedy generation produces readable text
- [ ] Per-layer MSE < 1e-4 vs Python reference
- [ ] Greedy output matches Python for 20 tokens
- [ ] Memory usage < 2GB (model + activations)
- [ ] No panics or undefined behavior

---

## Estimated Timeline

| Task Group | Tasks | Est. Time | Parallelism |
|------------|-------|-----------|-------------|
| Foundation | T01-T02 | 1h | Sequential |
| Tensor core | T03 | 3h | Single |
| Tensor ops | T04-T07 | 4h | 3-way parallel |
| Config/Loading | T08-T10 | 3h | 3-way parallel |
| Model basics | T11-T12 | 2h | 2-way parallel |
| RoPE + Attention | T13-T14 | 4h | Sequential |
| MLP + Block | T15-T16 | 2h | Sequential |
| Cache + Mask | T17-T18 | 2h | 2-way parallel |
| Assembly | T19 | 3h | Single |
| Generation | T20 | 2h | Single |
| Python ref | T21 | 2h | Parallel with T13-T20 |
| Validation | T22 | 3h | Single |
| **Total** | **22 tasks** | **~20h** | **~12h with parallelism** |

---

## Journaling Protocol

Every task completion produces a journal entry in `docs/journal/phase0/`:

```
docs/journal/phase0/
  T01-error-types.md
  T02-bf16-support.md
  T03-tensor-core.md
  ...
  T22-validation.md
```

Each entry records:
- What was implemented
- Key decisions made (and why)
- Any surprises or deviations from plan
- Performance observations
- Implications for Qwen3-Omni (Phase 1)

After each task, update `logs/ledger.md` with:
```
## YYYY-MM-DD HH:MM
Done: T{XX} — {description}
Next: T{YY} — {next task}
Findings: {any notable findings}
```
