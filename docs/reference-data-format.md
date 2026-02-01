# Reference Data Format Specification

**Date:** 2026-02-01
**Purpose:** Document format of Python-generated reference data for Rust validation

---

## Overview

Reference data is extracted from HuggingFace Transformers Qwen3-0.6B model and saved in NumPy `.npz` format (compressed archive of `.npy` arrays). This data serves as ground truth for validating the Rust inference implementation.

## Directory Structure

```
reference_data/
├── prompt1/
│   └── reference.npz      # "Hello" (single token)
├── prompt2/
│   └── reference.npz      # "The capital of France is"
└── prompt3/
    └── reference.npz      # "What is 2+2?"
```

## NPZ File Contents

Each `reference.npz` contains a dictionary of NumPy arrays:

### Input and Output

| Key | Shape | DType | Description |
|-----|-------|-------|-------------|
| `input_ids` | `[1, L]` | `int64` | Tokenized input (L varies by prompt) |
| `generated_ids` | `[1, L+20]` | `int64` | Greedy generation output (20 new tokens) |

### Intermediate Activations

| Key | Shape | DType | Description |
|-----|-------|-------|-------------|
| `embedding_output` | `[1, L, 1024]` | `float32` | Token embeddings (before first layer) |
| `layer_00_output` | `[1, L, 1024]` | `float32` | Output of transformer layer 0 |
| `layer_01_output` | `[1, L, 1024]` | `float32` | Output of transformer layer 1 |
| ... | ... | ... | ... |
| `layer_27_output` | `[1, L, 1024]` | `float32` | Output of transformer layer 27 |
| `final_norm_output` | `[1, L, 1024]` | `float32` | After final RMSNorm |
| `logits` | `[1, L, 151936]` | `float32` | LM head output (full sequence) |

### RoPE Tables

| Key | Shape | DType | Description |
|-----|-------|-------|-------------|
| `rope_cos` | `[40960, 128]` | `float32` | Precomputed cosine table |
| `rope_sin` | `[40960, 128]` | `float32` | Precomputed sine table |

Where:
- `1` = batch size (always 1 for reference data)
- `L` = sequence length (varies: 1-10 tokens depending on prompt)
- `1024` = `hidden_size` (Qwen3-0.6B)
- `151936` = `vocab_size` (Qwen3 tokenizer)
- `40960` = `max_position_embeddings`
- `128` = `head_dim` (hidden_size / num_attention_heads = 1024 / 8)

## Data Precision

- **Model weights**: BF16 (as in production)
- **Activations**: FP32 (converted from BF16 for validation)
- **Logits**: FP32 (full precision for token selection)

This ensures Rust implementation can validate against high-precision reference while still using BF16 weights internally.

## Loading in Rust

Use `ndarray-npy` crate to load `.npz` files:

```toml
[dev-dependencies]
ndarray = "0.16"
ndarray-npy = "0.9"
```

Example:

```rust
use ndarray::Array3;
use ndarray_npy::NpzReader;
use std::fs::File;

fn load_reference_logits(path: &Path) -> Result<Array3<f32>> {
    let file = File::open(path)?;
    let mut npz = NpzReader::new(file)?;
    let logits = npz.by_name("logits.npy")?;
    Ok(logits)
}
```

## Validation Metrics

### Mean Squared Error (MSE)

```rust
fn mse(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    assert_eq!(a.shape(), b.shape());
    let diff = a - b;
    (&diff * &diff).mean().unwrap()
}
```

**Thresholds:**
- Embedding layer: MSE < 1e-6 (no computation yet)
- Early layers (0-10): MSE < 1e-5 (minimal error accumulation)
- Later layers (11-27): MSE < 1e-4 (acceptable accumulated error)
- Final logits: MSE < 1e-3 (BF16 precision limit)

### Cosine Similarity

```rust
fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();
    dot / (norm_a * norm_b)
}
```

**Thresholds:**
- All layers: cosine similarity > 0.999 (confirms correct direction)

### Max Absolute Difference

```rust
fn max_abs_diff(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    let diff = a - b;
    diff.iter().map(|x| x.abs()).fold(f32::MIN, f32::max)
}
```

**Thresholds:**
- All layers: max_abs_diff < 0.01 (no catastrophic outliers)

## Validation Strategy

### 1. Component-Level Validation

Validate each component in isolation using synthetic inputs:
- RMSNorm: Compare output against reference for known inputs
- RoPE: Verify `rope_cos` and `rope_sin` tables match Python
- Attention: Compare scores, weights, and output for small test case

### 2. Layer-by-Layer Validation

Run Rust model layer-by-layer, comparing each layer's output:

```rust
for i in 0..28 {
    let rust_output = model.forward_layer(i, input)?;
    let python_output = reference.get_layer_output(i)?;

    let mse = mse(&rust_output, &python_output);

    assert!(mse < 1e-4, "Layer {} MSE too high: {}", i, mse);
}
```

**Why layer-by-layer:**
- Isolates error source (identifies which layer diverges)
- Prevents error accumulation from hiding root cause
- Enables incremental debugging

### 3. End-to-End Validation

Compare final outputs:
- **Logits**: MSE and cosine similarity on full `[1, L, 151936]` tensor
- **Greedy tokens**: Exact match on `generated_ids` (first 20 tokens)

### 4. Greedy Generation Validation

```rust
let mut rust_tokens = vec![input_ids];

for _ in 0..20 {
    let logits = model.forward(&rust_tokens)?;
    let next_token = argmax(&logits);  // Greedy
    rust_tokens.push(next_token);
}

assert_eq!(rust_tokens, python_generated_ids);
```

**Critical:** Generation must be deterministic (greedy, no sampling).

## Common Issues and Debugging

### MSE Too High on Early Layers

**Likely causes:**
1. Embedding lookup incorrect (wrong row selection)
2. RMSNorm computation error (check epsilon handling)
3. RoPE implementation mismatch (rotation direction)

**Debug:**
```rust
// Compare element-wise to find divergence point
for (i, (&rust, &python)) in rust_output.iter().zip(python_output.iter()).enumerate() {
    if (rust - python).abs() > 1e-5 {
        println!("Divergence at index {}: rust={}, python={}", i, rust, python);
    }
}
```

### MSE Grows Rapidly Across Layers

**Likely causes:**
1. Residual connections missing or incorrect
2. Pre-norm vs post-norm mismatch
3. Attention mask application error

**Debug:**
Compare norms of activations:
```rust
println!("Layer {} norm: rust={}, python={}",
         i,
         rust_output.iter().map(|x| x*x).sum::<f32>().sqrt(),
         python_output.iter().map(|x| x*x).sum::<f32>().sqrt());
```

### Logits Match but Generated Tokens Differ

**Likely causes:**
1. Argmax implementation error (wrong axis)
2. KV cache not updated correctly between steps
3. Offset parameter wrong in generation loop

**Debug:**
```rust
// Print top-5 tokens and their probabilities
let top5 = logits.top_k(5);
println!("Rust top-5: {:?}", top5);
println!("Python top token: {}", python_generated_ids[step]);
```

### RoPE Tables Mismatch

**Likely causes:**
1. Theta value incorrect (should be 1000000.0 for Qwen3)
2. Frequency calculation wrong (inv_freq formula)
3. Position indexing off-by-one

**Debug:**
```rust
// Compare first few positions
for pos in 0..5 {
    println!("pos={}: rust_cos={:?}, python_cos={:?}",
             pos,
             &rust_cos.slice(s![pos, 0..4]),
             &python_cos.slice(s![pos, 0..4]));
}
```

## Performance Notes

- **Load time**: `.npz` loading is fast (<100ms for all three prompts)
- **File size**: ~50-100 MB per prompt (compressed)
- **Memory**: ~200-400 MB uncompressed in RAM
- **Validation time**: Layer-by-layer comparison ~1s per prompt

## Regenerating Reference Data

If model weights change or Python implementation updates:

```bash
# Re-extract reference data
python scripts/extract_reference.py

# Verify new data
python scripts/inspect_reference.py reference_data/prompt1/reference.npz

# Re-run Rust validation
cargo test --test integration -- --nocapture
```

## Reference Implementation

Python extraction script: `scripts/extract_reference.py`
Rust validation example: `examples/validate.rs`
Rust integration tests: `tests/integration.rs`

---

## Appendix: Test Prompts

### Prompt 1: "Hello"

- **Purpose**: Simplest case, single token
- **Expected tokens**: 1 input + 20 generated = 21 total
- **Use case**: Verify basic embedding and generation

### Prompt 2: "The capital of France is"

- **Purpose**: Multi-token, factual knowledge
- **Expected tokens**: ~6 input + 20 generated = 26 total
- **Use case**: Verify attention mask, KV cache growth

### Prompt 3: "What is 2+2?"

- **Purpose**: Question format, arithmetic
- **Expected tokens**: ~5 input + 20 generated = 25 total
- **Use case**: Verify chat template, reasoning

All prompts use greedy decoding for deterministic, reproducible outputs.
