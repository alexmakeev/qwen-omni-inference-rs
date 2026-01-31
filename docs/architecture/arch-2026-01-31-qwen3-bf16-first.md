# Architecture: Qwen3-Omni BF16-First Inference in Rust

**Date:** 2026-01-31
**Status:** Pending Approval
**Supersedes:** `arch-2026-01-31-qwen3-omni-q8-rust.md` (Q8_0-first approach)

## Goal

Build complete Qwen3-Omni inference in Rust with a **BF16-first** strategy.
Phase 0 develops all infrastructure on Qwen3-0.6B (1.4GB, dense, text-only).
Phase 1+ applies learnings to full Qwen3-Omni-30B (~70GB BF16) on Luda (128GB Strix Halo).
BF16 baseline established BEFORE any quantization to provide ground truth for quality comparison.

**Critical change from previous plan**: BF16 working first, Q8_0 second. Phase 0 on small model to iterate fast.

## Current State

### What exists

```
candle-16b/candle-transformers/src/models/
  qwen3.rs              (312 lines) -- BF16 dense Qwen3 (same arch as 0.6B) with q_norm/k_norm
  qwen3_moe.rs          (504 lines) -- BF16 MoE variant
  qwen3_omni/
    config.rs           (472 lines) -- All 4 component configs
    thinker.rs          (616 lines) -- BF16 MoE Thinker, custom RoPE, GQA, KV cache
    aut_encoder.rs      (382 lines) -- ConvStem + Transformer + VQ
    talker.rs           (437 lines) -- MoE speech decoder, multi-codebook
    code2wav.rs         (238 lines) -- HiFi-GAN vocoder
    audio.rs            (245 lines) -- Mel spectrogram
    mod.rs              (164 lines) -- Qwen3Omni orchestrator
  quantized_qwen3.rs    (350 lines) -- GGUF Q8_0 dense Qwen3 (reference pattern)
  quantized_qwen3_moe.rs            -- GGUF Q8_0 MoE variant

candle-16b/candle-examples/examples/
  qwen3_omni_text/      -- BF16 Thinker text-only (loads from SafeTensors)
  quantized-qwen3/      -- GGUF quantized dense Qwen3

models/Qwen3-0.6B/
  config.json           -- Qwen3ForCausalLM dense config
  model.safetensors     -- 1.4GB BF16 weights
  tokenizer.json        -- HF tokenizer
```

### Architecture comparison: Qwen3-0.6B vs Qwen3-Omni Thinker

| Feature | Qwen3-0.6B | Omni Thinker |
|---------|-----------|--------------|
| Architecture | Qwen3ForCausalLM | Qwen3OmniThinker |
| Type | Dense | MoE (dense + sparse layers) |
| hidden_size | 1024 | 4096 |
| num_layers | 28 | 40 |
| num_heads | 16 | 32 |
| num_kv_heads | 8 | 8 |
| head_dim | 128 | 128 |
| intermediate_size | 3072 | 11008 (dense) / 2816 (MoE) |
| RoPE theta | 1M | 1M |
| activation | SiLU | SiLU |
| tie_word_embeddings | true | false |
| q_norm/k_norm | **YES** (in qwen3.rs) | **NO** (in thinker.rs) |
| MoE experts | 0 | 64, top-4, every 2nd layer |
| audio_embed | N/A | Optional linear projection |
| talker_head | N/A | Optional linear projection |
| Params | 0.6B | 30B (3B active) |
| BF16 size | 1.4GB | ~61GB |

**Key insight**: The Omni Thinker's `thinker.rs` has its OWN RoPE/Attention/MLP implementations separate from `qwen3.rs`. They share the same mathematical operations but have different code paths. Phase 0 must decide which code path to build on.

### Critical gap in existing thinker.rs

The existing `thinker.rs` has a **different RoPE implementation** than `qwen3.rs`:
- `qwen3.rs` uses `candle_nn::rotary_emb::rope()` (library implementation)
- `thinker.rs` uses custom `apply_rotary()` with manual half-dim split
- `qwen3.rs` has `q_norm`/`k_norm` per attention head; `thinker.rs` does NOT

This means `thinker.rs` may not match Qwen3-0.6B's exact inference behavior. Phase 0 gives us a way to validate which is correct.

### Hardware

- **Luda**: AMD Ryzen AI Max+ 395, 128GB unified memory
- **Qwen3-Omni BF16**: ~70GB total (fits in 128GB with room for KV cache)
- **Qwen3-0.6B BF16**: 1.4GB (trivially fits anywhere, even on dev laptop)

## Proposed Options

### Option A: New Standalone Crate (lluda-inference)

**Approach:** Create a fresh `lluda-inference` crate in the workspace that does NOT depend on candle-transformers models. Build SafeTensors loading, transformer ops, tokenizer integration from scratch using only candle-core primitives. Maximum control, maximum learning.

**New crate structure:**
```
lluda-inference/
  Cargo.toml
  src/
    lib.rs
    safetensors.rs      -- SafeTensors loader with tensor name mapping
    transformer/
      mod.rs
      attention.rs       -- GQA with RoPE, KV cache
      mlp.rs             -- SiLU gated MLP
      layer_norm.rs      -- RMS norm wrapper
      rope.rs            -- Rotary position embeddings
      moe.rs             -- Sparse MoE (Phase 1)
    tokenizer.rs         -- HF tokenizer integration
    config.rs            -- Model config parsing
    backend.rs           -- CPU/GPU backend abstraction trait
    memory.rs            -- Memory management, mmap, arena allocation
    journal.rs           -- Phase 0 findings logger
  examples/
    qwen3_06b.rs         -- Phase 0: text generation with Qwen3-0.6B
    qwen3_omni_text.rs   -- Phase 1: Thinker text-only
    qwen3_omni_full.rs   -- Phase 1+: Full pipeline
```

**Pros:**
- Clean slate, no baggage from candle-transformers patterns
- Every component understood from first principles
- Maximum journaling opportunity -- every decision documented
- Backend abstraction designed for our hardware from day one
- No risk of breaking existing candle code
- Easy to compare against existing candle implementations for correctness

**Cons:**
- More code to write initially
- Duplicates some candle-transformers functionality
- Two parallel implementations to maintain
- Can't reuse existing `qwen3_omni/` code directly

---

### Option B: Extend Existing candle-transformers (Incremental)

**Approach:** Build Phase 0 directly into existing candle-transformers by creating a unified `Qwen3Model` that can load EITHER Qwen3-0.6B OR Qwen3-Omni Thinker from SafeTensors. Fix the existing `thinker.rs` to match `qwen3.rs` patterns. Add journaling as comments/docs.

**Changes:**
- `candle-transformers/src/models/qwen3_omni/thinker.rs`:
  - Refactor to use `candle_nn::rotary_emb::rope()` (match qwen3.rs)
  - Add optional q_norm/k_norm (present in 0.6B, maybe not in Omni)
  - Make MoE optional (num_experts=0 for dense)
  - Add config auto-detection from SafeTensors metadata

- `candle-examples/examples/qwen3_06b_bf16/main.rs`:
  - New example: load Qwen3-0.6B SafeTensors, generate text
  - Use Thinker with dense config (no MoE, no audio)

- Backend abstraction added to existing structs via trait

**Pros:**
- Less code to write
- Reuses proven candle infrastructure
- Single implementation path
- Existing tests/examples still work

**Cons:**
- Tied to candle-transformers conventions (VarBuilder, etc.)
- Harder to experiment with alternative memory management
- Backend abstraction must fit into existing Device enum
- Risk of breaking existing Omni code while refactoring
- Less learning -- just wiring, not building

---

### Option C: Hybrid -- New Crate with candle-core (Recommended)

**Approach:** New `lluda-inference` crate that depends on `candle-core` (tensors, device, dtype) but NOT on `candle-transformers` or `candle-nn`. Build transformer layers using candle-core primitives directly. This gives us candle's tensor/device infrastructure without being locked into its model patterns.

**Key principle: Use candle for what it's good at (tensor ops, device abstraction), build model-level code ourselves.**

**Crate structure:**
```
lluda-inference/
  Cargo.toml              -- depends on candle-core, tokenizers, safetensors, memmap2
  src/
    lib.rs                 -- public API
    error.rs               -- Result<T, LludaError> (not candle::Result)

    # Core tensor operations (thin wrappers over candle-core)
    ops/
      mod.rs
      rope.rs              -- RoPE implementation (validated against qwen3.rs)
      rms_norm.rs           -- RMS LayerNorm
      attention.rs          -- Scaled dot-product attention with GQA
      softmax.rs            -- numerically stable softmax

    # Model components (generic, reusable)
    model/
      mod.rs
      config.rs             -- Unified config: can represent 0.6B, Omni Thinker, etc.
      embedding.rs          -- Token + positional embeddings
      mlp.rs                -- SiLU gated MLP
      moe.rs                -- Sparse MoE (expert routing, top-k)
      transformer.rs        -- Full transformer block (attn + ffn + norm)
      kv_cache.rs           -- KV cache with configurable strategy

    # Weight loading
    weights/
      mod.rs
      safetensors.rs        -- SafeTensors loader with prefix mapping
      tensor_map.rs         -- HF name <-> internal name mapping

    # Backend abstraction
    backend/
      mod.rs                -- ComputeBackend trait
      cpu.rs                -- CPU backend (BF16, F32)

    # Tokenizer
    tokenizer.rs            -- HF tokenizer wrapper

    # Generation
    generate.rs             -- Autoregressive generation loop
    sampling.rs             -- Temperature, top-k, top-p, repetition penalty

    # Journaling
    journal/
      mod.rs                -- Structured findings log
      entries.rs            -- Finding types (perf, correctness, memory, etc.)

  # Phase 0 examples
  examples/
    qwen3_06b_generate.rs   -- Text generation, validates correctness
    qwen3_06b_bench.rs       -- Benchmark: tokens/s, memory, first-token latency
    qwen3_06b_validate.rs    -- Compare output against HF transformers reference

  # Phase 1 examples (stubs, filled in later)
  examples/
    omni_thinker_text.rs     -- Thinker text-only (BF16)
    omni_audio_to_text.rs    -- AuT + Thinker
    omni_full_pipeline.rs    -- Full pipeline

  # Journal output
  journal/                   -- gitignored, findings written here
    README.md
```

**Dependencies (Cargo.toml):**
```toml
[dependencies]
candle-core = { path = "../candle-16b/candle-core" }
safetensors = "0.4"
memmap2 = "0.9"
tokenizers = { version = "0.21", default-features = false, features = ["onig"] }
half = { version = "2.4", features = ["num-traits"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tracing = "0.1"

[dev-dependencies]
anyhow = "1"
clap = { version = "4", features = ["derive"] }
```

**Phase 0 Implementation Plan (Qwen3-0.6B):**

1. **Scaffold crate** (Day 1)
   - Cargo.toml, lib.rs, error.rs
   - Journal infrastructure
   - Basic test framework

2. **SafeTensors loading** (Day 1-2)
   - Memory-mapped SafeTensors reader
   - Tensor name discovery (print all names from model.safetensors)
   - BF16 tensor loading into candle Tensor
   - **Journal**: tensor layout, memory patterns, load times

3. **Core ops** (Day 2-3)
   - RMS norm (validated against candle-nn version)
   - RoPE (validated against BOTH qwen3.rs and thinker.rs implementations)
   - GQA attention with KV cache
   - SiLU gated MLP
   - **Journal**: numerical accuracy, performance, memory allocation patterns

4. **Transformer assembly** (Day 3-4)
   - Stack layers with config-driven parameters
   - Causal attention mask generation
   - Token embedding with optional tie
   - Forward pass: tokens -> logits
   - **Journal**: per-layer timing, memory growth, correctness vs reference

5. **Generation loop** (Day 4-5)
   - Greedy decoding
   - Temperature + top-p sampling
   - KV cache reuse for autoregressive steps
   - **Journal**: tokens/s, memory steady-state, cache behavior

6. **Validation** (Day 5-6)
   - Run same prompt through HF transformers (Python) and our Rust
   - Compare logits (not just output text) for first N tokens
   - Fix any discrepancies
   - **Journal**: exact match report, any divergence analysis

7. **Benchmark** (Day 6-7)
   - First-token latency
   - Steady-state tokens/s
   - Peak memory usage
   - Compare against candle's existing qwen3.rs on same model
   - **Journal**: full performance report

**Phase 1 Plan (Qwen3-Omni BF16 on Luda):**

After Phase 0, we have validated infrastructure. Phase 1 applies it:

1. **Thinker BF16** -- extend config for MoE, add expert routing, load 61GB SafeTensors
2. **AuT Encoder BF16** -- add Conv1d, mel spectrogram, VQ encoder
3. **Talker BF16** -- MoE speech decoder with multi-codebook output
4. **Code2Wav BF16** -- HiFi-GAN Conv1d/ConvTranspose1d
5. **Full pipeline BF16** -- end-to-end audio-to-audio
6. **Validation** -- compare against HF inference

**Phase 2 Plan (Q8_0 Quantization):**

Only AFTER BF16 works and is validated:

1. **Quantize weights** -- SafeTensors BF16 -> GGUF Q8_0 per component
2. **QMatMul integration** -- use candle-core quantized ops
3. **Benchmark** -- quality (perplexity/BLEU vs BF16) + speed + memory

**Phase 3 Plan (GPU Migration):**

4. **wgpu backend** -- integrate existing candle-wgpu for matmul/attention
5. **Mixed placement** -- Thinker on GPU, others on CPU

**Pros:**
- candle-core gives us battle-tested tensor ops + device abstraction
- No dependency on candle-nn/candle-transformers model patterns
- Full control over model loading, memory layout, computation order
- Every component built with understanding, documented in journal
- Journal becomes invaluable reference for Phase 1 Omni implementation
- Clean backend trait -- CPU now, wgpu slot ready
- Can validate against existing candle qwen3.rs for correctness
- BF16 baseline provides ground truth before any quantization

**Cons:**
- More initial work than Option B
- Must implement RMS norm, RoPE, attention ourselves (but these are ~50 lines each)
- Two implementations of Qwen3 in the repo (candle-transformers + lluda-inference)
  - Mitigation: lluda-inference is the "production" path; candle-transformers is reference
- Need to resolve candle-core version compatibility

---

## Recommendation

**Option C: Hybrid -- New Crate with candle-core**

Reasoning:

1. **Phase 0 is about learning, not shipping.** A new crate maximizes learning. Every op implemented = one more thing understood. The journal captures everything for Phase 1.

2. **BF16-first requires SafeTensors loading.** The existing candle-transformers loads via VarBuilder which is fine, but building our own loader teaches us tensor memory layout, mmap patterns, and BF16 handling -- all critical for 70GB models on 128GB hardware.

3. **The existing thinker.rs has correctness questions.** Different RoPE, missing q_norm/k_norm. Rather than debug someone else's code, we build our own and validate against the known-good `qwen3.rs` reference AND Python HF transformers.

4. **Backend abstraction is a first-class concern.** We need CPU for Phase 0-1, wgpu for Phase 2+. Designing this into the crate from day one is cleaner than retrofitting candle-transformers.

5. **Memory management matters at 70GB.** On 128GB Luda, loading 70GB of weights leaves ~58GB for KV cache + activations + OS. We need explicit control over memory mapping, pre-allocation, and arena strategies. A fresh crate lets us design this properly.

6. **Qwen3-0.6B at 1.4GB is the perfect development target.** Fast iteration, fits anywhere, shares all base components with the Thinker. We can run the validation cycle (build -> test -> journal) many times per day.

## Risks

- **candle-core API changes**: We depend on candle-core's tensor API. If it changes, we must update.
  **Mitigation**: Pin candle-core version. Our usage is limited to basic tensor ops.

- **BF16 CPU performance**: Native BF16 compute is slow on CPU (no hardware BF16 ALU on x86). Candle upcasts to F32 for compute.
  **Mitigation**: This is expected and acceptable for Phase 0-1. F32 compute with BF16 storage is the standard approach. 70GB BF16 stored, computed in F32.

- **70GB model loading time**: Loading 70GB of SafeTensors takes significant time.
  **Mitigation**: Memory-mapped loading (memmap2). Only read pages as needed. Phase 0 journal will capture exact timings on 1.4GB to extrapolate.

- **KV cache memory at long context**: BF16 KV cache for 40 layers at 16K context could be ~16GB.
  **Mitigation**: Start with short context (512-2048). Profile memory growth. Consider F16/F32 KV cache strategy.

- **Validation complexity**: Comparing Rust vs Python logits requires matching tokenization, RoPE, and attention exactly.
  **Mitigation**: Compare token-by-token. Use greedy decoding (deterministic). Start with single-token forward pass before generation.

## Open Questions

1. **Should lluda-inference be in the lluda repo root or inside candle-16b workspace?**
   Suggestion: lluda repo root (sibling to candle-16b). It depends on candle-core via path, but is NOT a workspace member of candle-16b. This keeps it independent.

2. **Journal format: Markdown files or structured JSON?**
   Suggestion: Markdown files in `journal/` directory. One file per topic (e.g., `safetensors-loading.md`, `rope-validation.md`). Human-readable, diffable, git-trackable.

3. **Should Phase 0 validate against Python HF transformers or against existing candle qwen3.rs?**
   Suggestion: Both. First validate against candle qwen3.rs (same environment, easy). Then validate against Python (ground truth, harder setup). Journal both comparisons.

4. **Config format: reuse Qwen3-0.6B's config.json directly or define our own?**
   Suggestion: Parse HF config.json directly for Phase 0. Map to internal config struct. This ensures we handle the real config format from day one.

## Journaling Strategy

Every Phase 0 activity produces a journal entry in `lluda-inference/journal/`:

| Activity | Journal File | Contents |
|----------|-------------|----------|
| SafeTensors exploration | `01-safetensors.md` | Tensor names, shapes, dtypes, memory layout |
| RoPE implementation | `02-rope.md` | Algorithm choices, validation results, perf |
| Attention implementation | `03-attention.md` | GQA details, KV cache design, mask strategy |
| MLP implementation | `04-mlp.md` | SiLU gated, memory patterns |
| Full model assembly | `05-model-assembly.md` | Layer stacking, config mapping, load times |
| Generation loop | `06-generation.md` | Autoregressive behavior, KV cache reuse |
| Correctness validation | `07-validation.md` | Logit comparison, divergence analysis |
| Performance benchmarks | `08-benchmarks.md` | t/s, latency, memory, vs candle baseline |
| Omni planning | `09-omni-plan.md` | What changes for MoE, audio, talker |

Each entry follows format:
```markdown
# [Topic]

## Date
YYYY-MM-DD

## Goal
What we're trying to learn

## Approach
What we did

## Findings
- Finding 1: [detail]
- Finding 2: [detail]

## Implications for Omni
- How this applies to Phase 1

## Performance Data
| Metric | Value |
|--------|-------|
| ... | ... |

## Code References
- `file.rs:123` -- relevant code
```
