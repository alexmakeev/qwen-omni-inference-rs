# Development Ledger — qwen-omni-inference-rs

Project journal documenting all decisions, research, and progress.

---

## 2026-01-31 19:15 — Project Initialization

### Goal
Build pure Rust inference for Qwen Omni multimodal models, optimized for AMD Strix Halo (128GB unified memory, 128 CU RDNA 3.5 iGPU).

### Key Decisions

**Architecture: Variant A+ (Lightweight own abstraction)**
- NO candle-core dependency (full control for quantization)
- Minimal tensor abstraction (layout, storage, ops)
- Direct wgpu integration for GPU (Phase 3+)
- Maximum flexibility for Q8_0 custom kernels

**Implementation Strategy: BF16-first, then Q8_0**
- Phase 0: Infrastructure on Qwen3-0.6B (1.4GB, already local)
- Phase 1: BF16 full precision on Luda (CPU)
- Phase 2: Q8_0 quantization
- Phase 3: wgpu/Vulkan GPU acceleration

**Why BF16 before Q8_0:**
- Need ground truth for quantization quality comparison
- Numerical precision validation
- Layer-by-layer debugging reference

**Reference Tensor Extraction:**
- Use HuggingFace Transformers `output_hidden_states=True`
- Store in NumPy .npz format (Rust: ndarray-npy)
- Phase 0: Extract from local Qwen3-0.6B
- Phase 0.5: Extract from Qwen3-Omni 30B via HuggingFace Inference API
  - All modalities: text, speech-to-text, text-to-speech, image
  - Free tier or ~$2 budget
- Validation metrics: MSE < 1e-5, Cosine similarity > 0.999

---

### Research Summary

**1. ROCm vs Vulkan on Strix Halo**
- ROCm 7.2.0 supports gfx1151 but unstable (GPU hangs, compute corruption)
- Vulkan (RADV) 1.8x faster on long contexts (32K tokens)
- Verdict: wgpu/Vulkan path

**2. Qwen3-Omni Ecosystem**
- No existing Rust implementation (we're first)
- llama.cpp: text-only, no audio support
- vLLM: Python only, full Omni support
- Our position: first full Omni in Rust with Q8_0

**3. Candle Framework**
- NO Vulkan/ROCm support (issues open 1-2 years)
- NO Qwen3-Omni support
- Focus: CUDA/Metal
- Verdict: Not suitable

**4. NPU (AMD XDNA2)**
- Limited to ≤8B models, 2-3K context
- Qwen3-Omni 30B-A3B = 30B total, 3B active (MoE)
- Critical blockers:
  - Software immaturity (Linux VAI EP unstable)
  - MoE dynamic routing vs NPU static AOT compilation
  - Memory hierarchy: 30B experts don't fit in NPU SRAM → PCIe thrashing (41.7% I/O stalls)
  - Performance: NPU 3-6 t/s vs GPU 20-40 t/s target
- Verdict: GPU-only path confirmed

**5. Reference Tensor Extraction**
- HuggingFace Transformers: `output_hidden_states=True` works
- Format: NumPy .npz (Rust support via ndarray-npz)
- Qwen3-0.6B: local, Apache 2.0, no authentication needed
- Qwen3-Omni 30B: on Luda (already downloaded), or use HF Inference API
- Validation: layer-by-layer MSE/cosine similarity

---

### Technology Stack

**Language:** Rust (edition 2021)
**Backend:** wgpu/Vulkan (GPU), pure Rust (CPU)
**Quantization:** Q8_0 custom kernels
**Models:** Qwen3-0.6B (dev), Qwen3-Omni-30B-A3B (production)
**Hardware:** AMD Strix Halo (primary), universal wgpu (secondary)

---

### Repository

**Name:** qwen-omni-inference-rs
**URL:** https://github.com/alexmakeev/qwen-omni-inference-rs
**License:** MIT OR Apache-2.0 (Rust standard)

---

### Next Steps

Phase 0 implementation: Qwen3-0.6B infrastructure
- SafeTensors loading
- Transformer operations (attention, RoPE, MLP)
- Reference tensor extraction
- Validation pipeline
- Document all findings for Omni migration

---

## Development Log

[Future entries will go here chronologically]
