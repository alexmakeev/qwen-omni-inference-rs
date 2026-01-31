# Qwen Omni Inference in Rust

Pure Rust implementation of Qwen Omni multimodal models with first-class support for AMD Strix Halo APUs.

## 🎯 Primary Goal

Bring industrial-grade multimodal AI inference to **AMD Strix Halo** hardware. This project exists to give Strix Halo "proper brains" — leveraging its unique 128GB unified memory architecture for running large language models efficiently.

## 🔧 Hardware Focus

**Primary target:** AMD Ryzen AI Max 395 (Strix Halo)
- 128GB unified memory (CPU + iGPU shared pool)
- 128 CU RDNA 3.5 integrated GPU
- Optimized for wgpu/Vulkan UMA (Unified Memory Architecture)

**Also runs on:** Any hardware supporting wgpu (NVIDIA, Intel, other AMD GPUs)

## 🗺️ Roadmap

- [ ] **Phase 0:** Infrastructure development (Qwen3-0.6B validation)
- [ ] **Phase 0.5:** Reference tensor extraction (HuggingFace API)
- [ ] **Phase 1:** BF16 full precision inference (CPU)
- [ ] **Phase 1.5:** Audio components (speech-to-speech)
- [ ] **Phase 2:** Q8_0 quantization (memory efficiency)
- [ ] **Phase 3:** GPU acceleration (wgpu/Vulkan)

## 🏗️ Architecture

**Pure Rust** — no candle-core, full control for optimization
- Custom tensor abstraction (minimal, focused)
- Direct wgpu integration for GPU kernels
- Q8_0 quantization with custom WGSL shaders

**BF16-first approach:**
1. Build BF16 baseline (ground truth)
2. Implement Q8_0 quantization
3. Validate quality (MSE, cosine similarity)
4. Optimize for Strix Halo UMA

## 📚 Documentation

- **Architecture:** [`docs/architecture/`](docs/architecture/) — design documents
- **Research:** [`docs/research/`](docs/research/) — technology evaluations (ROCm, NPU, Candle, etc.)
- **Development Log:** [`logs/ledger.md`](logs/ledger.md) — daily progress journal
- **Task Tracking:** [`beads/`](beads/) — granular task management

## 🚀 Current Status

**Phase 0** — Infrastructure development on Qwen3-0.6B (1.4GB test model).
All findings journaled in `docs/journal/phase0/` for later application to full Qwen3-Omni-30B.

## 🌟 Why This Project?

- **First** pure Rust implementation of Qwen Omni (text + audio + vision)
- **First** Q8_0 quantized Omni inference (38GB vs 70GB BF16)
- **Optimized** for AMD Strix Halo's unique UMA architecture
- **Open research** — all decisions documented, reproducible

## 📖 Development Journey

This project follows a meticulous approach:
1. Build infrastructure on small models
2. Extract reference tensors for validation
3. Document every finding in journal
4. Scale to production with confidence

See [`logs/ledger.md`](logs/ledger.md) for the full development story.

## 🤝 Contributing

This is an open research project. Contributions, issues, and discussions welcome!

## 📜 License

MIT OR Apache-2.0 — see LICENSE-MIT and LICENSE-APACHE files
