# Architecture: Qwen3-Omni Full Inference in Rust (CPU-first, GPU-ready)

**Date:** 2026-01-31
**Status:** Pending Approval

## Goal

Build the first complete Qwen3-Omni implementation in Rust with Q8_0 quantization.
All four components (AuT Encoder + Thinker + Talker + Code2Wav) working end-to-end.
CPU-first with clean component separation for gradual wgpu/Vulkan GPU migration.

**Key differentiator**: No other Rust implementation exists. llama.cpp doesn't support Qwen3-Omni.
vLLM supports it but requires ROCm (unstable on Strix Halo). This is greenfield.

## Current State

### What exists in candle-16b

```
candle-transformers/src/models/qwen3_omni/
  config.rs       (472 lines) -- All 4 component configs, well-structured
  thinker.rs      (616 lines) -- Full Thinker MoE with KV cache, RoPE, GQA
  aut_encoder.rs  (382 lines) -- ConvStem + Transformer encoder + VQ quantizer
  talker.rs       (437 lines) -- MoE speech decoder with multi-codebook output
  code2wav.rs     (238 lines) -- HiFi-GAN-style vocoder
  audio.rs        (245 lines) -- Mel spectrogram, audio loading (hound)
  mod.rs          (164 lines) -- Qwen3Omni orchestrator struct, forward pipeline
```

**Status**: All components are implemented in BF16 for SafeTensors loading. Text-only
inference via Thinker works (example: `qwen3_omni_text`). Full pipeline (audio in/out)
exists in code but is **untested** with real weights.

### Infrastructure

- **candle-wgpu/**: wgpu backend with Q8_0 matmul shader (DP4a), softmax, layer_norm, RoPE
- **candle-core/src/wgpu_backend/**: Backend integration into Device enum
- **candle-core/src/quantized/**: GGML quantization (Q8_0, Q4_0, etc.), CPU SIMD kernels
- **tensor-tools/**: SafeTensors-to-GGUF quantizer (supports Q8_0)

### Hardware Context

- **Target**: AMD Ryzen AI Max+ 395 (Strix Halo), 128GB unified memory
- **GPU**: Radeon 8060S, RDNA 3.5, Vulkan 1.3
- **ROCm**: Unstable (gfx1151 not officially supported, GPU hangs, firmware bugs)
- **Vulkan RADV**: 2.4x faster than ROCm HIP for short context prompt processing
- **NPU**: Limited to <8B models, not relevant for 30B MoE

### Model Sizes

| Component     | BF16 Size | Q8_0 Size | Active Params |
|---------------|-----------|-----------|---------------|
| Thinker       | ~61 GB    | ~33 GB    | 3B (of 30B MoE) |
| AuT Encoder   | ~2.6 GB   | ~1.4 GB   | 650M |
| Talker        | ~6 GB     | ~3.2 GB   | 0.3B (of 3B MoE) |
| Code2Wav      | ~0.4 GB   | ~0.2 GB   | ~200M |
| **Total**     | **~70 GB**| **~38 GB**| -- |

Q8_0 at 38GB fits comfortably in 128GB with room for KV cache and activations.

## Proposed Options

### Option A: Monolithic Quantized Pipeline (All-at-once)

**Approach:** Extend the existing `Qwen3Omni` struct to load quantized GGUF weights
for all components. Single binary that does everything.

**Changes:**
- `candle-transformers/src/models/qwen3_omni/thinker.rs`: Add quantized variant using `candle_core::quantized::QTensor`
- Same for `aut_encoder.rs`, `talker.rs`, `code2wav.rs`
- `tensor-tools/src/main.rs`: Add Qwen3-Omni-aware quantization (per-component GGUF)
- `candle-examples/`: New `qwen3_omni_full` example

**Pros:**
- Simplest mental model -- one GGUF, one binary
- Fastest path to working demo

**Cons:**
- Hard to test components independently
- Can't run Thinker on GPU while AuT runs on CPU
- Can't upgrade one component without rebuilding everything
- Difficult to profile bottlenecks per component

---

### Option B: Component-Isolated Architecture (Recommended)

**Approach:** Each component is an independent module with its own weight loading,
device placement, and execution. A thin orchestrator connects them via typed
intermediate representations. Each component can run on CPU or GPU independently.

**Architecture:**

```
                    ┌─────────────────────────────────────────────┐
                    │            OmniOrchestrator                  │
                    │  (routes data between components,            │
                    │   manages lifecycle, handles streaming)      │
                    └─────┬───────┬───────┬───────┬───────────────┘
                          │       │       │       │
              ┌───────────┤       │       │       ├────────────┐
              │           │       │       │       │            │
         ┌────▼────┐ ┌────▼────┐ ┌▼───────▼┐ ┌───▼─────┐     │
         │AudioProc│ │AuTEnc   │ │Thinker  │ │Talker   │ ┌───▼────┐
         │(CPU)    │ │(CPU/GPU)│ │(CPU/GPU)│ │(CPU/GPU)│ │Code2Wav│
         │mel spec │ │encoder  │ │MoE LLM  │ │MoE TTS  │ │vocoder │
         └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └───┬────┘
              │           │           │           │           │
              └───────────┘───────────┘───────────┘───────────┘
                    All share: ComputeBackend trait
                    Each loads own: weights (SafeTensors OR GGUF)
```

**Key Design Decisions:**

1. **ComputeBackend trait** -- abstraction over CPU/wgpu for each component:
   ```rust
   pub trait ComputeBackend: Send + Sync {
       fn device(&self) -> &Device;
       fn dtype(&self) -> DType;
       fn name(&self) -> &str;
   }
   ```

2. **Typed intermediate representations** between components:
   ```rust
   // AuT Encoder → Thinker
   pub struct AudioTokens { pub tokens: Tensor, pub attention_mask: Tensor }

   // Thinker → Talker (+ text output)
   pub struct ThinkerOutput { pub text_logits: Tensor, pub talker_tokens: Tensor }

   // Talker → Code2Wav
   pub struct CodecTokens { pub codes: Vec<Tensor> } // 4 codebooks

   // Code2Wav → output
   pub struct AudioOutput { pub waveform: Tensor, pub sample_rate: u32 }
   ```

3. **Weight loading strategy** -- each component loads independently:
   ```rust
   pub enum WeightFormat {
       SafeTensors { path: PathBuf, dtype: DType },
       Gguf { path: PathBuf },
   }
   ```

4. **Device placement per component**:
   ```rust
   pub struct OmniConfig {
       pub audio_proc: DevicePlacement,    // Always CPU (mel spectrogram)
       pub aut_encoder: DevicePlacement,    // CPU first, GPU later
       pub thinker: DevicePlacement,        // CPU first (Q8_0), GPU later
       pub talker: DevicePlacement,         // CPU first, GPU later
       pub code2wav: DevicePlacement,       // CPU first, GPU later
   }

   pub enum DevicePlacement {
       Cpu,
       Wgpu { device_id: usize },
       Auto,  // picks best available
   }
   ```

**Changes:**

- `candle-transformers/src/models/qwen3_omni/mod.rs`:
  - Refactor `Qwen3Omni` to `OmniOrchestrator`
  - Add intermediate types (`AudioTokens`, `CodecTokens`, `AudioOutput`)
  - Add `OmniConfig` with per-component device placement

- `candle-transformers/src/models/qwen3_omni/thinker.rs`:
  - Add `QuantizedThinker` variant using `QMatMul` from candle quantized
  - Preserve existing BF16 `Thinker` as-is
  - Both implement same `ThinkerModel` trait

- `candle-transformers/src/models/qwen3_omni/aut_encoder.rs`:
  - Add `QuantizedAuTEncoder` variant
  - Keep BF16 variant

- `candle-transformers/src/models/qwen3_omni/talker.rs`:
  - Add `QuantizedTalker` variant
  - Keep BF16 variant

- `candle-transformers/src/models/qwen3_omni/code2wav.rs`:
  - Add `QuantizedCode2Wav` variant
  - Keep BF16 variant

- `tensor-tools/src/main.rs`:
  - Add `quantize-omni` subcommand
  - Per-component quantization with separate GGUF files
  - Or single GGUF with component prefixes

- New examples:
  - `candle-examples/examples/qwen3_omni_audio/main.rs` -- audio-to-text
  - `candle-examples/examples/qwen3_omni_tts/main.rs` -- text-to-speech
  - `candle-examples/examples/qwen3_omni_full/main.rs` -- full pipeline
  - `candle-examples/examples/qwen3_omni_bench/main.rs` -- per-component benchmarks

**Pros:**
- Each component testable/benchmarkable independently
- Can run Thinker on GPU while Code2Wav stays on CPU (or vice versa)
- Clean GPU migration path: port one component at a time
- Can mix BF16 and Q8_0 per component (e.g., Q8_0 Thinker + BF16 Code2Wav)
- Intermediate types are self-documenting API
- Existing BF16 code unchanged

**Cons:**
- More code to write upfront
- Slight overhead from intermediate type creation (negligible vs compute)
- Need to manage cross-device tensor transfers

---

### Option C: Layered Migration Strategy

**Approach:** Same as Option B but with a strict phased delivery. Each phase produces
a usable artifact before proceeding to the next.

**This is Option B with an explicit ordering guarantee.**

**Phase 1 (Week 1): Q8_0 Text-Only Thinker**
- Quantize Thinker weights to GGUF Q8_0
- `QuantizedThinker` loads GGUF, runs on CPU
- Example: `qwen3_omni_text_q8` -- same as existing text example but Q8_0
- **Deliverable**: Text completion at ~33GB memory, ~8-12 t/s on Strix Halo CPU

**Phase 2 (Week 2): Audio Input (AuT Encoder)**
- Quantize AuT encoder to Q8_0
- `QuantizedAuTEncoder` loads GGUF
- Example: `qwen3_omni_audio` -- audio file → text output
- **Deliverable**: Speech-to-text via full Omni architecture

**Phase 3 (Week 3): Audio Output (Talker + Code2Wav)**
- Quantize Talker and Code2Wav
- Full pipeline: audio in → text + audio out
- Example: `qwen3_omni_full`
- **Deliverable**: End-to-end speech-to-speech

**Phase 4 (Week 4+): GPU Migration**
- Port Thinker matmul to wgpu (biggest impact: 80%+ of compute)
- Port softmax, layer_norm, RoPE (already have wgpu shaders)
- Port Code2Wav convolutions (nice-to-have)
- **Deliverable**: Mixed CPU+GPU inference

**Pros:**
- Everything from Option B
- Usable output after each week
- Natural priority ordering (Thinker is 87% of compute)
- Clear "definition of done" for each phase

**Cons:**
- Same as Option B
- Commits to specific ordering (but it's the right ordering)

## Recommendation

**Option C: Layered Migration Strategy**

Reasoning:
1. **Risk management**: Each phase produces a testable, shippable artifact. If GPU migration stalls (which is likely given ROCm instability), we still have a fully working CPU Q8_0 pipeline.

2. **Thinker first**: At 30B params (87% of total), the Thinker dominates compute and memory. Q8_0 cuts it from 61GB to 33GB -- this alone makes the model runnable on Strix Halo with room to spare.

3. **Component isolation is not optional**: The existing `candle-wgpu` has Q8_0 matmul shaders but no other components ported. We NEED per-component device placement to use GPU for matmul while CPU handles everything else.

4. **Q8_0 on CPU is fast**: With AVX2/AVX-512 on Zen 5 cores, Q8_0 matmul is highly efficient. The Strix Halo has 16 Zen 5 cores -- for a 3B active MoE, CPU Q8_0 should give 8-12 t/s.

5. **Existing code is solid**: The BF16 implementations of all 4 components already exist and compile. Adding quantized variants is additive, not destructive.

## Implementation Plan

### Phase 1: Q8_0 Thinker (Text-Only)

1. **Extend tensor-tools with Omni quantizer**
   - Add `quantize-omni` subcommand
   - Read SafeTensors weights with "thinker." prefix
   - Quantize to GGUF with Q8_0 (2d weights) / F16 (embeddings, norms)
   - Output: `thinker-q8_0.gguf`

2. **Create QuantizedThinker**
   - New file: `candle-transformers/src/models/qwen3_omni/quantized_thinker.rs`
   - Uses `candle_core::quantized::{QTensor, QMatMul}` for linear layers
   - MoE router uses F32 (small, accuracy-critical)
   - KV cache stays F32/BF16
   - Implement same `forward_text_only()` API as BF16 Thinker

3. **Add intermediate types to mod.rs**
   - `AudioTokens`, `ThinkerOutput` (already exists), `CodecTokens`, `AudioOutput`
   - `OmniConfig` with device placement
   - `WeightFormat` enum

4. **Create qwen3_omni_text_q8 example**
   - Load GGUF weights for Thinker
   - Text completion on CPU
   - Benchmark: tokens/s, memory usage, first-token latency

### Phase 2: Audio Input

5. **Create QuantizedAuTEncoder**
   - New file: `quantized_aut_encoder.rs`
   - ConvStem stays F16/F32 (small, not worth quantizing)
   - Transformer layers use Q8_0 linear
   - VQ codebook stays F32

6. **Extend quantizer for AuT encoder weights**
   - "aut_encoder." prefix in SafeTensors
   - Output: `aut_encoder-q8_0.gguf`

7. **Create qwen3_omni_audio example**
   - Load audio file (WAV/OGA via hound)
   - Mel spectrogram on CPU
   - AuT encoder → Thinker → text output
   - Benchmark: end-to-end latency, RTF

### Phase 3: Audio Output

8. **Create QuantizedTalker**
   - New file: `quantized_talker.rs`
   - Same pattern as QuantizedThinker (Q8_0 MoE)

9. **Create QuantizedCode2Wav**
   - New file: `quantized_code2wav.rs`
   - Conv layers: Q8_0 or stay F16 (small model, speed doesn't matter)

10. **Create qwen3_omni_full example**
    - Full pipeline: audio → AuT → Thinker → Talker → Code2Wav → audio
    - WAV output via hound
    - Streaming mode: start speaking before full response generated

### Phase 4: GPU Migration

11. **Port Thinker Q8_0 matmul to wgpu**
    - Use existing `candle-wgpu/src/quantized.rs` Q8_0 shader
    - Need: buffer management, weight upload, result readback
    - Gate behind `DevicePlacement::Wgpu`

12. **Port Thinker attention ops**
    - Softmax (already have wgpu shader)
    - RoPE (already have wgpu shader)
    - Layer norm (already have wgpu shader)
    - Challenge: KV cache management on GPU

13. **Port remaining components**
    - Code2Wav convolutions (Conv1d + ConvTranspose1d)
    - AuT encoder transformer layers
    - Talker MoE

## Risks

- **Weight name mapping**: SafeTensors key prefixes (e.g., `thinker.model.layers.0.`) may not
  match expected GGUF tensor names. Need careful mapping in quantizer.
  **Mitigation**: Print all tensor names during first quantization, create explicit mapping table.

- **MoE quantization quality**: 64 experts with Q8_0 may have worse quality than dense models.
  **Mitigation**: Q8_0 is high-quality (~99.5% of FP16). MoE-specific research shows Q8_0
  is safe. If issues arise, can use Q8_K or per-expert calibration.

- **Code2Wav audio quality**: Vocoder is sensitive to quantization artifacts.
  **Mitigation**: Keep Code2Wav at F16 if Q8_0 produces audible artifacts. It's only ~400MB.

- **Conv1d quantization**: candle's quantized module is designed for linear layers.
  Conv1d weights need reshape before quantization.
  **Mitigation**: Reshape conv weights to 2D matrix, quantize, store reshaped.

- **KV cache memory**: Even with Q8_0 weights, KV cache is F32 and grows with context.
  At 16K context, KV cache for Thinker alone is ~16GB.
  **Mitigation**: Start with 4K context, measure, optimize later (grouped-query helps).

- **wgpu shader limitations**: Current Q8_0 shader handles matmul only. Attention requires
  fused softmax, masking, and multi-head dispatch.
  **Mitigation**: Phase 4 can fall back to CPU for unsupported ops. wgpu shaders for
  softmax/layernorm/rope already exist in candle-wgpu.

## Open Questions

1. **Single GGUF or per-component?** Suggestion: per-component (thinker-q8_0.gguf,
   aut_encoder-q8_0.gguf, etc.) for flexibility. Can always merge later.

2. **Streaming TTS**: Should Talker start generating while Thinker is still producing tokens?
   This is how real-time speech works. Suggestion: implement non-streaming first, add streaming
   in Phase 3.

3. **Image support**: Qwen3-Omni has vision capability through the Thinker. Text+image is
   listed as "mandatory" in requirements. The existing Thinker code doesn't include vision
   preprocessing. Suggestion: defer to a Phase 2.5 -- add image encoder after audio works.

4. **Video support**: Listed as "optional". Suggestion: defer entirely until Phase 3+ is stable.
