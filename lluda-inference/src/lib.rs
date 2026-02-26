//! Qwen Omni Inference
//!
//! Pure Rust implementation of Qwen Omni multimodal models.
//! Optimized for AMD Strix Halo (128GB UMA, wgpu/Vulkan).

#![warn(missing_docs)]

// Phase 0: Infrastructure modules
pub mod error;
pub mod bf16;
pub mod tensor;
pub mod config;
pub mod tokenizer;
pub mod loader;

// Phase 1: GPU acceleration (optional feature)
#[cfg(feature = "gpu")]
pub mod gpu;

// Model components
pub mod embedding;
pub mod rms_norm;
pub mod layer_norm;
pub mod conv1d;
pub mod rope;
pub mod attention;
pub mod audio_attention;
pub mod audio_mlp;
pub mod audio_encoder_layer;
pub mod audio_encoder;
pub mod mlp;
pub mod transformer;
pub mod causal_mask;
pub mod model;
pub mod omni_attention;
pub mod talker;
pub mod omni_model;

// Audio preprocessing
pub mod audio_preprocess;

// Generation
pub mod generate;
