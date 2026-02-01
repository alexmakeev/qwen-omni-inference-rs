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

// Model components
pub mod embedding;
pub mod rms_norm;
pub mod rope;
pub mod attention;
