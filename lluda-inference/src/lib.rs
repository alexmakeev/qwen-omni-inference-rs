//! Qwen Omni Inference
//!
//! Pure Rust implementation of Qwen Omni multimodal models.
//! Optimized for AMD Strix Halo (128GB UMA, wgpu/Vulkan).

#![warn(missing_docs)]

// Phase 0: Infrastructure modules
pub mod error;
pub mod bf16;
pub mod tensor;
