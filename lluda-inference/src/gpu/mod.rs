//! GPU acceleration infrastructure using wgpu.
//!
//! This module provides GPU context management for accelerating tensor operations
//! on AMD RADV (Vulkan) hardware. The infrastructure gracefully handles cases where
//! no GPU is available.
//!
//! # Example
//!
//! ```rust,no_run
//! use lluda_inference::gpu::{GpuContext, init};
//!
//! // Initialize GPU context
//! let ctx = init().expect("Failed to initialize GPU");
//!
//! // Use the context for GPU operations
//! println!("GPU adapter: {}", ctx.adapter_info().name);
//! ```

pub mod buffer;
pub mod gemv;

use crate::error::{LludaError, Result};
use std::sync::OnceLock;
use wgpu;

/// GPU context holding device, queue, and adapter.
///
/// This struct encapsulates all wgpu resources needed for GPU acceleration.
/// It is created via the `init()` function and should be retained for the
/// lifetime of GPU operations.
#[derive(Debug)]
pub struct GpuContext {
    /// WGPU adapter (represents physical GPU)
    adapter: wgpu::Adapter,
    /// WGPU device (logical GPU interface)
    device: wgpu::Device,
    /// Command queue for GPU operations
    queue: wgpu::Queue,
}

impl GpuContext {
    /// Get adapter information (GPU name, backend, etc).
    pub fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }

    /// Get reference to the GPU device.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get reference to the command queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Get reference to the adapter.
    pub fn adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }
}

/// Initialize GPU context with wgpu.
///
/// This function:
/// 1. Creates a wgpu instance
/// 2. Requests a high-performance GPU adapter (prefers discrete GPU)
/// 3. Requests a device and command queue
/// 4. Returns a GpuContext with all resources
///
/// # Errors
///
/// Returns `LludaError::Msg` if:
/// - No suitable GPU adapter is found
/// - Device request fails (driver issues, insufficient resources)
///
/// # Example
///
/// ```rust,no_run
/// use lluda_inference::gpu::init;
///
/// match init() {
///     Ok(ctx) => println!("GPU initialized: {}", ctx.adapter_info().name),
///     Err(e) => eprintln!("GPU init failed: {}", e),
/// }
/// ```
pub fn init() -> Result<GpuContext> {
    // Create wgpu instance with default backend selection
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

    // Request adapter with high performance preference
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .map_err(|e| LludaError::Msg(format!("Failed to request GPU adapter: {}", e)))?;

    // Log adapter info for diagnostics
    let adapter_info = adapter.get_info();
    eprintln!(
        "GPU adapter selected: {} ({:?})",
        adapter_info.name, adapter_info.backend
    );

    // Request device and queue with default limits
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("lluda-inference-device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        memory_hints: wgpu::MemoryHints::default(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        trace: wgpu::Trace::Off,
    }))
    .map_err(|e| LludaError::Msg(format!("Failed to request GPU device: {}", e)))?;

    Ok(GpuContext {
        adapter,
        device,
        queue,
    })
}

/// Global GPU context singleton.
///
/// Lazily initialized on first access. If GPU initialization fails,
/// the singleton stores None and all subsequent calls will fail.
static GPU_CONTEXT: OnceLock<Option<GpuContext>> = OnceLock::new();

/// Get reference to the global GPU context singleton.
///
/// On first call, attempts to initialize GPU. If initialization fails,
/// stores None and all subsequent calls will return an error.
///
/// # Errors
///
/// Returns error if:
/// - GPU initialization failed (no suitable GPU, driver issues)
/// - GPU was previously initialized but failed
///
/// # Example
///
/// ```rust,no_run
/// use lluda_inference::gpu::get_context;
///
/// match get_context() {
///     Ok(ctx) => println!("GPU available: {}", ctx.adapter_info().name),
///     Err(e) => eprintln!("GPU not available: {}", e),
/// }
/// ```
pub fn get_context() -> Result<&'static GpuContext> {
    GPU_CONTEXT
        .get_or_init(|| match init() {
            Ok(ctx) => {
                eprintln!("GPU initialized: {:?}", ctx.adapter_info().name);
                Some(ctx)
            }
            Err(e) => {
                eprintln!("GPU init failed: {}, falling back to CPU", e);
                None
            }
        })
        .as_ref()
        .ok_or_else(|| LludaError::Msg("GPU not available".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_init() {
        // This test verifies that GPU initialization either succeeds
        // or fails gracefully. On machines without GPU, it should
        // return an error (not panic).
        match init() {
            Ok(ctx) => {
                // GPU available - verify context is usable
                let info = ctx.adapter_info();
                println!("GPU test: {} ({:?})", info.name, info.backend);
                assert!(!info.name.is_empty());

                // Verify we can access device and queue
                assert!(ctx.device().limits().max_texture_dimension_2d > 0);
            }
            Err(e) => {
                // No GPU available - verify error is reasonable
                let msg = format!("{}", e);
                assert!(
                    msg.contains("Failed to request GPU adapter") || msg.contains("Failed to request GPU device"),
                    "Unexpected error message: {}",
                    msg
                );
                println!("GPU test: No GPU available (expected on some machines)");
            }
        }
    }

    #[test]
    fn test_gpu_context_accessors() {
        // Only run if GPU is available
        if let Ok(ctx) = init() {
            // Test that all accessors return valid references
            let _device = ctx.device();
            let _queue = ctx.queue();
            let _adapter = ctx.adapter();
            let info = ctx.adapter_info();

            // Verify adapter info is populated
            assert!(!info.name.is_empty());
            println!("GPU accessors test passed: {}", info.name);
        } else {
            println!("GPU accessors test skipped: No GPU available");
        }
    }
}
