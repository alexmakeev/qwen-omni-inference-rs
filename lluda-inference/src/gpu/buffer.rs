//! GPU buffer management for tensor data.
//!
//! This module provides abstractions for uploading and downloading tensor data
//! between CPU and GPU memory. It handles BF16 and F32 data types with proper
//! byte conversion and buffer management.

use crate::bf16::BF16;
use crate::error::{LludaError, Result};
use crate::gpu::GpuContext;
use crate::tensor::DType;
use std::sync::Arc;
use wgpu;

/// GPU buffer wrapper for tensor data.
///
/// Encapsulates a wgpu buffer along with metadata about the data type
/// and size. Provides methods for CPU <-> GPU data transfer.
#[derive(Debug, Clone)]
pub struct GpuTensorBuffer {
    buffer: Arc<wgpu::Buffer>,
    size: usize,
    dtype: DType,
}

impl GpuTensorBuffer {
    /// Upload BF16 data from CPU to GPU.
    ///
    /// # Arguments
    ///
    /// * `ctx` - GPU context with device and queue
    /// * `data` - CPU BF16 data to upload
    /// * `shape` - Tensor shape (for validation)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Data length doesn't match shape
    /// - Buffer creation fails
    pub fn from_cpu_bf16(ctx: &GpuContext, data: &[BF16], shape: &[usize]) -> Result<Self> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(LludaError::Msg(format!(
                "Data length {} doesn't match shape {:?} (expected {})",
                data.len(),
                shape,
                expected_len
            )));
        }

        // BF16 is 2 bytes (stored as u16)
        let size = data.len() * 2;

        // Create GPU buffer with appropriate usage flags
        let buffer = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("tensor-bf16-buffer"),
            size: size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Convert BF16 to bytes (u16 representation)
        let bytes: Vec<u8> = data
            .iter()
            .flat_map(|bf| bf.to_bits().to_le_bytes())
            .collect();

        // Upload to GPU
        ctx.queue().write_buffer(&buffer, 0, &bytes);

        Ok(GpuTensorBuffer {
            buffer: Arc::new(buffer),
            size,
            dtype: DType::BF16,
        })
    }

    /// Upload F32 data from CPU to GPU.
    ///
    /// # Arguments
    ///
    /// * `ctx` - GPU context with device and queue
    /// * `data` - CPU F32 data to upload
    /// * `shape` - Tensor shape (for validation)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Data length doesn't match shape
    /// - Buffer creation fails
    pub fn from_cpu_f32(ctx: &GpuContext, data: &[f32], shape: &[usize]) -> Result<Self> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(LludaError::Msg(format!(
                "Data length {} doesn't match shape {:?} (expected {})",
                data.len(),
                shape,
                expected_len
            )));
        }

        // F32 is 4 bytes
        let size = data.len() * 4;

        // Create GPU buffer with appropriate usage flags
        let buffer = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("tensor-f32-buffer"),
            size: size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Convert F32 to bytes
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

        // Upload to GPU
        ctx.queue().write_buffer(&buffer, 0, &bytes);

        Ok(GpuTensorBuffer {
            buffer: Arc::new(buffer),
            size,
            dtype: DType::F32,
        })
    }

    /// Download BF16 data from GPU to CPU.
    ///
    /// # Arguments
    ///
    /// * `ctx` - GPU context with device and queue
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Buffer type is not BF16
    /// - Download fails
    pub fn to_cpu_bf16(&self, ctx: &GpuContext) -> Result<Vec<BF16>> {
        if self.dtype != DType::BF16 {
            return Err(LludaError::Msg(format!(
                "Cannot download BF16: buffer is {}",
                self.dtype
            )));
        }

        // Create staging buffer for readback
        let staging_buffer = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging-buffer"),
            size: self.size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create command encoder and copy GPU buffer to staging
        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy-encoder"),
            });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_buffer, 0, self.size as u64);
        ctx.queue().submit(Some(encoder.finish()));

        // Map staging buffer and read data
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        // Poll device until mapping is complete (wait indefinitely)
        let _ = ctx.device().poll(wgpu::PollType::wait_indefinitely());

        // Wait for mapping to complete
        receiver
            .recv()
            .map_err(|e| LludaError::Msg(format!("Failed to receive map result: {}", e)))?
            .map_err(|e| LludaError::Msg(format!("Failed to map buffer: {}", e)))?;

        // Read data from mapped buffer
        let data = buffer_slice.get_mapped_range();
        let bytes: Vec<u8> = data.to_vec();
        drop(data);
        staging_buffer.unmap();

        // Convert bytes back to BF16
        let num_elements = self.size / 2;
        let mut result = Vec::with_capacity(num_elements);
        for i in 0..num_elements {
            let u16_val = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
            result.push(BF16::from_bits(u16_val));
        }

        Ok(result)
    }

    /// Download F32 data from GPU to CPU.
    ///
    /// # Arguments
    ///
    /// * `ctx` - GPU context with device and queue
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Buffer type is not F32
    /// - Download fails
    pub fn to_cpu_f32(&self, ctx: &GpuContext) -> Result<Vec<f32>> {
        if self.dtype != DType::F32 {
            return Err(LludaError::Msg(format!(
                "Cannot download F32: buffer is {}",
                self.dtype
            )));
        }

        // Create staging buffer for readback
        let staging_buffer = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging-buffer"),
            size: self.size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create command encoder and copy GPU buffer to staging
        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy-encoder"),
            });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_buffer, 0, self.size as u64);
        ctx.queue().submit(Some(encoder.finish()));

        // Map staging buffer and read data
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        // Poll device until mapping is complete (wait indefinitely)
        let _ = ctx.device().poll(wgpu::PollType::wait_indefinitely());

        // Wait for mapping to complete
        receiver
            .recv()
            .map_err(|e| LludaError::Msg(format!("Failed to receive map result: {}", e)))?
            .map_err(|e| LludaError::Msg(format!("Failed to map buffer: {}", e)))?;

        // Read data from mapped buffer
        let data = buffer_slice.get_mapped_range();
        let bytes: Vec<u8> = data.to_vec();
        drop(data);
        staging_buffer.unmap();

        // Convert bytes back to F32
        let num_elements = self.size / 4;
        let mut result = Vec::with_capacity(num_elements);
        for i in 0..num_elements {
            let f32_bytes = [
                bytes[i * 4],
                bytes[i * 4 + 1],
                bytes[i * 4 + 2],
                bytes[i * 4 + 3],
            ];
            result.push(f32::from_le_bytes(f32_bytes));
        }

        Ok(result)
    }

    /// Create a GpuTensorBuffer from an existing Arc<wgpu::Buffer>.
    ///
    /// This is useful for reconstructing a buffer wrapper from tensor data.
    ///
    /// # Arguments
    ///
    /// * `buffer` - Shared reference to GPU buffer
    /// * `size` - Size in bytes
    /// * `dtype` - Data type (BF16 or F32)
    pub fn from_arc_buffer(buffer: Arc<wgpu::Buffer>, size: usize, dtype: DType) -> Self {
        GpuTensorBuffer {
            buffer,
            size,
            dtype,
        }
    }

    /// Get the underlying wgpu buffer as Arc (for sharing).
    pub fn buffer(&self) -> Arc<wgpu::Buffer> {
        Arc::clone(&self.buffer)
    }

    /// Get data type of this buffer.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get buffer size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu;

    #[test]
    fn test_bf16_roundtrip() {
        // Only run if GPU is available
        let ctx = match gpu::init() {
            Ok(ctx) => ctx,
            Err(_) => {
                println!("Skipping GPU test: No GPU available");
                return;
            }
        };

        // Create test data
        let data = vec![
            BF16::from(1.0f32),
            BF16::from(2.5f32),
            BF16::from(-3.75f32),
            BF16::from(0.0f32),
        ];
        let shape = vec![2, 2];

        // Upload to GPU
        let gpu_buffer = GpuTensorBuffer::from_cpu_bf16(&ctx, &data, &shape)
            .expect("Failed to upload BF16 data");

        assert_eq!(gpu_buffer.dtype(), DType::BF16);
        assert_eq!(gpu_buffer.size(), 8); // 4 elements * 2 bytes

        // Download from GPU
        let downloaded = gpu_buffer
            .to_cpu_bf16(&ctx)
            .expect("Failed to download BF16 data");

        // Verify data matches
        assert_eq!(downloaded.len(), data.len());
        for (orig, dl) in data.iter().zip(downloaded.iter()) {
            assert_eq!(orig.to_bits(), dl.to_bits(), "BF16 roundtrip mismatch");
        }
    }

    #[test]
    fn test_f32_roundtrip() {
        // Only run if GPU is available
        let ctx = match gpu::init() {
            Ok(ctx) => ctx,
            Err(_) => {
                println!("Skipping GPU test: No GPU available");
                return;
            }
        };

        // Create test data
        let data = vec![1.0f32, 2.5, -3.75, 0.0, 100.5, -200.25];
        let shape = vec![2, 3];

        // Upload to GPU
        let gpu_buffer =
            GpuTensorBuffer::from_cpu_f32(&ctx, &data, &shape).expect("Failed to upload F32 data");

        assert_eq!(gpu_buffer.dtype(), DType::F32);
        assert_eq!(gpu_buffer.size(), 24); // 6 elements * 4 bytes

        // Download from GPU
        let downloaded = gpu_buffer
            .to_cpu_f32(&ctx)
            .expect("Failed to download F32 data");

        // Verify data matches
        assert_eq!(downloaded.len(), data.len());
        for (orig, dl) in data.iter().zip(downloaded.iter()) {
            assert_eq!(orig, dl, "F32 roundtrip mismatch");
        }
    }

    #[test]
    fn test_shape_validation() {
        let ctx = match gpu::init() {
            Ok(ctx) => ctx,
            Err(_) => {
                println!("Skipping GPU test: No GPU available");
                return;
            }
        };

        // Create mismatched data and shape
        let data = vec![BF16::from(1.0f32), BF16::from(2.0f32)];
        let wrong_shape = vec![2, 2]; // Expects 4 elements, but data has 2

        let result = GpuTensorBuffer::from_cpu_bf16(&ctx, &data, &wrong_shape);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("doesn't match"));
    }

    #[test]
    fn test_type_mismatch_download() {
        let ctx = match gpu::init() {
            Ok(ctx) => ctx,
            Err(_) => {
                println!("Skipping GPU test: No GPU available");
                return;
            }
        };

        // Upload F32 data
        let data = vec![1.0f32, 2.0];
        let shape = vec![2];
        let gpu_buffer = GpuTensorBuffer::from_cpu_f32(&ctx, &data, &shape).unwrap();

        // Try to download as BF16 (should fail)
        let result = gpu_buffer.to_cpu_bf16(&ctx);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("buffer is F32"));
    }
}
