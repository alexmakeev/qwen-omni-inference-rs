//! GEMV (General Matrix-Vector Multiply) compute pipeline.
//!
//! This module provides GPU-accelerated matrix-vector multiplication:
//!   output = matrix × vector
//!
//! Where:
//!   - matrix is M×N (row-major storage)
//!   - vector is N×1
//!   - output is M×1
//!
//! The implementation uses BF16 storage with F32 compute for numerical stability.
//! This is critical for transformer layers where Q/K/V projections and FFN operations
//! account for ~80% of inference time.

use crate::error::{LludaError, Result};
use crate::gpu::GpuContext;
use wgpu;

/// GEMV compute pipeline for GPU-accelerated matrix-vector multiplication.
///
/// This pipeline computes Y = A × X where:
///   - A is an M×N matrix (stored as packed BF16)
///   - X is an N×1 vector (stored as packed BF16, 2 values per u32)
///   - Y is an M×1 output vector (stored as unpacked BF16, 1 value per u32 in low 16 bits)
///
/// # Performance
///
/// - Each thread computes one output element (one row dot product)
/// - Workgroup size: 256 threads (optimal for AMD RDNA architecture)
/// - No shared memory optimization yet (focus on correctness first)
///
/// # Example
///
/// ```rust,no_run
/// use lluda_inference::gpu::{GpuContext, init};
/// use lluda_inference::gpu::gemv::GemvPipeline;
///
/// let ctx = init().expect("Failed to initialize GPU");
/// let pipeline = GemvPipeline::new(&ctx).expect("Failed to create pipeline");
///
/// // Use pipeline.execute() with matrix/vector buffers
/// ```
pub struct GemvPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GemvPipeline {
    /// Create a new GEMV compute pipeline.
    ///
    /// This loads the WGSL shader and creates the compute pipeline
    /// with appropriate bind group layout for matrix, vector, output buffers,
    /// and uniform parameters.
    ///
    /// # Errors
    ///
    /// Returns error if shader compilation or pipeline creation fails.
    pub fn new(ctx: &GpuContext) -> Result<Self> {
        // Load WGSL shader source
        let shader_source = include_str!("shaders/gemv.wgsl");

        // Create shader module
        let shader_module = ctx.device().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gemv-shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create bind group layout
        // Binding 0: matrix (storage buffer, read-only)
        // Binding 1: vector (storage buffer, read-only)
        // Binding 2: output (storage buffer, read-write)
        // Binding 3: uniforms (uniform buffer)
        let bind_group_layout =
            ctx.device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("gemv-bind-group-layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        // Create pipeline layout
        let pipeline_layout = ctx
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gemv-pipeline-layout"),
                bind_group_layouts: &[&bind_group_layout],
                immediate_size: 0,
            });

        // Create compute pipeline
        let pipeline = ctx
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gemv-pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("gemv_main"),
                compilation_options: Default::default(),
                cache: Default::default(),
            });

        Ok(GemvPipeline {
            pipeline,
            bind_group_layout,
        })
    }

    /// Execute GEMV operation: output = matrix × vector
    ///
    /// # Arguments
    ///
    /// * `ctx` - GPU context
    /// * `matrix_buf` - Matrix buffer (M×N elements, packed BF16, 2 per u32)
    /// * `vector_buf` - Vector buffer (N elements, packed BF16, 2 per u32)
    /// * `output_buf` - Output buffer (M elements, unpacked BF16, will be written)
    /// * `M` - Number of matrix rows (= output vector length)
    /// * `N` - Number of matrix columns (= input vector length)
    ///
    /// # Buffer sizes
    ///
    /// - matrix_buf: ceil(M*N/2) * 4 bytes (packed BF16: 2 values per u32)
    /// - vector_buf: ceil(N/2) * 4 bytes (packed BF16: 2 values per u32)
    /// - output_buf: M * 4 bytes (unpacked BF16: 1 value per u32 in low 16 bits)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - M or N is zero
    /// - Buffer sizes are invalid
    pub fn execute(
        &self,
        ctx: &GpuContext,
        matrix_buf: &wgpu::Buffer,
        vector_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        m: u32,
        n: u32,
    ) -> Result<()> {
        // Validate dimensions
        if m == 0 || n == 0 {
            return Err(LludaError::Msg(format!(
                "Invalid GEMV dimensions: M={}, N={} (must be > 0)",
                m, n
            )));
        }

        // Create uniform buffer with M, N
        let uniforms = [m, n];
        let uniform_buffer = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("gemv-uniforms"),
            size: 8, // 2 × u32 = 8 bytes
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Upload uniforms
        ctx.queue().write_buffer(
            &uniform_buffer,
            0,
            bytemuck::cast_slice(&uniforms),
        );

        // Create bind group
        let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gemv-bind-group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: matrix_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: vector_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder
        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gemv-encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gemv-pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch enough workgroups to cover all M rows
            // Each workgroup has 256 threads, so we need ceil(M/256) workgroups
            let workgroups_x = m.div_ceil(256);
            compute_pass.dispatch_workgroups(workgroups_x, 1, 1);
        }

        // Submit commands
        ctx.queue().submit(Some(encoder.finish()));

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bf16::BF16;
    use crate::gpu;

    /// Helper to pack BF16 values into u32 array (2 BF16 per u32)
    fn pack_bf16(data: &[BF16]) -> Vec<u32> {
        let mut packed = Vec::with_capacity((data.len() + 1) / 2);
        for chunk in data.chunks(2) {
            let low = chunk[0].to_bits() as u32;
            let high = if chunk.len() > 1 {
                chunk[1].to_bits() as u32
            } else {
                0
            };
            packed.push(low | (high << 16));
        }
        packed
    }

    #[test]
    fn test_gemv_small() {
        // Only run if GPU is available
        let ctx = match gpu::init() {
            Ok(ctx) => ctx,
            Err(_) => {
                println!("Skipping GPU test: No GPU available");
                return;
            }
        };

        // Create pipeline
        let pipeline = GemvPipeline::new(&ctx).expect("Failed to create GEMV pipeline");

        // Test matrix: 4×3
        // [ 1.0  2.0  3.0 ]
        // [ 4.0  5.0  6.0 ]
        // [ 7.0  8.0  9.0 ]
        // [10.0 11.0 12.0 ]
        let matrix_data = vec![
            BF16::from(1.0f32),
            BF16::from(2.0f32),
            BF16::from(3.0f32),
            BF16::from(4.0f32),
            BF16::from(5.0f32),
            BF16::from(6.0f32),
            BF16::from(7.0f32),
            BF16::from(8.0f32),
            BF16::from(9.0f32),
            BF16::from(10.0f32),
            BF16::from(11.0f32),
            BF16::from(12.0f32),
        ];

        // Test vector: 3×1
        // [1.0]
        // [2.0]
        // [3.0]
        let vector_data = vec![BF16::from(1.0f32), BF16::from(2.0f32), BF16::from(3.0f32)];

        // Expected output: 4×1
        // [1*1 + 2*2 + 3*3] = [14.0]
        // [4*1 + 5*2 + 6*3] = [32.0]
        // [7*1 + 8*2 + 9*3] = [50.0]
        // [10*1 + 11*2 + 12*3] = [68.0]
        let expected = vec![14.0f32, 32.0, 50.0, 68.0];

        let m = 4u32;
        let n = 3u32;

        // Pack data
        let matrix_packed = pack_bf16(&matrix_data);
        let vector_packed = pack_bf16(&vector_data);


        // Create GPU buffers
        let matrix_bytes: Vec<u8> = matrix_packed
            .iter()
            .flat_map(|u| u.to_le_bytes())
            .collect();
        let vector_bytes: Vec<u8> = vector_packed
            .iter()
            .flat_map(|u| u.to_le_bytes())
            .collect();

        let matrix_buf = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("test-matrix"),
            size: matrix_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue().write_buffer(&matrix_buf, 0, &matrix_bytes);

        let vector_buf = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("test-vector"),
            size: vector_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue().write_buffer(&vector_buf, 0, &vector_bytes);

        // Allocate output buffer (one u32 per BF16, not packed, to avoid RMW issues)
        let output_size = (m as usize * 4) as u64;
        let output_buf = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("test-output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Execute GEMV
        pipeline
            .execute(&ctx, &matrix_buf, &vector_buf, &output_buf, m, n)
            .expect("GEMV execution failed");

        // Wait for GPU to complete
        let _ = ctx.device().poll(wgpu::PollType::wait_indefinitely());

        // Read back results
        let staging_buf = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: output_buf.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy-encoder"),
            });
        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, output_buf.size());
        ctx.queue().submit(Some(encoder.finish()));

        // Map and read
        let buffer_slice = staging_buf.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        let _ = ctx.device().poll(wgpu::PollType::wait_indefinitely());
        receiver.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let output_bytes: Vec<u8> = data.to_vec();
        drop(data);
        staging_buf.unmap();

        // Read output (unpacked: one u32 per BF16, value in low 16 bits)
        let output_u32s: Vec<u32> = output_bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        // Extract BF16 from low 16 bits of each u32
        let output_bf16: Vec<BF16> = output_u32s
            .iter()
            .map(|u| BF16::from_bits((u & 0xFFFF) as u16))
            .collect();

        // Compare results (allow small BF16 precision difference)
        for (i, (got, exp)) in output_bf16.iter().zip(expected.iter()).enumerate() {
            let got_f32: f32 = (*got).into();
            let diff = (got_f32 - exp).abs();
            assert!(
                diff < 0.1,
                "Output[{}] mismatch: got {}, expected {} (diff {})",
                i,
                got_f32,
                exp,
                diff
            );
        }

        println!("GEMV small test passed!");
    }

    #[test]
    fn test_gemv_validation() {
        let ctx = match gpu::init() {
            Ok(ctx) => ctx,
            Err(_) => {
                println!("Skipping GPU test: No GPU available");
                return;
            }
        };

        let pipeline = GemvPipeline::new(&ctx).expect("Failed to create pipeline");

        // Create dummy buffers
        let dummy_buf = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("dummy"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Test zero dimensions
        let result = pipeline.execute(&ctx, &dummy_buf, &dummy_buf, &dummy_buf, 0, 10);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must be > 0"));

        let result = pipeline.execute(&ctx, &dummy_buf, &dummy_buf, &dummy_buf, 10, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must be > 0"));

        println!("GEMV validation test passed!");
    }
}
