//! GEMM (General Matrix-Matrix Multiply) compute pipeline.
//!
//! This module provides GPU-accelerated matrix-matrix multiplication:
//!   C = A × B
//!
//! Where:
//!   - A is M×K (row-major storage)
//!   - B is K×N (row-major storage)
//!   - C is M×N (row-major storage)
//!
//! The implementation uses BF16 storage with F32 compute for numerical stability.
//! This is critical for transformer layers where Linear projections (Q/K/V, FFN)
//! account for ~80% of inference time.

use crate::error::{LludaError, Result};
use crate::gpu::GpuContext;
use wgpu;

/// GEMM compute pipeline for GPU-accelerated matrix-matrix multiplication.
///
/// This pipeline computes C = A × B where:
///   - A is an M×K matrix (stored as packed BF16, 2 values per u32)
///   - B is a K×N matrix (stored as packed BF16, 2 values per u32)
///   - C is an M×N output matrix (stored as unpacked BF16, 1 value per u32 in low 16 bits)
///
/// # Performance
///
/// - Tiled algorithm: 16×16 tiles with shared memory
/// - Each workgroup computes one output tile (16×16 elements)
/// - Workgroup size: 256 threads (16×16, optimal for AMD RDNA architecture)
/// - Expected speedup: 5-10x for large matrices (>1024×1024)
///
/// # Example
///
/// ```rust,no_run
/// use lluda_inference::gpu::{GpuContext, init};
/// use lluda_inference::gpu::gemm::GemmPipeline;
///
/// let ctx = init().expect("Failed to initialize GPU");
/// let pipeline = GemmPipeline::new(&ctx).expect("Failed to create pipeline");
///
/// // Use pipeline.execute() with matrix buffers
/// ```
pub struct GemmPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GemmPipeline {
    /// Create a new GEMM compute pipeline.
    ///
    /// This loads the WGSL shader and creates the compute pipeline
    /// with appropriate bind group layout for matrix A, B, C buffers,
    /// and uniform parameters.
    ///
    /// # Errors
    ///
    /// Returns error if shader compilation or pipeline creation fails.
    pub fn new(ctx: &GpuContext) -> Result<Self> {
        // Load WGSL shader source
        let shader_source = include_str!("shaders/gemm.wgsl");

        // Create shader module
        let shader_module = ctx.device().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gemm-shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create bind group layout
        // Binding 0: matrix_a (storage buffer, read-only)
        // Binding 1: matrix_b (storage buffer, read-only)
        // Binding 2: matrix_c (storage buffer, read-write)
        // Binding 3: uniforms (uniform buffer)
        let bind_group_layout =
            ctx.device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("gemm-bind-group-layout"),
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
                label: Some("gemm-pipeline-layout"),
                bind_group_layouts: &[&bind_group_layout],
                immediate_size: 0,
            });

        // Create compute pipeline
        let pipeline = ctx
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gemm-pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("gemm_main"),
                compilation_options: Default::default(),
                cache: Default::default(),
            });

        Ok(GemmPipeline {
            pipeline,
            bind_group_layout,
        })
    }

    /// Execute GEMM operation: C = A × B
    ///
    /// # Arguments
    ///
    /// * `ctx` - GPU context
    /// * `a_buf` - Matrix A buffer (M×K elements, packed BF16, 2 per u32)
    /// * `b_buf` - Matrix B buffer (K×N elements, packed BF16, 2 per u32)
    /// * `c_buf` - Matrix C buffer (M×N elements, unpacked BF16, will be written)
    /// * `m` - Number of rows in A (= rows in C)
    /// * `k` - Number of cols in A (= rows in B)
    /// * `n` - Number of cols in B (= cols in C)
    ///
    /// # Buffer sizes
    ///
    /// - a_buf: ceil(M*K/2) * 4 bytes (packed BF16: 2 values per u32)
    /// - b_buf: ceil(K*N/2) * 4 bytes (packed BF16: 2 values per u32)
    /// - c_buf: M*N * 4 bytes (unpacked BF16: 1 value per u32 in low 16 bits)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - M, K, or N is zero
    /// - Buffer sizes are invalid
    pub fn execute(
        &self,
        ctx: &GpuContext,
        a_buf: &wgpu::Buffer,
        b_buf: &wgpu::Buffer,
        c_buf: &wgpu::Buffer,
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<()> {
        // Validate dimensions
        if m == 0 || k == 0 || n == 0 {
            return Err(LludaError::Msg(format!(
                "Invalid GEMM dimensions: M={}, K={}, N={} (must be > 0)",
                m, k, n
            )));
        }

        // Create uniform buffer with M, K, N
        let uniforms = [m, k, n];
        let uniform_buffer = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("gemm-uniforms"),
            size: 12, // 3 × u32 = 12 bytes
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Upload uniforms
        ctx.queue()
            .write_buffer(&uniform_buffer, 0, bytemuck::cast_slice(&uniforms));

        // Create bind group
        let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gemm-bind-group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buf.as_entire_binding(),
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
                label: Some("gemm-encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gemm-pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups: each workgroup computes a 16×16 tile of output
            // We need ceil(N/16) × ceil(M/16) workgroups
            let workgroups_x = n.div_ceil(16);
            let workgroups_y = m.div_ceil(16);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Submit commands
        ctx.queue().submit(Some(encoder.finish()));

        Ok(())
    }
}

/// High-level wrapper for GEMM operation: C = A × B
///
/// This function handles BF16 tensor data upload, GPU computation, and result download.
/// Used for accelerating Linear layer forward passes (matrix-matrix multiplication).
///
/// # Arguments
///
/// * `ctx` - GPU context
/// * `matrix_a` - Matrix A tensor (M×K), must be BF16
/// * `matrix_b` - Matrix B tensor (K×N), must be BF16
///
/// # Returns
///
/// Output tensor C (M×N), BF16 dtype
///
/// # Errors
///
/// Returns error if:
/// - Shapes are incompatible (A cols != B rows)
/// - Data types are not BF16
/// - GPU buffer operations fail
///
/// # Example
///
/// ```rust,no_run
/// use lluda_inference::gpu::{get_context, gemm::gemm_forward};
/// use lluda_inference::tensor::Tensor;
/// use lluda_inference::bf16::BF16;
///
/// let ctx = get_context().unwrap();
///
/// // Create matrices: A (2×3), B (3×4)
/// let a = Tensor::from_bf16(vec![BF16::from(1.0); 6], vec![2, 3]).unwrap();
/// let b = Tensor::from_bf16(vec![BF16::from(1.0); 12], vec![3, 4]).unwrap();
///
/// let result = gemm_forward(ctx, &a, &b).unwrap();
/// // result shape: [2, 4]
/// ```
pub fn gemm_forward(
    ctx: &GpuContext,
    matrix_a: &crate::tensor::Tensor,
    matrix_b: &crate::tensor::Tensor,
) -> Result<crate::tensor::Tensor> {
    use crate::bf16::BF16;
    use crate::tensor::DType;

    let profiling_enabled = std::env::var("PROFILE").is_ok();
    let total_start = std::time::Instant::now();

    // Validate shapes
    let a_shape = matrix_a.shape();
    let b_shape = matrix_b.shape();

    // Both matrices must be 2D
    if a_shape.len() != 2 {
        return Err(LludaError::Msg(format!(
            "GEMM: matrix A must be 2D, got shape {:?}",
            a_shape
        )));
    }
    if b_shape.len() != 2 {
        return Err(LludaError::Msg(format!(
            "GEMM: matrix B must be 2D, got shape {:?}",
            b_shape
        )));
    }

    let m = a_shape[0];
    let k_a = a_shape[1];
    let k_b = b_shape[0];
    let n = b_shape[1];

    // Validate inner dimensions
    if k_a != k_b {
        return Err(LludaError::Msg(format!(
            "GEMM: shape mismatch: A {}×{}, B {}×{} (inner dimensions must match)",
            m, k_a, k_b, n
        )));
    }
    let k = k_a;

    // Both tensors must be BF16
    if matrix_a.dtype() != DType::BF16 || matrix_b.dtype() != DType::BF16 {
        return Err(LludaError::Msg(format!(
            "GEMM: requires BF16 tensors, got A={}, B={}",
            matrix_a.dtype(),
            matrix_b.dtype()
        )));
    }

    // Extract BF16 data
    let extract_start = std::time::Instant::now();
    let a_data = matrix_a.to_vec_bf16();
    let b_data = matrix_b.to_vec_bf16();
    if profiling_enabled {
        eprintln!(
            "[GPU]     Extract BF16 data: {:.2}ms",
            extract_start.elapsed().as_secs_f64() * 1000.0
        );
    }

    // Pack BF16 data (2 values per u32)
    let pack_start = std::time::Instant::now();
    let a_packed = pack_bf16(&a_data);
    let b_packed = pack_bf16(&b_data);
    if profiling_enabled {
        eprintln!(
            "[GPU]     Pack BF16 data: {:.2}ms",
            pack_start.elapsed().as_secs_f64() * 1000.0
        );
    }

    // Create GPU buffers
    let buffer_create_start = std::time::Instant::now();
    let a_bytes: Vec<u8> = a_packed.iter().flat_map(|u| u.to_le_bytes()).collect();
    let b_bytes: Vec<u8> = b_packed.iter().flat_map(|u| u.to_le_bytes()).collect();

    let a_buf = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("gemm-matrix-a"),
        size: a_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let b_buf = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("gemm-matrix-b"),
        size: b_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output_size = (m * n * 4) as u64; // Unpacked: 1 BF16 per u32
    let c_buf = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("gemm-matrix-c"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if profiling_enabled {
        eprintln!(
            "[GPU]     Create buffers: {:.2}ms",
            buffer_create_start.elapsed().as_secs_f64() * 1000.0
        );
    }

    // Upload data to GPU
    let upload_start = std::time::Instant::now();
    ctx.queue().write_buffer(&a_buf, 0, &a_bytes);
    ctx.queue().write_buffer(&b_buf, 0, &b_bytes);
    if profiling_enabled {
        eprintln!(
            "[GPU]     Upload to GPU ({} + {} bytes): {:.2}ms",
            a_bytes.len(),
            b_bytes.len(),
            upload_start.elapsed().as_secs_f64() * 1000.0
        );
    }

    // Create pipeline and execute
    let pipeline_start = std::time::Instant::now();
    let pipeline = GemmPipeline::new(ctx)?;
    if profiling_enabled {
        eprintln!(
            "[GPU]     Create pipeline: {:.2}ms",
            pipeline_start.elapsed().as_secs_f64() * 1000.0
        );
    }

    let execute_start = std::time::Instant::now();
    pipeline.execute(ctx, &a_buf, &b_buf, &c_buf, m as u32, k as u32, n as u32)?;
    if profiling_enabled {
        eprintln!(
            "[GPU]     Submit GPU kernel: {:.2}ms",
            execute_start.elapsed().as_secs_f64() * 1000.0
        );
    }

    // Wait for GPU completion
    let wait_start = std::time::Instant::now();
    let _ = ctx.device().poll(wgpu::PollType::wait_indefinitely());
    if profiling_enabled {
        eprintln!(
            "[GPU]     Wait for GPU completion: {:.2}ms",
            wait_start.elapsed().as_secs_f64() * 1000.0
        );
    }

    // Read back results
    let download_start = std::time::Instant::now();
    let staging_buf = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("gemm-staging"),
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gemm-copy"),
        });
    encoder.copy_buffer_to_buffer(&c_buf, 0, &staging_buf, 0, output_size);
    ctx.queue().submit(Some(encoder.finish()));

    // Map and read output
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
    if profiling_enabled {
        eprintln!(
            "[GPU]     Download from GPU ({} bytes): {:.2}ms",
            output_bytes.len(),
            download_start.elapsed().as_secs_f64() * 1000.0
        );
    }

    // Unpack output (one u32 per BF16, value in low 16 bits)
    let unpack_start = std::time::Instant::now();
    let output_u32s: Vec<u32> = output_bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let output_bf16: Vec<BF16> = output_u32s
        .iter()
        .map(|u| BF16::from_bits((u & 0xFFFF) as u16))
        .collect();
    if profiling_enabled {
        eprintln!(
            "[GPU]     Unpack output: {:.2}ms",
            unpack_start.elapsed().as_secs_f64() * 1000.0
        );
        eprintln!(
            "[GPU]     Total GPU operation time: {:.2}ms",
            total_start.elapsed().as_secs_f64() * 1000.0
        );
    }

    // Return as BF16 tensor with shape [M, N]
    crate::tensor::Tensor::from_bf16(output_bf16, vec![m, n])
}

/// Helper to pack BF16 values into u32 array (2 BF16 per u32).
fn pack_bf16(data: &[crate::bf16::BF16]) -> Vec<u32> {
    let mut packed = Vec::with_capacity(data.len().div_ceil(2));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bf16::BF16;
    use crate::gpu;

    #[test]
    fn test_gemm_small() {
        // Only run if GPU is available
        let ctx = match gpu::init() {
            Ok(ctx) => ctx,
            Err(_) => {
                println!("Skipping GPU test: No GPU available");
                return;
            }
        };

        // Create pipeline
        let pipeline = GemmPipeline::new(&ctx).expect("Failed to create GEMM pipeline");

        // Test matrices:
        // A: 4×3
        // [ 1.0  2.0  3.0 ]
        // [ 4.0  5.0  6.0 ]
        // [ 7.0  8.0  9.0 ]
        // [10.0 11.0 12.0 ]
        let a_data = vec![
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

        // B: 3×5
        // [1.0 0.0 0.0 0.0 1.0]
        // [0.0 1.0 0.0 1.0 0.0]
        // [0.0 0.0 1.0 0.0 0.0]
        let b_data = vec![
            BF16::from(1.0f32),
            BF16::from(0.0f32),
            BF16::from(0.0f32),
            BF16::from(0.0f32),
            BF16::from(1.0f32),
            BF16::from(0.0f32),
            BF16::from(1.0f32),
            BF16::from(0.0f32),
            BF16::from(1.0f32),
            BF16::from(0.0f32),
            BF16::from(0.0f32),
            BF16::from(0.0f32),
            BF16::from(1.0f32),
            BF16::from(0.0f32),
            BF16::from(0.0f32),
        ];

        // Expected output: 4×5
        // C = A × B
        // [1.0  2.0  3.0  2.0  1.0]
        // [4.0  5.0  6.0  5.0  4.0]
        // [7.0  8.0  9.0  8.0  7.0]
        // [10.0 11.0 12.0 11.0 10.0]
        let expected = vec![
            1.0f32, 2.0, 3.0, 2.0, 1.0, 4.0, 5.0, 6.0, 5.0, 4.0, 7.0, 8.0, 9.0, 8.0, 7.0, 10.0,
            11.0, 12.0, 11.0, 10.0,
        ];

        let m = 4u32;
        let k = 3u32;
        let n = 5u32;

        // Pack data
        let a_packed = pack_bf16(&a_data);
        let b_packed = pack_bf16(&b_data);

        // Create GPU buffers
        let a_bytes: Vec<u8> = a_packed.iter().flat_map(|u| u.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b_packed.iter().flat_map(|u| u.to_le_bytes()).collect();

        let a_buf = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("test-matrix-a"),
            size: a_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue().write_buffer(&a_buf, 0, &a_bytes);

        let b_buf = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("test-matrix-b"),
            size: b_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue().write_buffer(&b_buf, 0, &b_bytes);

        // Allocate output buffer (one u32 per BF16, not packed)
        let output_size = (m * n * 4) as u64;
        let c_buf = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("test-matrix-c"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Execute GEMM
        pipeline
            .execute(&ctx, &a_buf, &b_buf, &c_buf, m, k, n)
            .expect("GEMM execution failed");

        // Wait for GPU to complete
        let _ = ctx.device().poll(wgpu::PollType::wait_indefinitely());

        // Read back results
        let staging_buf = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: c_buf.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy-encoder"),
            });
        encoder.copy_buffer_to_buffer(&c_buf, 0, &staging_buf, 0, c_buf.size());
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

        println!("GEMM small test passed!");
    }

    #[test]
    fn test_gemm_validation() {
        let ctx = match gpu::init() {
            Ok(ctx) => ctx,
            Err(_) => {
                println!("Skipping GPU test: No GPU available");
                return;
            }
        };

        let pipeline = GemmPipeline::new(&ctx).expect("Failed to create pipeline");

        // Create dummy buffers
        let dummy_buf = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("dummy"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Test zero dimensions
        let result = pipeline.execute(&ctx, &dummy_buf, &dummy_buf, &dummy_buf, 0, 10, 10);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must be > 0"));

        let result = pipeline.execute(&ctx, &dummy_buf, &dummy_buf, &dummy_buf, 10, 0, 10);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must be > 0"));

        let result = pipeline.execute(&ctx, &dummy_buf, &dummy_buf, &dummy_buf, 10, 10, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must be > 0"));

        println!("GEMM validation test passed!");
    }
}
