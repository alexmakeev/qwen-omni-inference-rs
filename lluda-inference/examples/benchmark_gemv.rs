//! GEMV benchmark: CPU vs GPU performance comparison
//!
//! Usage:
//!   cargo run --features gpu --example benchmark_gemv --release
//!
//! This benchmark compares CPU and GPU execution time for matrix-vector multiplication
//! with realistic model dimensions (e.g., 512×2048 for Linear layer forward pass).

use lluda_inference::bf16::BF16;
use lluda_inference::tensor::Tensor;
use std::time::Instant;

fn main() {
    println!("=== GEMV Benchmark: CPU vs GPU ===\n");

    // Test configurations: (M, N) dimensions
    let configs = vec![
        (512, 256, "Small (512×256)"),
        (512, 512, "Medium (512×512)"),
        (512, 1024, "Large (512×1024)"),
        (512, 2048, "XLarge (512×2048)"),
        (1024, 2048, "XXLarge (1024×2048)"),
    ];

    for (m, n, name) in configs {
        println!("--- {} ---", name);

        // Create BF16 test data
        let matrix_data: Vec<BF16> = (0..m * n).map(|i| BF16::from((i % 100) as f32)).collect();
        let vector_data: Vec<BF16> = (0..n).map(|i| BF16::from((i % 10) as f32)).collect();

        let matrix = Tensor::from_bf16(matrix_data, vec![m, n]).unwrap();
        let vector = Tensor::from_bf16(vector_data.clone(), vec![n]).unwrap();

        // GPU benchmark (if available)
        #[cfg(feature = "gpu")]
        {
            use lluda_inference::gpu;

            if gpu::get_context().is_ok() {
                // Warmup
                let _ = matrix.matmul(&vector);

                // Benchmark
                let start = Instant::now();
                let iterations = 100;
                for _ in 0..iterations {
                    let _ = matrix.matmul(&vector).unwrap();
                }
                let elapsed = start.elapsed();
                let avg_time = elapsed.as_micros() as f64 / iterations as f64;

                println!("  GPU: {:.2} μs/iter ({:.2} GFLOPS)", avg_time, gflops(m, n, avg_time));
            } else {
                println!("  GPU: Not available");
            }
        }

        #[cfg(not(feature = "gpu"))]
        {
            println!("  GPU: Disabled (compile with --features gpu)");
        }

        // CPU benchmark (force small matrix to avoid GPU dispatch)
        // Create F32 tensors to force CPU path
        let matrix_f32_data: Vec<f32> = (0..m * n).map(|i| (i % 100) as f32).collect();
        let vector_f32_data: Vec<f32> = (0..n).map(|i| (i % 10) as f32).collect();

        let matrix_cpu = Tensor::new(matrix_f32_data, vec![m, n]).unwrap();
        let vector_cpu = Tensor::new(vector_f32_data, vec![n]).unwrap();

        // Warmup
        let _ = matrix_cpu.matmul(&vector_cpu);

        // Benchmark
        let start = Instant::now();
        let iterations = 100;
        for _ in 0..iterations {
            let _ = matrix_cpu.matmul(&vector_cpu).unwrap();
        }
        let elapsed = start.elapsed();
        let avg_time = elapsed.as_micros() as f64 / iterations as f64;

        println!("  CPU: {:.2} μs/iter ({:.2} GFLOPS)", avg_time, gflops(m, n, avg_time));
        println!();
    }

    println!("=== Benchmark Complete ===");
}

/// Calculate GFLOPS (Giga Floating Point Operations Per Second)
///
/// GEMV operation: Y = A × X requires M*N multiplications and M*(N-1) additions
/// Total FLOPs: M * (2*N - 1) ≈ 2*M*N
fn gflops(m: usize, n: usize, time_us: f64) -> f64 {
    let flops = 2.0 * m as f64 * n as f64;
    let gflops = flops / (time_us * 1000.0); // μs to seconds, then to GFLOPS
    gflops
}
