//! Validate GPU GEMV correctness against CPU reference
//!
//! Usage:
//!   cargo run --features gpu --example validate_gemv
//!
//! This example verifies that GPU GEMV produces identical results to CPU.

use lluda_inference::bf16::BF16;
use lluda_inference::tensor::Tensor;

fn main() {
    println!("=== GEMV Validation: GPU vs CPU ===\n");

    // Test matrix: 128×64 (large enough to trigger GPU)
    let m = 128;
    let n = 64;

    println!("Matrix size: {}×{}\n", m, n);

    // Create deterministic test data
    let matrix_data: Vec<BF16> = (0..m * n)
        .map(|i| BF16::from(((i * 7 + 13) % 100) as f32 / 10.0))
        .collect();
    let vector_data: Vec<BF16> = (0..n)
        .map(|i| BF16::from(((i * 3 + 5) % 20) as f32 / 10.0))
        .collect();

    let matrix_bf16 = Tensor::from_bf16(matrix_data.clone(), vec![m, n]).unwrap();
    let vector_bf16 = Tensor::from_bf16(vector_data.clone(), vec![n]).unwrap();

    // GPU result (if available)
    #[cfg(feature = "gpu")]
    let gpu_result = {
        use lluda_inference::gpu;

        match gpu::get_context() {
            Ok(_) => {
                println!("GPU available, running GPU GEMV...");
                Some(matrix_bf16.matmul(&vector_bf16).unwrap())
            }
            Err(e) => {
                println!("GPU not available: {}", e);
                None
            }
        }
    };

    #[cfg(not(feature = "gpu"))]
    let gpu_result: Option<Tensor> = None;

    // CPU reference (force CPU by using F32 tensors)
    println!("Running CPU reference...");
    let matrix_f32: Vec<f32> = matrix_data.iter().map(|&bf| bf.into()).collect();
    let vector_f32: Vec<f32> = vector_data.iter().map(|&bf| bf.into()).collect();

    let matrix_cpu = Tensor::new(matrix_f32, vec![m, n]).unwrap();
    let vector_cpu = Tensor::new(vector_f32, vec![n]).unwrap();
    let cpu_result = matrix_cpu.matmul(&vector_cpu).unwrap();

    // Compare results
    if let Some(gpu_result) = gpu_result {
        println!("\nComparing GPU vs CPU results...");

        let gpu_data = gpu_result.to_vec_f32();
        let cpu_data = cpu_result.to_vec_f32();

        assert_eq!(gpu_data.len(), cpu_data.len(), "Result length mismatch");

        let mut max_diff = 0.0f32;
        let mut max_rel_diff = 0.0f32;
        let mut num_mismatches = 0;

        for i in 0..gpu_data.len() {
            let diff = (gpu_data[i] - cpu_data[i]).abs();
            let rel_diff = if cpu_data[i].abs() > 1e-6 {
                diff / cpu_data[i].abs()
            } else {
                0.0
            };

            max_diff = max_diff.max(diff);
            max_rel_diff = max_rel_diff.max(rel_diff);

            if diff > 0.1 {
                // Allow small differences due to BF16 precision
                num_mismatches += 1;
                if num_mismatches <= 5 {
                    println!(
                        "  [{}] GPU={:.6}, CPU={:.6}, diff={:.6}, rel={:.2}%",
                        i,
                        gpu_data[i],
                        cpu_data[i],
                        diff,
                        rel_diff * 100.0
                    );
                }
            }
        }

        println!("\nValidation Summary:");
        println!("  Max absolute diff: {:.6}", max_diff);
        println!("  Max relative diff: {:.2}%", max_rel_diff * 100.0);
        println!("  Mismatches (>0.1): {}/{}", num_mismatches, gpu_data.len());

        if max_rel_diff < 0.01 {
            // Less than 1% relative error
            println!("\n✓ GPU GEMV matches CPU reference!");
        } else if max_rel_diff < 0.05 {
            // Less than 5% relative error (acceptable for BF16)
            println!(
                "\n✓ GPU GEMV close to CPU reference (within BF16 tolerance)"
            );
        } else {
            println!("\n✗ GPU GEMV differs significantly from CPU!");
            println!("  (Max relative error: {:.2}% exceeds 5% threshold)", max_rel_diff * 100.0);
            std::process::exit(1);
        }
    } else {
        println!("\nSkipping validation (GPU not available)");
    }

    println!("\n=== Validation Complete ===");
}
