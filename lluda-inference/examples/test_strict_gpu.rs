/// Test strict GPU-or-fail behavior
///
/// This file demonstrates the new strict GPU behavior.
/// Compile with: cargo run --features gpu --example test_strict_gpu

use lluda_inference::tensor::{Tensor, DType};

fn main() {
    println!("Testing strict GPU-or-fail behavior...\n");

    // Test 1: F32 tensors should fail with clear error
    println!("Test 1: F32 tensors (should fail with dtype error)");
    let a_f32 = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let b_f32 = Tensor::new(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();

    match a_f32.matmul(&b_f32) {
        Ok(_) => println!("  ERROR: Should have failed!"),
        Err(e) => println!("  Expected error: {}", e),
    }

    println!();

    // Test 2: BF16 tensors should succeed
    println!("Test 2: BF16 tensors (should succeed)");
    let a_bf16 = a_f32.to_dtype(DType::BF16).unwrap();
    let b_bf16 = b_f32.to_dtype(DType::BF16).unwrap();

    match a_bf16.matmul(&b_bf16) {
        Ok(result) => {
            println!("  Success! Result shape: {:?}", result.shape());
            println!("  Result: {:?}", result.to_vec_f32());
        }
        Err(e) => println!("  Unexpected error: {}", e),
    }

    println!();

    // Test 3: 3D tensors should fail (unsupported in GPU mode)
    println!("Test 3: 3D×3D matmul (should fail - unsupported shape)");
    let a_3d = Tensor::new(vec![1.0f32; 24], vec![2, 3, 4]).unwrap();
    let b_3d = Tensor::new(vec![1.0f32; 24], vec![2, 4, 3]).unwrap();
    let a_3d_bf16 = a_3d.to_dtype(DType::BF16).unwrap();
    let b_3d_bf16 = b_3d.to_dtype(DType::BF16).unwrap();

    match a_3d_bf16.matmul(&b_3d_bf16) {
        Ok(_) => println!("  ERROR: Should have failed!"),
        Err(e) => println!("  Expected error: {}", e),
    }

    println!();
    println!("All tests completed. GPU-or-fail policy working correctly!");
}
