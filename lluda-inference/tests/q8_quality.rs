//! Q8_0 quantization quality validation on real model weights.
//!
//! Tests load actual SafeTensors model weights, quantize them to Q8_0,
//! and verify numerical quality metrics (MSE, cosine similarity).
//!
//! These tests require model files on disk and are skipped if not found.

use std::path::PathBuf;

/// Helper: compute cosine similarity between two f32 slices
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for i in 0..a.len() {
        dot += a[i] as f64 * b[i] as f64;
        norm_a += (a[i] as f64) * (a[i] as f64);
        norm_b += (b[i] as f64) * (b[i] as f64);
    }
    if norm_a < 1e-20 || norm_b < 1e-20 {
        return if norm_a < 1e-20 && norm_b < 1e-20 { 1.0 } else { 0.0 };
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

/// Helper: compute MSE between two f32 slices
fn mse(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let sum: f64 = a.iter().zip(b.iter())
        .map(|(x, y)| { let d = *x as f64 - *y as f64; d * d })
        .sum();
    sum / a.len() as f64
}

#[test]
fn test_q8_quantization_quality_on_real_weights() {
    // Try to find a model shard
    let possible_paths = [
        PathBuf::from("/home/alexmak/lluda/models/Qwen2.5-Omni-3B/model-00001-of-00003.safetensors"),
        PathBuf::from("/home/alexmak/lluda/models/Qwen3-Omni-30B/model-00001-of-00015.safetensors"),
    ];

    let shard_path = possible_paths.iter().find(|p| p.exists());
    let shard_path = match shard_path {
        Some(p) => p,
        None => {
            eprintln!("SKIPPED: No model shard found for Q8 quality test");
            return;
        }
    };

    eprintln!("Loading shard: {}", shard_path.display());

    // Load weights in BF16 (original)
    let bf16_weights = lluda_inference::loader::ModelWeights::from_safetensors(shard_path)
        .expect("Failed to load BF16 weights");

    // Load weights with Q8 quantization
    let q8_weights = lluda_inference::loader::ModelWeights::from_safetensors_q8(shard_path)
        .expect("Failed to load Q8 weights");

    eprintln!("BF16 tensors: {}, Q8 tensors: {}", bf16_weights.len(), q8_weights.len());
    assert_eq!(bf16_weights.len(), q8_weights.len(), "Tensor count mismatch");

    // Collect names as owned strings to avoid simultaneous borrow conflicts
    let names: Vec<String> = bf16_weights.tensor_names()
        .into_iter()
        .map(|s| s.to_owned())
        .collect();

    let mut quantized_count = 0;
    let mut total_mse = 0.0f64;
    let mut min_cosine = 1.0f64;
    let mut max_mse = 0.0f64;

    for name in &names {
        let bf16_tensor = bf16_weights.get(name).unwrap();
        let q8_tensor = q8_weights.get(name).unwrap();

        // Check if this tensor was quantized
        if q8_tensor.dtype() == lluda_inference::tensor::DType::Q8_0 {
            quantized_count += 1;

            // Compare dequantized Q8 with original BF16
            let bf16_f32 = bf16_tensor.to_vec_f32();
            let q8_f32 = q8_tensor.to_vec_f32(); // dequantizes

            assert_eq!(bf16_f32.len(), q8_f32.len(),
                "Size mismatch for {}: {} vs {}", name, bf16_f32.len(), q8_f32.len());

            let layer_mse = mse(&bf16_f32, &q8_f32);
            let layer_cosine = cosine_similarity(&bf16_f32, &q8_f32);

            eprintln!("  Q8 {}: shape={:?}, MSE={:.4e}, cos={:.6}",
                name, q8_tensor.shape(), layer_mse, layer_cosine);

            // Quality thresholds
            assert!(layer_cosine > 0.995,
                "FAIL: {} cosine similarity {:.6} < 0.995", name, layer_cosine);

            total_mse += layer_mse;
            if layer_cosine < min_cosine { min_cosine = layer_cosine; }
            if layer_mse > max_mse { max_mse = layer_mse; }
        } else {
            // Non-quantized tensor — should be identical dtype
            assert_eq!(bf16_tensor.dtype(), q8_tensor.dtype(),
                "Non-Q8 tensor {} dtype mismatch", name);
        }
    }

    eprintln!("\n=== Q8 Quality Summary ===");
    eprintln!("Quantized layers: {}", quantized_count);
    if quantized_count > 0 {
        eprintln!("Average MSE: {:.4e}", total_mse / quantized_count as f64);
        eprintln!("Max MSE: {:.4e}", max_mse);
        eprintln!("Min cosine: {:.6}", min_cosine);
    }

    assert!(quantized_count > 0, "No tensors were quantized!");
    assert!(min_cosine > 0.995, "Some layer has cosine < 0.995: {:.6}", min_cosine);

    eprintln!("\nQ8 quality validation PASSED!");
}

#[test]
fn test_q8_matmul_on_real_weights() {
    // Load a single shard to get real weight matrices
    let possible_paths = [
        PathBuf::from("/home/alexmak/lluda/models/Qwen2.5-Omni-3B/model-00001-of-00003.safetensors"),
    ];

    let shard_path = possible_paths.iter().find(|p| p.exists());
    let shard_path = match shard_path {
        Some(p) => p,
        None => {
            eprintln!("SKIPPED: No model shard found");
            return;
        }
    };

    let q8_weights = lluda_inference::loader::ModelWeights::from_safetensors_q8(shard_path)
        .expect("Failed to load Q8 weights");
    let bf16_weights = lluda_inference::loader::ModelWeights::from_safetensors(shard_path)
        .expect("Failed to load BF16 weights");

    // Collect names as owned strings so we can query both weight sets freely
    let names: Vec<String> = q8_weights.tensor_names()
        .into_iter()
        .map(|s| s.to_owned())
        .collect();

    // Find a Q8_0 weight tensor to test matmul; pick one where in_features % 32 == 0
    let q8_name = names.iter().find(|n| {
        let t = q8_weights.get(n.as_str()).unwrap();
        if t.dtype() != lluda_inference::tensor::DType::Q8_0 {
            return false;
        }
        let shape = t.shape();
        // Must be 2D and in_features divisible by 32 (Q8 block size)
        shape.len() == 2 && shape[1] % 32 == 0
    });

    let q8_name = match q8_name {
        Some(n) => n,
        None => {
            eprintln!("SKIPPED: No suitable Q8 tensor found (need 2D with in_features % 32 == 0)");
            return;
        }
    };

    eprintln!("Testing matmul with: {}", q8_name);

    let q8_tensor = q8_weights.get(q8_name.as_str()).unwrap();
    let bf16_tensor = bf16_weights.get(q8_name.as_str()).unwrap();

    // Get shape before consuming the tensor clone
    let shape = q8_tensor.shape().to_vec(); // [out_features, in_features]
    let out_features = shape[0];
    let in_features = shape[1];

    eprintln!("  Shape: [{}, {}]", out_features, in_features);

    // Create a test activation vector
    let mut x = vec![0.0f32; in_features];
    for i in 0..in_features {
        x[i] = (i as f32 * 0.001).sin();
    }

    // Q8 matmul via matmul_f32_x_quant
    let q8_blocks = q8_tensor.clone().into_q8_blocks().unwrap();
    let q8_result = lluda_inference::quant::matmul_f32_x_quant::<lluda_inference::quant::Q8Block>(
        &x, &q8_blocks, 1, in_features, out_features
    );

    // BF16 reference: x @ W^T  (W is [out_features, in_features], row-major)
    let w_f32 = bf16_tensor.to_vec_f32(); // [out_features, in_features] row-major
    let mut ref_result = vec![0.0f32; out_features];
    for j in 0..out_features {
        let mut sum = 0.0f64;
        for k in 0..in_features {
            sum += x[k] as f64 * w_f32[j * in_features + k] as f64;
        }
        ref_result[j] = sum as f32;
    }

    // Compare
    let cos = cosine_similarity(&q8_result, &ref_result);
    let m = mse(&q8_result, &ref_result);

    eprintln!("  Matmul result: cosine={:.6}, MSE={:.4e}", cos, m);
    assert!(cos > 0.999, "Q8 matmul cosine {:.6} < 0.999", cos);

    eprintln!("Q8 matmul validation PASSED!");
}
