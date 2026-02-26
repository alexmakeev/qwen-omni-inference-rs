//! Diagnostic test to bisect where Rust AudioEncoder diverges from Python.
//!
//! Uses F32 reference data in /home/alexmak/lluda/reference_data/omni_asr_test/:
//! - `mel_input_cropped.npy`               — mel [128, 185]
//! - `audio_pre_transformer_f32.npy`       — after conv stem + pos embed [93, 1280]
//! - `audio_layer_NN_output_f32.npy`       — per-layer outputs [93, 1280]
//! - `audio_tower_output_f32.npy`          — final output [46, 2048]
//!
//! # Running
//!
//! ```text
//! cargo test --release --test audio_encoder_diagnostic -- --nocapture --ignored
//! ```

use ndarray::ArrayD;
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::path::Path;

use lluda_inference::audio_encoder::AudioEncoder;
use lluda_inference::config::OmniConfig;
use lluda_inference::loader::ModelWeights;
use lluda_inference::tensor::Tensor;

const MODEL_DIR: &str = "/home/alexmak/lluda/models/Qwen2.5-Omni-3B";
const REF_DIR: &str = "/home/alexmak/lluda/reference_data/omni_asr_test";

// ── npy helpers ───────────────────────────────────────────────────────────────

fn load_npy_f32(path: &str) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
    let reader = File::open(path)?;
    let arr: ArrayD<f32> = ArrayD::read_npy(reader)?;
    let shape = arr.shape().to_vec();
    // Ensure C-order (row-major) layout before extracting raw data.
    // Numpy files saved from PyTorch may be stored in Fortran order (column-major)
    // when the underlying tensor was non-contiguous (e.g., from .transpose()).
    // into_raw_vec_and_offset() returns storage-order bytes, NOT logical C-order,
    // so a Fortran-ordered array yields column-major data — wrong for our Tensor::new.
    let c_contiguous = arr.as_standard_layout();
    let (data, _) = c_contiguous.into_owned().into_raw_vec_and_offset();
    Ok((data, shape))
}

// ── Metrics ───────────────────────────────────────────────────────────────────

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "cosine: length mismatch {} vs {}", a.len(), b.len());
    let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x as f64 * y as f64).sum();
    let na: f64 = a.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
    if na > 0.0 && nb > 0.0 { dot / (na * nb) } else { 0.0 }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn mse(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    a.iter().zip(b.iter()).map(|(&x, &y)| { let d = (x - y) as f64; d * d }).sum::<f64>() / n
}

fn print_metrics(label: &str, rust: &[f32], reference: &[f32]) {
    let c = cosine(rust, reference);
    let m = mse(rust, reference);
    let mx = max_abs_diff(rust, reference);
    eprintln!(
        "  {:<50} cosine={:.6}  MSE={:.2e}  MaxDiff={:.4}",
        label, c, m, mx
    );
}

// ── Prerequisites check ───────────────────────────────────────────────────────

fn check_prerequisites() -> bool {
    let model_ok = Path::new(MODEL_DIR).join("config.json").exists();
    let mel_ok = Path::new(REF_DIR).join("mel_input_cropped.npy").exists();
    let pre_ok = Path::new(REF_DIR).join("audio_pre_transformer_f32.npy").exists();

    if !model_ok {
        eprintln!("SKIP: model not found at {}", MODEL_DIR);
    }
    if !mel_ok {
        eprintln!("SKIP: mel_input_cropped.npy not found in {}", REF_DIR);
    }
    if !pre_ok {
        eprintln!("SKIP: audio_pre_transformer_f32.npy not found in {}", REF_DIR);
    }
    model_ok && mel_ok && pre_ok
}

// ── Main diagnostic test ──────────────────────────────────────────────────────

/// Step-by-step AudioEncoder diagnostic against F32 Python reference.
///
/// Checks:
/// 1. Conv stem output (after conv1+gelu+conv2+gelu+transpose)
/// 2. After positional embedding addition
/// 3. Per-layer transformer outputs (all 32 layers)
/// 4. Final tower output
#[test]
#[ignore]
fn test_audio_encoder_diagnostic_f32() {
    if !check_prerequisites() {
        return;
    }

    // ── Load config ──────────────────────────────────────────────────────────
    let config_path = format!("{}/config.json", MODEL_DIR);
    let config = match OmniConfig::from_file(&config_path) {
        Ok(c) => c,
        Err(e) => { eprintln!("Failed to load config: {}", e); return; }
    };

    // ── Load weights ─────────────────────────────────────────────────────────
    eprintln!("\nLoading model weights from {} ...", MODEL_DIR);
    let weights = match ModelWeights::from_directory(MODEL_DIR) {
        Ok(w) => w,
        Err(e) => { eprintln!("Failed to load weights: {}", e); return; }
    };

    // ── Build AudioEncoder ───────────────────────────────────────────────────
    eprintln!("Building AudioEncoder ...");
    let encoder = match AudioEncoder::load(
        &config.thinker_config.audio_config,
        |name| weights.get(name).cloned(),
    ) {
        Ok(enc) => enc,
        Err(e) => { eprintln!("Failed to build AudioEncoder: {}", e); return; }
    };

    // ── Load mel input ────────────────────────────────────────────────────────
    let mel_path = format!("{}/mel_input_cropped.npy", REF_DIR);
    let (mel_data, mel_shape) = match load_npy_f32(&mel_path) {
        Ok(v) => v,
        Err(e) => { eprintln!("Failed to load mel: {}", e); return; }
    };
    eprintln!("Mel input shape: {:?}", mel_shape);

    // The mel may need to be read as f32; check if it's stored as something else
    // by verifying the shape makes sense
    assert_eq!(mel_shape.len(), 2, "Expected 2D mel [128, T]");
    assert_eq!(mel_shape[0], 128, "Expected 128 mel bins");
    let t_mel = mel_shape[1];
    eprintln!("Mel T={} ({})", t_mel, if t_mel % 2 == 0 { "even" } else { "ODD" });

    let mel = match Tensor::new(mel_data, mel_shape) {
        Ok(t) => t,
        Err(e) => { eprintln!("Failed to create mel tensor: {}", e); return; }
    };

    // Expected conv2 output length: floor((T + 2*1 - 3) / 2) + 1
    let expected_conv2_len = (t_mel + 2 * 1 - 3) / 2 + 1;
    eprintln!("Expected conv2 output length: {} (T={})", expected_conv2_len, t_mel);

    // ── Run diagnostic forward ────────────────────────────────────────────────
    eprintln!("\nRunning diagnostic forward pass ...");
    let (after_conv_stem, after_pos_embed, layer_outputs) =
        match encoder.forward_diagnostic(&mel) {
            Ok(v) => v,
            Err(e) => { eprintln!("forward_diagnostic failed: {}", e); return; }
        };

    eprintln!("After conv stem shape:    {:?}", after_conv_stem.shape());
    eprintln!("After pos embed shape:    {:?}", after_pos_embed.shape());
    eprintln!("Number of layer outputs:  {}", layer_outputs.len());

    // ── Load reference: audio_pre_transformer_f32.npy [93, 1280] ─────────────
    eprintln!("\n=== STEP 1: Conv stem (after conv1+gelu+conv2+gelu+transpose) ===");
    let pre_tf_path = format!("{}/audio_pre_transformer_f32.npy", REF_DIR);
    match load_npy_f32(&pre_tf_path) {
        Ok((ref_data, ref_shape)) => {
            eprintln!("Reference pre-transformer shape: {:?}", ref_shape);
            eprintln!("Rust after_conv_stem shape:      {:?}", after_conv_stem.shape());

            let rust_conv = after_conv_stem.to_vec_f32();
            if rust_conv.len() == ref_data.len() {
                // The reference is [93, 1280] = after pos embed, not just conv stem.
                // But we check conv stem first (before pos embed) then after pos embed.
                // The reference `audio_pre_transformer_f32.npy` is AFTER pos embed.
                // So step 1 = conv stem alone (no reference), step 2 = with pos embed.
                eprintln!(
                    "  Note: reference includes pos embed. \
                     Conv stem cosine vs reference (expecting some mismatch due to pos embed):"
                );
                print_metrics("after_conv_stem vs pre_transformer_ref", &rust_conv, &ref_data);
            } else {
                eprintln!(
                    "  Shape mismatch: Rust has {} elements, reference has {} elements",
                    rust_conv.len(), ref_data.len()
                );
                eprintln!("  This indicates a convolution length bug!");
            }

            // ── STEP 2: After positional embedding ────────────────────────────
            eprintln!("\n=== STEP 2: After positional embedding ===");
            let rust_pos = after_pos_embed.to_vec_f32();
            if rust_pos.len() == ref_data.len() {
                print_metrics("after_pos_embed vs pre_transformer_ref", &rust_pos, &ref_data);
            } else {
                eprintln!(
                    "  Shape mismatch: Rust has {} elements, reference has {} elements",
                    rust_pos.len(), ref_data.len()
                );
                eprintln!("  Rust seq_len={}, reference seq_len={}",
                    after_pos_embed.shape()[0], ref_shape[0]);
            }
        }
        Err(e) => eprintln!("Failed to load pre_transformer reference: {}", e),
    }

    // ── STEP 3: Layer-by-layer comparison ────────────────────────────────────
    eprintln!("\n=== STEP 3: Per-layer transformer outputs ===");

    let mut first_bad_layer: Option<usize> = None;

    for layer_idx in 0..layer_outputs.len() {
        let layer_path = format!(
            "{}/audio_layer_{:02}_output_f32.npy",
            REF_DIR, layer_idx
        );
        if !Path::new(&layer_path).exists() {
            eprintln!("  Layer {:02}: reference not found, stopping layer comparison", layer_idx);
            break;
        }

        match load_npy_f32(&layer_path) {
            Ok((ref_data, ref_shape)) => {
                let rust_data = layer_outputs[layer_idx].to_vec_f32();
                let rust_shape = layer_outputs[layer_idx].shape().to_vec();

                if rust_data.len() != ref_data.len() {
                    eprintln!(
                        "  Layer {:02}: shape mismatch! Rust {:?} vs Ref {:?}",
                        layer_idx, rust_shape, ref_shape
                    );
                    if first_bad_layer.is_none() { first_bad_layer = Some(layer_idx); }
                    continue;
                }

                let c = cosine(&rust_data, &ref_data);
                let mx = max_abs_diff(&rust_data, &ref_data);
                let m = mse(&rust_data, &ref_data);

                let status = if c > 0.999 { "OK " } else if c > 0.99 { "~~ " } else { "BAD" };
                eprintln!(
                    "  [{}] Layer {:02}: cosine={:.6}  MSE={:.2e}  MaxDiff={:.4}",
                    status, layer_idx, c, m, mx
                );

                if c < 0.99 && first_bad_layer.is_none() {
                    first_bad_layer = Some(layer_idx);
                }
            }
            Err(e) => eprintln!("  Layer {:02}: failed to load reference: {}", layer_idx, e),
        }
    }

    // ── STEP 4: Final tower output ────────────────────────────────────────────
    eprintln!("\n=== STEP 4: Final tower output ===");
    let tower_path = format!("{}/audio_tower_output_f32.npy", REF_DIR);
    if Path::new(&tower_path).exists() {
        match load_npy_f32(&tower_path) {
            Ok((ref_data, ref_shape)) => {
                eprintln!("Reference tower output shape: {:?}", ref_shape);
                // Run full forward to get final output
                match encoder.forward(&mel) {
                    Ok(output) => {
                        let rust_data = output.to_vec_f32();
                        eprintln!("Rust tower output shape: {:?}", output.shape());
                        if rust_data.len() == ref_data.len() {
                            print_metrics("final tower output vs f32 reference", &rust_data, &ref_data);
                        } else {
                            eprintln!(
                                "  Shape mismatch: Rust {} elements, Ref {} elements",
                                rust_data.len(), ref_data.len()
                            );
                        }
                    }
                    Err(e) => eprintln!("  encoder.forward failed: {}", e),
                }
            }
            Err(e) => eprintln!("  Failed to load tower output: {}", e),
        }
    } else {
        eprintln!("  Tower output reference not found, skipping.");
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    eprintln!("\n=== SUMMARY ===");
    eprintln!("Mel shape: [128, {}] ({})", t_mel, if t_mel % 2 == 0 { "even" } else { "ODD" });
    eprintln!("Expected conv2 output T: {}", expected_conv2_len);
    eprintln!("Actual Rust conv stem output shape: {:?}", after_conv_stem.shape());

    match first_bad_layer {
        Some(layer) => {
            eprintln!("First bad layer: {} (cosine < 0.99)", layer);
            if layer == 0 {
                eprintln!("=> Divergence starts BEFORE layer 0 (likely conv stem or pos embed)");
            } else {
                eprintln!("=> Divergence starts at layer {}", layer);
            }
        }
        None => {
            eprintln!("All checked layers OK (cosine >= 0.99)");
        }
    }
}

/// Focused test: verify conv stem output length for T=185 (odd) vs T=200 (even).
///
/// This tests the hypothesis that the bug is in Conv1d for odd-length input.
#[test]
#[ignore]
fn test_conv_stem_length_odd_vs_even() {
    use lluda_inference::conv1d::Conv1d;

    eprintln!("\n=== Conv1d output length test: odd vs even T ===");

    // Minimal conv2 kernel: [1, 1, 3], stride=2, padding=1
    let weight = Tensor::new(vec![0.0f32; 1 * 1 * 3], vec![1, 1, 3]).unwrap();
    let bias = Tensor::new(vec![0.0f32; 1], vec![1]).unwrap();
    let conv = Conv1d::new(weight, bias, 2, 1).unwrap();

    for t in [183, 184, 185, 186, 187, 188, 200, 201] {
        let x = Tensor::new(vec![1.0f32; 1 * t], vec![1, t]).unwrap();
        let y = conv.forward(&x).unwrap();
        let out_len = y.shape()[1];
        let expected = (t + 2 * 1 - 3) / 2 + 1;
        let matches = if out_len == expected { "OK" } else { "MISMATCH" };
        eprintln!(
            "  T={:4}  expected_out={}  actual_out={}  {}",
            t, expected, out_len, matches
        );
    }
}

/// Test: load conv1 and conv2 weights from Rust, print first 10 values for comparison with Python.
///
/// Expected Python values:
///   conv1 weight shape: [1280, 128, 3]
///   conv1 first 10: [-0.001091, -0.003052, -0.000641, 0.016357, 0.015198, 0.013916, 0.010925, 0.007172, 0.007324, 0.006104]
///   conv1 bias first 5: [0.089844, -0.007355, -0.031006, -0.045898, 0.002060]
///   conv2 weight shape: [1280, 1280, 3]
///   conv2 first 10: [-0.034180, -0.039795, -0.036133, 0.020630, 0.010010, 0.009155, -0.004486, 0.007721, 0.002808, 0.002655]
#[test]
#[ignore]
fn test_conv_weight_values() {
    if !Path::new(MODEL_DIR).join("config.json").exists() {
        eprintln!("SKIP: model not found at {}", MODEL_DIR);
        return;
    }

    let config_path = format!("{}/config.json", MODEL_DIR);
    let config = match lluda_inference::config::OmniConfig::from_file(&config_path) {
        Ok(c) => c,
        Err(e) => { eprintln!("Failed to load config: {}", e); return; }
    };

    eprintln!("\nLoading model weights ...");
    let weights = match lluda_inference::loader::ModelWeights::from_directory(MODEL_DIR) {
        Ok(w) => w,
        Err(e) => { eprintln!("Failed to load weights: {}", e); return; }
    };

    // Check conv1 weight
    let conv1_w_name = "thinker.audio_tower.conv1.weight";
    match weights.get(conv1_w_name) {
        Some(w) => {
            eprintln!("conv1 weight shape: {:?}", w.shape());
            let data = w.to_vec_f32();
            eprintln!("conv1 first 10: {:?}", &data[..10.min(data.len())]);
        }
        None => eprintln!("MISSING: {}", conv1_w_name),
    }

    let conv1_b_name = "thinker.audio_tower.conv1.bias";
    match weights.get(conv1_b_name) {
        Some(b) => {
            let data = b.to_vec_f32();
            eprintln!("conv1 bias first 5: {:?}", &data[..5.min(data.len())]);
        }
        None => eprintln!("MISSING: {}", conv1_b_name),
    }

    // Check conv2 weight
    let conv2_w_name = "thinker.audio_tower.conv2.weight";
    match weights.get(conv2_w_name) {
        Some(w) => {
            eprintln!("conv2 weight shape: {:?}", w.shape());
            let data = w.to_vec_f32();
            eprintln!("conv2 first 10: {:?}", &data[..10.min(data.len())]);
        }
        None => eprintln!("MISSING: {}", conv2_w_name),
    }

    let conv2_b_name = "thinker.audio_tower.conv2.bias";
    match weights.get(conv2_b_name) {
        Some(b) => {
            let data = b.to_vec_f32();
            eprintln!("conv2 bias first 5: {:?}", &data[..5.min(data.len())]);
        }
        None => eprintln!("MISSING: {}", conv2_b_name),
    }

    // Also run conv stem and compare directly with Python conv stem output
    eprintln!("\nRunning conv stem on mel input ...");
    let mel_path = format!("{}/mel_input_cropped.npy", REF_DIR);
    let (mel_data, mel_shape) = match load_npy_f32(&mel_path) {
        Ok(v) => v,
        Err(e) => { eprintln!("Failed to load mel: {}", e); return; }
    };

    eprintln!("mel shape: {:?}, first 10 values: {:?}", mel_shape, &mel_data[..10.min(mel_data.len())]);

    // Get conv1 weight directly and do manual single-element computation
    // Manual: conv1[co=0, t=0] = bias[0] + sum over ci, k (with padding=1, stride=1, kernel=3)
    let manual_v = if let (Some(w_tensor), Some(b_tensor)) = (
        weights.get("thinker.audio_tower.conv1.weight"),
        weights.get("thinker.audio_tower.conv1.bias"),
    ) {
        let w = w_tensor.to_vec_f32();  // [1280, 128, 3]
        let b = b_tensor.to_vec_f32();  // [1280]
        // co=0, t=0, padding=1, stride=1, kernel=3, seq_len=185
        let seq_len = mel_shape[1];
        let in_ch = mel_shape[0];
        let kernel = 3usize;
        let padding = 1usize;
        let stride = 1usize;
        let co = 0usize;
        let t = 0usize;
        let mut s = b[co];
        for ci in 0..in_ch {
            for k in 0..kernel {
                let input_t = (t * stride + k) as isize - padding as isize;
                if input_t >= 0 && (input_t as usize) < seq_len {
                    s += w[co * in_ch * kernel + ci * kernel + k] * mel_data[ci * seq_len + input_t as usize];
                }
            }
        }
        eprintln!("Manual conv1[co=0, t=0] = {:.8} (expected from Python: 0.13310495)", s);
        s
    } else {
        eprintln!("Could not get conv1 weights for manual check");
        0.0f32
    };

    let mel = Tensor::new(mel_data, mel_shape).unwrap();
    let encoder = match AudioEncoder::load(
        &config.thinker_config.audio_config,
        |name| weights.get(name).cloned(),
    ) {
        Ok(enc) => enc,
        Err(e) => { eprintln!("Failed to build encoder: {}", e); return; }
    };

    let (after_conv_stem, _, _) = encoder.forward_diagnostic(&mel).unwrap();
    eprintln!("Rust after_conv_stem shape: {:?}", after_conv_stem.shape());

    let rust_conv = after_conv_stem.to_vec_f32();
    eprintln!("Rust after_conv_stem first 10: {:?}", &rust_conv[..10.min(rust_conv.len())]);

    // Load Python conv stem reference (pure conv stem, before pos embed)
    let conv_stem_path = format!("{}/conv_stem_only_f32.npy", REF_DIR);
    if Path::new(&conv_stem_path).exists() {
        match load_npy_f32(&conv_stem_path) {
            Ok((ref_data, ref_shape)) => {
                eprintln!("Python conv_stem_only_f32 shape: {:?}", ref_shape);
                eprintln!("Python conv_stem first 10: {:?}", &ref_data[..10.min(ref_data.len())]);
                if rust_conv.len() == ref_data.len() {
                    print_metrics("Rust conv_stem vs Python conv_stem_only_f32", &rust_conv, &ref_data);
                } else {
                    eprintln!("LENGTH MISMATCH: Rust={} Python={}", rust_conv.len(), ref_data.len());
                }
            }
            Err(e) => eprintln!("Failed to load conv_stem_only_f32: {}", e),
        }
    } else {
        eprintln!("conv_stem_only_f32.npy not found at {}", conv_stem_path);
    }
}

/// Check that pre-transformer reference data shape is what we expect.
#[test]
#[ignore]
fn test_reference_data_shapes() {
    eprintln!("\n=== Reference data shapes ===");

    let files = [
        ("mel_input_cropped.npy", vec![128usize, 185]),
        ("audio_pre_transformer_f32.npy", vec![93, 1280]),
        ("audio_layer_00_output_f32.npy", vec![93, 1280]),
        ("audio_layer_31_output_f32.npy", vec![93, 1280]),
        ("audio_tower_output_f32.npy", vec![46, 2048]),
    ];

    for (fname, expected_shape) in &files {
        let path = format!("{}/{}", REF_DIR, fname);
        match load_npy_f32(&path) {
            Ok((data, shape)) => {
                let ok = shape == *expected_shape;
                eprintln!(
                    "  {} {} {:?}  ({} elements)",
                    if ok { "[OK ]" } else { "[ERR]" },
                    fname,
                    shape,
                    data.len()
                );
                if !ok {
                    eprintln!("    Expected: {:?}", expected_shape);
                }
            }
            Err(e) => eprintln!("  [MISS] {}: {}", fname, e),
        }
    }
}
