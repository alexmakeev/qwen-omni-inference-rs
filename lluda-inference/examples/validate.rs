//! Standalone validation tool for comparing Rust outputs against Python reference.
//!
//! # Usage
//!
//! ```bash
//! # First, generate reference data using the Python script (T21)
//! python scripts/extract_reference.py
//!
//! # Then run validation
//! cargo run --example validate -- reference_data/prompt1
//! ```
//!
//! This tool performs layer-by-layer comparison of the Rust model implementation
//! against HuggingFace Transformers reference data.

use ndarray::ArrayD;
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::path::{Path, PathBuf};

/// Validation metrics for a single component.
#[derive(Debug)]
struct ValidationMetrics {
    name: String,
    mse: f64,
    cosine_similarity: f64,
    max_abs_diff: f64,
    mean_abs_diff: f64,
}

impl ValidationMetrics {
    fn new(name: String, predicted: &[f32], reference: &[f32]) -> Self {
        assert_eq!(
            predicted.len(),
            reference.len(),
            "{}: length mismatch: {} vs {}",
            name,
            predicted.len(),
            reference.len()
        );

        let n = predicted.len() as f64;

        // Mean Squared Error
        let mse: f64 = predicted
            .iter()
            .zip(reference.iter())
            .map(|(&p, &r)| {
                let diff = (p - r) as f64;
                diff * diff
            })
            .sum::<f64>()
            / n;

        // Max and mean absolute difference
        let mut max_abs_diff = 0.0f64;
        let mut sum_abs_diff = 0.0f64;

        for (&p, &r) in predicted.iter().zip(reference.iter()) {
            let abs_diff = (p - r).abs() as f64;
            max_abs_diff = max_abs_diff.max(abs_diff);
            sum_abs_diff += abs_diff;
        }

        let mean_abs_diff = sum_abs_diff / n;

        // Cosine similarity
        let dot_product: f64 = predicted
            .iter()
            .zip(reference.iter())
            .map(|(&p, &r)| (p as f64) * (r as f64))
            .sum();

        let pred_norm: f64 = predicted.iter().map(|&p| (p as f64) * (p as f64)).sum::<f64>().sqrt();
        let ref_norm: f64 = reference
            .iter()
            .map(|&r| (r as f64) * (r as f64))
            .sum::<f64>()
            .sqrt();

        let cosine_similarity = if pred_norm > 0.0 && ref_norm > 0.0 {
            dot_product / (pred_norm * ref_norm)
        } else {
            0.0
        };

        Self {
            name,
            mse,
            cosine_similarity,
            max_abs_diff,
            mean_abs_diff,
        }
    }

    fn is_valid(&self, mse_threshold: f64, cosine_threshold: f64, max_diff_threshold: f64) -> bool {
        self.mse < mse_threshold
            && self.cosine_similarity > cosine_threshold
            && self.max_abs_diff < max_diff_threshold
    }

    fn print(&self) {
        let status = if self.is_valid(1e-3, 0.999, 1e-2) {
            "✓"
        } else {
            "✗"
        };

        println!(
            "  {} {:30} MSE={:.2e}  Cosine={:.6}  MaxDiff={:.2e}  MeanDiff={:.2e}",
            status, self.name, self.mse, self.cosine_similarity, self.max_abs_diff, self.mean_abs_diff
        );
    }
}

/// Load .npy file as flat f32 vector.
fn load_npy_flat(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let reader = File::open(path)?;
    let array: ArrayD<f32> = ArrayD::read_npy(reader)?;
    let (vec, _offset) = array.into_raw_vec_and_offset();
    Ok(vec)
}

/// Load input token IDs from .npy file.
fn load_input_ids(path: &Path) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let reader = File::open(path)?;
    let array: ArrayD<i64> = ArrayD::read_npy(reader)?;
    Ok(array.iter().map(|&x| x as u32).collect())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use lluda_inference::config::Qwen3Config;
    use lluda_inference::loader::ModelWeights;
    use lluda_inference::model::Qwen3ForCausalLM;

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    let reference_dir = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        // Try to find reference_data directory
        let candidates = vec![
            PathBuf::from("reference_data"),
            PathBuf::from("../reference_data"),
        ];

        let mut found = None;
        for candidate in candidates {
            if candidate.exists() && candidate.is_dir() {
                // Find first prompt subdirectory
                for entry in std::fs::read_dir(&candidate)? {
                    let entry = entry?;
                    if entry.path().is_dir() {
                        found = Some(entry.path());
                        break;
                    }
                }
                if found.is_some() {
                    break;
                }
            }
        }

        match found {
            Some(dir) => dir,
            None => {
                eprintln!("Usage: {} <reference_data_dir>", args[0]);
                eprintln!();
                eprintln!("Example: {} reference_data/prompt1", args[0]);
                eprintln!();
                eprintln!("First run the Python script to generate reference data:");
                eprintln!("  python scripts/extract_reference.py");
                std::process::exit(1);
            }
        }
    };

    println!("=== Rust vs Python Validation ===");
    println!();
    println!("Reference data: {}", reference_dir.display());
    println!();

    // Check if reference directory exists
    if !reference_dir.exists() {
        eprintln!("Error: Reference directory does not exist: {}", reference_dir.display());
        eprintln!();
        eprintln!("Run the T21 Python script first:");
        eprintln!("  python scripts/extract_reference.py");
        std::process::exit(1);
    }

    // Load input token IDs
    let input_ids_path = reference_dir.join("input_ids.npy");
    let input_ids = load_input_ids(&input_ids_path)?;
    println!("Input: {} tokens", input_ids.len());
    println!("Token IDs: {:?}", input_ids);
    println!();

    // Load model
    let model_dir = PathBuf::from("models/Qwen3-0.6B");
    if !model_dir.exists() {
        eprintln!("Error: Model directory not found at {}", model_dir.display());
        eprintln!("Please download Qwen3-0.6B model first");
        std::process::exit(1);
    }

    println!("Loading Qwen3-0.6B model...");
    let config_path = model_dir.join("config.json");
    let weights_path = model_dir.join("model.safetensors");

    let config = Qwen3Config::from_file(&config_path)?;
    let weights = ModelWeights::from_safetensors(&weights_path)?;
    let mut model = Qwen3ForCausalLM::load(&config, &weights)?;

    println!("Model loaded: {} layers, vocab_size={}", config.num_hidden_layers, config.vocab_size);
    println!();

    // Run Rust model forward pass
    println!("Running Rust model forward pass...");
    let rust_logits = model.forward(&input_ids, 0)?;
    println!("Rust logits shape: {:?}", rust_logits.shape());
    println!();

    // Load reference logits
    let logits_path = reference_dir.join("logits.npy");
    if !logits_path.exists() {
        eprintln!("Error: logits.npy not found in {}", reference_dir.display());
        std::process::exit(1);
    }

    println!("Loading reference logits...");
    let ref_logits_arr = load_npy_flat(&logits_path)?;

    // Reference logits are [1, seq_len, vocab_size] flattened
    // We need last token: skip to (seq_len-1) * vocab_size
    let seq_len = input_ids.len();
    let vocab_size = config.vocab_size;
    let ref_logits_last = &ref_logits_arr[(seq_len - 1) * vocab_size..seq_len * vocab_size];

    // Get Rust logits (should be [1, 1, vocab_size])
    let rust_logits_vec = rust_logits.to_vec_f32();

    println!("Comparing outputs...");
    println!("  Rust logits: {} values", rust_logits_vec.len());
    println!("  Python logits (last token): {} values", ref_logits_last.len());
    println!();

    // Compute validation metrics
    let metrics = ValidationMetrics::new(
        "Logits".to_string(),
        &rust_logits_vec,
        ref_logits_last,
    );

    println!("=== Validation Results ===");
    println!();
    metrics.print();
    println!();

    // Check thresholds
    let mse_threshold = 1e-3;
    let cosine_threshold = 0.999;
    let max_diff_threshold = 1e-2;

    let passes = metrics.is_valid(mse_threshold, cosine_threshold, max_diff_threshold);

    if passes {
        println!("✓ VALIDATION PASSED");
        println!();
        println!("Rust implementation matches Python reference within tolerances:");
        println!("  MSE < {:.0e}", mse_threshold);
        println!("  Cosine similarity > {}", cosine_threshold);
        println!("  Max diff < {:.0e}", max_diff_threshold);
    } else {
        println!("✗ VALIDATION FAILED");
        println!();
        println!("Thresholds:");
        println!("  MSE: {:.2e} (threshold: {:.0e})", metrics.mse, mse_threshold);
        println!("  Cosine: {:.6} (threshold: {})", metrics.cosine_similarity, cosine_threshold);
        println!("  Max diff: {:.2e} (threshold: {:.0e})", metrics.max_abs_diff, max_diff_threshold);
        println!();
        println!("This may indicate:");
        println!("  - Numerical precision differences (BF16 vs F32)");
        println!("  - Implementation differences in operators");
        println!("  - Bug in Rust implementation");
    }

    println!();

    // Check if most likely token matches
    let rust_argmax = rust_logits_vec
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    let ref_argmax = ref_logits_last
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    println!("Next token prediction:");
    println!("  Rust:   token {} (logit: {:.4})", rust_argmax, rust_logits_vec[rust_argmax]);
    println!("  Python: token {} (logit: {:.4})", ref_argmax, ref_logits_last[ref_argmax]);

    if rust_argmax == ref_argmax {
        println!("  ✓ Predictions match!");
    } else {
        println!("  ⚠ Predictions differ");
        println!();
        println!("Top 5 Rust predictions:");
        let mut rust_sorted: Vec<(usize, f32)> = rust_logits_vec
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        rust_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (i, (token, logit)) in rust_sorted.iter().take(5).enumerate() {
            println!("    {}. token {} (logit: {:.4})", i + 1, token, logit);
        }

        println!();
        println!("Top 5 Python predictions:");
        let mut ref_sorted: Vec<(usize, f32)> = ref_logits_last
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        ref_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (i, (token, logit)) in ref_sorted.iter().take(5).enumerate() {
            println!("    {}. token {} (logit: {:.4})", i + 1, token, logit);
        }
    }

    println!();

    Ok(())
}
