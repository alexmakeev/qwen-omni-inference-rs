//! Validation tests for Qwen2.5-Omni Talker (TTS decoder) against Python reference.
//!
//! Compares Rust Talker outputs (codec logits) against reference data extracted
//! from HuggingFace Transformers. The talker converts codec token embeddings
//! into codec vocabulary logits.
//!
//! # Reference data generation (Python)
//!
//! 1. embed_tokens(input_ids=[8293]) → codec_embed [1,1,2048]
//! 2. thinker_to_talker_proj(codec_embed) → [1,1,896]  (NO thinker_hidden added)
//! 3. Through 24 decoder layers → per-layer [1,1,896]
//! 4. RMSNorm → final_norm [1,1,896]
//! 5. codec_head → logits [1,1,8448]
//!
//! The Rust forward does: combined = embed_tokens(input_ids) + thinker_hidden.
//! To match the Python reference (no thinker_hidden), pass a ZERO tensor.
//!
//! # Running
//!
//! ```text
//! cargo test --test omni_talker_validation -- --nocapture
//! ```
//!
//! # Requirements
//!
//! - Model weights: `/home/alexmak/lluda/models/Qwen2.5-Omni-3B/`
//! - Reference data: `/home/alexmak/lluda/reference_data/omni_3b/`
//!   - `talker_input_ids.npy`         — codec input token IDs [1, 1] (i64)
//!   - `talker_logits.npy`            — codec logits [1, 1, 8448] (f32)
//!   - `talker_layer_NN_output.npy`   — per-layer outputs (optional, informational)
//!
//! # Tolerance thresholds
//!
//! - Cosine similarity: > 0.99 for logits
//! - MSE:               < 1e-1 (relaxed for BF16->Q8 quantization over 24 layers)
//! - MaxDiff:           < 2.0  (relaxed for BF16->Q8 quantization over 24 layers)
//! - Argmax match expected

use ndarray::ArrayD;
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::path::Path;

use lluda_inference::config::OmniConfig;
use lluda_inference::loader::ModelWeights;
use lluda_inference::talker::Talker;
use lluda_inference::tensor::{DType, Tensor};

const MODEL_DIR: &str = "/home/alexmak/lluda/models/Qwen2.5-Omni-3B";
const REF_DIR: &str = "/home/alexmak/lluda/reference_data/omni_3b";

// ── Validation helpers ────────────────────────────────────────────────────────

/// Validation metrics for a single tensor comparison.
#[derive(Debug, Clone)]
struct ValidationMetrics {
    name: String,
    mse: f64,
    cosine_similarity: f64,
    max_abs_diff: f64,
    mean_abs_diff: f64,
}

impl ValidationMetrics {
    fn compute(name: &str, predicted: &[f32], reference: &[f32]) -> Self {
        assert_eq!(
            predicted.len(),
            reference.len(),
            "{}: length mismatch: {} vs {}",
            name,
            predicted.len(),
            reference.len()
        );

        let n = predicted.len() as f64;

        let mse: f64 = predicted
            .iter()
            .zip(reference.iter())
            .map(|(&p, &r)| {
                let d = (p - r) as f64;
                d * d
            })
            .sum::<f64>()
            / n;

        let mut max_abs_diff = 0.0f64;
        let mut sum_abs_diff = 0.0f64;
        for (&p, &r) in predicted.iter().zip(reference.iter()) {
            let d = (p - r).abs() as f64;
            max_abs_diff = max_abs_diff.max(d);
            sum_abs_diff += d;
        }
        let mean_abs_diff = sum_abs_diff / n;

        let dot: f64 = predicted
            .iter()
            .zip(reference.iter())
            .map(|(&p, &r)| (p as f64) * (r as f64))
            .sum();
        let norm_p: f64 = predicted.iter().map(|&p| (p as f64) * (p as f64)).sum::<f64>().sqrt();
        let norm_r: f64 = reference.iter().map(|&r| (r as f64) * (r as f64)).sum::<f64>().sqrt();
        let cosine_similarity = if norm_p > 0.0 && norm_r > 0.0 {
            dot / (norm_p * norm_r)
        } else {
            0.0
        };

        ValidationMetrics {
            name: name.to_string(),
            mse,
            cosine_similarity,
            max_abs_diff,
            mean_abs_diff,
        }
    }

    fn is_valid(&self, mse_threshold: f64, cosine_threshold: f64) -> bool {
        self.mse < mse_threshold && self.cosine_similarity > cosine_threshold
    }

    fn print(&self) {
        eprintln!(
            "  {}: MSE={:.2e}, Cosine={:.6}, MaxDiff={:.2e}, MeanDiff={:.2e}",
            self.name, self.mse, self.cosine_similarity, self.max_abs_diff, self.mean_abs_diff,
        );
    }
}

// ── npy loading helpers ───────────────────────────────────────────────────────

fn load_npy_f32(path: &str) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
    let reader = File::open(path)?;
    let arr: ArrayD<f32> = ArrayD::read_npy(reader)?;
    let shape = arr.shape().to_vec();
    let (data, _) = arr.into_raw_vec_and_offset();
    Ok((data, shape))
}

fn load_npy_i64(path: &str) -> Result<(Vec<i64>, Vec<usize>), Box<dyn std::error::Error>> {
    let reader = File::open(path)?;
    let arr: ArrayD<i64> = ArrayD::read_npy(reader)?;
    let shape = arr.shape().to_vec();
    let (data, _) = arr.into_raw_vec_and_offset();
    Ok((data, shape))
}

// ── Prerequisite check ────────────────────────────────────────────────────────

fn check_prerequisites() -> bool {
    let model_ok = Path::new(MODEL_DIR).join("config.json").exists();
    let input_ids_ok = Path::new(REF_DIR).join("talker_input_ids.npy").exists();
    let logits_ok = Path::new(REF_DIR).join("talker_logits.npy").exists();

    if !model_ok {
        eprintln!("Skipping: model not found at {}", MODEL_DIR);
    }
    if !input_ids_ok {
        eprintln!(
            "Skipping: talker_input_ids.npy not found at {}/talker_input_ids.npy",
            REF_DIR
        );
    }
    if !logits_ok {
        eprintln!(
            "Skipping: talker_logits.npy not found at {}/talker_logits.npy",
            REF_DIR
        );
    }
    model_ok && input_ids_ok && logits_ok
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Validate Talker codec logits against the Python reference.
///
/// Loads reference input_ids (codec BOS token 8293), constructs a zero
/// thinker_hidden tensor to match the Python reference (which adds no
/// thinker hidden states), runs Talker::forward, and compares the resulting
/// [1, 1, 8448] logits tensor against the Python reference.
#[test]
fn test_talker_logits() {
    if !check_prerequisites() {
        return;
    }

    // Load configuration
    let config_path = format!("{}/config.json", MODEL_DIR);
    let config = match OmniConfig::from_file(&config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping: failed to load config: {}", e);
            return;
        }
    };
    let talker_config = &config.talker_config;
    eprintln!(
        "Talker config: hidden_size={}, embedding_size={}, vocab_size={}, num_layers={}",
        talker_config.hidden_size,
        talker_config.embedding_size,
        talker_config.vocab_size,
        talker_config.num_hidden_layers,
    );

    // Load model weights
    eprintln!("Loading model weights from {} ...", MODEL_DIR);
    let weights = match ModelWeights::from_directory(MODEL_DIR) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Skipping: failed to load weights: {}", e);
            return;
        }
    };

    // Build Talker
    eprintln!("Building Talker ...");
    let mut talker = match Talker::load(talker_config, |name| weights.get(name).cloned()) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Skipping: failed to build Talker: {}", e);
            return;
        }
    };

    // Load reference codec input_ids [1, 1] — should contain token 8293 (codec BOS)
    let ids_path = format!("{}/talker_input_ids.npy", REF_DIR);
    let (ids_data, ids_shape) = match load_npy_i64(&ids_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Skipping: failed to load talker_input_ids.npy: {}", e);
            return;
        }
    };
    eprintln!("talker_input_ids shape: {:?}, values: {:?}", ids_shape, ids_data);

    let input_ids: Vec<u32> = ids_data.iter().map(|&x| x as u32).collect();
    eprintln!("Codec input IDs: {:?}", input_ids);

    // Create a ZERO thinker_hidden tensor [1, seq, embedding_size].
    // The Python reference does NOT add thinker hidden states (embed only).
    // Since Rust forward does: combined = embed(input_ids) + thinker_hidden,
    // passing zeros makes the combined equal to embed(input_ids) alone.
    let seq_len = input_ids.len();
    let embedding_size = talker_config.embedding_size;
    let zero_hidden = match Tensor::zeros(&[1, seq_len, embedding_size], DType::F32) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Skipping: failed to create zero thinker_hidden tensor: {}", e);
            return;
        }
    };
    eprintln!(
        "Zero thinker_hidden shape: {:?} (embedding_size={})",
        zero_hidden.shape(),
        embedding_size
    );

    // Run Talker forward
    eprintln!("Running Talker forward pass ...");
    let logits = match talker.forward(&input_ids, &zero_hidden, 0) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Talker forward failed: {}", e);
            panic!("Talker forward failed: {}", e);
        }
    };
    eprintln!("Talker logits shape: {:?}", logits.shape());

    let our_logits = logits.to_vec_f32();

    // Load reference logits [1, 1, 8448]
    let ref_path = format!("{}/talker_logits.npy", REF_DIR);
    let (ref_data, ref_shape) = match load_npy_f32(&ref_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Skipping: failed to load reference logits: {}", e);
            return;
        }
    };
    eprintln!("Reference logits shape: {:?}", ref_shape);

    // Shape sanity check
    assert_eq!(
        our_logits.len(),
        ref_data.len(),
        "Output element count mismatch: Rust={} Reference={}",
        our_logits.len(),
        ref_data.len()
    );

    // Compute metrics
    let metrics = ValidationMetrics::compute("Talker logits", &our_logits, &ref_data);

    eprintln!("\n=== Talker Logits Validation ===");
    metrics.print();

    // Argmax comparison for the first (and only) codec token
    let vocab_size = talker_config.vocab_size;
    if our_logits.len() >= vocab_size && ref_data.len() >= vocab_size {
        let our_argmax = our_logits[..vocab_size]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let ref_argmax = ref_data[..vocab_size]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        eprintln!("\nFirst codec token argmax:");
        eprintln!("  Rust:   token {}", our_argmax);
        eprintln!("  Python: token {}", ref_argmax);
        if our_argmax == ref_argmax {
            eprintln!("  Argmax match.");
        } else {
            eprintln!("  Argmax differ.");
        }

        assert_eq!(
            our_argmax, ref_argmax,
            "Argmax mismatch: Rust={} Python={}",
            our_argmax, ref_argmax
        );
    }

    if metrics.is_valid(1e-1, 0.99) {
        eprintln!("\nPASS: Talker output matches Python reference.");
    } else {
        eprintln!("\nFAIL: Talker output diverged from Python reference.");
        eprintln!("  Thresholds: MSE < 1e-1, Cosine > 0.99, MaxDiff < 2.0");
        eprintln!(
            "  Actual:     MSE={:.2e}, Cosine={:.6}, MaxDiff={:.2e}",
            metrics.mse, metrics.cosine_similarity, metrics.max_abs_diff
        );
    }

    assert!(
        metrics.cosine_similarity > 0.99,
        "Talker logits cosine similarity too low: {:.6} (threshold 0.99)",
        metrics.cosine_similarity
    );
    assert!(
        metrics.mse < 1e-1,
        "Talker logits MSE too high: {:.2e} (threshold 1e-1)",
        metrics.mse
    );
    assert!(
        metrics.max_abs_diff < 2.0,
        "Talker logits max absolute diff too high: {:.2e} (threshold 2.0)",
        metrics.max_abs_diff
    );
}

/// Inspect shapes of all available talker reference files (informational).
///
/// Reports shapes of input_ids, codec embedding, projection, per-layer outputs,
/// final norm, and logits reference tensors for debugging and orientation.
#[test]
fn test_talker_layer_reference_shapes() {
    eprintln!("=== Talker Reference File Shapes ===");

    // Scalar/small reference files
    let scalar_files = [
        "talker_input_ids.npy",
        "talker_codec_embedding_output.npy",
        "talker_proj_output.npy",
        "talker_final_norm_output.npy",
        "talker_logits.npy",
    ];

    for filename in &scalar_files {
        let path = format!("{}/{}", REF_DIR, filename);
        if !Path::new(&path).exists() {
            eprintln!("  {}: NOT FOUND", filename);
            continue;
        }
        match load_npy_f32(&path) {
            Ok((data, shape)) => {
                eprintln!("  {}: shape {:?}, {} elements", filename, shape, data.len());
            }
            Err(_) => {
                // Might be i64 (input_ids) — try that
                match load_npy_i64(&path) {
                    Ok((data, shape)) => {
                        eprintln!(
                            "  {} (i64): shape {:?}, values {:?}",
                            filename, shape, data
                        );
                    }
                    Err(e) => {
                        eprintln!("  {}: failed to load — {}", filename, e);
                    }
                }
            }
        }
    }

    // Per-layer reference files
    eprintln!("\n  Per-layer outputs:");
    let mut found_any_layer = false;
    for i in 0..24 {
        let path = format!("{}/talker_layer_{:02}_output.npy", REF_DIR, i);
        if !Path::new(&path).exists() {
            break;
        }
        found_any_layer = true;
        match load_npy_f32(&path) {
            Ok((data, shape)) => {
                eprintln!("  talker_layer_{:02}_output.npy: shape {:?}, {} elements", i, shape, data.len());
            }
            Err(e) => {
                eprintln!("  talker_layer_{:02}_output.npy: failed to load — {}", i, e);
            }
        }
    }
    if !found_any_layer {
        eprintln!("  (no per-layer reference files found)");
    }
}
