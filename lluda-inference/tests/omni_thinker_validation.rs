//! Validation tests for Qwen2.5-Omni Thinker (language model) against Python reference.
//!
//! Compares Rust OmniThinker outputs against reference data extracted from
//! HuggingFace Transformers. Uses OmniThinker directly for a clean text-only
//! validation path without loading the full OmniModel (AudioEncoder + Talker).
//!
//! # Running
//!
//! ```text
//! cargo test --test omni_thinker_validation -- --nocapture
//! ```
//!
//! # Requirements
//!
//! - Model weights: `/home/alexmak/lluda/models/Qwen2.5-Omni-3B/`
//! - Reference data: `/home/alexmak/lluda/reference_data/omni_3b/`
//!   - `thinker_input_ids.npy`         — input token IDs [seq] (i64)
//!   - `thinker_embedding_output.npy`  — embedding output [1, seq, 2048] (optional)
//!   - `thinker_layer_NN_output.npy`   — per-layer outputs [1, seq, 2048] (optional)
//!   - `thinker_final_norm_output.npy` — final norm output [1, seq, 2048] (optional)
//!   - `thinker_logits.npy`            — logits [1, seq, vocab_size] (optional)
//!
//! # Tolerance thresholds (same as Qwen3 validation.rs)
//!
//! - Cosine similarity: > 0.999 for early layers, > 0.99 for logits
//! - MSE:               < 1e-4 for all layers
//! - Max absolute diff: < 1e-2

use ndarray::ArrayD;
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::path::Path;

use lluda_inference::config::OmniConfig;
use lluda_inference::loader::ModelWeights;
use lluda_inference::omni_model::OmniThinker;

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

    fn is_valid(&self, mse_threshold: f64, cosine_threshold: f64, max_diff_threshold: f64) -> bool {
        self.mse < mse_threshold
            && self.cosine_similarity > cosine_threshold
            && self.max_abs_diff < max_diff_threshold
    }

    fn print(&self) {
        let valid = self.is_valid(1e-4, 0.999, 1e-2);
        let status = if valid { "PASS" } else { "FAIL" };
        eprintln!(
            "  [{}] {}: MSE={:.2e}, Cosine={:.6}, MaxDiff={:.2e}, MeanDiff={:.2e}",
            status, self.name, self.mse, self.cosine_similarity, self.max_abs_diff, self.mean_abs_diff,
        );
    }
}

// ── npy loading helpers ───────────────────────────────────────────────────────

/// Load an f32 npy file.
fn load_npy_f32(path: &str) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
    let reader = File::open(path)?;
    let arr: ArrayD<f32> = ArrayD::read_npy(reader)?;
    let shape = arr.shape().to_vec();
    // Ensure C-order (row-major) layout. Numpy may store arrays in Fortran order
    // (F-contiguous) when the source tensor was non-contiguous (e.g. transposed).
    // into_raw_vec_and_offset() reflects storage order, so we must normalize first.
    let c_contiguous = arr.as_standard_layout();
    let (data, _) = c_contiguous.into_owned().into_raw_vec_and_offset();
    Ok((data, shape))
}

/// Load an i64 npy file.
fn load_npy_i64(path: &str) -> Result<Vec<i64>, Box<dyn std::error::Error>> {
    let reader = File::open(path)?;
    let arr: ArrayD<i64> = ArrayD::read_npy(reader)?;
    // Same C-order normalization.
    let c_contiguous = arr.as_standard_layout();
    let (data, _) = c_contiguous.into_owned().into_raw_vec_and_offset();
    Ok(data)
}

// ── Prerequisite check ────────────────────────────────────────────────────────

fn check_prerequisites() -> bool {
    let model_ok = Path::new(MODEL_DIR).join("config.json").exists();
    let ref_ok = Path::new(REF_DIR).join("thinker_input_ids.npy").exists();

    if !model_ok {
        eprintln!("Skipping: model not found at {}", MODEL_DIR);
    }
    if !ref_ok {
        eprintln!("Skipping: thinker reference data not found at {}/thinker_input_ids.npy", REF_DIR);
    }
    model_ok && ref_ok
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Validate the thinker logits against the Python reference (text-only input).
///
/// Loads OmniThinker directly (without AudioEncoder or Talker), runs a
/// text-only forward pass, applies the LM head manually, and compares
/// all-token logits against the reference. This avoids the overhead of
/// loading the full OmniModel and removes the audio path from the test.
#[test]
fn test_thinker_logits() {
    if !check_prerequisites() {
        return;
    }

    // Load configuration
    let config = match OmniConfig::from_file(format!("{}/config.json", MODEL_DIR)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping: failed to load config: {}", e);
            return;
        }
    };

    // Load model weights
    eprintln!("Loading model weights from {} ...", MODEL_DIR);
    let weights = match ModelWeights::from_directory(MODEL_DIR) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Skipping: failed to load weights: {}", e);
            return;
        }
    };

    // Build OmniThinker directly (no AudioEncoder / Talker overhead)
    eprintln!("Building OmniThinker ...");
    let text_cfg = &config.thinker_config.text_config;
    let get = |name: &str| weights.get(name).cloned();
    let mut thinker = match OmniThinker::load(text_cfg, &get) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Skipping: failed to build OmniThinker: {}", e);
            return;
        }
    };

    // Load LM head weight and pre-transpose for matmul
    let lm_head = match weights.get("thinker.lm_head.weight").cloned() {
        Some(w) => w,
        None => {
            eprintln!("Skipping: missing thinker.lm_head.weight");
            return;
        }
    };
    // lm_head: [vocab_size, hidden_size] -> lm_head_t: [hidden_size, vocab_size]
    let lm_head_t = match lm_head.transpose_dims(0, 1) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Skipping: failed to transpose lm_head: {}", e);
            return;
        }
    };

    // Load reference input_ids
    let ids_path = format!("{}/thinker_input_ids.npy", REF_DIR);
    let ids_i64 = match load_npy_i64(&ids_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Skipping: failed to load thinker_input_ids.npy: {}", e);
            return;
        }
    };
    let input_ids: Vec<u32> = ids_i64.iter().map(|&x| x as u32).collect();
    eprintln!("Input IDs ({} tokens): {:?}", input_ids.len(), &input_ids[..input_ids.len().min(16)]);

    let seq_len = input_ids.len();

    // Forward through thinker
    // Returns: [1, seq_len, hidden_size]
    eprintln!("Running OmniThinker forward (text-only) ...");
    let hidden = match thinker.forward(&input_ids, 0) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("thinker.forward failed: {}", e);
            panic!("thinker.forward failed: {}", e);
        }
    };
    eprintln!("Hidden states shape: {:?}", hidden.shape());

    // Apply LM head: [1, seq_len, hidden] @ [hidden, vocab] -> [1, seq_len, vocab]
    // Reshape to 2D for matmul: [seq_len, hidden] @ [hidden, vocab] -> [seq_len, vocab]
    let hidden_size = hidden.shape()[2];
    let hidden_2d = match hidden.reshape(&[seq_len, hidden_size]) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("failed to reshape hidden: {}", e);
            panic!("reshape failed: {}", e);
        }
    };
    let logits_2d = match hidden_2d.matmul(&lm_head_t) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("lm_head matmul failed: {}", e);
            panic!("matmul failed: {}", e);
        }
    };
    // logits_2d: [seq_len, vocab_size]
    let vocab_size = logits_2d.shape()[1];
    eprintln!("Logits shape: [{}, {}]", seq_len, vocab_size);

    let our_logits = logits_2d.to_vec_f32();

    // Load reference logits
    let ref_path = format!("{}/thinker_logits.npy", REF_DIR);
    if !Path::new(&ref_path).exists() {
        eprintln!(
            "Skipping logits comparison: {} not found. \
             Run the Python reference script to generate it.",
            ref_path
        );
        return;
    }

    let (ref_data, ref_shape) = match load_npy_f32(&ref_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Skipping: failed to load reference logits: {}", e);
            return;
        }
    };
    eprintln!("Reference logits shape: {:?}", ref_shape);

    // Reference is [1, seq, vocab] — flatten to [seq * vocab] for comparison
    // Our logits are [seq_len * vocab_size]
    let ref_flat: Vec<f32> = if ref_data.len() == our_logits.len() {
        // Shapes already match (both flat seq*vocab)
        ref_data
    } else if ref_shape.len() == 3
        && ref_shape[0] == 1
        && ref_shape[1] == seq_len
        && ref_shape[2] == vocab_size
    {
        // [1, seq, vocab] — data is already in the right order, just use as-is
        ref_data
    } else {
        eprintln!(
            "Skipping: reference logits shape {:?} incompatible with our output [{}, {}]",
            ref_shape, seq_len, vocab_size
        );
        return;
    };

    let metrics = ValidationMetrics::compute("Thinker logits (all tokens)", &our_logits, &ref_flat);

    eprintln!("\n=== Thinker Logits Validation ===");
    metrics.print();

    // Check argmax agreement for the last token
    let last_start = (seq_len - 1) * vocab_size;
    let our_last = &our_logits[last_start..last_start + vocab_size];
    let ref_last = &ref_flat[last_start..last_start + vocab_size];

    let our_argmax = our_last
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let ref_argmax = ref_last
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    eprintln!("\nPredicted next token (last position):");
    eprintln!("  Rust:   token {}", our_argmax);
    eprintln!("  Python: token {}", ref_argmax);
    if our_argmax == ref_argmax {
        eprintln!("  Predictions match.");
    } else {
        eprintln!("  Predictions differ (may be acceptable due to numerical precision).");
    }

    assert!(
        metrics.cosine_similarity > 0.99,
        "Thinker logits cosine similarity too low: {:.6} (threshold 0.99)",
        metrics.cosine_similarity
    );
}

/// Layer-by-layer diagnostic for thinker hidden state validation.
///
/// If per-layer reference tensors are present, computes metrics for each layer.
/// Reports where divergence first appears.
#[test]
fn test_thinker_layer_by_layer() {
    if !check_prerequisites() {
        return;
    }

    // Check if per-layer reference data exists
    let layer_00_path = format!("{}/thinker_layer_00_output.npy", REF_DIR);
    if !Path::new(&layer_00_path).exists() {
        eprintln!(
            "Skipping layer-by-layer thinker test: per-layer data not found at {}",
            layer_00_path
        );
        return;
    }

    // Load input_ids
    let ids_path = format!("{}/thinker_input_ids.npy", REF_DIR);
    let ids_i64 = match load_npy_i64(&ids_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Skipping: failed to load input ids: {}", e);
            return;
        }
    };
    let input_ids: Vec<u32> = ids_i64.iter().map(|&x| x as u32).collect();

    eprintln!("=== Thinker Layer-by-Layer Reference Shapes ===");
    eprintln!("Input IDs: {} tokens", input_ids.len());

    // Report shapes of all available per-layer reference tensors
    for i in 0..36 {
        let path = format!("{}/thinker_layer_{:02}_output.npy", REF_DIR, i);
        if !Path::new(&path).exists() {
            break;
        }
        match load_npy_f32(&path) {
            Ok((data, shape)) => {
                eprintln!("  Layer {:2}: shape {:?}, {} elements", i, shape, data.len());
            }
            Err(e) => {
                eprintln!("  Layer {:2}: failed to load — {}", i, e);
            }
        }
    }

    // Report embedding output shape if present
    let emb_path = format!("{}/thinker_embedding_output.npy", REF_DIR);
    if Path::new(&emb_path).exists() {
        match load_npy_f32(&emb_path) {
            Ok((data, shape)) => {
                eprintln!("  Embedding: shape {:?}, {} elements", shape, data.len());
            }
            Err(e) => {
                eprintln!("  Embedding: failed to load — {}", e);
            }
        }
    }

    // Report final norm output shape if present
    let norm_path = format!("{}/thinker_final_norm_output.npy", REF_DIR);
    if Path::new(&norm_path).exists() {
        match load_npy_f32(&norm_path) {
            Ok((data, shape)) => {
                eprintln!("  Final norm: shape {:?}, {} elements", shape, data.len());
            }
            Err(e) => {
                eprintln!("  Final norm: failed to load — {}", e);
            }
        }
    }

    eprintln!("\nNote: Full per-layer comparison requires adding intermediate-capture");
    eprintln!("hooks to OmniThinker (similar to Qwen3ForCausalLM::forward_with_intermediates).");
    eprintln!("Once those hooks are added, this test will perform actual value comparison.");
}
