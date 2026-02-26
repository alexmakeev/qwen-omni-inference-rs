//! End-to-end TTS pipeline validation tests for Qwen2.5-Omni-3B.
//!
//! Validates the full TTS pipeline against Python reference data:
//! 1. Thinker last hidden state against `tts_last_hidden.npy`
//! 2. Full TTS codec logits (first position) against `tts_codec_logits.npy`
//!
//! # Running
//!
//! ```text
//! cargo test --release --test omni_tts_pipeline_validation -- --nocapture --ignored
//! ```
//!
//! # Requirements
//!
//! - Model weights: `/home/alexmak/lluda/models/Qwen2.5-Omni-3B/`
//! - Reference data: `/home/alexmak/lluda/reference_data/omni_tts_test/`
//!   - `tts_input_ids.npy`      — text prompt tokens [1, 23] (f32 stored, integer values)
//!   - `tts_thinker_hidden.npy` — thinker hidden states [1, 23, 2048] (f32)
//!   - `tts_last_hidden.npy`    — last hidden state [1, 1, 2048] (f32)
//!   - `tts_proj_output.npy`    — after thinker_to_talker_proj [1, 1, 896] (f32)
//!   - `tts_codec_logits.npy`   — first codec logits [1, 1, 8448] (f32)
//!   - `tts_codec_argmax.txt`   — "5515"

use ndarray::ArrayD;
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::path::Path;

use lluda_inference::config::OmniConfig;
use lluda_inference::loader::ModelWeights;
use lluda_inference::omni_model::OmniModel;

const MODEL_DIR: &str = "/home/alexmak/lluda/models/Qwen2.5-Omni-3B";
const REF_DIR: &str = "/home/alexmak/lluda/reference_data/omni_tts_test";

// ── npy loading helpers ───────────────────────────────────────────────────────

/// Load an f32 npy file as a flat Vec<f32> with shape.
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

/// Load input_ids npy file — stored as float32 but contains integer values.
///
/// The Python reference saves token IDs as float32 tensors; we load as f32
/// and convert each value to u32 via truncation.
fn load_input_ids_as_u32(path: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let (data, _shape) = load_npy_f32(path)?;
    let ids: Vec<u32> = data.iter().map(|&v| v as u32).collect();
    Ok(ids)
}

// ── Validation metrics ────────────────────────────────────────────────────────

/// Validation metrics for comparing two tensors.
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
        let norm_p: f64 = predicted
            .iter()
            .map(|&p| (p as f64) * (p as f64))
            .sum::<f64>()
            .sqrt();
        let norm_r: f64 = reference
            .iter()
            .map(|&r| (r as f64) * (r as f64))
            .sum::<f64>()
            .sqrt();
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

    fn print(&self) {
        eprintln!(
            "  {}: MSE={:.2e}, Cosine={:.6}, MaxDiff={:.2e}, MeanDiff={:.2e}",
            self.name, self.mse, self.cosine_similarity, self.max_abs_diff, self.mean_abs_diff,
        );
    }
}

// ── Prerequisite check ────────────────────────────────────────────────────────

/// Return true when both model weights and TTS reference data are present.
fn check_prerequisites() -> bool {
    let model_ok = Path::new(MODEL_DIR).join("config.json").exists();
    let ref_ok = Path::new(REF_DIR).join("tts_input_ids.npy").exists();

    if !model_ok {
        eprintln!("Skipping: model not found at {}", MODEL_DIR);
    }
    if !ref_ok {
        eprintln!("Skipping: reference data not found at {}", REF_DIR);
    }
    model_ok && ref_ok
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Validate full TTS pipeline codec logits.
///
/// Loads `tts_input_ids.npy` [1, 23], runs `OmniModel::forward_tts`, and
/// compares the resulting codec logits [1, 1, 8448] against `tts_codec_logits.npy`.
///
/// Threshold: cosine similarity > 0.98 (error accumulates across 36 thinker +
/// 24 talker = 60 layers total with Q8 weights).
/// Also checks that argmax matches the reference value 5515.
#[test]
#[ignore]
fn test_tts_pipeline_codec_logits() {
    if !check_prerequisites() {
        return;
    }

    // Load config
    let config_path = format!("{}/config.json", MODEL_DIR);
    let mut config = match OmniConfig::from_file(&config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping: failed to load config: {}", e);
            return;
        }
    };

    // Ensure talker is enabled (required for TTS pipeline)
    if !config.enable_talker {
        eprintln!("Note: enable_talker=false in config; overriding to true for TTS test");
        config.enable_talker = true;
    }

    // Load weights
    eprintln!("Loading model weights from {} ...", MODEL_DIR);
    let weights = match ModelWeights::from_directory(MODEL_DIR) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Skipping: failed to load weights: {}", e);
            return;
        }
    };

    // Build OmniModel with talker enabled
    eprintln!("Building OmniModel (enable_talker=true) ...");
    let mut model = match OmniModel::load(&config, &weights) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Skipping: failed to build OmniModel: {}", e);
            return;
        }
    };

    // Load input_ids (stored as f32, values are integer token IDs)
    let ids_path = format!("{}/tts_input_ids.npy", REF_DIR);
    let input_ids = match load_input_ids_as_u32(&ids_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Skipping: failed to load tts_input_ids: {}", e);
            return;
        }
    };
    eprintln!("Input IDs ({}): {:?}", input_ids.len(), &input_ids[..5.min(input_ids.len())]);

    // Run forward_tts: thinker + talker → codec logits [1, 1, 8448]
    eprintln!("Running OmniModel::forward_tts ...");
    let codec_logits_tensor = match model.forward_tts(&input_ids, 0) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("forward_tts failed: {}", e);
            panic!("forward_tts failed: {}", e);
        }
    };
    eprintln!("forward_tts output shape: {:?}", codec_logits_tensor.shape());

    // Load reference codec logits [1, 1, 8448]
    let ref_logits_path = format!("{}/tts_codec_logits.npy", REF_DIR);
    let (ref_logits, ref_shape) = match load_npy_f32(&ref_logits_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Skipping: failed to load tts_codec_logits: {}", e);
            return;
        }
    };
    eprintln!("Reference codec logits shape: {:?}", ref_shape);

    let our_logits = codec_logits_tensor.to_vec_f32();
    eprintln!(
        "Our logits: {} values, Reference: {} values",
        our_logits.len(),
        ref_logits.len()
    );

    // Shape check: both should be [1, 1, 8448] = 8448 values
    assert_eq!(
        our_logits.len(),
        ref_logits.len(),
        "Codec logits element count mismatch: Rust={} Reference={}",
        our_logits.len(),
        ref_logits.len()
    );

    // Compute validation metrics
    let metrics = ValidationMetrics::compute("TTS codec logits", &our_logits, &ref_logits);

    eprintln!("\n=== Test 1: TTS Pipeline Codec Logits Validation ===");
    metrics.print();

    // Argmax check
    let our_argmax = our_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let ref_argmax = ref_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    eprintln!("  Argmax: Rust={} Reference={}", our_argmax, ref_argmax);
    eprintln!(
        "  Argmax match: {}",
        if our_argmax == ref_argmax { "YES" } else { "NO" }
    );
    eprintln!("  Expected argmax from tts_codec_argmax.txt: 5515");

    let is_valid = metrics.cosine_similarity > 0.98;
    if is_valid {
        eprintln!("PASS: TTS codec logits match reference (cosine > 0.98).");
    } else {
        eprintln!("FAIL: TTS codec logits diverged from reference.");
        eprintln!("  Threshold: Cosine > 0.98");
    }

    assert!(
        metrics.cosine_similarity > 0.98,
        "Cosine similarity too low: {:.6} (threshold 0.98). \
         Error may accumulate across 36 thinker + 24 talker = 60 layers.",
        metrics.cosine_similarity
    );

    assert_eq!(
        our_argmax,
        5515,
        "Argmax mismatch from expected 5515: got {}",
        our_argmax
    );
}

/// Validate thinker hidden states for TTS.
///
/// Loads `tts_input_ids.npy` [1, 23], runs only the thinker forward pass,
/// and compares the last hidden state [1, 1, 2048] against `tts_last_hidden.npy`.
///
/// Threshold: cosine similarity > 0.99 (36 thinker layers only).
#[test]
#[ignore]
fn test_tts_thinker_hidden_comparison() {
    if !check_prerequisites() {
        return;
    }

    // Load config
    let config_path = format!("{}/config.json", MODEL_DIR);
    let mut config = match OmniConfig::from_file(&config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping: failed to load config: {}", e);
            return;
        }
    };

    // Enable talker for consistent model construction, but we only use thinker here.
    if !config.enable_talker {
        eprintln!("Note: enable_talker=false in config; overriding to true");
        config.enable_talker = true;
    }

    // Load weights
    eprintln!("Loading model weights from {} ...", MODEL_DIR);
    let weights = match ModelWeights::from_directory(MODEL_DIR) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Skipping: failed to load weights: {}", e);
            return;
        }
    };

    // Build OmniModel
    eprintln!("Building OmniModel ...");
    let mut model = match OmniModel::load(&config, &weights) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Skipping: failed to build OmniModel: {}", e);
            return;
        }
    };

    // Load input_ids (stored as f32, values are integer token IDs)
    let ids_path = format!("{}/tts_input_ids.npy", REF_DIR);
    let input_ids = match load_input_ids_as_u32(&ids_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Skipping: failed to load tts_input_ids: {}", e);
            return;
        }
    };
    eprintln!("Input IDs ({}): {:?}", input_ids.len(), &input_ids[..5.min(input_ids.len())]);

    // Run thinker only to get full hidden states [1, seq_len, 2048]
    eprintln!("Running OmniModel::forward_thinker (thinker only) ...");
    let thinker_hidden = match model.forward_thinker(&input_ids, 0) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("forward_thinker failed: {}", e);
            panic!("forward_thinker failed: {}", e);
        }
    };
    eprintln!("Thinker hidden shape: {:?}", thinker_hidden.shape());

    // Extract last hidden state: [1, seq_len, 2048] -> [1, 1, 2048]
    let seq_len = thinker_hidden.shape()[1];
    let last_hidden = match thinker_hidden.narrow(1, seq_len - 1, 1) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("narrow failed: {}", e);
            panic!("narrow failed: {}", e);
        }
    };
    eprintln!("Last hidden shape (after narrow): {:?}", last_hidden.shape());

    // Load reference last hidden state [1, 1, 2048]
    let ref_path = format!("{}/tts_last_hidden.npy", REF_DIR);
    let (ref_data, ref_shape) = match load_npy_f32(&ref_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Skipping: failed to load tts_last_hidden: {}", e);
            return;
        }
    };
    eprintln!("Reference last_hidden shape: {:?}", ref_shape);

    let our_data = last_hidden.to_vec_f32();
    eprintln!(
        "Our last_hidden: {} values, Reference: {} values",
        our_data.len(),
        ref_data.len()
    );

    // Shape check: both should be [1, 1, 2048] = 2048 values
    assert_eq!(
        our_data.len(),
        ref_data.len(),
        "Last hidden element count mismatch: Rust={} Reference={}",
        our_data.len(),
        ref_data.len()
    );

    // Compute validation metrics
    let metrics = ValidationMetrics::compute("TTS thinker last hidden", &our_data, &ref_data);

    eprintln!("\n=== Test 2: TTS Thinker Hidden State Validation ===");
    metrics.print();

    let is_valid = metrics.cosine_similarity > 0.99;
    if is_valid {
        eprintln!("PASS: Thinker last hidden state matches reference (cosine > 0.99).");
    } else {
        eprintln!("FAIL: Thinker last hidden state diverged from reference.");
        eprintln!("  Threshold: Cosine > 0.99");
    }

    assert!(
        metrics.cosine_similarity > 0.99,
        "Cosine similarity too low: {:.6} (threshold 0.99). \
         Thinker only (36 layers) should match closely.",
        metrics.cosine_similarity
    );
}
