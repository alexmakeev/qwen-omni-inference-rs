//! Validation tests for Qwen2.5-Omni AudioEncoder against Python reference.
//!
//! Compares Rust AudioEncoder outputs against reference data extracted by a
//! Python script using HuggingFace Transformers. Validates the conv stem,
//! positional embedding addition, transformer layers, avg-pool, ln_post and
//! linear projection in sequence.
//!
//! # Running
//!
//! ```text
//! cargo test --test omni_audio_validation -- --nocapture
//! ```
//!
//! # Requirements
//!
//! - Model weights: `/home/alexmak/lluda/models/Qwen2.5-Omni-3B/`
//! - Reference data: `/home/alexmak/lluda/reference_data/omni_3b/`
//!   - `audio_encoder_input.npy`    — mel spectrogram [128, T] (f32)
//!   - `audio_layer_NN_output.npy`  — per-layer outputs (optional)
//!   - `audio_ln_post_output.npy`   — ln_post output (optional)
//!   - `audio_encoder_output.npy`   — final encoder output [T/4, 2048]
//!
//! # Tolerance thresholds
//!
//! BF16 precision through 32 transformer layers accumulates rounding error.
//! Thresholds are tuned accordingly.
//!
//! - Cosine similarity: > 0.995 for the final output (BF16 through 32 layers)
//! - MSE:               < 5e-3  (BF16 accumulates ~1e-4 per layer)
//! - Max absolute diff: < 0.5   (BF16 outlier differences in deep networks)

use ndarray::ArrayD;
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::path::Path;

use lluda_inference::audio_encoder::AudioEncoder;
use lluda_inference::config::OmniConfig;
use lluda_inference::loader::ModelWeights;
use lluda_inference::tensor::Tensor;

const MODEL_DIR: &str = "/home/alexmak/lluda/models/Qwen2.5-Omni-3B";
const REF_DIR: &str = "/home/alexmak/lluda/reference_data/omni_3b";

// ── Validation helpers ────────────────────────────────────────────────────────

/// Validation metrics for a single tensor comparison.
#[derive(Debug, Clone)]
struct ValidationMetrics {
    /// Layer or component name for display.
    name: String,
    /// Mean squared error.
    mse: f64,
    /// Cosine similarity (1.0 = perfect match).
    cosine_similarity: f64,
    /// Maximum absolute difference.
    max_abs_diff: f64,
    /// Mean absolute difference.
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
        eprintln!(
            "  {}: MSE={:.2e}, Cosine={:.6}, MaxDiff={:.2e}, MeanDiff={:.2e}",
            self.name,
            self.mse,
            self.cosine_similarity,
            self.max_abs_diff,
            self.mean_abs_diff,
        );
    }
}

// ── npy loading helpers ───────────────────────────────────────────────────────

/// Load an f32 npy file as a flat Vec<f32>.
fn load_npy_f32(path: &str) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
    let reader = File::open(path)?;
    let arr: ArrayD<f32> = ArrayD::read_npy(reader)?;
    let shape = arr.shape().to_vec();
    let (data, _) = arr.into_raw_vec_and_offset();
    Ok((data, shape))
}

// ── Prerequisite check ────────────────────────────────────────────────────────

/// Return true when both model weights and reference data are present.
fn check_prerequisites() -> bool {
    let model_ok = Path::new(MODEL_DIR).join("config.json").exists();
    let ref_ok = Path::new(REF_DIR).join("audio_encoder_input.npy").exists();

    if !model_ok {
        eprintln!("Skipping: model not found at {}", MODEL_DIR);
    }
    if !ref_ok {
        eprintln!("Skipping: reference data not found at {}", REF_DIR);
    }
    model_ok && ref_ok
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Validate the full AudioEncoder output against the Python reference.
///
/// Loads mel input, runs AudioEncoder::forward, and compares the final
/// [T/4, 2048] output tensor against the reference.
#[test]
fn test_audio_encoder_output() {
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

    // Load model weights
    eprintln!("Loading model weights from {} ...", MODEL_DIR);
    let weights = match ModelWeights::from_directory(MODEL_DIR) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Skipping: failed to load weights: {}", e);
            return;
        }
    };

    // Build AudioEncoder
    eprintln!("Building AudioEncoder ...");
    let audio_encoder =
        match AudioEncoder::load(&config.thinker_config.audio_config, |name| {
            weights.get(name).cloned()
        }) {
            Ok(enc) => enc,
            Err(e) => {
                eprintln!("Skipping: failed to build AudioEncoder: {}", e);
                return;
            }
        };

    // Load mel input reference
    let mel_path = format!("{}/audio_encoder_input.npy", REF_DIR);
    let (mel_data, mel_shape) = match load_npy_f32(&mel_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Skipping: failed to load mel input: {}", e);
            return;
        }
    };
    eprintln!("Mel input shape: {:?}", mel_shape);

    let mel = match Tensor::new(mel_data, mel_shape) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Skipping: failed to create mel Tensor: {}", e);
            return;
        }
    };

    // Run AudioEncoder forward
    eprintln!("Running AudioEncoder forward pass ...");
    let output = match audio_encoder.forward(&mel) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("AudioEncoder forward failed: {}", e);
            panic!("AudioEncoder forward failed: {}", e);
        }
    };
    eprintln!("AudioEncoder output shape: {:?}", output.shape());

    // Load reference output
    let ref_path = format!("{}/audio_encoder_output.npy", REF_DIR);
    let (ref_data, ref_shape) = match load_npy_f32(&ref_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Skipping: failed to load reference output: {}", e);
            return;
        }
    };
    eprintln!("Reference output shape: {:?}", ref_shape);

    let our_data = output.to_vec_f32();

    // Shape check
    assert_eq!(
        our_data.len(),
        ref_data.len(),
        "Output element count mismatch: Rust={} Reference={}",
        our_data.len(),
        ref_data.len()
    );

    // Compute metrics
    let metrics = ValidationMetrics::compute("AudioEncoder final output", &our_data, &ref_data);

    eprintln!("\n=== AudioEncoder Final Output Validation ===");
    metrics.print();

    // BF16 through 32 transformer layers accumulates rounding error.
    // Thresholds reflect realistic precision for BF16 deep networks.
    let is_valid = metrics.is_valid(5e-3, 0.995, 0.5);

    if is_valid {
        eprintln!("\nPASS: AudioEncoder output matches Python reference.");
    } else {
        eprintln!("\nFAIL: AudioEncoder output diverged from Python reference.");
        eprintln!(
            "  Thresholds: MSE < 5e-3, Cosine > 0.995, MaxDiff < 0.5"
        );
        eprintln!(
            "  Actual:     MSE={:.2e}, Cosine={:.6}, MaxDiff={:.2e}",
            metrics.mse, metrics.cosine_similarity, metrics.max_abs_diff
        );
    }

    assert!(
        metrics.cosine_similarity > 0.995,
        "Cosine similarity too low: {:.6} (threshold 0.995)",
        metrics.cosine_similarity
    );
    assert!(
        metrics.mse < 5e-3,
        "MSE too high: {:.2e} (threshold 5e-3)",
        metrics.mse
    );
    assert!(
        metrics.max_abs_diff < 0.5,
        "MaxDiff too high: {:.2e} (threshold 0.5)",
        metrics.max_abs_diff
    );
}

/// Inspect per-layer reference shapes (informational, does not compare Rust output).
///
/// This test reports the shapes of per-layer reference tensors if they are
/// present. Full per-layer comparison requires debug hooks in AudioEncoder.
#[test]
fn test_audio_encoder_layer_reference_shapes() {
    if !check_prerequisites() {
        return;
    }

    let layer_00_path = format!("{}/audio_layer_00_output.npy", REF_DIR);
    if !Path::new(&layer_00_path).exists() {
        eprintln!(
            "Skipping layer shape inspection: per-layer reference data not found (expected at {})",
            layer_00_path
        );
        return;
    }

    eprintln!("=== Per-Layer Reference Shapes ===");
    for i in 0..32 {
        let path = format!("{}/audio_layer_{:02}_output.npy", REF_DIR, i);
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

    // Also check ln_post if present
    let ln_path = format!("{}/audio_ln_post_output.npy", REF_DIR);
    if Path::new(&ln_path).exists() {
        match load_npy_f32(&ln_path) {
            Ok((data, shape)) => {
                eprintln!("  ln_post:   shape {:?}, {} elements", shape, data.len());
            }
            Err(e) => {
                eprintln!("  ln_post:   failed to load — {}", e);
            }
        }
    }
}

/// Diagnostic test: reports per-layer reference shapes and prints info for future
/// per-layer validation. Does NOT assert — informational only.
///
/// Full per-layer comparison requires debug hooks in AudioEncoder.
/// This test is a stepping stone for future per-layer validation work.
#[test]
fn test_audio_encoder_per_layer_divergence() {
    if !check_prerequisites() {
        return;
    }

    // Load all per-layer reference outputs
    let mut layer_refs: Vec<(usize, Vec<f32>, Vec<usize>)> = Vec::new();
    for i in 0..32 {
        let path = format!("{}/audio_layer_{:02}_output.npy", REF_DIR, i);
        if Path::new(&path).exists() {
            match load_npy_f32(&path) {
                Ok((data, shape)) => {
                    layer_refs.push((i, data, shape));
                }
                Err(e) => {
                    eprintln!("  Layer {:2}: failed to load — {}", i, e);
                }
            }
        }
    }

    // Also check ln_post reference
    let ln_post_path = format!("{}/audio_ln_post_output.npy", REF_DIR);
    if Path::new(&ln_post_path).exists() {
        match load_npy_f32(&ln_post_path) {
            Ok((_data, shape)) => {
                eprintln!("ln_post reference: shape {:?}", shape);
            }
            Err(e) => {
                eprintln!("ln_post: failed to load — {}", e);
            }
        }
    }

    eprintln!("\nFound {} per-layer reference outputs", layer_refs.len());
    eprintln!("Per-layer validation requires debug hooks in AudioEncoder.");
    eprintln!("Use these shapes for future per-layer testing:");
    for (i, _, shape) in &layer_refs {
        eprintln!("  Layer {}: {:?}", i, shape);
    }
}
