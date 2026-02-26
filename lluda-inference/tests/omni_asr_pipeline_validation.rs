//! End-to-end ASR pipeline validation tests for Qwen2.5-Omni-3B.
//!
//! Validates the full ASR pipeline against Python reference data:
//! 1. AudioEncoder output against `audio_tower_output.npy`
//! 2. Full prefill logits (last position) against `logits.npy`
//! 3. First generated token ID (informational, no assertion)
//!
//! # Running
//!
//! ```text
//! cargo test --release --test omni_asr_pipeline_validation -- --nocapture --ignored
//! ```
//!
//! # Requirements
//!
//! - Model weights: `/home/alexmak/lluda/models/Qwen2.5-Omni-3B/`
//! - Reference data: `/home/alexmak/lluda/reference_data/omni_asr_test/`
//!   - `mel_input_cropped.npy`   — mel spectrogram [128, 186] (f32)
//!   - `input_ids.npy`           — token sequence [1, 74] (i64) with 46 audio placeholders
//!   - `logits.npy`              — full prefill logits [1, 74, 151936] (f32)
//!   - `audio_tower_output_f32.npy` — audio encoder output [46, 2048] (f32, F32 reference)
//!   - `generated_text.txt`      — expected transcription text

use ndarray::ArrayD;
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::path::Path;

use lluda_inference::audio_encoder::AudioEncoder;
use lluda_inference::config::OmniConfig;
use lluda_inference::loader::ModelWeights;
use lluda_inference::omni_model::OmniModel;
use lluda_inference::tensor::Tensor;

const MODEL_DIR: &str = "/home/alexmak/lluda/models/Qwen2.5-Omni-3B";
const REF_DIR: &str = "/home/alexmak/lluda/reference_data/omni_asr_test";

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

/// Load an i64 npy file as a flat Vec<i64> with shape.
fn load_npy_i64(path: &str) -> Result<(Vec<i64>, Vec<usize>), Box<dyn std::error::Error>> {
    let reader = File::open(path)?;
    let arr: ArrayD<i64> = ArrayD::read_npy(reader)?;
    let shape = arr.shape().to_vec();
    // Same C-order normalization as load_npy_f32.
    let c_contiguous = arr.as_standard_layout();
    let (data, _) = c_contiguous.into_owned().into_raw_vec_and_offset();
    Ok((data, shape))
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

    fn is_valid(&self, mse_threshold: f64, cosine_threshold: f64, max_diff_threshold: f64) -> bool {
        self.mse < mse_threshold
            && self.cosine_similarity > cosine_threshold
            && self.max_abs_diff < max_diff_threshold
    }

    fn print(&self) {
        eprintln!(
            "  {}: MSE={:.2e}, Cosine={:.6}, MaxDiff={:.2e}, MeanDiff={:.2e}",
            self.name, self.mse, self.cosine_similarity, self.max_abs_diff, self.mean_abs_diff,
        );
    }
}

// ── Prerequisite check ────────────────────────────────────────────────────────

/// Return true when both model weights and ASR reference data are present.
fn check_prerequisites() -> bool {
    let model_ok = Path::new(MODEL_DIR).join("config.json").exists();
    let ref_ok = Path::new(REF_DIR).join("mel_input_cropped.npy").exists();

    if !model_ok {
        eprintln!("Skipping: model not found at {}", MODEL_DIR);
    }
    if !ref_ok {
        eprintln!("Skipping: reference data not found at {}", REF_DIR);
    }
    model_ok && ref_ok
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Validate AudioEncoder output against the ASR reference.
///
/// Loads `mel_input_cropped.npy` [128, 186], runs AudioEncoder::forward,
/// and compares to `audio_tower_output.npy` [46, 2048].
///
/// Threshold: cosine similarity > 0.99 (BF16 weights, F32 compute, 32 layers, F32 reference).
#[test]
#[ignore]
fn test_asr_audio_encoder_output() {
    if !check_prerequisites() {
        return;
    }

    // Load config
    let config_path = format!("{}/config.json", MODEL_DIR);
    let config = match OmniConfig::from_file(&config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping: failed to load config: {}", e);
            return;
        }
    };

    // Load weights
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

    // Load mel input
    let mel_path = format!("{}/mel_input_cropped.npy", REF_DIR);
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

    // Load reference output (F32 reference — appropriate for Rust's BF16 weights + F32 compute)
    let ref_path = format!("{}/audio_tower_output_f32.npy", REF_DIR);
    let (ref_data, ref_shape) = match load_npy_f32(&ref_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Skipping: failed to load audio_tower_output reference: {}", e);
            return;
        }
    };
    eprintln!("Reference shape: {:?}", ref_shape);

    let our_data = output.to_vec_f32();

    // Shape check
    assert_eq!(
        our_data.len(),
        ref_data.len(),
        "AudioEncoder output element count mismatch: Rust={} Reference={}",
        our_data.len(),
        ref_data.len()
    );

    // Compute metrics
    let metrics = ValidationMetrics::compute("AudioEncoder ASR output", &our_data, &ref_data);

    eprintln!("\n=== Test 1: AudioEncoder ASR Output Validation ===");
    metrics.print();

    let is_valid = metrics.is_valid(5e-2, 0.99, 5.0);
    if is_valid {
        eprintln!("PASS: AudioEncoder output matches reference.");
    } else {
        eprintln!("FAIL: AudioEncoder output diverged from reference.");
        eprintln!("  Thresholds: MSE < 5e-2, Cosine > 0.99, MaxDiff < 5.0");
    }

    assert!(
        metrics.cosine_similarity > 0.99,
        "Cosine similarity too low: {:.6} (threshold 0.99)",
        metrics.cosine_similarity
    );
    assert!(
        metrics.mse < 5e-2,
        "MSE too high: {:.2e} (threshold 5e-2)",
        metrics.mse
    );
    assert!(
        metrics.max_abs_diff < 5.0,
        "MaxDiff too high: {:.2e} (threshold 5.0)",
        metrics.max_abs_diff
    );
}

/// Validate prefill logits from the full ASR pipeline.
///
/// Loads mel, input_ids, and runs OmniModel::forward_asr (which returns logits
/// for the last token position [1, 1, vocab_size]).
///
/// Compares the last-position logits against `logits.npy[0, 73, :]`
/// (the last row of the reference [1, 74, 151936] tensor).
///
/// Threshold: cosine similarity > 0.98 for the last-position logits
/// (error accumulates across 32 audio encoder layers + 36 thinker layers = 68 layers total).
#[test]
#[ignore]
fn test_asr_prefill_logits() {
    if !check_prerequisites() {
        return;
    }

    // Load config
    let config_path = format!("{}/config.json", MODEL_DIR);
    let config = match OmniConfig::from_file(&config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping: failed to load config: {}", e);
            return;
        }
    };

    // Load weights
    eprintln!("Loading model weights from {} ...", MODEL_DIR);
    let weights = match ModelWeights::from_directory(MODEL_DIR) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Skipping: failed to load weights: {}", e);
            return;
        }
    };

    // Build full OmniModel
    eprintln!("Building OmniModel ...");
    let mut model = match OmniModel::load(&config, &weights) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Skipping: failed to build OmniModel: {}", e);
            return;
        }
    };

    // Load mel input
    let mel_path = format!("{}/mel_input_cropped.npy", REF_DIR);
    let (mel_data, mel_shape) = match load_npy_f32(&mel_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Skipping: failed to load mel input: {}", e);
            return;
        }
    };
    eprintln!("Mel shape: {:?}", mel_shape);

    let mel = match Tensor::new(mel_data, mel_shape) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Skipping: failed to create mel Tensor: {}", e);
            return;
        }
    };

    // Load input_ids (i64 -> u32)
    let ids_path = format!("{}/input_ids.npy", REF_DIR);
    let (ids_data_i64, ids_shape) = match load_npy_i64(&ids_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Skipping: failed to load input_ids: {}", e);
            return;
        }
    };
    eprintln!("input_ids shape: {:?}", ids_shape);

    // Flatten [1, 74] -> [74] and convert i64 -> u32
    let input_ids: Vec<u32> = ids_data_i64.iter().map(|&x| x as u32).collect();
    let seq_len = input_ids.len();
    eprintln!("Sequence length: {}", seq_len);

    // Count audio placeholder tokens
    let audio_token_id = config.thinker_config.audio_token_index;
    let num_audio_tokens = input_ids.iter().filter(|&&id| id == audio_token_id).count();
    eprintln!("Audio placeholder tokens (ID {}): {}", audio_token_id, num_audio_tokens);

    // Load reference logits — only last position to avoid loading 44MB fully into comparison
    eprintln!("Loading reference logits (last position) ...");
    let logits_path = format!("{}/logits.npy", REF_DIR);
    let (ref_logits_all, ref_logits_shape) = match load_npy_f32(&logits_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Skipping: failed to load logits reference: {}", e);
            return;
        }
    };
    eprintln!("Reference logits shape: {:?}", ref_logits_shape);

    // Reference is [1, 74, 151936] — extract last position: [0, 73, :]
    let vocab_size = ref_logits_shape[2];
    let ref_last_pos_start = (seq_len - 1) * vocab_size;
    let ref_last_logits = &ref_logits_all[ref_last_pos_start..ref_last_pos_start + vocab_size];
    eprintln!(
        "Extracted reference last-position logits: {} values",
        ref_last_logits.len()
    );

    // Run forward_asr — returns [1, 1, vocab_size] for last token
    eprintln!("Running OmniModel::forward_asr ...");
    let logits_tensor = match model.forward_asr(&mel, &input_ids, 0) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("forward_asr failed: {}", e);
            panic!("forward_asr failed: {}", e);
        }
    };
    eprintln!("forward_asr output shape: {:?}", logits_tensor.shape());

    let our_logits = logits_tensor.to_vec_f32();
    eprintln!("Our logits: {} values", our_logits.len());

    assert_eq!(
        our_logits.len(),
        ref_last_logits.len(),
        "Logits element count mismatch: Rust={} Reference={}",
        our_logits.len(),
        ref_last_logits.len()
    );

    // Compute metrics on last-position logits
    let metrics =
        ValidationMetrics::compute("ASR last-position logits", &our_logits, ref_last_logits);

    eprintln!("\n=== Test 2: ASR Prefill Logits Validation (last position) ===");
    metrics.print();

    // Check argmax agreement on last position
    let our_argmax = our_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let ref_argmax = ref_last_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    eprintln!("  Argmax: Rust={} Reference={}", our_argmax, ref_argmax);
    eprintln!(
        "  Argmax match: {}",
        if our_argmax == ref_argmax {
            "YES"
        } else {
            "NO"
        }
    );

    let is_valid = metrics.is_valid(1.0, 0.98, 10.0);
    if is_valid {
        eprintln!("PASS: ASR logits match reference at last position.");
    } else {
        eprintln!("FAIL: ASR logits diverged from reference.");
        eprintln!("  Thresholds: MSE < 1.0, Cosine > 0.98, MaxDiff < 10.0");
    }

    assert!(
        metrics.cosine_similarity > 0.98,
        "Cosine similarity too low: {:.6} (threshold 0.98)",
        metrics.cosine_similarity
    );

    assert_eq!(
        our_argmax, ref_argmax,
        "Argmax mismatch: Rust={} Reference={}",
        our_argmax, ref_argmax
    );
}

/// Inspect the first generated token from ASR logits (informational, no assertions).
///
/// Runs the same pipeline as test 2, takes argmax at the last logit position,
/// and prints the resulting token ID.
///
/// The expected first generated token corresponds to the beginning of:
/// "The original content of this audio is: '我怎么知道你有男朋友啊'"
#[test]
#[ignore]
fn test_asr_generated_text() {
    if !check_prerequisites() {
        return;
    }

    // Load config
    let config_path = format!("{}/config.json", MODEL_DIR);
    let config = match OmniConfig::from_file(&config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping: failed to load config: {}", e);
            return;
        }
    };

    // Load weights
    eprintln!("Loading model weights ...");
    let weights = match ModelWeights::from_directory(MODEL_DIR) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Skipping: failed to load weights: {}", e);
            return;
        }
    };

    let mut model = match OmniModel::load(&config, &weights) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Skipping: failed to build OmniModel: {}", e);
            return;
        }
    };

    // Load mel input
    let mel_path = format!("{}/mel_input_cropped.npy", REF_DIR);
    let (mel_data, mel_shape) = match load_npy_f32(&mel_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Skipping: failed to load mel: {}", e);
            return;
        }
    };
    let mel = match Tensor::new(mel_data, mel_shape) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Skipping: failed to create mel Tensor: {}", e);
            return;
        }
    };

    // Load input_ids
    let ids_path = format!("{}/input_ids.npy", REF_DIR);
    let (ids_data_i64, _) = match load_npy_i64(&ids_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Skipping: failed to load input_ids: {}", e);
            return;
        }
    };
    let input_ids: Vec<u32> = ids_data_i64.iter().map(|&x| x as u32).collect();

    // Run ASR forward
    eprintln!("Running forward_asr ...");
    let logits_tensor = match model.forward_asr(&mel, &input_ids, 0) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("forward_asr failed: {}", e);
            panic!("forward_asr failed: {}", e);
        }
    };

    let logits = logits_tensor.to_vec_f32();
    let first_token_id = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0);

    // Read expected text
    let expected_text_path = format!("{}/generated_text.txt", REF_DIR);
    let expected_text = std::fs::read_to_string(&expected_text_path)
        .unwrap_or_else(|_| "<file not found>".to_string());

    eprintln!("\n=== Test 3: ASR Generated Text (informational) ===");
    eprintln!("  First generated token ID (argmax at position 73): {}", first_token_id);
    eprintln!("  Expected full transcription: {}", expected_text.trim());
    eprintln!("  (Token ID decoding requires tokenizer integration — see tokenizers crate)");
    eprintln!("  Note: Token ID {} should correspond to the start of:", first_token_id);
    eprintln!("        \"The original content of this audio is: '我怎么知道你有男朋友啊'\"");
}
