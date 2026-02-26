//! Validation test: Rust mel_spectrogram vs WhisperFeatureExtractor reference.
//!
//! Loads raw PCM samples and a reference mel spectrogram produced by
//! HuggingFace WhisperFeatureExtractor, runs the Rust mel_spectrogram,
//! and asserts numerical agreement.
//!
//! # Running
//!
//! ```text
//! cargo test --release --test mel_spectrogram_validation -- --nocapture --ignored
//! ```
//!
//! # Reference data
//!
//! - `audio_samples.npy`          [29449] f32 — raw PCM at 16 kHz
//! - `mel_whisper_fe_cropped.npy` [128, 184] f32 — WhisperFE output cropped from 30000 frames
//!
//! # Thresholds
//!
//! The Rust and Python implementations share the same formula (power spectrum,
//! HTK mel filterbank, log10 + Whisper normalization), so agreement should be
//! very tight even in f32.
//!
//! - Shape:            [128, 184] (last STFT frame dropped, matching WhisperFE `stft[..., :-1]`)
//! - Cosine similarity: > 0.999
//! - MSE:               < 1e-4
//! - Max absolute diff: < 0.1

use ndarray::ArrayD;
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::path::Path;

use lluda_inference::audio_preprocess::mel_spectrogram;

const REF_DIR: &str = "/home/alexmak/lluda/reference_data/omni_mel_test";

// ── npy loading helper ────────────────────────────────────────────────────────

/// Load an f32 npy file into a flat Vec<f32> with shape.
///
/// Uses `as_standard_layout()` to normalise C-order (row-major) layout before
/// extracting the raw buffer. Numpy may store non-contiguous (e.g. transposed)
/// arrays in Fortran order; without this step the data would be in the wrong order.
fn load_npy_f32(path: &str) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
    let reader = File::open(path)?;
    let arr: ArrayD<f32> = ArrayD::read_npy(reader)?;
    let shape = arr.shape().to_vec();
    let c_contiguous = arr.as_standard_layout();
    let (data, _) = c_contiguous.into_owned().into_raw_vec_and_offset();
    Ok((data, shape))
}

// ── Validation metrics ────────────────────────────────────────────────────────

struct Metrics {
    mse: f64,
    cosine: f64,
    max_abs_diff: f64,
    mean_abs_diff: f64,
}

impl Metrics {
    fn compute(predicted: &[f32], reference: &[f32]) -> Self {
        assert_eq!(predicted.len(), reference.len(), "Length mismatch");
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
        let norm_p = predicted.iter().map(|&p| (p as f64) * (p as f64)).sum::<f64>().sqrt();
        let norm_r = reference.iter().map(|&r| (r as f64) * (r as f64)).sum::<f64>().sqrt();
        let cosine = if norm_p > 0.0 && norm_r > 0.0 {
            dot / (norm_p * norm_r)
        } else {
            0.0
        };

        Metrics { mse, cosine, max_abs_diff, mean_abs_diff }
    }

    fn print(&self) {
        eprintln!("  MSE:           {:.4e}", self.mse);
        eprintln!("  Cosine:        {:.8}", self.cosine);
        eprintln!("  Max abs diff:  {:.4e}", self.max_abs_diff);
        eprintln!("  Mean abs diff: {:.4e}", self.mean_abs_diff);
    }
}

// ── Test ──────────────────────────────────────────────────────────────────────

/// Validate Rust mel_spectrogram against WhisperFeatureExtractor reference.
///
/// This test is marked `#[ignore]` because it requires reference data files
/// that are not checked into the repository.  Run it explicitly with:
///
/// ```text
/// cargo test --release --test mel_spectrogram_validation -- --nocapture --ignored
/// ```
#[test]
#[ignore]
fn test_mel_spectrogram_matches_whisper_fe() {
    // Check reference files exist before doing any work.
    let audio_path = format!("{}/audio_samples.npy", REF_DIR);
    let ref_path = format!("{}/mel_whisper_fe_cropped.npy", REF_DIR);

    if !Path::new(&audio_path).exists() {
        eprintln!("Skipping: audio_samples.npy not found at {}", audio_path);
        return;
    }
    if !Path::new(&ref_path).exists() {
        eprintln!("Skipping: mel_whisper_fe_cropped.npy not found at {}", ref_path);
        return;
    }

    // --- Load audio samples ---
    eprintln!("=== mel_spectrogram validation vs WhisperFeatureExtractor ===");
    let (audio_data, audio_shape) = load_npy_f32(&audio_path)
        .expect("Failed to load audio_samples.npy");
    eprintln!("Audio samples: shape {:?}, {} values", audio_shape, audio_data.len());
    assert_eq!(audio_shape.len(), 1, "Expected 1-D audio array, got shape {:?}", audio_shape);
    assert_eq!(audio_data.len(), 29449, "Expected 29449 samples, got {}", audio_data.len());

    // --- Run Rust mel_spectrogram ---
    eprintln!("Running Rust mel_spectrogram ...");
    let mel = mel_spectrogram(&audio_data)
        .expect("mel_spectrogram failed");
    let mel_shape = mel.shape().to_vec();
    let mel_data = mel.to_vec_f32();
    eprintln!("Rust mel shape: {:?}", mel_shape);

    // --- Load reference ---
    let (ref_data, ref_shape) = load_npy_f32(&ref_path)
        .expect("Failed to load mel_whisper_fe_cropped.npy");
    eprintln!("Reference mel shape: {:?}", ref_shape);

    // --- Shape check ---
    assert_eq!(
        mel_shape,
        vec![128usize, 184],
        "Expected shape [128, 184] (WhisperFE drops last STFT frame), got {:?}",
        mel_shape
    );
    assert_eq!(
        ref_shape,
        vec![128usize, 184],
        "Reference shape should be [128, 184], got {:?}",
        ref_shape
    );
    assert_eq!(
        mel_data.len(),
        ref_data.len(),
        "Element count mismatch: Rust={}, Reference={}",
        mel_data.len(),
        ref_data.len()
    );

    // --- Numerical metrics ---
    let metrics = Metrics::compute(&mel_data, &ref_data);

    eprintln!("\nComparison results (Rust vs WhisperFE reference):");
    metrics.print();

    // --- Per-frame statistics for debugging ---
    eprintln!("\nPer-frame max absolute diff (first 10 frames):");
    let n_mels = 128usize;
    let n_frames = 184usize;
    for frame in 0..n_frames.min(10) {
        let mut frame_max = 0.0f32;
        for mel_idx in 0..n_mels {
            let idx = mel_idx * n_frames + frame;
            let d = (mel_data[idx] - ref_data[idx]).abs();
            if d > frame_max {
                frame_max = d;
            }
        }
        eprintln!("  frame {:3}: max_diff = {:.4e}", frame, frame_max);
    }

    // --- Assertions ---
    let pass_cosine = metrics.cosine > 0.999;
    let pass_mse = metrics.mse < 1e-4;
    let pass_max = metrics.max_abs_diff < 0.1;

    if pass_cosine && pass_mse && pass_max {
        eprintln!("\nPASS: Rust mel_spectrogram matches WhisperFeatureExtractor.");
    } else {
        eprintln!("\nFAIL: Rust mel_spectrogram diverged from reference.");
        eprintln!("  Thresholds: cosine > 0.999, MSE < 1e-4, MaxDiff < 0.1");
    }

    assert!(
        pass_cosine,
        "Cosine similarity too low: {:.8} (threshold 0.999)",
        metrics.cosine
    );
    assert!(
        pass_mse,
        "MSE too high: {:.4e} (threshold 1e-4)",
        metrics.mse
    );
    assert!(
        pass_max,
        "Max absolute diff too high: {:.4e} (threshold 0.1)",
        metrics.max_abs_diff
    );
}
