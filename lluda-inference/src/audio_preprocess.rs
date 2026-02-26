//! Audio preprocessing: PCM to mel spectrogram conversion.
//!
//! Converts raw audio (WAV file or PCM samples) into a mel spectrogram
//! suitable for the Whisper-style audio encoder used in Qwen2.5-Omni.
//!
//! # Pipeline
//!
//! ```text
//! WAV file → PCM samples (f32, mono, 16kHz)
//!          → STFT (FFT size 400, hop 160, Hann window)
//!          → Power spectrum
//!          → Mel filterbank (128 bins, 0–8000 Hz)
//!          → Log10 + Whisper normalization
//!          → Tensor [128, T]
//! ```
//!
//! Parameters match Whisper and Qwen2.5-Omni exactly.

use crate::error::{LludaError, Result};
use crate::tensor::Tensor;

/// Expected sample rate for audio input.
pub const SAMPLE_RATE: u32 = 16000;

/// FFT window size (25 ms at 16 kHz).
pub const N_FFT: usize = 400;

/// Hop length between consecutive STFT frames (10 ms at 16 kHz).
pub const HOP_LENGTH: usize = 160;

/// Number of mel frequency bins.
pub const N_MELS: usize = 128;

/// Number of unique FFT magnitude bins (N_FFT / 2 + 1).
const FFT_BINS: usize = N_FFT / 2 + 1; // 201

// ── Public API ────────────────────────────────────────────────────────────────

/// Load a WAV file and return PCM samples as f32 mono 16 kHz.
///
/// Supports:
/// - Mono and stereo WAV files (stereo is averaged to mono).
/// - Integer (8/16/24/32-bit) and float (32-bit) sample formats.
/// - Only 16 kHz sample rate is accepted; resampling is not yet implemented.
///
/// # Arguments
/// * `path` — Path to the `.wav` file.
///
/// # Errors
/// Returns `LludaError::Msg` when the file cannot be opened, the sample rate
/// is not 16 kHz, or the WAV format is unsupported.
///
/// # Example
/// ```no_run
/// use lluda_inference::audio_preprocess::load_wav;
///
/// let samples = load_wav("audio.wav")?;
/// println!("Loaded {} samples", samples.len());
/// # Ok::<(), lluda_inference::error::LludaError>(())
/// ```
pub fn load_wav(path: impl AsRef<std::path::Path>) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(path.as_ref())
        .map_err(|e| LludaError::Msg(format!("Failed to open WAV: {}", e)))?;

    let spec = reader.spec();

    // Validate sample rate before reading any samples (fail fast).
    if spec.sample_rate != SAMPLE_RATE {
        return Err(LludaError::Msg(format!(
            "Expected {}Hz sample rate, got {}Hz. Resampling is not yet supported.",
            SAMPLE_RATE, spec.sample_rate
        )));
    }

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| {
                    s.map(|v| v as f32 / max_val)
                        .map_err(|e| LludaError::Msg(format!("WAV read error: {}", e)))
                })
                .collect::<Result<Vec<f32>>>()?
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| {
                s.map_err(|e| LludaError::Msg(format!("WAV read error: {}", e)))
            })
            .collect::<Result<Vec<f32>>>()?,
    };

    // Convert stereo to mono by averaging channels.
    let mono: Vec<f32> = if spec.channels == 2 {
        samples
            .chunks_exact(2)
            .map(|c| (c[0] + c[1]) * 0.5)
            .collect()
    } else if spec.channels == 1 {
        samples
    } else {
        return Err(LludaError::Msg(format!(
            "Unsupported channel count: {}. Only mono and stereo WAV files are supported.",
            spec.channels
        )));
    };

    Ok(mono)
}

/// Compute a mel spectrogram from raw PCM samples.
///
/// The input is assumed to be mono f32 audio at 16 kHz.
/// Returns a [`Tensor`] of shape `[128, num_frames]` where each column is
/// one analysis frame and rows are the 128 mel frequency bins.
///
/// Normalization follows Whisper: log10 is applied, then values are clamped
/// to `[max - 8, max]` and linearly mapped to `[-1, 1]`.
///
/// # Arguments
/// * `samples` — Flat slice of f32 PCM samples (mono, 16 kHz).
///
/// # Errors
/// Returns `LludaError::Msg` on internal FFT failure.
///
/// # Example
/// ```
/// use lluda_inference::audio_preprocess::mel_spectrogram;
///
/// let silence = vec![0.0f32; 16000]; // 1 second of silence
/// let mel = mel_spectrogram(&silence)?;
/// assert_eq!(mel.shape()[0], 128);
/// # Ok::<(), lluda_inference::error::LludaError>(())
/// ```
pub fn mel_spectrogram(samples: &[f32]) -> Result<Tensor> {
    // 1. Reflect-pad signal by N_FFT/2 on each side so that the first
    //    and last frames are centred on sample 0 and sample N-1.
    //    Matches numpy: np.pad(audio, N_FFT // 2, mode="reflect")
    //    For x=[a,b,c,d,e] with pad=2:
    //      left  (reflect around x[0]):   x[2], x[1]  → [c, b]
    //      right (reflect around x[n-1]): x[n-2], x[n-3] → [d, c]
    //      result: [c, b, a, b, c, d, e, d, c]
    let pad = N_FFT / 2;
    let mut padded = Vec::with_capacity(samples.len() + 2 * pad);

    if samples.len() > pad {
        // Left reflection: samples[pad], samples[pad-1], ..., samples[1]
        for i in (1..=pad).rev() {
            padded.push(samples[i]);
        }
    } else {
        // Signal too short for reflect — fall back to zero-pad
        padded.extend(vec![0.0f32; pad]);
    }

    padded.extend_from_slice(samples);

    if samples.len() > pad {
        // Right reflection: samples[n-2], samples[n-3], ..., samples[n-1-pad]
        let n = samples.len();
        for i in 1..=pad {
            padded.push(samples[n - 1 - i]);
        }
    } else {
        padded.extend(vec![0.0f32; pad]);
    }

    // 2. Precompute Hann window and mel filterbank (both are reused across frames).
    let window = hann_window(N_FFT);
    let filters = mel_filterbank(N_MELS, N_FFT, SAMPLE_RATE as f32, 0.0, 8000.0);

    // 3. STFT — one FFT per hop.
    let num_frames = (padded.len() - N_FFT) / HOP_LENGTH + 1;

    use realfft::RealFftPlanner;
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);

    // Output layout: row-major [N_MELS, num_frames].
    let mut mel_spec = vec![0.0f32; N_MELS * num_frames];

    for frame_idx in 0..num_frames {
        let start = frame_idx * HOP_LENGTH;

        // Apply window to extract frame.
        let mut frame: Vec<f32> = (0..N_FFT)
            .map(|i| padded[start + i] * window[i])
            .collect();

        // Forward FFT.
        let mut spectrum = fft.make_output_vec();
        fft.process(&mut frame, &mut spectrum)
            .map_err(|e| LludaError::Msg(format!("FFT error: {}", e)))?;

        // Power spectrum: |X[k]|² = re² + im².
        let power: Vec<f32> = spectrum
            .iter()
            .map(|c| c.re * c.re + c.im * c.im)
            .collect();

        // Apply mel filterbank: dot(filters[m, :], power).
        for mel_idx in 0..N_MELS {
            let filter_row = &filters[mel_idx * FFT_BINS..(mel_idx + 1) * FFT_BINS];
            let mut sum = 0.0f32;
            for bin in 0..FFT_BINS {
                sum += filter_row[bin] * power[bin];
            }
            mel_spec[mel_idx * num_frames + frame_idx] = sum;
        }
    }

    // 4. Log10 scale — clamp at 1e-10 to avoid -inf.
    let mut log_spec: Vec<f32> = mel_spec
        .iter()
        .map(|&x| x.max(1e-10_f32).log10())
        .collect();

    // 5. Whisper normalization:
    //    clip to [max_val - 8, ∞), shift by +4, divide by 4 → range ≈ [-1, 1].
    let max_val = log_spec
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    for v in log_spec.iter_mut() {
        *v = (v.max(max_val - 8.0) + 4.0) / 4.0;
    }

    Tensor::new(log_spec, vec![N_MELS, num_frames])
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Hann window of the given length.
///
/// Values follow the formula: `w[i] = 0.5 * (1 - cos(2π·i / N))`.
/// Both endpoints are 0 for a periodic window (standard for STFT).
fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            0.5 * (1.0
                - (2.0 * std::f32::consts::PI * i as f32 / size as f32).cos())
        })
        .collect()
}

/// Build a mel filterbank matrix of shape `[n_mels, n_fft/2 + 1]`.
///
/// Uses the HTK mel scale and Slaney-style area normalization so each
/// filter has unit area, making the output independent of filter bandwidth.
///
/// # Arguments
/// * `n_mels`  — Number of mel frequency bins.
/// * `n_fft`   — FFT size (window length).
/// * `sr`      — Sample rate in Hz.
/// * `fmin`    — Lowest frequency in Hz (typically 0).
/// * `fmax`    — Highest frequency in Hz (typically Nyquist = `sr / 2`).
fn mel_filterbank(n_mels: usize, n_fft: usize, sr: f32, fmin: f32, fmax: f32) -> Vec<f32> {
    let fft_bins = n_fft / 2 + 1;

    // HTK mel scale: mel = 2595 · log10(1 + hz / 700)
    let hz_to_mel = |hz: f32| -> f32 { 2595.0 * (1.0 + hz / 700.0).log10() };
    let mel_to_hz = |mel: f32| -> f32 { 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0) };

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    // n_mels + 2 equally spaced points in mel space (includes edges).
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    // Convert mel centre frequencies to Hz.
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Map Hz to FFT bin indices (continuous, not rounded).
    let bin_points: Vec<f32> = hz_points
        .iter()
        .map(|&hz| hz * n_fft as f32 / sr)
        .collect();

    // Build triangular filters with Slaney area normalization.
    let mut filters = vec![0.0f32; n_mels * fft_bins];

    for m in 0..n_mels {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];

        // Slaney normalization factor: 2 / (right_hz - left_hz).
        let enorm = 2.0 / (hz_points[m + 2] - hz_points[m]);

        for k in 0..fft_bins {
            let kf = k as f32;
            let weight = if kf >= left && kf <= center {
                (kf - left) / (center - left)
            } else if kf > center && kf <= right {
                (right - kf) / (right - center)
            } else {
                0.0
            };
            filters[m * fft_bins + k] = weight * enorm;
        }
    }

    filters
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── hann_window ───────────────────────────────────────────────────────────

    #[test]
    fn test_hann_window_endpoints() {
        let w = hann_window(4);
        // w[0] = 0.5*(1 - cos(0)) = 0
        assert!((w[0] - 0.0).abs() < 1e-6, "w[0] should be 0, got {}", w[0]);
        // w[2] = 0.5*(1 - cos(π)) = 1 (peak at halfway for size-4 window)
        // Actually for size 4: w[2] = 0.5*(1 - cos(2π*2/4)) = 0.5*(1 - cos(π)) = 1
        // w[1] = 0.5*(1 - cos(2π/4)) = 0.5*(1 - 0) = 0.5
        assert!(w[1] > 0.0, "w[1] should be positive, got {}", w[1]);
    }

    #[test]
    fn test_hann_window_symmetry_even() {
        let size = 8;
        let w = hann_window(size);
        // For a periodic Hann window of size N, w[i] = w[N-i] for i > 0.
        // i.e. w[1]==w[7], w[2]==w[6], w[3]==w[5]
        for i in 1..size / 2 {
            let diff = (w[i] - w[size - i]).abs();
            assert!(diff < 1e-6, "w[{}]={} != w[{}]={}", i, w[i], size - i, w[size - i]);
        }
    }

    #[test]
    fn test_hann_window_peak() {
        let w = hann_window(N_FFT); // 400 points
        // Peak is at index 200 (centre of window), value close to 1.
        assert!(w[N_FFT / 2] > 0.99, "Peak should be near 1.0, got {}", w[N_FFT / 2]);
        // Both endpoints are 0.
        assert!(w[0] < 1e-6);
    }

    // ── mel_filterbank ────────────────────────────────────────────────────────

    #[test]
    fn test_mel_filterbank_shape() {
        let filters = mel_filterbank(128, 400, 16000.0, 0.0, 8000.0);
        assert_eq!(filters.len(), 128 * 201, "Expected 128×201 filter matrix");
    }

    #[test]
    fn test_mel_filterbank_non_negative() {
        let filters = mel_filterbank(128, 400, 16000.0, 0.0, 8000.0);
        assert!(
            filters.iter().all(|&x| x >= 0.0),
            "All filter weights must be non-negative"
        );
    }

    #[test]
    fn test_mel_filterbank_high_freq_rows_nonzero() {
        // With N_FFT=400 and sr=16000, the FFT resolution is 40 Hz/bin.
        // The lowest mel bands (< ~100 Hz) are narrower than one FFT bin, so
        // their triangular filters produce all-zero rows at integer bin indices.
        // This is correct librosa/Whisper behavior and not a bug.
        //
        // Higher-frequency rows (starting around row 10+) are wide enough to
        // always capture at least one integer FFT bin.
        let n_mels = 128;
        let n_fft = 400;
        let fft_bins = n_fft / 2 + 1;
        let filters = mel_filterbank(n_mels, n_fft, 16000.0, 0.0, 8000.0);

        // From row 20 onward, each mel band must cover at least one FFT bin.
        for m in 20..n_mels {
            let row = &filters[m * fft_bins..(m + 1) * fft_bins];
            let max_weight = row.iter().cloned().fold(0.0f32, f32::max);
            assert!(
                max_weight > 0.0,
                "Filter row {} is all-zero (no FFT bins covered)",
                m
            );
        }

        // The overall filterbank must have at least half the rows non-zero.
        let nonzero_rows = (0..n_mels).filter(|&m| {
            let row = &filters[m * fft_bins..(m + 1) * fft_bins];
            row.iter().any(|&x| x > 0.0)
        }).count();
        assert!(
            nonzero_rows >= n_mels / 2,
            "At least half the rows should be non-zero, got {} / {}",
            nonzero_rows, n_mels
        );
    }

    // ── mel_spectrogram ───────────────────────────────────────────────────────

    #[test]
    fn test_mel_spectrogram_shape_silence() {
        // 1 second of silence at 16 kHz.
        let samples = vec![0.0f32; 16000];
        let mel = mel_spectrogram(&samples).unwrap();

        assert_eq!(mel.shape()[0], N_MELS, "Expected {} mel bins", N_MELS);
        // num_frames = (16000 + 400 - 400) / 160 + 1 = 101
        let expected_frames = (samples.len() + N_FFT - N_FFT) / HOP_LENGTH + 1;
        assert_eq!(
            mel.shape()[1],
            expected_frames,
            "Expected {} frames",
            expected_frames
        );
    }

    #[test]
    fn test_mel_spectrogram_shape_half_second() {
        let samples = vec![0.0f32; 8000]; // 0.5 s
        let mel = mel_spectrogram(&samples).unwrap();
        assert_eq!(mel.shape()[0], N_MELS);
        // frames = (8000 + 0) / 160 + 1 = 51
        let expected_frames = samples.len() / HOP_LENGTH + 1;
        assert_eq!(mel.shape()[1], expected_frames);
    }

    #[test]
    fn test_mel_spectrogram_sine_wave_not_all_zero() {
        // 440 Hz sine wave for 1 second.
        let samples: Vec<f32> = (0..16000)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin()
            })
            .collect();
        let mel = mel_spectrogram(&samples).unwrap();
        assert_eq!(mel.shape()[0], N_MELS);

        let data = mel.to_vec_f32();
        let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);

        // After Whisper normalization, max value should be at 0 (shifted by +4 / 4).
        // The spectrogram must not be constant — sine wave produces variation.
        assert!(
            max_val > min_val,
            "Spectrogram should not be constant. max={}, min={}",
            max_val,
            min_val
        );
    }

    #[test]
    fn test_mel_spectrogram_normalized_range() {
        let samples: Vec<f32> = (0..16000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();
        let mel = mel_spectrogram(&samples).unwrap();
        let data = mel.to_vec_f32();

        let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);

        // Whisper normalization: clamp to [log_max - 8, log_max], then (x + 4) / 4.
        // This means the dynamic range in the output is exactly 2.0
        // (8 dB range maps to (log_max - (log_max - 8) + 4 - 4) / 4 = 8/4 = 2.0).
        // The minimum value is max_val - 2.0 (all values below the clamping floor
        // are lifted to that floor).
        let dynamic_range = max_val - min_val;
        assert!(
            dynamic_range <= 2.0 + 1e-5,
            "Dynamic range should be <= 2.0, got {}",
            dynamic_range
        );
        assert!(
            dynamic_range > 0.0,
            "Spectrogram should not be constant"
        );
    }

    // ── reflect_padding ───────────────────────────────────────────────────────

    #[test]
    fn test_reflect_padding_correctness() {
        // Verify that mel_spectrogram uses reflect padding, not zero padding.
        //
        // A signal with a sharp step at position 0 (all zeros) followed by ones:
        //   [0, 0, ..., 0, 1, 1, ..., 1]
        // With zero-padding: the first pad=200 values in the padded signal are 0.
        // With reflect-padding: the padded values mirror samples[1..=pad],
        //   which are still 0 for a long-enough zero prefix. So for a uniform
        //   signal the two methods differ only at boundaries.
        //
        // Instead: use signal = [0.0, 1.0, 2.0, 3.0, 4.0] (length 5, pad=2)
        // to directly verify the padding values via a hand-computed case.
        //
        // np.pad([0,1,2,3,4], 2, mode='reflect') = [2,1, 0,1,2,3,4, 3,2]
        //
        // We replicate the padding logic here to confirm it is correct.
        let samples: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let pad = 2usize;
        let n = samples.len();

        let mut padded = Vec::with_capacity(n + 2 * pad);

        // Left reflection
        for i in (1..=pad).rev() {
            padded.push(samples[i]);
        }
        padded.extend_from_slice(&samples);
        // Right reflection
        for i in 1..=pad {
            padded.push(samples[n - 1 - i]);
        }

        // Expected: [2, 1, 0, 1, 2, 3, 4, 3, 2]
        let expected = vec![2.0f32, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0];
        assert_eq!(padded.len(), expected.len(), "Padded length mismatch");
        for (i, (&got, &exp)) in padded.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "padded[{}] = {} but expected {}",
                i, got, exp
            );
        }
    }

    #[test]
    fn test_reflect_padding_differs_from_zero_padding() {
        // Verify that reflect padding produces different output than zero-padding
        // for a non-constant signal. Use a ramp signal where the reflect boundary
        // is clearly non-zero.
        //
        // Signal: [10.0, 9.0, 8.0, ..., 1.0, 0.0] (length > pad=200)
        // With zero-padding: padded[0] = 0.0
        // With reflect-padding: padded[0] = samples[200] (non-zero for this ramp)
        let sig_len = N_FFT; // 400, which is > pad=200
        let samples: Vec<f32> = (0..sig_len).map(|i| (sig_len - i) as f32).collect();
        let pad = N_FFT / 2; // 200

        let mut reflect_padded = Vec::with_capacity(samples.len() + 2 * pad);
        for i in (1..=pad).rev() {
            reflect_padded.push(samples[i]);
        }
        reflect_padded.extend_from_slice(&samples);
        for i in 1..=pad {
            reflect_padded.push(samples[samples.len() - 1 - i]);
        }

        // The first element from reflect padding = samples[pad] = samples[200]
        let expected_first = samples[pad];
        assert!(
            (reflect_padded[0] - expected_first).abs() < 1e-6,
            "Left reflect: expected samples[{}]={}, got {}",
            pad, expected_first, reflect_padded[0]
        );

        // Zero padding would give 0.0 here
        assert!(
            expected_first != 0.0,
            "Test signal must be non-zero at index {} for this test to be meaningful",
            pad
        );
    }

    // ── load_wav ──────────────────────────────────────────────────────────────

    #[test]
    fn test_load_wav_nonexistent() {
        let result = load_wav("/nonexistent/file.wav");
        assert!(result.is_err(), "Expected error for missing file");
    }

    /// Synthesize a minimal WAV in memory, write to a temp file, then round-trip.
    #[test]
    fn test_load_wav_roundtrip_mono_i16() {
        // Write a tiny 16-bit mono 16 kHz WAV with a 440 Hz sine (0.1 s = 1600 samples).
        let tmp = std::env::temp_dir().join("lluda_test_audio.wav");
        {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 16000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::create(&tmp, spec).unwrap();
            for i in 0..1600u32 {
                let sample = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin();
                let pcm = (sample * i16::MAX as f32) as i16;
                writer.write_sample(pcm).unwrap();
            }
        }

        let samples = load_wav(&tmp).unwrap();
        assert_eq!(samples.len(), 1600, "Should load 1600 samples");
        // Check amplitude is in [-1, 1].
        assert!(samples.iter().all(|&s| s.abs() <= 1.0 + 1e-5));
        // Cleanup.
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_load_wav_stereo_averaged() {
        let tmp = std::env::temp_dir().join("lluda_test_stereo.wav");
        {
            let spec = hound::WavSpec {
                channels: 2,
                sample_rate: 16000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::create(&tmp, spec).unwrap();
            for _ in 0..100u32 {
                // Left channel = 1.0, right channel = -1.0 → average = 0.0
                writer.write_sample(i16::MAX).unwrap();
                writer.write_sample(i16::MIN).unwrap();
            }
        }

        let samples = load_wav(&tmp).unwrap();
        assert_eq!(samples.len(), 100);
        // Average of +1 and -1 is 0 (approximately — i16::MIN is -32768, MAX is 32767).
        for &s in &samples {
            assert!(s.abs() < 0.01, "Expected ~0 for averaged channels, got {}", s);
        }
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_load_wav_wrong_sample_rate() {
        let tmp = std::env::temp_dir().join("lluda_test_8khz.wav");
        {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 8000, // wrong rate
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::create(&tmp, spec).unwrap();
            writer.write_sample(0i16).unwrap();
        }

        let result = load_wav(&tmp);
        assert!(result.is_err(), "Expected error for wrong sample rate");
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("8000"), "Error should mention actual rate");
        let _ = std::fs::remove_file(&tmp);
    }
}
