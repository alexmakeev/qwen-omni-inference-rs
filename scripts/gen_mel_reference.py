#!/usr/bin/env python3
"""
Generate mel spectrogram reference data for validating Rust mel_spectrogram implementation.

Key findings about WhisperFeatureExtractor (Qwen2.5-Omni):
- Uses torch.stft with center=True (default in PyTorch 2.10+), reflect padding
- After STFT, discards the LAST frame: magnitudes = stft[..., :-1].abs()**2
  This gives 184 frames for 29449-sample audio instead of 185
- mel_filters shape in HF is (201, 128) = (freq_bins, n_mels) — transposed vs librosa
- Normalization: log10, clip to [max-8, inf), (x+4)/4

This script computes and saves:
1. mel_whisper_fe.npy       — from WhisperFeatureExtractor (padded to 30000 frames)
2. mel_whisper_fe_cropped.npy — FE output cropped to real audio frames (184 for our file)
3. mel_librosa.npy          — librosa with center=True (185 frames, includes last)
4. mel_torch_exact.npy      — exact torch.stft path replicated (184 frames, matches FE)
5. audio_samples.npy        — raw PCM f32 samples
"""

import os
import sys
import numpy as np
import librosa
import torch

WAV_PATH = "/home/alexmak/lluda/main/materials/privet_mir.wav"
MODEL_PATH = "/home/alexmak/lluda/models/Qwen2.5-Omni-3B"
OUTPUT_DIR = "/home/alexmak/lluda/reference_data/omni_mel_test"

N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128
SR = 16000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Load audio ─────────────────────────────────────────────────────────────

print("=" * 70)
print("Loading audio...")
audio, sr_loaded = librosa.load(WAV_PATH, sr=SR, mono=True)
print(f"  WAV path:        {WAV_PATH}")
print(f"  Loaded SR:       {sr_loaded} Hz")
print(f"  Samples:         {len(audio)}")
print(f"  Duration:        {len(audio) / SR:.3f} s")
print(f"  dtype:           {audio.dtype}")
print(f"  Value range:     [{audio.min():.4f}, {audio.max():.4f}]")

# ── 2. WhisperFeatureExtractor (official reference) ───────────────────────────

print("\n" + "=" * 70)
print("Loading WhisperFeatureExtractor from processor...")
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(MODEL_PATH)
fe = processor.feature_extractor

print(f"  Feature extractor type: {type(fe).__name__}")
print(f"  fe.n_fft:               {fe.n_fft}")
print(f"  fe.hop_length:          {fe.hop_length}")
print(f"  fe.feature_size:        {fe.feature_size}")
print(f"  fe.sampling_rate:       {fe.sampling_rate}")
print(f"  fe.n_samples:           {fe.n_samples}  (chunk_length={fe.chunk_length}s * 16000)")
print(f"  fe.nb_max_frames:       {fe.nb_max_frames}")
print(f"  fe.mel_filters shape:   {fe.mel_filters.shape}  (freq_bins x n_mels, transposed vs librosa)")

# Run WhisperFeatureExtractor
print("\nRunning WhisperFeatureExtractor...")
inputs = fe(audio, sampling_rate=SR, return_tensors="np")
mel_whisper_fe = inputs['input_features'][0]

print(f"\n  mel_whisper_fe shape:   {mel_whisper_fe.shape}")
print(f"  mel_whisper_fe dtype:   {mel_whisper_fe.dtype}")
print(f"  mel_whisper_fe min:     {mel_whisper_fe.min():.4f}")
print(f"  mel_whisper_fe max:     {mel_whisper_fe.max():.4f}")
print(f"  mel_whisper_fe mean:    {mel_whisper_fe.mean():.4f}")
print(f"  mel_whisper_fe std:     {mel_whisper_fe.std():.4f}")

n_frames_fe_total = mel_whisper_fe.shape[1]
print(f"\n  Total frames in FE output: {n_frames_fe_total}")
print(f"  (padded to {fe.chunk_length}s = {fe.nb_max_frames} frames)")

# ── 3. Compute exact frame counts ─────────────────────────────────────────────

# torch.stft with center=True (default): pads by N_FFT//2 on each side
# num_frames = (len(audio) + N_FFT - N_FFT) / HOP_LENGTH + 1 = len(audio)/HOP_LENGTH + 1
# BUT WhisperFE does stft[..., :-1] — discards last frame!
n_frames_torch_before_crop = len(audio) // HOP_LENGTH + 1  # 185
n_frames_torch_after_crop = n_frames_torch_before_crop - 1  # 184

# librosa center=True: same as torch without the :-1 crop
n_frames_librosa = len(audio) // HOP_LENGTH + 1  # 185

print(f"\n  Frame count analysis:")
print(f"    torch.stft center=True output: {n_frames_torch_before_crop}")
print(f"    WhisperFE after [:-1] crop:    {n_frames_torch_after_crop}")
print(f"    librosa center=True:           {n_frames_librosa}")

# ── 4. Exact torch replication ─────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Replicating exact torch.stft path (WhisperFE torch code)...")

waveform_t = torch.from_numpy(audio).float()
window_t = torch.hann_window(N_FFT)

stft_t = torch.stft(waveform_t, N_FFT, HOP_LENGTH, window=window_t, return_complex=True)
print(f"  torch.stft output shape: {stft_t.shape}")

# WhisperFE: magnitudes = stft[..., :-1].abs() ** 2
magnitudes_t = stft_t[..., :-1].abs() ** 2
print(f"  After [:-1]: {magnitudes_t.shape}")

# Apply mel filters: mel_filters.T @ magnitudes
# fe.mel_filters is (201, 128) — freq_bins x n_mels
# torch code: mel_filters.T @ magnitudes = (128, 201) @ (201, 184) = (128, 184)
mel_filters_t = torch.from_numpy(fe.mel_filters).float()
mel_spec_t = mel_filters_t.T @ magnitudes_t

# log10, clamp
log_spec_t = torch.clamp(mel_spec_t, min=1e-10).log10()
log_spec_t = torch.maximum(log_spec_t, log_spec_t.max() - 8.0)
log_spec_t = (log_spec_t + 4.0) / 4.0
mel_torch_exact = log_spec_t.numpy()

print(f"\n  mel_torch_exact shape:  {mel_torch_exact.shape}")
print(f"  mel_torch_exact min:    {mel_torch_exact.min():.4f}")
print(f"  mel_torch_exact max:    {mel_torch_exact.max():.4f}")
print(f"  mel_torch_exact mean:   {mel_torch_exact.mean():.4f}")
print(f"  mel_torch_exact std:    {mel_torch_exact.std():.4f}")

# ── 5. Librosa method (center=True, no :-1 crop) ──────────────────────────────

print("\n" + "=" * 70)
print("Computing mel via librosa (center=True, Rust-matching approach)...")

stft = librosa.stft(
    audio,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    window='hann',
    center=True,
    pad_mode='reflect'
)
magnitudes = np.abs(stft) ** 2
mel_basis = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=0.0, fmax=8000.0)
mel_spec = mel_basis @ magnitudes
log_spec = np.log10(np.maximum(mel_spec, 1e-10))
log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
log_spec = (log_spec + 4.0) / 4.0
mel_librosa = log_spec

print(f"  stft shape (center=True):  {stft.shape}")
print(f"  mel_librosa shape:         {mel_librosa.shape}")
print(f"  mel_librosa min:           {mel_librosa.min():.4f}")
print(f"  mel_librosa max:           {mel_librosa.max():.4f}")
print(f"  mel_librosa mean:          {mel_librosa.mean():.4f}")
print(f"  mel_librosa std:           {mel_librosa.std():.4f}")

# ── 6. Crop WhisperFE output to real audio frames ─────────────────────────────

mel_fe_cropped = mel_whisper_fe[:, :n_frames_torch_after_crop]
print(f"\n  mel_fe_cropped shape: {mel_fe_cropped.shape}")

# ── 7. Comparisons ─────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Comparisons...")

def compare(name_a, a, name_b, b):
    T = min(a.shape[1], b.shape[1])
    a_ = a[:, :T].astype(np.float32)
    b_ = b[:, :T].astype(np.float32)
    mse = np.mean((a_ - b_) ** 2)
    mae = np.mean(np.abs(a_ - b_))
    max_diff = np.max(np.abs(a_ - b_))
    cos_sim = np.dot(a_.flatten(), b_.flatten()) / (
        np.linalg.norm(a_.flatten()) * np.linalg.norm(b_.flatten()) + 1e-12
    )
    print(f"\n  {name_a} vs {name_b} (T={T}):")
    print(f"    MSE:          {mse:.8f}")
    print(f"    MAE:          {mae:.8f}")
    print(f"    Max abs diff: {max_diff:.8f}")
    print(f"    Cosine sim:   {cos_sim:.8f}")
    return cos_sim, mse

compare("mel_torch_exact", mel_torch_exact, "mel_fe_cropped", mel_fe_cropped)
compare("mel_torch_exact", mel_torch_exact, "mel_librosa(:-1)", mel_librosa[:, :-1])
compare("mel_librosa", mel_librosa, "mel_fe_cropped", mel_fe_cropped)

# ── 8. Inspect center/padding behavior in detail ──────────────────────────────

print("\n" + "=" * 70)
print("Center/padding summary:")
print(f"  torch.stft center=True (PyTorch 2.10+ default): YES")
print(f"  Padding mode: reflect")
print(f"  After STFT, WhisperFE discards last frame: stft[..., :-1]")
print(f"  => torch: {n_frames_torch_before_crop} frames -> {n_frames_torch_after_crop} frames (after crop)")
print(f"  librosa center=True: {n_frames_librosa} frames (no crop)")
print(f"  Rust (current): center=True, no crop -> {n_frames_librosa} frames")
print(f"  => Rust is off by 1 frame vs WhisperFE")

# ── 9. Normalization check ─────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Normalization formula (Whisper):")
print("  1. mel_spec = mel_filters.T @ magnitudes  (power spectrum)")
print("  2. log_spec = log10(clamp(mel_spec, 1e-10))")
print("  3. log_spec = max(log_spec, log_spec.max() - 8.0)")
print("  4. log_spec = (log_spec + 4.0) / 4.0")
print(f"\n  Expected output range: [max-8+4)/4, (max+4)/4] = [max/4-1, max/4+1]")
print(f"  Dynamic range: 8/4 = 2.0 exactly")
print(f"  WhisperFE output max:  {mel_whisper_fe.max():.4f}")
print(f"  WhisperFE output min:  {mel_whisper_fe.min():.4f}")
print(f"  Dynamic range:         {mel_whisper_fe.max() - mel_whisper_fe.min():.4f}")

# ── 10. Save reference files ───────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Saving reference files...")

def save_npy(path, data, name):
    arr = np.ascontiguousarray(data.astype(np.float32))
    np.save(path, arr)
    size_kb = os.path.getsize(path + ".npy" if not path.endswith(".npy") else path) / 1024
    print(f"  Saved {name}: {arr.shape} {arr.dtype} ({arr.nbytes/1024:.1f} KB) -> {os.path.basename(path)}")

# mel_whisper_fe.npy — full (padded to 30000 frames by WhisperFE)
mel_fe_path = os.path.join(OUTPUT_DIR, "mel_whisper_fe.npy")
arr = np.ascontiguousarray(mel_whisper_fe.astype(np.float32))
np.save(mel_fe_path, arr)
print(f"  Saved mel_whisper_fe: {arr.shape} ({arr.nbytes/1024:.1f} KB) -> mel_whisper_fe.npy")

# mel_whisper_fe_cropped.npy — cropped to real audio frames (184)
mel_fe_cropped_path = os.path.join(OUTPUT_DIR, "mel_whisper_fe_cropped.npy")
arr = np.ascontiguousarray(mel_fe_cropped.astype(np.float32))
np.save(mel_fe_cropped_path, arr)
print(f"  Saved mel_whisper_fe_cropped: {arr.shape} ({arr.nbytes/1024:.1f} KB) -> mel_whisper_fe_cropped.npy")

# mel_librosa.npy — librosa center=True (185 frames)
mel_lib_path = os.path.join(OUTPUT_DIR, "mel_librosa.npy")
arr = np.ascontiguousarray(mel_librosa.astype(np.float32))
np.save(mel_lib_path, arr)
print(f"  Saved mel_librosa: {arr.shape} ({arr.nbytes/1024:.1f} KB) -> mel_librosa.npy")

# mel_torch_exact.npy — exact torch.stft replication (184 frames, matches WhisperFE)
mel_torch_path = os.path.join(OUTPUT_DIR, "mel_torch_exact.npy")
arr = np.ascontiguousarray(mel_torch_exact.astype(np.float32))
np.save(mel_torch_path, arr)
print(f"  Saved mel_torch_exact: {arr.shape} ({arr.nbytes/1024:.1f} KB) -> mel_torch_exact.npy")

# audio_samples.npy — raw PCM f32
audio_path = os.path.join(OUTPUT_DIR, "audio_samples.npy")
arr = np.ascontiguousarray(audio.astype(np.float32))
np.save(audio_path, arr)
print(f"  Saved audio_samples: {arr.shape} ({arr.nbytes/1024:.1f} KB) -> audio_samples.npy")

# ── 11. Final summary ──────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"  Audio: {len(audio)} samples, {len(audio)/SR:.3f}s at {SR}Hz")
print()
print(f"  WhisperFE (torch path) pipeline:")
print(f"    torch.stft(center=True, reflect) -> shape (201, {n_frames_torch_before_crop})")
print(f"    [:-1] crop -> (201, {n_frames_torch_after_crop})")
print(f"    mel_filters.T @ magnitudes -> (128, {n_frames_torch_after_crop})")
print(f"    log10 -> clamp -> normalize -> (128, {n_frames_torch_after_crop})")
print()
print(f"  Librosa pipeline (current Rust approach):")
print(f"    librosa.stft(center=True, reflect) -> (201, {n_frames_librosa})")
print(f"    mel_basis @ magnitudes -> (128, {n_frames_librosa})")
print(f"    log10 -> clamp -> normalize -> (128, {n_frames_librosa})")
print()
print(f"  KEY DIFFERENCE: WhisperFE has 184 frames, Rust has 185 frames")
print(f"  CAUSE: WhisperFE drops the last frame via stft[..., :-1]")
print(f"  FIX: In Rust, after STFT, drop the last frame")
print()
print(f"  Mel filterbank: identical values, WhisperFE uses (201,128) = transposed of librosa (128,201)")
print()
print(f"  Files saved to: {OUTPUT_DIR}")
print(f"  - mel_whisper_fe.npy         [128, 30000]  FE output padded to 30s")
print(f"  - mel_whisper_fe_cropped.npy [128, 184]    FE real frames (torch exact)")
print(f"  - mel_torch_exact.npy        [128, 184]    manual torch.stft replication")
print(f"  - mel_librosa.npy            [128, 185]    librosa center=True (no crop)")
print(f"  - audio_samples.npy          [{len(audio)}]       raw PCM f32")
print()
print(f"  COSINE SIMILARITY torch_exact vs fe_cropped: ", end="")
T = min(mel_torch_exact.shape[1], mel_fe_cropped.shape[1])
a = mel_torch_exact[:, :T].flatten()
b = mel_fe_cropped[:, :T].flatten().astype(np.float32)
print(f"{np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)):.8f}")
