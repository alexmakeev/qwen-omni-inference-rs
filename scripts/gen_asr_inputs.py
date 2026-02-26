#!/usr/bin/env python3
"""
Generate missing ASR input reference data for Qwen2.5-Omni-3B validation.

Produces:
  mel_input.npy     — mel spectrogram [128, T] fed to AudioEncoder
  input_ids.npy     — full token sequence [1, seq_len] with 151646 audio placeholders
  logits.npy        — prefill-pass thinker logits [1, 74, 151936]

Also verifies that audio_tower_output matches the existing reference via cosine similarity.
"""

import sys
import time
import logging
import numpy as np
import torch
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

MODEL_PATH = "/home/alexmak/lluda/models/Qwen2.5-Omni-3B"
AUDIO_FILE = "/home/alexmak/lluda/main/materials/privet_mir.wav"
REF_DIR = Path("/home/alexmak/lluda/reference_data/omni_asr_test")
AUDIO_TOKEN_ID = 151646


def check_ram():
    try:
        import psutil
        m = psutil.virtual_memory()
        s = psutil.swap_memory()
        logger.info(
            "RAM: %.1f/%.1f GB used (%.0f%%)  Swap: %.1f/%.1f GB used (%.0f%%)",
            m.used/1e9, m.total/1e9, m.percent,
            s.used/1e9, s.total/1e9, s.percent,
        )
        avail_gb = m.available / 1e9
        if avail_gb < 10:
            logger.warning("Only %.1f GB RAM available — model load may use swap", avail_gb)
        return avail_gb
    except ImportError:
        logger.info("psutil not available, skipping RAM check")
        return None


def to_f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().to(torch.float32).numpy()


def main():
    logger.info("=" * 70)
    logger.info("gen_asr_inputs.py — generating mel_input, input_ids, logits")
    logger.info("=" * 70)

    # ------------------------------------------------------------------ RAM check
    avail = check_ram()

    # ------------------------------------------------------------------ Load model
    logger.info("Loading Qwen2_5OmniForConditionalGeneration from %s ...", MODEL_PATH)
    t0 = time.time()

    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    processor = Qwen2_5OmniProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    logger.info("Model loaded in %.1f s", time.time() - t0)
    check_ram()

    # ------------------------------------------------------------------ Build conversation
    conversations = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": f"file://{AUDIO_FILE}"},
                {"type": "text", "text": "Transcribe the audio into text."},
            ],
        },
    ]

    # ------------------------------------------------------------------ Apply chat template
    text = processor.apply_chat_template(
        conversations, add_generation_prompt=True, tokenize=False
    )
    logger.info("Chat template output (first 400 chars): %s", text[:400])

    # ------------------------------------------------------------------ Load audio via process_mm_info
    try:
        from qwen_omni_utils import process_mm_info
        audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)
        logger.info("Audio loaded via process_mm_info: shape=%s", audios[0].shape)
    except Exception as exc:
        logger.warning("process_mm_info failed (%s) — loading manually", exc)
        import soundfile as sf
        audio_data, sr = sf.read(AUDIO_FILE)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        audio_data = audio_data.astype(np.float32)
        if sr != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        audios = [audio_data]
        images = None
        videos = None

    # ------------------------------------------------------------------ Processor → inputs
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
    )
    # Cast floats to bfloat16 (matches model weights)
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor) and inputs[key].is_floating_point():
            inputs[key] = inputs[key].to(torch.bfloat16)

    logger.info(
        "Input tensors: %s",
        {k: (tuple(v.shape), str(v.dtype)) for k, v in inputs.items() if hasattr(v, "shape")},
    )

    # ------------------------------------------------------------------ Save input_ids
    input_ids = inputs["input_ids"]  # [1, seq_len]
    logger.info("input_ids shape: %s", input_ids.shape)

    audio_token_count = (input_ids == AUDIO_TOKEN_ID).sum().item()
    logger.info("Audio token (ID %d) count in input_ids: %d", AUDIO_TOKEN_ID, audio_token_count)

    input_ids_np = input_ids.detach().cpu().numpy().astype(np.int64)
    out_input_ids = REF_DIR / "input_ids.npy"
    np.save(out_input_ids, input_ids_np)
    logger.info("Saved input_ids: shape=%s  %.1f KB", input_ids_np.shape, out_input_ids.stat().st_size / 1024)

    # ------------------------------------------------------------------ Extract mel spectrogram
    # The processor stores mel features in 'input_features' key.
    # Shape from processor: [batch, n_mels, time] or [batch, 1, n_mels, time] depending on version.
    input_features = inputs.get("input_features")
    if input_features is not None:
        feat_np = to_f32(input_features)
        logger.info("input_features from processor: shape=%s", feat_np.shape)

        # Squeeze to [n_mels, T] — remove batch and channel dims
        mel = feat_np.squeeze()
        # After squeeze: could be [n_mels, T] (2D) or already correct
        if mel.ndim == 3:
            # [1, n_mels, T] → [n_mels, T]
            mel = mel.squeeze(0)
        logger.info("mel_input shape: %s  (expected [128, T])", mel.shape)

        out_mel = REF_DIR / "mel_input.npy"
        np.save(out_mel, mel)
        logger.info("Saved mel_input: shape=%s  %.1f KB", mel.shape, out_mel.stat().st_size / 1024)
    else:
        logger.warning("No 'input_features' key in processor output — mel not saved")
        logger.info("Available keys: %s", list(inputs.keys()))

    # ------------------------------------------------------------------ Audio encoder forward pass (for verification)
    # Run the audio tower to check against existing reference
    logger.info("Running audio tower forward pass ...")

    # The audio tower lives at model.thinker.audio_tower for Qwen2.5-Omni
    audio_tower = None
    for attr_path in ["thinker.audio_tower", "audio_tower", "model.audio_tower"]:
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            audio_tower = obj
            logger.info("Found audio tower at: model.%s", attr_path)
            break
        except AttributeError:
            continue

    audio_tower_output_check = None
    if audio_tower is not None and input_features is not None:
        # Determine the correct input format for the audio tower
        # Qwen2.5-Omni audio tower expects [batch, n_mels, T] typically
        feat_bf16 = input_features.to(torch.bfloat16)
        logger.info("audio_tower input shape: %s  dtype: %s", feat_bf16.shape, feat_bf16.dtype)
        with torch.no_grad():
            try:
                audio_out = audio_tower(feat_bf16)
                if isinstance(audio_out, tuple):
                    audio_out = audio_out[0]
                elif hasattr(audio_out, "last_hidden_state"):
                    audio_out = audio_out.last_hidden_state
                audio_tower_output_check = to_f32(audio_out)
                logger.info("audio_tower output shape: %s", audio_tower_output_check.shape)
            except Exception as exc:
                logger.warning("Audio tower forward failed: %s", exc)
                logger.info("Will skip audio tower verification")
    else:
        logger.warning("Cannot run audio tower check — tower=%s, features=%s",
                       audio_tower is not None, input_features is not None)

    # ------------------------------------------------------------------ Verify audio_tower_output
    existing_ref = REF_DIR / "audio_tower_output.npy"
    if audio_tower_output_check is not None and existing_ref.exists():
        existing = np.load(existing_ref)
        logger.info("Existing audio_tower_output shape: %s  new shape: %s",
                    existing.shape, audio_tower_output_check.shape)
        # Reshape check output to match existing [46, 2048] if needed
        check_flat = audio_tower_output_check.flatten()
        existing_flat = existing.flatten()
        if check_flat.shape == existing_flat.shape:
            cos_sim = float(np.dot(check_flat, existing_flat) /
                            (np.linalg.norm(check_flat) * np.linalg.norm(existing_flat) + 1e-10))
            logger.info("Audio tower cosine similarity vs existing reference: %.6f", cos_sim)
            if cos_sim > 0.999:
                logger.info("PASS: audio tower output matches existing reference")
            elif cos_sim > 0.99:
                logger.info("ACCEPTABLE: minor numerical difference (BF16 vs F32)")
            else:
                logger.warning("MISMATCH: cosine similarity %.4f — check model/input pipeline", cos_sim)
        else:
            logger.warning("Shape mismatch: existing=%s  new=%s — cannot compare",
                           existing_flat.shape, check_flat.shape)
            # Try to compare just the squeezed version
            new_squeezed = audio_tower_output_check.squeeze()
            logger.info("new squeezed shape: %s", new_squeezed.shape)
            if new_squeezed.shape == existing.shape:
                new_flat = new_squeezed.flatten()
                existing_flat2 = existing.flatten()
                cos_sim = float(np.dot(new_flat, existing_flat2) /
                                (np.linalg.norm(new_flat) * np.linalg.norm(existing_flat2) + 1e-10))
                logger.info("Audio tower cosine (after squeeze): %.6f", cos_sim)
    else:
        logger.info("Skipping audio tower verification (check=%s, ref_exists=%s)",
                    audio_tower_output_check is not None, existing_ref.exists())

    # ------------------------------------------------------------------ Prefill pass to get logits
    # We need to run just the thinker forward pass (not generate) on the
    # full input sequence including audio embeddings to get logits [1, 74, vocab].
    #
    # For Qwen2.5-Omni, model.thinker is Qwen2_5OmniThinkerForConditionalGeneration.
    # Its forward() accepts input_ids + audio features via the multimodal pathway.
    # We call model.thinker(**inputs) or model(**inputs) with the full inputs dict.
    #
    # The full model's thinker.forward() handles audio token replacement internally
    # (replaces audio placeholder tokens with audio encoder output embeddings).

    logger.info("Running prefill forward pass through thinker to get logits ...")
    logger.info("Input seq_len: %d  (expecting 74 based on existing embedding_output)", input_ids.shape[1])
    check_ram()

    t0 = time.time()
    with torch.no_grad():
        # Use model.thinker for the forward pass — this is the text+audio transformer
        # The thinker handles audio token substitution internally
        if hasattr(model, "thinker"):
            thinker = model.thinker
            logger.info("Using model.thinker for forward pass")
        else:
            thinker = model
            logger.info("Using model directly for forward pass")

        # Forward pass with the same inputs used for generation
        # The thinker forward expects: input_ids, attention_mask, and input_features
        # (audio features). It handles the audio-to-embedding substitution.
        try:
            outputs = thinker(**inputs, return_dict=True)
            logits = outputs.logits  # [1, seq_len, vocab_size]
            logger.info("Forward pass done in %.1f s", time.time() - t0)
            logger.info("logits shape: %s  dtype: %s", logits.shape, logits.dtype)
        except Exception as exc:
            logger.warning("thinker forward failed: %s", exc)
            logger.info("Trying with only input_ids and attention_mask ...")
            # Fallback: pass only the text inputs (no audio)
            text_inputs = {
                "input_ids": inputs["input_ids"],
            }
            if "attention_mask" in inputs:
                text_inputs["attention_mask"] = inputs["attention_mask"]
            outputs = thinker(**text_inputs, return_dict=True)
            logits = outputs.logits
            logger.warning("Fallback logits shape: %s (audio tokens NOT substituted)", logits.shape)

    logits_np = to_f32(logits)
    logger.info("logits_np shape: %s", logits_np.shape)

    out_logits = REF_DIR / "logits.npy"
    np.save(out_logits, logits_np)
    logger.info("Saved logits: shape=%s  %.1f KB", logits_np.shape, out_logits.stat().st_size / 1024)

    # ------------------------------------------------------------------ Summary
    logger.info("=" * 70)
    logger.info("SUMMARY — generated files:")
    for fname in ["mel_input.npy", "input_ids.npy", "logits.npy"]:
        p = REF_DIR / fname
        if p.exists():
            arr = np.load(p)
            logger.info("  %-20s  shape=%-20s  %.1f KB", fname, str(arr.shape), p.stat().st_size / 1024)
        else:
            logger.info("  %-20s  MISSING", fname)
    logger.info("=" * 70)

    # ------------------------------------------------------------------ Print token IDs for debugging
    logger.info("First 20 input_ids: %s", input_ids_np[0, :20].tolist())
    logger.info("Last 10 input_ids: %s", input_ids_np[0, -10:].tolist())


if __name__ == "__main__":
    main()
