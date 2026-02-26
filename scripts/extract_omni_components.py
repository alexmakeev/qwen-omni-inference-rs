#!/usr/bin/env python3
"""Extract per-component reference data from Qwen2.5-Omni-3B for Rust validation.

This script loads individual components of Qwen2.5-Omni-3B and runs them with
controlled, deterministic inputs to produce reference .npy files. The Rust
implementation can then be validated component by component against these files.

Model structure (Qwen2.5-Omni-3B):
  Qwen2_5OmniForConditionalGeneration
    .thinker  (Qwen2_5OmniThinkerForConditionalGeneration)
      .audio_tower  (Qwen2_5OmniAudioEncoder)
        .conv1, .conv2  - CNN frontend
        .layers[0..31]  - 32 transformer blocks
        .ln_post        - final LayerNorm
        .avg_pooler     - average pooling
        .proj           - projection to output_dim
      .visual  (Qwen2_5OmniVisionEncoder)
      .model   (Qwen2_5OmniThinkerTextModel)
        .embed_tokens
        .layers[0..35]  - 36 decoder layers
        .norm           - final RMSNorm
      .lm_head
    .talker  (Qwen2_5OmniTalkerForConditionalGeneration)
      .thinker_to_talker_proj
      .model  (Qwen2_5OmniTalkerModel)
        .embed_tokens
        .layers[0..23]  - 24 decoder layers
        .norm

Audio encoder config:
  num_mel_bins=128, d_model=1280, encoder_layers=32, n_window=100
  Input: [mel_bins, time_frames] (no batch dim for direct call)
  forward(input_features, feature_lens, aftercnn_lens)

Thinker text model config:
  hidden_size=2048, num_hidden_layers=36, vocab_size=151936

Talker config:
  hidden_size=896, num_hidden_layers=24, vocab_size=8448, embedding_size=2048

Usage:
    source /home/alexmak/lluda/venv_omni/bin/activate
    python scripts/extract_omni_components.py --stage inspect
    python scripts/extract_omni_components.py --stage audio
    python scripts/extract_omni_components.py --stage thinker
    python scripts/extract_omni_components.py --stage talker
    python scripts/extract_omni_components.py --stage all
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def to_f32_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert any torch tensor (including BF16) to float32 numpy array."""
    return tensor.detach().cpu().to(torch.float32).numpy()


def save_npy(output_dir: Path, name: str, arr: np.ndarray) -> None:
    """Save numpy array and log shape + size."""
    path = output_dir / f"{name}.npy"
    np.save(path, arr)
    size_kb = path.stat().st_size / 1024
    logger.info("  Saved %s: shape=%s  %.1f KB", name, arr.shape, size_kb)


def log_ram() -> None:
    """Log current RAM usage if psutil is available."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        logger.info(
            "RAM: %.1f/%.1f GB used (%.0f%%)  |  Swap: %.1f/%.1f GB used (%.0f%%)",
            mem.used / 1e9, mem.total / 1e9, mem.percent,
            swap.used / 1e9, swap.total / 1e9, swap.percent,
        )
    except ImportError:
        pass


def find_attr(obj, attr_paths: List[str]):
    """Try multiple dotted attribute paths; return first that resolves."""
    for path in attr_paths:
        parts = path.split(".")
        cur = obj
        found = True
        for part in parts:
            if hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                found = False
                break
        if found:
            return cur
    return None


# ---------------------------------------------------------------------------
# Activation capture
# ---------------------------------------------------------------------------

class ActivationCapture:
    """Register forward hooks and collect first-call outputs by name."""

    def __init__(self, capture_once: bool = False):
        """
        capture_once: if True, each name is captured only on its first forward
                      call (useful when hooks remain across multiple generate steps).
        """
        self.capture_once = capture_once
        self.activations: Dict[str, np.ndarray] = {}
        self._hooks: list = []

    def register(self, name: str, module: torch.nn.Module) -> None:
        def hook_fn(mod, inp, output):
            if self.capture_once and name in self.activations:
                return
            if isinstance(output, tuple):
                out = output[0]
            elif hasattr(output, "last_hidden_state"):
                out = output.last_hidden_state
            elif isinstance(output, torch.Tensor):
                out = output
            else:
                # Unknown type — skip silently
                return
            self.activations[name] = to_f32_numpy(out)

        self._hooks.append(module.register_forward_hook(hook_fn))

    def clear(self) -> None:
        self.activations.clear()

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Stage: inspect (no weights loaded)
# ---------------------------------------------------------------------------

def inspect_model(model_path: str) -> None:
    """Print model config summary without loading weights."""
    logger.info("=" * 70)
    logger.info("STAGE: inspect")
    logger.info("=" * 70)

    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    model_type = config.get("model_type", "unknown")
    logger.info("model_type: %s", model_type)

    # Audio encoder
    ac = config.get("thinker_config", {}).get("audio_config", {})
    logger.info("\n--- Audio Encoder ---")
    logger.info("  d_model=%s  encoder_layers=%s  encoder_attention_heads=%s",
                ac.get("d_model"), ac.get("encoder_layers"), ac.get("encoder_attention_heads"))
    logger.info("  num_mel_bins=%s  max_source_positions=%s  n_window=%s",
                ac.get("num_mel_bins"), ac.get("max_source_positions"), ac.get("n_window"))
    logger.info("  output_dim=%s", ac.get("output_dim"))

    # Thinker text model
    txt = config.get("thinker_config", {}).get("text_config", {})
    logger.info("\n--- Thinker Text Model ---")
    logger.info("  hidden_size=%s  num_hidden_layers=%s  num_attention_heads=%s",
                txt.get("hidden_size"), txt.get("num_hidden_layers"), txt.get("num_attention_heads"))
    logger.info("  vocab_size=%s", txt.get("vocab_size"))

    # Talker
    tal = config.get("talker_config", {})
    logger.info("\n--- Talker ---")
    logger.info("  hidden_size=%s  num_hidden_layers=%s  vocab_size=%s  embedding_size=%s",
                tal.get("hidden_size"), tal.get("num_hidden_layers"),
                tal.get("vocab_size"), tal.get("embedding_size"))

    logger.info("\nFull model hierarchy:")
    logger.info("  model.thinker.audio_tower     (Qwen2_5OmniAudioEncoder)")
    logger.info("    .conv1, .conv2              - CNN frontend")
    logger.info("    .layers[0..%d]              - transformer blocks",
                (ac.get("encoder_layers", 32) - 1))
    logger.info("    .ln_post, .avg_pooler, .proj")
    logger.info("  model.thinker.model           (Qwen2_5OmniThinkerTextModel)")
    logger.info("    .embed_tokens")
    logger.info("    .layers[0..%d]              - decoder layers",
                (txt.get("num_hidden_layers", 36) - 1))
    logger.info("    .norm")
    logger.info("  model.talker.model            (Qwen2_5OmniTalkerModel)")
    logger.info("    .embed_tokens")
    logger.info("    .layers[0..%d]              - decoder layers",
                (tal.get("num_hidden_layers", 24) - 1))
    logger.info("    .norm")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str):
    """Load the full Qwen2.5-Omni-3B model on CPU."""
    logger.info("Loading Qwen2_5OmniForConditionalGeneration from %s ...", model_path)
    logger.info("(device_map=cpu, torch_dtype=bfloat16, low_cpu_mem_usage=True)")
    log_ram()
    t0 = time.time()

    from transformers import Qwen2_5OmniForConditionalGeneration
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()

    elapsed = time.time() - t0
    logger.info("Model loaded in %.1f s", elapsed)
    log_ram()
    return model


def load_thinker_only(model_path: str):
    """Load only the Thinker component (lighter, saves memory)."""
    logger.info("Loading Qwen2_5OmniThinkerForConditionalGeneration from %s ...", model_path)
    log_ram()
    t0 = time.time()

    from transformers import Qwen2_5OmniThinkerForConditionalGeneration
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()

    elapsed = time.time() - t0
    logger.info("Thinker loaded in %.1f s", elapsed)
    log_ram()
    return model


# ---------------------------------------------------------------------------
# Stage: audio encoder
# ---------------------------------------------------------------------------

def extract_audio_encoder(model, output_dir: Path) -> None:
    """Extract per-layer activations from the audio encoder.

    The audio encoder (Qwen2_5OmniAudioEncoder) lives at:
      model.thinker.audio_tower   (when model is the full Qwen2_5OmniForConditionalGeneration)
      model.audio_tower           (when model is the Thinker)

    Its forward signature:
      forward(input_features, feature_lens=None, aftercnn_lens=None)

    input_features shape: [mel_bins, time_frames]  (no explicit batch dim;
      internally the encoder calls split/pad which treats dim-1 as time).

    We use n_window=100, so the minimum viable input is 2*n_window=200 frames.
    We use 200 frames (the smallest single-chunk input) for speed.

    Saved files:
      audio_encoder_input.npy           [128, 200]  mel input
      audio_encoder_feature_lens.npy    [1]         = 200
      audio_encoder_aftercnn_lens.npy   [1]         = 100 (after CNN stride-2)
      audio_layer_{00..31}_output.npy   [seq, 1280] per-layer hidden states
      audio_ln_post_output.npy          captured before avg_pool+proj
      audio_encoder_output.npy          [tokens, 2048] final projected output
    """
    logger.info("=" * 70)
    logger.info("STAGE: audio encoder")
    logger.info("=" * 70)

    audio_tower = find_attr(model, ["thinker.audio_tower", "audio_tower"])
    if audio_tower is None:
        logger.error("Could not locate audio_tower in model hierarchy")
        return

    n_window = audio_tower.n_window  # 100 for Qwen2.5-Omni-3B
    num_mel_bins = audio_tower.num_mel_bins  # 128
    n_layers = len(audio_tower.layers)  # 32
    logger.info("audio_tower: n_window=%d, num_mel_bins=%d, layers=%d", n_window, num_mel_bins, n_layers)

    # Build deterministic input: use exactly 2*n_window frames (one full chunk)
    time_frames = 2 * n_window  # 200
    torch.manual_seed(42)
    # input_features shape for the encoder: [mel_bins, time_frames]
    # The encoder internally treats this as a single audio with feature_lens=[time_frames]
    mel_input = torch.randn(num_mel_bins, time_frames, dtype=torch.float32)

    # Compute lengths
    feature_lens = torch.tensor([time_frames], dtype=torch.long)
    # aftercnn_lens = (feature_lens - 1) // 2 + 1 (from _get_feat_extract_output_lengths)
    aftercnn_lens = (feature_lens - 1) // 2 + 1  # = 100

    logger.info("Input: mel_input shape=%s, feature_lens=%s, aftercnn_lens=%s",
                mel_input.shape, feature_lens.tolist(), aftercnn_lens.tolist())

    # Register hooks on each layer and ln_post
    capture = ActivationCapture(capture_once=False)
    for i, layer in enumerate(audio_tower.layers):
        capture.register(f"audio_layer_{i:02d}", layer)
    capture.register("audio_ln_post", audio_tower.ln_post)

    # Forward pass
    logger.info("Running audio encoder forward pass ...")
    t0 = time.time()
    with torch.no_grad():
        # Cast mel to match encoder dtype (bfloat16)
        mel_bf16 = mel_input.to(audio_tower.dtype)
        audio_out = audio_tower(
            input_features=mel_bf16,
            feature_lens=feature_lens,
            aftercnn_lens=aftercnn_lens,
        )
    elapsed = time.time() - t0
    logger.info("Audio encoder forward done in %.1f s", elapsed)

    # Extract output tensor
    if hasattr(audio_out, "last_hidden_state"):
        out_tensor = audio_out.last_hidden_state
    elif isinstance(audio_out, tuple):
        out_tensor = audio_out[0]
    else:
        out_tensor = audio_out
    logger.info("Audio encoder output shape: %s", out_tensor.shape)

    capture.remove_hooks()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    save_npy(output_dir, "audio_encoder_input", mel_input.numpy())
    save_npy(output_dir, "audio_encoder_feature_lens", feature_lens.numpy())
    save_npy(output_dir, "audio_encoder_aftercnn_lens", aftercnn_lens.numpy())
    save_npy(output_dir, "audio_encoder_output", to_f32_numpy(out_tensor))

    for name, arr in sorted(capture.activations.items()):
        save_npy(output_dir, f"{name}_output", arr)

    logger.info("Audio encoder: saved %d activation tensors", len(capture.activations) + 1)
    log_ram()


# ---------------------------------------------------------------------------
# Stage: thinker (text decoder)
# ---------------------------------------------------------------------------

def extract_thinker(model, output_dir: Path) -> None:
    """Extract per-layer activations from the Thinker text decoder.

    The Thinker text model (Qwen2_5OmniThinkerTextModel) lives at:
      model.thinker.model   (when model is the full Qwen2_5OmniForConditionalGeneration)
      model.model           (when model is the Thinker)

    We use output_hidden_states=True to get all per-layer hidden states
    without hooking (simpler and avoids hook interaction with cache).

    Input: "You are a helpful assistant." tokenized with Qwen2.5 tokenizer.
    Saved files:
      thinker_input_ids.npy                [1, seq_len]
      thinker_embedding_output.npy         [1, seq_len, 2048]
      thinker_layer_{00..35}_output.npy    [1, seq_len, 2048]
      thinker_final_norm_output.npy        [1, seq_len, 2048]
      thinker_logits.npy                   [1, seq_len, vocab_size]
    """
    logger.info("=" * 70)
    logger.info("STAGE: thinker text decoder")
    logger.info("=" * 70)

    # Navigate to the Thinker (ForConditionalGeneration level)
    thinker_cg = find_attr(model, ["thinker", ""])
    if thinker_cg is None or not hasattr(model, "thinker"):
        # model IS the thinker
        thinker_cg = model
    else:
        thinker_cg = model.thinker

    # Inner text model
    thinker_inner = find_attr(model, ["thinker.model", "model"])
    if thinker_inner is None:
        logger.error("Could not locate thinker text model")
        return

    n_layers = len(thinker_inner.layers)
    hidden_size = thinker_inner.embed_tokens.embedding_dim
    logger.info("Thinker text model: layers=%d, hidden_size=%d", n_layers, hidden_size)

    # Use a fixed, deterministic token sequence.
    # "You are a helpful assistant." — common system prompt tokens for Qwen2.5
    # We use raw token IDs instead of the processor to avoid loading it.
    # These are approximate Qwen2.5 tokenizations of the system prompt.
    # For precise tokens, the test is still valid as long as they're consistent.
    input_ids = torch.tensor(
        [[151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13]],
        dtype=torch.long,
    )  # <|im_start|>system\nYou are a helpful assistant.
    logger.info("Input IDs: shape=%s  tokens=%s", input_ids.shape, input_ids[0].tolist())

    # Forward through thinker (ForConditionalGeneration level)
    # Use output_hidden_states=True to get all layer outputs without hooks
    logger.info("Running thinker forward pass (output_hidden_states=True) ...")
    t0 = time.time()
    with torch.no_grad():
        outputs = thinker_cg(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    elapsed = time.time() - t0
    logger.info("Thinker forward done in %.1f s", elapsed)

    hidden_states = getattr(outputs, "hidden_states", None)
    logits = getattr(outputs, "logits", None)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_npy(output_dir, "thinker_input_ids", input_ids.numpy().astype(np.int64))

    if logits is not None:
        logger.info("logits shape: %s", logits.shape)
        save_npy(output_dir, "thinker_logits", to_f32_numpy(logits))

    if hidden_states is not None:
        logger.info(
            "Saving hidden states: %d tensors (embedding + %d layers)",
            len(hidden_states), len(hidden_states) - 1,
        )
        # hidden_states[0] = embedding output (before any transformer block)
        save_npy(output_dir, "thinker_embedding_output", to_f32_numpy(hidden_states[0]))
        # hidden_states[1..N] = output of each transformer layer
        for i, hs in enumerate(hidden_states[1:]):
            save_npy(output_dir, f"thinker_layer_{i:02d}_output", to_f32_numpy(hs))
        # hidden_states[-1] = output after final RMSNorm (same as last layer in most impls)
        save_npy(output_dir, "thinker_final_norm_output", to_f32_numpy(hidden_states[-1]))
    else:
        logger.warning(
            "No hidden_states in output. Falling back to forward hooks on thinker_inner."
        )
        _extract_thinker_via_hooks(thinker_inner, input_ids, output_dir)

    log_ram()


def _extract_thinker_via_hooks(thinker_inner, input_ids: torch.Tensor, output_dir: Path) -> None:
    """Fallback: extract thinker activations via forward hooks on the inner model."""
    capture = ActivationCapture(capture_once=False)
    capture.register("thinker_embedding", thinker_inner.embed_tokens)
    for i, layer in enumerate(thinker_inner.layers):
        capture.register(f"thinker_layer_{i:02d}", layer)
    capture.register("thinker_final_norm", thinker_inner.norm)

    logger.info("Running thinker_inner forward via hooks ...")
    with torch.no_grad():
        thinker_inner(input_ids=input_ids, use_cache=False)

    capture.remove_hooks()
    for name, arr in sorted(capture.activations.items()):
        save_npy(output_dir, f"{name}_output", arr)
    logger.info("Thinker (hooks): saved %d tensors", len(capture.activations))


# ---------------------------------------------------------------------------
# Stage: talker (TTS decoder)
# ---------------------------------------------------------------------------

def extract_talker(model, output_dir: Path) -> None:
    """Extract per-layer activations from the Talker TTS decoder.

    The Talker lives at model.talker (Qwen2_5OmniTalkerForConditionalGeneration).
    Its inner model: model.talker.model (Qwen2_5OmniTalkerModel)
      .embed_tokens: vocab_size=8448, embedding_size=2048
      .layers[0..23]: 24 decoder layers, hidden_size=896
      .norm: RMSNorm

    The talker takes:
      - input_ids: codec token IDs [1, seq_len] (vocab 8448)
      - thinker_reply_part: projected thinker hidden states [1, thinker_len, hidden_size]

    For isolated validation we use a minimal synthetic input:
      - input_ids = [[8293]]  (tts_codec_start_token_id)
      - thinker_reply_part = zeros [1, 1, 896]  (projected to talker hidden_size)

    IMPORTANT: The talker's forward applies thinker_to_talker_proj (2048->896) to
    thinker_reply_part internally only when concatenating with text embeddings.
    For a clean isolated test, we pass inputs_embeds directly to talker.model.

    Saved files:
      talker_input_ids.npy                 [1, 1]
      talker_codec_embedding_output.npy    [1, 1, 896]
      talker_layer_{00..23}_output.npy     [1, 1, 896]
      talker_final_norm_output.npy         [1, 1, 896]
      talker_logits.npy                    [1, 1, 8448]  (codec head output)
    """
    logger.info("=" * 70)
    logger.info("STAGE: talker TTS decoder")
    logger.info("=" * 70)

    talker_cg = find_attr(model, ["talker"])
    if talker_cg is None:
        logger.error("Could not locate talker in model hierarchy")
        return

    talker_inner = find_attr(model, ["talker.model"])
    if talker_inner is None:
        logger.error("Could not locate talker.model")
        return

    n_layers = len(talker_inner.layers)
    hidden_size = talker_cg.config.hidden_size
    embedding_size = talker_cg.config.embedding_size
    vocab_size = talker_cg.config.vocab_size
    codec_bos = talker_cg.codec_bos_token  # 8293

    logger.info(
        "Talker: layers=%d, hidden_size=%d, embedding_size=%d, vocab_size=%d, codec_bos=%d",
        n_layers, hidden_size, embedding_size, vocab_size, codec_bos,
    )

    # Single-token input: the codec BOS token
    torch.manual_seed(42)
    input_ids = torch.tensor([[codec_bos]], dtype=torch.long)
    logger.info("Input IDs: %s", input_ids.tolist())

    # Get the codec token embedding (embedding_size=2048)
    with torch.no_grad():
        codec_embed = talker_inner.embed_tokens(input_ids)  # [1, 1, 2048]
    logger.info("codec_embed shape: %s (embedding_size=%d)", codec_embed.shape, embedding_size)

    # The talker's forward always applies thinker_to_talker_proj (2048->896) before
    # passing inputs to talker.model. We replicate that projection here so we can call
    # talker_inner (self.model) directly with correct 896-dim embeddings.
    # This also lets us capture thinker_to_talker_proj output as a reference.
    proj = talker_cg.thinker_to_talker_proj  # Linear(2048, 896, bias=True)
    with torch.no_grad():
        projected_embed = proj(codec_embed.to(proj.weight.dtype))  # [1, 1, 896]
    logger.info(
        "thinker_to_talker_proj output shape: %s  (2048->%d)",
        projected_embed.shape, hidden_size,
    )

    # Register hooks on talker inner model layers
    capture = ActivationCapture(capture_once=False)
    for i, layer in enumerate(talker_inner.layers):
        capture.register(f"talker_layer_{i:02d}", layer)
    capture.register("talker_final_norm", talker_inner.norm)

    logger.info("Running talker inner model forward (projected 896-dim embeds) ...")
    t0 = time.time()
    with torch.no_grad():
        talker_out = talker_inner(
            inputs_embeds=projected_embed,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    elapsed = time.time() - t0
    logger.info("Talker inner forward done in %.1f s", elapsed)

    capture.remove_hooks()

    output_dir.mkdir(parents=True, exist_ok=True)
    save_npy(output_dir, "talker_input_ids", input_ids.numpy().astype(np.int64))
    save_npy(output_dir, "talker_codec_embedding_output", to_f32_numpy(codec_embed))
    save_npy(output_dir, "talker_proj_output", to_f32_numpy(projected_embed))

    # Save hidden states from output_hidden_states
    # hidden_states[0] = inputs_embeds (the projected embed, before any layer)
    # hidden_states[1..N] = per-layer outputs
    # hidden_states[-1] = after final RMSNorm (same as last layer output post-norm)
    hs_list = getattr(talker_out, "hidden_states", None)
    if hs_list is not None:
        logger.info("Talker hidden_states: %d tensors", len(hs_list))
        save_npy(output_dir, "talker_inputs_embeds_hs", to_f32_numpy(hs_list[0]))
        for i, hs in enumerate(hs_list[1:]):
            save_npy(output_dir, f"talker_layer_{i:02d}_output", to_f32_numpy(hs))
        # hs_list[-1] is the post-norm output (last entry includes norm applied to last layer)
        save_npy(output_dir, "talker_final_norm_output", to_f32_numpy(hs_list[-1]))
    else:
        logger.info("No hidden_states in talker output; using hook captures.")
        for name, arr in sorted(capture.activations.items()):
            save_npy(output_dir, f"{name}_output", arr)

    # Compute logits via codec_head on the last hidden state
    last_hidden = getattr(talker_out, "last_hidden_state", None)
    if last_hidden is None and hs_list is not None:
        last_hidden = hs_list[-1]
    if last_hidden is not None and hasattr(talker_cg, "codec_head"):
        with torch.no_grad():
            logits = talker_cg.codec_head(last_hidden)
        logger.info("talker logits shape: %s", logits.shape)
        save_npy(output_dir, "talker_logits", to_f32_numpy(logits))

    log_ram()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-component reference data from Qwen2.5-Omni-3B"
    )
    parser.add_argument(
        "--model-path",
        default="/home/alexmak/lluda/models/Qwen2.5-Omni-3B",
        help="Path to the Qwen2.5-Omni-3B model directory",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/alexmak/lluda/reference_data/omni_3b",
        help="Directory to save .npy reference files",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "audio", "thinker", "talker", "inspect"],
        default="all",
        help=(
            "Which component to extract: "
            "inspect (config only, no model load), "
            "audio (audio encoder), "
            "thinker (text decoder), "
            "talker (TTS decoder), "
            "all (audio + thinker + talker)"
        ),
    )
    parser.add_argument(
        "--thinker-only-load",
        action="store_true",
        help="Load only the Thinker (saves memory when talker stage is not needed)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # Inspect stage requires no model loading
    if args.stage == "inspect":
        inspect_model(args.model_path)
        return

    # Decide what to load
    need_talker = args.stage in ("all", "talker")
    if need_talker and not args.thinker_only_load:
        model = load_model(args.model_path)
    else:
        model = load_thinker_only(args.model_path)

    if args.stage in ("all", "audio"):
        try:
            extract_audio_encoder(model, output_dir)
        except Exception as e:
            logger.error("Audio encoder extraction failed: %s", e, exc_info=True)

    if args.stage in ("all", "thinker"):
        try:
            extract_thinker(model, output_dir)
        except Exception as e:
            logger.error("Thinker extraction failed: %s", e, exc_info=True)

    if args.stage in ("all", "talker"):
        if not hasattr(model, "talker"):
            logger.error(
                "Talker not found in model — was the full model loaded? "
                "Re-run without --thinker-only-load or use --stage audio/thinker."
            )
        else:
            try:
                extract_talker(model, output_dir)
            except Exception as e:
                logger.error("Talker extraction failed: %s", e, exc_info=True)

    logger.info("=" * 70)
    logger.info("Done. Reference data saved to: %s", output_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
