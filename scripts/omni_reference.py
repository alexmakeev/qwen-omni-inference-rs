#!/usr/bin/env python3
"""
Generate reference activation data from Qwen3-Omni-30B or Qwen2.5-Omni-3B
for validating a Rust inference implementation.

This script runs on a low-RAM machine (30GB RAM + swap). The model is 70.5GB
BF16 — inference will be heavily swapped and slow (minutes per token). This
is expected: we need activations, not throughput.

Usage:
    python scripts/omni_reference.py --mode text_test
    python scripts/omni_reference.py --mode asr_test
    python scripts/omni_reference.py --mode tts_test
    python scripts/omni_reference.py --mode audio_to_audio
    python scripts/omni_reference.py --mode full_reference
    python scripts/omni_reference.py --mode server [--server-port 8899]

Optional overrides:
    --model-path /path/to/Qwen3-Omni-30B  (or Qwen2.5-Omni-3B)
    --output-dir /path/to/reference_data
    --audio-file /path/to/input.wav
    --server-port 8899  (server mode only)

Model family is auto-detected from config.json (model_type field).
Supported: qwen3_omni_moe (Qwen3-Omni-30B), qwen2_5_omni (Qwen2.5-Omni-3B).
"""

import argparse
import http.server
import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = "/home/alexmak/lluda/models/Qwen3-Omni-30B"
DEFAULT_REFERENCE_DIR = "/home/alexmak/lluda/reference_data"
DEFAULT_AUDIO_FILE = "/home/alexmak/lluda/main/materials/privet_mir.wav"


# ---------------------------------------------------------------------------
# Model family detection
# ---------------------------------------------------------------------------

def detect_model_family(model_path: str) -> str:
    """Return 'qwen3_omni_moe' or 'qwen2_5_omni' based on config.json model_type."""
    import json
    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    model_type = config.get("model_type", "")
    if "qwen3_omni_moe" in model_type:
        return "qwen3_omni_moe"
    elif "qwen2_5_omni" in model_type or "qwen2.5_omni" in model_type:
        return "qwen2_5_omni"
    else:
        raise ValueError(
            f"Unknown model_type in {config_path!r}: {model_type!r}. "
            "Expected 'qwen3_omni_moe' or 'qwen2_5_omni'."
        )


# ---------------------------------------------------------------------------
# Config navigation helpers
# ---------------------------------------------------------------------------

def _get_text_config(config):
    """Navigate to text_config regardless of which config level we're at.

    Config hierarchy for Qwen3-Omni-30B (qwen3_omni_moe):
      Qwen3OmniMoeConfig            -> .thinker_config.text_config
      Qwen3OmniMoeThinkerConfig     -> .text_config
      Qwen3OmniMoeTextConfig        -> (already the text config)

    Config hierarchy for Qwen2.5-Omni-3B (qwen2_5_omni):
      Qwen2_5OmniConfig             -> .thinker_config.text_config
                                       OR .text_config directly
      Qwen2_5OmniThinkerConfig      -> .text_config OR already text config

    num_hidden_layers, vocab_size, hidden_size live in the text config.
    """
    # Full model config: config.thinker_config.text_config
    if hasattr(config, 'thinker_config'):
        tc = config.thinker_config
        if hasattr(tc, 'text_config'):
            return tc.text_config
        return tc
    # Thinker config or flat config with text_config nested
    if hasattr(config, 'text_config'):
        return config.text_config
    # Already text config (has num_hidden_layers directly)
    return config


def _get_audio_config(config):
    """Navigate to audio_config regardless of which config level we're at.

    Config hierarchy:
      Qwen3OmniMoeConfig            -> .thinker_config.audio_config
      Qwen3OmniMoeThinkerConfig     -> .audio_config

    num_hidden_layers=32 lives here.
    """
    # Full model config: config.thinker_config.audio_config
    if hasattr(config, 'thinker_config'):
        tc = config.thinker_config
        if hasattr(tc, 'audio_config'):
            return tc.audio_config
        return tc
    # Thinker config: config.audio_config
    if hasattr(config, 'audio_config'):
        return config.audio_config
    return config


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def log_ram():
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


def save_npy(save_dir: Path, name: str, tensor: np.ndarray):
    """Save a numpy array as .npy file, logging shape and size."""
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{name}.npy"
    np.save(path, tensor)
    size_kb = path.stat().st_size / 1024
    logger.info("  Saved %s: shape=%s  %.1f KB", name, tensor.shape, size_kb)


def save_text(save_dir: Path, name: str, text: str):
    """Save a text string to a file."""
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / name
    path.write_text(text, encoding="utf-8")
    logger.info("  Saved %s: %r", name, text[:120])


def to_f32_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert any torch tensor (including BF16) to float32 numpy array."""
    return tensor.detach().cpu().to(torch.float32).numpy()


# ---------------------------------------------------------------------------
# Activation capture via forward hooks
# ---------------------------------------------------------------------------

class ActivationCapture:
    """Register forward hooks on named modules and collect outputs."""

    def __init__(self):
        self.activations: Dict[str, np.ndarray] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

    def register(self, name: str, module: torch.nn.Module):
        def hook_fn(mod, inp, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            self.activations[name] = to_f32_numpy(out)

        self._hooks.append(module.register_forward_hook(hook_fn))

    def clear(self):
        self.activations.clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_thinker(model_path: str):
    """
    Load only the Thinker component (lighter, text-only inference).

    Auto-detects model family from config.json (qwen3_omni_moe or qwen2_5_omni).
    Returns (model, processor).
    """
    family = detect_model_family(model_path)
    logger.info("Detected model family: %s", family)

    if family == "qwen3_omni_moe":
        from transformers import Qwen3OmniMoeThinkerForConditionalGeneration as ThinkerCls
        from transformers import Qwen3OmniMoeProcessor as ProcessorCls
    elif family == "qwen2_5_omni":
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration as ThinkerCls
        from transformers import Qwen2_5OmniProcessor as ProcessorCls
    else:
        raise ValueError(f"Unsupported model family: {family}")

    logger.info("Loading Thinker from %s ...", model_path)
    logger.info("(device_map=cpu, low_cpu_mem_usage=True, bfloat16)")
    log_ram()
    t0 = time.time()

    processor = ProcessorCls.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    model = ThinkerCls.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()

    elapsed = time.time() - t0
    text_cfg = _get_text_config(model.config)
    num_layers = text_cfg.num_hidden_layers
    logger.info(
        "Thinker loaded in %.1f s — %d layers, vocab=%d",
        elapsed, num_layers, text_cfg.vocab_size,
    )
    log_ram()
    return model, processor


def load_full_model(model_path: str, enable_audio_output: bool = False):
    """
    Load the full Omni model (Thinker + Audio Encoder + Talker + Code2Wav).

    Auto-detects model family from config.json (qwen3_omni_moe or qwen2_5_omni).
    Returns (model, processor).
    """
    family = detect_model_family(model_path)
    logger.info("Detected model family: %s", family)

    if family == "qwen3_omni_moe":
        from transformers import Qwen3OmniMoeForConditionalGeneration as FullCls
        from transformers import Qwen3OmniMoeProcessor as ProcessorCls
    elif family == "qwen2_5_omni":
        from transformers import Qwen2_5OmniForConditionalGeneration as FullCls
        from transformers import Qwen2_5OmniProcessor as ProcessorCls
    else:
        raise ValueError(f"Unsupported model family: {family}")

    logger.info(
        "Loading FULL %s from %s (audio_output=%s) ...",
        family, model_path, enable_audio_output,
    )
    log_ram()
    t0 = time.time()

    processor = ProcessorCls.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    load_kwargs: dict = dict(
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    if enable_audio_output:
        load_kwargs["enable_audio_output"] = True

    model = FullCls.from_pretrained(model_path, **load_kwargs)
    model.eval()

    elapsed = time.time() - t0
    logger.info("Full model loaded in %.1f s", elapsed)
    log_ram()
    return model, processor


# ---------------------------------------------------------------------------
# Mode: text_test
# ---------------------------------------------------------------------------

def run_text_test(
    model_path: str,
    output_base: Path,
    model=None,
    processor=None,
    family: Optional[str] = None,
):
    """
    Load Thinker only, run a simple text prompt, save basic outputs and
    per-layer hidden states for all Thinker transformer layers.

    If model and processor are provided, skips loading (server mode).
    family: model family string ('qwen3_omni_moe' or 'qwen2_5_omni').
            Auto-detected from model_path if not provided.

    Saves:
        input_ids.npy
        embedding_output.npy       (hidden_states[0])
        thinker_layer_00_output.npy .. thinker_layer_47_output.npy
                                   (hidden_states[1..48])
        final_norm_output.npy      (hidden_states[-1], same as last layer output
                                    which is after the final RMS norm in most
                                    Qwen3/Qwen2.5 implementations)
        logits.npy
        generated_text.txt
    """
    logger.info("=" * 70)
    logger.info("MODE: text_test")
    logger.info("=" * 70)

    save_dir = output_base / "omni_text_test"
    max_new_tokens = 10

    if family is None:
        family = detect_model_family(model_path)
    logger.info("Model family: %s", family)

    if model is None or processor is None:
        model, processor = load_thinker(model_path)

    # Check if this is the full model or thinker-only.
    # Qwen3OmniMoeForConditionalGeneration has no forward() of its own;
    # the Thinker sub-model does. Use thinker for forward and generate.
    if hasattr(model, 'thinker'):
        forward_model = model.thinker  # Full model → use thinker for forward
        generate_model = model.thinker  # For text-only generation, use thinker directly
        logger.info("Server mode: using model.thinker for forward/generate in text_test")
    else:
        forward_model = model  # Already the thinker
        generate_model = model

    # Build prompt via chat template for proper assistant-format response
    messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    logger.info("Applying chat template for: %r", messages[0]["content"])
    # Qwen3-Omni has thinking mode enabled by default; disable it to avoid
    # safety refusals during reference data collection.
    template_kwargs = dict(tokenize=False, add_generation_prompt=True)
    if family == "qwen3_omni_moe":
        template_kwargs["enable_thinking"] = False
    if hasattr(processor, "apply_chat_template"):
        prompt_text = processor.apply_chat_template(messages, **template_kwargs)
    else:
        prompt_text = processor.tokenizer.apply_chat_template(messages, **template_kwargs)
    logger.info("Formatted prompt: %r", prompt_text[:200])

    # Tokenize
    logger.info("Tokenizing formatted prompt ...")
    inputs = processor.tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    logger.info("input_ids shape: %s  tokens: %s", input_ids.shape, input_ids[0].tolist())

    # Single forward pass — request hidden states from all layers
    logger.info("Running forward pass (output_hidden_states=True) ...")
    t0 = time.time()
    with torch.no_grad():
        outputs = forward_model(**inputs, output_hidden_states=True, return_dict=True)
    fwd_elapsed = time.time() - t0
    logger.info("Forward pass done in %.1f s", fwd_elapsed)

    logits = outputs.logits  # [1, seq_len, vocab]
    logger.info("logits shape: %s", logits.shape)

    # Extract hidden states
    hidden_states = getattr(outputs, "hidden_states", None)

    save_dir.mkdir(parents=True, exist_ok=True)

    # Save input_ids and logits
    save_npy(save_dir, "input_ids", input_ids.numpy().astype(np.int64))
    save_npy(save_dir, "logits", to_f32_numpy(logits))

    # Save per-layer hidden states
    if hidden_states is not None:
        logger.info(
            "Saving hidden states: %d tensors (embedding + %d layers)",
            len(hidden_states), len(hidden_states) - 1,
        )
        # hidden_states[0] = embedding output (input to layer 0)
        save_npy(save_dir, "embedding_output", to_f32_numpy(hidden_states[0]))
        # hidden_states[1..N] = output of each transformer layer
        for i, hs in enumerate(hidden_states[1:]):
            save_npy(save_dir, f"thinker_layer_{i:02d}_output", to_f32_numpy(hs))
        # Final norm output: last hidden state (after the model's final RMS norm)
        save_npy(save_dir, "final_norm_output", to_f32_numpy(hidden_states[-1]))
    else:
        logger.warning(
            "No hidden_states in output — model may not support output_hidden_states. "
            "Only logits saved."
        )

    # Generation
    logger.info("Generating up to %d tokens ...", max_new_tokens)
    with torch.no_grad():
        t_gen_start = time.time()
        generated = generate_model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
        total_gen = time.time() - t_gen_start

    n_new = generated.shape[1] - input_ids.shape[1]
    if n_new > 0:
        avg_tok_time = total_gen / n_new
    else:
        avg_tok_time = 0.0
    logger.info(
        "Generated %d tokens in %.1f s  (%.1f s/token)",
        n_new, total_gen, avg_tok_time,
    )

    # Decode only new tokens (skip the prompt)
    new_ids = generated[0, input_ids.shape[1]:]
    generated_text = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
    logger.info("Generated text (new tokens only): %r", generated_text)

    save_text(save_dir, "generated_text.txt", generated_text)
    log_ram()
    logger.info("text_test complete. Output: %s", save_dir)


# ---------------------------------------------------------------------------
# Mode: asr_test
# ---------------------------------------------------------------------------

def run_asr_test(
    model_path: str,
    audio_file: str,
    output_base: Path,
    model=None,
    processor=None,
    family: Optional[str] = None,
):
    """
    Load full model, transcribe audio file.

    If model and processor are provided, skips loading (server mode).
    family: model family string ('qwen3_omni_moe' or 'qwen2_5_omni').
            Auto-detected from model_path if not provided.

    Saves: generated_text.txt
    """
    logger.info("=" * 70)
    logger.info("MODE: asr_test")
    logger.info("=" * 70)

    save_dir = output_base / "omni_asr_test"
    max_new_tokens = 562  # 512 thinking + 50 answer budget

    if not Path(audio_file).exists():
        logger.error("Audio file not found: %s", audio_file)
        return

    if family is None:
        family = detect_model_family(model_path)
    logger.info("Model family: %s", family)

    if model is None or processor is None:
        model, processor = load_full_model(model_path, enable_audio_output=False)

    # System message is required — model was trained with it and behaves
    # unreliably without it (produces hallucinated short English phrases).
    # Use an explicit speech recognition system prompt to prevent safety refusals.
    # "You are a helpful assistant." is too generic and triggers refusals on
    # Qwen3-Omni when asked to transcribe Russian audio in thinking mode.
    conversations = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio_url": f"file://{audio_file}",
                },
                {
                    "type": "text",
                    "text": "Transcribe the audio into text.",
                },
            ],
        },
    ]

    logger.info("Applying chat template ...")
    # For Qwen3-Omni: build a forced thinking prefix so the model skips thinking
    # and goes directly to transcription. This is necessary because:
    # 1. enable_thinking=True + greedy decoding → degenerate "The." answer
    # 2. enable_thinking=False → model generates a new <|im_start|> turn instead of answering
    # 3. Forced thinking prefix: pre-fill <think>...</think> so the model continues
    #    with the actual transcription after the closing </think> tag.
    #
    # CRITICAL: We use model.generate() (NOT model.thinker.generate()) here,
    # so the audio tower IS invoked — audio features are processed correctly.
    template_kwargs = dict(add_generation_prompt=True, tokenize=False)
    if family == "qwen3_omni_moe":
        # For Qwen3-Omni: apply chat template then manually pre-fill a thinking
        # prefix. This is the confirmed-working approach from server_30b_v18.log.
        # Without the prefix:
        #   - enable_thinking=False → model generates empty response
        #   - default thinking mode + greedy → degenerate "The." / refusal
        # With the prefix the model skips its own <think> generation and continues
        # directly to the transcription after </think>.
        text = processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )
        # Pre-fill thinking to prevent degenerate early EOS or safety refusal.
        # IMPORTANT: apply_chat_template ends with "<|im_start|>assistant\n".
        # We must NOT strip the trailing "\n" before appending <think>, because
        # Qwen3 requires the format "assistant\n<think>" — if the newline is
        # missing ("assistant<think>"), the model does not recognize the turn
        # start and generates garbage (new <|im_start|>user turn instead of output).
        text = text.rstrip()  # strips trailing \n (template ends with "assistant\n")
        if not text.endswith("<think>"):
            # Re-add the newline that rstrip() removed, then append think block.
            # Result: "...<|im_start|>assistant\n<think>\n...\n</think>\n\n"
            text += "\n<think>\nThe user wants me to transcribe the audio content into text.\n</think>\n\n"
        logger.info("Qwen3: pre-filling thinking prefix to prevent degenerate early EOS")
    else:
        text = processor.apply_chat_template(conversations, **template_kwargs)
    logger.info("Chat template output (first 400 chars): %s", text[:400])

    # Load and preprocess audio via process_mm_info (official API).
    # This loads the file via librosa and resamples to 16 kHz automatically.
    # The return value is a list of numpy arrays (one per audio in the conversation).
    try:
        from qwen_omni_utils import process_mm_info
        audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)
        logger.info(
            "Audio loaded via process_mm_info: shape=%s, sr=16000",
            audios[0].shape,
        )
    except Exception as exc:
        logger.warning("process_mm_info failed (%s) — falling back to manual load", exc)
        raw_audio, raw_sr = _load_audio(audio_file)
        if raw_audio is None:
            logger.error("Could not load audio from %s", audio_file)
            return
        if raw_sr != 16000:
            try:
                import librosa
                logger.info("Resampling audio from %d to 16000 Hz", raw_sr)
                raw_audio = librosa.resample(raw_audio, orig_sr=raw_sr, target_sr=16000)
            except ImportError:
                logger.error("librosa not available for resampling. Install with: pip install librosa")
                return
        audios = [raw_audio]
        images = None
        videos = None

    logger.info("Processing inputs ...")
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
    )
    # Convert float32 tensors to bfloat16 to match model weights.
    # input_features (mel spectrograms) must also be bf16 — the audio encoder
    # conv2d receives them directly and requires matching dtype.
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor) and inputs[key].is_floating_point() and inputs[key].dtype != torch.bfloat16:
            inputs[key] = inputs[key].to(torch.bfloat16)
    logger.info(
        "Input tensors: %s",
        {k: (v.shape, v.dtype) for k, v in inputs.items() if hasattr(v, "shape")},
    )
    log_ram()

    # Debug: count audio pad tokens in input_ids
    audio_pad_id = 151675  # <|audio_pad|> token ID
    audio_pad_count = (inputs["input_ids"] == audio_pad_id).sum().item()
    logger.info("DEBUG: audio_pad tokens in input_ids: %d", audio_pad_count)

    # Use model.generate() for Qwen3-Omni.
    # The audio tower is part of the thinker (Qwen3OmniMoeThinkerForConditionalGeneration
    # contains self.audio_tower), so both model.generate() and model.thinker.generate()
    # process audio correctly.
    #
    # return_audio=False skips talker/code2wav.
    # thinker_max_new_tokens replaces max_new_tokens (wrapper ignores max_new_tokens).
    # With forced thinking prefix (think block pre-filled), we only need answer budget.
    # Use do_sample=True to avoid greedy collapse to degenerate outputs.
    logger.info("Generating via model.generate() (max_new_tokens: %d) ...", max_new_tokens)
    t0 = time.time()
    with torch.no_grad():
        if family == "qwen3_omni_moe":
            # Full model generate() routes audio through the audio tower.
            # return_audio=False skips talker/code2wav — text only.
            # thinker_max_new_tokens replaces max_new_tokens (wrapper ignores max_new_tokens).
            # do_sample=True avoids greedy collapse to short degenerate outputs.
            # model.generate() internally calls self.thinker.generate().
            # With forced thinking prefix in prompt, the model continues after
            # </think> and generates the actual transcription in a new turn.
            # thinker_max_new_tokens replaces max_new_tokens (wrapper ignores it).
            # thinker_do_sample=False for deterministic output.
            result = model.generate(
                **inputs,
                return_audio=False,
                thinker_max_new_tokens=max_new_tokens,
                thinker_do_sample=False,
                thinker_pad_token_id=processor.tokenizer.eos_token_id,
                thinker_eos_token_id=processor.tokenizer.eos_token_id,
            )
            if isinstance(result, tuple):
                generated = result[0]
                if hasattr(generated, 'sequences'):
                    generated = generated.sequences
            elif hasattr(result, 'sequences'):
                generated = result.sequences
            else:
                generated = result
        else:
            # Qwen2.5 does not have return_audio parameter
            result = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            # generate may return (text_ids, audio) tuple if talker is active
            if isinstance(result, tuple):
                generated = result[0]
            else:
                generated = result
    elapsed = time.time() - t0

    input_len = inputs["input_ids"].shape[1]
    n_new = generated.shape[1] - input_len
    logger.info(
        "Generated %d tokens in %.1f s  (%.1f s/token)",
        n_new, elapsed, elapsed / max(n_new, 1),
    )

    # Decode only new tokens.
    # For Qwen3-Omni with forced thinking prefix, after </think>\n\n the model may
    # generate a new <|im_start|>assistant\n turn before the actual transcription.
    # skip_special_tokens=True removes <|im_start|> but leaves "assistant\n".
    # We strip this prefix plus any <think>...</think> blocks from the output.
    new_ids = generated[:, input_len:]
    # Debug: log raw token IDs for analysis (always log first 20 tokens)
    raw_token_ids = new_ids[0].tolist()
    debug_ids = raw_token_ids[:20]
    debug_tokens = [processor.tokenizer.decode([tid]) for tid in debug_ids]
    logger.info("DEBUG generated token IDs (first %d): %s", len(debug_ids), debug_ids)
    logger.info("DEBUG decoded (first %d): %s", len(debug_ids), debug_tokens)
    raw_decoded = processor.tokenizer.decode(new_ids[0], skip_special_tokens=True)
    logger.info("DEBUG raw decoded: %r", raw_decoded[:200])

    # Post-processing pipeline for Qwen3-Omni ASR output:
    # 1. Strip any <think>...</think> blocks (if thinking was active)
    cleaned = re.sub(r'<think>.*?</think>', '', raw_decoded, flags=re.DOTALL)
    # 2. Strip "assistant\n" prefix (artifact of new-turn generation after forced prefix)
    cleaned = re.sub(r'^assistant\s*\n', '', cleaned.strip()).strip()
    # 3. Strip "Assistant:" prefix (another variant)
    cleaned = re.sub(r'^Assistant\s*:\s*', '', cleaned).strip()

    if cleaned != raw_decoded.strip():
        logger.info("Post-processed output (raw len=%d, cleaned len=%d)", len(raw_decoded), len(cleaned))

    transcribed = cleaned
    logger.info("Transcription: %r", transcribed)

    save_text(save_dir, "generated_text.txt", transcribed)
    log_ram()
    logger.info("asr_test complete. Output: %s", save_dir)


# ---------------------------------------------------------------------------
# Mode: tts_test
# ---------------------------------------------------------------------------

def run_tts_test(
    model_path: str,
    output_base: Path,
    model=None,
    processor=None,
    family: Optional[str] = None,
):
    """
    Load full model with audio output enabled, generate speech.

    If model and processor are provided, skips loading (server mode).
    Note: server mode loads a single full model; TTS requires enable_audio_output=True
    which may not match the server-loaded model. A warning is logged in that case.

    Saves: output.wav
    """
    logger.info("=" * 70)
    logger.info("MODE: tts_test")
    logger.info("=" * 70)

    save_dir = output_base / "omni_tts_test"
    save_dir.mkdir(parents=True, exist_ok=True)

    text_to_speak = "Привет, мир! Как дела?"

    if family is None:
        family = detect_model_family(model_path)
    logger.info("Model family: %s", family)

    if model is None or processor is None:
        model, processor = load_full_model(model_path, enable_audio_output=True)
    else:
        logger.warning(
            "tts_test: using pre-loaded model (server mode). "
            "Audio output may not be available if model was loaded without enable_audio_output=True."
        )

    conversations = [
        {
            "role": "user",
            "content": [{"type": "text", "text": text_to_speak}],
        }
    ]

    # Qwen3-Omni has thinking mode enabled by default; disable it to avoid
    # safety refusals during reference data collection.
    template_kwargs = dict(add_generation_prompt=True, tokenize=False)
    if family == "qwen3_omni_moe":
        template_kwargs["enable_thinking"] = False
    text = processor.apply_chat_template(conversations, **template_kwargs)
    logger.info("Chat template output (first 300 chars): %s", text[:300])
    inputs = processor(text=text, return_tensors="pt")
    # Convert float32 tensors to bfloat16 to match model weights dtype
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor) and inputs[key].dtype == torch.float32:
            inputs[key] = inputs[key].to(torch.bfloat16)

    logger.info("Generating TTS for: %r", text_to_speak)
    log_ram()
    t0 = time.time()

    # Check if model supports return_audio (requires enable_audio_output=True)
    has_talker = hasattr(model, "talker") or getattr(model, "has_talker", False)
    output_wav = save_dir / "output.wav"

    with torch.no_grad():
        if has_talker:
            # Full model loaded with enable_audio_output=True: returns (text_ids, audio_waveform)
            result = model.generate(
                **inputs,
                return_audio=True,
                speaker="Chelsie",
                thinker_max_new_tokens=256,
                do_sample=False,
            )
            if isinstance(result, tuple) and len(result) == 2:
                text_ids, audio = result
            else:
                # Unexpected shape — treat as text only
                text_ids = result
                audio = None
        else:
            # Thinker-only or model without talker: text generation only
            logger.warning(
                "Model does not have talker (has_talker=False). "
                "Generating text only — no audio output."
            )
            text_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
            audio = None

    elapsed = time.time() - t0
    logger.info("TTS generation done in %.1f s", elapsed)

    # Save generated text (new tokens only — skip the input prompt)
    if isinstance(text_ids, torch.Tensor):
        input_len = inputs["input_ids"].shape[1]
        new_ids = text_ids[0, input_len:] if text_ids.shape[1] > input_len else text_ids[0]
        generated_text = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
        save_text(save_dir, "generated_text.txt", generated_text)
        logger.info("Generated text: %r", generated_text[:200])

    # Save audio waveform if available
    if audio is not None:
        # audio shape: [1, 1, num_samples] or similar — flatten to 1D
        audio_np = audio.reshape(-1).detach().cpu().float().numpy()
        import soundfile as sf
        sf.write(str(output_wav), audio_np, samplerate=24000)
        duration = len(audio_np) / 24000
        logger.info("Audio saved: %.2f s at 24000 Hz -> %s", duration, output_wav)
    else:
        logger.warning("No audio waveform produced — check model has_talker status")

    log_ram()
    logger.info("tts_test complete. Output: %s", save_dir)


# ---------------------------------------------------------------------------
# Mode: audio_to_audio
# ---------------------------------------------------------------------------

def run_audio_to_audio_test(
    model_path: str,
    output_base: Path,
    audio_file: str,
    model=None,
    processor=None,
    family: Optional[str] = None,
):
    """
    Take audio input, generate a spoken response (audio-to-audio).

    Loads the full model with enable_audio_output=True (or reuses pre-loaded
    model in server mode). Saves both the text transcription and the audio
    waveform of the model's spoken response.

    If model and processor are provided, skips loading (server mode).

    Saves:
        generated_text.txt   — text content of the model's response
        output.wav           — spoken audio response at 24000 Hz
    """
    logger.info("=" * 70)
    logger.info("MODE: audio_to_audio")
    logger.info("=" * 70)

    save_dir = output_base / "omni_audio_to_audio"
    save_dir.mkdir(parents=True, exist_ok=True)

    if not Path(audio_file).exists():
        logger.error("Audio file not found: %s", audio_file)
        return

    if family is None:
        family = detect_model_family(model_path)
    logger.info("Model family: %s", family)

    if model is None or processor is None:
        model, processor = load_full_model(model_path, enable_audio_output=True)

    # Load and preprocess audio
    audio_array, sample_rate = _load_audio(audio_file)
    if audio_array is None:
        logger.error("Could not load audio from %s", audio_file)
        return

    # Resample to 16kHz for the feature extractor
    if sample_rate != 16000:
        try:
            import librosa
            logger.info("Resampling audio from %d to 16000 Hz", sample_rate)
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        except ImportError:
            logger.error("librosa not available for resampling. Install with: pip install librosa")
            return

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": f"file://{audio_file}"},
                {"type": "text", "text": "Please respond to what you hear in the audio."},
            ],
        },
    ]

    logger.info("Applying chat template ...")
    # Qwen3-Omni has thinking mode enabled by default; disable it to avoid
    # safety refusals during reference data collection.
    template_kwargs = dict(add_generation_prompt=True, tokenize=False)
    if family == "qwen3_omni_moe":
        template_kwargs["enable_thinking"] = False
    prompt_text = processor.apply_chat_template(messages, **template_kwargs)
    logger.info("Chat template output (first 300 chars): %s", prompt_text[:300])

    logger.info("Processing inputs via process_mm_info ...")
    from qwen_omni_utils import process_mm_info
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    inputs = processor(
        text=prompt_text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
    )
    # Convert float32 tensors to bfloat16 to match model weights.
    # input_features (mel spectrograms) must also be bf16 — the audio encoder
    # conv2d receives them directly and requires matching dtype.
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor) and inputs[key].is_floating_point() and inputs[key].dtype != torch.bfloat16:
            inputs[key] = inputs[key].to(torch.bfloat16)
    logger.info(
        "Input tensors: %s",
        {k: (v.shape, v.dtype) for k, v in inputs.items() if hasattr(v, "shape")},
    )
    log_ram()

    # Check if model supports audio output
    has_talker = hasattr(model, "talker") or getattr(model, "has_talker", False)

    logger.info("Generating spoken response (has_talker=%s) ...", has_talker)
    t0 = time.time()
    with torch.no_grad():
        if has_talker:
            result = model.generate(
                **inputs,
                return_audio=True,
                speaker="Chelsie",
                thinker_max_new_tokens=256,
                do_sample=False,
            )
            if isinstance(result, tuple) and len(result) == 2:
                text_ids, audio = result
            else:
                text_ids = result
                audio = None
        else:
            logger.warning(
                "Model does not have talker. Generating text response only."
            )
            text_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
            audio = None
    elapsed = time.time() - t0
    logger.info("Generation done in %.1f s", elapsed)

    # Decode and save text response
    input_len = inputs["input_ids"].shape[1]
    if isinstance(text_ids, torch.Tensor):
        new_ids = text_ids[0, input_len:] if text_ids.shape[1] > input_len else text_ids[0]
        response_text = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
        logger.info("Text response: %r", response_text[:200])
        save_text(save_dir, "generated_text.txt", response_text)

    # Save audio response
    if audio is not None:
        audio_np = audio.reshape(-1).detach().cpu().float().numpy()
        import soundfile as sf
        output_wav = save_dir / "output.wav"
        sf.write(str(output_wav), audio_np, samplerate=24000)
        duration = len(audio_np) / 24000
        logger.info("Audio response saved: %.2f s at 24000 Hz -> %s", duration, output_wav)
    else:
        logger.warning("No audio output produced — model may not have talker enabled")

    log_ram()
    logger.info("audio_to_audio complete. Output: %s", save_dir)


def _save_wav(path: Path, audio: np.ndarray, sample_rate: int):
    """Save float32 mono audio as 16-bit WAV."""
    try:
        import soundfile as sf
        sf.write(str(path), audio, sample_rate)
    except ImportError:
        # Fallback: write minimal WAV manually
        import struct
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        data_bytes = audio_int16.tobytes()
        num_samples = len(audio_int16)
        with open(path, "wb") as f:
            # RIFF header
            f.write(b"RIFF")
            f.write(struct.pack("<I", 36 + len(data_bytes)))
            f.write(b"WAVE")
            # fmt chunk
            f.write(b"fmt ")
            f.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
            # data chunk
            f.write(b"data")
            f.write(struct.pack("<I", len(data_bytes)))
            f.write(data_bytes)
    logger.info("WAV saved: %s  (%d samples @ %d Hz)", path, len(audio), sample_rate)


# ---------------------------------------------------------------------------
# Mode: full_reference (complete activation extraction)
# ---------------------------------------------------------------------------

TEXT_PROMPTS = [
    ("text_simple", "Hello"),
    ("text_factual", "The capital of France is"),
    ("text_russian", "Привет, как дела?"),
]

AUDIO_PROMPTS = [
    # (name, audio_url, text) — use official prompt text from HuggingFace model card
    ("audio_asr", None, "Transcribe the audio into text."),
]

TTS_PROMPTS = [
    ("tts_russian", "Привет, мир!"),
]


def run_full_reference(
    model_path: str,
    audio_file: str,
    output_base: Path,
    model=None,
    processor=None,
):
    """
    Full activation extraction across all prompt types.

    Text prompts: use Thinker only (lighter) — unless model is pre-loaded (server mode).
    Audio/TTS prompts: use full model.

    If model and processor are provided (server mode), they are used for all parts
    instead of loading separate thinker / full models.
    """
    logger.info("=" * 70)
    logger.info("MODE: full_reference")
    logger.info("=" * 70)

    server_mode = (model is not None and processor is not None)

    # ----------------------------------------------------------------
    # Part 1: Text prompts
    # In standalone mode: load Thinker (lighter).
    # In server mode: reuse already-loaded full model.
    # ----------------------------------------------------------------
    logger.info("--- Part 1: Text prompts ---")

    if server_mode:
        logger.info("Server mode: using pre-loaded full model for text prompts")
        text_model = model
        text_processor = processor
        _free_text_model = False
    else:
        text_model, text_processor = load_thinker(model_path)
        _free_text_model = True

    # Determine number of thinker layers
    text_cfg = _get_text_config(text_model.config)
    try:
        num_layers = text_cfg.num_hidden_layers
    except AttributeError:
        num_layers = 48  # Qwen3-Omni-30B default
    logger.info("Thinker layers: %d", num_layers)

    for prompt_name, prompt_text in TEXT_PROMPTS:
        logger.info("--- Prompt: %s (%r) ---", prompt_name, prompt_text)
        try:
            _extract_text_activations(
                model=text_model,
                processor=text_processor,
                prompt_name=prompt_name,
                prompt_text=prompt_text,
                num_layers=num_layers,
                save_dir=output_base / f"omni_{prompt_name}",
            )
        except KeyboardInterrupt:
            logger.warning("Interrupted by user — skipping remaining prompts")
            break
        except Exception as exc:
            logger.error("FAILED prompt %s: %s", prompt_name, exc, exc_info=True)

    if _free_text_model:
        # Free thinker model before loading full model
        logger.info("Freeing Thinker model from memory ...")
        del text_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        log_ram()

    # ----------------------------------------------------------------
    # Part 2: Audio prompts via full model
    # ----------------------------------------------------------------
    logger.info("--- Part 2: Audio prompts (Full model) ---")

    if not Path(audio_file).exists():
        logger.warning("Audio file not found: %s — skipping audio/TTS prompts", audio_file)
        return

    audio_array, sample_rate = _load_audio(audio_file)
    if audio_array is None:
        logger.warning("Could not load audio — skipping audio/TTS prompts")
        return

    try:
        if server_mode:
            logger.info("Server mode: using pre-loaded full model for ASR")
            full_model = model
            full_processor = processor
            _free_full_model = False
        else:
            full_model, full_processor = load_full_model(model_path, enable_audio_output=False)
            _free_full_model = True

        _extract_asr_activations(
            model=full_model,
            processor=full_processor,
            audio_array=audio_array,
            sample_rate=sample_rate,
            prompt_name="audio_asr",
            transcription_prompt="Transcribe this audio exactly as spoken.",
            save_dir=output_base / "omni_audio_asr",
        )

        if _free_full_model:
            del full_model
            log_ram()

    except KeyboardInterrupt:
        logger.warning("Interrupted — skipping TTS prompts")
        return
    except Exception as exc:
        logger.error("FAILED audio_asr: %s", exc, exc_info=True)

    # ----------------------------------------------------------------
    # Part 3: TTS prompts via full model with audio output
    # ----------------------------------------------------------------
    logger.info("--- Part 3: TTS prompts (Full model, audio output) ---")

    try:
        if server_mode:
            logger.info("Server mode: using pre-loaded full model for TTS")
            tts_model = model
            tts_processor = processor
            _free_tts_model = False
        else:
            tts_model, tts_processor = load_full_model(model_path, enable_audio_output=True)
            _free_tts_model = True

        for tts_name, tts_text in TTS_PROMPTS:
            logger.info("--- TTS: %s (%r) ---", tts_name, tts_text)
            try:
                _extract_tts_activations(
                    model=tts_model,
                    processor=tts_processor,
                    prompt_name=tts_name,
                    text=tts_text,
                    save_dir=output_base / f"omni_{tts_name}",
                )
            except KeyboardInterrupt:
                logger.warning("Interrupted")
                break
            except Exception as exc:
                logger.error("FAILED TTS %s: %s", tts_name, exc, exc_info=True)

        if _free_tts_model:
            del tts_model
            log_ram()

    except Exception as exc:
        logger.error("Could not load TTS model: %s", exc, exc_info=True)

    logger.info("full_reference complete. Output: %s", output_base)


def _extract_text_activations(
    model,
    processor,
    prompt_name: str,
    prompt_text: str,
    num_layers: int,
    save_dir: Path,
):
    """
    Extract full layer-by-layer activations for a text prompt.

    Registers forward hooks on:
      - embedding layer
      - each transformer block (layers 0..num_layers-1)
      - final RMS norm
    """
    capture = ActivationCapture()

    # Locate embedding and transformer layers.
    # Qwen3OmniMoeThinker wraps a thinker sub-model; try common attribute paths.
    thinker_inner = _get_thinker_inner(model)

    # Embedding
    embed = _find_attr(thinker_inner, ["embed_tokens", "model.embed_tokens"])
    if embed is not None:
        capture.register("embedding_output", embed)
    else:
        logger.warning("Could not locate embedding layer for hooks")

    # Transformer blocks
    layers = _find_attr(thinker_inner, ["layers", "model.layers"])
    if layers is not None and hasattr(layers, "__len__"):
        for i, layer in enumerate(layers):
            capture.register(f"thinker_layer_{i:02d}_output", layer)
        logger.info("Registered hooks on %d transformer layers", len(layers))
    else:
        logger.warning("Could not locate transformer layers for hooks")

    # Final norm
    norm = _find_attr(thinker_inner, ["norm", "model.norm"])
    if norm is not None:
        capture.register("final_norm_output", norm)
    else:
        logger.warning("Could not locate final norm for hooks")

    # Tokenize
    inputs = processor(text=prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    logger.info(
        "Prompt %r -> %d tokens: %s",
        prompt_text, input_ids.shape[1], input_ids[0].tolist(),
    )

    # Forward pass with hidden states
    logger.info("Running forward pass ...")
    t0 = time.time()
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
    fwd_elapsed = time.time() - t0
    logger.info("Forward pass done in %.1f s", fwd_elapsed)

    logits = outputs.logits  # [1, seq, vocab]
    logger.info("logits shape: %s", logits.shape)

    # Hidden states from output (more reliable than hooks for standard models)
    hidden_states = getattr(outputs, "hidden_states", None)

    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Save from output_hidden_states (preferred path) ---
    if hidden_states is not None:
        logger.info("Saving hidden states from output_hidden_states (%d tensors)", len(hidden_states))
        # hidden_states[0] = embedding output
        # hidden_states[1..N] = layer outputs
        save_npy(save_dir, "embedding_output", to_f32_numpy(hidden_states[0]))
        for i, hs in enumerate(hidden_states[1:]):
            save_npy(save_dir, f"thinker_layer_{i:02d}_output", to_f32_numpy(hs))
        # Final norm: apply manually if we have the module
        last_hs = hidden_states[-1]
        if norm is not None:
            with torch.no_grad():
                final_normed = norm(last_hs)
            save_npy(save_dir, "final_norm_output", to_f32_numpy(final_normed))
        else:
            logger.warning("No norm module found; skipping final_norm_output")
    else:
        # Fall back to hook-captured activations
        logger.info(
            "output_hidden_states not returned; using %d hook captures",
            len(capture.activations),
        )
        for name, arr in capture.activations.items():
            save_npy(save_dir, name, arr)

    # Always save input_ids and logits
    save_npy(save_dir, "input_ids", input_ids.numpy().astype(np.int64))
    save_npy(save_dir, "logits", to_f32_numpy(logits))

    # Generate a few tokens
    logger.info("Generating tokens for verification ...")
    t1 = time.time()
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    gen_elapsed = time.time() - t1
    n_new = generated.shape[1] - input_ids.shape[1]
    logger.info(
        "Generated %d new tokens in %.1f s  (%.1f s/token)",
        n_new, gen_elapsed, gen_elapsed / max(n_new, 1),
    )

    generated_ids = generated.numpy().astype(np.int64)
    generated_text = processor.tokenizer.decode(generated[0], skip_special_tokens=True)

    save_npy(save_dir, "generated_ids", generated_ids)
    save_text(save_dir, "generated_text.txt", generated_text)

    capture.remove_hooks()
    capture.clear()
    log_ram()
    logger.info("Done: %s", save_dir)


def _extract_asr_activations(
    model,
    processor,
    audio_array: np.ndarray,
    sample_rate: int,
    prompt_name: str,
    transcription_prompt: str,
    save_dir: Path,
):
    """Extract activations for audio ASR prompt."""

    capture = ActivationCapture()

    # Register hook on audio tower if present
    audio_tower = _find_attr(
        model,
        ["audio_tower", "model.audio_tower", "thinker.audio_tower"],
    )
    if audio_tower is not None:
        capture.register("audio_embeddings", audio_tower)
        logger.info("Registered hook on audio_tower")
    else:
        logger.warning("Could not locate audio_tower for hooks")

    # System message required for reliable ASR (same fix as run_asr_test).
    # Use explicit speech recognition system prompt to avoid safety refusals.
    # audio_url uses a real path here so process_mm_info can load it.
    # We pass audio_array directly since it was pre-loaded by the caller.
    conversations = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a speech recognition assistant. Your only task is to convert audio speech to text. Output only the spoken words, nothing else."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_array},  # numpy array directly
                {"type": "text", "text": transcription_prompt},
            ],
        },
    ]

    # Detect family from model class name (same pattern used in generate block below).
    # Do NOT use enable_thinking=False: it pre-fills <think></think> in the prompt,
    # causing Qwen3-Omni to generate only \n<|im_end|> (empty response).
    # Instead, pre-fill a thinking prefix (confirmed-working approach from v18 server).
    _model_cls = type(model).__name__
    template_kwargs = dict(add_generation_prompt=True, tokenize=False)
    text = processor.apply_chat_template(conversations, **template_kwargs)
    if "Qwen3" in _model_cls or "qwen3" in _model_cls:
        # Pre-fill thinking to prevent degenerate early EOS or safety refusal.
        # IMPORTANT: apply_chat_template ends with "<|im_start|>assistant\n".
        # We must NOT strip the trailing "\n" before appending <think>, because
        # Qwen3 requires "assistant\n<think>" format. Missing "\n" causes the model
        # to generate garbage instead of the transcription.
        text = text.rstrip()  # strips trailing \n (template ends with "assistant\n")
        if not text.endswith("<think>"):
            # Re-add the newline that rstrip() removed, then append think block.
            # Result: "...<|im_start|>assistant\n<think>\n...\n</think>\n\n"
            text += "\n<think>\nThe user wants me to transcribe the audio content into text.\n</think>\n\n"
        logger.info("Qwen3: pre-filling thinking prefix to prevent degenerate early EOS")

    # Resample to 16kHz — the feature extractor requires 16000 Hz
    if sample_rate != 16000:
        try:
            import librosa
            logger.info("Resampling audio from %d to 16000 Hz", sample_rate)
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        except ImportError:
            logger.error("librosa not available for resampling. Install with: pip install librosa")
            return

    from qwen_omni_utils import process_mm_info
    audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
    )
    # Convert float32 tensors to bfloat16 to match model weights.
    # input_features (mel spectrograms) must also be bf16 — the audio encoder
    # conv2d receives them directly and requires matching dtype.
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor) and inputs[key].is_floating_point() and inputs[key].dtype != torch.bfloat16:
            inputs[key] = inputs[key].to(torch.bfloat16)

    logger.info(
        "ASR inputs: %s",
        {k: (v.shape, v.dtype) for k, v in inputs.items() if hasattr(v, "shape")},
    )
    log_ram()

    logger.info("Running ASR forward + generate ...")
    t0 = time.time()
    with torch.no_grad():
        # Detect family from model class name to avoid requiring family parameter here.
        # Qwen3 supports return_audio=False; Qwen2.5 does not have this parameter.
        # For Qwen3: use thinker_max_new_tokens (max_new_tokens is ignored by wrapper).
        # Allow thinking (512) + answer (50) tokens.
        model_cls_name = type(model).__name__
        if "Qwen3" in model_cls_name or "qwen3" in model_cls_name:
            result = model.generate(
                **inputs,
                thinker_max_new_tokens=562,
                thinker_do_sample=False,
                return_audio=False,
            )
            # model.generate() may return a tuple (text_ids, ...) for Qwen3-Omni
            if isinstance(result, tuple):
                generated = result[0]
                if hasattr(generated, "sequences"):
                    generated = generated.sequences
            elif hasattr(result, "sequences"):
                generated = result.sequences
            else:
                generated = result
        else:
            result = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
            )
            # generate may return (text_ids, audio) tuple if talker is active
            if isinstance(result, tuple):
                generated = result[0]
            else:
                generated = result
    elapsed = time.time() - t0

    input_len = inputs["input_ids"].shape[1]
    n_new = generated.shape[1] - input_len
    logger.info(
        "ASR generated %d tokens in %.1f s  (%.1f s/token)",
        n_new, elapsed, elapsed / max(n_new, 1),
    )

    raw_decoded = processor.tokenizer.decode(generated[0, input_len:], skip_special_tokens=True)
    # Strip thinking block if present (Qwen3 with thinking enabled)
    if "<think>" in raw_decoded and "</think>" in raw_decoded:
        think_end = raw_decoded.find("</think>")
        transcribed = raw_decoded[think_end + len("</think>"):].strip()
    else:
        transcribed = raw_decoded.strip()
    logger.info("Transcription: %r", transcribed)

    save_dir.mkdir(parents=True, exist_ok=True)

    # Save hook captures (audio embeddings, any tower activations)
    for name, arr in capture.activations.items():
        save_npy(save_dir, name, arr)

    save_npy(save_dir, "input_ids", inputs["input_ids"].numpy().astype(np.int64))
    save_npy(save_dir, "generated_ids", generated.numpy().astype(np.int64))
    save_text(save_dir, "generated_text.txt", transcribed)

    capture.remove_hooks()
    capture.clear()
    log_ram()
    logger.info("Done: %s", save_dir)


def _extract_tts_activations(
    model,
    processor,
    prompt_name: str,
    text: str,
    save_dir: Path,
):
    """Extract activations for TTS prompt, save codec codes and audio."""

    capture = ActivationCapture()

    # Attempt to hook talker layers if present
    talker = _find_attr(model, ["talker", "model.talker"])
    if talker is not None:
        talker_layers = _find_attr(talker, ["layers", "model.layers"])
        if talker_layers is not None and hasattr(talker_layers, "__len__"):
            for i, layer in enumerate(talker_layers):
                capture.register(f"talker_layer_{i:02d}_output", layer)
            logger.info("Registered hooks on %d talker layers", len(talker_layers))
    else:
        logger.warning("Could not locate talker for hooks")

    conversations = [
        {
            "role": "user",
            "content": [{"type": "text", "text": text}],
        }
    ]

    # Qwen3-Omni has thinking mode enabled by default; disable it to avoid
    # safety refusals during reference data collection.
    _model_cls = type(model).__name__
    template_kwargs = dict(add_generation_prompt=True, tokenize=False)
    if "Qwen3" in _model_cls or "qwen3" in _model_cls:
        template_kwargs["enable_thinking"] = False
    tpl = processor.apply_chat_template(conversations, **template_kwargs)
    inputs = processor(text=tpl, return_tensors="pt")

    logger.info("Generating TTS for: %r", text)
    log_ram()
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
            return_dict_in_generate=True,
        )
    elapsed = time.time() - t0
    logger.info("TTS generate done in %.1f s", elapsed)

    save_dir.mkdir(parents=True, exist_ok=True)

    # Save talker layer activations from hooks
    for name, arr in capture.activations.items():
        save_npy(save_dir, name, arr)

    # Extract codec codes if available
    codec_codes = None
    if hasattr(outputs, "codec_codes"):
        codec_codes = outputs.codec_codes
    elif hasattr(outputs, "sequences") and hasattr(outputs, "audio_tokens"):
        codec_codes = outputs.audio_tokens

    if codec_codes is not None:
        save_npy(save_dir, "codec_codes", to_f32_numpy(codec_codes))
    else:
        logger.info("No codec_codes attribute found in generate output")

    # Extract waveform if available
    waveform = None
    audio_cfg = _get_audio_config(model.config)
    sample_rate_out = getattr(audio_cfg, "audio_sample_rate", 24000)
    if hasattr(outputs, "waveform"):
        waveform = outputs.waveform
    elif hasattr(outputs, "audio"):
        waveform = outputs.audio

    if waveform is not None:
        audio_np = to_f32_numpy(waveform.squeeze())
        duration = len(audio_np) / sample_rate_out
        logger.info("Audio waveform: %.2f s at %d Hz", duration, sample_rate_out)
        _save_wav(save_dir / "generated_audio.wav", audio_np, sample_rate_out)
    else:
        # Decode token IDs as text (TTS may output text tokens in some configurations)
        if hasattr(outputs, "sequences"):
            seq = outputs.sequences
        else:
            seq = outputs if isinstance(outputs, torch.Tensor) else None

        if seq is not None:
            decoded = processor.tokenizer.decode(seq[0], skip_special_tokens=True)
            save_text(save_dir, "generated_text.txt", decoded)
            logger.info("No waveform — saved generated_text.txt instead")

    capture.remove_hooks()
    capture.clear()
    log_ram()
    logger.info("Done: %s", save_dir)


# ---------------------------------------------------------------------------
# Helper: traverse model attributes safely
# ---------------------------------------------------------------------------

def _get_thinker_inner(model):
    """
    Navigate into the thinker sub-model for Qwen3-Omni wrappers.
    Tries common attribute paths.
    """
    for attr_path in ["thinker", "model"]:
        inner = _find_attr(model, [attr_path])
        if inner is not None:
            return inner
    return model


def _find_attr(obj, attr_paths: List[str]):
    """
    Try multiple dotted attribute paths on obj; return first that exists.

    E.g. _find_attr(model, ["model.embed_tokens", "embed_tokens"])
    """
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
# Audio loading helper
# ---------------------------------------------------------------------------

def _load_audio(path: str) -> Tuple[Optional[np.ndarray], int]:
    """Load mono audio. Returns (array, sample_rate) or (None, 0)."""
    try:
        import soundfile as sf
        arr, sr = sf.read(path, always_2d=False)
        if arr.ndim == 2:
            arr = arr.mean(axis=1)  # stereo -> mono
        logger.info("Audio: %.2f s at %d Hz via soundfile", len(arr) / sr, sr)
        return arr.astype(np.float32), sr
    except ImportError:
        pass

    try:
        import librosa
        arr, sr = librosa.load(path, sr=None, mono=True)
        logger.info("Audio: %.2f s at %d Hz via librosa", len(arr) / sr, sr)
        return arr.astype(np.float32), sr
    except ImportError:
        pass

    try:
        import wave
        import struct as _struct
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            raw = wf.readframes(n_frames)
        if sampwidth == 2:
            arr_int = np.frombuffer(raw, dtype=np.int16).reshape(-1, n_channels)
            arr = arr_int.mean(axis=1).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            arr_int = np.frombuffer(raw, dtype=np.int32).reshape(-1, n_channels)
            arr = arr_int.mean(axis=1).astype(np.float32) / 2147483648.0
        else:
            logger.error("Unsupported WAV sample width: %d", sampwidth)
            return None, 0
        logger.info("Audio: %.2f s at %d Hz via wave stdlib", len(arr) / sr, sr)
        return arr, sr
    except Exception as exc:
        logger.error("Failed to load audio via stdlib wave: %s", exc)

    return None, 0


# ---------------------------------------------------------------------------
# Mode: server (HTTP server with pre-loaded model)
# ---------------------------------------------------------------------------

DEFAULT_SERVER_PORT = 8899


class _OmniRequestHandler(http.server.BaseHTTPRequestHandler):
    """
    Simple synchronous HTTP request handler for the Omni server mode.

    Supported endpoints:
        POST /run      — run a named test using the pre-loaded model
        POST /status   — return memory usage and model info
        POST /shutdown — signal the server to stop

    All requests and responses use JSON bodies.
    """

    # Set by run_server before the server is started
    server_state: dict = {}

    def log_message(self, fmt, *args):
        # Route access log through our logger instead of stderr
        logger.info("HTTP %s", fmt % args)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length > 0:
            raw = self.rfile.read(length)
            try:
                return json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                logger.warning("Bad JSON in request body: %s", exc)
        return {}

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        t_start = time.time()
        path = self.path.rstrip("/")
        logger.info("POST %s  from %s", path, self.client_address)

        if path == "/run":
            self._handle_run()
        elif path == "/status":
            self._handle_status()
        elif path == "/shutdown":
            self._handle_shutdown()
        else:
            self._send_json(404, {"error": f"Unknown endpoint: {path}"})

        elapsed = time.time() - t_start
        logger.info("POST %s handled in %.1f s", path, elapsed)

    # ------------------------------------------------------------------
    # /run
    # ------------------------------------------------------------------

    def _handle_run(self):
        body = self._read_json_body()
        test_name = body.get("test")
        params = body.get("params", {})

        valid_tests = ["text_test", "asr_test", "tts_test", "full_reference", "audio_to_audio"]
        if test_name not in valid_tests:
            self._send_json(400, {
                "error": f"Unknown test {test_name!r}. Valid: {valid_tests}",
            })
            return

        state = self.__class__.server_state
        model = state["model"]
        processor = state["processor"]
        model_path = state["model_path"]
        output_base = state["output_base"]
        audio_file = params.get("audio_file", state["audio_file"])
        family = state.get("family")

        logger.info("Running test %r with pre-loaded model (family=%s) ...", test_name, family)
        if params.get("audio_file"):
            logger.info("Using audio_file from request params: %s", audio_file)
        t0 = time.time()
        error = None
        output_dir = None

        try:
            if test_name == "text_test":
                save_dir = output_base / "omni_text_test"
                run_text_test(
                    model_path=model_path,
                    output_base=output_base,
                    model=model,
                    processor=processor,
                    family=family,
                )
                output_dir = str(save_dir)

            elif test_name == "asr_test":
                save_dir = output_base / "omni_asr_test"
                run_asr_test(
                    model_path=model_path,
                    audio_file=audio_file,
                    output_base=output_base,
                    model=model,
                    processor=processor,
                    family=family,
                )
                output_dir = str(save_dir)

            elif test_name == "tts_test":
                save_dir = output_base / "omni_tts_test"
                run_tts_test(
                    model_path=model_path,
                    output_base=output_base,
                    model=model,
                    processor=processor,
                    family=family,
                )
                output_dir = str(save_dir)

            elif test_name == "full_reference":
                run_full_reference(
                    model_path=model_path,
                    audio_file=audio_file,
                    output_base=output_base,
                    model=model,
                    processor=processor,
                )
                output_dir = str(output_base)

            elif test_name == "audio_to_audio":
                save_dir = output_base / "omni_audio_to_audio"
                run_audio_to_audio_test(
                    model_path=model_path,
                    output_base=output_base,
                    audio_file=audio_file,
                    model=model,
                    processor=processor,
                    family=family,
                )
                output_dir = str(save_dir)

        except Exception as exc:
            logger.error("Test %r failed: %s", test_name, exc, exc_info=True)
            error = str(exc)

        elapsed = time.time() - t0

        if error:
            self._send_json(500, {
                "status": "error",
                "test": test_name,
                "error": error,
                "elapsed_s": round(elapsed, 2),
            })
        else:
            self._send_json(200, {
                "status": "ok",
                "test": test_name,
                "output_dir": output_dir,
                "elapsed_s": round(elapsed, 2),
            })

    # ------------------------------------------------------------------
    # /status
    # ------------------------------------------------------------------

    def _handle_status(self):
        state = self.__class__.server_state
        info: dict = {
            "model_loaded": state.get("model") is not None,
            "model_path": state.get("model_path"),
            "model_family": state.get("family"),
            "model_class": type(state["model"]).__name__ if state.get("model") else None,
        }
        try:
            import psutil
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            info["ram_used_gb"] = round(mem.used / 1e9, 2)
            info["ram_total_gb"] = round(mem.total / 1e9, 2)
            info["ram_percent"] = round(mem.percent, 1)
            info["swap_used_gb"] = round(swap.used / 1e9, 2)
            info["swap_total_gb"] = round(swap.total / 1e9, 2)
        except ImportError:
            info["ram_info"] = "psutil not available"
        self._send_json(200, info)

    # ------------------------------------------------------------------
    # /shutdown
    # ------------------------------------------------------------------

    def _handle_shutdown(self):
        logger.info("Shutdown requested via /shutdown endpoint")
        self._send_json(200, {"status": "shutting_down"})
        # Schedule server shutdown in a background thread so the response
        # is sent before the socket closes.
        state = self.__class__.server_state
        server_ref = state.get("_server_ref")
        if server_ref is not None:
            def _do_shutdown():
                time.sleep(0.2)
                server_ref.shutdown()
            threading.Thread(target=_do_shutdown, daemon=True).start()


def run_server(
    model_path: str,
    output_base: Path,
    audio_file: str,
    port: int = DEFAULT_SERVER_PORT,
):
    """
    Load the FULL Omni model once, then serve HTTP requests on `port`.

    Model family is auto-detected from config.json.

    Endpoints:
        POST /run      { "test": "text_test"|"asr_test"|"tts_test"|"full_reference",
                         "params": {} }
        POST /status   {}
        POST /shutdown {}

    The server handles one request at a time (no concurrency). This is
    intentional — inference is CPU-bound and sequential is correct here.
    """
    logger.info("=" * 70)
    logger.info("MODE: server  (port %d)", port)
    logger.info("=" * 70)

    family = detect_model_family(model_path)
    logger.info("Detected model family: %s", family)
    logger.info("Loading FULL model before accepting connections ...")

    model, processor = load_full_model(model_path, enable_audio_output=True)

    logger.info("Model loaded. Starting HTTP server on port %d ...", port)
    log_ram()

    # Share state with the request handler via class attribute
    _OmniRequestHandler.server_state = {
        "model": model,
        "processor": processor,
        "model_path": model_path,
        "output_base": output_base,
        "audio_file": audio_file,
        "family": family,
        "_server_ref": None,  # filled in below
    }

    server = http.server.HTTPServer(("0.0.0.0", port), _OmniRequestHandler)
    _OmniRequestHandler.server_state["_server_ref"] = server

    logger.info("Omni server listening on http://0.0.0.0:%d", port)
    logger.info("Endpoints: POST /run  POST /status  POST /shutdown")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server interrupted by user (Ctrl-C)")
    finally:
        server.server_close()
        logger.info("Server stopped.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reference activations from Qwen3-Omni-30B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["text_test", "asr_test", "tts_test", "full_reference", "audio_to_audio", "server"],
        required=True,
        help="Which mode to run",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to Qwen3-Omni-30B directory (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_REFERENCE_DIR,
        help=f"Root output directory for reference data (default: {DEFAULT_REFERENCE_DIR})",
    )
    parser.add_argument(
        "--audio-file",
        default=DEFAULT_AUDIO_FILE,
        help=f"Path to audio file for ASR modes (default: {DEFAULT_AUDIO_FILE})",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=DEFAULT_SERVER_PORT,
        help=f"TCP port for server mode (default: {DEFAULT_SERVER_PORT})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = args.model_path
    output_base = Path(args.output_dir)
    audio_file = args.audio_file

    logger.info("Qwen3-Omni Reference Extractor")
    logger.info("  mode       : %s", args.mode)
    logger.info("  model_path : %s", model_path)
    logger.info("  output_dir : %s", output_base)
    logger.info("  audio_file : %s", audio_file)
    if args.mode == "server":
        logger.info("  server_port: %d", args.server_port)
    log_ram()

    if not Path(model_path).exists():
        logger.error("Model path does not exist: %s", model_path)
        sys.exit(1)

    try:
        if args.mode == "text_test":
            run_text_test(model_path, output_base)

        elif args.mode == "asr_test":
            run_asr_test(model_path, audio_file, output_base)

        elif args.mode == "tts_test":
            run_tts_test(model_path, output_base)

        elif args.mode == "full_reference":
            run_full_reference(model_path, audio_file, output_base)

        elif args.mode == "audio_to_audio":
            run_audio_to_audio_test(model_path, output_base, audio_file)

        elif args.mode == "server":
            run_server(
                model_path=model_path,
                output_base=output_base,
                audio_file=audio_file,
                port=args.server_port,
            )

    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl-C). Partial results may exist in %s", output_base)
        sys.exit(130)
    except Exception as exc:
        logger.error("Mode %r failed: %s", args.mode, exc, exc_info=True)
        sys.exit(1)

    logger.info("Done.")


if __name__ == "__main__":
    main()
