#!/usr/bin/env python3
"""
Extract reference tensors from Qwen3-0.6B using HuggingFace Transformers.

Saves intermediate activations for validation of Rust implementation.
This script runs offline with locally cached models.

Usage:
    python scripts/extract_reference.py

Output:
    reference_data/prompt1/*.npz - NumPy compressed arrays with all activations
    reference_data/prompt2/*.npz
    reference_data/prompt3/*.npz
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_model(model_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer from local path."""
    print(f"Loading model from {model_path}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )

    # Load model with BF16 precision (matching production deployment)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        local_files_only=True
    )
    model.eval()

    print(f"Model loaded: {model.config.num_hidden_layers} layers, "
          f"vocab_size={model.config.vocab_size}")

    return model, tokenizer


def extract_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 20
) -> Dict[str, np.ndarray]:
    """
    Run forward pass and extract all intermediate activations.

    Returns:
        Dictionary with keys:
            - input_ids: [1, L] int64
            - embedding_output: [1, L, hidden_size] float32
            - layer_{i}_output: [1, L, hidden_size] float32 for i in 0..27
            - final_norm_output: [1, L, hidden_size] float32
            - logits: [1, L, vocab_size] float32
            - generated_ids: [1, L+max_new_tokens] int64
            - rope_cos: [max_position_embeddings, head_dim] float32
            - rope_sin: [max_position_embeddings, head_dim] float32
    """
    activations = {}

    # Tokenize input
    print(f"Tokenizing prompt: {prompt!r}")
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    activations["input_ids"] = input_ids.numpy().astype(np.int64)

    print(f"Input IDs shape: {input_ids.shape}")

    # Forward pass with output_hidden_states=True
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            output_attentions=False,  # Skip attention weights (too large)
            return_dict=True
        )

    # Extract logits (full sequence, not just last token)
    logits = outputs.logits.to(torch.float32).numpy()
    activations["logits"] = logits
    print(f"Logits shape: {logits.shape}")

    # Extract hidden states
    # hidden_states is tuple of (embedding_output, layer_0, ..., layer_27)
    # Total length: num_layers + 1 (embedding layer)
    hidden_states = outputs.hidden_states
    print(f"Hidden states: {len(hidden_states)} layers")

    # Embedding output (before first transformer block)
    embedding_output = hidden_states[0].to(torch.float32).numpy()
    activations["embedding_output"] = embedding_output
    print(f"Embedding output shape: {embedding_output.shape}")

    # Per-layer outputs (after each transformer block)
    for i, layer_output in enumerate(hidden_states[1:]):
        layer_name = f"layer_{i:02d}_output"
        layer_array = layer_output.to(torch.float32).numpy()
        activations[layer_name] = layer_array
        print(f"{layer_name} shape: {layer_array.shape}")

    # Final norm output (apply RMS norm to last hidden state)
    # hidden_states[-1] is the output of the last transformer layer
    # We need to apply model.model.norm to get the actual final_norm_output
    with torch.no_grad():
        final_norm_output = model.model.norm(hidden_states[-1]).to(torch.float32).numpy()
    activations["final_norm_output"] = final_norm_output
    print(f"final_norm_output shape: {final_norm_output.shape}")

    # Extract RoPE cos/sin tables
    # Compute manually using config parameters to match Rust implementation
    print("Computing RoPE cos/sin tables...")
    max_seq_len = model.config.max_position_embeddings
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    rope_theta = getattr(model.config, 'rope_theta', 1000000.0)

    print(f"RoPE config: max_seq_len={max_seq_len}, head_dim={head_dim}, theta={rope_theta}")

    # Compute inverse frequencies: inv_freq[i] = 1.0 / (theta^(2*i / head_dim))
    # Shape: [head_dim // 2]
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))

    # Create position indices: [0, 1, 2, ..., max_seq_len-1]
    # Shape: [max_seq_len]
    positions = torch.arange(max_seq_len, dtype=torch.float32)

    # Compute outer product: positions[:, None] * inv_freq[None, :]
    # Shape: [max_seq_len, head_dim // 2]
    freqs = torch.outer(positions, inv_freq)

    # Compute cos and sin tables
    # Shape: [max_seq_len, head_dim // 2]
    cos_half = torch.cos(freqs)
    sin_half = torch.sin(freqs)

    # Interleave to get full head_dim: [cos0, cos0, cos1, cos1, ...]
    # This matches the RoPE application pattern where pairs of dimensions share the same angle
    # Shape: [max_seq_len, head_dim]
    rope_cos = torch.stack([cos_half, cos_half], dim=-1).reshape(max_seq_len, head_dim)
    rope_sin = torch.stack([sin_half, sin_half], dim=-1).reshape(max_seq_len, head_dim)

    # Convert to numpy
    activations["rope_cos"] = rope_cos.numpy()
    activations["rope_sin"] = rope_sin.numpy()
    print(f"RoPE cos shape: {rope_cos.shape}, sin shape: {rope_sin.shape}")

    # Greedy generation (deterministic)
    print(f"Generating {max_new_tokens} tokens (greedy)...")
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=None,  # Ignored when do_sample=False
        )

    generated_ids = generated.numpy().astype(np.int64)
    activations["generated_ids"] = generated_ids
    print(f"Generated IDs shape: {generated_ids.shape}")

    # Decode generated text for verification
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text!r}")

    return activations


def save_activations(activations: Dict[str, np.ndarray], output_dir: Path):
    """Save all activations as individual .npy files (not .npz).

    This matches the Rust validation code expectations:
    - input_ids.npy
    - embedding_output.npy
    - layer_00_output.npy
    - ...
    - layer_27_output.npy
    - final_norm_output.npy
    - logits.npy
    - generated_ids.npy
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving to {output_dir}...")
    total_size = 0

    for key, array in activations.items():
        output_file = output_dir / f"{key}.npy"
        np.save(output_file, array)
        file_size = output_file.stat().st_size
        total_size += file_size
        print(f"  {key}.npy: {array.shape} ({file_size / 1024:.1f} KB)")

    # Print total size
    size_mb = total_size / (1024 * 1024)
    print(f"Saved {len(activations)} arrays ({size_mb:.2f} MB total)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract reference activations from Qwen3-0.6B"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/Qwen3-0.6B",
        help="Path to model directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reference_data",
        help="Output directory for reference data"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20,
        help="Number of tokens to generate"
    )

    args = parser.parse_args()

    # Test prompts (deterministic, diverse)
    test_prompts = [
        ("prompt1", "Hello"),  # Single token
        ("prompt2", "The capital of France is"),  # Multi-token factual
        ("prompt3", "What is 2+2?"),  # Question
    ]

    # Load model once
    model, tokenizer = setup_model(args.model_path)

    # Process each prompt
    output_base = Path(args.output_dir)

    for prompt_name, prompt_text in test_prompts:
        print(f"\n{'='*80}")
        print(f"Processing {prompt_name}: {prompt_text!r}")
        print('='*80)

        # Extract activations
        activations = extract_activations(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_text,
            max_new_tokens=args.max_new_tokens
        )

        # Save to disk
        output_dir = output_base / prompt_name
        save_activations(activations, output_dir)

        print(f"\n✓ {prompt_name} complete")

    print(f"\n{'='*80}")
    print("All reference data extracted successfully!")
    print(f"Output directory: {output_base.absolute()}")
    print('='*80)


if __name__ == "__main__":
    main()
