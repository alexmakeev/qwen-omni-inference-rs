#!/usr/bin/env python3
"""Compare BF16 and Q8_0 reference activations for quality assessment.

Usage:
    python scripts/compare_q8_quality.py --bf16-dir reference_data/omni_text_simple --q8-dir reference_data/omni_text_simple_q8
    python scripts/compare_q8_quality.py --bf16-dir reference_data/omni_text_factual --q8-dir reference_data/omni_text_factual_q8
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path


def cosine_similarity(a, b):
    """Compute cosine similarity between two flat arrays."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 1.0 if norm_a < 1e-10 and norm_b < 1e-10 else 0.0
    return dot / (norm_a * norm_b)


def compare_activations(bf16_dir, q8_dir):
    """Compare all matching .npy files between BF16 and Q8 directories."""
    bf16_path = Path(bf16_dir)
    q8_path = Path(q8_dir)

    if not bf16_path.exists():
        print(f"ERROR: BF16 directory not found: {bf16_dir}")
        sys.exit(1)
    if not q8_path.exists():
        print(f"ERROR: Q8 directory not found: {q8_dir}")
        sys.exit(1)

    bf16_files = sorted(bf16_path.glob("*.npy"))
    q8_files = sorted(q8_path.glob("*.npy"))

    bf16_names = {f.name for f in bf16_files}
    q8_names = {f.name for f in q8_files}

    common = sorted(bf16_names & q8_names)
    bf16_only = sorted(bf16_names - q8_names)
    q8_only = sorted(q8_names - bf16_names)

    if bf16_only:
        print(f"\nWARN: Files only in BF16: {bf16_only}")
    if q8_only:
        print(f"\nWARN: Files only in Q8: {q8_only}")

    print(f"\n{'='*80}")
    print(f"Comparing {len(common)} activation files")
    print(f"BF16: {bf16_dir}")
    print(f"Q8:   {q8_dir}")
    print(f"{'='*80}\n")

    results = []

    print(f"{'Name':<40} {'Shape':<20} {'MSE':>12} {'Cosine':>10} {'MaxDiff':>12} {'Status':>8}")
    print("-" * 102)

    for name in common:
        bf16_data = np.load(bf16_path / name)
        q8_data = np.load(q8_path / name)

        if bf16_data.shape != q8_data.shape:
            print(f"{name:<40} SHAPE MISMATCH: {bf16_data.shape} vs {q8_data.shape}")
            continue

        bf16_f64 = bf16_data.astype(np.float64)
        q8_f64 = q8_data.astype(np.float64)

        mse = np.mean((bf16_f64 - q8_f64) ** 2)
        cos = cosine_similarity(bf16_data, q8_data)
        max_diff = np.max(np.abs(bf16_f64 - q8_f64))

        # Thresholds from plan: MSE < 1e-3, cosine > 0.995
        status = "OK" if mse < 1e-3 and cos > 0.995 else "WARN" if cos > 0.99 else "FAIL"

        results.append({
            'name': name,
            'shape': str(bf16_data.shape),
            'mse': mse,
            'cosine': cos,
            'max_diff': max_diff,
            'status': status,
        })

        print(f"{name:<40} {str(bf16_data.shape):<20} {mse:>12.4e} {cos:>10.6f} {max_diff:>12.4e} {status:>8}")

    # Summary
    if results:
        mses = [r['mse'] for r in results]
        cosines = [r['cosine'] for r in results]
        fails = sum(1 for r in results if r['status'] == 'FAIL')
        warns = sum(1 for r in results if r['status'] == 'WARN')

        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"Total files compared: {len(results)}")
        print(f"MSE   — min: {min(mses):.4e}, max: {max(mses):.4e}, mean: {np.mean(mses):.4e}")
        print(f"Cosine — min: {min(cosines):.6f}, max: {max(cosines):.6f}, mean: {np.mean(cosines):.6f}")
        print(f"Status — OK: {len(results) - fails - warns}, WARN: {warns}, FAIL: {fails}")

        if fails > 0:
            print(f"\nFAILED layers:")
            for r in results:
                if r['status'] == 'FAIL':
                    print(f"  {r['name']}: MSE={r['mse']:.4e}, cos={r['cosine']:.6f}")
            sys.exit(1)

    # Check logits/tokens specially
    bf16_logits = bf16_path / "logits.npy"
    q8_logits = q8_path / "logits.npy"
    if bf16_logits.exists() and q8_logits.exists():
        bf16_l = np.load(bf16_logits)
        q8_l = np.load(q8_logits)
        bf16_token = np.argmax(bf16_l.reshape(-1, bf16_l.shape[-1])[-1])
        q8_token = np.argmax(q8_l.reshape(-1, q8_l.shape[-1])[-1])
        token_match = bf16_token == q8_token
        print(f"\nTop-1 token: BF16={bf16_token}, Q8={q8_token} — {'MATCH' if token_match else 'MISMATCH'}")

    # Check generated text
    bf16_text = bf16_path / "generated_text.txt"
    q8_text = q8_path / "generated_text.txt"
    if bf16_text.exists() and q8_text.exists():
        bf16_t = bf16_text.read_text().strip()
        q8_t = q8_text.read_text().strip()
        text_match = bf16_t == q8_t
        print(f"\nGenerated text:")
        print(f"  BF16: {bf16_t[:100]}...")
        print(f"  Q8:   {q8_t[:100]}...")
        print(f"  {'MATCH' if text_match else 'DIFFERENT'}")


def main():
    parser = argparse.ArgumentParser(description='Compare BF16 and Q8_0 reference activations')
    parser.add_argument('--bf16-dir', required=True, help='Directory with BF16 reference .npy files')
    parser.add_argument('--q8-dir', required=True, help='Directory with Q8_0 reference .npy files')
    args = parser.parse_args()

    compare_activations(args.bf16_dir, args.q8_dir)


if __name__ == '__main__':
    main()
