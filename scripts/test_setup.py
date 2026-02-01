#!/usr/bin/env python3
"""
Test script to verify Python environment and dependencies.

Checks:
- Required packages are installed
- Model files are accessible
- Basic tokenization works

Usage:
    python scripts/test_setup.py
"""

import sys
from pathlib import Path


def check_imports():
    """Check if required packages are importable."""
    print("Checking imports...")

    required = {
        "torch": "PyTorch",
        "transformers": "HuggingFace Transformers",
        "numpy": "NumPy",
    }

    missing = []

    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            missing.append(module)

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r scripts/requirements.txt")
        return False

    return True


def check_model_files(model_path: str):
    """Check if model files exist."""
    print(f"\nChecking model files at {model_path}...")

    path = Path(model_path)

    if not path.exists():
        print(f"  ✗ Model directory not found: {path}")
        return False

    required_files = [
        "config.json",
        "tokenizer.json",
        "model.safetensors",
    ]

    all_found = True

    for file_name in required_files:
        file_path = path / file_name
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {file_name} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {file_name} - NOT FOUND")
            all_found = False

    return all_found


def test_tokenizer(model_path: str):
    """Test basic tokenizer functionality."""
    print(f"\nTesting tokenizer...")

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )

        # Test encoding
        test_text = "Hello, world!"
        tokens = tokenizer.encode(test_text)

        print(f"  Input: {test_text!r}")
        print(f"  Tokens: {tokens}")
        print(f"  Token count: {len(tokens)}")

        # Test decoding
        decoded = tokenizer.decode(tokens)
        print(f"  Decoded: {decoded!r}")

        # Check special tokens
        print(f"  BOS token ID: {tokenizer.bos_token_id}")
        print(f"  EOS token ID: {tokenizer.eos_token_id}")

        print("  ✓ Tokenizer working")
        return True

    except Exception as e:
        print(f"  ✗ Tokenizer failed: {e}")
        return False


def main():
    print("="*80)
    print("Reference Extraction Setup Test")
    print("="*80)

    # Default model path
    model_path = "models/Qwen3-0.6B"

    # Run checks
    checks = [
        ("Imports", check_imports),
        ("Model files", lambda: check_model_files(model_path)),
        ("Tokenizer", lambda: test_tokenizer(model_path)),
    ]

    results = []

    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"\n✗ {check_name} failed with error: {e}")
            results.append((check_name, False))

    # Summary
    print("\n" + "="*80)
    print("Summary:")

    all_passed = True
    for check_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check_name}")
        all_passed = all_passed and passed

    print("="*80)

    if all_passed:
        print("\n✓ All checks passed! Ready to extract reference data.")
        print("\nRun: python scripts/extract_reference.py")
        return 0
    else:
        print("\n✗ Some checks failed. Fix issues before extracting reference data.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
