# Reference Data Extraction Scripts

## Overview

This directory contains Python scripts for extracting reference activations from HuggingFace Transformers models. These reference tensors are used to validate the Rust inference implementation.

## Prerequisites

1. Python 3.8 or later
2. Qwen3-0.6B model downloaded locally to `models/Qwen3-0.6B/`
3. Required Python packages (see Installation)

## Installation

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r scripts/requirements.txt
```

## Usage

### Extract Reference Data

Run the extraction script to generate reference activations for validation:

```bash
python scripts/extract_reference.py
```

This will create `reference_data/` directory with three test cases:
- `prompt1/` - Single token: "Hello"
- `prompt2/` - Multi-token: "The capital of France is"
- `prompt3/` - Question: "What is 2+2?"

Each subdirectory contains a `reference.npz` file with all intermediate activations.

### Command Line Options

```bash
# Custom model path
python scripts/extract_reference.py --model-path /path/to/model

# Custom output directory
python scripts/extract_reference.py --output-dir my_reference_data

# Generate more tokens
python scripts/extract_reference.py --max-new-tokens 50
```

## Output Format

Each `reference.npz` file contains the following NumPy arrays:

| Key | Shape | DType | Description |
|-----|-------|-------|-------------|
| `input_ids` | [1, L] | int64 | Tokenized input |
| `embedding_output` | [1, L, 1024] | float32 | Token embeddings |
| `layer_00_output` | [1, L, 1024] | float32 | After transformer layer 0 |
| `layer_01_output` | [1, L, 1024] | float32 | After transformer layer 1 |
| ... | ... | ... | ... |
| `layer_27_output` | [1, L, 1024] | float32 | After transformer layer 27 |
| `final_norm_output` | [1, L, 1024] | float32 | After final RMSNorm |
| `logits` | [1, L, 151936] | float32 | LM head output (full vocab) |
| `generated_ids` | [1, L+20] | int64 | Greedy generation output |
| `rope_cos` | [40960, 128] | float32 | RoPE cosine table |
| `rope_sin` | [40960, 128] | float32 | RoPE sine table |

Where:
- L = sequence length (varies by prompt)
- 1024 = hidden_size
- 151936 = vocab_size
- 40960 = max_position_embeddings
- 128 = head_dim

## Loading Reference Data (Python)

```python
import numpy as np

# Load reference data
data = np.load("reference_data/prompt1/reference.npz")

# Access individual arrays
input_ids = data["input_ids"]
logits = data["logits"]
layer_0 = data["layer_00_output"]

# List all available keys
print(list(data.keys()))
```

## Loading Reference Data (Rust)

See `examples/validate.rs` for Rust code to load and compare against these references.

```rust
use ndarray_npy::NpzReader;

let mut npz = NpzReader::new(File::open("reference_data/prompt1/reference.npz")?)?;
let logits: Array3<f32> = npz.by_name("logits.npy")?;
```

## Validation Strategy

1. **Per-layer validation**: Compare Rust intermediate activations against each `layer_XX_output`
2. **Metrics**: MSE (mean squared error), cosine similarity, max absolute difference
3. **Tolerance**: MSE < 1e-5 for BF16 precision
4. **Token-level validation**: Greedy generation must match `generated_ids` exactly

## Offline Operation

The script runs entirely offline using locally cached models:
- Set `local_files_only=True` in all HuggingFace calls
- No network access required after initial model download
- Reproducible outputs (greedy decoding is deterministic)

## Troubleshooting

### Model Not Found

```
Error: models/Qwen3-0.6B not found
```

Download the model first:
```bash
cd models
git clone https://huggingface.co/Qwen/Qwen3-0.6B
```

### Out of Memory

For large models, reduce batch size or use CPU-only inference:
```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # Use FP16 instead of BF16
    device_map="cpu",
    low_cpu_mem_usage=True
)
```

### Shape Mismatches

Verify model architecture matches expectations:
```python
print(model.config)  # Check num_hidden_layers, hidden_size, etc.
```

## Notes

- **BF16 Precision**: Model runs in BF16 (matching production), but activations are saved as FP32 for validation
- **Deterministic**: Greedy decoding ensures reproducible outputs across runs
- **No Attention Weights**: Attention matrices are not saved (too large), only layer outputs
- **RoPE Tables**: Pre-computed for full `max_position_embeddings` range (40960 positions)
