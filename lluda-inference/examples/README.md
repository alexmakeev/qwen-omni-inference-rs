# Examples

Standalone examples demonstrating lluda-inference functionality.

## `validate.rs` - Validation Tool (T22)

Standalone validation tool for comparing Rust model outputs against Python reference data.

### Purpose

This tool validates the Rust implementation by:
1. Loading reference activations from Python (HuggingFace Transformers)
2. Running the Rust model with the same input
3. Comparing intermediate activations layer-by-layer
4. Reporting validation metrics and pass/fail status

### Usage

#### Prerequisites

1. Generate reference data using the T21 Python script:
   ```bash
   python scripts/extract_reference.py
   ```

   This creates `reference_data/` with subdirectories for each test prompt.

2. Build the validation tool:
   ```bash
   cargo build --example validate
   ```

#### Running Validation

```bash
# Validate against specific prompt
cargo run --example validate -- reference_data/prompt1

# Auto-detect reference data (uses first prompt found)
cargo run --example validate
```

### Output

The tool displays:

1. **Reference data summary**:
   - Input tokens and their IDs
   - Available reference files (embedding, layers, logits)
   - File sizes and validation thresholds

2. **Validation framework status**:
   - Which components have reference data
   - Expected tolerance thresholds
   - Next steps for integration

3. **Example validation** (self-validation of reference data):
   - Demonstrates the validation metrics in action
   - Should show perfect match (MSE=0, Cosine=1.0)

### Example Output

```
=== Rust vs Python Validation ===

Reference data: reference_data/prompt1

Input: 5 tokens
Token IDs: [151643, 9906, 151645, ...]

Found 31 reference activation files

=== Validation Framework Ready ===

Available reference data:
  Embedding (12.34 KB) - thresholds: MSE<1e-5, Cos>0.999, MaxDiff<1e-3
  Layer 0 (12.34 KB) - thresholds: MSE<1e-5, Cos>0.999, MaxDiff<1e-3
  Layer 1 (12.34 KB) - thresholds: MSE<1e-5, Cos>0.999, MaxDiff<1e-3
  ...
  Layer 27 (12.34 KB) - thresholds: MSE<1e-4, Cos>0.999, MaxDiff<1e-3
  Final Norm (12.34 KB) - thresholds: MSE<1e-4, Cos>0.999, MaxDiff<1e-3
  Logits (593.75 KB) - thresholds: MSE<1e-3, Cos>0.999, MaxDiff<1e-2

Next steps:
  1. Implement Rust model forward pass (T19)
  2. Run model with input_ids: [151643, 9906, ...]
  3. Extract intermediate activations
  4. Compare against reference using ValidationMetrics

Example: Loading reference data for Embedding...
  Shape: 5120 elements
  First 5 values: [0.123, -0.456, 0.789, ...]

Self-validation (should be perfect):
  ✓ Embedding                       MSE=0.00e+00  Cosine=1.000000  MaxDiff=0.00e+00  MeanDiff=0.00e+00
```

### Integration Points

Once T19 (Full Model Assembly) and T20 (Generation Loop) are implemented, this tool will:

1. Load the Rust Qwen3 model
2. Run forward pass with reference input_ids
3. Extract activations at each layer
4. Compare with Python reference
5. Display detailed validation report

### Code Structure

#### Main Components

- **`ValidationMetrics`**: Computes and stores validation metrics
  - MSE, cosine similarity, max/mean absolute difference
  - `is_valid()` method with configurable thresholds
  - `print()` method for formatted output

- **`load_npy_flat()`**: Loads NumPy `.npy` files as flat f32 vectors
- **`load_input_ids()`**: Loads input token IDs from `.npy`

#### Implementation Status

Current implementation (T22):
- ✅ Reference data loading infrastructure
- ✅ Validation metrics computation
- ✅ Threshold checking and reporting
- ⏳ Rust model forward pass (T19 dependency)
- ⏳ Layer-by-layer activation extraction (T19 dependency)
- ⏳ Full validation comparison (blocked on T19)

### Validation Metrics

#### Mean Squared Error (MSE)

```
MSE = (1/N) * Σ(predicted[i] - reference[i])²
```

Lower is better. Thresholds:
- Early layers (0-9): < 1e-5
- Later layers (10-27): < 1e-4
- Logits: < 1e-3

#### Cosine Similarity

```
cosine_sim = (predicted · reference) / (||predicted|| * ||reference||)
```

Higher is better. Threshold: > 0.999

Measures directional similarity (invariant to magnitude scaling).

#### Max Absolute Difference

```
max_diff = max(|predicted[i] - reference[i]|)
```

Catches outliers. Threshold: < 1e-3 (based on BF16 precision)

#### Mean Absolute Difference

```
mean_diff = (1/N) * Σ|predicted[i] - reference[i]|
```

Robust to outliers (compared to MSE).

### BF16 Precision Considerations

The tolerance thresholds are based on BF16 (bfloat16) precision characteristics:

- **Mantissa**: 7 bits (~2 decimal digits)
- **Typical error**: ~1e-3 for values in range [-10, 10]
- **Error accumulation**: Increases through 28 transformer layers

Early layers should have very low error (< 1e-5), while later layers accumulate
numerical errors and may reach ~1e-4.

### Troubleshooting

#### "Reference directory does not exist"

Generate reference data first:
```bash
python scripts/extract_reference.py
```

#### No reference files found

Check that the Python script completed successfully and created `.npy` files
in the reference_data directory.

#### Permission errors

Ensure you have read permissions on the reference_data directory.

### See Also

- `tests/validation.rs` - Integration test version of validation
- `scripts/extract_reference.py` - T21 reference data extraction
- `docs/architecture/phase0-implementation-plan.md` - T22 specification
