# T22: Validation Tests

This directory contains integration tests for validating the Rust implementation against Python reference data (from T21).

## Test Structure

### `validation.rs`

Main validation test module with the following components:

#### Core Data Structures

- **`LayerValidation`**: Validation metrics for a single layer/component
  - MSE (Mean Squared Error)
  - Cosine similarity
  - Max absolute difference
  - Mean absolute difference

- **`ValidationReport`**: Complete validation report aggregating all layers
  - Embedding layer validation
  - Per-layer transformer validations (28 layers for Qwen3-0.6B)
  - Final normalization validation
  - Logits validation
  - Generated token comparison

#### Validation Metrics

Compares predicted (Rust) vs reference (Python) tensors using:

1. **Mean Squared Error (MSE)**:
   - Early layers: < 1e-5
   - All layers: < 1e-4
   - Logits: < 1e-3

2. **Cosine Similarity**:
   - All components: > 0.999

3. **Max Absolute Difference**:
   - All components: < 1e-3 (BF16 precision)

#### Utility Functions

- `load_npy_1d()`, `load_npy_2d()`, `load_npy_3d()`, `load_npy_dyn()`: Load NumPy arrays
- `compute_validation_metrics()`: Calculate all metrics for two tensors
- `find_reference_dir()`: Locate reference_data directory

#### Tests

1. **`test_validation_metrics`**: Basic metric computation
2. **`test_validation_metrics_small_diff`**: Metric computation with small differences
3. **`test_load_reference_data`**: Test loading .npy files
4. **`test_validation_report_format`**: Report formatting
5. **`test_validate_against_python_reference`**: Full integration test (requires reference data)

## Running Tests

### Basic unit tests (always run):

```bash
cargo test --test validation
```

### With reference data (requires T21 Python script output):

1. First generate reference data:
   ```bash
   python scripts/extract_reference.py
   ```

2. Run validation tests:
   ```bash
   cargo test --test validation
   ```

The tests will automatically detect if reference data is available and skip if not present.

## Expected Reference Data Format

```
reference_data/
  prompt1/
    input_ids.npy          # Input token IDs [1, L]
    embedding_output.npy   # Embedding output [1, L, 1024]
    layer_00_output.npy    # Layer 0 output [1, L, 1024]
    layer_01_output.npy    # Layer 1 output [1, L, 1024]
    ...
    layer_27_output.npy    # Layer 27 output [1, L, 1024]
    final_norm_output.npy  # Final norm output [1, L, 1024]
    logits.npy             # Logits [1, L, vocab_size]
    generated_ids.npy      # Generated tokens (greedy)
    metadata.json          # Prompt and generation info
  prompt2/
    ...
```

## Tolerance Thresholds

Thresholds are chosen based on BF16 precision characteristics:

- **BF16 mantissa**: 7 bits (~2 decimal digits of precision)
- **Maximum representable error**: ~1e-3 for typical values
- **Error accumulation**: Increases through 28 transformer layers

### Per-Component Thresholds

| Component | MSE Threshold | Cosine Threshold | Max Diff Threshold |
|-----------|---------------|------------------|--------------------|
| Embedding | 1e-5 | 0.999 | 1e-3 |
| Layers 0-9 | 1e-5 | 0.999 | 1e-3 |
| Layers 10-27 | 1e-4 | 0.999 | 1e-3 |
| Final Norm | 1e-4 | 0.999 | 1e-3 |
| Logits | 1e-3 | 0.999 | 1e-2 |

## Dependencies

Test-only dependencies (in `[dev-dependencies]`):

- `ndarray = "0.16"` - N-dimensional array library
- `ndarray-npy = "0.9"` - NumPy `.npy` file format support
- `approx = "0.5"` - Approximate float comparisons

## Integration with T19 and T20

Once T19 (Full Model Assembly) and T20 (Generation Loop) are complete, the validation test will:

1. Load the Rust model
2. Run forward pass with reference input_ids
3. Extract intermediate activations at each layer
4. Compare against Python reference using `compute_validation_metrics()`
5. Generate report showing which layers pass/fail

## Example Output

```
=== Validation Report ===

Embedding:
  Layer  0: MSE=1.50e-06, Cosine=0.999500, MaxDiff=5.00e-04, MeanDiff=1.00e-04

Transformer Layers:
  ✓ Layer  0: MSE=2.00e-06, Cosine=0.999600, MaxDiff=6.00e-04, MeanDiff=1.20e-04
  ✓ Layer  1: MSE=2.50e-06, Cosine=0.999550, MaxDiff=7.00e-04, MeanDiff=1.30e-04
  ...
  ✓ Layer 27: MSE=8.00e-05, Cosine=0.999200, MaxDiff=9.00e-04, MeanDiff=2.00e-04

Final Norm:
  Layer  0: MSE=9.00e-05, Cosine=0.999100, MaxDiff=9.50e-04, MeanDiff=2.10e-04

Logits:
  Layer  0: MSE=5.00e-04, Cosine=0.999200, MaxDiff=2.00e-03, MeanDiff=5.00e-04

Generated tokens match: ✓

Overall: PASS
```

## See Also

- `examples/validate.rs` - Standalone validation tool
- `scripts/extract_reference.py` - T21 reference data extraction script
- `docs/architecture/phase0-implementation-plan.md` - T22 specification
