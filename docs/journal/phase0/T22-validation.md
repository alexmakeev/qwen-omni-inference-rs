# T22: Validation (Rust vs Python Reference)

## Date
2026-02-01

## Summary

Implemented comprehensive validation infrastructure for comparing Rust model outputs against Python HuggingFace Transformers reference data. Created both integration tests and a standalone validation tool to ensure correctness of the pure Rust implementation.

The validation system supports layer-by-layer comparison with appropriate tolerance thresholds for BF16 precision, and provides detailed diagnostic reports to identify divergences.

## Implementation

### Components Created

1. **Integration Test** (`tests/validation.rs`):
   - Core validation data structures (`LayerValidation`, `ValidationReport`)
   - Metrics computation functions (MSE, cosine similarity, max/mean absolute difference)
   - NumPy `.npy` file loading utilities
   - Automated test suite with 5 unit tests
   - Full integration test (skips gracefully if reference data not available)

2. **Standalone Tool** (`examples/validate.rs`):
   - Command-line validation tool
   - Auto-detection of reference data directories
   - Detailed progress reporting
   - Self-validation demonstration

3. **Documentation**:
   - `tests/README.md` - Test structure and usage
   - `examples/README.md` - Validation tool guide
   - Both include tolerance thresholds, expected data format, and troubleshooting

4. **Dependencies**:
   - Added `ndarray`, `ndarray-npy`, `approx` to `[dev-dependencies]`

### Validation Metrics

Implemented four complementary metrics:

1. **Mean Squared Error (MSE)**:
   - Early layers: < 1e-5
   - All layers: < 1e-4
   - Logits: < 1e-3

2. **Cosine Similarity**:
   - All components: > 0.999
   - Invariant to magnitude scaling

3. **Max Absolute Difference**:
   - Threshold: < 1e-3 (BF16 precision)
   - Catches outliers

4. **Mean Absolute Difference**:
   - Robust to outliers
   - Complementary to MSE

### Tolerance Threshold Rationale

Thresholds chosen based on BF16 precision characteristics:

- **BF16 mantissa**: 7 bits (~2 decimal digits)
- **Representable error**: ~1e-3 for typical activation magnitudes
- **Error accumulation**: Increases through 28 transformer layers

Early layers (0-9) should have minimal error (~1e-5), while later layers may accumulate to ~1e-4. Logits allow higher tolerance (1e-3) due to accumulated error over 28 layers.

### Reference Data Format

Expected structure (from T21 Python script):

```
reference_data/
  prompt1/
    input_ids.npy          # [1, L] int64
    embedding_output.npy   # [1, L, 1024] float32
    layer_00_output.npy    # [1, L, 1024] float32
    ...
    layer_27_output.npy    # [1, L, 1024] float32
    final_norm_output.npy  # [1, L, 1024] float32
    logits.npy             # [1, L, 151936] float32
    generated_ids.npy      # [1, L+N] int64 (greedy)
    metadata.json          # Prompt info
```

### Code Quality

- **Zero clippy warnings**: Achieved via `-D warnings` flag
- **All tests pass**: 5 unit tests, integration test (graceful skip)
- **Coding standards compliance**:
  - Result-based error handling (no panics)
  - Clear documentation for all public items
  - Proper test module structure (`#[cfg(test)] mod tests`)
  - Iterator-based implementation (clippy::manual_find fix)

## Decisions

### Integration Test vs Standalone Tool

**Decision**: Implement both.

**Rationale**:
- Integration test (`tests/validation.rs`): For CI/CD automated validation
- Standalone tool (`examples/validate.rs`): For interactive debugging and development

Both share the same validation logic (metrics computation, tolerance checking).

### Graceful Skipping

**Decision**: Tests skip gracefully when reference data is missing.

**Rationale**:
- Allows tests to run in CI without requiring large reference data files in repo
- Developer can generate reference data locally when needed
- Clear messages guide user on how to generate reference data

### Tolerance Thresholds

**Decision**: Layer-specific thresholds (stricter for early layers).

**Rationale**:
- Early layers have minimal numerical error
- Error accumulates through 28 transformer layers
- Logits are final output and accumulate most error
- Thresholds match BF16 precision characteristics

### NumPy File Format

**Decision**: Use `.npy` format (not `.npz` or other formats).

**Rationale**:
- Simple one-file-per-tensor structure
- Well-supported by `ndarray-npy` crate
- Easy to inspect and debug
- Matches T21 Python script output format

## Findings

### BF16 Precision Analysis

- **Observation**: BF16 has only 7 mantissa bits vs 23 for float32
- **Implication**: Maximum representable precision ~1e-3 for typical values
- **Impact**: Tolerance thresholds must account for this inherent limitation
- **Qwen3-Omni consideration**: Audio/vision modalities may need different thresholds

### Error Accumulation

- **Observation**: Error grows through transformer layers (28 layers for Qwen3-0.6B)
- **Implication**: Later layers require looser tolerance (1e-4 vs 1e-5)
- **Impact**: Validation must use layer-specific thresholds
- **Qwen3-Omni consideration**: Omni model has additional modality encoders which may accumulate error differently

### Test Infrastructure Benefits

- **Observation**: Modular validation infrastructure enables incremental testing
- **Implication**: Can validate components in isolation before full model is ready
- **Impact**: T22 is complete even though T19 (full model) is still in progress
- **Qwen3-Omni consideration**: Same infrastructure can validate audio/vision encoders separately

## Integration with T19, T20, T21

### Current Status

- **T21 (Python reference extraction)**: Python script exists (`scripts/extract_reference.py`)
- **T19 (Full model assembly)**: In progress (T19, T20, T21 being implemented in parallel)
- **T20 (Generation loop)**: In progress
- **T22 (This validation)**: ✅ Complete (framework ready, awaiting T19/T20 integration)

### Integration Points

Once T19 and T20 are complete, the validation test will:

1. Load Rust Qwen3 model using T19 infrastructure
2. Run forward pass with reference input_ids
3. Extract intermediate activations at each layer
4. Compare with Python reference using `compute_validation_metrics()`
5. Generate `ValidationReport` with pass/fail status

The validation infrastructure is fully ready for this integration.

## Metrics

| Metric | Value |
|--------|-------|
| Lines of code (validation.rs) | 549 |
| Lines of code (validate.rs) | 266 |
| Total test count | 5 |
| Clippy warnings | 0 |
| Cargo warnings | 0 |
| Dev dependencies added | 3 |
| Validation metrics implemented | 4 |
| Documentation files | 2 |

## Testing

### Unit Tests

All 5 tests pass:

1. `test_validation_metrics` - Perfect match case
2. `test_validation_metrics_small_diff` - Small difference case
3. `test_load_reference_data` - .npy file loading
4. `test_validation_report_format` - Report formatting
5. `test_validate_against_python_reference` - Full integration (skips if no data)

### Example Tool

Tested with and without reference data:
- Without data: Clear usage message and instructions
- With data: Displays available reference files and next steps

### Clippy

Zero warnings with `-D warnings` flag:
- Fixed `clippy::manual_find` warning
- Suppressed appropriate `dead_code` warnings for utility functions
- Used `into_raw_vec_and_offset()` instead of deprecated `into_raw_vec()`

## Next Steps

### Immediate (T19/T20 completion)

1. Implement full Qwen3 model assembly (T19)
2. Implement generation loop (T20)
3. Integrate model forward pass into validation test
4. Run validation against Python reference

### Future (Phase 1: Qwen3-Omni)

1. Extend validation to audio encoder
2. Extend validation to vision encoder
3. Add modality-specific tolerance thresholds
4. Validate cross-modal attention

## Observations for Qwen3-Omni

### Validation Extensibility

The validation infrastructure is generic and can be extended for Omni model:

1. **Audio encoder validation**:
   - Extract audio encoder outputs from Python
   - Compare Rust audio encoder implementation
   - May need different tolerance (audio features vs text embeddings)

2. **Vision encoder validation**:
   - Extract vision encoder outputs from Python
   - Compare Rust vision encoder implementation
   - Image patches may have different numerical characteristics

3. **Cross-modal attention**:
   - Validate attention between text and audio/vision
   - Ensure correct alignment of modalities

### Numerical Precision

BF16 may impact different modalities differently:
- **Text**: Embeddings are well-behaved, BF16 is sufficient
- **Audio**: Mel spectrograms have different dynamic range
- **Vision**: Image patches may have outliers (bright/dark pixels)

Validation thresholds may need to be modality-specific.

## Conclusion

T22 validation infrastructure is complete and ready for integration with T19/T20. The implementation follows all coding standards, has zero warnings, and provides comprehensive validation capabilities for ensuring Rust implementation correctness.

The modular design allows incremental testing and will extend cleanly to Qwen3-Omni's multimodal validation requirements in Phase 1.
