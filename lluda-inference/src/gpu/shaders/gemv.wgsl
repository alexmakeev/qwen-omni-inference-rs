// GEMV (General Matrix-Vector Multiply) compute shader
//
// Computes: output = matrix × vector
// Where:
//   - matrix is M×N (stored in row-major order)
//   - vector is N×1
//   - output is M×1
//
// Data format:
//   - All data stored as BF16 (2 values packed into 1 u32)
//   - Compute happens in F32 for numerical stability
//   - Results converted back to BF16 for output
//
// Parallelization:
//   - Each thread computes one output element (one row dot product)
//   - Dispatch (M, 1, 1) workgroups

// Storage buffers
@group(0) @binding(0) var<storage, read> matrix: array<u32>;    // BF16 matrix data (packed)
@group(0) @binding(1) var<storage, read> vector: array<u32>;    // BF16 vector data (packed)
@group(0) @binding(2) var<storage, read_write> output: array<u32>;  // BF16 output data (packed)

// Uniform buffer with matrix dimensions
struct Uniforms {
    M: u32,  // Number of rows in matrix
    N: u32,  // Number of columns in matrix (= vector length)
}
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

// BF16 conversion functions
// BF16 format: 1 bit sign, 8 bits exponent, 7 bits mantissa (high 16 bits of F32)

/// Convert BF16 (as u32) to F32
fn bf16_to_f32(bf16: u32) -> f32 {
    let bits = bf16 << 16u;  // Shift BF16 to high 16 bits of 32-bit word
    return bitcast<f32>(bits);
}

/// Convert F32 to BF16 (as u32)
fn f32_to_bf16(val: f32) -> u32 {
    let bits = bitcast<u32>(val);
    return bits >> 16u;  // Take high 16 bits
}

/// Extract BF16 value from packed matrix array
fn extract_bf16_matrix(idx: u32) -> u32 {
    let packed_idx = idx / 2u;
    let packed_val = matrix[packed_idx];
    let shift = (idx % 2u) * 16u;
    return (packed_val >> shift) & 0xFFFFu;
}

/// Extract BF16 value from packed vector array
fn extract_bf16_vector(idx: u32) -> u32 {
    let packed_idx = idx / 2u;
    let packed_val = vector[packed_idx];
    let shift = (idx % 2u) * 16u;
    return (packed_val >> shift) & 0xFFFFu;
}

/// Store BF16 value into output array (unpacked, one BF16 per u32)
fn store_bf16_output(idx: u32, val: u32) {
    // Store BF16 value in low 16 bits of its own u32
    output[idx] = val & 0xFFFFu;
}

@compute @workgroup_size(256, 1, 1)
fn gemv_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;

    // Bounds check: only process valid rows
    if (row >= uniforms.M) {
        return;
    }

    // Compute dot product of matrix row with vector
    var sum: f32 = 0.0;
    for (var col = 0u; col < uniforms.N; col++) {
        // Matrix is row-major: index = row * N + col
        let mat_idx = row * uniforms.N + col;
        let mat_bf16 = extract_bf16_matrix(mat_idx);
        let mat_val = bf16_to_f32(mat_bf16);

        let vec_bf16 = extract_bf16_vector(col);
        let vec_val = bf16_to_f32(vec_bf16);

        sum += mat_val * vec_val;
    }

    // Convert result back to BF16 and store
    let result_bf16 = f32_to_bf16(sum);
    store_bf16_output(row, result_bf16);
}
