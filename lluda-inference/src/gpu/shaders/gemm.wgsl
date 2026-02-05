// GEMM (General Matrix-Matrix Multiply) compute shader
//
// Computes: C = A × B
// Where:
//   - A is M×K (stored in row-major order)
//   - B is K×N (stored in row-major order)
//   - C is M×N (output)
//
// Data format:
//   - Input data (A, B) stored as BF16 (2 values packed into 1 u32)
//   - Output data (C) stored as BF16 unpacked (1 value per u32, to avoid race conditions)
//   - Compute happens in F32 for numerical stability
//
// Algorithm: Tiled GEMM with shared memory
//   - Tile size: 16×16 (optimized for AMD RDNA architecture)
//   - Each workgroup computes one output tile (16×16 elements)
//   - Workgroup size: 256 threads (16×16)
//   - Each thread computes one output element

// Storage buffers
@group(0) @binding(0) var<storage, read> matrix_a: array<u32>;  // M×K, BF16 packed (2 per u32)
@group(0) @binding(1) var<storage, read> matrix_b: array<u32>;  // K×N, BF16 packed (2 per u32)
@group(0) @binding(2) var<storage, read_write> matrix_c: array<u32>;  // M×N, BF16 unpacked (1 per u32)

// Uniform buffer with matrix dimensions
struct Uniforms {
    M: u32,  // Number of rows in A, rows in C
    K: u32,  // Number of cols in A, rows in B
    N: u32,  // Number of cols in B, cols in C
}
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

// Shared memory for tiling (16×16 tiles)
var<workgroup> tile_a: array<array<f32, 16>, 16>;  // Shared memory for A tile
var<workgroup> tile_b: array<array<f32, 16>, 16>;  // Shared memory for B tile

// BF16 conversion functions (same as GEMV)
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

/// Extract BF16 value from packed matrix A
/// Matrix A is stored row-major: A[row, col] = matrix_a[(row * K + col) / 2]
fn load_bf16_a(row: u32, col: u32) -> f32 {
    let linear_idx = row * uniforms.K + col;
    let packed_idx = linear_idx / 2u;
    let packed_val = matrix_a[packed_idx];
    let shift = (linear_idx % 2u) * 16u;
    let bf16 = (packed_val >> shift) & 0xFFFFu;
    return bf16_to_f32(bf16);
}

/// Extract BF16 value from packed matrix B
/// Matrix B is stored row-major: B[row, col] = matrix_b[(row * N + col) / 2]
fn load_bf16_b(row: u32, col: u32) -> f32 {
    let linear_idx = row * uniforms.N + col;
    let packed_idx = linear_idx / 2u;
    let packed_val = matrix_b[packed_idx];
    let shift = (linear_idx % 2u) * 16u;
    let bf16 = (packed_val >> shift) & 0xFFFFu;
    return bf16_to_f32(bf16);
}

/// Store BF16 value into output matrix C (unpacked)
/// Matrix C is stored row-major: C[row, col] = matrix_c[row * N + col]
fn store_bf16_c(row: u32, col: u32, val: f32) {
    let linear_idx = row * uniforms.N + col;
    let bf16 = f32_to_bf16(val);
    matrix_c[linear_idx] = bf16 & 0xFFFFu;
}

@compute @workgroup_size(16, 16, 1)
fn gemm_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let row = global_id.y;  // Output row in C
    let col = global_id.x;  // Output col in C

    var sum: f32 = 0.0;

    // Tile loop over K dimension
    let num_tiles = (uniforms.K + 15u) / 16u;

    for (var t = 0u; t < num_tiles; t++) {
        // Load A tile cooperatively
        // Each thread loads one element: A[workgroup_row_base + local_y, tile_col_base + local_x]
        let a_row = workgroup_id.y * 16u + local_id.y;
        let a_col = t * 16u + local_id.x;
        if (a_row < uniforms.M && a_col < uniforms.K) {
            tile_a[local_id.y][local_id.x] = load_bf16_a(a_row, a_col);
        } else {
            tile_a[local_id.y][local_id.x] = 0.0;
        }

        // Load B tile cooperatively
        // Each thread loads one element: B[tile_row_base + local_y, workgroup_col_base + local_x]
        let b_row = t * 16u + local_id.y;
        let b_col = workgroup_id.x * 16u + local_id.x;
        if (b_row < uniforms.K && b_col < uniforms.N) {
            tile_b[local_id.y][local_id.x] = load_bf16_b(b_row, b_col);
        } else {
            tile_b[local_id.y][local_id.x] = 0.0;
        }

        // Wait for all threads to load their tiles
        workgroupBarrier();

        // Compute partial dot product using loaded tiles
        for (var k = 0u; k < 16u; k++) {
            sum += tile_a[local_id.y][k] * tile_b[k][local_id.x];
        }

        // Wait for all threads to finish computation before loading next tile
        workgroupBarrier();
    }

    // Bounds check before writing output
    if (row < uniforms.M && col < uniforms.N) {
        // Write result to output matrix
        store_bf16_c(row, col, sum);
    }
}
