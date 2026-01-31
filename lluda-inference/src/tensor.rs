//! Core tensor type for multi-dimensional array operations.
//!
//! The Tensor struct provides storage for neural network data with support for:
//! - Multiple data types (F32 for compute, BF16 for storage)
//! - Shape and stride tracking
//! - Automatic type conversion
//!
//! # Design Principle
//!
//! **All compute happens in F32. BF16 is for storage only.**
//!
//! This design avoids the complexity of BF16 arithmetic (which has no hardware support
//! on x86 CPUs) while still benefiting from reduced memory bandwidth when loading weights.
//!
//! # Example
//!
//! ```rust
//! use lluda_inference::tensor::{Tensor, DType};
//!
//! // Create a 2x3 F32 tensor
//! let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
//!
//! // Query properties
//! assert_eq!(t.shape(), &[2, 3]);
//! assert_eq!(t.numel(), 6);
//! assert_eq!(t.ndim(), 2);
//! assert_eq!(t.dtype(), DType::F32);
//!
//! // Convert to BF16 for storage
//! let t_bf16 = t.to_dtype(DType::BF16).unwrap();
//!
//! // Get data back as F32 for compute
//! let data = t_bf16.to_vec_f32();
//! assert_eq!(data.len(), 6);
//! ```

use crate::bf16::BF16;
use crate::error::{LludaError, Result};

/// Tensor data type.
///
/// Specifies how tensor data is stored in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// Brain Float 16 (16-bit floating point, storage only)
    BF16,
    /// Float 32 (32-bit floating point, used for compute)
    F32,
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DType::BF16 => write!(f, "BF16"),
            DType::F32 => write!(f, "F32"),
        }
    }
}

/// Internal storage for tensor data.
///
/// Holds either BF16 or F32 data in a type-safe enum.
#[derive(Debug, Clone)]
enum TensorData {
    /// BF16 storage (for weights loaded from disk)
    BF16(Vec<BF16>),
    /// F32 storage (for activations and compute results)
    F32(Vec<f32>),
}

impl TensorData {
    /// Get the data type of this storage.
    fn dtype(&self) -> DType {
        match self {
            TensorData::BF16(_) => DType::BF16,
            TensorData::F32(_) => DType::F32,
        }
    }
}

/// Multi-dimensional tensor with shape and data type information.
///
/// Tensors store data in row-major (C-contiguous) layout by default.
/// All compute operations happen in F32 - BF16 tensors are automatically
/// converted when extracting data via `to_vec_f32()`.
#[derive(Debug, Clone)]
pub struct Tensor {
    data: TensorData,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl Tensor {
    /// Create a new F32 tensor from data and shape.
    ///
    /// The tensor is stored in row-major (C-contiguous) layout.
    ///
    /// # Arguments
    ///
    /// * `data` - Flat vector of F32 values
    /// * `shape` - Dimensions of the tensor
    ///
    /// # Errors
    ///
    /// Returns `ShapeMismatch` if the product of shape dimensions doesn't match data length.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// assert_eq!(t.shape(), &[2, 2]);
    /// ```
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        // Validate shape matches data length
        let numel = shape.iter().product();
        if data.len() != numel {
            return Err(LludaError::ShapeMismatch {
                expected: vec![numel],
                got: vec![data.len()],
            });
        }

        // Compute row-major strides
        let strides = compute_strides(&shape);

        Ok(Tensor {
            data: TensorData::F32(data),
            shape,
            strides,
        })
    }

    /// Create a new BF16 tensor from data and shape.
    ///
    /// This is typically used when loading weights from disk in BF16 format.
    ///
    /// # Arguments
    ///
    /// * `data` - Flat vector of BF16 values
    /// * `shape` - Dimensions of the tensor
    ///
    /// # Errors
    ///
    /// Returns `ShapeMismatch` if the product of shape dimensions doesn't match data length.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::tensor::Tensor;
    /// use lluda_inference::bf16::BF16;
    ///
    /// let data = vec![BF16::from(1.0f32), BF16::from(2.0f32)];
    /// let t = Tensor::from_bf16(data, vec![2]).unwrap();
    /// assert_eq!(t.shape(), &[2]);
    /// ```
    pub fn from_bf16(data: Vec<BF16>, shape: Vec<usize>) -> Result<Self> {
        // Validate shape matches data length
        let numel = shape.iter().product();
        if data.len() != numel {
            return Err(LludaError::ShapeMismatch {
                expected: vec![numel],
                got: vec![data.len()],
            });
        }

        // Compute row-major strides
        let strides = compute_strides(&shape);

        Ok(Tensor {
            data: TensorData::BF16(data),
            shape,
            strides,
        })
    }

    /// Convert tensor to a different data type.
    ///
    /// Creates a new tensor with data converted to the target dtype.
    /// This is a no-op if the tensor is already in the target dtype.
    ///
    /// # Arguments
    ///
    /// * `dtype` - Target data type (BF16 or F32)
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::tensor::{Tensor, DType};
    ///
    /// let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    /// let t_bf16 = t.to_dtype(DType::BF16).unwrap();
    /// assert_eq!(t_bf16.dtype(), DType::BF16);
    /// ```
    pub fn to_dtype(&self, dtype: DType) -> Result<Self> {
        // No-op if already in target dtype
        if self.dtype() == dtype {
            return Ok(self.clone());
        }

        let new_data = match (&self.data, dtype) {
            (TensorData::F32(f32_vec), DType::BF16) => {
                // F32 -> BF16
                let bf16_vec = BF16::from_f32_slice(f32_vec);
                TensorData::BF16(bf16_vec)
            }
            (TensorData::BF16(bf16_vec), DType::F32) => {
                // BF16 -> F32
                let f32_vec = BF16::to_f32_slice(bf16_vec);
                TensorData::F32(f32_vec)
            }
            _ => unreachable!("Already checked for same dtype above"),
        };

        Ok(Tensor {
            data: new_data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        })
    }

    /// Get the shape of the tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    /// assert_eq!(t.shape(), &[2, 3]);
    /// ```
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the data type of the tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::tensor::{Tensor, DType};
    ///
    /// let t = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
    /// assert_eq!(t.dtype(), DType::F32);
    /// ```
    pub fn dtype(&self) -> DType {
        self.data.dtype()
    }

    /// Get the total number of elements in the tensor.
    ///
    /// This is the product of all dimensions in the shape.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    /// assert_eq!(t.numel(), 6);
    /// ```
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the number of dimensions (rank) of the tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    /// assert_eq!(t.ndim(), 2);
    /// ```
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get tensor data as a vector of F32 values.
    ///
    /// This is the primary way to extract data for computation.
    /// If the tensor is stored as BF16, it will be automatically converted to F32.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::tensor::{Tensor, DType};
    /// use lluda_inference::bf16::BF16;
    ///
    /// // BF16 tensor
    /// let data = vec![BF16::from(1.0f32), BF16::from(2.0f32), BF16::from(3.0f32)];
    /// let t = Tensor::from_bf16(data, vec![3]).unwrap();
    ///
    /// // Extract as F32 for compute
    /// let f32_data = t.to_vec_f32();
    /// assert_eq!(f32_data.len(), 3);
    /// assert_eq!(f32_data[0], 1.0f32);
    /// ```
    pub fn to_vec_f32(&self) -> Vec<f32> {
        match &self.data {
            TensorData::F32(v) => v.clone(),
            TensorData::BF16(v) => BF16::to_f32_slice(v),
        }
    }

    /// Reshape tensor to a new shape without copying data (Phase 0: copies data).
    ///
    /// The total number of elements must remain the same.
    ///
    /// # Arguments
    ///
    /// * `new_shape` - Target shape dimensions
    ///
    /// # Returns
    ///
    /// Reshaped tensor with same data in new shape.
    ///
    /// # Errors
    ///
    /// Returns `ShapeMismatch` if the new shape has a different total number of elements.
    ///
    /// # Performance
    ///
    /// PERF: Phase 0 copies data for simplicity. Phase 3 will implement zero-copy views
    /// with stride manipulation for contiguous tensors.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    /// let reshaped = t.reshape(&[3, 2]).unwrap();
    ///
    /// assert_eq!(reshaped.shape(), &[3, 2]);
    /// // Data order preserved: [1, 2, 3, 4, 5, 6]
    /// ```
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Tensor> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(LludaError::ShapeMismatch {
                expected: vec![new_numel],
                got: vec![self.numel()],
            });
        }

        // PERF: Phase 0 copies data. Phase 3: zero-copy views with stride magic.
        // Preserve dtype: BF16 stays BF16, F32 stays F32.
        match &self.data {
            TensorData::F32(data) => Tensor::new(data.clone(), new_shape.to_vec()),
            TensorData::BF16(data) => Tensor::from_bf16(data.clone(), new_shape.to_vec()),
        }
    }

    /// Transpose a 2D tensor (swap dimensions).
    ///
    /// For Phase 0, only 2D tensors are supported. The operation swaps rows and columns.
    ///
    /// # Returns
    ///
    /// Transposed tensor with dimensions swapped: (M, N) -> (N, M).
    ///
    /// # Errors
    ///
    /// Returns `DimOutOfRange` if the tensor is not 2D.
    ///
    /// # Performance
    ///
    /// PERF: Phase 0 copies and reorders data. Phase 3 will implement lazy transpose
    /// (just swap strides, no data copy).
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::tensor::Tensor;
    ///
    /// // Matrix: [[1, 2], [3, 4]]
    /// let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let transposed = t.transpose().unwrap();
    ///
    /// // Result: [[1, 3], [2, 4]]
    /// assert_eq!(transposed.shape(), &[2, 2]);
    /// let data = transposed.to_vec_f32();
    /// assert_eq!(data, vec![1.0, 3.0, 2.0, 4.0]);
    /// ```
    pub fn transpose(&self) -> Result<Tensor> {
        if self.ndim() != 2 {
            return Err(LludaError::DimOutOfRange {
                dim: 2,
                ndim: self.ndim(),
            });
        }

        let m = self.shape[0];
        let n = self.shape[1];
        let data = self.to_vec_f32();
        let mut result = vec![0.0; m * n];

        // Transpose: output[j, i] = input[i, j]
        // Row-major indexing: input[i * n + j] -> output[j * m + i]
        for i in 0..m {
            for j in 0..n {
                result[j * m + i] = data[i * n + j];
            }
        }

        Tensor::new(result, vec![n, m])
    }

    /// Transpose two dimensions of a tensor.
    ///
    /// Swaps dimensions `dim0` and `dim1`, reordering data to match the new layout.
    /// Works for tensors of any rank.
    ///
    /// # Arguments
    ///
    /// * `dim0` - First dimension to swap
    /// * `dim1` - Second dimension to swap
    ///
    /// # Returns
    ///
    /// Tensor with dimensions `dim0` and `dim1` swapped.
    ///
    /// # Errors
    ///
    /// Returns `DimOutOfRange` if either dimension index is out of range.
    ///
    /// # Performance
    ///
    /// PERF: Phase 0 copies and reorders data using multi-index iteration.
    /// Phase 3 will implement lazy transpose (swap strides, no data copy).
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::tensor::Tensor;
    ///
    /// // 3D tensor [2, 3, 4], transpose dims 1 and 2 -> [2, 4, 3]
    /// let t = Tensor::new(vec![0.0; 24], vec![2, 3, 4]).unwrap();
    /// let tr = t.transpose_dims(1, 2).unwrap();
    /// assert_eq!(tr.shape(), &[2, 4, 3]);
    /// ```
    #[allow(clippy::needless_range_loop)]
    pub fn transpose_dims(&self, dim0: usize, dim1: usize) -> Result<Tensor> {
        if dim0 >= self.ndim() {
            return Err(LludaError::DimOutOfRange {
                dim: dim0,
                ndim: self.ndim(),
            });
        }
        if dim1 >= self.ndim() {
            return Err(LludaError::DimOutOfRange {
                dim: dim1,
                ndim: self.ndim(),
            });
        }

        // No-op if same dimension
        if dim0 == dim1 {
            return Ok(self.clone());
        }

        let data = self.to_vec_f32();
        let ndim = self.ndim();

        // Build new shape with swapped dimensions
        let mut new_shape = self.shape.clone();
        new_shape.swap(dim0, dim1);

        // Compute destination strides
        let dst_strides = compute_strides(&new_shape);

        let numel = self.numel();
        let mut result = vec![0.0f32; numel];

        // PERF: Naive multi-index iteration. O(numel * ndim).
        // For each element, compute source and destination linear indices.
        let mut multi_idx = vec![0usize; ndim];
        for flat_src in 0..numel {
            // Decompose flat_src into multi-index using source strides
            if flat_src > 0 {
                // Increment multi-index (like an odometer)
                let mut carry = true;
                for d in (0..ndim).rev() {
                    if carry {
                        multi_idx[d] += 1;
                        if multi_idx[d] >= self.shape[d] {
                            multi_idx[d] = 0;
                        } else {
                            carry = false;
                        }
                    }
                }
            }

            // Swap the indices for dim0 and dim1 to get destination multi-index
            let mut dst_idx = multi_idx.clone();
            dst_idx.swap(dim0, dim1);

            // Compute destination flat index
            let flat_dst: usize = dst_idx
                .iter()
                .zip(dst_strides.iter())
                .map(|(&i, &s)| i * s)
                .sum();

            result[flat_dst] = data[flat_src];
        }

        Tensor::new(result, new_shape)
    }

    /// Remove dimensions of size 1 from the tensor shape.
    ///
    /// # Arguments
    ///
    /// * `dim` - Optional dimension index to squeeze. If None, removes all size-1 dimensions.
    ///   If Some(i), removes dimension i only if its size is 1.
    ///
    /// # Returns
    ///
    /// Squeezed tensor with size-1 dimensions removed.
    ///
    /// # Errors
    ///
    /// Returns error if `dim` is Some(i) and dimension i does not have size 1,
    /// or if dimension index is out of range.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::tensor::Tensor;
    ///
    /// // Shape: [1, 3, 1]
    /// let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3, 1]).unwrap();
    ///
    /// // Remove all size-1 dims -> [3]
    /// let squeezed = t.squeeze(None).unwrap();
    /// assert_eq!(squeezed.shape(), &[3]);
    ///
    /// // Remove specific dim 0 -> [3, 1]
    /// let squeezed_0 = t.squeeze(Some(0)).unwrap();
    /// assert_eq!(squeezed_0.shape(), &[3, 1]);
    /// ```
    pub fn squeeze(&self, dim: Option<usize>) -> Result<Tensor> {
        let new_shape: Vec<usize> = match dim {
            None => {
                // Remove all dimensions of size 1
                self.shape.iter().copied().filter(|&s| s != 1).collect()
            }
            Some(d) => {
                // Validate dimension index
                if d >= self.ndim() {
                    return Err(LludaError::DimOutOfRange {
                        dim: d,
                        ndim: self.ndim(),
                    });
                }

                // Check that dimension has size 1
                if self.shape[d] != 1 {
                    return Err(LludaError::Msg(format!(
                        "Cannot squeeze dimension {} with size {} (must be 1)",
                        d, self.shape[d]
                    )));
                }

                // Remove only the specified dimension
                self.shape
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != d)
                    .map(|(_, &s)| s)
                    .collect()
            }
        };

        self.reshape(&new_shape)
    }

    /// Add a dimension of size 1 at the specified position.
    ///
    /// # Arguments
    ///
    /// * `dim` - Position where the new dimension should be inserted
    ///
    /// # Returns
    ///
    /// Tensor with an additional dimension of size 1 at position `dim`.
    ///
    /// # Errors
    ///
    /// Returns `DimOutOfRange` if `dim` is greater than the number of dimensions.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::tensor::Tensor;
    ///
    /// // Shape: [3]
    /// let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    ///
    /// // Insert at position 0 -> [1, 3]
    /// let unsqueezed = t.unsqueeze(0).unwrap();
    /// assert_eq!(unsqueezed.shape(), &[1, 3]);
    ///
    /// // Insert at position 1 -> [3, 1]
    /// let unsqueezed_end = t.unsqueeze(1).unwrap();
    /// assert_eq!(unsqueezed_end.shape(), &[3, 1]);
    /// ```
    pub fn unsqueeze(&self, dim: usize) -> Result<Tensor> {
        // Can insert at position 0 to ndim (inclusive, to append)
        if dim > self.ndim() {
            return Err(LludaError::DimOutOfRange {
                dim,
                ndim: self.ndim(),
            });
        }

        let mut new_shape = self.shape.clone();
        new_shape.insert(dim, 1);

        self.reshape(&new_shape)
    }

    /// Matrix multiplication: self @ rhs
    ///
    /// Supports:
    /// - 2D: [M, K] @ [K, N] -> [M, N]
    /// - 3D batched: [B, M, K] @ [B, K, N] -> [B, M, N]
    /// - 4D batched: [B, H, M, K] @ [B, H, K, N] -> [B, H, M, N] (multi-head attention)
    ///
    /// # Arguments
    ///
    /// * `rhs` - Right-hand side tensor
    ///
    /// # Returns
    ///
    /// Result tensor in F32 dtype (never BF16, as all compute happens in F32).
    ///
    /// # Errors
    ///
    /// Returns `ShapeMismatch` if dimensions are incompatible for matrix multiplication.
    ///
    /// # Performance
    ///
    /// PERF: Naive O(M*N*K) triple-loop implementation. Sufficient for Qwen3-0.6B model
    /// (~1024 hidden size, 28 layers). Phase 3 will optimize with:
    /// - BLAS (cblas_sgemm)
    /// - SIMD vectorization
    /// - Cache tiling
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::tensor::Tensor;
    ///
    /// // 2D matmul
    /// let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
    /// let c = a.matmul(&b).unwrap();
    ///
    /// // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    /// //         = [[19, 22], [43, 50]]
    /// assert_eq!(c.shape(), &[2, 2]);
    /// ```
    pub fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        let lhs_shape = self.shape();
        let rhs_shape = rhs.shape();

        // Both tensors must be at least 2D
        if lhs_shape.len() < 2 {
            return Err(LludaError::ShapeMismatch {
                expected: vec![0, 0], // at least 2D
                got: lhs_shape.to_vec(),
            });
        }
        if rhs_shape.len() < 2 {
            return Err(LludaError::ShapeMismatch {
                expected: vec![0, 0], // at least 2D
                got: rhs_shape.to_vec(),
            });
        }

        // Convert both to F32 for computation
        let lhs_data = self.to_vec_f32();
        let rhs_data = rhs.to_vec_f32();

        // Dispatch based on dimensionality
        match (lhs_shape.len(), rhs_shape.len()) {
            (2, 2) => {
                // Standard 2D matmul: [M, K] @ [K, N] -> [M, N]
                matmul_2d(&lhs_data, lhs_shape, &rhs_data, rhs_shape)
            }
            (3, 3) => {
                // Batched 3D matmul: [B, M, K] @ [B, K, N] -> [B, M, N]
                matmul_3d(&lhs_data, lhs_shape, &rhs_data, rhs_shape)
            }
            (4, 4) => {
                // Batched 4D matmul: [B, H, M, K] @ [B, H, K, N] -> [B, H, M, N]
                // Used for multi-head attention: Q @ K^T and attn_weights @ V
                matmul_4d(&lhs_data, lhs_shape, &rhs_data, rhs_shape)
            }
            _ => Err(LludaError::Msg(format!(
                "matmul: unsupported dimensions {}D @ {}D (only 2D, 3D, and 4D batched supported)",
                lhs_shape.len(),
                rhs_shape.len()
            ))),
        }
    }

    /// Element-wise addition: self + rhs
    ///
    /// Supports:
    /// - Same shape: element-wise addition
    /// - Simple broadcasting: scalar [1] can broadcast to any shape
    ///
    /// # Arguments
    ///
    /// * `rhs` - Right-hand side tensor
    ///
    /// # Returns
    ///
    /// Result tensor in F32 dtype.
    ///
    /// # Errors
    ///
    /// Returns `ShapeMismatch` if shapes are incompatible for broadcasting.
    ///
    /// # Performance
    ///
    /// PERF: Iterator-based element-wise operation. Phase 3 optimizations:
    /// - SIMD vectorization (packed_simd or std::simd)
    /// - Parallel iteration (rayon)
    /// - In-place operations (avoid allocation)
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    /// let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]).unwrap();
    /// let c = a.add(&b).unwrap();
    ///
    /// assert_eq!(c.to_vec_f32(), vec![5.0, 7.0, 9.0]);
    /// ```
    pub fn add(&self, rhs: &Tensor) -> Result<Tensor> {
        // Convert both to F32 for computation
        let lhs_data = self.to_vec_f32();
        let rhs_data = rhs.to_vec_f32();

        // Fast path: same shape (no broadcast needed)
        if self.shape() == rhs.shape() {
            let result: Vec<f32> = lhs_data
                .iter()
                .zip(rhs_data.iter())
                .map(|(a, b)| a + b)
                .collect();
            return Tensor::new(result, self.shape().to_vec());
        }

        // General NumPy-style broadcasting
        broadcast_binary_op(&lhs_data, self.shape(), &rhs_data, rhs.shape(), |a, b| {
            a + b
        })
    }

    /// Element-wise multiplication: self * rhs
    ///
    /// Supports:
    /// - Same shape: element-wise multiplication
    /// - Simple broadcasting: scalar [1] can broadcast to any shape
    ///
    /// # Arguments
    ///
    /// * `rhs` - Right-hand side tensor
    ///
    /// # Returns
    ///
    /// Result tensor in F32 dtype.
    ///
    /// # Errors
    ///
    /// Returns `ShapeMismatch` if shapes are incompatible for broadcasting.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    /// let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]).unwrap();
    /// let c = a.mul(&b).unwrap();
    ///
    /// assert_eq!(c.to_vec_f32(), vec![4.0, 10.0, 18.0]);
    /// ```
    pub fn mul(&self, rhs: &Tensor) -> Result<Tensor> {
        // Convert both to F32 for computation
        let lhs_data = self.to_vec_f32();
        let rhs_data = rhs.to_vec_f32();

        // Fast path: same shape (no broadcast needed)
        if self.shape() == rhs.shape() {
            let result: Vec<f32> = lhs_data
                .iter()
                .zip(rhs_data.iter())
                .map(|(a, b)| a * b)
                .collect();
            return Tensor::new(result, self.shape().to_vec());
        }

        // General NumPy-style broadcasting
        broadcast_binary_op(&lhs_data, self.shape(), &rhs_data, rhs.shape(), |a, b| {
            a * b
        })
    }

    /// Element-wise division: self / rhs
    ///
    /// Supports:
    /// - Same shape: element-wise division
    /// - Simple broadcasting: scalar [1] can broadcast to any shape
    ///
    /// # Arguments
    ///
    /// * `rhs` - Right-hand side tensor (divisor)
    ///
    /// # Returns
    ///
    /// Result tensor in F32 dtype.
    ///
    /// # Errors
    ///
    /// Returns `ShapeMismatch` if shapes are incompatible for broadcasting.
    ///
    /// # Note
    ///
    /// Division by zero produces Inf or NaN according to IEEE 754 floating point rules.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let a = Tensor::new(vec![4.0, 10.0, 18.0], vec![3]).unwrap();
    /// let b = Tensor::new(vec![2.0, 5.0, 6.0], vec![3]).unwrap();
    /// let c = a.div(&b).unwrap();
    ///
    /// assert_eq!(c.to_vec_f32(), vec![2.0, 2.0, 3.0]);
    /// ```
    pub fn div(&self, rhs: &Tensor) -> Result<Tensor> {
        // Convert both to F32 for computation
        let lhs_data = self.to_vec_f32();
        let rhs_data = rhs.to_vec_f32();

        // Fast path: same shape (no broadcast needed)
        if self.shape() == rhs.shape() {
            let result: Vec<f32> = lhs_data
                .iter()
                .zip(rhs_data.iter())
                .map(|(a, b)| a / b)
                .collect();
            return Tensor::new(result, self.shape().to_vec());
        }

        // General NumPy-style broadcasting
        broadcast_binary_op(&lhs_data, self.shape(), &rhs_data, rhs.shape(), |a, b| {
            a / b
        })
    }

    /// Negate all elements: -self
    ///
    /// # Returns
    ///
    /// Result tensor in F32 dtype with all elements negated.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let a = Tensor::new(vec![1.0, -2.0, 3.0], vec![3]).unwrap();
    /// let b = a.neg().unwrap();
    ///
    /// assert_eq!(b.to_vec_f32(), vec![-1.0, 2.0, -3.0]);
    /// ```
    pub fn neg(&self) -> Result<Tensor> {
        let data = self.to_vec_f32();
        let result: Vec<f32> = data.iter().map(|x| -x).collect();
        Tensor::new(result, self.shape().to_vec())
    }

    /// Reciprocal of all elements: 1/self
    ///
    /// # Returns
    ///
    /// Result tensor in F32 dtype with reciprocals of all elements.
    ///
    /// # Note
    ///
    /// Division by zero produces Inf according to IEEE 754 floating point rules.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let a = Tensor::new(vec![2.0, 4.0, 8.0], vec![3]).unwrap();
    /// let b = a.recip().unwrap();
    ///
    /// assert_eq!(b.to_vec_f32(), vec![0.5, 0.25, 0.125]);
    /// ```
    pub fn recip(&self) -> Result<Tensor> {
        let data = self.to_vec_f32();
        let result: Vec<f32> = data.iter().map(|x| 1.0 / x).collect();
        Tensor::new(result, self.shape().to_vec())
    }

    /// Add scalar to all elements: self + scalar
    ///
    /// # Arguments
    ///
    /// * `scalar` - Scalar value to add to each element
    ///
    /// # Returns
    ///
    /// Result tensor in F32 dtype.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    /// let b = a.add_scalar(5.0).unwrap();
    ///
    /// assert_eq!(b.to_vec_f32(), vec![6.0, 7.0, 8.0]);
    /// ```
    pub fn add_scalar(&self, scalar: f32) -> Result<Tensor> {
        let data = self.to_vec_f32();
        let result: Vec<f32> = data.iter().map(|x| x + scalar).collect();
        Tensor::new(result, self.shape().to_vec())
    }

    /// Multiply all elements by scalar: self * scalar
    ///
    /// # Arguments
    ///
    /// * `scalar` - Scalar value to multiply each element by
    ///
    /// # Returns
    ///
    /// Result tensor in F32 dtype.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    /// let b = a.mul_scalar(2.0).unwrap();
    ///
    /// assert_eq!(b.to_vec_f32(), vec![2.0, 4.0, 6.0]);
    /// ```
    pub fn mul_scalar(&self, scalar: f32) -> Result<Tensor> {
        let data = self.to_vec_f32();
        let result: Vec<f32> = data.iter().map(|x| x * scalar).collect();
        Tensor::new(result, self.shape().to_vec())
    }

    /// Element-wise subtraction: self - rhs
    ///
    /// Supports NumPy-style broadcasting (same rules as `add`).
    ///
    /// # Arguments
    ///
    /// * `rhs` - Right-hand side tensor (subtrahend)
    ///
    /// # Returns
    ///
    /// Result tensor in F32 dtype.
    ///
    /// # Errors
    ///
    /// Returns `ShapeMismatch` if shapes are incompatible for broadcasting.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let a = Tensor::new(vec![5.0, 7.0, 9.0], vec![3]).unwrap();
    /// let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    /// let c = a.sub(&b).unwrap();
    ///
    /// assert_eq!(c.to_vec_f32(), vec![4.0, 5.0, 6.0]);
    /// ```
    pub fn sub(&self, rhs: &Tensor) -> Result<Tensor> {
        // Convert both to F32 for computation
        let lhs_data = self.to_vec_f32();
        let rhs_data = rhs.to_vec_f32();

        // Fast path: same shape (no broadcast needed)
        if self.shape() == rhs.shape() {
            let result: Vec<f32> = lhs_data
                .iter()
                .zip(rhs_data.iter())
                .map(|(a, b)| a - b)
                .collect();
            return Tensor::new(result, self.shape().to_vec());
        }

        // General NumPy-style broadcasting
        broadcast_binary_op(&lhs_data, self.shape(), &rhs_data, rhs.shape(), |a, b| {
            a - b
        })
    }

    /// Element-wise exponential: e^x for each element.
    ///
    /// # Returns
    ///
    /// Result tensor in F32 dtype with e^x applied to each element.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let a = Tensor::new(vec![0.0, 1.0, 2.0], vec![3]).unwrap();
    /// let b = a.exp().unwrap();
    ///
    /// let data = b.to_vec_f32();
    /// assert!((data[0] - 1.0).abs() < 1e-6);    // e^0 = 1
    /// assert!((data[1] - 2.71828).abs() < 1e-4); // e^1 ≈ 2.71828
    /// ```
    pub fn exp(&self) -> Result<Tensor> {
        let data = self.to_vec_f32();
        let result: Vec<f32> = data.iter().map(|x| x.exp()).collect();
        Tensor::new(result, self.shape().to_vec())
    }

    /// Reduce sum along a dimension.
    ///
    /// Sums elements along the specified dimension, removing that dimension
    /// from the output shape.
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimension to sum along
    ///
    /// # Returns
    ///
    /// Tensor with the specified dimension removed.
    ///
    /// # Errors
    ///
    /// Returns `DimOutOfRange` if `dim` is out of range.
    ///
    /// # Performance
    ///
    /// PERF: Naive multi-index iteration. Phase 3 will optimize with SIMD.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    ///
    /// // Sum along dim 0: [1+4, 2+5, 3+6] = [5, 7, 9]
    /// let s0 = t.sum(0).unwrap();
    /// assert_eq!(s0.shape(), &[3]);
    /// assert_eq!(s0.to_vec_f32(), vec![5.0, 7.0, 9.0]);
    ///
    /// // Sum along dim 1: [1+2+3, 4+5+6] = [6, 15]
    /// let s1 = t.sum(1).unwrap();
    /// assert_eq!(s1.shape(), &[2]);
    /// assert_eq!(s1.to_vec_f32(), vec![6.0, 15.0]);
    /// ```
    #[allow(clippy::needless_range_loop)]
    pub fn sum(&self, dim: usize) -> Result<Tensor> {
        if dim >= self.ndim() {
            return Err(LludaError::DimOutOfRange {
                dim,
                ndim: self.ndim(),
            });
        }

        let data = self.to_vec_f32();
        let ndim = self.ndim();

        // Build output shape (remove the summed dimension)
        let out_shape: Vec<usize> = self
            .shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != dim)
            .map(|(_, &s)| s)
            .collect();

        // Handle edge case: reducing a 1D tensor -> scalar-like [1]
        let out_shape = if out_shape.is_empty() {
            vec![1]
        } else {
            out_shape
        };

        let out_numel: usize = out_shape.iter().product();
        let mut result = vec![0.0f32; out_numel];

        let dim_size = self.shape[dim];
        let src_strides = &self.strides;

        // For each output element, sum along the reduce dimension
        let mut out_multi_idx = vec![0usize; out_shape.len()];

        for flat_out in 0..out_numel {
            if flat_out > 0 {
                let mut carry = true;
                for d in (0..out_shape.len()).rev() {
                    if carry {
                        out_multi_idx[d] += 1;
                        if out_multi_idx[d] >= out_shape[d] {
                            out_multi_idx[d] = 0;
                        } else {
                            carry = false;
                        }
                    }
                }
            }

            // Map output multi-index to source multi-index (insert dim at position)
            let mut sum = 0.0f32;
            for k in 0..dim_size {
                let mut src_flat = 0usize;
                let mut out_d = 0;
                for d in 0..ndim {
                    if d == dim {
                        src_flat += k * src_strides[d];
                    } else {
                        src_flat += out_multi_idx[out_d] * src_strides[d];
                        out_d += 1;
                    }
                }
                sum += data[src_flat];
            }
            result[flat_out] = sum;
        }

        Tensor::new(result, out_shape)
    }

    /// Reduce sum along a dimension, keeping the dimension with size 1.
    ///
    /// Like `sum(dim)` but the output has the same number of dimensions
    /// with the reduced dimension set to size 1. Useful for broadcasting.
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimension to sum along
    ///
    /// # Returns
    ///
    /// Tensor with the specified dimension set to size 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    /// let s = t.sum_keepdim(1).unwrap();
    /// assert_eq!(s.shape(), &[2, 1]);
    /// assert_eq!(s.to_vec_f32(), vec![6.0, 15.0]);
    /// ```
    pub fn sum_keepdim(&self, dim: usize) -> Result<Tensor> {
        let reduced = self.sum(dim)?;
        // Re-insert the dimension as size 1
        let mut new_shape = self.shape.clone();
        new_shape[dim] = 1;
        reduced.reshape(&new_shape)
    }

    /// Reduce max along a dimension.
    ///
    /// Finds the maximum element along the specified dimension, removing
    /// that dimension from the output shape.
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimension to find max along
    ///
    /// # Returns
    ///
    /// Tensor with the specified dimension removed.
    ///
    /// # Errors
    ///
    /// Returns `DimOutOfRange` if `dim` is out of range.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let t = Tensor::new(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3]).unwrap();
    ///
    /// // Max along dim 1: [max(1,5,3), max(4,2,6)] = [5, 6]
    /// let m = t.max(1).unwrap();
    /// assert_eq!(m.shape(), &[2]);
    /// assert_eq!(m.to_vec_f32(), vec![5.0, 6.0]);
    /// ```
    #[allow(clippy::needless_range_loop)]
    pub fn max(&self, dim: usize) -> Result<Tensor> {
        if dim >= self.ndim() {
            return Err(LludaError::DimOutOfRange {
                dim,
                ndim: self.ndim(),
            });
        }

        let data = self.to_vec_f32();
        let ndim = self.ndim();

        // Build output shape (remove the max dimension)
        let out_shape: Vec<usize> = self
            .shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != dim)
            .map(|(_, &s)| s)
            .collect();

        let out_shape = if out_shape.is_empty() {
            vec![1]
        } else {
            out_shape
        };

        let out_numel: usize = out_shape.iter().product();
        let mut result = vec![f32::NEG_INFINITY; out_numel];

        let dim_size = self.shape[dim];
        let src_strides = &self.strides;

        let mut out_multi_idx = vec![0usize; out_shape.len()];

        for flat_out in 0..out_numel {
            if flat_out > 0 {
                let mut carry = true;
                for d in (0..out_shape.len()).rev() {
                    if carry {
                        out_multi_idx[d] += 1;
                        if out_multi_idx[d] >= out_shape[d] {
                            out_multi_idx[d] = 0;
                        } else {
                            carry = false;
                        }
                    }
                }
            }

            let mut max_val = f32::NEG_INFINITY;
            for k in 0..dim_size {
                let mut src_flat = 0usize;
                let mut out_d = 0;
                for d in 0..ndim {
                    if d == dim {
                        src_flat += k * src_strides[d];
                    } else {
                        src_flat += out_multi_idx[out_d] * src_strides[d];
                        out_d += 1;
                    }
                }
                let val = data[src_flat];
                if val > max_val {
                    max_val = val;
                }
            }
            result[flat_out] = max_val;
        }

        Tensor::new(result, out_shape)
    }

    /// Reduce max along a dimension, keeping the dimension with size 1.
    ///
    /// Like `max(dim)` but the output has the same number of dimensions.
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimension to find max along
    ///
    /// # Returns
    ///
    /// Tensor with the specified dimension set to size 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::tensor::Tensor;
    ///
    /// let t = Tensor::new(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3]).unwrap();
    /// let m = t.max_keepdim(1).unwrap();
    /// assert_eq!(m.shape(), &[2, 1]);
    /// assert_eq!(m.to_vec_f32(), vec![5.0, 6.0]);
    /// ```
    pub fn max_keepdim(&self, dim: usize) -> Result<Tensor> {
        let reduced = self.max(dim)?;
        let mut new_shape = self.shape.clone();
        new_shape[dim] = 1;
        reduced.reshape(&new_shape)
    }

    /// Create a tensor filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `shape` - Desired tensor shape
    /// * `dtype` - Data type (F32 or BF16)
    ///
    /// # Returns
    ///
    /// Tensor filled with zeros in the specified dtype.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lluda_inference::tensor::{Tensor, DType};
    ///
    /// let t = Tensor::zeros(&[2, 3], DType::F32).unwrap();
    /// assert_eq!(t.shape(), &[2, 3]);
    /// assert_eq!(t.dtype(), DType::F32);
    /// assert_eq!(t.to_vec_f32(), vec![0.0; 6]);
    /// ```
    pub fn zeros(shape: &[usize], dtype: DType) -> Result<Tensor> {
        let numel: usize = shape.iter().product();
        match dtype {
            DType::F32 => Tensor::new(vec![0.0f32; numel], shape.to_vec()),
            DType::BF16 => {
                Tensor::from_bf16(vec![BF16::from(0.0f32); numel], shape.to_vec())
            }
        }
    }
}

/// Compute broadcast shape from two shapes using NumPy broadcasting rules.
///
/// Rules:
/// 1. Align shapes from the right
/// 2. Each dimension must be equal, or one of them is 1
/// 3. Missing dimensions are treated as 1
///
/// Returns the broadcast output shape, or error if shapes are incompatible.
fn broadcast_shape(a_shape: &[usize], b_shape: &[usize]) -> Result<Vec<usize>> {
    let max_ndim = a_shape.len().max(b_shape.len());
    let mut result = vec![0usize; max_ndim];

    for i in 0..max_ndim {
        let a_dim = if i < a_shape.len() {
            a_shape[a_shape.len() - 1 - i]
        } else {
            1
        };
        let b_dim = if i < b_shape.len() {
            b_shape[b_shape.len() - 1 - i]
        } else {
            1
        };

        if a_dim == b_dim {
            result[max_ndim - 1 - i] = a_dim;
        } else if a_dim == 1 {
            result[max_ndim - 1 - i] = b_dim;
        } else if b_dim == 1 {
            result[max_ndim - 1 - i] = a_dim;
        } else {
            return Err(LludaError::ShapeMismatch {
                expected: a_shape.to_vec(),
                got: b_shape.to_vec(),
            });
        }
    }

    Ok(result)
}

/// Apply a binary operation element-wise with NumPy-style broadcasting.
///
/// Both inputs must already be F32 data. The function broadcasts as needed
/// and applies the operation to produce the output.
///
/// PERF: Naive multi-index iteration. Phase 3 will optimize with
/// stride-based broadcasting (no data expansion).
#[allow(clippy::needless_range_loop)]
fn broadcast_binary_op(
    a_data: &[f32],
    a_shape: &[usize],
    b_data: &[f32],
    b_shape: &[usize],
    op: fn(f32, f32) -> f32,
) -> Result<Tensor> {
    let out_shape = broadcast_shape(a_shape, b_shape)?;
    let out_numel: usize = out_shape.iter().product();
    let ndim = out_shape.len();

    // Pad shapes to same length (prepend 1s)
    let a_padded: Vec<usize> = {
        let pad = ndim - a_shape.len();
        let mut s = vec![1; pad];
        s.extend_from_slice(a_shape);
        s
    };
    let b_padded: Vec<usize> = {
        let pad = ndim - b_shape.len();
        let mut s = vec![1; pad];
        s.extend_from_slice(b_shape);
        s
    };

    let a_strides = compute_strides(&a_padded);
    let b_strides = compute_strides(&b_padded);

    let mut result = vec![0.0f32; out_numel];

    // Iterate over all output elements
    let mut multi_idx = vec![0usize; ndim];
    for flat_out in 0..out_numel {
        if flat_out > 0 {
            // Increment multi-index (odometer)
            let mut carry = true;
            for d in (0..ndim).rev() {
                if carry {
                    multi_idx[d] += 1;
                    if multi_idx[d] >= out_shape[d] {
                        multi_idx[d] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
        }

        // Compute source indices (clamp to 0 for broadcast dims of size 1)
        let a_flat: usize = multi_idx
            .iter()
            .enumerate()
            .map(|(d, &idx)| {
                let a_idx = if a_padded[d] == 1 { 0 } else { idx };
                a_idx * a_strides[d]
            })
            .sum();
        let b_flat: usize = multi_idx
            .iter()
            .enumerate()
            .map(|(d, &idx)| {
                let b_idx = if b_padded[d] == 1 { 0 } else { idx };
                b_idx * b_strides[d]
            })
            .sum();

        result[flat_out] = op(a_data[a_flat], b_data[b_flat]);
    }

    Tensor::new(result, out_shape)
}

/// Compute row-major (C-contiguous) strides for a given shape.
///
/// In row-major layout, the last dimension has stride 1, and each preceding
/// dimension's stride is the product of all subsequent dimension sizes.
///
/// # Arguments
///
/// * `shape` - Tensor dimensions
///
/// # Returns
///
/// Vector of strides, same length as shape.
///
/// # Example
///
/// ```ignore
/// compute_strides(&[2, 3, 4]) => [12, 4, 1]
/// compute_strides(&[5]) => [1]
/// compute_strides(&[]) => []
/// ```
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    if ndim == 0 {
        return vec![];
    }

    let mut strides = vec![1; ndim];

    // Work backwards from the last dimension
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    strides
}

/// 2D matrix multiplication: [M, K] @ [K, N] -> [M, N]
///
/// # Algorithm
///
/// Naive triple-loop:
/// ```ignore
/// for i in 0..M {
///     for j in 0..N {
///         C[i,j] = sum_k(A[i,k] * B[k,j])
///     }
/// }
/// ```
///
/// PERF: O(M*N*K) complexity. No blocking, no SIMD. Sufficient for Phase 0.
fn matmul_2d(
    lhs_data: &[f32],
    lhs_shape: &[usize],
    rhs_data: &[f32],
    rhs_shape: &[usize],
) -> Result<Tensor> {
    let m = lhs_shape[0];
    let k_lhs = lhs_shape[1];
    let k_rhs = rhs_shape[0];
    let n = rhs_shape[1];

    // Validate inner dimensions match
    if k_lhs != k_rhs {
        return Err(LludaError::ShapeMismatch {
            expected: vec![m, k_rhs, k_rhs, n],
            got: vec![m, k_lhs, k_rhs, n],
        });
    }

    let k = k_lhs;
    let mut result = vec![0.0f32; m * n];

    // Naive triple-loop matmul
    // C[i,j] = sum_k A[i,k] * B[k,j]
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for k_idx in 0..k {
                let a_val = lhs_data[i * k + k_idx];
                let b_val = rhs_data[k_idx * n + j];
                sum += a_val * b_val;
            }
            result[i * n + j] = sum;
        }
    }

    Tensor::new(result, vec![m, n])
}

/// 3D batched matrix multiplication: [B, M, K] @ [B, K, N] -> [B, M, N]
///
/// Applies 2D matmul to each batch independently.
///
/// # Algorithm
///
/// ```ignore
/// for b in 0..B {
///     output[b] = matmul_2d(lhs[b], rhs[b])
/// }
/// ```
fn matmul_3d(
    lhs_data: &[f32],
    lhs_shape: &[usize],
    rhs_data: &[f32],
    rhs_shape: &[usize],
) -> Result<Tensor> {
    let b_lhs = lhs_shape[0];
    let b_rhs = rhs_shape[0];
    let m = lhs_shape[1];
    let k_lhs = lhs_shape[2];
    let k_rhs = rhs_shape[1];
    let n = rhs_shape[2];

    // Validate batch dimensions match
    if b_lhs != b_rhs {
        return Err(LludaError::ShapeMismatch {
            expected: vec![b_lhs, m, k_lhs],
            got: vec![b_rhs, k_rhs, n],
        });
    }

    // Validate inner dimensions match
    if k_lhs != k_rhs {
        return Err(LludaError::ShapeMismatch {
            expected: vec![b_lhs, m, k_rhs, k_rhs, n],
            got: vec![b_lhs, m, k_lhs, k_rhs, n],
        });
    }

    let b = b_lhs;
    let k = k_lhs;
    let mut result = vec![0.0f32; b * m * n];

    // Process each batch
    let lhs_batch_size = m * k;
    let rhs_batch_size = k * n;
    let out_batch_size = m * n;

    for batch_idx in 0..b {
        let lhs_offset = batch_idx * lhs_batch_size;
        let rhs_offset = batch_idx * rhs_batch_size;
        let out_offset = batch_idx * out_batch_size;

        // Slice for this batch
        let lhs_batch = &lhs_data[lhs_offset..lhs_offset + lhs_batch_size];
        let rhs_batch = &rhs_data[rhs_offset..rhs_offset + rhs_batch_size];

        // Perform 2D matmul for this batch
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for k_idx in 0..k {
                    let a_val = lhs_batch[i * k + k_idx];
                    let b_val = rhs_batch[k_idx * n + j];
                    sum += a_val * b_val;
                }
                result[out_offset + i * n + j] = sum;
            }
        }
    }

    Tensor::new(result, vec![b, m, n])
}

/// 4D batched matrix multiplication: [B, H, M, K] @ [B, H, K, N] -> [B, H, M, N]
///
/// Applies 2D matmul to each (batch, head) pair independently.
/// This is the core operation for multi-head attention: Q @ K^T and attn @ V.
///
/// # Algorithm
///
/// ```ignore
/// for b in 0..B {
///     for h in 0..H {
///         output[b, h] = matmul_2d(lhs[b, h], rhs[b, h])
///     }
/// }
/// ```
///
/// PERF: O(B*H*M*N*K) complexity. No blocking, no SIMD. Sufficient for Phase 0.
fn matmul_4d(
    lhs_data: &[f32],
    lhs_shape: &[usize],
    rhs_data: &[f32],
    rhs_shape: &[usize],
) -> Result<Tensor> {
    let batch = lhs_shape[0];
    let heads = lhs_shape[1];
    let m = lhs_shape[2];
    let k_lhs = lhs_shape[3];
    let b_batch = rhs_shape[0];
    let b_heads = rhs_shape[1];
    let k_rhs = rhs_shape[2];
    let n = rhs_shape[3];

    // Validate batch dimensions match
    if batch != b_batch {
        return Err(LludaError::ShapeMismatch {
            expected: vec![batch, heads, m, k_lhs],
            got: vec![b_batch, b_heads, k_rhs, n],
        });
    }

    // Validate head dimensions match
    if heads != b_heads {
        return Err(LludaError::ShapeMismatch {
            expected: vec![batch, heads, m, k_lhs],
            got: vec![b_batch, b_heads, k_rhs, n],
        });
    }

    // Validate inner dimensions match
    if k_lhs != k_rhs {
        return Err(LludaError::ShapeMismatch {
            expected: vec![batch, heads, m, k_rhs],
            got: vec![batch, heads, m, k_lhs],
        });
    }

    let k = k_lhs;
    let mut result = vec![0.0f32; batch * heads * m * n];

    let lhs_head_size = m * k;
    let rhs_head_size = k * n;
    let out_head_size = m * n;

    for b in 0..batch {
        for h in 0..heads {
            let a_offset = (b * heads + h) * lhs_head_size;
            let b_offset = (b * heads + h) * rhs_head_size;
            let r_offset = (b * heads + h) * out_head_size;

            let a_slice = &lhs_data[a_offset..a_offset + lhs_head_size];
            let b_slice = &rhs_data[b_offset..b_offset + rhs_head_size];

            // Inline 2D matmul for this (batch, head) slice
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for k_idx in 0..k {
                        sum += a_slice[i * k + k_idx] * b_slice[k_idx * n + j];
                    }
                    result[r_offset + i * n + j] = sum;
                }
            }
        }
    }

    Tensor::new(result, vec![batch, heads, m, n])
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Construction Tests ==========

    #[test]
    fn test_new_f32_tensor() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::new(data.clone(), vec![2, 3]).unwrap();

        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.ndim(), 2);

        let recovered = t.to_vec_f32();
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_from_bf16_tensor() {
        let data = vec![
            BF16::from(1.0f32),
            BF16::from(2.0f32),
            BF16::from(3.0f32),
            BF16::from(4.0f32),
        ];
        let t = Tensor::from_bf16(data, vec![2, 2]).unwrap();

        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.dtype(), DType::BF16);
        assert_eq!(t.numel(), 4);
        assert_eq!(t.ndim(), 2);

        let recovered = t.to_vec_f32();
        assert_eq!(recovered.len(), 4);
        assert_eq!(recovered[0], 1.0f32);
        assert_eq!(recovered[1], 2.0f32);
    }

    #[test]
    fn test_shape_mismatch_error() {
        // Data has 4 elements but shape requires 6
        let result = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 3]);

        assert!(result.is_err());
        match result.unwrap_err() {
            LludaError::ShapeMismatch { expected, got } => {
                assert_eq!(expected, vec![6]);
                assert_eq!(got, vec![4]);
            }
            _ => panic!("Expected ShapeMismatch error"),
        }
    }

    #[test]
    fn test_bf16_shape_mismatch_error() {
        let data = vec![BF16::from(1.0f32), BF16::from(2.0f32)];
        let result = Tensor::from_bf16(data, vec![3]);

        assert!(result.is_err());
        match result.unwrap_err() {
            LludaError::ShapeMismatch { expected, got } => {
                assert_eq!(expected, vec![3]);
                assert_eq!(got, vec![2]);
            }
            _ => panic!("Expected ShapeMismatch error"),
        }
    }

    // ========== Shape Query Tests ==========

    #[test]
    fn test_shape_queries_scalar() {
        let t = Tensor::new(vec![42.0], vec![1]).unwrap();
        assert_eq!(t.shape(), &[1]);
        assert_eq!(t.numel(), 1);
        assert_eq!(t.ndim(), 1);
    }

    #[test]
    fn test_shape_queries_vector() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        assert_eq!(t.shape(), &[5]);
        assert_eq!(t.numel(), 5);
        assert_eq!(t.ndim(), 1);
    }

    #[test]
    fn test_shape_queries_matrix() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.ndim(), 2);
    }

    #[test]
    fn test_shape_queries_3d() {
        let data = vec![1.0; 24]; // 2*3*4 = 24
        let t = Tensor::new(data, vec![2, 3, 4]).unwrap();
        assert_eq!(t.shape(), &[2, 3, 4]);
        assert_eq!(t.numel(), 24);
        assert_eq!(t.ndim(), 3);
    }

    // ========== Strides Tests ==========

    #[test]
    fn test_strides_computation_1d() {
        let strides = compute_strides(&[5]);
        assert_eq!(strides, vec![1]);
    }

    #[test]
    fn test_strides_computation_2d() {
        let strides = compute_strides(&[2, 3]);
        // Last dim stride = 1
        // First dim stride = 3 * 1 = 3
        assert_eq!(strides, vec![3, 1]);
    }

    #[test]
    fn test_strides_computation_3d() {
        let strides = compute_strides(&[2, 3, 4]);
        // Last dim (4): stride = 1
        // Middle dim (3): stride = 4 * 1 = 4
        // First dim (2): stride = 3 * 4 = 12
        assert_eq!(strides, vec![12, 4, 1]);
    }

    #[test]
    fn test_strides_computation_4d() {
        let strides = compute_strides(&[2, 3, 4, 5]);
        // Working backwards:
        // dim 3 (5): stride = 1
        // dim 2 (4): stride = 5 * 1 = 5
        // dim 1 (3): stride = 4 * 5 = 20
        // dim 0 (2): stride = 3 * 20 = 60
        assert_eq!(strides, vec![60, 20, 5, 1]);
    }

    #[test]
    fn test_strides_empty_shape() {
        let strides = compute_strides(&[]);
        let expected: Vec<usize> = vec![];
        assert_eq!(strides, expected);
    }

    #[test]
    fn test_tensor_strides_stored_correctly() {
        let t = Tensor::new(vec![0.0; 24], vec![2, 3, 4]).unwrap();
        assert_eq!(t.strides, vec![12, 4, 1]);
    }

    // ========== DType Conversion Tests ==========

    #[test]
    fn test_f32_to_bf16_conversion() {
        let t_f32 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let t_bf16 = t_f32.to_dtype(DType::BF16).unwrap();

        assert_eq!(t_bf16.dtype(), DType::BF16);
        assert_eq!(t_bf16.shape(), &[3]);

        let recovered = t_bf16.to_vec_f32();
        assert_eq!(recovered.len(), 3);
        // BF16 precision is ~3 decimal places
        assert!((recovered[0] - 1.0).abs() < 0.01);
        assert!((recovered[1] - 2.0).abs() < 0.01);
        assert!((recovered[2] - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_bf16_to_f32_conversion() {
        let data = vec![
            BF16::from(1.5f32),
            BF16::from(2.5f32),
            BF16::from(3.5f32),
        ];
        let t_bf16 = Tensor::from_bf16(data, vec![3]).unwrap();
        let t_f32 = t_bf16.to_dtype(DType::F32).unwrap();

        assert_eq!(t_f32.dtype(), DType::F32);
        assert_eq!(t_f32.shape(), &[3]);

        let recovered = t_f32.to_vec_f32();
        assert_eq!(recovered.len(), 3);
        assert!((recovered[0] - 1.5).abs() < 0.01);
        assert!((recovered[1] - 2.5).abs() < 0.01);
        assert!((recovered[2] - 3.5).abs() < 0.01);
    }

    #[test]
    fn test_f32_to_bf16_to_f32_round_trip() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let t_f32 = Tensor::new(original.clone(), vec![5]).unwrap();
        let t_bf16 = t_f32.to_dtype(DType::BF16).unwrap();
        let t_f32_recovered = t_bf16.to_dtype(DType::F32).unwrap();

        assert_eq!(t_f32_recovered.dtype(), DType::F32);

        let recovered = t_f32_recovered.to_vec_f32();
        assert_eq!(recovered.len(), 5);

        // BF16 should preserve values within tolerance
        for (i, &orig) in original.iter().enumerate() {
            assert!(
                (recovered[i] - orig).abs() < 0.01,
                "Value {} differs too much: {} vs {}",
                i,
                orig,
                recovered[i]
            );
        }
    }

    #[test]
    fn test_to_dtype_same_dtype_is_noop() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let t_same = t.to_dtype(DType::F32).unwrap();

        // Should return a clone (same values)
        assert_eq!(t.dtype(), t_same.dtype());
        assert_eq!(t.to_vec_f32(), t_same.to_vec_f32());
    }

    // ========== Edge Cases ==========

    #[test]
    fn test_single_element_tensor() {
        let t = Tensor::new(vec![42.0], vec![1]).unwrap();
        assert_eq!(t.numel(), 1);
        assert_eq!(t.to_vec_f32(), vec![42.0]);
        assert_eq!(t.strides, vec![1]);
    }

    #[test]
    fn test_large_tensor_construction() {
        // Just test that we can create a large tensor without errors
        let size = 1000 * 1000;
        let data = vec![1.0f32; size];
        let t = Tensor::new(data, vec![1000, 1000]).unwrap();

        assert_eq!(t.numel(), size);
        assert_eq!(t.shape(), &[1000, 1000]);
        assert_eq!(t.strides, vec![1000, 1]);
    }

    // ========== Data Access Tests ==========

    #[test]
    fn test_to_vec_f32_from_f32_tensor() {
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let t = Tensor::new(original.clone(), vec![4]).unwrap();

        let data = t.to_vec_f32();
        assert_eq!(data, original);
    }

    #[test]
    fn test_to_vec_f32_from_bf16_tensor() {
        let bf16_data = vec![
            BF16::from(1.0f32),
            BF16::from(2.0f32),
            BF16::from(3.0f32),
        ];
        let t = Tensor::from_bf16(bf16_data, vec![3]).unwrap();

        let data = t.to_vec_f32();
        assert_eq!(data.len(), 3);
        assert_eq!(data[0], 1.0f32);
        assert_eq!(data[1], 2.0f32);
        assert_eq!(data[2], 3.0f32);
    }

    // ========== Clone Test ==========

    #[test]
    fn test_clone_tensor() {
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let t2 = t1.clone();

        assert_eq!(t1.shape(), t2.shape());
        assert_eq!(t1.dtype(), t2.dtype());
        assert_eq!(t1.to_vec_f32(), t2.to_vec_f32());
    }

    // ========== Matrix Multiplication Tests ==========

    #[test]
    fn test_matmul_2d_basic() {
        // Test case from spec:
        // [[1, 2], [3, 4]] @ [[5, 6], [7, 8]]
        // = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        // = [[19, 22], [43, 50]]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let c = a.matmul(&b).unwrap();

        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.dtype(), DType::F32);

        let result = c.to_vec_f32();
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_2d_non_square() {
        // [3, 2] @ [2, 4] -> [3, 4]
        let a = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], // 3x2
            vec![3, 2],
        )
        .unwrap();
        let b = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], // 2x4
            vec![2, 4],
        )
        .unwrap();
        let c = a.matmul(&b).unwrap();

        assert_eq!(c.shape(), &[3, 4]);
        assert_eq!(c.dtype(), DType::F32);

        let result = c.to_vec_f32();
        // First row: [1, 2] @ [[1,2,3,4], [5,6,7,8]] = [1*1+2*5, 1*2+2*6, 1*3+2*7, 1*4+2*8]
        //                                               = [11, 14, 17, 20]
        // Second row: [3, 4] @ ... = [3*1+4*5, 3*2+4*6, 3*3+4*7, 3*4+4*8]
        //                           = [23, 30, 37, 44]
        // Third row: [5, 6] @ ... = [5*1+6*5, 5*2+6*6, 5*3+6*7, 5*4+6*8]
        //                          = [35, 46, 57, 68]
        assert_eq!(
            result,
            vec![11.0, 14.0, 17.0, 20.0, 23.0, 30.0, 37.0, 44.0, 35.0, 46.0, 57.0, 68.0]
        );
    }

    #[test]
    fn test_matmul_2d_identity() {
        // A @ I = A
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let identity = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let c = a.matmul(&identity).unwrap();

        assert_eq!(c.shape(), &[2, 2]);
        let result = c.to_vec_f32();
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_matmul_2d_zero_matrix() {
        // A @ 0 = 0
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let zero = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let c = a.matmul(&zero).unwrap();

        assert_eq!(c.shape(), &[2, 2]);
        let result = c.to_vec_f32();
        assert_eq!(result, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_matmul_2d_shape_mismatch() {
        // [2, 3] @ [4, 5] should error (k mismatch: 3 != 4)
        let a = Tensor::new(vec![1.0; 6], vec![2, 3]).unwrap();
        let b = Tensor::new(vec![1.0; 20], vec![4, 5]).unwrap();
        let result = a.matmul(&b);

        assert!(result.is_err());
        match result.unwrap_err() {
            LludaError::ShapeMismatch { .. } => (),
            other => panic!("Expected ShapeMismatch, got: {:?}", other),
        }
    }

    #[test]
    fn test_matmul_3d_batched() {
        // Batched: [2, 2, 3] @ [2, 3, 2] -> [2, 2, 2]
        // Batch 0: [[1,2,3], [4,5,6]] @ [[1,2], [3,4], [5,6]]
        // Batch 1: [[7,8,9], [10,11,12]] @ [[1,2], [3,4], [5,6]]

        let lhs = Tensor::new(
            vec![
                // Batch 0
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // Batch 1
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![2, 2, 3],
        )
        .unwrap();

        let rhs = Tensor::new(
            vec![
                // Batch 0
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // Batch 1 (same weights)
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            ],
            vec![2, 3, 2],
        )
        .unwrap();

        let c = lhs.matmul(&rhs).unwrap();

        assert_eq!(c.shape(), &[2, 2, 2]);
        assert_eq!(c.dtype(), DType::F32);

        let result = c.to_vec_f32();

        // Batch 0, row 0: [1,2,3] @ [[1,2],[3,4],[5,6]] = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
        // Batch 0, row 1: [4,5,6] @ [[1,2],[3,4],[5,6]] = [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
        // Batch 1, row 0: [7,8,9] @ [[1,2],[3,4],[5,6]] = [7*1+8*3+9*5, 7*2+8*4+9*6] = [76, 100]
        // Batch 1, row 1: [10,11,12] @ [[1,2],[3,4],[5,6]] = [10*1+11*3+12*5, 10*2+11*4+12*6]
        //                                                     = [103, 136]
        assert_eq!(
            result,
            vec![22.0, 28.0, 49.0, 64.0, 76.0, 100.0, 103.0, 136.0]
        );
    }

    #[test]
    fn test_matmul_3d_batch_mismatch() {
        // [2, 2, 3] @ [3, 3, 2] should error (batch size mismatch)
        let a = Tensor::new(vec![1.0; 12], vec![2, 2, 3]).unwrap();
        let b = Tensor::new(vec![1.0; 18], vec![3, 3, 2]).unwrap();
        let result = a.matmul(&b);

        assert!(result.is_err());
        match result.unwrap_err() {
            LludaError::ShapeMismatch { .. } => (),
            other => panic!("Expected ShapeMismatch, got: {:?}", other),
        }
    }

    #[test]
    fn test_matmul_bf16_inputs() {
        // BF16 @ BF16 -> F32 (auto-convert)
        let a_bf16 = Tensor::from_bf16(
            vec![
                BF16::from(1.0f32),
                BF16::from(2.0f32),
                BF16::from(3.0f32),
                BF16::from(4.0f32),
            ],
            vec![2, 2],
        )
        .unwrap();

        let b_bf16 = Tensor::from_bf16(
            vec![
                BF16::from(5.0f32),
                BF16::from(6.0f32),
                BF16::from(7.0f32),
                BF16::from(8.0f32),
            ],
            vec![2, 2],
        )
        .unwrap();

        let c = a_bf16.matmul(&b_bf16).unwrap();

        // Result should be F32
        assert_eq!(c.dtype(), DType::F32);
        assert_eq!(c.shape(), &[2, 2]);

        let result = c.to_vec_f32();
        // Same expected values as test_matmul_2d_basic
        // BF16 precision may introduce small errors, but these values should be exact
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_mixed_dtypes() {
        // F32 @ BF16 -> F32
        let a_f32 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b_bf16 = Tensor::from_bf16(
            vec![
                BF16::from(5.0f32),
                BF16::from(6.0f32),
                BF16::from(7.0f32),
                BF16::from(8.0f32),
            ],
            vec![2, 2],
        )
        .unwrap();

        let c = a_f32.matmul(&b_bf16).unwrap();

        assert_eq!(c.dtype(), DType::F32);
        let result = c.to_vec_f32();
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_1d_tensor_error() {
        // 1D tensor should error (need at least 2D)
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]).unwrap();
        let result = a.matmul(&b);

        assert!(result.is_err());
        match result.unwrap_err() {
            LludaError::ShapeMismatch { .. } => (),
            other => panic!("Expected ShapeMismatch, got: {:?}", other),
        }
    }

    #[test]
    fn test_matmul_4d_batched() {
        // [1, 2, 2, 3] @ [1, 2, 3, 2] -> [1, 2, 2, 2]
        // batch=1, heads=2, M=2, K=3, N=2
        let a = Tensor::new(
            vec![
                // batch 0, head 0: [[1,2,3],[4,5,6]]
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                // batch 0, head 1: [[7,8,9],[10,11,12]]
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![1, 2, 2, 3],
        )
        .unwrap();

        let b = Tensor::new(
            vec![
                // batch 0, head 0: [[1,2],[3,4],[5,6]]
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                // batch 0, head 1: [[1,0],[0,1],[1,0]]
                1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
            ],
            vec![1, 2, 3, 2],
        )
        .unwrap();

        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[1, 2, 2, 2]);
        assert_eq!(c.dtype(), DType::F32);

        let result = c.to_vec_f32();
        // Head 0: [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]]
        //   row 0: [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
        //   row 1: [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
        // Head 1: [[7,8,9],[10,11,12]] @ [[1,0],[0,1],[1,0]]
        //   row 0: [7*1+8*0+9*1, 7*0+8*1+9*0] = [16, 8]
        //   row 1: [10*1+11*0+12*1, 10*0+11*1+12*0] = [22, 11]
        assert_eq!(
            result,
            vec![22.0, 28.0, 49.0, 64.0, 16.0, 8.0, 22.0, 11.0]
        );
    }

    #[test]
    fn test_matmul_4d_shape_mismatch() {
        // Batch mismatch: [2, 2, 2, 3] @ [1, 2, 3, 2]
        let a = Tensor::new(vec![1.0; 24], vec![2, 2, 2, 3]).unwrap();
        let b = Tensor::new(vec![1.0; 12], vec![1, 2, 3, 2]).unwrap();
        let result = a.matmul(&b);
        assert!(result.is_err());

        // Inner dim mismatch: [1, 2, 2, 3] @ [1, 2, 4, 2]
        let a = Tensor::new(vec![1.0; 12], vec![1, 2, 2, 3]).unwrap();
        let b = Tensor::new(vec![1.0; 16], vec![1, 2, 4, 2]).unwrap();
        let result = a.matmul(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_matmul_5d_unsupported() {
        // 5D not supported
        let a = Tensor::new(vec![1.0; 48], vec![2, 2, 2, 2, 3]).unwrap();
        let b = Tensor::new(vec![1.0; 48], vec![2, 2, 2, 3, 2]).unwrap();
        let result = a.matmul(&b);

        assert!(result.is_err());
        match result.unwrap_err() {
            LludaError::Msg(msg) => {
                assert!(msg.contains("unsupported dimensions"));
            }
            other => panic!("Expected Msg error, got: {:?}", other),
        }
    }

    #[test]
    fn test_matmul_result_always_f32() {
        // Verify that result is always F32, never BF16
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let c = a.matmul(&b).unwrap();

        assert_eq!(c.dtype(), DType::F32, "matmul result must be F32");
    }

    // ========== Element-wise Addition Tests ==========

    #[test]
    fn test_add_same_shape() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]).unwrap();
        let c = a.add(&b).unwrap();

        assert_eq!(c.to_vec_f32(), vec![5.0, 7.0, 9.0]);
        assert_eq!(c.dtype(), DType::F32);
    }

    #[test]
    fn test_add_2d() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let c = a.add(&b).unwrap();

        assert_eq!(c.to_vec_f32(), vec![6.0, 8.0, 10.0, 12.0]);
        assert_eq!(c.shape(), &[2, 2]);
    }

    #[test]
    fn test_add_scalar_broadcast() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::new(vec![10.0], vec![1]).unwrap();
        let c = a.add(&b).unwrap();

        assert_eq!(c.to_vec_f32(), vec![11.0, 12.0, 13.0]);
        assert_eq!(c.shape(), &[3]);
    }

    #[test]
    fn test_add_scalar_broadcast_reverse() {
        let a = Tensor::new(vec![10.0], vec![1]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let c = a.add(&b).unwrap();

        assert_eq!(c.to_vec_f32(), vec![11.0, 12.0, 13.0]);
        assert_eq!(c.shape(), &[3]);
    }

    #[test]
    fn test_add_trailing_dim_broadcast() {
        // [2, 3] + [3] -> broadcast [3] to [2, 3]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]).unwrap();
        let c = a.add(&b).unwrap();

        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.to_vec_f32(), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn test_add_middle_dim_broadcast() {
        // [2, 1, 3] + [2, 4, 3] -> broadcast dim 1
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 1, 3]).unwrap();
        let b = Tensor::new(vec![0.0; 24], vec![2, 4, 3]).unwrap();
        let c = a.add(&b).unwrap();

        assert_eq!(c.shape(), &[2, 4, 3]);
        // First batch element [1,2,3] repeated 4 times, second [4,5,6] repeated 4 times
        let data = c.to_vec_f32();
        assert_eq!(&data[0..3], &[1.0, 2.0, 3.0]);
        assert_eq!(&data[3..6], &[1.0, 2.0, 3.0]);
        assert_eq!(&data[12..15], &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_add_shape_mismatch() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::new(vec![4.0, 5.0], vec![2]).unwrap();
        let result = a.add(&b);

        assert!(result.is_err());
        match result.unwrap_err() {
            LludaError::ShapeMismatch { .. } => (),
            other => panic!("Expected ShapeMismatch, got: {:?}", other),
        }
    }

    #[test]
    fn test_add_bf16_inputs() {
        let a = Tensor::from_bf16(
            vec![BF16::from(1.0f32), BF16::from(2.0f32), BF16::from(3.0f32)],
            vec![3],
        )
        .unwrap();
        let b = Tensor::from_bf16(
            vec![BF16::from(4.0f32), BF16::from(5.0f32), BF16::from(6.0f32)],
            vec![3],
        )
        .unwrap();
        let c = a.add(&b).unwrap();

        assert_eq!(c.dtype(), DType::F32);
        assert_eq!(c.to_vec_f32(), vec![5.0, 7.0, 9.0]);
    }

    // ========== Element-wise Multiplication Tests ==========

    #[test]
    fn test_mul_same_shape() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]).unwrap();
        let c = a.mul(&b).unwrap();

        assert_eq!(c.to_vec_f32(), vec![4.0, 10.0, 18.0]);
        assert_eq!(c.dtype(), DType::F32);
    }

    #[test]
    fn test_mul_2d() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();
        let c = a.mul(&b).unwrap();

        assert_eq!(c.to_vec_f32(), vec![2.0, 6.0, 12.0, 20.0]);
        assert_eq!(c.shape(), &[2, 2]);
    }

    #[test]
    fn test_mul_scalar_broadcast() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::new(vec![2.0], vec![1]).unwrap();
        let c = a.mul(&b).unwrap();

        assert_eq!(c.to_vec_f32(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_mul_scalar_broadcast_reverse() {
        let a = Tensor::new(vec![2.0], vec![1]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let c = a.mul(&b).unwrap();

        assert_eq!(c.to_vec_f32(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_mul_trailing_dim_broadcast() {
        // [2, 3] * [3] -> [2, 3]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]).unwrap();
        let c = a.mul(&b).unwrap();

        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.to_vec_f32(), vec![2.0, 6.0, 12.0, 8.0, 15.0, 24.0]);
    }

    #[test]
    fn test_mul_shape_mismatch() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::new(vec![4.0, 5.0], vec![2]).unwrap();
        let result = a.mul(&b);

        assert!(result.is_err());
        match result.unwrap_err() {
            LludaError::ShapeMismatch { .. } => (),
            other => panic!("Expected ShapeMismatch, got: {:?}", other),
        }
    }

    // ========== Element-wise Division Tests ==========

    #[test]
    fn test_div_same_shape() {
        let a = Tensor::new(vec![4.0, 10.0, 18.0], vec![3]).unwrap();
        let b = Tensor::new(vec![2.0, 5.0, 6.0], vec![3]).unwrap();
        let c = a.div(&b).unwrap();

        assert_eq!(c.to_vec_f32(), vec![2.0, 2.0, 3.0]);
        assert_eq!(c.dtype(), DType::F32);
    }

    #[test]
    fn test_div_scalar_broadcast() {
        let a = Tensor::new(vec![2.0, 4.0, 6.0], vec![3]).unwrap();
        let b = Tensor::new(vec![2.0], vec![1]).unwrap();
        let c = a.div(&b).unwrap();

        assert_eq!(c.to_vec_f32(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_div_scalar_broadcast_reverse() {
        let a = Tensor::new(vec![12.0], vec![1]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]).unwrap();
        let c = a.div(&b).unwrap();

        assert_eq!(c.to_vec_f32(), vec![6.0, 4.0, 3.0]);
    }

    #[test]
    fn test_div_by_zero() {
        // Division by zero produces Inf
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::new(vec![0.0, 0.0, 0.0], vec![3]).unwrap();
        let c = a.div(&b).unwrap();

        let result = c.to_vec_f32();
        assert!(result[0].is_infinite());
        assert!(result[1].is_infinite());
        assert!(result[2].is_infinite());
    }

    #[test]
    fn test_div_shape_mismatch() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::new(vec![4.0, 5.0], vec![2]).unwrap();
        let result = a.div(&b);

        assert!(result.is_err());
        match result.unwrap_err() {
            LludaError::ShapeMismatch { .. } => (),
            other => panic!("Expected ShapeMismatch, got: {:?}", other),
        }
    }

    // ========== Unary Operation Tests ==========

    #[test]
    fn test_neg() {
        let a = Tensor::new(vec![1.0, -2.0, 3.0], vec![3]).unwrap();
        let b = a.neg().unwrap();

        assert_eq!(b.to_vec_f32(), vec![-1.0, 2.0, -3.0]);
        assert_eq!(b.dtype(), DType::F32);
    }

    #[test]
    fn test_neg_2d() {
        let a = Tensor::new(vec![1.0, -2.0, 3.0, -4.0], vec![2, 2]).unwrap();
        let b = a.neg().unwrap();

        assert_eq!(b.to_vec_f32(), vec![-1.0, 2.0, -3.0, 4.0]);
        assert_eq!(b.shape(), &[2, 2]);
    }

    #[test]
    fn test_recip() {
        let a = Tensor::new(vec![2.0, 4.0, 8.0], vec![3]).unwrap();
        let b = a.recip().unwrap();

        assert_eq!(b.to_vec_f32(), vec![0.5, 0.25, 0.125]);
        assert_eq!(b.dtype(), DType::F32);
    }

    #[test]
    fn test_recip_2d() {
        let a = Tensor::new(vec![1.0, 2.0, 4.0, 5.0], vec![2, 2]).unwrap();
        let b = a.recip().unwrap();

        assert_eq!(b.to_vec_f32(), vec![1.0, 0.5, 0.25, 0.2]);
        assert_eq!(b.shape(), &[2, 2]);
    }

    #[test]
    fn test_recip_zero() {
        // Reciprocal of zero produces Inf
        let a = Tensor::new(vec![0.0], vec![1]).unwrap();
        let b = a.recip().unwrap();

        let result = b.to_vec_f32();
        assert!(result[0].is_infinite());
    }

    // ========== Scalar Operation Tests ==========

    #[test]
    fn test_add_scalar() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = a.add_scalar(5.0).unwrap();

        assert_eq!(b.to_vec_f32(), vec![6.0, 7.0, 8.0]);
        assert_eq!(b.dtype(), DType::F32);
    }

    #[test]
    fn test_add_scalar_negative() {
        let a = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]).unwrap();
        let b = a.add_scalar(-5.0).unwrap();

        assert_eq!(b.to_vec_f32(), vec![5.0, 15.0, 25.0]);
    }

    #[test]
    fn test_add_scalar_2d() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = a.add_scalar(10.0).unwrap();

        assert_eq!(b.to_vec_f32(), vec![11.0, 12.0, 13.0, 14.0]);
        assert_eq!(b.shape(), &[2, 2]);
    }

    #[test]
    fn test_mul_scalar() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = a.mul_scalar(2.0).unwrap();

        assert_eq!(b.to_vec_f32(), vec![2.0, 4.0, 6.0]);
        assert_eq!(b.dtype(), DType::F32);
    }

    #[test]
    fn test_mul_scalar_negative() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = a.mul_scalar(-2.0).unwrap();

        assert_eq!(b.to_vec_f32(), vec![-2.0, -4.0, -6.0]);
    }

    #[test]
    fn test_mul_scalar_2d() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = a.mul_scalar(3.0).unwrap();

        assert_eq!(b.to_vec_f32(), vec![3.0, 6.0, 9.0, 12.0]);
        assert_eq!(b.shape(), &[2, 2]);
    }

    #[test]
    fn test_mul_scalar_zero() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = a.mul_scalar(0.0).unwrap();

        assert_eq!(b.to_vec_f32(), vec![0.0, 0.0, 0.0]);
    }

    // ========== BF16 Input Tests for Element-wise Ops ==========

    #[test]
    fn test_elementwise_bf16_to_f32_conversion() {
        // Verify that BF16 inputs are automatically converted to F32
        let a = Tensor::from_bf16(
            vec![BF16::from(2.0f32), BF16::from(4.0f32)],
            vec![2],
        )
        .unwrap();
        let b = Tensor::from_bf16(
            vec![BF16::from(3.0f32), BF16::from(5.0f32)],
            vec![2],
        )
        .unwrap();

        let c = a.mul(&b).unwrap();
        assert_eq!(c.dtype(), DType::F32);
        assert_eq!(c.to_vec_f32(), vec![6.0, 20.0]);
    }

    // ========== Reshape Tests ==========

    #[test]
    fn test_reshape_basic() {
        let t = Tensor::new((1..=6).map(|x| x as f32).collect(), vec![2, 3]).unwrap();
        let r = t.reshape(&[3, 2]).unwrap();

        assert_eq!(r.shape(), &[3, 2]);
        // Data order should be preserved: [1, 2, 3, 4, 5, 6]
        assert_eq!(r.to_vec_f32(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_reshape_1d_to_2d() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]).unwrap();
        let r = t.reshape(&[2, 3]).unwrap();

        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(r.numel(), 6);
    }

    #[test]
    fn test_reshape_3d_to_2d() {
        let data = vec![1.0; 24]; // 2*3*4 = 24
        let t = Tensor::new(data, vec![2, 3, 4]).unwrap();
        let r = t.reshape(&[6, 4]).unwrap();

        assert_eq!(r.shape(), &[6, 4]);
        assert_eq!(r.numel(), 24);
    }

    #[test]
    fn test_reshape_error_numel_mismatch() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

        // Try to reshape 6 elements into 4 elements (2*2)
        let result = t.reshape(&[2, 2]);

        assert!(result.is_err());
        match result.unwrap_err() {
            LludaError::ShapeMismatch { expected, got } => {
                assert_eq!(expected, vec![4]);
                assert_eq!(got, vec![6]);
            }
            _ => panic!("Expected ShapeMismatch error"),
        }
    }

    #[test]
    fn test_reshape_preserves_data() {
        let original_data = vec![1.0, 2.0, 3.0, 4.0];
        let t = Tensor::new(original_data.clone(), vec![4]).unwrap();
        let r = t.reshape(&[2, 2]).unwrap();

        assert_eq!(r.to_vec_f32(), original_data);
    }

    #[test]
    fn test_reshape_preserves_bf16_dtype() {
        let data = vec![
            BF16::from(1.0f32),
            BF16::from(2.0f32),
            BF16::from(3.0f32),
            BF16::from(4.0f32),
        ];
        let t = Tensor::from_bf16(data, vec![4]).unwrap();
        assert_eq!(t.dtype(), DType::BF16);

        let r = t.reshape(&[2, 2]).unwrap();
        assert_eq!(r.dtype(), DType::BF16);
        assert_eq!(r.shape(), &[2, 2]);
        assert_eq!(r.to_vec_f32(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    // ========== Transpose Tests ==========

    #[test]
    fn test_transpose_2x2() {
        // Matrix: [[1, 2], [3, 4]]
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let tr = t.transpose().unwrap();

        assert_eq!(tr.shape(), &[2, 2]);
        // Transposed: [[1, 3], [2, 4]]
        let data = tr.to_vec_f32();
        assert_eq!(data, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_transpose_non_square() {
        // Matrix: [[1, 2, 3], [4, 5, 6]]  (2x3)
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let tr = t.transpose().unwrap();

        assert_eq!(tr.shape(), &[3, 2]);
        // Transposed: [[1, 4], [2, 5], [3, 6]]
        let data = tr.to_vec_f32();
        assert_eq!(data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_1x3() {
        // Row vector: [[1, 2, 3]]
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let tr = t.transpose().unwrap();

        assert_eq!(tr.shape(), &[3, 1]);
        // Column vector: [[1], [2], [3]]
        let data = tr.to_vec_f32();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_transpose_double_is_identity() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let original_data = t.to_vec_f32();

        let tr = t.transpose().unwrap().transpose().unwrap();

        assert_eq!(tr.shape(), &[2, 3]);
        assert_eq!(tr.to_vec_f32(), original_data);
    }

    #[test]
    fn test_transpose_dims_2d_same_as_transpose() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let tr1 = t.transpose().unwrap();
        let tr2 = t.transpose_dims(0, 1).unwrap();

        assert_eq!(tr1.shape(), tr2.shape());
        assert_eq!(tr1.to_vec_f32(), tr2.to_vec_f32());
    }

    #[test]
    fn test_transpose_dims_3d() {
        // [2, 3, 4] -> transpose(0, 2) -> [4, 3, 2]
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let t = Tensor::new(data, vec![2, 3, 4]).unwrap();
        let tr = t.transpose_dims(0, 2).unwrap();

        assert_eq!(tr.shape(), &[4, 3, 2]);

        // Verify: element at [i,j,k] in original = element at [k,j,i] in transposed
        let orig = t.to_vec_f32();
        let transposed = tr.to_vec_f32();
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    let src_idx = i * 12 + j * 4 + k;
                    let dst_idx = k * 6 + j * 2 + i;
                    assert_eq!(
                        orig[src_idx], transposed[dst_idx],
                        "Mismatch at [{},{},{}]",
                        i, j, k
                    );
                }
            }
        }
    }

    #[test]
    fn test_transpose_dims_4d_last_two() {
        // [1, 2, 3, 4] -> transpose(2, 3) -> [1, 2, 4, 3]
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let t = Tensor::new(data, vec![1, 2, 3, 4]).unwrap();
        let tr = t.transpose_dims(2, 3).unwrap();

        assert_eq!(tr.shape(), &[1, 2, 4, 3]);

        // Verify specific elements
        let orig = t.to_vec_f32();
        let transposed = tr.to_vec_f32();
        // Element at [0,0,0,0] should stay at [0,0,0,0]
        assert_eq!(orig[0], transposed[0]);
        // Element at [0,0,0,1] should go to [0,0,1,0]
        assert_eq!(orig[1], transposed[3]);
    }

    #[test]
    fn test_transpose_dims_noop() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let tr = t.transpose_dims(0, 0).unwrap();

        assert_eq!(tr.shape(), t.shape());
        assert_eq!(tr.to_vec_f32(), t.to_vec_f32());
    }

    #[test]
    fn test_transpose_dims_out_of_range() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let result = t.transpose_dims(0, 2);
        assert!(result.is_err());
        match result.unwrap_err() {
            LludaError::DimOutOfRange { dim, ndim } => {
                assert_eq!(dim, 2);
                assert_eq!(ndim, 1);
            }
            _ => panic!("Expected DimOutOfRange error"),
        }
    }

    #[test]
    fn test_transpose_dims_double_is_identity() {
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let t = Tensor::new(data.clone(), vec![2, 3, 4]).unwrap();
        let tr = t.transpose_dims(0, 2).unwrap().transpose_dims(0, 2).unwrap();

        assert_eq!(tr.shape(), &[2, 3, 4]);
        assert_eq!(tr.to_vec_f32(), data);
    }

    #[test]
    fn test_transpose_error_not_2d() {
        // 1D tensor
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let result1 = t1.transpose();
        assert!(result1.is_err());

        // 3D tensor
        let t3 = Tensor::new(vec![1.0; 24], vec![2, 3, 4]).unwrap();
        let result3 = t3.transpose();
        assert!(result3.is_err());

        match result3.unwrap_err() {
            LludaError::DimOutOfRange { dim, ndim } => {
                assert_eq!(dim, 2);
                assert_eq!(ndim, 3);
            }
            _ => panic!("Expected DimOutOfRange error"),
        }
    }

    // ========== Squeeze Tests ==========

    #[test]
    fn test_squeeze_all_dims() {
        // Shape: [1, 3, 1]
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3, 1]).unwrap();
        let squeezed = t.squeeze(None).unwrap();

        assert_eq!(squeezed.shape(), &[3]);
        assert_eq!(squeezed.to_vec_f32(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_squeeze_specific_dim_first() {
        // Shape: [1, 3, 1]
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3, 1]).unwrap();
        let squeezed = t.squeeze(Some(0)).unwrap();

        // Remove dim 0 -> [3, 1]
        assert_eq!(squeezed.shape(), &[3, 1]);
    }

    #[test]
    fn test_squeeze_specific_dim_last() {
        // Shape: [1, 3, 1]
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3, 1]).unwrap();
        let squeezed = t.squeeze(Some(2)).unwrap();

        // Remove dim 2 -> [1, 3]
        assert_eq!(squeezed.shape(), &[1, 3]);
    }

    #[test]
    fn test_squeeze_no_size_1_dims() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let squeezed = t.squeeze(None).unwrap();

        // No dimensions to remove
        assert_eq!(squeezed.shape(), &[2, 3]);
    }

    #[test]
    fn test_squeeze_error_dim_not_size_1() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

        // Try to squeeze dim 0 which has size 2 (not 1)
        let result = t.squeeze(Some(0));

        assert!(result.is_err());
        match result.unwrap_err() {
            LludaError::Msg(msg) => {
                assert!(msg.contains("Cannot squeeze"));
                assert!(msg.contains("size 2"));
            }
            _ => panic!("Expected Msg error"),
        }
    }

    #[test]
    fn test_squeeze_error_dim_out_of_range() {
        let t = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();

        // Try to squeeze dimension 5 (out of range for 1D tensor)
        let result = t.squeeze(Some(5));

        assert!(result.is_err());
        match result.unwrap_err() {
            LludaError::DimOutOfRange { dim, ndim } => {
                assert_eq!(dim, 5);
                assert_eq!(ndim, 1);
            }
            _ => panic!("Expected DimOutOfRange error"),
        }
    }

    #[test]
    fn test_squeeze_multiple_size_1_dims() {
        // Shape: [1, 1, 3, 1]
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 1, 3, 1]).unwrap();
        let squeezed = t.squeeze(None).unwrap();

        // All size-1 dims removed -> [3]
        assert_eq!(squeezed.shape(), &[3]);
    }

    // ========== Unsqueeze Tests ==========

    #[test]
    fn test_unsqueeze_at_beginning() {
        // Shape: [3]
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let unsqueezed = t.unsqueeze(0).unwrap();

        // Insert at position 0 -> [1, 3]
        assert_eq!(unsqueezed.shape(), &[1, 3]);
        assert_eq!(unsqueezed.to_vec_f32(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_unsqueeze_at_end() {
        // Shape: [3]
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let unsqueezed = t.unsqueeze(1).unwrap();

        // Insert at position 1 -> [3, 1]
        assert_eq!(unsqueezed.shape(), &[3, 1]);
        assert_eq!(unsqueezed.to_vec_f32(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_unsqueeze_middle() {
        // Shape: [2, 3]
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let unsqueezed = t.unsqueeze(1).unwrap();

        // Insert at position 1 -> [2, 1, 3]
        assert_eq!(unsqueezed.shape(), &[2, 1, 3]);
    }

    #[test]
    fn test_unsqueeze_multiple_times() {
        // Shape: [3]
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let unsqueezed = t.unsqueeze(0).unwrap().unsqueeze(2).unwrap();

        // [3] -> [1, 3] -> [1, 3, 1]
        assert_eq!(unsqueezed.shape(), &[1, 3, 1]);
    }

    #[test]
    fn test_unsqueeze_error_dim_out_of_range() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();

        // Try to insert at position 5 (valid range is 0..=1 for 1D tensor)
        let result = t.unsqueeze(5);

        assert!(result.is_err());
        match result.unwrap_err() {
            LludaError::DimOutOfRange { dim, ndim } => {
                assert_eq!(dim, 5);
                assert_eq!(ndim, 1);
            }
            _ => panic!("Expected DimOutOfRange error"),
        }
    }

    #[test]
    fn test_squeeze_unsqueeze_round_trip() {
        // Original: [2, 3]
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

        // [2, 3] -> unsqueeze(1) -> [2, 1, 3] -> squeeze(Some(1)) -> [2, 3]
        let modified = t.unsqueeze(1).unwrap().squeeze(Some(1)).unwrap();

        assert_eq!(modified.shape(), t.shape());
        assert_eq!(modified.to_vec_f32(), t.to_vec_f32());
    }

    // ========== Subtraction Tests ==========

    #[test]
    fn test_sub_same_shape() {
        let a = Tensor::new(vec![5.0, 7.0, 9.0], vec![3]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let c = a.sub(&b).unwrap();

        assert_eq!(c.to_vec_f32(), vec![4.0, 5.0, 6.0]);
        assert_eq!(c.dtype(), DType::F32);
    }

    #[test]
    fn test_sub_broadcast() {
        // [2, 3] - [3] -> broadcast
        let a = Tensor::new(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![2, 3]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let c = a.sub(&b).unwrap();

        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.to_vec_f32(), vec![9.0, 18.0, 27.0, 39.0, 48.0, 57.0]);
    }

    // ========== Exp Tests ==========

    #[test]
    fn test_exp_basic() {
        let a = Tensor::new(vec![0.0, 1.0], vec![2]).unwrap();
        let b = a.exp().unwrap();

        let data = b.to_vec_f32();
        assert!((data[0] - 1.0).abs() < 1e-6); // e^0 = 1
        assert!((data[1] - std::f32::consts::E).abs() < 1e-5); // e^1
    }

    #[test]
    fn test_exp_2d() {
        let a = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let b = a.exp().unwrap();

        assert_eq!(b.shape(), &[2, 2]);
        assert_eq!(b.to_vec_f32(), vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_exp_negative() {
        let a = Tensor::new(vec![-1.0], vec![1]).unwrap();
        let b = a.exp().unwrap();

        let data = b.to_vec_f32();
        assert!((data[0] - (-1.0f32).exp()).abs() < 1e-6);
    }

    // ========== Sum Tests ==========

    #[test]
    fn test_sum_dim0() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let s = t.sum(0).unwrap();

        assert_eq!(s.shape(), &[3]);
        assert_eq!(s.to_vec_f32(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sum_dim1() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let s = t.sum(1).unwrap();

        assert_eq!(s.shape(), &[2]);
        assert_eq!(s.to_vec_f32(), vec![6.0, 15.0]);
    }

    #[test]
    fn test_sum_1d() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let s = t.sum(0).unwrap();

        assert_eq!(s.shape(), &[1]);
        assert_eq!(s.to_vec_f32(), vec![6.0]);
    }

    #[test]
    fn test_sum_3d() {
        // [2, 2, 3], sum along dim 2
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let t = Tensor::new(data, vec![2, 2, 3]).unwrap();
        let s = t.sum(2).unwrap();

        assert_eq!(s.shape(), &[2, 2]);
        // [1+2+3, 4+5+6, 7+8+9, 10+11+12] = [6, 15, 24, 33]
        assert_eq!(s.to_vec_f32(), vec![6.0, 15.0, 24.0, 33.0]);
    }

    #[test]
    fn test_sum_keepdim() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let s = t.sum_keepdim(1).unwrap();

        assert_eq!(s.shape(), &[2, 1]);
        assert_eq!(s.to_vec_f32(), vec![6.0, 15.0]);
    }

    #[test]
    fn test_sum_dim_out_of_range() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let result = t.sum(1);
        assert!(result.is_err());
        match result.unwrap_err() {
            LludaError::DimOutOfRange { dim, ndim } => {
                assert_eq!(dim, 1);
                assert_eq!(ndim, 1);
            }
            _ => panic!("Expected DimOutOfRange error"),
        }
    }

    // ========== Max Tests ==========

    #[test]
    fn test_max_dim1() {
        let t = Tensor::new(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3]).unwrap();
        let m = t.max(1).unwrap();

        assert_eq!(m.shape(), &[2]);
        assert_eq!(m.to_vec_f32(), vec![5.0, 6.0]);
    }

    #[test]
    fn test_max_dim0() {
        let t = Tensor::new(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3]).unwrap();
        let m = t.max(0).unwrap();

        assert_eq!(m.shape(), &[3]);
        assert_eq!(m.to_vec_f32(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_max_1d() {
        let t = Tensor::new(vec![3.0, 1.0, 5.0, 2.0], vec![4]).unwrap();
        let m = t.max(0).unwrap();

        assert_eq!(m.shape(), &[1]);
        assert_eq!(m.to_vec_f32(), vec![5.0]);
    }

    #[test]
    fn test_max_keepdim() {
        let t = Tensor::new(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3]).unwrap();
        let m = t.max_keepdim(1).unwrap();

        assert_eq!(m.shape(), &[2, 1]);
        assert_eq!(m.to_vec_f32(), vec![5.0, 6.0]);
    }

    #[test]
    fn test_max_negative_values() {
        let t = Tensor::new(vec![-5.0, -1.0, -3.0], vec![3]).unwrap();
        let m = t.max(0).unwrap();

        assert_eq!(m.to_vec_f32(), vec![-1.0]);
    }

    #[test]
    fn test_max_dim_out_of_range() {
        let t = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
        let result = t.max(1);
        assert!(result.is_err());
    }

    // ========== Zeros Constructor Tests ==========

    #[test]
    fn test_zeros_f32() {
        let t = Tensor::zeros(&[2, 3], DType::F32).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.to_vec_f32(), vec![0.0; 6]);
    }

    #[test]
    fn test_zeros_bf16() {
        let t = Tensor::zeros(&[3], DType::BF16).unwrap();
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.dtype(), DType::BF16);
        assert_eq!(t.to_vec_f32(), vec![0.0, 0.0, 0.0]);
    }
}