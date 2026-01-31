//! Error types for lluda-inference.
//!
//! Comprehensive error handling covering all failure modes:
//! - IO errors (file operations)
//! - Parse errors (model format, config)
//! - Shape mismatches (tensor dimensions)
//! - Numerical errors (NaN, overflow)
//! - Model errors (missing weights, invalid architecture)

use thiserror::Error;

/// Main error type for lluda-inference.
///
/// All library functions return `Result<T, LludaError>`.
/// No panics in library code - all failures go through this error type.
#[derive(Error, Debug)]
pub enum LludaError {
    /// Shape mismatch between tensors or operations.
    ///
    /// Example: Matrix multiplication requires matching inner dimensions.
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        /// Expected shape dimensions
        expected: Vec<usize>,
        /// Actual shape dimensions received
        got: Vec<usize>,
    },

    /// Data type mismatch.
    ///
    /// Example: Operation requires F32 but received BF16.
    #[error("DType mismatch: expected {expected}, got {got}")]
    DTypeMismatch {
        /// Expected data type name
        expected: String,
        /// Actual data type received
        got: String,
    },

    /// Dimension index out of range.
    ///
    /// Example: Attempting to access dimension 3 of a 2D tensor.
    #[error("Dimension out of range: {dim} for tensor with {ndim} dimensions")]
    DimOutOfRange {
        /// Dimension index that was accessed
        dim: usize,
        /// Number of dimensions in the tensor
        ndim: usize,
    },

    /// IO operation failed.
    ///
    /// Wraps standard library IO errors (file not found, read failures, etc).
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// SafeTensors parsing or loading error.
    ///
    /// Example: Invalid SafeTensors format, missing tensor, corrupt data.
    #[error("SafeTensors error: {0}")]
    SafeTensors(String),

    /// Tokenizer operation failed.
    ///
    /// Example: Invalid encoding, unknown token, tokenizer loading error.
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    /// JSON parsing error.
    ///
    /// Wraps serde_json errors when parsing model config files.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Numerical error.
    ///
    /// Example: NaN encountered, overflow, division by zero.
    #[error("Numerical error: {0}")]
    Numerical(String),

    /// Model structure error.
    ///
    /// Example: Missing required weights, invalid architecture, config mismatch.
    #[error("Model error: {0}")]
    Model(String),

    /// Generic error with message.
    ///
    /// Used when no specific error variant applies.
    /// Prefer specific variants for better error context.
    #[error("{0}")]
    Msg(String),
}

/// Result type alias for lluda-inference.
///
/// Equivalent to `std::result::Result<T, LludaError>`.
/// Used throughout the library for consistent error handling.
pub type Result<T> = std::result::Result<T, LludaError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_mismatch_display() {
        let err = LludaError::ShapeMismatch {
            expected: vec![2, 3],
            got: vec![2, 4],
        };
        let msg = format!("{}", err);
        assert!(msg.contains("expected"));
        assert!(msg.contains("[2, 3]"));
        assert!(msg.contains("[2, 4]"));
    }

    #[test]
    fn test_dtype_mismatch_display() {
        let err = LludaError::DTypeMismatch {
            expected: "F32".to_string(),
            got: "BF16".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("F32"));
        assert!(msg.contains("BF16"));
    }

    #[test]
    fn test_dim_out_of_range_display() {
        let err = LludaError::DimOutOfRange { dim: 3, ndim: 2 };
        let msg = format!("{}", err);
        assert!(msg.contains("3"));
        assert!(msg.contains("2"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let lluda_err: LludaError = io_err.into();
        let msg = format!("{}", lluda_err);
        assert!(msg.contains("IO error"));
        assert!(msg.contains("file not found"));
    }

    #[test]
    fn test_json_error_conversion() {
        let json_str = r#"{"invalid": json}"#;
        let json_err = serde_json::from_str::<serde_json::Value>(json_str).unwrap_err();
        let lluda_err: LludaError = json_err.into();
        let msg = format!("{}", lluda_err);
        assert!(msg.contains("JSON error"));
    }

    #[test]
    fn test_safetensors_error() {
        let err = LludaError::SafeTensors("invalid format".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("SafeTensors error"));
        assert!(msg.contains("invalid format"));
    }

    #[test]
    fn test_tokenizer_error() {
        let err = LludaError::Tokenizer("encoding failed".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("Tokenizer error"));
        assert!(msg.contains("encoding failed"));
    }

    #[test]
    fn test_numerical_error() {
        let err = LludaError::Numerical("NaN encountered".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("Numerical error"));
        assert!(msg.contains("NaN"));
    }

    #[test]
    fn test_model_error() {
        let err = LludaError::Model("missing weight: layer.0.weight".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("Model error"));
        assert!(msg.contains("missing weight"));
    }

    #[test]
    fn test_generic_msg_error() {
        let err = LludaError::Msg("custom error message".to_string());
        let msg = format!("{}", err);
        assert_eq!(msg, "custom error message");
    }

    #[test]
    fn test_error_propagation_with_question_mark() {
        fn inner_fn() -> Result<String> {
            Err(LludaError::Msg("inner error".to_string()))
        }

        fn outer_fn() -> Result<String> {
            let _result = inner_fn()?;
            Ok("success".to_string())
        }

        let result = outer_fn();
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert_eq!(msg, "inner error");
    }

    #[test]
    fn test_io_error_propagation() {
        fn read_nonexistent_file() -> Result<String> {
            let _contents = std::fs::read_to_string("/nonexistent/path/file.txt")?;
            Ok("success".to_string())
        }

        let result = read_nonexistent_file();
        assert!(result.is_err());
        match result.unwrap_err() {
            LludaError::Io(_) => (),
            other => panic!("Expected IO error, got: {:?}", other),
        }
    }

    #[test]
    fn test_error_context_preserved() {
        let original_err = LludaError::ShapeMismatch {
            expected: vec![10, 20],
            got: vec![10, 25],
        };

        // Simulate wrapping with context
        let wrapped = LludaError::Msg(format!("During matmul: {}", original_err));
        let msg = format!("{}", wrapped);
        assert!(msg.contains("During matmul"));
        assert!(msg.contains("[10, 20]"));
        assert!(msg.contains("[10, 25]"));
    }
}
