//! SafeTensors model weight loader.
//!
//! Provides memory-mapped loading of model weights from SafeTensors format.
//! The loader uses mmap to avoid loading the entire model file into RAM.
//! Supports both single-file and multi-shard (sharded) safetensors files.
//!
//! # Example — single file
//!
//! ```rust,no_run
//! use lluda_inference::loader::ModelWeights;
//!
//! let weights = ModelWeights::from_safetensors("models/Qwen3-0.6B/model.safetensors")?;
//!
//! // Get embedding weights
//! let embed = weights.get("model.embed_tokens.weight").unwrap();
//! assert_eq!(embed.shape(), &[151936, 1024]);
//! # Ok::<(), lluda_inference::error::LludaError>(())
//! ```
//!
//! # Example — sharded model directory
//!
//! ```rust,no_run
//! use lluda_inference::loader::ModelWeights;
//!
//! // Loads model-00001-of-00003.safetensors, model-00002-of-00003.safetensors, ...
//! let weights = ModelWeights::from_directory("models/Qwen2.5-Omni-3B")?;
//! println!("Loaded {} tensors", weights.len());
//! # Ok::<(), lluda_inference::error::LludaError>(())
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;
use safetensors::SafeTensors;

use crate::bf16::BF16;
use crate::error::{LludaError, Result};
use crate::tensor::Tensor;

/// Container for model weights loaded from SafeTensors format.
///
/// Uses memory mapping to efficiently access large model files without
/// loading everything into RAM. All underlying mmaps are kept alive via Arc
/// for the lifetime of the weights container.
///
/// Supports both single-file and multi-shard (sharded) safetensors models.
#[derive(Debug)]
pub struct ModelWeights {
    weights: HashMap<String, Tensor>,
    #[allow(dead_code)]
    mmaps: Vec<Arc<Mmap>>, // Keep all mmaps alive for the lifetime of weights
}

impl ModelWeights {
    /// Load model weights from a SafeTensors file.
    ///
    /// The file is memory-mapped for efficient access. All tensors are
    /// loaded immediately but share the underlying mmap.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the .safetensors file
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - File not found or cannot be opened
    /// - Invalid SafeTensors format
    /// - Unsupported data type (only BF16 and F32 are supported)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use lluda_inference::loader::ModelWeights;
    ///
    /// let weights = ModelWeights::from_safetensors("models/Qwen3-0.6B/model.safetensors")?;
    /// println!("Loaded {} tensors", weights.len());
    /// # Ok::<(), lluda_inference::error::LludaError>(())
    /// ```
    pub fn from_safetensors(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_safetensors_shards(&[path])
    }

    /// Load weights from multiple safetensors shard files.
    ///
    /// Memory-maps each shard and builds a unified weight map across all shards.
    /// All mmaps are kept alive for the lifetime of the returned `ModelWeights`.
    ///
    /// Duplicate tensor names across shards are an error — each shard must
    /// contain disjoint sets of tensor names.
    ///
    /// # Arguments
    ///
    /// * `paths` - Ordered list of paths to shard files
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Any shard file is not found or cannot be opened
    /// - Any shard has an invalid SafeTensors format
    /// - A tensor name appears in more than one shard
    /// - Unsupported data type (only BF16 and F32 are supported)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use lluda_inference::loader::ModelWeights;
    ///
    /// let shards = [
    ///     "models/Qwen2.5-Omni-3B/model-00001-of-00003.safetensors",
    ///     "models/Qwen2.5-Omni-3B/model-00002-of-00003.safetensors",
    ///     "models/Qwen2.5-Omni-3B/model-00003-of-00003.safetensors",
    /// ];
    /// let weights = ModelWeights::from_safetensors_shards(&shards)?;
    /// println!("Loaded {} tensors from {} shards", weights.len(), shards.len());
    /// # Ok::<(), lluda_inference::error::LludaError>(())
    /// ```
    pub fn from_safetensors_shards(paths: &[impl AsRef<Path>]) -> Result<Self> {
        let mut mmaps = Vec::with_capacity(paths.len());
        let mut weights = HashMap::new();

        for path in paths {
            let path = path.as_ref();

            // Memory-map the shard file (don't load it all into RAM)
            let file = std::fs::File::open(path)?;
            let mmap = unsafe { Mmap::map(&file)? };
            let mmap = Arc::new(mmap);

            // Parse SafeTensors header and metadata
            let tensors = SafeTensors::deserialize(&mmap).map_err(|e| {
                LludaError::SafeTensors(format!(
                    "Failed to parse SafeTensors '{}': {}",
                    path.display(),
                    e
                ))
            })?;

            // Load each tensor from this shard into the unified weight map
            for (name, view) in tensors.tensors() {
                if weights.contains_key(&name) {
                    return Err(LludaError::SafeTensors(format!(
                        "Duplicate tensor '{}' found in shard '{}'",
                        name,
                        path.display()
                    )));
                }
                let tensor = load_tensor(&name, &view)?;
                weights.insert(name, tensor);
            }

            // Keep mmap alive — tensors reference its memory
            mmaps.push(mmap);
        }

        Ok(Self { weights, mmaps })
    }

    /// Load all safetensors files from a model directory.
    ///
    /// Finds all `*.safetensors` files in `dir`, sorts them by name
    /// (alphabetical order ensures correct shard ordering for files
    /// named like `model-00001-of-00003.safetensors`), and loads them all.
    ///
    /// # Arguments
    ///
    /// * `dir` - Path to the model directory
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Directory does not exist or cannot be read
    /// - No `.safetensors` files are found in the directory
    /// - Any shard fails to load (see [`from_safetensors_shards`])
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use lluda_inference::loader::ModelWeights;
    ///
    /// let weights = ModelWeights::from_directory("models/Qwen2.5-Omni-3B")?;
    /// println!("Loaded {} tensors", weights.len());
    /// # Ok::<(), lluda_inference::error::LludaError>(())
    /// ```
    pub fn from_directory(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();

        // Collect all *.safetensors paths in this directory
        let read_dir = std::fs::read_dir(dir).map_err(|e| {
            LludaError::Io(std::io::Error::new(
                e.kind(),
                format!("Cannot read directory '{}': {}", dir.display(), e),
            ))
        })?;

        let mut shard_paths: Vec<std::path::PathBuf> = read_dir
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();

        if shard_paths.is_empty() {
            return Err(LludaError::SafeTensors(format!(
                "No .safetensors files found in directory '{}'",
                dir.display()
            )));
        }

        // Sort alphabetically so shards load in the correct order
        // (e.g. model-00001-of-00003 before model-00002-of-00003)
        shard_paths.sort();

        Self::from_safetensors_shards(&shard_paths)
    }

    /// Get a tensor by name.
    ///
    /// Returns `None` if the tensor doesn't exist in the model.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use lluda_inference::loader::ModelWeights;
    ///
    /// let weights = ModelWeights::from_safetensors("models/Qwen3-0.6B/model.safetensors")?;
    /// let embed = weights.get("model.embed_tokens.weight").unwrap();
    /// # Ok::<(), lluda_inference::error::LludaError>(())
    /// ```
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.weights.get(name)
    }

    /// Get the number of tensors loaded.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Check if no tensors are loaded.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Get all tensor names.
    ///
    /// Useful for debugging and exploring model structure.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.weights.keys().map(|s| s.as_str()).collect()
    }
}

/// Decide whether a tensor should be quantized (e.g. BF16→Q8).
///
/// Only 2D weight matrices are candidates for quantization. Several
/// special-purpose tensors must stay in their original dtype:
///
/// - `embed_tokens` — token embedding tables: discrete lookups, not GEMM
/// - `positional_embedding` — sinusoidal lookup table, fixed values
/// - `audio_bos_eos_token` — special audio boundary tokens
///
/// Biases (1D) and convolution kernels (3D+) are not quantized either,
/// but that is enforced by the `shape.len() != 2` guard.
pub fn should_quantize(name: &str, shape: &[usize]) -> bool {
    // Only quantize 2D tensors (weight matrices)
    if shape.len() != 2 {
        return false;
    }

    // Skip embedding tables and lookup tensors: they are used for indexing,
    // not matrix-vector products, so quantization hurts quality without helping throughput.
    if name.contains("embed_tokens")
        || name.contains("positional_embedding")
        || name.contains("audio_bos_eos_token")
    {
        return false;
    }

    true
}

/// Load a single tensor from a SafeTensors tensor view.
///
/// Converts raw bytes to our Tensor type based on dtype.
/// The tensor name is provided so that quantization exclusion rules
/// (`should_quantize`) can be consulted when quantization is applied.
fn load_tensor(name: &str, view: &safetensors::tensor::TensorView) -> Result<Tensor> {
    let shape = view.shape().to_vec();
    let dtype = view.dtype();
    let data_bytes = view.data();

    match dtype {
        safetensors::Dtype::BF16 => {
            // Load as BF16.
            // SafeTensors stores BF16 as u16 in little-endian byte order.
            let num_elements = shape.iter().product::<usize>();

            if data_bytes.len() != num_elements * 2 {
                return Err(LludaError::SafeTensors(format!(
                    "BF16 tensor '{}' size mismatch: expected {} bytes, got {}",
                    name,
                    num_elements * 2,
                    data_bytes.len()
                )));
            }

            // Parse u16 values from little-endian bytes
            let mut bf16_data = Vec::with_capacity(num_elements);
            for chunk in data_bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                bf16_data.push(BF16::from_bits(bits));
            }

            // Note: quantization guard is in place for when Q8 quantization is added.
            // If should_quantize(name, &shape) is true, the tensor is a candidate for
            // BF16→Q8 conversion. Currently we keep all tensors as BF16.
            let _ = should_quantize(name, &shape); // documents intent; no-op for now

            Tensor::from_bf16(bf16_data, shape)
        }

        safetensors::Dtype::F32 => {
            // Load as F32
            let num_elements = shape.iter().product::<usize>();

            if data_bytes.len() != num_elements * 4 {
                return Err(LludaError::SafeTensors(format!(
                    "F32 tensor '{}' size mismatch: expected {} bytes, got {}",
                    name,
                    num_elements * 4,
                    data_bytes.len()
                )));
            }

            // Parse f32 values from little-endian bytes
            let mut f32_data = Vec::with_capacity(num_elements);
            for chunk in data_bytes.chunks_exact(4) {
                let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                f32_data.push(f32::from_bits(bits));
            }

            Tensor::new(f32_data, shape)
        }

        other => Err(LludaError::SafeTensors(format!(
            "Unsupported dtype {:?} for tensor '{}'. Only BF16 and F32 are supported.",
            other, name
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::DType;
    use std::path::PathBuf;

    /// Helper to get model path, skipping tests if not available
    fn get_model_path() -> Option<PathBuf> {
        let path = PathBuf::from("models/Qwen3-0.6B/model.safetensors");
        if path.exists() {
            Some(path)
        } else {
            eprintln!("Skipping: model files not found at {}", path.display());
            None
        }
    }

    // -------------------------------------------------------------------------
    // should_quantize unit tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_should_quantize() {
        // Regular weight matrix → should quantize
        assert!(should_quantize(
            "model.layers.0.self_attn.q_proj.weight",
            &[1024, 1024]
        ));

        // Token embedding table → must NOT quantize (discrete lookup)
        assert!(!should_quantize(
            "model.embed_tokens.weight",
            &[151936, 1024]
        ));

        // Positional embedding → must NOT quantize (sinusoidal lookup table)
        assert!(!should_quantize(
            "thinker.audio_tower.positional_embedding",
            &[1500, 1280]
        ));

        // Audio boundary token → must NOT quantize
        assert!(!should_quantize(
            "thinker.audio_tower.audio_bos_eos_token.weight",
            &[2, 1280]
        ));

        // Conv weight (3D) → must NOT quantize
        assert!(!should_quantize(
            "thinker.audio_tower.conv1.weight",
            &[1280, 128, 3]
        ));

        // Layer norm (1D) → must NOT quantize
        assert!(!should_quantize(
            "model.layers.0.input_layernorm.weight",
            &[1024]
        ));
    }

    // -------------------------------------------------------------------------
    // Directory loader tests (offline, no real model files needed)
    // -------------------------------------------------------------------------

    #[test]
    fn test_from_directory_nonexistent() {
        let result = ModelWeights::from_directory("/nonexistent/path");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_directory_empty_dir() {
        // Create a temp dir with no .safetensors files
        let tmp = std::env::temp_dir().join("lluda_test_empty_dir");
        let _ = std::fs::create_dir_all(&tmp);
        // Put a non-safetensors file in there so the dir exists and is readable
        let _ = std::fs::write(tmp.join("config.json"), "{}");

        let result = ModelWeights::from_directory(&tmp);
        assert!(result.is_err(), "Should error when no .safetensors files found");

        // Cleanup
        let _ = std::fs::remove_dir_all(&tmp);
    }

    // -------------------------------------------------------------------------
    // Single-file loader tests (require model files)
    // -------------------------------------------------------------------------

    #[test]
    fn test_load_qwen3_06b_weights() {
        let Some(path) = get_model_path() else {
            return;
        };

        let weights = ModelWeights::from_safetensors(&path).unwrap();

        // Model should have tensors
        assert!(!weights.is_empty(), "Model should have loaded tensors");

        // Check key tensors exist
        assert!(
            weights.get("model.embed_tokens.weight").is_some(),
            "Embedding weights should exist"
        );
        assert!(
            weights.get("model.layers.0.self_attn.q_proj.weight").is_some(),
            "Layer 0 q_proj should exist"
        );
    }

    #[test]
    fn test_embedding_shape() {
        let Some(path) = get_model_path() else {
            return;
        };

        let weights = ModelWeights::from_safetensors(&path).unwrap();
        let embed = weights.get("model.embed_tokens.weight").unwrap();

        // Qwen3-0.6B: vocab_size=151936, hidden_size=1024
        assert_eq!(
            embed.shape(),
            &[151936, 1024],
            "Embedding shape should be [vocab_size, hidden_size]"
        );
    }

    #[test]
    fn test_embedding_dtype() {
        let Some(path) = get_model_path() else {
            return;
        };

        let weights = ModelWeights::from_safetensors(&path).unwrap();
        let embed = weights.get("model.embed_tokens.weight").unwrap();

        // Model uses BF16 for weights
        assert_eq!(
            embed.dtype(),
            DType::BF16,
            "Embedding weights should be BF16"
        );
    }

    #[test]
    fn test_layer_shapes() {
        let Some(path) = get_model_path() else {
            return;
        };

        let weights = ModelWeights::from_safetensors(&path).unwrap();

        // Check attention projection shapes (layer 0)
        // Q: [num_heads * head_dim, hidden_size] = [16 * 128, 1024] = [2048, 1024]
        let q_proj = weights.get("model.layers.0.self_attn.q_proj.weight").unwrap();
        assert_eq!(q_proj.shape(), &[2048, 1024], "Q projection shape mismatch");

        // K, V: [num_kv_heads * head_dim, hidden_size] = [8 * 128, 1024] = [1024, 1024]
        let k_proj = weights.get("model.layers.0.self_attn.k_proj.weight").unwrap();
        assert_eq!(k_proj.shape(), &[1024, 1024], "K projection shape mismatch");

        let v_proj = weights.get("model.layers.0.self_attn.v_proj.weight").unwrap();
        assert_eq!(v_proj.shape(), &[1024, 1024], "V projection shape mismatch");

        // O: [hidden_size, num_heads * head_dim] = [1024, 2048]
        let o_proj = weights.get("model.layers.0.self_attn.o_proj.weight").unwrap();
        assert_eq!(
            o_proj.shape(),
            &[1024, 2048],
            "O projection shape mismatch"
        );

        // Q/K norm: [head_dim] = [128]
        let q_norm = weights.get("model.layers.0.self_attn.q_norm.weight").unwrap();
        assert_eq!(q_norm.shape(), &[128], "Q norm shape mismatch");

        let k_norm = weights.get("model.layers.0.self_attn.k_norm.weight").unwrap();
        assert_eq!(k_norm.shape(), &[128], "K norm shape mismatch");

        // Layer norms: [hidden_size] = [1024]
        let input_norm = weights.get("model.layers.0.input_layernorm.weight").unwrap();
        assert_eq!(input_norm.shape(), &[1024], "Input norm shape mismatch");

        let post_attn_norm = weights
            .get("model.layers.0.post_attention_layernorm.weight")
            .unwrap();
        assert_eq!(
            post_attn_norm.shape(),
            &[1024],
            "Post-attention norm shape mismatch"
        );

        // MLP projections
        // Gate/Up: [intermediate_size, hidden_size] = [3072, 1024]
        let gate_proj = weights.get("model.layers.0.mlp.gate_proj.weight").unwrap();
        assert_eq!(gate_proj.shape(), &[3072, 1024], "Gate proj shape mismatch");

        let up_proj = weights.get("model.layers.0.mlp.up_proj.weight").unwrap();
        assert_eq!(up_proj.shape(), &[3072, 1024], "Up proj shape mismatch");

        // Down: [hidden_size, intermediate_size] = [1024, 3072]
        let down_proj = weights.get("model.layers.0.mlp.down_proj.weight").unwrap();
        assert_eq!(
            down_proj.shape(),
            &[1024, 3072],
            "Down proj shape mismatch"
        );
    }

    #[test]
    fn test_final_norm_shape() {
        let Some(path) = get_model_path() else {
            return;
        };

        let weights = ModelWeights::from_safetensors(&path).unwrap();
        let norm = weights.get("model.norm.weight").unwrap();

        assert_eq!(norm.shape(), &[1024], "Final norm shape should be [1024]");
        assert_eq!(norm.dtype(), DType::BF16, "Final norm should be BF16");
    }

    #[test]
    fn test_all_layers_present() {
        let Some(path) = get_model_path() else {
            return;
        };

        let weights = ModelWeights::from_safetensors(&path).unwrap();

        // Qwen3-0.6B has 28 layers
        for i in 0..28 {
            let layer_prefix = format!("model.layers.{}", i);

            // Check each layer has all required weights
            assert!(
                weights
                    .get(&format!("{}.self_attn.q_proj.weight", layer_prefix))
                    .is_some(),
                "Layer {} missing q_proj",
                i
            );
            assert!(
                weights
                    .get(&format!("{}.self_attn.k_proj.weight", layer_prefix))
                    .is_some(),
                "Layer {} missing k_proj",
                i
            );
            assert!(
                weights
                    .get(&format!("{}.self_attn.v_proj.weight", layer_prefix))
                    .is_some(),
                "Layer {} missing v_proj",
                i
            );
            assert!(
                weights
                    .get(&format!("{}.self_attn.o_proj.weight", layer_prefix))
                    .is_some(),
                "Layer {} missing o_proj",
                i
            );
            assert!(
                weights
                    .get(&format!("{}.self_attn.q_norm.weight", layer_prefix))
                    .is_some(),
                "Layer {} missing q_norm",
                i
            );
            assert!(
                weights
                    .get(&format!("{}.self_attn.k_norm.weight", layer_prefix))
                    .is_some(),
                "Layer {} missing k_norm",
                i
            );
            assert!(
                weights
                    .get(&format!("{}.mlp.gate_proj.weight", layer_prefix))
                    .is_some(),
                "Layer {} missing gate_proj",
                i
            );
            assert!(
                weights
                    .get(&format!("{}.mlp.up_proj.weight", layer_prefix))
                    .is_some(),
                "Layer {} missing up_proj",
                i
            );
            assert!(
                weights
                    .get(&format!("{}.mlp.down_proj.weight", layer_prefix))
                    .is_some(),
                "Layer {} missing down_proj",
                i
            );
            assert!(
                weights
                    .get(&format!("{}.input_layernorm.weight", layer_prefix))
                    .is_some(),
                "Layer {} missing input_layernorm",
                i
            );
            assert!(
                weights
                    .get(&format!("{}.post_attention_layernorm.weight", layer_prefix))
                    .is_some(),
                "Layer {} missing post_attention_layernorm",
                i
            );
        }
    }

    #[test]
    fn test_no_lm_head() {
        let Some(path) = get_model_path() else {
            return;
        };

        let weights = ModelWeights::from_safetensors(&path).unwrap();

        // Qwen3 has tie_word_embeddings=true, so no separate lm_head weight
        assert!(
            weights.get("lm_head.weight").is_none(),
            "Should not have lm_head.weight (tied embeddings)"
        );
    }

    #[test]
    fn test_file_not_found() {
        let result = ModelWeights::from_safetensors("/nonexistent/path/model.safetensors");
        assert!(result.is_err(), "Should error on missing file");

        match result.unwrap_err() {
            LludaError::Io(_) => (), // Expected
            other => panic!("Expected IO error, got: {:?}", other),
        }
    }

    #[test]
    fn test_tensor_names() {
        let Some(path) = get_model_path() else {
            return;
        };

        let weights = ModelWeights::from_safetensors(&path).unwrap();
        let names = weights.tensor_names();

        assert!(!names.is_empty(), "Should have tensor names");
        assert!(
            names.contains(&"model.embed_tokens.weight"),
            "Should include embedding weight name"
        );
    }

    #[test]
    fn test_len_and_is_empty() {
        let Some(path) = get_model_path() else {
            return;
        };

        let weights = ModelWeights::from_safetensors(&path).unwrap();

        assert!(!weights.is_empty(), "Should have tensors");
        assert!(!weights.is_empty(), "Should not be empty");
    }

    // -------------------------------------------------------------------------
    // Multi-shard tests (require model files)
    // -------------------------------------------------------------------------

    #[test]
    fn test_from_safetensors_shards_single_shard() {
        // A single-element slice must behave identically to from_safetensors
        let Some(path) = get_model_path() else {
            return;
        };

        let single = ModelWeights::from_safetensors(&path).unwrap();
        let sharded = ModelWeights::from_safetensors_shards(&[&path]).unwrap();

        assert_eq!(
            single.len(),
            sharded.len(),
            "Single shard must have same tensor count as from_safetensors"
        );
    }

    #[test]
    fn test_from_safetensors_shards_missing_file() {
        let result = ModelWeights::from_safetensors_shards(&["/nonexistent/shard.safetensors"]);
        assert!(result.is_err(), "Should error on missing shard file");
    }
}
