//! Integration tests for Qwen2.5-Omni-3B inference validation
//!
//! Compares Rust inference output against Python reference data.

use ndarray::ArrayD;
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::path::Path;

const MODEL_DIR: &str = "/home/alexmak/lluda/models/Qwen2.5-Omni-3B";
const REFERENCE_DIR: &str = "/home/alexmak/lluda/reference_data";

fn skip_if_no_model() -> bool {
    if !Path::new(MODEL_DIR).join("config.json").exists() {
        eprintln!("Skipping: model not found at {MODEL_DIR}");
        return true;
    }
    false
}

fn skip_if_no_reference() -> bool {
    if !Path::new(REFERENCE_DIR).join("omni_text_simple").exists() {
        eprintln!("Skipping: reference data not found at {REFERENCE_DIR}");
        return true;
    }
    false
}

fn load_npy_f32(path: &Path) -> Vec<f32> {
    let reader = File::open(path)
        .unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display()));
    let array: ArrayD<f32> = ArrayD::read_npy(reader)
        .unwrap_or_else(|e| panic!("Failed to read npy {}: {e}", path.display()));
    array.into_raw_vec_and_offset().0
}

fn load_npy_i64(path: &Path) -> Vec<i64> {
    let reader = File::open(path)
        .unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display()));
    let array: ArrayD<i64> = ArrayD::<i64>::read_npy(reader)
        .unwrap_or_else(|e| panic!("Failed to read npy i64 {}: {e}", path.display()));
    array.into_raw_vec_and_offset().0
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for i in 0..a.len() {
        let va = a[i] as f64;
        let vb = b[i] as f64;
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt() + 1e-12)
}

#[test]
fn test_omni_bf16_text_simple_forward() {
    if skip_if_no_model() || skip_if_no_reference() {
        return;
    }

    use lluda_inference::config::Qwen3Config;
    use lluda_inference::loader::ModelWeights;
    use lluda_inference::model::Qwen3ForCausalLM;

    // Load config
    let config = Qwen3Config::from_file(Path::new(MODEL_DIR).join("config.json"))
        .expect("Failed to load Omni config");

    println!(
        "Config loaded: hidden_size={}, layers={}, heads={}, kv_heads={}, head_dim={}, prefix='{}'",
        config.hidden_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.tensor_prefix
    );

    assert_eq!(config.hidden_size, 2048);
    assert_eq!(config.num_hidden_layers, 36);
    assert_eq!(config.num_attention_heads, 16);
    assert_eq!(config.num_key_value_heads, 2);
    assert_eq!(config.head_dim, 128);
    assert_eq!(config.tensor_prefix, "thinker.");

    // Load model weights (BF16)
    let weights = ModelWeights::from_safetensors_dir(MODEL_DIR)
        .expect("Failed to load model weights");

    println!("Weights loaded: {} tensors", weights.len());

    // Build model
    let mut model = Qwen3ForCausalLM::load(&config, &weights)
        .expect("Failed to build model");

    // Load reference input_ids
    let ref_dir = Path::new(REFERENCE_DIR).join("omni_text_simple");
    let input_ids_i64 = load_npy_i64(&ref_dir.join("input_ids.npy"));
    let input_ids: Vec<u32> = input_ids_i64.iter().map(|&v| v as u32).collect();

    println!("Input tokens: {:?}", input_ids);

    // Run forward pass
    let logits_tensor = model.forward(&input_ids, 0).expect("Forward pass failed");

    println!(
        "Logits shape: {:?} (expected: [1, 1, {}])",
        logits_tensor.shape(),
        config.vocab_size
    );

    // The Rust model returns shape [1, 1, vocab_size] for last token only
    assert_eq!(logits_tensor.shape(), &[1, 1, config.vocab_size]);

    let rust_logits = logits_tensor.to_vec_f32();

    // Load reference logits: shape [1, 1, 151936]
    let ref_logits = load_npy_f32(&ref_dir.join("logits.npy"));

    assert_eq!(
        rust_logits.len(),
        ref_logits.len(),
        "Logits length mismatch: rust={}, ref={}",
        rust_logits.len(),
        ref_logits.len()
    );

    let cos = cosine_similarity(&rust_logits, &ref_logits);
    println!("Logits cosine similarity (last token): {:.6}", cos);

    // Check argmax matches
    let rust_argmax = rust_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let ref_argmax = ref_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;

    println!("Rust argmax: {}, Reference argmax: {}", rust_argmax, ref_argmax);

    // BF16 inference vs FP32 reference: cosine ~0.96 is expected for a 36-layer model.
    // The argmax (next-token prediction) is the key correctness metric.
    // Threshold 0.95 provides a safe margin below the observed ~0.96.
    assert!(
        cos > 0.95,
        "Logits cosine {cos:.6} too low (expected > 0.95 for BF16 inference)"
    );
    assert_eq!(rust_argmax, ref_argmax, "Next token prediction mismatch");
}
