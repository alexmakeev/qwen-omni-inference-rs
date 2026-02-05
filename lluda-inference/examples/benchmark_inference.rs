//! Inference benchmark: CPU vs GPU tokens/sec comparison.
//!
//! Measures inference throughput (tokens/sec) for Qwen3-0.6B model
//! on both CPU and GPU, comparing:
//! - Generation speed (tokens/sec)
//! - Generated text quality (should be identical for greedy sampling)
//!
//! # Usage
//!
//! ```bash
//! # CPU mode
//! cargo run --example benchmark_inference --release
//!
//! # GPU mode
//! cargo run --features gpu --example benchmark_inference --release
//! ```
//!
//! # Expected Output
//!
//! - Tokens/sec metric for performance comparison
//! - Generated text for quality verification
//! - CPU vs GPU mode indicator

use std::time::Instant;

use lluda_inference::config::Qwen3Config;
use lluda_inference::generate::{generate, GenerationConfig};
use lluda_inference::loader::ModelWeights;
use lluda_inference::model::Qwen3ForCausalLM;
use lluda_inference::tokenizer::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Qwen3-0.6B Inference Benchmark ===");
    println!();

    // Detect GPU mode
    let gpu_enabled = cfg!(feature = "gpu");
    println!("Mode: {}", if gpu_enabled { "GPU" } else { "CPU" });
    println!();

    // Model path
    let model_dir = std::path::PathBuf::from("../models/Qwen3-0.6B");
    if !model_dir.exists() {
        eprintln!("Error: Model directory not found at {}", model_dir.display());
        eprintln!("Please ensure Qwen3-0.6B model is at ../models/Qwen3-0.6B/");
        std::process::exit(1);
    }

    println!("Model: {}", model_dir.display());

    // Test prompt
    let prompt = "The capital of France is";
    println!("Prompt: \"{}\"", prompt);

    // Generation parameters
    let max_new_tokens = 50;
    println!("Tokens to generate: {}", max_new_tokens);
    println!();

    // Load model
    println!("Loading model...");
    let load_start = Instant::now();

    let config_path = model_dir.join("config.json");
    let weights_path = model_dir.join("model.safetensors");
    let tokenizer_path = model_dir.join("tokenizer.json");

    let config = Qwen3Config::from_file(&config_path)?;
    let weights = ModelWeights::from_safetensors(&weights_path)?;
    let mut model = Qwen3ForCausalLM::load(&config, &weights)?;

    let tokenizer = Tokenizer::from_file(tokenizer_path, 151643, vec![151645, 151643])?;

    let load_elapsed = load_start.elapsed();
    println!(
        "Model loaded in {:.2}s ({} layers, vocab_size={})",
        load_elapsed.as_secs_f64(),
        config.num_hidden_layers,
        config.vocab_size
    );
    println!();

    // Generation config (greedy sampling for deterministic comparison)
    let gen_config = GenerationConfig {
        max_new_tokens,
        temperature: 0.0, // Greedy decoding
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
    };

    // Warmup: single token generation to initialize caches
    println!("Warmup: generating 1 token...");
    let warmup_config = GenerationConfig {
        max_new_tokens: 1,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
    };
    let _ = generate(&mut model, &tokenizer, prompt, &warmup_config)?;
    println!("Warmup complete");
    println!();

    // Benchmark: generate tokens and measure time
    println!("Benchmarking generation of {} tokens...", max_new_tokens);
    let bench_start = Instant::now();

    let output = generate(&mut model, &tokenizer, prompt, &gen_config)?;

    let bench_elapsed = bench_start.elapsed();
    let bench_secs = bench_elapsed.as_secs_f64();

    // Calculate tokens/sec
    let tokens_per_sec = max_new_tokens as f64 / bench_secs;

    // Results
    println!();
    println!("=== Benchmark Results ===");
    println!();
    println!("Generation time: {:.3}s", bench_secs);
    println!("Tokens/sec: {:.2}", tokens_per_sec);
    println!();

    // Display generated text
    println!("=== Generated Text ===");
    println!();
    println!("{}", output);
    println!();

    // Statistics
    let output_tokens = tokenizer.encode(&output, false)?;
    let prompt_tokens = tokenizer.encode(prompt, true)?;
    let actual_new_tokens = output_tokens.len().saturating_sub(prompt_tokens.len());

    println!("=== Token Statistics ===");
    println!();
    println!("Prompt tokens: {}", prompt_tokens.len());
    println!("Generated tokens: {}", actual_new_tokens);
    println!("Total tokens: {}", output_tokens.len());
    println!();

    // Performance summary
    println!("=== Summary ===");
    println!();
    println!("Mode: {}", if gpu_enabled { "GPU" } else { "CPU" });
    println!("Performance: {:.2} tokens/sec", tokens_per_sec);
    println!(
        "Average time per token: {:.0}ms",
        (bench_secs / actual_new_tokens as f64) * 1000.0
    );
    println!();

    if gpu_enabled {
        println!("GPU acceleration enabled via --features gpu");
    } else {
        println!("Running on CPU (compile with --features gpu for GPU acceleration)");
    }

    println!();
    println!("Benchmark complete!");

    Ok(())
}
