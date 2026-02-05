//! Inference benchmark with detailed profiling: CPU vs GPU tokens/sec comparison.
//!
//! Measures inference throughput (tokens/sec) for Qwen3-0.6B model
//! on both CPU and GPU with detailed timing breakdown to identify bottlenecks.
//!
//! # Usage
//!
//! ```bash
//! # CPU mode
//! PROFILE=1 cargo run --example benchmark_inference_profile --release
//!
//! # GPU mode with profiling
//! PROFILE=1 cargo run --features gpu --example benchmark_inference_profile --release
//! ```
//!
//! # Expected Output
//!
//! - Detailed timing for each operation
//! - GPU usage verification logs
//! - Upload/download/compute breakdown
//! - Token-by-token profiling

use std::time::Instant;

use lluda_inference::config::Qwen3Config;
use lluda_inference::generate::GenerationConfig;
use lluda_inference::loader::ModelWeights;
use lluda_inference::model::Qwen3ForCausalLM;
use lluda_inference::tokenizer::Tokenizer;

// Profiling macro - only active when PROFILE env var is set
macro_rules! time_operation {
    ($label:expr, $op:expr) => {{
        let profiling_enabled = std::env::var("PROFILE").is_ok();
        let start = Instant::now();
        let result = $op;
        let elapsed = start.elapsed();
        if profiling_enabled {
            eprintln!(
                "[PROFILE] {}: {:.2}ms",
                $label,
                elapsed.as_secs_f64() * 1000.0
            );
        }
        result
    }};
}

// Manual generation with detailed profiling
fn generate_with_profiling(
    model: &mut Qwen3ForCausalLM,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let profiling_enabled = std::env::var("PROFILE").is_ok();

    // Clear KV cache
    model.clear_kv_cache();

    // Encode prompt
    let mut input_ids = time_operation!("Tokenize prompt", tokenizer.encode(prompt, true)?);

    if profiling_enabled {
        eprintln!("[PROFILE] Prompt tokens: {}", input_ids.len());
    }

    // Prefill: forward pass with all prompt tokens
    if profiling_enabled {
        eprintln!("\n[PROFILE] === Prefill phase ===");
    }
    let prefill_start = Instant::now();
    let prefill_len = input_ids.len();
    let logits = time_operation!("  Prefill forward pass", model.forward(&input_ids, 0)?);
    if profiling_enabled {
        eprintln!(
            "[PROFILE]   Prefill total: {:.2}ms",
            prefill_start.elapsed().as_secs_f64() * 1000.0
        );
    }

    // Sample first new token
    let last_logits = {
        let logits_data = logits.to_vec_f32();
        let vocab_size = logits.shape()[2];
        let start_idx = (logits.shape()[1] - 1) * vocab_size;
        logits_data[start_idx..start_idx + vocab_size].to_vec()
    };

    let mut next_token = last_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx as u32)
        .unwrap();

    let mut generated_tokens = vec![next_token];
    input_ids.push(next_token);

    // Check if we hit EOS immediately
    if tokenizer.eos_token_ids().contains(&next_token) {
        let output = tokenizer.decode(&input_ids, true)?;
        return Ok(output);
    }

    // Generation loop
    for token_idx in 1..max_new_tokens {
        if profiling_enabled {
            eprintln!("\n[PROFILE] === Token {} ===", token_idx + 1);
        }

        let token_start = Instant::now();

        // Forward pass with just the new token (KV cache handles context)
        let offset = prefill_len + token_idx - 1;
        let logits = time_operation!(
            format!("  Token {} forward pass", token_idx + 1),
            model.forward(&[next_token], offset)?
        );

        // Get last token logits
        let last_logits = time_operation!(
            format!("  Token {} extract logits", token_idx + 1),
            {
                let logits_data = logits.to_vec_f32();
                let vocab_size = logits.shape()[2];
                let start_idx = (logits.shape()[1] - 1) * vocab_size;
                logits_data[start_idx..start_idx + vocab_size].to_vec()
            }
        );

        // Greedy sampling
        next_token = time_operation!(
            format!("  Token {} sampling", token_idx + 1),
            {
                last_logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as u32)
                    .unwrap()
            }
        );

        let token_elapsed = token_start.elapsed();
        if profiling_enabled {
            eprintln!(
                "[PROFILE]   Token {} total: {:.2}ms",
                token_idx + 1,
                token_elapsed.as_secs_f64() * 1000.0
            );
            eprintln!(
                "[PROFILE]   Throughput so far: {:.2} tok/s",
                (token_idx + 1) as f64 / token_start.elapsed().as_secs_f64()
            );
        }

        generated_tokens.push(next_token);
        input_ids.push(next_token);

        // Check for EOS
        if tokenizer.eos_token_ids().contains(&next_token) {
            if profiling_enabled {
                eprintln!("[PROFILE] EOS token detected, stopping generation");
            }
            break;
        }
    }

    // Decode
    let output = time_operation!("Decode output", tokenizer.decode(&input_ids, true)?);

    Ok(output)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let profiling_enabled = std::env::var("PROFILE").is_ok();

    println!("=== Qwen3-0.6B Inference Benchmark (Profiling Mode) ===");
    println!();

    if profiling_enabled {
        eprintln!("[PROFILE] Profiling enabled via PROFILE env var");
        eprintln!();
    }

    // Detect GPU mode
    let gpu_enabled = cfg!(feature = "gpu");
    println!("Mode: {}", if gpu_enabled { "GPU" } else { "CPU" });
    if profiling_enabled {
        if gpu_enabled {
            eprintln!("[GPU] GPU mode enabled");
        } else {
            eprintln!("[CPU] CPU mode");
        }
    }
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

    // Generation parameters (reduced for profiling)
    let max_new_tokens = 10;
    println!("Tokens to generate: {}", max_new_tokens);
    println!();

    // Load model
    println!("Loading model...");
    let load_start = Instant::now();

    let config_path = model_dir.join("config.json");
    let weights_path = model_dir.join("model.safetensors");
    let tokenizer_path = model_dir.join("tokenizer.json");

    let config = time_operation!("Load config", Qwen3Config::from_file(&config_path)?);
    let weights = time_operation!(
        "Load weights",
        ModelWeights::from_safetensors(&weights_path)?
    );

    #[cfg(feature = "gpu")]
    {
        if profiling_enabled {
            eprintln!("[GPU] Initializing GPU context...");
        }
    }

    let mut model = time_operation!("Initialize model", Qwen3ForCausalLM::load(&config, &weights)?);

    let tokenizer = time_operation!(
        "Load tokenizer",
        Tokenizer::from_file(tokenizer_path, 151643, vec![151645, 151643])?
    );

    let load_elapsed = load_start.elapsed();
    println!(
        "Model loaded in {:.2}s ({} layers, vocab_size={})",
        load_elapsed.as_secs_f64(),
        config.num_hidden_layers,
        config.vocab_size
    );
    println!();

    // Warmup: single token generation to initialize caches
    println!("Warmup: generating 1 token...");
    let warmup_config = GenerationConfig {
        max_new_tokens: 1,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
    };

    if profiling_enabled {
        eprintln!("[PROFILE] Starting warmup...");
    }

    let _ = time_operation!(
        "Warmup generation",
        lluda_inference::generate::generate(&mut model, &tokenizer, prompt, &warmup_config)?
    );

    println!("Warmup complete");
    println!();

    // Benchmark: generate tokens with detailed profiling
    println!("Benchmarking generation of {} tokens...", max_new_tokens);
    if profiling_enabled {
        eprintln!("\n[PROFILE] ========================================");
        eprintln!("[PROFILE] Starting profiled generation");
        eprintln!("[PROFILE] ========================================\n");
    }

    let bench_start = Instant::now();

    let output = generate_with_profiling(&mut model, &tokenizer, prompt, max_new_tokens)?;

    let bench_elapsed = bench_start.elapsed();
    let bench_secs = bench_elapsed.as_secs_f64();

    if profiling_enabled {
        eprintln!("\n[PROFILE] ========================================");
        eprintln!("[PROFILE] Generation complete");
        eprintln!("[PROFILE] ========================================\n");
    }

    // Calculate tokens/sec
    let output_tokens = tokenizer.encode(&output, false)?;
    let prompt_tokens = tokenizer.encode(prompt, true)?;
    let actual_new_tokens = output_tokens.len().saturating_sub(prompt_tokens.len());
    let tokens_per_sec = actual_new_tokens as f64 / bench_secs;

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
        if profiling_enabled {
            eprintln!("\n[PROFILE] Check logs above for GPU usage verification");
            eprintln!("[PROFILE] Look for [GPU] markers to verify GPU path was taken");
        }
    } else {
        println!("Running on CPU (compile with --features gpu for GPU acceleration)");
    }

    println!();
    println!("Benchmark complete!");

    Ok(())
}
