#!/bin/bash
#
# Run CPU vs GPU inference benchmark comparison
#
# Usage:
#   ./run_benchmark_comparison.sh
#

set -e

echo "=========================================="
echo "Qwen3-0.6B: CPU vs GPU Inference Benchmark"
echo "=========================================="
echo ""

# Create results directory
RESULTS_DIR="benchmark_results"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Running CPU benchmark..."
echo "----------------------------------------"
cargo run --example benchmark_inference --release > "$RESULTS_DIR/cpu_${TIMESTAMP}.txt" 2>&1
echo ""

echo "Running GPU benchmark..."
echo "----------------------------------------"
cargo run --features gpu --example benchmark_inference --release > "$RESULTS_DIR/gpu_${TIMESTAMP}.txt" 2>&1
echo ""

echo "=========================================="
echo "Comparison Results"
echo "=========================================="
echo ""

# Extract tokens/sec from results
CPU_TOKENS_PER_SEC=$(grep "Tokens/sec:" "$RESULTS_DIR/cpu_${TIMESTAMP}.txt" | awk '{print $2}')
GPU_TOKENS_PER_SEC=$(grep "Tokens/sec:" "$RESULTS_DIR/gpu_${TIMESTAMP}.txt" | awk '{print $2}')

CPU_TIME=$(grep "Generation time:" "$RESULTS_DIR/cpu_${TIMESTAMP}.txt" | awk '{print $3}')
GPU_TIME=$(grep "Generation time:" "$RESULTS_DIR/gpu_${TIMESTAMP}.txt" | awk '{print $3}')

echo "CPU Performance: $CPU_TOKENS_PER_SEC tokens/sec (total time: $CPU_TIME)"
echo "GPU Performance: $GPU_TOKENS_PER_SEC tokens/sec (total time: $GPU_TIME)"
echo ""

# Calculate speedup
SPEEDUP=$(echo "scale=2; $GPU_TOKENS_PER_SEC / $CPU_TOKENS_PER_SEC" | bc)
echo "GPU Speedup: ${SPEEDUP}x"
echo ""

# Extract generated text for comparison
echo "CPU Generated Text:"
echo "----------------------------------------"
sed -n '/=== Generated Text ===/,/=== Token Statistics ===/p' "$RESULTS_DIR/cpu_${TIMESTAMP}.txt" | grep -v "===" | head -10
echo ""

echo "GPU Generated Text:"
echo "----------------------------------------"
sed -n '/=== Generated Text ===/,/=== Token Statistics ===/p' "$RESULTS_DIR/gpu_${TIMESTAMP}.txt" | grep -v "===" | head -10
echo ""

# Check if texts match
CPU_TEXT=$(sed -n '/=== Generated Text ===/,/=== Token Statistics ===/p' "$RESULTS_DIR/cpu_${TIMESTAMP}.txt" | grep -v "===" | tr -d '\n' | tr -s ' ')
GPU_TEXT=$(sed -n '/=== Generated Text ===/,/=== Token Statistics ===/p' "$RESULTS_DIR/gpu_${TIMESTAMP}.txt" | grep -v "===" | tr -d '\n' | tr -s ' ')

if [ "$CPU_TEXT" = "$GPU_TEXT" ]; then
    echo "✓ Generated texts are IDENTICAL (quality verified)"
else
    echo "⚠ Generated texts DIFFER (may indicate numerical precision issues)"
fi

echo ""
echo "Results saved to:"
echo "  - $RESULTS_DIR/cpu_${TIMESTAMP}.txt"
echo "  - $RESULTS_DIR/gpu_${TIMESTAMP}.txt"
echo ""
