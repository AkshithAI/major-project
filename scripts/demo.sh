#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# ADHIE End-to-End Demo Script
#
# Usage:
#   ./scripts/demo.sh MODEL_REF [DRAFT_MODEL_REF]
#
# Examples:
#   ./scripts/demo.sh Qwen/Qwen2.5-7B-Instruct-GGUF:q4_k_m
#   ./scripts/demo.sh Qwen/Qwen2.5-14B-Instruct-GGUF:q4_k_m Qwen/Qwen2.5-1.5B-Instruct-GGUF:q4_k_m
#
# Demonstrates the full ADHIE pipeline:
#   1. Hardware profiling
#   2. Model inspection
#   3. Execution plan generation
#   4. Memory estimation
#   5. Baseline benchmark (CPU-only)
#   6. ADHIE-optimised benchmark (autotuned)
#   7. Side-by-side comparison
#   8. Ablation study (if draft model provided)
# ──────────────────────────────────────────────────────────────────

set -euo pipefail

MODEL="${1:?Usage: $0 MODEL_REF [DRAFT_MODEL_REF]}"
DRAFT="${2:-}"
CONTEXT="${ADHIE_CONTEXT:-8192}"
RUNS="${ADHIE_RUNS:-3}"
TOKENS="${ADHIE_TOKENS:-128}"
RESULTS_DIR="benchmarks/results/demo"

divider() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
}

# ─── Step 1: Hardware Profile ─────────────────────────────────────
divider "Step 1/7: Hardware Profile"
adhie profile -o "${RESULTS_DIR}/hardware_profile.yaml"

# ─── Step 2: Model Inspection ────────────────────────────────────
divider "Step 2/7: Model Inspection"
adhie inspect "${MODEL}" -o "${RESULTS_DIR}/model_profile.yaml"

# ─── Step 3: Execution Plan ──────────────────────────────────────
divider "Step 3/7: Execution Plan Generation"
DRAFT_FLAG=""
if [ -n "${DRAFT}" ]; then
    DRAFT_FLAG="--draft ${DRAFT}"
fi
adhie plan "${MODEL}" --context "${CONTEXT}" ${DRAFT_FLAG} -o "${RESULTS_DIR}/plans.yaml"

# ─── Step 4: Memory Estimation ───────────────────────────────────
divider "Step 4/7: Memory Estimation"
adhie estimate "${MODEL}" --context "${CONTEXT}"

# ─── Step 5: Baseline Benchmark (CPU-only) ───────────────────────
divider "Step 5/7: Baseline Benchmark (CPU-only)"
adhie benchmark "${MODEL}" \
    --context "${CONTEXT}" \
    --mode baseline \
    --runs "${RUNS}" \
    --tokens "${TOKENS}" \
    -o "${RESULTS_DIR}"

# ─── Step 6: ADHIE-optimised Benchmark ──────────────────────────
divider "Step 6/7: ADHIE-optimised Benchmark"
DRAFT_BENCH=""
if [ -n "${DRAFT}" ]; then
    DRAFT_BENCH="--draft ${DRAFT}"
fi
adhie benchmark "${MODEL}" \
    --context "${CONTEXT}" \
    --mode auto \
    --runs "${RUNS}" \
    --tokens "${TOKENS}" \
    ${DRAFT_BENCH} \
    -o "${RESULTS_DIR}"

# ─── Step 7: Comparison ─────────────────────────────────────────
divider "Step 7/7: Results Comparison"
RESULT_FILES=$(find "${RESULTS_DIR}" -name "*.json" -not -name "spec_*" -not -name "ablation_*" | sort)
if [ -n "${RESULT_FILES}" ]; then
    # shellcheck disable=SC2086
    adhie compare ${RESULT_FILES} -o "${RESULTS_DIR}/comparison_report.md"
fi

# ─── Optional: Ablation Study ───────────────────────────────────
if [ -n "${DRAFT}" ]; then
    divider "Bonus: Ablation Study"
    adhie ablation "${MODEL}" \
        --draft "${DRAFT}" \
        --context "${CONTEXT}" \
        --runs "${RUNS}" \
        --tokens "${TOKENS}" \
        -o "${RESULTS_DIR}/ablation"
fi

# ─── Summary ────────────────────────────────────────────────────
divider "Demo Complete"
echo "Results saved to: ${RESULTS_DIR}/"
echo ""
echo "Files:"
find "${RESULTS_DIR}" -type f | sort | while read -r f; do
    echo "  $(du -h "$f" | cut -f1)  $f"
done
echo ""
echo "Generate plots with:"
echo "  python scripts/generate_plots.py --input ${RESULTS_DIR}"

