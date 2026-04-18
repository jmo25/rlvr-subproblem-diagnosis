#!/bin/bash
# =============================================================================
# CS639 Project: Subproblem-Level Diagnosis of RLVR
# Run all experiments end-to-end
# =============================================================================
#
# Usage:
#   ./run_all.sh --profile phase1        # N=40, k up to 128 (Exp 1 crossover)
#   ./run_all.sh --profile phase2        # N=500, k up to 16 (Exp 2/3, default)
#   ./run_all.sh --profile r1_test --model r1distill  # Long-CoT model
#   ./run_all.sh --debug                 # alias for --profile debug
#   ./run_all.sh --model base            # Run only base model
#   ./run_all.sh --skip-inference        # Skip inference, just run analysis
#
# Profiles are defined in config.py (PROFILES dict). They set MAX_PROBLEMS,
# K_VALUES, MAX_NEW_TOKENS, MAX_MODEL_LEN. Override individual values with
# --max-problems / --num-samples / etc.
#
# Prerequisites:
#   - CUDA-capable GPU(s)
#   - conda activate sparkle
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults
MODEL="all"
PROFILE=""
MAX_PROBLEMS=""
NUM_SAMPLES=""
BATCH_SIZE=64
TP=1
SKIP_INFERENCE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --profile)
            PROFILE="$2"; shift 2;;
        --model)
            MODEL="$2"; shift 2;;
        --max-problems)
            MAX_PROBLEMS="$2"; shift 2;;
        --num-samples)
            NUM_SAMPLES="$2"; shift 2;;
        --batch-size)
            BATCH_SIZE="$2"; shift 2;;
        --tp)
            TP="$2"; shift 2;;
        --skip-inference)
            SKIP_INFERENCE=true; shift;;
        --debug)
            PROFILE="debug"; shift;;
        *)
            echo "Unknown option: $1"; exit 1;;
    esac
done

# Export profile for config.py to pick up
if [ -n "$PROFILE" ]; then
    export EXP_PROFILE="$PROFILE"
fi

# Build CLI overrides (non-empty args override profile defaults)
COMMON_ARGS=""
if [ -n "$MAX_PROBLEMS" ]; then
    COMMON_ARGS="$COMMON_ARGS --max-problems $MAX_PROBLEMS"
fi
INFERENCE_ARGS="$COMMON_ARGS"
if [ -n "$NUM_SAMPLES" ]; then
    INFERENCE_ARGS="$INFERENCE_ARGS --num-samples $NUM_SAMPLES"
fi

echo "============================================================"
echo " CS639: Subproblem-Level Diagnosis of RLVR"
echo "============================================================"
echo " Profile:     ${PROFILE:-phase2 (default)}"
echo " Model:       $MODEL"
echo " TP:          $TP"
echo " Max probs:   ${MAX_PROBLEMS:-from profile}"
echo " Num samples: ${NUM_SAMPLES:-from profile MAX_K}"
echo " Skip inf:    $SKIP_INFERENCE"
echo "============================================================"

# Ensure output directories exist
mkdir -p results figures

# ── Step 1: Inference ─────────────────────────────────────────────────────
if [ "$SKIP_INFERENCE" = false ]; then
    echo ""
    echo ">>> Step 1: Running vLLM inference..."
    echo "    This may take several hours depending on GPU and dataset size."
    echo ""

    # Set vLLM backend
    export VLLM_ATTENTION_BACKEND=XFORMERS

    python inference.py \
        --model "$MODEL" \
        --batch-size "$BATCH_SIZE" \
        --tp "$TP" \
        $INFERENCE_ARGS

    echo ">>> Inference complete."
else
    echo ""
    echo ">>> Skipping inference (using cached results)."
fi

# ── Step 2: Experiment 1 — Subproblem Pass@k ─────────────────────────────
echo ""
echo ">>> Step 2: Running Experiment 1 (Subproblem Pass@k)..."
python exp1_subproblem_passk.py $COMMON_ARGS
echo ">>> Experiment 1 complete."

# ── Step 3: Experiment 2 — Failure Localization ──────────────────────────
echo ""
echo ">>> Step 3: Running Experiment 2 (Failure Localization)..."
python exp2_failure_localization.py $COMMON_ARGS
echo ">>> Experiment 2 complete."

# ── Step 4: Experiment 3 — Path Divergence ───────────────────────────────
echo ""
echo ">>> Step 4: Running Experiment 3 (Path Divergence)..."
python exp3_path_divergence.py $COMMON_ARGS
echo ">>> Experiment 3 complete."

# ── Step 5: Visualizations ───────────────────────────────────────────────
echo ""
echo ">>> Step 5: Generating visualizations..."
python visualize.py --exp all
echo ">>> Visualizations complete."

# ── Summary ──────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " All experiments complete!"
echo "============================================================"
echo " Results:  $SCRIPT_DIR/results/"
echo " Figures:  $SCRIPT_DIR/figures/"
echo ""
echo " Key output files:"
echo "   results/exp1_subproblem_passk.json"
echo "   results/exp2_failure_localization.json"
echo "   results/exp3_path_divergence.json"
echo "   results/exp3_manual_review.json"
echo ""
echo " Figures:"
echo "   figures/exp1_passk_by_level.pdf"
echo "   figures/exp1_passk_by_model.pdf"
echo "   figures/exp2_success_heatmap.pdf"
echo "   figures/exp2_transition_gains.pdf"
echo "   figures/exp3_divergence_classification.pdf"
echo "   figures/exp3_divergence_rates.pdf"
echo "============================================================"
