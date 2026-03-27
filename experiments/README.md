# CS639 Experiments: Subproblem-Level Diagnosis of RLVR

## Overview

Three experiments investigating whether RLVR produces genuine step-by-step reasoning, using SPARKLE's verifiable subproblem decompositions.

**Models:** Qwen-2.5-Math-7B (base), SparkleRL-Stage1, SparkleRL-Stage2-aug

**Data:** SPARKLE hardmath benchmark with subproblem chains (thinking levels 0-4)

## Quick Start

```bash
# Full run (requires GPU)
conda activate sparkle
cd experiments
./run_all.sh

# Debug mode (5 problems, fast)
./run_all.sh --debug

# Run single model
./run_all.sh --model base

# Skip inference, only re-analyze
./run_all.sh --skip-inference
```

## Files

| File | Description |
|------|-------------|
| `config.py` | Models, paths, hyperparameters |
| `data_loader.py` | Load SPARKLE benchmark, create subproblem chains |
| `inference.py` | vLLM inference engine with caching |
| `grading.py` | Answer extraction and grading (reuses SPARKLE) |
| `exp1_subproblem_passk.py` | **Exp 1**: Subproblem Pass@k analysis |
| `exp2_failure_localization.py` | **Exp 2**: Failure localization across reasoning steps |
| `exp3_path_divergence.py` | **Exp 3**: Path divergence analysis (stretch goal) |
| `visualize.py` | Generate all figures |
| `run_all.sh` | End-to-end orchestration |

## Experiments

### Experiment 1: Subproblem Pass@k (H1)

Tests whether base models catch up to RL models at high pass@k at the **subproblem level**. If RLVR only improves sampling efficiency, the crossover should occur at both outcome and subproblem levels. If RLVR genuinely improves step-level reasoning, the RL advantage persists at the subproblem level.

### Experiment 2: Failure Localization (H2)

Tracks which reasoning step is the first failure/success point for each problem. Reveals whether RLVR improves **execution** (carrying out steps) or **planning** (knowing what steps to take). Stratified by difficulty and mathematical domain.

### Experiment 3: Path Divergence (H3, stretch goal)

Analyzes cases where models answer the full problem correctly but fail subproblem levels. Uses automated metrics (n-gram overlap, step coverage) and selects cases for manual review to classify divergence as: execution error, genuine alternative strategy, or pattern matching shortcut.

## Output

- `results/` — JSON result files for each experiment
- `figures/` — PDF figures for the paper
