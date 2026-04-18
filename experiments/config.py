"""
Configuration for CS639 Subproblem-Level Diagnosis Experiments.

Models: Qwen-2.5-Math-7B (base), SparkleRL-Stage1, SparkleRL-Stage2-aug
Data: SPARKLE benchmark (hardmath) with subproblem decompositions
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
FIGURES_DIR = os.path.join(EXPERIMENT_DIR, "figures")

# ── Models ─────────────────────────────────────────────────────────────────
MODELS = {
    "base": "/root/autodl-tmp/hf_cache/Qwen2.5-Math-7B",
    "stage1": "/root/autodl-tmp/hf_cache/SparkleRL-7B-Stage1",
    "stage2": "/root/autodl-tmp/hf_cache/SparkleRL-7B-Stage2-aug",
    "r1distill": "/root/autodl-tmp/hf_cache/DeepSeek-R1-Distill-Qwen-7B",
}

# ── Dataset ────────────────────────────────────────────────────────────────
DATASET_NAME = "sparkle-reasoning/hardmath"
DATASET_SPLIT = "train"  # hardmath only has train split

# ── Subproblem decomposition ──────────────────────────────────────────────
NUM_THINKING_LEVELS = 4  # number of progressive reasoning steps (levels 1-4)
# Level 0 = original problem (no partial reasoning)
# Level 1..4 = progressive amounts of reasoning provided

# ── Profiles ──────────────────────────────────────────────────────────────
# Select via EXP_PROFILE env var (default: phase2). Each profile sets
# MAX_PROBLEMS / K_VALUES / MAX_NEW_TOKENS / MAX_MODEL_LEN. Add new profiles
# here rather than sed-editing config.py for each run.
PROFILES = {
    "debug": {
        "MAX_PROBLEMS": 5,
        "K_VALUES": [1, 2, 4],
        "MAX_NEW_TOKENS": 3000,
        "MAX_MODEL_LEN": 4096,
    },
    "phase1": {  # Exp 1 crossover detection: small N, high k
        "MAX_PROBLEMS": 40,
        "K_VALUES": [1, 2, 4, 8, 16, 32, 64, 128],
        "MAX_NEW_TOKENS": 3000,
        "MAX_MODEL_LEN": 4096,
    },
    "phase2": {  # Exp 2/3 stratification: large N, moderate k
        "MAX_PROBLEMS": 500,
        "K_VALUES": [1, 2, 4, 8, 16],
        "MAX_NEW_TOKENS": 3000,
        "MAX_MODEL_LEN": 4096,
    },
    "r1_test": {  # Long-CoT models (DeepSeek-R1-Distill, etc.)
        "MAX_PROBLEMS": 500,
        "K_VALUES": [1, 2, 4, 8, 16],
        "MAX_NEW_TOKENS": 8000,
        "MAX_MODEL_LEN": 16384,
    },
}

_profile_name = os.environ.get("EXP_PROFILE", "phase2")
if _profile_name not in PROFILES:
    raise ValueError(
        f"Unknown EXP_PROFILE={_profile_name!r}. Available: {list(PROFILES)}"
    )
_profile = PROFILES[_profile_name]

# ── Sampling ───────────────────────────────────────────────────────────────
MAX_PROBLEMS = _profile["MAX_PROBLEMS"]
K_VALUES = _profile["K_VALUES"]
MAX_K = max(K_VALUES)  # we sample MAX_K responses and compute pass@k for subsets
TEMPERATURE = 0.6
TOP_P = 1.0
MAX_NEW_TOKENS = _profile["MAX_NEW_TOKENS"]
MAX_PROMPT_TOKENS = 1024

# ── vLLM ───────────────────────────────────────────────────────────────────
TENSOR_PARALLEL_SIZE = 1  # adjust based on available GPUs
GPU_MEMORY_UTILIZATION = 0.90
DTYPE = "bfloat16"
MAX_MODEL_LEN = _profile["MAX_MODEL_LEN"]

# ── Instruction prompt (same as SPARKLE) ──────────────────────────────────
INSTRUCTION = (
    "A conversation between User and Assistant. The user asks a math question, "
    "and the Assistant solves it step by step. The Assistant first thinks about "
    "the complete reasoning process in the mind enclosed within <think> </think> "
    "tags. Then the Assistant provides a clear, concise answer to the user within "
    "<answer> </answer> tags, with the final result enclosed in \\boxed{} notation."
    "\n\nFor example:\n<think>\nreasoning process here\n</think>\n<answer>\n"
    "The answer is \\boxed{...}.\n</answer>"
)

# ── Experiment 3 (path divergence) ────────────────────────────────────────
PATH_DIVERGENCE_SAMPLE_SIZE = 100  # number of cases to analyze manually
