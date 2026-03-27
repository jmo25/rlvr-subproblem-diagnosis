# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SPARKLE is a fine-grained evaluation framework for analyzing LLM reasoning improvements under reinforcement learning (RL). It uses a two-stage curriculum training approach built on top of ByteDance's VERL framework (in `verl/`). The base model is Qwen2.5-Math-7B, trained with PPO/GRPO on mathematical reasoning tasks.

Paper: arXiv 2506.04723

## Setup

```bash
conda create -n sparkle python==3.12
conda activate sparkle
pip3 install torch==2.4.0
pip install psutil numpy
pip3 install flash-attn --no-build-isolation
cd verl && pip3 install -e .
pip install wandb IPython matplotlib vertexai latex2sympy2
pip3 install -U antlr4-python3-runtime==4.9.3
```

## Common Commands

### Data Preparation
```bash
python scripts/data/prepare_stage_one_data.py              # -> data/sparkle_dsr40k.parquet
python scripts/data/prepare_stage_two_data_aug.py --aug_version all  # -> data/sparkle_hardmath_aug_all_train.parquet
```

### Training (requires 8x A100-80GB GPUs)
```bash
export VLLM_ATTENTION_BACKEND=XFORMERS  # Required on every machine before Ray cluster start
./scripts/train/stage_one.sh --model Qwen/Qwen2.5-Math-7B
./scripts/train/stage_two_aug.sh --model /path/to/stage1/checkpoint
```

Training is launched via `python3 -m verl.trainer.main_ppo` with Hydra config overrides. Default config: `verl/verl/trainer/config/ppo_trainer.yaml`.

### Evaluation
```bash
# Convert FSDP checkpoint to HuggingFace format
python eval/fsdp2hf.py --fsdp_path /path/to/checkpoint/actor --base_model Qwen/Qwen2.5-Math-7B --output_path /path/to/output

# Install eval harness and run benchmarks
cd eval/lm-evaluation-harness && pip install -e .
./scripts/eval/eval_all_vllm.sh
```

### Tests
```bash
cd verl && pytest tests/           # VERL framework tests
cd eval/lm-evaluation-harness && pytest tests/  # Eval harness tests
```

## Architecture

### Three Main Components

1. **VERL Training Framework** (`verl/verl/`) - Forked/vendored ByteDance VERL. Core RL training with FSDP distributed training, Ray orchestration, and vLLM rollout generation.
   - `trainer/main_ppo.py` - Entry point. Contains `_select_rm_score_fn()` which maps reward types to scoring functions. This is where SPARKLE-specific reward logic lives.
   - `trainer/ppo/ray_trainer.py` - Ray-based distributed PPO trainer
   - `trainer/ppo/core_algos.py` - PPO/GRPO algorithm implementation
   - `utils/reward_score/sparkle_score.py` - Math grading with `MathScorer` class (answer extraction, LaTeX normalization, grading)
   - `workers/` - Actor, Critic, Rollout, and Reward Manager workers for distributed training

2. **Training Scripts** (`scripts/`) - Shell scripts that invoke `main_ppo.py` with specific Hydra config overrides for each training stage.

3. **Evaluation** (`eval/`) - EleutherAI's LM Evaluation Harness plus `fsdp2hf.py` for checkpoint conversion. Benchmarks: AIME 2024, AMC 2023, MATH500, GSM8K, OlympiadBench.

### Reward Types

Defined in `main_ppo.py:_select_rm_score_fn()`, selected via `+reward_type=` Hydra override:
- `spk_s` - Standard: format + answer scored independently
- `spk_g` - Granular: partial credit for format
- `spk_h` - **Hierarchical (paper default)**: correct answer+format=2, answer only=1, wrong=-1
- `spk_h_aug` - Hierarchical with augmented partial format handling (used in Stage 2)

### Training Pipeline

```
prepare_stage_one_data.py -> sparkle_dsr40k.parquet (40.3k problems)
    -> main_ppo.py Stage 1 (spk_h, 30 epochs, KL=0.001)
        -> checkpoint
prepare_stage_two_data_aug.py -> sparkle_hardmath_aug.parquet (6.5k hard problems)
    -> main_ppo.py Stage 2 (spk_h_aug, 350 epochs, KL=0.01)
        -> final model
```

### Key Training Parameters
- Batch size: 128, mini-batch: 64
- Max prompt: 1024 tokens, max response: 3000 tokens
- Rollout: vLLM with temperature 0.6, n=32 samples
- FSDP with gradient checkpointing, ref model param offloaded

### Datasets (HuggingFace)
- `sparkle-reasoning/dsr40k` - 40.3k training problems (Stage 1)
- `sparkle-reasoning/hardmath` - 6.5k hard problems with augmented partial steps (Stage 2)

### Key Dependencies
PyTorch 2.4.0, vLLM <=0.6.3, Ray >=2.10, flash-attn, Hydra, W&B
