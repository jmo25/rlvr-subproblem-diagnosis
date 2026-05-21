# Subproblem-Level Diagnosis of RLVR

A recipe-conditional view of reasoning narrowing.

Two post-training recipes can leave a model with similar whole-problem pass@k yet very different intermediate reasoning behavior. This project uses the verifiable subproblem decompositions of [SPARKLE](https://github.com/sparkle-reasoning/sparkle) [Wang et al., 2025] to compare four 7B models descended from Qwen-2.5-Math at every scaffold level, and characterizes where outcome RLVR and long-CoT distillation diverge.

Inference-only code for the CS 639 project. The full writeup is `CS639_final_report.tex`. The `verl/`, `scripts/train/`, and `eval/` directories are vendored SPARKLE training code, included for reference but not invoked here.

## Motivation

[Yue et al. (2025)](https://arxiv.org/abs/2504.13837) report that base models surpass RL-tuned models at high pass@k, and conclude that RLVR narrows the set of solvable problems. Wen et al. (2025) reproduce the crossover on math benchmarks but argue it is misleading. Base models often reach correct answers through incorrect reasoning, and once those samples are filtered out RLVR models keep their lead at every k. Both papers evaluate the whole problem. We measure the same phenomenon at each scaffold level L₀ through L₄, where L₀ is the unscaffolded problem and L₄ provides every reasoning step except the last.

## Hypotheses

**H1.** The pass@k crossover holds at every subproblem level, but the crossover point varies with level in a pattern that does not simply track scaffold depth.

**H2.** Different post-training recipes produce different per-transition gain profiles. Some models gain disproportionately from near-complete scaffolds (an execution advantage), others perform strongest unscaffolded (a planning advantage).

**H3.** Among samples that solve L₀ correctly, the extent to which the chain of thought follows the SPARKLE decomposition differs across training recipes.

## Three diagnostics

**Subproblem pass@k.** Apply the pass@k crossover of Yue et al. at every scaffold level. Locates where, and at which k, the base model overtakes each RL-tuned checkpoint.

**Transition gain.** Δₖ = pass@1(Lₖ) − pass@1(Lₖ₋₁), the marginal value of one additional scaffold step. Gains concentrated late in the chain indicate an execution bottleneck. Flat profiles with strong L₀ indicate a planning advantage.

**Path alignment.** Rule-based classifier of L₀-correct chains. The n-gram overlap is the Jaccard coefficient J = |C ∩ R| / |C ∪ R| between generated CoT tokens C and reference decomposition tokens R. Step coverage is the mean per-step recall over reference steps. All verification is rule-based, with no LLM-as-judge.

## Key results

On a shared Qwen-2.5-Math-7B base, SparkleRL (outcome RLVR with partial-solution scaffolding) and DeepSeek-R1-Distill (long-CoT SFT distillation) show opposite behavioral signatures that whole-problem pass@k conflates.

| Diagnostic | BASE | STAGE1 | STAGE2 | R1-Distill |
|---|---|---|---|---|
| L₀ pass@1 | 0.022 | 0.002 | **0.064** | 0.042 |
| L₀ pass@16 | 0.158 | 0.040 | 0.128 | **0.262** |
| L₃→L₄ gain | +0.014 | +0.285 | **+0.395** | +0.015 |
| Path alignment (consistent fraction, L₀-correct) | 40.3% | **70.7%** | 59.4% | 32.7% |
| Tentative label | reference | *executor* | *executor* | *planner* |

SparkleRL turns near-complete scaffolds into correct answers far more reliably than the base. Its L₃→L₄ gain is 20 to 28 times that of BASE, and among L₀-correct samples it follows the human decomposition more often. R1-Distill has the highest L₀ pass@k at every k we tested but gains almost nothing from incremental hints.

We call these signatures *executor* and *planner*, but only provisionally. With one pipeline per side we cannot tell whether they generalize to outcome RLVR and long-CoT distillation as broader categories. Report §4.4 lists the controls we would need: cross-base validation, an SFT control, and a reward-versus-data intervention.

Numbers are Phase 2 (N=500, k≤16) under the `r1_test` profile.

## Models

All four are 7B and descend from Qwen-2.5-Math-7B, so differences across the diagnostics are attributable to post-training recipe rather than base capacity or pretraining.

- **BASE**: `Qwen/Qwen2.5-Math-7B`. Math-specific continued pretraining, no post-training.
- **STAGE1**: `sparkle-reasoning/SparkleRL-7B-Stage1`. GRPO with a rule-based outcome reward (+2 for answer plus format, +1 for answer only, −1 otherwise).
- **STAGE2**: `sparkle-reasoning/SparkleRL-7B-Stage2-aug`. Initialized from STAGE1 and trained further on hard problems with the **same** reward. The change is in the training data: each reference trace is split into four segments and a progressively longer prefix (0 to 4 segments) is prepended as partial-solution scaffolding.
- **R1-Distill**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`. SFT distillation from the long-CoT traces of DeepSeek-R1. No reinforcement learning.

## Data

`sparkle-reasoning/hardmath` (train split, 6,501 problems). Each problem comes with a final answer and 1 to 4 verifiable subproblem decompositions. At level Lₖ the model receives the ground-truth partial reasoning for steps 0 through k−1, prepended to the prompt. L₀ is unscaffolded, L₄ provides all but the final step. Subproblem chains are assembled by `experiments/data_loader.py` following SPARKLE's protocol. Subsampling is deterministic (first N of the fixed train order), not randomized.

We use hardmath rather than the SPARKLE paper's 2,564-problem eval set because the latter has no subproblem decomposition. Grading uses SPARKLE's `MathScorer` (LaTeX normalization and boxed-answer extraction).

## Setup

```bash
conda create -n sparkle python==3.12 && conda activate sparkle
pip3 install torch==2.4.0
pip3 install flash-attn --no-build-isolation
pip install vllm==0.6.3 transformers datasets latex2sympy2 \
            antlr4-python3-runtime==4.9.3
```

Inference runs on a single A800-80GB GPU. The shim `experiments/_vllm_patch.py` monkey-patches vLLM 0.6.3 to handle Qwen2.5-Math's newer `rope_scaling` dict (which omits the `factor` key). It is imported by `inference.py` and must not be removed.

## Profiles

The full grid would cost about 18 GPU-days, so experiments are split into compute-aware profiles. Select via `--profile <name>` on `run_all.sh` or the `EXP_PROFILE` env var. Profiles live in `experiments/config.py`. Do not edit `K_VALUES` or `MAX_NEW_TOKENS` by hand.

| Profile | Purpose | N | K_VALUES | Max new tokens | Max context |
|---|---|---|---|---|---|
| `debug` | smoke test | 5 | 1, 2, 4 | 3000 | 4096 |
| `phase1` | H1 pass@k crossover sweep | 40 | 1, 2, 4, 8, 16, 32, 64, 128 | 3000 | 4096 |
| `phase2` (default) | H2/H3 on 3 Qwen-Math models | 500 | 1, 2, 4, 8, 16 | 3000 | 4096 |
| `r1_test` | adds R1-Distill (long-CoT safe) | 500 | 1, 2, 4, 8, 16 | 8000 | 16384 |

Phase 1's 40 problems are a strict prefix of Phase 2's 500.

The token budget is asymmetric on purpose. R1-Distill uses 8000 tokens and a 16k context window to match the long-CoT distribution it was distilled from. The other three use 3000 tokens and a 4k context window to match SparkleRL's training distribution; BASE gets the same configuration for comparability. Truncating R1-Distill to 3000 tokens would cut reasoning off before the answer and measure compression rather than reasoning. The SparkleRL checkpoints already stay within 3000 tokens, so the larger budget is wasted on them. Cross-family comparisons of absolute pass@k should therefore be read conservatively (report §4.3 spells this out).

Within a profile every model uses the same temperature (0.6), top-p (1.0), N, and K, so within-profile pass@k comparisons remain valid.

## Usage

```bash
cd experiments
conda activate sparkle

# Smoke test, ~5 min
./run_all.sh --debug

# Phase 1: pass@k crossover, 3 Qwen-Math models, k up to 128
./run_all.sh --profile phase1

# Phase 2 with R1-Distill: stratified evaluation, k up to 16
./run_all.sh --profile r1_test

# Re-run analysis only, using cached vLLM outputs
./run_all.sh --profile r1_test --skip-inference

# Per-GPU parallel inference, one model per GPU
EXP_PROFILE=r1_test CUDA_VISIBLE_DEVICES=0 python inference.py --model base   &
EXP_PROFILE=r1_test CUDA_VISIBLE_DEVICES=1 python inference.py --model stage1 &
# ...then re-aggregate with --skip-inference
```

Outputs go to `experiments/results/` and `experiments/figures/`. Archive between profile runs (`mv results results_phase1/`) so the next profile starts clean. The repo's `.gitignore` excludes `results*/`, `figures*/`, and per-profile output subdirs.

### Path classifier

For every L₀-correct sample, `experiments/exp3_path_divergence.py` assigns one of four categories.

- **`consistent`**: L₀ and L₁ through L₄ are all correct (no subproblem failure anywhere in the chain). The consistent fraction in the results table above is `|consistent| / |L₀-correct samples|`, computed per model over all rollouts. The Phase 2 totals are 1,066 L₀-correct samples across the four models.
- **`high_overlap_failure`**: J ≥ 0.5 with at least one Lᵢ incorrect. Same strategy, execution slip.
- **`low_overlap_success`**: J < 0.5 with at least one Lᵢ incorrect. A different valid path.
- **`shortcut`**: J < 0.2 and CoT length < 200 characters. Possible pattern matching.

Thresholds (`HIGH_OVERLAP=0.5`, `LOW_OVERLAP=0.2`, `SHORT_COT=200`) are at `exp3_path_divergence.py:131`. The classifier is a coarse proxy and is not validated against human labels; report §4.3 discusses limitations.

## Citation

```bibtex
@misc{mo2026subproblem,
  title={Subproblem-Level Diagnosis of RLVR: A Recipe-Conditional View of Reasoning Narrowing},
  author={Jiaqi Mo},
  year={2026},
  note={CS 639 Final Report, University of Wisconsin-Madison}
}
```

## Acknowledgments

Uses the [SPARKLE](https://github.com/sparkle-reasoning/sparkle) benchmark and evaluation infrastructure by [Wang et al. (2025)](https://arxiv.org/abs/2506.04723). Course project for CS 639 (Foundations of LLMs), taught by Prof. Frederic Sala at UW-Madison.
