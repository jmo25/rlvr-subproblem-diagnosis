"""
Experiment 3: Path Divergence Analysis (stretch goal)

For cases where the RL model answers the full problem correctly (level 0)
but fails >= 1 subproblem level:
  - Extract the model's chain-of-thought from its correct level-0 response
  - Compare the reasoning path against the SPARKLE subproblem chain
  - Classify divergence types:
    (a) Same strategy, execution error on subproblem formulation
    (b) Genuinely different valid strategy
    (c) Pattern matching / shortcut without coherent intermediate reasoning

This analysis combines automated metrics with cases for manual review.
"""

import os
import sys
import json
import re
import argparse
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Tuple

from config import (
    MODELS, RESULTS_DIR, NUM_THINKING_LEVELS, PATH_DIVERGENCE_SAMPLE_SIZE,
)
from data_loader import load_sparkle_benchmark, ProblemWithChain
from inference import load_all_results
from grading import grade_response, extract_answer_from_response


def extract_cot_from_response(response: str) -> str:
    """Extract the chain-of-thought reasoning from a model response."""
    # Try to extract from <think>...</think> tags
    match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If continuing from partial <think>, the CoT is everything before </think>
    match = re.search(r"^(.*?)</think>", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: everything before <answer> or \boxed
    match = re.search(r"^(.*?)(?:<answer>|\\boxed)", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    return response.strip()


def compute_reasoning_overlap(cot: str, reference_steps: List[str]) -> dict:
    """
    Compute overlap between model's CoT and the reference reasoning steps.

    Uses n-gram overlap and key concept matching as automated proxies
    for reasoning path similarity.
    """
    if not cot or not reference_steps:
        return {"ngram_overlap": 0, "step_coverage": 0, "cot_length": len(cot)}

    # Tokenize (simple whitespace + lowercase)
    def tokenize(text):
        return set(re.findall(r'\b\w+\b', text.lower()))

    cot_tokens = tokenize(cot)
    full_ref = " ".join(reference_steps)
    ref_tokens = tokenize(full_ref)

    # Overall token overlap (Jaccard)
    if not cot_tokens or not ref_tokens:
        ngram_overlap = 0.0
    else:
        intersection = cot_tokens & ref_tokens
        union = cot_tokens | ref_tokens
        ngram_overlap = len(intersection) / len(union) if union else 0.0

    # Per-step coverage: what fraction of each reference step's tokens appear in CoT
    step_coverages = []
    for step in reference_steps:
        step_tokens = tokenize(step)
        if step_tokens:
            coverage = len(cot_tokens & step_tokens) / len(step_tokens)
            step_coverages.append(coverage)
        else:
            step_coverages.append(0.0)

    # Detect mathematical expressions/equations in both
    math_pattern = r'\\(?:frac|sqrt|sum|int|prod|lim|boxed)\{.*?\}|\d+\s*[+\-*/=<>]\s*\d+'
    cot_math = set(re.findall(math_pattern, cot))
    ref_math = set(re.findall(math_pattern, full_ref))
    math_overlap = len(cot_math & ref_math) / len(ref_math) if ref_math else 0.0

    return {
        "ngram_overlap": float(ngram_overlap),
        "step_coverage": float(np.mean(step_coverages)) if step_coverages else 0.0,
        "per_step_coverage": [float(c) for c in step_coverages],
        "math_overlap": float(math_overlap),
        "cot_length": len(cot),
        "ref_length": len(full_ref),
    }


def classify_divergence_auto(
    overlap: dict,
    level0_correct: bool,
    level_results: Dict[int, bool],
) -> str:
    """
    Automated classification of divergence type.

    Returns one of:
      "consistent" - correct at level 0 AND all subproblem levels
      "high_overlap_failure" - uses similar strategy but fails some subproblems
                               (likely execution error)
      "low_overlap_success" - correct at level 0 with different strategy
                              (genuinely different path)
      "shortcut" - correct at level 0 but very low reasoning overlap and short CoT
                   (possible pattern matching)
      "no_divergence" - fails at level 0 too
    """
    if not level0_correct:
        return "no_divergence"

    # Check if all levels are correct
    all_correct = all(level_results.get(l, False) for l in range(NUM_THINKING_LEVELS + 1))
    if all_correct:
        return "consistent"

    # Thresholds for classification
    HIGH_OVERLAP = 0.5
    LOW_OVERLAP = 0.2
    SHORT_COT = 200  # characters

    ngram = overlap.get("ngram_overlap", 0)
    cot_len = overlap.get("cot_length", 0)

    if ngram >= HIGH_OVERLAP:
        return "high_overlap_failure"  # same strategy, execution error
    elif cot_len < SHORT_COT and ngram < LOW_OVERLAP:
        return "shortcut"  # pattern matching
    else:
        return "low_overlap_success"  # different valid strategy


def analyze_path_divergence(
    problems: List[ProblemWithChain],
    model_name: str,
) -> dict:
    """
    Analyze path divergence for a single model.

    For each problem:
      - Check *all* L0 samples (not just the first) for correctness.
        Every L0-correct sample becomes a separate case record — we classify
        its CoT's overlap with the SPARKLE decomposition independently.
      - For subproblem levels 1..N, use pass@K (any of K samples correct) to
        determine whether the model can complete that step given scaffold.
      - "has_divergence" = problem has at least one L0-correct sample, but
        at least one subproblem level has zero correct samples.

    Returns per-problem stats (n_problems_*) and per-sample classification
    counts over the L0-correct solutions, giving >4x the data for H3 vs.
    the prior pass@1-only implementation.
    """
    cached = load_all_results(model_name)
    if not cached:
        return {}

    divergence_cases = []
    classification_counts = Counter()
    overlap_stats = defaultdict(list)

    n_problems_examined = 0
    n_problems_l0_correct = 0
    n_problems_divergent = 0
    n_problems_l0_fail = 0

    for p in problems:
        gt = p.ground_truth

        key0 = f"{p.problem_id}:0"
        if key0 not in cached:
            continue
        n_problems_examined += 1

        responses_l0 = cached[key0]
        # Check ALL L0 samples, not just the first
        l0_correct_indices = [
            i for i, r in enumerate(responses_l0) if grade_response(r, gt)
        ]
        any_l0_correct = len(l0_correct_indices) > 0

        # For levels 1..N, use pass@K (any sample correct) as the success signal.
        # This answers "can the model complete this step given ground-truth scaffold?",
        # which is independent of which L0 sample we picked.
        level_results = {}
        for level in range(NUM_THINKING_LEVELS + 1):
            key = f"{p.problem_id}:{level}"
            if key not in cached:
                continue
            responses = cached[key]
            level_results[level] = any(grade_response(r, gt) for r in responses)

        has_divergence = any_l0_correct and any(
            not level_results.get(l, True)
            for l in range(1, NUM_THINKING_LEVELS + 1)
        )

        if not any_l0_correct:
            # Problem the model can't solve unscaffolded — no path to analyze.
            n_problems_l0_fail += 1
            continue

        n_problems_l0_correct += 1
        if has_divergence:
            n_problems_divergent += 1

        # Per-sample classification: every correct L0 sample contributes one
        # classification and one overlap data point.
        for idx in l0_correct_indices:
            cot = extract_cot_from_response(responses_l0[idx])
            overlap = compute_reasoning_overlap(cot, p.thinking_splits)
            classification = classify_divergence_auto(
                overlap, True, level_results
            )
            classification_counts[classification] += 1

            overlap_stats["ngram_overlap"].append(overlap["ngram_overlap"])
            overlap_stats["step_coverage"].append(overlap["step_coverage"])

            divergence_cases.append({
                "problem_id": p.problem_id,
                "sample_idx": idx,
                "difficulty": p.difficulty,
                "domain": p.domain,
                "level0_correct": True,
                "level_results": level_results,
                "has_divergence": has_divergence,
                "classification": classification,
                "overlap": overlap,
                "question_preview": p.question_raw[:200],
                "cot_preview": cot[:500],
                "failed_levels": [
                    l for l in range(1, NUM_THINKING_LEVELS + 1)
                    if not level_results.get(l, True)
                ],
            })

    n_l0_correct_samples = len(divergence_cases)

    return {
        "model": model_name,
        "n_problems_examined": n_problems_examined,
        "n_problems_l0_correct": n_problems_l0_correct,
        "n_problems_l0_fail": n_problems_l0_fail,
        "n_problems_divergent": n_problems_divergent,
        "n_l0_correct_samples": n_l0_correct_samples,
        "divergence_rate": (
            n_problems_divergent / n_problems_l0_correct
            if n_problems_l0_correct else 0
        ),
        # Classifications are over L0-correct samples (not problems).
        "classification_counts": dict(classification_counts),
        "overlap_stats": {
            "mean_ngram_overlap": float(np.mean(overlap_stats["ngram_overlap"])) if overlap_stats["ngram_overlap"] else 0,
            "mean_step_coverage": float(np.mean(overlap_stats["step_coverage"])) if overlap_stats["step_coverage"] else 0,
        },
        "divergence_cases": divergence_cases,
        # Backward-compatible aliases (downstream scripts/figures used these)
        "n_total": n_problems_examined,
        "n_level0_correct": n_problems_l0_correct,
        "n_divergent": n_problems_divergent,
    }


def select_manual_review_cases(
    all_results: Dict[str, dict],
    n_cases: int = PATH_DIVERGENCE_SAMPLE_SIZE,
) -> List[dict]:
    """
    Select representative cases for manual review.

    Prioritizes: divergent cases from RL models, balanced across
    difficulty levels and divergence types.
    """
    review_cases = []

    for model_name in ["stage2", "stage1", "base"]:
        if model_name not in all_results:
            continue

        cases = all_results[model_name].get("divergence_cases", [])
        # Filter to divergent cases
        divergent = [c for c in cases if c.get("has_divergence")]

        # Sort by diversity of classification
        for c in divergent[:n_cases // len(MODELS)]:
            review_cases.append({
                "model": model_name,
                **c,
            })

    return review_cases[:n_cases]


def print_divergence_summary(all_results: Dict[str, dict]):
    """Print formatted path divergence summary."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Path Divergence Analysis")
    print("=" * 80)

    for model_name, result in all_results.items():
        n_examined = result.get("n_problems_examined", 0)
        n_l0 = result.get("n_problems_l0_correct", 0)
        n_div = result.get("n_problems_divergent", 0)
        n_samples = result.get("n_l0_correct_samples", 0)
        rate = result.get("divergence_rate", 0)

        print(f"\n--- {model_name} ---")
        print(f"  Problems examined:     {n_examined}")
        if n_examined:
            print(f"  Problems with ≥1 L0-correct sample: {n_l0} ({n_l0/n_examined*100:.1f}%)")
        print(f"  L0-correct samples (total):          {n_samples}")
        print(f"  Divergent problems:    {n_div} ({rate*100:.1f}% of L0-correct problems)")

        # Per-sample classification over L0-correct solutions
        total_samples = sum(result.get("classification_counts", {}).values())
        print(f"\n  Classification breakdown (per L0-correct sample, n={total_samples}):")
        for cls, count in sorted(result.get("classification_counts", {}).items()):
            pct = count / total_samples * 100 if total_samples else 0
            print(f"    {cls:<25} {count:>5} ({pct:.1f}%)")

        print(f"\n  Overlap statistics (over L0-correct samples):")
        stats = result.get("overlap_stats", {})
        print(f"    Mean n-gram overlap:  {stats.get('mean_ngram_overlap', 0):.3f}")
        print(f"    Mean step coverage:   {stats.get('mean_step_coverage', 0):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Path Divergence")
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--n-review", type=int, default=PATH_DIVERGENCE_SAMPLE_SIZE,
                        help="Number of cases to select for manual review")
    args = parser.parse_args()

    # Load data
    cache_path = os.path.join(RESULTS_DIR, "benchmark_data.json")
    problems = load_sparkle_benchmark(
        max_problems=args.max_problems,
        cache_path=cache_path,
    )

    # Analyze each model
    all_results = {}
    for model_name in MODELS:
        print(f"\n--- Analyzing {model_name} ---")
        result = analyze_path_divergence(problems, model_name)
        if result:
            all_results[model_name] = result

    # Print summary
    print_divergence_summary(all_results)

    # Select manual review cases
    review_cases = select_manual_review_cases(all_results, n_cases=args.n_review)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Summary (without full case details)
    summary = {
        "experiment": "exp3_path_divergence",
        "models": list(MODELS.keys()),
        "results": {
            m: {k: v for k, v in r.items() if k != "divergence_cases"}
            for m, r in all_results.items()
        },
    }
    summary_path = os.path.join(RESULTS_DIR, "exp3_path_divergence.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=convert)

    # Full divergence cases (for detailed analysis)
    cases_path = os.path.join(RESULTS_DIR, "exp3_divergence_cases.json")
    cases_data = {m: r.get("divergence_cases", []) for m, r in all_results.items()}
    with open(cases_path, "w") as f:
        json.dump(cases_data, f, indent=2, default=convert)

    # Manual review cases
    review_path = os.path.join(RESULTS_DIR, "exp3_manual_review.json")
    with open(review_path, "w") as f:
        json.dump(review_cases, f, indent=2, default=convert)

    print(f"\nSummary saved to {summary_path}")
    print(f"All divergence cases saved to {cases_path}")
    print(f"Manual review cases ({len(review_cases)}) saved to {review_path}")


if __name__ == "__main__":
    main()
