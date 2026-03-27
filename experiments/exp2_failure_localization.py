"""
Experiment 2: Failure Localization

For the subproblem chain (level 0 -> 1 -> 2 -> 3 -> 4):
  - Track which level is the first success point for each problem
  - Compute per-level success rates
  - Stratify by difficulty and mathematical domain
  - Compare failure patterns between base and RL models

Hypothesis H2: The subproblem gap is not uniformly distributed — certain
steps are systematically harder. This reveals whether RLVR improves
execution (carrying out steps) or planning (knowing what steps to take).
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Tuple

from config import MODELS, K_VALUES, RESULTS_DIR, NUM_THINKING_LEVELS
from data_loader import load_sparkle_benchmark, ProblemWithChain
from inference import load_all_results
from grading import grade_response


def analyze_failure_localization(
    problems: List[ProblemWithChain],
    model_name: str,
    k: int = 1,
) -> dict:
    """
    For each problem, determine at which thinking level the model first
    produces a correct answer (using pass@k criterion).

    Args:
        problems: list of problems with subproblem chains
        model_name: which model's results to analyze
        k: number of samples to consider (pass@k)

    Returns:
        dict with failure localization analysis
    """
    cached = load_all_results(model_name)
    if not cached:
        print(f"  No results for {model_name}")
        return {}

    # Per-problem analysis
    problem_analyses = []
    # Per-level success counts
    level_success = defaultdict(int)
    level_total = defaultdict(int)
    # First-success-level distribution
    first_success_dist = Counter()
    # Never-succeed count
    never_succeed = 0
    # Always-succeed (even at level 0) count
    always_succeed = 0

    # Stratified analysis
    by_difficulty = defaultdict(lambda: defaultdict(lambda: {"success": 0, "total": 0}))
    by_domain = defaultdict(lambda: defaultdict(lambda: {"success": 0, "total": 0}))

    for p in problems:
        gt = p.ground_truth
        level_results = {}  # level -> is_correct (pass@k)
        first_success = None

        for level in range(NUM_THINKING_LEVELS + 1):
            key = f"{p.problem_id}:{level}"
            if key not in cached:
                continue

            responses = cached[key][:k]
            is_correct = any(grade_response(r, gt) for r in responses)

            level_results[level] = is_correct
            level_success[level] += int(is_correct)
            level_total[level] += 1

            # Stratify
            if p.difficulty:
                by_difficulty[p.difficulty][level]["total"] += 1
                by_difficulty[p.difficulty][level]["success"] += int(is_correct)
            if p.domain:
                by_domain[p.domain][level]["total"] += 1
                by_domain[p.domain][level]["success"] += int(is_correct)

            if is_correct and first_success is None:
                first_success = level

        if first_success is not None:
            first_success_dist[first_success] += 1
            if first_success == 0:
                always_succeed += 1
        else:
            never_succeed += 1

        problem_analyses.append({
            "problem_id": p.problem_id,
            "difficulty": p.difficulty,
            "domain": p.domain,
            "level_results": level_results,
            "first_success_level": first_success,
        })

    # Compute per-level success rates
    level_success_rates = {}
    for level in range(NUM_THINKING_LEVELS + 1):
        if level_total[level] > 0:
            level_success_rates[level] = {
                "rate": level_success[level] / level_total[level],
                "correct": level_success[level],
                "total": level_total[level],
            }

    # Compute transition gains (how much does each additional level help?)
    transition_gains = {}
    for level in range(1, NUM_THINKING_LEVELS + 1):
        prev_rate = level_success_rates.get(level - 1, {}).get("rate", 0)
        curr_rate = level_success_rates.get(level, {}).get("rate", 0)
        transition_gains[f"{level-1}->{level}"] = {
            "gain": curr_rate - prev_rate,
            "prev_rate": prev_rate,
            "curr_rate": curr_rate,
        }

    # Stratified success rates
    stratified_difficulty = {}
    for diff, levels in by_difficulty.items():
        stratified_difficulty[diff] = {}
        for level, counts in levels.items():
            if counts["total"] > 0:
                stratified_difficulty[diff][level] = {
                    "rate": counts["success"] / counts["total"],
                    "n": counts["total"],
                }

    stratified_domain = {}
    for dom, levels in by_domain.items():
        stratified_domain[dom] = {}
        for level, counts in levels.items():
            if counts["total"] > 0:
                stratified_domain[dom][level] = {
                    "rate": counts["success"] / counts["total"],
                    "n": counts["total"],
                }

    return {
        "model": model_name,
        "k": k,
        "n_problems": len(problem_analyses),
        "level_success_rates": level_success_rates,
        "transition_gains": transition_gains,
        "first_success_distribution": dict(first_success_dist),
        "always_succeed_count": always_succeed,
        "never_succeed_count": never_succeed,
        "stratified_by_difficulty": stratified_difficulty,
        "stratified_by_domain": stratified_domain,
        "problem_analyses": problem_analyses,
    }


def compare_failure_patterns(all_results: Dict[str, dict]) -> dict:
    """
    Compare failure localization patterns across models.

    Identifies:
    - Steps where RL models gain the most over base
    - Whether RL benefit is concentrated on early vs late reasoning
    - Domain-specific RL advantages
    """
    if "base" not in all_results:
        return {}

    comparisons = {}

    for rl_model in ["stage1", "stage2"]:
        if rl_model not in all_results:
            continue

        base = all_results["base"]
        rl = all_results[rl_model]

        # Per-level RL advantage
        level_advantages = {}
        for level in range(NUM_THINKING_LEVELS + 1):
            base_rate = base.get("level_success_rates", {}).get(level, {}).get("rate", 0)
            rl_rate = rl.get("level_success_rates", {}).get(level, {}).get("rate", 0)
            level_advantages[level] = {
                "base_rate": base_rate,
                "rl_rate": rl_rate,
                "advantage": rl_rate - base_rate,
            }

        # Find which level has max RL advantage
        max_adv_level = max(level_advantages, key=lambda l: level_advantages[l]["advantage"])

        # Per-difficulty RL advantage at level 0
        difficulty_advantages = {}
        for diff in set(list(base.get("stratified_by_difficulty", {}).keys()) +
                       list(rl.get("stratified_by_difficulty", {}).keys())):
            base_l0 = base.get("stratified_by_difficulty", {}).get(diff, {}).get(0, {}).get("rate", 0)
            rl_l0 = rl.get("stratified_by_difficulty", {}).get(diff, {}).get(0, {}).get("rate", 0)
            difficulty_advantages[diff] = {
                "base_rate": base_l0,
                "rl_rate": rl_l0,
                "advantage": rl_l0 - base_l0,
            }

        comparisons[f"base_vs_{rl_model}"] = {
            "level_advantages": level_advantages,
            "max_advantage_level": max_adv_level,
            "max_advantage_value": level_advantages[max_adv_level]["advantage"],
            "difficulty_advantages": difficulty_advantages,
        }

    return comparisons


def print_failure_summary(all_results: Dict[str, dict]):
    """Print formatted failure localization summary."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Failure Localization Summary")
    print("=" * 80)

    # Per-level success rates table
    print("\n--- Per-Level Success Rates (pass@1) ---")
    header = f"{'Level':<8}" + "".join(f"{m:<15}" for m in MODELS)
    print(header)
    print("-" * len(header))

    for level in range(NUM_THINKING_LEVELS + 1):
        row = f"{level:<8}"
        for model_name in MODELS:
            if model_name in all_results:
                rate = all_results[model_name].get("level_success_rates", {}).get(
                    level, {}
                ).get("rate", 0)
                row += f"{rate:.3f}          "
            else:
                row += f"{'N/A':<15}"
        print(row)

    # Transition gains
    print("\n--- Transition Gains (per-level improvement) ---")
    for model_name in MODELS:
        if model_name not in all_results:
            continue
        print(f"\n  {model_name}:")
        gains = all_results[model_name].get("transition_gains", {})
        for trans, data in sorted(gains.items()):
            print(f"    {trans}: +{data['gain']:.3f} "
                  f"({data['prev_rate']:.3f} -> {data['curr_rate']:.3f})")

    # First success distribution
    print("\n--- First Success Level Distribution ---")
    for model_name in MODELS:
        if model_name not in all_results:
            continue
        dist = all_results[model_name].get("first_success_distribution", {})
        never = all_results[model_name].get("never_succeed_count", 0)
        total = all_results[model_name].get("n_problems", 0)
        print(f"\n  {model_name} (n={total}):")
        for level in sorted(dist.keys()):
            pct = dist[level] / total * 100 if total else 0
            print(f"    Level {level}: {dist[level]} ({pct:.1f}%)")
        pct_never = never / total * 100 if total else 0
        print(f"    Never:   {never} ({pct_never:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Failure Localization")
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--k", type=int, default=1,
                        help="k value for pass@k criterion (default: 1)")
    args = parser.parse_args()

    # Load data
    cache_path = os.path.join(RESULTS_DIR, "benchmark_data.json")
    problems = load_sparkle_benchmark(
        max_problems=args.max_problems,
        cache_path=cache_path,
    )

    # Run analysis for each model
    all_results = {}
    for model_name in MODELS:
        print(f"\n--- Analyzing {model_name} ---")
        result = analyze_failure_localization(problems, model_name, k=args.k)
        if result:
            all_results[model_name] = result

    # Compare patterns
    comparisons = compare_failure_patterns(all_results)

    # Print summary
    print_failure_summary(all_results)

    if comparisons:
        print("\n--- Model Comparison ---")
        for comp_name, comp_data in comparisons.items():
            print(f"\n{comp_name}:")
            print(f"  Max RL advantage at level {comp_data['max_advantage_level']}: "
                  f"+{comp_data['max_advantage_value']:.3f}")
            for level, adv in sorted(comp_data["level_advantages"].items()):
                print(f"  Level {level}: base={adv['base_rate']:.3f}, "
                      f"rl={adv['rl_rate']:.3f}, advantage={adv['advantage']:+.3f}")

    # Save results
    output = {
        "experiment": "exp2_failure_localization",
        "k": args.k,
        "models": list(MODELS.keys()),
        "results": {m: {k: v for k, v in r.items() if k != "problem_analyses"}
                    for m, r in all_results.items()},
        "comparisons": comparisons,
    }
    # Save detailed per-problem analyses separately (large)
    detail_output = {m: r.get("problem_analyses", []) for m, r in all_results.items()}

    output_path = os.path.join(RESULTS_DIR, "exp2_failure_localization.json")
    detail_path = os.path.join(RESULTS_DIR, "exp2_failure_detail.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=convert)
    with open(detail_path, "w") as f:
        json.dump(detail_output, f, indent=2, default=convert)

    print(f"\nResults saved to {output_path}")
    print(f"Detailed per-problem results saved to {detail_path}")


if __name__ == "__main__":
    main()
