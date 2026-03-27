"""
Experiment 1: Subproblem Pass@k Analysis

For each subproblem level (0-4) and model (base, stage1, stage2):
  - Compute pass@k for k = 1, 2, 4, 8, 16, 32
  - Compare pass@k curves between base and RL models
  - Identify crossover points where base overtakes RL at subproblem level
  - Compare with outcome-level (level 0) crossover point

Hypothesis H1: If RLVR only improves sampling efficiency, base models should
catch up to RL models at high pass@k even at the subproblem level. If RLVR
genuinely improves step-level reasoning, the RL advantage persists.
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict
from typing import List, Dict

from config import MODELS, K_VALUES, RESULTS_DIR, FIGURES_DIR, NUM_THINKING_LEVELS
from data_loader import load_sparkle_benchmark, ProblemWithChain
from inference import load_all_results
from grading import compute_pass_at_k, compute_majority_at_k, grade_response


def run_exp1(problems: List[ProblemWithChain], max_k: int = None):
    """
    Run Experiment 1: Subproblem Pass@k Analysis.

    For each model and thinking level, compute pass@k and maj@k
    across all problems.
    """
    if max_k:
        k_values = [k for k in K_VALUES if k <= max_k]
    else:
        k_values = K_VALUES

    results = {}  # model -> level -> {"pass@k": {k: mean}, "maj@k": {k: mean}, ...}

    for model_name in MODELS:
        print(f"\n--- Analyzing model: {model_name} ---")
        cached = load_all_results(model_name)

        if not cached:
            print(f"  No inference results found for {model_name}. Run inference.py first.")
            continue

        results[model_name] = {}

        for level in range(NUM_THINKING_LEVELS + 1):  # 0..4
            level_pass_k = defaultdict(list)  # k -> list of 0/1
            level_maj_k = defaultdict(list)
            level_correct_counts = []
            level_total = 0

            for p in problems:
                key = f"{p.problem_id}:{level}"
                if key not in cached:
                    continue

                responses = cached[key]
                gt = p.ground_truth
                level_total += 1

                # Compute pass@k
                pass_k = compute_pass_at_k(responses, gt, k_values)
                for k, val in pass_k.items():
                    if val is not None:
                        level_pass_k[k].append(val)

                # Compute maj@k
                maj_k = compute_majority_at_k(responses, gt, k_values)
                for k, val in maj_k.items():
                    if val is not None:
                        level_maj_k[k].append(val)

                # Count correct responses
                n_correct = sum(1 for r in responses if grade_response(r, gt))
                level_correct_counts.append(n_correct)

            if level_total == 0:
                print(f"  Level {level}: no data")
                continue

            # Aggregate
            level_results = {
                "n_problems": level_total,
                "pass_at_k": {},
                "maj_at_k": {},
                "avg_correct_rate": np.mean([c / len(cached[f"{problems[0].problem_id}:{level}"])
                                              for c in level_correct_counts]) if level_correct_counts else 0,
            }

            for k in k_values:
                if level_pass_k[k]:
                    level_results["pass_at_k"][k] = {
                        "mean": float(np.mean(level_pass_k[k])),
                        "std": float(np.std(level_pass_k[k])),
                        "n": len(level_pass_k[k]),
                    }
                if level_maj_k[k]:
                    level_results["maj_at_k"][k] = {
                        "mean": float(np.mean(level_maj_k[k])),
                        "std": float(np.std(level_maj_k[k])),
                        "n": len(level_maj_k[k]),
                    }

            results[model_name][level] = level_results

            # Print summary
            pass_str = ", ".join(
                f"k={k}: {level_results['pass_at_k'].get(k, {}).get('mean', 0):.3f}"
                for k in k_values
            )
            print(f"  Level {level} ({level_total} problems): pass@k = [{pass_str}]")

    return results


def find_crossover_points(results: dict) -> dict:
    """
    Find where the base model's pass@k overtakes the RL model's pass@k,
    both at the outcome level (level 0) and at each subproblem level.

    Returns:
        dict with crossover analysis
    """
    crossovers = {}

    if "base" not in results:
        return crossovers

    for rl_model in ["stage1", "stage2"]:
        if rl_model not in results:
            continue

        crossovers[f"base_vs_{rl_model}"] = {}

        for level in results["base"]:
            if level not in results[rl_model]:
                continue

            base_pass = results["base"][level].get("pass_at_k", {})
            rl_pass = results[rl_model][level].get("pass_at_k", {})

            crossover_k = None
            advantages = {}

            for k in sorted(K_VALUES):
                k_str = k  # keys might be int or str
                base_val = base_pass.get(k_str, {}).get("mean", 0)
                rl_val = rl_pass.get(k_str, {}).get("mean", 0)
                diff = rl_val - base_val  # positive = RL better
                advantages[k] = {
                    "base": base_val,
                    "rl": rl_val,
                    "diff": diff,
                }
                if diff < 0 and crossover_k is None:
                    crossover_k = k

            crossovers[f"base_vs_{rl_model}"][level] = {
                "crossover_k": crossover_k,
                "advantages": advantages,
            }

    return crossovers


def print_summary_table(results: dict):
    """Print a summary table of pass@k across models and levels."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Subproblem Pass@k Summary")
    print("=" * 80)

    for k in K_VALUES:
        print(f"\n--- Pass@{k} ---")
        header = f"{'Level':<8}" + "".join(f"{m:<15}" for m in MODELS)
        print(header)
        print("-" * len(header))

        for level in range(NUM_THINKING_LEVELS + 1):
            row = f"{level:<8}"
            for model_name in MODELS:
                if model_name in results and level in results[model_name]:
                    val = results[model_name][level].get("pass_at_k", {}).get(k, {})
                    if isinstance(val, dict):
                        row += f"{val.get('mean', 0):.3f}          "
                    else:
                        row += f"{'N/A':<15}"
                else:
                    row += f"{'N/A':<15}"
            print(row)


def save_results(results: dict, crossovers: dict, output_path: str):
    """Save experiment results to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output = {
        "experiment": "exp1_subproblem_passk",
        "k_values": K_VALUES,
        "models": list(MODELS.keys()),
        "results": results,
        "crossovers": crossovers,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=convert)

    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Subproblem Pass@k")
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--max-k", type=int, default=None,
                        help="Maximum k value to evaluate")
    args = parser.parse_args()

    # Load data
    cache_path = os.path.join(RESULTS_DIR, "benchmark_data.json")
    problems = load_sparkle_benchmark(
        max_problems=args.max_problems,
        cache_path=cache_path,
    )

    # Run analysis
    results = run_exp1(problems, max_k=args.max_k)

    # Crossover analysis
    crossovers = find_crossover_points(results)

    # Print summary
    print_summary_table(results)

    if crossovers:
        print("\n--- Crossover Analysis ---")
        for comparison, levels in crossovers.items():
            print(f"\n{comparison}:")
            for level, data in sorted(levels.items()):
                ck = data["crossover_k"]
                print(f"  Level {level}: crossover at k={ck if ck else 'never (RL always better)'}")

    # Save
    output_path = os.path.join(RESULTS_DIR, "exp1_subproblem_passk.json")
    save_results(results, crossovers, output_path)


if __name__ == "__main__":
    main()
