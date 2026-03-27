"""
vLLM-based inference engine for sampling multiple responses per prompt.

Supports batch inference across all models and subproblem levels,
with caching to avoid redundant computation.
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Optional
from dataclasses import asdict

from config import (
    MODELS, MAX_K, TEMPERATURE, TOP_P, MAX_NEW_TOKENS,
    TENSOR_PARALLEL_SIZE, GPU_MEMORY_UTILIZATION, DTYPE,
    RESULTS_DIR,
)
from data_loader import (
    load_sparkle_benchmark, ProblemWithChain, SubproblemInstance,
    get_subproblems_by_level,
)


def get_cache_path(model_name: str) -> str:
    """Get path for caching inference results."""
    return os.path.join(RESULTS_DIR, f"inference_{model_name}.jsonl")


def load_cached_results(model_name: str) -> Dict[str, List[str]]:
    """
    Load cached inference results.

    Returns:
        dict mapping "problem_id:thinking_level" -> list of responses
    """
    cache_path = get_cache_path(model_name)
    cached = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                entry = json.loads(line)
                key = f"{entry['problem_id']}:{entry['thinking_level']}"
                cached[key] = entry["responses"]
    return cached


def save_result(model_name: str, problem_id: int, thinking_level: int,
                responses: List[str]):
    """Append a single result to the cache file."""
    cache_path = get_cache_path(model_name)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    entry = {
        "problem_id": problem_id,
        "thinking_level": thinking_level,
        "responses": responses,
    }
    with open(cache_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run_inference_vllm(
    model_path: str,
    model_name: str,
    problems: List[ProblemWithChain],
    num_samples: int = MAX_K,
    batch_size: int = 64,
):
    """
    Run vLLM inference for all subproblems of all problems.

    For each subproblem, generates num_samples responses.
    Results are cached incrementally to RESULTS_DIR.

    Args:
        model_path: HuggingFace model path or local path
        model_name: short name for caching (e.g., "base", "stage1")
        problems: list of ProblemWithChain
        num_samples: number of responses per subproblem
        batch_size: number of prompts per vLLM batch
    """
    from vllm import LLM, SamplingParams

    # Load cached results to skip already-computed
    cached = load_cached_results(model_name)
    print(f"[{model_name}] Cached results: {len(cached)} subproblems")

    # Collect all subproblems that need inference
    todo: List[SubproblemInstance] = []
    for p in problems:
        for s in p.subproblems:
            key = f"{s.problem_id}:{s.thinking_level}"
            if key not in cached:
                todo.append(s)

    if not todo:
        print(f"[{model_name}] All subproblems already cached. Skipping.")
        return

    print(f"[{model_name}] Running inference on {len(todo)} subproblems "
          f"(x{num_samples} samples each)")

    # Initialize vLLM
    llm = LLM(
        model=model_path,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        dtype=DTYPE,
        trust_remote_code=True,
        max_model_len=MAX_NEW_TOKENS + 2048,  # prompt + generation
    )

    sampling_params = SamplingParams(
        n=num_samples,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_NEW_TOKENS,
    )

    # Process in batches
    total_batches = (len(todo) + batch_size - 1) // batch_size
    t0 = time.time()

    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(todo))
        batch = todo[batch_start:batch_end]

        prompts = [s.prompt for s in batch]

        print(f"  Batch {batch_idx + 1}/{total_batches} "
              f"({len(prompts)} prompts)...", end="", flush=True)

        outputs = llm.generate(prompts, sampling_params)

        for sub, output in zip(batch, outputs):
            responses = [o.text for o in output.outputs]
            save_result(model_name, sub.problem_id, sub.thinking_level, responses)

        elapsed = time.time() - t0
        rate = (batch_end) / elapsed if elapsed > 0 else 0
        print(f" done ({rate:.1f} subproblems/s)")

    total_time = time.time() - t0
    print(f"[{model_name}] Inference complete: {len(todo)} subproblems in {total_time:.1f}s")


def load_all_results(model_name: str) -> Dict[str, List[str]]:
    """
    Load all cached inference results for a model.

    Returns:
        dict mapping "problem_id:thinking_level" -> list of responses
    """
    return load_cached_results(model_name)


def main():
    parser = argparse.ArgumentParser(description="Run vLLM inference for experiments")
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"],
                        default="all", help="Which model to run")
    parser.add_argument("--max-problems", type=int, default=None,
                        help="Limit number of problems (for debugging)")
    parser.add_argument("--num-samples", type=int, default=MAX_K,
                        help="Number of samples per subproblem")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tp", type=int, default=TENSOR_PARALLEL_SIZE,
                        help="Tensor parallel size")
    args = parser.parse_args()

    # Override TP if specified
    global TENSOR_PARALLEL_SIZE
    TENSOR_PARALLEL_SIZE = args.tp

    # Load data
    cache_path = os.path.join(RESULTS_DIR, "benchmark_data.json")
    problems = load_sparkle_benchmark(
        max_problems=args.max_problems,
        cache_path=cache_path,
    )

    # Run inference
    model_list = list(MODELS.keys()) if args.model == "all" else [args.model]

    for model_name in model_list:
        model_path = MODELS[model_name]
        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({model_path})")
        print(f"{'='*60}")
        run_inference_vllm(
            model_path=model_path,
            model_name=model_name,
            problems=problems,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
