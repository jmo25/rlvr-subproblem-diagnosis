"""
Data loader for SPARKLE benchmark with subproblem decomposition.

Loads the hardmath dataset and creates subproblem chains by splitting
each problem's reasoning into progressive thinking levels (0-4).
"""

import os
import sys
import re
import json
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict

import datasets

# Add project root for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.data.prepare_stage_two_data_aug import smart_split_thinking_process

from config import (
    DATASET_NAME, DATASET_SPLIT, NUM_THINKING_LEVELS, INSTRUCTION,
    RESULTS_DIR
)


@dataclass
class SubproblemInstance:
    """A single subproblem instance at a specific thinking level."""
    problem_id: int
    thinking_level: int          # 0 = no hints, 1..4 = progressive reasoning
    total_levels: int
    question_raw: str            # original math question
    prompt: str                  # full prompt to send to model
    ground_truth: str            # correct final answer
    is_augmented: bool           # True if thinking_level > 0
    # metadata from dataset
    difficulty: str = ""
    domain: str = ""
    source: str = ""


@dataclass
class ProblemWithChain:
    """A full problem with its subproblem chain."""
    problem_id: int
    question_raw: str
    ground_truth: str
    difficulty: str
    domain: str
    source: str
    subproblems: List[SubproblemInstance] = field(default_factory=list)
    # The reasoning splits (for path divergence analysis)
    thinking_splits: List[str] = field(default_factory=list)
    full_solution: str = ""


def build_prompt_level0(question_raw: str) -> str:
    """Build prompt for level 0 (no partial reasoning provided)."""
    return f"{INSTRUCTION}\n\nUser: {question_raw} Assistant:"


def build_prompt_augmented(question_raw: str, partial_thinking: str) -> str:
    """Build prompt for augmented levels (partial reasoning provided)."""
    return (
        f"{INSTRUCTION}\n\nUser: {question_raw} "
        f"Assistant: <think>\n{partial_thinking}\n"
    )


def load_sparkle_benchmark(
    max_problems: Optional[int] = None,
    cache_path: Optional[str] = None,
) -> List[ProblemWithChain]:
    """
    Load the SPARKLE hardmath dataset and create subproblem chains.

    Each problem is decomposed into thinking_level 0..N where:
      - Level 0: original problem, no hints
      - Level 1..N: progressively more reasoning steps provided

    Args:
        max_problems: limit number of problems (for debugging)
        cache_path: path to cache the processed data

    Returns:
        List of ProblemWithChain objects
    """
    # Check cache
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        return _load_cache(cache_path)

    print(f"Loading dataset: {DATASET_NAME}")
    dataset = datasets.load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    if max_problems:
        dataset = dataset.select(range(min(max_problems, len(dataset))))

    problems = []
    skipped = 0

    for idx, example in enumerate(dataset):
        question_raw = example.get("problem", "")
        ground_truth = example.get("answer", "")

        # Extract metadata
        difficulty = example.get("difficulty", example.get("level", ""))
        domain = example.get("domain", example.get("type", ""))
        source = example.get("source", example.get("data_source", ""))

        # Get the reasoning process
        thinking_process = example.get("CoT") or example.get("solution") or ""

        if not thinking_process:
            skipped += 1
            continue

        # Split reasoning into progressive steps
        thinking_splits = smart_split_thinking_process(
            thinking_process, NUM_THINKING_LEVELS
        )

        # Build problem chain
        chain = ProblemWithChain(
            problem_id=idx,
            question_raw=question_raw,
            ground_truth=str(ground_truth),
            difficulty=str(difficulty),
            domain=str(domain),
            source=str(source),
            thinking_splits=thinking_splits,
            full_solution=thinking_process,
        )

        # Level 0: no partial reasoning
        chain.subproblems.append(SubproblemInstance(
            problem_id=idx,
            thinking_level=0,
            total_levels=len(thinking_splits),
            question_raw=question_raw,
            prompt=build_prompt_level0(question_raw),
            ground_truth=str(ground_truth),
            is_augmented=False,
            difficulty=str(difficulty),
            domain=str(domain),
            source=str(source),
        ))

        # Levels 1..N: progressively more reasoning
        for i in range(len(thinking_splits)):
            partial = "\n".join(thinking_splits[:i + 1])
            chain.subproblems.append(SubproblemInstance(
                problem_id=idx,
                thinking_level=i + 1,
                total_levels=len(thinking_splits),
                question_raw=question_raw,
                prompt=build_prompt_augmented(question_raw, partial),
                ground_truth=str(ground_truth),
                is_augmented=True,
                difficulty=str(difficulty),
                domain=str(domain),
                source=str(source),
            ))

        problems.append(chain)

    print(f"Loaded {len(problems)} problems ({skipped} skipped, no solution)")
    print(f"Total subproblem instances: {sum(len(p.subproblems) for p in problems)}")

    # Cache
    if cache_path:
        _save_cache(problems, cache_path)

    return problems


def _save_cache(problems: List[ProblemWithChain], path: str):
    """Save processed problems to JSON cache."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = []
    for p in problems:
        d = asdict(p)
        data.append(d)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Cached {len(problems)} problems to {path}")


def _load_cache(path: str) -> List[ProblemWithChain]:
    """Load processed problems from JSON cache."""
    with open(path) as f:
        data = json.load(f)
    problems = []
    for d in data:
        subs = [SubproblemInstance(**s) for s in d.pop("subproblems")]
        p = ProblemWithChain(**d)
        p.subproblems = subs
        problems.append(p)
    return problems


def get_all_subproblems(problems: List[ProblemWithChain]) -> List[SubproblemInstance]:
    """Flatten all subproblem instances from all problems."""
    return [s for p in problems for s in p.subproblems]


def get_subproblems_by_level(
    problems: List[ProblemWithChain], level: int
) -> List[SubproblemInstance]:
    """Get all subproblem instances at a specific thinking level."""
    return [s for p in problems for s in p.subproblems if s.thinking_level == level]


if __name__ == "__main__":
    # Quick test
    problems = load_sparkle_benchmark(max_problems=5)
    for p in problems[:2]:
        print(f"\nProblem {p.problem_id}: {p.question_raw[:80]}...")
        print(f"  Ground truth: {p.ground_truth}")
        print(f"  Difficulty: {p.difficulty}, Domain: {p.domain}")
        print(f"  Subproblem levels: {len(p.subproblems)}")
        for s in p.subproblems:
            print(f"    Level {s.thinking_level}: prompt length = {len(s.prompt)}")
