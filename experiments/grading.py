"""
Grading utilities for experiment evaluation.

Reuses SPARKLE's answer extraction and grading functions from the
evaluation harness for consistency with the original paper.
"""

import os
import sys
from typing import List, Optional, Union
from collections import Counter

# Add the eval harness to path so we can import SPARKLE's grading
EVAL_TASKS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "eval",
    "lm-evaluation-harness", "lm_eval", "tasks", "sparkle"
)
sys.path.insert(0, EVAL_TASKS_DIR)

from utils import (
    extract_solution,
    extract_answer,
    grade_answer_mathd,
    grade_answer_sympy,
    extract_boxed_answer,
)


def grade_response(response: str, ground_truth: Union[str, List[str]]) -> bool:
    """
    Grade a single model response against ground truth.

    Uses SPARKLE's grading pipeline:
      1. Extract answer from response (boxed -> answer tags -> patterns)
      2. Compare against ground truth using mathd and sympy graders

    Args:
        response: model-generated text
        ground_truth: correct answer(s)

    Returns:
        True if the extracted answer matches ground truth
    """
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    elif isinstance(ground_truth, (int, float)):
        ground_truth = [str(ground_truth)]

    # Normalize ground truths
    gt_list = []
    for gt in ground_truth:
        gt = str(gt)
        if "\\boxed" in gt:
            extracted = extract_boxed_answer(gt)
            if extracted is not None:
                gt_list.append(extracted)
        else:
            gt_list.append(gt)

    if not gt_list:
        return False

    # Extract answer from response
    answer_text = extract_solution(response)
    if not answer_text:
        return False

    # Grade against each possible ground truth
    for gt in gt_list:
        if grade_answer_mathd(answer_text, gt) or grade_answer_sympy(answer_text, gt):
            return True

    return False


def extract_answer_from_response(response: str) -> Optional[str]:
    """Extract the answer text from a model response."""
    return extract_solution(response)


def compute_pass_at_k(responses: List[str], ground_truth: Union[str, List[str]],
                      k_values: List[int]) -> dict:
    """
    Compute pass@k for multiple k values given a list of responses.

    pass@k = 1 if at least one of the first k responses is correct.

    Args:
        responses: list of model-generated responses
        ground_truth: correct answer(s)
        k_values: list of k values to evaluate

    Returns:
        dict mapping k -> pass@k (0 or 1)
    """
    # Grade each response
    correct = []
    extracted = []
    for resp in responses:
        is_correct = grade_response(resp, ground_truth)
        correct.append(is_correct)
        extracted.append(extract_answer_from_response(resp))

    results = {}
    for k in k_values:
        if k > len(responses):
            results[k] = None  # not enough samples
        else:
            results[k] = int(any(correct[:k]))

    return results


def compute_majority_at_k(responses: List[str], ground_truth: Union[str, List[str]],
                          k_values: List[int]) -> dict:
    """
    Compute maj@k for multiple k values.

    maj@k = 1 if the majority-vote answer among the first k responses is correct.
    """
    if isinstance(ground_truth, str):
        ground_truth_list = [ground_truth]
    elif isinstance(ground_truth, (int, float)):
        ground_truth_list = [str(ground_truth)]
    else:
        ground_truth_list = ground_truth

    # Normalize ground truths
    gt_list = []
    for gt in ground_truth_list:
        gt = str(gt)
        if "\\boxed" in gt:
            extracted = extract_boxed_answer(gt)
            if extracted:
                gt_list.append(extracted)
        else:
            gt_list.append(gt)

    # Extract all answers
    extracted_answers = []
    for resp in responses:
        ans = extract_answer_from_response(resp)
        if ans:
            extracted_answers.append(ans)
        else:
            extracted_answers.append("")  # placeholder for no answer

    results = {}
    for k in k_values:
        if k > len(responses):
            results[k] = None
        else:
            subset = [a for a in extracted_answers[:k] if a]
            if not subset:
                results[k] = 0
            else:
                most_common = Counter(subset).most_common(1)[0][0]
                majority_correct = any(
                    grade_answer_mathd(most_common, gt) or
                    grade_answer_sympy(most_common, gt)
                    for gt in gt_list
                )
                results[k] = int(majority_correct)

    return results


def grade_responses_batch(
    responses_list: List[List[str]],
    ground_truths: List[Union[str, List[str]]],
    k_values: List[int],
) -> List[dict]:
    """
    Batch compute pass@k and maj@k for multiple problems.

    Args:
        responses_list: list of (list of responses) per problem
        ground_truths: list of ground truth(s) per problem
        k_values: k values to evaluate

    Returns:
        list of dicts with pass@k and maj@k for each problem
    """
    all_results = []
    for responses, gt in zip(responses_list, ground_truths):
        pass_k = compute_pass_at_k(responses, gt, k_values)
        maj_k = compute_majority_at_k(responses, gt, k_values)
        all_results.append({
            "pass_at_k": pass_k,
            "maj_at_k": maj_k,
            "num_correct": sum(1 for r in responses if grade_response(r, gt)),
            "num_total": len(responses),
        })
    return all_results
