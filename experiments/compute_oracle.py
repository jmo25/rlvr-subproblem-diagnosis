"""
Compute L0 oracle (set-union) pass@1 / pass@16 across the 4 models from
phase2_r1 results, leveraging exp3_divergence_cases.json which already
records every L0-correct (problem_id, sample_idx) pair.

L0 pass@1 set per model  = problems where sample_idx 0 is L0-correct.
L0 pass@16 set per model = unique problem_ids appearing in the model's list.
"""

import json
import os
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "phase2_r1", "results")
N_TOTAL = 500
MODELS = ["base", "stage1", "stage2", "r1distill"]


def load_correct_sets():
    with open(os.path.join(RESULTS_DIR, "exp3_divergence_cases.json")) as f:
        cases = json.load(f)

    pass1, pass16 = {}, {}
    for m in MODELS:
        # (problem_id, sample_idx) pairs where L0 was correct
        pairs = [(c["problem_id"], c["sample_idx"]) for c in cases.get(m, [])]
        pass1[m] = {pid for pid, sid in pairs if sid == 0}
        pass16[m] = {pid for pid, _ in pairs}
    return pass1, pass16


def fmt_set(s):
    return f"{len(s):>3}/500 = {len(s)/N_TOTAL:.3f}"


def union(*sets):
    out = set()
    for s in sets:
        out |= s
    return out


def report(name, sets):
    print(f"\n=== L0 {name} ===")
    for m in MODELS:
        print(f"  {m:<10} {fmt_set(sets[m])}")
    print()
    pairs = [
        ("STAGE2 ∪ R1DISTILL",        union(sets["stage2"], sets["r1distill"])),
        ("STAGE1 ∪ R1DISTILL",        union(sets["stage1"], sets["r1distill"])),
        ("STAGE1 ∪ STAGE2 (executors)", union(sets["stage1"], sets["stage2"])),
        ("STAGE2 ∪ R1DISTILL ∪ BASE", union(sets["stage2"], sets["r1distill"], sets["base"])),
        ("ALL FOUR",                  union(*[sets[m] for m in MODELS])),
    ]
    print(f"  {'Oracle union':<32} {'count':>10}  {'gain over best':>15}")
    print(f"  {'-'*32} {'-'*10} {'-'*15}")
    for label, s in pairs:
        best_individual = max(len(sets[m]) for m in MODELS)
        gain_pp = (len(s) - best_individual) / N_TOTAL * 100
        print(f"  {label:<32} {fmt_set(s)}    +{gain_pp:.1f}pp")


def pairwise_overlap(sets):
    print("\n=== Pairwise problem-set overlap (Jaccard) ===")
    print(f"  {'pair':<25} {'A∩B':>5} {'A∪B':>5} {'Jacc':>6}")
    for i, a in enumerate(MODELS):
        for b in MODELS[i+1:]:
            inter = sets[a] & sets[b]
            uni = sets[a] | sets[b]
            j = len(inter) / len(uni) if uni else 0
            print(f"  {a + ' / ' + b:<25} {len(inter):>5} {len(uni):>5} {j:>6.3f}")


def main():
    pass1, pass16 = load_correct_sets()
    report("pass@1", pass1)
    report("pass@16", pass16)
    print("\n--- pass@16 pairwise ---")
    pairwise_overlap(pass16)


if __name__ == "__main__":
    main()
