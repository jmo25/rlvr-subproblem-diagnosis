"""
Visualization scripts for all three experiments.

Generates publication-quality figures for:
  - Exp1: Pass@k curves per model and subproblem level
  - Exp2: Failure localization heatmaps and transition gain charts
  - Exp3: Path divergence classification pie charts and overlap distributions
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['figure.dpi'] = 150

from config import MODELS, K_VALUES, NUM_THINKING_LEVELS, RESULTS_DIR, FIGURES_DIR

# Color scheme
MODEL_COLORS = {
    "base": "#1f77b4",     # blue
    "stage1": "#ff7f0e",   # orange
    "stage2": "#2ca02c",   # green
}
MODEL_LABELS = {
    "base": "Qwen-2.5-Math-7B (Base)",
    "stage1": "SparkleRL Stage 1",
    "stage2": "SparkleRL Stage 2 (Aug)",
}


def load_exp_results(exp_name: str) -> dict:
    """Load experiment results JSON."""
    path = os.path.join(RESULTS_DIR, f"{exp_name}.json")
    if not os.path.exists(path):
        print(f"Results not found: {path}")
        return {}
    with open(path) as f:
        return json.load(f)


# ── Experiment 1 Visualizations ───────────────────────────────────────────

def plot_passk_curves_by_level(data: dict):
    """
    Plot pass@k curves for each subproblem level, comparing models.
    One subplot per level.
    """
    results = data.get("results", {})
    n_levels = NUM_THINKING_LEVELS + 1
    fig, axes = plt.subplots(1, n_levels, figsize=(4 * n_levels, 4), sharey=True)

    if n_levels == 1:
        axes = [axes]

    for level_idx, ax in enumerate(axes):
        level = level_idx  # levels 0..4

        for model_name in MODELS:
            if model_name not in results:
                continue
            level_data = results[model_name].get(str(level), {})
            pass_k = level_data.get("pass_at_k", {})

            ks = []
            vals = []
            for k in K_VALUES:
                entry = pass_k.get(str(k), {})
                if isinstance(entry, dict) and "mean" in entry:
                    ks.append(k)
                    vals.append(entry["mean"])

            if ks:
                ax.plot(ks, vals, "o-",
                        color=MODEL_COLORS.get(model_name, "gray"),
                        label=MODEL_LABELS.get(model_name, model_name),
                        linewidth=2, markersize=5)

        level_label = "Full Problem" if level == 0 else f"Level {level}"
        ax.set_title(level_label)
        ax.set_xlabel("k")
        ax.set_xscale("log", base=2)
        ax.set_xticks(K_VALUES)
        ax.set_xticklabels([str(k) for k in K_VALUES])
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        if level_idx == 0:
            ax.set_ylabel("Pass@k")

    axes[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    fig.suptitle("Experiment 1: Subproblem Pass@k Analysis", fontsize=14, y=1.02)
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "exp1_passk_by_level.pdf")
    fig.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def plot_passk_curves_by_model(data: dict):
    """
    Plot pass@k curves for each model, comparing subproblem levels.
    One subplot per model.
    """
    results = data.get("results", {})
    model_list = [m for m in MODELS if m in results]
    n_models = len(model_list)

    if n_models == 0:
        return

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]

    level_colors = plt.cm.viridis(np.linspace(0.1, 0.9, NUM_THINKING_LEVELS + 1))

    for model_idx, (model_name, ax) in enumerate(zip(model_list, axes)):
        model_data = results[model_name]

        for level in range(NUM_THINKING_LEVELS + 1):
            level_data = model_data.get(str(level), {})
            pass_k = level_data.get("pass_at_k", {})

            ks = []
            vals = []
            for k in K_VALUES:
                entry = pass_k.get(str(k), {})
                if isinstance(entry, dict) and "mean" in entry:
                    ks.append(k)
                    vals.append(entry["mean"])

            label = "Full Problem" if level == 0 else f"Level {level}"
            if ks:
                ax.plot(ks, vals, "o-", color=level_colors[level],
                        label=label, linewidth=2, markersize=5)

        ax.set_title(MODEL_LABELS.get(model_name, model_name))
        ax.set_xlabel("k")
        ax.set_xscale("log", base=2)
        ax.set_xticks(K_VALUES)
        ax.set_xticklabels([str(k) for k in K_VALUES])
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        if model_idx == 0:
            ax.set_ylabel("Pass@k")

    axes[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    fig.suptitle("Subproblem Levels vs. Pass@k per Model", fontsize=14, y=1.02)
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "exp1_passk_by_model.pdf")
    fig.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def plot_crossover_analysis(data: dict):
    """Plot the RL advantage (RL - base) as a function of k, per subproblem level."""
    crossovers = data.get("crossovers", {})

    for comp_name, levels in crossovers.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        level_colors = plt.cm.viridis(np.linspace(0.1, 0.9, NUM_THINKING_LEVELS + 1))

        for level in sorted(levels.keys(), key=lambda x: int(x)):
            advantages = levels[level].get("advantages", {})
            ks = []
            diffs = []
            for k in K_VALUES:
                entry = advantages.get(str(k), {})
                if "diff" in entry:
                    ks.append(k)
                    diffs.append(entry["diff"])

            label_str = "Full Problem" if int(level) == 0 else f"Level {level}"
            if ks:
                ax.plot(ks, diffs, "o-", color=level_colors[int(level)],
                        label=label_str, linewidth=2, markersize=5)

        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Parity')
        ax.set_xlabel("k")
        ax.set_ylabel("RL Advantage (RL pass@k - Base pass@k)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(K_VALUES)
        ax.set_xticklabels([str(k) for k in K_VALUES])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"RL Advantage vs k ({comp_name})")

        fig.tight_layout()
        path = os.path.join(FIGURES_DIR, f"exp1_crossover_{comp_name}.pdf")
        fig.savefig(path, bbox_inches='tight')
        print(f"Saved: {path}")
        plt.close()


# ── Experiment 2 Visualizations ───────────────────────────────────────────

def plot_success_rate_heatmap(data: dict):
    """
    Heatmap of success rates: models (rows) x subproblem levels (columns).
    """
    results = data.get("results", {})
    model_list = [m for m in MODELS if m in results]
    n_levels = NUM_THINKING_LEVELS + 1

    if not model_list:
        return

    matrix = np.zeros((len(model_list), n_levels))
    for i, model_name in enumerate(model_list):
        for level in range(n_levels):
            rate = results[model_name].get("level_success_rates", {}).get(
                str(level), {}
            ).get("rate", 0)
            matrix[i, level] = rate

    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(n_levels))
    ax.set_xticklabels(["Full Problem"] + [f"Level {l}" for l in range(1, n_levels)])
    ax.set_yticks(range(len(model_list)))
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in model_list])

    # Annotate cells
    for i in range(len(model_list)):
        for j in range(n_levels):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha='center', va='center',
                    color='white' if matrix[i, j] > 0.5 else 'black', fontsize=10)

    plt.colorbar(im, ax=ax, label='Success Rate')
    ax.set_title("Experiment 2: Per-Level Success Rates (pass@1)")

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "exp2_success_heatmap.pdf")
    fig.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def plot_transition_gains(data: dict):
    """Bar chart of transition gains (how much each level helps)."""
    results = data.get("results", {})
    model_list = [m for m in MODELS if m in results]

    if not model_list:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    n_transitions = NUM_THINKING_LEVELS
    x = np.arange(n_transitions)
    width = 0.25

    for i, model_name in enumerate(model_list):
        gains = results[model_name].get("transition_gains", {})
        values = []
        for level in range(1, NUM_THINKING_LEVELS + 1):
            key = f"{level-1}->{level}"
            gain = gains.get(key, {}).get("gain", 0)
            values.append(gain)

        ax.bar(x + i * width, values, width,
               label=MODEL_LABELS.get(model_name, model_name),
               color=MODEL_COLORS.get(model_name, "gray"))

    ax.set_xlabel("Reasoning Step Transition")
    ax.set_ylabel("Success Rate Gain")
    ax.set_title("Experiment 2: Per-Transition Success Rate Gains")
    ax.set_xticks(x + width)
    labels = [f"L{l-1}→L{l}" for l in range(1, NUM_THINKING_LEVELS + 1)]
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "exp2_transition_gains.pdf")
    fig.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def plot_first_success_distribution(data: dict):
    """Stacked bar chart of first-success-level distribution."""
    results = data.get("results", {})
    model_list = [m for m in MODELS if m in results]

    if not model_list:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    n_levels = NUM_THINKING_LEVELS + 1
    level_colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_levels + 1))

    bottom = np.zeros(len(model_list))

    # "Never succeed" first (at bottom)
    never_vals = []
    for model_name in model_list:
        n_total = results[model_name].get("n_problems", 1)
        never = results[model_name].get("never_succeed_count", 0)
        never_vals.append(never / n_total * 100)

    ax.barh(range(len(model_list)), never_vals, color='lightgray', label='Never')
    bottom = np.array(never_vals)

    for level in range(n_levels):
        vals = []
        for model_name in model_list:
            n_total = results[model_name].get("n_problems", 1)
            dist = results[model_name].get("first_success_distribution", {})
            count = dist.get(str(level), 0)
            vals.append(count / n_total * 100)

        label = "Full Problem" if level == 0 else f"Level {level}"
        ax.barh(range(len(model_list)), vals, left=bottom,
                color=level_colors[level], label=label)
        bottom += np.array(vals)

    ax.set_yticks(range(len(model_list)))
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in model_list])
    ax.set_xlabel("Percentage of Problems (%)")
    ax.set_title("Experiment 2: First Success Level Distribution")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "exp2_first_success_dist.pdf")
    fig.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


# ── Experiment 3 Visualizations ───────────────────────────────────────────

def plot_divergence_classification(data: dict):
    """Pie charts of divergence classification per model."""
    results = data.get("results", {})
    model_list = [m for m in MODELS if m in results]

    if not model_list:
        return

    fig, axes = plt.subplots(1, len(model_list), figsize=(5 * len(model_list), 4))
    if len(model_list) == 1:
        axes = [axes]

    cls_colors = {
        "consistent": "#2ca02c",
        "high_overlap_failure": "#ff7f0e",
        "low_overlap_success": "#1f77b4",
        "shortcut": "#d62728",
        "no_divergence": "#7f7f7f",
    }

    for ax, model_name in zip(axes, model_list):
        counts = results[model_name].get("classification_counts", {})
        if not counts:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            ax.set_title(MODEL_LABELS.get(model_name, model_name))
            continue

        labels = list(counts.keys())
        values = list(counts.values())
        colors = [cls_colors.get(l, "gray") for l in labels]

        # Clean labels for display
        display_labels = [l.replace("_", " ").title() for l in labels]

        ax.pie(values, labels=display_labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 8})
        ax.set_title(MODEL_LABELS.get(model_name, model_name), fontsize=10)

    fig.suptitle("Experiment 3: Path Divergence Classification", fontsize=14, y=1.02)
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "exp3_divergence_classification.pdf")
    fig.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def plot_divergence_rates(data: dict):
    """Bar chart comparing divergence rates across models."""
    results = data.get("results", {})
    model_list = [m for m in MODELS if m in results]

    if not model_list:
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    rates = [results[m].get("divergence_rate", 0) * 100 for m in model_list]
    labels = [MODEL_LABELS.get(m, m) for m in model_list]
    colors = [MODEL_COLORS.get(m, "gray") for m in model_list]

    ax.bar(range(len(model_list)), rates, color=colors)
    ax.set_xticks(range(len(model_list)))
    ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel("Divergence Rate (%)")
    ax.set_title("Exp 3: % of Correct-at-Level-0 Problems That Fail Subproblem Levels")
    ax.grid(True, alpha=0.3, axis='y')

    for i, v in enumerate(rates):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha='center', fontsize=9)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "exp3_divergence_rates.pdf")
    fig.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate experiment visualizations")
    parser.add_argument("--exp", choices=["1", "2", "3", "all"], default="all")
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)

    if args.exp in ("1", "all"):
        print("\n=== Experiment 1 Visualizations ===")
        data = load_exp_results("exp1_subproblem_passk")
        if data:
            plot_passk_curves_by_level(data)
            plot_passk_curves_by_model(data)
            plot_crossover_analysis(data)

    if args.exp in ("2", "all"):
        print("\n=== Experiment 2 Visualizations ===")
        data = load_exp_results("exp2_failure_localization")
        if data:
            plot_success_rate_heatmap(data)
            plot_transition_gains(data)
            plot_first_success_distribution(data)

    if args.exp in ("3", "all"):
        print("\n=== Experiment 3 Visualizations ===")
        data = load_exp_results("exp3_path_divergence")
        if data:
            plot_divergence_classification(data)
            plot_divergence_rates(data)

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
