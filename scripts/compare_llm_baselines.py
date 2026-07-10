"""
compare_llm_baselines.py — Compare LLM zero-shot baselines (Haiku vs Sonnet vs GRPO).

Reads all available zero-shot result files and generates a comparison table + figure.

Usage:
  python scripts/compare_llm_baselines.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT   = Path("/home/simran/allspark-data-exploration/CPS-Debugger")
P3_OUT = ROOT / "outputs/phase3"

MODEL_FILES = {
    "Haiku (baseline)": P3_OUT / "anomaly_zero_shot_haiku_results.json",
    "Sonnet (zero-shot)": P3_OUT / "anomaly_zero_shot_claude-sonnet-4-6_results.json",
    "GRPO-tuned": P3_OUT / "grpo_eval_results.json",  # if available
}

ANOMALY_TYPES = ["NORMAL", "TRACE_SWAP", "THRESHOLD_BUG", "TIMING_DELAY", "STUCK_SENSOR"]


def load_results(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"  Warning: could not load {path.name}: {e}")
        return None


def print_comparison_table(models_data: dict):
    print("\n" + "=" * 80)
    print("LLM ZERO-SHOT ANOMALY DETECTION — MODEL COMPARISON")
    print("=" * 80)

    metrics = ["detection_acc", "typing_acc", "localization_acc", "reasoning_score", "total_score"]
    metric_labels = {
        "detection_acc":    "Detection Acc",
        "typing_acc":       "Typing Acc",
        "localization_acc": "Localization",
        "reasoning_score":  "Reasoning",
        "total_score":      "Total Score",
    }

    # Header
    col_w = 20
    header = f"{'Metric':<20}" + "".join(f"{name:>{col_w}}" for name in models_data)
    print(header)
    print("-" * len(header))

    for m in metrics:
        row = f"{metric_labels[m]:<20}"
        for name, data in models_data.items():
            v = data["overall"].get(m, float("nan"))
            row += f"{v:>{col_w}.3f}"
        print(row)

    print("\nPer-type total score:")
    type_header = f"{'Anomaly Type':<20}" + "".join(f"{name:>{col_w}}" for name in models_data)
    print(type_header)
    print("-" * len(type_header))

    for atype in ANOMALY_TYPES:
        row = f"{atype:<20}"
        for name, data in models_data.items():
            per = data.get("per_type", {}).get(atype, {})
            v = per.get("mean_score", float("nan"))
            row += f"{v:>{col_w}.3f}"
        print(row)

    print("=" * 80)


def make_comparison_figure(models_data: dict, out_path: Path):
    metrics = ["detection_acc", "typing_acc", "localization_acc", "reasoning_score", "total_score"]
    metric_labels = ["Detection", "Typing", "Localization", "Reasoning", "Total"]

    model_names = list(models_data.keys())
    n_models    = len(model_names)
    n_metrics   = len(metrics)

    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"][:n_models]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("LLM Zero-shot Anomaly Detection: Model Comparison", fontsize=14, y=1.02)

    # ── Panel 1: Overall metrics comparison ──────────────────────────────────
    ax = axes[0]
    x = np.arange(n_metrics)
    bar_w = 0.8 / n_models
    for i, (name, data) in enumerate(models_data.items()):
        vals = [data["overall"].get(m, 0) for m in metrics]
        xpos = x + (i - n_models / 2 + 0.5) * bar_w
        bars = ax.bar(xpos, vals, bar_w, label=name, color=colors[i],
                      edgecolor="black", linewidth=0.5, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=15, ha="right")
    ax.set_ylabel("Score (0–1)")
    ax.set_title("Overall Metrics by Model")
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.legend(loc="upper right", fontsize=9)

    # ── Panel 2: Per-type total score ─────────────────────────────────────────
    ax2 = axes[1]
    x2 = np.arange(len(ANOMALY_TYPES))
    for i, (name, data) in enumerate(models_data.items()):
        vals = [data.get("per_type", {}).get(t, {}).get("mean_score", 0)
                for t in ANOMALY_TYPES]
        xpos = x2 + (i - n_models / 2 + 0.5) * bar_w
        bars = ax2.bar(xpos, vals, bar_w, label=name, color=colors[i],
                       edgecolor="black", linewidth=0.5, alpha=0.85)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(ANOMALY_TYPES, rotation=20, ha="right")
    ax2.set_ylabel("Mean Total Score (0–1)")
    ax2.set_title("Per-Anomaly-Type Performance")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.name}")


def main():
    print("Loading LLM zero-shot comparison data...")

    models_data = {}
    for label, path in MODEL_FILES.items():
        data = load_results(path)
        if data:
            print(f"  ✓ {label}: {data['n_examples']} examples, "
                  f"model={data.get('model', '?')}")
            models_data[label] = data
        else:
            print(f"  ✗ {label}: not found ({path.name})")

    if not models_data:
        print("No results found. Run zero-shot evaluations first.")
        return

    print_comparison_table(models_data)

    out_fig = P3_OUT / "fig_llm_comparison.png"
    make_comparison_figure(models_data, out_fig)

    # Save comparison JSON
    comparison = {
        m: {
            "model": d.get("model"),
            "n_examples": d.get("n_examples"),
            "overall": d.get("overall"),
            "per_type": d.get("per_type"),
        }
        for m, d in models_data.items()
    }
    (P3_OUT / "llm_comparison.json").write_text(json.dumps(comparison, indent=2))
    print("Saved llm_comparison.json")


if __name__ == "__main__":
    main()
