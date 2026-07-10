#!/usr/bin/env python3
"""Generate results summary figure for the CPS debugging agent PoC."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent
RESULTS_FILE = OUT_DIR / "evaluation_results.json"


def load_results():
    with open(RESULTS_FILE) as f:
        return json.load(f)


def main():
    results = load_results()
    n = len(results)

    # Aggregate metrics
    metrics = {
        "diagnosis": sum(r["scores"]["diagnosis_correct"] for r in results),
        "semantic_fix": sum(r["scores"].get("semantic_fix", False) for r in results),
        "exact_fix": sum(r["scores"]["exact_fix"] for r in results),
        "partial_fix": sum(r["scores"]["partial_fix"] for r in results),
    }

    # Per-difficulty
    by_diff = {}
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in results if r["difficulty"] == diff]
        if subset:
            by_diff[diff] = {
                "n": len(subset),
                "diagnosis": sum(r["scores"]["diagnosis_correct"] for r in subset),
                "semantic_fix": sum(r["scores"].get("semantic_fix", False) for r in subset),
                "partial_fix": sum(r["scores"]["partial_fix"] for r in subset),
            }

    model = results[0].get("model", "unknown") if results else "unknown"
    total_tokens = sum(r.get("tokens", {}).get("total", 0) for r in results)
    avg_turns = np.mean([r["turns"] for r in results])

    # ── Figure ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: overall metrics
    ax = axes[0]
    labels = ["Diagnosis\nCorrect", "Semantic\nFix", "Exact\nFix", "Partial\nFix"]
    values = [metrics["diagnosis"], metrics["semantic_fix"],
              metrics["exact_fix"], metrics["partial_fix"]]
    pcts = [100 * v / n for v in values]
    colors = ["#378ADD", "#1D9E75", "#2E8B57", "#90C695"]
    bars = ax.bar(labels, pcts, color=colors, edgecolor="white", width=0.6)
    for bar, v, p in zip(bars, values, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{v}/{n}\n({p:.0f}%)", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"CPS Debugging Agent — Zero-shot Baseline\n"
                 f"Model: {model} | {n} bugs | avg {avg_turns:.1f} turns | "
                 f"{total_tokens:,} tokens")
    ax.axhline(y=10, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(3.5, 12, "Task 5 proxy (10%)", fontsize=8, color="gray", ha="right")

    # Right: by difficulty
    ax = axes[1]
    diff_labels = []
    diff_diag = []
    diff_fix = []
    for diff in ["easy", "medium", "hard"]:
        if diff in by_diff:
            d = by_diff[diff]
            diff_labels.append(f"{diff.capitalize()}\n(n={d['n']})")
            diff_diag.append(100 * d["diagnosis"] / d["n"])
            diff_fix.append(100 * d["semantic_fix"] / d["n"])

    x = np.arange(len(diff_labels))
    w = 0.35
    bars1 = ax.bar(x - w/2, diff_diag, w, label="Diagnosis", color="#378ADD")
    bars2 = ax.bar(x + w/2, diff_fix, w, label="Semantic Fix", color="#1D9E75")
    ax.set_xticks(x)
    ax.set_xticklabels(diff_labels)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Performance by Difficulty")
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 2,
                        f"{h:.0f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig_path = OUT_DIR / "fig_baseline_results.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {fig_path}")

    # ── Per-bug table ────────────────────────────────────────────────
    print(f"\n{'Bug ID':<35} {'Diff':<8} {'Diag':>5} {'S.Fix':>6} {'Turns':>6} {'Time':>6}")
    print("-" * 70)
    for r in results:
        s = r["scores"]
        print(f"{r['bug_id']:<35} {r['difficulty']:<8} "
              f"{'Y' if s['diagnosis_correct'] else 'N':>5} "
              f"{'Y' if s.get('semantic_fix', False) else 'N':>6} "
              f"{r['turns']:>6} "
              f"{r.get('elapsed_sec', 0):>5.0f}s")

    print(f"\nOverall: diagnosis={metrics['diagnosis']}/{n}, "
          f"semantic_fix={metrics['semantic_fix']}/{n}, "
          f"exact_fix={metrics['exact_fix']}/{n}")


if __name__ == "__main__":
    main()
