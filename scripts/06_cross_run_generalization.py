"""
06_cross_run_generalization.py — Cross-run leave-one-out evaluation.

Tests whether the physical→trace mapping generalizes across runs.
Trains on all runs except one, evaluates on the held-out run.
Repeats for each run as test set.

This is a stronger test than within-run temporal CV:
  - Different time of day, different initial conditions
  - Tests true generalization, not just temporal extrapolation

Outputs:
  fig_06a_cross_run_micro_f1.png
  fig_06b_cross_run_r2.png
  cross_run_results.json
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score, r2_score

OUT_DIR = Path(
    "/home/simran/allspark-data-exploration/CPS-Debugger/outputs/experiments"
)

plt.rcParams.update(
    {
        "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 13,
        "legend.fontsize": 10, "figure.dpi": 150, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False,
    }
)

PHYSICAL_COLS = [
    "dl_current_x_mean", "dl_current_x_std", "dl_current_x_delta",
    "dl_angle_mean", "dl_angle_std", "dl_angle_delta",
    "dl_velocity_mean", "dl_ang_vel_mean", "dl_ang_vel_std",
]
SYSLOG_COLS = ["sl_cpu", "sl_mem", "sl_load1", "sl_load5", "sl_load15"]
ALL_FEATURE_COLS = PHYSICAL_COLS + SYSLOG_COLS


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet = OUT_DIR / "aligned_dataset.parquet"
    vocab_path = OUT_DIR / "vocab_info.json"

    if not parquet.exists():
        raise FileNotFoundError(f"Run 00_preprocess.py first.")

    print("Loading aligned dataset...")
    df = pd.read_parquet(parquet)
    with open(vocab_path) as f:
        vocab_info = json.load(f)
    vocab = vocab_info["vocab"]

    branch_matrix = df[vocab].values.astype(float)
    valid = branch_matrix.sum(axis=1) > 0
    df = df[valid].reset_index(drop=True)
    branch_matrix = branch_matrix[valid]

    runs = df["run"].unique().tolist()
    print(f"Runs: {runs}")
    print(f"Dataset: {len(df):,} windows, {len(vocab)} branch targets")

    X = df[ALL_FEATURE_COLS].values.astype(float)
    X_phy = df[PHYSICAL_COLS].values.astype(float)
    Y = branch_matrix

    # Binary edge presence (for F1 comparison with existing code)
    Y_binary = (Y > 0).astype(int)

    loo_results = []
    for test_run in runs:
        train_mask = df["run"] != test_run
        test_mask = df["run"] == test_run

        X_train, X_test = X[train_mask], X[test_mask]
        X_phy_train, X_phy_test = X_phy[train_mask], X_phy[test_mask]
        Y_train, Y_test = Y[train_mask], Y[test_mask]
        Yb_train, Yb_test = Y_binary[train_mask], Y_binary[test_mask]

        print(f"\nHolding out: {test_run} ({test_mask.sum():,} windows)")
        print(f"  Training on: {train_mask.sum():,} windows from {len(runs)-1} runs")

        # RF regression (multi-output native, all 100 targets share tree structure)
        rf_reg = RandomForestRegressor(
            n_estimators=50, n_jobs=-1, random_state=42, max_depth=12, max_features="sqrt"
        )
        rf_reg.fit(X_train, Y_train)
        Y_pred = rf_reg.predict(X_test)
        r2_per_branch = [r2_score(Y_test[:, j], Y_pred[:, j]) for j in range(Y.shape[1])]
        mean_r2 = float(np.mean(r2_per_branch))
        frac_r2_0_3 = float((np.array(r2_per_branch) >= 0.3).mean())

        # Binary classification from thresholded regression predictions
        Yb_pred = (Y_pred > 0.5).astype(int)
        micro_f1 = float(f1_score(Yb_test.ravel(), Yb_pred.ravel(), zero_division=0))
        macro_f1 = float(f1_score(Yb_test, Yb_pred, average="macro", zero_division=0))

        # Physical only regression
        rf_phy = RandomForestRegressor(
            n_estimators=50, n_jobs=-1, random_state=42, max_depth=12, max_features="sqrt"
        )
        rf_phy.fit(X_phy_train, Y_train)
        Y_pred_phy = rf_phy.predict(X_phy_test)
        r2_phy = float(np.mean([
            r2_score(Y_test[:, j], Y_pred_phy[:, j]) for j in range(Y.shape[1])
        ]))

        result = {
            "test_run": test_run,
            "n_test": int(test_mask.sum()),
            "n_train": int(train_mask.sum()),
            "regression_all_features": {
                "mean_r2": mean_r2,
                "frac_r2_above_0.3": frac_r2_0_3,
            },
            "regression_physical_only": {
                "mean_r2": r2_phy,
            },
            "classification": {
                "micro_f1": micro_f1,
                "macro_f1": macro_f1,
            },
        }
        loo_results.append(result)
        print(f"  All features R²={mean_r2:.3f}, Physical R²={r2_phy:.3f}")
        print(f"  Micro-F1={micro_f1:.3f}, Macro-F1={macro_f1:.3f}")

    # ── Figure 6a: Micro-F1 across held-out runs ──────────────────────────────
    run_labels = [r["test_run"].replace("2025-", "").replace("_", "\n") for r in loo_results]
    micro_f1s = [r["classification"]["micro_f1"] for r in loo_results]
    macro_f1s = [r["classification"]["macro_f1"] for r in loo_results]
    r2s = [r["regression_all_features"]["mean_r2"] for r in loo_results]
    r2_phys = [r["regression_physical_only"]["mean_r2"] for r in loo_results]

    x = np.arange(len(runs))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.bar(x - w/2, micro_f1s, w, label="Micro-F1", color="#2196F3", edgecolor="none")
    ax.bar(x + w/2, macro_f1s, w, label="Macro-F1", color="#4CAF50", edgecolor="none")
    ax.axhline(np.mean(micro_f1s), color="#1565C0", linestyle="--", linewidth=1.5,
               label=f"Mean Micro-F1={np.mean(micro_f1s):.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels, fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 Score")
    ax.set_title("Cross-Run Generalization\n(Binary Branch Presence)")
    ax.legend(fontsize=9)
    for i, (mf, maf) in enumerate(zip(micro_f1s, macro_f1s)):
        ax.text(i - w/2, mf + 0.02, f"{mf:.2f}", ha="center", fontsize=7)
        ax.text(i + w/2, maf + 0.02, f"{maf:.2f}", ha="center", fontsize=7)

    ax = axes[1]
    ax.bar(x - w/2, r2s, w, label="All features", color="#9C27B0", edgecolor="none")
    ax.bar(x + w/2, r2_phys, w, label="Physical only", color="#E91E63", edgecolor="none")
    ax.axhline(np.mean(r2s), color="#6A1B9A", linestyle="--", linewidth=1.5,
               label=f"Mean R²={np.mean(r2s):.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels, fontsize=8)
    ax.set_ylabel("Mean R² (branch count regression)")
    ax.set_title("Cross-Run Generalization\n(Branch Count Regression)")
    ax.legend(fontsize=9)
    ax.set_ylim(min(-0.1, min(r2s) - 0.1), 1.1)
    for i, (r2, r2p) in enumerate(zip(r2s, r2_phys)):
        ax.text(i - w/2, r2 + 0.02, f"{r2:.2f}", ha="center", fontsize=7)
        ax.text(i + w/2, r2p + 0.02, f"{r2p:.2f}", ha="center", fontsize=7)

    plt.suptitle("Leave-One-Run-Out Evaluation", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_06_cross_run_generalization.png")
    plt.close()

    # Save results
    summary = {
        "mean_micro_f1": float(np.mean(micro_f1s)),
        "std_micro_f1": float(np.std(micro_f1s)),
        "mean_r2_all": float(np.mean(r2s)),
        "std_r2_all": float(np.std(r2s)),
        "mean_r2_physical": float(np.mean(r2_phys)),
        "per_run": loo_results,
    }
    (OUT_DIR / "cross_run_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\n── Summary ──")
    print(f"Mean Micro-F1: {summary['mean_micro_f1']:.3f} ± {summary['std_micro_f1']:.3f}")
    print(f"Mean R² (all): {summary['mean_r2_all']:.3f} ± {summary['std_r2_all']:.3f}")
    print(f"Mean R² (phy): {summary['mean_r2_physical']:.3f}")
    print(f"\nOutputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
