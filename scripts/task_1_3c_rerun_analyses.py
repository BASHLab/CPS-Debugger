"""
task_1_3c_rerun_analyses.py — Re-run key analyses on the expanded dataset.

Compares within-BALANCE R² and cross-run generalization results
between the original 6-run dataset and the expanded N-run dataset.

Output:
  outputs/phase3/expanded_results_summary.json
  outputs/phase3/fig_expanded_comparison.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

ROOT    = Path("/home/simran/allspark-data-exploration/CPS-Debugger")
P3_OUT  = ROOT / "outputs/phase3"
P3_OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150,
                     "savefig.dpi": 300})

PHYSICAL_COLS = [
    "dl_current_x_mean", "dl_current_x_std", "dl_current_x_delta",
    "dl_angle_mean", "dl_angle_std", "dl_angle_delta",
    "dl_velocity_mean", "dl_ang_vel_mean", "dl_ang_vel_std",
]


def temporal_cv_r2(X, y, n_folds=5):
    """5-fold contiguous block cross-validation, returns mean R²."""
    n = len(X)
    fold_size = n // n_folds
    r2s = []
    for k in range(n_folds):
        test_start = k * fold_size
        test_end   = (k + 1) * fold_size if k < n_folds - 1 else n
        idx_test   = np.arange(test_start, test_end)
        idx_train  = np.concatenate([np.arange(0, test_start),
                                     np.arange(test_end, n)])
        if len(idx_train) < 10 or len(idx_test) < 5:
            continue
        rf = RandomForestRegressor(n_estimators=100, max_depth=10,
                                   n_jobs=4, random_state=42)
        rf.fit(X[idx_train], y[idx_train])
        pred = rf.predict(X[idx_test])
        r2s.append(max(-1.0, r2_score(y[idx_test], pred)))
    return float(np.mean(r2s)) if r2s else 0.0


def within_balance_r2(df, vocab, n_branches=30):
    """Compute within-BALANCE R² for top n_branches."""
    bal = df[df["dl_state"] == 1].sort_values(["run", "win"]).reset_index(drop=True)
    if len(bal) < 100:
        return 0.0
    X = bal[[c for c in PHYSICAL_COLS if c in bal.columns]].values
    r2s = []
    for branch in vocab[:n_branches]:
        if branch not in bal.columns:
            continue
        y = bal[branch].values
        if y.std() < 1e-6:
            continue
        r2 = temporal_cv_r2(X, y)
        r2s.append(r2)
    return float(np.mean(r2s)) if r2s else 0.0


def cross_run_r2(df, vocab, n_branches=20):
    """Leave-one-run-out cross-validation with Ridge."""
    runs = df["run"].unique()
    if len(runs) < 2:
        return 0.0
    r2s = []
    for test_run in runs:
        train_df = df[df["run"] != test_run]
        test_df  = df[df["run"] == test_run]
        if len(test_df) < 20:
            continue
        X_train = train_df[[c for c in PHYSICAL_COLS if c in df.columns]].values
        X_test  = test_df[[c for c in PHYSICAL_COLS if c in df.columns]].values
        run_r2s = []
        for branch in vocab[:n_branches]:
            if branch not in df.columns:
                continue
            y_train = train_df[branch].values
            y_test  = test_df[branch].values
            if y_train.std() < 1e-6:
                continue
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train, y_train)
            pred = ridge.predict(X_test)
            run_r2s.append(max(-1.0, r2_score(y_test, pred)))
        if run_r2s:
            r2s.append(float(np.mean(run_r2s)))
    return float(np.mean(r2s)) if r2s else 0.0


def main():
    expanded_path = P3_OUT / "aligned_dataset_expanded.parquet"
    if not expanded_path.exists():
        print("ERROR: aligned_dataset_expanded.parquet not found.")
        print("Run task_1_3b_expand_preprocess.py first.")
        return

    df_exp = pd.read_parquet(expanded_path)
    df_6   = pd.read_parquet(ROOT / "outputs/experiments/aligned_dataset.parquet")
    vocab  = json.loads((ROOT / "outputs/experiments/vocab_info.json").read_text())["vocab"]

    print(f"6-run dataset:     {len(df_6):,} windows, {df_6['run'].nunique()} runs")
    print(f"Expanded dataset:  {len(df_exp):,} windows, {df_exp['run'].nunique()} runs")
    print(f"Vocabulary:        {len(vocab)} branches")

    N_BRANCHES = 30  # Use 30 branches for speed

    # ── Within-BALANCE R² ─────────────────────────────────────────────────────
    print("\nComputing within-BALANCE R² (6-run)...")
    r2_6 = within_balance_r2(df_6, vocab, n_branches=N_BRANCHES)
    print(f"  6-run:    {r2_6:.4f}")

    print("Computing within-BALANCE R² (expanded)...")
    r2_exp = within_balance_r2(df_exp, vocab, n_branches=N_BRANCHES)
    print(f"  expanded: {r2_exp:.4f}  (Δ = {r2_exp - r2_6:+.4f})")

    # ── Cross-run generalization ──────────────────────────────────────────────
    print("\nComputing cross-run R² (6-run, Ridge LOO)...")
    cr_6 = cross_run_r2(df_6, vocab, n_branches=N_BRANCHES)
    print(f"  6-run:    {cr_6:.4f}")

    print("Computing cross-run R² (expanded, Ridge LOO)...")
    # Subsample for speed if too many runs
    runs_exp = df_exp["run"].unique()
    if len(runs_exp) > 12:
        np.random.seed(42)
        sampled_runs = np.random.choice(runs_exp, 12, replace=False)
        df_exp_sub = df_exp[df_exp["run"].isin(sampled_runs)]
        print(f"  (subsampled to {len(sampled_runs)} runs for speed)")
    else:
        df_exp_sub = df_exp

    cr_exp = cross_run_r2(df_exp_sub, vocab, n_branches=N_BRANCHES)
    print(f"  expanded: {cr_exp:.4f}  (Δ = {cr_exp - cr_6:+.4f})")

    # ── Save results ──────────────────────────────────────────────────────────
    out = {
        "n_branches_evaluated": N_BRANCHES,
        "within_balance_r2": {
            "6_run":    r2_6,
            "expanded": r2_exp,
            "delta":    r2_exp - r2_6,
        },
        "cross_run_r2_ridge": {
            "6_run":    cr_6,
            "expanded": cr_exp,
            "delta":    cr_exp - cr_6,
        },
        "n_runs": {
            "6_run":    int(df_6["run"].nunique()),
            "expanded": int(df_exp["run"].nunique()),
        },
        "n_windows": {
            "6_run":    int(len(df_6)),
            "expanded": int(len(df_exp)),
        },
        "interpretation": (
            "Positive delta = more runs improve results (good for generalization claims). "
            "Near-zero delta = results are stable (good for robustness). "
            "Negative delta = overfitting to specific runs."
        ),
    }
    (P3_OUT / "expanded_results_summary.json").write_text(json.dumps(out, indent=2))
    print("\nSaved expanded_results_summary.json")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["6 runs", f"{df_exp['run'].nunique()} runs (expanded)"]

    ax = axes[0]
    vals = [r2_6, r2_exp]
    bars = ax.bar(labels, vals, color=["#2196F3", "#4CAF50"], edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Within-BALANCE R² (mean, top-30 branches)")
    ax.set_title("Does more data improve within-state R²?")
    ax.set_ylim(0, 1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.3f}",
                ha="center", fontsize=12, fontweight="bold")

    ax2 = axes[1]
    vals2 = [cr_6, cr_exp]
    bars2 = ax2.bar(labels, vals2, color=["#2196F3", "#4CAF50"], edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Cross-run R² (Ridge LOO, mean, top-30 branches)")
    ax2.set_title("Does more data improve cross-run generalization?")
    ax2.set_ylim(0, 1)
    for b, v in zip(bars2, vals2):
        ax2.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.3f}",
                 ha="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(P3_OUT / "fig_expanded_comparison.png")
    plt.close(fig)
    print("Saved fig_expanded_comparison.png")
    print("Done.")


if __name__ == "__main__":
    main()
