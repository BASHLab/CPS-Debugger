"""
01_branch_count_regression.py — Core novelty experiment.

Predicts branch EXECUTION COUNTS (from cf_table) from physical + syslog features.
This goes beyond binary presence/absence (existing code) to continuous regression.

Key question: which specific branch decisions (src_pc → dst_pc within each function)
are predictable from physical sensor observations?

Outputs:
  outputs/experiments/fig_01a_branch_r2_by_branch.png
  outputs/experiments/fig_01b_branch_prediction_scatter.png
  outputs/experiments/fig_01c_feature_importance.png
  outputs/experiments/fig_01d_r2_by_state.png
  outputs/experiments/branch_regression_results.json
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

OUT_DIR = Path(
    "/home/simran/allspark-data-exploration/CPS-Debugger/outputs/experiments"
)

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

PHYSICAL_COLS = [
    "dl_current_x_mean",
    "dl_current_x_std",
    "dl_current_x_delta",
    "dl_angle_mean",
    "dl_angle_std",
    "dl_angle_delta",
    "dl_velocity_mean",
    "dl_ang_vel_mean",
    "dl_ang_vel_std",
]
SYSLOG_COLS = ["sl_cpu", "sl_mem", "sl_load1", "sl_load5", "sl_load15"]
ALL_FEATURE_COLS = PHYSICAL_COLS + SYSLOG_COLS


def temporal_block_cv(n: int, n_folds: int = 5):
    """Generate (train_idx, test_idx) for n_folds contiguous time-block CV."""
    indices = np.arange(n)
    fold_size = n // n_folds
    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])
        yield train_idx, test_idx


def run_regression(X: np.ndarray, Y: np.ndarray, n_folds: int = 5) -> dict:
    """Run temporal-block CV and return per-branch R²."""
    n = len(X)
    all_preds = np.zeros_like(Y, dtype=float)
    all_truths = Y.copy().astype(float)

    for train_idx, test_idx in temporal_block_cv(n, n_folds):
        rf = RandomForestRegressor(
            n_estimators=100, n_jobs=-1, random_state=42, max_depth=12
        )
        rf.fit(X[train_idx], Y[train_idx])
        all_preds[test_idx] = rf.predict(test_idx.reshape(-1, 1) * 0 + X[test_idx])

    # Per-branch R²
    r2_per_branch = []
    for j in range(Y.shape[1]):
        r2 = r2_score(all_truths[:, j], all_preds[:, j])
        r2_per_branch.append(r2)

    # Overall cosine similarity per window
    dot = (all_truths * all_preds).sum(axis=1)
    norm_t = np.linalg.norm(all_truths, axis=1) + 1e-9
    norm_p = np.linalg.norm(all_preds, axis=1) + 1e-9
    cosine_sim = dot / (norm_t * norm_p)

    return {
        "r2_per_branch": r2_per_branch,
        "preds": all_preds,
        "truths": all_truths,
        "cosine_sim": cosine_sim,
    }


def run_regression_full(X: np.ndarray, Y: np.ndarray, n_folds: int = 5) -> dict:
    """Run temporal-block CV with MultiOutputRegressor for feature importance."""
    n = len(X)
    all_preds = np.zeros_like(Y, dtype=float)
    all_truths = Y.copy().astype(float)
    importances_list = []

    for train_idx, test_idx in temporal_block_cv(n, n_folds):
        rf = RandomForestRegressor(
            n_estimators=100, n_jobs=-1, random_state=42, max_depth=12
        )
        rf.fit(X[train_idx], Y[train_idx])
        all_preds[test_idx] = rf.predict(X[test_idx])
        importances_list.append(rf.feature_importances_)

    mean_importances = np.mean(importances_list, axis=0)

    r2_per_branch = []
    for j in range(Y.shape[1]):
        r2 = r2_score(all_truths[:, j], all_preds[:, j])
        r2_per_branch.append(r2)

    dot = (all_truths * all_preds).sum(axis=1)
    norm_t = np.linalg.norm(all_truths, axis=1) + 1e-9
    norm_p = np.linalg.norm(all_preds, axis=1) + 1e-9
    cosine_sim = dot / (norm_t * norm_p)

    return {
        "r2_per_branch": r2_per_branch,
        "preds": all_preds,
        "truths": all_truths,
        "cosine_sim": cosine_sim,
        "feature_importances": mean_importances,
    }


def plot_r2_by_branch(r2_vals, labels, title, fname, top_k=50):
    """Sorted bar chart of R² per branch."""
    idx = np.argsort(r2_vals)[::-1][:top_k]
    sorted_r2 = np.array(r2_vals)[idx]
    sorted_labels = [labels[i] for i in idx]

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#2196F3" if v >= 0.5 else "#90CAF9" if v >= 0.3 else "#E0E0E0"
              for v in sorted_r2]
    bars = ax.bar(range(len(sorted_r2)), sorted_r2, color=colors, edgecolor="none")
    ax.axhline(0.5, color="#F44336", linestyle="--", linewidth=1, label="R²=0.5")
    ax.axhline(0.3, color="#FF9800", linestyle="--", linewidth=1, label="R²=0.3")
    ax.set_xticks(range(len(sorted_r2)))
    ax.set_xticklabels(sorted_labels, rotation=90, fontsize=7)
    ax.set_ylabel("R²")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(min(-0.05, sorted_r2.min() - 0.05), 1.05)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    return idx, sorted_r2


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet = OUT_DIR / "aligned_dataset.parquet"
    vocab_path = OUT_DIR / "vocab_info.json"

    if not parquet.exists():
        raise FileNotFoundError(
            f"Run 00_preprocess.py first. Expected: {parquet}"
        )

    print("Loading aligned dataset...")
    df = pd.read_parquet(parquet)
    with open(vocab_path) as f:
        vocab_info = json.load(f)

    vocab = vocab_info["vocab"]
    labels_map = vocab_info["labels"]
    labels = [labels_map.get(b, b) for b in vocab]

    print(f"Dataset: {len(df):,} windows, {len(vocab)} branch targets")
    print(f"State distribution: {df['dl_state'].value_counts().to_dict()}")

    # Drop windows with all-zero branch vectors (no trace data)
    branch_matrix = df[vocab].values.astype(float)
    valid = branch_matrix.sum(axis=1) > 0
    df = df[valid].reset_index(drop=True)
    branch_matrix = branch_matrix[valid]
    print(f"After filtering zero-trace windows: {len(df):,}")

    X_all = df[ALL_FEATURE_COLS].values.astype(float)
    X_phy = df[PHYSICAL_COLS].values.astype(float)
    X_sys = df[SYSLOG_COLS].values.astype(float)
    Y = branch_matrix

    # ── Experiment 1: ALL features → branch counts ────────────────────────────
    print("\nRunning regression: ALL features → branch counts...")
    res_all = run_regression_full(X_all, Y, n_folds=5)

    # ── Experiment 2: Physical only ───────────────────────────────────────────
    print("Running regression: Physical only → branch counts...")
    res_phy = run_regression_full(X_phy, Y, n_folds=5)

    # ── Experiment 3: Syslog only ─────────────────────────────────────────────
    print("Running regression: Syslog only → branch counts...")
    res_sys = run_regression_full(X_sys, Y, n_folds=5)

    r2_all = res_all["r2_per_branch"]
    r2_phy = res_phy["r2_per_branch"]
    r2_sys = res_sys["r2_per_branch"]

    # ── Figure 1a: R² sorted bar chart ───────────────────────────────────────
    idx_sorted, sorted_r2 = plot_r2_by_branch(
        r2_all,
        labels,
        "Branch Count Prediction (R²) — All Features",
        OUT_DIR / "fig_01a_branch_r2_all_features.png",
    )
    plot_r2_by_branch(
        r2_phy,
        labels,
        "Branch Count Prediction (R²) — Physical Sensors Only",
        OUT_DIR / "fig_01a_branch_r2_physical_only.png",
    )

    # ── Figure 1b: Scatter plots for top-6 most predictable branches ─────────
    top6_idx = np.argsort(r2_all)[::-1][:6]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, j in zip(axes.flat, top6_idx):
        truth = res_all["truths"][:, j]
        pred = res_all["preds"][:, j]
        r2 = r2_all[j]
        ax.scatter(truth, pred, alpha=0.3, s=5, rasterized=True)
        lim = max(truth.max(), pred.max()) * 1.05
        ax.plot([0, lim], [0, lim], "r--", linewidth=1)
        ax.set_xlabel("True count")
        ax.set_ylabel("Predicted count")
        short_label = labels[j].split(":")[0][:20]
        ax.set_title(f"{short_label}\nR²={r2:.3f}", fontsize=10)
    fig.suptitle("Predicted vs Actual Branch Counts (Top-6 Branches)", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_01b_branch_prediction_scatter.png")
    plt.close()

    # ── Figure 1c: Feature importance ────────────────────────────────────────
    importances = res_all["feature_importances"]
    feat_idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(importances)), importances[feat_idx], color="#2196F3", edgecolor="none")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(
        [ALL_FEATURE_COLS[i] for i in feat_idx], rotation=45, ha="right", fontsize=9
    )
    ax.set_ylabel("Mean Feature Importance")
    ax.set_title("Feature Importance for Branch Count Prediction")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_01c_feature_importance.png")
    plt.close()

    # ── Figure 1d: R² by state ────────────────────────────────────────────────
    state_names = {0: "SWINGUP", 1: "BALANCE", 2: "RESET"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, (state_id, state_name) in zip(axes, state_names.items()):
        mask = df["dl_state"].values == state_id
        if mask.sum() < 50:
            ax.set_title(f"{state_name}\n(insufficient data)")
            continue
        X_s = X_all[mask]
        Y_s = Y[mask]
        res_s = run_regression_full(X_s, Y_s, n_folds=min(5, mask.sum() // 100))
        r2_s = res_s["r2_per_branch"]
        ax.bar(range(len(r2_s)), sorted(r2_s, reverse=True),
               color="#4CAF50", edgecolor="none")
        ax.axhline(0.5, color="red", linestyle="--", linewidth=1)
        ax.axhline(0.3, color="orange", linestyle="--", linewidth=1)
        ax.set_title(f"{state_name} (n={mask.sum():,})")
        ax.set_xlabel("Branch rank")
        ax.set_ylabel("R²")
        med = np.median(r2_s)
        frac_05 = (np.array(r2_s) >= 0.5).mean()
        ax.text(0.98, 0.98, f"median={med:.2f}\n>0.5: {frac_05:.0%}",
                transform=ax.transAxes, va="top", ha="right", fontsize=9)
    fig.suptitle("Branch Count Prediction R² by Controller State", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_01d_r2_by_state.png")
    plt.close()

    # ── Figure 1e: Physical vs Syslog R² scatter ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(r2_phy, r2_sys, c=r2_all, cmap="viridis",
                    alpha=0.8, s=50, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="R² (all features)")
    ax.axhline(0.3, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0.3, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("R² — Physical sensors only")
    ax.set_ylabel("R² — Syslog only")
    ax.set_title("Branch Predictability: Physical vs System-Health Sensors")
    # Annotate quadrants
    ax.text(0.75, 0.05, "Physical only\nUseful", transform=ax.transAxes,
            ha="center", fontsize=9, color="navy")
    ax.text(0.05, 0.75, "Syslog only\nUseful", transform=ax.transAxes,
            ha="center", fontsize=9, color="navy")
    ax.text(0.75, 0.75, "Both\nUseful", transform=ax.transAxes,
            ha="center", fontsize=9, color="darkgreen")
    ax.text(0.05, 0.05, "Neither\nUseful", transform=ax.transAxes,
            ha="center", fontsize=9, color="gray")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_01e_physical_vs_syslog_r2.png")
    plt.close()

    # ── Summary stats ─────────────────────────────────────────────────────────
    r2_all_arr = np.array(r2_all)
    r2_phy_arr = np.array(r2_phy)
    r2_sys_arr = np.array(r2_sys)

    physical_unique = (r2_phy_arr >= 0.3) & (r2_sys_arr < 0.1)
    syslog_unique = (r2_sys_arr >= 0.3) & (r2_phy_arr < 0.1)
    both_useful = (r2_phy_arr >= 0.3) & (r2_sys_arr >= 0.3)
    neither = (r2_all_arr < 0.1)

    results = {
        "n_windows": int(len(df)),
        "n_branches": int(len(vocab)),
        "all_features": {
            "mean_r2": float(r2_all_arr.mean()),
            "median_r2": float(np.median(r2_all_arr)),
            "frac_r2_above_0.3": float((r2_all_arr >= 0.3).mean()),
            "frac_r2_above_0.5": float((r2_all_arr >= 0.5).mean()),
            "frac_r2_above_0.7": float((r2_all_arr >= 0.7).mean()),
            "mean_cosine_sim": float(res_all["cosine_sim"].mean()),
        },
        "physical_only": {
            "mean_r2": float(r2_phy_arr.mean()),
            "frac_r2_above_0.3": float((r2_phy_arr >= 0.3).mean()),
        },
        "syslog_only": {
            "mean_r2": float(r2_sys_arr.mean()),
            "frac_r2_above_0.3": float((r2_sys_arr >= 0.3).mean()),
        },
        "information_decomposition": {
            "n_physical_unique": int(physical_unique.sum()),
            "n_syslog_unique": int(syslog_unique.sum()),
            "n_both_useful": int(both_useful.sum()),
            "n_neither": int(neither.sum()),
        },
        "top_10_branches_by_r2": [
            {
                "label": labels[i],
                "r2_all": float(r2_all[i]),
                "r2_physical": float(r2_phy[i]),
                "r2_syslog": float(r2_sys[i]),
            }
            for i in np.argsort(r2_all)[::-1][:10]
        ],
    }
    (OUT_DIR / "branch_regression_results.json").write_text(
        json.dumps(results, indent=2)
    )

    print("\n── Results ──")
    print(f"All features:   mean R²={results['all_features']['mean_r2']:.3f},  "
          f"R²>0.5: {results['all_features']['frac_r2_above_0.5']:.0%}")
    print(f"Physical only:  mean R²={results['physical_only']['mean_r2']:.3f},  "
          f"R²>0.3: {results['physical_only']['frac_r2_above_0.3']:.0%}")
    print(f"Syslog only:    mean R²={results['syslog_only']['mean_r2']:.3f},  "
          f"R²>0.3: {results['syslog_only']['frac_r2_above_0.3']:.0%}")
    print(f"\nInformation decomposition (out of {len(vocab)} branches):")
    print(f"  Physical-unique (phy≥0.3, sys<0.1): {physical_unique.sum()}")
    print(f"  Syslog-unique  (sys≥0.3, phy<0.1):  {syslog_unique.sum()}")
    print(f"  Both useful    (both≥0.3):            {both_useful.sum()}")
    print(f"  Neither (<0.1 all):                   {neither.sum()}")
    print("\nTop-5 most predictable branches:")
    for item in results["top_10_branches_by_r2"][:5]:
        print(f"  {item['label'][:50]:<50} R²={item['r2_all']:.3f} "
              f"(phy={item['r2_physical']:.2f}, sys={item['r2_syslog']:.2f})")
    print(f"\nOutputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
