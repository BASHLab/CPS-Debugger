"""
parallel_B4_temporal_stability.py — Does the model generalize across time?

Trains RF on first 3 runs chronologically, tests on last 3.
Also tracks per-run consistency score distribution over time.

Outputs:
  outputs/parallel/fig_temporal_stability.png
  outputs/parallel/temporal_stability.json
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

ROOT = Path("/home/simran/allspark-data-exploration/CPS-Debugger")
EXP  = ROOT / "outputs/experiments"
OUT  = ROOT / "outputs/parallel"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300,
})

PHYSICAL_COLS = [
    "dl_current_x_mean", "dl_current_x_std", "dl_current_x_delta",
    "dl_angle_mean", "dl_angle_std", "dl_angle_delta",
    "dl_velocity_mean", "dl_ang_vel_mean", "dl_ang_vel_std",
]


def loro_r2(df, vocab, runs):
    """Leave-one-run-out R² using Ridge regression."""
    r2s = {}
    for test_run in runs:
        train_df = df[df["run"] != test_run]
        test_df  = df[df["run"] == test_run]
        if len(test_df) < 20:
            continue
        X_tr = train_df[PHYSICAL_COLS].values
        Y_tr = train_df[vocab].values
        X_te = test_df[PHYSICAL_COLS].values
        Y_te = test_df[vocab].values
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr, Y_tr)
        r2s[test_run] = float(r2_score(Y_te, ridge.predict(X_te), multioutput="uniform_average"))
    return r2s


def main():
    df = pd.read_parquet(EXP / "aligned_dataset.parquet")
    vi = json.loads((EXP / "vocab_info.json").read_text())
    vocab = vi["vocab"]

    # Runs sorted chronologically by run_id (which is a timestamp)
    runs = sorted(df["run"].unique())
    print(f"Runs in chronological order:")
    for r in runs:
        n = (df["run"] == r).sum()
        print(f"  {r}: {n:,} windows")

    if len(runs) < 4:
        print("Not enough runs for temporal stability analysis (need ≥4). Using 80/20 split instead.")
        n_train = max(1, int(len(runs) * 0.6))
        train_runs = runs[:n_train]
        test_runs  = runs[n_train:]
    else:
        mid = len(runs) // 2
        train_runs = runs[:mid]
        test_runs  = runs[mid:]

    print(f"\nTrain runs (early): {train_runs}")
    print(f"Test runs  (late):  {test_runs}")

    train_df = df[df["run"].isin(train_runs)]
    test_df  = df[df["run"].isin(test_runs)]

    # ── Temporal train→test R² ─────────────────────────────────────────────────
    print("\nTraining RF on early runs, testing on late runs...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=8, random_state=42)
    rf.fit(train_df[PHYSICAL_COLS].values, train_df[vocab].values)

    Y_pred_late = rf.predict(test_df[PHYSICAL_COLS].values)
    r2_temporal = float(r2_score(test_df[vocab].values, Y_pred_late, multioutput="uniform_average"))
    print(f"Temporal R² (train early → test late): {r2_temporal:.4f}")

    # ── LORO R² for comparison ────────────────────────────────────────────────
    print("\nComputing LORO R² for comparison...")
    loro_scores = loro_r2(df, vocab, runs)
    r2_loro_mean = float(np.mean(list(loro_scores.values())))
    print(f"LORO mean R²: {r2_loro_mean:.4f}")

    # ── Per-run consistency score distribution ─────────────────────────────────
    print("\nComputing per-run consistency scores (trained on early, tested across all)...")
    per_run_stats = {}
    for run in runs:
        rdf = df[df["run"] == run]
        Y_pred_run = rf.predict(rdf[PHYSICAL_COLS].values)
        scores_run = np.linalg.norm(rdf[vocab].values - Y_pred_run, axis=1)
        per_run_stats[run] = {
            "mean":     float(scores_run.mean()),
            "std":      float(scores_run.std()),
            "p95":      float(np.percentile(scores_run, 95)),
            "is_train": run in train_runs,
            "n_windows": int(len(rdf)),
        }

    # ── Check for trend ───────────────────────────────────────────────────────
    run_indices = list(range(len(runs)))
    means = [per_run_stats[r]["mean"] for r in runs]
    if len(runs) > 2:
        from numpy.polynomial.polynomial import polyfit
        coef = polyfit(run_indices, means, 1)  # linear fit
        drift_per_run = float(coef[1])
    else:
        drift_per_run = 0.0

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Panel 1: Per-run LORO R²
    ax1 = axes[0, 0]
    run_labels = [r.split("_")[0] + "\n" + "_".join(r.split("_")[1:]) for r in runs]
    loro_r2s   = [loro_scores.get(r, 0) for r in runs]
    bar_colors = ["#1565C0" if r in train_runs else "#B71C1C" for r in runs]
    bars = ax1.bar(run_labels, loro_r2s, color=bar_colors, edgecolor="black", lw=0.5)
    ax1.axhline(r2_loro_mean, ls="--", color="#6A1B9A", lw=1.2,
                label=f"LORO mean={r2_loro_mean:.3f}")
    ax1.set_ylabel("LORO R² (Ridge)")
    ax1.set_title("Per-Run Generalization (LORO)", fontsize=10, fontweight="bold")
    ax1.set_xticklabels(run_labels, fontsize=7, rotation=30, ha="right")
    ax1.legend(fontsize=8)
    # Add legend for train/test color
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(facecolor="#1565C0", label="Train (early)"),
                         Patch(facecolor="#B71C1C", label="Test (late)"),
                         plt.Line2D([0], [0], ls="--", color="#6A1B9A", label=f"Mean={r2_loro_mean:.3f}")],
               fontsize=7)
    for b, v in zip(bars, loro_r2s):
        ax1.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)

    # Panel 2: Temporal train→test vs LORO
    ax2 = axes[0, 1]
    comp_labels = [f"LORO\nmean", f"Temporal\n(early→late)"]
    comp_vals   = [r2_loro_mean, r2_temporal]
    comp_colors = ["#1565C0", "#B71C1C" if r2_temporal < r2_loro_mean * 0.85 else "#4CAF50"]
    bars2 = ax2.bar(comp_labels, comp_vals, color=comp_colors, edgecolor="black", lw=0.5, width=0.4)
    ax2.set_ylabel("R² (Ridge regression)")
    ax2.set_title("Temporal Stability: LORO vs Train-Early→Test-Late", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    for b, v in zip(bars2, comp_vals):
        ax2.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    drift_pct = (r2_loro_mean - r2_temporal) / max(r2_loro_mean, 1e-6) * 100
    ax2.text(0.5, 0.1, f"Degradation: {drift_pct:.1f}%", transform=ax2.transAxes,
             ha="center", fontsize=10, color="#B71C1C" if drift_pct > 15 else "#2E8B57")

    # Panel 3: Per-run consistency score mean ± std
    ax3 = axes[1, 0]
    means_arr = np.array([per_run_stats[r]["mean"] for r in runs])
    stds_arr  = np.array([per_run_stats[r]["std"]  for r in runs])
    colors3   = ["#1565C0" if r in train_runs else "#B71C1C" for r in runs]
    ax3.bar(run_labels, means_arr, color=colors3, edgecolor="black", lw=0.5, alpha=0.8)
    ax3.errorbar(range(len(runs)), means_arr, yerr=stds_arr, fmt="none",
                 ecolor="black", capsize=4, lw=1.5)
    # Linear trend
    if len(runs) > 2:
        x_fit = np.array(range(len(runs)))
        y_fit = np.polyval([drift_per_run, means_arr[0] - drift_per_run * x_fit[0]], x_fit)
        ax3.plot(x_fit, y_fit, ls="--", color="#6A1B9A", lw=1.5,
                 label=f"Trend: {drift_per_run:+.3f}/run")
        ax3.legend(fontsize=8)
    ax3.set_ylabel("Consistency score (L2) — mean ± std")
    ax3.set_title("Per-Run Consistency Score Distribution\n(RF trained on early runs)", fontsize=10, fontweight="bold")
    ax3.set_xticklabels(run_labels, fontsize=7, rotation=30, ha="right")

    # Panel 4: Score distribution per run (violin / box)
    ax4 = axes[1, 1]
    score_data = []
    for run in runs:
        rdf = df[df["run"] == run]
        Y_pred_run = rf.predict(rdf[PHYSICAL_COLS].values)
        scores_run = np.linalg.norm(rdf[vocab].values - Y_pred_run, axis=1)
        score_data.append(scores_run)

    bp = ax4.boxplot(score_data, patch_artist=True, notch=False, widths=0.5,
                     medianprops=dict(color="white", lw=2))
    for patch, run in zip(bp["boxes"], runs):
        patch.set_facecolor("#1565C0" if run in train_runs else "#B71C1C")
        patch.set_alpha(0.7)
    ax4.set_xticklabels(run_labels, fontsize=7, rotation=30, ha="right")
    ax4.set_ylabel("Consistency score (L2)")
    ax4.set_title("Consistency Score Distribution Per Run", fontsize=10, fontweight="bold")
    ax4.axhline(17.61, ls="--", color="#F57F17", lw=1.0, label="p95 (training set)")
    ax4.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT / "fig_temporal_stability.png", bbox_inches="tight")
    plt.close(fig)

    # ── Interpretation ────────────────────────────────────────────────────────
    drift_note = ""
    if abs(drift_per_run) < 0.5:
        drift_note = "The consistency score distribution is stable across runs (low temporal drift)."
    elif drift_per_run > 0.5:
        drift_note = f"Consistency score drifts upward (+{drift_per_run:.2f}/run), suggesting the model degrades over time."
    else:
        drift_note = f"Consistency score drifts downward ({drift_per_run:.2f}/run), suggesting the system becomes more predictable over time."

    r2_stable = abs(r2_temporal - r2_loro_mean) / max(r2_loro_mean, 1e-6) < 0.15
    relationship_note = (
        f"The physical-to-trace relationship {'is' if r2_stable else 'is NOT'} stable "
        f"across the {len(runs)} experimental sessions. "
        f"Temporal R²={r2_temporal:.3f} vs LORO R²={r2_loro_mean:.3f} "
        f"({'within 15%' if r2_stable else f'{drift_pct:.0f}% degradation'})."
    )

    output = {
        "runs_chronological":      runs,
        "train_runs_early":        train_runs,
        "test_runs_late":          test_runs,
        "r2_loro_mean":            r2_loro_mean,
        "r2_temporal_early_late":  r2_temporal,
        "r2_degradation_pct":      float(drift_pct),
        "loro_per_run":            loro_scores,
        "per_run_consistency":     per_run_stats,
        "drift_per_run":           drift_per_run,
        "interpretation": {
            "relationship_stability": relationship_note,
            "score_drift":            drift_note,
            "verdict": "stable" if r2_stable else "unstable",
        },
    }
    (OUT / "temporal_stability.json").write_text(json.dumps(output, indent=2))

    print(f"\n── Temporal Stability Summary ──")
    print(f"  {relationship_note}")
    print(f"  {drift_note}")
    print("\nSaved fig_temporal_stability.png and temporal_stability.json")
    print("Done.")


if __name__ == "__main__":
    main()
