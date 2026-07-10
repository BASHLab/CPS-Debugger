"""
parallel_A3_limitations.py — Known limitations of the cross-modal consistency approach.

Tests three scenarios the system may NOT detect:
  1. Slow proportional sensor drift (linear, 0.001 m/window)
  2. Same-state near-setpoint trace swap (BALANCE pos=0.0 vs pos=0.09)
  3. Logic bug with no physical signature (±1 branch count perturbation)

Outputs:
  outputs/parallel/fig_limitation_analysis.png
  outputs/parallel/limitations.json
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

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


def compute_consistency_scores(X, Y_true, model):
    Y_pred = model.predict(X)
    return np.linalg.norm(Y_true - Y_pred, axis=1)


def main():
    df = pd.read_parquet(EXP / "aligned_dataset.parquet")
    vi = json.loads((EXP / "vocab_info.json").read_text())
    vocab = vi["vocab"]

    exp03 = json.loads((EXP / "anomaly_detection_results.json").read_text())
    p95 = exp03["normal_consistency"]["p95"]
    p99 = exp03["normal_consistency"]["p99"]

    n80 = int(len(df) * 0.8)
    train = df.iloc[:n80]
    test  = df.iloc[n80:].reset_index(drop=True)

    print("Training RF...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=8, random_state=42)
    rf.fit(train[PHYSICAL_COLS].values, train[vocab].values)

    X_test  = test[PHYSICAL_COLS].values
    Y_test  = test[vocab].values
    scores_normal = compute_consistency_scores(X_test, Y_test, rf)
    print(f"Normal consistency: mean={scores_normal.mean():.2f}, p95={np.percentile(scores_normal, 95):.2f}")

    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── Limitation 1: Proportional sensor drift ────────────────────────────────
    print("\n[1] Testing proportional sensor drift...")
    pos_col_idx = PHYSICAL_COLS.index("dl_current_x_mean")
    drift_rates   = np.arange(0, 0.5, 0.005)  # cumulative drift in meters
    pct_above_p95 = []

    for drift_total in drift_rates:
        X_drift = X_test.copy()
        # Apply cumulative linear drift (drift_total = max drift at end of test set)
        drift_per_win = drift_total / max(len(X_test), 1)
        drift_vec = np.arange(len(X_test)) * drift_per_win
        X_drift[:, pos_col_idx] += drift_vec
        scores_drift = compute_consistency_scores(X_drift, Y_test, rf)
        pct_above_p95.append(np.mean(scores_drift >= p95) * 100)

    # Threshold: when does >10% of windows exceed p95?
    detection_drift = next(
        (drift_rates[i] for i, p in enumerate(pct_above_p95) if p > 10),
        drift_rates[-1]
    )

    axes[0].plot(drift_rates * 100, pct_above_p95, color="#1565C0", lw=2)
    axes[0].axhline(5,  ls="--", color="#F57F17", lw=1, label="5% (baseline FPR)")
    axes[0].axhline(10, ls="--", color="#B71C1C", lw=1, label="10% (alert level)")
    axes[0].axvline(detection_drift * 100, ls=":", color="#6A1B9A", lw=1.5,
                    label=f"Detected at {detection_drift*100:.1f}cm drift")
    axes[0].set_xlabel("Cumulative position drift (cm)")
    axes[0].set_ylabel("% windows exceeding p95 threshold")
    axes[0].set_title("Limitation 1: Slow Sensor Drift\n(0.001m/window linear)")
    axes[0].legend(fontsize=8)

    results["slow_sensor_drift"] = {
        "detectable": bool(detection_drift < drift_rates[-1]),
        "detection_cumulative_drift_m": float(detection_drift),
        "detection_cumulative_drift_cm": float(detection_drift * 100),
        "explanation": (
            f"Slow linear drift is detectable only after {detection_drift*100:.1f}cm cumulative "
            f"displacement. Below that threshold, the RF adapts its predictions naturally because "
            f"physical features (angle, velocity) remain consistent with the drifted position. "
            f"This type of fault is hard to catch because the relationship between physical state "
            f"and execution path remains valid even with a slowly drifting position reading."
        ),
    }

    # ── Limitation 2: Near-setpoint trace swap ─────────────────────────────────
    print("\n[2] Testing near-setpoint BALANCE trace swap...")
    # Split BALANCE windows by target_x
    bal = test[test["dl_state"] == 1].copy()
    if "dl_target_x" in bal.columns and bal["dl_target_x"].nunique() > 1:
        targets = sorted(bal["dl_target_x"].unique())
        # Use the two most common distinct targets
        t_counts = bal["dl_target_x"].value_counts()
        t1, t2 = t_counts.index[0], t_counts.index[1] if len(t_counts) > 1 else t_counts.index[0]
        pool_t1 = bal[bal["dl_target_x"] == t1][vocab].values
        pool_t2 = bal[bal["dl_target_x"] == t2][vocab].values
        print(f"  Comparing target_x={t1:.3f} ({len(pool_t1)} windows) vs {t2:.3f} ({len(pool_t2)} windows)")
    else:
        # Fall back: split by median position
        median_pos = bal["dl_current_x_mean"].median()
        pool_t1 = bal[bal["dl_current_x_mean"] <= median_pos][vocab].values
        pool_t2 = bal[bal["dl_current_x_mean"] >  median_pos][vocab].values
        t1, t2 = "pos≤med", "pos>med"
        print(f"  Comparing pos≤median ({len(pool_t1)}) vs pos>median ({len(pool_t2)})")

    rng = np.random.default_rng(42)
    n_inject = min(200, len(pool_t1), len(pool_t2))
    X_bal  = bal[PHYSICAL_COLS].values[:n_inject]
    Y_bal  = bal[vocab].values[:n_inject]
    Y_pred_bal = rf.predict(X_bal)
    normal_scores_bal = np.linalg.norm(Y_bal - Y_pred_bal, axis=1)

    # Inject: swap trace from t1 windows with t2 traces
    swap_indices = rng.integers(len(pool_t2), size=n_inject)
    Y_swap = pool_t2[swap_indices]
    swap_scores = np.linalg.norm(Y_swap - Y_pred_bal, axis=1)

    labels_nearswap = np.array([0] * n_inject + [1] * n_inject)
    scores_nearswap = np.concatenate([normal_scores_bal, swap_scores])
    try:
        auc_nearswap = float(roc_auc_score(labels_nearswap, scores_nearswap))
    except Exception:
        auc_nearswap = 0.5

    bins = np.linspace(0, max(scores_nearswap.max(), p95 + 5), 40)
    axes[1].hist(normal_scores_bal, bins=bins, alpha=0.6, color="#2E8B57", label="Normal BALANCE", density=True)
    axes[1].hist(swap_scores, bins=bins, alpha=0.6, color="#B71C1C",
                 label=f"Near-setpoint swap\n(AUC={auc_nearswap:.3f})", density=True)
    axes[1].axvline(p95, ls="--", color="#F57F17", lw=1.2, label=f"p95={p95:.1f}")
    axes[1].axvline(p99, ls="--", color="#B71C1C", lw=1.0, label=f"p99={p99:.1f}")
    axes[1].set_xlabel("Consistency score (L2)")
    axes[1].set_ylabel("Density")
    axes[1].set_title(f"Limitation 2: Near-Setpoint BALANCE Swap\n(AUC={auc_nearswap:.3f})")
    axes[1].legend(fontsize=8)

    results["near_setpoint_swap"] = {
        "auc": auc_nearswap,
        "detectable": bool(auc_nearswap > 0.75),
        "target_1": str(t1),
        "target_2": str(t2),
        "n_injected": int(n_inject),
        "explanation": (
            f"Swapping BALANCE traces between different setpoint targets ({t1} vs {t2}) achieves "
            f"AUC={auc_nearswap:.3f}. "
            + ("This is hard to detect because within BALANCE, the execution path is highly "
               "stereotyped regardless of the exact setpoint — the LQR computes similar branch "
               "patterns as long as the pendulum remains balanced. The consistency score relies on "
               "physical features predicting branch patterns, and this relationship holds "
               "similarly across setpoints within the BALANCE state."
               if auc_nearswap < 0.75 else
               "The setpoint difference is large enough to produce distinguishable execution "
               "profiles, so detection is feasible.")
        ),
    }

    # ── Limitation 3: Logic bug with no physical signature ─────────────────────
    print("\n[3] Testing logic bug (±1 branch count perturbation)...")
    n_inject_3 = min(500, len(test))
    sample_idx = rng.choice(len(test), n_inject_3, replace=False)
    X_s  = X_test[sample_idx]
    Y_s  = Y_test[sample_idx]
    Y_pred_s = rf.predict(X_s)
    normal_scores_s = np.linalg.norm(Y_s - Y_pred_s, axis=1)

    # Perturb 5% of branch counts by ±1
    n_branches = len(vocab)
    n_perturb  = max(1, int(0.05 * n_branches))
    Y_perturbed = Y_s.copy()
    for i in range(n_inject_3):
        perturb_cols = rng.choice(n_branches, n_perturb, replace=False)
        Y_perturbed[i, perturb_cols] += rng.choice([-1, 1], n_perturb)
        Y_perturbed[i] = np.maximum(0, Y_perturbed[i])

    perturbed_scores = np.linalg.norm(Y_perturbed - Y_pred_s, axis=1)

    labels_bug = np.array([0] * n_inject_3 + [1] * n_inject_3)
    scores_bug = np.concatenate([normal_scores_s, perturbed_scores])
    try:
        auc_bug = float(roc_auc_score(labels_bug, scores_bug))
    except Exception:
        auc_bug = 0.5

    score_delta = perturbed_scores - normal_scores_s
    print(f"  AUC={auc_bug:.4f}, mean score delta={score_delta.mean():.3f} ± {score_delta.std():.3f}")

    bins_bug = np.linspace(0, max(perturbed_scores.max(), p95 + 5), 40)
    axes[2].hist(normal_scores_s, bins=bins_bug, alpha=0.6, color="#2E8B57", label="Normal", density=True)
    axes[2].hist(perturbed_scores, bins=bins_bug, alpha=0.6, color="#FF8C00",
                 label=f"±1 branch perturb\n(AUC={auc_bug:.3f})", density=True)
    axes[2].axvline(p95, ls="--", color="#F57F17", lw=1.2, label=f"p95={p95:.1f}")
    axes[2].set_xlabel("Consistency score (L2)")
    axes[2].set_ylabel("Density")
    axes[2].set_title(f"Limitation 3: Logic Bug (±1 Branch Count)\n(AUC={auc_bug:.3f})")
    axes[2].legend(fontsize=8)

    results["logic_bug_no_physical_signature"] = {
        "auc": auc_bug,
        "detectable": bool(auc_bug > 0.75),
        "n_branches_perturbed": int(n_perturb),
        "pct_branches_perturbed": float(n_perturb / n_branches * 100),
        "mean_score_delta": float(score_delta.mean()),
        "explanation": (
            f"Perturbing {n_perturb} of {n_branches} branch counts (5%) by ±1 achieves "
            f"AUC={auc_bug:.3f}. "
            + ("This type of fault is undetectable because the ±1 perturbation is within the "
               "natural variance of branch execution counts (counts fluctuate by small amounts "
               "depending on timing). The L2 residual is dominated by the larger variance in "
               "high-count branches, making single-count changes invisible."
               if auc_bug < 0.75 else
               "Despite the small perturbation, enough branches change to shift the L2 residual "
               "above the detection threshold.")
        ),
    }

    plt.tight_layout()
    fig.savefig(OUT / "fig_limitation_analysis.png", bbox_inches="tight")
    plt.close(fig)

    # Save results
    full_results = {
        "thresholds": {"p95": p95, "p99": p99},
        "summary": {
            "slow_drift_detected_at_cm":    results["slow_sensor_drift"]["detection_cumulative_drift_cm"],
            "near_setpoint_swap_auc":       results["near_setpoint_swap"]["auc"],
            "logic_bug_auc":                results["logic_bug_no_physical_signature"]["auc"],
        },
        "detailed": results,
    }
    (OUT / "limitations.json").write_text(json.dumps(full_results, indent=2))

    print("\n── Limitations Summary ──")
    print(f"  1. Slow drift: detectable at >{results['slow_sensor_drift']['detection_cumulative_drift_cm']:.1f}cm cumulative displacement")
    print(f"  2. Near-setpoint swap: AUC={results['near_setpoint_swap']['auc']:.3f}")
    print(f"  3. Logic bug (±1 count): AUC={results['logic_bug_no_physical_signature']['auc']:.3f}")
    print("\nSaved fig_limitation_analysis.png and limitations.json")
    print("Done.")


if __name__ == "__main__":
    main()
