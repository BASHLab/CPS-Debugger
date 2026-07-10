"""
03_anomaly_detection.py — Cross-modal consistency scoring + synthetic anomaly injection.

Trains a branch-count predictor on the first 80% of data (temporal).
Evaluates consistency (prediction residual) on the last 20%.
Injects 3 synthetic anomaly types and measures detection via ROC curves.

Anomaly types:
  1. Trace swap: replace BALANCE branch vectors with SWINGUP ones (wrong code path)
  2. Physical noise: add 3σ Gaussian noise to physical sensor features
  3. Temporal shift: misalign physical features by +5ms, +10ms, +50ms

Outputs:
  fig_03a_consistency_timeline.png
  fig_03b_consistency_distribution.png
  fig_03c_anomaly_roc_curves.png
  fig_03d_anomaly_score_histograms.png
  anomaly_detection_results.json
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score, roc_curve

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
    "dl_current_x_mean", "dl_current_x_std", "dl_current_x_delta",
    "dl_angle_mean", "dl_angle_std", "dl_angle_delta",
    "dl_velocity_mean", "dl_ang_vel_mean", "dl_ang_vel_std",
]
SYSLOG_COLS = ["sl_cpu", "sl_mem", "sl_load1", "sl_load5", "sl_load15"]
ALL_FEATURE_COLS = PHYSICAL_COLS + SYSLOG_COLS
STATE_NAMES = {0: "SWINGUP", 1: "BALANCE", 2: "RESET"}
STATE_COLORS = {0: "#FF9800", 1: "#4CAF50", 2: "#9C27B0"}


def consistency_score(model: RandomForestRegressor, X: np.ndarray,
                      Y_true: np.ndarray) -> np.ndarray:
    """Compute L2 norm of residual between predicted and actual branch vector."""
    Y_pred = model.predict(X)
    residual = Y_true - Y_pred
    return np.linalg.norm(residual, axis=1)


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

    # Filter zero-trace windows
    branch_matrix = df[vocab].values.astype(float)
    valid = branch_matrix.sum(axis=1) > 0
    df = df[valid].reset_index(drop=True)
    branch_matrix = branch_matrix[valid]
    print(f"Dataset: {len(df):,} windows")

    X = df[ALL_FEATURE_COLS].values.astype(float)
    Y = branch_matrix
    states = df["dl_state"].values

    # ── Train/test split: first 80% for training ──────────────────────────────
    n = len(df)
    n_train = int(0.8 * n)
    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]
    states_test = states[n_train:]

    print(f"Train: {n_train:,} windows, Test: {n - n_train:,} windows")
    print("Training consistency model (RF regressor)...")
    model = RandomForestRegressor(
        n_estimators=100, n_jobs=-1, random_state=42, max_depth=12
    )
    model.fit(X_train, Y_train)

    # Baseline consistency scores on normal test data
    scores_normal = consistency_score(model, X_test, Y_test)
    p95 = np.percentile(scores_normal, 95)
    p99 = np.percentile(scores_normal, 99)
    print(f"Normal consistency: mean={scores_normal.mean():.2f}, "
          f"p95={p95:.2f}, p99={p99:.2f}")

    # ── Anomaly injection ─────────────────────────────────────────────────────
    # For each anomaly type, create 100 randomly selected windows with injected anomaly
    rng = np.random.default_rng(42)
    n_test = len(X_test)
    anomaly_indices = rng.choice(n_test, size=min(200, n_test // 2), replace=False)

    # Type 1: Trace swap — replace branch vectors with those from different state
    # In test set, swap BALANCE windows' branch vectors with SWINGUP-like (from training)
    swingup_train_idx = np.where(states[:n_train] == 0)[0]
    balance_train_idx = np.where(states[:n_train] == 1)[0]

    Y_swap = Y_test.copy()
    if len(swingup_train_idx) > 0:
        swap_source_idx = rng.choice(swingup_train_idx, size=len(anomaly_indices), replace=True)
        Y_swap[anomaly_indices] = Y_train[swap_source_idx]
    scores_swap = consistency_score(model, X_test, Y_swap)

    # Type 2: Physical noise — add 3σ Gaussian noise to physical features
    sigma_phy = X_train[:, :len(PHYSICAL_COLS)].std(axis=0) * 3.0
    X_noisy = X_test.copy()
    noise = rng.normal(0, sigma_phy, size=(len(anomaly_indices), len(PHYSICAL_COLS)))
    X_noisy[np.ix_(anomaly_indices, np.arange(len(PHYSICAL_COLS)))] += noise
    scores_noisy = consistency_score(model, X_noisy, Y_test)

    # Type 3: Temporal shift — shift physical features by K windows
    def shifted_score(shift_windows: int) -> np.ndarray:
        X_shifted = X_test.copy()
        n_t = len(X_test)
        shift = shift_windows
        X_shifted_phy = np.roll(X_test[:, :len(PHYSICAL_COLS)], shift, axis=0)
        if shift > 0:
            X_shifted_phy[:shift] = X_test[:shift, :len(PHYSICAL_COLS)]
        X_shifted[:, :len(PHYSICAL_COLS)] = X_shifted_phy
        scores = consistency_score(model, X_shifted, Y_test)
        return scores

    # At 10ms windows: 1 window = 10ms, so 5ms=0.5→1, 10ms=1, 50ms=5 windows
    scores_shift5ms = shifted_score(1)    # 10ms (1 window)
    scores_shift10ms = shifted_score(1)   # 10ms (practical minimum)
    scores_shift50ms = shifted_score(5)   # 50ms

    # ── Figure 3a: Consistency score timeline ─────────────────────────────────
    t_idx = np.arange(len(scores_normal))
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax = axes[0]
    for state_id, name, col in [(0, "SWINGUP", "#FF9800"), (1, "BALANCE", "#4CAF50"), (2, "RESET", "#9C27B0")]:
        mask = states_test == state_id
        ax.scatter(t_idx[mask], scores_normal[mask], s=2, alpha=0.4, color=col,
                   label=name, rasterized=True)
    ax.axhline(p95, color="orange", linestyle="--", linewidth=1.2, label=f"95th pct ({p95:.1f})")
    ax.axhline(p99, color="red", linestyle="--", linewidth=1.2, label=f"99th pct ({p99:.1f})")
    ax.set_ylabel("Consistency Score\n(L2 residual)")
    ax.set_title("Cross-Modal Consistency Score on Test Set (Normal Data)")
    ax.legend(markerscale=3, loc="upper right")

    ax = axes[1]
    ax.stackplot(t_idx,
                 [(states_test == 0).astype(float)],
                 [(states_test == 1).astype(float)],
                 [(states_test == 2).astype(float)],
                 labels=["SWINGUP", "BALANCE", "RESET"],
                 colors=["#FF9800", "#4CAF50", "#9C27B0"],
                 alpha=0.7)
    ax.set_xlabel("Window index (10ms each)")
    ax.set_ylabel("State")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_03a_consistency_timeline.png")
    plt.close()

    # ── Figure 3b: Score distribution by state ────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    for ax, (state_id, name) in zip(axes, STATE_NAMES.items()):
        mask = states_test == state_id
        if mask.sum() < 10:
            ax.set_title(f"{name} (insufficient data)")
            continue
        s = scores_normal[mask]
        ax.hist(s, bins=50, color=STATE_COLORS[state_id], alpha=0.8, edgecolor="none", density=True)
        ax.axvline(np.percentile(s, 95), color="orange", linestyle="--", linewidth=1.5, label="95th pct")
        ax.axvline(np.percentile(s, 99), color="red", linestyle="--", linewidth=1.5, label="99th pct")
        ax.set_title(f"{name} (n={mask.sum():,})")
        ax.set_xlabel("Consistency score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
    fig.suptitle("Consistency Score Distribution by State", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_03b_consistency_distribution.png")
    plt.close()

    # ── Compute ROC curves ─────────────────────────────────────────────────────
    # Binary labels: normal (0) vs anomalous (1)
    y_normal = np.zeros(n_test)
    y_anomaly = np.zeros(n_test)
    y_anomaly[anomaly_indices] = 1.0

    def compute_roc(scores_normal_full, scores_anomaly_at_idx):
        """Mix normal and anomalous scores, compute ROC."""
        combined_scores = scores_normal_full.copy()
        combined_scores[anomaly_indices] = scores_anomaly_at_idx[anomaly_indices]
        fpr, tpr, _ = roc_curve(y_anomaly, combined_scores)
        auc = roc_auc_score(y_anomaly, combined_scores)
        return fpr, tpr, auc

    anomaly_types = {
        "Trace Swap\n(wrong code path)": (scores_swap, "#F44336"),
        "Physical Noise\n(3σ sensor noise)": (scores_noisy, "#2196F3"),
        "Temporal Shift 10ms\n(timing error)": (scores_shift10ms, "#4CAF50"),
        "Temporal Shift 50ms\n(large timing error)": (scores_shift50ms, "#FF9800"),
    }

    roc_results = {}
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.5)")

    for anom_name, (scores_anom, color) in anomaly_types.items():
        fpr, tpr, auc = compute_roc(scores_normal, scores_anom)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{anom_name.split(chr(10))[0]} (AUC={auc:.3f})")
        roc_results[anom_name.replace('\n', ' ')] = {
            "auc": float(auc),
            "n_anomalies": int(len(anomaly_indices)),
        }

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Anomaly Detection via Cross-Modal Consistency\n(ROC Curves)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_03c_anomaly_roc_curves.png")
    plt.close()

    # ── Figure 3d: Score histograms (normal vs each anomaly type) ────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (anom_name, (scores_anom, color)) in zip(axes.flat, anomaly_types.items()):
        # Normal scores at non-anomaly positions
        normal_at_clean = scores_normal[y_anomaly == 0]
        anom_at_anom = scores_anom[anomaly_indices]
        bins = np.linspace(0, max(scores_normal.max(), scores_anom[anomaly_indices].max()) * 1.05, 60)
        ax.hist(normal_at_clean, bins=bins, alpha=0.6, color="#90CAF9", label="Normal", density=True, edgecolor="none")
        ax.hist(anom_at_anom, bins=bins, alpha=0.6, color=color, label="Anomaly", density=True, edgecolor="none")
        ax.axvline(p95, color="black", linestyle="--", linewidth=1, label=f"Normal 95th pct")
        fpr, tpr, auc = compute_roc(scores_normal, scores_anom)
        ax.set_title(f"{anom_name}\nAUC={auc:.3f}", fontsize=10)
        ax.set_xlabel("Consistency score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    plt.suptitle("Consistency Score: Normal vs Anomalous Windows", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_03d_anomaly_score_histograms.png")
    plt.close()

    # ── Save results ──────────────────────────────────────────────────────────
    results_out = {
        "n_train": int(n_train),
        "n_test": int(n - n_train),
        "normal_consistency": {
            "mean": float(scores_normal.mean()),
            "std": float(scores_normal.std()),
            "p95": float(p95),
            "p99": float(p99),
        },
        "anomaly_results": roc_results,
        "n_injected_anomalies": int(len(anomaly_indices)),
    }
    (OUT_DIR / "anomaly_detection_results.json").write_text(
        json.dumps(results_out, indent=2)
    )
    print("\n── Anomaly Detection Results ──")
    for name, res in roc_results.items():
        print(f"  {name[:40]:<40}: AUC={res['auc']:.3f}")
    print(f"\nOutputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
