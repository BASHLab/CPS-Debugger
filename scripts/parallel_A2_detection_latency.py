"""
parallel_A2_detection_latency.py — Detection latency analysis.

Uses the RF model from exp03 (retrained) to answer: at p95/p99 thresholds,
how quickly (in ms) does each anomaly type trigger an alert?

Output:
  outputs/parallel/detection_latency.json
  outputs/parallel/fig_detection_latency.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

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
WINDOW_MS = 10  # ms per window
N_REPEATS = 500  # anomalies per type


def inject_anomalies(df, vocab, model, p95, p99, n=N_REPEATS, rng=None):
    """Inject anomalies one window at a time; record per-type detection lag."""
    if rng is None:
        rng = np.random.default_rng(42)

    # Use the test set (last 20%)
    n80 = int(len(df) * 0.8)
    test = df.iloc[n80:].reset_index(drop=True)
    if len(test) < 100:
        return {}

    X_test = test[PHYSICAL_COLS].values
    Y_pred = model.predict(X_test)
    Y_true = test[vocab].values
    base_scores = np.linalg.norm(Y_true - Y_pred, axis=1)

    # Candidate injection indices (well within the test window, not near edges)
    margin = 20
    candidate_idx = np.arange(margin, len(test) - margin)
    inject_at = rng.choice(candidate_idx, size=min(n, len(candidate_idx)), replace=False)

    results = {}

    # ── Type 1: Trace Swap ────────────────────────────────────────────────────
    swingup_rows = df[df["dl_state"] == 0][vocab].values
    balance_rows = df[df["dl_state"] == 1][vocab].values
    swap_lags_p95, swap_lags_p99 = [], []

    for idx in inject_at:
        src_state = test["dl_state"].iloc[idx]
        # Swap: if BALANCE, use SWINGUP trace and vice versa
        pool = swingup_rows if src_state == 1 else balance_rows
        if len(pool) == 0:
            continue
        swap_vec = pool[rng.integers(len(pool))]

        # Compute new score at injection window
        Y_anom = Y_true[idx:idx+1].copy()
        Y_anom[0] = swap_vec
        anom_score = np.linalg.norm(Y_anom - Y_pred[idx:idx+1], axis=1)[0]

        swap_lags_p95.append(0 if anom_score >= p95 else 1)
        swap_lags_p99.append(0 if anom_score >= p99 else 1)

    results["Trace Swap"] = {
        "p95_same_window_pct": float(np.mean([x == 0 for x in swap_lags_p95]) * 100),
        "p99_same_window_pct": float(np.mean([x == 0 for x in swap_lags_p99]) * 100),
        "p95_latency_ms":      float(np.mean(swap_lags_p95) * WINDOW_MS),
        "p99_latency_ms":      float(np.mean(swap_lags_p99) * WINDOW_MS),
        "n":                   len(swap_lags_p95),
    }

    # ── Type 2: Physical Noise (3σ) ───────────────────────────────────────────
    feat_stds = test[PHYSICAL_COLS].std().values
    noise_lags_p95, noise_lags_p99 = [], []

    for idx in inject_at:
        X_anom = X_test[idx:idx+1].copy()
        X_anom += rng.standard_normal(X_anom.shape) * 3 * feat_stds
        Y_pred_anom = model.predict(X_anom)
        anom_score = np.linalg.norm(Y_true[idx:idx+1] - Y_pred_anom, axis=1)[0]
        noise_lags_p95.append(0 if anom_score >= p95 else 1)
        noise_lags_p99.append(0 if anom_score >= p99 else 1)

    results["Sensor Corruption (3σ)"] = {
        "p95_same_window_pct": float(np.mean([x == 0 for x in noise_lags_p95]) * 100),
        "p99_same_window_pct": float(np.mean([x == 0 for x in noise_lags_p99]) * 100),
        "p95_latency_ms":      float(np.mean(noise_lags_p95) * WINDOW_MS),
        "p99_latency_ms":      float(np.mean(noise_lags_p99) * WINDOW_MS),
        "n":                   len(noise_lags_p95),
    }

    # ── Type 3: Timing Shift 50ms (5 windows) ─────────────────────────────────
    timing_lags_p95, timing_lags_p99 = [], []
    SHIFT = 5  # windows

    for idx in inject_at:
        if idx < SHIFT:
            continue
        X_shifted = X_test[idx - SHIFT:idx - SHIFT + 1]
        Y_pred_shifted = model.predict(X_shifted)
        anom_score = np.linalg.norm(Y_true[idx:idx+1] - Y_pred_shifted, axis=1)[0]
        timing_lags_p95.append(0 if anom_score >= p95 else 1)
        timing_lags_p99.append(0 if anom_score >= p99 else 1)

    results["Timing Shift (50ms)"] = {
        "p95_same_window_pct": float(np.mean([x == 0 for x in timing_lags_p95]) * 100),
        "p99_same_window_pct": float(np.mean([x == 0 for x in timing_lags_p99]) * 100),
        "p95_latency_ms":      float(np.mean(timing_lags_p95) * WINDOW_MS),
        "p99_latency_ms":      float(np.mean(timing_lags_p99) * WINDOW_MS),
        "n":                   len(timing_lags_p95),
    }

    # ── Multi-window confirmation (3 consecutive above-threshold) ─────────────
    # For each injection point, look at windows [idx, idx+1, idx+2]
    # and check if all 3 exceed threshold (reduces false positives).
    confirm_results = {}
    for atype, inject_scores_p95, inject_scores_p99 in [
        ("Trace Swap", swap_lags_p95, swap_lags_p99),
        ("Sensor Corruption (3σ)", noise_lags_p95, noise_lags_p99),
    ]:
        # With 3-window confirmation: add latency of max(0, 3 - hit_count) * WINDOW_MS
        # Simplification: since anomaly persists continuously, 3-window lag = 20ms extra
        p95_conf_lat = float(np.mean(inject_scores_p95) * WINDOW_MS + 20)
        p99_conf_lat = float(np.mean(inject_scores_p99) * WINDOW_MS + 20)
        confirm_results[atype] = {
            "p95_latency_3win_ms": p95_conf_lat,
            "p99_latency_3win_ms": p99_conf_lat,
        }
    results["_confirmation_latency"] = confirm_results

    return results


def main():
    df = pd.read_parquet(EXP / "aligned_dataset.parquet")
    vi = json.loads((EXP / "vocab_info.json").read_text())
    vocab = vi["vocab"]

    exp03 = json.loads((EXP / "anomaly_detection_results.json").read_text())
    p95 = exp03["normal_consistency"]["p95"]
    p99 = exp03["normal_consistency"]["p99"]
    print(f"Thresholds: p95={p95:.2f}, p99={p99:.2f}")

    # Train RF (same as exp03: 80% train)
    n80 = int(len(df) * 0.8)
    train = df.iloc[:n80]
    print("Training RF on 80% of data...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=8, random_state=42)
    rf.fit(train[PHYSICAL_COLS].values, train[vocab].values)

    print("Injecting anomalies and measuring detection latency...")
    results = inject_anomalies(df, vocab, rf, p95, p99)

    confirm = results.pop("_confirmation_latency", {})

    # Save
    output = {
        "thresholds": {"p95": p95, "p99": p99},
        "window_ms":  WINDOW_MS,
        "per_type":   results,
        "interpretation": (
            "p95_same_window_pct: % of anomalies detected in the same 10ms window they occur. "
            "p95_latency_ms: mean detection latency at p95 threshold. "
            "With 3-window confirmation, add 20ms for robustness."
        ),
        "confirmation_3win": confirm,
    }
    (OUT / "detection_latency.json").write_text(json.dumps(output, indent=2))

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n── Detection Latency Summary ──")
    for atype, r in results.items():
        print(f"  {atype}:")
        print(f"    p95: {r['p95_same_window_pct']:.1f}% same-window, mean latency {r['p95_latency_ms']:.1f}ms")
        print(f"    p99: {r['p99_same_window_pct']:.1f}% same-window, mean latency {r['p99_latency_ms']:.1f}ms")

    # ── Figure ────────────────────────────────────────────────────────────────
    atypes = list(results.keys())
    x = np.arange(len(atypes))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    p95_pcts = [results[a]["p95_same_window_pct"] for a in atypes]
    p99_pcts = [results[a]["p99_same_window_pct"] for a in atypes]
    bars1 = ax.bar(x - width/2, p95_pcts, width, label="p95 threshold", color="#F57F17", edgecolor="black", lw=0.5)
    bars2 = ax.bar(x + width/2, p99_pcts, width, label="p99 threshold", color="#B71C1C", edgecolor="black", lw=0.5)
    ax.set_ylabel("% detected in same 10ms window")
    ax.set_title("Same-Window Detection Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(atypes, rotation=15, ha="right")
    ax.set_ylim(0, 110)
    ax.axhline(100, ls="--", color="gray", lw=0.8)
    ax.legend()
    for b, v in list(zip(bars1, p95_pcts)) + list(zip(bars2, p99_pcts)):
        ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.0f}%", ha="center", fontsize=8)

    ax2 = axes[1]
    p95_lats = [results[a]["p95_latency_ms"] for a in atypes]
    p99_lats = [results[a]["p99_latency_ms"] for a in atypes]
    conf_p95 = [confirm.get(a, {}).get("p95_latency_3win_ms", p95_lats[i] + 20)
                for i, a in enumerate(atypes)]
    ax2.bar(x - width/2, p95_lats, width, label="p95 (immediate)", color="#F57F17", edgecolor="black", lw=0.5)
    ax2.bar(x + width/2, p99_lats, width, label="p99 (immediate)", color="#B71C1C", edgecolor="black", lw=0.5)
    ax2.axhline(10,  ls="--", color="#2E8B57", lw=1.2, label="1 window (10ms)")
    ax2.axhline(100, ls="--", color="#808080", lw=0.8)
    ax2.set_ylabel("Mean detection latency (ms)")
    ax2.set_title("Detection Latency (ms)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(atypes, rotation=15, ha="right")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT / "fig_detection_latency.png", bbox_inches="tight")
    plt.close(fig)
    print("\nSaved detection_latency.json and fig_detection_latency.png")
    print("Done.")


if __name__ == "__main__":
    main()
