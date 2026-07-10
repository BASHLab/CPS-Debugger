"""
parallel_B3_false_positive_rate.py — Steady-state false positive rate analysis.

Answers: "What's your FPR in normal operation?"
  - FPR at p95 and p99 thresholds
  - Cluster FPs by proximity to state transitions
  - Effect of 3-window confirmation requirement

Outputs:
  outputs/parallel/fig_false_positive_analysis.png
  outputs/parallel/false_positive_results.json
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
WINDOW_MS = 10


def distance_to_nearest_transition(states):
    """For each index, compute distance (in windows) to nearest state transition."""
    n = len(states)
    transitions = np.where(states[:-1] != states[1:])[0]  # transition AFTER index i
    if len(transitions) == 0:
        return np.full(n, n)
    dists = np.full(n, n, dtype=int)
    for i in range(n):
        d = np.abs(transitions - i)
        dists[i] = d.min()
    return dists


def sliding_window_confirmation(scores, threshold, k=3):
    """Return binary alert array: 1 if k consecutive windows all >= threshold."""
    alerts = np.zeros(len(scores), dtype=int)
    for i in range(k - 1, len(scores)):
        if all(scores[i - j] >= threshold for j in range(k)):
            alerts[i] = 1
    return alerts


def main():
    df = pd.read_parquet(EXP / "aligned_dataset.parquet")
    vi = json.loads((EXP / "vocab_info.json").read_text())
    vocab = vi["vocab"]

    exp03 = json.loads((EXP / "anomaly_detection_results.json").read_text())
    p95 = exp03["normal_consistency"]["p95"]
    p99 = exp03["normal_consistency"]["p99"]

    # Train on 80%, evaluate on last 20% (held-out, NO injected anomalies)
    n80 = int(len(df) * 0.8)
    train = df.iloc[:n80]
    test  = df.iloc[n80:].reset_index(drop=True)

    print("Training RF...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=8, random_state=42)
    rf.fit(train[PHYSICAL_COLS].values, train[vocab].values)

    print("Computing consistency scores on held-out test set (no anomalies)...")
    Y_pred = rf.predict(test[PHYSICAL_COLS].values)
    scores = np.linalg.norm(test[vocab].values - Y_pred, axis=1)
    states = test["dl_state"].values
    times  = np.arange(len(scores)) * WINDOW_MS / 1000  # seconds

    # ── FPR at different thresholds ───────────────────────────────────────────
    fpr_p95 = float(np.mean(scores >= p95) * 100)
    fpr_p99 = float(np.mean(scores >= p99) * 100)
    print(f"FPR at p95: {fpr_p95:.2f}%")
    print(f"FPR at p99: {fpr_p99:.2f}%")

    # ── 3-window confirmation ─────────────────────────────────────────────────
    alerts_p95_3win = sliding_window_confirmation(scores, p95, k=3)
    alerts_p99_3win = sliding_window_confirmation(scores, p99, k=3)
    fpr_p95_3win = float(alerts_p95_3win.mean() * 100)
    fpr_p99_3win = float(alerts_p99_3win.mean() * 100)
    print(f"FPR at p95 (3-win): {fpr_p95_3win:.2f}%")
    print(f"FPR at p99 (3-win): {fpr_p99_3win:.2f}%")

    # ── FP proximity to state transitions ─────────────────────────────────────
    fp_mask_p95 = scores >= p95
    fp_mask_p99 = scores >= p99
    dists = distance_to_nearest_transition(states)

    fp_dists_p95  = dists[fp_mask_p95]
    all_dists     = dists

    NEAR_TRANSITION_MS = 50  # 5 windows = 50ms
    n_near_p95 = int(np.sum(fp_dists_p95 <= NEAR_TRANSITION_MS // WINDOW_MS))
    n_far_p95  = len(fp_dists_p95) - n_near_p95
    pct_near_p95 = float(n_near_p95 / len(fp_dists_p95) * 100) if len(fp_dists_p95) > 0 else 0.0

    fp_dists_p99  = dists[fp_mask_p99]
    n_near_p99    = int(np.sum(fp_dists_p99 <= NEAR_TRANSITION_MS // WINDOW_MS))
    pct_near_p99  = float(n_near_p99 / len(fp_dists_p99) * 100) if len(fp_dists_p99) > 0 else 0.0

    print(f"FPs near transitions (<{NEAR_TRANSITION_MS}ms) at p95: {pct_near_p95:.1f}%")
    print(f"FPs near transitions (<{NEAR_TRANSITION_MS}ms) at p99: {pct_near_p99:.1f}%")

    # ── FPR by state ──────────────────────────────────────────────────────────
    fpr_by_state = {}
    for s, name in [(0, "SWINGUP"), (1, "BALANCE"), (2, "RESET")]:
        mask = states == s
        if mask.any():
            fpr_by_state[name] = float(np.mean(scores[mask] >= p95) * 100)

    # ── Figures ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=False,
                              gridspec_kw={"hspace": 0.35})

    STATE_COLORS = {0: "#FF8C00", 1: "#2E8B57", 2: "#DC143C"}
    STATE_ALPHA  = 0.12

    # Panel 1: consistency score over time (test set)
    ax = axes[0]
    # Add state bands
    prev_s, seg_t = states[0], times[0]
    for i, (t, s) in enumerate(zip(times[1:], states[1:]), 1):
        if s != prev_s:
            ax.axvspan(seg_t, t, color=STATE_COLORS[prev_s], alpha=STATE_ALPHA, lw=0)
            seg_t, prev_s = t, s
    ax.axvspan(seg_t, times[-1], color=STATE_COLORS[prev_s], alpha=STATE_ALPHA, lw=0)

    ax.plot(times, scores, color="#455A64", lw=0.6, alpha=0.8, label="Consistency score")
    ax.axhline(p95, ls="--", color="#F57F17", lw=1.2, label=f"p95={p95:.1f} (FPR={fpr_p95:.1f}%)")
    ax.axhline(p99, ls="--", color="#B71C1C", lw=1.0, label=f"p99={p99:.1f} (FPR={fpr_p99:.1f}%)")

    # Mark FP windows
    fp_times_p95 = times[fp_mask_p95]
    fp_scores_p95 = scores[fp_mask_p95]
    ax.scatter(fp_times_p95, fp_scores_p95, color="#F57F17", s=8, alpha=0.5, zorder=5, label="FP (p95)")

    ax.set_ylabel("Consistency score (L2)")
    ax.set_xlabel("Time (s) — held-out test set (no injected anomalies)")
    ax.set_title("Steady-State False Positive Analysis — Normal Operation", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, ncol=4)
    ax.margins(x=0)

    # Panel 2: histogram of FP distance to nearest transition
    ax2 = axes[1]
    bin_edges = np.arange(0, 51, 5) * WINDOW_MS  # 0, 50, 100, ..., 500ms
    dist_ms_all = all_dists * WINDOW_MS
    dist_ms_fp  = fp_dists_p95 * WINDOW_MS

    ax2.hist(dist_ms_all[dist_ms_all <= 500], bins=bin_edges, alpha=0.5, color="#90CAF9",
             density=True, label="All windows")
    ax2.hist(dist_ms_fp[dist_ms_fp <= 500], bins=bin_edges, alpha=0.7, color="#F57F17",
             density=True, label=f"FP windows (p95)")
    ax2.axvline(NEAR_TRANSITION_MS, ls="--", color="#6A1B9A", lw=1.2,
                label=f"{NEAR_TRANSITION_MS}ms boundary")
    ax2.set_xlabel("Distance to nearest state transition (ms)")
    ax2.set_ylabel("Density")
    ax2.set_title(f"False Positives Cluster Near State Transitions\n({pct_near_p95:.0f}% of FPs within {NEAR_TRANSITION_MS}ms of transition)")
    ax2.legend(fontsize=8)

    # Panel 3: FPR comparison bar chart
    ax3 = axes[2]
    configs  = ["p95\n(immediate)", "p99\n(immediate)", "p95\n(3-win confirm)", "p99\n(3-win confirm)"]
    fpr_vals = [fpr_p95, fpr_p99, fpr_p95_3win, fpr_p99_3win]
    bar_cols = ["#F57F17", "#B71C1C", "#FFB74D", "#EF9A9A"]
    bars = ax3.bar(configs, fpr_vals, color=bar_cols, edgecolor="black", lw=0.5)
    ax3.axhline(5.0, ls="--", color="#B71C1C", lw=1, label="p95 nominal FPR (5%)")
    ax3.axhline(1.0, ls="--", color="#F57F17", lw=1, label="p99 nominal FPR (1%)")
    for b, v in zip(bars, fpr_vals):
        ax3.text(b.get_x() + b.get_width()/2, v + 0.05, f"{v:.2f}%",
                 ha="center", fontsize=10, fontweight="bold")
    ax3.set_ylabel("False Positive Rate (%)")
    ax3.set_title("FPR: Immediate vs. 3-Window Confirmation")
    ax3.legend(fontsize=8)
    ax3.set_ylim(0, max(fpr_vals) * 1.4)

    plt.savefig(OUT / "fig_false_positive_analysis.png", bbox_inches="tight")
    plt.close(fig)

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "thresholds": {"p95": p95, "p99": p99},
        "fpr_pct": {
            "p95_immediate":  fpr_p95,
            "p99_immediate":  fpr_p99,
            "p95_3win":       fpr_p95_3win,
            "p99_3win":       fpr_p99_3win,
        },
        "fp_near_transitions": {
            "boundary_ms":    NEAR_TRANSITION_MS,
            "pct_near_p95":   pct_near_p95,
            "pct_near_p99":   pct_near_p99,
        },
        "fpr_by_state": fpr_by_state,
        "summary": (
            f"At p95, the steady-state FPR is {fpr_p95:.2f}%. "
            f"At p99, it is {fpr_p99:.2f}%. "
            f"{pct_near_p95:.0f}% of false positives occur within {NEAR_TRANSITION_MS}ms of a state transition. "
            f"With a 3-window confirmation requirement, FPR drops to {fpr_p95_3win:.2f}% (p95) "
            f"and {fpr_p99_3win:.2f}% (p99)."
        ),
    }
    (OUT / "false_positive_results.json").write_text(json.dumps(output, indent=2))

    print(f"\n── Summary ──")
    print(f"  {output['summary']}")
    print("\nSaved fig_false_positive_analysis.png and false_positive_results.json")
    print("Done.")


if __name__ == "__main__":
    main()
