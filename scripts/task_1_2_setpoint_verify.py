"""
task_1_2_setpoint_verify.py — Verify setpoint label accuracy.

Resolves the +0.05 vs +0.08 m label inconsistency: checks whether the controller
actually reaches the nominal target_x values in the datalayer data.

Output:
  outputs/phase3/setpoint_verification.json
  outputs/phase3/fig_setpoint_verification.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT   = Path("/home/simran/allspark-data-exploration/CPS-Debugger")
P3_OUT = ROOT / "outputs/phase3"
P3_OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150,
                     "savefig.dpi": 300})


def main():
    data_path = ROOT / "outputs/experiments/aligned_dataset.parquet"
    if not data_path.exists():
        print(f"ERROR: {data_path} not found.")
        return

    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} windows from aligned_dataset.parquet")

    # Filter to BALANCE state
    bal = df[df["dl_state"] == 1].copy()
    print(f"BALANCE windows: {len(bal):,}")

    target_vals = sorted(bal["dl_target_x"].dropna().unique())
    print(f"Target_x unique values in BALANCE: {target_vals}")

    # ── Stats per target_x class ──────────────────────────────────────────────
    stats = []
    for tv in target_vals:
        sub = bal[bal["dl_target_x"] == tv]
        mean_x  = sub["dl_current_x_mean"].mean()
        std_x   = sub["dl_current_x_mean"].std()
        error   = mean_x - tv
        n       = len(sub)
        # How well is the setpoint tracked? (% windows within 2cm)
        within_2cm = ((sub["dl_current_x_mean"] - tv).abs() < 0.02).mean()
        stats.append({
            "target_x_label":     float(tv),
            "n_windows":          int(n),
            "mean_achieved_x":    float(mean_x),
            "std_achieved_x":     float(std_x),
            "mean_tracking_error": float(error),
            "frac_within_2cm":    float(within_2cm),
        })
        print(f"  target_x={tv:+.3f}: mean_pos={mean_x:+.4f}±{std_x:.4f} "
              f"err={error:+.4f}  within_2cm={within_2cm:.1%}  n={n}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out = {
        "n_balance_windows": int(len(bal)),
        "target_x_stats": stats,
        "interpretation": (
            "target_x_label is the commanded setpoint from datalayer; "
            "mean_achieved_x is the actual cart position during that setpoint. "
            "Labels are exactly as reported by the controller, not rounded."
        ),
    }
    (P3_OUT / "setpoint_verification.json").write_text(json.dumps(out, indent=2))
    print("\nSaved setpoint_verification.json")

    # ── Figure 1: Position distribution per setpoint class ───────────────────
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: violin/box plot
    ax = axes[0]
    data_for_plot = [bal[bal["dl_target_x"] == tv]["dl_current_x_mean"].values
                     for tv in target_vals]
    parts = ax.violinplot(data_for_plot, positions=range(len(target_vals)),
                          showmedians=True, showextrema=False)
    for i, (pc, c) in enumerate(zip(parts["bodies"], colors[:len(target_vals)])):
        pc.set_facecolor(c)
        pc.set_alpha(0.7)
    ax.set_xticks(range(len(target_vals)))
    ax.set_xticklabels([f"{v:+.2f} m" for v in target_vals], rotation=30)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.7)
    for tv in target_vals:
        ax.axhline(tv, color="red", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.set_ylabel("Achieved cart position (m)")
    ax.set_xlabel("target_x label")
    ax.set_title("Actual cart position distribution per setpoint")

    # Right: time-series for one run's BALANCE segment
    ax2 = axes[1]
    run_id = bal["run"].value_counts().index[0]  # largest run
    run_bal = bal[bal["run"] == run_id].sort_values("win").head(2000)
    t = run_bal["win"].values * 0.01  # 10ms windows → seconds (relative)
    t = t - t[0]
    ax2.plot(t, run_bal["dl_current_x_mean"].values, lw=0.6,
             color="#2196F3", label="cart pos (m)")
    for tv, c in zip(target_vals, colors):
        ax2.axhline(tv, linestyle="--", color=c, linewidth=1.5,
                    label=f"target={tv:+.2f} m")
    ax2.set_xlabel("Time (s, relative)")
    ax2.set_ylabel("Cart position (m)")
    ax2.set_title(f"Time-series: {run_id[:20]} (BALANCE)")
    ax2.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    fig.savefig(P3_OUT / "fig_setpoint_verification.png")
    plt.close(fig)
    print("Saved fig_setpoint_verification.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
