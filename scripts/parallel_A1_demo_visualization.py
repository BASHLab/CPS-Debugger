"""
parallel_A1_demo_visualization.py — Operator dashboard and detection demo figures.

Outputs:
  outputs/parallel/fig_operator_dashboard.png  — 4-panel 30s segment with one state transition
  outputs/parallel/fig_detection_example.png   — ±500ms zoom around injected anomaly
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

ROOT   = Path("/home/simran/allspark-data-exploration/CPS-Debugger")
EXP    = ROOT / "outputs/experiments"
OUT    = ROOT / "outputs/parallel"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "font.family": "DejaVu Sans",
})

PHYSICAL_COLS = [
    "dl_current_x_mean", "dl_current_x_std", "dl_current_x_delta",
    "dl_angle_mean", "dl_angle_std", "dl_angle_delta",
    "dl_velocity_mean", "dl_ang_vel_mean", "dl_ang_vel_std",
]
STATE_COLORS = {0: "#FF8C00", 1: "#2E8B57", 2: "#DC143C"}
STATE_NAMES  = {0: "SWINGUP", 1: "BALANCE", 2: "RESET"}
STATE_ALPHA  = 0.15

def add_state_bands(ax, times, states):
    """Shade background by controller state."""
    if len(times) == 0:
        return
    prev_state = states[0]
    seg_start  = times[0]
    for t, s in zip(times[1:], states[1:]):
        if s != prev_state:
            ax.axvspan(seg_start, t, color=STATE_COLORS[prev_state], alpha=STATE_ALPHA, lw=0)
            seg_start = t
            prev_state = s
    ax.axvspan(seg_start, times[-1], color=STATE_COLORS[prev_state], alpha=STATE_ALPHA, lw=0)


def find_good_segment(df_run, win_len=300):
    """Find a 30s segment (300 × 10ms windows) that contains ≥1 state transition."""
    states = df_run["dl_state"].values
    wins   = df_run["win"].values
    n = len(states)
    best_i, best_score = 0, 0
    for i in range(n - win_len):
        seg = states[i:i + win_len]
        n_transitions = np.sum(seg[:-1] != seg[1:])
        # Prefer segments with a SWINGUP→BALANCE or BALANCE→RESET transition
        has_balance = 1 in seg
        has_swingup = 0 in seg
        score = n_transitions + 2 * int(has_balance and has_swingup)
        if score > best_score:
            best_score = score
            best_i = i
    return best_i, best_i + win_len


def compute_branch_labels(vocab, fn_names):
    """Map branch IDs to functional group labels."""
    labels = {}
    for b in vocab:
        parts = b.split(":")
        fid = parts[0] if parts else ""
        fn  = fn_names.get(fid, fn_names.get(str(fid), b))
        labels[b] = fn
    return labels


def make_dashboard(df_run, seg, vocab, fn_names, rf, p95, p99):
    """Build the 4-panel operator dashboard figure."""
    s, e = seg
    seg_df = df_run.iloc[s:e].copy().reset_index(drop=True)
    t = np.arange(len(seg_df)) * 0.01  # seconds

    states = seg_df["dl_state"].values
    angle  = seg_df["dl_angle_mean"].values
    pos    = seg_df["dl_current_x_mean"].values

    # Branch totals and function group presence
    bc = seg_df[vocab].values
    total_bc = bc.sum(axis=1)

    fn_labels = compute_branch_labels(vocab, fn_names)
    # Group into: LQR (balance), standup, other
    lqr_cols     = [v for v, l in fn_labels.items() if "lqr"     in l.lower()]
    standup_cols = [v for v, l in fn_labels.items() if "standup" in l.lower() or "swing" in l.lower()]
    lqr_sum     = seg_df[lqr_cols].sum(axis=1).values     if lqr_cols     else np.zeros(len(seg_df))
    standup_sum = seg_df[standup_cols].sum(axis=1).values if standup_cols else np.zeros(len(seg_df))

    # Consistency score: RF predictions on this segment
    X = seg_df[PHYSICAL_COLS].values
    Y_true = seg_df[vocab].values
    Y_pred = rf.predict(X)
    consistency = np.linalg.norm(Y_true - Y_pred, axis=1)

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True,
                             gridspec_kw={"hspace": 0.05, "height_ratios": [1.2, 1, 1, 1]})

    # ── Panel 1: Physical state ────────────────────────────────────────────────
    ax1 = axes[0]
    add_state_bands(ax1, t, states)
    l1, = ax1.plot(t, np.degrees(angle), color="#1565C0", lw=1.2, label="Angle (°)")
    ax1r = ax1.twinx()
    ax1r.spines["top"].set_visible(False)
    l2, = ax1r.plot(t, pos * 100, color="#B71C1C", lw=1.0, ls="--", label="Position (cm)")
    ax1.set_ylabel("Pendulum angle (°)", color="#1565C0")
    ax1r.set_ylabel("Cart position (cm)", color="#B71C1C")
    ax1.set_title("Operator Dashboard — Physical State & Execution Trace Monitor", fontsize=12, fontweight="bold")
    # State legend
    patches = [mpatches.Patch(color=STATE_COLORS[s], alpha=0.5, label=STATE_NAMES[s])
               for s in [0, 1, 2]]
    ax1.legend(handles=patches + [l1, l2], loc="upper right", fontsize=8, ncol=5)

    # ── Panel 2: Execution trace ───────────────────────────────────────────────
    ax2 = axes[1]
    add_state_bands(ax2, t, states)
    ax2.fill_between(t, 0, total_bc, color="#546E7A", alpha=0.7, label="Total branches")
    if lqr_cols:
        ax2.fill_between(t, 0, lqr_sum, color="#1565C0", alpha=0.8, label="LQR/balance")
    if standup_cols:
        ax2.fill_between(t, 0, standup_sum, color="#FF8C00", alpha=0.8, label="Standup")
    ax2.set_ylabel("Branch\nexecutions / 10ms")
    ax2.legend(loc="upper right", fontsize=8, ncol=3)

    # ── Panel 3: Consistency score ─────────────────────────────────────────────
    ax3 = axes[2]
    add_state_bands(ax3, t, states)
    ax3.plot(t, consistency, color="#37474F", lw=1.0, label="Consistency score (L2)")
    ax3.axhline(p95, ls="--", color="#F57F17", lw=1.2, label=f"p95 = {p95:.1f}")
    ax3.axhline(p99, ls="--", color="#B71C1C", lw=1.2, label=f"p99 = {p99:.1f}")
    above_p95 = consistency >= p95
    above_p99 = consistency >= p99
    ax3.fill_between(t, p95, consistency, where=above_p95 & ~above_p99,
                     color="#FFF176", alpha=0.8)
    ax3.fill_between(t, p99, consistency, where=above_p99,
                     color="#EF9A9A", alpha=0.8)
    ax3.set_ylabel("Consistency\nscore (L2)")
    ax3.legend(loc="upper right", fontsize=8, ncol=3)

    # ── Panel 4: Injected anomaly ──────────────────────────────────────────────
    ax4 = axes[3]
    add_state_bands(ax4, t, states)

    # Find a BALANCE window ~20% into the segment, inject trace-swap anomaly
    bal_idx = np.where(states == 1)[0]
    if len(bal_idx) == 0:
        bal_idx = np.arange(len(states) // 4, len(states) // 2)

    # Choose injection point in the middle of longest BALANCE run
    inject_idx = bal_idx[len(bal_idx) // 2]

    # Build anomaly consistency scores: inject a SWINGUP trace at inject_idx
    swingup_rows = df_run[df_run["dl_state"] == 0]
    if len(swingup_rows) > 0:
        swap_vec = swingup_rows[vocab].sample(1, random_state=42).values[0]
    else:
        swap_vec = np.zeros(len(vocab))

    Y_anom = Y_true.copy()
    Y_anom[inject_idx] = swap_vec
    consistency_anom = np.linalg.norm(Y_anom - Y_pred, axis=1)

    ax4.plot(t, consistency_anom, color="#37474F", lw=1.0, label="Consistency (with anomaly)")
    ax4.axhline(p95, ls="--", color="#F57F17", lw=1.2)
    ax4.axhline(p99, ls="--", color="#B71C1C", lw=1.2)
    above_99_anom = consistency_anom >= p99
    ax4.fill_between(t, p99, consistency_anom, where=above_99_anom,
                     color="#EF9A9A", alpha=0.9)
    inject_t = t[inject_idx]
    ax4.axvline(inject_t, color="#6A1B9A", lw=2, ls="-", label="Anomaly injected")
    ax4.annotate("Anomaly injected:\nbranch vector replaced\nwith SWINGUP trace",
                 xy=(inject_t, consistency_anom[inject_idx]),
                 xytext=(inject_t + 0.5, max(consistency_anom) * 0.85),
                 fontsize=8, color="#6A1B9A", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#6A1B9A", lw=1.5))
    ax4.set_ylabel("Consistency\nscore (L2)")
    ax4.set_xlabel("Time (s)")
    ax4.legend(loc="upper right", fontsize=8, ncol=2)

    for ax in axes:
        ax.margins(x=0)

    plt.savefig(OUT / "fig_operator_dashboard.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_operator_dashboard.png")

    return inject_idx, inject_t, consistency_anom, t, seg_df, states


def make_zoom_figure(t, seg_df, states, inject_idx, consistency_anom, vocab, fn_names, rf, p95, p99):
    """±500ms zoom around injected anomaly."""
    WIN_MS = 50  # ±500ms = ±50 windows
    s_idx = max(0, inject_idx - WIN_MS)
    e_idx = min(len(t), inject_idx + WIN_MS)

    t_z     = t[s_idx:e_idx] - t[inject_idx]  # center on injection
    states_z = states[s_idx:e_idx]
    seg_z   = seg_df.iloc[s_idx:e_idx].reset_index(drop=True)

    angle_z = seg_z["dl_angle_mean"].values
    pos_z   = seg_z["dl_current_x_mean"].values
    bc_z    = seg_z[vocab].values
    total_z = bc_z.sum(axis=1)

    X_z = seg_z[PHYSICAL_COLS].values
    Y_true_z = bc_z
    Y_pred_z = rf.predict(X_z)

    # Inject at the center window
    c_idx = inject_idx - s_idx
    swingup_rows = seg_df[states == 0]
    if len(swingup_rows) > 0:
        swap_vec = swingup_rows[vocab].sample(1, random_state=42).values[0]
    else:
        swap_vec = np.zeros(len(vocab))
    Y_anom_z = Y_true_z.copy()
    if c_idx < len(Y_anom_z):
        Y_anom_z[c_idx] = swap_vec

    cons_z      = np.linalg.norm(Y_true_z - Y_pred_z, axis=1)
    cons_anom_z = np.linalg.norm(Y_anom_z - Y_pred_z, axis=1)

    fn_labels = compute_branch_labels(vocab, fn_names)
    lqr_cols     = [v for v, l in fn_labels.items() if "lqr"     in l.lower()]
    standup_cols = [v for v, l in fn_labels.items() if "standup" in l.lower() or "swing" in l.lower()]
    lqr_z     = seg_z[lqr_cols].sum(axis=1).values     if lqr_cols     else np.zeros(len(seg_z))
    standup_z = seg_z[standup_cols].sum(axis=1).values if standup_cols else np.zeros(len(seg_z))

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True,
                             gridspec_kw={"hspace": 0.05, "height_ratios": [1.2, 1, 1, 1]})

    for i, ax in enumerate(axes):
        ax.axvline(0, color="#6A1B9A", lw=2, ls="--", alpha=0.8,
                   label="Anomaly injected" if i == 0 else None)
        add_state_bands(ax, t_z, states_z)

    axes[0].plot(t_z, np.degrees(angle_z), color="#1565C0", lw=1.5, label="Angle (°)")
    ax0r = axes[0].twinx()
    ax0r.spines["top"].set_visible(False)
    ax0r.plot(t_z, pos_z * 100, color="#B71C1C", lw=1.2, ls="--", label="Position (cm)")
    axes[0].set_ylabel("Pendulum angle (°)")
    axes[0].set_title("Detection Example — ±500ms Around Injected Trace-Swap Anomaly",
                       fontsize=11, fontweight="bold")
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].fill_between(t_z, 0, total_z, color="#546E7A", alpha=0.6, label="Total branches")
    if lqr_cols:
        axes[1].fill_between(t_z, 0, lqr_z, color="#1565C0", alpha=0.7, label="LQR/balance")
    if standup_cols:
        axes[1].fill_between(t_z, 0, standup_z, color="#FF8C00", alpha=0.7, label="Standup")
    axes[1].set_ylabel("Branch\nexecutions / 10ms")
    axes[1].legend(loc="upper left", fontsize=8)

    axes[2].plot(t_z, cons_z, color="#78909C", lw=1.2, ls="--", label="Normal (no anomaly)")
    axes[2].plot(t_z, cons_anom_z, color="#37474F", lw=1.5, label="With injected anomaly")
    axes[2].axhline(p95, ls=":", color="#F57F17", lw=1.2, label=f"p95={p95:.1f}")
    axes[2].axhline(p99, ls=":", color="#B71C1C", lw=1.2, label=f"p99={p99:.1f}")
    above_99 = cons_anom_z >= p99
    axes[2].fill_between(t_z, p99, cons_anom_z, where=above_99, color="#EF9A9A", alpha=0.9)
    axes[2].set_ylabel("Consistency\nscore (L2)")
    axes[2].legend(loc="upper left", fontsize=8, ncol=2)

    # Panel 4: difference highlighting
    axes[3].fill_between(t_z, 0, cons_anom_z - cons_z,
                         color="#6A1B9A", alpha=0.7, label="Score increase from anomaly")
    axes[3].axhline(0, color="black", lw=0.8)
    axes[3].set_ylabel("Score\ndelta (L2)")
    axes[3].set_xlabel("Time relative to anomaly injection (s)")
    axes[3].legend(loc="upper left", fontsize=8)

    # Annotate the anomaly window in panel 3
    if c_idx < len(t_z):
        axes[2].annotate(
            f"Score: {cons_anom_z[c_idx]:.1f}\n(>{p99:.0f} = detected)",
            xy=(t_z[c_idx], cons_anom_z[c_idx]),
            xytext=(t_z[c_idx] + 0.15, cons_anom_z[c_idx] + 2),
            fontsize=8, color="#6A1B9A", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#6A1B9A"),
        )

    for ax in axes:
        ax.margins(x=0)

    plt.savefig(OUT / "fig_detection_example.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_detection_example.png")


def main():
    df = pd.read_parquet(EXP / "aligned_dataset.parquet")
    vi = json.loads((EXP / "vocab_info.json").read_text())
    vocab    = vi["vocab"]
    fn_names = vi.get("fn_names", {})

    # Load exp03 thresholds
    exp03 = json.loads((EXP / "anomaly_detection_results.json").read_text())
    p95 = exp03["normal_consistency"]["p95"]
    p99 = exp03["normal_consistency"]["p99"]

    # Pick the run with the most windows (best for finding transitions)
    run_sizes = df.groupby("run").size()
    best_run  = run_sizes.idxmax()
    df_run    = df[df["run"] == best_run].sort_values("win").reset_index(drop=True)
    print(f"Using run: {best_run} ({len(df_run):,} windows)")

    # Find segment with transitions
    seg_start, seg_end = find_good_segment(df_run, win_len=300)
    seg_df_all = df_run.iloc[seg_start:seg_end]
    transitions = (seg_df_all["dl_state"].values[:-1] != seg_df_all["dl_state"].values[1:]).sum()
    print(f"Segment [{seg_start}:{seg_end}] — {transitions} state transitions")

    # Train RF on ALL windows in this run (first 80%) to get a good model
    n80 = int(len(df_run) * 0.8)
    train = df_run.iloc[:n80]
    X_tr = train[PHYSICAL_COLS].values
    Y_tr = train[vocab].values
    print("Training RF...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=8, random_state=42)
    rf.fit(X_tr, Y_tr)
    # Quick R² check on test set
    test = df_run.iloc[n80:]
    r2 = r2_score(test[vocab].values, rf.predict(test[PHYSICAL_COLS].values),
                  multioutput="uniform_average")
    print(f"RF test R²: {r2:.4f}")

    inject_idx, inject_t, consistency_anom, t, seg_df, states = make_dashboard(
        df_run, (seg_start, seg_end), vocab, fn_names, rf, p95, p99
    )
    make_zoom_figure(t, seg_df, states, inject_idx, consistency_anom,
                     vocab, fn_names, rf, p95, p99)
    print("Done.")


if __name__ == "__main__":
    main()
