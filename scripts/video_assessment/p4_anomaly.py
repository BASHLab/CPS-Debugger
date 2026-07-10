"""
Phase 4: Cross-modal anomaly detection.

Implements 7 anomaly types (A1–A7) as data-level injections and
5 detection monitors:
  1. datalayer_only      — RF predicts traces from physical features
  2. video_only          — RF predicts traces from best video tier
  3. combined            — RF predicts traces from physical + video
  4. video_to_dl         — RF predicts datalayer from video (sensor fault detector)
  5. dl_to_video         — RF predicts video from datalayer (physical change detector)

Experiments:
  4.1  AUC-ROC per anomaly × monitor
  4.2  Cross-monitoring heatmap
  4.3  Detection latency for A1 (bias) and A3 (drift)

Outputs:
  outputs/video_assessment/p4_anomaly_results.json
  outputs/video_assessment/fig_04_cross_monitoring_matrix.png
  outputs/video_assessment/fig_05_anomaly_detection_bars.png
  outputs/video_assessment/fig_09_detection_latency.png
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

ROOT     = Path(__file__).resolve().parents[2]
OUT_DIR  = ROOT / "outputs/video_assessment"
ALIGNED_PARQUET = ROOT / "outputs/phase3/aligned_dataset_expanded.parquet"

N_ESTIMATORS  = 200
N_ANOMALY_WIN = 200      # windows injected per anomaly
TRAIN_FRAC    = 0.80
BALANCE_STATE = 1
RANDOM_STATE  = 42

# Consecutive windows above p99 threshold to declare alert
ALERT_CONSECUTIVE = 3


# ── Anomaly injection ─────────────────────────────────────────────────────────

def inject_anomaly(data: pd.DataFrame, anomaly_type: str, params: dict) -> pd.DataFrame:
    """Return a modified copy with the anomaly injected."""
    d = data.copy()
    start = params.get("start_window", len(d) // 2)

    if anomaly_type == "sensor_bias":
        col    = params["column"]
        offset = params["offset"]
        d.loc[d.index[start:], col] = d.loc[d.index[start:], col] + offset

    elif anomaly_type == "sensor_freeze":
        col   = params["column"]
        frozen = float(d.iloc[start][col])
        d.loc[d.index[start:], col] = frozen

    elif anomaly_type == "sensor_drift":
        col      = params["column"]
        rate     = params["rate_per_window"]
        n_after  = len(d) - start
        d.loc[d.index[start:], col] = (
            d.loc[d.index[start:], col].values + np.arange(n_after) * rate
        )

    elif anomaly_type == "timing_delay":
        trace_cols = params["trace_cols"]
        shift      = params["shift_windows"]
        d[trace_cols] = d[trace_cols].shift(shift)
        d.dropna(subset=trace_cols[:1], inplace=True)

    elif anomaly_type == "mode_confusion":
        trace_cols    = params["trace_cols"]
        donor_traces  = params["donor_traces"]
        n             = min(len(d) - start, len(donor_traces))
        d.loc[d.index[start:start+n], trace_cols] = donor_traces.values[:n]

    elif anomaly_type == "added_mass":
        # Scale angle and velocity to simulate heavier pendulum
        scale_factor = params.get("scale", 1.3)
        for col in params.get("columns", []):
            d.loc[d.index[start:], col] = d.loc[d.index[start:], col] * scale_factor

    elif anomaly_type == "actuator_degrade":
        # Reduce velocity response
        for col in params.get("columns", []):
            scale = params.get("scale", 0.7)
            d.loc[d.index[start:], col] = d.loc[d.index[start:], col] * scale

    return d


# ── Monitor building ──────────────────────────────────────────────────────────

def build_monitors(X_train: dict, Y_train_traces: np.ndarray,
                   Y_train_dl: np.ndarray, Y_train_vid: np.ndarray):
    """
    Train 5 monitors on normal data.
    Returns dict of {monitor_name: (rf, X_test_key, Y_test_key)}.
    """
    monitors = {}

    def train_rf(X, Y, name):
        # Fill NaN
        col_med = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            X[np.isnan(X[:, j]), j] = col_med[j]
        rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, n_jobs=-1,
                                   random_state=RANDOM_STATE)
        rf.fit(X, Y)
        return rf, col_med

    if X_train.get("phys") is not None:
        rf, med = train_rf(X_train["phys"].copy(), Y_train_traces, "dl_only")
        monitors["datalayer_only"] = {"rf": rf, "input_key": "phys", "target_key": "traces", "col_med": med}

    if X_train.get("vid") is not None:
        rf, med = train_rf(X_train["vid"].copy(), Y_train_traces, "vid_only")
        monitors["video_only"] = {"rf": rf, "input_key": "vid", "target_key": "traces", "col_med": med}

    if X_train.get("both") is not None:
        rf, med = train_rf(X_train["both"].copy(), Y_train_traces, "combined")
        monitors["combined"] = {"rf": rf, "input_key": "both", "target_key": "traces", "col_med": med}

    if X_train.get("vid") is not None and Y_train_dl is not None and Y_train_dl.shape[1] > 0:
        rf, med = train_rf(X_train["vid"].copy(), Y_train_dl, "vid_to_dl")
        monitors["video_to_dl"] = {"rf": rf, "input_key": "vid", "target_key": "dl", "col_med": med}

    if X_train.get("phys") is not None and Y_train_vid is not None and Y_train_vid.shape[1] > 0:
        rf, med = train_rf(X_train["phys"].copy(), Y_train_vid, "dl_to_vid")
        monitors["dl_to_video"] = {"rf": rf, "input_key": "phys", "target_key": "vid", "col_med": med}

    return monitors


def compute_anomaly_score(monitor: dict, X_test: dict, Y_test: dict) -> np.ndarray:
    """Return per-window L2 residual scores."""
    rf      = monitor["rf"]
    in_key  = monitor["input_key"]
    tgt_key = monitor["target_key"]
    med     = monitor["col_med"]

    X = X_test[in_key].copy()
    Y = Y_test[tgt_key]

    # Fill NaN with training median
    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = med[j]

    Y_pred  = rf.predict(X)
    residuals = np.linalg.norm(Y - Y_pred, axis=1)
    return residuals


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 4: Cross-Modal Anomaly Detection")
    print("=" * 60)

    # Load data
    df_all = pd.read_parquet(ALIGNED_PARQUET)
    meta_dl   = {"dl_state", "dl_target_x", "dl_iteration"}
    phys_cols = [c for c in df_all.columns if c.startswith("dl_") and c not in meta_dl]
    branch_cols = [c for c in df_all.columns if ":" in c]

    # Use a session with video for injection experiment
    params_path = OUT_DIR / "dataset_params.json"
    if params_path.exists():
        params   = json.loads(params_path.read_text())
        sessions = params.get("full_video_sessions", [])
    else:
        sessions = [p.stem.replace("aligned_tier0_", "") for p in
                    sorted(OUT_DIR.glob("aligned_tier0_*.parquet"))]

    if not sessions:
        print("No video sessions found. Run Phase 0 and Tiers first.")
        return

    # Use first session for anomaly experiments (largest one preferred)
    test_session = sessions[0]
    print(f"Test session: {test_session}")

    df_sess = df_all[(df_all["run"] == test_session) & (df_all["dl_state"] == BALANCE_STATE)].copy()
    df_sess = df_sess.sort_values("win").reset_index(drop=True)

    # Load best available video tier for this session
    vid_cols = []
    for tier in [3, 1, 2, 0]:
        aligned_path = OUT_DIR / f"aligned_tier{tier}_{test_session}.parquet"
        if aligned_path.exists():
            vid_df = pd.read_parquet(aligned_path)
            vid_df = vid_df.merge(df_sess[["win"]], on="win", how="right")
            tier_cols = [c for c in vid_df.columns if c != "win"]
            df_sess = df_sess.merge(vid_df, on="win", how="left")
            vid_cols = [c for c in tier_cols if c in df_sess.columns]
            print(f"Using Tier {tier} video features: {len(vid_cols)} cols")
            break

    # Feature matrices
    n = len(df_sess)
    n_train = int(n * TRAIN_FRAC)

    X_phys_all = df_sess[phys_cols].fillna(0.0).values
    X_vid_all  = df_sess[vid_cols].fillna(0.0).values   if vid_cols else None
    X_both_all = np.hstack([X_phys_all, X_vid_all])     if vid_cols else X_phys_all
    Y_traces   = df_sess[branch_cols].fillna(0.0).values
    Y_dl       = X_phys_all   # for cross-prediction: predict dl from video

    # Detect primary angle column for sensor fault injection
    angle_col  = "dl_angle_mean" if "dl_angle_mean" in df_sess.columns else phys_cols[0]
    cart_col   = "dl_current_x_mean" if "dl_current_x_mean" in df_sess.columns else phys_cols[0]
    print(f"Angle col: {angle_col}, Cart col: {cart_col}")

    # Build X_train, Y_train
    X_train = {
        "phys": X_phys_all[:n_train],
        "vid":  X_vid_all[:n_train] if X_vid_all is not None else None,
        "both": X_both_all[:n_train],
    }
    monitors = build_monitors(
        X_train,
        Y_train_traces=Y_traces[:n_train],
        Y_train_dl=X_phys_all[:n_train],
        Y_train_vid=X_vid_all[:n_train] if X_vid_all is not None else None,
    )
    print(f"Monitors trained: {list(monitors.keys())}")

    # Normal scores (p99 threshold)
    X_test_normal  = {"phys": X_phys_all[n_train:], "vid": X_vid_all[n_train:] if X_vid_all is not None else None, "both": X_both_all[n_train:]}
    Y_test_normal  = {"traces": Y_traces[n_train:], "dl": X_phys_all[n_train:], "vid": X_vid_all[n_train:] if X_vid_all is not None else None}
    p99_thresholds = {}
    for mname, monitor in monitors.items():
        if X_test_normal.get(monitor["input_key"]) is not None and Y_test_normal.get(monitor["target_key"]) is not None:
            scores = compute_anomaly_score(monitor, X_test_normal, Y_test_normal)
            p99_thresholds[mname] = float(np.percentile(scores, 99))

    # Define anomaly configurations
    N_INJ   = N_ANOMALY_WIN
    start_w = n_train + (n - n_train) // 4   # inject in second quarter of test set

    donor_run = sessions[1] if len(sessions) > 1 else sessions[0]
    donor_df  = df_all[(df_all["run"] == donor_run) & (df_all["dl_state"] == BALANCE_STATE)].sort_values("win")

    anomaly_configs = {
        "A1_sensor_bias": {
            "type":   "sensor_bias",
            "params": {"column": angle_col, "offset": 3.0, "start_window": start_w},
            "affects_video": False,
        },
        "A2_sensor_freeze": {
            "type":   "sensor_freeze",
            "params": {"column": angle_col, "start_window": start_w},
            "affects_video": False,
        },
        "A3_sensor_drift": {
            "type":   "sensor_drift",
            "params": {"column": angle_col, "rate_per_window": 0.001, "start_window": start_w},
            "affects_video": False,
        },
        "A4_added_mass": {
            "type":   "added_mass",
            "params": {"columns": [angle_col, "dl_ang_vel_mean"], "scale": 1.4, "start_window": start_w},
            "affects_video": True,
        },
        "A5_actuator_degrade": {
            "type":   "actuator_degrade",
            "params": {"columns": ["dl_velocity_mean", cart_col], "scale": 0.65, "start_window": start_w},
            "affects_video": True,
        },
        "A6_mode_confusion": {
            "type":   "mode_confusion",
            "params": {
                "trace_cols":  branch_cols[:50],
                "start_window": start_w,
                "donor_traces": donor_df[branch_cols[:50]].fillna(0.0).head(N_INJ + 50),
            },
            "affects_video": False,
        },
        "A7_timing_delay": {
            "type":   "timing_delay",
            "params": {"trace_cols": branch_cols, "shift_windows": 5},
            "affects_video": False,
        },
    }

    # Experiment 4.1: AUC-ROC per anomaly × monitor
    print("\n=== Experiment 4.1: AUC-ROC per anomaly × monitor ===")
    results = {}
    monitor_names = ["datalayer_only", "video_only", "combined", "video_to_dl", "dl_to_video"]

    for anom_name, anom_cfg in anomaly_configs.items():
        print(f"\n  {anom_name}...")
        df_inj    = inject_anomaly(df_sess, anom_cfg["type"], anom_cfg["params"])
        n_inj     = min(N_INJ, len(df_inj) - start_w)
        start_idx = start_w - df_sess.index[0] if hasattr(df_sess.index, '__len__') else start_w
        start_idx = min(start_idx, len(df_inj) - N_INJ - 10)

        # Injected windows: the N_INJ windows after start
        X_phys_inj = df_inj[phys_cols].fillna(0.0).values
        X_vid_inj  = df_inj[vid_cols].fillna(0.0).values if vid_cols else None
        X_both_inj = np.hstack([X_phys_inj, X_vid_inj]) if vid_cols else X_phys_inj
        Y_traces_inj = df_inj[branch_cols].fillna(0.0).values

        # Combine normal test + injected anomalous windows for AUC
        n_normal_test = n - n_train
        X_phys_combo = np.vstack([X_phys_all[n_train:], X_phys_inj[start_idx:start_idx+N_INJ]])
        Y_traces_combo = np.vstack([Y_traces[n_train:], Y_traces_inj[start_idx:start_idx+N_INJ]])
        labels = np.array([0] * n_normal_test + [1] * N_INJ)

        if vid_cols:
            X_vid_combo  = np.vstack([X_vid_all[n_train:], X_vid_inj[start_idx:start_idx+N_INJ]])
            X_both_combo = np.vstack([X_both_all[n_train:], X_both_inj[start_idx:start_idx+N_INJ]])
        else:
            X_vid_combo  = None
            X_both_combo = X_phys_combo

        X_test_combo = {"phys": X_phys_combo, "vid": X_vid_combo, "both": X_both_combo}
        Y_test_combo = {"traces": Y_traces_combo, "dl": X_phys_combo, "vid": X_vid_combo}

        anom_aucs = {}
        for mname in monitor_names:
            if mname not in monitors:
                anom_aucs[mname] = None
                continue
            monitor = monitors[mname]
            if X_test_combo.get(monitor["input_key"]) is None:
                anom_aucs[mname] = None
                continue
            try:
                scores = compute_anomaly_score(monitor, X_test_combo, Y_test_combo)
                auc    = float(roc_auc_score(labels, scores))
            except Exception as e:
                auc = None
            anom_aucs[mname] = round(auc, 4) if auc is not None else None
            if auc is not None:
                print(f"    {mname:<20}: AUC = {auc:.4f}")

        results[anom_name] = anom_aucs

    # Experiment 4.3: Detection latency (A1, A3)
    latency_results = {}
    for anom_key in ["A1_sensor_bias", "A3_sensor_drift"]:
        if anom_key not in anomaly_configs:
            continue
        anom_cfg   = anomaly_configs[anom_key]
        df_inj     = inject_anomaly(df_sess, anom_cfg["type"], anom_cfg["params"])
        start_idx  = start_w

        latency_per_monitor = {}
        for mname, monitor in monitors.items():
            X_phys_inj   = df_inj[phys_cols].fillna(0.0).values[start_idx:]
            X_vid_inj    = df_inj[vid_cols].fillna(0.0).values[start_idx:] if vid_cols else None
            X_both_inj   = np.hstack([X_phys_inj, X_vid_inj]) if vid_cols else X_phys_inj
            Y_traces_inj = df_inj[branch_cols].fillna(0.0).values[start_idx:]

            X_post = {"phys": X_phys_inj, "vid": X_vid_inj, "both": X_both_inj}
            Y_post = {"traces": Y_traces_inj, "dl": X_phys_inj, "vid": X_vid_inj}

            if X_post.get(monitor["input_key"]) is None:
                continue
            try:
                scores    = compute_anomaly_score(monitor, X_post, Y_post)
                threshold = p99_thresholds.get(mname, np.percentile(scores, 90))
                # Find first window where 3 consecutive scores exceed threshold
                above     = (scores > threshold).astype(int)
                latency   = None
                consec    = 0
                for w, v in enumerate(above):
                    if v:
                        consec += 1
                        if consec >= ALERT_CONSECUTIVE:
                            latency = w - ALERT_CONSECUTIVE + 1
                            break
                    else:
                        consec = 0
                latency_per_monitor[mname] = latency
            except Exception:
                pass

        latency_results[anom_key] = latency_per_monitor
        print(f"\n{anom_key} detection latency (windows after injection):")
        for m, lat in latency_per_monitor.items():
            print(f"  {m:<20}: {lat if lat is not None else 'not detected'} windows")

    # Save results
    all_results = {
        "auc_results":     results,
        "latency_results": latency_results,
        "p99_thresholds":  p99_thresholds,
        "anomaly_configs": {k: {kk: vv for kk, vv in v.items() if kk != "params"
                                or not isinstance(vv, (pd.DataFrame, np.ndarray))}
                            for k, v in anomaly_configs.items()},
        "n_anomaly_windows": N_ANOMALY_WIN,
        "test_session": test_session,
    }
    (OUT_DIR / "p4_anomaly_results.json").write_text(json.dumps(all_results, indent=2))
    print("\nSaved p4_anomaly_results.json")

    # Figure 4: Cross-monitoring heatmap
    anom_names = list(results.keys())
    mon_names  = monitor_names
    matrix     = np.full((len(anom_names), len(mon_names)), np.nan)
    for i, anom in enumerate(anom_names):
        for j, mon in enumerate(mon_names):
            v = results[anom].get(mon)
            if v is not None:
                matrix[i, j] = v

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(matrix, vmin=0.4, vmax=1.0, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(mon_names)))
    ax.set_xticklabels([m.replace("_", "\n") for m in mon_names], fontsize=9)
    ax.set_yticks(range(len(anom_names)))
    ax.set_yticklabels(anom_names, fontsize=9)
    plt.colorbar(im, ax=ax, label="AUC-ROC")
    ax.set_title("Cross-Monitoring Matrix: 7 Anomalies × 5 Monitors")
    for i in range(len(anom_names)):
        for j in range(len(mon_names)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if v < 0.6 or v > 0.9 else "black")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_04_cross_monitoring_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_04_cross_monitoring_matrix.png")

    # Figure 5: Grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 5))
    x      = np.arange(len(anom_names))
    w      = 0.15
    colors = ["#1565C0", "#FF7043", "#2E8B57", "#AB47BC", "#F4B400"]
    for j, (mon, color) in enumerate(zip(mon_names, colors)):
        vals = [results[a].get(mon) for a in anom_names]
        vals = [v if v is not None else 0.5 for v in vals]
        ax.bar(x + j * w, vals, w, label=mon.replace("_", " "), color=color, alpha=0.85)
    ax.axhline(0.5, color="grey", ls="--", lw=0.8, label="Chance (0.5)")
    ax.set_xticks(x + w * 2)
    ax.set_xticklabels([a.replace("_", "\n") for a in anom_names], fontsize=8)
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Anomaly Detection: AUC by Anomaly Type and Monitor")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_05_anomaly_detection_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_05_anomaly_detection_bars.png")

    # Figure 9: Detection latency
    if latency_results:
        fig, axes = plt.subplots(1, len(latency_results), figsize=(12, 4), sharey=False)
        if len(latency_results) == 1:
            axes = [axes]
        for ax, (anom_key, latencies) in zip(axes, latency_results.items()):
            mons = list(latencies.keys())
            lats = [latencies[m] if latencies[m] is not None else 500 for m in mons]
            ax.bar(range(len(mons)), lats, color="#4FC3F7")
            ax.set_xticks(range(len(mons)))
            ax.set_xticklabels([m.replace("_", "\n") for m in mons], fontsize=8)
            ax.set_ylabel("Windows to detection (10ms each)")
            ax.set_title(f"{anom_key}\n(500 = not detected)")
        fig.suptitle("Detection Latency Analysis", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "fig_09_detection_latency.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved fig_09_detection_latency.png")

    print("\nPhase 4 complete.")


if __name__ == "__main__":
    main()
