"""
Task 1: Video-Based State Estimation for Cross-Monitoring.

Estimates pendulum angle, cart position, angular velocity from CoTracker
trajectories via Ridge regression. Then injects sensor faults (bias, freeze,
drift) and measures detection rates via video-datalayer disagreement.

Outputs:
  outputs/video_assessment/task1_state_estimation_results.json
  outputs/video_assessment/fig_task1_state_estimation.png
  outputs/video_assessment/fig_task1_bias_detection.png
  outputs/video_assessment/fig_task1_drift_latency.png
  outputs/video_assessment/fig_task1_freeze_example.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error

ROOT      = Path(__file__).resolve().parents[2]
OUT_DIR   = ROOT / "outputs/video_assessment"
FEAT_DIR  = OUT_DIR / "features"
ALIGNED_PARQUET = ROOT / "outputs/phase3/aligned_dataset_expanded.parquet"
WIN_TO_MS = 10   # win * WIN_TO_MS = Unix epoch ms

BALANCE_STATE = 1


# ── helpers ───────────────────────────────────────────────────────────────────

def get_video_sessions():
    p = OUT_DIR / "dataset_params.json"
    if p.exists():
        d = json.loads(p.read_text())
        return d.get("full_video_sessions", d.get("video_sessions", []))
    raise FileNotFoundError("dataset_params.json not found")


def load_session_data(run_id: str, df_all: pd.DataFrame):
    """Return (aligned_ct, dl_row) for a session, both indexed on win."""
    al_path = OUT_DIR / f"aligned_tier1_{run_id}.parquet"
    if not al_path.exists():
        return None, None
    ct = pd.read_parquet(al_path).set_index("win")
    dl = df_all[df_all["run"] == run_id].sort_values("win").set_index("win")
    common = ct.index.intersection(dl.index)
    return ct.loc[common], dl.loc[common]


def detect_target_cols(dl: pd.DataFrame):
    angle_cols    = [c for c in dl.columns if "angle" in c and "mean" in c]
    position_cols = [c for c in dl.columns if ("current_x" in c or "position" in c) and "mean" in c]
    velocity_cols = [c for c in dl.columns if "ang_vel" in c and "mean" in c]
    return (
        angle_cols[0]    if angle_cols    else None,
        position_cols[0] if position_cols else None,
        velocity_cols[0] if velocity_cols else None,
    )


def ct_feature_cols(ct: pd.DataFrame):
    return [c for c in ct.columns if c.startswith("t1_")]


# ── Task 1a: state estimation accuracy ────────────────────────────────────────

def run_state_estimation(sessions, df_all):
    print("\n=== Task 1a: State Estimation from CoTracker ===")
    all_results = {}

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    row_map = {"angle": 0, "position": 1, "velocity": 2}

    for i, run_id in enumerate(sessions):
        ct, dl = load_session_data(run_id, df_all)
        if ct is None:
            continue

        bal_mask = (dl["dl_state"] == BALANCE_STATE) & np.all(np.isfinite(ct.values), axis=1)
        ct_b = ct[bal_mask]
        dl_b = dl[bal_mask]
        if len(ct_b) < 200:
            continue

        fcols = ct_feature_cols(ct_b)
        X = ct_b[fcols].values
        angle_col, pos_col, vel_col = detect_target_cols(dl_b)

        n_train = int(len(X) * 0.8)
        sess_results = {}

        for name, col in [("angle", angle_col), ("position", pos_col), ("velocity", vel_col)]:
            if col is None:
                continue
            y = dl_b[col].values
            Xtr, Xte = X[:n_train], X[n_train:]
            ytr, yte = y[:n_train], y[n_train:]
            mask_tr = np.isfinite(ytr) & np.all(np.isfinite(Xtr), axis=1)
            mask_te = np.isfinite(yte) & np.all(np.isfinite(Xte), axis=1)
            if mask_tr.sum() < 50:
                continue
            mdl = Ridge(alpha=1.0)
            mdl.fit(Xtr[mask_tr], ytr[mask_tr])
            pred = mdl.predict(Xte[mask_te])
            r2  = r2_score(yte[mask_te], pred)
            mae = mean_absolute_error(yte[mask_te], pred)
            sess_results[name] = {"r2": round(r2, 4), "mae": round(mae, 6),
                                   "col": col, "n_test": int(mask_te.sum())}
            print(f"  [{run_id}] {name}: R²={r2:.3f}  MAE={mae:.4f}")

            if i == 0 and name in row_map:
                ax = axes[row_map[name]][0]
                t_ax = np.arange(len(yte[mask_te]))
                ax.plot(t_ax[:3000], yte[mask_te][:3000], label="Sensor", lw=0.8)
                ax.plot(t_ax[:3000], pred[:3000], label="Video est.", lw=0.8, alpha=0.8)
                ax.set_title(f"{name} estimate vs sensor (session 1)")
                ax.set_ylabel(col)
                ax.legend(fontsize=7)

        all_results[run_id] = sess_results

    # Summary over sessions
    agg = {}
    for tgt in ["angle", "position", "velocity"]:
        r2s  = [all_results[s][tgt]["r2"]  for s in all_results if tgt in all_results[s]]
        maes = [all_results[s][tgt]["mae"] for s in all_results if tgt in all_results[s]]
        if r2s:
            agg[tgt] = {"r2_mean": round(float(np.mean(r2s)), 4),
                        "r2_std":  round(float(np.std(r2s)),  4),
                        "mae_mean": round(float(np.mean(maes)), 6),
                        "mae_std":  round(float(np.std(maes)), 6)}
            print(f"  MEAN {tgt}: R²={agg[tgt]['r2_mean']:.3f}±{agg[tgt]['r2_std']:.3f}  MAE={agg[tgt]['mae_mean']:.4f}")

    # Right column: MAE bar chart
    tgts   = list(agg.keys())
    mae_v  = [agg[t]["mae_mean"] for t in tgts]
    mae_e  = [agg[t]["mae_std"]  for t in tgts]
    for row_idx, t in enumerate(tgts):
        ax = axes[row_idx][1]
        sessions_r2 = [all_results[s][t]["r2"] for s in all_results if t in all_results[s]]
        ax.bar(range(len(sessions_r2)), sessions_r2, color="#4FC3F7")
        ax.axhline(np.mean(sessions_r2), color="navy", ls="--", lw=1.2)
        ax.set_title(f"{t} R² per session")
        ax.set_ylabel("R²")
        ax.set_ylim(-0.2, 1.0)

    plt.suptitle("Task 1a: CoTracker → Physical State Estimation", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig_task1_state_estimation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_task1_state_estimation.png")

    return all_results, agg


# ── Task 1b: sensor bias detection ────────────────────────────────────────────

def run_bias_detection(sessions, df_all, angle_mae_mean):
    print("\n=== Task 1b: Sensor Bias Detection ===")
    bias_levels_rad  = [3 * np.pi/180, 5 * np.pi/180, 10 * np.pi/180]
    bias_levels_deg  = [3, 5, 10]
    threshold_factor = 2.5   # alarm if disagreement > threshold_factor * MAE

    results = {}

    for bias_rad, bias_deg in zip(bias_levels_rad, bias_levels_deg):
        detections, latencies = [], []

        for run_id in sessions:
            ct, dl = load_session_data(run_id, df_all)
            if ct is None:
                continue

            bal_mask = (dl["dl_state"] == BALANCE_STATE) & np.all(np.isfinite(ct.values), axis=1)
            ct_b, dl_b = ct[bal_mask], dl[bal_mask]
            if len(ct_b) < 400:
                continue

            fcols = ct_feature_cols(ct_b)
            X = ct_b[fcols].values
            angle_col, _, _ = detect_target_cols(dl_b)
            if angle_col is None:
                continue
            y = dl_b[angle_col].values

            n_train = int(len(X) * 0.8)
            Xtr, Xte = X[:n_train], X[n_train:]
            ytr, yte = y[:n_train], y[n_train:]
            mask_tr = np.isfinite(ytr) & np.all(np.isfinite(Xtr), axis=1)
            mask_te = np.isfinite(yte) & np.all(np.isfinite(Xte), axis=1)
            if mask_tr.sum() < 50 or mask_te.sum() < 50:
                continue

            mdl = Ridge(alpha=1.0)
            mdl.fit(Xtr[mask_tr], ytr[mask_tr])

            # Inject bias into sensor reading (test portion)
            yte_biased = yte[mask_te] + bias_rad
            pred = mdl.predict(Xte[mask_te])  # video estimate unaffected

            disagreement = np.abs(pred - yte_biased)
            threshold = threshold_factor * angle_mae_mean
            alarm = disagreement > threshold

            det = alarm.mean()
            latency = int(np.argmax(alarm)) if alarm.any() else len(alarm)
            detections.append(det)
            latencies.append(latency)

        det_mean = float(np.mean(detections)) if detections else 0.0
        lat_mean = float(np.mean(latencies)) if latencies else np.nan
        results[f"bias_{bias_deg}deg"] = {
            "detection_rate": round(det_mean, 4),
            "mean_latency_windows": round(lat_mean, 1),
            "threshold_rad": round(float(threshold_factor * angle_mae_mean), 5),
        }
        print(f"  Bias +{bias_deg}°: detection={det_mean:.2%}  latency={lat_mean:.0f} wins")

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    labels = [f"+{b}°" for b in bias_levels_deg]
    det_r  = [results[f"bias_{b}deg"]["detection_rate"]        for b in bias_levels_deg]
    lat_v  = [results[f"bias_{b}deg"]["mean_latency_windows"]  for b in bias_levels_deg]
    ax1.bar(labels, det_r, color=["#90CAF9","#1E88E5","#0D47A1"])
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel("Detection rate")
    ax1.set_title("Bias detection rate vs magnitude")
    ax2.bar(labels, lat_v, color=["#A5D6A7","#43A047","#1B5E20"])
    ax2.set_ylabel("Windows to first alarm (10ms each)")
    ax2.set_title("Detection latency vs bias magnitude")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig_task1_bias_detection.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_task1_bias_detection.png")

    return results


# ── Task 1c: sensor freeze detection ──────────────────────────────────────────

def run_freeze_detection(sessions, df_all, angle_mae_mean):
    print("\n=== Task 1c: Sensor Freeze Detection ===")
    threshold = 2.5 * angle_mae_mean
    results = {}

    freeze_example = None

    for run_id in sessions:
        ct, dl = load_session_data(run_id, df_all)
        if ct is None:
            continue

        bal_mask = (dl["dl_state"] == BALANCE_STATE) & np.all(np.isfinite(ct.values), axis=1)
        ct_b, dl_b = ct[bal_mask], dl[bal_mask]
        if len(ct_b) < 600:
            continue

        fcols = ct_feature_cols(ct_b)
        X = ct_b[fcols].values
        angle_col, _, _ = detect_target_cols(dl_b)
        if angle_col is None:
            continue
        y = dl_b[angle_col].values

        n_train = int(len(X) * 0.8)
        Xtr, Xte = X[:n_train], X[n_train:]
        ytr, yte = y[:n_train], y[n_train:]
        mask_tr = np.isfinite(ytr) & np.all(np.isfinite(Xtr), axis=1)
        mask_te = np.isfinite(yte) & np.all(np.isfinite(Xte), axis=1)
        if mask_tr.sum() < 50 or mask_te.sum() < 100:
            continue

        mdl = Ridge(alpha=1.0)
        mdl.fit(Xtr[mask_tr], ytr[mask_tr])
        pred = mdl.predict(Xte[mask_te])
        yte_v = yte[mask_te]

        # Inject freeze: sensor reads constant value from freeze_at onward
        freeze_at = len(yte_v) // 3
        yte_frozen = yte_v.copy()
        yte_frozen[freeze_at:] = yte_v[freeze_at]

        disagreement = np.abs(pred - yte_frozen)
        alarm = disagreement > threshold
        latency = int(np.argmax(alarm[freeze_at:])) if alarm[freeze_at:].any() else -1

        results[run_id] = {"freeze_at": freeze_at, "latency_windows": latency}
        print(f"  [{run_id}] freeze at win {freeze_at}, detected at +{latency} windows")

        if freeze_example is None and latency > 0:
            freeze_example = (yte_v, yte_frozen, pred, disagreement, threshold, freeze_at, run_id)

    # Figure: freeze example
    if freeze_example:
        yte_v, yte_frozen, pred, disag, thr, fa, run_id = freeze_example
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        t = np.arange(len(yte_v))
        axes[0].plot(t, np.degrees(yte_v),    label="True sensor",  lw=0.8)
        axes[0].plot(t, np.degrees(yte_frozen),label="Frozen sensor", lw=0.8, ls="--")
        axes[0].plot(t, np.degrees(pred),      label="Video estimate",lw=0.8, alpha=0.8)
        axes[0].axvline(fa, color="red", ls=":", label="Freeze start")
        axes[0].set_ylabel("Angle (deg)")
        axes[0].legend(fontsize=7)
        axes[0].set_title(f"Sensor freeze example ({run_id})")
        axes[1].plot(t, np.degrees(disag), lw=0.8, color="purple", label="Disagreement")
        axes[1].axhline(np.degrees(thr), color="red", ls="--", label=f"Threshold ({np.degrees(thr):.2f}°)")
        axes[1].axvline(fa, color="red", ls=":")
        axes[1].set_ylabel("Disagreement (deg)")
        axes[1].set_xlabel("Window (10ms each)")
        axes[1].legend(fontsize=7)
        plt.tight_layout()
        fig.savefig(OUT_DIR / "fig_task1_freeze_example.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved fig_task1_freeze_example.png")

    lat_vals = [v["latency_windows"] for v in results.values() if v["latency_windows"] > 0]
    mean_lat = float(np.mean(lat_vals)) if lat_vals else np.nan
    results["summary"] = {"mean_latency_windows": round(mean_lat, 1)}
    return results


# ── Task 1d: sensor drift detection ───────────────────────────────────────────

def run_drift_detection(sessions, df_all, angle_mae_mean):
    print("\n=== Task 1d: Sensor Drift Detection ===")
    drift_rates_deg_s = [0.1, 0.5, 1.0]   # degrees per second
    window_rate_hz    = 100.0              # datalayer at 100 Hz
    threshold         = 2.5 * angle_mae_mean

    results = {}
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#66BB6A", "#FFA726", "#EF5350"]

    for dr_deg, col in zip(drift_rates_deg_s, colors):
        dr_rad_per_win = (dr_deg * np.pi / 180) / window_rate_hz
        latencies = []

        for run_id in sessions:
            ct, dl = load_session_data(run_id, df_all)
            if ct is None:
                continue

            bal_mask = (dl["dl_state"] == BALANCE_STATE) & np.all(np.isfinite(ct.values), axis=1)
            ct_b, dl_b = ct[bal_mask], dl[bal_mask]
            if len(ct_b) < 600:
                continue

            fcols = ct_feature_cols(ct_b)
            X = ct_b[fcols].values
            angle_col, _, _ = detect_target_cols(dl_b)
            if angle_col is None:
                continue
            y = dl_b[angle_col].values

            n_train = int(len(X) * 0.8)
            Xtr, Xte = X[:n_train], X[n_train:]
            ytr, yte = y[:n_train], y[n_train:]
            mask_tr = np.isfinite(ytr) & np.all(np.isfinite(Xtr), axis=1)
            mask_te = np.isfinite(yte) & np.all(np.isfinite(Xte), axis=1)
            if mask_tr.sum() < 50 or mask_te.sum() < 100:
                continue

            mdl = Ridge(alpha=1.0)
            mdl.fit(Xtr[mask_tr], ytr[mask_tr])
            pred = mdl.predict(Xte[mask_te])
            yte_v = yte[mask_te]

            # Inject drift from beginning of test window
            drift_start = len(yte_v) // 4
            yte_drifted = yte_v.copy()
            n_test = len(yte_v) - drift_start
            yte_drifted[drift_start:] += np.arange(n_test) * dr_rad_per_win

            disagreement = np.abs(pred - yte_drifted)
            alarm_after  = disagreement[drift_start:]
            latency = int(np.argmax(alarm_after > threshold)) if (alarm_after > threshold).any() else -1
            if latency >= 0:
                latencies.append(latency)

        mean_lat = float(np.mean(latencies)) if latencies else np.nan
        lat_s    = mean_lat / window_rate_hz if not np.isnan(mean_lat) else np.nan
        results[f"drift_{dr_deg}deg_s"] = {
            "mean_latency_windows": round(mean_lat, 1),
            "mean_latency_seconds": round(lat_s, 2),
        }
        print(f"  Drift {dr_deg}°/s: latency={mean_lat:.0f} wins ({lat_s:.1f}s)")
        ax.bar([f"{dr_deg}°/s"], [lat_s if not np.isnan(lat_s) else 0],
               color=col, label=f"{dr_deg}°/s → {lat_s:.1f}s" if not np.isnan(lat_s) else f"{dr_deg}°/s → undetected")

    ax.set_ylabel("Detection latency (seconds)")
    ax.set_title("Drift detection latency vs drift rate\n(video-datalayer cross-check)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig_task1_drift_latency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_task1_drift_latency.png")

    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Task 1: Video-Based State Estimation for Cross-Monitoring")
    print("=" * 60)

    df_all   = pd.read_parquet(ALIGNED_PARQUET)
    sessions = get_video_sessions()
    print(f"Sessions: {sessions}")

    est_per_session, est_agg = run_state_estimation(sessions, df_all)

    angle_mae = est_agg.get("angle", {}).get("mae_mean", 0.05)
    print(f"\nAngle estimation MAE (mean over sessions): {np.degrees(angle_mae):.2f}°")

    bias_results   = run_bias_detection(sessions, df_all, angle_mae)
    freeze_results = run_freeze_detection(sessions, df_all, angle_mae)
    drift_results  = run_drift_detection(sessions, df_all, angle_mae)

    all_results = {
        "state_estimation_per_session": est_per_session,
        "state_estimation_summary":     est_agg,
        "angle_estimation_mae_deg":     round(float(np.degrees(angle_mae)), 4),
        "sensor_bias_detection":        bias_results,
        "sensor_freeze_detection":      freeze_results,
        "sensor_drift_detection":       drift_results,
    }
    (OUT_DIR / "task1_state_estimation_results.json").write_text(json.dumps(all_results, indent=2))
    print("\nSaved task1_state_estimation_results.json")


if __name__ == "__main__":
    main()
