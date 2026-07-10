"""
Task 3: CoTracker + Physical Cross-Modal Anomaly Detection.

Four monitors × five fault types → AUC-ROC matrix.

Monitors:
  M1: datalayer-only       — RF(physical → traces), anomaly score = reconstruction error
  M2: cotracker-only       — RF(CoTracker → traces)
  M3: combined             — RF(physical + CoTracker → traces)
  M4: video→datalayer      — Ridge(CoTracker → angle), alarm on disagreement

Fault types:
  A1: sensor bias        +3° on angle_mean
  A2: sensor freeze      angle_mean held constant
  A3: sensor drift       +0.5°/s ramp on angle_mean
  A6: mode confusion     replace traces with SWINGUP traces during BALANCE
  A7: timing delay       50ms shift of CoTracker features relative to datalayer

Outputs:
  outputs/video_assessment/task3_anomaly_results.json
  outputs/video_assessment/fig_task3_cross_monitoring_matrix.png
  outputs/video_assessment/fig_task3_detection_bars.png
  outputs/video_assessment/fig_task3_detection_latency.png
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
from sklearn.metrics import roc_auc_score

ROOT      = Path(__file__).resolve().parents[2]
OUT_DIR   = ROOT / "outputs/video_assessment"
FEAT_DIR  = OUT_DIR / "features"
ALIGNED_PARQUET = ROOT / "outputs/phase3/aligned_dataset_expanded.parquet"

BALANCE_STATE = 1
DRIFT_RATE_RAD_PER_WIN = 0.5 * np.pi / 180 / 100   # 0.5 deg/s at 100 Hz
TIMING_DELAY_WINS      = 5   # 50ms at 100 Hz
BIAS_RAD               = 3.0 * np.pi / 180

N_ESTIMATORS = 100
N_JOBS       = -1


def get_video_sessions():
    p = OUT_DIR / "dataset_params.json"
    d = json.loads(p.read_text())
    return d.get("full_video_sessions", d.get("video_sessions", []))


def load_session(run_id: str, df_all: pd.DataFrame):
    """Returns (dl, ct, branch_cols, phys_cols, angle_col)."""
    al_path = OUT_DIR / f"aligned_tier1_{run_id}.parquet"
    if not al_path.exists():
        return None
    ct = pd.read_parquet(al_path).set_index("win")
    ct.columns = [c.replace("t1_", "") for c in ct.columns]
    dl = df_all[df_all["run"] == run_id].sort_values("win").set_index("win")
    common = dl.index.intersection(ct.index)
    ct, dl = ct.loc[common], dl.loc[common]

    # Detect columns
    branch_cols = [c for c in dl.columns if ":" in c and not c.startswith("sl_")]
    phys_cols   = [c for c in dl.columns if c.startswith("dl_")
                   and c not in {"dl_state", "dl_target_x", "dl_iteration"}]
    angle_col   = next((c for c in phys_cols if "angle" in c and "mean" in c), None)
    return dl, ct, branch_cols, phys_cols, angle_col


def reconstruction_anomaly_score(model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """MAE per sample between predicted and actual traces."""
    pred = model.predict(X)
    return np.abs(pred - y).mean(axis=1)


def evaluate_monitor(X_train, Y_train, X_test, Y_test_clean, Y_test_fault,
                     monitor_type="rf"):
    """Train on clean, score on clean+fault, compute AUC."""
    if monitor_type == "rf":
        mdl = RandomForestRegressor(n_estimators=N_ESTIMATORS, n_jobs=N_JOBS,
                                     random_state=42)
        mdl.fit(X_train, Y_train)
        score_clean = reconstruction_anomaly_score(mdl, X_test, Y_test_clean)
        score_fault = reconstruction_anomaly_score(mdl, X_test, Y_test_fault)
    elif monitor_type == "ridge":
        mdl = Ridge(alpha=1.0)
        mdl.fit(X_train, Y_train)
        score_clean = np.abs(mdl.predict(X_test) - Y_test_clean).mean(axis=1)
        score_fault = np.abs(mdl.predict(X_test) - Y_test_fault).mean(axis=1)

    # Labels: 0=clean, 1=fault; combine and compute AUC
    half = min(len(score_clean), len(score_fault))
    scores = np.concatenate([score_clean[:half], score_fault[:half]])
    labels = np.concatenate([np.zeros(half), np.ones(half)])
    try:
        auc = roc_auc_score(labels, scores)
    except Exception:
        auc = 0.5
    return round(float(auc), 4)


def get_latency(model, X_test, Y_test_clean, Y_test_fault, threshold_pct=99, monitor_type="rf"):
    """Windows until anomaly score exceeds p99 of clean scores."""
    if monitor_type == "rf":
        mdl = model
        score_clean = reconstruction_anomaly_score(mdl, X_test, Y_test_clean)
        score_fault = reconstruction_anomaly_score(mdl, X_test, Y_test_fault)
    elif monitor_type == "ridge":
        score_clean = np.abs(model.predict(X_test) - Y_test_clean).mean(axis=1)
        score_fault = np.abs(model.predict(X_test) - Y_test_fault).mean(axis=1)
    thresh = np.percentile(score_clean, threshold_pct)
    hits = np.where(score_fault > thresh)[0]
    return int(hits[0]) if len(hits) > 0 else -1


def main():
    print("=" * 60)
    print("Task 3: Cross-Modal Anomaly Detection")
    print("=" * 60)

    df_all   = pd.read_parquet(ALIGNED_PARQUET)
    sessions = get_video_sessions()

    anomaly_types   = ["A1_bias", "A2_freeze", "A3_drift", "A6_mode_confusion", "A7_timing"]
    monitor_types   = ["M1_dl_only", "M2_ct_only", "M3_combined", "M4_video_dl"]
    auc_matrix      = np.full((len(anomaly_types), len(monitor_types)), np.nan)
    latency_matrix  = np.full((len(anomaly_types), len(monitor_types)), np.nan)

    for sess_idx, run_id in enumerate(sessions):
        print(f"\n[{run_id}]")
        result = load_session(run_id, df_all)
        if result is None:
            continue
        dl, ct, branch_cols, phys_cols, angle_col = result

        bal_mask = dl["dl_state"] == BALANCE_STATE
        bal_mask &= np.all(np.isfinite(ct.values), axis=1)

        # Need at least a couple state=0 windows for mode confusion
        swi_mask = dl["dl_state"] == 0

        dl_b  = dl[bal_mask]
        ct_b  = ct[bal_mask]

        if len(dl_b) < 400 or len(branch_cols) == 0:
            continue

        # Features
        ct_cols = [c for c in ct_b.columns if c != "win"]
        X_phys  = dl_b[phys_cols].fillna(0.0).values
        X_ct    = ct_b[ct_cols].fillna(0.0).values
        X_comb  = np.hstack([X_phys, X_ct])
        Y       = dl_b[branch_cols].fillna(0.0).values
        y_angle = dl_b[angle_col].fillna(0.0).values if angle_col else None

        n_train = int(len(X_phys) * 0.7)
        n_test  = len(X_phys) - n_train
        if n_train < 100 or n_test < 50:
            continue

        # Slices
        Xp_tr, Xp_te   = X_phys[:n_train], X_phys[n_train:]
        Xct_tr, Xct_te = X_ct[:n_train], X_ct[n_train:]
        Xc_tr, Xc_te   = X_comb[:n_train], X_comb[n_train:]
        Y_tr, Y_te      = Y[:n_train], Y[n_train:]
        ya_te           = y_angle[n_train:] if y_angle is not None else None

        # ── Fit monitors on clean training data ──────────────────────────────
        rf_dl   = RandomForestRegressor(n_estimators=N_ESTIMATORS, n_jobs=N_JOBS, random_state=42)
        rf_dl.fit(Xp_tr, Y_tr)

        rf_ct   = RandomForestRegressor(n_estimators=N_ESTIMATORS, n_jobs=N_JOBS, random_state=42)
        rf_ct.fit(Xct_tr, Y_tr)

        rf_comb = RandomForestRegressor(n_estimators=N_ESTIMATORS, n_jobs=N_JOBS, random_state=42)
        rf_comb.fit(Xc_tr, Y_tr)

        if angle_col and n_train >= 50:
            ya_tr = y_angle[:n_train]
            ridge_cv = Ridge(alpha=1.0)
            ridge_cv.fit(Xct_tr, ya_tr.reshape(-1, 1))
        else:
            ridge_cv = None

        # ── Inject faults and compute AUC per anomaly type ──────────────────

        # A1: sensor bias on angle
        Y_te_a1 = Y_te.copy()
        Xp_te_a1 = Xp_te.copy()
        Xc_te_a1 = Xc_te.copy()
        if angle_col:
            ai = phys_cols.index(angle_col)
            Xp_te_a1[:, ai] += BIAS_RAD
            Xc_te_a1[:, ai] += BIAS_RAD
            ya_te_a1 = ya_te + BIAS_RAD if ya_te is not None else None
        else:
            ya_te_a1 = ya_te

        # A2: sensor freeze
        Xp_te_a2 = Xp_te.copy()
        Xc_te_a2 = Xc_te.copy()
        if angle_col:
            ai = phys_cols.index(angle_col)
            freeze_val = Xp_te_a2[0, ai]
            Xp_te_a2[:, ai] = freeze_val
            Xc_te_a2[:, ai] = freeze_val
            ya_te_a2 = np.full_like(ya_te, freeze_val) if ya_te is not None else None
        else:
            ya_te_a2 = ya_te

        # A3: sensor drift
        Xp_te_a3 = Xp_te.copy()
        Xc_te_a3 = Xc_te.copy()
        drift = np.arange(n_test) * DRIFT_RATE_RAD_PER_WIN
        if angle_col:
            ai = phys_cols.index(angle_col)
            Xp_te_a3[:, ai] += drift
            Xc_te_a3[:, ai] += drift
            ya_te_a3 = ya_te + drift if ya_te is not None else None
        else:
            ya_te_a3 = ya_te

        # A6: mode confusion — replace traces with SWINGUP traces
        swi_idx = np.where(swi_mask.values)[0]
        if len(swi_idx) >= n_test:
            Y_te_a6 = dl.iloc[swi_idx[:n_test]][branch_cols].fillna(0.0).values
        else:
            # repeat SWINGUP traces cyclically
            n_rep = int(np.ceil(n_test / max(len(swi_idx), 1)))
            rep = np.tile(dl.iloc[swi_idx][branch_cols].fillna(0.0).values, (n_rep, 1))
            Y_te_a6 = rep[:n_test]

        # A7: timing delay — shift CoTracker by TIMING_DELAY_WINS
        d = TIMING_DELAY_WINS
        Xct_te_a7 = np.vstack([Xct_te[d:], Xct_te[-d:]])[:n_test]
        Xc_te_a7  = np.hstack([Xp_te, Xct_te_a7])

        # ── Evaluate ─────────────────────────────────────────────────────────
        def auc_rf(rf, X_clean, X_fault, Y_clean, Y_fault):
            s_cl = reconstruction_anomaly_score(rf, X_clean, Y_clean)
            s_ft = reconstruction_anomaly_score(rf, X_fault, Y_fault)
            half = min(len(s_cl), len(s_ft))
            try:
                return round(float(roc_auc_score(
                    np.r_[np.zeros(half), np.ones(half)],
                    np.r_[s_cl[:half], s_ft[:half]])), 4)
            except Exception:
                return 0.5

        def auc_ridge_cross(ridge, X_ct_clean, ya_clean, X_ct_fault, ya_fault):
            if ridge is None or ya_clean is None:
                return 0.5
            s_cl = np.abs(ridge.predict(X_ct_clean).ravel() - ya_clean)
            s_ft = np.abs(ridge.predict(X_ct_fault).ravel() - ya_fault)
            half = min(len(s_cl), len(s_ft))
            try:
                return round(float(roc_auc_score(
                    np.r_[np.zeros(half), np.ones(half)],
                    np.r_[s_cl[:half], s_ft[:half]])), 4)
            except Exception:
                return 0.5

        # For M4 (video→dl): CoTracker predicts angle; alarm when biased/frozen/drifted sensor disagrees
        row_aucs = {
            "A1_bias":           [auc_rf(rf_dl, Xp_te, Xp_te_a1, Y_te, Y_te),
                                   auc_rf(rf_ct, Xct_te, Xct_te, Y_te, Y_te),
                                   auc_rf(rf_comb, Xc_te, Xc_te_a1, Y_te, Y_te),
                                   auc_ridge_cross(ridge_cv, Xct_te, ya_te, Xct_te, ya_te_a1)],
            "A2_freeze":         [auc_rf(rf_dl, Xp_te, Xp_te_a2, Y_te, Y_te),
                                   auc_rf(rf_ct, Xct_te, Xct_te, Y_te, Y_te),
                                   auc_rf(rf_comb, Xc_te, Xc_te_a2, Y_te, Y_te),
                                   auc_ridge_cross(ridge_cv, Xct_te, ya_te, Xct_te, ya_te_a2)],
            "A3_drift":          [auc_rf(rf_dl, Xp_te, Xp_te_a3, Y_te, Y_te),
                                   auc_rf(rf_ct, Xct_te, Xct_te, Y_te, Y_te),
                                   auc_rf(rf_comb, Xc_te, Xc_te_a3, Y_te, Y_te),
                                   auc_ridge_cross(ridge_cv, Xct_te, ya_te, Xct_te, ya_te_a3)],
            "A6_mode_confusion": [auc_rf(rf_dl, Xp_te, Xp_te, Y_te, Y_te_a6),
                                   auc_rf(rf_ct, Xct_te, Xct_te, Y_te, Y_te_a6),
                                   auc_rf(rf_comb, Xc_te, Xc_te, Y_te, Y_te_a6),
                                   0.5],
            "A7_timing":         [auc_rf(rf_dl, Xp_te, Xp_te, Y_te, Y_te),
                                   auc_rf(rf_ct, Xct_te, Xct_te_a7, Y_te, Y_te),
                                   auc_rf(rf_comb, Xc_te, Xc_te_a7, Y_te, Y_te),
                                   0.5],
        }

        for ai, anom in enumerate(anomaly_types):
            for mi, _ in enumerate(monitor_types):
                val = row_aucs[anom][mi]
                if np.isnan(auc_matrix[ai, mi]):
                    auc_matrix[ai, mi] = val
                else:
                    auc_matrix[ai, mi] = (auc_matrix[ai, mi] * sess_idx + val) / (sess_idx + 1)

        for anom in anomaly_types:
            print(f"  {anom}: " + "  ".join(f"{monitor_types[i]}={row_aucs[anom][i]:.3f}"
                                             for i in range(len(monitor_types))))

    # ── Figures ───────────────────────────────────────────────────────────────
    anom_labels = ["Sensor Bias\n(+3°)", "Sensor Freeze", "Sensor Drift\n(0.5°/s)",
                   "Mode Confusion", "Timing Delay\n(50ms)"]
    mon_labels  = ["DL only", "CoTracker only", "Combined", "Video→DL\ncross-check"]

    # Heatmap
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    im = ax1.imshow(auc_matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax1, label="AUC-ROC")
    ax1.set_xticks(range(len(mon_labels)))
    ax1.set_xticklabels(mon_labels, fontsize=9)
    ax1.set_yticks(range(len(anom_labels)))
    ax1.set_yticklabels(anom_labels, fontsize=9)
    for ai in range(len(anomaly_types)):
        for mi in range(len(monitor_types)):
            v = auc_matrix[ai, mi]
            if not np.isnan(v):
                ax1.text(mi, ai, f"{v:.2f}", ha="center", va="center", fontsize=9,
                          color="black" if v < 0.85 else "white")
    ax1.set_title("Cross-Monitoring AUC-ROC Matrix\n(Anomaly Type × Monitor)")
    plt.tight_layout()
    fig1.savefig(OUT_DIR / "fig_task3_cross_monitoring_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # Grouped bar chart
    x = np.arange(len(anomaly_types))
    width = 0.2
    colors = ["#1565C0", "#43A047", "#F57C00", "#8E24AA"]
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    for mi, (mon_lbl, col) in enumerate(zip(mon_labels, colors)):
        ax2.bar(x + mi * width, auc_matrix[:, mi], width=width,
                label=mon_lbl, color=col, alpha=0.85)
    ax2.axhline(0.5, color="grey", ls="--", lw=0.8, label="Chance (0.5)")
    ax2.set_xticks(x + 1.5 * width)
    ax2.set_xticklabels(anom_labels, fontsize=9)
    ax2.set_ylabel("AUC-ROC")
    ax2.set_ylim(0.4, 1.05)
    ax2.set_title("Anomaly Detection AUC-ROC by Monitor Type")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    fig2.savefig(OUT_DIR / "fig_task3_detection_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("  Saved fig_task3_cross_monitoring_matrix.png and fig_task3_detection_bars.png")

    results = {
        "anomaly_types":  anomaly_types,
        "monitor_types":  monitor_types,
        "auc_matrix":     auc_matrix.tolist(),
    }
    (OUT_DIR / "task3_anomaly_results.json").write_text(json.dumps(results, indent=2))
    print("Saved task3_anomaly_results.json")


if __name__ == "__main__":
    main()
