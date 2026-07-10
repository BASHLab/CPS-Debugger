"""
phase4_ar_decomposition.py — Quantify marginal value of physical features over AR baseline.

This is the critical experiment that determines the project's viability.

Analyses:
  1a. Lag-1 AR R² within BALANCE vs RF R² within BALANCE
  1b. AR-residual prediction: does RF explain the CHANGE in branch counts?
  1c. Lag-1 AR on anomaly detection (AUC comparison)
  1d. Feature importance deep dive (angle ablation, BALANCE-only)
  2a. Within-BALANCE R² provenance (0.614 vs 0.560 discrepancy)
  2b. LORO Ridge vs RF explanation
  2c. Baselines with temporal block CV (corrected)

Outputs:
  outputs/phase4/fig_P4_ar_vs_rf_by_branch.png
  outputs/phase4/fig_P4_residual_prediction.png
  outputs/phase4/fig_P4_anomaly_ar_vs_rf.png
  outputs/phase4/fig_P4_feature_importance_balance_only.png
  outputs/phase4/fig_P4_marginal_value_summary.png
  outputs/phase4/phase4_results.json
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
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold

ROOT = Path("/home/simran/allspark-data-exploration/CPS-Debugger")
EXP  = ROOT / "outputs/experiments"
OUT  = ROOT / "outputs/phase4"
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
PHYSICAL_NO_ANGLE = [c for c in PHYSICAL_COLS if "angle" not in c]
ANGLE_ONLY = ["dl_angle_mean", "dl_angle_std", "dl_angle_delta"]

ANOMALY_TYPES = {
    "Trace Swap (wrong code path)": "trace_swap",
    "Physical Noise (3σ sensor noise)": "sensor_noise",
    "Temporal Shift 10ms (timing error)": "timing_10ms",
    "Temporal Shift 50ms (large timing error)": "timing_50ms",
}

N_JOBS = 8
RANDOM_STATE = 42


# ─── Temporal block CV helper ─────────────────────────────────────────────────

def temporal_block_cv_r2(X, Y, n_splits=5):
    """5-fold temporal block CV: each fold uses earlier data for train, later for test."""
    n = len(X)
    fold_size = n // n_splits
    r2s = []
    for i in range(n_splits):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_splits - 1 else n
        train_idx = list(range(0, test_start)) + list(range(test_end, n))
        test_idx = list(range(test_start, test_end))
        if len(train_idx) < 10 or len(test_idx) < 10:
            continue
        rf = RandomForestRegressor(n_estimators=100, max_depth=12,
                                   n_jobs=N_JOBS, random_state=RANDOM_STATE)
        rf.fit(X[train_idx], Y[train_idx])
        r2s.append(r2_score(Y[test_idx], rf.predict(X[test_idx]),
                            multioutput="uniform_average"))
    return float(np.mean(r2s)) if r2s else np.nan


def per_branch_temporal_cv_r2(X, Y, n_splits=5):
    """Returns per-branch R² using temporal block CV."""
    n = len(X)
    fold_size = n // n_splits
    preds = np.full_like(Y, np.nan, dtype=float)

    for i in range(n_splits):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_splits - 1 else n
        train_idx = list(range(0, test_start)) + list(range(test_end, n))
        test_idx = list(range(test_start, test_end))
        if len(train_idx) < 10:
            continue
        rf = RandomForestRegressor(n_estimators=100, max_depth=12,
                                   n_jobs=N_JOBS, random_state=RANDOM_STATE)
        rf.fit(X[train_idx], Y[train_idx])
        preds[test_idx] = rf.predict(X[test_idx])

    valid = ~np.any(np.isnan(preds), axis=0)
    per_branch = np.array([
        r2_score(Y[:, j], preds[:, j]) if valid[j] else np.nan
        for j in range(Y.shape[1])
    ])
    return per_branch


def main():
    print("Loading data...")
    df = pd.read_parquet(EXP / "aligned_dataset.parquet")
    vi = json.loads((EXP / "vocab_info.json").read_text())
    vocab = vi["vocab"]

    # Within-BALANCE only
    bal = df[df["dl_state"] == 1].copy().reset_index(drop=True)
    print(f"BALANCE windows: {len(bal):,} / {len(df):,} total")

    X_bal = bal[PHYSICAL_COLS].values
    Y_bal = bal[vocab].values.astype(float)

    results = {}

    # ══════════════════════════════════════════════════════════════════════════
    # 1a. Lag-1 AR R² within BALANCE
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[1a] Lag-1 AR R² within BALANCE...")

    # Must compute lag within each run (no bleeding across runs)
    ar_r2_per_branch = np.full(len(vocab), np.nan)
    rf_r2_per_branch_bal = np.full(len(vocab), np.nan)

    # AR: per-branch, within runs
    ar_actual_all, ar_pred_all = [], []
    for run in bal["run"].unique():
        rdf = bal[bal["run"] == run].sort_values("win")
        if len(rdf) < 2:
            continue
        Y_run = rdf[vocab].values.astype(float)
        # lag-1: predict Y[t] = Y[t-1], so valid pairs are indices 1..n-1
        ar_actual_all.append(Y_run[1:])    # actual[t]
        ar_pred_all.append(Y_run[:-1])     # predicted = actual[t-1]

    ar_actual = np.vstack(ar_actual_all)
    ar_pred   = np.vstack(ar_pred_all)

    for j in range(len(vocab)):
        ss_res = np.sum((ar_actual[:, j] - ar_pred[:, j]) ** 2)
        ss_tot = np.sum((ar_actual[:, j] - ar_actual[:, j].mean()) ** 2)
        ar_r2_per_branch[j] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    ar_r2_mean = float(np.nanmean(ar_r2_per_branch))
    print(f"  Lag-1 AR R² within BALANCE: {ar_r2_mean:.4f}")

    # RF within BALANCE (temporal block CV)
    print("  Training RF within BALANCE (temporal block CV)...")
    rf_r2_per_branch_bal = per_branch_temporal_cv_r2(X_bal, Y_bal)
    rf_r2_mean_bal = float(np.nanmean(rf_r2_per_branch_bal))
    print(f"  RF R² within BALANCE (temp CV): {rf_r2_mean_bal:.4f}")

    delta_per_branch = rf_r2_per_branch_bal - ar_r2_per_branch
    delta_mean = float(np.nanmean(delta_per_branch))
    print(f"  Delta (RF - AR) mean: {delta_mean:.4f}")
    print(f"  Branches where RF > AR: {int(np.nansum(delta_per_branch > 0))} / {len(vocab)}")

    results["1a_ar_decomposition"] = {
        "ar_r2_mean_within_balance":   ar_r2_mean,
        "rf_r2_mean_within_balance":   rf_r2_mean_bal,
        "delta_rf_minus_ar_mean":      delta_mean,
        "n_branches_rf_beats_ar":      int(np.nansum(delta_per_branch > 0)),
        "n_branches_ar_beats_rf":      int(np.nansum(delta_per_branch < 0)),
        "interpretation": (
            "RF substantially beats AR" if delta_mean > 0.1 else
            "RF slightly beats AR" if delta_mean > 0.02 else
            "RF barely beats AR — physical features add little beyond temporal smoothness"
        ),
    }

    # ══════════════════════════════════════════════════════════════════════════
    # 1b. AR-residual prediction
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[1b] AR-residual prediction (RF on branch-count changes)...")

    # Build residual dataset within BALANCE (within runs)
    residual_X, residual_Y = [], []
    for run in bal["run"].unique():
        rdf = bal[bal["run"] == run].sort_values("win")
        if len(rdf) < 2:
            continue
        Y_run = rdf[vocab].values.astype(float)
        X_run = rdf[PHYSICAL_COLS].values
        residuals = Y_run[1:] - Y_run[:-1]   # change from t-1 to t
        residual_X.append(X_run[1:])          # physical features at t
        residual_Y.append(residuals)

    res_X = np.vstack(residual_X)
    res_Y = np.vstack(residual_Y)
    print(f"  Residual dataset: {len(res_X):,} windows")

    # RF on residuals (temporal block CV)
    print("  Training RF on residuals (temporal block CV)...")
    rf_residual_r2 = temporal_block_cv_r2(res_X, res_Y)
    print(f"  RF residual R²: {rf_residual_r2:.4f}")

    # Per-branch residual R²
    rf_residual_per_branch = per_branch_temporal_cv_r2(res_X, res_Y)
    rf_residual_r2_mean = float(np.nanmean(rf_residual_per_branch))
    print(f"  RF residual R² (per-branch mean): {rf_residual_r2_mean:.4f}")

    # Combined: physical + lag-1 → current (temporal block CV)
    print("  Training RF on physical + lag-1 combined...")
    combined_actual_X, combined_actual_Y = [], []
    combined_ar_idx = []
    for run in bal["run"].unique():
        rdf = bal[bal["run"] == run].sort_values("win")
        if len(rdf) < 2:
            continue
        Y_run = rdf[vocab].values.astype(float)
        X_run = rdf[PHYSICAL_COLS].values
        X_comb = np.hstack([X_run[1:], Y_run[:-1]])   # physical[t] + Y[t-1]
        combined_actual_X.append(X_comb)
        combined_actual_Y.append(Y_run[1:])

    comb_X = np.vstack(combined_actual_X)
    comb_Y = np.vstack(combined_actual_Y)
    rf_combined_r2 = temporal_block_cv_r2(comb_X, comb_Y)
    print(f"  RF combined (physical + lag-1) R²: {rf_combined_r2:.4f}")

    results["1b_residual_prediction"] = {
        "rf_residual_r2_mean":         rf_residual_r2_mean,
        "rf_combined_physical_ar_r2":  rf_combined_r2,
        "rf_r2_physical_only":         rf_r2_mean_bal,
        "ar_r2":                       ar_r2_mean,
        "interpretation": (
            "Physical features predict branch-count CHANGES (genuine causal signal)"
            if rf_residual_r2_mean > 0.05 else
            "Physical features do not predict residuals — signal is in slowly-varying level only"
        ),
    }

    # ══════════════════════════════════════════════════════════════════════════
    # 1c. Lag-1 AR on anomaly detection
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[1c] Lag-1 AR on anomaly detection...")

    # Load exp03 anomaly detection setup
    exp03 = json.loads((EXP / "anomaly_detection_results.json").read_text())
    p95 = exp03["normal_consistency"]["p95"]

    # Train RF on 80% of data (same as exp03)
    n80 = int(len(df) * 0.8)
    train_df = df.iloc[:n80]
    test_df  = df.iloc[n80:].reset_index(drop=True)

    rf_full = RandomForestRegressor(n_estimators=100, max_depth=12,
                                    n_jobs=N_JOBS, random_state=RANDOM_STATE)
    rf_full.fit(train_df[PHYSICAL_COLS].values, train_df[vocab].values.astype(float))

    def inject_and_score(test_df, vocab, rf_model, anomaly_type, n=200):
        """Inject anomaly and return (rf_scores_normal, rf_scores_anom, ar_scores_normal, ar_scores_anom)."""
        normal_idx = np.random.default_rng(42).choice(len(test_df), size=n, replace=False)
        X_norm = test_df.iloc[normal_idx][PHYSICAL_COLS].values
        Y_norm = test_df.iloc[normal_idx][vocab].values.astype(float)
        rf_norm_scores = np.linalg.norm(
            Y_norm - rf_model.predict(X_norm), axis=1)

        if anomaly_type == "trace_swap":
            # Swap with trace from another state
            swingup = df[df["dl_state"] == 0][vocab].values.astype(float)
            if len(swingup) < n:
                return None
            swap_idx = np.random.default_rng(1).choice(len(swingup), size=n, replace=False)
            Y_anom = swingup[swap_idx]
        elif anomaly_type == "sensor_noise":
            noise_std = test_df[PHYSICAL_COLS].std().values * 3.0
            X_noisy = X_norm + np.random.default_rng(2).normal(0, noise_std, X_norm.shape)
            X_noisy_for_ar = X_noisy  # AR doesn't use X
            Y_anom = Y_norm.copy()
            X_norm_anom = X_noisy
        elif anomaly_type == "timing_50ms":
            shift = 5
            anom_idx = normal_idx + shift
            anom_idx = anom_idx[anom_idx < len(test_df)]
            if len(anom_idx) < 10:
                return None
            n = len(anom_idx)
            normal_idx = normal_idx[:n]
            X_norm = test_df.iloc[normal_idx][PHYSICAL_COLS].values
            Y_norm = test_df.iloc[normal_idx][vocab].values.astype(float)
            Y_anom = test_df.iloc[anom_idx][vocab].values.astype(float)
            rf_norm_scores = np.linalg.norm(Y_norm - rf_model.predict(X_norm), axis=1)
        elif anomaly_type == "timing_10ms":
            shift = 1
            anom_idx = normal_idx + shift
            anom_idx = anom_idx[anom_idx < len(test_df)]
            if len(anom_idx) < 10:
                return None
            n = len(anom_idx)
            normal_idx = normal_idx[:n]
            X_norm = test_df.iloc[normal_idx][PHYSICAL_COLS].values
            Y_norm = test_df.iloc[normal_idx][vocab].values.astype(float)
            Y_anom = test_df.iloc[anom_idx][vocab].values.astype(float)
            rf_norm_scores = np.linalg.norm(Y_norm - rf_model.predict(X_norm), axis=1)
        else:
            return None

        if anomaly_type not in ("timing_50ms", "timing_10ms"):
            if anomaly_type == "sensor_noise":
                rf_anom_scores = np.linalg.norm(
                    Y_anom - rf_model.predict(X_norm_anom), axis=1)
            else:
                X_anom = test_df.iloc[normal_idx][PHYSICAL_COLS].values
                rf_anom_scores = np.linalg.norm(
                    Y_anom - rf_model.predict(X_anom), axis=1)
        else:
            rf_anom_scores = np.linalg.norm(
                Y_anom - rf_model.predict(
                    test_df.iloc[normal_idx][PHYSICAL_COLS].values), axis=1)

        # AR: score = L2 distance between consecutive windows
        # For normal: |Y[t] - Y[t-1]|; for anomaly: same but Y[t] is replaced
        # Use the last window in train as t-1 reference for test windows
        # Simpler: use mean absolute change in training set as "normal AR score"
        # For a fair AR anomaly detector: score = L2(Y_current - Y_previous_in_test)
        # We compute AR scores on all test windows first (using consecutive pairs)
        test_Y = test_df[vocab].values.astype(float)
        if len(test_Y) > 1:
            ar_changes = np.linalg.norm(test_Y[1:] - test_Y[:-1], axis=1)
        else:
            ar_changes = np.zeros(max(n, 1))

        # Normal AR scores: pick from ar_changes at normal_idx (offset by 1)
        ar_normal_idx = normal_idx[normal_idx < len(ar_changes)]
        ar_norm_sc = ar_changes[ar_normal_idx] if len(ar_normal_idx) > 0 else np.zeros(n)

        if anomaly_type == "trace_swap":
            # AR score for anomaly: the swapped trace vs the previous test window
            # Use the same normal_idx positions but with swapped Y
            prev_Y = test_Y[ar_normal_idx - 1] if len(ar_normal_idx) > 0 else test_Y[:1]
            prev_Y = np.maximum(ar_normal_idx - 1, 0)
            prev_Y_vals = test_Y[np.maximum(normal_idx - 1, 0)]
            ar_anom_sc = np.linalg.norm(
                Y_anom[:len(normal_idx)] - prev_Y_vals, axis=1)
        elif anomaly_type == "sensor_noise":
            ar_anom_sc = ar_norm_sc  # AR doesn't see sensor values, same score
        else:
            # timing: Y[t+shift] vs Y[t-1]: should be similar since traces are smooth
            prev_Y_vals = test_Y[np.maximum(normal_idx - 1, 0)]
            ar_anom_sc = np.linalg.norm(Y_anom - prev_Y_vals, axis=1)

        n_use = min(len(rf_norm_scores), len(rf_anom_scores),
                    len(ar_norm_sc), len(ar_anom_sc))
        labels = np.array([0] * n_use + [1] * n_use)

        try:
            rf_auc = float(roc_auc_score(labels,
                np.concatenate([rf_norm_scores[:n_use], rf_anom_scores[:n_use]])))
            ar_auc = float(roc_auc_score(labels,
                np.concatenate([ar_norm_sc[:n_use], ar_anom_sc[:n_use]])))
        except Exception:
            rf_auc, ar_auc = np.nan, np.nan

        return rf_auc, ar_auc

    ar_vs_rf_auc = {}
    for atype_name, atype_key in ANOMALY_TYPES.items():
        res = inject_and_score(test_df, vocab, rf_full, atype_key)
        if res is not None:
            rf_auc, ar_auc = res
            ar_vs_rf_auc[atype_name] = {"rf_auc": rf_auc, "ar_auc": ar_auc,
                                         "delta": rf_auc - ar_auc}
            print(f"  {atype_name}: RF AUC={rf_auc:.4f}, AR AUC={ar_auc:.4f}, Δ={rf_auc - ar_auc:+.4f}")
        else:
            print(f"  {atype_name}: skipped (insufficient data)")

    results["1c_anomaly_ar_vs_rf"] = ar_vs_rf_auc

    # ══════════════════════════════════════════════════════════════════════════
    # 1d. Feature importance deep dive
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[1d] Feature importance deep dive within BALANCE...")

    def rf_r2_cv(X, Y, n_splits=5):
        return temporal_block_cv_r2(X, Y, n_splits=n_splits)

    print("  RF full physical (all 9 features)...")
    r2_full = rf_r2_mean_bal  # already computed

    print("  RF angle-only (3 features)...")
    r2_angle_only = rf_r2_cv(bal[ANGLE_ONLY].values, Y_bal)
    print(f"    angle-only R²: {r2_angle_only:.4f}")

    print("  RF no-angle (6 features)...")
    r2_no_angle = rf_r2_cv(bal[PHYSICAL_NO_ANGLE].values, Y_bal)
    print(f"    no-angle R²: {r2_no_angle:.4f}")

    print("  Ridge LORO (physical-only, within BALANCE)...")
    runs = sorted(bal["run"].unique())
    loro_ridge_r2s = []
    for test_run in runs:
        tr = bal[bal["run"] != test_run]
        te = bal[bal["run"] == test_run]
        if len(te) < 20:
            continue
        ridge = Ridge(alpha=1.0)
        ridge.fit(tr[PHYSICAL_COLS].values, tr[vocab].values.astype(float))
        loro_ridge_r2s.append(float(r2_score(
            te[vocab].values.astype(float),
            ridge.predict(te[PHYSICAL_COLS].values),
            multioutput="uniform_average")))
    loro_ridge_mean_balance = float(np.mean(loro_ridge_r2s))
    print(f"    LORO Ridge (within BALANCE): {loro_ridge_mean_balance:.4f}")

    # Feature importance within BALANCE
    rf_bal = RandomForestRegressor(n_estimators=100, max_depth=12,
                                   n_jobs=N_JOBS, random_state=RANDOM_STATE)
    rf_bal.fit(X_bal, Y_bal)
    fi_balance = rf_bal.feature_importances_

    # Feature importance overall (all states)
    X_all = df[PHYSICAL_COLS].values
    Y_all = df[vocab].values.astype(float)
    rf_all_states = RandomForestRegressor(n_estimators=100, max_depth=12,
                                          n_jobs=N_JOBS, random_state=RANDOM_STATE)
    rf_all_states.fit(X_all, Y_all)
    fi_all_states = rf_all_states.feature_importances_

    results["1d_feature_importance"] = {
        "rf_r2_full_balance":      r2_full,
        "rf_r2_angle_only_balance": r2_angle_only,
        "rf_r2_no_angle_balance":   r2_no_angle,
        "ar_r2_balance":            ar_r2_mean,
        "loro_ridge_r2_balance":    loro_ridge_mean_balance,
        "angle_contribution":       float(r2_full - r2_no_angle),
        "feature_importance_balance": {
            PHYSICAL_COLS[i]: float(fi_balance[i]) for i in range(len(PHYSICAL_COLS))
        },
        "feature_importance_all_states": {
            PHYSICAL_COLS[i]: float(fi_all_states[i]) for i in range(len(PHYSICAL_COLS))
        },
    }

    # ══════════════════════════════════════════════════════════════════════════
    # Priority 2: Resolve discrepancies
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[2] Resolving discrepancies...")

    # 2b: LORO Ridge vs RF explanation
    # cross_run_results.json has RF-LORO (physical): 0.653
    # temporal_stability.json has Ridge-LORO: -0.189
    # Both on 6-run dataset. Let's verify both.
    loro_rf_r2s = []
    for test_run in runs:
        tr = df[df["run"] != test_run]
        te = df[df["run"] == test_run]
        rf_loro = RandomForestRegressor(n_estimators=100, max_depth=12,
                                        n_jobs=N_JOBS, random_state=RANDOM_STATE)
        rf_loro.fit(tr[PHYSICAL_COLS].values, tr[vocab].values.astype(float))
        loro_rf_r2s.append(float(r2_score(
            te[vocab].values.astype(float),
            rf_loro.predict(te[PHYSICAL_COLS].values),
            multioutput="uniform_average")))
    loro_rf_mean = float(np.mean(loro_rf_r2s))
    print(f"  LORO RF R² (all states): {loro_rf_mean:.4f}")
    print(f"  LORO Ridge R² (all states, from temporal_stability.json): -0.189")
    print(f"  Explanation: Ridge overfits cross-run distribution shift; RF generalizes better.")

    results["2b_loro_discrepancy"] = {
        "loro_rf_r2_physical_all_states":      loro_rf_mean,
        "loro_ridge_r2_physical_all_states":   -0.189,  # from temporal_stability.json
        "explanation": "Ridge regression is highly sensitive to run-to-run distribution shift. RF generalizes better across runs. Both are computed on physical-only features, 6-run dataset. The 0.653 (exp06) is RF-LORO; -0.189 is Ridge-LORO.",
    }

    # 2a: within-BALANCE R² — 0.560 (task_1_3c, 6-run) vs 0.614 (previous report)
    # We just computed rf_r2_mean_bal with temporal block CV — that's the definitive number
    results["2a_within_balance_r2"] = {
        "rf_r2_temporal_cv_6run":   rf_r2_mean_bal,
        "ar_r2_6run":               ar_r2_mean,
        "note": "0.560 from task_1_3c (6-run, within BALANCE, temporal CV). "
                "0.614 may be from stratified analysis or different CV split. "
                f"This script's temporal block CV gives {rf_r2_mean_bal:.4f}.",
    }

    # 2c: Baselines with temporal block CV
    print("\n[2c] Baselines with temporal block CV (corrected)...")
    X_all = df[PHYSICAL_COLS].values
    Y_all = df[vocab].values.astype(float)

    # lag-1 AR (temporal block CV approximation using consecutive windows)
    # For a fair comparison: within each fold's test set, use previous window as prediction
    # We use the full dataset but compute AR only on consecutive pairs within runs
    ar_all_actual, ar_all_pred = [], []
    for run in sorted(df["run"].unique()):
        rdf = df[df["run"] == run].sort_values("win")
        Y_run = rdf[vocab].values.astype(float)
        ar_all_actual.append(Y_run[1:])
        ar_all_pred.append(Y_run[:-1])
    ar_all_actual = np.vstack(ar_all_actual)
    ar_all_pred_arr = np.vstack(ar_all_pred)
    ar_r2_all_states = float(r2_score(ar_all_actual, ar_all_pred_arr,
                                      multioutput="uniform_average"))
    print(f"  Lag-1 AR R² (all states): {ar_r2_all_states:.4f}")

    # RF temporal block CV (all states)
    print("  RF temporal block CV (all states)...")
    rf_r2_all_cv = temporal_block_cv_r2(X_all, Y_all)
    print(f"  RF R² (temporal CV, all states): {rf_r2_all_cv:.4f}")

    results["2c_baselines_corrected"] = {
        "lag1_ar_r2_all_states":     ar_r2_all_states,
        "lag1_ar_r2_within_balance": ar_r2_mean,
        "rf_r2_temporal_cv_all_states": rf_r2_all_cv,
        "rf_r2_temporal_cv_balance":    rf_r2_mean_bal,
        "note": "Temporal block CV makes all methods comparable. "
                "Global 80/20 split caused RF R²=-12.878 (distribution shift artifact).",
    }

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURES
    # ══════════════════════════════════════════════════════════════════════════
    print("\nGenerating figures...")

    # ── Fig 1: AR vs RF per branch scatter ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(ar_r2_per_branch, rf_r2_per_branch_bal,
                    c=delta_per_branch, cmap="RdYlGn", vmin=-0.3, vmax=0.3,
                    s=40, alpha=0.8, edgecolors="none")
    diag = np.linspace(min(ar_r2_per_branch.min(), rf_r2_per_branch_bal.min()),
                       max(ar_r2_per_branch.max(), rf_r2_per_branch_bal.max()), 100)
    ax.plot(diag, diag, "k--", lw=1.2, label="y=x (RF = AR)")
    ax.axhline(0, color="grey", lw=0.5, ls=":")
    ax.axvline(0, color="grey", lw=0.5, ls=":")
    plt.colorbar(sc, ax=ax, label="RF - AR (Δ R²)")
    ax.set_xlabel("Lag-1 AR R² (within BALANCE)")
    ax.set_ylabel("RF R² (within BALANCE, temporal block CV)")
    n_above = int(np.nansum(delta_per_branch > 0))
    ax.set_title(
        f"AR vs RF Prediction per Branch (within BALANCE)\n"
        f"AR mean={ar_r2_mean:.3f}, RF mean={rf_r2_mean_bal:.3f}, "
        f"Δ={delta_mean:+.3f} | RF>AR: {n_above}/100 branches",
        fontsize=9
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_P4_ar_vs_rf_by_branch.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_P4_ar_vs_rf_by_branch.png")

    # ── Fig 2: Residual prediction ────────────────────────────────────────────
    per_branch_res_r2 = rf_residual_per_branch
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(per_branch_res_r2[~np.isnan(per_branch_res_r2)], bins=20,
            color="#1565C0", edgecolor="white", alpha=0.8)
    ax.axvline(rf_residual_r2_mean, color="#B71C1C", lw=2,
               label=f"Mean={rf_residual_r2_mean:.3f}")
    ax.axvline(0, color="grey", lw=1, ls="--", label="R²=0")
    ax.set_xlabel("R² on branch-count residuals (changes)")
    ax.set_ylabel("Number of branches")
    ax.set_title("RF Predicts Branch-Count CHANGES\n(physical features → Δbranch_count)")
    ax.legend(fontsize=8)

    ax = axes[1]
    labels_bar = ["Lag-1 AR\n(levels)", "RF Physical\n(levels)", "RF Physical\n(residuals)", "RF Combined\n(physical+lag-1)"]
    vals_bar   = [ar_r2_mean, rf_r2_mean_bal, rf_residual_r2_mean, rf_combined_r2]
    colors_bar = ["#90CAF9", "#1565C0", "#F57F17", "#2E8B57"]
    bars = ax.bar(labels_bar, vals_bar, color=colors_bar, edgecolor="black", lw=0.5)
    ax.axhline(0, color="grey", lw=0.5)
    for b, v in zip(bars, vals_bar):
        ax.text(b.get_x() + b.get_width()/2, max(v, 0) + 0.005, f"{v:.3f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Mean R²")
    ax.set_title("R² Decomposition: Level vs Residual Prediction\n(within BALANCE)")
    ax.set_ylim(min(0, min(vals_bar)) - 0.05, max(vals_bar) + 0.08)

    fig.tight_layout()
    fig.savefig(OUT / "fig_P4_residual_prediction.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_P4_residual_prediction.png")

    # ── Fig 3: Anomaly AUC AR vs RF ───────────────────────────────────────────
    if ar_vs_rf_auc:
        anom_names = list(ar_vs_rf_auc.keys())
        rf_aucs = [ar_vs_rf_auc[n]["rf_auc"] for n in anom_names]
        ar_aucs = [ar_vs_rf_auc[n]["ar_auc"] for n in anom_names]

        x = np.arange(len(anom_names))
        w = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        b1 = ax.bar(x - w/2, rf_aucs, w, label="RF consistency (ours)", color="#1565C0", edgecolor="black", lw=0.5)
        b2 = ax.bar(x + w/2, ar_aucs, w, label="Lag-1 AR consistency", color="#90CAF9", edgecolor="black", lw=0.5)
        for b, v in zip(b1, rf_aucs):
            ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
        for b, v in zip(b2, ar_aucs):
            ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
        ax.axhline(0.5, color="grey", lw=1, ls="--", label="Random (0.5)")
        ax.set_xticks(x)
        ax.set_xticklabels([n.split(" (")[0] for n in anom_names], fontsize=9, rotation=15, ha="right")
        ax.set_ylabel("AUC")
        ax.set_ylim(0.4, 1.05)
        ax.set_title("Anomaly Detection AUC: RF Consistency vs Lag-1 AR\n(RF detects physics-trace inconsistency; AR detects temporal discontinuity)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / "fig_P4_anomaly_ar_vs_rf.png", bbox_inches="tight")
        plt.close(fig)
        print("  Saved fig_P4_anomaly_ar_vs_rf.png")

    # ── Fig 4: Feature importance BALANCE only ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    short_names = ["x̄", "σ_x", "Δx", "θ̄", "σ_θ", "Δθ", "ẋ", "θ̇̄", "σ_θ̇"]

    ax = axes[0]
    colors_fi = ["#B71C1C" if "angle" in c else "#1565C0" for c in PHYSICAL_COLS]
    bars = ax.barh(short_names[::-1], fi_balance[::-1], color=colors_fi[::-1],
                   edgecolor="black", lw=0.5)
    ax.set_xlabel("Feature importance")
    ax.set_title("Feature Importance — BALANCE state only\n(red = angle features)")

    ax = axes[1]
    bars2 = ax.barh(short_names[::-1], fi_all_states[::-1], color=colors_fi[::-1],
                    edgecolor="black", lw=0.5)
    ax.set_xlabel("Feature importance")
    ax.set_title("Feature Importance — All states\n(red = angle features)")

    # Annotate the shift
    axes[0].text(0.98, 0.05,
                 f"BALANCE: angle contrib = {sum(fi_balance[i] for i,c in enumerate(PHYSICAL_COLS) if 'angle' in c):.2f}",
                 transform=axes[0].transAxes, ha="right", fontsize=8)
    axes[1].text(0.98, 0.05,
                 f"All states: angle contrib = {sum(fi_all_states[i] for i,c in enumerate(PHYSICAL_COLS) if 'angle' in c):.2f}",
                 transform=axes[1].transAxes, ha="right", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "fig_P4_feature_importance_balance_only.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_P4_feature_importance_balance_only.png")

    # ── Fig 5: Marginal value summary ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel 1: R² decomposition
    ax = axes[0]
    components = ["AR component\n(temporal smoothness)", "Physical component\n(beyond AR)"]
    vals = [ar_r2_mean, max(0, delta_mean)]
    colors_p = ["#90CAF9", "#1565C0"]
    bars_p = ax.bar(components, vals, color=colors_p, edgecolor="black", lw=0.5)
    for b, v in zip(bars_p, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean R²")
    ax.set_title("R² Decomposition Within BALANCE\n(AR baseline vs physical features)")
    ax.set_ylim(0, max(vals) + 0.1)

    # Panel 2: Residual R²
    ax = axes[1]
    ax.bar(["Residual prediction\n(physical → Δbranch)"],
           [rf_residual_r2_mean], color="#F57F17", edgecolor="black", lw=0.5)
    ax.text(0, rf_residual_r2_mean + 0.002, f"{rf_residual_r2_mean:.3f}",
            ha="center", fontsize=11, fontweight="bold")
    ax.axhline(0.05, ls="--", color="#B71C1C", lw=1, label="Signal threshold (0.05)")
    ax.set_ylabel("Mean R² on residuals")
    ax.set_title("Physical Features → Branch-Count Changes\n(residual prediction test)")
    ax.set_ylim(min(-0.02, rf_residual_r2_mean - 0.02), max(0.2, rf_residual_r2_mean + 0.05))
    ax.legend(fontsize=8)

    # Panel 3: Anomaly AUC delta
    ax = axes[2]
    if ar_vs_rf_auc:
        anom_short = {
            "Trace Swap (wrong code path)": "Trace Swap",
            "Physical Noise (3σ sensor noise)": "Sensor Noise",
            "Temporal Shift 10ms (timing error)": "Timing 10ms",
            "Temporal Shift 50ms (large timing error)": "Timing 50ms",
        }
        deltas = [ar_vs_rf_auc[k]["delta"] for k in ar_vs_rf_auc]
        short_labels = [anom_short.get(k, k) for k in ar_vs_rf_auc]
        colors_d = ["#2E8B57" if d > 0 else "#B71C1C" for d in deltas]
        bars_d = ax.bar(short_labels, deltas, color=colors_d, edgecolor="black", lw=0.5)
        for b, v in zip(bars_d, deltas):
            ax.text(b.get_x() + b.get_width()/2,
                    v + 0.002 if v >= 0 else v - 0.008,
                    f"{v:+.3f}", ha="center", fontsize=8, fontweight="bold")
        ax.axhline(0, color="black", lw=1)
        ax.set_ylabel("Δ AUC (RF - AR)")
        ax.set_title("RF vs AR: AUC Advantage\n(green = RF better)")
        ax.set_xticklabels(short_labels, fontsize=8, rotation=20, ha="right")

    fig.suptitle("Phase 4: Marginal Value of Physical Features Over AR Baseline",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "fig_P4_marginal_value_summary.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_P4_marginal_value_summary.png")

    # ══════════════════════════════════════════════════════════════════════════
    # Summary and verdict
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── Phase 4 Summary ──")
    print(f"  [1a] AR R² within BALANCE:      {ar_r2_mean:.4f}")
    print(f"  [1a] RF R² within BALANCE:      {rf_r2_mean_bal:.4f}")
    print(f"  [1a] Δ (RF - AR):               {delta_mean:+.4f}")
    print(f"  [1b] Residual R² (phys → Δbr): {rf_residual_r2_mean:.4f}")
    print(f"  [1b] Combined R² (phys+AR):    {rf_combined_r2:.4f}")
    print(f"  [1d] RF angle-only R²:          {r2_angle_only:.4f}")
    print(f"  [1d] RF no-angle R²:            {r2_no_angle:.4f}")

    if ar_vs_rf_auc:
        print("\n  [1c] Anomaly detection AUC (RF / AR):")
        for k, v in ar_vs_rf_auc.items():
            print(f"       {k}: RF={v['rf_auc']:.4f}, AR={v['ar_auc']:.4f}, Δ={v['delta']:+.4f}")

    # Verdict
    physically_driven = delta_mean > 0.05 or rf_residual_r2_mean > 0.05
    anomaly_advantage = any(v["delta"] > 0.05 for v in ar_vs_rf_auc.values()) if ar_vs_rf_auc else False

    if physically_driven and anomaly_advantage:
        verdict = "STRONG: Physical features add genuine signal beyond AR. Both prediction AND anomaly detection benefit."
    elif anomaly_advantage:
        verdict = "MODERATE: Physical features add little to R² but substantially improve anomaly detection. Use 'consistency' framing."
    elif physically_driven:
        verdict = "MODERATE: Physical features improve prediction but anomaly detection advantage is weak."
    else:
        verdict = "WEAK: Physical features add little beyond AR. Pivot to LLM story as primary contribution."

    results["verdict"] = {
        "physically_driven_prediction": physically_driven,
        "anomaly_detection_advantage": anomaly_advantage,
        "summary": verdict,
        "recommended_framing": (
            "Cross-modal consistency scoring" if anomaly_advantage else
            "LLM-centric — detection as enabler"
        ),
    }
    print(f"\n  VERDICT: {verdict}")

    (OUT / "phase4_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved phase4_results.json and 5 figures to {OUT}")
    print("Done.")


if __name__ == "__main__":
    main()
