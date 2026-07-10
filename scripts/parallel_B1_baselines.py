"""
parallel_B1_baselines.py — Baseline comparisons for NeurIPS reviewers.

Implements four baselines vs. the RF cross-modal approach:
  1. State-mean baseline (no ML)
  2. Ridge regression (linear)
  3. Lag-1 autoregression (no physics features)
  4. Physics-based rule threshold

Outputs:
  outputs/parallel/fig_baseline_comparison.png
  outputs/parallel/baseline_results.json
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


def temporal_cv_score(model_fn, X, Y, n_folds=5):
    """5-fold temporal CV. Returns mean R² and predictions on held-out blocks."""
    n = len(X)
    fold = n // n_folds
    r2s, preds, trues = [], [], []
    for k in range(n_folds):
        ts = k * fold
        te = (k + 1) * fold if k < n_folds - 1 else n
        tr = np.concatenate([np.arange(0, ts), np.arange(te, n)])
        if len(tr) < 10 or te - ts < 5:
            continue
        m = model_fn()
        m.fit(X[tr], Y[tr])
        p = m.predict(X[ts:te])
        r2s.append(max(-1.0, r2_score(Y[ts:te], p, multioutput="uniform_average")))
        preds.append(p)
        trues.append(Y[ts:te])
    return float(np.mean(r2s)) if r2s else 0.0, np.vstack(preds) if preds else Y[:0], np.vstack(trues) if trues else Y[:0]


def cosine_sim(A, B):
    """Mean cosine similarity between rows of A and B."""
    nA = np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
    nB = np.linalg.norm(B, axis=1, keepdims=True) + 1e-9
    return float(np.mean(np.sum((A / nA) * (B / nB), axis=1)))


def compute_auc(scores_normal, scores_anomaly):
    """ROC AUC: normal=0, anomaly=1."""
    labels = np.array([0] * len(scores_normal) + [1] * len(scores_anomaly))
    scores = np.concatenate([scores_normal, scores_anomaly])
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return 0.5


def inject_trace_swap_anomalies(df, vocab, rng, n=200):
    """Return test-set normal and anomaly indices (BALANCE↔SWINGUP swap)."""
    n80 = int(len(df) * 0.8)
    test = df.iloc[n80:].reset_index(drop=True)
    bal_idx = test[test["dl_state"] == 1].index.tolist()
    sw_rows  = df[df["dl_state"] == 0][vocab].values
    if len(bal_idx) < n or len(sw_rows) < 1:
        return test, np.array([]), np.array([])
    inject_at = rng.choice(bal_idx, n, replace=False)
    Y_test = test[vocab].values.copy()
    Y_anom = Y_test.copy()
    for idx in inject_at:
        Y_anom[idx] = sw_rows[rng.integers(len(sw_rows))]
    return test, Y_test, Y_anom, inject_at


def main():
    df = pd.read_parquet(EXP / "aligned_dataset.parquet")
    vi = json.loads((EXP / "vocab_info.json").read_text())
    vocab = vi["vocab"]

    exp03 = json.loads((EXP / "anomaly_detection_results.json").read_text())
    p95 = exp03["normal_consistency"]["p95"]

    rng = np.random.default_rng(42)

    n80 = int(len(df) * 0.8)
    train = df.iloc[:n80].reset_index(drop=True)
    test  = df.iloc[n80:].reset_index(drop=True)

    X_tr = train[PHYSICAL_COLS].values
    Y_tr = train[vocab].values
    X_te = test[PHYSICAL_COLS].values
    Y_te = test[vocab].values

    # Pre-compute anomaly injection for AUC
    sw_rows = df[df["dl_state"] == 0][vocab].values
    bal_idx_te = np.where(test["dl_state"].values == 1)[0]
    n_inject    = min(200, len(bal_idx_te))
    inject_at   = rng.choice(bal_idx_te, n_inject, replace=False)
    Y_te_anom   = Y_te.copy()
    for idx in inject_at:
        Y_te_anom[idx] = sw_rows[rng.integers(len(sw_rows))]

    results = {}

    # ── 1. State-mean baseline ─────────────────────────────────────────────────
    print("[1] State-mean baseline...")
    state_means = {}
    for s in [0, 1, 2]:
        mask = train["dl_state"].values == s
        state_means[s] = Y_tr[mask].mean(axis=0) if mask.any() else np.zeros(len(vocab))

    Y_pred_sm = np.vstack([state_means.get(s, state_means[1]) for s in test["dl_state"].values])
    r2_sm = float(r2_score(Y_te, Y_pred_sm, multioutput="uniform_average"))
    cs_sm = cosine_sim(Y_te, Y_pred_sm)
    scores_normal_sm = np.linalg.norm(Y_te - Y_pred_sm, axis=1)
    scores_anom_sm   = np.linalg.norm(Y_te_anom - Y_pred_sm, axis=1)
    auc_sm = compute_auc(scores_normal_sm, scores_anom_sm[inject_at])

    results["state_mean"] = {"r2": r2_sm, "cosine_sim": cs_sm, "auc": auc_sm,
                              "description": "Predict branch vector as mean for controller state (no ML)"}
    print(f"  R²={r2_sm:.4f}, AUC={auc_sm:.4f}")

    # ── 2. Ridge regression ────────────────────────────────────────────────────
    print("[2] Ridge regression (linear)...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr, Y_tr)
    Y_pred_ridge = ridge.predict(X_te)
    r2_ridge = float(r2_score(Y_te, Y_pred_ridge, multioutput="uniform_average"))
    cs_ridge = cosine_sim(Y_te, Y_pred_ridge)
    scores_normal_ridge = np.linalg.norm(Y_te - Y_pred_ridge, axis=1)
    scores_anom_ridge   = np.linalg.norm(Y_te_anom - Y_pred_ridge, axis=1)
    auc_ridge = compute_auc(scores_normal_ridge, scores_anom_ridge[inject_at])

    results["ridge_regression"] = {"r2": r2_ridge, "cosine_sim": cs_ridge, "auc": auc_ridge,
                                    "description": "Ridge regression (physical → branch counts)"}
    print(f"  R²={r2_ridge:.4f}, AUC={auc_ridge:.4f}")

    # ── 3. Lag-1 autoregressive baseline ──────────────────────────────────────
    print("[3] Lag-1 autoregression (no physics)...")
    # Predict window t from window t-1 (previous branch counts)
    Y_lag_tr = Y_tr[:-1]  # features: t-1 branch counts
    Y_tgt_tr = Y_tr[1:]   # target: t branch counts
    Y_lag_te = Y_te[:-1]
    Y_tgt_te = Y_te[1:]

    ar_ridge = Ridge(alpha=1.0)
    ar_ridge.fit(Y_lag_tr, Y_tgt_tr)
    Y_pred_ar = ar_ridge.predict(Y_lag_te)
    r2_ar = float(r2_score(Y_tgt_te, Y_pred_ar, multioutput="uniform_average"))
    cs_ar = cosine_sim(Y_tgt_te, Y_pred_ar)

    # AUC: shift the anomaly indices to account for the -1 lag
    scores_normal_ar = np.linalg.norm(Y_tgt_te - Y_pred_ar, axis=1)
    inject_at_lag    = inject_at[inject_at > 0] - 1  # shift indices for lag
    inject_at_lag    = inject_at_lag[inject_at_lag < len(Y_tgt_te)]
    scores_anom_ar   = scores_normal_ar.copy()
    for idx in inject_at_lag:
        swap_vec = sw_rows[rng.integers(len(sw_rows))]
        Y_tgt_anom_row = Y_tgt_te[idx:idx+1].copy()
        Y_tgt_anom_row[0] = swap_vec
        scores_anom_ar[idx] = np.linalg.norm(Y_tgt_anom_row - Y_pred_ar[idx:idx+1], axis=1)[0]
    auc_ar = compute_auc(scores_normal_ar, scores_anom_ar[inject_at_lag]) if len(inject_at_lag) else 0.5

    results["lag1_autoregression"] = {"r2": r2_ar, "cosine_sim": cs_ar, "auc": auc_ar,
                                       "description": "Predict branch vector from previous window (no physics)"}
    print(f"  R²={r2_ar:.4f}, AUC={auc_ar:.4f}")

    # ── 4. Physics-based rule threshold ───────────────────────────────────────
    print("[4] Physics-based rule threshold...")
    # Rules: if |angle| < 0.1 and |pos| < 0.15 → expect BALANCE trace
    #        if |angle| > 0.5 → expect SWINGUP trace
    #        if |pos| > 0.17  → expect RESET trace
    # Vote by comparing expected state vs actual dl_state
    angle = test["dl_angle_mean"].abs().values
    pos   = test["dl_current_x_mean"].abs().values
    actual_state = test["dl_state"].values

    predicted_state = np.ones(len(test), dtype=int)  # default: BALANCE
    predicted_state[angle > 0.5] = 0   # SWINGUP
    predicted_state[pos > 0.17]  = 2   # RESET

    rule_acc = float(np.mean(predicted_state == actual_state))
    # AUC: anomaly = state mismatch after injection (predicted_state != actual after swap)
    # For the trace-swap anomaly, injected windows will have BALANCE state but SWINGUP trace
    # The rule baseline can't detect this because it doesn't see the trace
    rule_scores_normal = (predicted_state != actual_state).astype(float)
    rule_scores_anom   = rule_scores_normal.copy()
    # Injected windows: the rule doesn't change score (it only looks at physical state)
    auc_rule = compute_auc(rule_scores_normal, rule_scores_anom[inject_at])

    results["physics_rule"] = {
        "r2": float("nan"),  # not applicable
        "cosine_sim": float("nan"),
        "auc": auc_rule,
        "state_accuracy": rule_acc,
        "description": "Rule-based: expected function from angle/position thresholds",
    }
    print(f"  State accuracy={rule_acc:.4f}, AUC={auc_rule:.4f}")

    # ── 5. RF (our method) ─────────────────────────────────────────────────────
    print("[5] RF cross-modal (ours)...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=8, random_state=42)
    rf.fit(X_tr, Y_tr)
    Y_pred_rf = rf.predict(X_te)
    r2_rf = float(r2_score(Y_te, Y_pred_rf, multioutput="uniform_average"))
    cs_rf = cosine_sim(Y_te, Y_pred_rf)
    scores_normal_rf = np.linalg.norm(Y_te - Y_pred_rf, axis=1)
    scores_anom_rf   = np.linalg.norm(Y_te_anom - Y_pred_rf, axis=1)
    auc_rf = compute_auc(scores_normal_rf, scores_anom_rf[inject_at])

    results["random_forest"] = {"r2": r2_rf, "cosine_sim": cs_rf, "auc": auc_rf,
                                 "description": "RF regression (physical → branch counts) — ours"}
    print(f"  R²={r2_rf:.4f}, AUC={auc_rf:.4f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    methods   = ["State Mean\n(no ML)", "Ridge\n(linear)", "Lag-1 AR\n(no physics)",
                  "Rule-based\n(physics)", "RF\n(ours)"]
    keys      = ["state_mean", "ridge_regression", "lag1_autoregression", "physics_rule", "random_forest"]
    colors    = ["#90CAF9", "#64B5F6", "#42A5F5", "#FFA726", "#1565C0"]

    r2_vals  = [results[k]["r2"] for k in keys]
    auc_vals = [results[k]["auc"] for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(methods))
    for ax, vals, title, ylabel in [
        (axes[0], r2_vals, "R² Score (Branch Count Prediction)", "R² (mean over 100 branches)"),
        (axes[1], auc_vals, "Anomaly Detection AUC (Trace Swap)", "AUC (ROC)"),
    ]:
        bars = ax.bar(x, vals, color=colors, edgecolor="black", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axhline(0.5, ls="--", color="gray", lw=0.8, label="Chance (AUC=0.5)" if "AUC" in title else None)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width()/2, max(0, v) + 0.01, f"{v:.3f}",
                        ha="center", fontsize=9, fontweight="bold" if b.get_facecolor() == colors[-1] else "normal")
        if "R²" in title:
            ax.set_ylim(-0.05, 1.05)
        else:
            ax.set_ylim(0.4, 1.05)
            ax.legend(fontsize=8)
        # Highlight our method
        bars[-1].set_edgecolor("#B71C1C")
        bars[-1].set_linewidth(2)

    plt.tight_layout()
    fig.savefig(OUT / "fig_baseline_comparison.png", bbox_inches="tight")
    plt.close(fig)

    (OUT / "baseline_results.json").write_text(json.dumps({
        "methods": results,
        "key_finding": (
            f"RF achieves R²={r2_rf:.3f} vs lag-1 AR R²={r2_ar:.3f}. "
            f"If lag-1 AR were close to RF, the model would mainly be doing temporal smoothing. "
            f"The {'large' if r2_rf - r2_ar > 0.1 else 'small'} gap (Δ={r2_rf - r2_ar:.3f}) "
            f"{'demonstrates' if r2_rf - r2_ar > 0.1 else 'raises concerns that'} the physical "
            f"features contribute meaningful predictive information beyond temporal autocorrelation."
        ),
    }, indent=2))

    print("\n── Baseline Summary ──")
    for k, m in zip(keys, methods):
        r = results[k]
        r2_str = f"R²={r['r2']:.3f}" if not np.isnan(r["r2"]) else "R²=N/A"
        print(f"  {m.replace(chr(10), ' '):20s}: {r2_str}, AUC={r['auc']:.3f}")
    print("\nSaved fig_baseline_comparison.png and baseline_results.json")
    print("Done.")


if __name__ == "__main__":
    main()
