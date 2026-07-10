"""
phase2_A_stratified_analysis.py — State-stratified R² decomposition.

Answers the critical question: "How much of R²=0.748 is within-state fine-grained
prediction vs. between-state classification?"

Methods:
  A1. State-mean baseline (R²_between) — free lunch from knowing the state
  A2. Per-state RF (R²_within_state) — temporal CV inside each state
  A3. FWL residualization — OLS demean on state dummies, then RF on residuals
  A4. Confound leakage check — can residualised features still predict state?
  A5. Visualizations

Outputs in outputs/phase2/:
  stratified_r2.csv
  fig_A1_r2_decomposition.png
  fig_A2_within_state_r2_histogram.png
  fig_A3_balance_state_detail.png
  fig_A4_r2_between_vs_within_scatter.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path("/home/simran/allspark-data-exploration/CPS-Debugger")
OUT  = ROOT / "outputs/phase2"
EXP  = ROOT / "outputs/experiments"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 12,
    "legend.fontsize": 9, "figure.dpi": 150, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

PHYSICAL_COLS = [
    "dl_current_x_mean", "dl_current_x_std", "dl_current_x_delta",
    "dl_angle_mean", "dl_angle_std", "dl_angle_delta",
    "dl_velocity_mean", "dl_ang_vel_mean", "dl_ang_vel_std",
]
STATE_NAMES = {0: "SWINGUP", 1: "BALANCE", 2: "RESET"}
STATE_COLORS = {0: "#e07b54", 1: "#4c9be8", 2: "#6abf69"}


def temporal_block_cv(X, Y, n_splits=5):
    """5-fold contiguous time-block CV; returns list of (train_idx, test_idx)."""
    n = len(X)
    fold_size = n // n_splits
    folds = []
    for k in range(n_splits):
        test_start = k * fold_size
        test_end   = (k + 1) * fold_size if k < n_splits - 1 else n
        test_idx   = np.arange(test_start, test_end)
        train_idx  = np.concatenate([np.arange(0, test_start),
                                     np.arange(test_end, n)])
        if len(train_idx) > 0 and len(test_idx) > 0:
            folds.append((train_idx, test_idx))
    return folds


def cv_r2_per_branch(X, Y, n_estimators=100, max_depth=12, n_splits=5):
    """Temporal block CV → per-branch R² array."""
    folds = temporal_block_cv(X, Y, n_splits)
    all_preds = np.full_like(Y, np.nan, dtype=float)
    for tr_idx, te_idx in folds:
        rf = RandomForestRegressor(
            n_estimators=n_estimators, n_jobs=-1,
            random_state=42, max_depth=max_depth, max_features="sqrt"
        )
        rf.fit(X[tr_idx], Y[tr_idx])
        all_preds[te_idx] = rf.predict(X[te_idx])
    r2s = []
    for j in range(Y.shape[1]):
        mask = ~np.isnan(all_preds[:, j])
        r2s.append(r2_score(Y[mask, j], all_preds[mask, j]))
    return np.array(r2s)


def func_category(label):
    """Classify branch label into a high-level function category."""
    lower = label.lower()
    if "memcpy" in lower:
        return "memcpy"
    if "memset" in lower:
        return "memset"
    if any(x in lower for x in ["pou_lqr", "lqr"]):
        return "LQR"
    if any(x in lower for x in ["pou_standup", "standup"]):
        return "standup"
    if any(x in lower for x in ["tick", "pou_main", "pou_general"]):
        return "tick/main"
    if any(x in lower for x in ["printf", "vfprintf", "fwritex", "writev",
                                  "stdio", "log_data"]):
        return "logging/IO"
    if any(x in lower for x in ["fmin", "fmax", "sin", "cos", "rem_pio2",
                                  "math_"]):
        return "math"
    if any(x in lower for x in ["turn_on", "turn_off"]):
        return "timer"
    return "other"


def main():
    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading aligned dataset...")
    df = pd.read_parquet(EXP / "aligned_dataset.parquet")
    with open(EXP / "vocab_info.json") as f:
        vocab_info = json.load(f)
    vocab  = vocab_info["vocab"]
    labels = vocab_info.get("labels", {})

    # Filter to valid windows (non-zero branch sum)
    branch_mat = df[vocab].values.astype(float)
    valid = branch_mat.sum(axis=1) > 0
    df   = df[valid].reset_index(drop=True)
    branch_mat = branch_mat[valid]

    X_all  = df[PHYSICAL_COLS].values.astype(float)
    states = df["dl_state"].values.astype(int)
    n_branches = len(vocab)
    print(f"Dataset: {len(df):,} windows, {n_branches} branches")
    print(f"State distribution: { {STATE_NAMES[s]: int((states==s).sum()) for s in [0,1,2]} }")

    # ── A1. State-mean baseline → R²_between ─────────────────────────────────
    print("\n[A1] Computing state-mean baseline (R²_between)...")
    state_means = {}
    for s in [0, 1, 2]:
        mask = states == s
        state_means[s] = branch_mat[mask].mean(axis=0) if mask.sum() > 0 else np.zeros(n_branches)

    y_pred_state = np.array([state_means[s] for s in states])
    r2_between = np.array([
        r2_score(branch_mat[:, j], y_pred_state[:, j]) for j in range(n_branches)
    ])
    print(f"  Mean R²_between = {r2_between.mean():.3f}, Median = {np.median(r2_between):.3f}")

    # ── A1b. Total R² (RF, all data, temporal CV) ────────────────────────────
    print("\n[A1b] Computing R²_total (RF temporal CV on all data)...")
    r2_total = cv_r2_per_branch(X_all, branch_mat, n_estimators=100, max_depth=12)
    print(f"  Mean R²_total = {r2_total.mean():.3f}")

    # ── A1c. R²_within = incremental beyond state means ──────────────────────
    # Use residualised formulation: train RF on within-state residuals
    # First, the formula-based estimate:
    r2_within_formula = np.where(
        r2_between < 0.999,
        1.0 - (1.0 - r2_total) / np.maximum(1.0 - r2_between, 1e-6),
        0.0
    )
    print(f"\n[A1c] R²_within (formula) = {r2_within_formula.mean():.3f}")
    print(f"  Frac R²_within > 0.1: {(r2_within_formula > 0.1).mean():.0%}")
    print(f"  Frac R²_within > 0.2: {(r2_within_formula > 0.2).mean():.0%}")
    print(f"  Frac R²_within > 0.3: {(r2_within_formula > 0.3).mean():.0%}")

    # ── A2. Per-state RF (R²_within_direct) ──────────────────────────────────
    print("\n[A2] Per-state RF evaluation...")
    r2_per_state = {}
    for s, sname in STATE_NAMES.items():
        mask = states == s
        n_s  = mask.sum()
        if n_s < 200:
            print(f"  {sname}: only {n_s} windows, skipping")
            r2_per_state[s] = np.zeros(n_branches)
            continue
        X_s  = X_all[mask]
        Y_s  = branch_mat[mask]
        # Temporal CV within state
        n_splits = 5 if n_s >= 500 else 3
        r2s = cv_r2_per_branch(X_s, Y_s, n_splits=n_splits,
                               n_estimators=80, max_depth=10)
        r2_per_state[s] = r2s
        print(f"  {sname} (n={n_s:,}): mean R² = {r2s.mean():.3f}, "
              f"frac>0.3 = {(r2s>0.3).mean():.0%}")

    r2_within_balance = r2_per_state[1]  # BALANCE is the key state

    # ── A3. FWL Residualisation ───────────────────────────────────────────────
    print("\n[A3] FWL residualisation (demean on state dummies)...")
    state_dummies = np.zeros((len(df), 3))
    for i, s in enumerate([0, 1, 2]):
        state_dummies[states == s, i] = 1.0

    # Residualise X (physical features)
    X_res = X_all.copy()
    for j in range(X_all.shape[1]):
        for s in [0, 1, 2]:
            mask = states == s
            if mask.sum() > 0:
                X_res[mask, j] -= X_all[mask, j].mean()

    # Residualise Y (branch counts)
    Y_res = branch_mat.copy()
    for j in range(n_branches):
        for s in [0, 1, 2]:
            mask = states == s
            if mask.sum() > 0:
                Y_res[mask, j] -= branch_mat[mask, j].mean()

    r2_fwl = cv_r2_per_branch(X_res, Y_res, n_estimators=80, max_depth=10)
    print(f"  FWL mean R² = {r2_fwl.mean():.3f}, frac>0.2 = {(r2_fwl>0.2).mean():.0%}")

    # ── A4. Confound leakage check ────────────────────────────────────────────
    print("\n[A4] Confound leakage check (can residualised X predict state?)...")
    kf = KFold(n_splits=5, shuffle=False)
    state_preds = np.full(len(states), -1, dtype=int)
    for tr, te in kf.split(X_res):
        clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
        clf.fit(X_res[tr], states[tr])
        state_preds[te] = clf.predict(X_res[te])
    leakage_acc = (state_preds == states).mean()
    chance_acc  = (states == 1).mean()  # predicting BALANCE always
    print(f"  Leakage accuracy = {leakage_acc:.3f} "
          f"(chance={chance_acc:.3f} by always predicting BALANCE)")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    rows = []
    for j, bkey in enumerate(vocab):
        name   = labels.get(bkey, bkey)
        cat    = func_category(name)
        rows.append({
            "branch_id":          bkey,
            "function_name":      name,
            "category":           cat,
            "r2_total":           float(r2_total[j]),
            "r2_between":         float(r2_between[j]),
            "r2_within_formula":  float(r2_within_formula[j]),
            "r2_within_balance":  float(r2_within_balance[j]),
            "r2_fwl":             float(r2_fwl[j]),
            "r2_swingup":         float(r2_per_state[0][j]),
            "r2_reset":           float(r2_per_state[2][j]),
        })
    csv_df = pd.DataFrame(rows)
    csv_df.to_csv(OUT / "stratified_r2.csv", index=False)
    print(f"\nSaved stratified_r2.csv ({len(rows)} branches)")

    # Also save summary stats
    summary = {
        "n_windows": int(len(df)),
        "n_branches": n_branches,
        "r2_total_mean": float(r2_total.mean()),
        "r2_between_mean": float(r2_between.mean()),
        "r2_within_formula_mean": float(r2_within_formula.mean()),
        "r2_within_balance_mean": float(r2_within_balance.mean()),
        "r2_fwl_mean": float(r2_fwl.mean()),
        "frac_within_formula_gt_01": float((r2_within_formula > 0.1).mean()),
        "frac_within_formula_gt_02": float((r2_within_formula > 0.2).mean()),
        "frac_within_formula_gt_03": float((r2_within_formula > 0.3).mean()),
        "frac_within_balance_gt_03": float((r2_within_balance > 0.3).mean()),
        "leakage_accuracy": float(leakage_acc),
        "chance_accuracy_balance": float(chance_acc),
    }
    with open(OUT / "stratified_r2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved stratified_r2_summary.json")

    # ── A5. Visualisations ────────────────────────────────────────────────────
    print("\nGenerating figures...")

    # fig_A1: Stacked R² bars (R²_between + R²_within), sorted by R²_within desc
    top_n = 35
    sort_idx  = np.argsort(r2_within_formula)[::-1][:top_n]
    top_names = [labels.get(vocab[i], vocab[i]).replace("→", "→\n") for i in sort_idx]
    r2_b_top  = r2_between[sort_idx]
    r2_w_top  = np.clip(r2_within_formula[sort_idx], 0, None)
    r2_neg    = np.clip(r2_within_formula[sort_idx], None, 0)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(top_n)
    ax.bar(x, r2_b_top, label="R²_between (state identity)", color="#aec6cf", width=0.8)
    ax.bar(x, r2_w_top, bottom=r2_b_top, label="R²_within (fine-grained)", color="#2e6da4", width=0.8)
    ax.bar(x, r2_neg,   bottom=r2_b_top + r2_neg, color="#e07b54", alpha=0.6,
           label="R²_within < 0 (overfit to state)", width=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(top_names, rotation=90, ha="center", fontsize=6.5)
    ax.set_ylabel("R²")
    ax.set_title(f"R² Decomposition: Between-State vs Within-State (top {top_n} branches by R²_within)")
    ax.axhline(r2_total.mean(), color="k", ls="--", lw=1.0, label=f"Mean R²_total = {r2_total.mean():.3f}")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.2, 1.05)
    plt.tight_layout()
    fig.savefig(OUT / "fig_A1_r2_decomposition.png")
    plt.close(fig)
    print("  fig_A1 saved")

    # fig_A2: Histogram of R²_within
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(r2_within_formula, bins=30, color="#2e6da4", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="k", lw=1.2, ls="--")
    n_gt02 = (r2_within_formula > 0.2).sum()
    n_gt03 = (r2_within_formula > 0.3).sum()
    ax.axvline(0.2, color="#e07b54", lw=1.5, ls="--", label=f"0.2 threshold (N={n_gt02})")
    ax.axvline(0.3, color="#6abf69", lw=1.5, ls="--", label=f"0.3 threshold (N={n_gt03})")
    ax.set_xlabel("R²_within (after removing state-mean baseline)")
    ax.set_ylabel("Number of branches")
    ax.set_title(f"Distribution of Within-State R² across {n_branches} branches\n"
                 f"Mean = {r2_within_formula.mean():.3f}, Median = {np.median(r2_within_formula):.3f}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT / "fig_A2_within_state_r2_histogram.png")
    plt.close(fig)
    print("  fig_A2 saved")

    # fig_A3: BALANCE-state-only R², top 25 branches
    top_n3 = 25
    sort_b  = np.argsort(r2_within_balance)[::-1][:top_n3]
    cats    = [func_category(labels.get(vocab[i], vocab[i])) for i in sort_b]
    cat_colors = {"memcpy": "#2e6da4", "memset": "#7bc8f6", "LQR": "#e07b54",
                  "standup": "#f4a261", "tick/main": "#6abf69",
                  "logging/IO": "#aec6cf", "math": "#c77dff",
                  "timer": "#f9c74f", "other": "#999"}
    colors_b3 = [cat_colors.get(c, "#999") for c in cats]
    names_b3  = [labels.get(vocab[i], vocab[i]) for i in sort_b]

    fig, ax = plt.subplots(figsize=(12, 5))
    x3 = np.arange(top_n3)
    ax.bar(x3, r2_within_balance[sort_b], color=colors_b3, width=0.8)
    ax.set_xticks(x3)
    ax.set_xticklabels(names_b3, rotation=90, ha="center", fontsize=7.5)
    ax.set_ylabel("R² (within BALANCE state only)")
    ax.set_title(f"Within-BALANCE Branch Prediction (top {top_n3}, n=130k windows)\n"
                 f"Mean R²_within_BALANCE = {r2_within_balance.mean():.3f}, "
                 f"frac>0.3 = {(r2_within_balance>0.3).mean():.0%}")
    patches = [mpatches.Patch(color=v, label=k) for k, v in cat_colors.items()
               if k in cats]
    ax.legend(handles=patches, fontsize=8, ncol=2)
    plt.tight_layout()
    fig.savefig(OUT / "fig_A3_balance_state_detail.png")
    plt.close(fig)
    print("  fig_A3 saved")

    # fig_A4: Scatter R²_between vs R²_within, colored by category
    fig, ax = plt.subplots(figsize=(7, 6))
    cats_all = [func_category(labels.get(b, b)) for b in vocab]
    for cat, color in cat_colors.items():
        idx = [i for i, c in enumerate(cats_all) if c == cat]
        if idx:
            ax.scatter(r2_between[idx], r2_within_formula[idx],
                       color=color, s=45, alpha=0.8, label=cat, zorder=3)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("R²_between (state-mean baseline)")
    ax.set_ylabel("R²_within (fine-grained, after state removal)")
    ax.set_title("Per-Branch R² Decomposition by Function Category")
    ax.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    fig.savefig(OUT / "fig_A4_r2_between_vs_within_scatter.png")
    plt.close(fig)
    print("  fig_A4 saved")

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("PHASE A SUMMARY")
    print("=" * 55)
    print(f"  R²_total (all features, temporal CV): {r2_total.mean():.3f}")
    print(f"  R²_between (state-mean baseline):     {r2_between.mean():.3f}")
    print(f"  R²_within (formula, incremental):     {r2_within_formula.mean():.3f}")
    print(f"  R²_within_BALANCE (direct, per-state):{r2_within_balance.mean():.3f}")
    print(f"  R²_FWL (residualised features+targets):{r2_fwl.mean():.3f}")
    frac_total = r2_between.mean() / max(r2_total.mean(), 1e-6)
    print(f"\n  → {frac_total:.0%} of mean R²_total is accounted for by state identity")
    print(f"  → {1-frac_total:.0%} is genuine within-state fine-grained signal")
    print(f"\n  Branches with R²_within > 0.1: {(r2_within_formula>0.1).sum()}")
    print(f"  Branches with R²_within > 0.2: {(r2_within_formula>0.2).sum()}")
    print(f"  Branches with R²_within > 0.3: {(r2_within_formula>0.3).sum()}")
    print(f"\n  Within-BALANCE R² > 0.3: {(r2_within_balance>0.3).sum()} branches")
    print(f"  FWL leakage check: residual accuracy = {leakage_acc:.3f} "
          f"(vs chance = {chance_acc:.3f})")
    print("=" * 55)
    print(f"\nAll outputs saved to {OUT}/")


if __name__ == "__main__":
    main()
