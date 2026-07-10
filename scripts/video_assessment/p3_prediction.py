"""
Phase 3: RF prediction ladder — trace prediction from video feature tiers.

Configurations A–L:
  A) physical_only               BASELINE
  B) tier0_flow                  raw motion alone
  C) tier1_cotracker             explicit kinematics alone
  D) tier2_dinov2_pca            spatial scene alone
  E) tier3_vjepa2_pca            temporal dynamics alone
  F) physical + tier0            does flow add to datalayer?
  G) physical + tier1            do trajectories add?
  H) physical + tier2            does DINOv2 add?
  I) physical + tier3            does V-JEPA 2 add?
  J) physical + tier1 + tier3    best combination?
  K) all_video_no_physical       can video replace datalayer?
  L) everything                  ceiling

All within BALANCE (dl_state == 1), 5-fold contiguous temporal block CV.
Same fold indices for all configurations (per session, then pooled).
RF: n_estimators=200, n_jobs=-1.

Phase 3.2: Residual analysis — video's unique contribution.
Phase 3.3: Per-branch R² delta.

Outputs:
  outputs/video_assessment/p3_prediction_results.json
  outputs/video_assessment/fig_02_prediction_ladder.png
  outputs/video_assessment/fig_03_per_branch_improvement.png
  outputs/video_assessment/fig_06_residual_analysis.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

ROOT      = Path(__file__).resolve().parents[2]
FEAT_DIR  = ROOT / "outputs/video_assessment"
OUT_DIR   = ROOT / "outputs/video_assessment"
ALIGNED_PARQUET = ROOT / "outputs/phase3/aligned_dataset_expanded.parquet"

N_ESTIMATORS          = 200
N_ESTIMATORS_RESIDUAL = 50     # leaner for residual phase — avoids OOM on 277k×100
N_FOLDS               = 5
N_JOBS                = -1
N_JOBS_RESIDUAL       = 8      # cap residual parallelism to control peak RAM
BALANCE_STATE = 1
RANDOM_STATE  = 42

CONFIG_NAMES = {
    "A": "physical_only",
    "B": "tier0_flow",
    "C": "tier1_cotracker",
    "D": "tier2_dinov2_pca",
    "E": "tier3_vjepa2_pca",
    "F": "physical+tier0",
    "G": "physical+tier1",
    "H": "physical+tier2",
    "I": "physical+tier3",
    "J": "physical+tier1+tier3",
    "K": "all_video_no_physical",
    "L": "everything",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_aligned_video_features(sessions: list) -> dict:
    """Load per-tier per-session aligned parquet files."""
    tier_data = {0: {}, 1: {}, 2: {}, 3: {}}
    for tier in [0, 1, 2, 3]:
        for run_id in sessions:
            path = OUT_DIR / f"aligned_tier{tier}_{run_id}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                tier_data[tier][run_id] = df
    return tier_data


def build_window_dataset(df_all: pd.DataFrame, tier_data: dict,
                         sessions: list, phys_cols: list,
                         branch_cols: list) -> pd.DataFrame:
    """
    Merge base dataset with aligned video features for all sessions.
    Returns combined DataFrame indexed by (run, win) with all feature columns.
    """
    parts = []
    for run_id in sessions:
        run_df = df_all[df_all["run"] == run_id].copy()
        for tier, run_feats in tier_data.items():
            if run_id in run_feats:
                feat_df = run_feats[run_id]
                # Merge on win column
                run_df = run_df.merge(feat_df, on="win", how="left")
        parts.append(run_df)

    combined = pd.concat(parts, ignore_index=True)
    # Restrict to BALANCE state
    combined = combined[combined["dl_state"] == BALANCE_STATE].copy()
    return combined


def get_fold_indices(n: int, n_folds: int = N_FOLDS) -> list:
    """Contiguous temporal block fold indices."""
    fold_size = n // n_folds
    folds = []
    for k in range(n_folds):
        test_start = k * fold_size
        test_end   = (k + 1) * fold_size if k < n_folds - 1 else n
        train_idx  = list(range(0, test_start)) + list(range(test_end, n))
        test_idx   = list(range(test_start, test_end))
        folds.append((train_idx, test_idx))
    return folds


def cv_r2(X: np.ndarray, Y: np.ndarray, n_folds: int = N_FOLDS) -> tuple:
    """5-fold temporal block CV; returns (mean_r2, std_r2, per_branch_r2_mean)."""
    n           = len(X)
    folds       = get_fold_indices(n, n_folds)
    fold_r2s    = []
    branch_r2s  = np.zeros((n_folds, Y.shape[1]))

    for k, (train_idx, test_idx) in enumerate(folds):
        if len(train_idx) < 10 or len(test_idx) < 5:
            continue
        X_tr, X_te = X[train_idx], X[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]

        # Fill NaN with training median
        col_median = np.nanmedian(X_tr, axis=0)
        for j in range(X_tr.shape[1]):
            X_tr[np.isnan(X_tr[:, j]), j] = col_median[j]
            X_te[np.isnan(X_te[:, j]), j] = col_median[j]

        rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, n_jobs=N_JOBS,
                                   random_state=RANDOM_STATE)
        rf.fit(X_tr, Y_tr)
        Y_pred = rf.predict(X_te)
        fold_r2 = float(r2_score(Y_te, Y_pred, multioutput="uniform_average"))
        fold_r2s.append(fold_r2)

        # Per-branch R²
        for b in range(Y.shape[1]):
            branch_r2s[k, b] = float(r2_score(Y_te[:, b], Y_pred[:, b]))

    mean_r2     = float(np.mean(fold_r2s)) if fold_r2s else float("nan")
    std_r2      = float(np.std(fold_r2s))  if fold_r2s else float("nan")
    per_branch  = branch_r2s[:len(fold_r2s)].mean(axis=0).tolist() if fold_r2s else []
    return mean_r2, std_r2, per_branch


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 3: Trace Prediction Ladder")
    print("=" * 60)

    # Load dataset
    df_all = pd.read_parquet(ALIGNED_PARQUET)
    print(f"Loaded: {df_all.shape[0]:,} rows × {df_all.shape[1]} cols")

    # Detect columns
    meta_dl   = {"dl_state", "dl_target_x", "dl_iteration"}
    phys_cols = [c for c in df_all.columns if c.startswith("dl_") and c not in meta_dl]
    branch_cols = [c for c in df_all.columns if ":" in c]
    print(f"Physical: {len(phys_cols)}, Branches: {len(branch_cols)}")

    # Load session list
    params_path = OUT_DIR / "dataset_params.json"
    if params_path.exists():
        params   = json.loads(params_path.read_text())
        sessions = params.get("full_video_sessions", [])
    else:
        # Use all sessions that have at least tier0 video features
        sessions = [p.stem.replace("aligned_tier0_", "") for p in
                    sorted(OUT_DIR.glob("aligned_tier0_*.parquet"))]
    print(f"Video sessions for analysis: {len(sessions)} — {sessions}")

    # Load aligned video features
    tier_data = load_aligned_video_features(sessions)
    for t in [0, 1, 2, 3]:
        n = len(tier_data[t])
        print(f"  Tier {t}: {n}/{len(sessions)} sessions loaded")

    # Build combined dataset
    combined = build_window_dataset(df_all, tier_data, sessions, phys_cols, branch_cols)
    print(f"\nCombined BALANCE dataset: {len(combined):,} rows")

    # Feature column sets per tier
    t0_cols = [c for c in combined.columns if c.startswith("t0_")]
    t1_cols = [c for c in combined.columns if c.startswith("t1_")]
    t2_cols = [c for c in combined.columns if c.startswith("t2_")]
    t3_cols = [c for c in combined.columns if c.startswith("t3_")]
    print(f"  Tier 0 cols: {len(t0_cols)}")
    print(f"  Tier 1 cols: {len(t1_cols)}")
    print(f"  Tier 2 cols: {len(t2_cols)}")
    print(f"  Tier 3 cols: {len(t3_cols)}")

    # Y: branch targets
    branch_avail = [c for c in branch_cols if c in combined.columns]
    Y = combined[branch_avail].fillna(0.0).values
    print(f"\nBranch targets: {len(branch_avail)}")

    # Feature set definitions
    feature_sets = {
        "A": phys_cols,
        "B": t0_cols,
        "C": t1_cols,
        "D": t2_cols,
        "E": t3_cols,
        "F": phys_cols + t0_cols,
        "G": phys_cols + t1_cols,
        "H": phys_cols + t2_cols,
        "I": phys_cols + t3_cols,
        "J": phys_cols + t1_cols + t3_cols,
        "K": t0_cols + t1_cols + t2_cols + t3_cols,
        "L": phys_cols + t0_cols + t1_cols + t2_cols + t3_cols,
    }

    # Run CV for each config
    results = {}
    print("\n=== RF 5-fold Temporal Block CV (within BALANCE) ===")
    print(f"{'Config':<4} {'Name':<28} {'Feats':>6} {'R² mean':>8} {'R² std':>8}")
    print("-" * 60)

    best_combined_name  = None
    best_combined_r2    = -np.inf
    baseline_r2         = None
    baseline_per_branch = None

    for cfg, feat_cols in feature_sets.items():
        avail = [c for c in feat_cols if c in combined.columns]
        if not avail:
            print(f"  {cfg}: no features available, skipping")
            results[cfg] = {"r2_mean": None, "r2_std": None, "n_features": 0}
            continue

        X = combined[avail].values

        mean_r2, std_r2, per_branch = cv_r2(X, Y)
        results[cfg] = {
            "config_name": CONFIG_NAMES[cfg],
            "feature_cols": avail,
            "n_features":   len(avail),
            "r2_mean":      round(mean_r2, 4),
            "r2_std":       round(std_r2,  4),
            "per_branch_r2": per_branch,
        }
        print(f"  {cfg:<4} {CONFIG_NAMES[cfg]:<28} {len(avail):>6} {mean_r2:>8.4f} ± {std_r2:.4f}")

        if cfg == "A":
            baseline_r2         = mean_r2
            baseline_per_branch = per_branch
        if "physical" in CONFIG_NAMES[cfg] and mean_r2 > best_combined_r2:
            best_combined_r2   = mean_r2
            best_combined_name = cfg

    # Save phase 3.1 results early so a crash in 3.2 doesn't lose them
    early_results = {
        "configs":          results,
        "residual_analysis": {},
        "per_branch_delta": {},
        "n_video_sessions": len(sessions),
        "n_balance_windows": len(combined),
        "n_branch_targets": len(branch_avail),
    }
    (OUT_DIR / "p3_prediction_results.json").write_text(json.dumps(early_results, indent=2))
    print("Saved p3_prediction_results.json (phase 3.1 checkpoint)")

    # Phase 3.2: Residual analysis
    print("\n=== Phase 3.2: Residual Analysis ===")
    residual_results = {}
    if baseline_r2 is not None and "A" in results and results["A"]["r2_mean"] is not None:
        X_phys = combined[[c for c in phys_cols if c in combined.columns]].fillna(0.0).values
        folds  = get_fold_indices(len(combined))

        residuals = np.zeros_like(Y, dtype=float)
        for train_idx, test_idx in folds:
            X_tr, X_te = X_phys[train_idx], X_phys[test_idx]
            Y_tr, Y_te = Y[train_idx], Y[test_idx]
            col_med = np.nanmedian(X_tr, axis=0)
            for j in range(X_tr.shape[1]):
                X_tr[np.isnan(X_tr[:, j]), j] = col_med[j]
                X_te[np.isnan(X_te[:, j]), j] = col_med[j]
            rf = RandomForestRegressor(n_estimators=N_ESTIMATORS_RESIDUAL,
                                       n_jobs=N_JOBS_RESIDUAL,
                                       random_state=RANDOM_STATE)
            rf.fit(X_tr, Y_tr)
            residuals[test_idx] = Y_te - rf.predict(X_te)
            del rf  # free immediately

        for tier, tier_cols in [(0, t0_cols), (1, t1_cols), (2, t2_cols), (3, t3_cols)]:
            avail = [c for c in tier_cols if c in combined.columns]
            if not avail:
                continue
            X_vid = combined[avail].fillna(0.0).values
            res_r2, res_std, _ = cv_r2(X_vid, residuals)
            residual_results[f"tier{tier}"] = {
                "residual_r2_mean": round(res_r2, 4),
                "residual_r2_std":  round(res_std, 4),
            }
            print(f"  Tier {tier} residual R²: {res_r2:.4f} ± {res_std:.4f}")

    # Phase 3.3: Per-branch analysis
    per_branch_delta = {}
    if (baseline_per_branch and best_combined_name and
            results.get(best_combined_name, {}).get("per_branch_r2")):
        best_pb  = results[best_combined_name]["per_branch_r2"]
        base_pb  = baseline_per_branch
        n        = min(len(best_pb), len(base_pb))
        deltas   = [best_pb[i] - base_pb[i] for i in range(n)]
        n_improved = sum(d > 0.05 for d in deltas)
        per_branch_delta = {
            "branch_cols":   branch_avail[:n],
            "baseline_r2":   base_pb[:n],
            "best_combo_r2": best_pb[:n],
            "delta":         deltas,
            "n_improved_0.05": n_improved,
        }
        print(f"\nBranches with delta R² > 0.05: {n_improved}/{n}")

    # Save results
    all_results = {
        "configs":          results,
        "residual_analysis": residual_results,
        "per_branch_delta": per_branch_delta,
        "n_video_sessions": len(sessions),
        "n_balance_windows": len(combined),
        "n_branch_targets": len(branch_avail),
    }
    (OUT_DIR / "p3_prediction_results.json").write_text(json.dumps(all_results, indent=2))
    print("\nSaved p3_prediction_results.json")

    # Figure 2: Prediction ladder
    cfg_list = [c for c in CONFIG_NAMES if c in results and results[c]["r2_mean"] is not None]
    r2_means = [results[c]["r2_mean"] for c in cfg_list]
    r2_stds  = [results[c]["r2_std"]  for c in cfg_list]
    labels   = [f"{c}: {CONFIG_NAMES[c]}" for c in cfg_list]

    colors = []
    for c in cfg_list:
        name = CONFIG_NAMES[c]
        if name == "physical_only":
            colors.append("#1565C0")
        elif "physical" in name:
            colors.append("#2E8B57")
        elif name.startswith("all_video"):
            colors.append("#8B0000")
        else:
            colors.append("#90CAF9")

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(cfg_list)), r2_means, color=colors, alpha=0.85)
    ax.errorbar(range(len(cfg_list)), r2_means, yerr=r2_stds,
                fmt="none", color="black", capsize=4, lw=1.5)
    if baseline_r2 is not None:
        ax.axhline(baseline_r2, color="#1565C0", ls="--", lw=1.5,
                   label=f"Physical only baseline ({baseline_r2:.3f})")
    ax.set_xticks(range(len(cfg_list)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Mean R² (5-fold CV, within BALANCE)")
    ax.set_title("Trace Prediction: Feature Tier Ladder (RF, n=200 trees)")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=min(0, min(r2_means) - 0.05) if r2_means else 0)
    # Annotate n_features
    for i, c in enumerate(cfg_list):
        n_f = results[c]["n_features"]
        ax.text(i, (r2_means[i] or 0) + (r2_stds[i] or 0) + 0.005,
                f"n={n_f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_02_prediction_ladder.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_02_prediction_ladder.png")

    # Figure 6: Residual analysis
    if residual_results:
        tiers   = list(residual_results.keys())
        res_r2s = [residual_results[t]["residual_r2_mean"] for t in tiers]
        res_std = [residual_results[t]["residual_r2_std"] for t in tiers]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(len(tiers)), res_r2s, color=["#90CAF9","#4FC3F7","#AB47BC","#FF7043"])
        ax.errorbar(range(len(tiers)), res_r2s, yerr=res_std,
                    fmt="none", color="black", capsize=4)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xticks(range(len(tiers)))
        ax.set_xticklabels(tiers)
        ax.set_ylabel("Residual R² (video predicting physical-RF residuals)")
        ax.set_title("Unique Video Contribution After Physical Features")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "fig_06_residual_analysis.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved fig_06_residual_analysis.png")

    # Figure 3: Per-branch improvement scatter
    if per_branch_delta and per_branch_delta.get("delta"):
        base_pb = per_branch_delta["baseline_r2"]
        best_pb = per_branch_delta["best_combo_r2"]
        deltas  = per_branch_delta["delta"]
        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(base_pb, best_pb, c=deltas, cmap="RdYlGn", alpha=0.7,
                        vmin=-0.1, vmax=0.2, s=30)
        lo = min(min(base_pb), min(best_pb))
        hi = max(max(base_pb), max(best_pb))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="No change")
        ax.axhline(0, color="grey", lw=0.3)
        ax.axvline(0, color="grey", lw=0.3)
        plt.colorbar(sc, ax=ax, label="Delta R² (combined - physical)")
        ax.set_xlabel(f"Physical-only R² (config A)")
        ax.set_ylabel(f"Best combined R² (config {best_combined_name})")
        ax.set_title("Per-branch R² improvement from video features")
        n_imp = per_branch_delta["n_improved_0.05"]
        ax.text(0.02, 0.97, f"Branches improved (Δ>0.05): {n_imp}/{len(deltas)}",
                transform=ax.transAxes, va="top", fontsize=9)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "fig_03_per_branch_improvement.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved fig_03_per_branch_improvement.png")

    print("\nPhase 3 complete.")


if __name__ == "__main__":
    main()
