"""
Phase 5: Cross-session Leave-One-Session-Out (LOSO) evaluation.

For each of the 6 video sessions:
  - Train on remaining 5 sessions (video + physical features)
  - Test on held-out session
  - PCA fitted on training sessions only (no leakage)

Compares:
  - physical_only (config A) — baseline
  - best_combined (e.g. physical + tier0 or tier1) — from Phase 3 results

Paper framing decision:
  combined LOSO R² > physical LOSO R²  → video adds generalizable signal
  combined ≈ physical                  → video is redundant
  combined < physical                  → video overfits to session-specific visuals

Outputs:
  outputs/video_assessment/p5_loso_results.json
  outputs/video_assessment/fig_07_loso_generalization.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

ROOT      = Path(__file__).resolve().parents[2]
OUT_DIR   = ROOT / "outputs/video_assessment"
ALIGNED_PARQUET = ROOT / "outputs/phase3/aligned_dataset_expanded.parquet"

N_ESTIMATORS = 200
BALANCE_STATE = 1
RANDOM_STATE  = 42
PCA_DIMS      = 32
N_JOBS        = -1


def load_aligned_tier_for_session(tier: int, run_id: str) -> pd.DataFrame | None:
    """Load the per-window aligned features for a given tier and session."""
    path = OUT_DIR / f"aligned_tier{tier}_{run_id}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def fit_pca_on_train(train_raw_paths: list, pca_dims: int) -> tuple:
    """Fit PCA on training session raw NPZ files. Returns (pca, fit_ok)."""
    from sklearn.decomposition import PCA as SKLearnPCA
    all_feats = []
    for path in train_raw_paths:
        if not path.exists():
            continue
        data = np.load(path)
        key  = "features"
        if key in data:
            valid_mask = data.get("timestamp_ms", data.get("clip_center_ms", None))
            if valid_mask is not None:
                valid = valid_mask > 0
                all_feats.append(data[key][valid])
            else:
                all_feats.append(data[key])

    if not all_feats:
        return None, False
    X = np.vstack(all_feats)
    pca = SKLearnPCA(n_components=min(pca_dims, X.shape[1]), random_state=RANDOM_STATE)
    pca.fit(X)
    return pca, True


def project_with_pca(pca, raw_path: Path, run_id: str, tier: int,
                     ts_key: str = "timestamp_ms") -> pd.DataFrame | None:
    """Project a raw NPZ through the pre-fitted PCA."""
    if not raw_path.exists():
        return None
    data    = np.load(raw_path)
    feats   = data["features"]
    ts      = data[ts_key]
    proj    = pca.transform(feats)
    cols    = {f"t{tier}_pca_{i:02d}": proj[:, i] for i in range(proj.shape[1])}
    df      = pd.DataFrame({"timestamp_ms": ts, **cols})
    return df[df["timestamp_ms"] > 0]


def build_session_feature_matrix(run_id: str, df_all: pd.DataFrame,
                                  phys_cols: list, branch_cols: list,
                                  tiers_available: dict,
                                  pca_models: dict) -> tuple:
    """
    Build (X_phys, X_vid, Y) for a single session within BALANCE state.
    pca_models: {tier: pca_object} — PCA fitted on OTHER sessions.
    """
    run_df = df_all[(df_all["run"] == run_id) & (df_all["dl_state"] == BALANCE_STATE)].copy()
    run_df = run_df.sort_values("win").reset_index(drop=True)

    wins     = run_df["win"].values
    win_ms_a = wins * 10  # Unix epoch ms

    Y      = run_df[branch_cols].fillna(0.0).values
    X_phys = run_df[phys_cols].fillna(0.0).values

    # Build video feature matrix
    vid_parts = []

    for tier in [0, 1, 2, 3]:
        if tier not in tiers_available or run_id not in tiers_available[tier]:
            continue

        if tier in (0, 1):
            # Already aligned per-window
            feat_df   = load_aligned_tier_for_session(tier, run_id)
            if feat_df is None:
                continue
            feat_df   = feat_df.merge(pd.DataFrame({"win": wins}), on="win", how="right")
            feat_cols = [c for c in feat_df.columns if c != "win"]
            vid_parts.append(feat_df[feat_cols].fillna(0.0).values)

        elif tier in (2, 3):
            # Raw NPZ needs PCA projection
            pca = pca_models.get(tier)
            if pca is None:
                continue
            ts_key   = "timestamp_ms" if tier == 2 else "clip_center_ms"
            raw_path = OUT_DIR / f"features/tier{tier}_raw_{run_id}.npz"
            proj_df  = project_with_pca(pca, raw_path, run_id, tier, ts_key)
            if proj_df is None:
                continue

            # Align to win_ms_a
            from scipy.interpolate import interp1d
            ts_vid   = proj_df["timestamp_ms"].values
            feat_vid = proj_df[[c for c in proj_df.columns if c != "timestamp_ms"]].values
            aligned  = np.full((len(wins), feat_vid.shape[1]), np.nan)
            sort_idx = np.argsort(ts_vid)
            ts_s     = ts_vid[sort_idx]
            val_s    = feat_vid[sort_idx]
            for d in range(feat_vid.shape[1]):
                f = interp1d(ts_s, val_s[:, d], kind="nearest",
                             bounds_error=False, fill_value=np.nan)
                aligned[:, d] = f(win_ms_a)
            vid_parts.append(np.nan_to_num(aligned, nan=0.0))

    X_vid = np.hstack(vid_parts) if vid_parts else None
    return X_phys, X_vid, Y


def main():
    print("=" * 60)
    print("Phase 5: Cross-Session LOSO")
    print("=" * 60)

    df_all = pd.read_parquet(ALIGNED_PARQUET)
    meta_dl = {"dl_state", "dl_target_x", "dl_iteration"}
    phys_cols   = [c for c in df_all.columns if c.startswith("dl_") and c not in meta_dl]
    branch_cols = [c for c in df_all.columns if ":" in c]
    print(f"Physical: {len(phys_cols)}, Branches: {len(branch_cols)}")

    params_path = OUT_DIR / "dataset_params.json"
    if params_path.exists():
        params   = json.loads(params_path.read_text())
        sessions = params.get("full_video_sessions", [])
    else:
        sessions = [p.stem.replace("aligned_tier0_", "") for p in
                    sorted(OUT_DIR.glob("aligned_tier0_*.parquet"))]
    print(f"Video sessions: {sessions}")

    if len(sessions) < 2:
        print("Need at least 2 sessions for LOSO. Exiting.")
        return

    # Determine which tiers have data for at least half the sessions
    tiers_available = {}
    for tier in [0, 1, 2, 3]:
        present = {}
        for run in sessions:
            if tier in (0, 1):
                path = OUT_DIR / f"aligned_tier{tier}_{run}.parquet"
            else:
                path = OUT_DIR / f"features/tier{tier}_raw_{run}.npz"
            if path.exists():
                present[run] = path
        if len(present) >= 2:
            tiers_available[tier] = present
            print(f"  Tier {tier}: available for {len(present)}/{len(sessions)} sessions")

    # Load Phase 3 results to find best combined config
    p3_path = OUT_DIR / "p3_prediction_results.json"
    best_config_name = "physical+tier0"
    if p3_path.exists():
        p3 = json.loads(p3_path.read_text())
        best_r2   = -np.inf
        for cfg, v in p3.get("configs", {}).items():
            if v.get("r2_mean") and "physical" in v.get("config_name", ""):
                if v["r2_mean"] > best_r2:
                    best_r2          = v["r2_mean"]
                    best_config_name = v["config_name"]
        print(f"Best combined config from Phase 3: {best_config_name} (R²={best_r2:.4f})")

    # LOSO
    loso_results = {}
    loso_physical = []
    loso_combined = []

    print("\n=== LOSO Results ===")
    print(f"{'Session':<30} {'Phys R²':>10} {'Comb R²':>10} {'Delta':>8}")
    print("-" * 60)

    for held_out in sessions:
        train_sessions = [s for s in sessions if s != held_out]

        # Fit PCA on training sessions (tiers 2, 3 only)
        pca_models = {}
        for tier in [2, 3]:
            if tier not in tiers_available:
                continue
            ts_key   = "timestamp_ms" if tier == 2 else "clip_center_ms"
            raw_paths = [tiers_available[tier][s] for s in train_sessions
                         if s in tiers_available[tier]]
            pca, ok  = fit_pca_on_train(raw_paths, PCA_DIMS)
            if ok:
                pca_models[tier] = pca

        # Build train matrix (concatenate all training sessions)
        X_phys_train_list, X_vid_train_list, Y_train_list = [], [], []
        for run in train_sessions:
            xp, xv, y = build_session_feature_matrix(
                run, df_all, phys_cols, branch_cols, tiers_available, pca_models
            )
            X_phys_train_list.append(xp)
            if xv is not None:
                X_vid_train_list.append(xv)
            Y_train_list.append(y)

        X_phys_tr = np.vstack(X_phys_train_list)
        Y_tr      = np.vstack(Y_train_list)
        X_vid_tr  = np.vstack(X_vid_train_list) if X_vid_train_list else None

        # Build test matrix
        X_phys_te, X_vid_te, Y_te = build_session_feature_matrix(
            held_out, df_all, phys_cols, branch_cols, tiers_available, pca_models
        )

        # Physical-only
        rf_phys = RandomForestRegressor(n_estimators=N_ESTIMATORS, n_jobs=N_JOBS,
                                        random_state=RANDOM_STATE)
        rf_phys.fit(np.nan_to_num(X_phys_tr, nan=0.0), Y_tr)
        r2_phys = float(r2_score(Y_te, rf_phys.predict(np.nan_to_num(X_phys_te, nan=0.0)),
                                  multioutput="uniform_average"))
        loso_physical.append(r2_phys)

        # Combined (physical + video)
        r2_comb = None
        if X_vid_tr is not None and X_vid_te is not None:
            X_both_tr = np.hstack([X_phys_tr, X_vid_tr])
            X_both_te = np.hstack([X_phys_te, X_vid_te])
            rf_comb   = RandomForestRegressor(n_estimators=N_ESTIMATORS, n_jobs=N_JOBS,
                                               random_state=RANDOM_STATE)
            rf_comb.fit(np.nan_to_num(X_both_tr, nan=0.0), Y_tr)
            r2_comb = float(r2_score(Y_te, rf_comb.predict(np.nan_to_num(X_both_te, nan=0.0)),
                                      multioutput="uniform_average"))
            loso_combined.append(r2_comb)

        loso_results[held_out] = {
            "r2_physical": round(r2_phys, 4),
            "r2_combined": round(r2_comb, 4) if r2_comb is not None else None,
            "delta":       round(r2_comb - r2_phys, 4) if r2_comb is not None else None,
        }
        delta_str = f"{r2_comb-r2_phys:+.4f}" if r2_comb is not None else "n/a"
        r2_comb_str = f"{r2_comb:.4f}" if r2_comb is not None else "n/a"
        print(f"  {held_out:<30} {r2_phys:>10.4f} {r2_comb_str:>10} {delta_str:>8}")

    mean_phys = float(np.mean(loso_physical)) if loso_physical else None
    mean_comb = float(np.mean(loso_combined)) if loso_combined else None

    if mean_phys is not None:
        print(f"\n  Mean physical: {mean_phys:.4f}")
    if mean_comb is not None:
        print(f"  Mean combined: {mean_comb:.4f}")
        delta = mean_comb - mean_phys
        if delta > 0.03:
            framing = "Video adds generalizable prediction signal (Δ R² > 0.03)"
        elif abs(delta) <= 0.03:
            framing = "Video is redundant with datalayer (Δ R² ≈ 0)"
        else:
            framing = "Video overfits to session-specific visuals (Δ R² < 0)"
        print(f"  Framing: {framing}")
        loso_results["_summary"] = {
            "mean_physical": round(mean_phys, 4),
            "mean_combined": round(mean_comb, 4),
            "delta":         round(delta, 4),
            "framing":       framing,
        }

    (OUT_DIR / "p5_loso_results.json").write_text(json.dumps(loso_results, indent=2))
    print("\nSaved p5_loso_results.json")

    # Figure 7: Grouped bar chart
    run_labels = [s[-8:] for s in sessions if s in loso_results and "_summary" not in s]
    phys_vals  = [loso_results[s]["r2_physical"] for s in sessions if s in loso_results and "_summary" not in s]
    comb_vals  = [loso_results[s].get("r2_combined") for s in sessions if s in loso_results and "_summary" not in s]

    x  = np.arange(len(run_labels))
    w  = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, phys_vals, w, label="Physical only", color="#1565C0", alpha=0.85)
    valid_comb = [v if v is not None else 0 for v in comb_vals]
    ax.bar(x + w/2, valid_comb, w, label="Physical + video", color="#2E8B57", alpha=0.85)
    if mean_phys is not None:
        ax.axhline(mean_phys, color="#1565C0", ls="--", lw=1.5, alpha=0.6,
                   label=f"Phys mean ({mean_phys:.3f})")
    if mean_comb is not None:
        ax.axhline(mean_comb, color="#2E8B57", ls="--", lw=1.5, alpha=0.6,
                   label=f"Comb mean ({mean_comb:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels, rotation=25, ha="right")
    ax.set_ylabel("R² (held-out session, within BALANCE)")
    ax.set_title("LOSO Generalization: Physical Only vs Physical + Video")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_07_loso_generalization.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_07_loso_generalization.png")

    print("\nPhase 5 complete.")


if __name__ == "__main__":
    main()
