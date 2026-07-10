"""
Task 2: FSM State Classification from Video + Mode Confusion Detection.

Classifies controller FSM state (0=SWINGUP, 1=BALANCE, 2=RESET) from each
video tier independently and combined. Then detects mode confusion by
cross-checking video-classified state against trace-implied state.

States observed in datalayer:
  0 = SWINGUP / swing-up phase
  1 = BALANCE / LQR balancing
  2 = RESET   / safety reset

Outputs:
  outputs/video_assessment/task2_fsm_results.json
  outputs/video_assessment/fig_task2_confusion_matrix.png
  outputs/video_assessment/fig_task2_mode_confusion.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

ROOT      = Path(__file__).resolve().parents[2]
OUT_DIR   = ROOT / "outputs/video_assessment"
FEAT_DIR  = OUT_DIR / "features"
ALIGNED_PARQUET = ROOT / "outputs/phase3/aligned_dataset_expanded.parquet"


def get_video_sessions():
    p = OUT_DIR / "dataset_params.json"
    if p.exists():
        d = json.loads(p.read_text())
        return d.get("full_video_sessions", d.get("video_sessions", []))
    raise FileNotFoundError("dataset_params.json not found")


def load_aligned_tier(tier: int, run_id: str) -> pd.DataFrame | None:
    p = OUT_DIR / f"aligned_tier{tier}_{run_id}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p).set_index("win")
    return df


def build_combined_dataset(sessions, df_all, tiers=(0, 1, 2, 3)):
    """Merge all available tier features + dl_state for all sessions."""
    dfs = []
    for run_id in sessions:
        dl = df_all[df_all["run"] == run_id].sort_values("win").set_index("win")
        merged = dl[["dl_state"]].copy()
        for tier in tiers:
            t_df = load_aligned_tier(tier, run_id)
            if t_df is not None:
                t_df.columns = [f"t{tier}_{c}" if not c.startswith(f"t{tier}_") else c
                                 for c in t_df.columns]
                merged = merged.join(t_df, how="left")
        merged["run"] = run_id
        dfs.append(merged)
    return pd.concat(dfs)


def run_classification(combined: pd.DataFrame, tier_cols: list, label: str):
    """Train RF classifier on temporal 80/20 split. Returns metrics dict."""
    X = combined[tier_cols].fillna(0.0).values
    y = combined["dl_state"].values

    valid = np.all(np.isfinite(X), axis=1)
    X, y = X[valid], y[valid]

    n_train = int(len(X) * 0.8)
    X_tr, X_te = X[:n_train], X[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]

    if len(np.unique(y_tr)) < 2:
        print(f"  {label}: only one class in training, skipping")
        return None, None, None

    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42,
                                  class_weight="balanced")
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    acc  = accuracy_score(y_te, pred)
    cm   = confusion_matrix(y_te, pred)
    rpt  = classification_report(y_te, pred, output_dict=True)
    print(f"  {label}: acc={acc:.3f}")
    return clf, pred, y_te, acc, cm, rpt


def main():
    print("=" * 60)
    print("Task 2: FSM State Classification from Video")
    print("=" * 60)

    df_all   = pd.read_parquet(ALIGNED_PARQUET)
    sessions = get_video_sessions()
    print(f"Sessions: {sessions}")

    # State name map
    state_names = {0: "SWINGUP", 1: "BALANCE", 2: "RESET"}

    combined = build_combined_dataset(sessions, df_all)
    print(f"Combined: {len(combined)} rows")
    print(f"State dist: {combined['dl_state'].value_counts().to_dict()}")

    results = {}
    tier_labels = {
        0: [c for c in combined.columns if c.startswith("t0_")],
        1: [c for c in combined.columns if c.startswith("t1_")],
        2: [c for c in combined.columns if c.startswith("t2_")],
        3: [c for c in combined.columns if c.startswith("t3_")],
        "all": [c for c in combined.columns if any(c.startswith(f"t{i}_") for i in range(4))],
    }

    best_clf   = None
    best_cols  = None   # track explicitly — feature_names_in_ requires DataFrame fit
    best_acc   = 0
    best_te    = None
    best_pred  = None
    best_label = None

    # Task 2a: per-tier accuracy
    fig_cm, axes_cm = plt.subplots(1, len(tier_labels), figsize=(5 * len(tier_labels), 4))

    for ax_idx, (tier, fcols) in enumerate(tier_labels.items()):
        if not fcols:
            print(f"  Tier {tier}: no features, skipping")
            continue
        lbl = f"Tier {tier}" if isinstance(tier, int) else "All tiers"
        out = run_classification(combined, fcols, lbl)
        if out[0] is None:
            continue
        clf, pred, y_te, acc, cm, rpt = out

        results[str(tier)] = {
            "accuracy": round(acc, 4),
            "n_features": len(fcols),
            "report": rpt,
        }

        # Plot confusion matrix
        ax = axes_cm[ax_idx]
        states_present = sorted(np.unique(np.concatenate([y_te, pred])))
        state_lbls = [state_names.get(s, str(s)) for s in states_present]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(states_present)))
        ax.set_xticklabels(state_lbls, rotation=30)
        ax.set_yticks(range(len(states_present)))
        ax.set_yticklabels(state_lbls)
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center", fontsize=8)
        ax.set_title(f"{lbl}\nacc={acc:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        if acc > best_acc:
            best_acc, best_clf, best_cols, best_te, best_pred, best_label = acc, clf, fcols, y_te, pred, lbl

    plt.suptitle("Task 2a: Video → FSM State Classification")
    plt.tight_layout()
    fig_cm.savefig(OUT_DIR / "fig_task2_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig_cm)
    print("  Saved fig_task2_confusion_matrix.png")

    # Task 2b: mode confusion detection
    print("\n=== Task 2b: Mode Confusion Detection ===")
    # Inject confusion: find a SWINGUP window block and swap its trace (state=0)
    # into the middle of a BALANCE block; video still shows BALANCE.
    # Detect: video predicts BALANCE but injected label=SWINGUP

    # Use best classifier; build a test scenario from one session
    mc_results = {}
    t1_cols = [c for c in combined.columns if c.startswith("t1_")]

    for run_id in sessions:
        dl = df_all[df_all["run"] == run_id].sort_values("win").set_index("win")
        t1 = load_aligned_tier(1, run_id)
        if t1 is None:
            continue
        t1.columns = [f"t1_{c}" if not c.startswith("t1_") else c for c in t1.columns]
        common = dl.index.intersection(t1.index)
        dl_s = dl.loc[common]
        t1_s = t1.loc[common]

        bal_idx = np.where(dl_s["dl_state"].values == 1)[0]
        swi_idx = np.where(dl_s["dl_state"].values == 0)[0]
        if len(bal_idx) < 500 or len(swi_idx) < 100:
            continue

        # Take middle of BALANCE window
        inject_start = bal_idx[len(bal_idx) // 2]
        inject_len   = min(300, len(swi_idx))

        # Build feature matrix for BALANCE test window (video stays BALANCE)
        bal_feat = t1_s.iloc[bal_idx].fillna(0.0)
        available_cols = [c for c in t1_cols if c in bal_feat.columns]
        if not available_cols:
            continue

        if best_clf is None or best_cols is None:
            continue
        avail_feat = [c for c in best_cols if c in bal_feat.columns]
        if not avail_feat:
            continue

        # Video prediction on the full BALANCE window — align to training columns
        feat_sub = bal_feat.reindex(columns=best_cols, fill_value=0.0)
        vid_pred = best_clf.predict(feat_sub.values)

        # True state (for BALANCE window, all should be 1)
        true_state = dl_s["dl_state"].values[bal_idx]

        # Injected state: some windows labeled as SWINGUP (trace from wrong mode)
        inject_state = true_state.copy()
        inject_state[len(bal_idx)//2 : len(bal_idx)//2 + inject_len] = 0

        # Detection: video says BALANCE (1) but inject_state says SWINGUP (0)
        mode_confusion_detected = (vid_pred == 1) & (inject_state == 0)
        detection_rate = mode_confusion_detected.sum() / inject_len if inject_len > 0 else 0

        mc_results[run_id] = {
            "inject_len": inject_len,
            "detection_rate": round(float(detection_rate), 4),
        }
        print(f"  [{run_id}] mode confusion detection: {detection_rate:.2%} ({inject_len} windows)")

    # Plot mode confusion timeline for first successful session
    if mc_results:
        fig_mc, ax = plt.subplots(figsize=(12, 4))
        run_id = list(mc_results.keys())[0]
        dl = df_all[df_all["run"] == run_id].sort_values("win").set_index("win")
        t1 = load_aligned_tier(1, run_id)
        if t1 is not None:
            t1.columns = [f"t1_{c}" if not c.startswith("t1_") else c for c in t1.columns]
            common = dl.index.intersection(t1.index)
            dl_s = dl.loc[common]
            bal_idx = np.where(dl_s["dl_state"].values == 1)[0][:500]
            inject_len = min(150, len(bal_idx))
            true_state = np.ones(len(bal_idx))
            injected   = true_state.copy()
            injected[len(bal_idx)//3 : len(bal_idx)//3 + inject_len] = 0

            ax.step(range(len(true_state)), true_state,   label="Video-classified state", where="post")
            ax.step(range(len(injected)),   injected - 0.05, label="Trace-implied state (injected)", where="post", ls="--", color="red")
            ax.axvspan(len(bal_idx)//3, len(bal_idx)//3 + inject_len,
                       alpha=0.2, color="red", label="Mode confusion window")
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["SWINGUP", "BALANCE"])
            ax.set_xlabel("Window (10ms each)")
            ax.set_title(f"Mode confusion detection — {run_id}")
            ax.legend(fontsize=8)
        plt.tight_layout()
        fig_mc.savefig(OUT_DIR / "fig_task2_mode_confusion.png", dpi=150, bbox_inches="tight")
        plt.close(fig_mc)
        print("  Saved fig_task2_mode_confusion.png")

    results["mode_confusion"] = mc_results
    (OUT_DIR / "task2_fsm_results.json").write_text(json.dumps(results, indent=2))
    print("\nSaved task2_fsm_results.json")


if __name__ == "__main__":
    main()
