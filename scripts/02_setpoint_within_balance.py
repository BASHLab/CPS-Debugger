"""
02_setpoint_within_balance.py — Fine-grained within-state prediction.

During BALANCE, the controller cycles through 4 target cart positions:
  0.0m → 0.05m → -0.09m → 0.09m

This experiment tests: can physical/syslog/trace features predict
WHICH setpoint is currently active — from observations alone?

This is the cleanest "fine-grained" demonstration: same top-level state,
different sub-behavior, distinguished purely by sensor observations.

Outputs:
  outputs/experiments/fig_02a_setpoint_accuracy_comparison.png
  outputs/experiments/fig_02b_setpoint_confusion_best.png
  outputs/experiments/fig_02c_setpoint_feature_importance.png
  outputs/experiments/fig_02d_setpoint_prediction_timeline.png
  outputs/experiments/fig_02e_setpoint_umap.png
  outputs/experiments/setpoint_results.json
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler

OUT_DIR = Path(
    "/home/simran/allspark-data-exploration/CPS-Debugger/outputs/experiments"
)

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

PHYSICAL_COLS = [
    "dl_current_x_mean",
    "dl_current_x_std",
    "dl_current_x_delta",
    "dl_angle_mean",
    "dl_angle_std",
    "dl_angle_delta",
    "dl_velocity_mean",
    "dl_ang_vel_mean",
    "dl_ang_vel_std",
]
SYSLOG_COLS = ["sl_cpu", "sl_mem", "sl_load1", "sl_load5", "sl_load15"]

TARGET_X_VALUES = [0.0, 0.05, -0.09, 0.09]
TARGET_X_LABELS = ["0.0 m", "+0.05 m", "−0.09 m", "+0.09 m"]
TARGET_X_COLORS = ["#2196F3", "#4CAF50", "#F44336", "#FF9800"]


def temporal_block_cv(n: int, n_folds: int = 5):
    indices = np.arange(n)
    fold_size = n // n_folds
    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])
        yield train_idx, test_idx


def encode_target_x(target_x_col: pd.Series) -> np.ndarray:
    """Encode target_x values as class indices 0-3."""
    classes = np.full(len(target_x_col), -1, dtype=int)
    for i, val in enumerate(TARGET_X_VALUES):
        mask = np.abs(target_x_col.values - val) < 0.005
        classes[mask] = i
    return classes


def cv_classify(X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> dict:
    """Temporal block CV → accuracy, macro F1, per-fold scores."""
    n = len(X)
    all_preds = np.full(n, -1, dtype=int)
    fold_accs, fold_f1s = [], []
    importances_list = []

    for train_idx, test_idx in temporal_block_cv(n, n_folds):
        if len(np.unique(y[train_idx])) < 2:
            continue
        clf = RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=42, max_depth=None
        )
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        all_preds[test_idx] = preds
        fold_accs.append(accuracy_score(y[test_idx], preds))
        fold_f1s.append(
            f1_score(y[test_idx], preds, average="macro", zero_division=0)
        )
        importances_list.append(clf.feature_importances_)

    # Final model on all data for confusion matrix
    clf_full = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=42, max_depth=None
    )
    clf_full.fit(X, y)

    valid = all_preds >= 0
    return {
        "accuracy": float(np.mean(fold_accs)),
        "accuracy_std": float(np.std(fold_accs)),
        "macro_f1": float(np.mean(fold_f1s)),
        "macro_f1_std": float(np.std(fold_f1s)),
        "preds": all_preds,
        "valid_mask": valid,
        "importances": np.mean(importances_list, axis=0) if importances_list else None,
        "clf_full": clf_full,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet = OUT_DIR / "aligned_dataset.parquet"
    vocab_path = OUT_DIR / "vocab_info.json"

    if not parquet.exists():
        raise FileNotFoundError(f"Run 00_preprocess.py first. Expected: {parquet}")

    print("Loading aligned dataset...")
    df = pd.read_parquet(parquet)
    with open(vocab_path) as f:
        vocab_info = json.load(f)
    vocab = vocab_info["vocab"]

    # Filter to BALANCE state only
    df_bal = df[df["dl_state"] == 1].reset_index(drop=True)
    print(f"BALANCE windows: {len(df_bal):,} (of {len(df):,} total)")

    # Encode target_x
    y = encode_target_x(df_bal["dl_target_x"])
    valid_y = y >= 0
    df_bal = df_bal[valid_y].reset_index(drop=True)
    y = y[valid_y]
    print(f"After target_x encoding: {len(df_bal):,} windows")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Branch features (top-20 most variable from vocab)
    branch_matrix = df_bal[vocab].values.astype(float)

    # Feature sets
    X_phy = df_bal[PHYSICAL_COLS].values.astype(float)
    X_sys = df_bal[SYSLOG_COLS].values.astype(float)
    X_all = df_bal[PHYSICAL_COLS + SYSLOG_COLS].values.astype(float)
    X_trace = branch_matrix  # branch count vector as feature
    X_fused = np.hstack([X_all, branch_matrix])

    feature_sets = {
        "Physical only": (X_phy, PHYSICAL_COLS),
        "Syslog only": (X_sys, SYSLOG_COLS),
        "Physical + Syslog": (X_all, PHYSICAL_COLS + SYSLOG_COLS),
        "Trace (branch counts)": (X_trace, vocab[:20]),
        "All (physical+syslog+trace)": (X_fused, PHYSICAL_COLS + SYSLOG_COLS + vocab),
    }

    results_by_fs = {}
    print("\nRunning classification for each feature set...")
    for name, (X, feat_cols) in feature_sets.items():
        print(f"  {name}...")
        res = cv_classify(X, y)
        results_by_fs[name] = res
        print(f"    accuracy={res['accuracy']:.3f}±{res['accuracy_std']:.3f}, "
              f"macro_f1={res['macro_f1']:.3f}")

    # ── Figure 2a: Accuracy/F1 comparison bar chart ───────────────────────────
    names = list(feature_sets.keys())
    accs = [results_by_fs[n]["accuracy"] for n in names]
    acc_stds = [results_by_fs[n]["accuracy_std"] for n in names]
    f1s = [results_by_fs[n]["macro_f1"] for n in names]
    f1_stds = [results_by_fs[n]["macro_f1_std"] for n in names]

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - w/2, accs, w, yerr=acc_stds, capsize=4,
                color="#2196F3", label="Accuracy", error_kw={"linewidth": 1.5})
    b2 = ax.bar(x + w/2, f1s, w, yerr=f1_stds, capsize=4,
                color="#4CAF50", label="Macro F1", error_kw={"linewidth": 1.5})
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=1, label="Random baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("Within-BALANCE Setpoint Prediction (4-class)")
    ax.legend()
    for bar_grp in [b1, b2]:
        for bar in bar_grp:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_02a_setpoint_accuracy_comparison.png")
    plt.close()

    # ── Figure 2b: Confusion matrix for best feature set ─────────────────────
    best_name = max(names, key=lambda n: results_by_fs[n]["macro_f1"])
    best_res = results_by_fs[best_name]
    valid = best_res["valid_mask"]
    y_true = y[valid]
    y_pred = best_res["preds"][valid]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=TARGET_X_LABELS
    )
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {best_name}\n(acc={best_res['accuracy']:.3f})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_02b_setpoint_confusion_best.png")
    plt.close()

    # ── Figure 2c: Feature importance (best physical-only model) ──────────────
    phy_imp = results_by_fs["Physical + Syslog"]["importances"]
    if phy_imp is not None:
        feat_cols = PHYSICAL_COLS + SYSLOG_COLS
        idx = np.argsort(phy_imp)[::-1]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(feat_cols)), phy_imp[idx], color="#2196F3", edgecolor="none")
        ax.set_xticks(range(len(feat_cols)))
        ax.set_xticklabels([feat_cols[i] for i in idx], rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Feature Importance")
        ax.set_title("Feature Importance for Setpoint Prediction (Physical + Syslog)")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "fig_02c_setpoint_feature_importance.png")
        plt.close()

    # ── Figure 2d: Prediction timeline for one run ────────────────────────────
    # Use the last 500 BALANCE windows from the first run for clarity
    run_name = df_bal["run"].iloc[0]
    run_mask = df_bal["run"] == run_name
    df_run = df_bal[run_mask].reset_index(drop=True)
    y_run = y[run_mask.values]
    if len(df_run) > 500:
        df_run = df_run.iloc[:500]
        y_run = y_run[:500]

    X_run = df_run[PHYSICAL_COLS + SYSLOG_COLS].values.astype(float)
    clf_best = results_by_fs["Physical + Syslog"]["clf_full"]
    y_pred_run = clf_best.predict(X_run)

    t_ms = (df_run["win"].values - df_run["win"].values[0]) * 10  # ms offset
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    # Panel 1: physical position vs true target
    ax = axes[0]
    ax.plot(t_ms, df_run["dl_current_x_mean"].values, color="#2196F3", linewidth=1, label="cart position")
    for cls, val, lbl, col in zip(range(4), TARGET_X_VALUES, TARGET_X_LABELS, TARGET_X_COLORS):
        mask = y_run == cls
        ax.scatter(t_ms[mask], np.full(mask.sum(), val), s=3, alpha=0.5, color=col, label=f"target {lbl}")
    ax.set_ylabel("Position (m)")
    ax.set_title(f"Within-BALANCE Timeline — {run_name}")
    ax.legend(loc="upper right", fontsize=8, ncol=3)

    # Panel 2: true vs predicted target class
    ax = axes[1]
    for cls, col in enumerate(TARGET_X_COLORS):
        mask_true = y_run == cls
        ax.scatter(t_ms[mask_true], np.full(mask_true.sum(), cls + 0.1), s=4, color=col, alpha=0.6)
        mask_pred = y_pred_run == cls
        ax.scatter(t_ms[mask_pred], np.full(mask_pred.sum(), cls - 0.1), s=4, color=col, alpha=0.3, marker="x")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(TARGET_X_LABELS, fontsize=9)
    ax.set_ylabel("Target class")
    patches = [mpatches.Patch(color="gray", label="True (upper) / Predicted (lower)")]
    ax.legend(handles=patches, fontsize=8)

    # Panel 3: correct/incorrect
    correct = (y_pred_run == y_run).astype(int)
    ax = axes[2]
    ax.fill_between(t_ms, correct, step="pre", alpha=0.4, color="#4CAF50", label="Correct")
    ax.fill_between(t_ms, 1 - correct, step="pre", alpha=0.4, color="#F44336", label="Incorrect")
    ax.set_ylabel("Prediction")
    ax.set_xlabel("Time (ms)")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_02d_setpoint_prediction_timeline.png")
    plt.close()

    # ── Figure 2e: UMAP of BALANCE traces colored by setpoint ────────────────
    try:
        import umap
        sc = StandardScaler()
        X_umap_in = sc.fit_transform(X_phy[:, :])  # physical only
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30)
        embedding = reducer.fit_transform(X_umap_in)

        fig, ax = plt.subplots(figsize=(8, 7))
        for cls, col, lbl in zip(range(4), TARGET_X_COLORS, TARGET_X_LABELS):
            mask = y == cls
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       s=3, alpha=0.5, color=col, label=lbl, rasterized=True)
        ax.set_title("UMAP of BALANCE Windows (physical features)\nColored by Target Setpoint")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.legend(title="Target", markerscale=3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "fig_02e_setpoint_umap.png")
        plt.close()
        print("UMAP figure saved.")
    except ImportError:
        print("umap-learn not installed, skipping UMAP figure.")

    # ── Save results ──────────────────────────────────────────────────────────
    results_out = {
        "n_balance_windows": int(len(df_bal)),
        "class_distribution": {
            TARGET_X_LABELS[i]: int((y == i).sum()) for i in range(4)
        },
        "results_by_feature_set": {
            name: {
                "accuracy": float(res["accuracy"]),
                "accuracy_std": float(res["accuracy_std"]),
                "macro_f1": float(res["macro_f1"]),
                "macro_f1_std": float(res["macro_f1_std"]),
            }
            for name, res in results_by_fs.items()
        },
        "best_feature_set": best_name,
    }
    (OUT_DIR / "setpoint_results.json").write_text(json.dumps(results_out, indent=2))
    print(f"\nBest feature set: {best_name} (F1={best_res['macro_f1']:.3f})")
    print(f"Outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
