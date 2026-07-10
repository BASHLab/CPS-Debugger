"""
05_audio_trace_prediction.py — Audio-only → execution trace prediction.

Uses the Feb04 run (no physical state sensors).
Tests: can audio features alone predict which branches execute?

This demonstrates a key modality: acoustic observations encode execution trace
details even without direct physical state measurements.

Outputs:
  fig_05a_audio_branch_r2.png
  fig_05b_audio_feature_importance.png
  fig_05c_audio_spectrogram_with_trace.png
  audio_trace_results.json
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


def temporal_block_cv(n: int, n_folds: int = 5):
    indices = np.arange(n)
    fold_size = n // n_folds
    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])
        yield train_idx, test_idx


def run_regression_cv(X: np.ndarray, Y: np.ndarray, n_folds: int = 5) -> dict:
    n = len(X)
    all_preds = np.zeros_like(Y, dtype=float)
    importances = []

    for train_idx, test_idx in temporal_block_cv(n, n_folds):
        rf = RandomForestRegressor(
            n_estimators=100, n_jobs=-1, random_state=42, max_depth=12
        )
        rf.fit(X[train_idx], Y[train_idx])
        all_preds[test_idx] = rf.predict(X[test_idx])
        importances.append(rf.feature_importances_)

    r2_per_branch = [r2_score(Y[:, j], all_preds[:, j]) for j in range(Y.shape[1])]
    cosine_sim = (Y * all_preds).sum(axis=1) / (
        np.linalg.norm(Y, axis=1) * np.linalg.norm(all_preds, axis=1) + 1e-9
    )

    return {
        "r2_per_branch": r2_per_branch,
        "mean_cosine_sim": float(cosine_sim.mean()),
        "importances": np.mean(importances, axis=0),
        "preds": all_preds,
        "truths": Y,
    }


def main():
    parquet = OUT_DIR / "audio_aligned_dataset.parquet"
    vocab_path = OUT_DIR / "vocab_info_feb04.json"

    if not parquet.exists():
        raise FileNotFoundError(
            f"Run 04_audio_preprocess.py first. Expected: {parquet}"
        )

    print("Loading Feb04 audio-aligned dataset...")
    df = pd.read_parquet(parquet)
    with open(vocab_path) as f:
        vocab_info = json.load(f)

    vocab = vocab_info["vocab"]
    audio_cols = vocab_info["audio_cols"]
    syslog_cols = vocab_info.get("syslog_cols", ["sl_cpu", "sl_mem", "sl_load1"])

    branch_matrix = df[vocab].values.astype(float)
    valid = branch_matrix.sum(axis=1) > 0
    df = df[valid].reset_index(drop=True)
    branch_matrix = branch_matrix[valid]
    print(f"Dataset: {len(df):,} windows, {len(vocab)} branch targets")

    # Feature sets
    audio_present = [c for c in audio_cols if c in df.columns]
    syslog_present = [c for c in syslog_cols if c in df.columns]

    X_audio = df[audio_present].values.astype(float)
    X_sys = df[syslog_present].values.astype(float) if syslog_present else None
    X_all = df[audio_present + syslog_present].values.astype(float)
    Y = branch_matrix

    print(f"Audio features: {len(audio_present)}")
    print(f"Syslog features: {len(syslog_present)}")

    # Run regression for each feature set
    print("\nRunning: Audio only → branch counts...")
    res_audio = run_regression_cv(X_audio, Y)

    print("Running: Audio + Syslog → branch counts...")
    res_all = run_regression_cv(X_all, Y)

    r2_audio = res_audio["r2_per_branch"]
    r2_all = res_all["r2_per_branch"]
    r2_audio_arr = np.array(r2_audio)
    r2_all_arr = np.array(r2_all)

    # ── Figure 5a: R² bar chart ───────────────────────────────────────────────
    idx_sorted = np.argsort(r2_audio_arr)[::-1]
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(vocab))
    ax.bar(x - 0.2, r2_audio_arr[idx_sorted], 0.35, label="Audio only",
           color="#E91E63", edgecolor="none", alpha=0.85)
    ax.bar(x + 0.2, r2_all_arr[idx_sorted], 0.35, label="Audio + Syslog",
           color="#9C27B0", edgecolor="none", alpha=0.85)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1)
    ax.axhline(0.3, color="orange", linestyle="--", linewidth=1)
    ax.set_xticks(x[::5])
    ax.set_xticklabels([vocab[idx_sorted[i]][:15] for i in range(0, len(vocab), 5)],
                       rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("R²")
    ax.set_title(f"Branch Count Prediction from Audio — Feb04 Run\n"
                 f"Audio R²>0.3: {(r2_audio_arr>=0.3).sum()}/{len(vocab)} branches, "
                 f"mean R²={r2_audio_arr.mean():.3f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_05a_audio_branch_r2.png")
    plt.close()

    # ── Figure 5b: Feature importance ────────────────────────────────────────
    imp = res_audio["importances"]
    idx_imp = np.argsort(imp)[::-1][:20]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(idx_imp)), imp[idx_imp], color="#E91E63", edgecolor="none")
    ax.set_xticks(range(len(idx_imp)))
    ax.set_xticklabels([audio_present[i] for i in idx_imp], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Feature Importance")
    ax.set_title("Top Audio Features for Branch Count Prediction")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_05b_audio_feature_importance.png")
    plt.close()

    # ── Figure 5c: Timeline of audio RMS vs top branch ────────────────────────
    if "audio_rms" in df.columns and len(vocab) > 0:
        top_branch = vocab[np.argmax(r2_audio)]
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        t = np.arange(len(df)) * 10  # ms
        axes[0].plot(t, df["audio_rms"].values, color="#E91E63", linewidth=0.8)
        axes[0].set_ylabel("Audio RMS")
        axes[0].set_title("Audio Energy vs Execution Trace (Feb04 Run)")
        axes[1].plot(t, df[top_branch].values, color="#9C27B0", linewidth=0.8)
        axes[1].set_ylabel(f"Branch count\n({top_branch[:30]})")
        axes[1].set_xlabel("Time (ms)")
        axes[1].set_title(f"Top Predicted Branch (R²={max(r2_audio):.3f})")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "fig_05c_audio_trace_timeline.png")
        plt.close()

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "n_windows": int(len(df)),
        "n_audio_features": len(audio_present),
        "n_branch_targets": len(vocab),
        "audio_only": {
            "mean_r2": float(r2_audio_arr.mean()),
            "frac_r2_above_0.3": float((r2_audio_arr >= 0.3).mean()),
            "frac_r2_above_0.5": float((r2_audio_arr >= 0.5).mean()),
            "mean_cosine_sim": float(res_audio["mean_cosine_sim"]),
        },
        "audio_plus_syslog": {
            "mean_r2": float(r2_all_arr.mean()),
            "frac_r2_above_0.3": float((r2_all_arr >= 0.3).mean()),
            "mean_cosine_sim": float(res_all["mean_cosine_sim"]),
        },
        "top_5_branches_by_audio_r2": [
            {"branch": vocab[i], "r2_audio": float(r2_audio[i]), "r2_all": float(r2_all[i])}
            for i in np.argsort(r2_audio)[::-1][:5]
        ],
    }
    (OUT_DIR / "audio_trace_results.json").write_text(json.dumps(results, indent=2))

    print("\n── Audio → Trace Results ──")
    print(f"Audio only:   mean R²={results['audio_only']['mean_r2']:.3f}, "
          f"R²>0.3: {results['audio_only']['frac_r2_above_0.3']:.0%}")
    print(f"Audio+Syslog: mean R²={results['audio_plus_syslog']['mean_r2']:.3f}, "
          f"R²>0.3: {results['audio_plus_syslog']['frac_r2_above_0.3']:.0%}")
    print(f"\nOutputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
