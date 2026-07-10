"""
task_2_2_audio_features.py — Extract audio features and correlate with trace.

Finds runs with audio + datalayer + trace, extracts MFCC features at 10ms windows,
and runs RF regression (audio → branch counts within BALANCE).

Output:
  outputs/phase3/audio_features.parquet
  outputs/phase3/audio_summary.json
  outputs/phase3/fig_audio_r2.png
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT    = Path("/home/simran/allspark-data-exploration/CPS-Debugger")
P3_OUT  = ROOT / "outputs/phase3"
P3_OUT.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path("/home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs")

PHYSICAL_COLS = [
    "dl_current_x_mean", "dl_current_x_std", "dl_current_x_delta",
    "dl_angle_mean", "dl_angle_std", "dl_angle_delta",
    "dl_velocity_mean", "dl_ang_vel_mean", "dl_ang_vel_std",
]


def extract_audio_features(wav_path: Path, win_ms: int = 10):
    """Extract MFCC + spectral features at 10ms windows."""
    import librosa
    import warnings
    warnings.filterwarnings("ignore")

    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    hop_length = int(sr * win_ms / 1000)  # samples per 10ms window
    win_length = hop_length * 2           # 20ms analysis window

    print(f"    Audio: {len(y)/sr:.1f}s at {sr}Hz, "
          f"{len(y)//hop_length} windows of {win_ms}ms")

    # MFCC (13 coefficients + deltas = 26 features)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13,
                                  hop_length=hop_length, win_length=win_length)
    mfcc_delta = librosa.feature.delta(mfcc)

    # Spectral features (6)
    rms       = librosa.feature.rms(y=y, hop_length=hop_length)
    zcr       = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)
    spec_flat = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)

    # Stack: (n_features, n_frames) → transpose to (n_frames, n_features)
    features = np.vstack([
        mfcc, mfcc_delta,
        rms, zcr, spec_cent, spec_roll, spec_flat
    ]).T  # (n_frames, 31)

    n_frames = features.shape[0]
    feature_names = (
        [f"mfcc_{i}" for i in range(13)]
        + [f"mfcc_d_{i}" for i in range(13)]
        + ["rms", "zcr", "spec_centroid", "spec_rolloff", "spec_flatness"]
    )
    return features, feature_names, sr, hop_length


def find_audio_runs():
    """Find runs with both audio and datalayer/trace."""
    inv_path = P3_OUT / "data_inventory.csv"
    if inv_path.exists():
        inv = pd.read_csv(inv_path)
        audio_runs = inv[inv["has_audio"] == True]
        complete   = audio_runs[audio_runs["status"].str.startswith("COMPLETE")]
        result = []
        for _, row in complete.iterrows():
            p = Path(row["path"])
            if not p.exists():
                p = DATA_ROOT / "extracted" / row["run_id"]
            result.append(p)
        return result

    # Fallback: scan manually
    result = []
    for r in sorted((DATA_ROOT / "extracted").glob("*")):
        if r.is_dir() and list(r.rglob("*.wav")):
            if (r / "datalayer" / "datalayer.csv").exists():
                result.append(r)
    return result


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150,
                         "savefig.dpi": 300})

    audio_runs = find_audio_runs()
    if not audio_runs:
        print("No runs found with audio + complete datalayer/trace.")
        print("The Feb04 run has audio but no datalayer — audio analysis deferred.")
        (P3_OUT / "audio_summary.json").write_text(json.dumps({
            "status": "NO_COMPLETE_AUDIO_RUNS",
            "message": "No runs found with audio + datalayer + trace. "
                       "Feb04 run has audio but no physical sensor data.",
            "n_audio_runs_found": 0,
        }, indent=2))
        return

    # Load aligned dataset
    expanded = P3_OUT / "aligned_dataset_expanded.parquet"
    baseline = ROOT / "outputs/experiments/aligned_dataset.parquet"
    dataset_path = expanded if expanded.exists() else baseline
    if not dataset_path.exists():
        print("ERROR: No aligned dataset found.")
        return

    df = pd.read_parquet(dataset_path)
    vocab = json.loads((ROOT / "outputs/experiments/vocab_info.json").read_text())["vocab"]
    vocab = vocab[:50]  # use top 50 for speed

    all_results = []

    for run_dir in audio_runs:
        run_id = run_dir.name
        print(f"\n── {run_id} ──")
        run_df = df[df["run"] == run_id]
        if len(run_df) == 0:
            print(f"  Not in aligned dataset, skipping")
            continue

        # Find WAV file
        wav_files = list(run_dir.rglob("*.wav"))
        if not wav_files:
            print(f"  No WAV files found")
            continue
        wav = wav_files[0]
        print(f"  WAV: {wav}")

        try:
            features, feat_names, sr, hop = extract_audio_features(wav)
        except Exception as e:
            print(f"  Audio extraction failed: {e}")
            continue

        # Align audio frames to 10ms windows
        n_frames  = features.shape[0]
        first_win = run_df["win"].min()
        frame_wins = np.arange(n_frames) + first_win

        feat_df = pd.DataFrame(features, columns=feat_names)
        feat_df["win"] = frame_wins

        # Merge with branch counts
        merged = feat_df.merge(run_df[["win", "dl_state"] + vocab], on="win", how="inner")
        bal    = merged[merged["dl_state"] == 1]
        print(f"  Matched {len(merged)} windows ({len(bal)} BALANCE)")

        if len(bal) < 30:
            print("  Insufficient BALANCE windows, skipping")
            continue

        X_audio = bal[feat_names].values
        X_phys  = bal[[c for c in PHYSICAL_COLS if c in bal.columns]].values
        split   = int(0.8 * len(bal))

        r2_audio, r2_phys = [], []
        for branch in vocab:
            y = bal[branch].values
            if y.std() < 1e-6:
                r2_audio.append(0.0)
                r2_phys.append(0.0)
                continue
            rf_a = RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=4)
            rf_a.fit(X_audio[:split], y[:split])
            r2_audio.append(max(-1.0, r2_score(y[split:], rf_a.predict(X_audio[split:]))))

            rf_p = RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=4)
            rf_p.fit(X_phys[:split], y[:split])
            r2_phys.append(max(-1.0, r2_score(y[split:], rf_p.predict(X_phys[split:]))))

        print(f"  Audio R² mean={np.mean(r2_audio):.3f}  "
              f"Physical R² mean={np.mean(r2_phys):.3f}")

        all_results.append({
            "run_id": run_id,
            "n_balance_windows": int(len(bal)),
            "audio_r2_mean":    float(np.mean(r2_audio)),
            "audio_r2_frac_gt_03": float(np.mean([r > 0.3 for r in r2_audio])),
            "physical_r2_mean": float(np.mean(r2_phys)),
            "per_branch_r2_audio": [float(r) for r in r2_audio],
            "per_branch_r2_phys":  [float(r) for r in r2_phys],
        })

    if not all_results:
        (P3_OUT / "audio_summary.json").write_text(json.dumps({
            "status": "NO_DATA",
            "message": "Audio runs found but no matched windows in aligned dataset.",
        }, indent=2))
        return

    # Summary
    out = {
        "n_audio_runs_analyzed": len(all_results),
        "results": all_results,
        "overall_audio_r2_mean": float(np.mean([r["audio_r2_mean"] for r in all_results])),
        "overall_physical_r2_mean": float(np.mean([r["physical_r2_mean"] for r in all_results])),
    }
    (P3_OUT / "audio_summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved audio_summary.json")

    # Figure
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in all_results:
        r2s = sorted(r["per_branch_r2_audio"], reverse=True)
        ax.plot(r2s, label=f"{r['run_id'][:15]} (mean={r['audio_r2_mean']:.3f})")
    ax.axhline(0.3, linestyle="--", color="gray", label="R²=0.3")
    ax.set_xlabel("Branch rank")
    ax.set_ylabel("R² (audio → branch count, within BALANCE)")
    ax.set_title("Audio modality: branch count predictability")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(P3_OUT / "fig_audio_r2.png")
    plt.close(fig)
    print("Saved fig_audio_r2.png")
    print("Done.")


if __name__ == "__main__":
    main()
