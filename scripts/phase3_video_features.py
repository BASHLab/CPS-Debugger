"""
phase3_video_features.py — Extract physically-motivated features from pendulum video.

Two approaches:
  A. Optical flow (Farneback): dense flow magnitude → proxy for angular velocity and
     motion energy. No model required.
  B. Direct pendulum tracking (OpenCV): extract angle θ, cart position x, angular
     velocity dθ/dt from each frame using contour detection.

For each run with video+datalayer+trace:
  1. Builds 10ms datalayer windows with physical features + dl_timestamp_ms
  2. Builds branch-count targets from processed_trace.csv (top-50 branches by variance
     within BALANCE state, shared vocabulary across runs)
  3. Extracts video features and aligns to datalayer windows via timestamps
  4. Compares RF R²: video-only vs physical-only vs combined (within BALANCE state)

Outputs:
  outputs/phase3/video_features.parquet   — per-window aligned features
  outputs/phase3/video_r2_summary.json    — RF R² comparison
  outputs/phase3/fig_video_rf_r2.png      — bar chart
"""

import json
import time
import warnings
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT      = Path("/home/simran/allspark-data-exploration/CPS-Debugger")
DATA_ROOT = Path("/home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs/extracted")
OUT       = ROOT / "outputs/phase3"
OUT.mkdir(parents=True, exist_ok=True)

WIN_MS      = 10       # 10ms datalayer windows
TOP_K_VOCAB = 50       # branches by variance within BALANCE
N_JOBS      = 8
RANDOM_STATE = 42
BALANCE_STATE = 1      # pendulum_state == 1 → BALANCE

PHYSICAL_COLS = [
    "dl_current_x_mean", "dl_current_x_std", "dl_current_x_delta",
    "dl_angle_mean",     "dl_angle_std",     "dl_angle_delta",
    "dl_velocity_mean",  "dl_ang_vel_mean",  "dl_ang_vel_std",
]

VIDEO_RUNS = [
    "2025-03-17_10-51-58",
    "2025-03-17_11-06-39",
    "2025-03-18_12-39-10",
    "2025-03-19_10-05-47",
    "2025-03-19_10-20-35",
    "2025-03-19_10-37-56",
]


# ─── Datalayer preprocessing ──────────────────────────────────────────────────

def load_datalayer_windows(run_dir: Path) -> pd.DataFrame:
    """
    Load datalayer.csv → aggregate into WIN_MS windows.
    Returns DataFrame with columns:
      win, dl_state, dl_current_x_mean, ..., dl_timestamp_ms (window start in ms)
    """
    win_us = WIN_MS * 1000
    df = pd.read_csv(run_dir / "datalayer" / "datalayer.csv")
    df["t_us"] = (df["timestamp"] // 1000).astype(np.int64)   # ns → µs
    df["win"]  = df["t_us"] // win_us

    agg = df.groupby("win").agg(
        dl_state         = ("pendulum_state",   "first"),
        dl_target_x      = ("target_x",         "first"),
        dl_current_x_mean= ("current_x",        "mean"),
        dl_current_x_std = ("current_x",        "std"),
        dl_current_x_delta=("current_x",        lambda x: float(x.iloc[-1]-x.iloc[0]) if len(x)>1 else 0.0),
        dl_angle_mean    = ("current_angle",     "mean"),
        dl_angle_std     = ("current_angle",     "std"),
        dl_angle_delta   = ("current_angle",     lambda x: float(x.iloc[-1]-x.iloc[0]) if len(x)>1 else 0.0),
        dl_velocity_mean = ("velocity",          "mean"),
        dl_ang_vel_mean  = ("angular_velocity",  "mean"),
        dl_ang_vel_std   = ("angular_velocity",  "std"),
        dl_timestamp_ms  = ("t_us",              lambda x: int(x.min()) // 1000),  # µs → ms
    ).reset_index()
    agg["dl_state"] = agg["dl_state"].fillna(-1).astype(int)
    return agg.fillna(0.0)


def load_trace_branch_counts(run_dir: Path) -> pd.DataFrame:
    """
    Parse processed_trace.csv → per-window branch count dict.
    Returns DataFrame with columns: win + one column per branch.
    """
    win_us = WIN_MS * 1000
    tr = pd.read_csv(run_dir / "processed_trace" / "processed_trace.csv")
    tr["t_us"] = tr["timestamp"].astype(np.int64)
    tr["win"]  = tr["t_us"] // win_us

    counts_rows = []
    for win_id, group in tqdm(tr.groupby("win"), desc="  trace windows", leave=False):
        window_counts: dict = {}
        for cf_str in group["cf_table"].values:
            try:
                cf = json.loads(cf_str)
            except Exception:
                continue
            for func_id, src_dict in cf.items():
                for src_pc, dst_dict in src_dict.items():
                    for dst_pc, count in dst_dict.items():
                        key = f"{func_id}:{src_pc}:{dst_pc}"
                        window_counts[key] = window_counts.get(key, 0) + int(count)
        window_counts["win"] = win_id
        counts_rows.append(window_counts)

    counts_df = pd.DataFrame(counts_rows).fillna(0.0)
    counts_df["win"] = counts_df["win"].astype(np.int64)
    return counts_df


def select_vocab(all_counts: list, balance_windows_per_run: list, top_k: int) -> list:
    """Select top-K branches by coefficient of variation within BALANCE windows."""
    balance_frames = []
    for counts_df, bal_wins in zip(all_counts, balance_windows_per_run):
        bal = counts_df[counts_df["win"].isin(bal_wins)]
        balance_frames.append(bal)

    combined = pd.concat(balance_frames, ignore_index=True)
    branch_cols = [c for c in combined.columns if c != "win"]
    n_total = len(combined)

    stats = []
    for col in branch_cols:
        vals = combined[col].values
        presence = (vals > 0).sum()
        if presence < 0.02 * n_total:
            continue
        mean_val = vals.mean()
        std_val  = vals.std()
        cv = std_val / (mean_val + 1e-9)
        stats.append({"branch": col, "cv": cv})

    stats_df = pd.DataFrame(stats).sort_values("cv", ascending=False)
    return stats_df.head(top_k)["branch"].tolist()


# ─── Video feature extraction ─────────────────────────────────────────────────

def load_video_timestamps(ts_path: Path) -> dict:
    """Parse output_timestamps_session_*.txt → {frame_idx: timestamp_ms}."""
    frame_ts = {}
    for line in ts_path.read_text().strip().split("\n"):
        parts = line.split(":")
        if len(parts) == 2:
            try:
                frame_ts[int(parts[0].strip())] = int(parts[1].strip())
            except ValueError:
                pass
    return frame_ts


def extract_optical_flow_features(mp4_path: Path, frame_ts: dict,
                                   subsample: int = 3) -> pd.DataFrame:
    """
    Extract per-frame optical flow magnitude and direction via Farneback.
    Returns DataFrame: frame_idx, timestamp_ms, flow_mag_mean, flow_mag_max,
                       flow_mag_std, flow_x_mean, flow_y_mean.
    """
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {mp4_path}")

    prev_gray = None
    records = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % subsample == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                records.append({
                    "frame_idx":    frame_idx,
                    "timestamp_ms": frame_ts.get(frame_idx, -1),
                    "flow_mag_mean": float(mag.mean()),
                    "flow_mag_max":  float(mag.max()),
                    "flow_mag_std":  float(mag.std()),
                    "flow_x_mean":   float(flow[..., 0].mean()),
                    "flow_y_mean":   float(flow[..., 1].mean()),
                })
            prev_gray = gray

        frame_idx += 1

    cap.release()
    return pd.DataFrame(records)


def extract_pendulum_tracking_features(mp4_path: Path, frame_ts: dict,
                                        subsample: int = 3) -> pd.DataFrame:
    """
    Direct pendulum tracking via brightness-based blob detection.
    Returns DataFrame: frame_idx, timestamp_ms, bob_x, bob_y, bob_area,
                       estimated_angle, estimated_angvel.
    """
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {mp4_path}")

    height = width = None
    prev_bob_x = None
    records = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % subsample == 0:
            if height is None:
                height, width = frame.shape[:2]
                pivot_x = width / 2
                pivot_y = height * 0.15

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)

            bob_x = bob_y = bob_area = np.nan
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                if area > 50:
                    M = cv2.moments(largest)
                    if M["m00"] > 0:
                        bob_x    = M["m10"] / M["m00"] / width
                        bob_y    = M["m01"] / M["m00"] / height
                        bob_area = area / (width * height)

            est_angle = np.nan
            if not np.isnan(bob_x):
                dx = bob_x * width  - pivot_x
                dy = bob_y * height - pivot_y
                est_angle = float(np.arctan2(dx, dy))

            ang_vel = np.nan
            if prev_bob_x is not None and not np.isnan(bob_x) and not np.isnan(prev_bob_x):
                ang_vel = float(bob_x - prev_bob_x) * 30

            records.append({
                "frame_idx":        frame_idx,
                "timestamp_ms":     frame_ts.get(frame_idx, -1),
                "bob_x":            bob_x,
                "bob_y":            bob_y,
                "bob_area":         bob_area,
                "estimated_angle":  est_angle,
                "estimated_angvel": ang_vel,
            })
            prev_bob_x = bob_x

        frame_idx += 1

    cap.release()
    return pd.DataFrame(records)


# ─── Alignment ────────────────────────────────────────────────────────────────

def align_frames_to_windows(frame_df: pd.DataFrame,
                             win_df: pd.DataFrame,
                             feature_cols: list) -> pd.DataFrame:
    """
    Merge frame-level features (timestamp_ms) into datalayer windows
    (dl_timestamp_ms, win). Each window covers [dl_timestamp_ms, dl_timestamp_ms+WIN_MS).
    """
    valid = frame_df[frame_df["timestamp_ms"] > 0].copy()
    if valid.empty:
        return pd.DataFrame()

    # Sort both by time for merge_asof
    valid = valid.sort_values("timestamp_ms")
    win_sorted = win_df[["win", "dl_timestamp_ms"]].sort_values("dl_timestamp_ms")

    # Assign each frame to the nearest window start (backward direction)
    merged = pd.merge_asof(
        valid, win_sorted,
        left_on="timestamp_ms", right_on="dl_timestamp_ms",
        direction="backward"
    )
    # Keep only frames within the window duration
    merged = merged[
        merged["timestamp_ms"] < merged["dl_timestamp_ms"] + WIN_MS
    ]

    if merged.empty:
        return pd.DataFrame()

    agg = merged.groupby("win")[feature_cols].mean().reset_index()
    return agg


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 3: Video Feature Analysis")
    print("=" * 60)

    # ── Step 1: Preprocess datalayer + trace for all video runs ──────────────
    print("\n[Step 1] Building datalayer windows + trace branch counts...")
    run_dfs    = {}   # run → merged datalayer+trace DataFrame
    all_counts = []   # for vocab building
    bal_wins   = []   # BALANCE window IDs per run

    for run in VIDEO_RUNS:
        run_dir = DATA_ROOT / run
        if not run_dir.exists():
            print(f"  [{run}] Directory not found, skipping")
            continue

        print(f"\n  [{run}]")
        t0 = time.time()

        # Datalayer windows (includes dl_timestamp_ms)
        dl = load_datalayer_windows(run_dir)
        print(f"    Datalayer: {len(dl)} windows, {(dl['dl_state']==BALANCE_STATE).sum()} BALANCE")

        # Trace branch counts
        counts = load_trace_branch_counts(run_dir)
        print(f"    Trace: {len(counts)} windows, {counts.shape[1]-1} unique branches ({time.time()-t0:.0f}s)")

        # Merge
        merged = dl.merge(counts, on="win", how="inner")
        merged["run"] = run

        run_dfs[run]  = merged
        all_counts.append(counts.drop(columns=["win"], errors="ignore"))
        bal_wins.append(set(merged[merged["dl_state"] == BALANCE_STATE]["win"].tolist()))

    if not run_dfs:
        print("No runs processed. Exiting.")
        return

    # ── Step 2: Build shared vocabulary ──────────────────────────────────────
    print(f"\n[Step 2] Selecting top-{TOP_K_VOCAB} branches by CoV within BALANCE...")
    vocab = select_vocab(
        [run_dfs[r] for r in run_dfs],
        [bal_wins[i] for i, r in enumerate(run_dfs)],
        TOP_K_VOCAB
    )
    # Recompute vocab using only columns that exist across runs
    all_branch_cols = set(vocab)
    for r, df in run_dfs.items():
        all_branch_cols &= set(df.columns)
    vocab = [b for b in vocab if b in all_branch_cols]
    print(f"  Final shared vocabulary: {len(vocab)} branches")

    if len(vocab) < 5:
        print("  WARNING: Very few shared branches across runs. Expanding search...")
        all_cols = set.intersection(*[set(df.columns) for df in run_dfs.values()])
        branch_candidates = [c for c in all_cols if c not in {"win","run","dl_state","dl_timestamp_ms"} and ":" in c]
        vocab = branch_candidates[:TOP_K_VOCAB]
        print(f"  Using {len(vocab)} branch columns")

    # ── Step 3: Process each run's video ─────────────────────────────────────
    print("\n[Step 3] Extracting video features per run...")
    run_results    = {}
    all_merged_dfs = []

    FLOW_COLS  = ["flow_mag_mean","flow_mag_max","flow_mag_std","flow_x_mean","flow_y_mean"]
    TRACK_COLS = ["bob_x","bob_y","bob_area","estimated_angle","estimated_angvel"]

    for run in run_dfs:
        run_dir   = DATA_ROOT / run
        video_dir = run_dir / "camera-video"
        mp4s      = sorted(video_dir.glob("output_*.mp4"))
        ts_files  = sorted(video_dir.glob("output_timestamps_session_*.txt"))

        if not mp4s or not ts_files:
            print(f"\n  [{run}] No video/timestamp files found, skipping")
            continue

        # Use camera 0
        mp4     = mp4s[0]
        ts_file = ts_files[0]
        print(f"\n  [{run}] Camera: {mp4.name} ({mp4.stat().st_size//1_000_000:.0f}MB)")

        frame_ts = load_video_timestamps(ts_file)
        if not frame_ts:
            print("    No timestamps loaded, skipping")
            continue

        # Optical flow
        print(f"    Optical flow...")
        try:
            flow_df = extract_optical_flow_features(mp4, frame_ts, subsample=3)
            print(f"    → {len(flow_df)} frames with flow features")
        except Exception as e:
            print(f"    Flow extraction failed: {e}")
            flow_df = pd.DataFrame()

        # Pendulum tracking
        print(f"    Pendulum tracking...")
        try:
            track_df = extract_pendulum_tracking_features(mp4, frame_ts, subsample=3)
            print(f"    → {len(track_df)} frames with tracking features")
        except Exception as e:
            print(f"    Tracking failed: {e}")
            track_df = pd.DataFrame()

        # Align to windows
        run_df = run_dfs[run]
        video_win_df = pd.DataFrame()

        if not flow_df.empty:
            af = align_frames_to_windows(flow_df, run_df[["win","dl_timestamp_ms"]], FLOW_COLS)
            if not af.empty:
                video_win_df = af if video_win_df.empty else video_win_df.merge(af, on="win", how="outer")
                print(f"    Aligned flow: {len(af)} windows")

        if not track_df.empty:
            at = align_frames_to_windows(track_df, run_df[["win","dl_timestamp_ms"]], TRACK_COLS)
            if not at.empty:
                video_win_df = at if video_win_df.empty else video_win_df.merge(at, on="win", how="outer")
                print(f"    Aligned tracking: {len(at)} windows")

        if video_win_df.empty:
            print("    No aligned video windows, skipping R²")
            continue

        # Merge with datalayer+trace data
        merged = run_df.merge(video_win_df, on="win", how="inner")

        # Filter to BALANCE state
        bal = merged[merged["dl_state"] == BALANCE_STATE].copy()
        video_cols = [c for c in FLOW_COLS + TRACK_COLS if c in bal.columns]
        phys_cols  = [c for c in PHYSICAL_COLS if c in bal.columns]
        vocab_avail = [c for c in vocab if c in bal.columns]

        bal = bal.dropna(subset=video_cols[:1])  # require at least one video feature
        print(f"    BALANCE windows with video: {len(bal)}")

        if len(bal) < 30 or len(vocab_avail) < 3:
            print(f"    Too few windows ({len(bal)}) or vocab cols ({len(vocab_avail)}), skipping")
            continue

        X_vid  = bal[video_cols].fillna(0.0).values
        X_phys = bal[phys_cols].fillna(0.0).values
        X_both = np.hstack([X_phys, X_vid])
        Y      = bal[vocab_avail].fillna(0.0).values

        rf_kwargs = dict(n_estimators=100, max_depth=12, n_jobs=N_JOBS, random_state=RANDOM_STATE)

        rf_vid = RandomForestRegressor(**rf_kwargs).fit(X_vid, Y)
        r2_vid = float(r2_score(Y, rf_vid.predict(X_vid), multioutput="uniform_average"))

        rf_phys = RandomForestRegressor(**rf_kwargs).fit(X_phys, Y)
        r2_phys = float(r2_score(Y, rf_phys.predict(X_phys), multioutput="uniform_average"))

        rf_both = RandomForestRegressor(**rf_kwargs).fit(X_both, Y)
        r2_both = float(r2_score(Y, rf_both.predict(X_both), multioutput="uniform_average"))

        run_results[run] = {
            "n_balance_windows": len(bal),
            "n_video_features":  len(video_cols),
            "n_vocab_branches":  len(vocab_avail),
            "r2_video_only":    r2_vid,
            "r2_physical_only": r2_phys,
            "r2_combined":      r2_both,
        }
        print(f"    R²: video={r2_vid:.4f} | physical={r2_phys:.4f} | combined={r2_both:.4f}")

        # Save merged df for later export
        vid_export_cols = ["win","run","dl_state","dl_timestamp_ms"] + phys_cols + video_cols + vocab_avail[:10]
        all_merged_dfs.append(merged[[c for c in vid_export_cols if c in merged.columns]])

    # ── Step 4: Save outputs ──────────────────────────────────────────────────
    print("\n[Step 4] Saving outputs...")

    if all_merged_dfs:
        combined = pd.concat(all_merged_dfs, ignore_index=True)
        combined.to_parquet(OUT / "video_features.parquet", index=False)
        print(f"  Saved video_features.parquet ({len(combined)} rows)")

    summary = {
        "per_run":           run_results,
        "vocab_size":        len(vocab),
        "mean_r2_video":     float(np.mean([v["r2_video_only"]    for v in run_results.values()])) if run_results else None,
        "mean_r2_physical":  float(np.mean([v["r2_physical_only"] for v in run_results.values()])) if run_results else None,
        "mean_r2_combined":  float(np.mean([v["r2_combined"]       for v in run_results.values()])) if run_results else None,
    }
    (OUT / "video_r2_summary.json").write_text(json.dumps(summary, indent=2))
    print("  Saved video_r2_summary.json")

    # Figure
    if run_results:
        runs  = list(run_results.keys())
        r2_v  = [run_results[r]["r2_video_only"]    for r in runs]
        r2_p  = [run_results[r]["r2_physical_only"] for r in runs]
        r2_b  = [run_results[r]["r2_combined"]      for r in runs]

        x = np.arange(len(runs))
        w = 0.25
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - w, r2_v, w, label="Video only (flow+tracking)", color="#90CAF9")
        ax.bar(x,     r2_p, w, label="Physical sensors only",       color="#1565C0")
        ax.bar(x + w, r2_b, w, label="Combined",                    color="#2E8B57")
        ax.set_xticks(x)
        ax.set_xticklabels([r[-8:] for r in runs], rotation=20, ha="right")
        ax.set_ylabel("R² (within BALANCE, train set)")
        ax.set_title("Video Features vs Physical Sensors → Branch Count Prediction")
        ax.legend()
        ax.axhline(0, color="grey", lw=0.5)
        fig.tight_layout()
        fig.savefig(OUT / "fig_video_rf_r2.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("  Saved fig_video_rf_r2.png")

    print("\nSummary:")
    for r, v in run_results.items():
        print(f"  {r[-8:]}: video={v['r2_video_only']:.4f} | phys={v['r2_physical_only']:.4f} | combined={v['r2_combined']:.4f}")
    if summary["mean_r2_video"] is not None:
        print(f"\n  Mean video R²:    {summary['mean_r2_video']:.4f}")
        print(f"  Mean physical R²: {summary['mean_r2_physical']:.4f}")
        print(f"  Mean combined R²: {summary['mean_r2_combined']:.4f}")


if __name__ == "__main__":
    main()
