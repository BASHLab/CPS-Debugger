"""
Tier 0: Optical flow feature extraction (CPU-only, Farneback).

Per-frame on 320×240 resized frames:
  flow_mag_mean, flow_mag_std, flow_x_mean, flow_y_mean,
  flow_x_std, flow_y_std, motion_centroid_x, motion_centroid_y,
  motion_area_frac

Run one session at a time:
  python tier0_optical_flow.py --run 2025-03-17_10-51-58
  python tier0_optical_flow.py --all   (process all video sessions)

Outputs (per session):
  outputs/video_assessment/features/tier0_{run_id}.parquet
  columns: frame_idx, timestamp_ms, flow_mag_mean, ..., motion_area_frac
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT      = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs/extracted")
OUT_DIR   = ROOT / "outputs/video_assessment/features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESIZE_W, RESIZE_H = 320, 240
MOTION_THRESH      = 1.0   # px/frame magnitude threshold for "motion" mask


def load_timestamps(ts_path: Path) -> dict:
    """Parse output_timestamps_session_*.txt → {frame_idx: timestamp_ms}."""
    frame_ts = {}
    for line in ts_path.read_text().strip().split("\n"):
        parts = line.strip().split(":")
        if len(parts) == 2:
            try:
                frame_ts[int(parts[0].strip())] = int(parts[1].strip())
            except ValueError:
                pass
    return frame_ts


def extract_flow(mp4_path: Path, frame_ts: dict) -> pd.DataFrame:
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {mp4_path}")

    prev_gray = None
    records   = []
    frame_idx = 0
    total_pixels = RESIZE_W * RESIZE_H

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, (RESIZE_W, RESIZE_H))
        gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            fx, fy = flow[..., 0], flow[..., 1]
            mag    = np.sqrt(fx**2 + fy**2)

            # Motion mask: pixels with significant motion
            motion_mask  = mag > MOTION_THRESH
            n_motion     = motion_mask.sum()
            area_frac    = float(n_motion) / total_pixels

            if n_motion > 0:
                ys, xs = np.where(motion_mask)
                centroid_x = float(xs.mean()) / RESIZE_W   # [0,1]
                centroid_y = float(ys.mean()) / RESIZE_H   # [0,1]
            else:
                centroid_x = centroid_y = 0.5

            records.append({
                "frame_idx":        frame_idx,
                "timestamp_ms":     frame_ts.get(frame_idx, -1),
                "flow_mag_mean":    float(mag.mean()),
                "flow_mag_std":     float(mag.std()),
                "flow_x_mean":      float(fx.mean()),
                "flow_y_mean":      float(fy.mean()),
                "flow_x_std":       float(fx.std()),
                "flow_y_std":       float(fy.std()),
                "motion_centroid_x": centroid_x,
                "motion_centroid_y": centroid_y,
                "motion_area_frac": area_frac,
            })

        prev_gray  = gray
        frame_idx += 1

    cap.release()
    return pd.DataFrame(records)


def process_session(run_id: str) -> bool:
    out_path = OUT_DIR / f"tier0_{run_id}.parquet"
    if out_path.exists():
        print(f"  [{run_id}] Already exists, skipping.")
        return True

    run_dir    = DATA_ROOT / run_id
    video_dir  = run_dir / "camera-video"
    mp4s       = sorted(video_dir.glob("output_*.mp4"))
    ts_files   = sorted(video_dir.glob("output_timestamps_session_*.txt"))

    if not mp4s:
        print(f"  [{run_id}] No MP4 found.")
        return False
    if not ts_files:
        print(f"  [{run_id}] No timestamp file found.")
        return False

    mp4     = mp4s[0]   # camera 0
    ts_file = ts_files[0]

    print(f"  [{run_id}] Processing {mp4.name} ({mp4.stat().st_size//1_000_000:.0f}MB)...")
    frame_ts = load_timestamps(ts_file)
    print(f"    Timestamps: {len(frame_ts)} frames")

    df = extract_flow(mp4, frame_ts)
    valid = df[df["timestamp_ms"] > 0]
    print(f"    Extracted: {len(df)} flow frames ({len(valid)} with valid timestamps)")

    df.to_parquet(out_path, index=False)
    print(f"    Saved → {out_path.name} ({len(df)} rows)")

    # Summary stats
    feat_cols = [c for c in df.columns if c not in {"frame_idx","timestamp_ms"}]
    summary   = {
        "run_id":        run_id,
        "n_frames":      len(df),
        "n_valid_ts":    int((df["timestamp_ms"] > 0).sum()),
        "features":      feat_cols,
    }
    if len(valid) > 0:
        ts = valid["timestamp_ms"].values
        fps = 1000.0 / float(np.median(np.diff(ts))) if len(ts) > 1 else 0.0
        summary.update({
            "fps":        round(fps, 2),
            "ts_start":   int(ts[0]),
            "ts_end":     int(ts[-1]),
            "duration_s": round((ts[-1]-ts[0])/1000.0, 1),
        })
    (OUT_DIR / f"tier0_{run_id}_meta.json").write_text(json.dumps(summary, indent=2))
    return True


def get_video_sessions() -> list:
    params_path = ROOT / "outputs/video_assessment/dataset_params.json"
    if params_path.exists():
        params = json.loads(params_path.read_text())
        return params.get("full_video_sessions", params.get("video_sessions", []))
    # Fallback: scan extracted dir
    sessions = []
    for d in sorted(DATA_ROOT.iterdir()):
        if (d / "camera-video").exists() and (d / "datalayer").exists() and (d / "processed_trace").exists():
            sessions.append(d.name)
    return sessions


def main():
    parser = argparse.ArgumentParser(description="Tier 0: Optical flow extraction")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run",  help="Single run ID to process")
    group.add_argument("--all",  action="store_true", help="Process all video sessions")
    args   = parser.parse_args()

    print("=" * 60)
    print("Tier 0: Optical Flow Feature Extraction")
    print("=" * 60)

    if args.all:
        sessions = get_video_sessions()
        print(f"Found {len(sessions)} video session(s): {sessions}")
    else:
        sessions = [args.run]

    for run in sessions:
        process_session(run)

    print("\nTier 0 complete.")


if __name__ == "__main__":
    main()
