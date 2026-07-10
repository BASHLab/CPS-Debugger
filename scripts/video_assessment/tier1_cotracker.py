"""
Tier 1: CoTracker3 — explicit point trajectory extraction.

Tracks a 7×7 grid of points across frames.
Per-frame features:
  ct_mean_x, ct_mean_y, ct_std_x, ct_std_y,
  ct_vel_x, ct_vel_y, ct_vel_mag,
  ct_spread_y, ct_top_y, ct_bottom_y, ct_visibility_frac

Usage (one session per job):
  python tier1_cotracker.py --run 2025-03-17_10-51-58

Outputs:
  outputs/video_assessment/features/tier1_{run_id}.parquet
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT      = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs/extracted")
OUT_DIR   = ROOT / "outputs/video_assessment/features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GRID_SIZE   = 7      # 7×7 = 49 tracked points
CHUNK_FRAMES = 300   # process in chunks to manage memory
RESIZE_W    = 640
RESIZE_H    = 480


def load_timestamps(ts_path: Path) -> dict:
    frame_ts = {}
    for line in ts_path.read_text().strip().split("\n"):
        parts = line.strip().split(":")
        if len(parts) == 2:
            try:
                frame_ts[int(parts[0].strip())] = int(parts[1].strip())
            except ValueError:
                pass
    return frame_ts


def build_query_points(grid_size: int, W: int, H: int) -> torch.Tensor:
    """Create grid_size x grid_size query points uniformly placed across frame."""
    xs = torch.linspace(0.1 * W, 0.9 * W, grid_size)
    ys = torch.linspace(0.1 * H, 0.9 * H, grid_size)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    # queries: [N, 3] = (t=0, x, y)
    N      = grid_size * grid_size
    t_zero = torch.zeros(N, 1)
    coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
    return torch.cat([t_zero, coords], dim=1).unsqueeze(0)  # [1, N, 3]


def load_video_frames(mp4_path: Path, start: int, end: int) -> tuple:
    """Load frames [start, end) from video. Returns (frames_tensor, frame_indices)."""
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {mp4_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    indices = []
    for i in range(start, end):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(cv2.resize(frame, (RESIZE_W, RESIZE_H)), cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        indices.append(i)
    cap.release()

    if not frames:
        return None, []
    # [T, H, W, C] → [1, T, C, H, W] float [0,1]
    arr    = np.stack(frames).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0)
    return tensor, indices


def extract_frame_features(tracks: np.ndarray, visibility: np.ndarray) -> list:
    """
    tracks:     [T, N, 2] (x, y in pixel coords)
    visibility: [T, N]    bool
    Returns list of T dicts, one per frame.
    """
    T, N, _ = tracks.shape
    records  = []

    for t in range(T):
        vis_mask = visibility[t]  # [N]
        n_vis    = int(vis_mask.sum())

        if n_vis > 0:
            pts   = tracks[t][vis_mask]   # [n_vis, 2]
            mean_x = float(pts[:, 0].mean()) / RESIZE_W
            mean_y = float(pts[:, 1].mean()) / RESIZE_H
            std_x  = float(pts[:, 0].std())  / RESIZE_W  if n_vis > 1 else 0.0
            std_y  = float(pts[:, 1].std())  / RESIZE_H  if n_vis > 1 else 0.0
            spread_y = float(pts[:, 1].max() - pts[:, 1].min()) / RESIZE_H
            top_y    = float(pts[:, 1].min()) / RESIZE_H
            bottom_y = float(pts[:, 1].max()) / RESIZE_H
        else:
            mean_x = mean_y = std_x = std_y = spread_y = top_y = bottom_y = float("nan")

        if t > 0:
            prev_vis = visibility[t-1]
            both_vis = vis_mask & prev_vis
            if both_vis.sum() > 0:
                dx      = tracks[t][both_vis, 0] - tracks[t-1][both_vis, 0]
                dy      = tracks[t][both_vis, 1] - tracks[t-1][both_vis, 1]
                vel_x   = float(dx.mean()) / RESIZE_W
                vel_y   = float(dy.mean()) / RESIZE_H
                vel_mag = float(np.sqrt(dx**2 + dy**2).mean()) / RESIZE_W
            else:
                vel_x = vel_y = vel_mag = float("nan")
        else:
            vel_x = vel_y = vel_mag = float("nan")

        records.append({
            "ct_mean_x":          mean_x,
            "ct_mean_y":          mean_y,
            "ct_std_x":           std_x,
            "ct_std_y":           std_y,
            "ct_vel_x":           vel_x,
            "ct_vel_y":           vel_y,
            "ct_vel_mag":         vel_mag,
            "ct_spread_y":        spread_y,
            "ct_top_y":           top_y,
            "ct_bottom_y":        bottom_y,
            "ct_visibility_frac": float(n_vis) / N,
        })

    return records


def process_session(run_id: str) -> bool:
    out_path = OUT_DIR / f"tier1_{run_id}.parquet"
    if out_path.exists():
        print(f"  [{run_id}] Already exists, skipping.")
        return True

    run_dir   = DATA_ROOT / run_id
    video_dir = run_dir / "camera-video"
    mp4s      = sorted(video_dir.glob("output_*.mp4"))
    ts_files  = sorted(video_dir.glob("output_timestamps_session_*.txt"))

    if not mp4s or not ts_files:
        print(f"  [{run_id}] Missing video/timestamp files.")
        return False

    mp4     = mp4s[0]
    ts_file = ts_files[0]
    print(f"  [{run_id}] {mp4.name} ({mp4.stat().st_size//1_000_000:.0f}MB)")

    # Total frames
    cap        = cv2.VideoCapture(str(mp4))
    n_total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"    Total frames: {n_total}")

    frame_ts = load_timestamps(ts_file)

    # Load CoTracker3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    Loading CoTracker3 on {device}...")
    try:
        cotracker = torch.hub.load(
            "facebookresearch/co-tracker", "cotracker3_offline",
            trust_repo=True
        ).to(device)
        cotracker.eval()
    except Exception as e:
        print(f"    FAILED to load CoTracker3: {e}")
        return False

    # Build query points (first frame coordinates)
    queries = build_query_points(GRID_SIZE, RESIZE_W, RESIZE_H).to(device)

    # Process in chunks
    all_records   = []
    all_frame_idx = []

    for chunk_start in range(0, n_total, CHUNK_FRAMES):
        chunk_end = min(chunk_start + CHUNK_FRAMES, n_total)
        print(f"    Chunk [{chunk_start}, {chunk_end}) of {n_total}...", end=" ", flush=True)

        video_tensor, frame_indices = load_video_frames(mp4, chunk_start, chunk_end)
        if video_tensor is None:
            print("empty")
            break

        video_tensor = video_tensor.to(device)

        # Use init queries only for the first chunk; subsequent chunks initialize
        # from the last known point positions (simplification: re-use grid queries)
        with torch.no_grad():
            try:
                pred_tracks, pred_visibility = cotracker(
                    video_tensor,
                    queries=queries
                )
                # pred_tracks:     [1, T, N, 2]
                # pred_visibility: [1, T, N]
                tracks     = pred_tracks[0].cpu().numpy()       # [T, N, 2]
                visibility = pred_visibility[0].cpu().numpy()   # [T, N] bool
            except Exception as e:
                print(f"inference error: {e}")
                break

        chunk_records = extract_frame_features(tracks, visibility)
        all_records.extend(chunk_records)
        all_frame_idx.extend(frame_indices)
        print(f"{len(chunk_records)} frames")

    if not all_records:
        print(f"    No records extracted.")
        return False

    df = pd.DataFrame(all_records)
    df.insert(0, "frame_idx",    all_frame_idx)
    df.insert(1, "timestamp_ms", [frame_ts.get(i, -1) for i in all_frame_idx])

    df.to_parquet(out_path, index=False)
    print(f"    Saved → {out_path.name} ({len(df)} rows)")

    valid = df[df["timestamp_ms"] > 0]
    summary = {
        "run_id":     run_id,
        "n_frames":   len(df),
        "n_valid_ts": int(len(valid)),
        "grid_size":  GRID_SIZE,
        "n_points":   GRID_SIZE * GRID_SIZE,
        "features":   [c for c in df.columns if c not in {"frame_idx","timestamp_ms"}],
    }
    (OUT_DIR / f"tier1_{run_id}_meta.json").write_text(json.dumps(summary, indent=2))
    return True


def main():
    parser = argparse.ArgumentParser(description="Tier 1: CoTracker3 extraction")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run",  help="Single run ID")
    group.add_argument("--all",  action="store_true", help="All video sessions")
    args   = parser.parse_args()

    print("=" * 60)
    print("Tier 1: CoTracker3 Feature Extraction")
    print("=" * 60)

    params_path = ROOT / "outputs/video_assessment/dataset_params.json"
    if args.all:
        if params_path.exists():
            sessions = json.loads(params_path.read_text()).get("full_video_sessions", [])
        else:
            sessions = sorted(d.name for d in DATA_ROOT.iterdir()
                              if (d/"camera-video").exists() and (d/"datalayer").exists())
        print(f"Processing {len(sessions)} sessions")
    else:
        sessions = [args.run]

    for run in sessions:
        process_session(run)

    print("\nTier 1 complete.")


if __name__ == "__main__":
    main()
