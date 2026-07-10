"""
Phase 0: Reconnaissance — detect dataset parameters, build session manifest.

Outputs:
  outputs/video_assessment/session_manifest.csv
  outputs/video_assessment/dataset_params.json
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT      = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs/extracted")
OUT       = ROOT / "outputs/video_assessment"
OUT.mkdir(parents=True, exist_ok=True)

ALIGNED_PARQUET = ROOT / "outputs/phase3/aligned_dataset_expanded.parquet"
WIN_MS          = 10   # 10ms windows


# ── 0.1  Detect dataset parameters ────────────────────────────────────────────

def detect_params(df: pd.DataFrame) -> dict:
    # Physical features: dl_ prefix, excluding metadata-like cols
    meta_dl = {"dl_state", "dl_target_x", "dl_iteration"}
    physical_cols = [c for c in df.columns
                     if c.startswith("dl_") and c not in meta_dl]

    # Syslog features
    syslog_cols = [c for c in df.columns if c.startswith("sl_")]

    # Branch targets: colon-delimited numeric columns
    branch_cols = [c for c in df.columns if ":" in c]

    # Window rate from win column (win = t_us // 10000, so win*10 = ms)
    # Compute across a single session to avoid cross-session gaps
    sample_run = df[df["run"] == df["run"].iloc[0]].copy()
    wins = np.sort(sample_run["win"].values)
    diffs = np.diff(wins)
    median_diff_wins = float(np.median(diffs[diffs > 0]))
    # Each win unit = 10ms, so median diff of 1 → 10ms interval → 100 Hz
    rate_hz = 1000.0 / (median_diff_wins * WIN_MS)

    # Session list
    sessions = sorted(df["run"].unique().tolist())

    return {
        "n_sessions":        len(sessions),
        "sessions":          sessions,
        "n_windows_total":   len(df),
        "physical_cols":     physical_cols,
        "n_physical_cols":   len(physical_cols),
        "syslog_cols":       syslog_cols,
        "n_syslog_cols":     len(syslog_cols),
        "branch_cols":       branch_cols,
        "n_branch_cols":     len(branch_cols),
        "win_ms":            WIN_MS,
        "datalayer_rate_hz": rate_hz,
        "win_to_ms":         "win * 10 = Unix epoch ms",
    }


# ── 0.2  ffprobe helper ────────────────────────────────────────────────────────

def ffprobe_video(mp4_path: Path) -> dict:
    """Extract fps, resolution, duration, codec using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-select_streams", "v:0", str(mp4_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        info = json.loads(result.stdout)
        stream = info["streams"][0]
        # Parse FPS (fraction string like "30/1" or "125/1")
        fps_str = stream.get("r_frame_rate", "0/1")
        num, den = fps_str.split("/")
        fps = int(num) / int(den) if int(den) != 0 else 0.0
        return {
            "width":    int(stream.get("width", 0)),
            "height":   int(stream.get("height", 0)),
            "fps":      fps,
            "duration": float(stream.get("duration", 0.0)),
            "codec":    stream.get("codec_name", "unknown"),
            "n_frames": int(stream.get("nb_frames", -1)),
        }
    except Exception as e:
        return {"error": str(e)}


def detect_fps_from_timestamps(ts_path: Path) -> dict:
    """Fallback: compute FPS from camera timestamp file."""
    ts = []
    for line in ts_path.read_text().strip().split("\n"):
        parts = line.strip().split(":")
        if len(parts) == 2:
            try:
                ts.append(int(parts[1].strip()))
            except ValueError:
                pass
    if len(ts) < 2:
        return {"n_frames": len(ts), "fps": 0.0, "duration_s": 0.0}
    ts = np.array(ts)
    diffs = np.diff(ts)
    fps = 1000.0 / float(np.median(diffs)) if np.median(diffs) > 0 else 0.0
    return {
        "n_frames":    len(ts),
        "fps":         round(fps, 2),
        "duration_s":  round((ts[-1] - ts[0]) / 1000.0, 2),
        "ts_start_ms": int(ts[0]),
        "ts_end_ms":   int(ts[-1]),
    }


# ── 0.3  Build session manifest ────────────────────────────────────────────────

def build_manifest(sessions: list, df_full: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run in sessions:
        run_dir = DATA_ROOT / run

        # Datalayer
        dl_path   = run_dir / "datalayer" / "datalayer.csv"
        has_dl    = dl_path.exists()
        dl_rows   = None
        if has_dl:
            dl_rows = sum(1 for _ in open(dl_path)) - 1  # quick line count

        # Processed trace
        tr_path    = run_dir / "processed_trace" / "processed_trace.csv"
        has_trace  = tr_path.exists()

        # Video
        video_dir  = run_dir / "camera-video"
        has_video  = video_dir.exists() and bool(list(video_dir.glob("output_*.mp4")))
        mp4s       = sorted(video_dir.glob("output_*.mp4")) if has_video else []
        ts_files   = sorted(video_dir.glob("output_timestamps_session_*.txt")) if has_video else []

        # Per-camera info
        cameras = []
        for cam_idx, mp4 in enumerate(mp4s):
            cam_info = {"cam": cam_idx, "mp4": mp4.name, "size_mb": mp4.stat().st_size // 1_000_000}
            # Try ffprobe first
            fp_info = ffprobe_video(mp4)
            if "error" not in fp_info:
                cam_info.update(fp_info)
            elif cam_idx < len(ts_files):
                # Fall back to timestamp file
                ts_info = detect_fps_from_timestamps(ts_files[cam_idx])
                cam_info.update(ts_info)
            cameras.append(cam_info)

        # Datalayer windows in df_full
        run_df      = df_full[df_full["run"] == run] if run in df_full["run"].values else pd.DataFrame()
        n_windows   = len(run_df)
        n_balance   = int((run_df["dl_state"] == 1).sum()) if len(run_df) else 0

        # Alignment: overlap between video ts and datalayer win timestamps
        align_info  = {}
        if cameras and "ts_start_ms" in cameras[0] and n_windows > 0:
            wins       = run_df["win"].values
            dl_start   = int(wins.min()) * 10  # win * 10 = ms
            dl_end     = int(wins.max()) * 10
            vid_start  = cameras[0]["ts_start_ms"]
            vid_end    = cameras[0]["ts_end_ms"]
            overlap_s  = max(0, min(vid_end, dl_end) - max(vid_start, dl_start)) / 1000.0
            align_info = {
                "dl_start_ms":  dl_start,
                "dl_end_ms":    dl_end,
                "vid_start_ms": vid_start,
                "vid_end_ms":   vid_end,
                "overlap_s":    round(overlap_s, 1),
                "offset_ms":    vid_start - dl_start,  # positive = video starts after DL
            }

        row = {
            "run":          run,
            "has_video":    has_video,
            "has_dl":       has_dl,
            "has_trace":    has_trace,
            "dl_rows":      dl_rows,
            "n_windows":    n_windows,
            "n_balance":    n_balance,
            "n_cameras":    len(mp4s),
        }
        if cameras:
            row["fps"]        = cameras[0].get("fps",       None)
            row["width"]      = cameras[0].get("width",     None)
            row["height"]     = cameras[0].get("height",    None)
            row["n_frames"]   = cameras[0].get("n_frames",  None)
            row["duration_s"] = cameras[0].get("duration_s", None)
        row.update(align_info)
        rows.append(row)

    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 0: Reconnaissance")
    print("=" * 60)

    # Load aligned dataset
    print(f"\nLoading {ALIGNED_PARQUET.name}...")
    df = pd.read_parquet(ALIGNED_PARQUET)
    print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # Detect parameters
    params = detect_params(df)
    print(f"\nDetected parameters:")
    print(f"  Sessions:         {params['n_sessions']}")
    print(f"  Total windows:    {params['n_windows_total']:,}")
    print(f"  Physical cols:    {params['n_physical_cols']} → {params['physical_cols']}")
    print(f"  Syslog cols:      {params['n_syslog_cols']} → {params['syslog_cols']}")
    print(f"  Branch targets:   {params['n_branch_cols']}")
    print(f"  Datalayer rate:   {params['datalayer_rate_hz']:.1f} Hz ({WIN_MS}ms windows)")

    # Build session manifest
    print(f"\nBuilding session manifest for {len(params['sessions'])} sessions...")
    manifest = build_manifest(params["sessions"], df)

    video_sessions = manifest[manifest["has_video"]]["run"].tolist()
    full_sessions  = manifest[manifest["has_video"] & manifest["has_dl"] & manifest["has_trace"]]["run"].tolist()

    print(f"\nSession summary:")
    print(f"  Total sessions:                {len(manifest)}")
    print(f"  Sessions with video:           {manifest['has_video'].sum()}")
    print(f"  Sessions with video+DL+trace:  {len(full_sessions)}")
    print(f"\nVideo sessions:")
    for run in video_sessions:
        row = manifest[manifest["run"] == run].iloc[0]
        fps_str   = f"{row.get('fps', '?'):.0f}" if pd.notna(row.get("fps")) else "?"
        overlap   = f"{row.get('overlap_s', '?'):.0f}s" if pd.notna(row.get("overlap_s")) else "?"
        n_frames  = int(row.get("n_frames", 0)) if pd.notna(row.get("n_frames")) else "?"
        print(f"  {run}: fps={fps_str}, frames={n_frames}, overlap={overlap}, "
              f"balance={int(row.get('n_balance',0)):,}")

    # Alignment parameters
    print(f"\nAlignment parameters:")
    if video_sessions:
        sample     = manifest[manifest["run"] == video_sessions[0]].iloc[0]
        sample_fps = sample.get("fps", 0) or 0
        dl_hz      = params["datalayer_rate_hz"]
        if sample_fps > 0 and dl_hz > 0:
            ratio = dl_hz / sample_fps
            if ratio > 1:
                method = f"nearest-frame ({ratio:.1f} datalayer windows per video frame)"
            else:
                method = f"subsample ({1/ratio:.1f} video frames per datalayer window)"
            print(f"  Video FPS: {sample_fps:.1f} Hz")
            print(f"  Datalayer: {dl_hz:.1f} Hz")
            print(f"  Alignment: {method}")

    # Save outputs
    manifest.to_csv(OUT / "session_manifest.csv", index=False)
    print(f"\nSaved session_manifest.csv ({len(manifest)} rows)")

    # Add useful lists to params
    params["video_sessions"]      = video_sessions
    params["full_video_sessions"] = full_sessions
    (OUT / "dataset_params.json").write_text(json.dumps(params, indent=2))
    print("Saved dataset_params.json")

    print("\nPhase 0 complete.")


if __name__ == "__main__":
    main()
