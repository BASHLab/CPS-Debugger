"""
task_2_1a_video_inventory.py — Video-specific inventory.

Finds all runs with video, counts frames, checks alignment with datalayer,
and identifies the best candidate for DINOv2 analysis.

Output:
  outputs/phase3/video_inventory.json
"""

import json
from pathlib import Path

import numpy as np

ROOT     = Path("/home/simran/allspark-data-exploration")
DATA_ROOT = ROOT / "Pittsburgh-pendulum-datalogs"
P3_OUT   = ROOT / "CPS-Debugger/outputs/phase3"
P3_OUT.mkdir(parents=True, exist_ok=True)

# All places to look for video
SEARCH_ROOTS = [
    DATA_ROOT / "extracted",
    DATA_ROOT / "2025",
    DATA_ROOT,
    ROOT / "CPS-Debugger/data",
]


def count_video_frames(mp4_path: Path) -> int:
    """Count frames in an MP4 using OpenCV. Returns -1 if cv2 unavailable."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(mp4_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return n, fps
    except ImportError:
        # Fallback: estimate from file size (rough)
        return -1, -1
    except Exception:
        return -1, -1


def find_datalayer(run_dir: Path):
    """Try to find datalayer.csv for a run directory."""
    candidates = [
        run_dir / "datalayer" / "datalayer.csv",
        run_dir.parent / run_dir.name / "datalayer" / "datalayer.csv",
    ]
    # Also look in extracted/
    extracted = DATA_ROOT / "extracted" / run_dir.name / "datalayer" / "datalayer.csv"
    candidates.append(extracted)
    for c in candidates:
        if c.exists():
            return c
    return None


def find_trace(run_dir: Path):
    """Try to find processed_trace.csv for a run directory."""
    candidates = [
        run_dir / "processed_trace" / "processed_trace.csv",
        DATA_ROOT / "extracted" / run_dir.name / "processed_trace" / "processed_trace.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def main():
    print("Scanning for video files...")
    video_runs = {}  # run_id → info dict

    for search_root in SEARCH_ROOTS:
        if not search_root.exists():
            continue
        for mp4 in sorted(search_root.rglob("*.mp4")):
            # Determine run_id from directory name
            # Look for timestamp-like name in path components
            run_id = None
            for part in mp4.parts[::-1]:
                if len(part) >= 16 and part.startswith("20") and "-" in part:
                    run_id = part
                    break
            if run_id is None:
                run_id = mp4.parent.parent.name

            if run_id not in video_runs:
                video_runs[run_id] = {
                    "run_id": run_id,
                    "mp4_files": [],
                    "total_frames": 0,
                    "fps": -1.0,
                    "duration_s": -1.0,
                    "has_datalayer": False,
                    "has_trace": False,
                    "datalayer_rows": 0,
                    "trace_rows": 0,
                }

            n_frames, fps = count_video_frames(mp4)
            video_runs[run_id]["mp4_files"].append({
                "path": str(mp4),
                "n_frames": n_frames,
                "fps": fps,
            })
            if n_frames > 0:
                video_runs[run_id]["total_frames"] += n_frames
                video_runs[run_id]["fps"] = fps
                if fps > 0:
                    video_runs[run_id]["duration_s"] = n_frames / fps

    # Check datalayer/trace availability for each video run
    print(f"\nFound {len(video_runs)} runs with video. Checking modalities...")
    for run_id, info in video_runs.items():
        # Try to find run directory
        for root in SEARCH_ROOTS:
            run_dir = root / run_id
            if run_dir.exists():
                dl = find_datalayer(run_dir)
                tr = find_trace(run_dir)
                if dl:
                    info["has_datalayer"] = True
                    try:
                        with open(dl) as f:
                            info["datalayer_rows"] = sum(1 for _ in f) - 1
                    except Exception:
                        pass
                if tr:
                    info["has_trace"] = True
                    try:
                        with open(tr) as f:
                            info["trace_rows"] = sum(1 for _ in f) - 1
                    except Exception:
                        pass
                break

        print(f"  {run_id:30s}  frames={info['total_frames']:6d}  "
              f"fps={info['fps']:.1f}  dur={info['duration_s']:.1f}s  "
              f"dl={info['has_datalayer']}  tr={info['has_trace']}")

    # ── Identify best candidate for DINOv2 ───────────────────────────────────
    # Prefer: has datalayer + trace + max frames
    def score(info):
        s = info["total_frames"]
        if info["has_datalayer"]:
            s += 100000
        if info["has_trace"]:
            s += 100000
        return s

    best = max(video_runs.values(), key=score) if video_runs else None

    print("\n── Video Inventory Summary ──")
    print(f"  Total runs with video: {len(video_runs)}")
    if best:
        print(f"  Best candidate for DINOv2: {best['run_id']}")
        print(f"    Frames: {best['total_frames']}, Duration: {best['duration_s']:.1f}s")
        print(f"    Has datalayer: {best['has_datalayer']}, Has trace: {best['has_trace']}")
        if not best["has_datalayer"]:
            print("    WARNING: Best video run has no datalayer — "
                  "DINOv2 analysis will be visual only, cannot correlate with trace.")

    # Save JSON
    out = {
        "n_runs_with_video": len(video_runs),
        "best_dino_candidate": best["run_id"] if best else None,
        "best_candidate_info": best,
        "all_runs": list(video_runs.values()),
    }
    (P3_OUT / "video_inventory.json").write_text(json.dumps(out, indent=2))
    print("\nSaved video_inventory.json")
    print("Done.")


if __name__ == "__main__":
    main()
