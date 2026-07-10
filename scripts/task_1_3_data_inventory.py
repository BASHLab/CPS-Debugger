"""
task_1_3_data_inventory.py — Full dataset inventory.

Scans all known data locations to find available runs, their modalities,
and extraction status.

Output:
  outputs/phase3/data_inventory.csv
  stdout summary
"""

import json
import subprocess
import tarfile
from pathlib import Path

import pandas as pd

# ── Search roots ──────────────────────────────────────────────────────────────
EXTRACT_ROOT  = Path("/home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs/extracted")
TARBALL_DIR25 = Path("/home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs/2025")
TARBALL_ROOT  = Path("/home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs")
CPS_DATA      = Path("/home/simran/allspark-data-exploration/CPS-Debugger/data")
P3_OUT        = Path("/home/simran/allspark-data-exploration/CPS-Debugger/outputs/phase3")
P3_OUT.mkdir(parents=True, exist_ok=True)


def check_run_dir(run_dir: Path) -> dict:
    """Assess an extracted run directory for available modalities."""
    record = {
        "run_id":       run_dir.name,
        "path":         str(run_dir),
        "has_datalayer": False,
        "has_trace":    False,
        "has_syslog":   False,
        "has_audio":    False,
        "has_video":    False,
        "has_code":     False,
        "n_datalayer_rows": 0,
        "n_trace_rows": 0,
        "status":       "UNKNOWN",
    }

    dl = run_dir / "datalayer" / "datalayer.csv"
    tr = run_dir / "processed_trace" / "processed_trace.csv"
    sl = run_dir / "syslog" / "syslog.csv"

    if dl.exists():
        record["has_datalayer"] = True
        try:
            with open(dl) as f:
                record["n_datalayer_rows"] = sum(1 for _ in f) - 1
        except Exception:
            pass
    if tr.exists():
        record["has_trace"] = True
        try:
            with open(tr) as f:
                record["n_trace_rows"] = sum(1 for _ in f) - 1
        except Exception:
            pass
    if sl.exists():
        record["has_syslog"] = True

    # Audio: any wav files
    audio_dir = run_dir / "audio"
    if audio_dir.exists() and any(audio_dir.glob("*.wav")):
        record["has_audio"] = True

    # Video: any mp4 files anywhere under run_dir
    if list(run_dir.rglob("*.mp4")):
        record["has_video"] = True

    # Code layout.json
    if (run_dir / "code" / "layout.json").exists():
        record["has_code"] = True

    # Determine status
    core = record["has_datalayer"] and record["has_trace"] and record["has_syslog"]
    if core and record["has_audio"] and record["has_video"]:
        record["status"] = "COMPLETE+ALL"
    elif core and record["has_audio"]:
        record["status"] = "COMPLETE+AUDIO"
    elif core and record["has_video"]:
        record["status"] = "COMPLETE+VIDEO"
    elif core:
        record["status"] = "COMPLETE"
    elif record["has_trace"] and not record["has_datalayer"]:
        record["status"] = "PARTIAL_NO_DATALAYER"
    else:
        record["status"] = "PARTIAL"

    return record


def peek_tarball(tb: Path) -> dict:
    """Peek inside a tarball to see what run it contains."""
    record = {
        "run_id":   tb.stem.replace(".tar", ""),
        "path":     str(tb),
        "status":   "UNEXTRACTED",
        "size_mb":  round(tb.stat().st_size / 1e6, 1),
        "has_datalayer": None,
        "has_trace": None,
        "has_syslog": None,
        "has_audio": None,
        "has_video": None,
        "has_code":  None,
        "n_datalayer_rows": 0,
        "n_trace_rows": 0,
    }
    try:
        with tarfile.open(tb, "r:gz") as tf:
            names = tf.getnames()
            record["has_datalayer"] = any("datalayer.csv" in n for n in names)
            record["has_trace"]     = any("processed_trace.csv" in n for n in names)
            record["has_syslog"]    = any("syslog.csv" in n for n in names)
            record["has_audio"]     = any(n.endswith(".wav") for n in names)
            record["has_video"]     = any(n.endswith(".mp4") for n in names)
            record["has_code"]      = any("layout.json" in n for n in names)
    except Exception as e:
        record["status"] = f"TARBALL_ERROR: {e}"
    return record


def main():
    records = []

    # ── 1. Extracted runs ─────────────────────────────────────────────────────
    print("Scanning extracted runs...")
    if EXTRACT_ROOT.exists():
        for run_dir in sorted(EXTRACT_ROOT.iterdir()):
            if run_dir.is_dir() and run_dir.name.startswith("20"):
                rec = check_run_dir(run_dir)
                records.append(rec)
                print(f"  {rec['run_id']:30s} {rec['status']:20s} "
                      f"dl={rec['n_datalayer_rows']:6d} tr={rec['n_trace_rows']:6d}")

    # ── 2. Extracted dirs inside TARBALL_DIR25 ────────────────────────────────
    print("\nScanning 2025/ for extracted dirs...")
    if TARBALL_DIR25.exists():
        for item in sorted(TARBALL_DIR25.iterdir()):
            if item.is_dir() and item.name.startswith("20") and not item.name.endswith("videos"):
                # Check if it's a valid run dir (has datalayer/trace subdirs)
                run_ids_seen = {r["run_id"] for r in records}
                if item.name not in run_ids_seen:
                    rec = check_run_dir(item)
                    records.append(rec)
                    print(f"  {rec['run_id']:30s} {rec['status']:20s}")

    # ── 3. Video-only dirs ────────────────────────────────────────────────────
    print("\nScanning for video-only dirs...")
    for search_root in [TARBALL_DIR25, TARBALL_ROOT]:
        if not search_root.exists():
            continue
        for item in sorted(search_root.rglob("*.mp4")):
            run_dir = item.parent.parent  # mp4 is usually in a subdir
            run_ids_seen = {r["run_id"] for r in records}
            if run_dir.name not in run_ids_seen and run_dir.name.startswith("20"):
                rec = {
                    "run_id": run_dir.name,
                    "path": str(run_dir),
                    "has_datalayer": False,
                    "has_trace": False,
                    "has_syslog": False,
                    "has_audio": False,
                    "has_video": True,
                    "has_code": False,
                    "n_datalayer_rows": 0,
                    "n_trace_rows": 0,
                    "status": "VIDEO_ONLY",
                }
                records.append(rec)
                print(f"  {rec['run_id']:30s} VIDEO_ONLY")

    # ── 4. CPS-Debugger/data/ (Feb04 partial run) ─────────────────────────────
    print("\nScanning CPS-Debugger/data/...")
    if CPS_DATA.exists():
        for run_dir in sorted(CPS_DATA.iterdir()):
            if run_dir.is_dir() and run_dir.name.startswith("20"):
                run_ids_seen = {r["run_id"] for r in records}
                if run_dir.name not in run_ids_seen:
                    rec = check_run_dir(run_dir)
                    records.append(rec)
                    print(f"  {rec['run_id']:30s} {rec['status']:20s}")

    # ── 5. Tarballs ───────────────────────────────────────────────────────────
    print("\nScanning tarballs...")
    all_tarball_dirs = [TARBALL_ROOT, TARBALL_DIR25]
    for tbd in all_tarball_dirs:
        if not tbd.exists():
            continue
        for tb in sorted(tbd.glob("*.tar.gz")):
            run_id = tb.stem.replace(".tar", "")
            run_ids_seen = {r["run_id"] for r in records}
            if run_id not in run_ids_seen:
                print(f"  Peeking {tb.name} ({tb.stat().st_size/1e6:.0f} MB)...", end=" ", flush=True)
                rec = peek_tarball(tb)
                records.append(rec)
                print(f"{rec['status']}  dl={rec['has_datalayer']} tr={rec['has_trace']}")

    # ── 6. Report 2025.zip ────────────────────────────────────────────────────
    big_zip = TARBALL_ROOT / "2025.zip"
    if big_zip.exists():
        size_gb = big_zip.stat().st_size / 1e9
        print(f"\n2025.zip: {size_gb:.1f} GB — requires extraction "
              f"(corrupt central directory; use streaming extraction)")
        records.append({
            "run_id": "2025.zip_ARCHIVE",
            "path": str(big_zip),
            "status": f"ARCHIVE_NEEDS_EXTRACTION_{size_gb:.1f}GB",
            "has_datalayer": None, "has_trace": None, "has_syslog": None,
            "has_audio": None, "has_video": None, "has_code": None,
            "n_datalayer_rows": 0, "n_trace_rows": 0,
        })

    # ── Save and summarize ────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    df.to_csv(P3_OUT / "data_inventory.csv", index=False)
    print(f"\nSaved data_inventory.csv ({len(df)} entries)")

    print("\n── SUMMARY ──")
    status_counts = df["status"].value_counts()
    for status, cnt in status_counts.items():
        print(f"  {status:40s}: {cnt}")

    complete = df[df["status"].str.startswith("COMPLETE")]
    total_dl_rows = complete["n_datalayer_rows"].sum()
    approx_windows = total_dl_rows // 10  # ~10 datalayer rows per 10ms window (~1kHz)
    print(f"\n  Complete runs: {len(complete)}")
    print(f"  Total datalayer rows: {total_dl_rows:,}")
    print(f"  Approx 10ms windows: {approx_windows:,}")
    print(f"  Runs with audio:  {complete['has_audio'].sum()}")
    print(f"  Runs with video:  {complete['has_video'].sum()}")

    unextracted = df[df["status"] == "UNEXTRACTED"]
    if len(unextracted):
        total_mb = unextracted.get("size_mb", pd.Series(dtype=float)).sum() if "size_mb" in df.columns else 0
        print(f"\n  Unextracted tarballs: {len(unextracted)} ({total_mb:.0f} MB total)")
        for _, row in unextracted.iterrows():
            print(f"    {row['run_id']:30s}  dl={row['has_datalayer']} tr={row['has_trace']}")

    # Save structured summary JSON
    summary = {
        "n_complete_runs": int(len(complete)),
        "n_unextracted_tarballs": int(len(unextracted)),
        "approx_10ms_windows_complete": int(approx_windows),
        "complete_with_audio": int(complete["has_audio"].sum() if len(complete) else 0),
        "complete_with_video": int(complete["has_video"].sum() if len(complete) else 0),
        "complete_run_ids": complete["run_id"].tolist(),
        "unextracted_run_ids": unextracted["run_id"].tolist() if len(unextracted) else [],
    }
    (P3_OUT / "data_inventory_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nDone.")


if __name__ == "__main__":
    main()
