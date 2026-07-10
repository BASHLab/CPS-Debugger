"""
task_1_3b_expand_preprocess.py — Preprocess all valid runs into an expanded dataset.

Reads data_inventory.csv from task_1_3, collects all COMPLETE runs,
and runs the existing 00_preprocess.py pipeline on them. Uses the same
100-branch vocabulary from the original 6-run analysis for consistent targets.

Output:
  outputs/phase3/aligned_dataset_expanded.parquet
  outputs/phase3/expand_preprocess_summary.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT    = Path("/home/simran/allspark-data-exploration/CPS-Debugger")
P3_OUT  = ROOT / "outputs/phase3"
P3_OUT.mkdir(parents=True, exist_ok=True)

# Add scripts dir to path for imports from 00_preprocess.py
sys.path.insert(0, str(ROOT / "scripts"))

DATA_ROOT = Path("/home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs/extracted")
VOCAB_PATH = ROOT / "outputs/experiments/vocab_info.json"


def main():
    # ── Load inventory ────────────────────────────────────────────────────────
    inv_path = P3_OUT / "data_inventory.csv"
    if not inv_path.exists():
        print("ERROR: data_inventory.csv not found. Run task_1_3_data_inventory.py first.")
        sys.exit(1)

    inv = pd.read_csv(inv_path)
    complete = inv[inv["status"].str.startswith("COMPLETE")]
    print(f"Found {len(complete)} complete runs in inventory")

    # Resolve actual run directories
    run_dirs = []
    for _, row in complete.iterrows():
        path = Path(row["path"])
        if path.exists() and (path / "datalayer" / "datalayer.csv").exists():
            run_dirs.append(path)
        else:
            # Try default extracted location
            fallback = DATA_ROOT / row["run_id"]
            if fallback.exists():
                run_dirs.append(fallback)
            else:
                print(f"  WARNING: Cannot find directory for {row['run_id']}, skipping")

    print(f"Resolved {len(run_dirs)} run directories")
    for r in run_dirs:
        print(f"  {r.name}")

    if len(run_dirs) == 0:
        print("ERROR: No run directories found.")
        sys.exit(1)

    # ── Load existing vocabulary ──────────────────────────────────────────────
    if not VOCAB_PATH.exists():
        print("ERROR: vocab_info.json not found.")
        sys.exit(1)

    vocab_info = json.loads(VOCAB_PATH.read_text())
    vocab      = vocab_info["vocab"]          # list of 100 branch IDs
    fn_names   = vocab_info["fn_names"]       # func_id → name
    print(f"Using existing vocabulary of {len(vocab)} branches")

    # ── Import preprocessing helpers ──────────────────────────────────────────
    from preprocess_00 import (load_datalayer, load_syslog, load_trace_counts,
                                branch_label)

    # ── Process each run ─────────────────────────────────────────────────────
    WIN_MS = 10
    win_us = WIN_MS * 1000
    all_aligned = []

    for run in run_dirs:
        print(f"\n── {run.name} ──")
        t0 = time.time()
        try:
            dl     = load_datalayer(run, win_us)
            sl     = load_syslog(run, win_us)
            counts_df, _ = load_trace_counts(run, win_us)

            # Reindex to existing vocabulary
            for col in vocab:
                if col not in counts_df.columns:
                    counts_df[col] = 0.0
            bc = counts_df[["win"] + vocab]

            merged = dl.merge(
                sl[["win", "sl_cpu", "sl_mem", "sl_load1", "sl_load5", "sl_load15"]],
                on="win", how="left"
            )
            merged = merged.merge(bc, on="win", how="inner")
            merged = merged.ffill().bfill().fillna(0.0)
            merged["run"] = run.name
            all_aligned.append(merged)
            print(f"  Done in {time.time()-t0:.1f}s — {len(merged):,} windows")
        except Exception as e:
            print(f"  ERROR processing {run.name}: {e}")
            continue

    if not all_aligned:
        print("ERROR: No runs processed successfully.")
        sys.exit(1)

    combined = pd.concat(all_aligned, ignore_index=True)
    combined = combined.sort_values(["run", "win"]).reset_index(drop=True)

    print(f"\nTotal windows: {len(combined):,}")
    print(f"Runs: {combined['run'].nunique()}")
    print(f"State distribution: {combined['dl_state'].value_counts().to_dict()}")

    combined.to_parquet(P3_OUT / "aligned_dataset_expanded.parquet", index=False)

    summary = {
        "n_runs": int(combined["run"].nunique()),
        "n_windows": int(len(combined)),
        "run_ids": sorted(combined["run"].unique().tolist()),
        "state_distribution": {
            str(k): int(v) for k, v in combined["dl_state"].value_counts().items()
        },
        "vocab_size": len(vocab),
    }
    (P3_OUT / "expand_preprocess_summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved aligned_dataset_expanded.parquet + expand_preprocess_summary.json")
    print("Done.")


if __name__ == "__main__":
    # Rename 00_preprocess import alias
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "preprocess_00",
        str(ROOT / "scripts/00_preprocess.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["preprocess_00"] = mod
    main()
