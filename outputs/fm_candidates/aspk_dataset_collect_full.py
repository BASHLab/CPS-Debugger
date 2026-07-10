#!/usr/bin/env python3
"""ASPK Full-Vocab Dataset Collect.

Read all per-session parquets + meta JSONs from the full-vocab build,
produce manifest.csv and dataset_summary.md. Print statistics comparison
with old (ctrl+math prefix-stripped) dataset.

Usage:
    python3 aspk_dataset_collect_full.py \
        --data-dir /path/to/train_data_full \
        --token-mapping /path/to/auto_token_mapping.json \
        --sessions s1 s2 s3 ... \
        [--old-data-dir /path/to/old/train_data]
"""
import argparse, json, time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

STATE_MAP = {0: "SWINGUP", 1: "BALANCE", 2: "RESET"}
SENSOR_COLS = ["pendulum_state", "iteration", "target_x", "current_x",
               "velocity", "current_angle", "angular_velocity"]


def main():
    ap = argparse.ArgumentParser(
        description="Collect per-session parquets and print dataset statistics.")
    ap.add_argument("--data-dir", required=True,
        help="Directory containing per-session .parquet and _meta.json files")
    ap.add_argument("--token-mapping", required=True,
        help="Path to auto_token_mapping.json")
    ap.add_argument("--sessions", nargs="+", required=True,
        help="List of session IDs")
    ap.add_argument("--old-data-dir", default=None,
        help="Path to old train_data/ directory for comparison (optional)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    tm = json.loads(Path(args.token_mapping).read_text())
    vocab_size = tm["vocab_size"]

    # Collect per-session stats — use metadata files and lightweight parquet reads
    # to avoid O(N*L) token iteration across ~13B tokens
    manifest_rows = []
    all_trace_lens = []
    global_state_counts = Counter()
    global_nan_count = 0
    total_ticks = 0
    total_unknown = 0

    for sess in args.sessions:
        pq_path = data_dir / f"{sess}.parquet"
        meta_path = data_dir / f"{sess}_meta.json"
        if not pq_path.exists():
            print(f"  MISSING: {pq_path}")
            continue
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        # Read only scalar columns (skip the trace list column for speed)
        df = pd.read_parquet(pq_path, columns=[
            "trace_length", "pendulum_state", "iteration",
            "target_x", "current_x", "velocity",
            "current_angle", "angular_velocity"
        ])
        n = len(df)
        total_ticks += n
        total_unknown += meta.get("n_unknown_tokens", 0)

        # trace lengths from metadata or fast column read
        tl = df["trace_length"].values
        all_trace_lens.extend(tl.tolist())

        # FSM states
        if "pendulum_state" in df.columns:
            vc = df["pendulum_state"].astype(int).value_counts()
            for s, cnt in vc.items():
                global_state_counts[s] += cnt

        # NaN check
        for c in SENSOR_COLS:
            if c in df.columns:
                global_nan_count += int(df[c].isna().sum())

        sc = meta.get("fsm_state_counts", {})
        manifest_rows.append({
            "session": sess,
            "n_ticks": n,
            "n_unaligned_skipped": meta.get("n_unaligned", 0),
            "trace_len_mean": float(tl.mean()) if n else 0,
            "trace_len_std": float(tl.std()) if n else 0,
            "fsm_SWINGUP": sc.get("0", sc.get(0, 0)),
            "fsm_BALANCE": sc.get("1", sc.get(1, 0)),
            "fsm_RESET": sc.get("2", sc.get(2, 0)),
            "parquet_MB": pq_path.stat().st_size / 1e6,
        })
        print(f"  {sess}: {n} ticks  trace_len={tl.mean():.1f}+/-{tl.std():.1f}  "
              f"({pq_path.stat().st_size/1e6:.1f} MB)")

    mdf = pd.DataFrame(manifest_rows)
    mdf.to_csv(data_dir / "manifest.csv", index=False)
    print(f"\nManifest: {data_dir / 'manifest.csv'}  ({len(mdf)} sessions)")

    # Global stats
    trace_lens = np.array(all_trace_lens)
    print(f"\n{'='*60}")
    print(f"  NEW DATASET (full vocabulary)")
    print(f"{'='*60}")
    print(f"  Sessions: {len(mdf)}")
    print(f"  Total ticks: {total_ticks}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Total unknown tokens across all sessions: {total_unknown}")
    if len(trace_lens):
        print(f"  Trace length: min={trace_lens.min()}  max={trace_lens.max()}  "
              f"mean={trace_lens.mean():.1f}  median={int(np.median(trace_lens))}  "
              f"std={trace_lens.std():.1f}")
    idle = (trace_lens < 16).sum() if len(trace_lens) else 0
    print(f"  Idle ticks (trace_length < 16): {idle} ({100*idle/max(total_ticks,1):.1f}%)")
    print(f"  FSM state counts:")
    for s in sorted(global_state_counts):
        pct = 100 * global_state_counts[s] / max(total_ticks, 1)
        print(f"    {STATE_MAP.get(s,'?')} (state={s}): {global_state_counts[s]} ({pct:.1f}%)")
    print(f"  Total parquet size: {mdf['parquet_MB'].sum():.1f} MB")
    print(f"  NaN sensor values: {global_nan_count}")

    # ── Comparison with old dataset ──
    if args.old_data_dir:
        old_dir = Path(args.old_data_dir)
        old_manifest = old_dir / "manifest.csv"
        old_tm_path = old_dir / "token_mapping.json"
        old_pf_path = old_dir / "fixed_prefix.json"

        if old_manifest.exists() and old_tm_path.exists():
            old_mdf = pd.read_csv(old_manifest)
            old_tm = json.loads(old_tm_path.read_text())
            old_pf = json.loads(old_pf_path.read_text()) if old_pf_path.exists() else {}
            old_total = int(old_mdf["n_ticks"].sum())
            old_vtl_mean = old_mdf["var_trace_len_mean"].mean() if "var_trace_len_mean" in old_mdf.columns else 0
            old_vtl_std = old_mdf["var_trace_len_std"].mean() if "var_trace_len_std" in old_mdf.columns else 0

            print(f"\n{'='*60}")
            print(f"  COMPARISON: Old vs New")
            print(f"{'='*60}")
            print(f"  {'':30s} {'Old':>12s} {'New':>12s}")
            print(f"  {'Sessions':30s} {len(old_mdf):>12d} {len(mdf):>12d}")
            print(f"  {'Total ticks':30s} {old_total:>12d} {total_ticks:>12d}")
            print(f"  {'Vocab size':30s} {old_tm['vocab_size']:>12d} {vocab_size:>12d}")
            print(f"  {'Prefix stripped':30s} {'yes':>12s} {'no':>12s}")
            prefix_len = old_pf.get("global_prefix_length", "?")
            print(f"  {'Prefix length':30s} {str(prefix_len):>12s} {'N/A':>12s}")
            print(f"  {'Idle ticks included':30s} {'no':>12s} {'yes':>12s}")
            print(f"  {'Mean trace length':30s} {old_vtl_mean:>12.1f} {trace_lens.mean():>12.1f}")
            total_mb = mdf['parquet_MB'].sum()
            old_pq = sum(
                (old_dir / f"{s}.parquet").stat().st_size / 1e6
                for s in old_mdf["session"] if (old_dir / f"{s}.parquet").exists()
            )
            print(f"  {'Total parquet size (MB)':30s} {old_pq:>12.1f} {total_mb:>12.1f}")
        else:
            print(f"\n  Old dataset dir exists but missing manifest/token_mapping.")

    # ── Write summary ──
    R = []
    R.append("# ASPK Full-Vocabulary Training Dataset Summary")
    R.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    R.append("")
    R.append("## Overview")
    R.append(f"- Sessions: {len(mdf)}")
    R.append(f"- Total ticks: {total_ticks}")
    R.append(f"- Token vocabulary size: {vocab_size}")
    R.append(f"- Prefix stripping: NO (full trace)")
    R.append(f"- Idle ticks: INCLUDED")
    R.append(f"- NaN sensor values: {global_nan_count}")
    R.append("")
    R.append("## Trace Length (full)")
    if len(trace_lens):
        R.append(f"- min: {trace_lens.min()}")
        R.append(f"- max: {trace_lens.max()}")
        R.append(f"- mean: {trace_lens.mean():.1f}")
        R.append(f"- median: {int(np.median(trace_lens))}")
        R.append(f"- std: {trace_lens.std():.1f}")
    R.append("")
    R.append("## Per-Session Manifest")
    R.append("")
    R.append("| Session | Ticks | Unaligned | TraceLen mean+/-std | SWINGUP | BALANCE | RESET | MB |")
    R.append("|---------|------:|----------:|--------------------:|--------:|--------:|------:|---:|")
    for _, r in mdf.iterrows():
        R.append(f"| {r['session']} | {r['n_ticks']} | "
                 f"{r['n_unaligned_skipped']} | {r['trace_len_mean']:.1f}+/-{r['trace_len_std']:.1f} | "
                 f"{r['fsm_SWINGUP']} | {r['fsm_BALANCE']} | {r['fsm_RESET']} | "
                 f"{r['parquet_MB']:.1f} |")
    R.append("")

    (data_dir / "dataset_summary.md").write_text("\n".join(R) + "\n")
    print(f"\nSaved -> {data_dir / 'dataset_summary.md'}")


if __name__ == "__main__":
    main()
