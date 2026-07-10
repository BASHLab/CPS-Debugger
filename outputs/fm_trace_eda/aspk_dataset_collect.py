#!/usr/bin/env python3
"""ASPK Dataset Collect — Task 4.

Read all per-session parquets + meta JSONs, produce:
  train_data/manifest.csv
  train_data/dataset_summary.md
"""
import json, time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

EDA_DIR = Path("/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_trace_eda")
TRAIN_DIR = EDA_DIR / "train_data"

STATE_MAP = {0: "SWINGUP", 1: "BALANCE", 2: "RESET"}

SESSIONS = [
    "2025-03-17_10-36-44", "2025-03-17_10-51-58", "2025-03-17_11-06-39",
    "2025-03-18_12-39-10",
    "2025-03-19_10-05-47", "2025-03-19_10-20-35", "2025-03-19_10-37-56",
    "2025-03-19_11-10-13",
    "2025-03-20_09-31-56", "2025-03-20_09-46-30", "2025-03-20_10-03-00",
    "2025-03-21_11-21-42",
    "2025-03-25_13-23-42", "2025-03-25_13-39-06",
    "2025-03-26_11-03-04",
]

def main():
    # Load token mapping
    tm = json.loads((TRAIN_DIR / "token_mapping.json").read_text())
    pf = json.loads((TRAIN_DIR / "fixed_prefix.json").read_text())
    vocab_size = tm["vocab_size"]
    prefix_len = pf["global_prefix_length"]

    # Collect per-session stats
    manifest_rows = []
    all_var_lens = []
    global_token_counts = Counter()
    global_state_counts = Counter()
    global_nan_count = 0
    total_ticks = 0
    per_session_frames = {}

    for sess in SESSIONS:
        pq_path = TRAIN_DIR / f"{sess}.parquet"
        meta_path = TRAIN_DIR / f"{sess}_meta.json"
        if not pq_path.exists():
            print(f"  MISSING: {pq_path}")
            continue
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        df = pd.read_parquet(pq_path)
        per_session_frames[sess] = df
        n = len(df)
        total_ticks += n

        # variable trace lengths
        vtl = df["variable_trace_len"].values
        all_var_lens.extend(vtl.tolist())

        # token freq
        for traces in df["variable_trace"]:
            for t in traces:
                if t >= 0:
                    global_token_counts[t] += 1

        # FSM states
        if "pendulum_state" in df.columns:
            for s in df["pendulum_state"].astype(int):
                global_state_counts[s] += 1

        # NaN check
        sensor_cols = ["pendulum_state","iteration","target_x","current_x",
                       "velocity","current_angle","angular_velocity"]
        for c in sensor_cols:
            if c in df.columns:
                global_nan_count += int(df[c].isna().sum())

        # Manifest row
        sc = meta.get("fsm_state_counts", {})
        manifest_rows.append({
            "session": sess,
            "n_ticks": n,
            "n_idle_skipped": meta.get("n_idle", 0),
            "n_unaligned_skipped": meta.get("n_unaligned", 0),
            "var_trace_len_mean": float(vtl.mean()) if n else 0,
            "var_trace_len_std": float(vtl.std()) if n else 0,
            "fsm_SWINGUP": sc.get("0", sc.get(0, 0)),
            "fsm_BALANCE": sc.get("1", sc.get(1, 0)),
            "fsm_RESET": sc.get("2", sc.get(2, 0)),
        })
        print(f"  {sess}: {n} ticks  var_len={vtl.mean():.1f}±{vtl.std():.1f}")

    mdf = pd.DataFrame(manifest_rows)
    mdf.to_csv(TRAIN_DIR / "manifest.csv", index=False)
    print(f"\nManifest: {TRAIN_DIR / 'manifest.csv'}  ({len(mdf)} sessions)")

    # Global stats
    var_lens = np.array(all_var_lens)
    print(f"\n--- Global Dataset Statistics ---")
    print(f"  Total ticks: {total_ticks}")
    print(f"  Variable trace length:")
    if len(var_lens):
        print(f"    min={var_lens.min()}  max={var_lens.max()}  "
              f"mean={var_lens.mean():.1f}  median={int(np.median(var_lens))}  "
              f"std={var_lens.std():.1f}")
    print(f"  FSM state counts:")
    for s in sorted(global_state_counts):
        print(f"    {s} ({STATE_MAP.get(s,'?')}): {global_state_counts[s]}")
    print(f"  NaN sensor values: {global_nan_count}")
    print(f"  Token vocab size: {vocab_size}")
    print(f"  Unique tokens seen: {len(global_token_counts)}")

    # Top tokens
    top = global_token_counts.most_common(20)
    triples = tm["triples"]
    func_names = json.loads((EDA_DIR / "func_id_to_name.json").read_text())
    print(f"\n  Top 20 tokens (by frequency):")
    print(f"    {'id':>4} {'func':>4} {'name':25s} {'from':>6}->{'to':<6}  {'count':>10}")
    for tid, cnt in top:
        f, fp, tp = triples[tid]
        print(f"    {tid:>4} {f:>4} {func_names.get(str(f),'?'):25s} {fp:>6}->{tp:<6}  {cnt:>10}")

    # Suggested train/val/test split
    print(f"\n--- Suggested Train/Val/Test Split ---")
    # Sort sessions by date, interleave FSM coverage
    # Group by date prefix for diversity
    mdf_sorted = mdf.sort_values("n_ticks", ascending=False).reset_index(drop=True)
    n_sess = len(mdf_sorted)
    # Simple approach: 12/2/1 or 11/2/2
    train_n = n_sess - 4
    val_n = 2
    test_n = n_sess - train_n - val_n
    # For diverse splits, pick val/test from different dates
    all_dates = [s[:10] for s in mdf_sorted["session"]]
    unique_dates = sorted(set(all_dates))
    # Pick test from last 2 dates, val from middle
    test_sessions = [s for s in SESSIONS if s.startswith("2025-03-25") or s.startswith("2025-03-26")][:test_n]
    val_sessions = [s for s in SESSIONS if s.startswith("2025-03-21")][:val_n]
    if len(val_sessions) < val_n:
        val_sessions += [s for s in SESSIONS if s.startswith("2025-03-20") and s not in test_sessions][:val_n-len(val_sessions)]
    train_sessions = [s for s in SESSIONS if s not in test_sessions and s not in val_sessions]

    for name, slist in [("TRAIN", train_sessions), ("VAL", val_sessions), ("TEST", test_sessions)]:
        sub = mdf[mdf["session"].isin(slist)]
        tot = int(sub["n_ticks"].sum())
        sw = int(sub["fsm_SWINGUP"].sum())
        bal = int(sub["fsm_BALANCE"].sum())
        rst = int(sub["fsm_RESET"].sum())
        print(f"  {name}: {len(slist)} sessions, {tot} ticks  "
              f"(SWINGUP={sw}, BALANCE={bal}, RESET={rst})")
        for s in slist:
            print(f"    {s}")

    # Write summary
    R = []
    R.append("# ASPK Training Dataset Summary")
    R.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    R.append("")
    R.append("## Overview")
    R.append(f"- Sessions: {len(mdf)}")
    R.append(f"- Total ticks (busy only): {total_ticks}")
    R.append(f"- Token vocabulary size: {vocab_size}")
    R.append(f"- Deterministic prefix length: {prefix_len}")
    R.append(f"- NaN sensor values: {global_nan_count}")
    R.append("")
    R.append("## Variable Trace Length (after prefix stripping)")
    if len(var_lens):
        R.append(f"- min: {var_lens.min()}")
        R.append(f"- max: {var_lens.max()}")
        R.append(f"- mean: {var_lens.mean():.1f}")
        R.append(f"- median: {int(np.median(var_lens))}")
        R.append(f"- std: {var_lens.std():.1f}")
    R.append("")
    R.append("## FSM State Distribution")
    for s in sorted(global_state_counts):
        pct = 100 * global_state_counts[s] / max(total_ticks, 1)
        R.append(f"- {STATE_MAP.get(s,'?')} (state={s}): {global_state_counts[s]} ({pct:.1f}%)")
    R.append("")
    R.append("## Per-Session Manifest")
    R.append("")
    R.append("| Session | Ticks | Idle | Unaligned | VarLen mean±std | SWINGUP | BALANCE | RESET |")
    R.append("|---------|------:|-----:|----------:|----------------:|--------:|--------:|------:|")
    for _, r in mdf.iterrows():
        R.append(f"| {r['session']} | {r['n_ticks']} | {r['n_idle_skipped']} | "
                 f"{r['n_unaligned_skipped']} | {r['var_trace_len_mean']:.1f}±{r['var_trace_len_std']:.1f} | "
                 f"{r['fsm_SWINGUP']} | {r['fsm_BALANCE']} | {r['fsm_RESET']} |")
    R.append("")
    R.append("## Top 20 Tokens (by frequency)")
    R.append("")
    R.append("| Rank | Token | Func | Name | Edge | Count |")
    R.append("|-----:|------:|-----:|------|------|------:|")
    for rank, (tid, cnt) in enumerate(top, 1):
        f, fp, tp = triples[tid]
        R.append(f"| {rank} | {tid} | {f} | {func_names.get(str(f),'?')} | {fp}->{tp} | {cnt} |")
    R.append("")
    R.append("## Suggested Split")
    R.append("")
    for name, slist in [("Train", train_sessions), ("Val", val_sessions), ("Test", test_sessions)]:
        sub = mdf[mdf["session"].isin(slist)]
        tot = int(sub["n_ticks"].sum())
        R.append(f"### {name} ({len(slist)} sessions, {tot} ticks)")
        for s in slist:
            R.append(f"- {s}")
    R.append("")

    (TRAIN_DIR / "dataset_summary.md").write_text("\n".join(R) + "\n")
    print(f"\nSaved → {TRAIN_DIR / 'dataset_summary.md'}")

if __name__ == "__main__":
    main()
