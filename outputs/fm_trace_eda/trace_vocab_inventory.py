#!/usr/bin/env python3
"""Trace Vocabulary Inventory — 5-task analysis of the 163-token vocabulary.

Tasks:
  1. Token inventory by function (category classification)
  2. Per-token frequency and variability in training data
  3. Math token determinism check
  4. Sensor-predictable vs computation-internal tokens
  5. Proposed target subsets

Outputs:
  train_data/trace_vocabulary_inventory.csv
  train_data/trace_inventory_summary.md
"""
import json, time, sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ── paths ─────────────────────────────────────────────────────────────
EDA_DIR = Path("/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_trace_eda")
TRAIN_DIR = EDA_DIR / "train_data"

# ── load metadata ─────────────────────────────────────────────────────
tm = json.loads((TRAIN_DIR / "token_mapping.json").read_text())
pf = json.loads((TRAIN_DIR / "fixed_prefix.json").read_text())
func_names = json.loads((EDA_DIR / "func_id_to_name.json").read_text())
VOCAB_SIZE = tm["vocab_size"]
TRIPLES = tm["triples"]  # list of [func, from_pc, to_pc]
PREFIX_LEN = pf["global_prefix_length"]

# ── category classification ───────────────────────────────────────────
CONTROLLER_FUNC_IDS = set(range(20, 47))  # 20..46
MATH_FUNC_IDS = {63, 64, 70, 71, 72, 73, 74, 77, 78, 99}

def classify_func(fid):
    if fid in CONTROLLER_FUNC_IDS:
        return "controller"
    elif fid in MATH_FUNC_IDS:
        return "math"
    else:
        return "other"

# Sessions to analyze (4 diverse: typical balance, high swingup/reset,
# no reset, highest reset)
ANALYSIS_SESSIONS = [
    "2025-03-18_12-39-10",
    "2025-03-19_10-05-47",
    "2025-03-21_11-21-42",
    "2025-03-25_13-23-42",
]

SENSOR_COLS = ["pendulum_state", "iteration", "target_x", "current_x",
               "velocity", "current_angle", "angular_velocity"]

SAMPLE_PER_SESSION = 80_000  # sample for speed; ~320K total


def load_sessions(sessions, sample_n=None):
    """Load and optionally sample parquet sessions. Returns combined DataFrame."""
    frames = []
    for sess in sessions:
        pq_path = TRAIN_DIR / f"{sess}.parquet"
        if not pq_path.exists():
            print(f"  MISSING: {pq_path}")
            continue
        df = pd.read_parquet(pq_path)
        if sample_n and len(df) > sample_n:
            df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
        frames.append(df)
        print(f"  Loaded {sess}: {len(df)} rows")
    return pd.concat(frames, ignore_index=True)


def traces_to_count_matrix(traces, vocab_size):
    """Convert list-of-lists traces to (n_ticks, vocab_size) count matrix."""
    n = len(traces)
    counts = np.zeros((n, vocab_size), dtype=np.int16)
    for i, tr in enumerate(traces):
        for t in tr:
            if 0 <= t < vocab_size:
                counts[i, t] += 1
    return counts


def traces_to_padded(traces, pad_val=-1):
    """Convert list-of-lists to padded 2D array."""
    max_len = max(len(tr) for tr in traces)
    arr = np.full((len(traces), max_len), pad_val, dtype=np.int16)
    for i, tr in enumerate(traces):
        arr[i, :len(tr)] = tr
    return arr


def entropy(counts_array):
    """Shannon entropy of a discrete count distribution (in bits)."""
    vals, freqs = np.unique(counts_array, return_counts=True)
    p = freqs / freqs.sum()
    return -np.sum(p * np.log2(p + 1e-15))


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Token inventory by function
# ══════════════════════════════════════════════════════════════════════
def task1_inventory():
    print("\n" + "="*70)
    print("TASK 1: Token Inventory by Function")
    print("="*70)

    rows = []
    for tid, (fid, fp, tp) in enumerate(TRIPLES):
        cat = classify_func(fid)
        fname = func_names.get(str(fid), "?")
        rows.append({
            "token_id": tid,
            "func_id": fid,
            "func_name": fname,
            "from_pc": fp,
            "to_pc": tp,
            "category": cat,
        })

    df = pd.DataFrame(rows)

    # Count by category
    cat_counts = df["category"].value_counts()
    print(f"\nCategory counts:")
    for cat in ["controller", "math", "other"]:
        n = cat_counts.get(cat, 0)
        print(f"  {cat}: {n} tokens")

    # Count by function within each category
    print(f"\nTokens per function:")
    for cat in ["controller", "math"]:
        sub = df[df["category"] == cat]
        func_counts = sub.groupby(["func_id", "func_name"]).size().reset_index(name="n_tokens")
        func_counts = func_counts.sort_values("func_id")
        print(f"\n  [{cat.upper()}]")
        for _, r in func_counts.iterrows():
            print(f"    func {r['func_id']:>3d} {r['func_name']:30s}: {r['n_tokens']} tokens")

    return df


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Per-token frequency and variability
# ══════════════════════════════════════════════════════════════════════
def task2_frequency(data_df, inv_df):
    print("\n" + "="*70)
    print("TASK 2: Per-token Frequency and Variability")
    print("="*70)

    traces = data_df["variable_trace"].tolist()
    n_ticks = len(traces)
    print(f"  Building count matrix for {n_ticks} ticks x {VOCAB_SIZE} tokens...")
    t0 = time.time()
    counts = traces_to_count_matrix(traces, VOCAB_SIZE)
    print(f"  Count matrix built in {time.time()-t0:.1f}s")

    # Also build padded position array for position analysis
    print(f"  Building padded position array...")
    padded = traces_to_padded(traces)
    max_len = padded.shape[1]
    print(f"  Padded array: {padded.shape}, built in {time.time()-t0:.1f}s")

    results = []
    for tid in range(VOCAB_SIZE):
        col = counts[:, tid]
        mean_count = col.mean()
        pct_present = (col > 0).mean() * 100
        count_entropy = entropy(col)

        # Position distribution: where does this token appear?
        # Find all positions where this token occurs
        pos_mask = (padded == tid)
        positions = []
        if pos_mask.any():
            row_idx, col_idx = np.where(pos_mask)
            positions = col_idx
        if len(positions) > 0:
            pos_mean = positions.mean()
            pos_std = positions.std()
            pos_min = int(positions.min())
            pos_max = int(positions.max())
        else:
            pos_mean = pos_std = 0.0
            pos_min = pos_max = -1

        results.append({
            "token_id": tid,
            "mean_count_per_tick": round(mean_count, 4),
            "pct_ticks_present": round(pct_present, 2),
            "count_entropy_bits": round(count_entropy, 4),
            "count_min": int(col.min()),
            "count_max": int(col.max()),
            "pos_mean": round(pos_mean, 1),
            "pos_std": round(pos_std, 1),
            "pos_min": pos_min,
            "pos_max": pos_max,
        })

    freq_df = pd.DataFrame(results)

    # Sort by entropy descending (most variable = most interesting)
    freq_df = freq_df.sort_values("count_entropy_bits", ascending=False)
    print(f"\nTop 20 most variable tokens (by count entropy):")
    print(f"  {'TID':>4} {'Func':25s} {'Cat':10s} {'MeanCnt':>8} {'%Present':>9} "
          f"{'Entropy':>8} {'CntRange':>10} {'PosRange':>15}")
    for _, r in freq_df.head(20).iterrows():
        tid = int(r["token_id"])
        fid = TRIPLES[tid][0]
        fname = func_names.get(str(fid), "?")
        cat = classify_func(fid)
        crange = f"{int(r['count_min'])}-{int(r['count_max'])}"
        prange = f"{int(r['pos_min'])}-{int(r['pos_max'])}" if r['pos_min'] >= 0 else "n/a"
        print(f"  {tid:>4d} {fname:25s} {cat:10s} {r['mean_count_per_tick']:>8.3f} "
              f"{r['pct_ticks_present']:>8.1f}% {r['count_entropy_bits']:>8.4f} "
              f"{crange:>10} {prange:>15}")

    # Reset sort for later merge
    freq_df = freq_df.sort_values("token_id").reset_index(drop=True)

    # Summary stats
    print(f"\n  Always-present tokens (>99.9% ticks): "
          f"{(freq_df['pct_ticks_present'] > 99.9).sum()}")
    print(f"  Rare tokens (<10% ticks): "
          f"{(freq_df['pct_ticks_present'] < 10).sum()}")
    print(f"  Fixed-count tokens (entropy < 0.1): "
          f"{(freq_df['count_entropy_bits'] < 0.1).sum()}")
    print(f"  Variable-count tokens (entropy > 1.0): "
          f"{(freq_df['count_entropy_bits'] > 1.0).sum()}")

    # Verify mean sequence length
    trace_lens = data_df["variable_trace_len"].values
    print(f"\n  Variable trace lengths: mean={trace_lens.mean():.1f}, "
          f"median={np.median(trace_lens):.0f}, std={trace_lens.std():.1f}")
    total_tokens = freq_df["mean_count_per_tick"].sum()
    print(f"  Sum of mean counts across all tokens: {total_tokens:.1f} "
          f"(should ≈ mean trace length)")

    return freq_df, counts


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Math token determinism check
# ══════════════════════════════════════════════════════════════════════
def task3_math_determinism(data_df, inv_df, counts):
    print("\n" + "="*70)
    print("TASK 3: Math Token Determinism Check")
    print("="*70)

    math_tids = [tid for tid, (fid, _, _) in enumerate(TRIPLES)
                 if classify_func(fid) == "math"]
    ctrl_tids = [tid for tid, (fid, _, _) in enumerate(TRIPLES)
                 if classify_func(fid) == "controller"]

    print(f"  Controller tokens: {len(ctrl_tids)}")
    print(f"  Math tokens: {len(math_tids)}")

    n_ticks = counts.shape[0]
    # Use a larger sample for determinism check
    sample_n = min(n_ticks, 200_000)
    if sample_n < n_ticks:
        idx = np.random.RandomState(42).choice(n_ticks, sample_n, replace=False)
        ctrl_counts = counts[idx][:, ctrl_tids]
        math_counts = counts[idx][:, math_tids]
    else:
        ctrl_counts = counts[:, ctrl_tids]
        math_counts = counts[:, math_tids]

    print(f"  Analyzing {sample_n} ticks...")

    # Hash controller token count vectors
    # For each unique controller pattern, check if math pattern is deterministic
    ctrl_hashes = {}
    for i in range(sample_n):
        ctrl_key = tuple(ctrl_counts[i])
        math_val = tuple(math_counts[i])
        if ctrl_key not in ctrl_hashes:
            ctrl_hashes[ctrl_key] = set()
        ctrl_hashes[ctrl_key].add(math_val)

    n_unique_ctrl = len(ctrl_hashes)
    n_deterministic = sum(1 for v in ctrl_hashes.values() if len(v) == 1)
    n_nondeterministic = n_unique_ctrl - n_deterministic

    # Count ticks in deterministic vs non-deterministic groups
    ticks_in_det = 0
    ticks_in_nondet = 0
    for i in range(sample_n):
        ctrl_key = tuple(ctrl_counts[i])
        if len(ctrl_hashes[ctrl_key]) == 1:
            ticks_in_det += 1
        else:
            ticks_in_nondet += 1

    print(f"\n  Unique controller count patterns: {n_unique_ctrl}")
    print(f"  Deterministic (math fully determined by ctrl): "
          f"{n_deterministic} ({100*n_deterministic/max(n_unique_ctrl,1):.1f}%)")
    print(f"  Non-deterministic: "
          f"{n_nondeterministic} ({100*n_nondeterministic/max(n_unique_ctrl,1):.1f}%)")
    print(f"\n  Ticks in deterministic groups: "
          f"{ticks_in_det} ({100*ticks_in_det/sample_n:.1f}%)")
    print(f"  Ticks in non-deterministic groups: "
          f"{ticks_in_nondet} ({100*ticks_in_nondet/sample_n:.1f}%)")

    # For non-deterministic groups, how much does math vary?
    math_variation = []
    for ctrl_key, math_set in ctrl_hashes.items():
        if len(math_set) > 1:
            math_variation.append(len(math_set))
    if math_variation:
        print(f"\n  Non-deterministic groups: math variants per ctrl pattern:")
        print(f"    min={min(math_variation)}, max={max(math_variation)}, "
              f"mean={np.mean(math_variation):.1f}, median={np.median(math_variation):.0f}")

    # Also check: are math tokens determined by controller tokens' PRESENCE
    # (binary) rather than counts?
    print(f"\n  Checking binary (presence/absence) determinism...")
    ctrl_binary = (ctrl_counts > 0).astype(np.int8)
    ctrl_bin_hashes = {}
    for i in range(sample_n):
        ctrl_key = tuple(ctrl_binary[i])
        math_val = tuple(math_counts[i])
        if ctrl_key not in ctrl_bin_hashes:
            ctrl_bin_hashes[ctrl_key] = set()
        ctrl_bin_hashes[ctrl_key].add(math_val)

    n_unique_bin = len(ctrl_bin_hashes)
    n_det_bin = sum(1 for v in ctrl_bin_hashes.values() if len(v) == 1)
    ticks_det_bin = sum(1 for i in range(sample_n)
                        if len(ctrl_bin_hashes[tuple(ctrl_binary[i])]) == 1)
    print(f"  Unique binary ctrl patterns: {n_unique_bin}")
    print(f"  Deterministic (binary): {n_det_bin} ({100*n_det_bin/max(n_unique_bin,1):.1f}%)")
    print(f"  Ticks in deterministic binary groups: "
          f"{ticks_det_bin} ({100*ticks_det_bin/sample_n:.1f}%)")

    # Per math-token: is each one individually predictable from ctrl?
    print(f"\n  Per math-token predictability from controller counts:")
    print(f"  {'TID':>4} {'Func':25s} {'Edge':>12} {'CtrlDet%':>10} {'Variants':>10}")
    math_det_flags = {}
    for j, mtid in enumerate(math_tids):
        fid, fp, tp = TRIPLES[mtid]
        fname = func_names.get(str(fid), "?")
        # For each ctrl pattern, check if this specific math token count is fixed
        per_token_map = {}
        for i in range(sample_n):
            ctrl_key = tuple(ctrl_counts[i])
            mval = int(math_counts[i, j])
            if ctrl_key not in per_token_map:
                per_token_map[ctrl_key] = set()
            per_token_map[ctrl_key].add(mval)
        n_det = sum(1 for v in per_token_map.values() if len(v) == 1)
        max_variants = max(len(v) for v in per_token_map.values())
        det_pct = 100 * n_det / len(per_token_map)
        math_det_flags[mtid] = det_pct
        print(f"  {mtid:>4d} {fname:25s} {fp:>5}->{tp:<5} {det_pct:>9.1f}% {max_variants:>10}")

    return {
        "n_unique_ctrl_patterns": n_unique_ctrl,
        "pct_deterministic_patterns": 100*n_deterministic/max(n_unique_ctrl,1),
        "pct_ticks_deterministic": 100*ticks_in_det/sample_n,
        "math_det_flags": math_det_flags,
    }


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Sensor-predictable vs computation-internal tokens
# ══════════════════════════════════════════════════════════════════════
def task4_sensor_correlation(data_df, counts):
    print("\n" + "="*70)
    print("TASK 4: Sensor-predictable vs Computation-internal Tokens")
    print("="*70)

    # Get sensor values
    available_sensors = [c for c in SENSOR_COLS if c in data_df.columns]
    print(f"  Available sensors: {available_sensors}")

    # Continuous sensors only (exclude pendulum_state which is categorical)
    continuous_sensors = [c for c in available_sensors if c != "pendulum_state"]

    sensor_vals = {}
    for c in continuous_sensors:
        v = data_df[c].values.astype(np.float64)
        # Standardize for correlation
        std = v.std()
        if std > 0:
            sensor_vals[c] = (v - v.mean()) / std
        else:
            sensor_vals[c] = v - v.mean()

    n_ticks = counts.shape[0]

    # Compute Pearson correlation between each token count and each sensor
    print(f"  Computing correlations ({VOCAB_SIZE} tokens x {len(continuous_sensors)} sensors)...")
    corr_results = []
    for tid in range(VOCAB_SIZE):
        col = counts[:, tid].astype(np.float64)
        col_std = col.std()
        row = {"token_id": tid}
        max_abs_r = 0.0
        max_sensor = ""
        for sname, sval in sensor_vals.items():
            if col_std > 0:
                r = np.corrcoef(col, sval)[0, 1]
            else:
                r = 0.0
            row[f"r_{sname}"] = round(r, 4)
            if abs(r) > max_abs_r:
                max_abs_r = abs(r)
                max_sensor = sname
        row["max_abs_r"] = round(max_abs_r, 4)
        row["max_sensor"] = max_sensor

        # Classify
        if max_abs_r > 0.3:
            row["sensor_class"] = "sensor_predictable"
        elif max_abs_r > 0.1:
            row["sensor_class"] = "weakly_predictable"
        else:
            row["sensor_class"] = "computation_internal"

        corr_results.append(row)

    corr_df = pd.DataFrame(corr_results)

    # Summary
    class_counts = corr_df["sensor_class"].value_counts()
    print(f"\n  Token classification:")
    for cls in ["sensor_predictable", "weakly_predictable", "computation_internal"]:
        n = class_counts.get(cls, 0)
        print(f"    {cls}: {n} tokens")

    # Top sensor-predictable tokens
    sp = corr_df[corr_df["sensor_class"] == "sensor_predictable"].sort_values(
        "max_abs_r", ascending=False)
    if len(sp):
        print(f"\n  Sensor-predictable tokens (|r| > 0.3):")
        print(f"  {'TID':>4} {'Func':25s} {'Cat':10s} {'MaxR':>7} {'Sensor':20s}")
        for _, r in sp.iterrows():
            tid = int(r["token_id"])
            fid = TRIPLES[tid][0]
            fname = func_names.get(str(fid), "?")
            cat = classify_func(fid)
            print(f"  {tid:>4d} {fname:25s} {cat:10s} {r['max_abs_r']:>7.3f} {r['max_sensor']:20s}")

    # For computation-internal tokens: check inter-token predictability
    print(f"\n  Checking inter-token correlations for computation-internal tokens...")
    ci_tids = corr_df[corr_df["sensor_class"] == "computation_internal"]["token_id"].values.astype(int)
    sp_tids = corr_df[corr_df["sensor_class"] != "computation_internal"]["token_id"].values.astype(int)

    if len(ci_tids) > 0 and len(sp_tids) > 0:
        ci_counts = counts[:, ci_tids].astype(np.float64)
        sp_counts = counts[:, sp_tids].astype(np.float64)

        # For each CI token, max |r| with any sensor-predictable token
        inter_max_r = []
        for j, tid in enumerate(ci_tids):
            col = ci_counts[:, j]
            if col.std() == 0:
                inter_max_r.append(0.0)
                continue
            max_r = 0.0
            for k in range(sp_counts.shape[1]):
                if sp_counts[:, k].std() > 0:
                    r = abs(np.corrcoef(col, sp_counts[:, k])[0, 1])
                    if r > max_r:
                        max_r = r
            inter_max_r.append(max_r)

        inter_arr = np.array(inter_max_r)
        print(f"    CI tokens with max inter-token |r| > 0.3: "
              f"{(inter_arr > 0.3).sum()}/{len(ci_tids)}")
        print(f"    CI tokens with max inter-token |r| > 0.1: "
              f"{(inter_arr > 0.1).sum()}/{len(ci_tids)}")
        print(f"    CI tokens truly unpredictable (max |r| < 0.1 with anything): "
              f"{(inter_arr < 0.1).sum()}/{len(ci_tids)}")

        # Add inter-token correlation to corr_df
        inter_map = dict(zip(ci_tids, inter_max_r))
        corr_df["inter_token_max_r"] = corr_df["token_id"].map(
            lambda t: round(inter_map.get(t, np.nan), 4))

    # Also: per-FSM-state analysis
    if "pendulum_state" in data_df.columns:
        print(f"\n  Per-FSM-state token presence analysis:")
        states = data_df["pendulum_state"].astype(int).values
        state_map = {0: "SWINGUP", 1: "BALANCE", 2: "RESET"}
        state_token_means = {}
        for s in sorted(np.unique(states)):
            mask = states == s
            if mask.sum() < 100:
                continue
            s_counts = counts[mask]
            means = s_counts.mean(axis=0)
            state_token_means[s] = means
            sname = state_map.get(s, f"state{s}")
            print(f"    {sname} ({mask.sum()} ticks): "
                  f"mean seq len = {means.sum():.1f}")

        # Tokens that differ most between states
        if len(state_token_means) >= 2:
            print(f"\n  Tokens with largest BALANCE vs non-BALANCE difference:")
            if 1 in state_token_means and 0 in state_token_means:
                bal = state_token_means[1]
                nonbal_states = [s for s in state_token_means if s != 1]
                nonbal = np.mean([state_token_means[s] for s in nonbal_states], axis=0)
                diff = np.abs(bal - nonbal)
                top_diff_idx = np.argsort(diff)[::-1][:15]
                print(f"  {'TID':>4} {'Func':25s} {'BAL mean':>10} {'NonBAL mean':>12} {'|Diff|':>8}")
                for tid in top_diff_idx:
                    tid = int(tid)
                    fid = TRIPLES[tid][0]
                    fname = func_names.get(str(fid), "?")
                    print(f"  {tid:>4d} {fname:25s} {bal[tid]:>10.3f} {nonbal[tid]:>12.3f} "
                          f"{diff[tid]:>8.3f}")

        # Add FSM sensitivity to corr_df
        if 1 in state_token_means and 0 in state_token_means:
            fsm_diffs = np.abs(state_token_means[1] - nonbal)
            corr_df["fsm_sensitivity"] = [round(fsm_diffs[t], 4) for t in range(VOCAB_SIZE)]

    return corr_df


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Proposed target subsets
# ══════════════════════════════════════════════════════════════════════
def task5_subsets(inv_df, freq_df, corr_df, math_det_info, counts):
    print("\n" + "="*70)
    print("TASK 5: Proposed Target Subsets")
    print("="*70)

    # Merge all info
    merged = inv_df.merge(freq_df, on="token_id").merge(
        corr_df[["token_id", "max_abs_r", "max_sensor", "sensor_class",
                 "inter_token_max_r"]].rename(columns={}),
        on="token_id", how="left")

    if "fsm_sensitivity" in corr_df.columns:
        merged = merged.merge(
            corr_df[["token_id", "fsm_sensitivity"]], on="token_id", how="left")

    n_ticks = counts.shape[0]
    trace_lens = counts.sum(axis=1)

    # Subset 1: Full (all 163)
    full_tids = list(range(VOCAB_SIZE))
    full_len = trace_lens.mean()
    full_bits = sum(freq_df["count_entropy_bits"])

    # Subset 2: Controller-only (exclude math)
    ctrl_tids = [tid for tid, (fid, _, _) in enumerate(TRIPLES)
                 if classify_func(fid) == "controller"]
    ctrl_lens = counts[:, ctrl_tids].sum(axis=1)
    ctrl_mean_len = ctrl_lens.mean()
    ctrl_bits = sum(freq_df.loc[freq_df["token_id"].isin(ctrl_tids), "count_entropy_bits"])

    # Subset 3: Sensor-driven (|r| > 0.1)
    sensor_tids = corr_df[corr_df["max_abs_r"] > 0.1]["token_id"].tolist()
    sensor_lens = counts[:, sensor_tids].sum(axis=1)
    sensor_mean_len = sensor_lens.mean()
    sensor_bits = sum(freq_df.loc[freq_df["token_id"].isin(sensor_tids), "count_entropy_bits"])

    # Subset 4: Variable tokens only (entropy > 0.1 bits)
    variable_tids = freq_df[freq_df["count_entropy_bits"] > 0.1]["token_id"].tolist()
    variable_lens = counts[:, variable_tids].sum(axis=1)
    variable_mean_len = variable_lens.mean()
    variable_bits = sum(freq_df.loc[freq_df["token_id"].isin(variable_tids), "count_entropy_bits"])

    subsets = [
        ("Full (all tokens)", full_tids, full_len, full_bits),
        ("Controller-only", ctrl_tids, ctrl_mean_len, ctrl_bits),
        ("Sensor-driven (|r|>0.1)", sensor_tids, sensor_mean_len, sensor_bits),
        ("Variable-only (H>0.1 bits)", variable_tids, variable_mean_len, variable_bits),
    ]

    print(f"\n  {'Subset':35s} {'Vocab':>6} {'MeanLen':>8} {'TotalH(bits)':>13} {'%FullH':>7}")
    for name, tids, mean_len, bits in subsets:
        pct = 100 * bits / max(full_bits, 1e-10)
        print(f"  {name:35s} {len(tids):>6} {mean_len:>8.1f} {bits:>13.1f} {pct:>6.1f}%")

    # Math determinism summary for recommendation
    det_pct = math_det_info["pct_ticks_deterministic"]
    print(f"\n  Math determinism: {det_pct:.1f}% of ticks have math fully determined by ctrl")

    return merged, subsets


# ══════════════════════════════════════════════════════════════════════
# Write outputs
# ══════════════════════════════════════════════════════════════════════
def write_csv(merged):
    out = TRAIN_DIR / "trace_vocabulary_inventory.csv"
    merged.to_csv(out, index=False, float_format="%.4f")
    print(f"\nSaved → {out}")


def write_summary(inv_df, freq_df, corr_df, math_info, subsets_info, merged):
    merged_df, subsets = subsets_info

    R = []
    R.append("# Trace Vocabulary Inventory")
    R.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    R.append(f"Analysis sessions: {', '.join(ANALYSIS_SESSIONS)}")
    R.append(f"Sample per session: {SAMPLE_PER_SESSION}")
    R.append("")

    # Task 1
    R.append("## Task 1: Token Inventory by Function")
    R.append("")
    cat_counts = inv_df["category"].value_counts()
    for cat in ["controller", "math", "other"]:
        R.append(f"- **{cat}**: {cat_counts.get(cat, 0)} tokens")
    R.append("")
    R.append("### Tokens per function")
    R.append("")
    R.append("| Category | Func ID | Name | # Tokens |")
    R.append("|----------|--------:|------|--------:|")
    for cat in ["controller", "math"]:
        sub = inv_df[inv_df["category"] == cat]
        for (fid, fname), grp in sub.groupby(["func_id", "func_name"]):
            R.append(f"| {cat} | {fid} | {fname} | {len(grp)} |")
    R.append("")

    # Task 2
    R.append("## Task 2: Per-token Frequency and Variability")
    R.append("")
    n_always = (freq_df["pct_ticks_present"] > 99.9).sum()
    n_rare = (freq_df["pct_ticks_present"] < 10).sum()
    n_fixed = (freq_df["count_entropy_bits"] < 0.1).sum()
    n_variable = (freq_df["count_entropy_bits"] > 1.0).sum()
    R.append(f"- Always-present tokens (>99.9%): **{n_always}**")
    R.append(f"- Rare tokens (<10%): **{n_rare}**")
    R.append(f"- Fixed-count tokens (H < 0.1 bits): **{n_fixed}**")
    R.append(f"- High-variability tokens (H > 1.0 bits): **{n_variable}**")
    R.append("")
    R.append("### Top 20 most variable tokens")
    R.append("")
    R.append("| Rank | TID | Function | Category | Mean Count | % Present | Entropy (bits) | Count Range | Pos Range |")
    R.append("|-----:|----:|----------|----------|----------:|----------:|--------------:|------------|----------|")
    top20 = freq_df.sort_values("count_entropy_bits", ascending=False).head(20)
    for rank, (_, r) in enumerate(top20.iterrows(), 1):
        tid = int(r["token_id"])
        fid = TRIPLES[tid][0]
        fname = func_names.get(str(fid), "?")
        cat = classify_func(fid)
        R.append(f"| {rank} | {tid} | {fname} | {cat} | {r['mean_count_per_tick']:.3f} | "
                 f"{r['pct_ticks_present']:.1f} | {r['count_entropy_bits']:.3f} | "
                 f"{int(r['count_min'])}-{int(r['count_max'])} | {int(r['pos_min'])}-{int(r['pos_max'])} |")
    R.append("")

    # Task 3
    R.append("## Task 3: Math Token Determinism")
    R.append("")
    R.append(f"- Unique controller count patterns: **{math_info['n_unique_ctrl_patterns']}**")
    R.append(f"- Patterns where math is fully determined: "
             f"**{math_info['pct_deterministic_patterns']:.1f}%**")
    R.append(f"- Ticks in deterministic groups: "
             f"**{math_info['pct_ticks_deterministic']:.1f}%**")
    R.append("")
    if math_info['pct_ticks_deterministic'] > 90:
        R.append("> **Conclusion**: Math tokens are largely deterministic given controller "
                 "tokens. A controller-only model could derive math tokens post-hoc.")
    elif math_info['pct_ticks_deterministic'] > 50:
        R.append("> **Conclusion**: Math tokens are partially deterministic. Some information "
                 "is lost by excluding them, but a controller-only model captures most behavior.")
    else:
        R.append("> **Conclusion**: Math tokens carry substantial independent information. "
                 "Excluding them would lose significant signal.")
    R.append("")

    # Task 4
    R.append("## Task 4: Sensor Predictability")
    R.append("")
    class_counts = corr_df["sensor_class"].value_counts()
    for cls in ["sensor_predictable", "weakly_predictable", "computation_internal"]:
        R.append(f"- **{cls}** (|r| {'> 0.3' if cls=='sensor_predictable' else '0.1-0.3' if cls=='weakly_predictable' else '< 0.1'}): "
                 f"{class_counts.get(cls, 0)} tokens")
    R.append("")

    sp = corr_df[corr_df["sensor_class"] == "sensor_predictable"].sort_values(
        "max_abs_r", ascending=False)
    if len(sp):
        R.append("### Sensor-predictable tokens")
        R.append("")
        R.append("| TID | Function | Category | Max |r| | Best Sensor |")
        R.append("|----:|----------|----------|-------:|-------------|")
        for _, r in sp.iterrows():
            tid = int(r["token_id"])
            fid = TRIPLES[tid][0]
            fname = func_names.get(str(fid), "?")
            cat = classify_func(fid)
            R.append(f"| {tid} | {fname} | {cat} | {r['max_abs_r']:.3f} | {r['max_sensor']} |")
        R.append("")

    # Task 5
    R.append("## Task 5: Proposed Target Subsets")
    R.append("")
    R.append("| Subset | Vocab Size | Mean Seq Len | Total Entropy (bits) | % Full Entropy |")
    R.append("|--------|----------:|------------:|---------:|---------:|")
    for name, tids, mean_len, bits in subsets:
        pct = 100 * bits / max(subsets[0][3], 1e-10)
        R.append(f"| {name} | {len(tids)} | {mean_len:.1f} | {bits:.1f} | {pct:.1f}% |")
    R.append("")

    R.append("### Recommendations")
    R.append("")
    R.append("1. **Start with Full (163 tokens)**: The vocabulary is small enough that "
             "there is no computational reason to reduce it. A small transformer can "
             "handle 163 tokens efficiently.")
    R.append("")
    R.append("2. **Controller-only as ablation**: If math tokens are >90% deterministic "
             "given controller tokens, training controller-only and deriving math post-hoc "
             "is a valid simplification. Use as an ablation study.")
    R.append("")
    R.append("3. **Variable-only for focused modeling**: Tokens with entropy < 0.1 bits "
             "are essentially constant (always present at count=1 or always absent). "
             "A model could hard-code these and focus capacity on variable tokens.")
    R.append("")

    out = TRAIN_DIR / "trace_inventory_summary.md"
    out.write_text("\n".join(R) + "\n")
    print(f"Saved → {out}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("Trace Vocabulary Inventory")
    print(f"  Vocab size: {VOCAB_SIZE}, Prefix length: {PREFIX_LEN}")

    # Task 1
    inv_df = task1_inventory()

    # Load data
    print(f"\nLoading {len(ANALYSIS_SESSIONS)} sessions (sample={SAMPLE_PER_SESSION})...")
    data_df = load_sessions(ANALYSIS_SESSIONS, sample_n=SAMPLE_PER_SESSION)
    print(f"  Total rows: {len(data_df)}")

    # Task 2
    freq_df, counts = task2_frequency(data_df, inv_df)

    # Task 3
    math_info = task3_math_determinism(data_df, inv_df, counts)

    # Task 4
    corr_df = task4_sensor_correlation(data_df, counts)

    # Task 5
    subsets_info = task5_subsets(inv_df, freq_df, corr_df, math_info, counts)

    # Write outputs
    merged = subsets_info[0]
    write_csv(merged)
    write_summary(inv_df, freq_df, corr_df, math_info, subsets_info, merged)

    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
