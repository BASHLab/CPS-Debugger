#!/usr/bin/env python3
"""ASPK Dataset Prep — Tasks 1+2.

Task 1: Profile the deterministic prefix of controller+math edge sequences
        across all 16 intact-aspk sessions.
Task 2: Verify library determinism — does (ctrl+math sequence, FSM state)
        determine the library edge counts?

Outputs:
  fixed_prefix.json   — deterministic prefix of ctrl+math edge sequences
  token_mapping.json  — global (func, from_pc, to_pc) → token_id mapping (0..N-1)
"""
import json, time, hashlib
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────
EXTRACTED = Path("/home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs/extracted")
EDA_DIR = Path("/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_trace_eda")
TRAIN_DIR = EDA_DIR / "train_data"
TRAIN_DIR.mkdir(exist_ok=True)
PREP_OUT = EDA_DIR / "aspk_eda"
PREP_OUT.mkdir(exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────
CONTROLLER_FUNCS = {20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46}
MATH_FUNCS = {63,64,70,71,72,73,74,77,78,99}
CTRL_MATH = CONTROLLER_FUNCS | MATH_FUNCS
CTRL_MATH_ARR = np.array(sorted(CTRL_MATH), dtype=np.int64)

SESSIONS = [
    "2025-03-17_10-36-44", "2025-03-17_10-51-58", "2025-03-17_11-06-39",
    "2025-03-18_12-39-10",
    "2025-03-19_10-05-47", "2025-03-19_10-20-35", "2025-03-19_10-37-56",
    "2025-03-19_11-10-13",                     # 10-54-34 dropped: no datalayer
    "2025-03-20_09-31-56", "2025-03-20_09-46-30", "2025-03-20_10-03-00",
    "2025-03-21_11-21-42",
    "2025-03-25_13-23-42", "2025-03-25_13-39-06",
    "2025-03-26_11-03-04",
]
# Busy-tick threshold: ticks with fewer ctrl+math entries are "idle" (32-byte files)
BUSY_MIN = 16

# ── decode ─────────────────────────────────────────────────────────────
def decode_aspk_np(data):
    """Numpy decoder. Returns (N,3) int64 array of [func, from_pc, to_pc].
    Skips 8-byte header."""
    n = (len(data) - 8) // 8
    if n <= 0:
        return np.empty((0, 3), dtype=np.int64)
    arr = np.frombuffer(data[8:8 + n*8], dtype=np.uint8).reshape(n, 8).astype(np.int64)
    to_pc   = arr[:, 0] | (arr[:, 1] << 8) | (arr[:, 2] << 16)
    from_pc = arr[:, 3] | (arr[:, 4] << 8) | (arr[:, 5] << 16)
    func    = arr[:, 6] | (arr[:, 7] << 8)
    return np.stack([func, from_pc, to_pc], axis=1)

def decode_aspk_split(data):
    """Returns (ctrl_arr, lib_arr) — each (N,3) — preserving order."""
    arr = decode_aspk_np(data)
    if len(arr) == 0:
        return arr, arr
    mask_ctrl = np.isin(arr[:, 0], CTRL_MATH_ARR)
    return arr[mask_ctrl], arr[~mask_ctrl]

def parse_aspk_filename(name):
    base = name[:-5] if name.endswith(".aspk") else name
    parts = base.split("_ts:")
    cc = int(parts[0].split(":", 1)[1])
    ts = int(parts[1])
    return cc, ts

def list_session_aspk(session):
    trace = EXTRACTED / session / "trace"
    items = []
    for f in trace.iterdir():
        if not f.name.endswith(".aspk"): continue
        try:
            cc, ts = parse_aspk_filename(f.name)
            items.append((cc, ts, f))
        except Exception:
            pass
    items.sort(key=lambda x: x[0])
    return items

def load_datalayer(session):
    dl = pd.read_csv(EXTRACTED / session / "datalayer" / "datalayer.csv")
    dl["ts_us"] = (dl["timestamp"] // 1000).astype(np.int64)
    dl = dl.sort_values("ts_us").reset_index(drop=True)
    return dl

# ══════════════════════════════════════════════════════════════════════
# TASK 1 — Deterministic prefix profiling
# ══════════════════════════════════════════════════════════════════════
def task_1():
    print("\n" + "="*80)
    print("# TASK 1: Deterministic prefix across 16 sessions")
    print("="*80)

    n_sess = len(SESSIONS)
    target_total = 5000
    per_sess = target_total // n_sess  # ~312
    print(f"\nSampling {per_sess} ticks/session × {n_sess} sessions = {per_sess*n_sess} ticks")

    all_ctrl_seqs = []          # list[list[tuple]]
    per_session_seqs = {}       # session -> list[list[tuple]]
    global_triples = set()

    t0 = time.time()
    for sess in SESSIONS:
        print(f"  [{sess}] listing & sampling...")
        files = list_session_aspk(sess)
        if not files:
            print(f"    SKIP: no files")
            continue
        idxs = np.linspace(0, len(files)-1, per_sess, dtype=int)
        seqs = []
        for i in idxs:
            cc, ts, p = files[i]
            data = p.read_bytes()
            ctrl, _ = decode_aspk_split(data)
            seq = [tuple(int(x) for x in row) for row in ctrl]
            seqs.append(seq)
            for tr in seq:
                global_triples.add(tr)
        per_session_seqs[sess] = seqs
        all_ctrl_seqs.extend(seqs)
        print(f"    sampled {len(seqs)} ticks, "
              f"vocab so far: {len(global_triples)}  ({time.time()-t0:.1f}s)")

    # Separate idle vs busy ticks
    busy_seqs = [s for s in all_ctrl_seqs if len(s) >= BUSY_MIN]
    idle_seqs = [s for s in all_ctrl_seqs if len(s) <  BUSY_MIN]
    print(f"\n--- Tick mode split (BUSY_MIN={BUSY_MIN}) ---")
    print(f"  total: {len(all_ctrl_seqs)}  busy: {len(busy_seqs)}  idle: {len(idle_seqs)} "
          f"({100*len(idle_seqs)/len(all_ctrl_seqs):.2f}% idle)")
    if idle_seqs:
        idle_lens = Counter(len(s) for s in idle_seqs)
        print(f"  idle length histogram: {dict(idle_lens)}")
        print(f"  example idle tick: {idle_seqs[0]}")

    # Compute longest common prefix on BUSY ticks
    print(f"\n--- Global longest common prefix (busy ticks only) ---")
    if not busy_seqs:
        print("ERROR: no busy sequences"); return None, None
    min_len = min(len(s) for s in busy_seqs)
    print(f"  busy sequences: {len(busy_seqs)}")
    print(f"  min seq length:  {min_len}")
    print(f"  max seq length:  {max(len(s) for s in busy_seqs)}")

    prefix = []
    for pos in range(min_len):
        first = busy_seqs[0][pos]
        if all(s[pos] == first for s in busy_seqs):
            prefix.append(first)
        else:
            break
    print(f"  Global busy prefix length: {len(prefix)}")

    # Per-session prefix (busy ticks only)
    print(f"\n--- Per-session prefix lengths (busy only) ---")
    sess_prefixes = {}
    for sess, seqs in per_session_seqs.items():
        bseqs = [s for s in seqs if len(s) >= BUSY_MIN]
        slen = min(len(s) for s in bseqs) if bseqs else 0
        plen = 0
        if bseqs:
            for pos in range(slen):
                first = bseqs[0][pos]
                if all(s[pos] == first for s in bseqs):
                    plen += 1
                else:
                    break
            sess_prefixes[sess] = plen
        print(f"  {sess}: busy_prefix={plen}  (n_busy={len(bseqs)}/{len(seqs)}, min_len={slen})")

    # Print the prefix
    print(f"\n--- Fixed prefix sequence ({len(prefix)} edges) ---")
    func_names = json.loads((EDA_DIR / "func_id_to_name.json").read_text())
    for i, (f, fp, tp) in enumerate(prefix):
        fname = func_names.get(str(f), "?")
        print(f"  [{i:3d}] func={f:>3} {fname:25s}  {fp:>6} -> {tp:<6}")

    # Save
    prefix_obj = {
        "n_sequences_sampled": len(all_ctrl_seqs),
        "n_busy_sequences": len(busy_seqs),
        "n_idle_sequences": len(idle_seqs),
        "busy_min_threshold": BUSY_MIN,
        "n_sessions": len(per_session_seqs),
        "global_prefix_length": len(prefix),
        "global_prefix": [list(t) for t in prefix],
        "per_session_prefix_length": sess_prefixes,
    }
    out_path = TRAIN_DIR / "fixed_prefix.json"
    out_path.write_text(json.dumps(prefix_obj, indent=2))
    print(f"\n  Saved prefix → {out_path}")

    # Token mapping (global vocab — sort by (func, from_pc, to_pc) for stability)
    sorted_triples = sorted(global_triples)
    token_map = {f"{f}:{fp}:{tp}": i for i, (f, fp, tp) in enumerate(sorted_triples)}
    print(f"\n--- Global ctrl+math token vocab: {len(sorted_triples)} ---")
    tm_obj = {
        "vocab_size": len(sorted_triples),
        "triples": [list(t) for t in sorted_triples],
        "triple_to_id": token_map,
    }
    tm_path = TRAIN_DIR / "token_mapping.json"
    tm_path.write_text(json.dumps(tm_obj, indent=2))
    print(f"  Saved token map → {tm_path}")

    # Sanity: per-session unique-func diversity
    print(f"\n--- Per-session unique ctrl+math triples ---")
    for sess, seqs in per_session_seqs.items():
        triples = set()
        for s in seqs:
            triples.update(s)
        print(f"  {sess}: {len(triples)} unique triples")

    return prefix, sorted_triples

# ══════════════════════════════════════════════════════════════════════
# TASK 2 — Library determinism
# ══════════════════════════════════════════════════════════════════════
def task_2():
    print("\n" + "="*80)
    print("# TASK 2: Library determinism given ctrl+math sequence + FSM state")
    print("="*80)

    n_per_sess = max(1, 1000 // len(SESSIONS))
    print(f"\nSampling {n_per_sess}/session × {len(SESSIONS)} = {n_per_sess*len(SESSIONS)} ticks")

    rows = []  # ctrl_key, state, lib_key
    t0 = time.time()
    for sess in SESSIONS:
        files = list_session_aspk(sess)
        if not files: continue
        dl = load_datalayer(sess)
        dl_ts = dl["ts_us"].values
        states = dl["pendulum_state"].values

        idxs = np.linspace(0, len(files)-1, n_per_sess, dtype=int)
        for i in idxs:
            cc, ts, p = files[i]
            # nearest dl row
            j = np.searchsorted(dl_ts, ts)
            if j >= len(dl_ts): j = len(dl_ts) - 1
            if j > 0 and abs(dl_ts[j-1] - ts) < abs(dl_ts[j] - ts):
                j -= 1
            if abs(int(dl_ts[j]) - ts) > 10_000:
                continue
            state = float(states[j])

            data = p.read_bytes()
            ctrl, lib = decode_aspk_split(data)
            ctrl_key = hashlib.md5(ctrl.tobytes()).hexdigest()
            # library count multiset
            if len(lib) == 0:
                lib_key = "EMPTY"
            else:
                lc = Counter(map(tuple, lib.tolist()))
                # canonicalise
                lib_key = hashlib.md5(
                    repr(sorted(lc.items())).encode()
                ).hexdigest()
            rows.append((ctrl_key, state, lib_key))
        print(f"  {sess}: {len(rows)} ticks total ({time.time()-t0:.1f}s)")

    # Group by (ctrl_key, state) and check if lib_key is constant within each group
    groups = defaultdict(set)
    for ck, st, lk in rows:
        groups[(ck, st)].add(lk)

    n_groups = len(groups)
    n_singleton_groups = sum(1 for v in groups.values() if len(v) == 1)
    sizes = Counter(len(v) for v in groups.values())
    print(f"\n  Total ticks: {len(rows)}")
    print(f"  Unique (ctrl_seq, state) keys: {n_groups}")
    print(f"  Groups with 1 unique lib_key (deterministic): {n_singleton_groups} "
          f"({100*n_singleton_groups/max(n_groups,1):.2f}%)")
    print(f"  Group lib_key cardinality histogram: {dict(sizes)}")

    # Per-tick deterministic fraction (a tick is "predictable" if its group has 1 lib key)
    n_predict = sum(1 for ck, st, lk in rows if len(groups[(ck, st)]) == 1)
    print(f"  Ticks in deterministic groups: {n_predict}/{len(rows)} "
          f"({100*n_predict/max(len(rows),1):.2f}%)")

    # Try the weaker test: is lib_key determined by ctrl_key alone (ignore state)?
    groups_ctrl = defaultdict(set)
    for ck, st, lk in rows:
        groups_ctrl[ck].add(lk)
    n_singleton_ctrl = sum(1 for v in groups_ctrl.values() if len(v) == 1)
    print(f"\n  Weaker test (ctrl_key only):")
    print(f"    Unique ctrl_keys: {len(groups_ctrl)}")
    print(f"    Deterministic (1 lib_key): {n_singleton_ctrl}/{len(groups_ctrl)} "
          f"({100*n_singleton_ctrl/max(len(groups_ctrl),1):.2f}%)")

    return groups

# ══════════════════════════════════════════════════════════════════════
def main():
    task_1()
    task_2()

if __name__ == "__main__":
    main()
