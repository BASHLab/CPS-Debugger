#!/usr/bin/env python3
"""ASPK Session Builder — Task 3.

Process all aspk files in one session, decode → filter to ctrl+math →
tokenize → strip prefix → match to datalayer → write a parquet shard.

Usage:
    python3 aspk_build_session.py --session 2025-03-18_12-39-10 [--workers 8]
"""
import argparse, json, os, sys, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ── paths ──────────────────────────────────────────────────────────────
EXTRACTED = Path("/home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs/extracted")
EDA_DIR = Path("/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_trace_eda")
TRAIN_DIR = EDA_DIR / "train_data"
TRAIN_DIR.mkdir(exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────
CONTROLLER_FUNCS = {20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46}
MATH_FUNCS = {63,64,70,71,72,73,74,77,78,99}
CTRL_MATH = CONTROLLER_FUNCS | MATH_FUNCS
CTRL_MATH_ARR = np.array(sorted(CTRL_MATH), dtype=np.int64)
BUSY_MIN = 16
ALIGN_MAX_US = 10_000

SENSOR_COLS = ["pendulum_state","iteration","target_x","current_x",
               "velocity","current_angle","angular_velocity"]

# Loaded once per worker (and main process)
_TOKEN_MAP = None
_PREFIX_LEN = None

def load_assets():
    """Load token mapping and prefix info into module globals."""
    global _TOKEN_MAP, _PREFIX_LEN
    tm = json.loads((TRAIN_DIR / "token_mapping.json").read_text())
    pf = json.loads((TRAIN_DIR / "fixed_prefix.json").read_text())
    _TOKEN_MAP = {tuple(t): i for i, t in enumerate(tm["triples"])}
    _PREFIX_LEN = pf["global_prefix_length"]
    return _TOKEN_MAP, _PREFIX_LEN

# ── decoding ───────────────────────────────────────────────────────────
def decode_aspk_ctrl_tokens(data):
    """Decode aspk binary, filter to ctrl+math, tokenize. Returns list[int] | None.
    Returns None for files smaller than the busy threshold or with unknown triples."""
    n = (len(data) - 8) // 8
    if n <= 0:
        return None, 0
    arr = np.frombuffer(data[8:8 + n*8], dtype=np.uint8).reshape(n, 8).astype(np.int64)
    to_pc   = arr[:, 0] | (arr[:, 1] << 8) | (arr[:, 2] << 16)
    from_pc = arr[:, 3] | (arr[:, 4] << 8) | (arr[:, 5] << 16)
    func    = arr[:, 6] | (arr[:, 7] << 8)
    mask = np.isin(func, CTRL_MATH_ARR)
    cf = func[mask]; cfp = from_pc[mask]; ctp = to_pc[mask]
    n_ctrl = len(cf)
    if n_ctrl < BUSY_MIN:
        return None, n_ctrl  # idle tick
    tokens = []
    for i in range(n_ctrl):
        key = (int(cf[i]), int(cfp[i]), int(ctp[i]))
        tid = _TOKEN_MAP.get(key)
        if tid is None:
            # Unknown triple — leave token slot for debugging; skip with -1
            tokens.append(-1)
        else:
            tokens.append(tid)
    return tokens, n_ctrl

def parse_aspk_filename(name):
    base = name[:-5] if name.endswith(".aspk") else name
    parts = base.split("_ts:")
    cc = int(parts[0].split(":", 1)[1])
    ts = int(parts[1])
    return cc, ts

# ── per-file worker ────────────────────────────────────────────────────
def process_file(path_str):
    """Worker. Returns dict or None (idle/error)."""
    try:
        with open(path_str, "rb") as f:
            data = f.read()
        name = os.path.basename(path_str)
        cc, ts = parse_aspk_filename(name)
        tokens, full_len = decode_aspk_ctrl_tokens(data)
        if tokens is None:
            return {"_skip": True, "cc": cc, "ts_us": ts, "n_ctrl": full_len}
        # Strip the deterministic prefix
        suffix = tokens[_PREFIX_LEN:]
        return {
            "cc": cc, "ts_us": ts,
            "tokens": suffix,
            "full_trace_length": full_len,
            "prefix_length": _PREFIX_LEN,
            "n_unknown": sum(1 for t in suffix if t < 0),
        }
    except Exception as e:
        return {"_error": True, "path": path_str, "err": str(e)}

def init_worker():
    load_assets()

# ── session driver ─────────────────────────────────────────────────────
def list_session_aspk(session):
    trace = EXTRACTED / session / "trace"
    paths = []
    for entry in os.scandir(trace):
        n = entry.name
        if n.endswith(".aspk"):
            paths.append(entry.path)
    return paths

def load_datalayer(session):
    dl = pd.read_csv(EXTRACTED / session / "datalayer" / "datalayer.csv")
    dl["ts_us"] = (dl["timestamp"] // 1000).astype(np.int64)
    dl = dl.sort_values("ts_us").reset_index(drop=True)
    return dl

def build_session(session, workers, log_every=50_000):
    t0 = time.time()
    print(f"[{session}] starting build (workers={workers})")
    print(f"[{session}] listing aspk files...")
    paths = list_session_aspk(session)
    print(f"[{session}]   found {len(paths)} files in {time.time()-t0:.1f}s")

    print(f"[{session}] loading datalayer...")
    dl = load_datalayer(session)
    dl_ts = dl["ts_us"].values
    print(f"[{session}]   {len(dl)} rows  ({time.time()-t0:.1f}s)")

    sensor_arrs = {c: dl[c].values for c in SENSOR_COLS if c in dl.columns}
    have_cols = list(sensor_arrs.keys())

    # Process files in parallel
    rows = []
    n_idle = n_err = n_unaligned = 0
    n_unknown_tok = 0

    def accept(res, i):
        nonlocal n_idle, n_err, n_unaligned, n_unknown_tok
        if res.get("_error"):
            n_err += 1; return
        if res.get("_skip"):
            n_idle += 1; return
        ts = res["ts_us"]
        j = np.searchsorted(dl_ts, ts)
        if j >= len(dl_ts): j = len(dl_ts) - 1
        if j > 0 and abs(int(dl_ts[j-1]) - ts) < abs(int(dl_ts[j]) - ts):
            j -= 1
        d = abs(int(dl_ts[j]) - ts)
        if d > ALIGN_MAX_US:
            n_unaligned += 1; return
        row = {
            "session_id": session,
            "cycle_count": res["cc"],
            "timestamp_us": ts,
            "align_err_us": d,
            "full_trace_length": res["full_trace_length"],
            "prefix_length": res["prefix_length"],
            "variable_trace": res["tokens"],
            "variable_trace_len": len(res["tokens"]),
        }
        for c in have_cols:
            row[c] = float(sensor_arrs[c][j])
        rows.append(row)
        n_unknown_tok += res["n_unknown"]
        if i % log_every == 0:
            print(f"[{session}]   processed {i}/{len(paths)}  "
                  f"kept={len(rows)}  idle={n_idle}  unaligned={n_unaligned}  "
                  f"({time.time()-t0:.1f}s)", flush=True)

    print(f"[{session}] decoding {len(paths)} files...")
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as ex:
            for i, res in enumerate(ex.map(process_file, paths, chunksize=2000), 1):
                accept(res, i)
    else:
        for i, p in enumerate(paths, 1):
            accept(process_file(p), i)

    print(f"[{session}] decode pass done in {time.time()-t0:.1f}s")
    print(f"[{session}]   kept={len(rows)}  idle={n_idle}  unaligned={n_unaligned}  err={n_err}")
    print(f"[{session}]   unknown tokens (across all kept rows): {n_unknown_tok}")

    # Build pyarrow table directly to avoid pandas list-column slowness
    print(f"[{session}] building parquet...")
    df = pd.DataFrame(rows)
    df = df.sort_values("cycle_count").reset_index(drop=True)
    out_path = TRAIN_DIR / f"{session}.parquet"
    df.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    print(f"[{session}] wrote {out_path}  ({len(df)} rows, "
          f"{out_path.stat().st_size/1e6:.1f} MB)")

    # Per-session metadata
    state_col = "pendulum_state"
    state_counts = {}
    if state_col in df.columns:
        sc = df[state_col].astype(int).value_counts().to_dict()
        state_counts = {int(k): int(v) for k, v in sc.items()}

    meta = {
        "session": session,
        "n_files_total": len(paths),
        "n_kept": len(df),
        "n_idle": n_idle,
        "n_unaligned": n_unaligned,
        "n_err": n_err,
        "n_unknown_tokens": n_unknown_tok,
        "fsm_state_counts": state_counts,
        "variable_trace_len": {
            "mean": float(df["variable_trace_len"].mean()) if len(df) else 0,
            "median": float(df["variable_trace_len"].median()) if len(df) else 0,
            "min": int(df["variable_trace_len"].min()) if len(df) else 0,
            "max": int(df["variable_trace_len"].max()) if len(df) else 0,
            "std": float(df["variable_trace_len"].std()) if len(df) else 0,
        },
        "elapsed_s": time.time()-t0,
    }
    meta_path = TRAIN_DIR / f"{session}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[{session}] DONE in {time.time()-t0:.1f}s")
    return meta

# ── main ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    load_assets()
    print(f"Loaded token vocab: {len(_TOKEN_MAP)}  prefix length: {_PREFIX_LEN}")

    build_session(args.session, args.workers)

if __name__ == "__main__":
    main()
