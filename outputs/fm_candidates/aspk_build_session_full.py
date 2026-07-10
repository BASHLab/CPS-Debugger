#!/usr/bin/env python3
"""ASPK Full-Vocabulary Session Builder.

Process all aspk files in one session: decode ALL edges (no function filter),
tokenize with full vocab, keep idle ticks, no prefix stripping.

Usage:
    python3 aspk_build_session_full.py \
        --session 2025-03-18_12-39-10 \
        --extracted-dir /path/to/extracted \
        --token-mapping /path/to/auto_token_mapping.json \
        --output-dir /path/to/output \
        [--workers 8]
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

# ── constants ──────────────────────────────────────────────────────────
ALIGN_MAX_US = 10_000

SENSOR_COLS = ["pendulum_state", "iteration", "target_x", "current_x",
               "velocity", "current_angle", "angular_velocity"]

# Loaded once per worker
_TOKEN_MAP = None
_VOCAB_SIZE = None
_TOKEN_MAP_PATH = None


def load_assets(token_mapping_path):
    """Load token mapping into module globals."""
    global _TOKEN_MAP, _VOCAB_SIZE, _TOKEN_MAP_PATH
    _TOKEN_MAP_PATH = token_mapping_path
    tm = json.loads(Path(token_mapping_path).read_text())
    _TOKEN_MAP = {tuple(t): i for i, t in enumerate(tm["triples"])}
    _VOCAB_SIZE = tm["vocab_size"]
    return _TOKEN_MAP, _VOCAB_SIZE


# ── decoding ───────────────────────────────────────────────────────────
def decode_aspk_all_tokens(data):
    """Decode aspk binary, tokenize ALL edges. Returns (list[int], n_entries).

    No function filter, no busy-min threshold. Idle ticks are kept.
    Returns (tokens, n_entries) where tokens may be empty for 0-entry files.
    """
    n = (len(data) - 8) // 8
    if n <= 0:
        return [], 0
    arr = np.frombuffer(data[8:8 + n * 8], dtype=np.uint8).reshape(n, 8)
    arr = arr.astype(np.int64)
    to_pc   = arr[:, 0] | (arr[:, 1] << 8) | (arr[:, 2] << 16)
    from_pc = arr[:, 3] | (arr[:, 4] << 8) | (arr[:, 5] << 16)
    func    = arr[:, 6] | (arr[:, 7] << 8)

    tokens = []
    n_skipped = 0
    for i in range(n):
        key = (int(func[i]), int(from_pc[i]), int(to_pc[i]))
        tid = _TOKEN_MAP.get(key)
        if tid is None:
            n_skipped += 1  # edge not in vocab → skip (e.g. pruned library function)
        else:
            tokens.append(tid)
    return tokens, n, n_skipped


def parse_aspk_filename(name):
    base = name[:-5] if name.endswith(".aspk") else name
    parts = base.split("_ts:")
    cc = int(parts[0].split(":", 1)[1])
    ts = int(parts[1])
    return cc, ts


# ── per-file worker ────────────────────────────────────────────────────
def process_file(path_str):
    """Worker. Returns dict."""
    try:
        with open(path_str, "rb") as f:
            data = f.read()
        name = os.path.basename(path_str)
        cc, ts = parse_aspk_filename(name)
        tokens, n_entries, n_skipped = decode_aspk_all_tokens(data)
        return {
            "cc": cc, "ts_us": ts,
            "tokens": tokens,
            "trace_length": len(tokens),
            "n_raw_entries": n_entries,
            "n_skipped": n_skipped,
        }
    except Exception as e:
        return {"_error": True, "path": path_str, "err": str(e)}


def init_worker():
    load_assets(_TOKEN_MAP_PATH)


# ── session driver ─────────────────────────────────────────────────────
def list_session_aspk(extracted_dir, session):
    trace = Path(extracted_dir) / session / "trace"
    return [e.path for e in os.scandir(trace) if e.name.endswith(".aspk")]


def load_datalayer(extracted_dir, session):
    dl = pd.read_csv(Path(extracted_dir) / session / "datalayer" / "datalayer.csv")
    dl["ts_us"] = (dl["timestamp"] // 1000).astype(np.int64)
    dl = dl.sort_values("ts_us").reset_index(drop=True)
    return dl


def build_session(session, extracted_dir, output_dir, workers, log_every=50_000):
    from concurrent.futures import ProcessPoolExecutor

    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{session}] starting full-vocab build (workers={workers})")
    paths = list_session_aspk(extracted_dir, session)
    print(f"[{session}]   found {len(paths)} files in {time.time()-t0:.1f}s")

    print(f"[{session}] loading datalayer...")
    dl = load_datalayer(extracted_dir, session)
    dl_ts = dl["ts_us"].values
    print(f"[{session}]   {len(dl)} rows  ({time.time()-t0:.1f}s)")

    sensor_arrs = {c: dl[c].values for c in SENSOR_COLS if c in dl.columns}
    have_cols = list(sensor_arrs.keys())

    rows = []
    n_err = n_unaligned = 0
    n_unknown_tok = 0

    def accept(res, i):
        nonlocal n_err, n_unaligned, n_unknown_tok
        if res.get("_error"):
            n_err += 1
            return
        ts = res["ts_us"]
        j = np.searchsorted(dl_ts, ts)
        if j >= len(dl_ts):
            j = len(dl_ts) - 1
        if j > 0 and abs(int(dl_ts[j-1]) - ts) < abs(int(dl_ts[j]) - ts):
            j -= 1
        d = abs(int(dl_ts[j]) - ts)
        if d > ALIGN_MAX_US:
            n_unaligned += 1
            return
        row = {
            "session_id": session,
            "cycle_count": res["cc"],
            "timestamp_us": ts,
            "align_err_us": d,
            "trace_length": res["trace_length"],
            "trace": res["tokens"],
        }
        for c in have_cols:
            row[c] = float(sensor_arrs[c][j])
        rows.append(row)
        n_unknown_tok += res.get("n_unknown", 0) + res.get("n_skipped", 0)
        if i % log_every == 0:
            print(f"[{session}]   processed {i}/{len(paths)}  "
                  f"kept={len(rows)}  unaligned={n_unaligned}  "
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
    print(f"[{session}]   kept={len(rows)}  unaligned={n_unaligned}  err={n_err}")
    print(f"[{session}]   skipped tokens (pruned/unknown, across all kept rows): {n_unknown_tok}")

    # Write parquet
    print(f"[{session}] building parquet...")
    df = pd.DataFrame(rows)
    df = df.sort_values("cycle_count").reset_index(drop=True)
    out_path = output_dir / f"{session}.parquet"
    df.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    print(f"[{session}] wrote {out_path}  ({len(df)} rows, "
          f"{out_path.stat().st_size/1e6:.1f} MB)")

    # Per-session metadata
    state_col = "pendulum_state"
    state_counts = {}
    if state_col in df.columns:
        sc = df[state_col].astype(int).value_counts().to_dict()
        state_counts = {int(k): int(v) for k, v in sc.items()}

    trace_lens = df["trace_length"].values if len(df) else np.array([])
    meta = {
        "session": session,
        "n_files_total": len(paths),
        "n_kept": len(df),
        "n_unaligned": n_unaligned,
        "n_err": n_err,
        "n_skipped_tokens": n_unknown_tok,
        "vocab_size": _VOCAB_SIZE,
        "fsm_state_counts": state_counts,
        "trace_length": {
            "mean": float(trace_lens.mean()) if len(trace_lens) else 0,
            "median": float(np.median(trace_lens)) if len(trace_lens) else 0,
            "min": int(trace_lens.min()) if len(trace_lens) else 0,
            "max": int(trace_lens.max()) if len(trace_lens) else 0,
            "std": float(trace_lens.std()) if len(trace_lens) else 0,
        },
        "elapsed_s": time.time() - t0,
    }
    meta_path = output_dir / f"{session}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[{session}] DONE in {time.time()-t0:.1f}s")
    return meta


# ── main ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Build full-vocabulary parquet from one ASPK session.")
    ap.add_argument("--session", required=True,
        help="Session directory name (e.g. 2025-03-18_12-39-10)")
    ap.add_argument("--extracted-dir", required=True,
        help="Path to extracted/ directory containing session folders")
    ap.add_argument("--token-mapping", required=True,
        help="Path to auto_token_mapping.json")
    ap.add_argument("--output-dir", required=True,
        help="Directory for output parquet and metadata files")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    load_assets(args.token_mapping)
    print(f"Loaded token vocab: {_VOCAB_SIZE}")

    build_session(args.session, args.extracted_dir, args.output_dir, args.workers)


if __name__ == "__main__":
    main()
