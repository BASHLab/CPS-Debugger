#!/usr/bin/env python3
"""Execution Trace EDA — characterize data for FM sequence generation.

Outputs:
  - session_manifest.json: per-session data availability
  - edge_vocabulary.json: all unique (func, from_pc, to_pc) triplets with stats
  - sequence_length_stats.json: distribution of per-tick sequence lengths
  - fig_sequence_stats.png: visualization of key metrics
"""

import json
import struct
import os
import glob
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs")
EXTRACTED = BASE / "extracted"
EXTRACTED_FEB = BASE / "extracted_feb04" / "data"
OUT_DIR = Path(__file__).resolve().parent
ALIGNED = Path("/home/simran/allspark-data-exploration/CPS-Debugger/outputs/experiments/aligned_dataset.parquet")


def build_session_manifest():
    """Build manifest of all available sessions and their data."""
    manifest = {}

    for sess_dir in sorted(EXTRACTED.iterdir()):
        if not sess_dir.is_dir():
            continue
        sess = sess_dir.name
        pt = sess_dir / "processed_trace" / "processed_trace.csv"
        dl = sess_dir / "datalayer" / "datalayer.csv"
        layout = sess_dir / "code" / "layout.json"
        trace = sess_dir / "trace"

        entry = {
            "path": str(sess_dir),
            "has_processed_trace": pt.exists(),
            "processed_trace_mb": round(pt.stat().st_size / 1e6, 1) if pt.exists() else 0,
            "has_datalayer": dl.exists(),
            "has_layout": layout.exists(),
            "has_raw_aspk": trace.is_dir() and any(trace.iterdir()),
        }

        if pt.exists():
            df = pd.read_csv(pt, usecols=["timestamp"])
            entry["n_rows"] = len(df)
            entry["duration_sec"] = round((df["timestamp"].max() - df["timestamp"].min()) / 1e6, 1)
        manifest[sess] = entry

    # Feb04 session
    feb_dir = EXTRACTED_FEB / "2025-02-04_09-33-16"
    if feb_dir.exists():
        pt = feb_dir / "processed_trace" / "processed_trace.csv"
        if pt.exists():
            df = pd.read_csv(pt, usecols=["timestamp"])
            manifest["2025-02-04_09-33-16"] = {
                "path": str(feb_dir),
                "has_processed_trace": True,
                "processed_trace_mb": round(pt.stat().st_size / 1e6, 1),
                "has_datalayer": (feb_dir / "datalayer" / "datalayer.csv").exists(),
                "has_layout": (feb_dir / "code" / "layout.json").exists(),
                "has_raw_aspk": False,
                "n_rows": len(df),
                "duration_sec": round((df["timestamp"].max() - df["timestamp"].min()) / 1e6, 1),
            }

    return manifest


def parse_aspk_file(filepath):
    """Parse a raw .aspk binary file into a list of (func_id, from_pc, to_pc) tuples."""
    with open(filepath, "rb") as f:
        data = f.read()
    n_entries = len(data) // 8 - 1
    sequence = []
    for i in range(1, n_entries + 1):
        entry = data[i * 8:(i + 1) * 8]
        target_pc = int.from_bytes(entry[0:3], "little")
        source_pc = int.from_bytes(entry[3:6], "little")
        func_idx = int.from_bytes(entry[6:8], "little")
        sequence.append((func_idx, source_pc, target_pc))
    return sequence


def build_edge_vocabulary(session_dir, n_sample=1000):
    """Build edge vocabulary from sampled raw .aspk files."""
    trace_dir = session_dir / "trace"
    files = sorted(glob.glob(str(trace_dir / "*.aspk")))
    if not files:
        return None, None

    indices = np.linspace(0, len(files) - 1, min(n_sample, len(files)), dtype=int)
    edge_counter = Counter()
    seq_lengths = []

    for idx in indices:
        seq = parse_aspk_file(files[idx])
        seq_lengths.append(len(seq))
        for triplet in seq:
            edge_counter[triplet] += 1

    return edge_counter, seq_lengths


def load_layout(session_dir):
    """Load layout.json and return function mapping."""
    layout_path = session_dir / "code" / "layout.json"
    if not layout_path.exists():
        return {}
    with open(layout_path) as f:
        layout = json.load(f)
    func_map = {}
    for k, v in layout.items():
        if k.isdigit():
            func_map[int(k)] = v.get("name", f"func_{k}")
    return func_map


def main():
    print("=== Building session manifest ===")
    manifest = build_session_manifest()

    manifest_path = OUT_DIR / "session_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {len(manifest)} sessions to {manifest_path}")

    # Print summary
    aspk_sessions = [s for s, m in manifest.items() if m.get("has_raw_aspk")]
    pt_only = [s for s, m in manifest.items() if m.get("has_processed_trace") and not m.get("has_raw_aspk")]
    print(f"  Sessions with raw .aspk: {len(aspk_sessions)}")
    print(f"  Sessions with processed_trace only: {len(pt_only)}")

    # Build edge vocabulary from first session with raw aspk
    if aspk_sessions:
        print(f"\n=== Building edge vocabulary from {aspk_sessions[0]} ===")
        sess_dir = Path(manifest[aspk_sessions[0]]["path"])
        func_map = load_layout(sess_dir)

        edge_counter, seq_lengths = build_edge_vocabulary(sess_dir, n_sample=1000)
        if edge_counter:
            # Save vocabulary
            vocab = {}
            for (func_id, from_pc, to_pc), count in edge_counter.most_common():
                key = f"{func_id}:{from_pc}:{to_pc}"
                func_name = func_map.get(func_id, f"func_{func_id}")
                vocab[key] = {
                    "func_id": func_id,
                    "func_name": func_name,
                    "from_pc": from_pc,
                    "to_pc": to_pc,
                    "total_count": count,
                }

            vocab_path = OUT_DIR / "edge_vocabulary.json"
            with open(vocab_path, "w") as f:
                json.dump(vocab, f, indent=2)
            print(f"Saved {len(vocab)} unique edges to {vocab_path}")

            # Categorize
            controller_funcs = {22, 33, 34, 35, 36, 37, 38, 39, 40, 42, 63, 64}
            ctrl_edges = {k: v for k, v in vocab.items() if v["func_id"] in controller_funcs}
            lib_edges = {k: v for k, v in vocab.items() if v["func_id"] not in controller_funcs}
            print(f"  Controller edges: {len(ctrl_edges)}")
            print(f"  Library edges: {len(lib_edges)}")

            # Sequence length stats
            sl = np.array(seq_lengths)
            stats = {
                "n_samples": len(sl),
                "mean": float(sl.mean()),
                "std": float(sl.std()),
                "min": int(sl.min()),
                "max": int(sl.max()),
                "p25": float(np.percentile(sl, 25)),
                "p50": float(np.percentile(sl, 50)),
                "p75": float(np.percentile(sl, 75)),
            }

            # Estimate controller-only sequence lengths
            ctrl_total = sum(v["total_count"] for v in ctrl_edges.values())
            all_total = sum(v["total_count"] for v in vocab.values())
            ctrl_ratio = ctrl_total / all_total if all_total else 0
            stats["ctrl_only_estimated_mean"] = float(sl.mean() * ctrl_ratio)
            stats["ctrl_ratio"] = round(ctrl_ratio, 3)

            stats_path = OUT_DIR / "sequence_length_stats.json"
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"\nSequence stats: mean={stats['mean']:.0f}, "
                  f"ctrl_only_est={stats['ctrl_only_estimated_mean']:.0f}")

            # === Figure ===
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # 1. Sequence length distribution
            ax = axes[0, 0]
            ax.hist(sl, bins=50, color="#378ADD", edgecolor="white", alpha=0.8)
            ax.axvline(sl.mean(), color="red", linestyle="--", label=f"mean={sl.mean():.0f}")
            ax.set_xlabel("Sequence length (branches per tick)")
            ax.set_ylabel("Count")
            ax.set_title("Per-Tick Sequence Length Distribution")
            ax.legend()

            # 2. Edge frequency (log scale)
            ax = axes[0, 1]
            counts = sorted(edge_counter.values(), reverse=True)
            ax.bar(range(len(counts)), counts, color="#1D9E75", width=1.0)
            ax.set_yscale("log")
            ax.set_xlabel("Edge rank")
            ax.set_ylabel("Total count (log)")
            ax.set_title(f"Edge Frequency Distribution ({len(counts)} unique edges)")
            ax.axvline(len(ctrl_edges), color="red", linestyle="--",
                        label=f"Controller edges: {len(ctrl_edges)}")
            ax.legend()

            # 3. Edges by function
            ax = axes[1, 0]
            func_edges = defaultdict(int)
            func_counts = defaultdict(int)
            for (fid, _, _), cnt in edge_counter.items():
                func_edges[fid] += 1
                func_counts[fid] += cnt
            sorted_funcs = sorted(func_counts.keys(), key=lambda x: func_counts[x], reverse=True)[:15]
            labels = [f"{fid}\n{func_map.get(fid, '?')[:12]}" for fid in sorted_funcs]
            values = [func_counts[fid] for fid in sorted_funcs]
            colors = ["#378ADD" if fid in controller_funcs else "#90C695" for fid in sorted_funcs]
            ax.bar(labels, values, color=colors, edgecolor="white")
            ax.set_ylabel("Total branch executions")
            ax.set_title("Branch Executions by Function (blue=controller, green=library)")
            ax.tick_params(axis="x", rotation=45)

            # 4. Session durations
            ax = axes[1, 1]
            sess_names = sorted(manifest.keys())
            durations = [manifest[s].get("duration_sec", 0) for s in sess_names]
            colors = ["#378ADD" if manifest[s].get("has_raw_aspk") else "#CCCCCC" for s in sess_names]
            short_names = [s[5:10] for s in sess_names]  # just the date part
            ax.barh(short_names, durations, color=colors, edgecolor="white")
            ax.set_xlabel("Duration (seconds)")
            ax.set_title("Session Durations (blue=has raw .aspk, gray=processed only)")

            plt.tight_layout()
            fig_path = OUT_DIR / "fig_trace_eda.png"
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"\nSaved figure to {fig_path}")


if __name__ == "__main__":
    main()
