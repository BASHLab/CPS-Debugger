"""
00_preprocess.py — Build the unified aligned dataset from raw run CSVs.

Reads datalayer, syslog, processed_trace CSVs from each run directory,
aggregates all modalities to 10ms windows, parses cf_table branch counts,
and saves parquet files for fast downstream loading.

Run: python3 scripts/00_preprocess.py [--runs run1 run2 ...] [--win-ms 10] [--top-k 100]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── defaults ──────────────────────────────────────────────────────────────────
DATA_ROOT = Path(
    "/home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs/extracted"
)
OUT_DIR = Path(
    "/home/simran/allspark-data-exploration/CPS-Debugger/outputs/experiments"
)
WIN_MS = 10
TOP_K = 100  # top-K most-variable branches to keep in vocabulary


# ── helpers ───────────────────────────────────────────────────────────────────
def us_to_win(t_us: pd.Series, win_us: int) -> pd.Series:
    return (t_us // win_us).astype(np.int64)


def load_datalayer(run: Path, win_us: int) -> pd.DataFrame:
    df = pd.read_csv(run / "datalayer" / "datalayer.csv")
    df["t_us"] = (df["timestamp"] // 1000).astype(np.int64)  # ns → us
    df["win"] = us_to_win(df["t_us"], win_us)
    agg = df.groupby("win").agg(
        dl_state=("pendulum_state", "first"),
        dl_target_x=("target_x", "first"),
        dl_iteration=("iteration", "first"),
        dl_current_x_mean=("current_x", "mean"),
        dl_current_x_std=("current_x", "std"),
        dl_current_x_delta=(
            "current_x",
            lambda x: float(x.iloc[-1] - x.iloc[0]) if len(x) > 1 else 0.0,
        ),
        dl_angle_mean=("current_angle", "mean"),
        dl_angle_std=("current_angle", "std"),
        dl_angle_delta=(
            "current_angle",
            lambda x: float(x.iloc[-1] - x.iloc[0]) if len(x) > 1 else 0.0,
        ),
        dl_velocity_mean=("velocity", "mean"),
        dl_ang_vel_mean=("angular_velocity", "mean"),
        dl_ang_vel_std=("angular_velocity", "std"),
    ).reset_index()
    agg["dl_state"] = agg["dl_state"].fillna(-1).astype(int)
    return agg.fillna(0.0)


def load_syslog(run: Path, win_us: int) -> pd.DataFrame:
    df = pd.read_csv(run / "syslog" / "syslog.csv")
    df["t_us"] = df["Timestamp"].astype(np.int64)
    df["win"] = us_to_win(df["t_us"], win_us)
    agg = (
        df.groupby("win")
        .agg(
            sl_cpu=("CPU_Usage(%)", "mean"),
            sl_mem=("Memory_Usage(%)", "mean"),
            sl_load1=("Load_1m", "mean"),
            sl_load5=("Load_5m", "mean"),
            sl_load15=("Load_15m", "mean"),
        )
        .reset_index()
    )
    return agg.ffill().fillna(0.0)


def parse_cf_row(cf_str: str) -> dict:
    """Parse one cf_table cell → {func_id:src_pc:dst_pc: count}."""
    try:
        cf = json.loads(cf_str)
    except Exception:
        return {}
    result = {}
    for func_id, src_dict in cf.items():
        for src_pc, dst_dict in src_dict.items():
            for dst_pc, count in dst_dict.items():
                key = f"{func_id}:{src_pc}:{dst_pc}"
                result[key] = result.get(key, 0) + int(count)
    return result


def load_trace_counts(run: Path, win_us: int) -> tuple:
    """Parse processed_trace.csv → (branch_counts_df, edges_df)."""
    tr = pd.read_csv(run / "processed_trace" / "processed_trace.csv")
    tr["t_us"] = tr["timestamp"].astype(np.int64)
    tr["win"] = us_to_win(tr["t_us"], win_us)

    n = len(tr)
    print(f"  Parsing {n:,} trace rows from {run.name}...")
    t0 = time.time()

    counts_rows = []
    edges_rows = []
    for win_id, group in tqdm(tr.groupby("win"), desc="  windows", leave=False):
        # Branch counts: sum over all 1ms rows in this 10ms window
        window_counts: dict = {}
        for cf_str in group["cf_table"].values:
            for k, v in parse_cf_row(cf_str).items():
                window_counts[k] = window_counts.get(k, 0) + v
        window_counts["win"] = win_id
        counts_rows.append(window_counts)

        # Function call graph edges: union over window
        edge_set: set = set()
        for cell in group["function_graph_edges"].values:
            try:
                for a, b in json.loads(cell):
                    edge_set.add(f"{a}->{b}")
            except Exception:
                pass
        edges_rows.append({"win": win_id, "edges": sorted(edge_set)})

    print(f"  Done in {time.time()-t0:.1f}s")
    counts_df = pd.DataFrame(counts_rows).fillna(0.0)
    counts_df["win"] = counts_df["win"].astype(np.int64)
    edges_df = pd.DataFrame(edges_rows)
    edges_df["win"] = edges_df["win"].astype(np.int64)
    return counts_df, edges_df


def build_vocabulary(all_counts: list, top_k: int) -> tuple:
    """Select top-K branches by coefficient of variation (std/mean).
    Must appear in >5% of windows."""
    combined = pd.concat(all_counts, ignore_index=True)
    n_total = len(combined)
    branch_cols = [c for c in combined.columns if c != "win"]

    stats = []
    for col in branch_cols:
        vals = combined[col].values
        presence = (vals > 0).sum()
        if presence < 0.05 * n_total:
            continue
        mean_val = vals.mean()
        std_val = vals.std()
        cv = std_val / (mean_val + 1e-9)
        stats.append(
            {
                "branch": col,
                "cv": cv,
                "mean": mean_val,
                "std": std_val,
                "presence_frac": presence / n_total,
            }
        )

    stats_df = pd.DataFrame(stats).sort_values("cv", ascending=False)
    vocab = stats_df.head(top_k)["branch"].tolist()
    return vocab, stats_df


def load_layout(run: Path) -> dict:
    layout_path = run / "code" / "layout.json"
    if not layout_path.exists():
        return {}
    try:
        layout = json.loads(layout_path.read_text())
        return {
            k: v.get("name", k) if isinstance(v, dict) else k
            for k, v in layout.items()
        }
    except Exception:
        return {}


def branch_label(key: str, fn_names: dict) -> str:
    parts = key.split(":")
    if len(parts) == 3:
        func_id, src_pc, dst_pc = parts
        name = fn_names.get(func_id, func_id)
        return f"{name}:{src_pc}→{dst_pc}"
    return key


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, default=None)
    parser.add_argument("--win-ms", type=int, default=WIN_MS)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    win_us = args.win_ms * 1000
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    if args.runs is None:
        # All 6 extracted March runs
        runs = sorted(DATA_ROOT.glob("2025-03-*"))
        print(f"Auto-detected {len(runs)} extracted runs")
    else:
        runs = [r.resolve() for r in args.runs]

    print(
        f"Processing {len(runs)} run(s), window={args.win_ms}ms, top-k={args.top_k}"
    )

    # Pass 1: load all modalities per run
    all_dl, all_sl, all_counts, all_edges, fn_names = [], [], [], [], {}

    for run in runs:
        print(f"\n── {run.name} ──")
        dl = load_datalayer(run, win_us)
        sl = load_syslog(run, win_us)
        counts_df, edges_df = load_trace_counts(run, win_us)

        for df in (dl, sl, counts_df, edges_df):
            df["run"] = run.name

        all_dl.append(dl)
        all_sl.append(sl)
        all_counts.append(counts_df.drop(columns=["run"]))  # for vocab building
        all_edges.append(edges_df)

        if not fn_names:
            fn_names = load_layout(run)

    # Pass 2: build vocabulary across all runs
    print("\nBuilding branch vocabulary...")
    vocab, vocab_stats = build_vocabulary(all_counts, args.top_k)
    print(f"Selected {len(vocab)} branches (top-{args.top_k} by CV)")

    # Pass 3: build aligned datasets per run
    all_aligned = []
    for i, run in enumerate(runs):
        dl = all_dl[i]
        sl = all_sl[i]
        counts_df = all_counts[i].copy()
        counts_df["run"] = run.name

        # Reindex to vocab (fill missing branches with 0)
        for col in vocab:
            if col not in counts_df.columns:
                counts_df[col] = 0.0
        bc = counts_df[["win", "run"] + vocab]

        merged = dl.merge(
            sl[["win", "sl_cpu", "sl_mem", "sl_load1", "sl_load5", "sl_load15"]],
            on="win",
            how="left",
        )
        merged = merged.merge(bc[["win"] + vocab], on="win", how="inner")
        merged = merged.ffill().bfill().fillna(0.0)
        all_aligned.append(merged)

    combined = pd.concat(all_aligned, ignore_index=True)
    # Sort by window index within run for temporal correctness
    combined = combined.sort_values(["run", "win"]).reset_index(drop=True)

    print(f"\nTotal windows: {len(combined):,}")
    print(f"State distribution: {combined['dl_state'].value_counts().to_dict()}")

    # Save outputs
    combined.to_parquet(out / "aligned_dataset.parquet", index=False)

    vocab_stats["label"] = vocab_stats["branch"].apply(
        lambda k: branch_label(k, fn_names)
    )
    vocab_stats.to_parquet(out / "vocab_stats.parquet", index=False)

    vocab_info = {
        "vocab": vocab,
        "labels": {b: branch_label(b, fn_names) for b in vocab},
        "fn_names": fn_names,
        "win_ms": args.win_ms,
        "top_k": args.top_k,
        "runs": [r.name for r in runs],
    }
    (out / "vocab_info.json").write_text(json.dumps(vocab_info, indent=2))

    print(f"\nSaved: {out / 'aligned_dataset.parquet'}")
    print(f"Saved: {out / 'vocab_stats.parquet'}")
    print(f"Saved: {out / 'vocab_info.json'}")
    print("\nTop-10 most variable branches:")
    print(
        vocab_stats[["label", "cv", "mean", "presence_frac"]]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
