import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPS debugger baseline for pendulum traces.")
    parser.add_argument("--run", type=Path, required=True, help="Path to one run directory.")
    parser.add_argument(
        "--vocab",
        type=Path,
        default=None,
        help="Optional vocabulary file (supports JSON or Python literal dict/list).",
    )
    parser.add_argument(
        "--label-mode",
        choices=["auto", "edges", "triplets"],
        default="auto",
        help="Trace label type. auto=triplets if tuple vocab is provided, else edges.",
    )
    parser.add_argument("--topk", type=int, default=300, help="Top-K labels to keep when building vocab.")
    parser.add_argument("--win-us", type=int, default=10_000, help="Window size in microseconds.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--random-state", type=int, default=0, help="Random seed.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold.")
    parser.add_argument("--max-iter", type=int, default=400, help="Max iterations for logistic regression.")
    parser.add_argument("--top-anoms", type=int, default=10, help="How many anomalies to report.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for artifacts. Defaults to <run>/cps_debugger_outputs.",
    )
    return parser.parse_args()


def load_csvs(run: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dl = pd.read_csv(run / "datalayer" / "datalayer.csv")
    sl = pd.read_csv(run / "syslog" / "syslog.csv")
    tr = pd.read_csv(run / "processed_trace" / "processed_trace.csv")
    return dl, sl, tr


def safe_delta(x: pd.Series) -> float:
    if x.empty:
        return 0.0
    return float(x.iloc[-1] - x.iloc[0])


def to_win(s: pd.Series, win_us: int) -> pd.Series:
    return (s // win_us).astype(np.int64)


def build_features(dl: pd.DataFrame, sl: pd.DataFrame, tr: pd.DataFrame, win_us: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dl["t_us"] = (dl["timestamp"] // 1000).astype(np.int64)  # ns -> us
    sl["t_us"] = sl["Timestamp"].astype(np.int64)
    tr["t_us"] = tr["timestamp"].astype(np.int64)

    dl["win"] = to_win(dl["t_us"], win_us)
    sl["win"] = to_win(sl["t_us"], win_us)
    tr["win"] = to_win(tr["t_us"], win_us)

    dl_feat = dl.groupby("win").agg(
        dl_target_x_mean=("target_x", "mean"),
        dl_current_x_mean=("current_x", "mean"),
        dl_velocity_mean=("velocity", "mean"),
        dl_angle_mean=("current_angle", "mean"),
        dl_ang_vel_mean=("angular_velocity", "mean"),
        dl_current_x_std=("current_x", "std"),
        dl_angle_std=("current_angle", "std"),
        dl_d_current_x=("current_x", safe_delta),
        dl_d_angle=("current_angle", safe_delta),
    )

    sl_feat = sl.groupby("win").agg(
        sl_cpu_mean=("CPU_Usage(%)", "mean"),
        sl_mem_mean=("Memory_Usage(%)", "mean"),
        sl_load1_mean=("Load_1m", "mean"),
        sl_load5_mean=("Load_5m", "mean"),
        sl_load15_mean=("Load_15m", "mean"),
    )

    return dl_feat, sl_feat


def parse_edges(cell: str) -> Set[str]:
    try:
        edges = json.loads(cell)
        return {f"{a}->{b}" for a, b in edges}
    except Exception:
        return set()


def parse_triplet_keys(cell: str) -> Set[Tuple[str, str, str]]:
    result: Set[Tuple[str, str, str]] = set()
    try:
        cf = json.loads(cell)
    except Exception:
        return result
    for func_id, src_dict in cf.items():
        for src_pc, dst_dict in src_dict.items():
            for dst_pc in dst_dict.keys():
                result.add((str(func_id), str(src_pc), str(dst_pc)))
    return result


def load_vocab(vocab_path: Path) -> Tuple[Optional[List], str]:
    if vocab_path is None or not vocab_path.exists():
        return None, "none"

    text = vocab_path.read_text()
    data = None
    source_format = "json"
    try:
        data = json.loads(text)
    except Exception:
        source_format = "py_literal"
        try:
            data = ast.literal_eval(text)
        except Exception:
            return None, "unreadable"

    if isinstance(data, dict):
        if not data:
            return [], source_format
        # Typical format: label -> idx
        if all(isinstance(v, int) for v in data.values()):
            ordered = [k for k, _ in sorted(data.items(), key=lambda kv: kv[1])]
            return ordered, source_format
        # Fallback: keep keys in insertion order
        return list(data.keys()), source_format

    if isinstance(data, list):
        return data, source_format

    return None, "unsupported"


def choose_label_mode(requested: str, vocab: Optional[List]) -> str:
    if requested in {"edges", "triplets"}:
        return requested
    if vocab and isinstance(vocab[0], tuple) and len(vocab[0]) == 3:
        return "triplets"
    return "edges"


def build_labels(
    tr: pd.DataFrame,
    label_mode: str,
    topk: int,
    vocab_from_file: Optional[List],
) -> Tuple[np.ndarray, np.ndarray, List]:
    if label_mode == "edges":
        tr["label_set"] = tr["function_graph_edges"].apply(parse_edges)
    else:
        tr["label_set"] = tr["cf_table"].apply(parse_triplet_keys)

    by_win = tr.groupby("win")["label_set"].apply(lambda sets: set().union(*sets))
    wins = by_win.index.to_numpy(dtype=np.int64)

    if vocab_from_file:
        vocab = list(vocab_from_file)
    else:
        ctr = Counter()
        for s in by_win:
            ctr.update(s)
        vocab = [item for item, _ in ctr.most_common(topk)]

    v2i = {label: idx for idx, label in enumerate(vocab)}
    y = np.zeros((len(wins), len(vocab)), dtype=np.int8)
    for row_idx, labels in enumerate(by_win):
        for label in labels:
            idx = v2i.get(label)
            if idx is not None:
                y[row_idx, idx] = 1
    return y, wins, vocab


def load_layout_names(layout_path: Path) -> Dict[str, str]:
    if not layout_path.exists():
        return {}
    try:
        layout = json.loads(layout_path.read_text())
    except Exception:
        return {}
    names = {}
    for k, v in layout.items():
        if isinstance(v, dict) and "name" in v:
            names[str(k)] = str(v["name"])
    return names


def pretty_label(label, label_mode: str, fn_names: Dict[str, str]) -> str:
    if label_mode == "edges":
        a, b = str(label).split("->", 1)
        return f"{fn_names.get(a, a)}({a})->{fn_names.get(b, b)}({b})"
    func_id, src, dst = label
    func_name = fn_names.get(str(func_id), str(func_id))
    return f"{func_name}({func_id}):{src}->{dst}"


def main() -> None:
    args = parse_args()
    run = args.run.resolve()
    out_dir = args.output_dir.resolve() if args.output_dir else run / "cps_debugger_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    dl, sl, tr = load_csvs(run)
    dl_feat, sl_feat = build_features(dl, sl, tr, args.win_us)

    tr["t_us"] = tr["timestamp"].astype(np.int64)
    tr["win"] = to_win(tr["t_us"], args.win_us)

    loaded_vocab, vocab_format = load_vocab(args.vocab) if args.vocab else (None, "none")
    label_mode = choose_label_mode(args.label_mode, loaded_vocab)
    y, y_wins, vocab = build_labels(
        tr=tr,
        label_mode=label_mode,
        topk=args.topk,
        vocab_from_file=loaded_vocab,
    )

    x = dl_feat.join(sl_feat, how="left")
    x = x.reindex(y_wins).sort_index()
    x = x.ffill().bfill().fillna(0.0)
    feature_names = x.columns.tolist()
    x_vals = x.values

    x_train, x_test, y_train, y_test = train_test_split(
        x_vals, y, test_size=args.test_size, random_state=args.random_state
    )

    # Keep labels that are trainable for binary logistic models.
    train_pos = y_train.sum(axis=0)
    valid_labels = (train_pos > 0) & (train_pos < y_train.shape[0])
    if not np.any(valid_labels):
        raise RuntimeError("No trainable labels found after split; try a larger top-k or different run.")

    vocab_used = [vocab[i] for i, ok in enumerate(valid_labels) if ok]
    y_train_used = y_train[:, valid_labels]
    y_test_used = y_test[:, valid_labels]
    y_used = y[:, valid_labels]

    sc = StandardScaler()
    x_train_s = sc.fit_transform(x_train)
    x_test_s = sc.transform(x_test)

    clf = OneVsRestClassifier(LogisticRegression(max_iter=args.max_iter, n_jobs=-1))
    clf.fit(x_train_s, y_train_used)

    y_pred = (clf.predict_proba(x_test_s) >= args.threshold).astype(int)
    micro = float(f1_score(y_test_used.ravel(), y_pred.ravel(), zero_division=0))
    macro = float(f1_score(y_test_used, y_pred, average="macro", zero_division=0))

    sys_cols = [i for i, c in enumerate(feature_names) if c.startswith("sl_")]
    micro_missing_syslog = None
    if sys_cols:
        x_test_miss = x_test.copy()
        x_test_miss[:, sys_cols] = 0.0
        y_pred_miss = (clf.predict_proba(sc.transform(x_test_miss)) >= args.threshold).astype(int)
        micro_missing_syslog = float(f1_score(y_test_used.ravel(), y_pred_miss.ravel(), zero_division=0))

    probs_all = clf.predict_proba(sc.transform(x_vals))
    eps = 1e-6
    score = (
        y_used * (-np.log(np.clip(probs_all, eps, 1 - eps)))
        + (1 - y_used) * (-np.log(np.clip(1 - probs_all, eps, 1 - eps)))
    ).sum(axis=1)

    top_idx = np.argsort(score)[-args.top_anoms :][::-1]
    fn_names = load_layout_names(run / "code" / "layout.json")
    anomalies: List[Dict] = []
    for rank, idx in enumerate(top_idx, start=1):
        surprises = []
        for j in np.where(y_used[idx] == 1)[0]:
            label = vocab_used[j]
            surprises.append(
                {
                    "label": pretty_label(label, label_mode, fn_names),
                    "raw_label": str(label),
                    "surprise": float(-np.log(np.clip(probs_all[idx, j], eps, 1.0))),
                }
            )
        surprises.sort(key=lambda item: item["surprise"], reverse=True)
        anomalies.append(
            {
                "rank": rank,
                "window": int(y_wins[idx]),
                "score": float(score[idx]),
                "top_surprises": surprises[:5],
            }
        )

    metrics = {
        "run": str(run),
        "win_us": int(args.win_us),
        "label_mode": label_mode,
        "vocab_path": str(args.vocab) if args.vocab else None,
        "vocab_format": vocab_format,
        "num_windows": int(len(y_wins)),
        "num_features": int(x_vals.shape[1]),
        "num_labels_total": int(len(vocab)),
        "num_labels_used": int(len(vocab_used)),
        "train_size": int(x_train.shape[0]),
        "test_size": int(x_test.shape[0]),
        "micro_f1": micro,
        "macro_f1": macro,
        "micro_f1_missing_syslog": micro_missing_syslog,
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(
        [
            {
                "rank": a["rank"],
                "window": a["window"],
                "score": a["score"],
                "top_surprises": json.dumps(a["top_surprises"]),
            }
            for a in anomalies
        ]
    ).to_csv(out_dir / "anomalies.csv", index=False)

    run_start_us = int(y_wins.min()) * args.win_us
    run_end_us = (int(y_wins.max()) + 1) * args.win_us
    run_start_iso = pd.to_datetime(run_start_us, unit="us", utc=True).strftime("%Y-%m-%d %H:%M:%S.%f UTC")
    run_end_iso = pd.to_datetime(run_end_us, unit="us", utc=True).strftime("%Y-%m-%d %H:%M:%S.%f UTC")

    missing_line = "- Micro-F1 (missing syslog): `N/A`"
    robustness_line = "- Robustness note: syslog ablation not available."
    if micro_missing_syslog is not None:
        drop_abs = micro - micro_missing_syslog
        drop_rel = (drop_abs / micro) * 100.0 if micro > 0 else 0.0
        missing_line = f"- Micro-F1 (missing syslog): `{micro_missing_syslog:.4f}`"
        robustness_line = (
            f"- Robustness note: removing syslog reduces Micro-F1 by "
            f"`{drop_abs:.4f}` (`{drop_rel:.1f}%` relative)."
        )

    report_lines = [
        "# CPS Debugger Report",
        "",
        "## Task",
        "",
        "Build a lightweight CPS debugger baseline that predicts execution-level behavior from synchronized",
        "telemetry, then highlights time windows where observed execution is unlikely under the learned model.",
        "",
        "## Why This Matters",
        "",
        "- Cyber-physical incidents often appear first as cross-modal inconsistency (physical state looks normal, but control flow changes).",
        "- Full trace inspection at every cycle is expensive; a model can triage suspicious windows quickly.",
        "- Missing modalities are common in production capture pipelines, so robustness to absent inputs matters.",
        "",
        "## Run Context",
        "",
        f"- Run path: `{run}`",
        f"- Approximate run span: `{run_start_iso}` to `{run_end_iso}`",
        f"- Window size: `{args.win_us}` us (`{args.win_us / 1000:.1f}` ms)",
        f"- Label mode: `{label_mode}`",
        f"- Vocabulary source: `{vocab_format}`",
        (
            f"- Vocabulary file: `{args.vocab}`"
            if args.vocab is not None
            else "- Vocabulary file: `None (auto-built from this run)`"
        ),
        "",
        "## Method Summary",
        "",
        "- Aggregate `datalayer` and `syslog` signals per time window into statistical/dynamic features.",
        "- Convert trace data into multi-label execution targets (`triplets` from `cf_table` here).",
        "- Train a one-vs-rest logistic regression baseline to map sensor features -> execution labels.",
        "- Score each window with negative log-likelihood; larger score means more surprising execution.",
        "- Localize anomalies by listing labels with the highest surprise within each top window.",
        "",
        "## Quantitative Results",
        "",
        f"- Micro-F1: `{micro:.4f}`",
        f"- Macro-F1: `{macro:.4f}`",
        missing_line,
        robustness_line,
        f"- Windows: `{len(y_wins)}`",
        f"- Features: `{x_vals.shape[1]}`",
        f"- Labels (used / total): `{len(vocab_used)} / {len(vocab)}`",
        f"- Train/Test windows: `{x_train.shape[0]} / {x_test.shape[0]}`",
        "",
        "## Interpretation",
        "",
        "- High Micro-F1 indicates strong performance on frequent control-flow patterns.",
        "- Lower Macro-F1 indicates rarer labels are harder and remain a key improvement area.",
        "- The syslog ablation gap quantifies dependence on system-health context for prediction stability.",
        "",
        "## Top Anomalous Windows",
        "",
        "These are ranked by total negative log-likelihood. Each row lists the most surprising localized labels.",
        "",
    ]

    for a in anomalies:
        anom_ts_us = int(a["window"]) * args.win_us
        anom_ts_iso = pd.to_datetime(anom_ts_us, unit="us", utc=True).strftime("%Y-%m-%d %H:%M:%S.%f UTC")
        report_lines.append(
            f"### Rank {a['rank']} | win={a['window']} | ts={anom_ts_iso} | score={a['score']:.3f}"
        )
        if not a["top_surprises"]:
            report_lines.append("- No observed labels in trained vocabulary for this window.")
        else:
            for s in a["top_surprises"]:
                report_lines.append(f"- `{s['label']}` (surprise={s['surprise']:.3f})")
        report_lines.append("")

    report_lines.extend(
        [
            "## Artifacts",
            "",
            f"- Metrics JSON: `{out_dir / 'metrics.json'}`",
            f"- Anomaly table CSV: `{out_dir / 'anomalies.csv'}`",
            f"- This report: `{out_dir / 'run_report.md'}`",
            "",
            "## Caveats",
            "",
            "- This baseline is linear and window-local; it does not model longer temporal dependencies.",
            "- Anomaly scores are relative to this run and split, not global calibrated probabilities.",
            "- Stronger baselines could include sequence models, class balancing, and calibrated thresholds.",
        ]
    )

    (out_dir / "run_report.md").write_text("\n".join(report_lines))

    print(f"Run: {run}")
    print(f"Output dir: {out_dir}")
    print(f"Label mode: {label_mode}")
    print(f"Vocab size used/total: {len(vocab_used)}/{len(vocab)}")
    print(f"Micro-F1: {micro:.4f}")
    print(f"Macro-F1: {macro:.4f}")
    if micro_missing_syslog is not None:
        print(f"Micro-F1 (missing syslog): {micro_missing_syslog:.4f}")
    print("Top anomalous windows:")
    for a in anomalies:
        print(f"  rank={a['rank']} win={a['window']} score={a['score']:.3f}")


if __name__ == "__main__":
    main()
