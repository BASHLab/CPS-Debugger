#!/usr/bin/env python3
"""Generate a concise markdown report for trace prediction experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from cps_trace_pred.pipeline import ExperimentConfig, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate markdown report for CPS trace prediction.")
    parser.add_argument("--results_json", type=Path, default=None, help="Load precomputed results from JSON.")
    parser.add_argument("--save_json", type=Path, default=None, help="Save computed results to JSON.")
    parser.add_argument("--save_md", type=Path, default=None, help="Save markdown report to file.")

    # If --results_json is not provided, use the same run config as train.py.
    parser.add_argument("--runs", nargs="+", default=None, help="Run directories.")
    parser.add_argument("--split_mode", choices=["blocked", "run_holdout"], default="blocked")
    parser.add_argument("--test_runs", nargs="*", default=None)
    parser.add_argument("--win_ms", type=int, default=10)
    parser.add_argument("--feature_lag", type=int, default=1)
    parser.add_argument("--target", choices=["functions", "edges"], default="edges")
    parser.add_argument("--topk_edges", type=int, default=100)
    parser.add_argument("--features", choices=["datalayer", "syslog", "fused"], default="fused")
    parser.add_argument("--model", choices=["logreg", "mlp"], default="logreg")
    parser.add_argument("--missing_mode", choices=["none", "test_drop", "train_dropout"], default="none")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_iter", type=int, default=300)
    parser.add_argument("--mlp_hidden", type=int, default=256)
    parser.add_argument("--mlp_dropout", type=float, default=0.2)
    parser.add_argument("--mlp_lr", type=float, default=1e-3)
    parser.add_argument("--mlp_batch_size", type=int, default=256)
    parser.add_argument("--mlp_epochs", type=int, default=25)
    parser.add_argument("--train_dropout_prob", type=float, default=0.2)
    return parser


def render_markdown(result: Dict) -> str:
    cfg = result["config"]
    stats = result["dataset_stats"]
    metrics = result["metrics"]
    missing = result.get("missing_metrics", {})
    per_class = result.get("per_class_f1_top10", [])

    lines = [
        "# Trace Prediction Report",
        "",
        "## Setup",
        "",
        f"- Runs: `{len(cfg['runs'])}`",
        f"- Split mode: `{result['split'].get('split_mode', 'random')}`",
        f"- Train runs: `{len(result['split'].get('train_runs', []))}`",
        f"- Test runs: `{len(result['split'].get('test_runs', []))}`",
        f"- Target: `{cfg['target']}`",
        f"- Features: `{cfg['features']}`",
        f"- Feature lag: `{cfg.get('feature_lag', 1)}`",
        f"- Model: `{cfg['model']}`",
        f"- Missing mode: `{cfg['missing_mode']}`",
        f"- Window size: `{cfg['win_ms']} ms`",
        "",
        "## Dataset Stats",
        "",
        f"- Windows: `{stats['num_windows']}`",
        f"- Features: `{stats['num_features']}`",
        f"- Label cardinality (avg positives/window): `{stats['label_cardinality']:.3f}`",
        f"- Label space (total/used): `{result['label_space']['labels_total']}/{result['label_space']['labels_used_for_training']}`",
        "",
        "Top labels:",
    ]
    for row in stats.get("top_labels", []):
        lines.append(f"- `{row['pretty_label']}`: {row['count']}")

    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Micro-F1 | {metrics['micro_f1']:.4f} |",
            f"| Macro-F1 | {metrics['macro_f1']:.4f} |",
            f"| Precision@5 | {metrics['precision_at_5']:.4f} |",
            f"| Precision@10 | {metrics['precision_at_10']:.4f} |",
        ]
    )

    if "micro_f1_missing_syslog" in missing:
        lines.append(f"| Micro-F1 (missing syslog) | {missing['micro_f1_missing_syslog']:.4f} |")
    if "macro_f1_missing_syslog" in missing:
        lines.append(f"| Macro-F1 (missing syslog) | {missing['macro_f1_missing_syslog']:.4f} |")
    if "micro_f1_missing_datalayer" in missing:
        lines.append(f"| Micro-F1 (missing datalayer) | {missing['micro_f1_missing_datalayer']:.4f} |")
    if "macro_f1_missing_datalayer" in missing:
        lines.append(f"| Macro-F1 (missing datalayer) | {missing['macro_f1_missing_datalayer']:.4f} |")

    lines.extend(["", "## Interpretability Summary", ""])
    if "micro_f1_missing_syslog" in missing:
        delta = metrics["micro_f1"] - missing["micro_f1_missing_syslog"]
        lines.append(f"- Removing syslog decreases Micro-F1 by `{delta:.4f}`.")
    if "micro_f1_missing_datalayer" in missing:
        delta = metrics["micro_f1"] - missing["micro_f1_missing_datalayer"]
        lines.append(f"- Removing datalayer decreases Micro-F1 by `{delta:.4f}`.")
    if not missing:
        lines.append("- Missing-modality stress test not enabled for this run.")

    if cfg["target"] == "functions" and per_class:
        lines.extend(["", "Top-function per-class F1 (top-10 by frequency):"])
        for row in per_class:
            lines.append(f"- `{row['pretty_label']}`: F1={row['f1']:.4f}, support={row['support']}")

    lag_compare = result.get("lag_compare", {})
    if lag_compare:
        lines.extend(["", "## Lag Leakage Check", ""])
        for lag_key in ["lag_0", "lag_1"]:
            if lag_key not in lag_compare:
                continue
            m = lag_compare[lag_key]
            lines.append(
                f"- `{lag_key.replace('_', '=')}`: "
                f"micro={m['micro_f1']:.4f}, macro={m['macro_f1']:.4f}, "
                f"p@5={m['precision_at_5']:.4f}, p@10={m['precision_at_10']:.4f}"
            )

    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    if args.results_json is not None:
        if not args.results_json.exists():
            raise FileNotFoundError(f"Missing results file: {args.results_json}")
        result = json.loads(args.results_json.read_text())
    else:
        if not args.runs:
            raise ValueError("Either provide --results_json or supply --runs and experiment settings.")
        cfg = ExperimentConfig(
            runs=args.runs,
            split_mode=args.split_mode,
            test_runs=args.test_runs,
            win_ms=args.win_ms,
            feature_lag=args.feature_lag,
            target=args.target,
            topk_edges=args.topk_edges,
            features=args.features,
            model=args.model,
            missing_mode=args.missing_mode,
            test_size=args.test_size,
            random_state=args.random_state,
            threshold=args.threshold,
            max_iter=args.max_iter,
            mlp_hidden=args.mlp_hidden,
            mlp_dropout=args.mlp_dropout,
            mlp_lr=args.mlp_lr,
            mlp_batch_size=args.mlp_batch_size,
            mlp_epochs=args.mlp_epochs,
            train_dropout_prob=args.train_dropout_prob,
        )
        result = run_experiment(cfg)

    markdown = render_markdown(result)
    print(markdown)

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(result, indent=2))
    if args.save_md is not None:
        args.save_md.parent.mkdir(parents=True, exist_ok=True)
        args.save_md.write_text(markdown)


if __name__ == "__main__":
    main()
