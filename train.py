#!/usr/bin/env python3
"""CLI entrypoint for training/evaluating CPS trace prediction baselines."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from cps_trace_pred.pipeline import ExperimentConfig, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate CPS trace prediction baselines.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories.")
    parser.add_argument(
        "--split_mode",
        choices=["blocked", "run_holdout"],
        default="blocked",
        help="Blocked time split within run, or explicit held-out run split.",
    )
    parser.add_argument(
        "--test_runs",
        nargs="*",
        default=None,
        help="Run directory names/paths to hold out when --split_mode=run_holdout.",
    )
    parser.add_argument("--win_ms", type=int, default=10, help="Window size in milliseconds.")
    parser.add_argument(
        "--feature_lag",
        type=int,
        default=1,
        help="Predict trace(t) from telemetry(t-lag). Default 1.",
    )
    parser.add_argument("--target", choices=["functions", "edges"], required=True)
    parser.add_argument("--topk_edges", type=int, default=100, help="Top-K edges for edge target.")
    parser.add_argument("--features", choices=["datalayer", "syslog", "fused"], required=True)
    parser.add_argument("--model", choices=["logreg", "mlp"], required=True)
    parser.add_argument(
        "--missing_mode",
        choices=["none", "test_drop", "train_dropout"],
        default="none",
    )

    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_iter", type=int, default=300)

    parser.add_argument("--mlp_hidden", type=int, default=256)
    parser.add_argument("--mlp_dropout", type=float, default=0.2)
    parser.add_argument("--mlp_lr", type=float, default=1e-3)
    parser.add_argument("--mlp_weight_decay", type=float, default=1e-4)
    parser.add_argument("--mlp_batch_size", type=int, default=256)
    parser.add_argument("--mlp_epochs", type=int, default=25)
    parser.add_argument("--train_dropout_prob", type=float, default=0.2)
    parser.add_argument(
        "--mlp_no_pos_weight",
        action="store_true",
        help="Disable class-imbalance positive weighting for MLP BCE loss.",
    )
    parser.add_argument("--mlp_max_pos_weight", type=float, default=20.0)
    parser.add_argument("--mlp_grad_clip_norm", type=float, default=1.0)

    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional JSON path to save full experiment results.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
        mlp_weight_decay=args.mlp_weight_decay,
        mlp_batch_size=args.mlp_batch_size,
        mlp_epochs=args.mlp_epochs,
        train_dropout_prob=args.train_dropout_prob,
        mlp_use_pos_weight=not args.mlp_no_pos_weight,
        mlp_max_pos_weight=args.mlp_max_pos_weight,
        mlp_grad_clip_norm=args.mlp_grad_clip_norm,
    )

    result = run_experiment(cfg)

    sanity = result.get("sanity", {})
    print(f"Feature lag:    {sanity.get('feature_lag', cfg.feature_lag)}")
    print("Final feature columns:")
    for col in sanity.get("feature_columns", []):
        print(f"  - {col}")
    print(f"Leakage assertions passed: {sanity.get('leakage_assertions_passed', False)}")
    print("Feature-label_cardinality correlation:")
    corr_map = sanity.get("feature_label_cardinality_corr", {})
    for col, corr in sorted(corr_map.items()):
        print(f"  {col}: {corr:.6f}")

    print(f"Split mode:     {result['split']['split_mode']}")
    print(f"Train runs:     {', '.join(result['split']['train_runs'])}")
    print(f"Test runs:      {', '.join(result['split']['test_runs'])}")
    print(f"Train windows: {result['split']['train_windows']}")
    print(f"Test windows:  {result['split']['test_windows']}")
    print(f"Micro-F1:      {result['metrics']['micro_f1']:.4f}")
    print(f"Macro-F1:      {result['metrics']['macro_f1']:.4f}")
    print(f"Precision@5:   {result['metrics']['precision_at_5']:.4f}")
    print(f"Precision@10:  {result['metrics']['precision_at_10']:.4f}")

    missing_syslog = result["missing_metrics"].get("micro_f1_missing_syslog")
    missing_syslog_macro = result["missing_metrics"].get("macro_f1_missing_syslog")
    missing_dl = result["missing_metrics"].get("micro_f1_missing_datalayer")
    missing_dl_macro = result["missing_metrics"].get("macro_f1_missing_datalayer")
    print(
        "Micro-F1 missing syslog (drop): "
        + (f"{missing_syslog:.4f}" if missing_syslog is not None else "N/A")
    )
    print(
        "Macro-F1 missing syslog (drop): "
        + (f"{missing_syslog_macro:.4f}" if missing_syslog_macro is not None else "N/A")
    )
    print(
        "Micro-F1 missing datalayer (drop): "
        + (f"{missing_dl:.4f}" if missing_dl is not None else "N/A")
    )
    print(
        "Macro-F1 missing datalayer (drop): "
        + (f"{missing_dl_macro:.4f}" if missing_dl_macro is not None else "N/A")
    )

    # Always print lag-0 vs lag-1 metrics to inspect possible post-decision leakage.
    lag0_result = run_experiment(replace(cfg, feature_lag=0))
    lag1_result = run_experiment(replace(cfg, feature_lag=1))
    print("Lag comparison:")
    print(
        f"  lag=0 -> micro={lag0_result['metrics']['micro_f1']:.4f}, "
        f"macro={lag0_result['metrics']['macro_f1']:.4f}, "
        f"p@5={lag0_result['metrics']['precision_at_5']:.4f}, "
        f"p@10={lag0_result['metrics']['precision_at_10']:.4f}"
    )
    print(
        f"  lag=1 -> micro={lag1_result['metrics']['micro_f1']:.4f}, "
        f"macro={lag1_result['metrics']['macro_f1']:.4f}, "
        f"p@5={lag1_result['metrics']['precision_at_5']:.4f}, "
        f"p@10={lag1_result['metrics']['precision_at_10']:.4f}"
    )
    result["lag_compare"] = {
        "lag_0": lag0_result["metrics"],
        "lag_1": lag1_result["metrics"],
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2))
        print(f"Saved results JSON: {args.output_json}")


if __name__ == "__main__":
    main()
