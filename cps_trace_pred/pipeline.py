"""Training/evaluation pipeline for CPS trace prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dataset import RunDatasetBuilder
from .metrics import compute_metrics, per_class_f1_top_n, threshold_predictions
from .models import MLPBaseline, MLPConfig, LogRegOVR, apply_modality_dropout_numpy, set_global_seed
from .trace_utils import pretty_edge, pretty_function


@dataclass
class ExperimentConfig:
    runs: List[str]
    split_mode: str = "blocked"  # {"blocked", "run_holdout"}
    test_runs: Optional[List[str]] = None
    win_ms: int = 10
    feature_lag: int = 1
    target: str = "edges"
    topk_edges: int = 100
    features: str = "fused"
    model: str = "logreg"
    missing_mode: str = "none"
    test_size: float = 0.2
    random_state: int = 0
    threshold: float = 0.5
    max_iter: int = 300
    mlp_hidden: int = 256
    mlp_dropout: float = 0.2
    mlp_lr: float = 1e-3
    mlp_weight_decay: float = 1e-4
    mlp_batch_size: int = 256
    mlp_epochs: int = 25
    train_dropout_prob: float = 0.2
    mlp_use_pos_weight: bool = True
    mlp_max_pos_weight: float = 20.0
    mlp_grad_clip_norm: float = 1.0


def _pretty_label(label: str, target: str, layout_map: Dict[str, str]) -> str:
    if target == "functions":
        return pretty_function(label, layout_map)
    return pretty_edge(label, layout_map)


def _drop_and_score(
    predictor,
    x_test: np.ndarray,
    y_test: np.ndarray,
    modality_slice: Tuple[int, int],
    threshold: float,
) -> Dict[str, float]:
    dropped = x_test.copy()
    start, end = modality_slice
    dropped[:, start:end] = 0.0
    probs = predictor.predict_proba(dropped)
    metrics = compute_metrics(y_test, probs, threshold=threshold)
    return {
        "micro_f1": float(metrics["micro_f1"]),
        "macro_f1": float(metrics["macro_f1"]),
    }


def run_experiment(cfg: ExperimentConfig) -> Dict:
    set_global_seed(cfg.random_state)
    builder = RunDatasetBuilder(
        run_dirs=cfg.runs,
        win_ms=cfg.win_ms,
        target=cfg.target,
        topk_edges=cfg.topk_edges,
        features=cfg.features,
        feature_lag=cfg.feature_lag,
    )
    dataset = builder.build()

    x = dataset.X
    y = dataset.Y
    if y.shape[1] == 0:
        raise RuntimeError("No labels found for the chosen target formulation.")

    indices = np.arange(x.shape[0])
    run_names = dataset.metadata["run"].astype(str).to_numpy()
    run_wins = dataset.metadata["win"].astype(np.int64).to_numpy()

    # Sanity checks for leakage-prone features.
    feature_cols = list(dataset.feature_names)
    disallowed_substrings = [
        "dl_target_x",
        "dl_velocity_",
        "dl_pendulum_state",
        "dl_iteration",
        "cf_table",
        "function_graph_edges",
        "trace_edge",
        "trace_branch",
        "unique_function",
    ]
    for col in feature_cols:
        for token in disallowed_substrings:
            if token in col:
                raise AssertionError(f"Leaky feature detected: {col}")
    if cfg.feature_lag == 0:
        if any("dl_angular_velocity_" in c for c in feature_cols):
            raise AssertionError("angular_velocity features are only allowed when feature_lag > 0")

    label_cardinality = y.sum(axis=1).astype(np.float32) if y.size else np.zeros((x.shape[0],), dtype=np.float32)
    feature_label_card_corr: Dict[str, float] = {}
    for i, col in enumerate(feature_cols):
        x_col = x[:, i].astype(np.float32)
        if np.allclose(x_col, x_col[0]):
            corr = 0.0
        else:
            corr = float(np.corrcoef(x_col, label_cardinality)[0, 1])
            if np.isnan(corr):
                corr = 0.0
        feature_label_card_corr[col] = corr

    if cfg.split_mode == "blocked":
        train_parts = []
        test_parts = []
        for run_name in sorted(set(run_names.tolist())):
            run_idx = indices[run_names == run_name]
            run_idx = run_idx[np.argsort(run_wins[run_idx])]
            if len(run_idx) < 2:
                continue
            cut = int(np.floor(0.7 * len(run_idx)))
            cut = min(max(cut, 1), len(run_idx) - 1)
            train_parts.append(run_idx[:cut])
            test_parts.append(run_idx[cut:])
        if not train_parts or not test_parts:
            raise ValueError("Blocked split failed; not enough windows per run.")
        idx_train = np.concatenate(train_parts)
        idx_test = np.concatenate(test_parts)
    elif cfg.split_mode == "run_holdout":
        if not cfg.test_runs:
            raise ValueError("split_mode=run_holdout requires --test_runs.")
        holdout_names = {Path(r).name for r in cfg.test_runs}
        test_mask = np.isin(run_names, list(holdout_names))
        idx_test = indices[test_mask]
        idx_train = indices[~test_mask]
        if len(idx_test) == 0:
            raise ValueError(
                f"No test windows matched holdout runs: {sorted(holdout_names)}. "
                f"Available runs: {sorted(set(run_names))}"
            )
        if len(idx_train) == 0:
            raise ValueError("Holdout split produced zero training windows.")
    else:
        raise ValueError(f"Unsupported split_mode: {cfg.split_mode}")

    x_train, x_test = x[idx_train], x[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    # Remove labels that are constant in training split to avoid degenerate classifiers.
    train_support = y_train.sum(axis=0)
    valid_mask = (train_support > 0) & (train_support < y_train.shape[0])
    if not np.any(valid_mask):
        raise RuntimeError("No trainable labels after split. Try more runs or a larger test split.")

    y_train_used = y_train[:, valid_mask]
    y_test_used = y_test[:, valid_mask]
    used_label_names = [name for i, name in enumerate(dataset.label_names) if valid_mask[i]]
    used_label_counts = {
        name: int(dataset.label_counts.get(name, 0))
        for i, name in enumerate(dataset.label_names)
        if valid_mask[i]
    }

    if cfg.model == "logreg":
        x_train_fit = x_train
        if cfg.missing_mode == "train_dropout":
            x_train_fit = apply_modality_dropout_numpy(
                x_train,
                modality_slices=dataset.modality_slices,
                drop_prob=cfg.train_dropout_prob,
                rng=np.random.default_rng(cfg.random_state),
            )
        predictor = LogRegOVR(max_iter=cfg.max_iter)
        predictor.fit(x_train_fit, y_train_used)
    elif cfg.model == "mlp":
        mlp_cfg = MLPConfig(
            hidden_dim=cfg.mlp_hidden,
            dropout=cfg.mlp_dropout,
            lr=cfg.mlp_lr,
            weight_decay=cfg.mlp_weight_decay,
            batch_size=cfg.mlp_batch_size,
            epochs=cfg.mlp_epochs,
            train_dropout_prob=cfg.train_dropout_prob,
            use_pos_weight=cfg.mlp_use_pos_weight,
            max_pos_weight=cfg.mlp_max_pos_weight,
            grad_clip_norm=cfg.mlp_grad_clip_norm,
            seed=cfg.random_state,
        )
        predictor = MLPBaseline(
            input_dim=x_train.shape[1],
            output_dim=y_train_used.shape[1],
            cfg=mlp_cfg,
            modality_slices=dataset.modality_slices,
            enable_train_modality_dropout=(cfg.missing_mode == "train_dropout"),
        )
        predictor.fit(x_train, y_train_used)
    else:
        raise ValueError(f"Unsupported model: {cfg.model}")

    probs = predictor.predict_proba(x_test)
    metrics = compute_metrics(y_test_used, probs, threshold=cfg.threshold, ks=(5, 10))
    y_pred = threshold_predictions(probs, threshold=cfg.threshold)

    per_class_rows = []
    if cfg.target == "functions":
        per_class_rows = per_class_f1_top_n(
            y_true=y_test_used,
            y_pred=y_pred,
            label_names=used_label_names,
            n=10,
        )
        for row in per_class_rows:
            row["pretty_label"] = _pretty_label(row["label"], cfg.target, dataset.layout_map)

    # Missing-modality robustness at test time (zeroing modality blocks).
    missing_metrics: Dict[str, float] = {}
    if cfg.missing_mode in {"test_drop", "train_dropout"}:
        if "syslog" in dataset.modality_slices:
            dropped_metrics = _drop_and_score(
                predictor,
                x_test=x_test,
                y_test=y_test_used,
                modality_slice=dataset.modality_slices["syslog"],
                threshold=cfg.threshold,
            )
            missing_metrics["micro_f1_missing_syslog"] = dropped_metrics["micro_f1"]
            missing_metrics["macro_f1_missing_syslog"] = dropped_metrics["macro_f1"]
        if "datalayer" in dataset.modality_slices:
            dropped_metrics = _drop_and_score(
                predictor,
                x_test=x_test,
                y_test=y_test_used,
                modality_slice=dataset.modality_slices["datalayer"],
                threshold=cfg.threshold,
            )
            missing_metrics["micro_f1_missing_datalayer"] = dropped_metrics["micro_f1"]
            missing_metrics["macro_f1_missing_datalayer"] = dropped_metrics["macro_f1"]

    top_labels = []
    for label, count in sorted(dataset.label_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]:
        top_labels.append(
            {
                "label": label,
                "pretty_label": _pretty_label(label, cfg.target, dataset.layout_map),
                "count": int(count),
            }
        )

    label_cardinality_avg = float(y.sum(axis=1).mean()) if y.size else 0.0
    result = {
        "config": asdict(cfg),
        "split": {
            "split_mode": cfg.split_mode,
            "train_windows": int(len(idx_train)),
            "test_windows": int(len(idx_test)),
            "train_runs": sorted(set(run_names[idx_train].tolist())),
            "test_runs": sorted(set(run_names[idx_test].tolist())),
        },
        "dataset_stats": {
            "num_runs": int(dataset.metadata["run"].nunique()),
            "num_windows": int(x.shape[0]),
            "num_features": int(x.shape[1]),
            "label_cardinality": label_cardinality_avg,
            "target_type": cfg.target,
            "top_labels": top_labels,
        },
        "label_space": {
            "labels_total": int(len(dataset.label_names)),
            "labels_used_for_training": int(len(used_label_names)),
        },
        "metrics": metrics,
        "missing_metrics": missing_metrics,
        "per_class_f1_top10": per_class_rows,
        "sanity": {
            "feature_columns": feature_cols,
            "feature_label_cardinality_corr": feature_label_card_corr,
            "leakage_assertions_passed": True,
            "feature_lag": cfg.feature_lag,
        },
    }
    return result
