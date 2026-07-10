"""Windowed run loader for CPS trace prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .trace_utils import edge_to_str, functions_from_edges, load_layout_map, parse_function_graph_edges


def _delta_last_first(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.iloc[-1] - series.iloc[0])


@dataclass
class WindowedDataset:
    """Container returned by ``RunDatasetBuilder.build``."""

    X: np.ndarray
    Y: np.ndarray
    feature_names: List[str]
    label_names: List[str]
    label_counts: Dict[str, int]
    modality_slices: Dict[str, Tuple[int, int]]
    metadata: pd.DataFrame
    layout_map: Dict[str, str]
    target: str
    win_us: int
    feature_lag: int


class RunDatasetBuilder:
    """Build a windowed multi-run dataset for trace prediction.

    Assumptions:
    - Each run directory contains:
      - ``datalayer/datalayer.csv``
      - ``syslog/syslog.csv``
      - ``processed_trace/processed_trace.csv``
    - Timestamps follow the provided spec (datalayer in ns, others in us).
    """

    # Strictly allowed telemetry fields for model inputs.
    DL_BASE_COLUMNS = ["current_x", "current_angle"]
    DL_LAG_ONLY_COLUMNS = ["angular_velocity"]
    SL_COLUMNS = ["CPU_Usage(%)", "Memory_Usage(%)", "Load_1m", "Load_5m", "Load_15m"]

    def __init__(
        self,
        run_dirs: Sequence[str],
        win_ms: int = 10,
        target: str = "edges",
        topk_edges: int = 100,
        features: str = "fused",
        feature_lag: int = 1,
    ) -> None:
        self.run_dirs = [Path(r).resolve() for r in run_dirs]
        self.win_us = int(win_ms) * 1000
        self.target = target
        self.topk_edges = int(topk_edges)
        self.features = features
        self.feature_lag = int(feature_lag)
        if self.feature_lag < 0:
            raise ValueError("feature_lag must be >= 0")

    def _load_run_csvs(self, run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        dl_path = run_dir / "datalayer" / "datalayer.csv"
        sl_path = run_dir / "syslog" / "syslog.csv"
        tr_path = run_dir / "processed_trace" / "processed_trace.csv"
        if not dl_path.exists():
            raise FileNotFoundError(f"Missing datalayer file: {dl_path}")
        if not sl_path.exists():
            raise FileNotFoundError(f"Missing syslog file: {sl_path}")
        if not tr_path.exists():
            raise FileNotFoundError(f"Missing trace file: {tr_path}")
        return pd.read_csv(dl_path), pd.read_csv(sl_path), pd.read_csv(tr_path)

    def _to_win(self, series: pd.Series) -> pd.Series:
        return (series // self.win_us).astype(np.int64)

    def _allowed_datalayer_columns(self) -> List[str]:
        cols = list(self.DL_BASE_COLUMNS)
        if self.feature_lag > 0:
            cols.extend(self.DL_LAG_ONLY_COLUMNS)
        return cols

    def _aggregate_datalayer(self, dl: pd.DataFrame) -> pd.DataFrame:
        dl = dl.copy()
        dl["t_us"] = (dl["timestamp"] // 1000).astype(np.int64)
        dl["win"] = self._to_win(dl["t_us"])

        agg_spec = {}
        for col in self._allowed_datalayer_columns():
            if col not in dl.columns:
                raise ValueError(f"Required datalayer column missing: {col}")
            agg_spec[f"dl_{col}_mean"] = (col, "mean")
            agg_spec[f"dl_{col}_std"] = (col, "std")
            agg_spec[f"dl_{col}_delta"] = (col, _delta_last_first)

        dl_feat = dl.groupby("win").agg(**agg_spec).sort_index()
        dl_feat = dl_feat.fillna(0.0)
        return dl_feat

    def _aggregate_syslog(self, sl: pd.DataFrame) -> pd.DataFrame:
        sl = sl.copy()
        sl["t_us"] = sl["Timestamp"].astype(np.int64)
        sl["win"] = self._to_win(sl["t_us"])
        sl_feat = sl.groupby("win").agg(
            sl_cpu_mean=("CPU_Usage(%)", "mean"),
            sl_mem_mean=("Memory_Usage(%)", "mean"),
            sl_load1_mean=("Load_1m", "mean"),
            sl_load5_mean=("Load_5m", "mean"),
            sl_load15_mean=("Load_15m", "mean"),
        )
        return sl_feat.sort_index()

    def _aggregate_trace_labels(self, tr: pd.DataFrame) -> pd.Series:
        tr = tr.copy()
        tr["t_us"] = tr["timestamp"].astype(np.int64)
        tr["win"] = self._to_win(tr["t_us"])

        if self.target == "edges":
            tr["label_set"] = tr["function_graph_edges"].apply(
                lambda cell: {edge_to_str(a, b) for a, b in parse_function_graph_edges(cell)}
            )
        elif self.target == "functions":
            tr["label_set"] = tr["function_graph_edges"].apply(
                lambda cell: set(functions_from_edges(parse_function_graph_edges(cell)))
            )
        else:
            raise ValueError(f"Unknown target type: {self.target}")

        return tr.groupby("win")["label_set"].apply(lambda items: set().union(*items))

    def _select_feature_columns(self, full_cols: List[str]) -> List[str]:
        dl_cols = [c for c in full_cols if c.startswith("dl_")]
        sl_cols = [c for c in full_cols if c.startswith("sl_")]

        if self.features == "datalayer":
            return dl_cols
        if self.features == "syslog":
            return sl_cols
        if self.features == "fused":
            return dl_cols + sl_cols
        raise ValueError(f"Unknown features setting: {self.features}")

    def build(self) -> WindowedDataset:
        run_tables: List[pd.DataFrame] = []
        all_label_sets: List[Set[str]] = []
        layout_map: Dict[str, str] = {}

        for run_dir in self.run_dirs:
            dl, sl, tr = self._load_run_csvs(run_dir)
            dl_feat = self._aggregate_datalayer(dl)
            sl_feat = self._aggregate_syslog(sl)
            labels_by_win = self._aggregate_trace_labels(tr)

            full_feat = dl_feat.join(sl_feat, how="outer").sort_index()
            sl_cols_full = [c for c in full_feat.columns if c.startswith("sl_")]
            if sl_cols_full:
                # Causal fill: carry only past syslog information forward.
                full_feat[sl_cols_full] = full_feat[sl_cols_full].ffill()

            # Temporal causality: features for window t come from telemetry at t - lag.
            target_wins = labels_by_win.index.astype(np.int64)
            feature_wins = target_wins - self.feature_lag
            aligned = full_feat.reindex(feature_wins).copy()
            aligned.index = target_wins

            # Drop windows with no telemetry context after lag shift.
            valid_mask = aligned.notna().any(axis=1)
            aligned = aligned.loc[valid_mask]
            labels_aligned = labels_by_win.reindex(aligned.index)

            aligned = aligned.fillna(0.0)

            frame = aligned.copy()
            frame["run"] = run_dir.name
            frame["win"] = frame.index.astype(np.int64)
            frame["label_set"] = labels_aligned.values
            run_tables.append(frame.reset_index(drop=True))
            all_label_sets.extend(frame["label_set"].tolist())

            run_layout_map = load_layout_map(run_dir / "code" / "layout.json")
            for func_id, func_name in run_layout_map.items():
                layout_map.setdefault(func_id, func_name)

        if not run_tables:
            raise RuntimeError("No run data loaded.")

        all_df = pd.concat(run_tables, ignore_index=True)
        all_feature_cols = [c for c in all_df.columns if c.startswith("dl_") or c.startswith("sl_")]
        feature_cols = self._select_feature_columns(all_feature_cols)
        if not feature_cols:
            raise RuntimeError("No features selected. Check --features argument.")

        # Build label vocabulary from window-level label presence.
        label_counter: Dict[str, int] = {}
        for label_set in all_label_sets:
            for label in label_set:
                label_counter[label] = label_counter.get(label, 0) + 1

        sorted_labels = sorted(label_counter.items(), key=lambda kv: kv[1], reverse=True)
        if self.target == "edges":
            sorted_labels = sorted_labels[: self.topk_edges]
        label_names = [label for label, _ in sorted_labels]
        label_counts = {label: count for label, count in sorted_labels}

        label_to_idx = {label: idx for idx, label in enumerate(label_names)}
        n_samples = len(all_df)
        y = np.zeros((n_samples, len(label_names)), dtype=np.int8)
        for row_idx, label_set in enumerate(all_df["label_set"]):
            for label in label_set:
                idx = label_to_idx.get(label)
                if idx is not None:
                    y[row_idx, idx] = 1

        x = all_df[feature_cols].astype(np.float32).to_numpy()

        # Track slices for modality drop experiments.
        dl_cols = [c for c in feature_cols if c.startswith("dl_")]
        sl_cols = [c for c in feature_cols if c.startswith("sl_")]
        modality_slices: Dict[str, Tuple[int, int]] = {}
        start = 0
        if dl_cols:
            modality_slices["datalayer"] = (start, start + len(dl_cols))
            start += len(dl_cols)
        if sl_cols:
            modality_slices["syslog"] = (start, start + len(sl_cols))
            start += len(sl_cols)

        metadata = all_df[["run", "win"]].copy()
        return WindowedDataset(
            X=x,
            Y=y,
            feature_names=feature_cols,
            label_names=label_names,
            label_counts=label_counts,
            modality_slices=modality_slices,
            metadata=metadata,
            layout_map=layout_map,
            target=self.target,
            win_us=self.win_us,
            feature_lag=self.feature_lag,
        )
