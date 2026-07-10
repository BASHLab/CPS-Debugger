"""Multi-modal data extension for BL-3 / BL-4 / BL-5.

Wraps TraceDataset with pre-aligned V-JEPA 2.1 + CoTracker3 lookups. For
each tick at timestamp T, we use the most recent frozen-extracted V-JEPA
embedding with end_timestamp_us ≤ T, and the most recent CoTracker
trajectory entry with timestamp_us ≤ T (strict causality).

This is the FROZEN-feature path. BL-3b (LoRA V-JEPA) requires running
V-JEPA inside the training graph — not handled here.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data import TraceDataset

VIDEO_DIR     = Path("/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates/video")
VJEPA_DIR     = VIDEO_DIR / "vjepa_embeddings"
COTRACKER_DIR = VIDEO_DIR / "cotracker_trajectories"

VJEPA_DIM     = 1024
COTRACKER_DIM = 60   # 20 points × (x, y, vis)


def _load_vjepa_session(session: str):
    """Load a session's V-JEPA parquet → (sorted_end_ts_us[N], features[N, 1024])."""
    path = VJEPA_DIR / f"{session}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"V-JEPA features missing: {path}")
    df = pd.read_parquet(path)
    end_ts = df["end_timestamp_us"].to_numpy(dtype=np.int64)
    feats  = np.stack(
        [np.asarray(x, dtype=np.float32) for x in df["features"].to_list()], axis=0
    )
    # Drop rows with invalid timestamps; keep monotone-sorted by end_ts
    keep = end_ts > 0
    end_ts, feats = end_ts[keep], feats[keep]
    order = np.argsort(end_ts, kind="stable")
    return end_ts[order], feats[order]


def _load_cotracker_session(session: str):
    """Load a session's CoTracker parquet → (sorted_ts_us[N], features[N, 60])."""
    path = COTRACKER_DIR / f"{session}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"CoTracker features missing: {path}")
    df = pd.read_parquet(path)
    ts = df["timestamp_us"].to_numpy(dtype=np.int64)
    coords = np.stack(
        [np.asarray(x, dtype=np.float32) for x in df["point_coords"].to_list()], axis=0
    )    # (N, 40) interleaved [x0,y0, ..., x19,y19]
    vis = np.stack(
        [np.asarray(x, dtype=np.float32) for x in df["point_visibility"].to_list()], axis=0
    )    # (N, 20)
    feats = np.concatenate([coords, vis], axis=1)   # (N, 60)
    keep = ts > 0
    ts, feats = ts[keep], feats[keep]
    order = np.argsort(ts, kind="stable")
    return ts[order], feats[order]


def _causal_index(sorted_ts: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    """For each q in query_ts, return idx of the largest sorted_ts ≤ q.
    Returns -1 if no such entry exists. searchsorted-right then -1."""
    idx = np.searchsorted(sorted_ts, query_ts, side="right") - 1
    return idx


class VideoConditionedTraceDataset(Dataset):
    """Wraps a TraceDataset and adds V-JEPA + CoTracker per-tick lookups.

    Args:
      base: a fully-loaded TraceDataset (or a torch.utils.data.Subset of one).
      use_vjepa: include V-JEPA embeddings.
      use_cotracker: include CoTracker trajectory features.

    __getitem__ returns the base dict plus:
      vjepa_emb     (1024,)  — zero vector if no causal V-JEPA available
      vjepa_valid   bool     — True iff ≥ 1 V-JEPA window finished by tick time
      cotracker_emb (60,)    — zero vector if no causal CoTracker available
      cotracker_valid bool   — True iff ≥ 1 CoTracker frame ≤ tick time
    """

    def __init__(
        self,
        base: Dataset,
        use_vjepa: bool = True,
        use_cotracker: bool = True,
    ):
        self.base = base
        self.use_vjepa = use_vjepa
        self.use_cotracker = use_cotracker

        # Resolve the underlying TraceDataset and the index map for any Subset wrapper
        td = base
        index_map: Optional[np.ndarray] = None
        if hasattr(base, "indices") and hasattr(base, "dataset"):       # torch Subset
            td = base.dataset
            index_map = np.asarray(base.indices, dtype=np.int64)
        if not isinstance(td, TraceDataset):
            raise TypeError("VideoConditionedTraceDataset expects TraceDataset (or Subset of one)")
        self._underlying = td

        sessions_in_order: list = sorted(set(td.session_ids))
        self._vjepa_data:     dict = {}      # session -> (end_ts, feats)
        self._cotracker_data: dict = {}      # session -> (ts, feats)

        for sess in sessions_in_order:
            if use_vjepa:
                self._vjepa_data[sess]     = _load_vjepa_session(sess)
            if use_cotracker:
                self._cotracker_data[sess] = _load_cotracker_session(sess)

        # Pre-compute per-base-sample lookups (vjepa_idx, cotracker_idx)
        n_base = len(td)
        ts_us  = np.asarray(td.timestamps_us, dtype=np.int64)
        sess_arr = np.asarray(td.session_ids)
        self._vjepa_idx     = np.full(n_base, -1, dtype=np.int64)
        self._cotracker_idx = np.full(n_base, -1, dtype=np.int64)
        self._vjepa_session_offsets = {}     # cache of feats handles
        self._cotracker_session_offsets = {}

        for sess in sessions_in_order:
            mask = sess_arr == sess
            if not mask.any():
                continue
            sub_ts = ts_us[mask]
            if use_vjepa:
                v_ts, _ = self._vjepa_data[sess]
                v_idx = _causal_index(v_ts, sub_ts)
                self._vjepa_idx[mask] = v_idx
            if use_cotracker:
                c_ts, _ = self._cotracker_data[sess]
                c_idx = _causal_index(c_ts, sub_ts)
                self._cotracker_idx[mask] = c_idx

        self._index_map = index_map        # base→underlying mapping for Subset; None if no subset
        # Quick sanity print
        n_v_invalid = int((self._vjepa_idx < 0).sum()) if use_vjepa else 0
        n_c_invalid = int((self._cotracker_idx < 0).sum()) if use_cotracker else 0
        print(f"  VideoConditionedTraceDataset: {n_base} samples, "
              f"V-JEPA invalid={n_v_invalid}, CoTracker invalid={n_c_invalid}")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        # Resolve i to underlying-TraceDataset idx (subset or direct)
        u_idx = int(self._index_map[i]) if self._index_map is not None else int(i)
        sess  = self._underlying.session_ids[u_idx]

        if self.use_vjepa:
            v_idx = int(self._vjepa_idx[u_idx])
            if v_idx >= 0:
                _, feats = self._vjepa_data[sess]
                v = torch.from_numpy(feats[v_idx]).float()
                v_valid = True
            else:
                v = torch.zeros(VJEPA_DIM, dtype=torch.float32)
                v_valid = False
            item["vjepa_emb"] = v
            item["vjepa_valid"] = v_valid

        if self.use_cotracker:
            c_idx = int(self._cotracker_idx[u_idx])
            if c_idx >= 0:
                _, feats = self._cotracker_data[sess]
                c = torch.from_numpy(feats[c_idx]).float()
                c_valid = True
            else:
                c = torch.zeros(COTRACKER_DIM, dtype=torch.float32)
                c_valid = False
            item["cotracker_emb"] = c
            item["cotracker_valid"] = c_valid

        return item


def assemble_conditioning(
    batch: dict,
    use_vjepa: bool,
    use_cotracker: bool,
) -> torch.Tensor:
    """Concatenate the requested modalities into a single conditioning tensor.

    Always includes Chronos-2 (`sensor_emb`). Order is fixed:
      [sensor_emb, vjepa_emb, cotracker_emb]   — drop modalities not selected.
    """
    parts = [batch["sensor_emb"]]
    if use_vjepa:
        parts.append(batch["vjepa_emb"])
    if use_cotracker:
        parts.append(batch["cotracker_emb"])
    return torch.cat(parts, dim=-1)


def conditioning_dim(d_sensor: int, use_vjepa: bool, use_cotracker: bool) -> int:
    return d_sensor + (VJEPA_DIM if use_vjepa else 0) + (COTRACKER_DIM if use_cotracker else 0)
