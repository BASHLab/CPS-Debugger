"""BL-2 data loading — trace + precomputed sensor embeddings."""
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    DATA_DIR, EMB_DIR, TOKEN_MAPPING, SENSOR_COLS, ModelConfig,
)


def load_token_mapping(path: Path = TOKEN_MAPPING):
    tm = json.loads(path.read_text())
    return tm["vocab_size"], tm["triples"], tm["triple_to_id"]


class TraceDataset(Dataset):
    """Dataset of (trace_tokens, sensor_embedding) pairs.

    Loads per-session parquets and precomputed sensor embeddings.
    Traces are padded/truncated to max_len.
    """

    def __init__(
        self,
        sessions: List[str],
        max_len: int,
        mask_id: int,
        emb_dir: Path = EMB_DIR,
        data_dir: Path = DATA_DIR,
        limit: Optional[int] = None,
    ):
        self.max_len = max_len
        self.mask_id = mask_id
        self.traces = []      # list of np.array[int32]
        self.embeddings = []   # list of np.array[float32] shape (proj_dim,)
        self.trace_lens = []   # original lengths
        self.fsm_states = []   # int FSM state (pendulum_state) per sample
        self.timestamps_us = []  # camera/sensor timestamp_us per sample
        self.parquet_rows  = []  # int row index within the source session parquet
        self.session_ids = []

        # Phase 1: collect indices and memory-map embeddings
        self._emb_maps = []    # list of mmap arrays per session
        self._emb_indices = [] # (emb_map_idx, row_idx) per sample
        self._state_pred_maps = []  # parallel list of state-pred arrays (E5; None if file absent)

        for sess in sessions:
            pq_path = data_dir / f"{sess}.parquet"
            emb_path = emb_dir / f"{sess}_emb.npy"

            df = pd.read_parquet(pq_path,
                                 columns=["trace", "trace_length", "pendulum_state",
                                          "timestamp_us"])
            emb = np.load(emb_path, mmap_mode='r')  # memory-mapped
            assert len(df) == len(emb), f"{sess}: {len(df)} ticks but {len(emb)} embeddings"

            emb_idx = len(self._emb_maps)
            self._emb_maps.append(emb)

            # E5: load per-tick state predictions if available
            sp_path = emb_dir / f"{sess}_state_pred.npy"
            if sp_path.exists():
                sp = np.load(sp_path, mmap_mode="r")
                assert len(sp) == len(emb), (
                    f"{sess}: {len(sp)} state preds vs {len(emb)} embeddings")
                self._state_pred_maps.append(sp)
            else:
                self._state_pred_maps.append(None)

            for i in range(len(df)):
                if limit is not None and len(self.traces) >= limit:
                    break
                row = df.iloc[i]
                tr = np.array(row["trace"], dtype=np.int32)
                self.traces.append(tr)
                self._emb_indices.append((emb_idx, i))
                self.trace_lens.append(len(tr))
                self.fsm_states.append(int(row["pendulum_state"]))
                self.timestamps_us.append(int(row["timestamp_us"]))
                self.parquet_rows.append(i)
                self.session_ids.append(sess)

            del df  # free parquet memory

            if limit is not None and len(self.traces) >= limit:
                break

        print(f"  Loaded {len(self.traces)} samples from {len(sessions)} sessions, "
              f"emb dim: {self._emb_maps[0].shape[1]}")

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        tr = self.traces[idx]
        tl = self.trace_lens[idx]
        emb_map_idx, emb_row = self._emb_indices[idx]
        emb = np.array(self._emb_maps[emb_map_idx][emb_row])  # copy from mmap

        # Pass through whatever shape was precomputed. Chronos-2 / MOMENT emit
        # (d_emb,) per tick; BL-7 emits (K, d_emb). Each decoder handles its
        # own dimensionality — ARModernModel mean-pools internally if 3D;
        # ARModernCrossModel consumes the K dense tokens via cross-attention.

        # Pad or truncate trace to max_len
        if len(tr) >= self.max_len:
            tr = tr[:self.max_len]
            tl = self.max_len
        else:
            pad = np.full(self.max_len - len(tr), self.mask_id, dtype=np.int32)
            tr = np.concatenate([tr, pad])

        out = {
            "trace": torch.from_numpy(tr).long(),           # (max_len,)
            "trace_len": tl,
            "sensor_emb": torch.from_numpy(emb).float(),    # (d_emb,) or (K, d_emb)
            "fsm_state": self.fsm_states[idx],
        }
        # E5: include per-tick probe-predicted state if a *_state_pred.npy was found
        sp_map = self._state_pred_maps[emb_map_idx]
        if sp_map is not None:
            out["state_pred"] = int(sp_map[emb_row])
        return out


class JointTraceWindowDataset(Dataset):
    """Dataset of (trace_tokens, raw_sensor_window) pairs for E2 joint training.

    Returns the raw 500-tick × 7-channel sensor window ending at each tick,
    instead of a precomputed encoder embedding. Used when the encoder is
    being jointly fine-tuned with the AR decoder (E2 K-sweep) — the model
    runs encoder.encode_dense(window) inside its forward pass.

    Memory: per-tick raw window is 500 × 7 × 4 = 14 KB vs 4 × 256 × 4 = 4 KB
    for a precomputed embedding. We keep one float32 sensor array per
    session in RAM (~50 MB/session × 13 sessions = ~650 MB total) and slice
    a fresh window per call.
    """

    def __init__(
        self,
        sessions: List[str],
        max_len: int,
        mask_id: int,
        window_size: int = 500,
        data_dir: Path = DATA_DIR,
        limit: Optional[int] = None,
    ):
        self.max_len = max_len
        self.mask_id = mask_id
        self.window_size = window_size
        self.traces = []
        self.trace_lens = []
        self.fsm_states = []
        self.session_ids = []

        self._sensors = []                # (n_ticks, 7) float32 per session
        self._tick_indices = []           # (session_idx, tick) per sample

        for sess in sessions:
            pq_path = data_dir / f"{sess}.parquet"
            # Dedupe columns: pendulum_state is in both the trace metadata
            # block and in SENSOR_COLS — listing it twice produces a duplicate
            # column DataFrame whose row-lookup returns a Series, not a scalar.
            extra_cols = ["trace", "trace_length", "pendulum_state",
                          "timestamp_us"]
            cols = list(dict.fromkeys(extra_cols + list(SENSOR_COLS)))
            df = pd.read_parquet(pq_path, columns=cols)
            sensors = df[list(SENSOR_COLS)].values.astype(np.float32)  # (n, 7)
            sess_idx = len(self._sensors)
            self._sensors.append(sensors)

            for i in range(len(df)):
                if limit is not None and len(self.traces) >= limit:
                    break
                row = df.iloc[i]
                tr = np.array(row["trace"], dtype=np.int32)
                self.traces.append(tr)
                self.trace_lens.append(len(tr))
                self.fsm_states.append(int(row["pendulum_state"]))
                self.session_ids.append(sess)
                self._tick_indices.append((sess_idx, i))

            del df
            if limit is not None and len(self.traces) >= limit:
                break

        print(f"  JointTraceWindowDataset: {len(self.traces)} samples from "
              f"{len(sessions)} sessions, window_size={window_size}")

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        tr = self.traces[idx]
        tl = self.trace_lens[idx]
        sess_idx, tick = self._tick_indices[idx]
        n = self._sensors[sess_idx].shape[0]
        w = self.window_size

        # Slice window of w ticks ending at `tick`, zero-pad if dipping below 0.
        ws = max(0, tick - w + 1)
        window = self._sensors[sess_idx][ws:tick + 1]    # (<= w, 7)
        if window.shape[0] < w:
            pad = np.zeros((w - window.shape[0], 7), dtype=np.float32)
            window = np.concatenate([pad, window], axis=0)
        window = window.T.copy()                          # (7, w)

        if len(tr) >= self.max_len:
            tr = tr[:self.max_len]
            tl = self.max_len
        else:
            pad = np.full(self.max_len - len(tr), self.mask_id, dtype=np.int32)
            tr = np.concatenate([tr, pad])

        return {
            "trace": torch.from_numpy(tr).long(),                  # (max_len,)
            "trace_len": tl,
            "sensor_window": torch.from_numpy(window).float(),     # (7, w)
            "fsm_state": self.fsm_states[idx],
        }


def make_dataloaders(
    train_sessions, val_sessions, max_len, mask_id,
    batch_size, num_workers=4, limit=None,
):
    train_ds = TraceDataset(train_sessions, max_len, mask_id, limit=limit)
    val_ds = TraceDataset(val_sessions, max_len, mask_id, limit=limit)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_dl, val_dl
