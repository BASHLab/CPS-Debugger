"""BL-7 sensor-only dataset for JEPA pretraining.

Loads the 7 sensor channels from session parquets (no traces, no embeddings).
Each __getitem__ call returns a random 500-tick window from a random session
weighted by session length. Window-end positions are sampled fresh per call
(ignoring the dataset index) — set worker RNGs from the global seed so the
sequence of samples is deterministic given the seed.

Per BL-7 plan §4.3 / §4.4: no input augmentations during JEPA pretraining
(data2vec 2.0 recipe). Augmentations are reserved for Tier 2 contrastive aux.

For Tier 2 (class-balanced contrastive), this dataset also exposes the
per-tick FSM state via `pendulum_state`. Tier 1B does not use it.
"""
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import DATA_DIR, SENSOR_COLS, BL7EncoderConfig


class SensorWindowDataset(Dataset):
    """Random-window sensor dataset for BL-7 pretraining.

    Sessions are loaded once into RAM as float32 (N_ticks, 7) arrays. Each
    call to __getitem__ draws a random session weighted by length, then a
    random window-end inside that session, and returns the 500-tick slice
    ending at that position. If the slice would dip before t=0 the start is
    zero-padded.

    Class-balanced mode (Tier 2): when class_balanced=True, the sampler
    picks a target state uniformly from {0, 1, 2}, then picks a session
    weighted by that state's count in that session, then picks a tick whose
    state matches. Builds per-state-per-session tick-index lists at
    construction time (~80 MB for the full 22 M-tick dataset).
    """

    def __init__(
        self,
        sessions: List[str],
        cfg: BL7EncoderConfig,
        samples_per_epoch: int,
        data_dir: Path = DATA_DIR,
        return_state: bool = False,
        class_balanced: bool = False,
        return_hubert: bool = False,
        hubert_codebook_path: str = None,
    ):
        self.cfg = cfg
        self.samples_per_epoch = samples_per_epoch
        self.return_state = return_state or class_balanced
        self.class_balanced = class_balanced
        # HuBERT per-patch discrete labels (computed on the fly from each
        # window's own 98-patch grid — same relative tiling the conv uses, so
        # no window-position alignment problem).
        self.return_hubert = return_hubert
        self._hb_cb = None
        if return_hubert:
            from precompute_hubert_labels import load_codebook
            self._hb_cb = load_codebook(hubert_codebook_path)
            assert int(self._hb_cb["window_size"]) == cfg.window_size, (
                f"codebook window {int(self._hb_cb['window_size'])} != "
                f"cfg.window_size {cfg.window_size}")
            assert int(self._hb_cb["n_patches"]) == cfg.n_time_patches, (
                f"codebook n_patches {int(self._hb_cb['n_patches'])} != "
                f"cfg.n_time_patches {cfg.n_time_patches}")

        self._sensors = []   # list of (n_ticks, 7) float32 arrays
        self._states = []    # list of (n_ticks,) int8 arrays (always loaded
                             # if class_balanced; otherwise only if return_state)
        self._lengths = []
        self._session_names = []

        cols = list(SENSOR_COLS)
        for sess in sessions:
            df = pd.read_parquet(data_dir / f"{sess}.parquet", columns=cols)
            arr = df[cols].values.astype(np.float32)
            self._sensors.append(arr)
            self._lengths.append(len(arr))
            self._session_names.append(sess)
            if self.return_state:
                # pendulum_state is column 0 of SENSOR_COLS
                self._states.append(df["pendulum_state"].values.astype(np.int8))
            del df

        self._lengths = np.array(self._lengths, dtype=np.int64)
        self._cumlen = self._lengths.cumsum()
        self._total = int(self._lengths.sum())
        # Worker RNG, seeded in worker_init_fn
        self._rng = np.random.default_rng(0)

        # Class-balanced index: per state, per session, list of valid tick-end indices
        # (i.e. ticks where state == target_state AND tick >= window_size-1).
        self._per_state_session_ticks = None
        self._per_state_session_lens = None
        if self.class_balanced:
            w = self.cfg.window_size
            self._per_state_session_ticks = {0: [], 1: [], 2: []}
            self._per_state_session_lens = {0: [], 1: [], 2: []}
            for s_idx, state_arr in enumerate(self._states):
                valid_range = state_arr[w - 1:]  # only ticks where full window fits
                base = w - 1
                for st in (0, 1, 2):
                    ticks = np.where(valid_range == st)[0].astype(np.int64) + base
                    self._per_state_session_ticks[st].append(ticks)
                    self._per_state_session_lens[st].append(len(ticks))
            for st in (0, 1, 2):
                self._per_state_session_lens[st] = np.array(
                    self._per_state_session_lens[st], dtype=np.int64)
            counts = {st: int(self._per_state_session_lens[st].sum()) for st in (0,1,2)}
            total = sum(counts.values())
            pct = {st: 100*counts[st]/total for st in counts}
            print(f"  Per-state counts: state-0={counts[0]:,} ({pct[0]:.1f}%), "
                  f"state-1={counts[1]:,} ({pct[1]:.1f}%), "
                  f"state-2={counts[2]:,} ({pct[2]:.1f}%)")

        print(f"  SensorWindowDataset: {len(sessions)} sessions, "
              f"{self._total:,} ticks, {samples_per_epoch:,} samples/epoch"
              + (f", class-balanced 33/33/33" if self.class_balanced else ""))

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_seed(self, seed: int) -> None:
        """Called from a worker_init_fn so each worker sees a different stream."""
        self._rng = np.random.default_rng(seed)

    def _sample_class_balanced(self):
        """Pick state uniformly from {0,1,2}, then session weighted, then tick uniformly."""
        st = int(self._rng.integers(0, 3))
        sess_lens = self._per_state_session_lens[st]
        # Session weighted by count of this state in that session
        cum = sess_lens.cumsum()
        total_for_state = int(cum[-1])
        r = self._rng.integers(0, total_for_state)
        sess_idx = int(np.searchsorted(cum, r, side="right"))
        ticks = self._per_state_session_ticks[st][sess_idx]
        end = int(ticks[self._rng.integers(0, len(ticks))])
        return st, sess_idx, end

    def _sample_uniform(self):
        """Pick (session, tick) uniformly weighted by session length."""
        r = self._rng.integers(0, self._total)
        sess_idx = int(np.searchsorted(self._cumlen, r, side="right"))
        n = self._lengths[sess_idx]
        w = self.cfg.window_size
        end = int(self._rng.integers(w - 1, n))
        return sess_idx, end

    def __getitem__(self, idx: int):
        # idx is ignored — windows are drawn fresh from the worker RNG.
        if self.class_balanced:
            st_label, sess_idx, end = self._sample_class_balanced()
        else:
            sess_idx, end = self._sample_uniform()
            st_label = None

        w = self.cfg.window_size
        start = end - w + 1
        window = self._sensors[sess_idx][start:end + 1]   # (w, 7)
        window = window.T.copy()                          # (7, w)

        out = {"sensors": torch.from_numpy(window).float()}
        if self.return_state:
            if st_label is None:
                st_label = int(self._states[sess_idx][end])
            out["state"] = st_label
        if self.return_hubert:
            from precompute_hubert_labels import assign_clusters
            labels = assign_clusters(window.astype(np.float64), self._hb_cb)  # (C*Tp,)
            out["hubert_labels"] = torch.from_numpy(labels).long()
        return out


def make_worker_init_fn(global_seed: int):
    """Seed each worker with a unique stream so they don't draw identical windows.

    Each worker_init_fn call sets the worker's RNG to seed (global_seed + worker_id).
    Combined with multi-rank DDP — pass rank into global_seed at the call site —
    this gives every (rank, worker) pair a distinct stream.
    """
    def init(worker_id: int):
        info = torch.utils.data.get_worker_info()
        # Same dataset object exists per-worker; mutate its RNG.
        info.dataset.set_seed(global_seed + worker_id)
    return init
