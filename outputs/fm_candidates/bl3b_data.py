"""Dataset for BL-3b (LoRA-adapted V-JEPA).

Each tick now returns a raw 64-frame video clip (uint8, [64, 3, 384, 384])
ending at the same end-frame as the cached V-JEPA embedding from BL-3 — so
when LoRA passes are turned off this is bit-identical-conditioning to BL-3.

Frames are mmapped from `video/frames_30fps/{session}.npy` produced by
`extract_frames_30fps.py`. V-JEPA timestamps come from the same V-JEPA
parquet that BL-3 uses (we only need its `end_frame_idx` for alignment).
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
FRAMES_DIR    = VIDEO_DIR / "frames_30fps"
CLIP_FRAMES   = 64
NATIVE_TO_30  = 4         # 120fps -> 30fps stride


class BL3bDataset(Dataset):
    """Wraps a TraceDataset (or a torch Subset of one). Adds a
    `clip` field (uint8 tensor, shape (64, 3, 384, 384)) per item, taken
    causally from the 30fps frame cache so it ends at the same frame the
    V-JEPA embedding was computed at. Fully zero-padded at the head if a
    tick is too early in the session for a complete 64-frame window."""

    def __init__(self, base: Dataset):
        self.base = base
        td = base
        index_map: Optional[np.ndarray] = None
        if hasattr(base, "indices") and hasattr(base, "dataset"):
            td = base.dataset
            index_map = np.asarray(base.indices, dtype=np.int64)
        if not isinstance(td, TraceDataset):
            raise TypeError("BL3bDataset expects TraceDataset (or Subset of one)")
        self._underlying = td
        self._index_map = index_map

        self._frames: dict = {}     # session -> mmap (T_30, 3, 384, 384) uint8
        self._end_native_by_vjepa_idx: dict = {}    # session -> int64[N_v]
        self._vjepa_end_ts: dict = {}              # session -> int64[N_v]

        sessions = sorted(set(td.session_ids))
        for sess in sessions:
            v_df = pd.read_parquet(
                VJEPA_DIR / f"{sess}.parquet",
                columns=["end_frame_idx", "end_timestamp_us"],
            )
            ts = v_df["end_timestamp_us"].to_numpy(dtype=np.int64)
            ef = v_df["end_frame_idx"].to_numpy(dtype=np.int64)
            keep = ts > 0
            ts, ef = ts[keep], ef[keep]
            order = np.argsort(ts, kind="stable")
            self._vjepa_end_ts[sess] = ts[order]
            self._end_native_by_vjepa_idx[sess] = ef[order]
            self._frames[sess] = np.load(FRAMES_DIR / f"{sess}.npy", mmap_mode="r")

        n_base = len(td)
        ts_us  = np.asarray(td.timestamps_us, dtype=np.int64)
        sess_arr = np.asarray(td.session_ids)
        self._vjepa_idx = np.full(n_base, -1, dtype=np.int64)
        for sess in sessions:
            mask = sess_arr == sess
            if not mask.any():
                continue
            v_ts = self._vjepa_end_ts[sess]
            sub_ts = ts_us[mask]
            self._vjepa_idx[mask] = np.searchsorted(v_ts, sub_ts, side="right") - 1
        n_invalid = int((self._vjepa_idx < 0).sum())
        print(f"  BL3bDataset: {n_base} samples, V-JEPA-invalid={n_invalid}")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        u_idx = int(self._index_map[i]) if self._index_map is not None else int(i)
        sess  = self._underlying.session_ids[u_idx]
        v_idx = int(self._vjepa_idx[u_idx])

        if v_idx < 0:
            clip = np.zeros((CLIP_FRAMES, 3, 384, 384), dtype=np.uint8)
        else:
            end_native = int(self._end_native_by_vjepa_idx[sess][v_idx])
            end_30 = end_native // NATIVE_TO_30
            start_30 = max(0, end_30 - CLIP_FRAMES + 1)
            chunk = np.asarray(self._frames[sess][start_30:end_30 + 1])  # (<=64, 3, 384, 384)
            if chunk.shape[0] < CLIP_FRAMES:
                pad = np.zeros((CLIP_FRAMES - chunk.shape[0], 3, 384, 384), dtype=np.uint8)
                clip = np.concatenate([pad, chunk], axis=0)
            else:
                clip = chunk
        item["clip"] = torch.from_numpy(clip)         # uint8 (64, 3, 384, 384)
        return item
