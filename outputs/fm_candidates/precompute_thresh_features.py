"""E1-revised: precompute threshold + libm derived features per tick.

Produces `sensor_embeddings_bl7_thresh/<session>_emb.npy` of shape
(N_ticks, 4, 272) by concatenating 16 derived features onto each of the
K=4 BL-7 tokens. Decoder picks them up via the existing emb pipeline;
no model code changes.

Feature set (per tick, computed from raw 7-channel sensors):

  PLC thresholds (5):
    |theta - pi|, |theta|, |theta_d|, |v|, |target_x - current_x|

  RESET features (2):
    |x|/MAX_X_SAFE, sign(x)

  libm-input features (4):
    cos(theta), sin(theta),
    log(1 - 0.9999 * |x|/MAX_X_DISPL),
    log(1 - 0.9999 * |v|/V_MAX)

  Lagged PLC (5, at t-5 with zero pad for first 5 ticks)

PLC and libm-log channels are z-scored against per-channel statistics
computed over the TRAIN sessions, clipped to +/- 5 sigma. cos/sin,
|x|/MAX_X_SAFE, sign(x), libm-arg-ratio terms are bounded by
construction and used as-is.

Usage:
    python3 precompute_thresh_features.py
    python3 precompute_thresh_features.py --sessions 2025-03-17_10-36-44
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DATA_DIR, BASE_DIR, ALL_SESSIONS, TRAIN_SESSIONS, SENSOR_COLS,
)

SRC_EMB_DIR = BASE_DIR / "sensor_embeddings_bl7"
DST_EMB_DIR = BASE_DIR / "sensor_embeddings_bl7_thresh"
NORM_FILE = BASE_DIR / "thresh_features_norm.json"

# Controller-source constants (sys_params.h)
MAX_X_DISPL = 0.18
X_SAFETY_BUFFER = 0.03
MAX_X_SAFE = MAX_X_DISPL - X_SAFETY_BUFFER
V_MAX = 1.0
LAG = 5
CLIP_SIGMA = 5.0
N_FEATURES = 16  # 5 PLC + 2 RESET + 4 libm + 5 lagged PLC


def load_raw(sess: str) -> dict:
    """Return dict of channel name -> np.ndarray for one session."""
    df = pd.read_parquet(DATA_DIR / f"{sess}.parquet", columns=list(SENSOR_COLS))
    df = df.loc[:, ~df.columns.duplicated()]
    return {c: df[c].values.astype(np.float64) for c in SENSOR_COLS}


def compute_plc(raw: dict) -> np.ndarray:
    """5 PLC threshold-style features per tick (N, 5). Raw scale."""
    theta = raw["current_angle"]
    theta_d = raw["angular_velocity"]
    v = raw["velocity"]
    x = raw["current_x"]
    tx = raw["target_x"]
    return np.stack([
        np.abs(theta - np.pi),
        np.abs(theta),
        np.abs(theta_d),
        np.abs(v),
        np.abs(tx - x),
    ], axis=1)  # (N, 5)


def compute_reset(raw: dict) -> np.ndarray:
    """2 RESET features per tick (N, 2). Bounded, no normalization needed."""
    x = raw["current_x"]
    return np.stack([
        np.abs(x) / MAX_X_SAFE,
        np.sign(x),
    ], axis=1)  # (N, 2)


def compute_libm(raw: dict) -> np.ndarray:
    """4 libm-input features per tick (N, 4). cos/sin bounded; log args
    clamped to (-inf, 0]."""
    theta = raw["current_angle"]
    x = raw["current_x"]
    v = raw["velocity"]
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    # log(1 - 0.9999 * |x|/MAX_X_DISPL): log of distance-to-saturation
    log_relpos = np.log(np.clip(1.0 - 0.9999 * np.abs(x) / MAX_X_DISPL, 1e-6, None))
    log_relvel = np.log(np.clip(1.0 - 0.9999 * np.abs(v) / V_MAX, 1e-6, None))
    return np.stack([cos_t, sin_t, log_relpos, log_relvel], axis=1)  # (N, 4)


def lag_features(plc: np.ndarray, lag: int = LAG) -> np.ndarray:
    """Lagged PLC features (N, 5). Pad first `lag` rows with zeros."""
    n = plc.shape[0]
    out = np.zeros_like(plc)
    if n > lag:
        out[lag:] = plc[:-lag]
    return out


def gather_train_stats(sessions=TRAIN_SESSIONS) -> dict:
    """Per-channel mean/std for the PLC + log channels (the ones that
    need z-scoring). Computed across the TRAIN sessions."""
    plc_accum, log_accum = [], []
    for sess in sessions:
        raw = load_raw(sess)
        plc_accum.append(compute_plc(raw))               # (N, 5)
        log_accum.append(compute_libm(raw)[:, 2:])       # (N, 2)
    plc_all = np.concatenate(plc_accum, axis=0)
    log_all = np.concatenate(log_accum, axis=0)
    stats = {
        "plc_mean": plc_all.mean(axis=0).tolist(),
        "plc_std": plc_all.std(axis=0).clip(min=1e-6).tolist(),
        "log_mean": log_all.mean(axis=0).tolist(),
        "log_std": log_all.std(axis=0).clip(min=1e-6).tolist(),
    }
    print(f"train stats over {len(sessions)} sessions:")
    print(f"  plc mean: {[f'{m:.3f}' for m in stats['plc_mean']]}")
    print(f"  plc std:  {[f'{s:.3f}' for s in stats['plc_std']]}")
    print(f"  log mean: {[f'{m:.3f}' for m in stats['log_mean']]}")
    print(f"  log std:  {[f'{s:.3f}' for s in stats['log_std']]}")
    return stats


def normalize_plc(plc: np.ndarray, stats: dict) -> np.ndarray:
    mean = np.asarray(stats["plc_mean"], dtype=np.float64)
    std = np.asarray(stats["plc_std"], dtype=np.float64)
    z = (plc - mean) / std
    return np.clip(z, -CLIP_SIGMA, CLIP_SIGMA)


def normalize_log(log_chs: np.ndarray, stats: dict) -> np.ndarray:
    mean = np.asarray(stats["log_mean"], dtype=np.float64)
    std = np.asarray(stats["log_std"], dtype=np.float64)
    z = (log_chs - mean) / std
    return np.clip(z, -CLIP_SIGMA, CLIP_SIGMA)


def features_for_session(sess: str, stats: dict) -> np.ndarray:
    """Returns (N, 16) of float32 derived features."""
    raw = load_raw(sess)
    plc = compute_plc(raw)               # (N, 5) raw
    libm = compute_libm(raw)             # (N, 4) cos, sin, log_rp, log_rv
    reset = compute_reset(raw)           # (N, 2) ratio, sign
    plc_lag = lag_features(plc)          # (N, 5) raw lagged

    plc_z = normalize_plc(plc, stats)
    plc_lag_z = normalize_plc(plc_lag, stats)
    # libm: cos/sin as-is, log channels z-scored
    libm_norm = np.concatenate([
        libm[:, :2],                      # cos, sin (bounded)
        normalize_log(libm[:, 2:], stats),  # log channels
    ], axis=1)                            # (N, 4)
    out = np.concatenate([plc_z, reset, libm_norm, plc_lag_z], axis=1)
    assert out.shape[1] == N_FEATURES, f"expected {N_FEATURES} features, got {out.shape[1]}"
    return out.astype(np.float32)


def process_session(sess: str, stats: dict,
                    src_dir: Path, dst_dir: Path) -> None:
    src = src_dir / f"{sess}_emb.npy"
    dst = dst_dir / f"{sess}_emb.npy"
    if not src.is_file():
        print(f"  [{sess}] SKIP: missing {src}")
        return
    emb = np.load(src)                    # (N, K, d) or (N, d)
    if emb.ndim == 2:
        emb = emb[:, None, :]             # (N, 1, d)
    n, k, d = emb.shape
    feats = features_for_session(sess, stats)
    if feats.shape[0] != n:
        raise RuntimeError(f"{sess}: emb has {n} rows, features have {feats.shape[0]}")
    # Broadcast features across the K dim
    feats_bk = np.broadcast_to(feats[:, None, :], (n, k, N_FEATURES)).copy()
    new = np.concatenate([emb, feats_bk], axis=-1).astype(np.float32)
    assert new.shape == (n, k, d + N_FEATURES), \
        f"unexpected output shape {new.shape}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    np.save(dst, new)
    print(f"  [{sess}] wrote {new.shape} -> {dst.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", default=None,
                    help="Optional session subset; default is all 15 sessions.")
    ap.add_argument("--stats-only", action="store_true",
                    help="Only compute and save train normalization stats.")
    ap.add_argument("--src-emb-dir", default=None,
                    help=f"Source embedding dir (default: {SRC_EMB_DIR}).")
    ap.add_argument("--dst-emb-dir", default=None,
                    help=f"Destination dir for thresh-augmented emb "
                         f"(default: {DST_EMB_DIR}).")
    args = ap.parse_args()
    sessions = args.sessions or ALL_SESSIONS
    src_dir = Path(args.src_emb_dir) if args.src_emb_dir else SRC_EMB_DIR
    dst_dir = Path(args.dst_emb_dir) if args.dst_emb_dir else DST_EMB_DIR

    if NORM_FILE.is_file() and not args.stats_only:
        stats = json.loads(NORM_FILE.read_text())
        print(f"loaded stats from {NORM_FILE}")
    else:
        stats = gather_train_stats(TRAIN_SESSIONS)
        NORM_FILE.write_text(json.dumps(stats, indent=2))
        print(f"saved stats to {NORM_FILE}")
        if args.stats_only:
            return

    print(f"\nprocessing {len(sessions)} sessions; "
          f"src={src_dir} -> dst={dst_dir}")
    for sess in sessions:
        process_session(sess, stats, src_dir, dst_dir)
    print("\ndone.")


if __name__ == "__main__":
    main()
