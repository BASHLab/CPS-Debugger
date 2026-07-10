"""HuBERT-style discrete codebook for the per-patch non-moving anchor.

Fits an offline k-means codebook over hand-crafted per-patch features of the
raw 7 sensor channels. The codebook is COMPLETELY independent of any encoder,
so it is a provably non-moving target (the point: anchor the per-position JEPA
prediction against the data2vec moving-target collapse).

Design choice — labels are assigned ON THE FLY in the dataset, not precomputed
per session. Each training window is tiled into exactly `n_time_patches` (98 at
4000-tick) contiguous blocks via the SAME relative grid the conv frontend uses,
so there is no sliding-grid alignment problem. This script therefore only
fits + saves the codebook; the dataset imports `compute_patch_features` /
`assign_clusters` / `load_codebook` from here.

Per (time-patch x channel) token feature (5-dim): [mean, std, first, last,
slope] over that patch's tick block. Features are z-scored per (channel,feature)
on TRAIN statistics, then a single SHARED k-means (k=256) is fit over all
(patch x channel) vectors pooled together. Each token -> one cluster id; the
686 = 7*98 labels are laid out channel-major (flat = c*n_patches + tp) to match
the teacher_targets reshape in bl7_pretrain.py.

Usage:
    python3 precompute_hubert_labels.py --fit --window-size 4000 \
        --conv-strides 5,4,2,1 --out models/bl7_w4000_hubert/hubert_codebook.npz
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BL7EncoderConfig, DATA_DIR, BASE_DIR, TRAIN_SESSIONS, SENSOR_COLS,
)

N_FEAT = 5          # mean, std, first, last, slope
CLIP_SIGMA = 5.0


def _patch_edges(window_size: int, n_patches: int) -> np.ndarray:
    """Contiguous block edges tiling [0, window_size) into n_patches blocks.
    Matches the conv frontend's relative patch grid (used identically for every
    window, so window-position alignment is automatic)."""
    return np.linspace(0, window_size, n_patches + 1).astype(np.int64)


def compute_patch_features(window: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """window: (C, W) float. Returns (n_patches, C, 5) raw per-patch features."""
    C, W = window.shape
    n_patches = len(edges) - 1
    out = np.zeros((n_patches, C, N_FEAT), dtype=np.float64)
    for tp in range(n_patches):
        lo, hi = int(edges[tp]), int(edges[tp + 1])
        seg = window[:, lo:hi]                       # (C, blk)
        if seg.shape[1] == 0:
            continue
        out[tp, :, 0] = seg.mean(axis=1)
        out[tp, :, 1] = seg.std(axis=1)
        out[tp, :, 2] = seg[:, 0]
        out[tp, :, 3] = seg[:, -1]
        out[tp, :, 4] = seg[:, -1] - seg[:, 0]       # slope
    return out


def _zscore(feats: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """feats (..., C, 5); mean/std (C, 5)."""
    z = (feats - mean) / std
    return np.clip(z, -CLIP_SIGMA, CLIP_SIGMA)


def assign_clusters(window: np.ndarray, cb: dict) -> np.ndarray:
    """window (C, W) -> (C * n_patches,) int64 cluster ids, channel-major
    (flat = c*n_patches + tp). `cb` is a loaded codebook dict."""
    edges = cb["edges"]
    feats = compute_patch_features(window, edges)            # (Tp, C, 5)
    z = _zscore(feats, cb["feat_mean"], cb["feat_std"])      # (Tp, C, 5)
    Tp, C, _ = z.shape
    flat = z.transpose(1, 0, 2).reshape(C * Tp, N_FEAT)      # channel-major (c*Tp+tp)
    cents = cb["centroids"]                                  # (k, 5)
    # nearest centroid by squared euclidean
    d = ((flat[:, None, :] - cents[None, :, :]) ** 2).sum(-1)  # (C*Tp, k)
    return d.argmin(axis=1).astype(np.int64)


def load_codebook(path) -> dict:
    z = np.load(path, allow_pickle=False)
    cb = {k: z[k] for k in z.files}
    cb["edges"] = cb["edges"].astype(np.int64)
    return cb


def _sample_window_feats(sessions, window_size, edges, n_windows, seed):
    rng = np.random.default_rng(seed)
    arrays = []
    for s in sessions:
        df = pd.read_parquet(DATA_DIR / f"{s}.parquet", columns=list(SENSOR_COLS))
        df = df.loc[:, ~df.columns.duplicated()]
        arrays.append(df[list(SENSOR_COLS)].values.astype(np.float64).T)  # (C, N)
    lens = np.array([a.shape[1] for a in arrays])
    feat_list = []
    per = max(1, n_windows // len(arrays))
    for a in arrays:
        N = a.shape[1]
        for _ in range(per):
            end = int(rng.integers(window_size - 1, N))
            w = a[:, end - window_size + 1: end + 1]          # (C, W)
            feat_list.append(compute_patch_features(w, edges))  # (Tp, C, 5)
    return np.stack(feat_list)                                 # (nw, Tp, C, 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--window-size", type=int, default=4000)
    ap.add_argument("--conv-strides", default="5,4,2,1")
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--n-windows", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    strides = tuple(int(x) for x in args.conv_strides.split(","))
    ecfg = BL7EncoderConfig(window_size=args.window_size, conv_strides=strides)
    n_patches = ecfg.n_time_patches
    edges = _patch_edges(args.window_size, n_patches)
    print(f"window={args.window_size} strides={strides} -> n_patches={n_patches}, "
          f"C={len(SENSOR_COLS)}, tokens={len(SENSOR_COLS)*n_patches}")

    if not args.fit:
        raise SystemExit("Only --fit is implemented (labels assigned on the fly).")

    from sklearn.cluster import MiniBatchKMeans
    feats = _sample_window_feats(TRAIN_SESSIONS, args.window_size, edges,
                                 args.n_windows, args.seed)        # (nw, Tp, C, 5)
    C = feats.shape[2]
    # per-(channel,feature) z-stats over the sampled blocks
    feat_mean = feats.mean(axis=(0, 1))                            # (C, 5)
    feat_std = feats.std(axis=(0, 1)).clip(min=1e-6)               # (C, 5)
    z = _zscore(feats, feat_mean, feat_std)
    pool = z.reshape(-1, N_FEAT)                                   # (nw*Tp*C, 5)
    print(f"fitting MiniBatchKMeans k={args.k} on {pool.shape[0]:,} vectors ...")
    km = MiniBatchKMeans(n_clusters=args.k, random_state=args.seed,
                         n_init="auto", batch_size=4096, max_iter=200)
    km.fit(pool)
    counts = np.bincount(km.labels_, minlength=args.k)
    print(f"cluster usage: min={counts.min()} max={counts.max()} "
          f"empty={(counts == 0).sum()}/{args.k}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out,
             centroids=km.cluster_centers_.astype(np.float32),
             feat_mean=feat_mean.astype(np.float32),
             feat_std=feat_std.astype(np.float32),
             edges=edges.astype(np.int64),
             window_size=np.int64(args.window_size),
             n_patches=np.int64(n_patches),
             conv_strides=np.array(strides, dtype=np.int64),
             k=np.int64(args.k),
             seed=np.int64(args.seed))
    print(f"saved codebook -> {out}")


if __name__ == "__main__":
    main()
