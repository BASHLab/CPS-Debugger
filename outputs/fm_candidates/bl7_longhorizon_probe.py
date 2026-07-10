"""Long-horizon decodability probe for the BL-7 encoder.

Tests whether the encoder represents long-horizon accumulator state — a proxy
for the controller counters (counter_xref / balanced_counter / setpoint_counter)
that drive the branch targets the decoder gets wrong — that a 500-tick window
structurally cannot see.

Target: ticks-since-last-FSM-transition (capped). Banded by horizon so we see
WHERE decodability breaks:
  - [0,500):    both 500-tick and 4000-tick windows contain the transition.
  - [500,4000): only a 4000-tick window contains it.

Features compared via closed-form ridge regression:
  - axial tokens mean-pooled (the full pre-pool representation, d_model)
  - the K=4 Perceiver pool flattened (the decoder's actual prefix input)
Reports R2 and MAE (ticks) overall and per band.

Reading:
  * 4000-tick encoder recovers [500,4000) well  -> long-horizon info IS in the
    representation; the bottleneck is decoder coupling, not the encoder.
  * 4000-tick encoder fails [500,4000) (≈ the 500-tick encoder) -> the encoder /
    pretraining is not capturing the long-horizon state.

Usage:
  python3 bl7_longhorizon_probe.py --bl7-ckpt models/bl7_w4000/bl7_best.pt \
      --out results/bl7_w4000_probes/longhorizon_step1000.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import BL7EncoderConfig, DATA_DIR, SENSOR_COLS
from bl7_model import BL7Encoder, EMATeacher


def load_enc(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ecfg = BL7EncoderConfig(**ckpt["encoder_config"])
    enc = BL7Encoder(ecfg).to(device)
    if "ema" in ckpt:
        ema = EMATeacher(enc).to(device)
        ema.load_state_dict(ckpt["ema"])
        ema.copy_to(enc)
        print(f"Loaded EMA-teacher from {ckpt_path}")
    else:
        enc.load_state_dict(ckpt["student"])
        print(f"Loaded student from {ckpt_path}")
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, ecfg


def ticks_since_transition(states: np.ndarray) -> np.ndarray:
    """For each tick t, ticks since the most recent state change (0 at change)."""
    n = len(states)
    out = np.zeros(n, dtype=np.int64)
    changed = np.nonzero(states[1:] != states[:-1])[0] + 1  # indices where a change lands
    last = 0
    ci = 0
    for t in range(1, n):
        if ci < len(changed) and changed[ci] == t:
            last = t
            ci += 1
        out[t] = t - last
    return out


def ridge_eval(X: np.ndarray, y: np.ndarray, bands, reg: float = 1e-2):
    Xc = X - X.mean(axis=0, keepdims=True)
    ybar = y.mean()
    yc = y - ybar
    A = Xc.T @ Xc + reg * np.eye(Xc.shape[1], dtype=np.float64)
    W = np.linalg.solve(A, Xc.T @ yc)
    yhat = Xc @ W + ybar

    def metrics(mask):
        if int(mask.sum()) < 5:
            return None
        yt, yp = y[mask], yhat[mask]
        mse = float(((yp - yt) ** 2).mean())
        var = float(((yt - yt.mean()) ** 2).mean())
        return {"n": int(mask.sum()), "mae": float(np.abs(yp - yt).mean()),
                "rmse": float(mse ** 0.5), "r2": float(1.0 - mse / max(var, 1e-9))}

    res = {"overall": metrics(np.ones(len(y), dtype=bool))}
    for lo, hi in bands:
        res[f"[{lo},{hi})"] = metrics((y >= lo) & (y < hi))
    return res


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bl7-ckpt", required=True)
    ap.add_argument("--val-sessions", nargs="+",
                    default=["2025-03-21_11-21-42", "2025-03-25_13-23-42"])
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-samples", type=int, default=8000)
    ap.add_argument("--cap", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    enc, ecfg = load_enc(args.bl7_ckpt, device)
    w = ecfg.window_size

    sensor_arrays, tst_arrays = [], []
    for s in args.val_sessions:
        df = pd.read_parquet(Path(args.data_dir) / f"{s}.parquet",
                             columns=list(SENSOR_COLS))
        sensor_arrays.append(df[list(SENSOR_COLS)].values.astype(np.float32))
        tst_arrays.append(ticks_since_transition(df["pendulum_state"].values.astype(np.int64)))

    lens = np.array([len(a) for a in sensor_arrays])
    cum = lens.cumsum()
    total = int(lens.sum())
    rng = np.random.default_rng(args.seed)

    n = args.n_samples
    ax = np.zeros((n, ecfg.d_model), dtype=np.float32)
    de = np.zeros((n, ecfg.K * ecfg.d_emb), dtype=np.float32)
    y = np.zeros(n, dtype=np.float64)

    cur = 0
    while cur < n:
        bs = min(args.batch_size, n - cur)
        windows, ys = [], []
        for _ in range(bs):
            r = rng.integers(0, total)
            si = int(np.searchsorted(cum, r, side="right"))
            L = lens[si]
            end = int(rng.integers(w - 1, L))
            windows.append(sensor_arrays[si][end - w + 1: end + 1].T)
            ys.append(min(int(tst_arrays[si][end]), args.cap))
        b = torch.from_numpy(np.stack(windows)).float().to(device)
        with torch.amp.autocast(device.type, dtype=torch.bfloat16):
            a = enc.encode_axial(b).float().mean(dim=1)        # (bs, d_model)
            d = enc.encode_dense(b).float().reshape(bs, -1)    # (bs, K*d_emb)
        ax[cur:cur + bs] = a.cpu().numpy()
        de[cur:cur + bs] = d.cpu().numpy()
        y[cur:cur + bs] = ys
        cur += bs

    bands = [(0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000)]
    out = {
        "bl7_ckpt": str(args.bl7_ckpt),
        "window_size": int(w),
        "n_samples": int(n),
        "cap": int(args.cap),
        "target": "ticks_since_last_fsm_transition",
        "target_mean": float(y.mean()),
        "target_frac_ge_500": float((y >= 500).mean()),
        "axial_meanpool": ridge_eval(ax.astype(np.float64), y, bands),
        "dense_K4_pool": ridge_eval(de.astype(np.float64), y, bands),
    }
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
