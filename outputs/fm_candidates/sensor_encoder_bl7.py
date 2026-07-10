"""BL-7 frozen-encoder embedding precompute.

Mirrors sensor_encoder.py:139-181 but:
  - Loads a BL-7 student/EMA checkpoint (use the EMA-teacher weights, not student;
    they are the published artifact in V-JEPA / data2vec).
  - Emits dense (N_ticks, K=4, d_emb=256) per session.
  - Output path: sensor_embeddings_bl7/<session>_emb.npy

The AR-modern decoder consumes either (N_ticks, d) [back-compat] or
(N_ticks, K, d) [BL-7]; data.py auto-detects via shape.

Usage:
    python3 sensor_encoder_bl7.py \\
        --ckpt models/bl7/bl7_best.pt \\
        --sessions 2025-03-25_13-39-06 2025-03-26_11-03-04 \\
        --output-dir sensor_embeddings_bl7/
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import BL7EncoderConfig, DATA_DIR, BASE_DIR, SENSOR_COLS
from bl7_model import BL7Encoder, EMATeacher


def load_bl7_encoder(ckpt_path: Path, device: torch.device,
                     use_ema: bool = True) -> BL7Encoder:
    """Load a BL-7 encoder from a pretrain checkpoint.

    By default uses the EMA-teacher weights (published artifact in
    V-JEPA / data2vec). Set --use-ema=false to load the raw student.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ecfg = BL7EncoderConfig(**ckpt["encoder_config"])
    enc = BL7Encoder(ecfg).to(device)

    if use_ema and "ema" in ckpt:
        # Apply EMA shadow weights
        ema = EMATeacher(enc).to(device)
        ema.load_state_dict(ckpt["ema"])
        ema.copy_to(enc)
        print(f"  Loaded EMA-teacher weights from {ckpt_path}")
    else:
        enc.load_state_dict(ckpt["student"])
        print(f"  Loaded raw student weights from {ckpt_path}")

    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


@torch.no_grad()
def precompute_session(
    session: str, encoder: BL7Encoder, output_dir: Path,
    data_dir: Path, batch_size: int, device: torch.device,
):
    cfg = encoder.cfg
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_dir / f"{session}.parquet", columns=list(SENSOR_COLS))
    sensors = df[list(SENSOR_COLS)].values.astype(np.float32)  # (n, 7)
    n = len(sensors)

    out = np.zeros((n, cfg.K, cfg.d_emb), dtype=np.float32)

    w = cfg.window_size
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        windows = []
        for t in range(start, end):
            ws = max(0, t - w + 1)
            window = sensors[ws:t + 1, :]                      # (<=w, 7)
            if window.shape[0] < w:
                pad = np.zeros((w - window.shape[0], cfg.n_channels), dtype=np.float32)
                window = np.concatenate([pad, window], axis=0)
            windows.append(window.T)                           # (7, w)
        batch = torch.from_numpy(np.stack(windows)).to(device)  # (B, 7, w)

        with torch.amp.autocast(device.type, dtype=torch.bfloat16):
            emb = encoder.encode_dense(batch)                  # (B, K, d_emb)
        out[start:end] = emb.float().cpu().numpy()

        if (start // batch_size) % 50 == 0:
            elapsed = time.time() - t0
            rate = end / elapsed if elapsed > 0 else 0
            eta = (n - end) / rate if rate > 0 else 0
            print(f"  [{session}] {end}/{n} ticks ({elapsed:.0f}s, {rate:.0f} ticks/s, ETA {eta:.0f}s)",
                  flush=True)

    out_path = output_dir / f"{session}_emb.npy"
    np.save(out_path, out)
    print(f"  [{session}] Saved {out_path} ({out.shape}, {time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to bl7_best.pt")
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--output-dir", default=str(BASE_DIR / "sensor_embeddings_bl7"))
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--use-ema", type=lambda s: s.lower() not in {"0", "false", "no"},
                    default=True, help="Use EMA-teacher weights (default true)")
    args = ap.parse_args()

    device = torch.device(args.device)
    encoder = load_bl7_encoder(Path(args.ckpt), device, use_ema=args.use_ema)
    n_total = sum(p.numel() for p in encoder.parameters())
    print(f"  Encoder params (total, frozen): {n_total:,}")

    for sess in args.sessions:
        precompute_session(
            sess, encoder, Path(args.output_dir), Path(args.data_dir),
            args.batch_size, device,
        )


if __name__ == "__main__":
    main()
