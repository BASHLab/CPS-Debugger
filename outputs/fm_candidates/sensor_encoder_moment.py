"""BL-6 Stage 2 — Precompute MOMENT-1-large per-tick sensor embeddings.

Apples-to-apples to `sensor_encoder.py` (Chronos-2 path):
  * Same 7 sensor channels (SENSOR_COLS).
  * Same per-tick sliding-window approach: at tick t we feed the last
    MOMENT_INPUT_LEN=512 timesteps (left-padded with zeros for early ticks).
  * Same fixed-random-init projection MLP (deterministic via seed) — Chronos-2
    used 5376→1024→512; MOMENT uses 7168→2048→1024 (same expansion ratio,
    output dim matches MOMENT's encoder dim instead of MDLM d_model).
  * Same per-channel z-norm before forward (MOMENT pretraining expects it,
    matched to what `finetune_moment.py` does).

Output: sensor_embeddings_moment/{session}_emb.npy of shape (n_ticks, 1024).

Usage:
    python3 sensor_encoder_moment.py \\
        --sessions 2025-03-17_10-36-44 ... \\
        --output-dir sensor_embeddings_moment/ \\
        --moment-ckpt models/moment_finetuned/moment_best.pt
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import (
    SensorConfig, DATA_DIR, MODEL_DIR, SENSOR_COLS,
)

MOMENT_MODEL_NAME = "AutonLab/MOMENT-1-large"
MOMENT_INPUT_LEN  = 512
MOMENT_DIM        = 1024     # encoder output dim per channel
PROJ_OUT_DIM      = 1024     # save dim per tick (≠ Chronos-2's 512; we keep MOMENT-native dim)
PROJ_MID_DIM      = 2048     # 2× output, matching Chronos-2's expansion ratio
PROJ_SEED         = 42       # fixed seed for random-init MLP — DO NOT CHANGE


def encode_batch(model, batch_chw: torch.Tensor, device) -> torch.Tensor:
    """batch_chw: (B, C=7, T=512) per-channel z-normalized.

    MOMENT processes channels independently. We reshape (B, C, T) -> (B*C, 1, T),
    forward through the embedding head, reshape back to (B, C, MOMENT_DIM).
    """
    B, C, T = batch_chw.shape
    flat = batch_chw.reshape(B * C, 1, T).to(device, dtype=torch.float32)
    with torch.no_grad():
        out = model(x_enc=flat)         # MOMENTOutputs
    emb = out.embeddings                # (B*C, MOMENT_DIM)
    return emb.reshape(B, C, -1)        # (B, C, MOMENT_DIM)


def precompute_session(
    session: str,
    model,
    proj: nn.Module,
    cfg: SensorConfig,
    data_dir: Path,
    output_dir: Path,
    batch_size: int,
    device,
):
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_dir / f"{session}.parquet", columns=SENSOR_COLS)
    n = len(df)
    sensors = df[SENSOR_COLS].to_numpy(dtype=np.float32)        # (n, 7)
    print(f"  [{session}] n={n}, channels={sensors.shape[1]}")

    out = np.zeros((n, PROJ_OUT_DIM), dtype=np.float32)
    proj = proj.eval()

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_windows = []
        for t in range(start, end):
            w_start = max(0, t - MOMENT_INPUT_LEN + 1)
            window = sensors[w_start:t + 1]                          # (<=512, 7)
            if window.shape[0] < MOMENT_INPUT_LEN:
                pad = np.zeros((MOMENT_INPUT_LEN - window.shape[0], 7), dtype=np.float32)
                window = np.concatenate([pad, window], axis=0)        # (512, 7)
            batch_windows.append(window.T)                            # (7, 512)
        batch = torch.from_numpy(np.stack(batch_windows))            # (B, 7, 512)

        # Per-channel z-norm to match MOMENT pretraining
        mu = batch.mean(dim=-1, keepdim=True)
        sd = batch.std(dim=-1, keepdim=True).clamp(min=1e-6)
        batch_norm = (batch - mu) / sd

        emb = encode_batch(model, batch_norm, device)                 # (B, 7, MOMENT_DIM)
        flat = emb.reshape(emb.shape[0], -1).float().cpu()            # (B, 7168)
        with torch.no_grad():
            projected = proj(flat).numpy()                            # (B, 1024)
        out[start:end] = projected

        if (start // batch_size) % 25 == 0:
            elapsed = time.time() - t0
            rate = end / max(elapsed, 1e-3)
            eta = (n - end) / max(rate, 1e-3)
            print(f"    {end}/{n} ({elapsed:.0f}s, {rate:.0f} ticks/s, ETA {eta:.0f}s)", flush=True)

    out_path = output_dir / f"{session}_emb.npy"
    np.save(out_path, out)
    print(f"  [{session}] Saved {out_path} ({out.shape}, {time.time()-t0:.0f}s)")


def load_finetuned_moment(ckpt_path: Path | None, device: str):
    from momentfm import MOMENTPipeline
    print(f"Loading {MOMENT_MODEL_NAME} (task=embedding)...")
    model = MOMENTPipeline.from_pretrained(
        MOMENT_MODEL_NAME,
        model_kwargs={"task_name": "embedding"},
    )
    model.init()
    if ckpt_path is not None and ckpt_path.exists():
        print(f"  loading fine-tuned weights: {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu")
        # Saved dict is the full state_dict including the forecasting head.
        # Strip head-only keys; the encoder/embedder/normalizer weights remain.
        full = sd["model_state"] if isinstance(sd, dict) and "model_state" in sd else sd
        # load_state_dict will silently skip mismatched keys with strict=False.
        missing, unexpected = model.load_state_dict(full, strict=False)
        print(f"  missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device).eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--output-dir", default=str(MODEL_DIR.parent / "sensor_embeddings_moment"))
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--moment-ckpt", default=None,
                    help="Path to fine-tuned MOMENT checkpoint .pt (default: models/moment_finetuned/moment_best.pt)")
    ap.add_argument("--seed", type=int, default=PROJ_SEED,
                    help="Random seed for MLP projection init (must match across sessions)")
    args = ap.parse_args()

    cfg = SensorConfig(batch_size=args.batch_size)

    ckpt = Path(args.moment_ckpt) if args.moment_ckpt else (MODEL_DIR / "moment_finetuned" / "moment_best.pt")
    model = load_finetuned_moment(ckpt, args.device)

    # Fixed-random-init projection MLP — deterministic via seed
    torch.manual_seed(args.seed)
    input_dim = 7 * MOMENT_DIM    # 7168
    proj = nn.Sequential(
        nn.Linear(input_dim, PROJ_MID_DIM),
        nn.GELU(),
        nn.Linear(PROJ_MID_DIM, PROJ_OUT_DIM),
    ).to("cpu")     # projection runs on CPU after we pull encoder output back

    output_dir = Path(args.output_dir)
    device_t = torch.device(args.device)
    for sess in args.sessions:
        precompute_session(sess, model, proj, cfg,
                           data_dir=Path(args.data_dir),
                           output_dir=output_dir,
                           batch_size=args.batch_size,
                           device=device_t)

    print("\nDone.")


if __name__ == "__main__":
    main()
