"""Sensor encoder — Chronos-2 multivariate embedding extraction.

Two-stage approach:
  Stage 1: Fine-tune Chronos-2 on domain sensor data (finetune_chronos.py)
  Stage 2: Freeze fine-tuned encoder, precompute embeddings for MDLM training

Uses Chronos-2's native multivariate support (GroupSelfAttention) to encode
all 7 sensor channels jointly, enabling cross-channel information sharing.
Calls model.encode() directly with group_ids for efficient batched inference.

Usage (precompute):
    python3 sensor_encoder.py \
        --sessions 2025-03-17_10-36-44 2025-03-19_10-05-47 \
        --output-dir sensor_embeddings/
        [--chronos-ckpt models/chronos2_finetuned/]
"""
import argparse, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from config import (
    SensorConfig, DATA_DIR, EMB_DIR, MODEL_DIR, SENSOR_COLS,
)


def encode_multivariate_batch(model, batch: torch.Tensor, n_channels: int = 7):
    """Encode a batch of multivariate sensor windows via Chronos-2.

    Calls model.encode() directly with group_ids for cross-channel attention,
    avoiding the overhead of pipeline.embed() (dataset/dataloader creation).

    Args:
        model: Chronos2Model (the .model attribute of Chronos2Pipeline)
        batch: (B, n_channels, T) — e.g. (32, 7, 500)
        n_channels: number of variates per sample
    Returns:
        (B, n_channels, 768) — mean-pooled per-channel embeddings with cross-attention
    """
    B, C, T = batch.shape
    device = model.device

    # Flatten: (B, 7, T) → (B*7, T)
    flat = batch.reshape(B * C, T).to(device=device, dtype=torch.float32)

    # group_ids: channels from the same sample share a group
    # [0,0,0,0,0,0,0, 1,1,1,1,1,1,1, ..., B-1,...,B-1]
    group_ids = torch.arange(B, device=device).repeat_interleave(C)

    with torch.no_grad():
        encoder_out, loc_scale, _, num_ctx_patches = model.encode(
            context=flat,
            group_ids=group_ids,
            num_output_patches=1,  # matches embed() default
        )
    # hidden: (B*7, num_patches+2, 768)
    hidden = encoder_out.last_hidden_state

    # Reshape: (B, 7, num_patches+2, 768)
    hidden = hidden.reshape(B, C, -1, hidden.shape[-1])

    # Mean pool over patches → (B, 7, 768)
    return hidden.mean(dim=2)


def precompute_session_embeddings(
    session: str,
    model,
    proj: nn.Module,
    cfg: SensorConfig = SensorConfig(),
    data_dir: Path = DATA_DIR,
    output_dir: Path = EMB_DIR,
):
    """Precompute sensor embeddings for one session."""
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_dir / f"{session}.parquet", columns=SENSOR_COLS)
    n = len(df)
    sensors = df[SENSOR_COLS].values.astype(np.float32)  # (n, 7)

    proj = proj.cpu().eval()
    embeddings = np.zeros((n, cfg.proj_dim), dtype=np.float32)

    for start in range(0, n, cfg.batch_size):
        end = min(start + cfg.batch_size, n)
        batch_windows = []

        for t in range(start, end):
            w_start = max(0, t - cfg.window_size + 1)
            window = sensors[w_start:t + 1, :]  # (<=500, 7)

            if window.shape[0] < cfg.window_size:
                pad = np.zeros((cfg.window_size - window.shape[0], cfg.n_channels),
                               dtype=np.float32)
                window = np.concatenate([pad, window], axis=0)

            batch_windows.append(window.T)  # (7, 500)

        batch = torch.from_numpy(np.stack(batch_windows))  # (B, 7, 500)

        # Encode with cross-channel attention
        with torch.amp.autocast(model.device.type, dtype=torch.float16):
            pooled = encode_multivariate_batch(model, batch, cfg.n_channels)
        # pooled: (B, 7, 768)

        flat = pooled.reshape(pooled.shape[0], -1).float().cpu()  # (B, 5376)
        with torch.no_grad():
            emb = proj(flat)  # (B, 512)
        embeddings[start:end] = emb.numpy()

        if (start // cfg.batch_size) % 50 == 0:
            elapsed = time.time() - t0
            rate = end / elapsed if elapsed > 0 else 0
            eta = (n - end) / rate if rate > 0 else 0
            print(f"  [{session}] {end}/{n} ticks ({elapsed:.0f}s, {rate:.0f} ticks/s, ETA {eta:.0f}s)",
                  flush=True)

    out_path = output_dir / f"{session}_emb.npy"
    np.save(out_path, embeddings)
    print(f"  [{session}] Saved {out_path} ({embeddings.shape}, {time.time()-t0:.0f}s)")
    return embeddings


def resolve_checkpoint(ckpt_path: str) -> str | None:
    """Resolve fine-tuned Chronos-2 checkpoint path."""
    p = Path(ckpt_path)
    finetuned = p / "finetuned-ckpt"
    if finetuned.exists():
        return str(finetuned)
    if p.exists() and (p / "config.json").exists():
        return str(p)
    return None


def _apply_chronos2_attn_patch() -> None:
    """Workaround: Chronos2*Config classes lack `_attn_implementation` that
    the installed transformers (4.33.3) expects. Class-level patch so all
    instances (incl. nested configs) inherit a safe default. Verified with
    chronos-forecasting 2.2.2 + transformers 4.33.3.
    """
    import chronos as _chronos
    seen = set()
    def walk(mod, depth=0):
        if depth > 3 or id(mod) in seen: return
        seen.add(id(mod))
        for name in dir(mod):
            if name.startswith('_'): continue
            try: obj = getattr(mod, name)
            except Exception: continue
            if isinstance(obj, type) and 'Config' in obj.__name__:
                if not hasattr(obj, '_attn_implementation'):
                    obj._attn_implementation = 'eager'
            elif hasattr(obj, '__path__'):
                walk(obj, depth + 1)
    walk(_chronos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--output-dir", default=str(EMB_DIR))
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--chronos-ckpt", default=None)
    ap.add_argument("--window-size", type=int, default=None,
                    help="Override SensorConfig.window_size (default 500). "
                         "Pair with --chronos-ckpt pointing to the matching "
                         "per-window finetune, and --output-dir for emb dest.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for MLP projection init (must match across sessions)")
    args = ap.parse_args()

    cfg = SensorConfig(batch_size=args.batch_size)
    if args.window_size is not None:
        cfg.window_size = args.window_size
    ckpt = args.chronos_ckpt or str(MODEL_DIR / "chronos2_finetuned")

    resolved = resolve_checkpoint(ckpt)
    if resolved:
        print(f"Loading fine-tuned Chronos-2 from {resolved}...")
        ckpt = resolved
    else:
        print(f"Fine-tuned checkpoint not found at {ckpt}, using base model: {cfg.chronos_model}")
        ckpt = cfg.chronos_model

    from chronos import Chronos2Pipeline
    _apply_chronos2_attn_patch()
    pipe = Chronos2Pipeline.from_pretrained(ckpt, device_map=args.device)
    model = pipe.model
    print(f"  Device: {model.device}")

    # MLP projection — fixed random init (same seed across all sessions)
    torch.manual_seed(args.seed)
    input_dim = cfg.n_channels * cfg.chronos_d_model  # 5376
    proj = nn.Sequential(
        nn.Linear(input_dim, cfg.proj_dim * 2),
        nn.GELU(),
        nn.Linear(cfg.proj_dim * 2, cfg.proj_dim),
    )

    for sess in args.sessions:
        precompute_session_embeddings(
            sess, model, proj, cfg,
            data_dir=Path(args.data_dir),
            output_dir=Path(args.output_dir),
        )


if __name__ == "__main__":
    main()
