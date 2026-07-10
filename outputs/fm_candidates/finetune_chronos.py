"""BL-2 Stage 1 — Fine-tune Chronos-2 on domain sensor data.

Uses Chronos-2's built-in .fit() API with forecasting (quantile loss)
objective on the 7 sensor channels from the ASPK pendulum system.

Each channel is treated as an independent univariate time series.
The fine-tuned model is saved for Stage 2 (frozen embedding precomputation).

Usage:
    python3 finetune_chronos.py --device cuda
    python3 finetune_chronos.py --device cuda --sessions s1 s2 ...
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import (
    SensorConfig, TRAIN_SESSIONS, VAL_SESSIONS,
    DATA_DIR, MODEL_DIR, SENSOR_COLS,
)


def _apply_chronos2_attn_patch() -> None:
    """Workaround: Chronos2*Config classes lack `_attn_implementation` that
    the installed transformers (4.33.3) expects. Class-level patch so all
    instances (incl. nested configs) inherit a safe default.
    Verified with chronos-forecasting 2.2.2 + transformers 4.33.3.
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


def load_sensor_series(
    sessions: list[str],
    data_dir: Path,
    window_size: int = 500,
    stride: int = 250,
) -> list[dict]:
    """Load sensor data as multivariate series for Chronos2Pipeline.fit().

    Extracts overlapping windows with all 7 channels as a single
    multivariate time series. Chronos-2's GroupSelfAttention enables
    cross-channel information sharing during fine-tuning.

    Returns:
        list of dicts with "target" key (2D arrays of shape (7, window_size))
    """
    all_series = []
    for sess in sessions:
        pq_path = data_dir / f"{sess}.parquet"
        df = pd.read_parquet(pq_path, columns=SENSOR_COLS)
        sensors = df[SENSOR_COLS].values.astype(np.float32)  # (n, 7)
        n = len(sensors)

        for start in range(0, n - window_size, stride):
            window = sensors[start:start + window_size, :]  # (window_size, 7)
            all_series.append({"target": window.T})  # (7, window_size)

    return all_series


def finetune(args):
    from chronos import Chronos2Pipeline
    _apply_chronos2_attn_patch()

    cfg = SensorConfig()
    if args.window_size is not None:
        cfg.window_size = args.window_size  # override per-sweep
    if args.num_steps is not None:
        cfg.ft_num_steps = args.num_steps
    if args.batch_size is not None:
        cfg.ft_batch_size = args.batch_size
    device = torch.device(args.device)

    # Sessions
    train_sessions = args.sessions if args.sessions else TRAIN_SESSIONS
    val_sessions = args.val_sessions if args.val_sessions else VAL_SESSIONS

    print(f"Stage 1: Fine-tune Chronos-2 on sensor data")
    print(f"  Model: {cfg.chronos_model}")
    print(f"  Train sessions: {len(train_sessions)}")
    print(f"  Val sessions: {len(val_sessions)}")
    print(f"  Prediction length: {cfg.ft_prediction_length}")
    print(f"  Num steps: {cfg.ft_num_steps}")
    print(f"  Learning rate: {cfg.ft_learning_rate}")
    print(f"  Batch size: {cfg.ft_batch_size}")
    print()

    # Load training data
    print("Loading training sensor series...")
    t0 = time.time()
    train_data = load_sensor_series(
        train_sessions, Path(args.data_dir),
        window_size=cfg.window_size, stride=args.stride,
    )
    print(f"  Loaded {len(train_data)} training windows ({time.time()-t0:.1f}s)")

    # Load validation data
    val_data = None
    if val_sessions:
        print("Loading validation sensor series...")
        val_data = load_sensor_series(
            val_sessions, Path(args.data_dir),
            window_size=cfg.window_size, stride=args.stride,
        )
        print(f"  Loaded {len(val_data)} validation windows")

    # Load base model
    print(f"\nLoading {cfg.chronos_model}...")
    pipeline = Chronos2Pipeline.from_pretrained(
        cfg.chronos_model,
        device_map=args.device,
    )

    # Fine-tune: per-window output dir (chronos2_finetuned_w{W}); 500 keeps
    # the legacy name "chronos2_finetuned" for backward compatibility.
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    elif cfg.window_size == 500:
        output_dir = MODEL_DIR / "chronos2_finetuned"
    else:
        output_dir = MODEL_DIR / f"chronos2_finetuned_w{cfg.window_size}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFine-tuning for {cfg.ft_num_steps} steps...")
    print(f"  Output: {output_dir}")
    t0 = time.time()

    # Strategy B: grad-accum keeps effective batch constant across W.
    # pipeline.fit absorbs extras via **extra_trainer_kwargs (line 220 in
    # chronos/chronos2/pipeline.py: training_kwargs.update(extra_trainer_kwargs))
    # which then go to TrainingArguments; pass gradient_accumulation_steps
    # directly as a kwarg.
    extra_kwargs = {}
    if args.grad_accum_steps and args.grad_accum_steps > 1:
        extra_kwargs["gradient_accumulation_steps"] = args.grad_accum_steps
    eff_bs = cfg.ft_batch_size * max(args.grad_accum_steps or 1, 1)
    print(f"  per-step batch = {cfg.ft_batch_size}, grad-accum = "
          f"{args.grad_accum_steps or 1}, effective batch = {eff_bs}")

    finetuned = pipeline.fit(
        inputs=train_data,
        prediction_length=cfg.ft_prediction_length,
        validation_inputs=val_data,
        finetune_mode=args.finetune_mode,
        learning_rate=cfg.ft_learning_rate,
        num_steps=cfg.ft_num_steps,
        batch_size=cfg.ft_batch_size,
        output_dir=str(output_dir),
        **extra_kwargs,
    )

    elapsed = time.time() - t0
    print(f"\nFine-tuning complete ({elapsed:.0f}s)")
    print(f"  Model saved to: {output_dir}")

    # Save a marker file with config
    import json
    meta = {
        "base_model": cfg.chronos_model,
        "finetune_mode": args.finetune_mode,
        "prediction_length": cfg.ft_prediction_length,
        "num_steps": cfg.ft_num_steps,
        "learning_rate": cfg.ft_learning_rate,
        "batch_size": cfg.ft_batch_size,
        "n_train_windows": len(train_data),
        "n_val_windows": len(val_data) if val_data else 0,
        "train_sessions": train_sessions,
        "val_sessions": val_sessions,
        "stride": args.stride,
        "elapsed_s": elapsed,
    }
    (output_dir / "finetune_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  Metadata saved to: {output_dir / 'finetune_meta.json'}")


def main():
    ap = argparse.ArgumentParser(description="BL-2 Stage 1: Fine-tune Chronos-2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--sessions", nargs="+", default=None,
                    help="Train session IDs (default: TRAIN_SESSIONS)")
    ap.add_argument("--val-sessions", nargs="+", default=None,
                    help="Val session IDs (default: VAL_SESSIONS)")
    ap.add_argument("--stride", type=int, default=250,
                    help="Window stride for extracting training series")
    ap.add_argument("--finetune-mode", default="full", choices=["full", "lora"],
                    help="Fine-tuning mode (default: full)")
    ap.add_argument("--window-size", type=int, default=None,
                    help="Override SensorConfig.window_size (default: 500). "
                         "Output ckpt -> models/chronos2_finetuned_w<W>/ unless "
                         "--output-dir is given; W=500 uses legacy "
                         "models/chronos2_finetuned/.")
    ap.add_argument("--output-dir", default=None,
                    help="Override finetune output dir.")
    ap.add_argument("--num-steps", type=int, default=None,
                    help="Override SensorConfig.ft_num_steps (for smoke tests).")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Override SensorConfig.ft_batch_size.")
    ap.add_argument("--grad-accum-steps", type=int, default=1,
                    help="Gradient accumulation steps. Use with reduced "
                         "--batch-size to keep effective batch constant "
                         "(strategy B for the window sweep).")
    args = ap.parse_args()
    finetune(args)


if __name__ == "__main__":
    main()
