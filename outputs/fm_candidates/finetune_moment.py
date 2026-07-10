"""BL-6 Stage 1 — Fine-tune MOMENT-1-large on the same forecast task as Chronos-2.

Apples-to-apples to the Chronos-2 pipeline: same 7 sensor channels
(SENSOR_COLS), same window_size=500 prefix → the last 512 timesteps go in,
forecast horizon=64. Standard MSE loss. Saved encoder is the input to
sensor_encoder_moment.py (Stage 2).

Differences vs `finetune_chronos.py`:
  - MOMENT input length is fixed at 512 (its pretraining context). We use
    the same 500-window stride pattern but pad/truncate to 512 in the
    forward pass.
  - MOMENT processes channels independently (no GroupSelfAttention), so we
    just reshape (B, 7, 512) → (B*7, 1, 512) for the forward and reshape
    back, treating each channel as a univariate series — matching the
    standard MOMENT usage.
  - MSE loss (MOMENT default) instead of Chronos-2's quantile loss.

Usage:
    python3 finetune_moment.py --device cuda
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config import (
    SensorConfig, TRAIN_SESSIONS, VAL_SESSIONS,
    DATA_DIR, MODEL_DIR, SENSOR_COLS,
)
from determinism import DEFAULT_SEED, set_deterministic

MOMENT_INPUT_LEN  = 512
MOMENT_MODEL_NAME = "AutonLab/MOMENT-1-large"
FORECAST_HORIZON  = 64        # matches SensorConfig.ft_prediction_length
NUM_STEPS         = 5000      # matches SensorConfig.ft_num_steps
LR                = 1e-4      # matches SensorConfig.ft_learning_rate
BATCH_SIZE        = 64        # MOMENT-1-large is bigger than Chronos-2; halve
WINDOW_STRIDE     = 250       # half the 500-input stride


class MomentForecastWindows(Dataset):
    """Sliding-window dataset over a list of session parquets.

    Each sample: x = sensor[t : t+512] (input), y = sensor[t+512 : t+512+H]
    (forecast target). Both shape (n_channels, T).
    """
    def __init__(self, sessions, data_dir: Path, horizon: int):
        self.windows = []        # list of (channels, time) np.ndarray
        self.targets = []        # list of (channels, horizon) np.ndarray
        total_w = 0
        for sess in sessions:
            df = pd.read_parquet(data_dir / f"{sess}.parquet", columns=SENSOR_COLS)
            arr = df[SENSOR_COLS].to_numpy(dtype=np.float32)        # (n, C)
            n = len(arr)
            for start in range(0, n - MOMENT_INPUT_LEN - horizon, WINDOW_STRIDE):
                x = arr[start:start + MOMENT_INPUT_LEN].T            # (C, 512)
                y = arr[start + MOMENT_INPUT_LEN:
                        start + MOMENT_INPUT_LEN + horizon].T        # (C, H)
                self.windows.append(x)
                self.targets.append(y)
                total_w += 1
            print(f"  {sess}: n={n} windows so far={total_w}")
        print(f"Total windows: {total_w}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.windows[idx]).float(),
                torch.from_numpy(self.targets[idx]).float())


def normalize_per_channel(x: torch.Tensor):
    """Per-channel mean/std normalize (MOMENT expects standardized inputs).

    sd is clamped to a non-trivial floor (1.0) — when a channel is nearly
    constant inside the 512-window (e.g. early-session `iteration` ≈ 0,
    or `pendulum_state` stuck in one state), 1e-6 floors the std and any
    out-of-window y target then explodes to ~1e6 in normalized space,
    producing useless MSE in the millions. Floor=1 keeps such windows
    pass-through-scaled instead.
    """
    mean = x.mean(dim=-1, keepdim=True)
    std  = x.std(dim=-1, keepdim=True).clamp(min=1.0)
    return (x - mean) / std, mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--model-dir", default=str(MODEL_DIR / "moment_finetuned"))
    ap.add_argument("--sessions", nargs="+", default=None)
    ap.add_argument("--val-sessions", nargs="+", default=None)
    ap.add_argument("--num-steps", type=int, default=NUM_STEPS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    set_deterministic(args.seed)
    device = torch.device(args.device)
    model_dir = Path(args.model_dir); model_dir.mkdir(parents=True, exist_ok=True)

    train_sessions = args.sessions     or TRAIN_SESSIONS
    val_sessions   = args.val_sessions or VAL_SESSIONS
    print(f"Stage 1: Fine-tune {MOMENT_MODEL_NAME}")
    print(f"  train: {len(train_sessions)} sessions, val: {len(val_sessions)}")
    print(f"  steps={args.num_steps}, lr={args.lr}, batch={args.batch_size}")
    print(f"  forecast_horizon={FORECAST_HORIZON}, window_size={MOMENT_INPUT_LEN}")

    print("Loading training windows...")
    t0 = time.time()
    train_ds = MomentForecastWindows(train_sessions, Path(args.data_dir), FORECAST_HORIZON)
    val_ds   = MomentForecastWindows(val_sessions,   Path(args.data_dir), FORECAST_HORIZON)
    print(f"  loaded in {time.time()-t0:.1f}s")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)

    print("Loading MOMENT-1-large...")
    from momentfm import MOMENTPipeline
    model = MOMENTPipeline.from_pretrained(
        MOMENT_MODEL_NAME,
        model_kwargs={
            "task_name":         "forecasting",
            "forecast_horizon":  FORECAST_HORIZON,
            "head_dropout":      0.1,
            "weight_decay":      0,
            "freeze_encoder":    False,
            "freeze_embedder":   False,
            "freeze_head":       False,
        },
    )
    model.init()
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params/1e6:.1f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_steps)
    loss_fn = nn.MSELoss()

    step = 0
    best_val = float("inf")
    running, t0 = 0.0, time.time()
    iter_dl = iter(train_dl)
    while step < args.num_steps:
        try:
            x, y = next(iter_dl)
        except StopIteration:
            iter_dl = iter(train_dl)
            x, y = next(iter_dl)

        x = x.to(device, non_blocking=True)        # (B, C, 512)
        y = y.to(device, non_blocking=True)        # (B, C, H)

        # Per-channel normalize input; same scale applied to target so MSE is meaningful
        x_norm, mu, sd = normalize_per_channel(x)
        y_norm = (y - mu) / sd

        opt.zero_grad()
        out = model(x_enc=x_norm)
        pred = out.forecast                        # (B, C, H)
        loss = loss_fn(pred, y_norm)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        running += loss.item()
        step += 1
        if step % 50 == 0:
            elapsed = time.time() - t0
            print(f"step {step:>5d}/{args.num_steps} | mse {running/50:.4f} | "
                  f"lr {sched.get_last_lr()[0]:.2e} | "
                  f"{50/elapsed:.2f} steps/s")
            running = 0.0; t0 = time.time()

        if step % 500 == 0 or step == args.num_steps:
            model.eval()
            vlosses = []
            with torch.no_grad():
                for vi, (vx, vy) in enumerate(val_dl):
                    if vi >= 50: break
                    vx = vx.to(device); vy = vy.to(device)
                    vx_n, mu, sd = normalize_per_channel(vx)
                    vy_n = (vy - mu) / sd
                    vp = model(x_enc=vx_n).forecast
                    vlosses.append(loss_fn(vp, vy_n).item())
            val_mse = float(np.mean(vlosses)) if vlosses else float("inf")
            print(f"  [VAL] step {step} | val_mse {val_mse:.4f}")
            if val_mse < best_val:
                best_val = val_mse
                torch.save({
                    "step": step,
                    "model_state": model.state_dict(),
                    "val_mse": val_mse,
                    "moment_model_name": MOMENT_MODEL_NAME,
                    "forecast_horizon": FORECAST_HORIZON,
                    "input_len": MOMENT_INPUT_LEN,
                }, model_dir / "moment_best.pt")
                print(f"  saved best -> {model_dir/'moment_best.pt'}")
            model.train()

    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "val_mse": best_val,
        "moment_model_name": MOMENT_MODEL_NAME,
        "forecast_horizon": FORECAST_HORIZON,
        "input_len": MOMENT_INPUT_LEN,
    }, model_dir / f"moment_final_{step}.pt")
    print(f"\nDone. Best val MSE: {best_val:.4f}")


if __name__ == "__main__":
    main()
