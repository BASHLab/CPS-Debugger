"""MDLM training loop — supports multi-GPU DDP.

Usage:
    # Single GPU
    python3 mdlm_train.py --device cuda

    # Multi-GPU (4 GPUs)
    torchrun --nproc_per_node=4 mdlm_train.py

    # Sanity check
    python3 mdlm_train.py --device cuda --sanity-check --limit 100

    # Resume from checkpoint
    torchrun --nproc_per_node=4 mdlm_train.py --resume models/mdlm_step_10000.pt
"""
import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from config import (
    ModelConfig, TrainConfig, SensorConfig,
    TRAIN_SESSIONS, VAL_SESSIONS, MODEL_DIR, EMB_DIR, DATA_DIR, TOKEN_MAPPING,
)
from data import TraceDataset, load_token_mapping
from mdlm_model import MDLM


# ── EMA ──────────────────────────────────────────────────────────────────
class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, parameters, decay: float):
        self.decay = decay
        self.num_updates = 0
        self.shadow = [p.clone().detach() for p in parameters if p.requires_grad]

    def update(self, parameters):
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        with torch.no_grad():
            params = [p for p in parameters if p.requires_grad]
            for s, p in zip(self.shadow, params):
                s.lerp_(p.data, 1 - decay)

    def copy_to(self, parameters):
        params = [p for p in parameters if p.requires_grad]
        for s, p in zip(self.shadow, params):
            p.data.copy_(s.data)

    def store(self, parameters):
        self._backup = [p.clone() for p in parameters if p.requires_grad]

    def restore(self, parameters):
        params = [p for p in parameters if p.requires_grad]
        for b, p in zip(self._backup, params):
            p.data.copy_(b.data)

    def to(self, device):
        self.shadow = [s.to(device) for s in self.shadow]
        return self

    def state_dict(self):
        return {"decay": self.decay, "num_updates": self.num_updates, "shadow": self.shadow}

    def load_state_dict(self, d):
        self.decay = d["decay"]
        self.num_updates = d["num_updates"]
        self.shadow = d["shadow"]


# ── LR Schedule ──────────────────────────────────────────────────────────
def get_lr(step: int, warmup: int, max_steps: int, peak_lr: float) -> float:
    """Cosine decay with linear warmup."""
    if step < warmup:
        return peak_lr * step / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return peak_lr * 0.5 * (1 + math.cos(math.pi * progress))


# ── Validation ───────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, val_dl: DataLoader, device: torch.device, mask_id: int,
             max_batches: int = 50):
    """Run validation and return mean metrics."""
    # Unwrap DDP if needed
    raw = model.module if hasattr(model, "module") else model
    raw.eval()
    totals = {}
    n = 0
    for i, batch in enumerate(val_dl):
        if i >= max_batches:
            break
        x0 = batch["trace"].to(device)
        emb = batch["sensor_emb"].to(device)
        pad_mask = (x0 != mask_id).float()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, metrics = raw.compute_loss(x0, emb, pad_mask)

        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1

    raw.train()
    return {k: v / max(n, 1) for k, v in totals.items()}


# ── Checkpointing ────────────────────────────────────────────────────────
def save_checkpoint(path: Path, step: int, model, optimizer, ema: EMA,
                    mcfg: ModelConfig, tcfg: TrainConfig, best_val_loss: float):
    # Unwrap DDP
    raw = model.module if hasattr(model, "module") else model
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema": ema.state_dict(),
        "model_config": mcfg.__dict__,
        "train_config": tcfg.__dict__,
        "best_val_loss": best_val_loss,
    }, path)
    print(f"  Saved checkpoint: {path} (step {step})")


def load_checkpoint(path: Path, model, optimizer, ema: EMA, device):
    raw = model.module if hasattr(model, "module") else model
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    ema.load_state_dict(ckpt["ema"])
    ema.to(device)
    print(f"  Resumed from {path} at step {ckpt['step']}")
    return ckpt["step"], ckpt.get("best_val_loss", float("inf"))


# ── DDP Setup ────────────────────────────────────────────────────────────
def setup_ddp():
    """Initialize DDP if launched via torchrun."""
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


# ── Training Loop ────────────────────────────────────────────────────────
def train(args):
    rank, local_rank, world_size = setup_ddp()
    is_main = (rank == 0)
    device = torch.device(f"cuda:{local_rank}")

    # Load vocab
    vocab_size, _, _ = load_token_mapping()
    if is_main:
        print(f"Vocab size: {vocab_size}")
        print(f"World size: {world_size}")

    # Configs
    mcfg = ModelConfig(vocab_size=vocab_size, mask_token_id=vocab_size)
    tcfg = TrainConfig()

    if args.sanity_check:
        tcfg.max_steps = 2000
        tcfg.eval_every = 200
        tcfg.save_every = 500
        tcfg.log_every = 10
        tcfg.batch_size = 16
        tcfg.grad_accum = 1
        tcfg.warmup_steps = 100

    # Session overrides
    train_sessions = args.train_sessions or TRAIN_SESSIONS
    val_sessions = args.val_sessions or VAL_SESSIONS
    model_dir = Path(args.model_dir) if args.model_dir else MODEL_DIR

    # Data
    if is_main:
        print(f"Train sessions: {train_sessions}")
        print(f"Val sessions:   {val_sessions}")
        print("Loading datasets...")
    train_ds = TraceDataset(
        train_sessions, mcfg.max_seq_len, mcfg.vocab_size,
        emb_dir=Path(args.emb_dir), data_dir=Path(args.data_dir),
        limit=args.limit,
    )
    val_ds = TraceDataset(
        val_sessions, mcfg.max_seq_len, mcfg.vocab_size,
        emb_dir=Path(args.emb_dir), data_dir=Path(args.data_dir),
        limit=args.limit,
    )

    # DDP sampler for training
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                        shuffle=True) if world_size > 1 else None
    train_dl = DataLoader(
        train_ds, batch_size=tcfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    val_dl = DataLoader(
        val_ds, batch_size=tcfg.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    if is_main:
        eff_batch = tcfg.batch_size * tcfg.grad_accum * world_size
        print(f"Train: {len(train_ds)} samples, {len(train_dl)} batches/epoch")
        print(f"Val:   {len(val_ds)} samples, {len(val_dl)} batches/epoch")

    # Model
    model = MDLM(mcfg).to(device)
    if is_main:
        print(f"Model parameters: {model.param_count():,}")

    # Gradient checkpointing
    if tcfg.grad_checkpoint:
        for block in model.backbone.blocks:
            block._orig_forward = block.forward
            block.forward = lambda *a, _b=block, **kw: torch.utils.checkpoint.checkpoint(
                _b._orig_forward, *a, use_reentrant=False, **kw
            )

    # torch.compile for speed
    if not args.sanity_check and not args.no_compile:
        if is_main:
            print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # DDP wrapper
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # Optimizer (fused for speed; no GradScaler needed with bf16)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay,
        betas=(0.9, 0.999), fused=True,
    )

    # EMA (on the raw model parameters)
    raw = model.module if hasattr(model, "module") else model
    ema = EMA(raw.parameters(), decay=tcfg.ema_decay)
    ema.to(device)

    # Resume
    step = 0
    best_val_loss = float("inf")
    if args.resume:
        step, best_val_loss = load_checkpoint(
            Path(args.resume), model, optimizer, ema, device
        )

    # Training
    model.train()
    optimizer.zero_grad()
    running = {}
    t0 = time.time()
    epoch = 0
    data_iter = iter(train_dl)
    mask_id = mcfg.vocab_size  # = mask_token_id

    if is_main:
        eff_batch = tcfg.batch_size * tcfg.grad_accum * world_size
        print(f"\nStarting training from step {step}...")
        print(f"  Effective batch: {tcfg.batch_size} × {tcfg.grad_accum} × {world_size}gpu = {eff_batch}")
        print(f"  Max steps: {tcfg.max_steps}")
        print(f"  bf16: {tcfg.bf16}, grad_clip: {tcfg.grad_clip}")
        print(f"  Gradient checkpointing: {tcfg.grad_checkpoint}")
        print(f"  torch.compile: {not args.sanity_check and not args.no_compile}")
        print()

    while step < tcfg.max_steps:
        # Accumulation loop
        for accum_step in range(tcfg.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                data_iter = iter(train_dl)
                batch = next(data_iter)

            x0 = batch["trace"].to(device, non_blocking=True)
            emb = batch["sensor_emb"].to(device, non_blocking=True)
            pad_mask = (x0 != mask_id).float()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=tcfg.bf16):
                if hasattr(model, "module"):
                    loss, metrics = model.module.compute_loss(x0, emb, pad_mask)
                else:
                    loss, metrics = model.compute_loss(x0, emb, pad_mask)
                loss = loss / tcfg.grad_accum

            loss.backward()

            # Accumulate metrics
            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + v / tcfg.grad_accum

        # Optimizer step
        if tcfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)

        optimizer.step()
        optimizer.zero_grad()

        # LR schedule
        step += 1
        lr = get_lr(step, tcfg.warmup_steps, tcfg.max_steps, tcfg.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # EMA update
        ema.update(raw.parameters())

        # Logging (main process only)
        if is_main and step % tcfg.log_every == 0:
            elapsed = time.time() - t0
            steps_per_sec = tcfg.log_every / elapsed if elapsed > 0 else 0
            n = tcfg.log_every
            print(
                f"step {step:>7d}/{tcfg.max_steps} | "
                f"loss {running.get('loss', 0) / n:.4f} | "
                f"acc {running.get('acc_masked', 0) / n:.3f} | "
                f"σ {running.get('mean_sigma', 0) / n:.3f} | "
                f"frac_masked {running.get('frac_masked', 0) / n:.3f} | "
                f"lr {lr:.2e} | "
                f"{steps_per_sec:.1f} steps/s"
            )
            running = {}
            t0 = time.time()

        # Validation
        if step % tcfg.eval_every == 0:
            ema.store(raw.parameters())
            ema.copy_to(raw.parameters())

            val_metrics = validate(model, val_dl, device, mask_id)
            val_loss = val_metrics.get("loss", float("inf"))
            if is_main:
                print(
                    f"  [VAL] step {step} | "
                    f"loss {val_loss:.4f} | "
                    f"acc {val_metrics.get('acc_masked', 0):.3f} | "
                    f"σ {val_metrics.get('mean_sigma', 0):.3f}"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model_dir / "mdlm_best.pt", step, model, optimizer, ema, mcfg, tcfg,
                        best_val_loss,
                    )

            ema.restore(raw.parameters())
            model.train()

        # Periodic checkpoint
        if is_main and step % tcfg.save_every == 0:
            save_checkpoint(
                model_dir / f"mdlm_step_{step}.pt", step, model, optimizer, ema, mcfg, tcfg,
                best_val_loss,
            )

    # Final save
    if is_main:
        save_checkpoint(
            model_dir / f"mdlm_final_{step}.pt", step, model, optimizer, ema, mcfg, tcfg,
            best_val_loss,
        )
        print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

    cleanup_ddp()


def main():
    ap = argparse.ArgumentParser(description="BL-2 MDLM training")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    ap.add_argument("--emb-dir", default=str(EMB_DIR))
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit samples per session (for testing)")
    ap.add_argument("--sanity-check", action="store_true",
                    help="Quick overfit run with reduced steps")
    ap.add_argument("--train-sessions", nargs="+", default=None,
                    help="Override train sessions (e.g. 2025-03-17_10-36-44)")
    ap.add_argument("--val-sessions", nargs="+", default=None,
                    help="Override val sessions")
    ap.add_argument("--model-dir", default=None,
                    help="Override model output directory")
    ap.add_argument("--no-compile", action="store_true",
                    help="Disable torch.compile")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
