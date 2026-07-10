"""AR baseline training loop — mirrors mdlm_train.py exactly, swaps MDLM → ARModel.

Usage:
    python3 ar_train.py --device cuda
    python3 ar_train.py --device cuda --sanity-check --limit 100
"""
import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from config import (
    ModelConfig, TrainConfig,
    TRAIN_SESSIONS, VAL_SESSIONS, MODEL_DIR, EMB_DIR, DATA_DIR,
)
from data import TraceDataset, load_token_mapping
from ar_model import ARModel
from determinism import DEFAULT_SEED, make_generator, make_worker_init_fn, set_deterministic
from mdlm_train import EMA, get_lr, setup_ddp, cleanup_ddp  # reuse


# ── Validation ───────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, val_dl, device, pad_id, max_batches=50):
    raw = model.module if hasattr(model, "module") else model
    raw.eval()
    totals, n = {}, 0
    for i, batch in enumerate(val_dl):
        if i >= max_batches:
            break
        x0 = batch["trace"].to(device)
        emb = batch["sensor_emb"].to(device)
        pad_mask = (x0 != pad_id).float()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, metrics = raw.compute_loss(x0, emb, pad_mask)

        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1

    raw.train()
    return {k: v / max(n, 1) for k, v in totals.items()}


# ── Checkpointing ────────────────────────────────────────────────────────
def save_checkpoint(path, step, model, optimizer, ema, mcfg, tcfg, best_val_loss, seed=None):
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
        "objective": "autoregressive",
        "seed": seed,
        "torch_version": torch.__version__,
    }, path)
    print(f"  Saved checkpoint: {path} (step {step})")


def load_checkpoint(path, model, optimizer, ema, device):
    raw = model.module if hasattr(model, "module") else model
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    ema.load_state_dict(ckpt["ema"])
    ema.to(device)
    print(f"  Resumed from {path} at step {ckpt['step']}")
    return ckpt["step"], ckpt.get("best_val_loss", float("inf"))


# ── Training Loop ────────────────────────────────────────────────────────
def train(args):
    rank, local_rank, world_size = setup_ddp()
    is_main = (rank == 0)
    device = torch.device(f"cuda:{local_rank}")

    vocab_size, _, _ = load_token_mapping()
    if is_main:
        print(f"Vocab size: {vocab_size}")
        print(f"World size: {world_size}")

    mcfg = ModelConfig(vocab_size=vocab_size, mask_token_id=vocab_size)
    tcfg = TrainConfig()

    # Apply CLI overrides (AR converges in ~3K steps on 2-session data)
    if args.max_steps is not None:
        tcfg.max_steps = args.max_steps
    if args.warmup_steps is not None:
        tcfg.warmup_steps = args.warmup_steps
    if args.eval_every is not None:
        tcfg.eval_every = args.eval_every
    if args.save_every is not None:
        tcfg.save_every = args.save_every

    if args.sanity_check:
        tcfg.max_steps = 2000
        tcfg.eval_every = 200
        tcfg.save_every = 500
        tcfg.log_every = 10
        tcfg.batch_size = 16
        tcfg.grad_accum = 1
        tcfg.warmup_steps = 100

    train_sessions = args.train_sessions or TRAIN_SESSIONS
    val_sessions = args.val_sessions or VAL_SESSIONS
    model_dir = Path(args.model_dir) if args.model_dir else MODEL_DIR

    pad_id = mcfg.vocab_size  # TraceDataset pads with this

    if is_main:
        print(f"Train sessions: {train_sessions}")
        print(f"Val sessions:   {val_sessions}")
        print("Loading datasets...")
    train_ds = TraceDataset(
        train_sessions, mcfg.max_seq_len, pad_id,
        emb_dir=Path(args.emb_dir), data_dir=Path(args.data_dir),
        limit=args.limit,
    )
    val_ds = TraceDataset(
        val_sessions, mcfg.max_seq_len, pad_id,
        emb_dir=Path(args.emb_dir), data_dir=Path(args.data_dir),
        limit=args.limit,
    )

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                        shuffle=True, seed=args.seed) if world_size > 1 else None
    worker_init = make_worker_init_fn(args.seed)
    train_dl = DataLoader(
        train_ds, batch_size=tcfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
        worker_init_fn=worker_init,
        generator=make_generator(args.seed),
    )
    val_dl = DataLoader(
        val_ds, batch_size=tcfg.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
        worker_init_fn=worker_init,
    )

    if is_main:
        eff_batch = tcfg.batch_size * tcfg.grad_accum * world_size
        print(f"Train: {len(train_ds)} samples, {len(train_dl)} batches/epoch")
        print(f"Val:   {len(val_ds)} samples, {len(val_dl)} batches/epoch")

    model = ARModel(mcfg).to(device)
    if is_main:
        print(f"Model parameters: {model.param_count():,}")

    if tcfg.grad_checkpoint:
        for block in model.backbone.blocks:
            block._orig_forward = block.forward
            block.forward = lambda *a, _b=block, **kw: torch.utils.checkpoint.checkpoint(
                _b._orig_forward, *a, use_reentrant=False, **kw
            )

    if not args.sanity_check and not args.no_compile:
        if is_main:
            print("Compiling model with torch.compile...")
        model = torch.compile(model)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay,
        betas=(0.9, 0.999), fused=True,
    )

    raw = model.module if hasattr(model, "module") else model
    ema = EMA(raw.parameters(), decay=tcfg.ema_decay)
    ema.to(device)

    step = 0
    best_val_loss = float("inf")
    if args.resume:
        step, best_val_loss = load_checkpoint(Path(args.resume), model, optimizer, ema, device)

    model.train()
    optimizer.zero_grad()
    running, t0, epoch = {}, time.time(), 0
    data_iter = iter(train_dl)

    if is_main:
        eff_batch = tcfg.batch_size * tcfg.grad_accum * world_size
        print(f"\nStarting AR training from step {step}...")
        print(f"  Effective batch: {tcfg.batch_size} × {tcfg.grad_accum} × {world_size}gpu = {eff_batch}")
        print(f"  Max steps: {tcfg.max_steps}")
        print(f"  Pad id: {pad_id}, BOS id: {pad_id + 1}, vocab_total: {mcfg.vocab_size + 2}")
        print()

    while step < tcfg.max_steps:
        for _ in range(tcfg.grad_accum):
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
            pad_mask = (x0 != pad_id).float()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=tcfg.bf16):
                if hasattr(model, "module"):
                    loss, metrics = model.module.compute_loss(x0, emb, pad_mask)
                else:
                    loss, metrics = model.compute_loss(x0, emb, pad_mask)
                loss = loss / tcfg.grad_accum

            loss.backward()

            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + v / tcfg.grad_accum

        if tcfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        step += 1
        lr = get_lr(step, tcfg.warmup_steps, tcfg.max_steps, tcfg.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        ema.update(raw.parameters())

        if is_main and step % tcfg.log_every == 0:
            elapsed = time.time() - t0
            steps_per_sec = tcfg.log_every / elapsed if elapsed > 0 else 0
            n = tcfg.log_every
            ppl = math.exp(min(running.get("loss", 0) / n, 20))
            print(
                f"step {step:>7d}/{tcfg.max_steps} | "
                f"loss {running.get('loss', 0) / n:.4f} | "
                f"ppl {ppl:.2f} | "
                f"tf_acc {running.get('acc_teacher_forced', 0) / n:.3f} | "
                f"lr {lr:.2e} | "
                f"{steps_per_sec:.1f} steps/s"
            )
            running = {}
            t0 = time.time()

        if step % tcfg.eval_every == 0:
            ema.store(raw.parameters())
            ema.copy_to(raw.parameters())

            val_metrics = validate(model, val_dl, device, pad_id)
            val_loss = val_metrics.get("loss", float("inf"))
            if is_main:
                val_ppl = math.exp(min(val_loss, 20))
                print(
                    f"  [VAL] step {step} | "
                    f"loss {val_loss:.4f} | "
                    f"ppl {val_ppl:.2f} | "
                    f"tf_acc {val_metrics.get('acc_teacher_forced', 0):.3f}"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model_dir / "ar_best.pt", step, model, optimizer,
                                    ema, mcfg, tcfg, best_val_loss, seed=args.seed)

            ema.restore(raw.parameters())
            model.train()

        if is_main and step % tcfg.save_every == 0:
            save_checkpoint(model_dir / f"ar_step_{step}.pt", step, model, optimizer,
                            ema, mcfg, tcfg, best_val_loss, seed=args.seed)

    if is_main:
        save_checkpoint(model_dir / f"ar_final_{step}.pt", step, model, optimizer,
                        ema, mcfg, tcfg, best_val_loss, seed=args.seed)
        print(f"\nAR training complete. Best val loss: {best_val_loss:.4f}")

    cleanup_ddp()


def main():
    ap = argparse.ArgumentParser(description="BL-2 AR baseline training")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--emb-dir", default=str(EMB_DIR))
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sanity-check", action="store_true")
    ap.add_argument("--train-sessions", nargs="+", default=None)
    ap.add_argument("--val-sessions", nargs="+", default=None)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--no-compile", action="store_true")
    # TrainConfig overrides (AR converges in ~3K steps on 2-session data)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--warmup-steps", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="Global seed for determinism (RNGs + CUBLAS + cuDNN)")
    args = ap.parse_args()
    set_deterministic(args.seed)
    train(args)


if __name__ == "__main__":
    main()
