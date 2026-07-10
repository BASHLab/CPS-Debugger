"""Training entry point for BL-3 / BL-4 / BL-5 (frozen video features).

Same architecture and optimizer setup as `ar_modern_train.py`. Only differences:
  * dataset wraps `TraceDataset` with `VideoConditionedTraceDataset` so each
    item carries pre-aligned V-JEPA / CoTracker embeddings;
  * model is `VideoConditionedARModern` with a wider conditioning projection;
  * conditioning vector is the concatenation of the modalities selected by
    --variant.

  bl3   = sensor + V-JEPA            (1536-d conditioning)
  bl4   = sensor + CoTracker         ( 572-d conditioning)
  bl5   = sensor + V-JEPA + CoTracker (1596-d conditioning)

BL-3b (LoRA-adapted V-JEPA) is NOT this script — it requires V-JEPA in the
training graph and online clip loading.
"""
import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from config import (
    ModelConfig, TrainConfig,
    TRAIN_SESSIONS, VAL_SESSIONS, MODEL_DIR, EMB_DIR, DATA_DIR,
)
from data import TraceDataset, load_token_mapping
from ar_video_model import VideoConditionedARModern
from determinism import DEFAULT_SEED, make_generator, make_worker_init_fn, set_deterministic
from mdlm_train import EMA, get_lr, setup_ddp, cleanup_ddp
from ar_modern_train import (
    Muon, split_muon_adam_params, validate as _ar_validate,
)
from video_data import (
    VideoConditionedTraceDataset, assemble_conditioning, conditioning_dim, VJEPA_DIM, COTRACKER_DIM,
)


# Per-variant config: sensor encoder choice + which video modalities to concat.
# Sensor encoders cache to different emb_dir / dim. BL-6b's video subset is
# tentatively (vjepa, cotracker); we will narrow to the BL-3/3b/4/5 winner
# once those results land — adjust use_vjepa / use_cotracker here at that time.
VARIANT_TO_CONFIG = {
    "bl3":  dict(sensor_kind="chronos2", use_vjepa=True,  use_cotracker=False),
    "bl4":  dict(sensor_kind="chronos2", use_vjepa=False, use_cotracker=True),
    "bl5":  dict(sensor_kind="chronos2", use_vjepa=True,  use_cotracker=True),
    "bl6":  dict(sensor_kind="moment",   use_vjepa=False, use_cotracker=False),
    "bl6b": dict(sensor_kind="moment",   use_vjepa=True,  use_cotracker=True),
}
SENSOR_KIND_TO_DIR_DIM = {
    "chronos2": (str(EMB_DIR), 512),
    "moment":   (str(EMB_DIR.parent / "sensor_embeddings_moment"), 1024),
}
OBJECTIVE_NAME = {v: f"ar_modern_video_{v}" for v in VARIANT_TO_CONFIG}

# Video flags subset of variant config (for VideoConditionedTraceDataset)
def _video_flags(variant: str) -> dict:
    cfg = VARIANT_TO_CONFIG[variant]
    return {"use_vjepa": cfg["use_vjepa"], "use_cotracker": cfg["use_cotracker"]}


def video_validate(model, val_dl, device, pad_id, flags, max_batches=50):
    raw = model.module if hasattr(model, "module") else model
    raw.eval()
    totals, n = {}, 0
    for i, batch in enumerate(val_dl):
        if i >= max_batches:
            break
        x0 = batch["trace"].to(device)
        cond = assemble_conditioning(batch, **flags).to(device)
        pad_mask = (x0 != pad_id).float()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, metrics = raw.compute_loss(x0, cond, pad_mask)
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1
    raw.train()
    return {k: v / max(n, 1) for k, v in totals.items()}


def save_video_checkpoint(path, step, model, opt_muon, opt_adam, ema, mcfg, tcfg,
                          best_val_loss, *, variant: str, conditioning_dim_val: int,
                          flags: dict, sensor_emb_dir: str, seed=None):
    raw = model.module if hasattr(model, "module") else model
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "model": raw.state_dict(),
        "opt_muon": opt_muon.state_dict(),
        "opt_adam": opt_adam.state_dict(),
        "ema": ema.state_dict(),
        "model_config": mcfg.__dict__,
        "train_config": tcfg.__dict__,
        "best_val_loss": best_val_loss,
        "objective": OBJECTIVE_NAME[variant],
        "variant": variant,
        "conditioning_dim": conditioning_dim_val,
        "video_flags": flags,
        "sensor_emb_dir": sensor_emb_dir,
        "sensor_kind": VARIANT_TO_CONFIG[variant]["sensor_kind"],
        "seed": seed,
        "torch_version": torch.__version__,
    }, path)
    print(f"  Saved checkpoint: {path} (step {step})")


def train(args):
    set_deterministic(args.seed)
    rank, local_rank, world_size = setup_ddp()
    is_main = (rank == 0)
    device = torch.device(f"cuda:{local_rank}")

    vocab_size, _, _ = load_token_mapping()
    if is_main:
        print(f"Variant: {args.variant}")
        print(f"Vocab size: {vocab_size}, World size: {world_size}")

    mcfg = ModelConfig(vocab_size=vocab_size, mask_token_id=vocab_size)
    tcfg = TrainConfig()
    if args.max_steps is not None:    tcfg.max_steps = args.max_steps
    if args.warmup_steps is not None: tcfg.warmup_steps = args.warmup_steps
    if args.eval_every is not None:   tcfg.eval_every = args.eval_every
    if args.save_every is not None:   tcfg.save_every = args.save_every

    train_sessions = args.train_sessions or TRAIN_SESSIONS
    val_sessions   = args.val_sessions   or VAL_SESSIONS
    model_dir = Path(args.model_dir) if args.model_dir else MODEL_DIR
    pad_id = mcfg.vocab_size

    vcfg = VARIANT_TO_CONFIG[args.variant]
    flags = _video_flags(args.variant)
    sensor_emb_dir, sensor_dim = SENSOR_KIND_TO_DIR_DIM[vcfg["sensor_kind"]]
    if args.emb_dir is not None:
        sensor_emb_dir = args.emb_dir          # CLI override
    cond_dim = conditioning_dim(sensor_dim, **flags)
    needs_video = flags["use_vjepa"] or flags["use_cotracker"]
    if is_main:
        print(f"  Train sessions: {train_sessions}")
        print(f"  Val sessions:   {val_sessions}")
        print(f"  Sensor encoder: {vcfg['sensor_kind']} (emb_dir={sensor_emb_dir}, dim={sensor_dim})")
        print(f"  Video flags: {flags}; conditioning_dim = {cond_dim}")
        print("Loading datasets...")

    base_train_ds = TraceDataset(
        train_sessions, mcfg.max_seq_len, pad_id,
        emb_dir=Path(sensor_emb_dir), data_dir=Path(args.data_dir),
        limit=args.limit,
    )
    base_val_ds = TraceDataset(
        val_sessions, mcfg.max_seq_len, pad_id,
        emb_dir=Path(sensor_emb_dir), data_dir=Path(args.data_dir),
        limit=args.limit,
    )
    if needs_video:
        train_ds = VideoConditionedTraceDataset(base_train_ds, **flags)
        val_ds   = VideoConditionedTraceDataset(base_val_ds,   **flags)
    else:
        train_ds, val_ds = base_train_ds, base_val_ds

    train_sampler = (DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                        shuffle=True, seed=args.seed)
                     if world_size > 1 else None)
    worker_init = make_worker_init_fn(args.seed)
    train_dl = DataLoader(
        train_ds, batch_size=tcfg.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
        worker_init_fn=worker_init, generator=make_generator(args.seed),
    )
    val_dl = DataLoader(
        val_ds, batch_size=tcfg.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
        worker_init_fn=worker_init,
    )

    model = VideoConditionedARModern(mcfg, conditioning_dim=cond_dim).to(device)
    if is_main:
        print(f"Model parameters: {model.param_count():,}")

    if not args.no_compile:
        if is_main:
            print("Compiling model with torch.compile...")
        model = torch.compile(model)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    muon_params, adam_params = split_muon_adam_params(
        model.module if hasattr(model, "module") else model
    )
    opt_muon = Muon(muon_params, lr=args.muon_lr, momentum=0.95, nesterov=True,
                    weight_decay=tcfg.weight_decay)
    opt_adam = torch.optim.AdamW(adam_params, lr=tcfg.lr, weight_decay=tcfg.weight_decay,
                                 betas=(0.9, 0.999), fused=True)

    raw = model.module if hasattr(model, "module") else model
    ema = EMA(raw.parameters(), decay=tcfg.ema_decay)
    ema.to(device)

    step = 0
    best_val_loss = float("inf")
    opt_muon.zero_grad()
    opt_adam.zero_grad()
    running, t0, epoch = {}, time.time(), 0
    data_iter = iter(train_dl)

    if is_main:
        eff_batch = tcfg.batch_size * tcfg.grad_accum * world_size
        print(f"\nStarting {args.variant.upper()} training...")
        print(f"  Effective batch: {tcfg.batch_size} × {tcfg.grad_accum} × {world_size}gpu = {eff_batch}")
        print(f"  Max steps: {tcfg.max_steps}, Adam LR: {tcfg.lr:.2e}, Muon LR: {args.muon_lr:.2e}")

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
            cond = assemble_conditioning(batch, **flags).to(device, non_blocking=True)
            pad_mask = (x0 != pad_id).float()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=tcfg.bf16):
                if hasattr(model, "module"):
                    loss, metrics = model.module.compute_loss(x0, cond, pad_mask)
                else:
                    loss, metrics = model.compute_loss(x0, cond, pad_mask)
                loss = loss / tcfg.grad_accum
            loss.backward()
            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + v / tcfg.grad_accum

        if tcfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt_muon.step(); opt_adam.step()
        opt_muon.zero_grad(); opt_adam.zero_grad()

        step += 1
        adam_lr = get_lr(step, tcfg.warmup_steps, tcfg.max_steps, tcfg.lr)
        muon_lr = get_lr(step, tcfg.warmup_steps, tcfg.max_steps, args.muon_lr)
        for pg in opt_adam.param_groups: pg["lr"] = adam_lr
        for pg in opt_muon.param_groups: pg["lr"] = muon_lr

        ema.update(raw.parameters())

        if is_main and step % tcfg.log_every == 0:
            elapsed = time.time() - t0
            steps_per_sec = tcfg.log_every / elapsed if elapsed > 0 else 0
            n = tcfg.log_every
            ppl = math.exp(min(running.get("loss", 0) / n, 20))
            print(f"step {step:>7d}/{tcfg.max_steps} | loss {running.get('loss', 0)/n:.4f} | ppl {ppl:.2f} | "
                  f"tf_acc {running.get('acc_teacher_forced', 0)/n:.3f} | "
                  f"adam_lr {adam_lr:.2e} | muon_lr {muon_lr:.2e} | {steps_per_sec:.1f} steps/s")
            running = {}; t0 = time.time()

        if step % tcfg.eval_every == 0:
            ema.store(raw.parameters()); ema.copy_to(raw.parameters())
            val_metrics = video_validate(model, val_dl, device, pad_id, flags)
            val_loss = val_metrics.get("loss", float("inf"))
            if is_main:
                val_ppl = math.exp(min(val_loss, 20))
                print(f"  [VAL] step {step} | loss {val_loss:.4f} | ppl {val_ppl:.2f} | "
                      f"tf_acc {val_metrics.get('acc_teacher_forced', 0):.3f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_video_checkpoint(
                        model_dir / f"{args.variant}_best.pt",
                        step, model, opt_muon, opt_adam, ema, mcfg, tcfg, best_val_loss,
                        variant=args.variant, conditioning_dim_val=cond_dim,
                        flags=flags, sensor_emb_dir=sensor_emb_dir, seed=args.seed,
                    )
            ema.restore(raw.parameters())
            model.train()

        if is_main and step % tcfg.save_every == 0:
            save_video_checkpoint(
                model_dir / f"{args.variant}_step_{step}.pt",
                step, model, opt_muon, opt_adam, ema, mcfg, tcfg, best_val_loss,
                variant=args.variant, conditioning_dim_val=cond_dim,
                flags=flags, sensor_emb_dir=sensor_emb_dir, seed=args.seed,
            )

    if is_main:
        save_video_checkpoint(
            model_dir / f"{args.variant}_final_{step}.pt",
            step, model, opt_muon, opt_adam, ema, mcfg, tcfg, best_val_loss,
            variant=args.variant, conditioning_dim_val=cond_dim,
            flags=flags, sensor_emb_dir=sensor_emb_dir, seed=args.seed,
        )
        print(f"\n{args.variant.upper()} training complete. Best val loss: {best_val_loss:.4f}")
    cleanup_ddp()


def main():
    ap = argparse.ArgumentParser(description="Train BL-3..6b video / MOMENT-conditioned AR-modern")
    ap.add_argument("--variant", required=True, choices=list(VARIANT_TO_CONFIG.keys()))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--emb-dir", default=None,
                    help="Override sensor emb dir (default: per-variant from SENSOR_KIND_TO_DIR_DIM)")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--train-sessions", nargs="+", default=None)
    ap.add_argument("--val-sessions",   nargs="+", default=None)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--warmup-steps", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
