"""BL-3b training entry point — Chronos-2 sensor + LoRA-adapted V-JEPA 2.1.

Mirrors the BL-3 training loop (Muon for 2D AR-decoder block weights, AdamW
for the rest) but adds a third optimizer group for the LoRA adapters
(AdamW). V-JEPA base weights are frozen; only LoRA adapters + projector +
AR decoder train.

Key knobs vs BL-3:
  * batch_size halved (8 default vs 16) — V-JEPA forward dominates memory.
  * grad_accum=2 to keep effective batch the same.
  * grad-checkpointing on V-JEPA blocks.
  * No torch.compile (peft + custom fwd doesn't trace cleanly here).

Usage:
    python3 bl3b_train.py --train-sessions ... --val-sessions ... --model-dir ...
"""
import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import (
    ModelConfig, TrainConfig,
    TRAIN_SESSIONS, VAL_SESSIONS, MODEL_DIR, EMB_DIR, DATA_DIR,
)
from data import TraceDataset, load_token_mapping
from determinism import DEFAULT_SEED, make_generator, make_worker_init_fn, set_deterministic
from mdlm_train import EMA, get_lr, setup_ddp, cleanup_ddp
from ar_modern_train import Muon, split_muon_adam_params
from bl3b_data import BL3bDataset
from bl3b_model import BL3bModel, COND_DIM


def save_bl3b_checkpoint(path, step, model, opt_muon, opt_adam, opt_lora, ema, mcfg, tcfg,
                         best_val_loss, sensor_emb_dir, seed=None):
    raw = model.module if hasattr(model, "module") else model
    path.parent.mkdir(parents=True, exist_ok=True)
    # Save AR weights, LoRA adapter weights (compact), and sensor proj.
    # The base V-JEPA is frozen; we don't save it (it's loaded from hub at eval).
    torch.save({
        "step": step,
        "ar_state":   raw.ar.state_dict(),
        "lora_state": {k: v for k, v in raw.vjepa.state_dict().items() if "lora" in k.lower()},
        "opt_muon": opt_muon.state_dict(),
        "opt_adam": opt_adam.state_dict(),
        "opt_lora": opt_lora.state_dict(),
        "ema": ema.state_dict(),
        "model_config": mcfg.__dict__,
        "train_config": tcfg.__dict__,
        "best_val_loss": best_val_loss,
        "objective": "ar_modern_video_bl3b",
        "variant": "bl3b",
        "conditioning_dim": COND_DIM,
        "video_flags": {"use_vjepa": True, "use_cotracker": False, "lora_vjepa": True},
        "sensor_emb_dir": sensor_emb_dir,
        "sensor_kind": "chronos2",
        "seed": seed,
        "torch_version": torch.__version__,
    }, path)
    print(f"  Saved checkpoint: {path} (step {step})")


@torch.no_grad()
def validate(model, val_dl, device, pad_id, max_batches=30):
    raw = model.module if hasattr(model, "module") else model
    raw.eval()
    totals, n = {}, 0
    for i, batch in enumerate(val_dl):
        if i >= max_batches:
            break
        x0 = batch["trace"].to(device)
        emb = batch["sensor_emb"].to(device)
        clip = batch["clip"].to(device)
        pad_mask = (x0 != pad_id).float()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, metrics = raw.compute_loss(x0, emb, clip, pad_mask)
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1
    raw.train()
    return {k: v / max(n, 1) for k, v in totals.items()}


def main():
    ap = argparse.ArgumentParser(description="Train BL-3b (LoRA V-JEPA + Chronos-2 sensor)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--emb-dir", default=str(EMB_DIR))
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--train-sessions", nargs="+", default=None)
    ap.add_argument("--val-sessions",   nargs="+", default=None)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=8)        # halved vs BL-3
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--lora-lr", type=float, default=1e-4)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    set_deterministic(args.seed)
    rank, local_rank, world_size = setup_ddp()
    is_main = (rank == 0)
    device = torch.device(f"cuda:{local_rank}")

    vocab_size, _, _ = load_token_mapping()
    mcfg = ModelConfig(vocab_size=vocab_size, mask_token_id=vocab_size)
    tcfg = TrainConfig()
    tcfg.batch_size = args.batch_size
    tcfg.grad_accum = args.grad_accum
    tcfg.max_steps = args.max_steps
    tcfg.warmup_steps = args.warmup_steps
    tcfg.eval_every = args.eval_every
    tcfg.save_every = args.save_every

    train_sessions = args.train_sessions or TRAIN_SESSIONS
    val_sessions   = args.val_sessions   or VAL_SESSIONS
    model_dir = Path(args.model_dir) if args.model_dir else MODEL_DIR
    pad_id = mcfg.vocab_size

    if is_main:
        print(f"BL-3b: LoRA V-JEPA + Chronos-2")
        print(f"  Train sessions: {train_sessions}")
        print(f"  Val sessions:   {val_sessions}")
        print(f"  conditioning_dim = {COND_DIM} (sensor 512 + v-jepa 1024)")
        print(f"  batch={tcfg.batch_size}, grad_accum={tcfg.grad_accum}")
        print("Loading datasets...")

    base_train_ds = TraceDataset(train_sessions, mcfg.max_seq_len, pad_id,
                                 emb_dir=Path(args.emb_dir), data_dir=Path(args.data_dir),
                                 limit=args.limit)
    base_val_ds   = TraceDataset(val_sessions,   mcfg.max_seq_len, pad_id,
                                 emb_dir=Path(args.emb_dir), data_dir=Path(args.data_dir),
                                 limit=args.limit)
    train_ds = BL3bDataset(base_train_ds)
    val_ds   = BL3bDataset(base_val_ds)

    worker_init = make_worker_init_fn(args.seed)
    train_dl = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True,
                          persistent_workers=args.num_workers > 0,
                          worker_init_fn=worker_init,
                          generator=make_generator(args.seed))
    val_dl = DataLoader(val_ds, batch_size=tcfg.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0,
                        worker_init_fn=worker_init)

    model = BL3bModel(mcfg).to(device)
    train_n, total_n = model.param_count()
    if is_main:
        print(f"Model: {train_n/1e6:.1f}M trainable / {total_n/1e6:.1f}M total")

    # Optimizers:
    #   Muon  -> 2D weights inside AR-modern decoder blocks
    #   AdamW -> rest of AR-modern (embeddings, lm_head, RMSNorm, sensor_proj)
    #   AdamW -> LoRA adapter params (peft tags them as `lora_A`/`lora_B`)
    muon_params, ar_adam_params = split_muon_adam_params(model.ar)
    lora_params = [p for n, p in model.vjepa.named_parameters()
                   if p.requires_grad and "lora" in n.lower()]
    if is_main:
        print(f"  Muon  AR params:    {len(muon_params)} tensors, "
              f"{sum(p.numel() for p in muon_params):,} elems")
        print(f"  AdamW AR params:    {len(ar_adam_params)} tensors, "
              f"{sum(p.numel() for p in ar_adam_params):,} elems")
        print(f"  AdamW LoRA params:  {len(lora_params)} tensors, "
              f"{sum(p.numel() for p in lora_params):,} elems")

    opt_muon = Muon(muon_params, lr=args.muon_lr, momentum=0.95, nesterov=True,
                    weight_decay=tcfg.weight_decay)
    opt_adam = torch.optim.AdamW(ar_adam_params, lr=tcfg.lr, weight_decay=tcfg.weight_decay,
                                 betas=(0.9, 0.999), fused=True)
    opt_lora = torch.optim.AdamW(lora_params, lr=args.lora_lr, weight_decay=0.0,
                                 betas=(0.9, 0.999), fused=True)

    raw = model
    ema = EMA(raw.parameters(), decay=tcfg.ema_decay)
    ema.to(device)

    step = 0
    best_val_loss = float("inf")
    opt_muon.zero_grad(); opt_adam.zero_grad(); opt_lora.zero_grad()
    running, t0, epoch = {}, time.time(), 0
    data_iter = iter(train_dl)

    if is_main:
        eff_batch = tcfg.batch_size * tcfg.grad_accum
        print(f"\nStarting BL-3b training. Effective batch: {eff_batch}, max_steps: {tcfg.max_steps}")

    while step < tcfg.max_steps:
        for _ in range(tcfg.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                data_iter = iter(train_dl)
                batch = next(data_iter)
            x0   = batch["trace"].to(device, non_blocking=True)
            emb  = batch["sensor_emb"].to(device, non_blocking=True)
            clip = batch["clip"].to(device, non_blocking=True)
            pad_mask = (x0 != pad_id).float()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, metrics = model.compute_loss(x0, emb, clip, pad_mask)
                loss = loss / tcfg.grad_accum
            loss.backward()
            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + v / tcfg.grad_accum

        if tcfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt_muon.step(); opt_adam.step(); opt_lora.step()
        opt_muon.zero_grad(); opt_adam.zero_grad(); opt_lora.zero_grad()

        step += 1
        adam_lr = get_lr(step, tcfg.warmup_steps, tcfg.max_steps, tcfg.lr)
        muon_lr = get_lr(step, tcfg.warmup_steps, tcfg.max_steps, args.muon_lr)
        lora_lr = get_lr(step, tcfg.warmup_steps, tcfg.max_steps, args.lora_lr)
        for pg in opt_adam.param_groups: pg["lr"] = adam_lr
        for pg in opt_muon.param_groups: pg["lr"] = muon_lr
        for pg in opt_lora.param_groups: pg["lr"] = lora_lr

        ema.update(raw.parameters())

        if is_main and step % tcfg.log_every == 0:
            elapsed = time.time() - t0
            sps = tcfg.log_every / elapsed if elapsed > 0 else 0
            n = tcfg.log_every
            ppl = math.exp(min(running.get("loss", 0)/n, 20))
            print(f"step {step:>6d}/{tcfg.max_steps} | loss {running.get('loss',0)/n:.4f} | ppl {ppl:.2f} | "
                  f"tf_acc {running.get('acc_teacher_forced',0)/n:.3f} | "
                  f"lora_lr {lora_lr:.2e} | {sps:.2f} steps/s")
            running = {}; t0 = time.time()

        if step % tcfg.eval_every == 0:
            ema.store(raw.parameters()); ema.copy_to(raw.parameters())
            val_metrics = validate(model, val_dl, device, pad_id)
            val_loss = val_metrics.get("loss", float("inf"))
            if is_main:
                val_ppl = math.exp(min(val_loss, 20))
                print(f"  [VAL] step {step} | loss {val_loss:.4f} | ppl {val_ppl:.2f} | "
                      f"tf_acc {val_metrics.get('acc_teacher_forced', 0):.3f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_bl3b_checkpoint(model_dir / "bl3b_best.pt", step, model,
                                         opt_muon, opt_adam, opt_lora, ema, mcfg, tcfg,
                                         best_val_loss, sensor_emb_dir=str(args.emb_dir),
                                         seed=args.seed)
            ema.restore(raw.parameters()); model.train()

        if is_main and step % tcfg.save_every == 0:
            save_bl3b_checkpoint(model_dir / f"bl3b_step_{step}.pt", step, model,
                                 opt_muon, opt_adam, opt_lora, ema, mcfg, tcfg,
                                 best_val_loss, sensor_emb_dir=str(args.emb_dir),
                                 seed=args.seed)

    if is_main:
        save_bl3b_checkpoint(model_dir / f"bl3b_final_{step}.pt", step, model,
                             opt_muon, opt_adam, opt_lora, ema, mcfg, tcfg,
                             best_val_loss, sensor_emb_dir=str(args.emb_dir),
                             seed=args.seed)
        print(f"\nBL-3b training complete. Best val loss: {best_val_loss:.4f}")
    cleanup_ddp()


if __name__ == "__main__":
    main()
