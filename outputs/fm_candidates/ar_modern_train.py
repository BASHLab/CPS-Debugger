"""AR-modern training loop.

Mirrors ar_train.py exactly, but:
  - Trains ARModernModel (RMSNorm + SwiGLU + QK-Norm)
  - Uses Muon optimizer for hidden 2D weights (attention + FFN projection matrices)
    and AdamW for everything else (embeddings, LM head, norm scales, biases).

Muon (Keller Jordan, https://kellerjordan.github.io/posts/muon/) — MIT licensed —
applies SGD-momentum then orthogonalizes the update via Newton-Schulz iteration.
At our scale (~26-50M params) it's expected to give ~1.3-1.5× sample efficiency
over AdamW on 2D weights, per the modded-nanogpt speedrun records.

Convention (per Keller's blog and confirmed in modded-nanogpt):
  - Muon → 2D weights inside transformer blocks (qkv, out_proj, gate, up, down)
  - AdamW → embeddings, LM head, RMSNorm scales, any 1D params

Usage:
    python3 ar_modern_train.py --device cuda
    python3 ar_modern_train.py --device cuda --sanity-check --limit 100
"""
import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from config import (
    ModelConfig, TrainConfig,
    TRAIN_SESSIONS, VAL_SESSIONS, MODEL_DIR, EMB_DIR, DATA_DIR,
)
from data import TraceDataset, JointTraceWindowDataset, load_token_mapping
from ar_modern_model import ARModernModel
from ar_modern_cross_model import ARModernCrossModel
from ar_modern_state_model import ARModernStateModel
from ar_modern_factored_model import ARModernFactoredModel
from determinism import DEFAULT_SEED, make_generator, make_worker_init_fn, set_deterministic
from mdlm_train import EMA, get_lr, setup_ddp, cleanup_ddp


# ── Muon optimizer ───────────────────────────────────────────────────────
def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute zero-power (orthogonalization) of G.

    Quintic iteration: X <- a*X + b*(X X^T) X + c*(X X^T)^2 X
    Coefficients tuned by Keller Jordan to maximize the slope at 0 while
    keeping the iteration stable. Operates on (m, n) matrices, m >= n.
    Returns an orthogonal matrix U V^T (the polar factor of G).
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if X.size(0) > X.size(1):
        X = X.T
    # Normalize spectral norm to <= 1 so Newton-Schulz is stable
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B_ = b * A + c * (A @ A)
        X = a * X + B_ @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """MomentUm Orthogonalized by Newton-schulz.

    Reference: https://kellerjordan.github.io/posts/muon/

    Apply ONLY to 2D hidden weights (attention + MLP projections). Use AdamW
    for embeddings, LM head, biases, and norm scales.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, weight_decay: float = 0.0, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                assert g.ndim == 2, f"Muon expects 2D weights, got {g.shape}"

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)
                update = g.add(buf, alpha=mom) if nesterov else buf

                # Decoupled weight decay
                if wd != 0:
                    p.data.mul_(1 - lr * wd)

                # Orthogonalize the update
                ortho = _zeropower_via_newtonschulz5(update, steps=ns_steps)
                # LR scale: max(d_in, d_out) / d_in  (Keller's recommended scaling)
                # Equivalent to scaling by sqrt(max/min) — see modded-nanogpt
                scale = max(1.0, ortho.size(0) / ortho.size(1)) ** 0.5
                p.data.add_(ortho, alpha=-lr * scale)

        return loss


# ── Param partitioning ───────────────────────────────────────────────────
def split_muon_adam_params(model: torch.nn.Module):
    """Return (muon_params, adam_params) by Keller's convention.

    Muon: 2D weights inside transformer blocks (qkv, out_proj, gate, up, down).
    Adam: embeddings, LM head (out_proj at top level), sensor_proj, norm scales,
          1D biases — anything that is not a hidden 2D weight.

    Param-name shapes recognised:
      ARModernModel:        backbone.blocks.<i>.<...>.weight
      ARModernCrossModel:   backbone.cross_blocks.<i>.<...>.weight
                            backbone.self_blocks.<i>.<...>.weight
    Substrings ".blocks.", ".cross_blocks.", ".self_blocks." all qualify.
    """
    block_substrings = (".blocks.", ".cross_blocks.", ".self_blocks.")
    muon, adam = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        in_block = any(s in name for s in block_substrings)
        # 2D weights *inside* blocks → Muon. Norm scales (RMSNorm.weight) are
        # 1D and skip this check.
        if in_block and p.ndim == 2 and "weight" in name:
            muon.append(p)
        else:
            adam.append(p)
    return muon, adam


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
        # Joint mode batches carry sensor_window; precomputed-emb mode carries sensor_emb.
        emb = batch["sensor_window"].to(device) if "sensor_window" in batch \
              else batch["sensor_emb"].to(device)
        pad_mask = (x0 != pad_id).float()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # E5 state-token mode: compute_loss needs state_id argument.
            # Gate on the model's expected signature, not on dataset presence;
            # otherwise stray state_pred files in the emb dir crash other models.
            if isinstance(raw, ARModernStateModel) and "state_pred" in batch:
                state_id = batch["state_pred"].to(device).long()
                loss, metrics = raw.compute_loss(x0, emb, pad_mask, state_id)
            else:
                loss, metrics = raw.compute_loss(x0, emb, pad_mask)

        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1

    raw.train()
    return {k: v / max(n, 1) for k, v in totals.items()}


# ── Checkpointing ────────────────────────────────────────────────────────
def save_checkpoint(path, step, model, opt_muon, opt_adam, ema, mcfg, tcfg, best_val_loss,
                    seed=None, use_cross_attn=False, use_state_token=False,
                    use_factored_head=False):
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
        "objective": "autoregressive_modern",
        "use_cross_attn": use_cross_attn,
        "use_state_token": use_state_token,
        "use_factored_head": use_factored_head,
        "seed": seed,
        "torch_version": torch.__version__,
    }, path)
    print(f"  Saved checkpoint: {path} (step {step})")


def load_checkpoint(path, model, opt_muon, opt_adam, ema, device):
    raw = model.module if hasattr(model, "module") else model
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw.load_state_dict(ckpt["model"])
    opt_muon.load_state_dict(ckpt["opt_muon"])
    opt_adam.load_state_dict(ckpt["opt_adam"])
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

    mcfg = ModelConfig(vocab_size=vocab_size, mask_token_id=vocab_size,
                       d_sensor_emb=args.d_sensor_emb,
                       n_sensor_tokens=args.n_sensor_tokens,
                       cond_dropout_p=args.cond_dropout_p)
    tcfg = TrainConfig()

    # Apply CLI overrides
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

    pad_id = mcfg.vocab_size

    if is_main:
        print(f"Train sessions: {train_sessions}")
        print(f"Val sessions:   {val_sessions}")
        print("Loading datasets...")

    # E2 K-sweep joint mode: load raw sensor windows (encode on the fly)
    # instead of precomputed embeddings. Triggered by --bl7-encoder-ckpt.
    joint_mode = bool(args.bl7_encoder_ckpt)
    if joint_mode:
        train_ds = JointTraceWindowDataset(
            train_sessions, mcfg.max_seq_len, pad_id,
            window_size=args.bl7_window_size,
            data_dir=Path(args.data_dir),
            limit=args.limit,
        )
        val_ds = JointTraceWindowDataset(
            val_sessions, mcfg.max_seq_len, pad_id,
            window_size=args.bl7_window_size,
            data_dir=Path(args.data_dir),
            limit=args.limit,
        )
    else:
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

    if args.state_balanced and world_size == 1:
        from torch.utils.data import WeightedRandomSampler
        fsm = np.asarray(train_ds.fsm_states, dtype=np.int64)
        counts = np.bincount(fsm, minlength=3).astype(np.float64)
        counts = np.clip(counts, 1.0, None)
        weights = 1.0 / counts[fsm]
        train_sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights).double(),
            num_samples=len(fsm),
            replacement=True,
            generator=make_generator(args.seed),
        )
        if is_main:
            cnt0, cnt1, cnt2 = (int((fsm == s).sum()) for s in (0, 1, 2))
            print(f"  state-balanced sampler: counts (0/1/2) = {cnt0}/{cnt1}/{cnt2}; "
                  f"each batch will be ~1/3 from each state in expectation")
    elif args.state_balanced and world_size > 1:
        raise RuntimeError("--state-balanced requires single-GPU (world_size=1); "
                           "multi-GPU weighted sampling is not implemented here.")

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

    if args.use_state_token:
        ar_model = ARModernStateModel(mcfg, cond_dropout_p=args.cond_dropout_p).to(device)
        if is_main:
            print(f"Using ARModernStateModel (state-token prefix + CFG)")
    elif args.use_cross_attn:
        ar_model = ARModernCrossModel(mcfg).to(device)
        if is_main:
            print(f"Using ARModernCrossModel (Flamingo cross-attention decoder)")
    elif args.use_factored_head:
        ar_model = ARModernFactoredModel(mcfg).to(device)
        if is_main:
            print(f"Using ARModernFactoredModel (factored opcode/op1/op2 heads)")
    else:
        ar_model = ARModernModel(mcfg).to(device)
        if is_main:
            print(f"Using ARModernModel (prefix-token decoder)")

    # E2 K-sweep joint mode: wrap the AR model with a frozen-backbone BL-7
    # encoder. The pool head trains with the AR decoder; backbone stays frozen.
    if joint_mode:
        from bl7_ksweep import build_ksweep_encoder, JointEncoderARModel
        K_arg = None if args.bl7_axial else args.bl7_K
        bl7_enc = build_ksweep_encoder(args.bl7_encoder_ckpt, K=K_arg, device=device)
        model = JointEncoderARModel(bl7_enc, ar_model, axial_mode=args.bl7_axial)
        if is_main:
            trainable_enc = sum(p.numel() for p in bl7_enc.parameters() if p.requires_grad)
            frozen_enc = sum(p.numel() for p in bl7_enc.parameters() if not p.requires_grad)
            print(f"Joint mode: BL-7 encoder loaded from {args.bl7_encoder_ckpt}")
            print(f"  Encoder trainable params: {trainable_enc:,} (pool only)")
            print(f"  Encoder frozen   params: {frozen_enc:,} (conv + transformer + out_norm)")
            print(f"  K = {K_arg if K_arg is not None else 'axial (no Perceiver)'}")
    else:
        model = ar_model

    if is_main:
        print(f"Model parameters (total trainable): "
              f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    if tcfg.grad_checkpoint:
        # ARModernModel: backbone.blocks (causal self-attn).
        # ARModernCrossModel: backbone.cross_blocks (cross-attn) + backbone.self_blocks (causal).
        backbone = model.backbone
        block_lists = []
        if hasattr(backbone, "blocks"):
            block_lists.append(backbone.blocks)
        if hasattr(backbone, "self_blocks"):
            block_lists.append(backbone.self_blocks)
        if hasattr(backbone, "cross_blocks"):
            block_lists.append(backbone.cross_blocks)
        for blocks in block_lists:
            for block in blocks:
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

    # ── Two-optimizer setup: Muon for 2D block weights, AdamW for the rest ──
    muon_params, adam_params = split_muon_adam_params(
        model.module if hasattr(model, "module") else model
    )
    n_muon = sum(p.numel() for p in muon_params)
    n_adam = sum(p.numel() for p in adam_params)
    if is_main:
        print(f"  Muon  params: {len(muon_params):>4d} tensors, {n_muon:>11,} elems")
        print(f"  AdamW params: {len(adam_params):>4d} tensors, {n_adam:>11,} elems")

    # Muon LR is typically ~10× larger than Adam LR per Keller's recipe.
    # We use 0.02 (Keller default) vs Adam tcfg.lr=3e-4.
    opt_muon = Muon(
        muon_params, lr=args.muon_lr, momentum=0.95, nesterov=True,
        weight_decay=tcfg.weight_decay,
    )
    opt_adam = torch.optim.AdamW(
        adam_params, lr=tcfg.lr, weight_decay=tcfg.weight_decay,
        betas=(0.9, 0.999), fused=True,
    )

    raw = model.module if hasattr(model, "module") else model
    ema = EMA(raw.parameters(), decay=tcfg.ema_decay)
    ema.to(device)

    step = 0
    best_val_loss = float("inf")
    if args.resume:
        step, best_val_loss = load_checkpoint(
            Path(args.resume), model, opt_muon, opt_adam, ema, device
        )

    model.train()
    opt_muon.zero_grad()
    opt_adam.zero_grad()
    running, t0, epoch = {}, time.time(), 0
    data_iter = iter(train_dl)

    if is_main:
        eff_batch = tcfg.batch_size * tcfg.grad_accum * world_size
        print(f"\nStarting AR-modern training from step {step}...")
        print(f"  Effective batch: {tcfg.batch_size} × {tcfg.grad_accum} × {world_size}gpu = {eff_batch}")
        print(f"  Max steps: {tcfg.max_steps}")
        print(f"  Adam LR: {tcfg.lr:.2e} | Muon LR: {args.muon_lr:.2e}")
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
            # Joint mode batches carry sensor_window; precomputed-emb mode carries sensor_emb.
            emb = batch["sensor_window"].to(device, non_blocking=True) \
                  if "sensor_window" in batch \
                  else batch["sensor_emb"].to(device, non_blocking=True)
            pad_mask = (x0 != pad_id).float()
            # E5 state-token mode: also pass state_id (gated on model type).
            raw_model = model.module if hasattr(model, "module") else model
            state_id = (batch["state_pred"].to(device, non_blocking=True).long()
                        if (isinstance(raw_model, ARModernStateModel)
                            and "state_pred" in batch) else None)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=tcfg.bf16):
                if state_id is not None:
                    loss, metrics = raw_model.compute_loss(x0, emb, pad_mask, state_id)
                else:
                    loss, metrics = raw_model.compute_loss(x0, emb, pad_mask)
                loss = loss / tcfg.grad_accum

            loss.backward()

            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + v / tcfg.grad_accum

        if tcfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt_muon.step()
        opt_adam.step()
        opt_muon.zero_grad()
        opt_adam.zero_grad()

        step += 1
        # LR schedule applied to both optimizers proportionally
        adam_lr = get_lr(step, tcfg.warmup_steps, tcfg.max_steps, tcfg.lr)
        muon_lr = get_lr(step, tcfg.warmup_steps, tcfg.max_steps, args.muon_lr)
        for pg in opt_adam.param_groups:
            pg["lr"] = adam_lr
        for pg in opt_muon.param_groups:
            pg["lr"] = muon_lr

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
                f"adam_lr {adam_lr:.2e} | muon_lr {muon_lr:.2e} | "
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
                    save_checkpoint(model_dir / "ar_modern_best.pt", step, model,
                                    opt_muon, opt_adam, ema, mcfg, tcfg, best_val_loss,
                                    seed=args.seed, use_cross_attn=args.use_cross_attn, use_state_token=args.use_state_token, use_factored_head=args.use_factored_head)

            ema.restore(raw.parameters())
            model.train()

        if is_main and step % tcfg.save_every == 0:
            save_checkpoint(model_dir / f"ar_modern_step_{step}.pt", step, model,
                            opt_muon, opt_adam, ema, mcfg, tcfg, best_val_loss,
                            seed=args.seed, use_cross_attn=args.use_cross_attn, use_state_token=args.use_state_token, use_factored_head=args.use_factored_head)

    if is_main:
        save_checkpoint(model_dir / f"ar_modern_final_{step}.pt", step, model,
                        opt_muon, opt_adam, ema, mcfg, tcfg, best_val_loss,
                        seed=args.seed, use_cross_attn=args.use_cross_attn, use_state_token=args.use_state_token, use_factored_head=args.use_factored_head)
        print(f"\nAR-modern training complete. Best val loss: {best_val_loss:.4f}")

    cleanup_ddp()


def main():
    ap = argparse.ArgumentParser(description="BL-2 AR-modern training (RMSNorm+SwiGLU+QK-Norm+Muon)")
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
    ap.add_argument("--muon-lr", type=float, default=0.02,
                    help="Muon LR (Keller default 0.02 ≈ 60× Adam's 3e-4)")
    # TrainConfig overrides (AR converges in ~3K steps on 2-session data)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--warmup-steps", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="Global seed for determinism (RNGs + CUBLAS + cuDNN)")
    ap.add_argument("--d-sensor-emb", type=int, default=512,
                    help="Encoder output dim (Chronos-2/MOMENT: 512, BL-7: 256). "
                         "Must match the precomputed embeddings under --emb-dir.")
    ap.add_argument("--use-cross-attn", action="store_true",
                    help="Use ARModernCrossModel (Flamingo dense cross-attention "
                         "decoder, Tier 1A.1) instead of ARModernModel (prefix token).")
    ap.add_argument("--n-sensor-tokens", type=int, default=1,
                    help="Number of dense sensor tokens. Prefix decoder: 1 "
                         "(handles 3D via mean-pool internally). Cross-attn "
                         "decoder + BL-7: 4.")
    ap.add_argument("--cond-dropout-p", type=float, default=0.10,
                    help="Conditioning dropout prob for cross-attn decoder "
                         "(CFG prereq). Ignored for prefix decoder.")
    # E2 K-sweep joint training mode: load a frozen BL-7 encoder and
    # encode raw sensor windows on the fly. Pool head trains with the AR.
    ap.add_argument("--bl7-encoder-ckpt", default=None,
                    help="Path to BL-7 v1 encoder checkpoint. If set, enter "
                         "joint training mode: frozen backbone + trainable "
                         "pool + AR-modern. Uses JointTraceWindowDataset.")
    ap.add_argument("--bl7-K", type=int, default=4,
                    help="Number of Perceiver-pool query tokens (E2 K-sweep). "
                         "Ignored if --bl7-axial is set.")
    ap.add_argument("--bl7-axial", action="store_true",
                    help="No-Perceiver mode: use raw (B, 154, 512) axial "
                         "tokens. Requires --use-cross-attn (prefix decoder "
                         "can't take 154-token prefix). E2 K=none variant.")
    ap.add_argument("--bl7-window-size", type=int, default=500,
                    help="Sensor-window size for the joint encoder. Default 500.")
    # E5 state-token + CFG mode
    ap.add_argument("--use-state-token", action="store_true",
                    help="Use ARModernStateModel — adds a state-token prefix "
                         "to the AR decoder, sourced from per-tick state_pred "
                         "files (created by `bl7_probes.py dump-state-predictions`). "
                         "Enables CFG sampling at eval (guidance_scale > 1.0).")

    # State-balanced training-batch sampler (across pendulum_state)
    ap.add_argument("--state-balanced", action="store_true",
                    help="Sample training batches with class-balanced weights "
                         "across pendulum_state (0/1/2). Uses WeightedRandomSampler; "
                         "single-GPU only. Does not change the encoder.")

    # Factored opcode/operand decoder head
    ap.add_argument("--use-factored-head", action="store_true",
                    help="Use ARModernFactoredModel — replaces the joint 648-way "
                         "softmax with three conditional heads "
                         "(opcode, op1 | opcode, op2 | opcode, op1) with "
                         "legal-triple masking at sampling time. Loss is the "
                         "sum of three CEs.")
    args = ap.parse_args()
    set_deterministic(args.seed)
    train(args)


if __name__ == "__main__":
    main()
