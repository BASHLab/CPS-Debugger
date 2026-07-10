"""BL-7 JEPA pretraining loop (Tier 1B of the BL-7 plan).

data2vec / V-JEPA-style self-supervised pretraining on sensor windows:
  - student encoder processes (B, 7, 500) → per-position (B, 154, 512) activations
  - EMA teacher (no_grad) encodes the same input → averaged top-K layer activations
  - block-span masking (30–40 % ratio) over the 154-token sequence
  - predictor takes the student's full output with mask-tokens at masked positions
    and predicts the teacher's contextualised hidden state at masked positions
  - L1 (smooth-L1) loss on masked positions only
  - EMA τ annealed 0.999 → 0.9999 over the first 30 % of training

Tier 1B uses **only** the JEPA loss. Tier 2 will add a class-balanced
contrastive auxiliary; the dataset already exposes per-tick state for that.

Usage:
    python3 bl7_pretrain.py --device cuda
    python3 bl7_pretrain.py --device cuda --sanity-check
"""
import argparse
import math
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import (
    BL7EncoderConfig, BL7TrainConfig, MODEL_DIR,
    TRAIN_SESSIONS, VAL_SESSIONS,
)
from bl7_data import SensorWindowDataset, make_worker_init_fn
from bl7_model import BL7Encoder, JEPAPredictor, KoopmanPredictor, EMATeacher, HuBERTHead
from determinism import DEFAULT_SEED, set_deterministic


# ── Mask sampler ─────────────────────────────────────────────────────────
def sample_block_span_mask(
    B: int, L: int, ratio_lo: float, ratio_hi: float,
    span_min: int, span_max: int,
    device: torch.device,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """Sample a block-span mask for a (B, L) token sequence.

    For each sample: lay down spans of random lengths in [span_min, span_max]
    starting at random positions until the masked fraction is in
    [ratio_lo, ratio_hi]. Spans may overlap.

    Returns a bool tensor of shape (B, L); True where masked.
    """
    mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    target_min = int(ratio_lo * L)
    target_max = int(ratio_hi * L)
    for b in range(B):
        attempts = 0
        while mask[b].sum().item() < target_min and attempts < 200:
            span = torch.randint(span_min, span_max + 1, (1,), generator=generator).item()
            start = torch.randint(0, max(1, L - span + 1), (1,), generator=generator).item()
            mask[b, start:start + span] = True
            attempts += 1
            if mask[b].sum().item() > target_max:
                break
    return mask


# ── LR schedule (tri-stage 3/90/7) ───────────────────────────────────────
def tri_stage_lr(step: int, max_steps: int, peak_lr: float,
                 warmup_pct: float = 0.03, flat_pct: float = 0.90) -> float:
    """Linear warmup → flat at peak → linear decay to 0.01·peak."""
    warm = int(warmup_pct * max_steps)
    flat_end = int((warmup_pct + flat_pct) * max_steps)
    if step < warm:
        return peak_lr * (step + 1) / max(1, warm)
    if step < flat_end:
        return peak_lr
    # decay
    progress = (step - flat_end) / max(1, max_steps - flat_end)
    return peak_lr * (1.0 - 0.99 * min(progress, 1.0))


def ema_tau_at(step: int, max_steps: int, tau_start: float, tau_end: float,
               anneal_pct: float) -> float:
    anneal_end = int(anneal_pct * max_steps)
    if step >= anneal_end:
        return tau_end
    p = step / max(1, anneal_end)
    return tau_start + (tau_end - tau_start) * p


# ── Time-reversal symmetry aux ────────────────────────────────────────────
def time_reversal_loss(student: BL7Encoder, x: torch.Tensor) -> torch.Tensor:
    """Encourages encoder(forward) ≈ time-reversed encoder(time-reversed input).

    Closed-loop control signals minus active forcing are partially
    time-reversal-invariant: a pendulum's free oscillation is symmetric
    under t → -t. The LQR control input and friction break exact
    reversibility, but the encoder *features* should still place
    forward-time and reversed-time windows of the same physical event in
    related parts of the latent space.

    Methodology: compare the encoder's full per-position output on x and
    on flip(x), with one of them time-reversed before comparison. We use
    cosine similarity averaged over channels and tokens, then take 1 - sim
    so smaller is better.

    This is a Tier-9 ablation auxiliary; no published precedent in TS SSL.
    Kempner-flavored, derived from physical symmetry of damped oscillators.
    """
    # x: (B, C=7, T=500)
    x_rev = torch.flip(x, dims=[2])

    # Run encoder on both forward and reversed inputs.
    out_fwd, _ = student.encode_full(x)         # (B, C, T_patches, d_model)
    out_rev, _ = student.encode_full(x_rev)

    # Reverse the time-axis of out_rev so it aligns with out_fwd's time-axis.
    out_rev_aligned = torch.flip(out_rev, dims=[2])

    # Cosine similarity per position; minimise (1 - sim) ∈ [0, 2].
    sim = F.cosine_similarity(out_fwd, out_rev_aligned, dim=-1)  # (B, C, T_patches)
    return (1.0 - sim).mean()


# ── E3: prototype-based SupCon with natural-rate sampling ────────────────
class ProtoSupCon:
    """Prototype-based supervised contrastive loss for E3 (BL-8a).

    Fixes the failure mode of the Tier-2 contrastive (BL-7 B/C): that one used
    33/33/33 state-balanced *resampling*, which distorted the encoder's view
    of natural dynamics and made downstream NED 48-65% worse. E3 instead:
      - samples at the NATURAL state rate (no resampling distortion),
      - maintains a per-class running prototype (EMA, momentum 0.999),
      - pulls each sample toward its class prototype with a per-class
        temperature (warmer for the rare state-2 to counter small-batch noise).

    State labels are used only to route samples to prototypes — never as a
    prediction target, so the encoder stays trace-blind.

    Usage: construct once before the train loop; call .loss(dense, states)
    each step (it updates prototypes in-place under no_grad).
    """

    def __init__(self, d_emb: int, device: torch.device,
                 momentum: float = 0.999,
                 temps: tuple = (0.07, 0.07, 0.20)):  # τ for states 0,1,2
        self.momentum = momentum
        self.temps = torch.tensor(temps, device=device)
        # L2-normalised prototypes, one per FSM state. Lazy-init on first batch.
        self.prototypes = torch.zeros(3, d_emb, device=device)
        self._initialised = False

    @torch.no_grad()
    def _update_prototypes(self, z: torch.Tensor, states: torch.Tensor):
        # Build a FRESH tensor (no in-place [c]= writes) — in-place mutation of
        # self.prototypes corrupts the autograd version counter of any forward
        # pass that referenced it. Rebinding to a new tensor is autograd-safe.
        new_protos = self.prototypes.clone()
        for c in range(3):
            mask = (states == c)
            if not mask.any():
                continue
            class_mean = F.normalize(z[mask].mean(dim=0), dim=-1)
            if not self._initialised:
                new_protos[c] = class_mean
            else:
                p = self.momentum * self.prototypes[c] + (1 - self.momentum) * class_mean
                new_protos[c] = F.normalize(p, dim=-1)
        self.prototypes = new_protos
        self._initialised = True

    def loss(self, dense_tokens: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """dense_tokens: (B, K, d_emb); states: (B,) int in {0,1,2}."""
        B, K, D = dense_tokens.shape
        device = dense_tokens.device
        rand_k = torch.randint(0, K, (B,), device=device)
        z = F.normalize(dense_tokens[torch.arange(B, device=device), rand_k], dim=-1)

        # Warm up prototypes on the very first call so the first loss is sane.
        if not self._initialised:
            self._update_prototypes(z.detach(), states)

        # Per-sample temperature picked by that sample's state.
        tau = self.temps[states]                             # (B,)
        # Prototypes never receive gradient — use a detached snapshot in the
        # forward so backward doesn't try to traverse the EMA-update history.
        protos = self.prototypes.detach()
        sims = (z @ protos.T) / tau.unsqueeze(1)              # (B, 3)
        log_prob = F.log_softmax(sims, dim=1)                 # (B, 3)
        # Pull toward own-class prototype
        loss = F.nll_loss(log_prob, states)

        # EMA-update prototypes AFTER computing the loss (no grad through update)
        self._update_prototypes(z.detach(), states)
        return loss

    def state_dict(self):
        return {"prototypes": self.prototypes.clone(),
                "initialised": self._initialised}

    def load_state_dict(self, sd):
        self.prototypes = sd["prototypes"].clone()
        self._initialised = sd["initialised"]


# ── Class-balanced SupCon-style contrastive on random-1-of-K dense tokens ─
def supcon_contrastive_loss(dense_tokens: torch.Tensor, states: torch.Tensor,
                            temperature: float = 0.10) -> torch.Tensor:
    """SupCon (Khosla et al. NeurIPS 2020) over per-tick dense BL-7 tokens.

    For each anchor i, pick a random index k_i ∈ {0,...,K-1} (random-per-anchor
    per the BL-7 plan §4.3 — prevents the model from pushing class-info into
    a single fixed token slot). Use dense_tokens[i, k_i] as the anchor's
    contrastive embedding. Positives are samples j ≠ i with state[j] == state[i];
    negatives are samples with different state. State labels enter via
    sampling only — never as a prediction target. The other K-1 tokens are
    untouched, preserving fine-grained per-position info for the JEPA loss.

    Args:
      dense_tokens: (B, K, d_emb) — output of BL7Encoder.encode_dense
      states:       (B,)         — int class labels (0, 1, 2)
      temperature:  scalar — InfoNCE softmax temperature

    Returns:
      scalar loss (mean over anchors that have at least one positive).
    """
    B, K, D = dense_tokens.shape
    device = dense_tokens.device

    # Random index per anchor — drawn fresh each call so positions rotate
    rand_k = torch.randint(0, K, (B,), device=device)
    selected = dense_tokens[torch.arange(B, device=device), rand_k]  # (B, D)
    selected = F.normalize(selected, dim=-1)

    sim = selected @ selected.T / temperature                       # (B, B)

    # Mask self
    self_mask = torch.eye(B, dtype=torch.bool, device=device)
    # Same-state pairs (excluding self) are positives
    state_eq = (states.unsqueeze(0) == states.unsqueeze(1))
    positives = state_eq & ~self_mask

    sim = sim.masked_fill(self_mask, -1e9)
    log_prob = F.log_softmax(sim, dim=1)

    n_pos = positives.sum(dim=1).clamp(min=1)
    loss_per_anchor = -(log_prob * positives.float()).sum(dim=1) / n_pos
    # Skip anchors with zero positives (rare: state class with only 1 sample in batch)
    has_pos = (positives.sum(dim=1) > 0).float()
    return (loss_per_anchor * has_pos).sum() / has_pos.sum().clamp(min=1)


# ── Teacher target extraction ────────────────────────────────────────────
@torch.no_grad()
def teacher_targets(
    teacher: BL7Encoder, x: torch.Tensor, target_layers: int
) -> torch.Tensor:
    """Run teacher encoder on x; return averaged top-K layer activations.

    x: (B, C, T)
    Returns: (B, C*T_patches, d_model) — the prediction target.
    """
    cfg = teacher.cfg
    layer_idx = list(range(cfg.n_layers - target_layers, cfg.n_layers))
    _, layer_outs = teacher.encode_full(x, return_layers=layer_idx)
    # Each is (B, C, T_patches, d_model); average across layers.
    target = torch.stack(layer_outs, dim=0).mean(dim=0)
    B, C, Tp, D = target.shape
    target = target.reshape(B, C * Tp, D)
    # Layer-norm targets per V-JEPA recipe (helps loss scale)
    target = F.layer_norm(target, (D,))
    return target


# ── Build a teacher copy of the encoder, frozen and EMA-updated ──────────
def build_teacher(student: BL7Encoder, device: torch.device) -> Tuple[BL7Encoder, EMATeacher]:
    cfg = student.cfg
    teacher = BL7Encoder(cfg).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    # Initialise teacher = student
    with torch.no_grad():
        for tp, sp in zip(teacher.parameters(), student.parameters()):
            tp.data.copy_(sp.data)
    ema = EMATeacher(student).to(device)
    return teacher, ema


@torch.no_grad()
def sync_teacher_from_ema(teacher: BL7Encoder, ema: EMATeacher) -> None:
    """Copy EMA shadow params into the teacher module."""
    for name, p in teacher.named_parameters():
        if name in ema.shadow:
            p.data.copy_(ema.shadow[name])


# ── Validation ───────────────────────────────────────────────────────────
@torch.no_grad()
def validate(student, predictor, teacher, val_dl, ecfg, tcfg, device, max_batches=20):
    student.eval()
    predictor.eval()
    L = ecfg.n_channels * ecfg.n_time_patches
    losses = []
    for i, batch in enumerate(val_dl):
        if i >= max_batches:
            break
        x = batch["sensors"].to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=tcfg.bf16):
            target = teacher_targets(teacher, x, tcfg.teacher_target_layers)

            full_final, _ = student.encode_full(x)
            B, C, Tp, D = full_final.shape
            student_seq = full_final.reshape(B, C * Tp, D)

            mask = sample_block_span_mask(
                B, L,
                ratio_lo=tcfg.mask_ratio - 0.05, ratio_hi=tcfg.mask_ratio + 0.05,
                span_min=tcfg.mask_span_min, span_max=tcfg.mask_span_max,
                device=device,
            )
            pred = predictor(student_seq, mask)
            loss = F.smooth_l1_loss(
                pred[mask], target[mask], reduction="mean"
            )
        losses.append(loss.item())
    student.train()
    predictor.train()
    return float(np.mean(losses)) if losses else float("inf")


# ── Train loop ───────────────────────────────────────────────────────────
def train(args):
    set_deterministic(args.seed)
    device = torch.device(args.device)

    # Build encoder config, applying any long-window overrides BEFORE
    # construction so __post_init__ derives n_time_patches from the right
    # window + conv schedule.
    enc_kwargs = {}
    if args.window_size is not None:
        enc_kwargs["window_size"] = args.window_size
    if args.conv_strides is not None:
        strides = tuple(int(s) for s in args.conv_strides.split(","))
        assert len(strides) == 4, "--conv-strides must be 4 comma-separated ints"
        enc_kwargs["conv_strides"] = strides
    ecfg = BL7EncoderConfig(**enc_kwargs)
    tcfg = BL7TrainConfig()
    if args.mask_span_min is not None:
        tcfg.mask_span_min = args.mask_span_min
    if args.mask_span_max is not None:
        tcfg.mask_span_max = args.mask_span_max
    print(f"  window_size={ecfg.window_size}, conv_strides={ecfg.conv_strides}, "
          f"n_time_patches={ecfg.n_time_patches}")

    if args.max_steps is not None:
        tcfg.max_steps = args.max_steps
    if args.batch_size is not None:
        tcfg.batch_size = args.batch_size
    if args.grad_accum is not None:
        tcfg.grad_accum = args.grad_accum
    if args.use_contrastive:
        tcfg.use_contrastive = True
    if args.use_time_reversal:
        tcfg.use_time_reversal = True
    if args.use_proto_supcon:
        tcfg.use_proto_supcon = True
    if args.proto_supcon_weight is not None:
        tcfg.proto_supcon_weight = args.proto_supcon_weight
    if args.use_hubert_aux:
        tcfg.use_hubert_aux = True
        if args.hubert_k is not None:
            tcfg.hubert_k = args.hubert_k
        if args.hubert_weight is not None:
            tcfg.hubert_weight = args.hubert_weight
        _md = Path(args.model_dir) if args.model_dir else MODEL_DIR
        tcfg.hubert_codebook_path = (
            args.hubert_codebook
            or str(_md / args.ckpt_subdir / "hubert_codebook.npz")
        )
    if args.K is not None:
        ecfg.K = args.K
    if args.channel_mix_every is not None:
        ecfg.channel_mix_every = args.channel_mix_every
    if args.sanity_check:
        tcfg.max_steps = 200
        tcfg.batch_size = 4
        tcfg.grad_accum = 1
        tcfg.eval_every = 50
        tcfg.save_every = 100
        tcfg.log_every = 5
        samples_per_epoch = 1000
    else:
        samples_per_epoch = args.samples_per_epoch

    print(f"BL-7 JEPA pretraining")
    print(f"  Encoder: {ecfg}")
    print(f"  Train:   max_steps={tcfg.max_steps}, batch={tcfg.batch_size}×{tcfg.grad_accum}, "
          f"lr={tcfg.lr:.0e}, mask_ratio={tcfg.mask_ratio}")

    use_contrastive = tcfg.use_contrastive
    use_proto_supcon = tcfg.use_proto_supcon
    # Both aux losses need per-tick state labels. But ONLY the (failed) Tier-2
    # contrastive resamples to 33/33/33; E3 proto-SupCon samples at the
    # natural state rate — that's the whole point of E3.
    need_state = use_contrastive or use_proto_supcon
    class_balanced = use_contrastive  # E3 deliberately does NOT class-balance
    use_hubert = tcfg.use_hubert_aux
    hb_cb_path = tcfg.hubert_codebook_path if use_hubert else None
    train_ds = SensorWindowDataset(
        sessions=TRAIN_SESSIONS,
        cfg=ecfg,
        samples_per_epoch=samples_per_epoch,
        return_state=need_state,
        class_balanced=class_balanced,
        return_hubert=use_hubert,
        hubert_codebook_path=hb_cb_path,
    )
    val_ds = SensorWindowDataset(
        sessions=VAL_SESSIONS,
        cfg=ecfg,
        samples_per_epoch=max(2 * tcfg.batch_size * 20, 1024),
        return_state=need_state,
        class_balanced=class_balanced,
        return_hubert=use_hubert,
        hubert_codebook_path=hb_cb_path,
    )
    worker_init = make_worker_init_fn(args.seed)
    train_dl = DataLoader(
        train_ds, batch_size=tcfg.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        worker_init_fn=worker_init,
        persistent_workers=args.num_workers > 0,
    )
    val_dl = DataLoader(
        val_ds, batch_size=tcfg.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=worker_init,
        persistent_workers=args.num_workers > 0,
    )

    # Models
    student = BL7Encoder(ecfg).to(device)
    teacher, ema = build_teacher(student, device)
    if args.predictor_type == "koopman":
        # Tier 9: Mariotti / Ruiz-Morales AAAI 2026 near-identity low-rank
        # linear predictor. Forces JEPA optimum onto Koopman eigenfunctions.
        predictor = KoopmanPredictor(ecfg, rank=args.koopman_rank).to(device)
        print(f"  Predictor: Koopman near-identity low-rank linear (rank={args.koopman_rank})")
    else:
        predictor = JEPAPredictor(
            ecfg, n_layers=tcfg.pred_n_layers,
            d_model=tcfg.pred_d_model, n_heads=tcfg.pred_n_heads,
        ).to(device)
        print(f"  Predictor: shallow transformer ({tcfg.pred_n_layers}L x {tcfg.pred_d_model})")
    print(f"  Student params:   {student.param_count():,}")
    print(f"  Predictor params: {sum(p.numel() for p in predictor.parameters()):,}")

    # HuBERT per-patch discrete head (non-moving anchor); pretraining-only.
    hubert_head = None
    if use_hubert:
        hubert_head = HuBERTHead(ecfg, k=tcfg.hubert_k).to(device)
        print(f"  HuBERT aux: ON (k={tcfg.hubert_k}, weight={tcfg.hubert_weight}, "
              f"codebook={tcfg.hubert_codebook_path})")

    _opt_params = list(student.parameters()) + list(predictor.parameters())
    if use_hubert:
        _opt_params += list(hubert_head.parameters())
    opt = torch.optim.AdamW(
        _opt_params,
        lr=tcfg.lr, weight_decay=tcfg.weight_decay, betas=tcfg.betas, fused=True,
    )

    # E3: prototype-based SupCon (per-class EMA prototypes, natural-rate sampling)
    proto_supcon = None
    if use_proto_supcon:
        proto_supcon = ProtoSupCon(
            d_emb=ecfg.d_emb, device=device,
            momentum=tcfg.proto_supcon_momentum,
        )
        print(f"  E3 proto-SupCon: ON (weight={tcfg.proto_supcon_weight}, "
              f"momentum={tcfg.proto_supcon_momentum}, natural-rate sampling)")

    L = ecfg.n_channels * ecfg.n_time_patches
    model_dir = Path(args.model_dir) if args.model_dir else MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = model_dir / args.ckpt_subdir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    best_val = float("inf")
    student.train()
    predictor.train()
    running, t0, epoch = {"loss": 0.0, "n": 0}, time.time(), 0
    data_iter = iter(train_dl)

    print(f"\nStarting BL-7 pretrain (sequence length L={L} tokens) ...")

    while step < tcfg.max_steps:
        for _ in range(tcfg.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                data_iter = iter(train_dl)
                batch = next(data_iter)
            x = batch["sensors"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=tcfg.bf16):
                # Teacher target (no grad)
                target = teacher_targets(teacher, x, tcfg.teacher_target_layers)

                # Student forward (full per-position activations) and dense pool
                full_final, _ = student.encode_full(x)
                B, C, Tp, D = full_final.shape
                student_seq = full_final.reshape(B, C * Tp, D)

                # Mask
                mask = sample_block_span_mask(
                    B, L,
                    ratio_lo=tcfg.mask_ratio - 0.05, ratio_hi=tcfg.mask_ratio + 0.05,
                    span_min=tcfg.mask_span_min, span_max=tcfg.mask_span_max,
                    device=device,
                )

                # Predictor on full (with mask token at masked positions)
                pred = predictor(student_seq, mask)

                # JEPA L1 on masked positions only
                jepa_loss = F.smooth_l1_loss(
                    pred[mask], target[mask], reduction="mean"
                )

                # Optional Tier 2: class-balanced contrastive on random-1-of-K
                if use_contrastive:
                    dense = student.pool(student_seq)             # (B, K, d_emb)
                    states = batch["state"].to(device)
                    con_loss = supcon_contrastive_loss(
                        dense, states, temperature=tcfg.contrastive_temperature,
                    )
                else:
                    con_loss = torch.tensor(0.0, device=device)

                # Optional E3: prototype-based SupCon (natural-rate sampling)
                if use_proto_supcon:
                    dense = student.pool(student_seq)             # (B, K, d_emb)
                    states = batch["state"].to(device)
                    proto_loss = proto_supcon.loss(dense, states)
                else:
                    proto_loss = torch.tensor(0.0, device=device)

                # Optional Tier 9: time-reversal symmetry aux
                if tcfg.use_time_reversal:
                    tr_loss = time_reversal_loss(student, x)
                else:
                    tr_loss = torch.tensor(0.0, device=device)

                # Optional: HuBERT per-patch discrete CE (non-moving per-position
                # anchor). Predict the offline k-means cluster of each MASKED
                # patch token from the student's per-position features, reusing
                # the SAME JEPA mask.
                if use_hubert:
                    hb_logits = hubert_head(student_seq)              # (B, L, k)
                    hb_labels = batch["hubert_labels"].to(device)     # (B, L)
                    hb_loss = F.cross_entropy(hb_logits[mask], hb_labels[mask])
                else:
                    hb_loss = torch.tensor(0.0, device=device)

                loss_unscaled = (
                    jepa_loss
                    + (tcfg.contrastive_weight * con_loss if use_contrastive else 0)
                    + (tcfg.proto_supcon_weight * proto_loss if use_proto_supcon else 0)
                    + (tcfg.time_reversal_weight * tr_loss if tcfg.use_time_reversal else 0)
                    + (tcfg.hubert_weight * hb_loss if use_hubert else 0)
                )
                loss = loss_unscaled / tcfg.grad_accum

            loss.backward()
            running["loss"] += loss_unscaled.item()
            running["jepa"] = running.get("jepa", 0.0) + jepa_loss.item()
            running["con"] = running.get("con", 0.0) + con_loss.item()
            running["proto"] = running.get("proto", 0.0) + proto_loss.item()
            running["tr"] = running.get("tr", 0.0) + tr_loss.item()
            running["hb"] = running.get("hb", 0.0) + hb_loss.item()
            running["n"] += 1

        if tcfg.grad_clip > 0:
            _clip_params = list(student.parameters()) + list(predictor.parameters())
            if use_hubert:
                _clip_params += list(hubert_head.parameters())
            torch.nn.utils.clip_grad_norm_(_clip_params, tcfg.grad_clip)
        opt.step()
        opt.zero_grad()
        step += 1

        # LR
        lr = tri_stage_lr(step, tcfg.max_steps, tcfg.lr,
                          warmup_pct=tcfg.warmup_pct, flat_pct=tcfg.flat_pct)
        for pg in opt.param_groups:
            pg["lr"] = lr

        # EMA update
        tau = ema_tau_at(step, tcfg.max_steps,
                         tcfg.ema_tau_start, tcfg.ema_tau_end, tcfg.ema_anneal_pct)
        ema.update(student, tau=tau)
        sync_teacher_from_ema(teacher, ema)

        if step % tcfg.log_every == 0:
            elapsed = time.time() - t0
            steps_per_sec = tcfg.log_every / elapsed if elapsed > 0 else 0
            n = max(running["n"], 1)
            avg_loss = running["loss"] / n
            avg_jepa = running["jepa"] / n
            avg_con = running["con"] / n
            avg_tr = running["tr"] / n
            avg_proto = running["proto"] / n
            avg_hb = running.get("hb", 0.0) / n
            parts = []
            if use_contrastive:
                parts.append(f"con {avg_con:.4f}")
            if use_proto_supcon:
                parts.append(f"proto {avg_proto:.4f}")
            if tcfg.use_time_reversal:
                parts.append(f"tr {avg_tr:.4f}")
            if use_hubert:
                parts.append(f"hb {avg_hb:.4f}")
            has_aux = use_contrastive or use_proto_supcon or tcfg.use_time_reversal or use_hubert
            extra = (f" | jepa {avg_jepa:.4f}" + "".join(" " + p for p in parts)) if has_aux else ""
            print(
                f"step {step:>7d}/{tcfg.max_steps} | "
                f"loss {avg_loss:.4f}{extra} | lr {lr:.2e} | tau {tau:.5f} | "
                f"{steps_per_sec:.1f} steps/s"
            )
            running = {"loss": 0.0, "jepa": 0.0, "con": 0.0, "proto": 0.0, "tr": 0.0, "hb": 0.0, "n": 0}
            t0 = time.time()

        if step % tcfg.eval_every == 0:
            val_loss = validate(student, predictor, teacher, val_dl, ecfg, tcfg, device)
            print(f"  [VAL] step {step} | val_loss {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                _save(ckpt_dir / "bl7_best.pt", step, student, predictor, ema, ecfg, tcfg, best_val, args.seed, hubert_head=hubert_head)

        if step % tcfg.save_every == 0:
            _save(ckpt_dir / f"bl7_step_{step}.pt", step, student, predictor, ema, ecfg, tcfg, best_val, args.seed, hubert_head=hubert_head)

    _save(ckpt_dir / f"bl7_final_{step}.pt", step, student, predictor, ema, ecfg, tcfg, best_val, args.seed, hubert_head=hubert_head)
    print(f"\nBL-7 pretraining complete. Best val loss: {best_val:.4f}")


def _save(path, step, student, predictor, ema, ecfg, tcfg, best_val, seed,
          hubert_head=None):
    ckpt = {
        "step": step,
        "student": student.state_dict(),
        "predictor": predictor.state_dict(),
        "ema": ema.state_dict(),
        "encoder_config": ecfg.__dict__,
        "train_config": tcfg.__dict__,
        "best_val_loss": best_val,
        "seed": seed,
        "torch_version": torch.__version__,
    }
    if hubert_head is not None:
        ckpt["hubert_head"] = hubert_head.state_dict()
    torch.save(ckpt, path)
    print(f"  Saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None,
                    help="Override grad accumulation steps. For the 4000-tick "
                         "window, reduce --batch-size and raise this to hold "
                         "effective batch ~128.")
    ap.add_argument("--samples-per-epoch", type=int, default=5_000_000)
    ap.add_argument("--sanity-check", action="store_true")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--use-contrastive", action="store_true",
                    help="Enable Tier 2 class-balanced contrastive aux. "
                         "Uses SupCon-style InfoNCE over a random-1-of-K dense "
                         "token per anchor with state-balanced sampling.")
    ap.add_argument("--use-time-reversal", action="store_true",
                    help="Enable Tier 9 time-reversal symmetry aux. Penalises "
                         "encoder(forward) vs reversed(encoder(time-reversed)) "
                         "discrepancy. Cosine-similarity-based.")
    ap.add_argument("--use-proto-supcon", action="store_true",
                    help="Enable E3 (BL-8a) prototype-based SupCon aux. Uses "
                         "per-class EMA prototypes + per-class temperature + "
                         "NATURAL-RATE sampling (no 33/33/33 resampling — the "
                         "distortion that made Tier-2 contrastive backfire).")
    ap.add_argument("--proto-supcon-weight", type=float, default=None,
                    help="Override proto-SupCon aux loss weight (default 0.10).")
    ap.add_argument("--predictor-type", choices=["transformer", "koopman"],
                    default="transformer",
                    help="JEPA predictor type. 'transformer' = default 6-layer "
                         "shallow Transformer. 'koopman' = Tier 9 near-identity "
                         "low-rank linear (Mariotti/Ruiz-Morales AAAI 2026).")
    ap.add_argument("--koopman-rank", type=int, default=8,
                    help="Rank of the Koopman predictor's low-rank update.")
    ap.add_argument("--K", type=int, default=None,
                    help="Override BL7EncoderConfig.K (number of dense tokens "
                         "per tick). Tier 9 K-sweep: K ∈ {1, 2, 4 [default], 8, 16}.")
    ap.add_argument("--channel-mix-every", type=int, default=None,
                    help="Override channel-mix-every-K-layers. Tier 9 ablation: "
                         "every=1 → channel attention every layer.")
    ap.add_argument("--ckpt-subdir", default="bl7",
                    help="Checkpoint subdirectory under --model-dir. Default "
                         "'bl7' (the 500-tick baseline). Use a distinct name "
                         "(e.g. 'bl7_w4000') for new variants so the baseline "
                         "checkpoint is never overwritten.")
    ap.add_argument("--window-size", type=int, default=None,
                    help="Override encoder window length in ticks (default 500). "
                         "Long-horizon variant uses 4000.")
    ap.add_argument("--conv-strides", default=None,
                    help="Comma-separated 4-layer conv stride schedule, e.g. "
                         "'5,4,2,1' for the 4000-tick window. Default keeps the "
                         "500-tick schedule (5,2,2,1). n_time_patches is derived.")
    ap.add_argument("--mask-span-min", type=int, default=None,
                    help="Override JEPA mask span min (patches). Rescale up for "
                         "longer patch sequences.")
    ap.add_argument("--mask-span-max", type=int, default=None,
                    help="Override JEPA mask span max (patches).")
    ap.add_argument("--use-hubert-aux", action="store_true",
                    help="Enable the HuBERT-style per-patch discrete-codebook "
                         "auxiliary: predict the offline k-means cluster of each "
                         "MASKED patch token (a non-moving per-position anchor).")
    ap.add_argument("--hubert-k", type=int, default=None,
                    help="Override codebook size (default 256).")
    ap.add_argument("--hubert-weight", type=float, default=None,
                    help="Override HuBERT CE weight (default 1.0, co-equal anchor).")
    ap.add_argument("--hubert-codebook", default=None,
                    help="Path to hubert_codebook.npz (default "
                         "<ckpt_dir>/hubert_codebook.npz).")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
