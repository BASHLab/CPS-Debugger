"""Tier 3 fine-tuning helpers: sandwich-N (unfreeze last N encoder blocks)
and LoRA-rank-R (low-rank adapters on encoder attention).

Both are joint-training modes: the BL-7 encoder is loaded from a pretraining
checkpoint, then either (a) the last N blocks are unfrozen and trained
alongside the AR-modern decoder, or (b) LoRA adapters are added to encoder
QKV+O projections and trained jointly. The "frozen" baseline (the default
elsewhere in this codebase) does neither — encoder weights stay fixed.

Usage from ar_modern_train.py:

    from bl7_finetune_helpers import (
        load_bl7_for_finetune, freeze_encoder, sandwich_n_finetune,
        attach_lora_to_encoder,
    )

    encoder = load_bl7_for_finetune("models/bl7/bl7_best.pt", device)
    if args.encoder_mode == "frozen":
        freeze_encoder(encoder)
    elif args.encoder_mode == "sandwich2":
        sandwich_n_finetune(encoder, n=2)
    elif args.encoder_mode == "lora32":
        attach_lora_to_encoder(encoder, rank=32)

    # encoder.train() / .encode_dense(x) used inside the AR forward pass
    # to produce sensor_emb on-the-fly instead of reading from disk.
"""
import math
from typing import List

import torch
import torch.nn as nn

from bl7_model import BL7Encoder, EMATeacher
from config import BL7EncoderConfig


# ── Loading ──────────────────────────────────────────────────────────────
def load_bl7_for_finetune(ckpt_path, device, use_ema: bool = True) -> BL7Encoder:
    """Load a BL-7 encoder, preferring EMA-teacher weights.

    Returned encoder is in eval mode and has requires_grad=False on all
    parameters. Call one of freeze_encoder / sandwich_n_finetune /
    attach_lora_to_encoder to set up the desired finetune regime.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ecfg = BL7EncoderConfig(**ckpt["encoder_config"])
    enc = BL7Encoder(ecfg).to(device)
    if use_ema and "ema" in ckpt:
        ema = EMATeacher(enc).to(device)
        ema.load_state_dict(ckpt["ema"])
        ema.copy_to(enc)
    else:
        enc.load_state_dict(ckpt["student"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def freeze_encoder(enc: BL7Encoder) -> None:
    for p in enc.parameters():
        p.requires_grad_(False)
    enc.eval()


# ── Sandwich-N (unfreeze last N encoder transformer blocks) ───────────────
def sandwich_n_finetune(enc: BL7Encoder, n: int = 2) -> List[nn.Parameter]:
    """Unfreeze the last `n` transformer blocks + the perceiver pool.

    Leaves the conv frontend + patch embed + early transformer blocks frozen
    so that the lower-level features stay anchored to the pretrained encoder.
    Returns the list of newly-trainable parameters (caller's optimizer
    consumes these).
    """
    # Re-enable training mode globally; specific freezes below.
    for p in enc.parameters():
        p.requires_grad_(False)
    enc.train()  # affects dropout / norm running stats only

    total_layers = len(enc.blocks)
    unfreeze_from = max(0, total_layers - n)
    trainable_params = []

    # Last n blocks
    for i, block in enumerate(enc.blocks):
        if i >= unfreeze_from:
            for p in block.parameters():
                p.requires_grad_(True)
                trainable_params.append(p)

    # Perceiver pool (Tier 9 ablation: pool may be the "thinnest" layer to retune)
    for p in enc.pool.parameters():
        p.requires_grad_(True)
        trainable_params.append(p)

    # Output norm
    for p in enc.out_norm.parameters():
        p.requires_grad_(True)
        trainable_params.append(p)

    return trainable_params


# ── LoRA (rank-R adapters on encoder attention) ──────────────────────────
class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear with a learnable low-rank adapter.

    The original weight stays frozen; only A ∈ R^{r×in}, B ∈ R^{out×r} train.
    Output: y = W x + (B @ A) x · α / r.  α defaults to r for unit-scale init.
    """

    def __init__(self, base: nn.Linear, rank: int = 32, alpha: int = 32,
                 dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        in_f, out_f = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B init zero so the adapter is identity at start of training.
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        delta = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return y + delta * self.scale


def attach_lora_to_encoder(
    enc: BL7Encoder, rank: int = 32, alpha: int = 32, dropout: float = 0.0,
) -> List[nn.Parameter]:
    """Wrap encoder attention QKV+O linears with LoRA adapters.

    Targets the qkv and out_proj of every BidirBlockModern inside each
    AxialAttnBlock. Returns the list of newly-trainable LoRA parameters
    (the rest of the encoder remains frozen).
    """
    for p in enc.parameters():
        p.requires_grad_(False)

    trainable: List[nn.Parameter] = []
    for axial in enc.blocks:                  # AxialAttnBlock
        block = axial.block                   # BidirBlockModern inside
        # Replace block.qkv and block.out_proj with LoRA-wrapped versions
        qkv = block.qkv
        out_proj = block.out_proj
        block.qkv = LoRALinear(qkv, rank=rank, alpha=alpha, dropout=dropout)
        block.out_proj = LoRALinear(out_proj, rank=rank, alpha=alpha, dropout=dropout)
        # Move new params to the same device as the wrapped layer
        block.qkv.to(qkv.weight.device)
        block.out_proj.to(out_proj.weight.device)
        for name, p in block.qkv.named_parameters(recurse=True):
            if name.startswith("lora_"):
                trainable.append(p)
        for name, p in block.out_proj.named_parameters(recurse=True):
            if name.startswith("lora_"):
                trainable.append(p)

    enc.train()  # dropout / norm running stats; weights still frozen except LoRA
    return trainable


def count_trainable(enc: BL7Encoder) -> int:
    return sum(p.numel() for p in enc.parameters() if p.requires_grad)
