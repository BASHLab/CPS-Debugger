"""BL-7 K-sweep + Perceiver bypass — Tier 9 / Rare-state plan E2.

Builds a frozen-backbone wrapper around the BL-7 v1 encoder checkpoint with
a fresh Perceiver pool head of configurable K. The 12-layer axial
transformer + conv frontend + output norm are frozen; only the pool is
trainable (alongside the downstream AR decoder).

The no-Perceiver path bypasses the pool entirely and returns the full
(B, 154, 512) flat axial token sequence. Downstream must consume a long
prefix — use `ARModernCrossModel` cross-attention for this case.

Usage:

    from bl7_ksweep import build_ksweep_encoder

    # K=8 variant
    enc = build_ksweep_encoder("models/bl7/bl7_best.pt", K=8, device="cuda")
    out = enc.encode_dense(sensor_window)   # (B, 8, 256)

    # No-Perceiver path
    enc = build_ksweep_encoder("models/bl7/bl7_best.pt", K=None, device="cuda")
    out = enc.encode_axial(sensor_window)   # (B, 154, 512)

The returned encoder has backbone parameters frozen (`requires_grad_=False`)
and the new pool's parameters trainable.
"""
from pathlib import Path
from typing import Optional, Union

import torch

from config import BL7EncoderConfig
from bl7_model import BL7Encoder, EMATeacher


def build_ksweep_encoder(
    ckpt_path: Union[str, Path],
    K: Optional[int],
    device: Union[str, torch.device] = "cuda",
    use_ema: bool = True,
) -> BL7Encoder:
    """Load BL-7 v1 backbone, swap in a fresh K-pool, freeze backbone.

    Args:
        ckpt_path: path to bl7_best.pt (e.g. "models/bl7/bl7_best.pt")
        K: number of Perceiver query tokens to use; or None for the
            no-Perceiver / raw-axial-token path.
        device: where to place the encoder.
        use_ema: load EMA-teacher weights (the published artifact) rather
            than the student. Recommended.

    Returns:
        BL7Encoder with frozen backbone and trainable pool (or unused pool
        if K is None — caller will use encode_axial() instead of encode_dense).
    """
    device = torch.device(device)
    ckpt = torch.load(Path(ckpt_path), map_location=device, weights_only=False)

    # Build a fresh encoder with the ORIGINAL K so the state-dict loads
    # cleanly; we'll swap the pool afterwards if requested.
    orig_cfg = BL7EncoderConfig(**ckpt["encoder_config"])
    enc = BL7Encoder(orig_cfg).to(device)

    # Apply EMA-teacher (or raw student) weights.
    if use_ema and "ema" in ckpt:
        ema = EMATeacher(enc).to(device)
        ema.load_state_dict(ckpt["ema"])
        ema.copy_to(enc)
    else:
        enc.load_state_dict(ckpt["student"])

    if K is not None and int(K) != orig_cfg.K:
        # Swap the pool to the new K. The new pool is randomly initialised
        # and will be trained alongside the AR decoder.
        enc.replace_pool(int(K))

    # Freeze backbone; only the new pool head (if any) trains.
    enc.freeze_backbone()
    return enc


def trainable_param_count(enc: BL7Encoder) -> int:
    return sum(p.numel() for p in enc.parameters() if p.requires_grad)


def frozen_param_count(enc: BL7Encoder) -> int:
    return sum(p.numel() for p in enc.parameters() if not p.requires_grad)


# ── Joint encoder-AR wrapper for E2 K-sweep training ────────────────────
class JointEncoderARModel(torch.nn.Module):
    """Wraps a frozen-backbone BL-7 encoder + a trainable AR-modern decoder.

    The encoder's backbone is frozen (params have requires_grad=False) but
    the Perceiver pool is trainable. The AR decoder is fully trainable.
    Loss is computed by the AR decoder; gradients flow back through both
    the AR decoder and the pool (but not into the frozen backbone weights —
    forward still runs through the backbone since the pool needs its
    features).

    Use with the JointTraceWindowDataset from data.py — batches yield
    raw sensor windows (B, 7, w) which this module encodes on the fly.

    Signature mirrors ARModernModel/ARModernCrossModel so existing
    training loops work unchanged.
    """

    def __init__(self, encoder: BL7Encoder, ar_model: torch.nn.Module,
                 axial_mode: bool = False):
        super().__init__()
        self.encoder = encoder
        self.ar = ar_model
        self.axial_mode = axial_mode
        # Mirror useful attributes that ar_modern_train.py reads from the model
        self.cfg = ar_model.cfg

    @property
    def backbone(self):
        # ar_modern_train.py iterates model.backbone.blocks for gradient
        # checkpointing — delegate to the AR model's backbone.
        return self.ar.backbone

    def encode(self, sensor_window: torch.Tensor) -> torch.Tensor:
        # Backbone is frozen; using no_grad() saves ~4× backward memory.
        # The pool runs WITH grad enabled (no torch.no_grad block here).
        if self.axial_mode:
            return self.encoder.encode_axial(sensor_window)
        return self.encoder.encode_dense(sensor_window)

    def compute_loss(self, x0, sensor_window, pad_mask):
        sensor_emb = self.encode(sensor_window)
        return self.ar.compute_loss(x0, sensor_emb, pad_mask)

    @torch.no_grad()
    def sample(self, sensor_window: torch.Tensor, seq_len: int,
               temperature: float = 1.0, greedy: bool = False,
               guidance_scale: float = 1.0):
        sensor_emb = self.encode(sensor_window)
        if hasattr(self.ar, "sample"):
            # ARModernCrossModel takes guidance_scale; ARModernModel ignores it.
            kwargs = dict(seq_len=seq_len, temperature=temperature, greedy=greedy)
            if guidance_scale != 1.0:
                kwargs["guidance_scale"] = guidance_scale
            return self.ar.sample(sensor_emb, **kwargs)
        raise NotImplementedError("Underlying AR model has no sample() method")

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
