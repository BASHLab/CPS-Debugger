"""BL-7 sensor encoder.

Channel-aware ViT-Small with conv frontend + Perceiver-style 4-query pool.
Outputs K=4 dense tokens per tick at d_emb=256 — meant to drop into the
existing AR-modern decoder once the dense-conditioning interface lands
(or via a mean-pool stopgap until then).

Pretraining objective is data2vec / V-JEPA-style EMA latent-target prediction
(see bl7_pretrain.py); this file only defines the encoder + predictor + EMA
teacher wrappers.

Architecture (per BL-7 plan §4.2):

    INPUT (B, 7 channels, 500 ticks)
      │
      │ Conv frontend (weight-shared across channels): 500 → 100 → 50 → 25
      │   blocks of (Conv1d, GELU, GroupNorm). Output (B, 7, 25, 256).
      │
      │ Patch projection (256 → 512) + learned channel-id embedding +
      │ sinusoidal time embedding. Output (B, 7, 25, 512).
      │
      │ 12 axial-attention layers, alternating:
      │   layers 0, 3, 6, 9          → time-axis attention   (within each channel)
      │   layers 1,2,4,5,7,8,10,11   → channel-axis attention (within each time patch)
      │
      │ Perceiver pool: 4 learned queries × cross-attention over flat
      │   (B, 175, 512) tokens. Output (B, 4, 512).
      │
      │ Output projection: Linear(512 → 256). Output (B, 4, 256).
      ▼
    sensor_embeddings_bl7/<session>_emb.npy   shape (N_ticks, 4, 256)

The encoder has both `.encode_dense(x)` (returns K=4 dense tokens) and
`.encode_full(x)` (returns the full per-token contextualised representation
before pooling — used by the JEPA loss to supervise per-position features).
"""
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BL7EncoderConfig
from ar_modern_model import RMSNorm, SwiGLU


# ── Conv frontend ────────────────────────────────────────────────────────
class ConvFrontend(nn.Module):
    """Per-channel weight-shared conv stack: 500 ticks → 25 patches @ 256-d.

    All 7 channels see the same conv weights — channel identity is added
    later via a learned channel-id embedding. This is the "speech foundation
    model on sensors" lesson from Apple 2025 (arXiv:2509.00221): the
    conv frontend is what transports cross-modally.
    """

    def __init__(self, cfg: BL7EncoderConfig):
        super().__init__()
        # 4-layer stack; channel widths fixed, (kernel, stride, padding) driven
        # by the config schedule so the same module covers both the 500-tick
        # design (strides 5,2,2,1 → 22 patches) and the 4000-tick design
        # (strides 5,4,2,1 → 98 patches). GroupNorm num_groups=8 (wav2vec 2.0).
        widths = [1, 64, 128, 256, cfg.conv_out_dim]
        kernels, strides, paddings = cfg.conv_kernels, cfg.conv_strides, cfg.conv_paddings
        assert len(kernels) == len(strides) == len(paddings) == 4, (
            "ConvFrontend expects 4-layer kernel/stride/padding schedules")
        layers = []
        for i in range(4):
            layers += [
                nn.Conv1d(widths[i], widths[i + 1], kernel_size=kernels[i],
                          stride=strides[i], padding=paddings[i], bias=False),
                nn.GELU(),
                nn.GroupNorm(num_groups=8, num_channels=widths[i + 1]),
            ]
        self.blocks = nn.Sequential(*layers)
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T). Reshape so each channel is its own "sample".
        B, C, T = x.shape
        assert C == self.cfg.n_channels and T == self.cfg.window_size, (
            f"expected ({self.cfg.n_channels}, {self.cfg.window_size}) got ({C}, {T})")
        x = x.reshape(B * C, 1, T)              # (B*C, 1, window_size)
        x = self.blocks(x)                      # (B*C, conv_out_dim, n_time_patches)
        assert x.shape[-1] == self.cfg.n_time_patches, (
            f"conv produced {x.shape[-1]} patches, config says "
            f"{self.cfg.n_time_patches}; check conv schedule vs window_size")
        x = x.reshape(B, C, self.cfg.conv_out_dim, self.cfg.n_time_patches)
        return x.transpose(2, 3).contiguous()   # (B, C, T_patches, conv_out_dim)


# ── Patch embedding with positional info ─────────────────────────────────
class PatchEmbed(nn.Module):
    """conv frontend output → d_model tokens with channel-id + time positional embeddings."""

    def __init__(self, cfg: BL7EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.conv_out_dim, cfg.d_model, bias=False)
        # Learned channel-id (small vocab, 7 entries)
        self.channel_emb = nn.Parameter(
            torch.randn(cfg.n_channels, cfg.d_model) * 0.02)
        # Sinusoidal time-axis positional (fixed)
        self.register_buffer(
            "time_pos",
            _sinusoidal_position(cfg.n_time_patches, cfg.d_model),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T_patches, conv_out_dim)
        x = self.proj(x)                                     # (B, C, T, d)
        x = x + self.channel_emb[None, :, None, :]           # broadcast over T
        x = x + self.time_pos[None, None, :, :]              # broadcast over B,C
        return x


def _sinusoidal_position(length: int, dim: int) -> torch.Tensor:
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / dim))
    out = torch.zeros(length, dim)
    out[:, 0::2] = torch.sin(pos * div)
    out[:, 1::2] = torch.cos(pos * div)
    return out


# ── Bidirectional self-attention block ───────────────────────────────────
class BidirBlockModern(nn.Module):
    """Pre-norm RMSNorm + SDPA + RMSNorm + SwiGLU. No causal mask, no RoPE.

    Mirrors CausalBlockModern in ar_modern_model.py except that attention is
    bidirectional and there's no rotary position encoding (positional info
    enters once at the patch-embed level via channel + time embeddings).
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm_attn = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.attn_drop_p = dropout

        self.norm_ff = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x_norm = self.norm_attn(x)
        qkv = self.qkv(x_norm).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # (B, H, L, Dh) each

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).reshape(B, L, D)
        x = x + self.out_proj(attn)
        x = x + self.ff(self.norm_ff(x))
        return x


class AxialAttnBlock(nn.Module):
    """Wraps BidirBlockModern to attend along time-axis or channel-axis.

    Input/Output shape: (B, C, T, D).
    axis="time"    → attend within each channel over T tokens.
    axis="channel" → attend within each time patch over C tokens.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float,
                 axis: str):
        super().__init__()
        assert axis in ("time", "channel")
        self.axis = axis
        self.block = BidirBlockModern(d_model, n_heads, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, D = x.shape
        if self.axis == "time":
            y = x.reshape(B * C, T, D)
            y = self.block(y)
            return y.reshape(B, C, T, D)
        else:  # channel
            y = x.permute(0, 2, 1, 3).reshape(B * T, C, D)
            y = self.block(y)
            return y.reshape(B, T, C, D).permute(0, 2, 1, 3).contiguous()


# ── Perceiver-style 4-query pool ─────────────────────────────────────────
class PerceiverPool(nn.Module):
    """K learned latent queries cross-attend to (B, C*T, D) flat tokens.

    Output is (B, K, d_emb) — the dense per-tick representation that goes
    to disk and is consumed by the AR decoder.
    """

    def __init__(self, cfg: BL7EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.queries = nn.Parameter(torch.randn(cfg.K, cfg.d_model) * 0.02)
        self.norm_q = RMSNorm(cfg.d_model)
        self.norm_kv = RMSNorm(cfg.d_model)
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.kv_proj = nn.Linear(cfg.d_model, 2 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        # Per-head normalisation
        self.q_norm = RMSNorm(cfg.d_model // cfg.n_heads)
        self.k_norm = RMSNorm(cfg.d_model // cfg.n_heads)
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        # FFN after pool, then project to output dim
        self.norm_ff = RMSNorm(cfg.d_model)
        self.ff = SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.out_dim_proj = nn.Linear(cfg.d_model, cfg.d_emb, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C*T, d_model) — caller flattens before passing in.
        B, L, D = x.shape
        K = self.cfg.K

        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, K, D)
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(x)

        q_p = self.q_proj(q_norm).reshape(B, K, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(kv_norm).reshape(B, L, 2, self.n_heads, self.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)  # (B, H, L, Dh)

        q_p = self.q_norm(q_p)
        k = self.k_norm(k)

        attn = F.scaled_dot_product_attention(
            q_p, k, v,
            dropout_p=self.cfg.dropout if self.training else 0.0,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).reshape(B, K, D)
        out = q + self.out_proj(attn)
        out = out + self.ff(self.norm_ff(out))
        return self.out_dim_proj(out)        # (B, K, d_emb)


# ── BL-7 encoder ─────────────────────────────────────────────────────────
class BL7Encoder(nn.Module):
    """Sensor encoder. Returns either dense (B, K, d_emb) per-tick tokens
    (production / inference) or the full per-position activations across
    all transformer layers (training-time / JEPA target).
    """

    # layers 0,3,6,9 → time, others → channel (when channel_mix_every=3).
    # With channel_mix_every=1, every layer mixes channels (no per-channel
    # time-axis layers) — Tier 9 ablation.
    @staticmethod
    def _axis_for_layer(i: int, every: int) -> str:
        if every <= 1:
            return "channel"
        return "time" if i % every == 0 else "channel"

    def __init__(self, cfg: BL7EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.frontend = ConvFrontend(cfg)
        self.patch_embed = PatchEmbed(cfg)
        self.blocks = nn.ModuleList([
            AxialAttnBlock(cfg.d_model, cfg.n_heads, cfg.d_ff,
                           cfg.dropout, self._axis_for_layer(i, cfg.channel_mix_every))
            for i in range(cfg.n_layers)
        ])
        self.out_norm = RMSNorm(cfg.d_model)
        self.pool = PerceiverPool(cfg)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_full(self, x: torch.Tensor,
                    return_layers: Optional[List[int]] = None
                    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run the full encoder; return final pre-norm features + selected per-layer activations.

        x: (B, C, T)
        Returns:
          final: (B, C, T_patches, d_model) post-out_norm
          layer_outs: list of selected layer activations as (B, C, T_patches, d_model)
        """
        x = self.frontend(x)               # (B, C, T_patches, conv_out_dim)
        x = self.patch_embed(x)            # (B, C, T_patches, d_model)
        layer_outs: List[torch.Tensor] = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if return_layers is not None and i in return_layers:
                layer_outs.append(x)
        final = self.out_norm(x)
        return final, layer_outs

    def encode_dense(self, x: torch.Tensor) -> torch.Tensor:
        """Production path: (B, C, T) → (B, K, d_emb)."""
        final, _ = self.encode_full(x)
        B, C, T_patches, D = final.shape
        flat = final.reshape(B, C * T_patches, D)
        return self.pool(flat)

    def encode_axial(self, x: torch.Tensor) -> torch.Tensor:
        """No-pool path: (B, C, T) → (B, C*T_patches, d_model).

        Returns the full flattened axial token sequence after the encoder's
        output norm, bypassing the Perceiver pool entirely. Used by the
        Tier 9 K-sweep "no-Perceiver" path (E2 in the rare-state recovery
        plan) — the downstream decoder must support a long prefix
        (e.g. ARModernCrossModel cross-attention).
        """
        final, _ = self.encode_full(x)
        B, C, T_patches, D = final.shape
        return final.reshape(B, C * T_patches, D)

    def replace_pool(self, new_K: int) -> "PerceiverPool":
        """Swap the Perceiver pool for one with a different K.

        Used by the E2 K-sweep: load BL-7 v1 backbone, then resize the pool
        head independently for K ∈ {4, 8, 16, 32, 64}. Backbone weights are
        preserved; only the new pool's parameters are trainable.

        The new pool is randomly initialised (the original K=4 pool's query
        Parameter is the wrong shape for the new K). Caller should freeze
        the rest of the encoder and train the new pool alongside whatever
        downstream module consumes the dense tokens.

        Returns the new pool module for the caller's convenience.
        """
        new_cfg = type(self.cfg)(**{**self.cfg.__dict__, "K": int(new_K)})
        new_pool = PerceiverPool(new_cfg)
        device = next(self.parameters()).device
        new_pool = new_pool.to(device)
        self.pool = new_pool
        # Update cfg in-place to reflect the new K
        self.cfg = new_cfg
        return new_pool

    def freeze_backbone(self) -> None:
        """Freeze everything except the Perceiver pool. For K-sweep training."""
        for name, p in self.named_parameters():
            if name.startswith("pool."):
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_dense(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Predictor (data2vec / V-JEPA-style) ──────────────────────────────────
class JEPAPredictor(nn.Module):
    """Shallow Transformer that predicts EMA-teacher latents at masked positions.

    Following V-JEPA 2.1 / data2vec 2.0: takes the student's visible tokens
    plus a learned mask token at masked positions, predicts the teacher's
    contextualised hidden state at masked positions only.
    """

    def __init__(self, encoder_cfg: BL7EncoderConfig, n_layers: int,
                 d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.encoder_d = encoder_cfg.d_model
        self.d_model = d_model
        self.in_proj = nn.Linear(encoder_cfg.d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, encoder_cfg.d_model, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.blocks = nn.ModuleList([
            BidirBlockModern(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_layers)
        ])
        self.out_norm = RMSNorm(d_model)

    def forward(self, visible: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        visible: (B, L, d_enc) — full-length encoder output at every position
                  with masked positions replaced by 0 (or any placeholder).
        mask: (B, L) — bool, True where position is masked (predict here).
        Returns: (B, L, d_enc) — predictions at every position; caller
                  selects masked positions via mask.
        """
        B, L, _ = visible.shape
        x = self.in_proj(visible)
        # Replace masked positions with the learned mask token
        x = torch.where(mask.unsqueeze(-1), self.mask_token.expand(B, L, -1), x)
        for blk in self.blocks:
            x = blk(x)
        x = self.out_norm(x)
        return self.out_proj(x)


# ── HuBERT-style discrete-codebook head (non-moving per-patch anchor) ────
class HuBERTHead(nn.Module):
    """Predicts the offline k-means cluster id of each patch token from the
    student's per-position encoder features (B, L, d_enc) -> logits (B, L, k).

    Pretraining-only (discarded at inference, like JEPAPredictor). The CE loss
    is taken at MASKED positions only, reusing the JEPA mask, so it anchors the
    same per-position prediction the moving EMA target corrupts.
    """

    def __init__(self, encoder_cfg: BL7EncoderConfig, k: int):
        super().__init__()
        self.proj = nn.Linear(encoder_cfg.d_model, k, bias=True)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.proj(feats)


# ── Koopman-constrained predictor (Tier 9 ablation) ─────────────────────
class KoopmanPredictor(nn.Module):
    """Near-identity low-rank linear predictor for the JEPA loss.

    Implements the Mariotti / Ruiz-Morales AAAI 2026 (arXiv:2511.09783)
    recipe: the JEPA predictor P is constrained to lie in a
    near-identity family P(z) = z + tanh(g) · (B @ A @ z), where A, B are
    learned low-rank factors of an additive update and g is a learnable
    scalar gate initialised at 0 so training starts at identity.

    Per the paper, this constraint makes the JEPA loss's optimum represent
    Koopman eigenfunctions / regime-indicator functions of the underlying
    dynamical system — which maps onto our 3-state FSM as a clean test.
    Note: published validation is autonomous/ergodic systems only; we
    are testing the extension to actuated systems.
    """

    def __init__(self, encoder_cfg: BL7EncoderConfig, rank: int = 8):
        super().__init__()
        self.encoder_d = encoder_cfg.d_model
        self.rank = rank
        # Low-rank update: A maps d_model → rank, B maps rank → d_model.
        self.A = nn.Linear(encoder_cfg.d_model, rank, bias=False)
        self.B = nn.Linear(rank, encoder_cfg.d_model, bias=False)
        # Zero-init gate so the predictor is identity at the start of training.
        self.gate = nn.Parameter(torch.zeros(1))
        # Mask token (predictor needs to insert *something* at masked positions
        # since visible-only inputs would lose positional info). Use a small
        # learned vector; identity is applied to it just like to visible
        # positions, so masked-position outputs come from gate·B·A·mask_token.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, encoder_cfg.d_model))

    def forward(self, visible: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """visible: (B, L, d_enc) — student output, masked positions zeroed.
        mask:    (B, L) bool — True where masked.
        Returns: (B, L, d_enc) — near-identity-projected per-position output.
        """
        B, L, _ = visible.shape
        x = torch.where(mask.unsqueeze(-1),
                        self.mask_token.expand(B, L, -1), visible)
        delta = self.B(self.A(x))                              # (B, L, d_model)
        return x + torch.tanh(self.gate) * delta


# ── EMA teacher wrapper ──────────────────────────────────────────────────
class EMATeacher:
    """Maintains an EMA copy of an encoder's parameters.

    Stored as a buffer of detached parameter tensors. Update in-place after
    each optimizer step. Decay τ is annealed externally via update(tau).
    """

    def __init__(self, model: nn.Module):
        self.shadow = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    def to(self, device):
        for k in self.shadow:
            self.shadow[k] = self.shadow[k].to(device)
        return self

    @torch.no_grad()
    def update(self, model: nn.Module, tau: float):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[name].mul_(tau).add_(p.detach(), alpha=1.0 - tau)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name])

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, sd):
        for k, v in sd.items():
            self.shadow[k] = v.clone()
