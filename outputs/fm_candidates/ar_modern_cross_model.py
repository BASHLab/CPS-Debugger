"""AR-modern with Flamingo-style gated cross-attention conditioning.

Replaces the single-prefix-token conditioning of ar_modern_model.py with a
gated cross-attention block per decoder layer (Flamingo recipe — Alayrac et
al. arXiv:2204.14198). Adds a learned null-token + conditioning dropout so
classifier-free guidance is available at inference time (Sanchez et al.
arXiv:2306.17806).

This is Tier 1A.1 of the BL-7 plan. The existing ARModernModel is kept
untouched so the paper has a clean ablation row separating
"prefix-token conditioning" from "dense cross-attention conditioning".

Key differences from ar_modern_model.py:
  - sensor_emb is (B, K, d_emb) instead of (B, d_model). For Tier 1A.1 with
    the existing Chronos-2 encoder, K=1 and d_emb=d_model=512. For BL-7 (Tier
    1B onwards), K=4 and d_emb=256.
  - No prefix concatenation. Each self-attention block is preceded by a
    GatedCrossAttn block whose residual is zero-initialized via tanh(gate)
    so training begins at the unconditioned model and the conditioning
    pathway opens gradually.
  - Conditioning dropout: with prob p during training, replace sensor_emb
    with a learned null_emb. Required for CFG. p=0 → no dropout, exact
    behaviour of the always-conditioned case.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig
from ar_model import _precompute_freqs, _apply_rope
from ar_modern_model import RMSNorm, SwiGLU, CausalBlockModern


# ── Gated cross-attention (Flamingo recipe) ──────────────────────────────
class GatedCrossAttn(nn.Module):
    """One Flamingo-style cross-attention block.

    Query: trace tokens (B, L_trace, d_model).
    Key/Value: sensor tokens (B, K, d_emb), projected to d_model internally.

    Two zero-initialised tanh gates (one for cross-attn residual, one for
    FFN residual). At init both gates are 0 → tanh(0)=0 → block is identity.
    During training the gates relax and the conditioning pathway opens.
    Same gating recipe as Flamingo §2.2.
    """

    def __init__(self, d_model: int, n_heads: int, d_emb: int, d_ff: int,
                 dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Pre-norms
        self.norm_q = RMSNorm(d_model)
        self.norm_kv = RMSNorm(d_emb)

        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_emb, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # QK-Norm (matches CausalBlockModern stability convention)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        # Zero-init Flamingo gate
        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.attn_drop_p = dropout

        # Cross-attn FFN (Flamingo also gates a per-block FFN)
        self.norm_ff = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff, dropout)
        self.ff_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model). kv: (B, K, d_emb).
        B, L, D = x.shape
        K = kv.shape[1]

        # Cross-attention with zero-init gate
        x_norm = self.norm_q(x)
        kv_norm = self.norm_kv(kv)

        q = self.q_proj(x_norm).reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        kv_p = self.kv_proj(kv_norm).reshape(B, K, 2, self.n_heads, self.head_dim)
        k, v = kv_p.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, H, K, Dh)

        # QK-Norm before attention (no RoPE on cross-attn — keys live in a
        # different sequence space than queries)
        q = self.q_norm(q)
        k = self.k_norm(k)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).reshape(B, L, D)

        x = x + torch.tanh(self.attn_gate) * self.out_proj(attn)

        # Gated FFN
        x = x + torch.tanh(self.ff_gate) * self.ff(self.norm_ff(x))
        return x


# ── Backbone ─────────────────────────────────────────────────────────────
class CausalBackboneCross(nn.Module):
    """Decoder backbone with one GatedCrossAttn block before each self-attn block.

    Layout per layer:
      x = GatedCrossAttn(x, kv=sensor_emb)   # zero-init, identity at start
      x = CausalBlockModern(x)               # standard causal self-attn

    Conditioning dropout: with prob cond_dropout_p (default 0 — set per-run),
    replace the supplied sensor_emb with a learned null_emb (B, K, d_emb).
    Done per-sample, not per-batch, so the same batch can mix conditioned
    and unconditioned forward passes.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.pad_id = cfg.vocab_size
        self.bos_id = cfg.vocab_size + 1
        vocab_total = cfg.vocab_size + 2

        # Required new fields on cfg (set via CLI in train script):
        #   cfg.K               — number of dense sensor tokens (1 today, 4 for BL-7)
        #   cfg.d_sensor_emb    — encoder output dim (512 today, 256 for BL-7)
        #   cfg.cond_dropout_p  — conditioning dropout prob during training
        K = cfg.n_sensor_tokens
        d_emb = getattr(cfg, "d_sensor_emb", cfg.d_model)
        self.K = K
        self.d_emb = d_emb
        self.cond_dropout_p = getattr(cfg, "cond_dropout_p", 0.10)

        self.token_emb = nn.Embedding(vocab_total, cfg.d_model)

        # Learned null token (for CFG and conditioning dropout)
        # Same shape as a single-sample sensor_emb minus batch dim.
        self.null_emb = nn.Parameter(torch.randn(1, K, d_emb) * 0.02)

        # Stack: cross-attn + self-attn per layer
        self.cross_blocks = nn.ModuleList([
            GatedCrossAttn(cfg.d_model, cfg.n_heads, d_emb, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.self_blocks = nn.ModuleList([
            CausalBlockModern(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])

        self.out_norm = RMSNorm(cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, vocab_total, bias=False)

        # RoPE on self-attn over trace tokens only (no prefix concat now)
        max_pos = cfg.max_seq_len + 1  # +1 for BOS
        cos, sin = _precompute_freqs(cfg.d_model // cfg.n_heads, max_pos)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        # gates already zero-initialised in GatedCrossAttn.__init__

    def _normalise_sensor(self, sensor_emb: torch.Tensor) -> torch.Tensor:
        """Accept (B, d_emb) or (B, K, d_emb); always return (B, K, d_emb)."""
        if sensor_emb.dim() == 2:
            sensor_emb = sensor_emb.unsqueeze(1)  # (B, 1, d_emb)
        assert sensor_emb.dim() == 3, f"expected 3D (B,K,d_emb), got {sensor_emb.shape}"
        assert sensor_emb.shape[1] == self.K, (
            f"K mismatch: got {sensor_emb.shape[1]}, model expects {self.K}")
        assert sensor_emb.shape[2] == self.d_emb, (
            f"d_emb mismatch: got {sensor_emb.shape[2]}, model expects {self.d_emb}")
        return sensor_emb

    def _apply_cond_dropout(self, sensor_emb: torch.Tensor) -> torch.Tensor:
        """Replace per-sample sensor_emb with null_emb with prob cond_dropout_p.

        Per-sample mask gives gradient signal on null_emb in every batch and
        keeps the per-batch noise from being all-or-nothing.
        """
        if not self.training or self.cond_dropout_p <= 0:
            return sensor_emb
        B = sensor_emb.shape[0]
        device = sensor_emb.device
        drop = (torch.rand(B, device=device) < self.cond_dropout_p).view(B, 1, 1)
        null = self.null_emb.expand(B, -1, -1)
        return torch.where(drop, null, sensor_emb)

    def _force_unconditional(self, sensor_emb: torch.Tensor) -> torch.Tensor:
        """For CFG: always use null_emb regardless of input."""
        B = sensor_emb.shape[0]
        return self.null_emb.expand(B, -1, -1)

    def forward(self, input_ids: torch.Tensor, sensor_emb: torch.Tensor,
                force_unconditional: bool = False) -> torch.Tensor:
        # input_ids: (B, L). sensor_emb: (B, K, d_emb) or (B, d_emb).
        sensor_emb = self._normalise_sensor(sensor_emb)
        if force_unconditional:
            sensor_emb = self._force_unconditional(sensor_emb)
        else:
            sensor_emb = self._apply_cond_dropout(sensor_emb)

        x = self.token_emb(input_ids)  # (B, L, d_model)

        for cross, self_blk in zip(self.cross_blocks, self.self_blocks):
            x = cross(x, sensor_emb)
            x = self_blk(x, self.rope_cos, self.rope_sin)

        x = self.out_norm(x)
        return self.out_proj(x)


# ── ARModernCrossModel ───────────────────────────────────────────────────
class ARModernCrossModel(nn.Module):
    """AR-modern with Flamingo-style cross-attention conditioning.

    Same I/O contract as ARModernModel but with two new behaviours:
      - sensor_emb may be (B, d_emb) [back-compat] or (B, K, d_emb) [dense].
      - sample() supports classifier-free guidance via guidance_scale.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.pad_id = cfg.vocab_size
        self.bos_id = cfg.vocab_size + 1
        self.backbone = CausalBackboneCross(cfg)

    def compute_loss(self, x0: torch.Tensor, sensor_emb: torch.Tensor,
                     pad_mask: torch.Tensor):
        B, L = x0.shape
        bos = torch.full((B, 1), self.bos_id, dtype=x0.dtype, device=x0.device)
        input_ids = torch.cat([bos, x0], dim=1)              # (B, L+1)
        logits = self.backbone(input_ids, sensor_emb)         # (B, L+1, V)
        logits = logits[:, :L, :]                             # predict x0[t] from prefix[:t+1]

        loss_per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            x0.reshape(-1),
            reduction="none",
        ).reshape(B, L)
        n_tokens = pad_mask.sum().clamp(min=1)
        loss = (loss_per_token * pad_mask).sum() / n_tokens

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct = ((preds == x0).float() * pad_mask).sum()
            tf_acc = correct / n_tokens

        return loss, {
            "loss": loss.item(),
            "acc_teacher_forced": tf_acc.item(),
            "n_tokens": n_tokens.item(),
        }

    @torch.no_grad()
    def sample(self, sensor_emb: torch.Tensor, seq_len: int,
               temperature: float = 1.0, greedy: bool = False,
               guidance_scale: float = 1.0):
        """Token-by-token AR sampling. guidance_scale > 1 enables CFG."""
        B = sensor_emb.shape[0]
        device = sensor_emb.device

        generated = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)
        do_cfg = guidance_scale != 1.0

        for _ in range(seq_len):
            cond_logits = self.backbone(generated, sensor_emb)[:, -1, :]
            if do_cfg:
                uncond_logits = self.backbone(
                    generated, sensor_emb, force_unconditional=True
                )[:, -1, :]
                # CFG: log p_w(x|c) = (1+w)*log p(x|c) - w*log p(x)
                # equivalently logits_w = uncond + (1+w)*(cond - uncond)
                next_logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
            else:
                next_logits = cond_logits
            next_logits[:, self.pad_id] = -1e9
            next_logits[:, self.bos_id] = -1e9

            if greedy:
                next_tok = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits / max(temperature, 1e-6), dim=-1)
                next_tok = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_tok], dim=1)

        return generated[:, 1:]

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
