"""AR-modern baseline — causal Transformer with 2026 consensus upgrades.

Same I/O contract and sensor-as-virtual-token conditioning as ar_model.py,
but with three architectural upgrades validated by the modded-nanogpt speedrun
records (Keller Jordan, MIT) and the 2025 small-LM literature (Raschka survey,
OLMo 2, Gemma 3, Mamba-3 paper):

  1. RMSNorm            — replaces LayerNorm (Llama / OLMo / Gemma / Qwen / Mamba-3)
  2. SwiGLU FFN         — replaces GELU MLP (same lineage)
  3. QK-Norm            — RMSNorm on Q/K before RoPE (OLMo 2, Gemma 3, Mamba-3,
                          modded-nanogpt records/2024-10-14_ModernArch)

The Muon optimizer (~1.3-1.5× sample efficiency over AdamW on 2D weights) lives
in the trainer, not here.

Vocab layout (matches dataset's pad convention) — identical to ar_model.py:
  - ids 0 .. V-1          : real tokens (V = cfg.vocab_size)
  - id V                  : PAD       (= cfg.mask_token_id, produced by TraceDataset padding)
  - id V+1                : BOS
  vocab_total = V + 2

SwiGLU FFN sizing: SwiGLU has 3 projections (gate, up, down) vs GELU's 2
(in, out). To keep parameter count roughly comparable to ar_model.py at the
same cfg.d_ff, we shrink the inner dim to 2/3 × d_ff rounded to a multiple of
64 — Llama convention.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig
from ar_model import _precompute_freqs, _apply_rope


# ── RMSNorm ──────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 for numerical stability under bf16 autocast
        x_fp32 = x.float()
        rms = x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x_fp32 * rms).to(x.dtype) * self.weight


# ── SwiGLU FFN ───────────────────────────────────────────────────────────
class SwiGLU(nn.Module):
    """Llama-style SwiGLU: down(silu(gate(x)) * up(x))."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        # 2/3 × d_ff rounded to multiple of 64 — matches Llama's param-budget convention
        d_inner = max(64, int(d_ff * 2 / 3) // 64 * 64)
        self.gate = nn.Linear(d_model, d_inner, bias=False)
        self.up = nn.Linear(d_model, d_inner, bias=False)
        self.down = nn.Linear(d_inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


# ── Causal block: pre-norm RMS + QK-Norm + SwiGLU ────────────────────────
class CausalBlockModern(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm_attn = RMSNorm(d_model)
        self.norm_ff = RMSNorm(d_model)

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop_p = dropout

        # QK-Norm: per-head RMSNorm applied to Q and K *before* RoPE
        # (OLMo 2 / Gemma 3 ordering — stabilizes attention logits at bf16)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.ff = SwiGLU(d_model, d_ff, dropout)

    def forward(self, x, rope_cos, rope_sin):
        x_norm = self.norm_attn(x)
        B, L, D = x_norm.shape
        qkv = self.qkv(x_norm).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # (B, H, L, Dh)

        # QK-Norm (per-head), then RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = _apply_rope(q, rope_cos, rope_sin)
        k = _apply_rope(k, rope_cos, rope_sin)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            is_causal=True,
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        x = x + self.out_proj(attn_out)
        x = x + self.ff(self.norm_ff(x))
        return x


# ── Backbone ─────────────────────────────────────────────────────────────
class CausalBackboneModern(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.pad_id = cfg.vocab_size
        self.bos_id = cfg.vocab_size + 1
        vocab_total = cfg.vocab_size + 2

        self.token_emb = nn.Embedding(vocab_total, cfg.d_model)
        # sensor_proj maps the encoder-output dim (d_sensor_emb) to d_model.
        # Chronos-2 and MOMENT pre-existing setups have d_sensor_emb == d_model
        # (both = 512), so the projection is square; back-compat preserved.
        # BL-7 uses d_sensor_emb = 256, so this becomes a 256 → 512 linear.
        d_sensor = getattr(cfg, "d_sensor_emb", cfg.d_model)
        self.sensor_proj = nn.Linear(d_sensor, cfg.d_model, bias=False)

        self.blocks = nn.ModuleList([
            CausalBlockModern(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.out_norm = RMSNorm(cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, vocab_total, bias=False)

        max_pos = cfg.n_sensor_tokens + 1 + cfg.max_seq_len
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

    def forward(self, input_ids: torch.Tensor, sensor_emb: torch.Tensor) -> torch.Tensor:
        B, L_in = input_ids.shape
        x = self.token_emb(input_ids)

        # Prefix-token decoder expects (B, d_sensor_emb). If we're handed a
        # dense (B, K, d_sensor_emb) BL-7 embedding, mean-pool to single vector.
        # The dense-cross-attn decoder (ARModernCrossModel) handles K tokens
        # natively; this branch is the stopgap for prefix-token consumers.
        if sensor_emb.dim() == 3:
            sensor_emb = sensor_emb.mean(dim=1)

        sensor_tok = self.sensor_proj(sensor_emb).unsqueeze(1)
        x = torch.cat([sensor_tok, x], dim=1)

        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)

        x = x[:, 1:, :]
        x = self.out_norm(x)
        return self.out_proj(x)


# ── ARModernModel ────────────────────────────────────────────────────────
class ARModernModel(nn.Module):
    """AR baseline + RMSNorm + SwiGLU + QK-Norm. Same I/O as ARModel."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.pad_id = cfg.vocab_size
        self.bos_id = cfg.vocab_size + 1
        self.backbone = CausalBackboneModern(cfg)

    def compute_loss(self, x0: torch.Tensor, sensor_emb: torch.Tensor,
                     pad_mask: torch.Tensor):
        B, L = x0.shape
        bos = torch.full((B, 1), self.bos_id, dtype=x0.dtype, device=x0.device)
        input_ids = torch.cat([bos, x0], dim=1)
        logits = self.backbone(input_ids, sensor_emb)
        logits = logits[:, :L, :]

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
               temperature: float = 1.0, greedy: bool = False):
        B = sensor_emb.shape[0]
        device = sensor_emb.device

        generated = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)
        for _ in range(seq_len):
            logits = self.backbone(generated, sensor_emb)
            next_logits = logits[:, -1, :]
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
