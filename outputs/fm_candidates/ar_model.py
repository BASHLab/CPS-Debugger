"""Autoregressive baseline — causal Transformer w/ sensor prefix.

Reference point for the MDLM diffusion objective. Same sensor-encoder,
same parquets, same eval harness. Differences vs mdlm_model.py:
  - Causal self-attention (no bidirectional context)
  - Cross-entropy next-token loss, shift-by-1 (no diffusion time, no AdaLN)
  - Pre-LN transformer blocks (no AdaLN since no time conditioning)
  - Vocab adds BOS on top of the existing PAD token
  - Autoregressive greedy / temperature sampling

Vocab layout (matches dataset's pad convention):
  - ids 0 .. V-1          : real tokens (V = cfg.vocab_size)
  - id V                  : PAD       (= cfg.mask_token_id, produced by TraceDataset padding)
  - id V+1                : BOS
  vocab_total = V + 2
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


# ── RoPE (same as mdlm_model.py) ───────────────────────────────────────
def _precompute_freqs(dim: int, max_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos = cos[:x.shape[-2], :d].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[-2], :d].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ── Causal Transformer block (pre-LN) ─────────────────────────────────
class CausalBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop_p = dropout

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, rope_cos, rope_sin):
        # Self-attention (causal)
        x_norm = self.norm_attn(x)
        B, L, D = x_norm.shape
        qkv = self.qkv(x_norm).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # (B, H, L, Dh)
        q = _apply_rope(q, rope_cos, rope_sin)
        k = _apply_rope(k, rope_cos, rope_sin)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            is_causal=True,
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        x = x + self.out_proj(attn_out)

        # FFN
        x = x + self.ff(self.norm_ff(x))
        return x


# ── AR backbone ───────────────────────────────────────────────────────
class CausalBackbone(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.pad_id = cfg.vocab_size           # matches TraceDataset padding
        self.bos_id = cfg.vocab_size + 1
        vocab_total = cfg.vocab_size + 2

        self.token_emb = nn.Embedding(vocab_total, cfg.d_model)
        # Sensor prefix projection (sensor_emb → d_model, to match MDLM layout)
        self.sensor_proj = nn.Linear(cfg.d_model, cfg.d_model)

        self.blocks = nn.ModuleList([
            CausalBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, vocab_total)

        # RoPE covers sensor + BOS + max_seq_len tokens
        # (n_sensor_tokens is 1 in cfg; +1 for BOS; +max_seq_len for tokens)
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
        """
        Args:
            input_ids: (B, L_in) — [BOS, x0, x1, ..., x_{L-1}] (caller prepends BOS)
            sensor_emb: (B, d_model)
        Returns:
            logits: (B, L_in, vocab_total) — position k predicts token k+1
                    (sensor prefix is stripped before returning)
        """
        B, L_in = input_ids.shape
        x = self.token_emb(input_ids)                               # (B, L_in, D)
        sensor_tok = self.sensor_proj(sensor_emb).unsqueeze(1)      # (B, 1, D)
        x = torch.cat([sensor_tok, x], dim=1)                       # (B, 1+L_in, D)

        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)

        x = x[:, 1:, :]  # drop sensor prefix → (B, L_in, D)
        x = self.out_norm(x)
        return self.out_proj(x)


# ── AR Model ──────────────────────────────────────────────────────────
class ARModel(nn.Module):
    """Autoregressive baseline with sensor conditioning."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.pad_id = cfg.vocab_size
        self.bos_id = cfg.vocab_size + 1
        self.backbone = CausalBackbone(cfg)

    def compute_loss(self, x0: torch.Tensor, sensor_emb: torch.Tensor,
                     pad_mask: torch.Tensor):
        """Next-token CE loss with BOS shift.

        Args:
            x0: (B, L) — trace tokens, padded with pad_id
            sensor_emb: (B, d_model)
            pad_mask: (B, L) — 1 for real tokens, 0 for padding
        Returns:
            loss: scalar
            metrics: dict
        """
        B, L = x0.shape
        bos = torch.full((B, 1), self.bos_id, dtype=x0.dtype, device=x0.device)
        # Input: [BOS, x0, x1, ..., x_{L-2}] (length L — last token not an input)
        # Target: [x0, x1, ..., x_{L-1}] — same positions as output logits
        # Actually simpler: input = [BOS, x0, ..., x_{L-1}], logits at positions 0..L,
        # targets = [x0, ..., x_{L-1}, ignored]. Slice logits[:, :L] to align with x0.
        input_ids = torch.cat([bos, x0], dim=1)                     # (B, 1+L)

        logits = self.backbone(input_ids, sensor_emb)               # (B, 1+L, V+2)
        logits = logits[:, :L, :]                                   # (B, L, V+2) — predict x0..x_{L-1}

        # CE on real-token positions only
        loss_per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            x0.reshape(-1),
            reduction="none",
        ).reshape(B, L)                                             # (B, L)
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
        """Autoregressive generation conditioned on sensor embedding.

        Args:
            sensor_emb: (B, d_model)
            seq_len: number of tokens to generate
            temperature: sampling temperature (ignored if greedy)
            greedy: if True, argmax; else multinomial sample
        Returns:
            x: (B, seq_len) of generated token IDs
        """
        B = sensor_emb.shape[0]
        device = sensor_emb.device

        generated = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)
        for _ in range(seq_len):
            logits = self.backbone(generated, sensor_emb)            # (B, cur_len, V+2)
            next_logits = logits[:, -1, :]                           # (B, V+2)
            # Block PAD and BOS from being generated
            next_logits[:, self.pad_id] = -1e9
            next_logits[:, self.bos_id] = -1e9
            if greedy:
                next_tok = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits / max(temperature, 1e-6), dim=-1)
                next_tok = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_tok], dim=1)

        return generated[:, 1:]  # drop BOS

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
