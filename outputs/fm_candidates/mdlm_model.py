"""BL-2 Conditional MDLM — DiT backbone with absorbing-state diffusion.

Architecture:
  - Token embedding (vocab + MASK)
  - Timestep embedding (diffusion time σ via MLP)
  - Sensor prefix (1 token, NEVER masked — LLaDA SFT pattern)
  - Bidirectional DiT blocks with AdaLN + RoPE
  - Log-linear noise schedule, SUBS parameterization
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


# ── RoPE (Rotary Position Embeddings) ─────────────────────────────────
def _precompute_freqs(dim: int, max_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, freqs)  # (max_len, dim//2)
    return torch.cos(freqs), torch.sin(freqs)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply RoPE to (B, H, L, D) tensor."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos = cos[:x.shape[-2], :d].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[-2], :d].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ── AdaLN (Adaptive Layer Norm) ───────────────────────────────────────
class AdaLN(nn.Module):
    """Adaptive LayerNorm conditioned on sigma embedding."""

    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 3 * d_model)  # scale, shift, gate

    def forward(self, x, c):
        """x: (B, L, D), c: (B, cond_dim) → (x_normed, gate)"""
        scale, shift, gate = self.proj(c).unsqueeze(1).chunk(3, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x, gate


# ── DiT Block ─────────────────────────────────────────────────────────
class DiTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, cond_dim: int, dropout: float):
        super().__init__()
        self.adaln_attn = AdaLN(d_model, cond_dim)
        self.adaln_ff = AdaLN(d_model, cond_dim)

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, c, rope_cos, rope_sin):
        # Self-attention with AdaLN
        x_norm, gate_attn = self.adaln_attn(x, c)
        B, L, D = x_norm.shape
        qkv = self.qkv(x_norm).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, H, L, Dh)
        # Apply RoPE
        q = _apply_rope(q, rope_cos, rope_sin)
        k = _apply_rope(k, rope_cos, rope_sin)
        # Scaled dot-product attention (uses Flash Attention when available)
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0 if not self.training else self.attn_drop.p)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        attn_out = self.out_proj(attn_out)
        x = x + gate_attn * attn_out

        # Feed-forward with AdaLN
        x_norm, gate_ff = self.adaln_ff(x, c)
        x = x + gate_ff * self.ff(x_norm)
        return x


# ── DiT Backbone ──────────────────────────────────────────────────────
class DiTBackbone(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        vocab_total = cfg.vocab_size + 1  # +1 for MASK token
        cond_dim = cfg.d_model

        # Embeddings
        self.token_emb = nn.Embedding(vocab_total, cfg.d_model)
        self.sigma_emb = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Sensor prefix projection (from sensor encoder output)
        self.sensor_proj = nn.Linear(cfg.d_model, cfg.d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cond_dim, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])

        # Output head
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, vocab_total)

        # RoPE
        cos, sin = _precompute_freqs(cfg.d_model // cfg.n_heads, cfg.max_seq_len + cfg.n_sensor_tokens)
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

    def forward(self, x_ids: torch.Tensor, sigma: torch.Tensor,
                sensor_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_ids: (B, L) — token IDs (possibly masked)
            sigma: (B,) — diffusion noise level
            sensor_emb: (B, d_model) — precomputed sensor embedding
        Returns:
            logits: (B, L, vocab_total) — predictions for trace positions only
        """
        B, L = x_ids.shape

        # Token embeddings
        x = self.token_emb(x_ids)  # (B, L, D)

        # Prepend sensor token
        sensor_tok = self.sensor_proj(sensor_emb).unsqueeze(1)  # (B, 1, D)
        x = torch.cat([sensor_tok, x], dim=1)  # (B, 1+L, D)

        # Sigma conditioning
        c = self.sigma_emb(sigma.unsqueeze(-1))  # (B, D)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c, self.rope_cos, self.rope_sin)

        # Output: remove sensor prefix, project to vocab
        x = x[:, 1:, :]  # (B, L, D) — trace positions only
        x = self.out_norm(x)
        logits = self.out_proj(x)  # (B, L, vocab_total)
        return logits


# ── Noise Schedule ────────────────────────────────────────────────────
class LogLinearNoise:
    """Log-linear noise schedule: σ(t) = -log(1 - (1-ε)t), t ∈ [0,1]."""

    def __init__(self, eps: float = 1e-3):
        self.eps = eps

    def __call__(self, t: torch.Tensor):
        """Returns (sigma, dsigma) both shape (B,)."""
        sigma = -torch.log1p(-(1 - self.eps) * t)
        dsigma = (1 - self.eps) / (1 - (1 - self.eps) * t)
        return sigma, dsigma


class CosineNoise:
    """Cosine noise schedule."""

    def __init__(self, eps: float = 1e-3):
        self.eps = eps

    def __call__(self, t: torch.Tensor):
        s = 0.008
        f_t = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        f_0 = math.cos(s / (1 + s) * math.pi / 2) ** 2
        sigma = -torch.log(f_t / f_0 + 1e-8)
        dsigma = (math.pi * (1 - self.eps)) / (2 * (1 + s)) * torch.tan(
            (t + s) / (1 + s) * math.pi / 2
        )
        return sigma, dsigma


def get_noise_schedule(name: str):
    if name == "loglinear":
        return LogLinearNoise()
    elif name == "cosine":
        return CosineNoise()
    raise ValueError(f"Unknown noise schedule: {name}")


# ── MDLM Diffusion ───────────────────────────────────────────────────
class MDLM(nn.Module):
    """Masked Diffusion Language Model with sensor conditioning.

    Forward process: absorbing state (tokens → MASK)
    Reverse: iterative denoising, unmasking highest-confidence positions
    Loss: SUBS parameterization (Rao-Blackwellized ELBO)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.mask_id = cfg.vocab_size  # MASK = last token in vocab_total
        self.backbone = DiTBackbone(cfg)
        self.noise = get_noise_schedule(cfg.noise_schedule)

    def q_xt(self, x0: torch.Tensor, move_chance: torch.Tensor):
        """Forward diffusion: randomly mask tokens.

        Args:
            x0: (B, L) — clean tokens
            move_chance: (B, 1) — probability of masking each token
        Returns:
            xt: (B, L) — noisy (partially masked) tokens
        """
        move = torch.rand_like(x0.float()) < move_chance
        return torch.where(move, self.mask_id, x0)

    def compute_loss(self, x0: torch.Tensor, sensor_emb: torch.Tensor,
                     pad_mask: torch.Tensor):
        """Compute SUBS diffusion loss (Rao-Blackwellized ELBO).

        Args:
            x0: (B, L) — clean token IDs
            sensor_emb: (B, d_model) — precomputed sensor embeddings
            pad_mask: (B, L) — 1 for real tokens, 0 for padding
        Returns:
            loss: scalar
            metrics: dict of logging metrics
        """
        B, L = x0.shape

        # Sample time uniformly
        t = torch.rand(B, device=x0.device)
        sigma, dsigma = self.noise(t)
        move_chance = 1 - torch.exp(-sigma)  # (B,)

        # Corrupt tokens
        xt = self.q_xt(x0, move_chance.unsqueeze(1))

        # Predict
        logits = self.backbone(xt, sigma, sensor_emb)  # (B, L, V+1)

        # SUBS loss: only compute on masked positions
        # log p(x0 | xt) at masked positions, weighted by dsigma / expm1(sigma)
        log_probs = F.log_softmax(logits, dim=-1)  # (B, L, V+1)

        # Gather log prob of true token at each position
        log_p_x0 = log_probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)  # (B, L)

        # SUBS weight: dsigma / (exp(sigma) - 1)
        weight = dsigma / torch.expm1(sigma)  # (B,)

        # Loss per token (only masked positions contribute)
        is_masked = (xt == self.mask_id).float()  # (B, L)
        token_loss = -log_p_x0 * is_masked * pad_mask  # (B, L)

        # Normalize: average over masked tokens, weight by dsigma
        n_masked = (is_masked * pad_mask).sum(dim=1).clamp(min=1)  # (B,)
        per_sample_loss = (token_loss.sum(dim=1) / n_masked) * weight  # (B,)
        loss = per_sample_loss.mean()

        # Metrics
        with torch.no_grad():
            # Token accuracy on masked positions
            preds = logits.argmax(dim=-1)  # (B, L)
            correct = ((preds == x0) * is_masked * pad_mask).sum()
            total_masked = (is_masked * pad_mask).sum()
            acc = correct / total_masked.clamp(min=1)

        return loss, {
            "loss": loss.item(),
            "acc_masked": acc.item(),
            "mean_sigma": sigma.mean().item(),
            "frac_masked": (is_masked * pad_mask).sum().item() / pad_mask.sum().item(),
        }

    @torch.no_grad()
    def sample(self, sensor_emb: torch.Tensor, seq_len: int,
               n_steps: int = 256, temperature: float = 1.0):
        """Sample traces via iterative denoising.

        Args:
            sensor_emb: (B, d_model) — precomputed sensor embeddings
            seq_len: trace length to generate
            n_steps: number of denoising steps
            temperature: sampling temperature
        Returns:
            x: (B, seq_len) — generated token IDs
        """
        B = sensor_emb.shape[0]
        device = sensor_emb.device

        # Start fully masked
        x = torch.full((B, seq_len), self.mask_id, dtype=torch.long, device=device)

        # Timesteps from 1 to eps
        eps = 1e-5
        timesteps = torch.linspace(1.0, eps, n_steps + 1, device=device)

        for i in range(n_steps):
            t = timesteps[i].expand(B)
            sigma, _ = self.noise(t)

            logits = self.backbone(x, sigma, sensor_emb)  # (B, L, V+1)

            # Only update masked positions
            is_masked = (x == self.mask_id)

            if temperature != 1.0:
                logits = logits / temperature

            probs = F.softmax(logits, dim=-1)

            # For masked positions, sample or take argmax
            # Use confidence-based unmasking: unmask fraction of positions
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(eps, device=device)
            sigma_next, _ = self.noise(t_next.expand(B))
            # Fraction to remain masked at next step
            frac_masked_next = 1 - torch.exp(-sigma_next[0])
            n_to_remain_masked = max(0, int(frac_masked_next * seq_len))

            # Get confidence scores for masked positions
            max_probs = probs.max(dim=-1).values  # (B, L)
            max_probs = torch.where(is_masked, max_probs, torch.tensor(float('inf'), device=device))

            # Find threshold: keep n_to_remain_masked lowest-confidence positions masked
            if n_to_remain_masked > 0 and n_to_remain_masked < is_masked.sum(dim=1).min():
                sorted_probs, _ = max_probs.sort(dim=1)
                threshold = sorted_probs[:, n_to_remain_masked - 1].unsqueeze(1)
                unmask = is_masked & (max_probs > threshold)
            else:
                unmask = is_masked  # unmask everything

            # Sample tokens for unmasked positions
            sampled = torch.multinomial(
                probs.view(-1, probs.shape[-1]), 1
            ).view(B, seq_len)

            x = torch.where(unmask, sampled, x)

        return x

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
