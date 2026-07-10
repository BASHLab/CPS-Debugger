"""AR-modern with FSM-state-token prefix + classifier-free guidance (E5).

Extends the prefix-token decoder: prepends a learned state-token to the
existing sensor-emb prefix. The state-token comes from a frozen linear
probe's prediction on the BL-7 sensor embedding (probe was trained on
ground-truth FSM state during E5 prep).

State vocabulary: {0=SWINGUP, 1=BALANCE, 2=RESET, 3=NO_COND}.
Conditioning dropout (10% by default) replaces the predicted state with
NO_COND during training, enabling classifier-free guidance at inference.

Inference CFG:
    logits_final = uncond + γ · (cond - uncond)
where `cond` uses the probe-predicted state and `uncond` uses NO_COND.
Sensor-emb is identical in both passes — only the state-token swaps.

Reuses the rest of ARModernModel's machinery (RMSNorm + SwiGLU + QK-Norm
+ Muon-compatible 2D weights inside `.blocks.`). Saves checkpoints
distinguishable from vanilla ARModernModel via `use_state_token=True`.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig
from ar_modern_model import CausalBackboneModern, RMSNorm
from ar_model import _precompute_freqs


# State vocabulary: 0=SWINGUP, 1=BALANCE, 2=RESET, 3=NO_COND
STATE_VOCAB_SIZE = 4
NO_COND_ID = 3


class CausalBackboneStateModern(nn.Module):
    """Variant of CausalBackboneModern with an additional state-token prefix.

    Prefix layout:  [state_tok, sensor_tok, BOS, x_0, ..., x_{L-1}]
    Output is the AR logits for predicting [x_0, ..., x_{L-1}] given the
    prefix and BOS — strip the first 2 (state + sensor) prefix tokens
    after the backbone, then the BOS-conditioned next-token logits start
    at index 0.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.pad_id = cfg.vocab_size
        self.bos_id = cfg.vocab_size + 1
        vocab_total = cfg.vocab_size + 2

        self.token_emb = nn.Embedding(vocab_total, cfg.d_model)
        d_sensor = getattr(cfg, "d_sensor_emb", cfg.d_model)
        self.sensor_proj = nn.Linear(d_sensor, cfg.d_model, bias=False)
        # State token embedding: states + NO_COND
        self.state_emb = nn.Embedding(STATE_VOCAB_SIZE, cfg.d_model)

        # Reuse the same block class as ARModernModel
        from ar_modern_model import CausalBlockModern
        self.blocks = nn.ModuleList([
            CausalBlockModern(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.out_norm = RMSNorm(cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, vocab_total, bias=False)

        # RoPE buffers — prefix is 2 tokens (state + sensor), then BOS + trace
        max_pos = 2 + 1 + cfg.max_seq_len
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

    def forward(self, input_ids: torch.Tensor, sensor_emb: torch.Tensor,
                state_id: torch.Tensor) -> torch.Tensor:
        # Prefix-decoder mean-pool stopgap for dense (B, K, d_emb) BL-7 emb
        if sensor_emb.dim() == 3:
            sensor_emb = sensor_emb.mean(dim=1)
        B, L_in = input_ids.shape
        x = self.token_emb(input_ids)
        state_tok = self.state_emb(state_id).unsqueeze(1)             # (B, 1, d_model)
        sensor_tok = self.sensor_proj(sensor_emb).unsqueeze(1)        # (B, 1, d_model)
        x = torch.cat([state_tok, sensor_tok, x], dim=1)

        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)

        x = x[:, 2:, :]                                                # strip both prefix tokens
        x = self.out_norm(x)
        return self.out_proj(x)


class ARModernStateModel(nn.Module):
    """AR-modern + state-token prefix + classifier-free guidance.

    Forward signature mirrors ARModernModel but takes an extra `state_id`
    tensor of shape (B,) long, in [0..STATE_VOCAB_SIZE-1].
    """

    def __init__(self, cfg: ModelConfig, cond_dropout_p: float = 0.10):
        super().__init__()
        self.cfg = cfg
        self.pad_id = cfg.vocab_size
        self.bos_id = cfg.vocab_size + 1
        self.cond_dropout_p = cond_dropout_p
        self.backbone = CausalBackboneStateModern(cfg)

    def _apply_state_dropout(self, state_id: torch.Tensor) -> torch.Tensor:
        if not self.training or self.cond_dropout_p <= 0:
            return state_id
        drop = (torch.rand(state_id.shape, device=state_id.device)
                < self.cond_dropout_p)
        return torch.where(drop, torch.full_like(state_id, NO_COND_ID), state_id)

    def compute_loss(self, x0: torch.Tensor, sensor_emb: torch.Tensor,
                     pad_mask: torch.Tensor, state_id: torch.Tensor):
        """state_id: (B,) long in [0..3]."""
        state_id = self._apply_state_dropout(state_id)
        B, L = x0.shape
        bos = torch.full((B, 1), self.bos_id, dtype=x0.dtype, device=x0.device)
        input_ids = torch.cat([bos, x0], dim=1)               # (B, L+1)
        logits = self.backbone(input_ids, sensor_emb, state_id)  # (B, L+1, V)
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
    def sample(self, sensor_emb: torch.Tensor, state_id: torch.Tensor,
               seq_len: int, temperature: float = 1.0, greedy: bool = False,
               guidance_scale: float = 1.0):
        """Generate trace tokens. If guidance_scale != 1.0, runs CFG:
        2 forward passes per step (state_id vs NO_COND), mixed by γ.
        """
        B = sensor_emb.shape[0]
        device = sensor_emb.device
        do_cfg = guidance_scale != 1.0
        no_cond = torch.full_like(state_id, NO_COND_ID)

        generated = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)
        for _ in range(seq_len):
            cond_logits = self.backbone(generated, sensor_emb, state_id)[:, -1, :]
            if do_cfg:
                uncond_logits = self.backbone(generated, sensor_emb, no_cond)[:, -1, :]
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

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
