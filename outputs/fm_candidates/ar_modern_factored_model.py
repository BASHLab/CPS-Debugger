"""AR-modern with factored (opcode / source_pc / target_pc) output head (E2).

The vocabulary is the 648 unique triples (wasm_function_id, source_pc,
target_pc). Instead of a single softmax over the joint, this model has
three hierarchical heads:

  P(opcode)            ~ 30 dense classes + PAD + BOS
  P(op1 | opcode)      ~ 344 dense classes + PAD + BOS, masked legal-set
  P(op2 | opcode, op1) ~ 480 dense classes + PAD + BOS, masked legal-set

Loss is CE_opcode + CE_op1 + CE_op2 (each averaged over non-pad
positions). At sampling time, each head's logits are masked to the
legal set for the chosen prefix, so the emitted (opcode, op1, op2)
always forms a legal triple in the vocab.

The motivation is from the Wave-3 / Wave-4 diagnostic chain: ~99.9 %
of state-1 token errors are operand-2-only, with ~7 distinct op2
values covering 80 % of all errors. Giving op2 a dedicated head with
conditioning on (opcode, op1) lets the decoder allocate capacity
where the error mass is, regardless of encoder representation.

Saved checkpoints carry `use_factored_head=True` for dispatch in
evaluate.py.
"""
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig, BASE_DIR
from ar_modern_model import CausalBlockModern, RMSNorm
from ar_model import _precompute_freqs


# Vocab geometry — fixed by `pruned_token_mapping.json`. Loaded lazily on
# first construction so importing this module is free.
_VOCAB_CACHE = {}


def _load_vocab():
    if _VOCAB_CACHE:
        return _VOCAB_CACHE
    tm_path = BASE_DIR / "train_data_full" / "pruned_token_mapping.json"
    tm = json.loads(tm_path.read_text())
    triples = tm["triples"]                                  # list of [op, op1, op2]
    vocab_size = tm["vocab_size"]                            # 648

    opcodes = sorted({t[0] for t in triples})
    op1s = sorted({t[1] for t in triples})
    op2s = sorted({t[2] for t in triples})

    op_to_idx = {v: i for i, v in enumerate(opcodes)}
    op1_to_idx = {v: i for i, v in enumerate(op1s)}
    op2_to_idx = {v: i for i, v in enumerate(op2s)}

    n_op, n_op1, n_op2 = len(opcodes), len(op1s), len(op2s)
    # specials live AFTER the real classes
    PAD_OP = n_op
    BOS_OP = n_op + 1
    PAD_OP1 = n_op1
    BOS_OP1 = n_op1 + 1
    PAD_OP2 = n_op2
    BOS_OP2 = n_op2 + 1

    # Per-token-id triples (id 0..647) plus PAD (=vocab_size) and BOS
    # (=vocab_size+1). PAD/BOS triples map to (PAD_*,PAD_*,PAD_*) and
    # (BOS_*,BOS_*,BOS_*) so loss-masking with pad_mask cleanly excludes
    # them.
    op_lut = torch.full((vocab_size + 2,), PAD_OP, dtype=torch.long)
    op1_lut = torch.full((vocab_size + 2,), PAD_OP1, dtype=torch.long)
    op2_lut = torch.full((vocab_size + 2,), PAD_OP2, dtype=torch.long)
    op_lut[vocab_size + 1] = BOS_OP
    op1_lut[vocab_size + 1] = BOS_OP1
    op2_lut[vocab_size + 1] = BOS_OP2
    for tid, (op, p1, p2) in enumerate(triples):
        op_lut[tid] = op_to_idx[op]
        op1_lut[tid] = op1_to_idx[p1]
        op2_lut[tid] = op2_to_idx[p2]

    # Legal masks (bool):
    #   legal_op1[op_idx, op1_idx] = True if (op_val, op1_val) is a prefix
    #     of any real triple.
    #   legal_op2[op_idx, op1_idx, op2_idx] = True if (op,op1,op2) is in vocab.
    n_op_total = n_op + 2
    n_op1_total = n_op1 + 2
    n_op2_total = n_op2 + 2
    legal_op1 = torch.zeros((n_op_total, n_op1_total), dtype=torch.bool)
    legal_op2 = torch.zeros((n_op_total, n_op1_total, n_op2_total), dtype=torch.bool)
    triple_to_tid = torch.full((n_op_total, n_op1_total, n_op2_total), -1, dtype=torch.long)
    for tid, (op, p1, p2) in enumerate(triples):
        oi, p1i, p2i = op_to_idx[op], op1_to_idx[p1], op2_to_idx[p2]
        legal_op1[oi, p1i] = True
        legal_op2[oi, p1i, p2i] = True
        triple_to_tid[oi, p1i, p2i] = tid

    _VOCAB_CACHE.update({
        "vocab_size": vocab_size,
        "n_op": n_op, "n_op1": n_op1, "n_op2": n_op2,
        "n_op_total": n_op_total,
        "n_op1_total": n_op1_total,
        "n_op2_total": n_op2_total,
        "op_lut": op_lut,
        "op1_lut": op1_lut,
        "op2_lut": op2_lut,
        "legal_op1": legal_op1,
        "legal_op2": legal_op2,
        "triple_to_tid": triple_to_tid,
        "PAD_OP": PAD_OP, "BOS_OP": BOS_OP,
        "PAD_OP1": PAD_OP1, "BOS_OP1": BOS_OP1,
        "PAD_OP2": PAD_OP2, "BOS_OP2": BOS_OP2,
    })
    return _VOCAB_CACHE


class FactoredBackbone(nn.Module):
    """Same as CausalBackboneModern through the transformer stack, but
    ending at `out_norm` and returning hidden states (B, L, d_model)
    instead of vocab logits."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.pad_id = cfg.vocab_size
        self.bos_id = cfg.vocab_size + 1
        vocab_total = cfg.vocab_size + 2
        self.token_emb = nn.Embedding(vocab_total, cfg.d_model)
        d_sensor = getattr(cfg, "d_sensor_emb", cfg.d_model)
        self.sensor_proj = nn.Linear(d_sensor, cfg.d_model, bias=False)
        self.blocks = nn.ModuleList([
            CausalBlockModern(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.out_norm = RMSNorm(cfg.d_model)
        # No out_proj here — heads consume out_norm output.
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

    def forward(self, input_ids: torch.Tensor, sensor_emb: torch.Tensor):
        B, L_in = input_ids.shape
        if sensor_emb.dim() == 3:
            sensor_emb = sensor_emb.mean(dim=1)
        x = self.token_emb(input_ids)
        sensor_tok = self.sensor_proj(sensor_emb).unsqueeze(1)
        x = torch.cat([sensor_tok, x], dim=1)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        x = x[:, 1:, :]              # strip sensor-prefix output
        return self.out_norm(x)      # (B, L_in, d_model)


class ARModernFactoredModel(nn.Module):
    """AR-modern decoder with factored (opcode/op1/op2) output head."""

    OP_EMB_DIM = 64
    OP1_EMB_DIM = 64

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.pad_id = cfg.vocab_size
        self.bos_id = cfg.vocab_size + 1
        self.backbone = FactoredBackbone(cfg)

        v = _load_vocab()
        self.n_op_total = v["n_op_total"]
        self.n_op1_total = v["n_op1_total"]
        self.n_op2_total = v["n_op2_total"]

        d = cfg.d_model
        self.opcode_head = nn.Linear(d, self.n_op_total, bias=False)
        self.op_cond_emb = nn.Embedding(self.n_op_total, self.OP_EMB_DIM)
        self.op1_head = nn.Linear(d + self.OP_EMB_DIM, self.n_op1_total, bias=False)
        self.op1_cond_emb = nn.Embedding(self.n_op1_total, self.OP1_EMB_DIM)
        self.op2_head = nn.Linear(
            d + self.OP_EMB_DIM + self.OP1_EMB_DIM, self.n_op2_total, bias=False)

        for m in (self.opcode_head, self.op1_head, self.op2_head):
            nn.init.xavier_uniform_(m.weight)
        nn.init.normal_(self.op_cond_emb.weight, std=0.02)
        nn.init.normal_(self.op1_cond_emb.weight, std=0.02)

        # Vocab lookups + legal-set masks as buffers (move with .to(device)).
        self.register_buffer("op_lut", v["op_lut"], persistent=False)
        self.register_buffer("op1_lut", v["op1_lut"], persistent=False)
        self.register_buffer("op2_lut", v["op2_lut"], persistent=False)
        self.register_buffer("legal_op1", v["legal_op1"], persistent=False)
        self.register_buffer("legal_op2", v["legal_op2"], persistent=False)
        self.register_buffer("triple_to_tid", v["triple_to_tid"], persistent=False)
        self.PAD_OP = v["PAD_OP"]; self.BOS_OP = v["BOS_OP"]
        self.PAD_OP1 = v["PAD_OP1"]; self.BOS_OP1 = v["BOS_OP1"]
        self.PAD_OP2 = v["PAD_OP2"]; self.BOS_OP2 = v["BOS_OP2"]

    def _mask_specials(self, logits: torch.Tensor, pad_idx: int, bos_idx: int):
        logits = logits.clone()
        logits[..., pad_idx] = -1e9
        logits[..., bos_idx] = -1e9
        return logits

    def compute_loss(self, x0: torch.Tensor, sensor_emb: torch.Tensor,
                     pad_mask: torch.Tensor):
        B, L = x0.shape
        bos = torch.full((B, 1), self.bos_id, dtype=x0.dtype, device=x0.device)
        input_ids = torch.cat([bos, x0], dim=1)
        h = self.backbone(input_ids, sensor_emb)[:, :L, :]   # (B, L, d)

        gt_op = self.op_lut[x0]                              # (B, L) op_idx
        gt_op1 = self.op1_lut[x0]
        gt_op2 = self.op2_lut[x0]

        # Stage 1: opcode logits
        op_logits = self.opcode_head(h)                       # (B, L, n_op_total)
        # Stage 2: condition op1 head on ground-truth opcode embedding
        op_emb = self.op_cond_emb(gt_op)                      # (B, L, OP_EMB_DIM)
        op1_logits = self.op1_head(torch.cat([h, op_emb], dim=-1))
        # Stage 3: condition op2 head on (gt opcode, gt op1) embeddings
        op1_emb = self.op1_cond_emb(gt_op1)
        op2_logits = self.op2_head(torch.cat([h, op_emb, op1_emb], dim=-1))

        # CE per stage; weight by pad_mask
        def _masked_ce(logits, target):
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
                reduction="none",
            ).reshape(B, L)
            n_tokens = pad_mask.sum().clamp(min=1)
            return (ce * pad_mask).sum() / n_tokens

        loss_op = _masked_ce(op_logits, gt_op)
        loss_op1 = _masked_ce(op1_logits, gt_op1)
        loss_op2 = _masked_ce(op2_logits, gt_op2)
        loss = loss_op + loss_op1 + loss_op2

        with torch.no_grad():
            pred_op = op_logits.argmax(dim=-1)
            pred_op1 = op1_logits.argmax(dim=-1)
            pred_op2 = op2_logits.argmax(dim=-1)
            joint_correct = ((pred_op == gt_op) & (pred_op1 == gt_op1) & (pred_op2 == gt_op2)).float()
            n_tokens = pad_mask.sum().clamp(min=1)
            tf_acc = (joint_correct * pad_mask).sum() / n_tokens

        return loss, {
            "loss": loss.item(),
            "loss_opcode": loss_op.item(),
            "loss_op1": loss_op1.item(),
            "loss_op2": loss_op2.item(),
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
            h = self.backbone(generated, sensor_emb)[:, -1, :]   # (B, d)
            # opcode
            op_logits = self.opcode_head(h)
            op_logits = self._mask_specials(op_logits, self.PAD_OP, self.BOS_OP)
            op_idx = self._pick(op_logits, temperature, greedy)   # (B,)
            # op1 | opcode
            op_emb = self.op_cond_emb(op_idx)
            op1_logits = self.op1_head(torch.cat([h, op_emb], dim=-1))
            # mask to legal_op1[op_idx]
            mask1 = ~self.legal_op1[op_idx]                       # (B, n_op1_total)
            op1_logits = op1_logits.masked_fill(mask1, -1e9)
            op1_idx = self._pick(op1_logits, temperature, greedy)
            # op2 | opcode, op1
            op1_emb = self.op1_cond_emb(op1_idx)
            op2_logits = self.op2_head(torch.cat([h, op_emb, op1_emb], dim=-1))
            mask2 = ~self.legal_op2[op_idx, op1_idx]              # (B, n_op2_total)
            op2_logits = op2_logits.masked_fill(mask2, -1e9)
            op2_idx = self._pick(op2_logits, temperature, greedy)
            # Resolve to token id; -1 means illegal (sampling guard)
            tids = self.triple_to_tid[op_idx, op1_idx, op2_idx]   # (B,)
            # Fall back to a uniform legal triple under any illegal idx (defensive).
            illegal = tids < 0
            if illegal.any():
                # Replace with the lowest-id legal triple for that (op_idx, op1_idx)
                for b in torch.where(illegal)[0].tolist():
                    legal_ids = torch.where(self.legal_op2[op_idx[b], op1_idx[b]])[0]
                    if len(legal_ids) > 0:
                        tids[b] = self.triple_to_tid[op_idx[b], op1_idx[b], legal_ids[0]]
                    else:
                        tids[b] = 0
            generated = torch.cat([generated, tids.unsqueeze(1)], dim=1)
        return generated[:, 1:]

    @staticmethod
    def _pick(logits: torch.Tensor, temperature: float, greedy: bool):
        if greedy:
            return logits.argmax(dim=-1)
        probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
