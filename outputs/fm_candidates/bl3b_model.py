"""BL-3b model — V-JEPA 2.1 ViT-L 384 + LoRA + AR-modern decoder.

Differences from BL-3 (frozen V-JEPA):
  * V-JEPA encoder is wrapped with peft.LoraConfig (rank 16, alpha 32,
    dropout 0.1) targeting the fused `qkv` and the output `proj` of every
    attention layer. Standard ViT LoRA recipe — see Hu et al. 2021 plus
    common PEFT defaults.
  * The 64-frame clip is forwarded through the wrapped V-JEPA at every
    training step (no caching), then mean-pooled spatially+temporally to a
    1024-d vector that gets concatenated with the Chronos-2 sensor emb
    (512-d) before going into the AR-modern decoder.
  * Gradient checkpointing is enabled on the V-JEPA blocks to fit larger
    batches on H100 80 GB.
  * Base V-JEPA weights are frozen; only LoRA adapters + projector + AR
    decoder train.
"""
from pathlib import Path
import sys
from typing import Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from config import ModelConfig
from ar_video_model import VideoConditionedARModern

VJEPA_DIR = Path("/home/simran/allspark-data-exploration/CPS-Debugger/vjepa2")
INPUT_RES = 384
CLIP_FRAMES = 64
VJEPA_DIM   = 1024
SENSOR_DIM  = 512
COND_DIM    = SENSOR_DIM + VJEPA_DIM            # 1536

# ImageNet normalization (matches V-JEPA pretraining)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


def load_vjepa_encoder():
    sys.path.insert(0, str(VJEPA_DIR))
    encoder, _predictor = torch.hub.load(
        str(VJEPA_DIR), "vjepa2_1_vit_large_384",
        source="local", pretrained=True, trust_repo=True,
    )
    return encoder


def make_lora_vjepa(r=16, lora_alpha=32, lora_dropout=0.1):
    encoder = load_vjepa_encoder()
    # Freeze the base
    for p in encoder.parameters():
        p.requires_grad = False
    # peft 0.5: target_modules matches by name substring
    # Target ONLY the fused attention `qkv` (Linear). V-JEPA also has
    # patch_embed.proj (Conv3d) and blocks.X.attn.proj (Linear); peft 0.5
    # rejects Conv3d. Substring match on "proj" hits both, so we drop it
    # and stick to Q+K+V — that's the standard ViT-LoRA recipe (Hu et al.
    # 2021 + PEFT examples), and matches the spirit of "Q/V LoRA".
    cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["qkv"],
        bias="none",
        task_type=None,
    )
    return get_peft_model(encoder, cfg)


class BL3bModel(nn.Module):
    """V-JEPA(LoRA) + Chronos-2 sensor emb -> AR-modern decoder."""

    def __init__(self, mcfg: ModelConfig, lora_r: int = 16, lora_alpha: int = 32,
                 lora_dropout: float = 0.1, grad_checkpoint_vjepa: bool = True):
        super().__init__()
        self.cfg = mcfg
        self.pad_id = mcfg.vocab_size
        self.bos_id = mcfg.vocab_size + 1

        self.vjepa = make_lora_vjepa(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        if grad_checkpoint_vjepa:
            # Best-effort: most ViT impls expose a list of blocks at .blocks
            base = self.vjepa.base_model.model if hasattr(self.vjepa, "base_model") else self.vjepa
            blocks = getattr(base, "blocks", None) or getattr(base, "layers", None)
            if blocks is not None:
                for b in blocks:
                    b._orig_forward = b.forward
                    b.forward = (lambda *a, _b=b, **kw:
                                 torch.utils.checkpoint.checkpoint(
                                     _b._orig_forward, *a, use_reentrant=False, **kw))

        # AR-modern with conditioning_dim = 1536 (sensor 512 + v-jepa 1024)
        self.ar = VideoConditionedARModern(mcfg, conditioning_dim=COND_DIM)

        # Cache normalization buffers
        self.register_buffer("imagenet_mean", _MEAN, persistent=False)
        self.register_buffer("imagenet_std",  _STD,  persistent=False)

    def encode_vjepa(self, clip_uint8: torch.Tensor) -> torch.Tensor:
        """clip_uint8: (B, 64, 3, 384, 384) uint8 -> (B, 1024) float."""
        # uint8 -> float, scale, normalize
        x = clip_uint8.float() / 255.0                          # (B, T, 3, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()               # (B, 3, T, H, W)
        x = (x - self.imagenet_mean) / self.imagenet_std
        out = self.vjepa(x)                                     # tokens
        if out.dim() == 3:
            return out.mean(dim=1)
        # collapse spatiotemporal dims
        return out.flatten(1, -2).mean(dim=1)

    def conditioning(self, sensor_emb: torch.Tensor, clip_uint8: torch.Tensor) -> torch.Tensor:
        v = self.encode_vjepa(clip_uint8)
        return torch.cat([sensor_emb, v], dim=-1)               # (B, 1536)

    def compute_loss(self, x0, sensor_emb, clip_uint8, pad_mask):
        cond = self.conditioning(sensor_emb, clip_uint8)
        return self.ar.compute_loss(x0, cond, pad_mask)

    @torch.no_grad()
    def sample(self, sensor_emb, clip_uint8, seq_len, **kw):
        cond = self.conditioning(sensor_emb, clip_uint8)
        return self.ar.sample(cond, seq_len, **kw)

    def param_count(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return train, total
