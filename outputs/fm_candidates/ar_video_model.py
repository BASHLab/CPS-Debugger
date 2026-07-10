"""AR-modern with multi-modal conditioning (BL-3 / BL-3b / BL-4 / BL-5).

Same backbone as `ar_modern_model.ARModernModel` (RMSNorm + SwiGLU + QK-Norm
+ RoPE), differing only in the conditioning projection: the input vector can
be arbitrary-dim (concat of Chronos-2, V-JEPA, CoTracker, ...) instead of
fixed d_model.

The conditioning vector is projected to d_model and prepended as a single
virtual token at position 0, exactly like AR-modern's sensor_emb. The
training loop is responsible for assembling the conditioning vector per
variant; this class is variant-agnostic.

Variant   | conditioning_dim | composition
----------|------------------|------------------------------------------
BL-3      | 1536             | sensor (512) || vjepa (1024)
BL-3b     | 1536             | sensor (512) || vjepa-with-LoRA (1024)
BL-4      |  572             | sensor (512) || cotracker (60)
BL-5      | 1596             | sensor (512) || vjepa (1024) || cotracker (60)
"""
import torch
import torch.nn as nn

from config import ModelConfig
from ar_modern_model import ARModernModel


class VideoConditionedARModern(ARModernModel):
    """Drop-in for ARModernModel with a wider conditioning projection.

    `compute_loss` / `sample` / `param_count` are inherited unchanged — they
    pass `sensor_emb` straight to `backbone.sensor_proj`. We just replace
    that linear with one that takes `conditioning_dim` inputs."""

    def __init__(self, cfg: ModelConfig, conditioning_dim: int):
        super().__init__(cfg)
        self.conditioning_dim = conditioning_dim
        self.backbone.sensor_proj = nn.Linear(
            conditioning_dim, cfg.d_model, bias=False
        )
        nn.init.xavier_uniform_(self.backbone.sensor_proj.weight)
