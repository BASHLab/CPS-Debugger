"""Pipeline-wide determinism controls.

Call `set_deterministic(seed)` at process start. This is the *reproducibility*
contract: same seed + same GPU SKU + same PyTorch build → bit-exact outputs.

What this handles:
  - Python hash seed, `random`, `numpy`, `torch` CPU + CUDA seeds
  - cuDNN: deterministic=True, benchmark=False (stable kernel selection)
  - CUBLAS workspace config (required for deterministic matmul under CUDA ≥10.2)
  - Deterministic algorithms flag (falls back where no deterministic impl exists)
  - TF32 kept ON (bf16 autocast dominates; TF32 matmul is still deterministic)

What this does NOT handle (see docstring of `caveats()` for details):
  - Cross-GPU-SKU numerical identity (A100 vs Blackwell → different kernels)
  - Multi-worker DataLoader ordering (use `make_dataloader_generator()`)
  - Flash/SDPA backend drift (call `pin_sdpa_backend()` before timing)
"""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


DEFAULT_SEED = 42


def set_deterministic(seed: int = DEFAULT_SEED,
                      warn_only: bool = True,
                      tf32: bool = True) -> None:
    """Fix every RNG the BL-2 pipeline touches.

    Args:
        seed: base seed. DataLoader workers should derive from it via
              `make_worker_init_fn(seed)`.
        warn_only: if True, fall back to non-deterministic ops where no
                   deterministic implementation exists (prints a warning).
                   Set False to hard-fail instead — useful for debugging.
        tf32: keep TF32 matmul enabled. TF32 is deterministic at the kernel
              level; only turn off if you need bit-exact fp32 comparison.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Required by PyTorch for deterministic cuBLAS under CUDA ≥10.2.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32

    # warn_only=True lets ops without a deterministic impl run anyway, with a
    # stderr warning. For inference on our architecture (no scatter_add on the
    # hot path, no conv), the warning set should be empty.
    torch.use_deterministic_algorithms(True, warn_only=warn_only)


def make_worker_init_fn(seed: int = DEFAULT_SEED):
    """DataLoader worker seeding for multi-worker determinism.

    Usage:
        DataLoader(..., num_workers=N,
                   worker_init_fn=make_worker_init_fn(SEED),
                   generator=make_generator(SEED))
    """
    def _init(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return _init


def make_generator(seed: int = DEFAULT_SEED) -> torch.Generator:
    """Fixed-seed generator for passing into DataLoader(generator=...)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def pin_sdpa_backend(prefer: str = "math") -> None:
    """Pin the scaled_dot_product_attention backend so kernel dispatch is stable.

    "math" is the most portable and deterministic. "flash" and "mem_efficient"
    are faster but their dispatch can vary across PyTorch patch versions and
    GPU SKUs. For timing comparability across AR-base vs AR-modern the single
    requirement is that both models use the same backend — so call this once
    at process start with the same value for both models.
    """
    from torch.nn.attention import sdpa_kernel, SDPBackend
    # Return a context-manager-like object is not what we want here;
    # instead enable/disable backends globally via the older API.
    torch.backends.cuda.enable_flash_sdp(prefer == "flash")
    torch.backends.cuda.enable_mem_efficient_sdp(prefer == "mem_efficient")
    torch.backends.cuda.enable_math_sdp(prefer == "math")


def caveats() -> str:
    """Return a human-readable summary of residual non-determinism."""
    return (
        "Residual sources of variance after set_deterministic():\n"
        "  1. GPU SKU: A100 vs RTX PRO 6000 B → different Tensor Core kernels;\n"
        "     bf16/fp16 GEMM results differ in last bit. Pin SKU via SLURM\n"
        "     --constraint or --nodelist for bit-exact comparison.\n"
        "  2. PyTorch build: patch-level upgrades can change kernel dispatch.\n"
        "     Record torch.__version__ alongside results.\n"
        "  3. Sampling ops (torch.multinomial) are SEED-reproducible, not\n"
        "     deterministic-without-seed. Seed before each call if you need\n"
        "     identical sampled outputs across runs.\n"
        "  4. Flash/SDPA dispatch: call pin_sdpa_backend() if comparing models\n"
        "     whose attention patterns differ."
    )
