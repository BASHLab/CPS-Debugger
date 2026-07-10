"""BL-2 MDLM configuration."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path("/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates")
# Full-vocab parquets (2124-token vocab, L~1876) — kept for ablation
DATA_DIR_FULL = BASE_DIR / "train_data_full"
# Pruned-vocab parquets (648-token vocab, L~91) — default for training
DATA_DIR_PRUNED = BASE_DIR / "train_data_pruned"
DATA_DIR = DATA_DIR_PRUNED      # active dataset
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"
EMB_DIR = BASE_DIR / "sensor_embeddings"

EXTRACTED = Path("/home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs/extracted")

# Token mappings live in DATA_DIR_FULL (both point to same WASM binary)
TOKEN_MAPPING = DATA_DIR_FULL / "pruned_token_mapping.json"
TOKEN_MAPPING_FULL = DATA_DIR_FULL / "auto_token_mapping.json"

# ── Data split ─────────────────────────────────────────────────────────
# Mirrors Chronos-2 FT splits. Test sessions are the 2 that Chronos-2 FT
# never saw (neither train nor val) — true held-out evaluation.
# Train/val match Chronos-2 FT so sensor encoder never sees test-session data.
TRAIN_SESSIONS = [
    "2025-03-17_10-36-44", "2025-03-17_10-51-58", "2025-03-17_11-06-39",
    "2025-03-18_12-39-10",
    "2025-03-19_10-05-47", "2025-03-19_10-20-35", "2025-03-19_10-37-56",
    "2025-03-19_11-10-13",
    "2025-03-20_09-31-56", "2025-03-20_09-46-30", "2025-03-20_10-03-00",
]
VAL_SESSIONS = [
    "2025-03-21_11-21-42",
    "2025-03-25_13-23-42",
]
TEST_SESSIONS = [
    "2025-03-25_13-39-06",
    "2025-03-26_11-03-04",
]
ALL_SESSIONS = TRAIN_SESSIONS + VAL_SESSIONS + TEST_SESSIONS

SENSOR_COLS = [
    "pendulum_state", "iteration", "target_x", "current_x",
    "velocity", "current_angle", "angular_velocity",
]
N_SENSORS = len(SENSOR_COLS)


# ── Sensor encoder ─────────────────────────────────────────────────────
@dataclass
class SensorConfig:
    chronos_model: str = "amazon/chronos-2"
    chronos_d_model: int = 768      # chronos-2 encoder dim
    n_patches: int = 32             # 500 timesteps / 16 patch_size ≈ 32
    window_size: int = 500          # sensor context window
    n_channels: int = N_SENSORS     # 7 sensor channels
    proj_dim: int = 512             # output embedding dim (= MDLM d_model)
    freeze_chronos: bool = True     # freeze during MDLM training (Stage 2)
    batch_size: int = 64            # batch size for precomputing embeddings
    # Stage 1 fine-tuning
    ft_prediction_length: int = 64  # forecast horizon for fine-tuning (4 output patches × 16)
    ft_num_steps: int = 5000        # ~4 epochs over 150K windows
    ft_learning_rate: float = 1e-4  # Chronos pre-training uses 1e-3; 1e-4 is conservative for FT
    ft_batch_size: int = 128        # fine-tuning batch size


# ── MDLM model ────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    vocab_size: int = 648            # from pruned_token_mapping.json (set at runtime)
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 200           # ~87 pruned trace + ~1 sensor prefix + margin
    # Diffusion
    noise_schedule: str = "loglinear"  # loglinear | cosine
    parameterization: str = "subs"     # subs (substitution, Rao-Blackwellized)
    mask_token_id: int = -1            # set at runtime = vocab_size
    # Sensor conditioning
    n_sensor_tokens: int = 1           # number of prepended sensor tokens
    d_sensor_emb: int = 512            # encoder output dim (Chronos-2: 512, BL-7: 256)
    cond_dropout_p: float = 0.0        # conditioning dropout for cross-attn decoder (Tier 1A.1)


# ── Training ───────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    # Optimizer
    lr: float = 3e-4
    weight_decay: float = 0.01       # paper uses 0; kept nonzero for small-data regularization
    warmup_steps: int = 2500         # MDLM paper default
    max_steps: int = 150_000         # ~19 epochs on 2-session train (MDLM benefits from long training)
    # Batch
    batch_size: int = 128
    grad_accum: int = 1              # effective batch = 128
    # Mixed precision
    bf16: bool = True
    grad_clip: float = 1.0
    grad_checkpoint: bool = True     # needed for BS=128 on L40S (46GB VRAM)
    # Logging
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 5000
    # EMA
    ema_decay: float = 0.9999
    # Sampling
    sampling_steps: int = 256        # T for inference


@dataclass
class EvalConfig:
    sampling_steps: int = 1000       # MDLM paper default for final eval (ddpm_cache predictor)
    batch_size: int = 8
    n_samples: int = 1000            # per-split evaluation samples


# ── BL-7 sensor encoder (Tier 1B / 2 of the BL-7 plan) ────────────────────
@dataclass
class BL7EncoderConfig:
    """Channel-aware ViT-Small + conv frontend + Perceiver pool. ~50 M params."""
    # Input
    n_channels: int = N_SENSORS         # 7
    window_size: int = 500              # ticks per encoder window
    # Conv frontend (per-channel, weight-shared). The stride schedule maps the
    # raw window to n_time_patches; defaults are the 500-tick design
    # (500 → 99 → 46 → 22 → 22). Override conv_strides for longer windows, e.g.
    # the 4000-tick variant uses strides (5,4,2,1) → 799 → 198 → 98 → 98.
    conv_out_dim: int = 256             # output channel count after final block
    conv_kernels: tuple = (10, 8, 4, 3)
    conv_strides: tuple = (5, 2, 2, 1)
    conv_paddings: tuple = (0, 0, 0, 1)
    # n_time_patches is DERIVED from window_size + conv schedule in
    # __post_init__; the default here is a placeholder, overwritten on init.
    n_time_patches: int = 22
    # Transformer (ViT-Small)
    d_model: int = 512
    n_heads: int = 8                    # head_dim = 64
    n_layers: int = 12
    d_ff: int = 2048                    # MLP×4
    dropout: float = 0.10
    # Channel-axis attention every 3 layers (layers 1,2,4,5,7,8,10,11)
    # → time-axis at layers 0,3,6,9.
    # Output: K=4 dense tokens per tick at d_emb=256
    K: int = 4
    d_emb: int = 256
    # Axial attention: time-axis attention every channel_mix_every layers
    # (the rest are channel-axis attention). Default 3 = layers 0,3,6,9 time;
    # 1,2,4,5,7,8,10,11 channel. Tier 9 ablation: channel_mix_every=1.
    channel_mix_every: int = 3

    def __post_init__(self):
        # Derive n_time_patches from the conv schedule so it can never drift
        # from window_size / strides. Conv1d length formula per layer:
        #   L_out = floor((L_in - kernel + 2*padding) / stride) + 1
        L = self.window_size
        for k, s, p in zip(self.conv_kernels, self.conv_strides, self.conv_paddings):
            L = (L - k + 2 * p) // s + 1
        self.n_time_patches = L


@dataclass
class BL7TrainConfig:
    """JEPA pretraining recipe for BL-7. Tier 1B = main loss only."""
    # Optimizer
    lr: float = 5e-4                    # peak (tri-stage 3/90/7)
    weight_decay: float = 0.05
    betas: tuple = (0.9, 0.95)
    # Schedule
    warmup_pct: float = 0.03
    flat_pct: float = 0.90              # decay starts at step 0.93*max
    max_steps: int = 100_000            # ~300 epochs × 5 M sub-sampled windows / batch
    # Batch
    batch_size: int = 64
    grad_accum: int = 2                 # effective batch = 128
    bf16: bool = True
    grad_clip: float = 1.0
    # JEPA — span-block masking
    mask_span_min: int = 10             # patches
    mask_span_max: int = 20
    mask_ratio: float = 0.35            # 30–40 % audio-style
    teacher_target_layers: int = 8      # average top-K teacher layers
    # EMA teacher
    ema_tau_start: float = 0.999
    ema_tau_end: float = 0.9999
    ema_anneal_pct: float = 0.30        # τ_start → τ_end over first 30% of training
    # Predictor (separate small transformer)
    pred_n_layers: int = 6
    pred_d_model: int = 384
    pred_n_heads: int = 6
    # Logging / checkpoint
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 5000
    # Class-balanced contrastive aux (Tier 2, off in Tier 1B)
    use_contrastive: bool = False
    contrastive_weight: float = 0.20
    contrastive_temperature: float = 0.10
    # Time-reversal symmetry aux (Tier 9 ablation, off by default)
    use_time_reversal: bool = False
    time_reversal_weight: float = 0.10
    # E3 (BL-8a): prototype-based SupCon with natural-rate sampling.
    # Distinct from use_contrastive (which has the 33/33/33 resampling
    # distortion that made Tier-2 backfire). off by default.
    use_proto_supcon: bool = False
    proto_supcon_weight: float = 0.10
    proto_supcon_momentum: float = 0.999
    # HuBERT-style per-patch discrete-codebook auxiliary (non-moving anchor).
    # Predicts the offline k-means cluster id of each MASKED patch token from
    # the student's per-position features. Unlike use_contrastive (pooled K=4,
    # a soft regularizer), this anchors the SAME per-position prediction the
    # EMA target corrupts, so it is a co-equal anchor (weight ~1.0). off by
    # default. Labels precomputed by precompute_hubert_labels.py.
    use_hubert_aux: bool = False
    hubert_k: int = 256
    hubert_weight: float = 1.0
    hubert_label_dir: str = None        # default: models/<ckpt_subdir>/hubert_labels
    hubert_codebook_path: str = None    # default: models/<ckpt_subdir>/hubert_codebook.npz
