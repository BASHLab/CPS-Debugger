"""Extract the BL-7 encoder (frozen backbone + trained pool) from a joint-trained
AR-modern checkpoint and save it in sensor_encoder_bl7.py-compatible format.

After E2 joint training, the AR-modern checkpoint contains a state_dict where
the encoder weights live under `encoder.*` (JointEncoderARModel structure).
This script pulls those weights out and writes a clean encoder ckpt that
sensor_encoder_bl7.py can load via its existing `use_ema=True` path.

We write the encoder state under `"ema"` key for backward compat (the
existing precompute script preferentially loads EMA weights). The
encoder_config is taken from the original BL-7 v1 ckpt but with K updated.

Usage:
    python3 extract_bl7_encoder_from_joint.py \\
        --joint-ckpt models/ar_modern_bl7_k8_fold_0/ar_modern_best.pt \\
        --output     models/ar_modern_bl7_k8_fold_0/bl7_encoder.pt
"""
import argparse
from pathlib import Path

import torch

from config import BL7EncoderConfig


def extract(joint_ckpt_path: Path, output_path: Path,
            orig_encoder_ckpt: Path = None) -> None:
    ckpt = torch.load(joint_ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    # Strip torch.compile prefix if present
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}

    # Joint state_dict has keys: encoder.*, ar.*  (JointEncoderARModel structure)
    encoder_state = {
        k.removeprefix("encoder."): v for k, v in sd.items() if k.startswith("encoder.")
    }
    if not encoder_state:
        raise RuntimeError(
            f"No 'encoder.*' keys found in {joint_ckpt_path}. "
            f"Is this a joint-trained checkpoint? Found keys (first 5): "
            f"{list(sd.keys())[:5]}"
        )

    # Derive encoder_config from the loaded state to capture the right K.
    # Pool query parameter is `pool.queries` with shape (K, d_model).
    queries = encoder_state.get("pool.queries")
    if queries is None:
        raise RuntimeError("Joint ckpt missing 'pool.queries' — pool absent?")
    K = queries.shape[0]
    d_model = queries.shape[1]

    # Source the full encoder_config (window / conv schedule / n_time_patches)
    # from the original pretrain ckpt when given, overriding only K and
    # d_model from the trained pool. Without it, fall back to the 500-tick
    # default config (correct only for the 500-tick K-sweep).
    if orig_encoder_ckpt is not None:
        base = torch.load(Path(orig_encoder_ckpt), map_location="cpu",
                          weights_only=False)
        cfg_dict = dict(base["encoder_config"])
        cfg_dict["K"] = K
        cfg_dict["d_model"] = d_model
        enc_cfg = BL7EncoderConfig(**cfg_dict)
    else:
        enc_cfg = BL7EncoderConfig(K=K, d_model=d_model)
    enc_cfg_dict = enc_cfg.__dict__

    torch.save({
        "ema": encoder_state,                # keyed `ema` so sensor_encoder_bl7.py loads it
        "encoder_config": enc_cfg_dict,
        "extracted_from": str(joint_ckpt_path),
        "extraction_note": "Encoder weights extracted from joint-trained AR-modern checkpoint",
    }, output_path)
    print(f"Extracted encoder ({len(encoder_state)} tensors, K={K}, d_model={d_model})")
    print(f"  -> {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint-ckpt", required=True, help="path to joint ar_modern_best.pt")
    ap.add_argument("--output", required=True, help="path to write extracted encoder ckpt")
    ap.add_argument("--orig-encoder-ckpt", default=None,
                    help="source full encoder_config (window/conv schedule) from this "
                         "pretrain ckpt; only K/d_model are overridden from the trained pool")
    args = ap.parse_args()
    extract(Path(args.joint_ckpt), Path(args.output),
            orig_encoder_ckpt=args.orig_encoder_ckpt)


if __name__ == "__main__":
    main()
