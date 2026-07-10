"""BL-7 Tier 0 + Tier 5 probes.

Tier 0 — randomization probe: is the AR-modern decoder actually using the
sensor encoder, or is it corpus-memorizing? Compute trace NLL on the val
split with sensor_emb replaced by:
  - "real":           baseline
  - "random":         iid Gaussian with matching shape (or pooled shape)
  - "zero":           all zeros
  - "cross-session":  embeddings drawn from a different session's mmap
If NLL is similar across conditions, the decoder isn't using the encoder.

Tier 5 — information-density probes on a frozen sensor encoder feature:
  (1) per-tick next-patch sensor-signal reconstruction MSE (linear head)
  (2) per-tick discrete-trace-vocab linear probe (small held-out subset)
  (3) Gram-matrix similarity to an earlier checkpoint (DINOv3 diagnostic)
  (4) SUPERB-style layer-weighted-sum linear probe (which layers carry info)

This file is self-contained on top of the existing data + models layers.
Usage:
    # Tier 0 on existing BL-7 v1 fold-0 checkpoint
    python3 bl7_probes.py randomization \\
        --ar-ckpt models/ar_modern_bl7_fold_0/ar_modern_best.pt \\
        --val-sessions 2025-03-21_11-21-42 2025-03-25_13-23-42 \\
        --emb-dir sensor_embeddings_bl7 --n-samples 1000

    # Tier 0 on BL-2 (sensor-only Chronos-2) for comparison
    python3 bl7_probes.py randomization \\
        --ar-ckpt models/ar_modern_fold_0/ar_modern_best.pt \\
        --val-sessions 2025-03-21_11-21-42 2025-03-25_13-23-42 \\
        --emb-dir sensor_embeddings --n-samples 1000

    # Tier 5 info-density probes
    python3 bl7_probes.py info-density \\
        --bl7-ckpt models/bl7/bl7_best.pt --fold 0 --n-samples 2000
"""
import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import (
    ModelConfig, DATA_DIR, EMB_DIR, MODEL_DIR, SENSOR_COLS,
    BL7EncoderConfig,
)
from data import TraceDataset, load_token_mapping
from ar_modern_model import ARModernModel
from ar_modern_cross_model import ARModernCrossModel


# ── Tier 0 ────────────────────────────────────────────────────────────────
@torch.no_grad()
def _classification_report(preds: torch.Tensor, labels: torch.Tensor,
                            n_classes: int = 3) -> Dict:
    """Per-class precision / recall / F1 + macro-F1 + balanced accuracy.

    With the CPS FSM imbalance (~94/3/3), plain accuracy is dominated by the
    majority class and misleading. F1 (needs both precision AND recall) and
    balanced accuracy (mean per-class recall) are the honest metrics.
    """
    preds = preds.long()
    labels = labels.long()
    report = {"per_class": {}}
    f1s, recalls = [], []
    for c in range(n_classes):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        support = int((labels == c).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        report["per_class"][c] = {
            "precision": precision, "recall": recall, "f1": f1, "support": support,
        }
        f1s.append(f1)
        recalls.append(recall)
    report["macro_f1"] = sum(f1s) / n_classes
    report["balanced_accuracy"] = sum(recalls) / n_classes
    report["overall_accuracy"] = float((preds == labels).float().mean())
    return report


def _trace_nll(model, batch_iter, device, replace_fn, max_batches=50):
    """Run model forward; return mean NLL over masked tokens.

    replace_fn(sensor_emb_batch) → sensor_emb_batch  (identity for baseline)
    """
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i, batch in enumerate(batch_iter):
        if i >= max_batches:
            break
        x0 = batch["trace"].to(device)
        emb = batch["sensor_emb"].to(device)
        emb = replace_fn(emb)
        pad_id = model.cfg.vocab_size
        pad_mask = (x0 != pad_id).float()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, metrics = model.compute_loss(x0, emb, pad_mask)
        total_loss += metrics["loss"] * metrics["n_tokens"]
        total_tokens += metrics["n_tokens"]
    return total_loss / max(total_tokens, 1)


def randomization_probe(args):
    device = torch.device(args.device)
    vocab_size, _, _ = load_token_mapping()

    # Load AR-modern checkpoint
    ckpt = torch.load(args.ar_ckpt, map_location=device, weights_only=False)
    mcfg_dict = ckpt["model_config"]
    mcfg = ModelConfig(**mcfg_dict)
    use_cross = bool(ckpt.get("use_cross_attn", False))
    model_cls = ARModernCrossModel if use_cross else ARModernModel
    model = model_cls(mcfg).to(device)
    # Strip torch.compile prefix from state_dict keys if present.
    sd = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()

    print(f"Loaded {type(model).__name__} from {args.ar_ckpt}")
    print(f"  d_sensor_emb = {mcfg.d_sensor_emb}, n_sensor_tokens = {mcfg.n_sensor_tokens}, "
          f"use_cross_attn = {use_cross}")

    # Build dataset on the val sessions
    val_ds = TraceDataset(
        args.val_sessions, mcfg.max_seq_len, mask_id=mcfg.vocab_size,
        emb_dir=Path(args.emb_dir), data_dir=Path(args.data_dir),
        limit=args.n_samples,
    )
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=False)

    # Cross-session embedding pool: load a different session's embeddings
    # and reuse them with replacement during the cross-session condition.
    cross_emb_pool = None
    if args.cross_session is not None:
        path = Path(args.emb_dir) / f"{args.cross_session}_emb.npy"
        cross_emb_pool = torch.from_numpy(np.load(path)).float().to(device)
        print(f"  cross-session pool: {tuple(cross_emb_pool.shape)}")

    rng = torch.Generator(device=device).manual_seed(args.seed)

    def baseline(emb):    return emb
    def gaussian(emb):    return torch.randn_like(emb)
    def zero(emb):        return torch.zeros_like(emb)
    def cross_session(emb):
        if cross_emb_pool is None:
            return emb
        idx = torch.randint(0, cross_emb_pool.shape[0], (emb.shape[0],),
                            generator=rng, device=device)
        return cross_emb_pool[idx]

    conditions = [("real", baseline), ("random_gaussian", gaussian), ("zero", zero)]
    if cross_emb_pool is not None:
        conditions.append(("cross_session", cross_session))

    results = {}
    for name, fn in conditions:
        t0 = time.time()
        nll = _trace_nll(model, val_dl, device, fn,
                         max_batches=max(args.n_samples // args.batch_size, 1))
        elapsed = time.time() - t0
        ppl = math.exp(min(nll, 20))
        results[name] = {"nll": float(nll), "ppl": float(ppl), "elapsed_s": elapsed}
        print(f"  {name:>20s}: NLL = {nll:.4f}  PPL = {ppl:.3f}  ({elapsed:.1f}s)")

    # Δ to baseline
    real = results["real"]["nll"]
    print("\nΔ-NLL vs real (positive = worse / model is sensitive to encoder):")
    for name, r in results.items():
        if name == "real":
            continue
        delta = r["nll"] - real
        verdict = "STRONGLY uses encoder" if delta > 0.5 else \
                  "uses encoder somewhat" if delta > 0.1 else \
                  "DOES NOT use encoder (corpus-memorising)"
        print(f"  {name:>20s}: Δ = {delta:+.4f} nats   ({verdict})")

    out_path = Path(args.out) if args.out else Path("results/tier0_randomization.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "ar_ckpt": str(args.ar_ckpt),
        "use_cross_attn": use_cross,
        "n_samples": args.n_samples,
        "val_sessions": list(args.val_sessions),
        "emb_dir": args.emb_dir,
        "results": results,
    }, indent=2))
    print(f"\nSaved {out_path}")


# ── Tier 5 ────────────────────────────────────────────────────────────────
@torch.no_grad()
def _encode_batches(encoder, sensor_arrays: List[np.ndarray], window_size: int,
                    device: torch.device, n_samples: int, batch_size: int = 32):
    """Sample n_samples random windows across the provided sensor arrays and
    return (features, sensor_targets). features: (n, K*d_emb) flat; sensor_targets:
    (n, 7) — next-tick sensor values (for Probe 1).
    """
    K, d_emb = encoder.cfg.K, encoder.cfg.d_emb
    feat = np.zeros((n_samples, K, d_emb), dtype=np.float32)
    targets = np.zeros((n_samples, 7), dtype=np.float32)
    rng = np.random.default_rng(0)

    # Build random (session, end) draws
    lens = np.array([len(a) for a in sensor_arrays])
    cum = lens.cumsum()
    total = lens.sum()

    cursor = 0
    while cursor < n_samples:
        bs = min(batch_size, n_samples - cursor)
        windows, t_targets = [], []
        for _ in range(bs):
            # Need window plus next tick
            r = rng.integers(0, total)
            si = int(np.searchsorted(cum, r, side="right"))
            n = lens[si]
            end = int(rng.integers(window_size - 1, n - 1))  # leave room for next tick
            window = sensor_arrays[si][end - window_size + 1: end + 1]
            windows.append(window.T)
            t_targets.append(sensor_arrays[si][end + 1])
        batch = torch.from_numpy(np.stack(windows)).float().to(device)
        with torch.amp.autocast(device.type, dtype=torch.bfloat16):
            dense = encoder.encode_dense(batch)         # (bs, K, d_emb)
        feat[cursor: cursor + bs] = dense.float().cpu().numpy()
        targets[cursor: cursor + bs] = np.stack(t_targets)
        cursor += bs

    return feat, targets


def info_density_probes(args):
    from bl7_model import BL7Encoder, EMATeacher
    device = torch.device(args.device)

    # Load BL-7 EMA-teacher
    ckpt = torch.load(args.bl7_ckpt, map_location=device, weights_only=False)
    ecfg = BL7EncoderConfig(**ckpt["encoder_config"])
    enc = BL7Encoder(ecfg).to(device)
    if "ema" in ckpt:
        ema = EMATeacher(enc).to(device)
        ema.load_state_dict(ckpt["ema"])
        ema.copy_to(enc)
        print(f"Loaded EMA-teacher BL-7 from {args.bl7_ckpt}")
    else:
        enc.load_state_dict(ckpt["student"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    # Build sensor data for val sessions
    sessions = args.val_sessions
    sensor_arrays = []
    state_arrays = []
    for sess in sessions:
        df = pd.read_parquet(Path(args.data_dir) / f"{sess}.parquet",
                             columns=list(SENSOR_COLS))
        sensor_arrays.append(df[list(SENSOR_COLS)].values.astype(np.float32))
        state_arrays.append(df["pendulum_state"].values.astype(np.int64))

    # Probe 1: per-tick next-patch sensor-signal reconstruction MSE (linear head)
    print("\nProbe 1: per-tick next-tick sensor regression (linear)")
    feat, targets = _encode_batches(enc, sensor_arrays, ecfg.window_size,
                                    device, n_samples=args.n_samples)
    X = feat.reshape(feat.shape[0], -1)
    # Centre + ridge regression closed-form on features → targets
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = targets - targets.mean(axis=0, keepdims=True)
    reg = 1e-2
    # (X'X + λI)^-1 X' y
    A = Xc.T @ Xc + reg * np.eye(Xc.shape[1], dtype=np.float32)
    W = np.linalg.solve(A, Xc.T @ yc)
    y_hat = Xc @ W
    mse = float(((y_hat - yc) ** 2).mean())
    var = float((yc ** 2).mean())
    r2 = 1.0 - mse / max(var, 1e-9)
    print(f"  MSE = {mse:.4f}   R² = {r2:.4f}")

    # Probe 2: per-tick FSM state classification (linear)
    print("\nProbe 2: per-tick FSM state classification (linear, 3-way)")
    # Sample windows along with their ending state
    K, d_emb = ecfg.K, ecfg.d_emb
    rng = np.random.default_rng(1)
    lens = np.array([len(a) for a in sensor_arrays])
    cum = lens.cumsum()
    total = lens.sum()
    n_cls = args.n_samples
    feat_cls = np.zeros((n_cls, K, d_emb), dtype=np.float32)
    y_cls = np.zeros(n_cls, dtype=np.int64)
    cursor = 0
    batch_size = 32
    while cursor < n_cls:
        bs = min(batch_size, n_cls - cursor)
        windows, ys = [], []
        for _ in range(bs):
            r = rng.integers(0, total)
            si = int(np.searchsorted(cum, r, side="right"))
            n = lens[si]
            end = int(rng.integers(ecfg.window_size - 1, n))
            windows.append(sensor_arrays[si][end - ecfg.window_size + 1: end + 1].T)
            ys.append(int(state_arrays[si][end]))
        b = torch.from_numpy(np.stack(windows)).float().to(device)
        with torch.amp.autocast(device.type, dtype=torch.bfloat16):
            dense = enc.encode_dense(b)
        feat_cls[cursor: cursor + bs] = dense.float().cpu().numpy()
        y_cls[cursor: cursor + bs] = ys
        cursor += bs

    # Per-state balanced linear probe (multinomial logistic regression via PyTorch SGD)
    Xf = torch.from_numpy(feat_cls.reshape(n_cls, -1)).to(device)
    yt = torch.from_numpy(y_cls).to(device)
    n_features = Xf.shape[1]
    head = nn.Linear(n_features, 3, bias=True).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    # Class weights for balanced training (state imbalance heavy)
    counts = torch.bincount(yt, minlength=3).float().clamp(min=1)
    weights = (1.0 / counts) / (1.0 / counts).sum()  # normalised inverse freq
    for it in range(500):
        opt.zero_grad()
        logits = head(Xf)
        loss = F.cross_entropy(logits, yt, weight=weights.to(device))
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds = head(Xf).argmax(dim=-1)
        report = _classification_report(preds.cpu(), yt.cpu(), n_classes=3)
    print(f"  Overall acc {report['overall_accuracy']:.4f} | "
          f"macro-F1 {report['macro_f1']:.4f} | "
          f"balanced-acc {report['balanced_accuracy']:.4f}")
    for c in (0, 1, 2):
        pc = report["per_class"][c]
        print(f"    state-{c}: P={pc['precision']:.4f} R={pc['recall']:.4f} "
              f"F1={pc['f1']:.4f} (support={pc['support']})")

    # Save results
    out_path = Path(args.out) if args.out else Path("results/tier5_info_density.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "bl7_ckpt": str(args.bl7_ckpt),
        "val_sessions": list(args.val_sessions),
        "n_samples": args.n_samples,
        "probe_1_next_tick_regression": {"mse": mse, "r2": r2},
        "probe_2_fsm_classification": report,
    }, indent=2))
    print(f"\nSaved {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────
def dump_state_predictions(args):
    """E5 prep — train a 3-way linear FSM-state probe on frozen BL-7 features,
    then predict per-tick state for every session and save alongside the
    existing precomputed embeddings.

    For each session, this writes <emb-dir>/<session>_state_pred.npy of shape
    (N_ticks,) int8. The values are in {0, 1, 2} (FSM states) — caller's
    NO_COND token id is added separately at training time.

    Probe state_dict is saved to <out-probe> for downstream introspection.
    """
    device = torch.device(args.device)

    # Build training data from VAL_SESSIONS (the probe's training set —
    # ground-truth state is read from parquet `pendulum_state` column).
    train_sess = args.train_sessions
    pred_sess  = args.pred_sessions

    print(f"Training probe on sessions: {train_sess}")
    # A 3-way linear probe over 1024-dim features does not need every tick —
    # subsample per session to cap memory (the full 11-session set is ~22 GB).
    # We stratify-cap per session so all 3 FSM states stay represented.
    per_session_cap = args.train_cap // max(len(train_sess), 1)
    rng = np.random.default_rng(0)
    X_train, y_train = [], []
    for sess in train_sess:
        emb = np.load(Path(args.emb_dir) / f"{sess}_emb.npy", mmap_mode="r")
        df = pd.read_parquet(Path(args.data_dir) / f"{sess}.parquet",
                             columns=["pendulum_state"])
        states = df["pendulum_state"].values.astype(np.int64)
        n = len(states)
        # Stratified subsample: take up to per_session_cap/3 from each state,
        # backfilling from the majority class if a rare state is short.
        idx_pick = []
        for st in (0, 1, 2):
            st_idx = np.where(states == st)[0]
            take = min(len(st_idx), per_session_cap // 3)
            if take > 0:
                idx_pick.append(rng.choice(st_idx, size=take, replace=False))
        idx = np.concatenate(idx_pick) if idx_pick else np.arange(min(n, per_session_cap))
        idx.sort()
        emb_sub = np.array(emb[idx])  # materialise only the picked rows
        if emb_sub.ndim == 3:
            emb_sub = emb_sub.reshape(emb_sub.shape[0], -1)
        X_train.append(emb_sub)
        y_train.append(states[idx])
        del emb
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    print(f"  Training set (subsampled): {X_train.shape}, label dist: {np.bincount(y_train)}")

    # Train linear probe (multinomial logreg via PyTorch AdamW + class weights)
    Xf = torch.from_numpy(X_train).float().to(device)
    yt = torch.from_numpy(y_train).to(device)
    head = nn.Linear(Xf.shape[1], 3, bias=True).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    counts = torch.bincount(yt, minlength=3).float().clamp(min=1)
    cw = (1.0 / counts) / (1.0 / counts).sum()
    for it in range(800):
        opt.zero_grad()
        logits = head(Xf)
        loss = F.cross_entropy(logits, yt, weight=cw.to(device))
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds = head(Xf).argmax(dim=-1)
        report = _classification_report(preds.cpu(), yt.cpu(), n_classes=3)
    acc = report["overall_accuracy"]
    print(f"  Probe trained: overall acc {acc:.4f} | "
          f"macro-F1 {report['macro_f1']:.4f} | "
          f"balanced-acc {report['balanced_accuracy']:.4f}")
    for c in (0, 1, 2):
        pc = report["per_class"][c]
        print(f"    state-{c}: P={pc['precision']:.4f} R={pc['recall']:.4f} "
              f"F1={pc['f1']:.4f} (support={pc['support']})")

    # Save probe checkpoint
    probe_path = Path(args.out_probe)
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "head_state_dict": head.state_dict(),
        "in_features": Xf.shape[1],
        "n_classes": 3,
        "training_acc": acc,
        "classification_report": report,
        "train_sessions": list(train_sess),
        "emb_dir": str(args.emb_dir),
    }, probe_path)
    print(f"  Saved probe -> {probe_path}")

    # Predict for all requested sessions and save per-tick state predictions.
    # Batched per session to keep peak memory bounded.
    head.eval()
    BATCH = 100_000
    for sess in pred_sess:
        emb = np.load(Path(args.emb_dir) / f"{sess}_emb.npy", mmap_mode="r")
        n = emb.shape[0]
        preds = np.zeros(n, dtype=np.int8)
        with torch.no_grad():
            for start in range(0, n, BATCH):
                end = min(start + BATCH, n)
                chunk = np.array(emb[start:end])
                if chunk.ndim == 3:
                    chunk = chunk.reshape(chunk.shape[0], -1)
                logits = head(torch.from_numpy(chunk).float().to(device))
                preds[start:end] = logits.argmax(dim=-1).cpu().numpy().astype(np.int8)
        out_path = Path(args.emb_dir) / f"{sess}_state_pred.npy"
        np.save(out_path, preds)
        print(f"  [{sess}] state predictions saved -> {out_path} "
              f"({np.bincount(preds, minlength=3)} per-state)")
        del emb


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("randomization", help="Tier 0 randomization probe")
    p0.add_argument("--ar-ckpt", required=True)
    p0.add_argument("--val-sessions", nargs="+", required=True)
    p0.add_argument("--emb-dir", default=str(EMB_DIR))
    p0.add_argument("--data-dir", default=str(DATA_DIR))
    p0.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p0.add_argument("--batch-size", type=int, default=32)
    p0.add_argument("--n-samples", type=int, default=1000)
    p0.add_argument("--cross-session", default=None,
                    help="Optional session name whose embeddings act as cross-session pool")
    p0.add_argument("--seed", type=int, default=42)
    p0.add_argument("--out", default=None)

    p5 = sub.add_parser("info-density", help="Tier 5 info-density probes")
    p5.add_argument("--bl7-ckpt", required=True)
    p5.add_argument("--val-sessions", nargs="+",
                    default=["2025-03-21_11-21-42", "2025-03-25_13-23-42"])
    p5.add_argument("--data-dir", default=str(DATA_DIR))
    p5.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p5.add_argument("--n-samples", type=int, default=2000)
    p5.add_argument("--out", default=None)

    pdump = sub.add_parser("dump-state-predictions",
                            help="E5: train probe on val sessions, predict per-tick state for all")
    pdump.add_argument("--emb-dir", required=True)
    pdump.add_argument("--data-dir", default=str(DATA_DIR))
    pdump.add_argument("--train-sessions", nargs="+", required=True,
                       help="Sessions to fit the probe on (ground-truth state)")
    pdump.add_argument("--pred-sessions", nargs="+", required=True,
                       help="Sessions to predict state on (writes <emb-dir>/<sess>_state_pred.npy)")
    pdump.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    pdump.add_argument("--out-probe", default="models/bl7_state_probe.pt")
    pdump.add_argument("--train-cap", type=int, default=300_000,
                       help="Max total training points for the linear probe "
                            "(stratified-subsampled per session to cap memory).")

    args = ap.parse_args()
    if args.cmd == "randomization":
        randomization_probe(args)
    elif args.cmd == "info-density":
        info_density_probes(args)
    elif args.cmd == "dump-state-predictions":
        dump_state_predictions(args)


if __name__ == "__main__":
    main()
