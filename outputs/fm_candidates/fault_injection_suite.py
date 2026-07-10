"""E1a + E1b — fault detection with the paper's own system.

Two detectors, one perturbation pipeline:

  E1a encoder one-class (deployed mode, sensors only): Mahalanobis
      distance of the Chronos-2 window embedding to the training
      distribution. Reuses the generation encoder; no anomaly labels.
  E1b trace consistency (twin mode): NED between the trace decoded from
      the (possibly perturbed) sensor window and the observed trace.

Faults injected at the SENSOR level on held-out test ticks:
  sensor_noise : add N(0, 3*sigma_train) per channel across the window
  stuck_sensor : freeze the angle channel at its value 1000 ticks ago
  timing_50ms  : shift all channels 50 ticks within the window

Fidelity check: re-encoded UNPERTURBED windows must match the stored
sensor_embeddings_w4000 rows (same model ckpt + same seeded projection).

Outputs:
  results/anomaly_encoder_oneclass/results.json   {"auc": {...}}
  results/anomaly_trace_consistency/results.json  merged {"auc": {...}}

Run under slimllm_bw (chronos needs transformers>=4.51); torch code is
env-agnostic.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import ModelConfig, SensorConfig, DATA_DIR, SENSOR_COLS
from data import TraceDataset, load_token_mapping
from metrics import normalized_edit_distance
from sensor_encoder import (
    encode_multivariate_batch, resolve_checkpoint, _apply_chronos2_attn_patch,
)

FM = Path(__file__).parent
RES = FM / "results"

TEST_SESSIONS = ["2025-03-25_13-39-06", "2025-03-26_11-03-04"]
FOLD_TRAIN_0 = ["2025-03-17_10-36-44", "2025-03-20_09-31-56"]
W = 4000
ANGLE_CH = SENSOR_COLS.index("current_angle")


def load_session_sensors(sess: str) -> np.ndarray:
    import pandas as pd
    df = pd.read_parquet(Path(DATA_DIR) / f"{sess}.parquet",
                         columns=SENSOR_COLS)
    return df[SENSOR_COLS].values.astype(np.float32)      # (n, 7)


def window_at(sensors: np.ndarray, t: int) -> np.ndarray:
    w = sensors[max(0, t - W + 1):t + 1]
    if w.shape[0] < W:
        w = np.concatenate([np.zeros((W - w.shape[0], w.shape[1]),
                                     dtype=np.float32), w])
    return w.T.copy()                                      # (7, W)


def perturb(win: np.ndarray, fault: str, sigma: np.ndarray,
            rng: np.random.Generator) -> np.ndarray:
    w = win.copy()
    if fault == "sensor_noise":
        w += rng.normal(0.0, 3.0 * sigma[:, None], size=w.shape).astype(np.float32)
    elif fault == "stuck_sensor":
        w[ANGLE_CH, -1000:] = w[ANGLE_CH, -1000]
    elif fault == "timing_50ms":
        w = np.roll(w, 50, axis=1)
        w[:, :50] = w[:, [50]]
    else:
        raise ValueError(fault)
    return w


@torch.no_grad()
def encode_windows(model, proj, wins: np.ndarray, device,
                   batch: int = 8) -> np.ndarray:
    out = []
    for i in range(0, len(wins), batch):
        b = torch.from_numpy(wins[i:i + batch]).to(device)
        pooled = encode_multivariate_batch(model, b, 7)     # (B, 7, 768)
        flat = pooled.reshape(pooled.shape[0], -1).float().cpu()
        out.append(proj(flat).numpy())
    return np.concatenate(out)                              # (N, 512)


class Mahalanobis:
    def __init__(self, X: np.ndarray, ridge: float = 1e-3):
        self.mu = X.mean(0)
        C = np.cov(X.T) + ridge * np.eye(X.shape[1])
        self.P = np.linalg.inv(C)

    def score(self, X: np.ndarray) -> np.ndarray:
        d = X - self.mu
        return np.einsum("ij,jk,ik->i", d, self.P, d)


def auc_rank(normal, anomalous) -> float:
    from scipy.stats import mannwhitneyu
    u, _ = mannwhitneyu(anomalous, normal, alternative="greater")
    return float(u / (len(normal) * len(anomalous)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ticks", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chronos-ckpt",
                    default=str(FM / "models/chronos2_finetuned_w4000"))
    ap.add_argument("--decoder-ckpt",
                    default=str(FM / "models/ar_modern_bl2_w4000_fold_0/ar_modern_best.pt"))
    ap.add_argument("--proj-seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")

    # ── Chronos encoder + seeded projection (identical to precompute) ──
    from chronos import Chronos2Pipeline
    _apply_chronos2_attn_patch()
    ckpt = resolve_checkpoint(args.chronos_ckpt) or args.chronos_ckpt
    pipe = Chronos2Pipeline.from_pretrained(ckpt, device_map="cuda")
    model = pipe.model
    cfg = SensorConfig()
    torch.manual_seed(args.proj_seed)
    input_dim = 7 * 768
    proj = nn.Sequential(
        nn.Linear(input_dim, cfg.proj_dim * 2), nn.GELU(),
        nn.Linear(cfg.proj_dim * 2, cfg.proj_dim),
    ).cpu().eval()

    # ── Sample test ticks; build windows ───────────────────────────────
    sensors = {s: load_session_sensors(s) for s in TEST_SESSIONS}
    train_stats = np.concatenate(
        [load_session_sensors(s) for s in FOLD_TRAIN_0])
    sigma = train_stats.std(0).astype(np.float32)

    picks = []
    for s in TEST_SESSIONS:
        n = sensors[s].shape[0]
        idx = rng.choice(np.arange(W, n), size=args.n_ticks // 2,
                         replace=False)
        picks += [(s, int(t)) for t in idx]
    print(f"{len(picks)} test ticks sampled")

    normal_w = np.stack([window_at(sensors[s], t) for s, t in picks])
    variants = {"normal": normal_w}
    for f in ("sensor_noise", "stuck_sensor", "timing_50ms"):
        variants[f] = np.stack([perturb(w, f, sigma, rng) for w in normal_w])

    embs = {k: encode_windows(model, proj, v, device)
            for k, v in variants.items()}

    # Fidelity check vs stored embeddings.
    stored = {s: np.load(FM / f"sensor_embeddings_w{W}" / f"{s}_emb.npy",
                         mmap_mode="r") for s in TEST_SESSIONS}
    ref_rows = np.stack([np.asarray(stored[s][t]) for s, t in picks[:64]])
    diff = np.abs(ref_rows - embs["normal"][:64]).max()
    print(f"fidelity vs stored embeddings: max|diff| = {diff:.2e}")
    assert diff < 1e-3, "re-encoding does not reproduce stored embeddings"

    # ── E1a: one-class on train embeddings ─────────────────────────────
    train_emb = np.concatenate([
        np.asarray(np.load(FM / f"sensor_embeddings_w{W}" / f"{s}_emb.npy",
                           mmap_mode="r")[::10]) for s in FOLD_TRAIN_0])
    print(f"one-class fit on {len(train_emb)} train embeddings")
    # The embedding distribution is multimodal (three controller states), so
    # a single-Gaussian Mahalanobis washes out per-state structure; kNN
    # distance is the primary detector, Mahalanobis kept as robustness.
    oc = Mahalanobis(train_emb)
    knn_bank = train_emb[::5].astype(np.float32)          # ~20k refs

    def knn_score(X, k=5, block=256):
        out = np.empty(len(X))
        bank = torch.from_numpy(knn_bank).to(device)
        for i in range(0, len(X), block):
            q = torch.from_numpy(X[i:i + block].astype(np.float32)).to(device)
            d = torch.cdist(q, bank)
            out[i:i + block] = d.kthvalue(k, dim=1).values.cpu().numpy()
        return out

    e1a, e1a_maha = {}, {}
    s_norm_knn = knn_score(embs["normal"])
    s_norm_maha = oc.score(embs["normal"])
    for f in ("sensor_noise", "stuck_sensor", "timing_50ms"):
        e1a[f] = auc_rank(s_norm_knn, knn_score(embs[f]))
        e1a_maha[f] = auc_rank(s_norm_maha, oc.score(embs[f]))
        print(f"E1a {f}: kNN AUC {e1a[f]:.4f}  (Mahalanobis {e1a_maha[f]:.4f})")

    # ── Baseline: one-class on raw per-channel window statistics (no
    # foundation encoder). Isolates what the Chronos encoder buys for
    # detection. Features: mean/std/min/max/last per channel over the
    # window -> 7*5 = 35-d, Mahalanobis on train windows. ─────────────────
    def raw_stats(wins):                         # wins: (N, 7, W) -> (N, 35)
        return np.concatenate([
            wins.mean(2), wins.std(2), wins.min(2), wins.max(2), wins[:, :, -1],
        ], axis=1)
    # Fit on a sample of train windows.
    tr_sess = FOLD_TRAIN_0
    tr_wins = []
    for s in tr_sess:
        sn = load_session_sensors(s)
        ti = rng.choice(np.arange(W, sn.shape[0]), size=1000, replace=False)
        tr_wins += [window_at(sn, int(t)) for t in ti]
    base_oc = Mahalanobis(raw_stats(np.stack(tr_wins)))
    base_norm = base_oc.score(raw_stats(variants["normal"]))
    e1base = {}
    for f in ("sensor_noise", "stuck_sensor", "timing_50ms"):
        e1base[f] = auc_rank(base_norm, base_oc.score(raw_stats(variants[f])))
        print(f"baseline (raw stats) {f}: AUC {e1base[f]:.4f}")

    # ── E1b: decode perturbed embeddings, NED vs observed traces ───────
    from ar_modern_model import ARModernModel
    vocab_size, _, _ = load_token_mapping()
    mcfg = ModelConfig(vocab_size=vocab_size, mask_token_id=vocab_size,
                       d_sensor_emb=512)
    dck = torch.load(args.decoder_ckpt, map_location=device,
                     weights_only=False)
    sd = {k.removeprefix("_orig_mod."): v for k, v in dck["model"].items()}
    dec = ARModernModel(ModelConfig(**dck["model_config"])).to(device)
    dec.load_state_dict(sd)
    # evaluate.py applies EMA weights; without them the decoder is far from
    # its reported quality (bug found 2026-07-02: NED 0.55 vs 0.02).
    if dck.get("ema") is not None:
        from mdlm_train import EMA
        ema = EMA(dec.parameters(), decay=dck["ema"]["decay"])
        ema.load_state_dict(dck["ema"])
        ema.to(device)
        ema.copy_to(dec.parameters())
        print("applied EMA weights to decoder")
    dec.eval()

    ds = TraceDataset(TEST_SESSIONS, mcfg.max_seq_len, mcfg.vocab_size,
                      emb_dir=FM / f"sensor_embeddings_w{W}",
                      data_dir=Path(DATA_DIR))
    sess_off, off = {}, 0
    for s in TEST_SESSIONS:
        sess_off[s] = off
        off += sensors[s].shape[0]
    observed = [ds.traces[sess_off[s] + t] for s, t in picks]

    @torch.no_grad()
    def decode(emb: np.ndarray) -> list:
        outs = []
        for i in range(0, len(emb), 64):
            e = torch.from_numpy(emb[i:i + 64]).float().to(device)
            g = dec.sample(e, seq_len=mcfg.max_seq_len, temperature=1.0,
                           greedy=True)
            outs += [row.tolist() for row in g.cpu()]
        return outs

    def crop(seq):
        out = []
        for t in seq:
            if t >= vocab_size:
                break
            out.append(int(t))
        return out

    ned = {}
    for k in variants:
        gen = decode(embs[k])
        ned[k] = np.array([
            normalized_edit_distance(crop(g), list(o))
            for g, o in zip(gen, observed)])
        print(f"E1b {k}: NED mean {ned[k].mean():.4f}")

    e1b = {f: auc_rank(ned["normal"], ned[f])
           for f in ("sensor_noise", "stuck_sensor", "timing_50ms")}
    for f, a in e1b.items():
        print(f"E1b {f}: AUC {a:.4f}")

    # ── Write results ───────────────────────────────────────────────────
    d1 = RES / "anomaly_encoder_oneclass"
    d1.mkdir(parents=True, exist_ok=True)
    (d1 / "results.json").write_text(json.dumps(
        {"n_ticks": len(picks), "auc": e1a,
         "auc_mahalanobis": e1a_maha,
         "auc_rawstats_baseline": e1base}, indent=2))
    d2 = RES / "anomaly_trace_consistency"
    d2.mkdir(parents=True, exist_ok=True)
    prev = {}
    if (d2 / "results.json").exists():
        prev = json.loads((d2 / "results.json").read_text())
    prev.setdefault("auc", {}).update(e1b)
    (d2 / "results.json").write_text(json.dumps(prev, indent=2))
    print("results written")


if __name__ == "__main__":
    main()
