"""BL-7 within-state-1 gap — pre-research diagnostics (Diag 1-3).

Diagnostic 1 (HARD GATE): per-class precision/recall/F1 of a linear FSM-state
  probe on frozen BL-7 v1 features, evaluated on the stratified-100K test set,
  both natural-distribution and class-balanced.
Diagnostic 2: within-state-1 mutual-information sketch — which sensor-window
  features carry information about (a) trace template, (b) trace length.
Diagnostic 3: trace-length predictability — can a GBM predict state-1 trace
  length from sensor-window features?

Diag 4 (convergence-trajectory log parsing) is a separate, login-node task.

All diagnostics use the frozen BL-7 v1 encoder embeddings already on disk
(sensor_embeddings_bl7/<sess>_emb.npy) and the deterministic stratified-100K
subsetting (EVAL_SUBSET_SEED=1234567, see evaluate.py).

Usage:
    python3 bl7_diagnostics.py diag1
    python3 bl7_diagnostics.py diag2
    python3 bl7_diagnostics.py diag3
    python3 bl7_diagnostics.py all
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DATA_DIR, SENSOR_COLS, BASE_DIR

EVAL_SUBSET_SEED = 1234567
EMB_DIR = BASE_DIR / "sensor_embeddings_bl7"
RESULTS = BASE_DIR / "results"

# Fold-0 splits (from submit_ar_modern_bl7_5fold.sh)
FOLD0_TRAIN = ["2025-03-17_10-36-44", "2025-03-20_09-31-56"]
VAL_SESSIONS = ["2025-03-21_11-21-42", "2025-03-25_13-23-42"]
TEST_SESSIONS = ["2025-03-25_13-39-06", "2025-03-26_11-03-04"]
WINDOW = 500


# ── shared helpers ───────────────────────────────────────────────────────
def _classification_report(preds, labels, n_classes=3):
    """Per-class precision/recall/F1 + macro-F1 + balanced + overall accuracy."""
    preds = np.asarray(preds).astype(int)
    labels = np.asarray(labels).astype(int)
    rep = {"per_class": {}}
    f1s, recalls = [], []
    for c in range(n_classes):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        support = int((labels == c).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rep["per_class"][f"state_{c}"] = {
            "precision": prec, "recall": rec, "f1": f1, "support": support}
        f1s.append(f1); recalls.append(rec)
    rep["macro_f1"] = sum(f1s) / n_classes
    rep["balanced_accuracy"] = sum(recalls) / n_classes
    rep["accuracy"] = float((preds == labels).mean())
    return rep


def _load_session_embeddings(sess):
    """Load (N, K, d) BL-7 embeddings for a session, flattened to (N, K*d)."""
    emb = np.load(EMB_DIR / f"{sess}_emb.npy")
    if emb.ndim == 3:
        emb = emb.reshape(emb.shape[0], -1)
    return emb.astype(np.float32)


def _load_session_states(sess):
    df = pd.read_parquet(DATA_DIR / f"{sess}.parquet", columns=["pendulum_state"])
    df = df.loc[:, ~df.columns.duplicated()]
    return df["pendulum_state"].values.astype(np.int64)


def _load_session_sensors(sess):
    """Load (N, 7) raw sensor array for a session."""
    cols = list(dict.fromkeys(list(SENSOR_COLS)))
    df = pd.read_parquet(DATA_DIR / f"{sess}.parquet", columns=cols)
    df = df.loc[:, ~df.columns.duplicated()]
    return df[cols].values.astype(np.float32)


def _stratified_100k_indices(fsm_arr, target=100_000):
    """Replicate evaluate.py's stratified subsetting: ALL state-0 + ALL state-2
    + (target - those) state-1 sampled via EVAL_SUBSET_SEED. Returns sorted idx
    into the array fsm_arr was built from."""
    idx_0 = np.where(fsm_arr == 0)[0]
    idx_2 = np.where(fsm_arr == 2)[0]
    idx_1_pool = np.where(fsm_arr == 1)[0]
    n_state1 = max(0, target - len(idx_0) - len(idx_2))
    rng = np.random.default_rng(EVAL_SUBSET_SEED)
    idx_1 = rng.permutation(idx_1_pool)[:n_state1]
    return np.sort(np.concatenate([idx_0, idx_1, idx_2]))


def _window_features(sensors, end_idx, window=WINDOW):
    """7 features per channel over the `window`-tick window ending at end_idx.
    sensors: (N, 7). Returns dict {channel_name: {feature: value}} including a
    derived 'control_error' channel = target_x - current_x.
    """
    start = max(0, end_idx - window + 1)
    w = sensors[start:end_idx + 1]  # (<=window, 7)
    if w.shape[0] < window:
        pad = np.zeros((window - w.shape[0], 7), dtype=np.float32)
        w = np.concatenate([pad, w], axis=0)
    # derived control-error channel
    ti = SENSOR_COLS.index("target_x")
    ci = SENSOR_COLS.index("current_x")
    ctrl_err = (w[:, ti] - w[:, ci]).reshape(-1, 1)
    w8 = np.concatenate([w, ctrl_err], axis=1)  # (window, 8)
    names = list(SENSOR_COLS) + ["control_error_derived"]
    feats = {}
    for ci2, name in enumerate(names):
        col = w8[:, ci2]
        d10 = np.diff(col[-11:]) if len(col) >= 11 else np.array([0.0])
        feats[name] = {
            "raw_last_tick": float(col[-1]),
            "mean_window": float(col.mean()),
            "mean_last_50": float(col[-50:].mean()),
            "mean_last_10": float(col[-10:].mean()),
            "std_window": float(col.std()),
            "first_diff_mean_last_10": float(d10.mean()),
            "first_diff_std_last_10": float(d10.std()),
        }
    return feats


FEATURE_NAMES = ["raw_last_tick", "mean_window", "mean_last_50", "mean_last_10",
                 "std_window", "first_diff_mean_last_10", "first_diff_std_last_10"]
CHANNEL_NAMES = list(SENSOR_COLS) + ["control_error_derived"]


# ── Diagnostic 1 — per-class F1 (HARD GATE) ──────────────────────────────
def diag1(args):
    print("=== Diagnostic 1: per-class F1 of linear FSM-state probe ===")
    device = torch.device(args.device)

    # Fit probe on fold-0 train sessions (subsample to cap memory).
    Xtr, ytr = [], []
    per_sess_cap = args.train_cap // max(len(FOLD0_TRAIN), 1)
    rng = np.random.default_rng(0)
    for sess in FOLD0_TRAIN:
        emb = _load_session_embeddings(sess)
        st = _load_session_states(sess)
        # stratified subsample so all 3 states represented
        picks = []
        for s in (0, 1, 2):
            si = np.where(st == s)[0]
            take = min(len(si), per_sess_cap // 3)
            if take:
                picks.append(rng.choice(si, size=take, replace=False))
        idx = np.sort(np.concatenate(picks)) if picks else np.arange(min(len(st), per_sess_cap))
        Xtr.append(emb[idx]); ytr.append(st[idx])
    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    print(f"  probe train set: {Xtr.shape}, label dist {np.bincount(ytr)}")

    Xt = torch.from_numpy(Xtr).float().to(device)
    yt = torch.from_numpy(ytr).to(device)
    head = nn.Linear(Xt.shape[1], 3).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    counts = torch.bincount(yt, minlength=3).float().clamp(min=1)
    cw = ((1.0 / counts) / (1.0 / counts).sum()).to(device)
    for _ in range(800):
        opt.zero_grad()
        loss = F.cross_entropy(head(Xt), yt, weight=cw)
        loss.backward(); opt.step()
    head.eval()

    # Build test set: stratified-100K over the 2 test sessions concatenated.
    emb_test, st_test = [], []
    for sess in TEST_SESSIONS:
        emb_test.append(_load_session_embeddings(sess))
        st_test.append(_load_session_states(sess))
    emb_test = np.concatenate(emb_test)
    st_test = np.concatenate(st_test)
    strat_idx = _stratified_100k_indices(st_test, target=100_000)
    Xte = emb_test[strat_idx]
    yte = st_test[strat_idx]
    print(f"  stratified-100K test set: {Xte.shape}, label dist {np.bincount(yte)}")

    with torch.no_grad():
        preds_nat = head(torch.from_numpy(Xte).float().to(device)).argmax(-1).cpu().numpy()
    nat_report = _classification_report(preds_nat, yte)

    # class-balanced resample: equal counts of each state
    min_n = int(min(np.bincount(yte)))
    bal_pick = []
    rng2 = np.random.default_rng(EVAL_SUBSET_SEED)
    for s in (0, 1, 2):
        si = np.where(yte == s)[0]
        bal_pick.append(rng2.choice(si, size=min_n, replace=False))
    bal_idx = np.concatenate(bal_pick)
    with torch.no_grad():
        preds_bal = head(torch.from_numpy(Xte[bal_idx]).float().to(device)).argmax(-1).cpu().numpy()
    bal_report = _classification_report(preds_bal, yte[bal_idx])

    out = {
        "fold": 0,
        "probe_train_sessions": FOLD0_TRAIN,
        "test_sessions": TEST_SESSIONS,
        "natural_distribution": nat_report,
        "class_balanced_distribution": {**bal_report, "per_class_n": min_n},
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "diag1_linear_probe_per_class.json").write_text(json.dumps(out, indent=2))

    # Routing
    s1_f1 = nat_report["per_class"]["state_1"]["f1"]
    s2_f1 = nat_report["per_class"]["state_2"]["f1"]
    acc = nat_report["accuracy"]
    bal_macro = bal_report["macro_f1"]
    print(f"\n  NATURAL: acc={acc:.4f} macro-F1={nat_report['macro_f1']:.4f}")
    for s in (0, 1, 2):
        pc = nat_report["per_class"][f"state_{s}"]
        print(f"    state-{s}: P={pc['precision']:.4f} R={pc['recall']:.4f} "
              f"F1={pc['f1']:.4f} (n={pc['support']})")
    print(f"  CLASS-BALANCED: acc={bal_report['accuracy']:.4f} macro-F1={bal_macro:.4f}")
    for s in (0, 1, 2):
        pc = bal_report["per_class"][f"state_{s}"]
        print(f"    state-{s}: P={pc['precision']:.4f} R={pc['recall']:.4f} F1={pc['f1']:.4f}")

    # Branch logic per the spec
    all_f1_near_acc = all(
        abs(nat_report["per_class"][f"state_{s}"]["f1"] - acc) < 0.05 for s in (0, 1, 2))
    branch_c = all_f1_near_acc and (acc - bal_macro) > 0.10
    if branch_c:
        branch = "C — STOP & ESCALATE: probe largely exploiting the class prior"
    elif s1_f1 >= 0.95 and s2_f1 >= 0.85:
        branch = "A — encoder genuinely nails coarse state; proceed to within-state-1 focus"
    elif s1_f1 >= 0.95 and s2_f1 < 0.7:
        branch = "B — state-2 features noisy; expand scope to minority representation"
    else:
        branch = "AMBIGUOUS — s1_f1/s2_f1 don't match A/B/C cleanly; review manually"
    out["routing_branch"] = branch
    (RESULTS / "diag1_linear_probe_per_class.json").write_text(json.dumps(out, indent=2))
    print(f"\n  ROUTING: Branch {branch}")
    return out


# ── Diagnostic 2 — within-state-1 MI sketch ──────────────────────────────
def diag2(args):
    print("=== Diagnostic 2: within-state-1 MI sketch ===")
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.cluster import KMeans

    # Collect state-1 ticks from the stratified-100K test set, with their
    # (session, row) so we can slice raw windows + read traces.
    rows = []  # (sess_idx, row, trace_tuple, trace_len)
    sensors_by_sess = {}
    for si, sess in enumerate(TEST_SESSIONS):
        sensors_by_sess[si] = _load_session_sensors(sess)
        df = pd.read_parquet(DATA_DIR / f"{sess}.parquet",
                             columns=["trace", "trace_length", "pendulum_state"])
        df = df.loc[:, ~df.columns.duplicated()]
        st = df["pendulum_state"].values.astype(np.int64)
        # this session's contribution to the concatenated array
        rows.append((si, df, st))
    # Build concatenated state array, get stratified idx, map back to (sess,row)
    st_all = np.concatenate([r[2] for r in rows])
    strat_idx = _stratified_100k_indices(st_all, target=100_000)
    # offsets
    lens = [len(r[2]) for r in rows]
    offsets = np.cumsum([0] + lens)
    state1_items = []  # (sess_idx, local_row, trace, trace_len)
    for gi in strat_idx:
        if st_all[gi] != 1:
            continue
        si = int(np.searchsorted(offsets, gi, side="right") - 1)
        local = int(gi - offsets[si])
        df = rows[si][1]
        tr = np.asarray(df.iloc[local]["trace"], dtype=np.int64)
        state1_items.append((si, local, tr, len(tr)))
    print(f"  state-1 windows in stratified-100K test: {len(state1_items)}")
    if len(state1_items) < 100:
        raise RuntimeError(f"Too few state-1 windows ({len(state1_items)}) — abort")

    # template_id via KMeans(5) on bag-of-token count vectors (cheap, scales).
    # Also report exact-match top-5 coverage as the "is there template structure" signal.
    VOCAB = 648
    bow = np.zeros((len(state1_items), VOCAB), dtype=np.float32)
    for i, (_, _, tr, _) in enumerate(state1_items):
        for t in tr:
            if 0 <= t < VOCAB:
                bow[i, t] += 1.0
    km = KMeans(n_clusters=5, n_init=4, random_state=0).fit(bow)
    template_id = km.labels_
    # exact-match coverage
    exact = Counter(tuple(tr.tolist()) for _, _, tr, _ in state1_items)
    top5_exact = exact.most_common(5)
    top5_cov = sum(c for _, c in top5_exact) / len(state1_items)
    print(f"  KMeans-5 cluster sizes: {np.bincount(template_id).tolist()}")
    print(f"  exact-match top-5 template coverage: {top5_cov:.3f}")

    # length_bucket: short/modal/long relative to modal length
    lengths = np.array([it[3] for it in state1_items])
    modal_len = Counter(lengths.tolist()).most_common(1)[0][0]
    length_bucket = np.where(lengths < modal_len - 2, 0,
                             np.where(lengths > modal_len + 2, 2, 1))  # short/modal/long

    # 56 features per window
    feat_mat = np.zeros((len(state1_items), len(CHANNEL_NAMES) * len(FEATURE_NAMES)),
                        dtype=np.float32)
    for i, (si, local, _, _) in enumerate(state1_items):
        f = _window_features(sensors_by_sess[si], local)
        col = 0
        for ch in CHANNEL_NAMES:
            for fn in FEATURE_NAMES:
                feat_mat[i, col] = f[ch][fn]
                col += 1

    mi_template = mutual_info_classif(feat_mat, template_id, random_state=0)
    mi_length = mutual_info_classif(feat_mat, length_bucket, random_state=0)

    # organise per (channel, feature)
    mi_per = {}
    flat = []
    col = 0
    for ch in CHANNEL_NAMES:
        mi_per[ch] = {}
        for fn in FEATURE_NAMES:
            mi_per[ch][fn] = {"vs_template": float(mi_template[col]),
                              "vs_length": float(mi_length[col])}
            flat.append((ch, fn, float(mi_template[col]), float(mi_length[col])))
            col += 1
    top5_template = sorted(flat, key=lambda x: -x[2])[:5]
    top5_length = sorted(flat, key=lambda x: -x[3])[:5]

    out = {
        "fold": 0,
        "n_state_1_windows": len(state1_items),
        "template_id_distribution": {str(i): int(c) for i, c in
                                     enumerate(np.bincount(template_id, minlength=5))},
        "exact_match_top5_coverage": top5_cov,
        "length_bucket_distribution": {
            "short": int((length_bucket == 0).sum()),
            "modal": int((length_bucket == 1).sum()),
            "long": int((length_bucket == 2).sum())},
        "modal_length": int(modal_len),
        "mi_per_channel_feature": mi_per,
        "top_5_features_by_template_mi": [
            {"channel": c, "feature": f, "mi": m} for c, f, m, _ in top5_template],
        "top_5_features_by_length_mi": [
            {"channel": c, "feature": f, "mi": m} for c, f, _, m in top5_length],
    }
    (RESULTS / "diag2_mi_sketch.json").write_text(json.dumps(out, indent=2))
    print(f"  top-5 features by template-MI:")
    for c, f, m, _ in top5_template:
        print(f"    {c}.{f}: {m:.4f}")
    print(f"  top-5 features by length-MI:")
    for c, f, _, m in top5_length:
        print(f"    {c}.{f}: {m:.4f}")
    return out


# ── Diagnostic 3 — trace-length predictability ───────────────────────────
def diag3(args):
    print("=== Diagnostic 3: trace-length predictability (GBM) ===")
    import lightgbm as lgb
    from sklearn.metrics import r2_score, mean_absolute_error

    def collect(sessions, cap):
        feats, lens = [], []
        rng = np.random.default_rng(0)
        for sess in sessions:
            sensors = _load_session_sensors(sess)
            df = pd.read_parquet(DATA_DIR / f"{sess}.parquet",
                                 columns=["trace", "pendulum_state"])
            df = df.loc[:, ~df.columns.duplicated()]
            st = df["pendulum_state"].values.astype(np.int64)
            s1 = np.where(st == 1)[0]
            if len(s1) > cap:
                s1 = np.sort(rng.choice(s1, size=cap, replace=False))
            for local in s1:
                f = _window_features(sensors, int(local))
                row = [f[ch][fn] for ch in CHANNEL_NAMES for fn in FEATURE_NAMES]
                feats.append(row)
                lens.append(len(df.iloc[local]["trace"]))
        return np.array(feats, dtype=np.float32), np.array(lens, dtype=np.float32)

    Xtr, ytr = collect(FOLD0_TRAIN, cap=args.gbm_cap)
    Xva, yva = collect(VAL_SESSIONS, cap=args.gbm_cap)
    print(f"  train {Xtr.shape}, val {Xva.shape}")

    model = lgb.LGBMRegressor(n_estimators=200, max_depth=4, random_state=0,
                              verbose=-1)
    model.fit(Xtr, ytr)
    pred = model.predict(Xva)
    r2 = float(r2_score(yva, pred))
    mae = float(mean_absolute_error(yva, pred))
    trivial_mae = float(mean_absolute_error(yva, np.full_like(yva, ytr.mean())))

    fnames = [f"{ch}.{fn}" for ch in CHANNEL_NAMES for fn in FEATURE_NAMES]
    imp = sorted(zip(fnames, model.feature_importances_.tolist()),
                 key=lambda x: -x[1])[:10]

    out = {
        "fold": 0,
        "n_train": int(len(ytr)), "n_val": int(len(yva)),
        "mean_trace_length": float(ytr.mean()),
        "std_trace_length": float(ytr.std()),
        "trivial_baseline_r2": 0.0,
        "trivial_baseline_mae": trivial_mae,
        "gbm_val_r2": r2,
        "gbm_val_mae": mae,
        "top_10_features_by_importance": [
            {"feature": n, "importance": int(v)} for n, v in imp],
    }
    (RESULTS / "diag3_length_predictability.json").write_text(json.dumps(out, indent=2))
    print(f"  GBM val R²={r2:.4f}  MAE={mae:.2f}  (trivial MAE={trivial_mae:.2f})")
    print(f"  top features: {[n for n, _ in imp[:5]]}")
    return out


# ── Diagnostic 4 — per-state TF-acc of the deployed checkpoint ───────────
# The aggregate val trajectory is already in the training logs (both variants
# plateau by ~step 2000; BL-7 mildly val-regresses after step 3000). Diag4's
# only unique contribution is the *per-state* split, and the checkpoint that
# actually produced the stratified-100K NED numbers is the val-loss-selected
# `ar_modern_best.pt` — so we evaluate exactly that, one per variant.
DIAG4_VARIANTS = [
    # (label, checkpoint dir, emb dir)
    ("bl7", "ar_modern_bl7_fold_0", "sensor_embeddings_bl7"),
    ("bl2", "ar_modern_fold_0", "sensor_embeddings"),
]
DIAG4_CKPT = "ar_modern_best.pt"


def _per_state_tf_acc(model, ds, device, pad_id, batch_size=128):
    """Teacher-forced forward over the whole dataset; returns per-state
    {correct_tokens, total_tokens, tf_acc} keyed by fsm_state."""
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    corr = {0: 0.0, 1: 0.0, 2: 0.0}
    tot = {0: 0.0, 1: 0.0, 2: 0.0}
    model.eval()
    with torch.no_grad():
        for batch in dl:
            x0 = batch["trace"].to(device)
            emb = batch["sensor_emb"].to(device)
            fsm = batch["fsm_state"].numpy()
            pad_mask = (x0 != pad_id).float()
            B, L = x0.shape
            bos = torch.full((B, 1), model.bos_id, dtype=x0.dtype, device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16) \
                    if device.type == "cuda" else _nullctx():
                logits = model.backbone(torch.cat([bos, x0], dim=1), emb)[:, :L, :]
            preds = logits.argmax(dim=-1)
            per_tok_corr = ((preds == x0).float() * pad_mask).sum(dim=1).cpu().numpy()
            per_tok_tot = pad_mask.sum(dim=1).cpu().numpy()
            for s in (0, 1, 2):
                m = fsm == s
                corr[s] += float(per_tok_corr[m].sum())
                tot[s] += float(per_tok_tot[m].sum())
    return {s: {"correct": corr[s], "total": tot[s],
                "tf_acc": (corr[s] / tot[s]) if tot[s] else None} for s in (0, 1, 2)}


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def diag4(args):
    print("=== Diagnostic 4: per-state TF-acc of deployed checkpoint ===")
    from config import ModelConfig
    from data import TraceDataset
    from ar_modern_model import ARModernModel

    device = torch.device(args.device)
    out = {"fold": 0, "checkpoint": DIAG4_CKPT, "variants": {}}
    for label, ckpt_dir, emb_dirname in DIAG4_VARIANTS:
        cdir = BASE_DIR / "models" / ckpt_dir
        emb_dir = BASE_DIR / emb_dirname
        ckpt = torch.load(cdir / DIAG4_CKPT, map_location="cpu", weights_only=False)
        mcfg = ModelConfig(**ckpt["model_config"])
        ds = TraceDataset(VAL_SESSIONS, mcfg.max_seq_len, mcfg.vocab_size,
                          emb_dir=emb_dir, data_dir=DATA_DIR)
        m = ARModernModel(mcfg).to(device)
        sd = ckpt["model"]
        sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
              for k, v in sd.items()}
        m.load_state_dict(sd)
        per_state = _per_state_tf_acc(m, ds, device, mcfg.vocab_size)
        agg_corr = sum(per_state[s]["correct"] for s in (0, 1, 2))
        agg_tot = sum(per_state[s]["total"] for s in (0, 1, 2))
        row = {"ckpt_step": int(ckpt.get("step", -1)),
               "tf_acc_all": agg_corr / agg_tot if agg_tot else None,
               "tf_acc_state_0": per_state[0]["tf_acc"],
               "tf_acc_state_1": per_state[1]["tf_acc"],
               "tf_acc_state_2": per_state[2]["tf_acc"],
               "n_tokens_state_0": per_state[0]["total"],
               "n_tokens_state_1": per_state[1]["total"],
               "n_tokens_state_2": per_state[2]["total"]}
        out["variants"][label] = row
        print(f"  [{label}] step {row['ckpt_step']}: all={row['tf_acc_all']:.4f} "
              f"s0={row['tf_acc_state_0']:.4f} s1={row['tf_acc_state_1']:.4f} "
              f"s2={row['tf_acc_state_2']:.4f}")

    if "bl7" in out["variants"] and "bl2" in out["variants"]:
        b7, b2 = out["variants"]["bl7"], out["variants"]["bl2"]
        out["gap_bl2_minus_bl7"] = {
            "tf_acc_all": b2["tf_acc_all"] - b7["tf_acc_all"],
            "tf_acc_state_0": b2["tf_acc_state_0"] - b7["tf_acc_state_0"],
            "tf_acc_state_1": b2["tf_acc_state_1"] - b7["tf_acc_state_1"],
            "tf_acc_state_2": b2["tf_acc_state_2"] - b7["tf_acc_state_2"],
        }
        g = out["gap_bl2_minus_bl7"]
        print(f"\n  GAP (BL2 - BL7): all={g['tf_acc_all']:+.4f} "
              f"s0={g['tf_acc_state_0']:+.4f} s1={g['tf_acc_state_1']:+.4f} "
              f"s2={g['tf_acc_state_2']:+.4f}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "diag4_convergence_trajectory.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["diag1", "diag2", "diag3", "diag4", "all"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--train-cap", type=int, default=300_000,
                    help="Diag1: cap on linear-probe training points.")
    ap.add_argument("--gbm-cap", type=int, default=120_000,
                    help="Diag3: per-session cap on GBM training points.")
    args = ap.parse_args()
    if args.cmd in ("diag1", "all"):
        diag1(args)
    if args.cmd in ("diag2", "all"):
        diag2(args)
    if args.cmd in ("diag3", "all"):
        diag3(args)
    if args.cmd in ("diag4", "all"):
        diag4(args)


if __name__ == "__main__":
    main()
