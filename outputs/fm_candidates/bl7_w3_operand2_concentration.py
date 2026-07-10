"""Wave-3 Probe 2 — operand-2 error concentration on state-1.

Under teacher forcing and free-running greedy decode, measure how
concentrated BL-7 v1's operand-2 prediction errors are: top-K shares of
the error distribution, entropy, top confusion pairs, and the most
error-prone ground-truth operand-2 values.

All on fold-0 val sessions, using the deployed BL-7 v1 decoder
checkpoint (`ar_modern_bl7_fold_0/ar_modern_best.pt`).

Usage:
    python3 bl7_w3_operand2_concentration.py
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import DATA_DIR, BASE_DIR, ModelConfig, VAL_SESSIONS
from data import TraceDataset, load_token_mapping
from ar_modern_model import ARModernModel

RESULTS = BASE_DIR / "results"

CKPT_DIR = "ar_modern_bl7_fold_0"
EMB_DIRNAME = "sensor_embeddings_bl7"
FREE_RUN_CAP = 5000


def _amp_ctx(device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    class _N:
        def __enter__(self): return None
        def __exit__(self, *a): return False
    return _N()


def _entropy_nats(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h


def _concentration(error_op2_counts, total_errors, total_corrects_by_op2):
    """Derived metrics for the operand-2 error distribution."""
    items = sorted(error_op2_counts.items(), key=lambda kv: -kv[1])
    top1 = items[0][1] / total_errors if items else 0.0
    top5 = sum(c for _, c in items[:5]) / total_errors if total_errors else 0.0
    top10 = sum(c for _, c in items[:10]) / total_errors if total_errors else 0.0
    # n values needed to cover 80% of errors
    running = 0
    n_for_80 = 0
    for i, (_, c) in enumerate(items):
        running += c
        if running >= 0.8 * total_errors:
            n_for_80 = i + 1
            break
    return {
        "top_1_operand2_error_share": top1,
        "top_5_operand2_error_share": top5,
        "top_10_operand2_error_share": top10,
        "n_values_to_cover_80pct_errors": n_for_80,
        "error_distribution_entropy_nats": _entropy_nats(error_op2_counts),
        "n_distinct_operand2_values_in_errors": len(error_op2_counts),
    }


def _top10_gt(error_op2_counts, total_corrects_by_op2):
    """Top-10 ground-truth operand-2 values by error count + error rate."""
    items = sorted(error_op2_counts.items(), key=lambda kv: -kv[1])[:10]
    out = []
    for op2, ec in items:
        corr = total_corrects_by_op2.get(op2, 0)
        total = ec + corr
        out.append({
            "gt_operand2": int(op2),
            "error_count": int(ec),
            "total_count": int(total),
            "error_rate": ec / total if total else 0.0,
        })
    return out


def _top20_confusion(conf_pair_counts):
    items = sorted(conf_pair_counts.items(), key=lambda kv: -kv[1])[:20]
    return [{"gt_operand2": int(g), "pred_operand2": int(p), "count": int(c)}
            for (g, p), c in items]


def collect_tf(m, ds, mcfg, triple_t, vs, device, batch_size=128, target_state=1):
    """Teacher-forced pass over the whole val set; collect operand-2
    error stats restricted to tokens whose tick is in `target_state`."""
    pad_id = mcfg.vocab_size
    bos_id = mcfg.vocab_size + 1

    err_op2 = Counter()           # gt operand-2 among operand-2-only errors
    correct_op2 = Counter()       # gt operand-2 where the model got the whole token right
    conf_pairs = Counter()        # (gt_op2, pred_op2) when only op2 differs

    # also per-position-bucket error counts to check W3 stop-and-escalate #5
    err_pos_bins = Counter()      # bucket → count
    POS_BUCKETS = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 91)]

    total_state1_tokens = 0
    total_state1_errors = 0
    total_state1_op2_only_errors = 0

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    with torch.no_grad():
        for batch in dl:
            x0 = batch["trace"].to(device)
            emb = batch["sensor_emb"].to(device)
            fsm = batch["fsm_state"].numpy()
            pad_mask = (x0 != pad_id)
            B, L = x0.shape
            bos = torch.full((B, 1), bos_id, dtype=x0.dtype, device=device)
            with _amp_ctx(device):
                logits = m.backbone(torch.cat([bos, x0], dim=1), emb)[:, :L, :]
            pred = logits.argmax(dim=-1)
            x_safe = x0.clamp(0, vs + 1)
            pred_safe = pred.clamp(0, vs + 1)
            gt_tri = triple_t[x_safe]      # (B, L, 3)
            pr_tri = triple_t[pred_safe]   # (B, L, 3)
            op_eq = (pr_tri[..., 0] == gt_tri[..., 0])
            o1_eq = (pr_tri[..., 1] == gt_tri[..., 1])
            o2_eq = (pr_tri[..., 2] == gt_tri[..., 2])
            # operand-2-only error: opcode + op1 right, op2 wrong
            op2_only_err = op_eq & o1_eq & ~o2_eq & pad_mask
            any_err = (~(op_eq & o1_eq & o2_eq)) & pad_mask
            # token-correct (joint)
            tok_ok = op_eq & o1_eq & o2_eq & pad_mask

            # move pieces to CPU once
            op2_only_err_cpu = op2_only_err.cpu().numpy()
            any_err_cpu = any_err.cpu().numpy()
            tok_ok_cpu = tok_ok.cpu().numpy()
            pad_cpu = pad_mask.cpu().numpy()
            gt_op2_cpu = gt_tri[..., 2].cpu().numpy()
            pr_op2_cpu = pr_tri[..., 2].cpu().numpy()

            for bi in range(B):
                if fsm[bi] != target_state:
                    continue
                total_state1_tokens += int(pad_cpu[bi].sum())
                total_state1_errors += int(any_err_cpu[bi].sum())
                # operand-2-only error positions in this row
                errs_here = np.where(op2_only_err_cpu[bi])[0]
                total_state1_op2_only_errors += len(errs_here)
                for pos in errs_here:
                    g = int(gt_op2_cpu[bi, pos])
                    p = int(pr_op2_cpu[bi, pos])
                    err_op2[g] += 1
                    conf_pairs[(g, p)] += 1
                    for lo, hi in POS_BUCKETS:
                        if lo <= pos < hi:
                            err_pos_bins[f"{lo}-{hi}"] += 1
                            break
                # operand-2 correct (whole token right) — credit the gt op2
                ok_here = np.where(tok_ok_cpu[bi])[0]
                for pos in ok_here:
                    g = int(gt_op2_cpu[bi, pos])
                    correct_op2[g] += 1

    out = _concentration(err_op2, total_state1_op2_only_errors, correct_op2)
    out.update({
        "n_state_1_tokens": int(total_state1_tokens),
        "n_state_1_total_errors": int(total_state1_errors),
        "n_state_1_op2_only_errors": int(total_state1_op2_only_errors),
        "n_distinct_operand2_values_in_corrects": len(correct_op2),
        "top_20_confusion_pairs": _top20_confusion(conf_pairs),
        "top_10_gt_operand2_in_errors": _top10_gt(err_op2, correct_op2),
        "operand2_error_position_bucket": dict(err_pos_bins),
    })
    return out


def _state1_indices(ds, cap, seed=0):
    idx = np.array([i for i, s in enumerate(ds.fsm_states) if s == 1])
    if cap is not None and len(idx) > cap:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=cap, replace=False))
    return idx


def _gather(ds, indices, device):
    trace = torch.stack([ds[i]["trace"] for i in indices])
    emb = torch.stack([ds[i]["sensor_emb"] for i in indices])
    return trace.to(device), emb.to(device)


def collect_fr(m, ds, mcfg, triple_t, vs, device, free_cap=FREE_RUN_CAP,
               batch_size=128, seed=0):
    """Greedy free-running decode, then compare per-position to ground
    truth and collect operand-2 error stats."""
    idx = _state1_indices(ds, cap=free_cap, seed=seed)
    pad_id = mcfg.vocab_size
    bos_id = mcfg.vocab_size + 1

    err_op2 = Counter()
    correct_op2 = Counter()
    conf_pairs = Counter()
    err_pos_bins = Counter()
    POS_BUCKETS = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 91)]

    total_state1_tokens = 0
    total_state1_errors = 0
    total_state1_op2_only_errors = 0

    L = mcfg.max_seq_len
    for s in range(0, len(idx), batch_size):
        sub = idx[s:s + batch_size]
        x0, emb = _gather(ds, sub, device)
        pad_mask = (x0 != pad_id)
        B = x0.shape[0]
        gen = torch.full((B, 1), bos_id, dtype=x0.dtype, device=device)
        for p in range(L):
            with torch.no_grad(), _amp_ctx(device):
                logits = m.backbone(gen, emb)[:, -1, :]
            logits[:, pad_id] = -1e9
            logits[:, bos_id] = -1e9
            nxt = logits.argmax(dim=-1, keepdim=True)
            gen = torch.cat([gen, nxt], dim=1)
        pred = gen[:, 1:]  # (B, L)

        x_safe = x0.clamp(0, vs + 1)
        pr_safe = pred.clamp(0, vs + 1)
        gt_tri = triple_t[x_safe]
        pr_tri = triple_t[pr_safe]
        op_eq = (pr_tri[..., 0] == gt_tri[..., 0])
        o1_eq = (pr_tri[..., 1] == gt_tri[..., 1])
        o2_eq = (pr_tri[..., 2] == gt_tri[..., 2])
        op2_only_err = op_eq & o1_eq & ~o2_eq & pad_mask
        any_err = (~(op_eq & o1_eq & o2_eq)) & pad_mask
        tok_ok = op_eq & o1_eq & o2_eq & pad_mask

        op2_only_err_cpu = op2_only_err.cpu().numpy()
        any_err_cpu = any_err.cpu().numpy()
        tok_ok_cpu = tok_ok.cpu().numpy()
        pad_cpu = pad_mask.cpu().numpy()
        gt_op2_cpu = gt_tri[..., 2].cpu().numpy()
        pr_op2_cpu = pr_tri[..., 2].cpu().numpy()

        for bi in range(B):
            total_state1_tokens += int(pad_cpu[bi].sum())
            total_state1_errors += int(any_err_cpu[bi].sum())
            errs_here = np.where(op2_only_err_cpu[bi])[0]
            total_state1_op2_only_errors += len(errs_here)
            for pos in errs_here:
                g = int(gt_op2_cpu[bi, pos])
                pr = int(pr_op2_cpu[bi, pos])
                err_op2[g] += 1
                conf_pairs[(g, pr)] += 1
                for lo, hi in POS_BUCKETS:
                    if lo <= pos < hi:
                        err_pos_bins[f"{lo}-{hi}"] += 1
                        break
            ok_here = np.where(tok_ok_cpu[bi])[0]
            for pos in ok_here:
                g = int(gt_op2_cpu[bi, pos])
                correct_op2[g] += 1

    out = _concentration(err_op2, total_state1_op2_only_errors, correct_op2)
    out.update({
        "n_sampled_windows": int(len(idx)),
        "n_state_1_tokens": int(total_state1_tokens),
        "n_state_1_total_errors": int(total_state1_errors),
        "n_state_1_op2_only_errors": int(total_state1_op2_only_errors),
        "n_distinct_operand2_values_in_corrects": len(correct_op2),
        "top_20_confusion_pairs": _top20_confusion(conf_pairs),
        "top_10_gt_operand2_in_errors": _top10_gt(err_op2, correct_op2),
        "operand2_error_position_bucket": dict(err_pos_bins),
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--free-cap", type=int, default=FREE_RUN_CAP)
    args = ap.parse_args()
    device = torch.device(args.device)
    print(f"BL-7 v1 op2 concentration on {device}")

    vs, triples, _ = load_token_mapping()
    triple_arr = np.zeros((vs + 2, 3), dtype=np.int64)
    triple_arr[:vs] = np.array(triples, dtype=np.int64)
    triple_arr[vs:] = -1
    triple_t = torch.from_numpy(triple_arr).to(device)

    # Load BL-7 v1 decoder + val ds
    cdir = BASE_DIR / "models" / CKPT_DIR
    emb_dir = BASE_DIR / EMB_DIRNAME
    ckpt = torch.load(cdir / "ar_modern_best.pt", map_location="cpu", weights_only=False)
    mcfg = ModelConfig(**ckpt["model_config"])
    ds = TraceDataset(VAL_SESSIONS, mcfg.max_seq_len, mcfg.vocab_size,
                      emb_dir=emb_dir, data_dir=DATA_DIR)
    m = ARModernModel(mcfg).to(device)
    sd = ckpt["model"]
    sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
          for k, v in sd.items()}
    m.load_state_dict(sd)
    m.eval()
    print(f"  loaded BL-7 v1 decoder best.pt step={ckpt.get('step', '?')}")

    print("=== TF pass ===")
    tf = collect_tf(m, ds, mcfg, triple_t, vs, device,
                    batch_size=args.batch_size)
    print(f"  state-1 tokens: {tf['n_state_1_tokens']}, "
          f"any-errors: {tf['n_state_1_total_errors']}, "
          f"op2-only-errors: {tf['n_state_1_op2_only_errors']}")
    print(f"  top-1: {tf['top_1_operand2_error_share']:.4f}  "
          f"top-5: {tf['top_5_operand2_error_share']:.4f}  "
          f"top-10: {tf['top_10_operand2_error_share']:.4f}  "
          f"n-for-80%: {tf['n_values_to_cover_80pct_errors']}  "
          f"H={tf['error_distribution_entropy_nats']:.4f} nats")
    print(f"  position buckets of op2-errors: {tf['operand2_error_position_bucket']}")

    print("=== Free-running pass ===")
    fr = collect_fr(m, ds, mcfg, triple_t, vs, device,
                    free_cap=args.free_cap, batch_size=args.batch_size)
    print(f"  windows: {fr['n_sampled_windows']}, state-1 tokens: "
          f"{fr['n_state_1_tokens']}, any-errors: "
          f"{fr['n_state_1_total_errors']}, op2-only-errors: "
          f"{fr['n_state_1_op2_only_errors']}")
    print(f"  top-1: {fr['top_1_operand2_error_share']:.4f}  "
          f"top-5: {fr['top_5_operand2_error_share']:.4f}  "
          f"top-10: {fr['top_10_operand2_error_share']:.4f}  "
          f"n-for-80%: {fr['n_values_to_cover_80pct_errors']}  "
          f"H={fr['error_distribution_entropy_nats']:.4f} nats")
    print(f"  position buckets of op2-errors: {fr['operand2_error_position_bucket']}")

    # overlap of top-10 gt op2 between TF and FR
    tf_top10 = {d["gt_operand2"] for d in tf["top_10_gt_operand2_in_errors"]}
    fr_top10 = {d["gt_operand2"] for d in fr["top_10_gt_operand2_in_errors"]}
    overlap = len(tf_top10 & fr_top10)

    # routing flag (TF-based, per the spec)
    top5 = tf["top_5_operand2_error_share"]
    n80 = tf["n_values_to_cover_80pct_errors"]
    if top5 >= 0.60 or n80 < 15:
        route = "CONCENTRATED"
    elif top5 < 0.30 or n80 > 50:
        route = "DIFFUSE"
    else:
        route = "MIXED"

    # also report position-distribution flag (stop-and-escalate #5)
    pos_buckets = tf["operand2_error_position_bucket"]
    if pos_buckets:
        share_75plus = pos_buckets.get("75-91", 0) / max(sum(pos_buckets.values()), 1)
        # NOTE: amplification predicts most errors in later positions under
        # free-running, but under TF the errors could be uniform — that's
        # actually expected.  We report this number; flag only if TF errors
        # are *suspiciously* uniform AND late positions are saturated.
    else:
        share_75plus = None

    out = {
        "fold": 0,
        "checkpoint": "ar_modern_bl7_fold_0/ar_modern_best.pt",
        "teacher_forced": tf,
        "free_running_greedy": fr,
        "tf_vs_free_running": {
            "operand2_value_overlap_top10": overlap,
            "comment": (f"{overlap}/10 of TF's top error-prone gt operand-2 "
                        f"values also appear in FR's top-10. "
                        f"Overlap ≥ 7 indicates amplification preserves the "
                        f"error concentration.")
        },
        "routing": {
            "tf_branch": route,
            "tf_top5_share": top5,
            "tf_n80": n80,
        },
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "probe_w3_2_operand2_concentration.json").write_text(json.dumps(out, indent=2))
    print(f"\nROUTING (TF-based): {route}")
    print(f"  top-5 share = {top5:.4f}, n-for-80% = {n80}")
    print(f"  top-10 overlap TF↔FR: {overlap}/10")


if __name__ == "__main__":
    main()
