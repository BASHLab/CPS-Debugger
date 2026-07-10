"""Wave-4 pre-flight: state-0 and state-2 operand-2 error decomposition.

Repeats Wave-3 Probe 2 (triple decomposition + concentration metrics)
on state-0 and state-2 windows, so we know whether the operand-2-only
story and the branch-site concentration generalise across FSM states.

Same machinery as `bl7_w3_operand2_concentration.py`, parameterised on
target state.

Usage:
    python3 bl7_w4_pre_state_decomp.py
Outputs:
    results/wave4_state_decomp.json  (one block per state)
"""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import DATA_DIR, BASE_DIR, ModelConfig, VAL_SESSIONS
from data import TraceDataset, load_token_mapping
from ar_modern_model import ARModernModel

RESULTS = BASE_DIR / "results"
CKPT_DIR_BL7 = "ar_modern_bl7_fold_0"
EMB_BL7 = "sensor_embeddings_bl7"
CKPT_DIR_BL2 = "ar_modern_fold_0"
EMB_BL2 = "sensor_embeddings"
FREE_RUN_CAP = 5000


def _amp(device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    class _N:
        def __enter__(self): return None
        def __exit__(self, *a): return False
    return _N()


def _entropy(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c/total) * math.log(c/total) for c in counts.values() if c > 0)


def _concentration_metrics(err_op2, total_err):
    items = sorted(err_op2.items(), key=lambda kv: -kv[1])
    if not items:
        return {"top_1": 0, "top_5": 0, "top_10": 0, "n80": 0, "H_nats": 0}
    s5 = sum(c for _, c in items[:5]) / total_err
    s10 = sum(c for _, c in items[:10]) / total_err
    running = 0
    n80 = 0
    for i, (_, c) in enumerate(items):
        running += c
        if running >= 0.8 * total_err:
            n80 = i + 1
            break
    return {
        "top_1": items[0][1] / total_err,
        "top_5": s5,
        "top_10": s10,
        "n_to_cover_80pct": n80,
        "H_nats": _entropy(err_op2),
    }


def collect_tf(m, ds, mcfg, triple_t, vs, device, target_state, batch_size=128):
    pad_id = mcfg.vocab_size
    bos_id = mcfg.vocab_size + 1
    err_op2 = Counter()
    conf = Counter()
    tot_tokens = 0
    tot_err = 0
    op2_only = 0
    POS = [(0,15),(15,30),(30,45),(45,60),(60,75),(75,91)]
    err_pos = Counter()

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    with torch.no_grad():
        for batch in dl:
            x0 = batch["trace"].to(device)
            emb = batch["sensor_emb"].to(device)
            fsm = batch["fsm_state"].numpy()
            pad = (x0 != pad_id)
            B, L = x0.shape
            bos = torch.full((B,1), bos_id, dtype=x0.dtype, device=device)
            with _amp(device):
                logits = m.backbone(torch.cat([bos, x0], dim=1), emb)[:, :L, :]
            pred = logits.argmax(-1)
            xs = x0.clamp(0, vs+1); ps = pred.clamp(0, vs+1)
            gt = triple_t[xs]; pr = triple_t[ps]
            op_eq = (gt[...,0] == pr[...,0])
            o1_eq = (gt[...,1] == pr[...,1])
            o2_eq = (gt[...,2] == pr[...,2])
            op2_only_m = op_eq & o1_eq & ~o2_eq & pad
            any_err = (~(op_eq & o1_eq & o2_eq)) & pad
            for arr in ('op2_only_cpu','any_err_cpu','gt_op2','pr_op2','pad_cpu'):
                pass
            op2_only_cpu = op2_only_m.cpu().numpy()
            any_err_cpu = any_err.cpu().numpy()
            gt_op2 = gt[...,2].cpu().numpy()
            pr_op2 = pr[...,2].cpu().numpy()
            pad_cpu = pad.cpu().numpy()
            for bi in range(B):
                if fsm[bi] != target_state:
                    continue
                tot_tokens += int(pad_cpu[bi].sum())
                tot_err += int(any_err_cpu[bi].sum())
                errs = np.where(op2_only_cpu[bi])[0]
                op2_only += len(errs)
                for pos in errs:
                    g = int(gt_op2[bi, pos]); p = int(pr_op2[bi, pos])
                    err_op2[g] += 1
                    conf[(g,p)] += 1
                    for lo, hi in POS:
                        if lo <= pos < hi:
                            err_pos[f"{lo}-{hi}"] += 1
                            break
    metrics = _concentration_metrics(err_op2, op2_only) if op2_only else {}
    return {
        "n_tokens": tot_tokens,
        "n_total_errors": tot_err,
        "n_op2_only_errors": op2_only,
        "frac_op2_only_of_errors": (op2_only / tot_err) if tot_err else None,
        "concentration": metrics,
        "top_20_confusion": [
            {"gt": int(g), "pred": int(p), "count": int(c)}
            for (g,p), c in sorted(conf.items(), key=lambda kv: -kv[1])[:20]
        ],
        "top_10_gt_op2": [
            {"op2": int(v), "count": int(c)}
            for v, c in sorted(err_op2.items(), key=lambda kv: -kv[1])[:10]
        ],
        "position_buckets": dict(err_pos),
    }


def _state_indices(ds, state, cap, seed=0):
    idx = np.array([i for i, s in enumerate(ds.fsm_states) if s == state])
    if cap and len(idx) > cap:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=cap, replace=False))
    return idx


def _gather(ds, indices, device):
    trace = torch.stack([ds[i]["trace"] for i in indices])
    emb = torch.stack([ds[i]["sensor_emb"] for i in indices])
    return trace.to(device), emb.to(device)


def collect_fr(m, ds, mcfg, triple_t, vs, device, target_state, cap=FREE_RUN_CAP, batch_size=128, seed=0):
    idx = _state_indices(ds, target_state, cap, seed=seed)
    if len(idx) == 0:
        return {"n_sampled_windows": 0, "skipped": True}
    pad_id = mcfg.vocab_size; bos_id = mcfg.vocab_size + 1
    err_op2 = Counter(); conf = Counter()
    tot_tokens = 0; tot_err = 0; op2_only = 0
    POS = [(0,15),(15,30),(30,45),(45,60),(60,75),(75,91)]
    err_pos = Counter()
    L = mcfg.max_seq_len
    for s in range(0, len(idx), batch_size):
        sub = idx[s:s+batch_size]
        x0, emb = _gather(ds, sub, device)
        pad = (x0 != pad_id)
        B = x0.shape[0]
        gen = torch.full((B,1), bos_id, dtype=x0.dtype, device=device)
        for p in range(L):
            with torch.no_grad(), _amp(device):
                logits = m.backbone(gen, emb)[:, -1, :]
            logits[:, pad_id] = -1e9
            logits[:, bos_id] = -1e9
            nxt = logits.argmax(-1, keepdim=True)
            gen = torch.cat([gen, nxt], dim=1)
        pred = gen[:, 1:]
        xs = x0.clamp(0, vs+1); ps = pred.clamp(0, vs+1)
        gt = triple_t[xs]; pr = triple_t[ps]
        op_eq = (gt[...,0] == pr[...,0])
        o1_eq = (gt[...,1] == pr[...,1])
        o2_eq = (gt[...,2] == pr[...,2])
        op2_only_m = op_eq & o1_eq & ~o2_eq & pad
        any_err = (~(op_eq & o1_eq & o2_eq)) & pad
        op2_only_cpu = op2_only_m.cpu().numpy()
        any_err_cpu = any_err.cpu().numpy()
        gt_op2 = gt[...,2].cpu().numpy()
        pr_op2 = pr[...,2].cpu().numpy()
        pad_cpu = pad.cpu().numpy()
        for bi in range(B):
            tot_tokens += int(pad_cpu[bi].sum())
            tot_err += int(any_err_cpu[bi].sum())
            errs = np.where(op2_only_cpu[bi])[0]
            op2_only += len(errs)
            for pos in errs:
                g = int(gt_op2[bi, pos]); p = int(pr_op2[bi, pos])
                err_op2[g] += 1
                conf[(g,p)] += 1
                for lo, hi in POS:
                    if lo <= pos < hi:
                        err_pos[f"{lo}-{hi}"] += 1
                        break
    metrics = _concentration_metrics(err_op2, op2_only) if op2_only else {}
    return {
        "n_sampled_windows": int(len(idx)),
        "n_tokens": tot_tokens,
        "n_total_errors": tot_err,
        "n_op2_only_errors": op2_only,
        "frac_op2_only_of_errors": (op2_only / tot_err) if tot_err else None,
        "concentration": metrics,
        "top_20_confusion": [
            {"gt": int(g), "pred": int(p), "count": int(c)}
            for (g,p), c in sorted(conf.items(), key=lambda kv: -kv[1])[:20]
        ],
        "top_10_gt_op2": [
            {"op2": int(v), "count": int(c)}
            for v, c in sorted(err_op2.items(), key=lambda kv: -kv[1])[:10]
        ],
        "position_buckets": dict(err_pos),
    }


def run_variant(label, ckpt_subdir, emb_subdir, triple_t, vs, device):
    cdir = BASE_DIR / "models" / ckpt_subdir
    emb_dir = BASE_DIR / emb_subdir
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
    print(f"[{label}] loaded best.pt step={ckpt.get('step','?')}")
    out = {"checkpoint": f"{ckpt_subdir}/ar_modern_best.pt"}
    for st in (0, 1, 2):
        print(f"  [{label}] state={st} TF…")
        tf = collect_tf(m, ds, mcfg, triple_t, vs, device, target_state=st)
        print(f"    n_tokens={tf['n_tokens']} any_err={tf['n_total_errors']} "
              f"op2_only={tf['n_op2_only_errors']} "
              f"frac_op2_only={tf['frac_op2_only_of_errors']}")
        if tf['concentration']:
            print(f"    top-5 share={tf['concentration']['top_5']:.4f} "
                  f"n80={tf['concentration']['n_to_cover_80pct']}")
        print(f"  [{label}] state={st} FR…")
        fr = collect_fr(m, ds, mcfg, triple_t, vs, device, target_state=st)
        if fr.get("skipped"):
            print("    skipped (no windows)")
        else:
            print(f"    windows={fr['n_sampled_windows']} any_err={fr['n_total_errors']} "
                  f"op2_only={fr['n_op2_only_errors']}")
        out[f"state_{st}"] = {"teacher_forced": tf, "free_running": fr}
    del m
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    vs, triples, _ = load_token_mapping()
    triple_arr = np.zeros((vs+2, 3), dtype=np.int64)
    triple_arr[:vs] = np.array(triples, dtype=np.int64)
    triple_arr[vs:] = -1
    triple_t = torch.from_numpy(triple_arr).to(device)

    out = {"fold": 0}
    print("=== BL-7 v1 ===")
    out["bl7_v1"] = run_variant("bl7_v1", CKPT_DIR_BL7, EMB_BL7, triple_t, vs, device)
    print("=== BL-2 ===")
    out["bl2"] = run_variant("bl2", CKPT_DIR_BL2, EMB_BL2, triple_t, vs, device)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "wave4_state_decomp.json").write_text(json.dumps(out, indent=2))
    print("\nSaved results/wave4_state_decomp.json")

    # Quick summary
    print("\n=== Summary (state × variant × {TF, FR}, n_op2_only / n_total_errors) ===")
    for st in (0, 1, 2):
        print(f"  state {st}:")
        for v in ("bl7_v1", "bl2"):
            tf = out[v][f"state_{st}"]["teacher_forced"]
            fr = out[v][f"state_{st}"]["free_running"]
            tf_frac = tf['frac_op2_only_of_errors']
            tf_top5 = tf['concentration'].get('top_5', 0) if tf['concentration'] else 0
            tf_n80 = tf['concentration'].get('n_to_cover_80pct', 0) if tf['concentration'] else 0
            print(f"    {v}: TF op2-only-frac={tf_frac}  top5={tf_top5:.3f}  n80={tf_n80}")


if __name__ == "__main__":
    main()
