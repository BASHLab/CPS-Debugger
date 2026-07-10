"""BL-7 within-state-1 gap — Wave 2 follow-up probes.

Three probes that test the AR-amplification reframing from Wave 1 (D4):

  Probe 1: Opcode vs operand TF-acc breakdown (HIGHEST PRIORITY).
           Decompose each predicted/target token into its
           (wasm_function_id, operand_1, operand_2) triple and compute
           a multiplicative TF-acc decomposition per FSM state.
  Probe 2: Position-wise TF-acc degradation under teacher forcing AND
           free-running (greedy) generation.
  Probe 3: Next-token entropy comparison between BL-7 v1 and BL-2 on
           matched contexts.

All probes run against the deployed `ar_modern_best.pt` per variant.
Encoder is frozen (we read precomputed `*_emb.npy` files).

Usage:
    python3 bl7_followup_probes.py {probe1,probe2,probe3,all}
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import DATA_DIR, BASE_DIR, ModelConfig, VAL_SESSIONS
from data import TraceDataset, load_token_mapping
from ar_modern_model import ARModernModel

RESULTS = BASE_DIR / "results"

VARIANTS = [
    # (label, ckpt dir, emb dir, json output label)
    ("bl2",    "ar_modern_fold_0",     "sensor_embeddings",     "bl2"),
    ("bl7_v1", "ar_modern_bl7_fold_0", "sensor_embeddings_bl7", "bl7_v1"),
]
POS_BUCKETS = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 91)]


# ── shared loaders ───────────────────────────────────────────────────────
def _bucket(p: int) -> str:
    for lo, hi in POS_BUCKETS:
        if lo <= p < hi:
            return f"{lo}-{hi}"
    return f"{POS_BUCKETS[-1][0]}-{POS_BUCKETS[-1][1]}"


def _load_variant(label, ckpt_dir, emb_dirname, device):
    """Return (model, val_dataset, mcfg) for one variant."""
    cdir = BASE_DIR / "models" / ckpt_dir
    emb_dir = BASE_DIR / emb_dirname
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
    print(f"  [{label}] loaded best.pt step={ckpt.get('step', '?')} on {device}")
    return m, ds, mcfg


def _amp_ctx(device):
    """bf16 autocast on CUDA, no-op on CPU."""
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    class _N:
        def __enter__(self): return None
        def __exit__(self, *a): return False
    return _N()


# ── Probe 1 — opcode/operand TF-acc decomposition ────────────────────────
def probe1(args):
    print("=== Probe 1: opcode/operand TF-acc decomposition ===")
    from torch.utils.data import DataLoader
    vs, triples, _ = load_token_mapping()
    # vocab lookup tensors: id → opcode, op1, op2 (with sentinel for PAD/BOS at vs..vs+1)
    triple_arr = np.zeros((vs + 2, 3), dtype=np.int64)
    triple_arr[:vs] = np.array(triples, dtype=np.int64)
    triple_arr[vs:] = -1  # PAD/BOS triples are -1 so they can't accidentally match
    device = torch.device(args.device)

    out = {"fold": 0, "n_state_1_tokens": 0}
    cache = {}
    for label, ckpt_dir, emb_dirname, json_label in VARIANTS:
        m, ds, mcfg = _load_variant(label, ckpt_dir, emb_dirname, device)
        trip_t = torch.from_numpy(triple_arr).to(device)  # (V+2, 3)
        pad_id = mcfg.vocab_size
        bos_id = mcfg.vocab_size + 1

        # per-state accumulators
        acc = {s: {"n": 0, "op_ok": 0, "op_ok_o1_ok": 0, "op_ok_o1_ok_o2_ok": 0}
               for s in (0, 1, 2)}

        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        with torch.no_grad():
            for batch in dl:
                x0 = batch["trace"].to(device)            # (B, L)
                emb = batch["sensor_emb"].to(device)
                fsm = batch["fsm_state"].numpy()
                pad_mask = (x0 != pad_id)
                B, L = x0.shape
                bos = torch.full((B, 1), bos_id, dtype=x0.dtype, device=device)
                with _amp_ctx(device):
                    logits = m.backbone(torch.cat([bos, x0], dim=1), emb)[:, :L, :]
                pred = logits.argmax(dim=-1)  # (B, L)

                # triple decode (vectorised gather)
                pred_tri = trip_t[pred.clamp(0, vs + 1)]   # (B, L, 3)
                gt_tri = trip_t[x0.clamp(0, vs + 1)]       # (B, L, 3)
                op_ok = (pred_tri[..., 0] == gt_tri[..., 0]) & pad_mask
                o1_ok = (pred_tri[..., 1] == gt_tri[..., 1]) & pad_mask
                o2_ok = (pred_tri[..., 2] == gt_tri[..., 2]) & pad_mask

                op_ok = op_ok.cpu().numpy()
                o1_ok = o1_ok.cpu().numpy()
                o2_ok = o2_ok.cpu().numpy()
                pad_np = pad_mask.cpu().numpy()
                for bi in range(B):
                    s = int(fsm[bi])
                    if s not in acc:
                        continue
                    n_tok = pad_np[bi].sum()
                    acc[s]["n"] += int(n_tok)
                    op = op_ok[bi]
                    acc[s]["op_ok"] += int(op.sum())
                    op_o1 = op & o1_ok[bi]
                    acc[s]["op_ok_o1_ok"] += int(op_o1.sum())
                    acc[s]["op_ok_o1_ok_o2_ok"] += int((op_o1 & o2_ok[bi]).sum())

        def _to_metrics(d):
            n = max(d["n"], 1)
            op = d["op_ok"] / n
            o1_cond = d["op_ok_o1_ok"] / max(d["op_ok"], 1)
            o2_cond = d["op_ok_o1_ok_o2_ok"] / max(d["op_ok_o1_ok"], 1)
            return {
                "n_tokens": d["n"],
                "opcode_tfacc": op,
                "operand1_tfacc_given_opcode": o1_cond,
                "operand2_tfacc_given_opcode_and_op1": o2_cond,
                "joint_tfacc": d["op_ok_o1_ok_o2_ok"] / n,
            }

        per = {f"state_{s}": _to_metrics(acc[s]) for s in (0, 1, 2)}
        # aggregate "all_states"
        all_acc = {"n": sum(acc[s]["n"] for s in acc),
                   "op_ok": sum(acc[s]["op_ok"] for s in acc),
                   "op_ok_o1_ok": sum(acc[s]["op_ok_o1_ok"] for s in acc),
                   "op_ok_o1_ok_o2_ok": sum(acc[s]["op_ok_o1_ok_o2_ok"] for s in acc)}
        per["all_states"] = _to_metrics(all_acc)
        cache[json_label] = per
        out[json_label] = per
        print(f"  [{label}] state-1: opcode={per['state_1']['opcode_tfacc']:.4f} "
              f"op1|op={per['state_1']['operand1_tfacc_given_opcode']:.4f} "
              f"op2|op,op1={per['state_1']['operand2_tfacc_given_opcode_and_op1']:.4f} "
              f"joint={per['state_1']['joint_tfacc']:.4f}")
        del m
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # gaps on state-1
    if "bl2" in cache and "bl7_v1" in cache:
        b2 = cache["bl2"]["state_1"]; b7 = cache["bl7_v1"]["state_1"]
        out["n_state_1_tokens"] = b2["n_tokens"]
        out["gap_state_1"] = {
            "opcode_gap": b2["opcode_tfacc"] - b7["opcode_tfacc"],
            "operand1_gap": b2["operand1_tfacc_given_opcode"]
                          - b7["operand1_tfacc_given_opcode"],
            "operand2_gap": b2["operand2_tfacc_given_opcode_and_op1"]
                          - b7["operand2_tfacc_given_opcode_and_op1"],
            "joint_gap": b2["joint_tfacc"] - b7["joint_tfacc"],
        }
        g = out["gap_state_1"]
        print(f"\n  STATE-1 GAP (BL2 - BL7): opcode={g['opcode_gap']:+.4f} "
              f"op1={g['operand1_gap']:+.4f} op2={g['operand2_gap']:+.4f} "
              f"joint={g['joint_gap']:+.4f}")
        # routing
        op_g = abs(g["opcode_gap"])
        operand_g = abs(g["operand1_gap"]) + abs(g["operand2_gap"])
        if operand_g >= 3 * op_g:
            branch = "OPERAND_DOMINATES — factored decoder strongly motivated"
        elif op_g >= 3 * operand_g:
            branch = "OPCODE_DOMINATES — encoder objective rethink, not decoder"
        else:
            branch = "ROUGHLY_EQUAL — diffuse gap; exposure-bias remedies generally"
        out["routing_branch"] = branch
        print(f"  ROUTING: {branch}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "probe1_opcode_operand_tfacc.json").write_text(json.dumps(out, indent=2))
    return out


# ── Probe 2 — position-wise TF-acc (TF + free-running) ───────────────────
def _state1_indices(ds, cap=None, seed=0):
    """Indices into ds where fsm_state == 1 (optionally capped)."""
    idx = np.array([i for i, s in enumerate(ds.fsm_states) if s == 1])
    if cap is not None and len(idx) > cap:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=cap, replace=False))
    return idx


def _state_indices(ds, state, cap=None, seed=0):
    idx = np.array([i for i, s in enumerate(ds.fsm_states) if s == state])
    if cap is not None and len(idx) > cap:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=cap, replace=False))
    return idx


def _gather_batch(ds, indices, device):
    """Stack samples at given indices into a batch dict."""
    trace = torch.stack([ds[i]["trace"] for i in indices])
    emb = torch.stack([ds[i]["sensor_emb"] for i in indices])
    return trace.to(device), emb.to(device)


def _tf_per_position(m, ds, indices, mcfg, device, batch_size=128):
    """Returns per-position arrays: correct[pos], total[pos] over the indices."""
    L = mcfg.max_seq_len
    pad_id = mcfg.vocab_size
    bos_id = mcfg.vocab_size + 1
    corr = np.zeros(L, dtype=np.int64)
    tot = np.zeros(L, dtype=np.int64)
    for s in range(0, len(indices), batch_size):
        sub = indices[s:s + batch_size]
        x0, emb = _gather_batch(ds, sub, device)
        pad_mask = (x0 != pad_id)
        B = x0.shape[0]
        bos = torch.full((B, 1), bos_id, dtype=x0.dtype, device=device)
        with torch.no_grad(), _amp_ctx(device):
            logits = m.backbone(torch.cat([bos, x0], dim=1), emb)[:, :L, :]
        pred = logits.argmax(dim=-1)
        ok = ((pred == x0) & pad_mask).cpu().numpy()  # (B, L)
        pm = pad_mask.cpu().numpy()
        corr += ok.sum(axis=0)
        tot += pm.sum(axis=0)
    return corr, tot


def _freerun_per_position(m, ds, indices, mcfg, device, batch_size=128):
    """Greedy free-running decode. Returns per-position correct[pos]/total[pos]."""
    L = mcfg.max_seq_len
    pad_id = mcfg.vocab_size
    bos_id = mcfg.vocab_size + 1
    corr = np.zeros(L, dtype=np.int64)
    tot = np.zeros(L, dtype=np.int64)
    for s in range(0, len(indices), batch_size):
        sub = indices[s:s + batch_size]
        x0, emb = _gather_batch(ds, sub, device)
        pad_mask = (x0 != pad_id)
        B = x0.shape[0]
        gen = torch.full((B, 1), bos_id, dtype=x0.dtype, device=device)
        # step-by-step greedy
        for p in range(L):
            with torch.no_grad(), _amp_ctx(device):
                logits = m.backbone(gen, emb)[:, -1, :]
            # disable special ids
            logits[:, pad_id] = -1e9
            logits[:, bos_id] = -1e9
            nxt = logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            gen = torch.cat([gen, nxt], dim=1)
        pred = gen[:, 1:]  # (B, L)
        ok = ((pred == x0) & pad_mask).cpu().numpy()
        pm = pad_mask.cpu().numpy()
        corr += ok.sum(axis=0)
        tot += pm.sum(axis=0)
    return corr, tot


def _bucketize(corr, tot):
    out = {}
    for lo, hi in POS_BUCKETS:
        c = int(corr[lo:hi].sum())
        t = int(tot[lo:hi].sum())
        out[f"{lo}-{hi}"] = (c / t) if t else None
    return out


def _bucket_n(tot):
    return {f"{lo}-{hi}": int(tot[lo:hi].sum()) for lo, hi in POS_BUCKETS}


def probe2(args):
    print("=== Probe 2: position-wise TF-acc (TF + free-running) ===")
    device = torch.device(args.device)
    out = {"fold": 0, "decode_recipe": "greedy",
           "free_running_n_state_1_windows": args.free_cap}
    n_per_bucket = None
    for label, ckpt_dir, emb_dirname, json_label in VARIANTS:
        m, ds, mcfg = _load_variant(label, ckpt_dir, emb_dirname, device)
        idx_all = _state1_indices(ds)
        idx_free = _state1_indices(ds, cap=args.free_cap, seed=0)
        print(f"  [{label}] state-1: TF over {len(idx_all)} windows, "
              f"free-running over {len(idx_free)} windows")
        # teacher forced
        c_tf, t_tf = _tf_per_position(m, ds, idx_all, mcfg, device,
                                      batch_size=args.batch_size)
        out[f"{json_label}_teacher_forced"] = _bucketize(c_tf, t_tf)
        if n_per_bucket is None:
            n_per_bucket = _bucket_n(t_tf)
        # free running (greedy)
        c_fr, t_fr = _freerun_per_position(m, ds, idx_free, mcfg, device,
                                           batch_size=args.batch_size)
        out[f"{json_label}_free_running"] = _bucketize(c_fr, t_fr)
        # also keep raw per-position arrays for plotting
        out[f"{json_label}_tf_per_position"] = (c_tf / np.maximum(t_tf, 1)).tolist()
        out[f"{json_label}_fr_per_position"] = (c_fr / np.maximum(t_fr, 1)).tolist()
        print(f"  [{label}] TF: " +
              "  ".join(f"{k}={v:.3f}" for k, v in out[f"{json_label}_teacher_forced"].items() if v is not None))
        print(f"  [{label}] FR: " +
              "  ".join(f"{k}={v:.3f}" for k, v in out[f"{json_label}_free_running"].items() if v is not None))
        del m
        torch.cuda.empty_cache() if device.type == "cuda" else None

    out["n_per_bucket_state_1"] = n_per_bucket

    # routing diagnosis (text only; numbers stand on their own)
    def _mean(d):
        vs = [v for v in d.values() if v is not None]
        return sum(vs) / len(vs) if vs else 0.0
    if "bl7_v1_teacher_forced" in out and "bl2_teacher_forced" in out:
        b7_tf = _mean(out["bl7_v1_teacher_forced"])
        b2_tf = _mean(out["bl2_teacher_forced"])
        b7_fr = _mean(out["bl7_v1_free_running"])
        b2_fr = _mean(out["bl2_free_running"])
        tf_gap_75 = (out["bl2_teacher_forced"].get("75-91") or 0) \
                  - (out["bl7_v1_teacher_forced"].get("75-91") or 0)
        if tf_gap_75 > 0.01:
            branch = "TF_DEGRADES_DIFFERENTIALLY — BL-7 loses long-horizon information"
        elif (b2_fr - b7_fr) - (b2_tf - b7_tf) > 0.01:
            branch = "PURE_AMPLIFICATION — BL-7 amplifies more under free-running"
        else:
            branch = "EQUAL_DEGRADATION — sequence-modeling problem, not encoder"
        out["routing_branch"] = branch
        out["bl2_mean_tf"], out["bl7_v1_mean_tf"] = b2_tf, b7_tf
        out["bl2_mean_fr"], out["bl7_v1_mean_fr"] = b2_fr, b7_fr
        print(f"  ROUTING: {branch}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "probe2_position_tfacc.json").write_text(json.dumps(out, indent=2))
    return out


# ── Probe 3 — next-token entropy ─────────────────────────────────────────
def _entropy_per_position(m, ds, indices, mcfg, device, batch_size=64):
    """Per-position mean Shannon entropy (nats) over the indices.
    Returns (entropy_sum[pos], count[pos]) — caller divides."""
    L = mcfg.max_seq_len
    pad_id = mcfg.vocab_size
    bos_id = mcfg.vocab_size + 1
    ent_sum = np.zeros(L, dtype=np.float64)
    n = np.zeros(L, dtype=np.int64)
    for s in range(0, len(indices), batch_size):
        sub = indices[s:s + batch_size]
        x0, emb = _gather_batch(ds, sub, device)
        pad_mask = (x0 != pad_id)
        B = x0.shape[0]
        bos = torch.full((B, 1), bos_id, dtype=x0.dtype, device=device)
        with torch.no_grad(), _amp_ctx(device):
            logits = m.backbone(torch.cat([bos, x0], dim=1), emb)[:, :L, :].float()
        # entropy of softmax over 648 logits at each position (exclude PAD/BOS heads)
        logits[..., pad_id:pad_id + 2] = -1e9  # mask PAD, BOS in the distribution
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        ent = -(p * logp).sum(dim=-1)  # (B, L) in nats
        pm = pad_mask.float()
        ent_sum += (ent * pm).sum(dim=0).cpu().numpy()
        n += pm.sum(dim=0).long().cpu().numpy()
    return ent_sum, n


def probe3(args):
    print("=== Probe 3: next-token entropy ===")
    device = torch.device(args.device)
    out = {"fold": 0, "n_state_1_windows_sampled": args.entropy_cap}
    for s in (0, 1, 2):
        out[f"state_{s}"] = {}

    for label, ckpt_dir, emb_dirname, json_label in VARIANTS:
        m, ds, mcfg = _load_variant(label, ckpt_dir, emb_dirname, device)
        for s in (0, 1, 2):
            idx = _state_indices(ds, s, cap=args.entropy_cap, seed=0)
            if len(idx) == 0:
                out[f"state_{s}"][f"{json_label}_entropy_per_position_bucket"] = {}
                continue
            ent_sum, n = _entropy_per_position(m, ds, idx, mcfg, device,
                                               batch_size=args.entropy_bs)
            bucket = {}
            for lo, hi in POS_BUCKETS:
                en = ent_sum[lo:hi].sum()
                nn = max(int(n[lo:hi].sum()), 1)
                bucket[f"{lo}-{hi}"] = float(en / nn)
            mean_ent = float(ent_sum.sum() / max(n.sum(), 1))
            out[f"state_{s}"][f"{json_label}_entropy_per_position_bucket"] = bucket
            out[f"state_{s}"][f"{json_label}_mean_entropy"] = mean_ent
            out[f"state_{s}"][f"{json_label}_n_windows"] = int(len(idx))
            print(f"  [{label}] state-{s} mean_entropy={mean_ent:.4f} (n={len(idx)}) "
                  f"buckets: " + "  ".join(f"{k}={v:.3f}" for k, v in bucket.items()))
        del m
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # state-1 gap & routing
    s1 = out["state_1"]
    if "bl2_mean_entropy" in s1 and "bl7_v1_mean_entropy" in s1:
        gap = s1["bl7_v1_mean_entropy"] - s1["bl2_mean_entropy"]
        s1["mean_entropy_gap"] = gap
        if gap >= 0.3:
            branch = "MECHANISM_1 — BL-7 distributions noticeably less peaky (≥ 0.3 nats); FSQ direction"
        elif gap < 0.1:
            branch = "MECHANISM_2 — entropies similar (< 0.1 nats); temporal-clocking direction"
        else:
            branch = "MIXED — both mechanisms contribute (gap 0.1–0.3 nats)"
        out["routing_branch"] = branch
        print(f"  STATE-1 entropy gap (BL-7 − BL-2): {gap:+.4f} nats  → {branch}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "probe3_token_entropy.json").write_text(json.dumps(out, indent=2))
    return out


# ── orchestration ────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe1", "probe2", "probe3", "all"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=128,
                    help="batch size for forward passes (TF mode in probe1/2)")
    ap.add_argument("--free-cap", type=int, default=5000,
                    help="probe2: max state-1 windows for free-running greedy decode")
    ap.add_argument("--entropy-cap", type=int, default=5000,
                    help="probe3: per-state cap on windows for entropy computation")
    ap.add_argument("--entropy-bs", type=int, default=64,
                    help="probe3: batch size (softmax fp32 is memory-heavy)")
    args = ap.parse_args()
    if args.cmd in ("probe1", "all"):
        probe1(args)
    if args.cmd in ("probe2", "all"):
        probe2(args)
    if args.cmd in ("probe3", "all"):
        probe3(args)


if __name__ == "__main__":
    main()
