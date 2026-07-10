"""A6 — hard-tick analysis: where does the model beat the template floor?

The corpus is ~98% self-similar, so overall NED understates what matters.
This analysis stratifies the stratified-100K evaluation by tick difficulty:

  S1 transition ticks: within +/-K ticks of an FSM state change,
  S2 path-change ticks: the executed trace differs from the previous
     tick's trace (same session, outside S1) -- within-state branch
     decisions, the diagnostically load-bearing ticks,
  S3 steady ticks: everything else.

For the featured model and the per-state modal-trace floor it reports NED
and exact-match per stratum. The claim this feeds: the model's advantage
concentrates exactly on the ticks where debugging happens.

Reproduces evaluate.py's stratified index selection bit-for-bit and
asserts agreement with the saved npz fsm_states before trusting the join.

Output: results/hard_tick_analysis/results.json
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from config import ModelConfig, DATA_DIR, EMB_DIR
from data import TraceDataset, load_token_mapping
from evaluate import EVAL_SUBSET_SEED, _load_train_modal_traces
from metrics import normalized_edit_distance

RES = Path(__file__).parent / "results"

TEST_SESSIONS = ["2025-03-25_13-39-06", "2025-03-26_11-03-04"]
FOLD_TRAIN_0 = ["2025-03-17_10-36-44", "2025-03-20_09-31-56"]


def stratified_indices(ds, target: int) -> np.ndarray:
    """Bit-for-bit copy of evaluate.py's stratified selection."""
    fsm_arr = np.array(ds.fsm_states, dtype=np.int64)
    idx_0 = np.where(fsm_arr == 0)[0]
    idx_2 = np.where(fsm_arr == 2)[0]
    idx_1_pool = np.where(fsm_arr == 1)[0]
    n_state1 = max(0, target - len(idx_0) - len(idx_2))
    rng = np.random.default_rng(EVAL_SUBSET_SEED)
    idx_1 = rng.permutation(idx_1_pool)[:n_state1]
    return np.sort(np.concatenate([idx_0, idx_1, idx_2]))


def load_npz(variant: str, fold: int):
    d = RES / f"{variant}_strat100k_fold_{fold}"
    refs, gens, states = [], [], []
    for chunk in sorted(d.glob(f"{variant}_eval_test_chunk_*.npz")):
        z = np.load(chunk, allow_pickle=True)
        refs.extend(list(z["references"]))
        gens.extend(list(z["greedy_generated"]))
        states.extend(list(z["fsm_states"]))
    return refs, gens, np.asarray(states)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="ar_modern_bl2_w4000_thresh")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--emb-dir", default=str(EMB_DIR))
    ap.add_argument("--target", type=int, default=100000)
    ap.add_argument("--k-transition", type=int, default=10)
    args = ap.parse_args()

    vocab_size, _, _ = load_token_mapping()
    mcfg = ModelConfig(vocab_size=vocab_size, mask_token_id=vocab_size)
    ds = TraceDataset(TEST_SESSIONS, mcfg.max_seq_len, mcfg.vocab_size,
                      emb_dir=Path(args.emb_dir), data_dir=Path(DATA_DIR))
    n_full = len(ds)
    sub_idx = stratified_indices(ds, args.target)
    refs, gens, npz_states = load_npz(args.variant, args.fold)
    assert len(refs) == len(sub_idx), (len(refs), len(sub_idx))
    ds_states = np.array(ds.fsm_states, dtype=np.int64)[sub_idx]
    assert (ds_states == npz_states).all(), "stratified join mismatch"
    print(f"join verified: {len(sub_idx)} rows")

    # ── Strata over the FULL test sequence ─────────────────────────────
    sess = np.asarray(ds.session_ids)
    fsm = np.array(ds.fsm_states, dtype=np.int64)
    same_sess = np.zeros(n_full, dtype=bool)
    same_sess[1:] = sess[1:] == sess[:-1]

    trans_pt = np.zeros(n_full, dtype=bool)
    trans_pt[1:] = (fsm[1:] != fsm[:-1]) & same_sess[1:]
    # Dilate by +/-K.
    K = args.k_transition
    trans = np.zeros(n_full, dtype=bool)
    for i in np.flatnonzero(trans_pt):
        lo, hi = max(0, i - K), min(n_full, i + K + 1)
        trans[lo:hi] = True

    path_change = np.zeros(n_full, dtype=bool)
    for i in range(1, n_full):
        if same_sess[i] and not trans[i]:
            a, b = ds.traces[i], ds.traces[i - 1]
            path_change[i] = len(a) != len(b) or not np.array_equal(a, b)

    strata_full = np.full(n_full, 3, dtype=np.int64)   # S3 steady
    strata_full[path_change] = 2                        # S2 path change
    strata_full[trans] = 1                              # S1 transition
    strata = strata_full[sub_idx]

    # ── Floor: per-state modal trace from the fold's train sessions ────
    fl_args = SimpleNamespace(emb_dir=args.emb_dir, data_dir=str(DATA_DIR))
    modal = _load_train_modal_traces(fl_args, mcfg, FOLD_TRAIN_0)
    fallback = next(iter(modal.values()))

    out = {"variant": args.variant, "fold": args.fold,
           "k_transition": K, "strata": {}}
    names = {1: "transition", 2: "path_change", 3: "steady"}
    for s, name in names.items():
        rows = np.flatnonzero(strata == s)
        if len(rows) == 0:
            continue
        m_ned, f_ned, m_em, f_em = [], [], [], []
        for i in rows:
            ref = list(refs[i])
            gen = list(gens[i])
            flo = list(modal.get(int(npz_states[i]), fallback))
            m = normalized_edit_distance(gen, ref)
            f = normalized_edit_distance(flo, ref)
            m_ned.append(m); f_ned.append(f)
            m_em.append(float(m == 0.0)); f_em.append(float(f == 0.0))
        out["strata"][name] = {
            "n": int(len(rows)),
            "share_of_eval": float(len(rows) / len(strata)),
            "model_ned_mean": float(np.mean(m_ned)),
            "floor_ned_mean": float(np.mean(f_ned)),
            "model_exact_match": float(np.mean(m_em)),
            "floor_exact_match": float(np.mean(f_em)),
        }
        r = out["strata"][name]
        print(f"{name:12s} n={r['n']:6d} ({r['share_of_eval']*100:5.1f}%)  "
              f"model NED {r['model_ned_mean']:.4f}  floor NED "
              f"{r['floor_ned_mean']:.4f}  model EM {r['model_exact_match']*100:5.1f}%  "
              f"floor EM {r['floor_exact_match']*100:5.1f}%")

    out_dir = RES / "hard_tick_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
