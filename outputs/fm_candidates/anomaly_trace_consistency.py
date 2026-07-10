"""Fault detection via trace consistency, using the paper's own generator.

Detector: the consistency score of a tick is the normalized edit distance
between the trace the generator reconstructs from sensors and the trace
observed on the machine. On a healthy system the two agree (low NED); a
fault that changes the executed path drives them apart. This replaces the
preliminary RF branch-count detector with the actual system, end to end.

Stage 1 (this script): code-path substitution, offline from the saved eval
npz chunks (references = observed traces, greedy_generated = reconstructed
traces). The injected fault replaces the observed trace of a tick with an
observed trace drawn from a different controller state, which models the
controller taking the wrong path while the plant looks nominal.

Later stages (separate runs): perturbed-sensor variants (require
re-encoding) and SIL buggy-controller streams (workstream C).

Output: results/anomaly_trace_consistency/results.json
        {"auc": {"trace_swap": ...}, ...}
"""
import argparse
import json
from pathlib import Path

import numpy as np

from metrics import normalized_edit_distance

RES = Path(__file__).parent / "results"


def load_chunks(variant: str, fold: int):
    d = RES / f"{variant}_strat100k_fold_{fold}"
    refs, gens, states = [], [], []
    for chunk in sorted(d.glob(f"{variant}_eval_test_chunk_*.npz")):
        z = np.load(chunk, allow_pickle=True)
        refs.extend(list(z["references"]))
        gens.extend(list(z["greedy_generated"]))
        states.extend(list(z["fsm_states"]))
    return refs, gens, np.asarray(states)


def ned_scores(gens, refs, idx):
    return np.array([
        normalized_edit_distance(list(gens[i]), list(refs[i])) for i in idx
    ])


def auc_rank(normal: np.ndarray, anomalous: np.ndarray) -> float:
    """Mann-Whitney AUC: P(anomalous score > normal score)."""
    from scipy.stats import mannwhitneyu
    u, _ = mannwhitneyu(anomalous, normal, alternative="greater")
    return float(u / (len(normal) * len(anomalous)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="ar_modern_bl2_w4000_thresh")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-normal", type=int, default=5000)
    ap.add_argument("--n-anomalies", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    refs, gens, states = load_chunks(args.variant, args.fold)
    n = len(refs)
    print(f"loaded {n} ticks from {args.variant} fold {args.fold}")
    assert n > 0, "no npz chunks found"

    # Normal consistency distribution.
    normal_idx = rng.choice(n, size=min(args.n_normal, n), replace=False)
    normal = ned_scores(gens, refs, normal_idx)
    print(f"normal NED: mean {normal.mean():.4f}  p95 {np.quantile(normal, .95):.4f}  "
          f"p99 {np.quantile(normal, .99):.4f}")

    # Code-path substitution: observed trace swapped with one from a
    # different controller state.
    by_state = {s: np.flatnonzero(states == s) for s in np.unique(states)}
    cand = rng.choice(n, size=min(args.n_anomalies * 2, n), replace=False)
    swapped = []
    for i in cand:
        others = [s for s in by_state if s != states[i] and len(by_state[s])]
        if not others:
            continue
        j = rng.choice(by_state[rng.choice(others)])
        swapped.append(normalized_edit_distance(list(gens[i]), list(refs[j])))
        if len(swapped) >= args.n_anomalies:
            break
    swapped = np.asarray(swapped)
    print(f"trace-swap NED: mean {swapped.mean():.4f}  n={len(swapped)}")

    auc = auc_rank(normal, swapped)
    print(f"AUC (trace_swap): {auc:.6f}")

    out_dir = RES / "anomaly_trace_consistency"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "variant": args.variant,
        "fold": args.fold,
        "n_normal": int(len(normal)),
        "n_anomalies": int(len(swapped)),
        "normal_ned": {"mean": float(normal.mean()),
                       "p95": float(np.quantile(normal, .95)),
                       "p99": float(np.quantile(normal, .99))},
        "auc": {"trace_swap": auc},
    }
    (out_dir / "results.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
