"""Learn the alignment's edit costs from fault-free operation.

Our alignment uses unit costs: every substitution, insertion and deletion costs
one. That encodes the assumption that the reconstruction is equally likely to
make any mistake, which is false and consequential. The alignment decides which
positions are substitutions and which are indels, and only substitutions are
eligible for attribution, so a mis-set cost silently changes what the localizer
can even consider. A habitually skipped block scored under unit costs can be
resolved as a run of substitutions rather than one deletion, manufacturing
exactly the spurious evidence the downstream filter then has to discard.

The variant-calling literature solves this by deriving penalties as negative
log-likelihoods of the observed error profile, so that the minimum-cost
alignment is the most likely explanation rather than the shortest edit script.
Nanopore basecallers do this for homopolymers, where a 0.8% overall indel rate
hides a 41.8% rate inside long repeats; setting one global cost there costs
about ten points of recall.

Here the same idea is: make the reconstruction's habitual mistakes cheap and
everything else expensive, with the costs counted on a session the fault study
never touches.

    sub(a -> b) = -log P(recon emits a | machine executed b)
    del(a)      = -log P(recon emits a | machine executed nothing there)
    ins(b)      = -log P(recon omits b | machine executed b)

Unseen pairs fall back to a floor set by the rarest observed event, so a
genuinely novel substitution stays expensive and remains attributable.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "../fm_candidates")
from waxi_host import WaxiHost
from replay import encode_input, load_session, CODE, DATA, VOCAB
import edge_localize_recon as ELR
from recon_align import align

HERE = Path(__file__).parent
MODULE = f"{CODE}/module.wasm"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="2025-03-26_11-03-04")
    ap.add_argument("--recon", default="../fm_candidates/results/recon_full/"
                                       "recon_2025-03-26_11-03-04_0_60568.npy")
    ap.add_argument("--ticks", type=int, default=16000)
    ap.add_argument("--out", default="results/edit_costs.json")
    args = ap.parse_args()
    assert args.session != "2025-03-25_13-39-06", "must not calibrate on the test session"

    rec = np.load(args.recon)[:args.ticks]
    angle, x, _ = load_session(f"{DATA}/{args.session}.parquet", 0, args.ticks)
    h = WaxiHost(MODULE, VOCAB, quiet=True); h.service_init()

    sub = Counter(); dele = Counter(); ins = Counter(); match = Counter()
    for t in range(min(len(angle), len(rec))):
        h.write_input(encode_input(angle[t], x[t]))
        obs = list(ELR.project(h.tick(decode=True)))
        ref = [int(v) for v in rec[t]][:len(obs)]
        for op, i, j in align(ref, obs):
            if op == "match":
                match[obs[j]] += 1
            elif op == "sub":
                sub[(ref[i], obs[j])] += 1
            elif op == "del":
                dele[ref[i]] += 1
            elif op == "ins":
                ins[obs[j]] += 1
        if (t + 1) % 4000 == 0:
            print(f"  {t+1} ticks", flush=True)

    total = sum(match.values()) + sum(sub.values()) + sum(dele.values()) + sum(ins.values())
    print(f"\n{total:,} aligned operations over {args.ticks} ticks")
    for nm, c in (("match", match), ("sub", sub), ("del", dele), ("ins", ins)):
        print(f"  {nm:<6} {sum(c.values()):>9,}  ({100*sum(c.values())/total:5.2f}%)"
              f"  {len(c)} distinct")

    # Costs as negative log frequencies. The floor for unseen events is one
    # count rarer than the rarest observed, so a novel substitution is always
    # more expensive than any habitual one and cannot be absorbed as noise.
    def cost_map(c, denom):
        return {f"{k}": float(-np.log(v / denom)) for k, v in c.items()}

    rarest = min([v for v in list(sub.values()) + list(dele.values())
                  + list(ins.values()) if v > 0] or [1])
    floor = float(-np.log((rarest * 0.5) / total))
    out = dict(session=args.session, ticks=int(args.ticks), n_ops=int(total),
               floor=floor,
               sub={f"{a}|{b}": float(-np.log(v / total)) for (a, b), v in sub.items()},
               dele=cost_map(dele, total), ins=cost_map(ins, total))
    Path(HERE, args.out).write_text(json.dumps(out, indent=1))
    print(f"\nunseen-event floor cost {floor:.2f}")
    cs = [v for v in out["sub"].values()]
    print(f"habitual substitution costs: min {min(cs):.2f} median "
          f"{np.median(cs):.2f} max {max(cs):.2f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
