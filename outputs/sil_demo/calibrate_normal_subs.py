"""Calibrate the reconstruction's normal substitution profile.

Runs on fault-free operation from a session the fault study never touches. For
each tick it aligns the reconstruction to the executed trace and records every
substitution the alignment reports. Those are the mistakes the reconstruction
makes when nothing is wrong, so at test time they must not be attributed to a
fault. This is the whole content of the separator, and it uses only artifacts a
commissioning run produces.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "../fm_candidates")
from waxi_host import WaxiHost
from replay import encode_input, load_session, CODE, DATA, VOCAB
import edge_localize_recon as ELR
from recon_align import NormalProfile, substitutions

HERE = Path(__file__).parent
MODULE = f"{CODE}/module.wasm"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="2025-03-26_11-03-04")
    ap.add_argument("--recon", default="../fm_candidates/results/recon_full/"
                                       "recon_2025-03-26_11-03-04_0_60568.npy")
    ap.add_argument("--ticks", type=int, default=16000)
    ap.add_argument("--out", default="results/normal_subs.json")
    args = ap.parse_args()
    assert args.session != "2025-03-25_13-39-06", \
        "calibration session must differ from the fault-study session"

    rec = np.load(args.recon)[:args.ticks]
    angle, x, _ = load_session(f"{DATA}/{args.session}.parquet", 0, args.ticks)
    state = pd.read_parquet(f"{DATA}/{args.session}.parquet",
                            columns=["pendulum_state"])["pendulum_state"] \
        .to_numpy()[:args.ticks].astype(int)
    h = WaxiHost(MODULE, VOCAB, quiet=True); h.service_init()

    prof = NormalProfile()
    n_sub = n_tick = 0
    for t in range(min(len(angle), len(rec))):
        h.write_input(encode_input(angle[t], x[t]))
        obs = list(ELR.project(h.tick(decode=True)))
        ref = [int(v) for v in rec[t]][:len(obs)]
        for a, b, _ in substitutions(ref, obs):
            prof.observe(int(state[t]), a, b)
            n_sub += 1
        n_tick += 1
        if n_tick % 4000 == 0:
            print(f"  {n_tick} ticks, {len(prof.counts)} distinct subs", flush=True)

    Path(HERE, args.out).write_text(json.dumps(
        dict(session=args.session, ticks=n_tick, n_substitutions=n_sub,
             distinct=len(prof.counts), counts=prof.to_json()), indent=1))
    print(f"\n{n_tick} ticks, {n_sub} substitutions, "
          f"{len(prof.counts)} distinct (state, ref, obs) triples")
    print(f"mean substitutions per tick: {n_sub/max(1,n_tick):.2f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
