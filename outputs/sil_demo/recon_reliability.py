"""Offline reliability map for the reconstruction, for deployable localization.

The composition localizer decides whether a divergence is the fault or a
reconstruction error. It has been doing so by checking, at each position,
whether the reconstruction agrees with a wasmtime replay of the SHIPPED binary
on the same tick. That replay is the artifact the reconstruction exists to
replace: a site that has it can localize by exact replay and needs no
reconstruction at all. Using it at scoring time makes the deployable number
depend on the golden trace.

What is genuinely available is calibration data from commissioning: normal
operation on the twin, where both the executed trace and the reconstruction can
be observed. From that we can learn WHERE the reconstruction is trustworthy,
freeze it, and carry it to a machine where no golden trace exists.

The map is keyed on what a deployed machine knows at scoring time: the
controller state (a logged sensor channel) and the position within the tick.
It is calibrated on a DIFFERENT held-out session from the one the fault study
replays, so nothing about the test session leaks into it.

Writes results/recon_reliability.json.
"""
import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from waxi_host import WaxiHost
from replay import encode_input, load_session, CODE, DATA, VOCAB
import edge_localize_recon as ELR

HERE = Path(__file__).parent
OUT = HERE / "results"
MODULE = f"{CODE}/module.wasm"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="2025-03-26_11-03-04",
                    help="calibration session; must NOT be the fault-study one")
    ap.add_argument("--recon", default="../fm_candidates/results/recon_full/"
                                       "recon_2025-03-26_11-03-04_0_60568.npy")
    ap.add_argument("--ticks", type=int, default=16000)
    ap.add_argument("--out", default="results/recon_reliability.json")
    args = ap.parse_args()
    assert args.session != "2025-03-25_13-39-06", \
        "calibration session must differ from the fault-study session"

    angle, x, _ = load_session(f"{DATA}/{args.session}.parquet", 0, args.ticks)
    # DATA is train_data_full, whose `trace` column is the RAW edge stream
    # (median 1889 tokens). The reconstruction lives in the pruned 648-edge
    # vocabulary (median 98). Comparing the two directly scores 2.7% and is
    # meaningless; the executed side must go through the same projection the
    # localizer uses. The state channel is fine to read from either.
    state = pd.read_parquet(f"{DATA}/{args.session}.parquet",
                            columns=["pendulum_state"])["pendulum_state"] \
        .to_numpy()[:args.ticks].astype(int)
    recon = np.load(args.recon)[:args.ticks]
    print(f"calibrating on {args.session}: {len(angle)} ticks, "
          f"recon {recon.shape}", flush=True)

    h = WaxiHost(MODULE, VOCAB, quiet=True); h.service_init()
    ok = defaultdict(int); tot = defaultdict(int)
    for t in range(min(len(angle), len(recon))):
        h.write_input(encode_input(angle[t], x[t]))
        obs = ELR.project(h.tick(decode=True))     # executed trace, normal run
        rec = [int(v) for v in recon[t]]
        s = int(state[t])
        for i in range(len(obs)):
            r = rec[i] if i < len(rec) else None
            tot[(s, i)] += 1
            ok[(s, i)] += int(r == obs[i])
        if (t + 1) % 4000 == 0:
            print(f"  {t+1} ticks", flush=True)

    # Serialize with string keys; record support so the consumer can require a
    # minimum before trusting a cell.
    m = {f"{s}|{i}": [ok[(s, i)], tot[(s, i)]] for (s, i) in tot}
    Path(HERE, args.out).write_text(json.dumps(
        dict(session=args.session, ticks=int(min(len(angle), len(recon))),
             key="state|position", map=m), indent=1))
    rel = np.array([ok[k] / tot[k] for k in tot])
    print(f"\n{len(m)} (state, position) cells")
    print(f"reliability: median {np.median(rel):.3f}, "
          f"frac cells >=0.99: {(rel >= 0.99).mean():.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
