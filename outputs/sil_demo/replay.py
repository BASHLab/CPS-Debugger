"""replay.py -- open-loop trace-differential analysis on real sensor data.

The pivot away from closed-loop SIL (see c2_findings.md): instead of
simulating the plant, we REPLAY a real session's logged sensors through the
shipped controller binary and through each mutated binary, and compare the
reconstructed control-flow traces. No plant model, no sim-to-real gap.

  * Detection: a seeded software fault takes different branches, so the
    mutant's per-tick trace diverges from the nominal's on the same inputs.
  * Localization: the functions/edges where they diverge name the fault site.

Both binaries start from the same fresh state and receive the identical real
sensor sequence, so any trace difference is attributable to the mutation, not
to initial-state drift. (Matching the ORIGINAL captured trace is a separate,
stricter fidelity check that additionally requires replaying a whole session
import os as _os
from pathlib import Path as _Path

# Roots. Override with environment variables; defaults assume this checkout and
# the dataset release unpacked as CPSD_DATA (see outputs/sil_demo/README.md).
REPO = _Path(_os.environ.get("CPSD_REPO", _Path(__file__).resolve().parents[2]))
CPSD_DATA = _Path(_os.environ.get("CPSD_DATA", REPO / "data"))
CPSD_CAPTURE = _Path(_os.environ.get("CPSD_CAPTURE", CPSD_DATA / "captures"))

from its swing-up onset so the controller's counters track the real run.)

CPU-only (wasmtime + pyarrow); no torch.
"""
import argparse
import struct
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from waxi_host import WaxiHost
from plant import (COUNTS_PER_RAD, COUNTS_PER_REV, ENC_MASK, POS_COUNTS_PER_M,
                   STATUS_WORD, OFF_ENCODER, OFF_STATUS, OFF_POSIW)

DATA = str(CPSD_DATA)
CODE = str(CPSD_CAPTURE / "2025-03-17_10-36-44" / "code")
VOCAB = str(CPSD_DATA / "auto_token_mapping.json")


def encode_input(angle, x):
    """Real logged (theta, x) -> 64-byte EtherCAT input image (exact inverse
    of the controller's decode; matches plant.py synthesis with +1 signs)."""
    buf = bytearray(64)
    masked = (round(angle * COUNTS_PER_RAD) + COUNTS_PER_REV // 2) & ENC_MASK
    struct.pack_into("<H", buf, OFF_ENCODER, masked)
    struct.pack_into("<H", buf, OFF_STATUS, STATUS_WORD)
    struct.pack_into("<i", buf, OFF_POSIW, int(round(x * POS_COUNTS_PER_M)))
    return bytes(buf)


NFUNC = 256  # controller has ~113 functions; func_ids observed well under 256


def replay(module_path, angle, x):
    """Replay the sensor sequence through a binary. Returns a per-tick
    function-count matrix of shape (n_ticks, NFUNC): entry [t, f] is how many
    control-flow edges in function f executed on tick t. The decode is
    vectorized with numpy (frombuffer + bit shift), so it is fast enough to
    replay a whole session per mutant."""
    host = WaxiHost(module_path, VOCAB, quiet=True)
    host.service_init()
    mat = np.zeros((len(angle), NFUNC), dtype=np.int32)
    for i, (a, xx) in enumerate(zip(angle, x)):
        host.write_input(encode_input(a, xx))
        raw = host.tick_raw()
        n8 = (len(raw) // 8) * 8
        if not n8:
            continue
        recs = np.frombuffer(raw[:n8], dtype="<i8")
        fids = ((recs >> 48) & 0xFFFF).astype(np.int64)
        fids = fids[fids < NFUNC]
        if fids.size:
            mat[i] = np.bincount(fids, minlength=NFUNC)[:NFUNC]
    return mat


def compare(nom, bug):
    """Compare two per-tick function-count matrices. Detection: fraction of
    ticks whose function counts differ. Localization: functions carrying the
    divergence, ranked by total count difference over the session."""
    diff = np.abs(nom.astype(np.int64) - bug.astype(np.int64))  # (T, NFUNC)
    per_tick = diff.sum(axis=1)
    divergent = per_tick > 0
    first = int(np.argmax(divergent)) if divergent.any() else -1
    func_delta = diff.sum(axis=0)  # (NFUNC,)
    order = np.argsort(func_delta)[::-1]
    top = [(int(f), int(func_delta[f])) for f in order if func_delta[f] > 0][:8]
    # Localize by the CAUSAL ORIGIN: the functions that first differ, at the
    # earliest divergent tick. Max-divergence functions reflect downstream
    # consequences (a rerouted trajectory diverges everywhere); the fault
    # branch itself is where divergence begins.
    origin = []
    if first >= 0:
        row = diff[first]
        for f in np.argsort(row)[::-1]:
            if row[f] > 0:
                origin.append((int(f), int(row[f])))
        origin = origin[:8]
    return dict(
        mean_delta=float(per_tick.mean()),
        frac_divergent=float(divergent.mean()),
        first_divergent_tick=first,
        top_divergent_funcs=top,
        origin_funcs=origin,
    )


def load_session(fpath, start, n):
    t = pq.read_table(fpath, columns=["current_angle", "current_x",
                                      "pendulum_state"]).to_pandas()
    sl = slice(start, start + n)
    return (t["current_angle"].values[sl].astype(float),
            t["current_x"].values[sl].astype(float),
            t["pendulum_state"].values[sl])


def main():
    import glob
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--ticks", type=int, default=4000)
    ap.add_argument("--nominal", default=f"{CODE}/module.wasm")
    ap.add_argument("--mutant", required=True, help="patched wasm path")
    args = ap.parse_args()
    fpath = args.session or sorted(glob.glob(f"{DATA}/*.parquet"))[0]
    angle, x, state = load_session(fpath, args.start, args.ticks)
    print(f"[replay] {Path(fpath).name} ticks {args.start}..{args.start+args.ticks}"
          f"  states={dict(zip(*np.unique(state, return_counts=True)))}")
    nom = replay(args.nominal, angle, x)
    bug = replay(args.mutant, angle, x)
    print(f"[replay] nominal mean edges/tick={nom.sum(axis=1).mean():.1f}"
          f"  mutant={bug.sum(axis=1).mean():.1f}")
    res = compare(nom, bug)
    print(f"[replay] mean tick-NED(nominal,mutant) = {res['mean_ned']:.4f}")
    print(f"[replay] fraction of ticks divergent   = {res['frac_divergent']:.4f}")
    print(f"[replay] first divergent tick          = {res['first_divergent_tick']}")
    print(f"[replay] top divergent func_ids (localization): {res['top_divergent_funcs']}")


if __name__ == "__main__":
    main()
