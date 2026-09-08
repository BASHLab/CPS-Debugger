"""Localization floors for the control-flow corpus, scored on the SAME detected
set and denominator as the edge-level localizer (edge_localize.json rows). A
result is meaningful only against a floor, so this reports what trivial
localizers achieve on the identical faults:

  * random          -- predict a uniformly random controller function; expected
                        hit@1 = 1 / |candidate functions|.
  * frequency-prior -- always predict the busiest controller function (most
                        executed edges in the nominal trace); hit@1 = fraction
                        of faults whose ground-truth function is that one.
  * automaton        -- an I/O-event automaton (NDAAO-lite) flags a deviant
                        cycle but names no branch, so at function granularity it
                        localizes nothing. This is the granularity contrast, not
                        a head-to-head.

Writes results/localize_baselines.json.
"""
import json
from pathlib import Path

import numpy as np

from replay import load_session, replay, CODE, DATA

MODULE = f"{CODE}/module.wasm"
SILRES = Path(__file__).parent / "results"
# Controller functions: the plausible localization targets. Everything else
# (libm, printf, malloc, wasi) is infrastructure a fault may reroute through
# but is never the fault site.
CONTROLLER = {20, 21, 22, 33, 36, 37, 38, 39, 40, 42}


def main():
    el = json.loads((SILRES / "edge_localize.json").read_text())
    rows = el["rows"]                                   # detected faults only
    gts = [r["gt"] for r in rows]
    det = len(rows)
    assert det == el["detected"], (det, el["detected"])

    # Nominal execution frequency over the identical window edge_localize used.
    angle, x, _ = load_session(sorted(Path(DATA).glob("*.parquet"))[0].as_posix(),
                               0, 15000)
    counts = replay(MODULE, angle, x).sum(axis=0)       # (NFUNC,) total edges
    cand = sorted(f for f in CONTROLLER if counts[f] > 0)
    freq_top = max(cand, key=lambda f: counts[f])

    random_pct = 100.0 / len(cand)
    random_hit = det / len(cand)                        # expected count
    freq_hit = sum(1 for g in gts if g == freq_top)

    out = dict(
        detected=det,
        n_candidates=len(cand),
        candidates=cand,
        freq_top=int(freq_top),
        random={"expected_hit1": round(random_hit, 2),
                "hit1_pct": round(random_pct, 1)},
        frequency_prior={"hit1": freq_hit,
                         "hit1_pct": round(freq_hit / det * 100, 1)},
        automaton={"hit1": 0, "hit1_pct": 0.0,
                   "note": "names a deviant cycle, not a branch"},
        # Report BOTH attribution frames. The raw frame lands within a point of
        # the frequency prior, so quoting it alone would understate the
        # localizer and misrepresent where its accuracy comes from.
        edge_localizer={"hit1": el["hit1"],
                        "hit1_pct": round(el["hit1"] / det * 100, 1)},
        edge_localizer_blame={"hit1": el["hit1_blame"],
                              "hit1_pct": round(el["hit1_blame"] / det * 100, 1)},
    )
    (SILRES / "localize_baselines.json").write_text(json.dumps(out, indent=1))
    print(f"detected faults        : {det}  (candidate funcs: {len(cand)})")
    print(f"random (expected)      : {random_hit:.2f}/{det} = {random_pct:.0f}%")
    print(f"frequency-prior (f{freq_top}) : {freq_hit}/{det} = "
          f"{freq_hit/det*100:.0f}%")
    print(f"automaton (NDAAO-lite) : names no branch = 0%")
    print(f"edge-level, raw frame  : {el['hit1']}/{det} = "
          f"{el['hit1']/det*100:.0f}%")
    print(f"edge-level, blame frame: {el['hit1_blame']}/{det} = "
          f"{el['hit1_blame']/det*100:.0f}%")


if __name__ == "__main__":
    main()
