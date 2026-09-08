"""Improved localizer, diagnosis-driven: rank divergence over CONTROLLER
functions only (exclude the libm/printf/malloc noise the rerouted trajectory
lights up). Re-measure hit@1 on the detected control faults vs the old
all-function ranking, and count the cause-not-effect cases (GT has zero own
divergence -> needs call-graph attribution).
"""
import json
import struct
from pathlib import Path
from collections import defaultdict

import numpy as np

from replay import load_session, replay, compare, CODE, DATA
from run_opcode import swap_byte
from run_corpus import patch_by_value

MODULE = f"{CODE}/module.wasm"
SITES = json.loads((Path(__file__).parent / "results" / "mutation_sites.json").read_text())
CMP_INV = {0x63: 0x64, 0x64: 0x63, 0x65: 0x66, 0x66: 0x65}
ARITH_INV = {0xA0: 0xA1, 0xA1: 0xA0}
# Controller functions (localization candidates); everything else (libm, printf,
# malloc, wasi) is infrastructure the fault may reroute through but is never the
# fault site.
CONTROLLER = {20, 21, 22, 33, 36, 37, 38, 39, 40, 42}


def mutant(base, s):
    if s["kind"] == "f64cmp":
        return swap_byte(base, s["off"], s["op"], CMP_INV[s["op"]])
    if s["kind"] == "f64arith":
        return swap_byte(base, s["off"], s["op"], ARITH_INV[s["op"]])
    if s["kind"] == "f64const":
        c = s["value"]; return patch_by_value(base, "f64", c, 1.0 if c == 0 else c * 2.0)
    return None


def main():
    angle, x, _ = load_session(sorted(Path(DATA).glob("*.parquet"))[0].as_posix(), 0, 15000)
    nom = replay(MODULE, angle, x)
    base = Path(MODULE).read_bytes()
    ctrl = [s for s in SITES if s["kind"] in ("f64cmp", "f64arith", "f64const")]
    agg = defaultdict(lambda: [0, 0, 0, 0])   # kind -> [det, hit_all, hit_ctrl, causeNoEffect]
    for s in ctrl:
        try:
            m = mutant(base, s)
        except Exception:
            continue
        p = f"/tmp/il_{s['off']}.wasm"; Path(p).write_bytes(m)
        try:
            ev = compare(nom, replay(p, angle, x))
        except Exception:
            Path(p).unlink(missing_ok=True); continue
        Path(p).unlink(missing_ok=True)
        if ev["frac_divergent"] == 0:
            continue
        gt = s["func"]; k = s["kind"]; agg[k][0] += 1
        ranked = ev["top_divergent_funcs"]                    # (func, weight) desc
        allids = [f for f, _ in ranked]
        ctrlids = [f for f, _ in ranked if f in CONTROLLER]
        gt_w = dict(ranked).get(gt, 0.0)
        if allids and allids[0] == gt:
            agg[k][1] += 1
        if ctrlids and ctrlids[0] == gt:
            agg[k][2] += 1
        if gt_w == 0.0:
            agg[k][3] += 1                                     # cause not effect
    print(f"{'kind':10} {'det':>4} {'hit@1 all':>10} {'hit@1 ctrl-only':>16} {'cause!=effect':>14}")
    tot = [0, 0, 0, 0]
    for k, (d, ha, hc, ce) in sorted(agg.items()):
        for i, v in enumerate((d, ha, hc, ce)):
            tot[i] += v
        print(f"{k:10} {d:4d} {ha:>4}/{d:<4}={ha/d*100 if d else 0:3.0f}% "
              f"{hc:>4}/{d:<4}={hc/d*100 if d else 0:3.0f}% {ce:>10}")
    d, ha, hc, ce = tot
    print(f"\nALL control faults det={d}: hit@1 all={ha/d*100:.0f}%  "
          f"hit@1 controller-only={hc/d*100:.0f}%  cause!=effect={ce} ({ce/d*100:.0f}%)")


if __name__ == "__main__":
    main()
