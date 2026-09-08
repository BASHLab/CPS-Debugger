"""Are the silent faults dormant, or inert?

The closed-loop corpus finds that 52% of the faults which reroute execution
never move the actuator under the trajectory the machine happened to run. A
reviewer can reasonably ask why such a fault is worth an alarm rather than a
behaviourally inert edit. The answer the incident record suggests is latency:
the Boeing 787 generator counter was silent for 248 days before it was not.

This tests it directly. Each silent fault is re-run from several initial
conditions. A fault that moves the actuator under any of them was dormant, not
inert: the observed trajectory simply never exercised it in a way that reached
a command.

Run:  SHARD=i NSHARD=n python3 dormancy.py
"""
import json
import math
import os
from pathlib import Path

import numpy as np

import cosim_faults as cf
from edge_localize import mutant, SITES
from plant import CartPole, PlantParams, PlantState
from waxi_host import WaxiHost
from run_sil import VOCAB

TICKS = int(os.environ.get("DORM_TICKS", 20000))
SHARD = int(os.environ.get("SHARD", 0))
NSHARD = int(os.environ.get("NSHARD", 1))
OUT = Path(f"results/dormancy/shard_{SHARD:03d}.json")

# Initial conditions. pi is hanging straight down, which is what the corpus ran.
# The others start the rod part-way up or near balance, so the controller enters
# different phases and exercises branches the hanging start never reaches.
ICS = {"hanging": math.pi, "hanging_off": math.pi - 0.30, "swung": 1.20,
       "near_top": 0.50, "balanced": 0.05}


def run_from(module, phi0, n, ref=None):
    """cf.run_closed, but from a chosen initial rod angle."""
    h = WaxiHost(module, VOCAB, quiet=True); h.service_init()
    p = CartPole(PlantParams(**cf.PLANT), PlantState(phi=phi0))
    edges, vcmd, diff, held = [], [], None, None
    for t in range(n):
        if held is None or t % cf.SENSOR_HOLD == 0:
            held = p.input_image()
        h.write_input(held)
        tk = h.tick(decode=True)
        if ref is None:
            edges.append(tk)
        elif diff is None and t < len(ref) and tk != ref[t]:
            diff = t
        v = p.decode_velsw(h.read_output())
        vcmd.append(float(v))
        p.step(v)
    del h
    return (edges if ref is None else diff), np.array(vcmd)


def main():
    full = json.loads(Path("results/cosim_faults_full.json").read_text())
    silent = [r for r in full["rows"]
              if r.get("status") == "ok" and r.get("detected")
              and r.get("first_act_tick", -1) < 0]
    by_off = {s["off"]: s for s in SITES}
    mine = silent[SHARD::NSHARD]
    print(f"shard {SHARD}/{NSHARD}: {len(mine)} of {len(silent)} silent faults, "
          f"{len(ICS)} initial conditions, {TICKS} ticks", flush=True)

    base = Path(cf.MODULE).read_bytes()
    # One nominal live at a time. Holding all five at 20k ticks is ~20 GB of
    # edge tuples, which is what killed the first attempt; iterating initial
    # conditions on the outside costs nothing extra in compute.
    per_site = {r["off"]: {} for r in mine}
    kinds = {r["off"]: (r["kind"], r["gt"]) for r in mine}
    for name, phi0 in ICS.items():
        e_nom, v_nom = run_from(cf.MODULE, phi0, TICKS)
        print(f"  nominal {name}: phi0={phi0:.2f}", flush=True)
        for n, r in enumerate(mine, 1):
            site = by_off.get(r["off"])
            if site is None:
                continue
            m = mutant(base, site)
            if m is None:
                per_site[r["off"]][name] = {"status": "unpatchable"}
                continue
            p_w = f"/tmp/dorm_{os.getpid()}_{site['off']}.wasm"
            Path(p_w).write_bytes(m)
            try:
                d, v_f = run_from(p_w, phi0, TICKS, ref=e_nom)
                Lv = min(len(v_nom), len(v_f))
                dv = np.asarray(v_f[:Lv]) != np.asarray(v_nom[:Lv])
                per_site[r["off"]][name] = {
                    "status": "ok", "trace_diverged": d is not None,
                    "act_moved": bool(dv.any()),
                    "first_act_tick": int(np.argmax(dv)) if dv.any() else -1}
            except Exception as exc:
                per_site[r["off"]][name] = {"status": f"runtime:{type(exc).__name__}"}
            finally:
                Path(p_w).unlink(missing_ok=True)
        del e_nom, v_nom
        import gc; gc.collect()

    rows = []
    for off, per_ic in per_site.items():
        act = [k for k, v in per_ic.items()
               if v.get("status") == "ok" and v.get("act_moved")]
        kind, gt = kinds[off]
        rows.append(dict(off=off, kind=kind, gt=gt, per_ic=per_ic,
                         activated_in=act, activated=bool(act)))
        print(f"  off={off} {kind} activated_in={act}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(shard=SHARD, nshard=NSHARD, ics=ICS, ticks=TICKS,
                   n_silent_total=len(silent), rows=rows), open(OUT, "w"), indent=1)
    print(f"wrote {OUT} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
