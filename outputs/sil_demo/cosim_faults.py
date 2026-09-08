"""Closed-loop fault demonstration on the identified/validated plant.

Unlike open-loop replay (real sensors fixed), here a control-flow fault reroutes
the controller, whose changed actuator command feeds back through the faithful
plant into FUTURE sensor readings. For each fault we run the shipped and mutated
binaries in closed loop from the same initial condition and report:
  * trace divergence + edge-level first-difference localization (fault -> branch);
  * the closed-loop signature open-loop cannot show: the tick the physical
    trajectory (theta) first departs the nominal, its peak departure, and whether
    the fault destabilizes the stand (nominal stays near the top; faulty falls).

Plant params are all grounded in data (see aspk_sil_plant_id): 2 ms control rate,
velocity servo tau_v = 5 ms, validated MSE friction.
"""
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np

from plant import CartPole, PlantParams, PlantState
from waxi_host import WaxiHost
from run_sil import CAPTURE, VOCAB
from edge_localize import mutant, SITES, blame

MODULE = str(CAPTURE / "module.wasm")
# The rig runs two clocks: the controller task cycles every 1 ms (cycle_count
# increments once per logged row) while the EtherCAT bus refreshes sensors every
# 2 ms (current_angle is identical across tick pairs 99.8% of the time, and the
# wire carries 244,242 bus cycles for 491,915 rows). Running the co-simulation at
# a 2 ms control period with a fresh input image every tick got both wrong and
# held balance for only ~240 ticks; with the two rates separated it sustains
# 5,788, past the 1,500-tick stays_balanced gate.
PLANT = dict(tau_v=0.005, dt_control=1.0e-3, substeps=8)
SENSOR_HOLD = 2            # controller ticks per EtherCAT sensor refresh
SILRES = Path(__file__).parent / "results"


import gc


def run_closed(module, n, ref=None):
    """Run the loop for n ticks.

    With ref=None the whole per-tick edge sequence is returned, which is what the
    nominal run needs. With ref supplied (the nominal sequence) the faulty run is
    compared tick by tick and its edges are discarded as they are consumed,
    returning only the first divergence. Holding both sequences for a long run
    exhausted memory: at ~91 edges a tick, 120k ticks is millions of tuples per
    run and two of them do not fit.
    """
    h = WaxiHost(module, VOCAB, quiet=True); h.service_init()
    p = CartPole(PlantParams(**PLANT), PlantState(phi=math.pi))
    edges, phi, vcmd = [], [], []
    diff = None
    held = None
    for t in range(n):
        if held is None or t % SENSOR_HOLD == 0:
            held = p.input_image()
        h.write_input(held)
        tk = h.tick(decode=True)
        if ref is None:
            edges.append(tk)
        elif diff is None and t < len(ref) and tk != ref[t]:
            _raw, _pre = _first_edge_attr(ref[t], tk)
            diff = (t, _raw, blame(_pre, _raw))
        _v = p.decode_velsw(h.read_output())
        vcmd.append(float(_v))
        p.step(_v)
        phi.append(p.s.phi)
    del h
    gc.collect()
    return (edges if ref is None else diff), np.array(phi), np.array(vcmd)


def _first_edge_attr(a, b):
    """(func of the first differing edge, faulty edges before it) in one tick.

    The prefix is what blame() walks back through, matching analyze()'s
    blame(e1[:i], raw) in the open-loop study.
    """
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else None
        y = b[i] if i < len(b) else None
        if x != y:
            return (y if y is not None else x)[0], list(b[:i])
    return None, []


def _first_edge_func(a, b):
    """Func of the first differing edge within one tick."""
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else None
        y = b[i] if i < len(b) else None
        if x != y:
            return (y if y is not None else x)[0]
    return None


def first_diff_edge(e0, e1):
    """First executed edge that differs -> (tick, func). The flipped branch lives
    in the fault's own function, so its func is the localization prediction."""
    for t in range(min(len(e0), len(e1))):
        if e0[t] != e1[t]:
            a, b = e0[t], e1[t]
            for i in range(max(len(a), len(b))):
                x0 = a[i] if i < len(a) else None
                y0 = b[i] if i < len(b) else None
                if x0 != y0:
                    return t, (y0 if y0 is not None else x0)[0]
    return None, None


def exercised_subset(e_nom, k=16):
    """Faults whose branch site sits in a function the nominal actually executes.

    This filter used to exclude the whole balance regime, because the co-sim
    never sustained balance and so never ran the LQR. With the rig's two clocks
    modelled the run does reach balance, so balance-phase functions now enter
    the subset on their own and the exclusion is no longer doing the work."""
    exe = {f for tk in e_nom for f, a, b in tk}
    ctrl = [s for s in SITES if s["kind"] in ("relational", "threshold")
            and s["func"] in exe]
    rel = [s for s in ctrl if s["kind"] == "relational"]
    thr = [s for s in ctrl if s["kind"] == "threshold"]
    out, seen = [], set()
    for s in rel[:k // 2] + thr[:k // 2]:
        if s["off"] not in seen:
            seen.add(s["off"]); out.append(s)
    return sorted(exe), out


def main():
    # Swing-up takes roughly 22,000 ticks at the 1 ms control rate before the
    # first balance episode, so a run must be long enough to contain it. The
    # previous 8,000 stopped short of balance entirely, which is part of why
    # the closed-loop corpus looked unreachable.
    n = int(os.environ.get("COSIM_TICKS", 40000))
    base = Path(MODULE).read_bytes()
    e_nom, phi_nom, _v_nom = run_closed(MODULE, n)
    nom_top = float(np.mean(np.abs(phi_nom[-2000:]) < 0.2))     # nominal near top?
    exe, subset = exercised_subset(e_nom)
    print(f"nominal: final|phi|={abs(phi_nom[-1]):.3f} near_top_frac={nom_top:.2f} "
          f"exercised_funcs={exe}", flush=True)
    print(f"testing {len(subset)} faults in exercised functions", flush=True)
    rows = []
    for s in subset:
        try:
            m = mutant(base, s)
        except Exception:
            m = None
        if m is None:
            continue
        p = f"/tmp/clf_{s['off']}.wasm"; Path(p).write_bytes(m)
        try:
            d_f, phi_f, _v_f = run_closed(p, n, ref=e_nom)
        except Exception:
            Path(p).unlink(missing_ok=True); continue
        Path(p).unlink(missing_ok=True)
        t_edge, _pred_raw, pred = d_f if d_f else (None, None, None)
        L = min(len(phi_nom), len(phi_f))
        dth = np.abs(phi_f[:L] - phi_nom[:L])
        t_traj = int(np.argmax(dth > 0.02)) if (dth > 0.02).any() else -1
        r = dict(off=s["off"], kind=s["kind"], gt=s["func"], pred=pred,
                 detected=t_edge is not None, hit=(t_edge is not None and pred == s["func"]),
                 first_edge_tick=t_edge, first_traj_tick=t_traj,
                 max_dtheta=float(dth.max()), final_phi_faulty=float(abs(phi_f[-1])))
        rows.append(r)
        print(f"off={r['off']:6d} {r['kind']:10} gt={r['gt']} pred={r['pred']} "
              f"det={r['detected']} hit={r['hit']} edge@{r['first_edge_tick']} "
              f"traj@{r['first_traj_tick']} maxdth={r['max_dtheta']:.2f} "
              f"final|phi|_f={r['final_phi_faulty']:.2f}", flush=True)
    det = [r for r in rows if r["detected"]]
    fed_back = [r for r in det if r["first_traj_tick"] >= 0]
    summary = dict(n_faults=len(rows), detected=len(det),
                   hit1=sum(r["hit"] for r in det),
                   fed_back=len(fed_back),
                   nominal_final_phi=float(abs(phi_nom[-1])))
    (SILRES / os.environ.get("COSIM_OUT", "cosim_faults.json")).write_text(
        json.dumps(dict(summary=summary, nominal_near_top=nom_top, rows=rows), indent=1))
    print("SUMMARY:", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
