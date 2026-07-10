"""run_sil.py — closed-loop software-in-the-loop driver.

Couples the shipped controller (WaxiHost) to the cart-pole plant (CartPole):
each tick, synthesize the sensor image from plant state, run one control tick,
decode the velocity setpoint, and integrate the plant forward.

Two entry points:
  smoke()  — scan the four (enc_sign, vel_sign) combinations from the hanging
             state and report which one lets the controller swing up and
             balance. This is the go/no-go for the physics coupling: a correct
             sign combo should drive |phi| toward 0 and hold it.
  run()    — run N ticks under fixed params, collecting the per-tick trace
             (decoded triples) and the logged sensor channels, and emit an
             extracted/-style session directory for the trace->token pipeline.

All heavy runs go through sbatch per repo convention; smoke() is light
(wasmtime, no torch) and may run inline for the go/no-go.
"""
import argparse
import json
import math
import struct
from collections import Counter
from pathlib import Path

from plant import CartPole, PlantParams, PlantState
from waxi_host import WaxiHost

CAPTURE = Path("/home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs"
               "/extracted/2025-03-17_10-36-44/code")
VOCAB = Path("/home/simran/allspark-data-exploration/CPS-Debugger"
             "/outputs/fm_candidates/train_data_full/auto_token_mapping.json")


def _drive(host, plant, n_ticks, collect_trace=False):
    """Run n_ticks of the closed loop. Returns per-tick telemetry."""
    tele = {"phi": [], "x": [], "v_cmd": [], "n_edges": []}
    triples_per_tick = [] if collect_trace else None
    for _ in range(n_ticks):
        host.write_input(plant.input_image())
        triples = host.tick(decode=collect_trace)
        if collect_trace:
            triples_per_tick.append(triples)
            n_edges = len(triples)
        else:
            n_edges = triples  # tick(decode=False) returns the count
        v_cmd = plant.decode_velsw(host.read_output())
        plant.step(v_cmd)
        tele["phi"].append(plant.s.phi)
        tele["x"].append(plant.s.x)
        tele["v_cmd"].append(v_cmd)
        tele["n_edges"].append(n_edges)
    return tele, triples_per_tick


def _balance_score(phi_trace, tail=2000):
    """Fraction of the last `tail` ticks with |phi| < 0.2 rad (near upright)."""
    tail_phi = phi_trace[-tail:] if len(phi_trace) >= tail else phi_trace
    if not tail_phi:
        return 0.0
    return sum(1 for p in tail_phi if abs(p) < 0.2) / len(tail_phi)


def smoke(n_ticks=8000, module=str(CAPTURE / "module.wasm")):
    """Scan the four sign combos; report which balances the pendulum."""
    results = {}
    for enc_sign in (1, -1):
        for vel_sign in (1, -1):
            host = WaxiHost(module, VOCAB, quiet=True)
            host.service_init()
            params = PlantParams(enc_sign=enc_sign, vel_sign=vel_sign)
            plant = CartPole(params, PlantState(phi=math.pi))  # start hanging
            tele, _ = _drive(host, plant, n_ticks, collect_trace=False)
            score = _balance_score(tele["phi"])
            min_abs_phi = min(abs(p) for p in tele["phi"])
            results[(enc_sign, vel_sign)] = {
                "balance_frac": round(score, 4),
                "min_abs_phi": round(min_abs_phi, 4),
                "final_abs_phi": round(abs(tele["phi"][-1]), 4),
                "mean_edges": round(sum(tele["n_edges"]) / len(tele["n_edges"]), 1),
            }
            print(f"enc_sign={enc_sign:+d} vel_sign={vel_sign:+d}  "
                  f"balance_frac={score:.3f}  min|phi|={min_abs_phi:.3f}  "
                  f"final|phi|={abs(tele['phi'][-1]):.3f}  "
                  f"edges/tick={results[(enc_sign,vel_sign)]['mean_edges']}")
    best = max(results, key=lambda k: results[k]["balance_frac"])
    print(f"\nbest sign combo: enc_sign={best[0]:+d} vel_sign={best[1]:+d} "
          f"(balance_frac={results[best]['balance_frac']:.3f})")
    go = results[best]["balance_frac"] >= 0.5 and results[best]["min_abs_phi"] < 0.07
    print(f"[smoke] {'GO: controller balances in SIL' if go else 'INVESTIGATE: no combo balances'}")
    return results, best, go


def run(n_ticks, out_dir, enc_sign, vel_sign, module=str(CAPTURE / "module.wasm")):
    """Full run with trace collection; writes telemetry + trace to out_dir."""
    host = WaxiHost(module, VOCAB, quiet=True)
    host.service_init()
    params = PlantParams(enc_sign=enc_sign, vel_sign=vel_sign)
    plant = CartPole(params, PlantState(phi=math.pi))
    tele, triples = _drive(host, plant, n_ticks, collect_trace=True)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vocab = {tuple(t) for t in json.loads(VOCAB.read_text())["triples"]}
    n_edges = sum(len(t) for t in triples)
    n_bad = sum(1 for tk in triples for tr in tk if tr not in vocab)
    (out / "telemetry.json").write_text(json.dumps({
        "n_ticks": n_ticks, "enc_sign": enc_sign, "vel_sign": vel_sign,
        "balance_frac": _balance_score(tele["phi"]),
        "n_edges": n_edges, "n_out_of_vocab": n_bad,
        "frac_in_vocab": (n_edges - n_bad) / max(n_edges, 1),
        "phi": tele["phi"], "x": tele["x"], "v_cmd": tele["v_cmd"],
    }))
    # Compact trace: one line of space-joined token ids per tick handled by the
    # downstream aspk_build_session_full.py; here we dump raw triples as JSONL.
    with (out / "trace_triples.jsonl").open("w") as f:
        for tk in triples:
            f.write(json.dumps([list(t) for t in tk]) + "\n")
    print(f"[run] {n_ticks} ticks -> {out}  frac_in_vocab="
          f"{(n_edges - n_bad) / max(n_edges, 1):.6f}  balance_frac="
          f"{_balance_score(tele['phi']):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "run"])
    ap.add_argument("--ticks", type=int, default=8000)
    ap.add_argument("--out", default="sil_runs/run0")
    ap.add_argument("--enc-sign", type=int, default=1)
    ap.add_argument("--vel-sign", type=int, default=1)
    args = ap.parse_args()
    if args.mode == "smoke":
        smoke(args.ticks)
    else:
        run(args.ticks, args.out, args.enc_sign, args.vel_sign)


if __name__ == "__main__":
    main()
