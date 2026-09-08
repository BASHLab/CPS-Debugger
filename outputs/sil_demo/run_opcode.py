"""run_opcode.py -- opcode-swap fault mutants (CWE-697 comparison, CWE-480
operator), extending the constant-only corpus to the other two mechanisms
the fault taxonomy names.

Each mutant is a single-byte swap at a verified code-section offset, located
by anchoring off a nearby f64 constant (the opcode a compiled comparison or
arithmetic emits sits a fixed distance from its operand constant). The swap
is layout-preserving by construction: one byte in, one byte out, no offsets
move. Detection and localization reuse the trace-differential replay.

CPU-only. Reuses the cached nominal replay matrix. Writes results/opcode_replay.json.
"""
import json
import os
from pathlib import Path

import numpy as np

from replay import load_session, replay, compare, CODE, DATA

REPO = Path(os.environ.get("CPSD_REPO",
                           Path(__file__).resolve().parents[2]))
NAMES = json.loads((REPO / "outputs/fm_trace_eda/func_id_to_name.json").read_text())

# name, byte offset, old opcode, new opcode, CWE, ground-truth function id, note
MUTANTS = [
    ("theta_cmp",    26760, 0x63, 0x64, "CWE-697", 39,
     "pou_standup_rel: balance-entry test f64.lt -> f64.gt (inverted comparison)"),
    ("deadband_cmp", 27175, 0x64, 0x63, "CWE-697", 40,
     "pou_lqr_sim: velocity-deadband test f64.gt -> f64.lt (inverted comparison)"),
    ("lqr_sign",     27245, 0xA0, 0xA1, "CWE-480", 40,
     "pou_lqr_sim: control-law term f64.add -> f64.sub (flipped sign)"),
]


def swap_byte(data, off, old, new):
    b = bytearray(data)
    if b[off] != old:
        raise ValueError(f"offset {off}: expected 0x{old:02x} got 0x{b[off]:02x}")
    b[off] = new
    return bytes(b)


def main():
    module = f"{CODE}/module.wasm"
    base = Path(module).read_bytes()
    angle, x, state = load_session(sorted(Path(DATA).glob("*.parquet"))[0].as_posix(), 0, 15000)
    cache = Path("/tmp/repair_mat_nom.npy")
    nom = np.load(cache) if cache.exists() else replay(module, angle, x)

    rows, ndet, norigin = [], 0, 0
    for name, off, old, new, cwe, gt_fid, note in MUTANTS:
        p = f"/tmp/op_{name}.wasm"
        Path(p).write_bytes(swap_byte(base, off, old, new))
        ev = compare(nom, replay(p, angle, x))
        origin = [f for f, _ in ev["origin_funcs"]]
        det = ev["frac_divergent"] > 0
        o1 = bool(origin) and origin[0] == gt_fid
        ndet += det; norigin += o1 and det
        rows.append(dict(name=name, cwe=cwe, detected=det,
                         frac_divergent=ev["frac_divergent"],
                         first_tick=ev["first_divergent_tick"],
                         gt_func=NAMES.get(str(gt_fid), str(gt_fid)),
                         origin=[NAMES.get(str(f), str(f)) for f in origin[:3]],
                         hit1_origin=bool(o1 and det), note=note))
        print(f"{name:14} {cwe} det={det!s:5} frac={ev['frac_divergent']:.4f} "
              f"gt={NAMES.get(str(gt_fid))} origin={[NAMES.get(str(f)) for f in origin[:2]]} "
              f"hit@1={o1 and det}")

    summary = dict(n=len(rows), detected=ndet, hit1_origin=norigin)
    (Path(__file__).parent / "results" / "opcode_replay.json").write_text(
        json.dumps({"mutants": rows, "summary": summary}, indent=2))
    print("\nsummary:", summary)


if __name__ == "__main__":
    main()
