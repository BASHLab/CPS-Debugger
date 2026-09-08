"""run_corpus.py -- mutation corpus over open-loop replay (workstream M).

Injects each fault as a layout-preserving constant mutation of the shipped
binary, replays a real session through the shipped and mutated binaries, and
scores:
  * detection  -- does the fault produce a trace divergence?
  * localization -- does the divergence point to the function that actually
    contains the mutated constant (ground truth from layout.json)? We score
    two signals: the ORIGIN function (first-diverging tick; the causal site)
    and the MAX function (largest total divergence; often a downstream
    consequence when the fault reroutes the trajectory).

Corpus = control constants uniquely locatable in the shipped binary, spanning
counter bounds, a deadband, thresholds, and a gain. A gain that only scales a
value changes no branch, so the trace is unchanged and detection is (correctly)
negative -- the complement the physical/encoder detector covers.

CPU-only (wasmtime + pyarrow). Writes results/corpus_replay.json.
"""
import os as _os
from pathlib import Path as _Path

# Roots. Override with environment variables; defaults assume this checkout and
# the dataset release unpacked as CPSD_DATA (see outputs/sil_demo/README.md).
REPO = _Path(_os.environ.get("CPSD_REPO", _Path(__file__).resolve().parents[2]))
CPSD_DATA = _Path(_os.environ.get("CPSD_DATA", REPO / "data"))
CPSD_CAPTURE = _Path(_os.environ.get("CPSD_CAPTURE", CPSD_DATA / "captures"))

import json
import struct
from pathlib import Path

import numpy as np

from replay import load_session, replay, compare, CODE, DATA

OUT = Path(__file__).parent / "results"
LAYOUT = f"{CODE}/layout.json"
NAMES = str(REPO / "outputs/fm_trace_eda/func_id_to_name.json")

# (name, kind, old, new, human-readable site)
CORPUS = [
    ("balanced_counter",  "i32", 1500,   150,   "stays_balanced (counter bound)"),
    ("setpoint_counter",  "i32", 1000,   100,   "tick, setpoint cycling"),
    ("counter_goodrange", "i32", 500,    100,   "pou_lqr_sim goodrange gate"),
    ("deadband_v",        "f64", 0.0001, 0.05,  "pou_lqr_sim velocity deadband"),
    ("theta_cap",         "f64", 0.07,   0.20,  "pou_standup_rel angle test"),
    ("x_safety",          "f64", 0.15,   0.05,  "tick x-safety abort"),
    ("Kx",                "f64", 100.0,  400.0, "pou_lqr_sim position gain"),
]


def _sleb(v):
    out = bytearray(); more = True
    while more:
        b = v & 0x7f; v >>= 7
        if (v == 0 and not (b & 0x40)) or (v == -1 and (b & 0x40)):
            more = False
        else:
            b |= 0x80
        out.append(b)
    return bytes(out)


def _encode(kind, val):
    return bytes([0x41]) + _sleb(val) if kind == "i32" \
        else bytes([0x44]) + struct.pack("<d", val)


def patch_by_value(data, kind, old, new):
    old_b, new_b = _encode(kind, old), _encode(kind, new)
    if len(old_b) != len(new_b):
        raise ValueError(f"width change {old}->{new} not layout-preserving")
    if data.count(old_b) != 1:
        raise ValueError(f"{old} not unique ({data.count(old_b)})")
    off = data.find(old_b)
    return data[:off] + new_b + data[off + len(new_b):]


def _disasm(entry):
    for blk in entry.get("br_blocks", {}).values():
        if isinstance(blk, list):
            for ins in blk:
                if isinstance(ins, dict):
                    yield ins.get("inst", "")


def gt_func(layout, kind, value):
    """Ground-truth fault function: the func whose disassembly contains the
    mutated constant. layout stores f64 constants as their raw u64 bits."""
    if kind == "i32":
        needle = f"i32.const {value}"
    else:
        needle = f"f64.const {struct.unpack('<Q', struct.pack('<d', value))[0]}"
    return [int(f) for f, e in layout.items()
            if isinstance(e, dict) and any(needle in i for i in _disasm(e))]


def main():
    module = f"{CODE}/module.wasm"
    base = Path(module).read_bytes()
    layout = json.loads(Path(LAYOUT).read_text())
    names = json.loads(Path(NAMES).read_text())
    fpath = sorted(Path(DATA).glob("*.parquet"))[0].as_posix()
    angle, x, state = load_session(fpath, 0, 15000)
    print(f"[corpus] window 0..15000  states="
          f"{dict(zip(*np.unique(state, return_counts=True)))}")
    nom = replay(module, angle, x)

    rows, n_det, hit1_orig, hit1_max, hitk = [], 0, 0, 0, 0
    for name, kind, old, new, site in CORPUS:
        try:
            mut = patch_by_value(base, kind, old, new)
        except ValueError as e:
            print(f"[corpus] {name}: SKIP ({e})"); continue
        Path(f"/tmp/mut_{name}.wasm").write_bytes(mut)
        res = compare(nom, replay(f"/tmp/mut_{name}.wasm", angle, x))
        gt = set(gt_func(layout, kind, old))
        origin = [f for f, _ in res["origin_funcs"]]
        top = [f for f, _ in res["top_divergent_funcs"]]
        det = res["frac_divergent"] > 0
        o1 = bool(origin) and origin[0] in gt
        m1 = bool(top) and top[0] in gt
        hk = bool(gt & set(top[:5] + origin[:5]))
        if det:
            n_det += 1; hit1_orig += o1; hit1_max += m1; hitk += hk
        rows.append(dict(name=name, detected=det, frac=res["frac_divergent"],
                         first_tick=res["first_divergent_tick"], gt=sorted(gt),
                         origin=origin[:4], top=top[:4],
                         hit1_origin=o1, hit1_max=m1, hitk=hk, site=site))
        gtn = "/".join(names.get(str(f), "?") for f in sorted(gt))
        print(f"[corpus] {name:17s} det={det!s:5} frac={res['frac_divergent']:.4f} "
              f"GT={sorted(gt)}({gtn})  origin={origin[:3]} max={top[:3]}  "
              f"hit@1_origin={o1} hit@1_max={m1} hit@k={hk}")

    exercised = int((nom.sum(axis=0) > 0).sum())
    summary = dict(n=len(rows), detected=n_det, hit1_origin=hit1_orig,
                   hit1_max=hit1_max, hitk=hitk, exercised=exercised)
    OUT.mkdir(exist_ok=True)
    (OUT / "corpus_replay.json").write_text(
        json.dumps({"mutants": rows, "summary": summary}, indent=2))
    print(f"\n[corpus] detection {n_det}/{len(rows)}; of detected: "
          f"hit@1(origin)={hit1_orig}/{n_det}  hit@1(max)={hit1_max}/{n_det}  "
          f"hit@k={hitk}/{n_det}")


if __name__ == "__main__":
    main()
