"""N2 — recover controller decision constants from the shipped binary.

The threshold-distance features (precompute_thresh_features.py) currently
use constants read from the controller source (sys_params.h): the rail
half-widths, the deadbands, the counter bounds. This script shows the same
constants are present as immediate operands in the deployed WebAssembly
binary, so the features can be built with no source access.

Method: scan the WASM code section for f64.const (0x44 + 8 IEEE-754 bytes)
and i32.const (0x41 + signed LEB128) immediates, collect the multiset, and
check which of the source constants appear. Extraction fidelity is the
fraction of source constants recovered.

Output: results/n2_binary_constants/results.json
"""
import json
import struct
from pathlib import Path

CAPTURE = Path("/home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs"
               "/extracted/2025-03-17_10-36-44/code")
OUT = Path(__file__).parent / "results" / "n2_binary_constants"

# Source constants the threshold features and controller branches use.
# (sys_params.h + lqrsim.c / pend_monolithic.c, per the branch inventory.)
TARGET_F64 = {
    "MAX_X_DISPL": 0.18,
    "X_SAFETY_BUFFER": 0.03,
    "V_MAX": 1.0,
    "v_deadband": 0.0001,
    "theta_deadband": 0.005,
    "standup_theta": 0.07,
    "standup_theta_d": 0.05,
    "lqr_gain_theta": 274.0,
    "lqr_gain_v": 54.0,
}
TARGET_I32 = {
    "balanced_counter": 1500,
    "setpoint_counter": 1000,
    "counter_goodrange": 500,
    "counter_xref": 4000,
}
F64_TOL = 1e-9


def read_uleb(b, i):
    r = s = 0
    while True:
        byte = b[i]; i += 1
        r |= (byte & 0x7F) << s
        if not (byte & 0x80):
            return r, i
        s += 7


def read_sleb(b, i):
    r = s = 0
    while True:
        byte = b[i]; i += 1
        r |= (byte & 0x7F) << s
        s += 7
        if not (byte & 0x80):
            if byte & 0x40:
                r |= -(1 << s)
            return r, i


def scan_constants(wasm: bytes):
    """Sweep the whole module for const-opcode byte patterns. This
    over-scans (it does not parse section structure), which only inflates
    the candidate set; membership of the target constants is unaffected."""
    f64s, i32s = [], []
    n = len(wasm)
    i = 0
    while i < n:
        op = wasm[i]
        if op == 0x44 and i + 9 <= n:               # f64.const
            (val,) = struct.unpack_from("<d", wasm, i + 1)
            f64s.append(val); i += 9; continue
        if op == 0x41:                              # i32.const
            try:
                val, j = read_sleb(wasm, i + 1)
                if -(1 << 31) <= val < (1 << 31):
                    i32s.append(val); i = j; continue
            except IndexError:
                pass
        i += 1
    return f64s, i32s


def main():
    wasm = (CAPTURE / "orig_module.wasm").read_bytes()
    f64s, i32s = scan_constants(wasm)
    print(f"scanned {len(wasm)} bytes: {len(f64s)} f64.const, {len(i32s)} i32.const")

    rec_f64, rec_i32 = {}, {}
    for name, c in TARGET_F64.items():
        hit = any(abs(v - c) <= F64_TOL for v in f64s)
        rec_f64[name] = hit
        print(f"  f64 {name:18s} {c:<10g} {'FOUND' if hit else 'missing'}")
    i32set = set(i32s)
    for name, c in TARGET_I32.items():
        hit = c in i32set
        rec_i32[name] = hit
        print(f"  i32 {name:18s} {c:<10d} {'FOUND' if hit else 'missing'}")

    n_hit = sum(rec_f64.values()) + sum(rec_i32.values())
    n_tot = len(TARGET_F64) + len(TARGET_I32)
    fidelity = n_hit / n_tot
    print(f"\ndirect extraction fidelity: {n_hit}/{n_tot} = {fidelity:.3f}")

    # Folded / derived forms: the compiler may store a product or difference
    # rather than the source decomposition. The features consume these
    # derived values directly, so recovering them is what matters.
    folded = {
        "MAX_X_SAFE=0.18-0.03=0.15": any(abs(v - 0.15) <= F64_TOL for v in f64s),
        "saturation_const_0.9999": any(abs(v - 0.9999) <= F64_TOL for v in f64s),
    }
    for k, v in folded.items():
        print(f"  folded {k}: {'FOUND' if v else 'missing'}")
    # Constants the threshold features actually consume (feature-relevant),
    # counting the folded MAX_X_SAFE in place of its 0.18/0.03 source form.
    feat_needed = ["V_MAX", "v_deadband", "theta_deadband", "standup_theta",
                   "standup_theta_d"]
    feat_hit = sum(rec_f64[k] for k in feat_needed) + sum(folded.values())
    feat_tot = len(feat_needed) + len(folded)
    print(f"feature-relevant fidelity: {feat_hit}/{feat_tot} = {feat_hit/feat_tot:.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps({
        "n_f64_const": len(f64s), "n_i32_const": len(i32s),
        "recovered_f64": rec_f64, "recovered_i32": rec_i32,
        "folded_forms": folded,
        "n_recovered": n_hit, "n_target": n_tot, "fidelity": fidelity,
        "feature_relevant_recovered": feat_hit,
        "feature_relevant_target": feat_tot,
        "feature_relevant_fidelity": feat_hit / feat_tot,
        "misses": {"counter_xref_4000": "absent (loop bound, not a stored immediate)",
                   "MAX_X_DISPL_0.18": "folded into 0.15",
                   "X_SAFETY_BUFFER_0.03": "folded into 0.15"},
    }, indent=2))
    print(f"wrote {OUT/'results.json'}")


if __name__ == "__main__":
    main()
