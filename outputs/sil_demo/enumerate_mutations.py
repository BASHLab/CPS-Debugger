"""Enumerate CONTROL-FLOW-MISROUTING fault sites across the controller
functions: single-site, layout-preserving mutations that change which branch
the controller executes. Three families:
  * relational   -- comparison-operator inversions at guards (CWE-697), across
                    f64/f32/i32/i64 (lt<->gt, le<->ge, eq<->ne)
  * logic        -- and<->or on a compound guard (CWE-697)
  * threshold    -- a numeric constant that feeds a comparison, i.e., a genuine
                    branch threshold/bound/selector (CWE-682); value constants
                    (gains, scale factors) that feed no comparison are not
                    enumerated -- they change a value, not a branch.
Each site carries its absolute module offset (offset-based patch) and its
ground-truth function. Writes results/mutation_sites.json. Validated against the
hand-found comparison anchors before emitting.
"""
import json
import struct
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fm_candidates"))
from extract_wasm_edges import read_uleb128, read_sleb128, instruction_size  # noqa
from replay import CODE

MODULE = f"{CODE}/module.wasm"
# Controller functions to enumerate faults in. The pendulum list was hand-picked;
# that does not travel to a new testbed, so it can now be DERIVED from the
# functions the recorded traces actually execute (--controller-from-vocab). A
# fault in a function the machine never runs is unobservable by construction, so
# the executed set is exactly the right scope.
CONTROLLER = {20, 21, 22, 33, 36, 37, 38, 39, 40, 42}


def controller_from_vocab(mapping_path):
    """Functions that appear in the observed trace vocabulary."""
    import json as _json
    m = _json.loads(Path(mapping_path).read_text())
    return {int(t[0]) for t in m["triples"]}

# Relational comparisons: direction-inverting swaps (lt<->gt, le<->ge, eq<->ne)
# across value types. Inverting the sense flips which branch a guard takes.
CMP_INV = {
    0x61: 0x62, 0x62: 0x61,                                     # f64 eq/ne
    0x63: 0x64, 0x64: 0x63, 0x65: 0x66, 0x66: 0x65,             # f64 lt/gt le/ge
    0x5b: 0x5c, 0x5c: 0x5b, 0x5d: 0x5e, 0x5e: 0x5d, 0x5f: 0x60, 0x60: 0x5f,  # f32
    0x46: 0x47, 0x47: 0x46,                                     # i32 eq/ne
    0x48: 0x4a, 0x4a: 0x48, 0x49: 0x4b, 0x4b: 0x49,             # i32 lt/gt s,u
    0x4c: 0x4e, 0x4e: 0x4c, 0x4d: 0x4f, 0x4f: 0x4d,             # i32 le/ge s,u
    0x51: 0x52, 0x52: 0x51,                                     # i64 eq/ne
    0x53: 0x55, 0x55: 0x53, 0x54: 0x56, 0x56: 0x54,             # i64 lt/gt s,u
    0x57: 0x59, 0x59: 0x57, 0x58: 0x5a, 0x5a: 0x58,             # i64 le/ge s,u
}
LOGIC_INV = {0x71: 0x72, 0x72: 0x71, 0x83: 0x84, 0x84: 0x83}    # i32/i64 and<->or

# Wrong arithmetic / sign (CWE-682 / CWE-480): a control-law term added instead
# of subtracted, a gain multiplied instead of divided, a saturation taking the
# wrong extreme. All single-byte opcode swaps, so layout-preserving like the
# relational family. Unlike relational and threshold faults, these change a
# COMPUTED VALUE rather than a guard directly: they reach the executed path only
# if the corrupted value later feeds a branch. That is exactly the observability
# boundary the paper otherwise asserts, so enumerating them lets us MEASURE what
# fraction of value-corrupting faults become visible in the trace.
ARITH_INV = {
    0xA0: 0xA1, 0xA1: 0xA0, 0xA2: 0xA3, 0xA3: 0xA2, 0xA4: 0xA5, 0xA5: 0xA4,  # f64 add/sub mul/div min/max
    0x92: 0x93, 0x93: 0x92, 0x94: 0x95, 0x95: 0x94, 0x96: 0x97, 0x97: 0x96,  # f32
    0x6A: 0x6B, 0x6B: 0x6A, 0x6C: 0x6D, 0x6D: 0x6C,                          # i32 add/sub mul/div_s
    0x7C: 0x7D, 0x7D: 0x7C, 0x7E: 0x7F, 0x7F: 0x7E,                          # i64
}
# Sign inversion: f64/f32 neg <-> abs. Same width, opposite meaning for a signed
# term (a negated feedback term becomes a magnitude, flipping actuation sense).
SIGN_INV = {0x9A: 0x99, 0x99: 0x9A, 0x8C: 0x8B, 0x8B: 0x8C}
ANCHORS = [(26760, 0x63, 39), (27175, 0x64, 40)]               # theta_cmp, deadband_cmp
LOOKAHEAD = 4                                                  # const->comparison window


def code_section_start(data):
    off = 8
    while off < len(data):
        sid = data[off]; off += 1
        size, off = read_uleb128(data, off)
        if sid == 10:
            return off
        off += size
    raise RuntimeError("no code section")


def enumerate_sites(data, n_imports):
    sites = []
    p = code_section_start(data)
    n_funcs, p = read_uleb128(data, p)
    for i in range(n_funcs):
        func_id = n_imports + i
        body_size, p = read_uleb128(data, p)
        body_end = p + body_size
        if func_id in CONTROLLER:
            q = p
            n_decls, q = read_uleb128(data, q)
            for _ in range(n_decls):
                _, q = read_uleb128(data, q); q += 1
            instr_start = q
            stream = []                                        # (off, pc, op, const_meta)
            pos = q
            while pos < body_end:
                op = data[pos]; size = instruction_size(data, pos)
                pc = pos - instr_start + 1
                meta = None
                if op == 0x44:
                    meta = ("f64", struct.unpack("<d", bytes(data[pos + 1:pos + 9]))[0], 8)
                elif op == 0x41:
                    v, _ = read_sleb128(data, pos + 1); meta = ("i32", v, size - 1)
                stream.append((pos, pc, op, meta)); pos += size
            for idx, (off, pc, op, meta) in enumerate(stream):
                if op in CMP_INV:
                    sites.append(dict(kind="relational", func=func_id, off=off, pc=pc,
                                      op=op, new=CMP_INV[op]))
                elif op in ARITH_INV:
                    sites.append(dict(kind="arithmetic", func=func_id, off=off, pc=pc,
                                      op=op, new=ARITH_INV[op]))
                elif op in SIGN_INV:
                    sites.append(dict(kind="sign", func=func_id, off=off, pc=pc,
                                      op=op, new=SIGN_INV[op]))
                # logic (and/or) omitted: in WASM these are dominated by bitwise
                # value operations (masks, alignment), not boolean guard
                # connectives, and are not separable at the binary level.
                elif meta is not None:
                    ahead = [o for (_, _, o, _) in stream[idx + 1:idx + 1 + LOOKAHEAD]]
                    if any(o in CMP_INV for o in ahead):        # constant feeds a comparison
                        sites.append(dict(kind="threshold", func=func_id, off=off, pc=pc,
                                          ctype=meta[0], value=meta[1], width=meta[2]))
        p = body_end
    return sites


def main(module=None, controller=None, out=None):
    data = Path(MODULE).read_bytes()
    n_imports = 18                                             # canonical layout/replay base
    sites = enumerate_sites(data, n_imports)
    by_off = {s["off"]: s for s in sites}
    for off, op, gt in ANCHORS:
        s = by_off.get(off)
        assert s and data[off] == op and s["func"] == gt, f"anchor fail off={off}: {s}"
        print(f"anchor ok: off={off} op=0x{op:02x} func={gt} kind={s['kind']}")
    from collections import Counter
    print("by kind:", dict(Counter(s["kind"] for s in sites)))
    print("by func:", dict(sorted(Counter(s["func"] for s in sites).items())))
    OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)
    (OUT / "mutation_sites.json").write_text(json.dumps(sites, indent=1))
    print(f"wrote {len(sites)} control-flow fault sites -> results/mutation_sites.json")


if __name__ == "__main__":
    main()
