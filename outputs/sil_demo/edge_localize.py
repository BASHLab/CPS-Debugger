"""Edge-level first-difference localizer (diagnosis-driven fix). Instead of
ranking functions by divergence magnitude (which finds the downstream
consequence), find the FIRST executed control-flow edge that differs between the
nominal and mutant per-tick traces: that edge is the branch the fault flipped,
and it lives in the fault's own function. Re-measure hit@1 on the detected
control faults vs the count-based localizer.
"""
import struct
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

from waxi_host import WaxiHost
from replay import encode_input, load_session, replay, compare, CODE, DATA, VOCAB

MODULE = f"{CODE}/module.wasm"
SITES = json.loads((Path(__file__).parent / "results" / "mutation_sites.json").read_text())


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


def mutant(base, s):
    """Layout-preserving mutation for a control-flow fault site."""
    b = bytearray(base); off = s["off"]
    if s["kind"] in ("relational", "arithmetic", "sign"):
        b[off] = s["new"]; return bytes(b)   # single-byte opcode swap
    if s["kind"] == "threshold":
        if s["ctype"] == "f64":
            c = s["value"]; b[off + 1:off + 9] = struct.pack("<d", 1.0 if c == 0 else c * 2.0)
            return bytes(b)
        c, w = s["value"], s["width"]
        for d in (c * 2, c // 2 if c else None, c + 1, c - 1, c + 10, c + 100, c + 1000):
            if d is None or d == c:
                continue
            enc = _sleb(int(d))
            if len(enc) == w:
                b[off + 1:off + 1 + w] = enc; return bytes(b)
        return None
    return None


# Blame only functions that could plausibly HOST a fault: application code with
# mutable guard/arithmetic sites. Runtime glue (waxi_service_dlr_factory),
# logging (log, log_data, printf_core) and libm (fmin, fmax) can be perturbed BY
# a fault but can never be its location, so naming them is an attribution error,
# not a localization failure. This is the blame-frame filtering used in crash
# triage. The candidate set is derived from the binary (functions carrying
# enumerated sites), not from the fault labels, so it leaks nothing.
CANDIDATE_FUNCS = {s["func"] for s in SITES}


def blame(edges_before, raw_func):
    """Innermost candidate frame: walk back from the divergence to the most
    recent edge in a function that could host a fault."""
    if raw_func in CANDIDATE_FUNCS:
        return raw_func
    for e in reversed(edges_before):
        if e[0] in CANDIDATE_FUNCS:
            return e[0]
    return raw_func


def analyze(module_mut, angle, x):
    """Lockstep nominal vs mutant; return (pred_func, detected) at the FIRST
    differing edge (the flipped branch). Early break => fast. On the control-flow
    corpus, pred_func == fault function is the localization hit."""
    h0 = WaxiHost(MODULE, VOCAB, quiet=True); h0.service_init()
    h1 = WaxiHost(module_mut, VOCAB, quiet=True); h1.service_init()
    for a, xx in zip(angle, x):
        img = encode_input(a, xx)
        h0.write_input(img); e0 = h0.tick(decode=True)
        h1.write_input(img); e1 = h1.tick(decode=True)
        if e0 != e1:
            for i in range(max(len(e0), len(e1))):
                a0 = e0[i] if i < len(e0) else None
                b1 = e1[i] if i < len(e1) else None
                if a0 != b1:
                    raw = (b1 if b1 is not None else a0)[0]
                    return raw, blame(e1[:i], raw), True
    return None, None, False


def _dump(by_kind, rows):
    det = sum(v[0] for v in by_kind.values())
    hit = sum(v[1] for v in by_kind.values())
    hitb = sum(v[2] for v in by_kind.values())
    (Path(__file__).parent / "results" / "edge_localize.json").write_text(
        json.dumps(dict(detected=det, hit1=hit, hit1_blame=hitb,
                        by_kind={k: list(v) for k, v in by_kind.items()},
                        rows=rows), indent=1))


def main():
    angle, x, _ = load_session(sorted(Path(DATA).glob("*.parquet"))[0].as_posix(), 0, 15000)
    base = Path(MODULE).read_bytes()
    ctrl = [s for s in SITES if s["kind"] in ("relational", "threshold",
                                              "arithmetic", "sign")]
    from collections import Counter
    by_kind = defaultdict(lambda: [0, 0, 0])   # kind -> [detected, hit1_raw, hit1_blame]
    rows = []
    for s in ctrl:
        try:
            m = mutant(base, s)
        except Exception:
            continue
        if m is None:
            continue
        p = f"/tmp/el_{s['off']}.wasm"; Path(p).write_bytes(m)
        try:
            pred, pred_blame, detected = analyze(p, angle, x)
        except Exception:
            Path(p).unlink(missing_ok=True); continue
        Path(p).unlink(missing_ok=True)
        if not detected:
            continue
        by_kind[s["kind"]][0] += 1
        by_kind[s["kind"]][1] += int(pred == s["func"])
        by_kind[s["kind"]][2] += int(pred_blame == s["func"])
        rows.append(dict(off=s["off"], kind=s["kind"], gt=s["func"],
                         pred=pred, pred_blame=pred_blame))
        # Checkpoint every site: a replay corpus is hours of compute and a
        # failure in the reporting code must never discard it.
        if len(rows) % 10 == 0:
            _dump(by_kind, rows)
    det = sum(v[0] for v in by_kind.values())
    hit = sum(v[1] for v in by_kind.values())
    hitb = sum(v[2] for v in by_kind.values())
    _dump(by_kind, rows)
    for k, v in sorted(by_kind.items()):
        d, h, hb = v[0], v[1], v[2]
        print(f"{k:12} detected={d:3d}  raw={h/d*100 if d else 0:5.1f}%  "
              f"blame-frame={hb/d*100 if d else 0:5.1f}%")
    print(f"ALL: detected={det}  raw hit@1={hit}/{det}={hit/det*100 if det else 0:.1f}%  "
          f"blame-frame hit@1={hitb}/{det}={hitb/det*100 if det else 0:.1f}%")


if __name__ == "__main__":
    main()
