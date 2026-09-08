"""Reconstruction-driven first-difference localizer (R1.2: close the composition
gap). The 73% hit@1 in edge_localize.py compares two EXACT wasmtime replays
(shipped h0 vs mutant h1); the generator's reconstructed trace never enters it.
Here the NOMINAL reference is the model's contiguous per-tick reconstruction on a
HELD-OUT session (reconstruct_session.py), and the OBSERVED-faulty side is the
exact mutant replay. We localize at the first control-flow edge where observed
diverges from the reconstructed reference, and report the deployable hit@1 next
to the exact-replay upper bound.

Comparison alphabet: the model's 648-edge pruned vocab. Observed (wasmtime) edges
are projected to pruned-token ids (rare out-of-vocab edges dropped, count logged);
h0 and h1 use the identical projection so the exact-replay number is measured in
the same alphabet, isolating the reconstruction tax from the pruning effect.

Deployable termination: the generator has no EOS (it emits max_seq_len real
edges), so each tick's comparison is bounded by the OBSERVED length, which is
legitimately available in twin mode (we observe the deployed machine's trace).
This localizes branch-flip faults (the relational/threshold CWE-682 corpus),
exactly the fault model.

Two localization rules, both scored on the exact-replay-detected fault set:
  * naive     - first tick with a positional token mismatch within observed len.
  * persistent(k) - first func whose first-divergent position holds for k
    consecutive ticks; filters transient single-tick reconstruction errors
    (design decision 3, the deployable mitigation).
"""
import argparse
import json
import os
import pickle
import struct
from collections import Counter, defaultdict
from pathlib import Path
from multiprocessing import Pool

import numpy as np

from waxi_host import WaxiHost
from replay import encode_input, load_session, CODE, DATA, VOCAB
from recon_align import (NormalProfile, first_fault_substitution as _align_first_fault,
                         substitutions as _subs)


def _blame_stack(prefix_toks, raw_func):
    """Innermost frame on the reconstructed call stack at this position.

    Two earlier rules bracket the right answer and neither is it. _blame walks
    back to the nearest candidate frame and stops, so a divergence inside a
    callee is attributed to the dispatcher that invoked it: on the 126 faults
    whose true site is a callee it scores 26.2%. _blame_deep prefers the deepest
    frame ever entered and lifts those to 39.7%, but it excludes the dispatcher
    outright and therefore scores 0.0% on the 72 faults that genuinely live in
    it.

    What both approximate is the innermost frame whose span contains the
    position, which is the top of the call stack there. Reconstruct the stack by
    walking the prefix: a function already on the stack means control returned
    to it, so pop back to it; a function not on the stack means it was called,
    so push. This needs no layout metadata and names the dispatcher exactly when
    the divergence is in the dispatcher's own code."""
    stack = []
    for tok in prefix_toks:
        f = TRIPLES[tok][0] if isinstance(tok, int) else tok[0]
        if f in stack:
            del stack[stack.index(f) + 1:]      # return: unwind to that frame
        else:
            stack.append(f)                      # call: enter a new frame
    if raw_func in stack:
        del stack[stack.index(raw_func) + 1:]
    elif raw_func is not None:
        stack.append(raw_func)
    for f in reversed(stack):
        if f in CANDIDATE_FUNCS:
            return f
    return _blame(prefix_toks, raw_func)


def _blame_deep(prefix_toks, raw_func):
    """Attribute to the innermost candidate function, descending past the scan
    dispatcher.

    _blame walks back to the nearest candidate frame and stops. `tick` is the
    top-level scan entry and is itself a candidate, so a divergence whose edge
    belongs to `tick` is attributed there even when the fault lives in a callee
    the scan invoked. That is the dominant error: on the 126 faults whose true
    site is not `tick`, the rule answers `tick` 59 times. This variant prefers
    the deepest candidate frame the prefix entered and has not returned from,
    falling back to _blame when the prefix carries no nesting information."""
    if raw_func in CANDIDATE_FUNCS and raw_func != DISPATCH_FUNC:
        return raw_func
    seen = []
    for tok in prefix_toks:
        f = TRIPLES[tok][0] if isinstance(tok, int) else tok[0]
        if f in CANDIDATE_FUNCS and f != DISPATCH_FUNC:
            if not seen or seen[-1] != f:
                seen.append(f)
    return seen[-1] if seen else _blame(prefix_toks, raw_func)


def _conf_fault(ref, obs, state, profile, conf, thresh):
    """First substitution the calibration never produced AND where the model
    emitted its own token with probability at least `thresh`.

    A low-confidence position is one the reconstruction was guessing at, so a
    disagreement there is weak evidence about the machine. Requiring confidence
    keeps the positions where the model committed and was contradicted."""
    if conf is None:
        return None
    for a, b, j in _subs(ref, obs):
        if profile.is_known_error(state, a, b):
            continue
        # The reconstruction's confidence is indexed by ITS own position, which
        # after alignment is not j; use the aligned ref index instead.
        k = min(j, len(conf) - 1)
        if float(conf[k]) >= thresh:
            return j
    return None


def _novel_fault(ref, obs, state, profile, succ, use_profile=True):
    """First substitution whose executed transition the controller never made
    during fault-free operation.

    Filtering to known branch sites was the wrong direction and cost 23 of 85
    correct attributions: the successor map is built from NORMAL traces, so a
    predecessor edge looks deterministic exactly when the fault's alternative
    was never taken, and requiring a known decision point discards the novel
    transitions a fault creates. Inverting the test uses that: a transition
    absent from the map is one the controller has not made before, which is
    what a flipped branch produces and what a reconstruction error, drawn from
    the model's fit to normal behaviour, mostly does not."""
    for a, b, j in _subs(ref, obs):
        if use_profile and profile.is_known_error(state, a, b):
            continue
        prev = obs[j - 1] if j > 0 else None
        if prev is None or b not in succ.get(prev, ()):
            return j
    return None


def _decision_fault(ref, obs, state, profile, decision):
    """First substitution that the calibration never produced AND that sits at
    a branch site, i.e. a position whose preceding edge has more than one legal
    successor. Position 0 counts as a decision because the tick's entry edge is
    itself chosen."""
    for a, b, j in _subs(ref, obs):
        if profile.is_known_error(state, a, b):
            continue
        prev = obs[j - 1] if j > 0 else None
        if prev is None or prev in decision:
            return j
    return None

MODULE = f"{CODE}/module.wasm"
HERE = Path(__file__).parent
SITES = json.loads((HERE / "results" / "mutation_sites.json").read_text())
REPO = Path(os.environ.get("CPSD_REPO",
                           Path(__file__).resolve().parents[2]))
PRUNED = json.loads((REPO / "outputs/fm_candidates/train_data_full/pruned_token_mapping.json").read_text())
TRIPLES = [tuple(t) for t in PRUNED["triples"]]                 # id -> (f,fr,to)
TRIPLE_TO_ID = {tuple(t): i for i, t in enumerate(TRIPLES)}     # (f,fr,to) -> id

# module-level state shared with forked workers (set in main before Pool)
_ANGLE = _X = None
_NOMINAL = None        # list[tick] -> tuple(token ids) projected to pruned vocab
_RELIAB = None         # {(state, pos): p_correct} calibrated offline, or None
_RELTHRESH = 0.99      # a position must be at least this reliable to be trusted
_STATE = None          # per-tick controller state of the fault-study session
_PROFILE = None        # NormalProfile: substitutions seen on fault-free operation
_DECISION = None       # edges with more than one legal successor (branch sites)
_SUCC = None           # edge -> set of successors observed in normal operation
_CONF = None           # (n_ticks, seq_len) model probability of each emitted token
_CONFTHRESH = 0.99     # a divergence counts only where the model was this sure
DISPATCH_FUNC = 22     # `tick`, the scan entry point that calls the POUs
_RECON = None          # np.int16 (n_ticks, seq_len); row i == absolute tick (START+i)
_START = _END = 0
_WINDOW = 1000         # replay this many ticks past the exact-divergence tick


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
    # Families must match edge_localize.py exactly. When this handled only
    # relational and threshold, the arithmetic and sign sites returned None and
    # dropped out silently, so the reconstruction-driven number was measured on
    # the two easy families while the exact-replay corpus covered all four.
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


def project(edges):
    """wasmtime edge triples -> pruned-vocab token ids (drop rare OOV edges)."""
    return tuple(TRIPLE_TO_ID[e] for e in edges if e in TRIPLE_TO_ID)


def replay_nominal(n):
    h = WaxiHost(MODULE, VOCAB, quiet=True); h.service_init()
    out = []
    for a, xx in zip(_ANGLE[:n], _X[:n]):
        h.write_input(encode_input(a, xx))
        out.append(project(h.tick(decode=True)))
    return out


# Blame only functions that could HOST a fault. Runtime glue, logging and libm
# can be perturbed by a fault but never be its location, so naming them is an
# attribution error. Derived from the binary (functions carrying enumerated
# sites), not from labels. Same rule as edge_localize.py; on the exact-replay
# corpus it lifted guard faults 75->100% and arithmetic 27->62%.
CANDIDATE_FUNCS = {s["func"] for s in SITES}


def _blame(prefix_toks, raw_func):
    if raw_func in CANDIDATE_FUNCS:
        return raw_func
    for tok in reversed(prefix_toks):
        f = TRIPLES[tok][0] if isinstance(tok, int) else tok[0]
        if f in CANDIDATE_FUNCS:
            return f
    return raw_func


def _first_mismatch_func(ref_toks, obs_toks):
    """Blamed func of the first positional token mismatch within observed length,
    or None if observed is a prefix-match of ref (no flip). ref may be longer
    (recon=200).

    The blame frame is applied HERE so that every localization rule below (exact
    replay, naive recon, persistent-k, excess) shares one attribution rule and
    the rows differ only in their divergence criterion. Scoring the exact-replay
    upper bound on raw first-mismatch while the deployable rules got the blame
    frame made the upper bound the weaker measurement, and the deployable path
    appeared to beat it.
    """
    L = len(obs_toks)
    for i in range(L):
        r = ref_toks[i] if i < len(ref_toks) else None
        if r != obs_toks[i]:
            tok = obs_toks[i] if obs_toks[i] is not None else r
            return _blame(obs_toks[:i], TRIPLES[tok][0])
    # observed fully matched the ref prefix; if ref (true-length h0) is shorter
    # than observed, the fault ADDED edges -> divergence at the tail.
    if len(ref_toks) < L:
        j = len(ref_toks)
        return _blame(obs_toks[:j], TRIPLES[obs_toks[j]][0])
    return None


def _analyze_site(s):
    # Worker-level guard: a mutant can trap inside wasmtime mid-replay; if that
    # exception propagated, its traceback would hold unpicklable wasmtime handles
    # and crash the whole pool. Catch everything and return a plain error dict.
    try:
        return _analyze_site_inner(s)
    except Exception as ex:
        return dict(off=s.get("off"), kind=s.get("kind"), gt=s.get("func"),
                    detected=False, error=f"{type(ex).__name__}: {ex}")


def _analyze_site_inner(s):
    try:
        m = mutant(bytes(BASE), s)
    except Exception:
        return None
    if m is None:
        return None
    p = f"/tmp/elr_{s['off']}_{s['kind']}.wasm"; Path(p).write_bytes(m)
    try:
        h1 = WaxiHost(p, VOCAB, quiet=True); h1.service_init()
    except Exception:
        Path(p).unlink(missing_ok=True); return None

    exact_div_tick = None
    exact_pred = None
    # per-tick recon first-divergent func over the replayed window (for persistent)
    recon_div_func = {}           # tick -> func (or None): first recon-vs-obs mismatch
    excess_func = {}              # tick -> func (or None): first FAULT-attributable
    calib_func = {}               # tick -> func: same, but offline-calibrated
    align_func = {}               # tick -> func: alignment + normal-sub profile
    dec_func = {}                 # tick -> func: the above, decision sites only
    conf_func = {}                # tick -> func: confidence-gated substitution
    deep_func = {}                # tick -> func: alignment + innermost-callee
    stack_func = {}               # tick -> func: alignment + call-stack top
    novel_func = {}               # tick -> func: unseen transition + profile
    novel_only_func = {}          # tick -> func: unseen transition alone
    n = min(_END - _START, len(_ANGLE))
    stop = n
    for t in range(n):
        h1.write_input(encode_input(_ANGLE[t], _X[t]))
        obs = project(h1.tick(decode=True))
        nom = _NOMINAL[t]
        # exact-replay localization (nominal = shipped-binary replay)
        if exact_div_tick is None:
            f = _first_mismatch_func(nom, obs)
            if f is not None:
                exact_div_tick = t; exact_pred = f
                stop = min(n, t + _WINDOW)      # bound the rest of the replay
        # recon-driven per-tick divergence (nominal reference = reconstruction)
        rec = [int(x) for x in _RECON[t]] if t < _RECON.shape[0] else None
        if rec is not None:
            recon_div_func[t] = _first_mismatch_func(rec, obs)
            # baseline-cancelled fault attribution: first position where the
            # mutant diverges from recon AND recon matched the nominal there
            # (so it is the fault, not a reconstruction error). Deployable: the
            # recon-vs-nominal baseline is calibrated offline on normal data.
            L = min(len(obs), len(rec), len(nom))
            ef = None
            for i in range(L):
                if rec[i] == nom[i] and rec[i] != obs[i]:
                    ef = _blame(obs[:i], TRIPLES[obs[i]][0]); break
            excess_func[t] = ef
            # Deployable variant: same idea, but the decision of whether a
            # position is trustworthy comes from a map calibrated offline on a
            # DIFFERENT normal session, not from the shipped binary's trace on
            # this tick. Nothing here reads `nom`.
            # Alignment-based deployable rule. Positional comparison is
            # confounded: on normal operation the reconstruction agrees
            # positionally only 53% of the time while its edit distance is
            # 0.058, because a skipped block shifts every later index. Aligning
            # first turns those into indels and leaves a flipped branch as a
            # substitution. A substitution the calibration also produces on
            # fault-free data is a reconstruction error, so only an unseen one
            # is attributed. Reads no shipped-binary trace.
            if _PROFILE is not None:
                st_ = int(_STATE[t]) if _STATE is not None and t < len(_STATE) else -1
                ref_ = rec[:len(obs)]
                jj = _align_first_fault(ref_, list(obs), st_, _PROFILE)
                align_func[t] = (_blame(obs[:jj], TRIPLES[obs[jj]][0])
                                 if jj is not None else None)
            # Branch-decision filter. A control-flow fault flips a BRANCH, so
            # the edge it changes must leave a site with more than one legal
            # successor. Most edges have exactly one (137 of 190 in the
            # empirical successor map), so a substitution at a deterministic
            # position cannot be the flipped branch and is either a
            # reconstruction error or a downstream consequence of one. Filtering
            # to decision positions applies the paper's own branch-decision
            # framing to attribution.
            # Confidence-gated attribution. The residual-diagnosis literature
            # states the problem directly: where the model is inaccurate, a
            # residual cannot be told from a fault, and the remedy is to weight
            # the residual by the model's own uncertainty. Here that means
            # attributing a substitution only where the reconstruction was
            # confident, so a disagreement is evidence about the machine rather
            # than about the model.
            if _PROFILE is not None:
                st5 = int(_STATE[t]) if _STATE is not None and t < len(_STATE) else -1
                ref5 = rec[:len(obs)]
                dj2 = _align_first_fault(ref5, list(obs), st5, _PROFILE)
                deep_func[t] = (_blame_deep(obs[:dj2], TRIPLES[obs[dj2]][0])
                                if dj2 is not None else None)
                stack_func[t] = (_blame_stack(obs[:dj2], TRIPLES[obs[dj2]][0])
                                 if dj2 is not None else None)
            if _PROFILE is not None and _CONF is not None:
                st4 = int(_STATE[t]) if _STATE is not None and t < len(_STATE) else -1
                ref4 = rec[:len(obs)]
                cj = _conf_fault(ref4, list(obs), st4, _PROFILE,
                                 _CONF[t] if t < len(_CONF) else None, _CONFTHRESH)
                conf_func[t] = (_blame(obs[:cj], TRIPLES[obs[cj]][0])
                                if cj is not None else None)
            if _PROFILE is not None and _SUCC is not None:
                st3 = int(_STATE[t]) if _STATE is not None and t < len(_STATE) else -1
                ref3 = rec[:len(obs)]
                nj = _novel_fault(ref3, list(obs), st3, _PROFILE, _SUCC)
                novel_func[t] = (_blame(obs[:nj], TRIPLES[obs[nj]][0])
                                 if nj is not None else None)
                nj2 = _novel_fault(ref3, list(obs), st3, _PROFILE, _SUCC, False)
                novel_only_func[t] = (_blame(obs[:nj2], TRIPLES[obs[nj2]][0])
                                      if nj2 is not None else None)
            if _PROFILE is not None and _DECISION is not None:
                st2 = int(_STATE[t]) if _STATE is not None and t < len(_STATE) else -1
                ref2 = rec[:len(obs)]
                dj = _decision_fault(ref2, list(obs), st2, _PROFILE, _DECISION)
                dec_func[t] = (_blame(obs[:dj], TRIPLES[obs[dj]][0])
                               if dj is not None else None)
            if _RELIAB is not None:
                st = int(_STATE[t]) if _STATE is not None and t < len(_STATE) else -1
                cf = None
                Lc = min(len(obs), len(rec))
                for i in range(Lc):
                    if rec[i] == obs[i]:
                        continue
                    if _RELIAB.get((st, i), 0.0) >= _RELTHRESH:
                        cf = _blame(obs[:i], TRIPLES[obs[i]][0]); break
                calib_func[t] = cf
        if exact_div_tick is not None and t >= stop:
            break
    Path(p).unlink(missing_ok=True)

    if exact_div_tick is None:
        return dict(off=s["off"], kind=s["kind"], gt=s["func"], detected=False)

    # naive recon: first tick (>= exact detection region start) with a mismatch
    recon_naive = None
    for t in sorted(recon_div_func):
        if recon_div_func[t] is not None:
            recon_naive = recon_div_func[t]; break
    # persistent(k): first func holding for k consecutive ticks
    def persistent(k):
        run_f, run_n = None, 0
        for t in sorted(recon_div_func):
            f = recon_div_func[t]
            if f is not None and f == run_f:
                run_n += 1
            else:
                run_f, run_n = f, (1 if f is not None else 0)
            if run_f is not None and run_n >= k:
                return run_f
        return None
    # longest-run: the func with the longest consecutive divergence run over the
    # window (the fault diverges every balance tick; recon noise is transient).
    longest = defaultdict(int); run_f, run_n = None, 0
    for t in sorted(recon_div_func):
        f = recon_div_func[t]
        if f is not None and f == run_f:
            run_n += 1
        else:
            run_f, run_n = f, (1 if f is not None else 0)
        if run_f is not None:
            longest[run_f] = max(longest[run_f], run_n)
    recon_longrun = max(longest, key=longest.get) if longest else None
    # mode: most frequent first-divergent func over ticks where recon diverges.
    votes = Counter(f for f in recon_div_func.values() if f is not None)
    recon_mode = votes.most_common(1)[0][0] if votes else None
    # DEPLOYABLE: baseline-cancelled fault attribution. Mode and longest-run of
    # the per-tick first fault-attributable func (recon noise cancels because it
    # also appears against the nominal baseline).
    ex_votes = Counter(f for f in excess_func.values() if f is not None)
    recon_excess = ex_votes.most_common(1)[0][0] if ex_votes else None
    ex_long = defaultdict(int); rf, rn = None, 0
    for t in sorted(excess_func):
        f = excess_func[t]
        if f is not None and f == rf:
            rn += 1
        else:
            rf, rn = f, (1 if f is not None else 0)
        if rf is not None:
            ex_long[rf] = max(ex_long[rf], rn)
    recon_excess_longrun = max(ex_long, key=ex_long.get) if ex_long else None
    n_excess = sum(1 for f in excess_func.values() if f is not None)
    cal_votes = Counter(f for f in calib_func.values() if f is not None)
    recon_calib = cal_votes.most_common(1)[0][0] if cal_votes else None
    # Spectrum-based aggregation. Taking the mode over ticks treats every tick
    # as an equal vote, but a reconstruction error is spread thinly over many
    # functions while a fault is concentrated in one. Ochiai is the standard
    # suspiciousness metric for exactly that asymmetry: a function scores high
    # when it is implicated in many divergent ticks AND appears in few others.
    def _ochiai(votes):
        if not votes:
            return None
        total = sum(votes.values())
        best, bs = None, -1.0
        for f, ef in votes.items():
            # ef: ticks implicating f. ep: divergent ticks implicating something
            # else. Ochiai = ef / sqrt((ef + nf) * (ef + ep)) with nf = 0 here,
            # since every scored tick that diverges implicates exactly one func.
            ep = total - ef
            sc = ef / ((ef * (ef + ep)) ** 0.5) if ef else 0.0
            if sc > bs:
                best, bs = f, sc
        return best

    sk_votes = Counter(f for f in stack_func.values() if f is not None)
    recon_stack = sk_votes.most_common(1)[0][0] if sk_votes else None
    dp_votes = Counter(f for f in deep_func.values() if f is not None)
    recon_deep = dp_votes.most_common(1)[0][0] if dp_votes else None
    cf_votes = Counter(f for f in conf_func.values() if f is not None)
    recon_conf = cf_votes.most_common(1)[0][0] if cf_votes else None
    al_votes_ = Counter(f for f in align_func.values() if f is not None)
    recon_align_ochiai = _ochiai(al_votes_)
    recon_conf_ochiai = _ochiai(cf_votes)
    nv_votes = Counter(f for f in novel_func.values() if f is not None)
    recon_novel = nv_votes.most_common(1)[0][0] if nv_votes else None
    no_votes = Counter(f for f in novel_only_func.values() if f is not None)
    recon_novel_only = no_votes.most_common(1)[0][0] if no_votes else None
    dc_votes = Counter(f for f in dec_func.values() if f is not None)
    recon_decision = dc_votes.most_common(1)[0][0] if dc_votes else None
    al_votes = Counter(f for f in align_func.values() if f is not None)
    recon_align = al_votes.most_common(1)[0][0] if al_votes else None
    recon_align_first = next((align_func[t] for t in sorted(align_func)
                              if align_func[t] is not None), None)
    return dict(off=s["off"], kind=s["kind"], gt=s["func"], detected=True,
                exact_pred=exact_pred, exact_tick=exact_div_tick,
                recon_naive=recon_naive,
                recon_k3=persistent(3), recon_k5=persistent(5),
                recon_k10=persistent(10),
                recon_longrun=recon_longrun, recon_mode=recon_mode,
                recon_excess=recon_excess, recon_excess_longrun=recon_excess_longrun,
                recon_calib=recon_calib, recon_align=recon_align,
                recon_align_first=recon_align_first,
                recon_decision=recon_decision, recon_novel=recon_novel,
                recon_novel_only=recon_novel_only, recon_conf=recon_conf,
                recon_align_ochiai=recon_align_ochiai,
                recon_conf_ochiai=recon_conf_ochiai, recon_deep=recon_deep,
                # Vote distribution over functions, not just its argmax. Taking
                # the mode discards how far ahead the winner was, which is the
                # only thing that separates a confident call from a coin flip
                # between a caller and its callee. Saved so margin rules and
                # set-valued answers can be computed without re-replaying.
                align_votes={str(k): v for k, v in al_votes_.items()},
                deep_votes={str(k): v for k, v in dp_votes.items()},
                stack_votes={str(k): v for k, v in sk_votes.items()},
                recon_stack=recon_stack,
                n_excess_ticks=n_excess)


BASE = None


def main():
    global _ANGLE, _X, _NOMINAL, _RECON, _START, _END, BASE, _WINDOW
    global _RELIAB, _RELTHRESH, _STATE, _PROFILE, _DECISION, _SUCC
    global _CONF, _CONFTHRESH
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", default="../fm_candidates/results/recon_localize/"
                    "recon_2025-03-25_13-39-06_0_16000.npy")
    ap.add_argument("--session", default="2025-03-25_13-39-06")
    ap.add_argument("--nominal-cache", default="results/nominal_recon_cache.pkl")
    ap.add_argument("--cache-nominal-only", action="store_true")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--window", type=int, default=1000)
    ap.add_argument("--out", default="results/edge_localize_recon.json")
    ap.add_argument("--reliability", default="results/recon_reliability.json",
                    help="frozen offline reliability map; enables the "
                         "deployable calibrated rule")
    ap.add_argument("--rel-thresh", type=float, default=0.99)
    ap.add_argument("--normal-subs", default="results/normal_subs.json",
                    help="frozen normal-operation substitution profile")
    ap.add_argument("--sub-min-count", type=int, default=1)
    ap.add_argument("--conf", default=None,
                    help="per-token confidence array saved alongside the "
                         "reconstruction (recon_*.conf.npy)")
    ap.add_argument("--conf-thresh", type=float, default=0.99)
    ap.add_argument("--succ-map",
                    default="../fm_candidates/results/cfg_succ_fold0.json")
    ap.add_argument("--summary-only", action="store_true",
                    help="recompute the paper summary from an existing --out "
                         "without re-replaying every site")
    args = ap.parse_args()
    _WINDOW = args.window

    if args.summary_only:
        prev = json.loads((HERE / args.out).read_text())
        _write_summary([r for r in prev["rows"] if r.get("detected")],
                       prev["rows"], args, *prev.get("window", (0, 16000)))
        return

    meta_path = Path(args.recon).with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text()); _START, _END = meta["start"], meta["end"]
    else:
        _START, _END = 0, 16000
    session_path = f"{DATA}/{args.session}.parquet"
    _ANGLE, _X, _ = load_session(session_path, _START, _END - _START)
    BASE = Path(MODULE).read_bytes()

    cache = HERE / args.nominal_cache
    if cache.exists() and not args.cache_nominal_only:
        _NOMINAL = pickle.loads(cache.read_bytes())
        print(f"loaded nominal cache: {len(_NOMINAL)} ticks")
    else:
        print(f"replaying nominal over [{_START},{_END})...", flush=True)
        _NOMINAL = replay_nominal(_END - _START)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps(_NOMINAL))
        print(f"cached nominal -> {cache} ({len(_NOMINAL)} ticks)")
    if args.cache_nominal_only:
        return

    relp = HERE / args.reliability
    if relp.exists():
        raw = json.loads(relp.read_text())
        assert raw["session"] != args.session, (
            f"reliability map was calibrated on the fault-study session "
            f"{args.session}; that leaks the test session into the rule")
        _RELTHRESH = args.rel_thresh
        _RELIAB = {}
        for k, (ok_, tot_) in raw["map"].items():
            st, pos = k.split("|")
            # Require real support before trusting a cell; a position seen a
            # handful of times cannot establish reliability.
            if tot_ >= 30:
                _RELIAB[(int(st), int(pos))] = ok_ / tot_
        import pandas as _pd
        _STATE = _pd.read_parquet(session_path, columns=["pendulum_state"]) \
            ["pendulum_state"].to_numpy()[_START:_END].astype(int)
        print(f"loaded reliability map from {raw['session']}: "
              f"{len(_RELIAB)} cells with support, thresh {_RELTHRESH}")
    else:
        print("no reliability map; calibrated rule disabled")

    nsp = HERE / args.normal_subs
    if nsp.exists():
        raw = json.loads(nsp.read_text())
        assert raw["session"] != args.session, (
            "normal-substitution profile was calibrated on the fault-study "
            "session; that leaks the test session into the rule")
        _PROFILE = NormalProfile.from_json(raw["counts"], args.sub_min_count)
        import pandas as _pd
        if _STATE is None:
            _STATE = _pd.read_parquet(session_path, columns=["pendulum_state"]) \
                ["pendulum_state"].to_numpy()[_START:_END].astype(int)
        print(f"loaded normal-sub profile from {raw['session']}: "
              f"{raw['distinct']} distinct substitutions over {raw['ticks']} ticks")
    else:
        print("no normal-sub profile; alignment rule disabled")

    smp = HERE / args.succ_map
    if smp.exists():
        sm = json.loads(smp.read_text())["succ"]
        _DECISION = {int(k) for k, v in sm.items() if len(v) > 1}
        _SUCC = {int(k): set(int(x) for x in v) for k, v in sm.items()}
        print(f"decision sites: {len(_DECISION)} of {len(sm)} edges")
    else:
        print("no successor map; decision filter disabled")

    cpath = args.conf or str(Path(args.recon).with_suffix("")) + ".conf.npy"
    if Path(cpath).exists():
        _CONF = np.load(cpath).astype(np.float32)
        _CONFTHRESH = args.conf_thresh
        print(f"loaded confidence {_CONF.shape}, threshold {_CONFTHRESH}")
    else:
        print(f"no confidence array at {cpath}; confidence rule disabled")

    _RECON = np.load(args.recon)                     # (n_ticks, seq_len)
    print(f"loaded recon {_RECON.shape}; sites={len(SITES)}; procs={args.procs}")

    with Pool(args.procs) as pool:
        rows = [r for r in pool.map(_analyze_site, SITES) if r is not None]

    det = [r for r in rows if r["detected"]]
    n_det = len(det)
    def hit(key):
        return sum(1 for r in det if r.get(key) == r["gt"])
    summary = dict(
        n_sites=len(rows), n_detected=n_det,
        exact_hit1=hit("exact_pred"),
        recon_naive_hit1=hit("recon_naive"),
        recon_k3_hit1=hit("recon_k3"),
        recon_k5_hit1=hit("recon_k5"),
        recon_k10_hit1=hit("recon_k10"),
        recon_longrun_hit1=hit("recon_longrun"),
        recon_mode_hit1=hit("recon_mode"),
        recon_excess_hit1=hit("recon_excess"),
        recon_excess_longrun_hit1=hit("recon_excess_longrun"),
        recon_calib_hit1=hit("recon_calib"),
        recon_align_hit1=hit("recon_align"),
        recon_align_first_hit1=hit("recon_align_first"),
        recon_decision_hit1=hit("recon_decision"),
        recon_novel_hit1=hit("recon_novel"),
        recon_novel_only_hit1=hit("recon_novel_only"),
        recon_conf_hit1=hit("recon_conf"),
        recon_align_ochiai_hit1=hit("recon_align_ochiai"),
        recon_conf_ochiai_hit1=hit("recon_conf_ochiai"),
        recon_deep_hit1=hit("recon_deep"),
        recon_stack_hit1=hit("recon_stack"),
    )
    (HERE / args.out).write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))
    print("\n=== R1.2 reconstruction-driven localization (held-out session) ===")
    print(f"detected control faults: {n_det}")
    for k in ("exact_pred", "recon_naive", "recon_k3", "recon_k5", "recon_k10",
              "recon_longrun", "recon_mode", "recon_excess",
              "recon_excess_longrun", "recon_calib", "recon_align",
              "recon_align_first", "recon_decision", "recon_novel",
              "recon_novel_only", "recon_conf", "recon_align_ochiai",
              "recon_conf_ochiai", "recon_deep", "recon_stack"):
        h = hit(k)
        print(f"  {k:12} hit@1 = {h}/{n_det} = {100*h/n_det if n_det else 0:.1f}%")

    _write_summary(det, rows, args, _START, _END)


def _write_summary(det, rows, args, start, end):
    """Emit the paper-facing summary from the run that produced the rows.

    This file used to be maintained by hand, so a corpus change could move every
    number in it while the paper kept quoting the old ones.
    """
    n_det = len(det)
    rng = np.random.default_rng(20260731)
    ok = {k: np.array([r.get(k) == r["gt"] for r in det], dtype=bool)
          for k in ("exact_pred", "recon_naive", "recon_excess", "recon_align",
                    "recon_calib", "recon_decision", "recon_novel",
                    "recon_novel_only", "recon_conf", "recon_align_ochiai",
                    "recon_conf_ochiai", "recon_deep", "recon_stack")}

    def stat(key):
        v = ok[key]
        n = len(v)
        if n == 0:
            return dict(hit=0, n=0, pct=0.0, ci_lo=0.0, ci_hi=0.0)
        boot = [v[rng.integers(0, n, n)].mean() * 100 for _ in range(10000)]
        return dict(hit=int(v.sum()), n=n, pct=round(100 * v.mean(), 1),
                    ci_lo=round(float(np.percentile(boot, 2.5)), 1),
                    ci_hi=round(float(np.percentile(boot, 97.5)), 1))

    by_kind = {}
    for k in sorted({r["kind"] for r in det}):
        sub = [r for r in det if r["kind"] == k]
        by_kind[k] = {
            "exact": dict(hit=sum(r.get("exact_pred") == r["gt"] for r in sub),
                          n=len(sub),
                          pct=round(100 * sum(r.get("exact_pred") == r["gt"]
                                              for r in sub) / len(sub), 1)),
            # Deployable arm, per family. Uses the alignment rule, NOT
            # recon_excess: that rule decides whether a divergence is the fault
            # by checking the reconstruction against a replay of the shipped
            # binary on the same tick, which is the artifact the reconstruction
            # exists to replace. See the note on recon_deployable below.
            "recon": dict(hit=sum(r.get("recon_align") == r["gt"] for r in sub),
                          n=len(sub),
                          pct=round(100 * sum(r.get("recon_align") == r["gt"]
                                              for r in sub) / len(sub), 1)),
        }
    comp = dict(session=args.session, held_out=True, window=[start, end],
                n_sites=len(rows), n_detected=n_det,
                exact_replay=stat("exact_pred"),
                recon_naive=stat("recon_naive"),
                # The deployable number must not consult the shipped binary.
                # recon_excess attributes a divergence by testing rec[i] ==
                # nom[i], where nom is a wasmtime replay of the shipped binary
                # on that tick; a site holding that replay can localize by exact
                # replay and needs no reconstruction. Implementing the same idea
                # honestly, as a reliability map calibrated offline on a
                # different normal session, reproduces the naive number (37.4%),
                # which shows the 60.1% came from the test-time oracle rather
                # than from calibration. recon_align is the deployable rule:
                # align the reconstruction to the executed trace, then attribute
                # only substitutions that fault-free calibration never produced.
                recon_deployable=stat("recon_align"),
                recon_oracle_gated=stat("recon_excess"),
                recon_calibrated=stat("recon_calib"),
                recon_decision=stat("recon_decision"),
                recon_conf=stat("recon_conf"),
                recon_align_ochiai=stat("recon_align_ochiai"),
                recon_conf_ochiai=stat("recon_conf_ochiai"),
                recon_deep=stat("recon_deep"),
                recon_stack=stat("recon_stack"),
                by_kind=by_kind)
    (HERE / "results" / "r1_2_composition_summary.json").write_text(
        json.dumps(comp, indent=1))
    print("wrote results/r1_2_composition_summary.json")


if __name__ == "__main__":
    main()
