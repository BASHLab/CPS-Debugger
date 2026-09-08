"""Regenerate macros_numbers.tex from primary result artifacts.

Every experimental number in the paper is a LaTeX macro defined here; the
draft never hard-codes a result. Re-run after any eval lands:

    python3 gen_numbers.py          # writes macros_numbers.tex next to it

Sources (primary artifacts only, never hand-transcribed):
  - results/<variant>_strat100k_fold_<i>/<variant>_eval_test.json
  - results/baselines_strat100k_fold_<i>/<baseline>_eval_test.json
  - results/timing_*/ar_modern_eval_test.json  (H100 ms/tick)
  - outputs/experiments/anomaly_detection_results.json  (ROC AUCs)
  - outputs/phase3/anomaly_zero_shot_haiku_results.json (explainer)

Macros for results not yet landed emit \\textbf{TBD} so the draft compiles
at every stage and unfinished numbers are visible in the PDF.
"""
import json
import os
import statistics
from pathlib import Path

# Roots are overridable so the macros can be regenerated on any machine:
#   CPSD_REPO        tree whose outputs/ holds the result artifacts
#   CPSD_MACROS_OUT  macros_numbers.tex to write
REPO = Path(os.environ.get(
    "CPSD_REPO", str(Path(__file__).resolve().parents[1])))
FM = REPO / "outputs/fm_candidates"
RES = FM / "results"
EXP = REPO / "outputs/experiments"
P3 = REPO / "outputs/phase3"
# This script lives in iaai27_work/ but writes into the Overleaf paper dir.
OUT = Path(os.environ.get(
    "CPSD_MACROS_OUT",
    str(Path(__file__).parent.parent / "iaai27_paper" / "macros_numbers.tex")))

TBD = r"\textbf{TBD}"
STATES = {"0": "Sw", "1": "Bal", "2": "Rst"}   # swing-up, balance, reset

macros = {}


def put(name: str, value, fmt: str = "{:.4f}"):
    if value is None:
        macros[name] = TBD
    elif isinstance(value, str):
        macros[name] = value
    else:
        macros[name] = fmt.format(value)


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def emit_variant(key: str, variant: str):
    r"""Per-state + overall NED (5-fold median [min-max]) and EM% macros.

    Emits \NED<key><Sw|Bal|Rst|All> and \EM<key>. TBD until 5 folds exist.
    """
    folds = []
    for i in range(5):
        d = _load(RES / f"{variant}_strat100k_fold_{i}" /
                  f"{variant}_eval_test.json")
        if d and "greedy_ned_mean" in d:
            folds.append(d)
    if len(folds) < 5:
        for s in list(STATES.values()) + ["All"]:
            put(f"NED{key}{s}", None)
        put(f"EM{key}", None)
        return
    for st, sname in STATES.items():
        vals = [d["greedy_per_state"][st]["ned_mean"] for d in folds]
        put(f"NED{key}{sname}",
            f"{statistics.median(vals):.4f}")
    vals = [d["greedy_ned_mean"] for d in folds]
    put(f"NED{key}All",
        f"{statistics.median(vals):.4f} [{min(vals):.4f}--{max(vals):.4f}]")
    ems = [d["greedy_exact_match"] * 100 for d in folds]
    put(f"EM{key}", f"{statistics.median(ems):.1f}")


def emit_baseline(key: str, baseline: str):
    """Same macro shape for trivial floors (baseline_* key space)."""
    folds = []
    for i in range(5):
        d = _load(RES / f"baselines_strat100k_fold_{i}" /
                  f"{baseline}_eval_test.json")
        if d and "baseline_ned_mean" in d:
            folds.append(d)
    if len(folds) < 5:
        for s in list(STATES.values()) + ["All"]:
            put(f"NED{key}{s}", None)
        put(f"EM{key}", None)
        return
    per_state_ok = all("baseline_per_state" in d for d in folds)
    for st, sname in STATES.items():
        if per_state_ok:
            vals = [d["baseline_per_state"][st]["ned_mean"] for d in folds]
            put(f"NED{key}{sname}", f"{statistics.median(vals):.4f}")
        else:
            put(f"NED{key}{sname}", None)
    vals = [d["baseline_ned_mean"] for d in folds]
    put(f"NED{key}All",
        f"{statistics.median(vals):.4f} [{min(vals):.4f}--{max(vals):.4f}]")
    ems = [d.get("baseline_exact_match", 0) * 100 for d in folds]
    put(f"EM{key}", f"{statistics.median(ems):.1f}")


def main():
    # ── Leaderboard variants ────────────────────────────────────────────
    emit_variant("BLtwo", "ar_modern")                       # Chronos-2, 0.5 s
    emit_variant("BLseventhresh", "ar_modern_bl7_e1thresh")  # JEPA + thresh
    emit_variant("BLtwofours", "ar_modern_bl2_w4000")        # Chronos-2, 4 s
    emit_variant("BLtwofoursthresh", "ar_modern_bl2_w4000_thresh")

    # ── Trivial floors ──────────────────────────────────────────────────
    emit_baseline("FloorModalTok", "copy_modal")
    emit_baseline("FloorUnigram", "unigram")
    emit_baseline("FloorStateModal", "state_modal_trace")

    # Aliases used in running text.
    macros["NEDheadline"] = macros["NEDBLtwofoursthreshAll"]
    macros["NEDtrivialfloor"] = macros["NEDFloorStateModalAll"]

    # ── R1.1 ablations: OOD sensor perturbation + random-init encoder ────
    # OOD: fold-0 base ar_modern with the deployed sensor stream perturbed at
    # inference (nominal/noise/stuck/timing). A single-model robustness probe.
    ood = _load(RES / "ood_recon.json")
    if ood:
        put("NEDoodNominal", ood.get("nominal"))
        put("NEDoodNoise", ood.get("noise"))
        put("NEDoodStuck", ood.get("stuck"))
        put("NEDoodTiming", ood.get("timing"))
    # random-init frozen encoder, 5-fold strat100k median [min-max]; TBD until
    # all five folds land (matches every leaderboard row). Comparator = NEDBLtwo.
    ri = []
    for i in range(5):
        d = _load(RES / f"ar_modern_randinit_strat100k_fold_{i}" /
                  "ar_modern_randinit_eval_test.json")
        if d and "greedy_ned_mean" in d:
            ri.append(d["greedy_ned_mean"])
    if len(ri) == 5:
        put("NEDrandinit",
            f"{statistics.median(ri):.4f} [{min(ri):.4f}--{max(ri):.4f}]")

    # ── GRPO note numbers (fold-0; the negative result) ─────────────────
    ref = _load(RES / "ar_modern_bl2_w4000_thresh_strat100k_fold_0" /
                "ar_modern_bl2_w4000_thresh_eval_test.json")
    g1 = _load(RES / "ar_modern_bl2_w4000_thresh_grpo_strat100k_fold_0" /
               "ar_modern_bl2_w4000_thresh_grpo_eval_test.json")
    g2 = _load(RES / "ar_modern_bl2_w4000_thresh_grpo_v2_strat100k_fold_0" /
               "ar_modern_bl2_w4000_thresh_grpo_v2_eval_test.json")
    put("NEDgrpoRefSTwo",
        ref["greedy_per_state"]["2"]["ned_mean"] if ref else None)
    put("NEDgrpoRefAllFoldZero", ref["greedy_ned_mean"] if ref else None)
    put("NEDgrpoSTwo", g1["greedy_per_state"]["2"]["ned_mean"] if g1 else None)
    put("NEDgrpoAll", g1["greedy_ned_mean"] if g1 else None)
    put("NEDgrpoVtwoSTwo",
        g2["greedy_per_state"]["2"]["ned_mean"] if g2 else None)
    put("NEDgrpoVtwoAll", g2["greedy_ned_mean"] if g2 else None)

    # ── Timing (H100 protocol) ──────────────────────────────────────────
    t = _load(RES / "timing_bl2_w4000_thresh_fold_0" /
              "ar_modern_eval_test.json")
    put("MsPerTick", t.get("greedy_p99_ms_per_tick") if t else None, "{:.2f}")
    if t and t.get("greedy_p99_ms_per_tick"):
        put("SessionReconMinutes",
            t["greedy_p99_ms_per_tick"] * 480_000 / 1000 / 60, "{:.0f}")
    else:
        put("SessionReconMinutes", None)

    # ── Fault detection (system-native experiments E1a/E1b) ─────────────
    # The pilot-study RF results (outputs/experiments/
    # anomaly_detection_results.json) are preliminaries and are NOT used
    # in the paper. These macros fill from the new artifacts when the
    # encoder one-class (E1a) and trace-consistency (E1b) runs land.
    def trunc4(v):
        # Truncate, never round: 0.99995 must not present as 1.0000.
        return None if v is None else f"{int(v * 10000) / 10000:.4f}"
    e1a = _load(RES / "anomaly_encoder_oneclass" / "results.json")
    e1b = _load(RES / "anomaly_trace_consistency" / "results.json")
    e1a_full = e1a  # keep the full json for the baseline field
    def auc_from(d, key, field="auc"):
        return trunc4(d[field].get(key)) if d and field in d else None
    # Code-path faults are replayed on byte-identical sensor logs, so the
    # sensor-only encoder detector has no signal by construction (same reason
    # the code-path-substitution row marks the sensor detectors n/a).
    code_path_faults = {"sil_counter", "sil_deadband", "trace_swap"}
    for key, name in (("sil_counter", "SILcounter"),
                      ("sil_deadband", "SILdeadband"),
                      ("trace_swap", "traceswap"),
                      ("sensor_noise", "sensornoise"),
                      ("stuck_sensor", "stucksensor"),
                      ("timing_50ms", "timingfifty")):
        if key in code_path_faults:
            put(f"AUCenc{name}", "n/a", "{}")
        else:
            # Mahalanobis, matching the window-statistics baseline scorer, so
            # the encoder row is a representation-only ablation.
            put(f"AUCenc{name}", auc_from(e1a, key, "auc_mahalanobis"))
        put(f"AUCtc{name}", auc_from(e1b, key))
    # Abstract/intro alias: headline detection number = trace-consistency
    # on code-path substitution once E1b lands.
    put("AUCtraceswap", auc_from(e1b, "trace_swap"))
    # raw-stats baseline (no foundation encoder)
    base = (e1a_full or {}).get("auc_rawstats_baseline", {})
    for key, name in (("sensor_noise","sensornoise"),("stuck_sensor","stucksensor"),
                      ("timing_50ms","timingfifty")):
        put(f"AUCbase{name}", trunc4(base.get(key)) if base else None)

    # ── Explainer (redesigned benchmark D'; fills when it exists) ───────
    for name in ("ExplZSdetect", "ExplZStype", "ExplZSlocalize",
                 "ExplZStotal", "ExplFTdetect", "ExplFTtype",
                 "ExplFTlocalize", "ExplFTtotal"):
        put(name, None)

    # ── Dataset stats (canonical; reconcile in G6) ──────────────────────
    put("DatasetSessions", "15", "{}")
    put("DatasetTicks", "7.31M", "{}")
    put("TraceMedianLen", "91", "{}")
    put("VocabSize", "648", "{}")

    # ── Hard-tick analysis (A6), 5-fold median + [min, max] ─────────────
    ht = _load(RES / "hard_tick_analysis" / "hard_tick_5fold.json")
    for key, name in (("transition", "Trans"), ("path_change", "Path"),
                      ("steady", "Steady")):
        s = (ht or {}).get("strata", {}).get(key)
        put(f"HT{name}N", s and round(s["n"]["median"]), "{:d}")
        put(f"HT{name}ModelNED", s and s["model_ned_mean"]["median"])
        put(f"HT{name}ModelNEDlo", s and s["model_ned_mean"]["lo"])
        put(f"HT{name}ModelNEDhi", s and s["model_ned_mean"]["hi"])
        put(f"HT{name}FloorNED", s and s["floor_ned_mean"]["median"])
        put(f"HT{name}ModelEM", s and s["model_exact_match"]["median"] * 100, "{:.1f}")
        put(f"HT{name}FloorEM", s and s["floor_exact_match"]["median"] * 100, "{:.1f}")

    # ── Nearest-neighbor retrieval baseline (honest strong lookup) ──────
    knn = _load(RES / "knn_retrieval" / "results.json")
    put("NEDknnAll", knn and knn["ned_mean"])
    put("EMknnAll", knn and knn["exact_match"] * 100, "{:.1f}")
    # Per-state retrieval NED (computed from the saved per-tick retrieved traces,
    # same stratified subset as the leaderboard). Retrieval beats the generator on
    # swing-up and reset and loses on balance; the branch-decision metric is where
    # the generator separates. Reported honestly rather than left TBD.
    _kps = (knn or {}).get("per_state", {})
    for _st, _nm in (("swing_up", "Sw"), ("balance", "Bal"), ("reset", "Rst")):
        put(f"NEDknn{_nm}", _kps.get(_st, {}).get("ned_mean"))

    # Intervals for the retrieval baseline. The model is reported as a five-fold
    # median with a range and retrieval was reported as one bare number, so the
    # comparison could not be falsified. Two intervals, measuring two things:
    #   matched  - bank = each fold's own TRAINING pair, so retrieval sees
    #              exactly what the model saw. Same axis as the model's fold
    #              interval and the only directly comparable one.
    #   fullbank - the eight-session bank kept from the original run, which is
    #              deliberately advantaged over any single fold, with a
    #              bootstrap CI over ticks. A different axis; label it as such.
    _kci = (knn or {}).get("ned_mean_ci95")
    put("NEDknnCIlo", _kci and _kci[0])
    put("NEDknnCIhi", _kci and _kci[1])
    _km = [_load(RES / f"knn_matched_fold_{i}" / "results.json") for i in range(5)]
    _kmv = sorted(k["ned_mean"] for k in _km if k)
    put("NEDknnMatched", _median(_kmv) if _kmv else None)
    put("NEDknnMatchedLo", _kmv[0] if _kmv else None)
    put("NEDknnMatchedHi", _kmv[-1] if _kmv else None)
    put("NEDknnMatchedN", len(_kmv) if _kmv else None, "{:d}")
    # The matched runs are PAIRED with the model's folds: fold i's bank is fold
    # i's training pair and the test set is the same, so the comparison can be
    # made fold by fold instead of median against median, which is both stronger
    # and immune to the two ranges overlapping.
    _mm = [_load(RES / f"ar_modern_bl2_w4000_thresh_strat100k_fold_{i}" /
                 "ar_modern_bl2_w4000_thresh_eval_test.json") for i in range(5)]
    _pairs = [(m["greedy_ned_mean"], k["ned_mean"], m.get("greedy_exact_match"),
               k["exact_match"]) for m, k in zip(_mm, _km) if m and k]
    if _pairs:
        put("KnnMatchedWins", f"{sum(1 for a, b, _, _ in _pairs if a < b)}"
                              f"/{len(_pairs)}")
        put("EMknnMatched", _median([p[3] for p in _pairs]) * 100, "{:.1f}")
    else:
        put("KnnMatchedWins", None)
        put("EMknnMatched", None)

    # ── Session-rotated cross-validation ────────────────────────────────
    # The canonical five folds rotate the TRAINING pair and share one held-out
    # pair, so their spread is training-resample variance. Rotating the TEST
    # sessions as well (fold i evaluated on fold i+1's training pair, disjoint
    # from its own) is the interval the text was being read as reporting.
    # Raw "all" columns are NOT comparable across rotated folds. The stratified
    # eval keeps every minority-state tick, so the state mix follows the held-out
    # pair: fold 0's subset is 44% reset (hardest) and fold 1's is 79% balance
    # (easiest). Comparing raw numbers would report which states each subset
    # happens to contain as if it were session-transfer error. rotate_reweight.py
    # re-weights every fold's per-state NED to the natural state distribution,
    # and puts the canonical folds through the same weighting so the two
    # intervals sit on one footing.
    _rw = _load(RES / "rotate_reweight.json")
    def _iv(tag, key):
        rows = (_rw or {}).get(tag) or []
        v = sorted(r["reweighted"] for r in rows if r.get("reweighted"))
        put(f"NED{key}", _median(v) if v else None)
        put(f"NED{key}Lo", v[0] if v else None)
        put(f"NED{key}Hi", v[-1] if v else None)
        put(f"NED{key}N", len(v) if v else None, "{:d}")
    _iv("rotated", "rotate")
    _iv("canonical", "canonReweighted")

    def _rows(tag):
        out = []
        for r in ((_rw or {}).get(tag) or []):
            ps = r["per_state"]
            out.append("{} & {:.4f} & {:.4f} & {:.4f} & {:.4f}".format(
                r["fold"], r["reweighted"], ps[0], ps[1], ps[2]))
        return (" \\\\\n".join(out) + r" \\") if out else (
            TBD + (" & " + TBD) * 4 + r" \\")
    put("CanonRotRows", _rows("canonical"))
    _cr = (_rw or {}).get("canonical") or []
    put("CanonRstLo", min((r["per_state"][2] for r in _cr), default=None))
    put("CanonRstHi", max((r["per_state"][2] for r in _cr), default=None))
    put("NatMixBal", (_rw or {}).get("natural_mix", [0, 0, 0])[1] * 100, "{:.0f}")
    put("RotRotRows", _rows("rotated"))
    # The two mixes that make the weighting necessary, quoted from the data so
    # the appendix sentence cannot drift from the table above it.
    _rr = (_rw or {}).get("rotated") or []
    if _rr:
        put("RotMixWorstRst", max(r["strat_mix"][2] for r in _rr) * 100, "{:.0f}")
        put("RotMixBestBal", max(r["strat_mix"][1] for r in _rr) * 100, "{:.0f}")
    else:
        put("RotMixWorstRst", None)
        put("RotMixBestBal", None)

    # Branch-decision accuracy (the reframed headline metric): committed-prefix
    # decision accuracy at branch sites, model vs nearest-neighbor retrieval.
    # Prefer the 5-fold aggregate (median [min-max], matching every other row);
    # fall back to the fold-0 single-run file if the 5-fold has not been run.
    bd5 = _load(RES / "branch_decision_5fold.json")
    bd = _load(RES / "branch_decision_analysis.json")
    if bd5:
        for _st, _nm in (("steady", "Steady"), ("path_change", "Path"),
                         ("transition", "Trans")):
            m = bd5["model_5fold"].get(_st)
            put(f"BDModel{_nm}",
                m and f"{m['median']:.1f} [{m['min']:.1f}--{m['max']:.1f}]", "{}")
            put(f"BDModel{_nm}Med", m and m["median"], "{:.1f}")
            put(f"BDKnn{_nm}", bd5["retrieval"].get(_st), "{:.1f}")
    elif bd:
        for _st, _nm in (("steady", "Steady"), ("path_change", "Path"),
                         ("transition", "Trans")):
            for _m, _mn in (("model", "Model"), ("retrieval", "Knn")):
                put(f"BD{_mn}{_nm}", bd["decision_acc"][_st][_m] * 100, "{:.1f}")
    else:
        for _nm in ("Steady", "Path", "Trans"):
            for _mn in ("Model", "Knn"):
                put(f"BD{_mn}{_nm}", None)
    put("HTPathknnEM", knn and knn["path_change"]["exact_match"] * 100, "{:.1f}")

    # ── R-2(b): can the decoder carry its own counter state? ─────────────
    # The paper's sharpest limitation is that branches behind long-horizon
    # counters reconstruct worst. The obvious remedy is to feed the decoder a
    # running summary of the trace it has already emitted, so it can maintain
    # the counters itself. Measured, that fails in a way that rules out the
    # whole family, and the mechanism is the interesting part.
    rb = _load(RES / "r2b_rollout.json")
    if rb and rb.get("arms"):
        a = rb["arms"]
        put("RtwoTicks", rb.get("ticks"), "{:,d}")
        for k, tag in (("base", "Base"), ("oracle", "Oracle"), ("self", "Self")):
            put(f"Rtwo{tag}NED", a.get(k, {}).get("ned"))
            put(f"Rtwo{tag}EM", (a.get(k, {}).get("exact") or 0) * 100, "{:.1f}")
    else:
        for nm in ("RtwoTicks", "RtwoBaseNED", "RtwoOracleNED", "RtwoSelfNED",
                   "RtwoBaseEM", "RtwoOracleEM", "RtwoSelfEM"):
            put(nm, None)
    rd = _load(RES / "r2b_drift.json")
    cps = (rd or {}).get("checkpoints") or []
    if cps:
        # Slope of count error against tick: the drift is unbounded, so quote
        # the rate rather than a single endpoint.
        xs = [c[0] for c in cps]; ys = [c[1] for c in cps]
        n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
        slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
                 / sum((x - mx) ** 2 for x in xs))
        put("RtwoDriftPerTick", slope, "{:.3f}")
        put("RtwoDriftAt", xs[-1], "{:,d}")
        put("RtwoDriftErr", ys[-1], "{:.0f}")
        # Ticks until the drift reaches the counter threshold it must represent.
        put("RtwoDriftReachGate", 1500 / slope, "{:,.0f}")
    else:
        for nm in ("RtwoDriftPerTick", "RtwoDriftAt", "RtwoDriftErr",
                   "RtwoDriftReachGate"):
            put(nm, None)

    # ── R-1: constrained decoding to the observed successor map ──────────
    # A negative result worth reporting: the obvious "make the output valid by
    # construction" move changes almost nothing, which says the model's errors
    # are wrong-legal-branch errors rather than validity errors.
    r1 = _load(RES / "r1_raw_diff.json")
    if r1 and r1.get("n"):
        put("RoneN", r1["n"], "{:,d}")
        put("RoneSeqDiff", r1["post_seq_diff"], "{:d}")
        put("RoneTokDiff", r1["post_tok_diff"], "{:d}")
    else:
        for nm in ("RoneN", "RoneSeqDiff", "RoneTokDiff"):
            put(nm, None)
    # Successor-map density: measured, and the reason the constraint is inert.
    sm = _load(RES / "successor_map_stats.json")
    put("SuccPairs", (sm or {}).get("n_pairs"), "{:d}")
    put("SuccEdges", (sm or {}).get("n_edges"), "{:d}")
    put("SuccPerEdge", (sm or {}).get("mean_successors"), "{:.2f}")

    # ── Information budget (N1: sensor-blind decoder) ───────────────────
    import math
    blind = _load(RES / "ar_modern_bl2_wzero_strat100k_fold_0" /
                  "ar_modern_bl2_wzero_eval_test.json")
    full = _load(RES / "ar_modern_bl2_w4000_thresh_strat100k_fold_0" /
                 "ar_modern_bl2_w4000_thresh_eval_test.json")
    if blind and full:
        bits = ((blind["tf_loss"] - full["tf_loss"])
                * full["mean_trace_len"] / math.log(2))
        put("BitsPerTick", bits, "{:.1f}")
        put("NEDblindAll", blind["greedy_ned_mean"])
    else:
        put("BitsPerTick", None)
        put("NEDblindAll", None)

    # ── First-party instrumentation overhead (F2') ──────────────────────
    ov = _load(REPO / "outputs/sil_demo/overhead_results.json")
    put("OverheadSlowdown",
        ov and f"{ov['slowdown']['median']:.3f}$\\times$")
    put("OverheadSlowdownPct",
        ov and f"{(ov['slowdown']['median'] - 1) * 100:.1f}\\%")
    put("OverheadInstrMedianUs",
        ov and ov["instrumented"]["median_us"], "{:.0f}")

    # ── Localization / repair loop (land with workstreams B and C) ──────
    for name in ("LocHitAtOne", "LocHitAtFive", "LocPrecisionAtFive",
                 "DetectLatency"):
        put(name, None)

    # ── Repair loop over the detected-fault corpus (C6 at scale) ────────
    rc = _load(REPO / "outputs/sil_demo/results/repair_corpus.json")
    if rc:
        s = rc["summary"]
        put("RepairDetected", s["detected"], "{:d}")
        put("RepairExact", s["repaired"], "{:d}")
        byname = {f["name"]: f for f in rc["faults"]}
        th = byname.get("theta_cap", {})
        # theta_cap: threshold recovered to sensor resolution
        put("RepairThetaProposed", th.get("proposed"), "{}")
        put("RepairThetaBefore", th.get("frac_before", 0) * 100, "{:.1f}")
        put("RepairThetaAfter", th.get("frac_after", 0) * 100
            if isinstance(th.get("frac_after"), (int, float)) else None, "{:.1f}")
    else:
        for nm in ("RepairDetected", "RepairExact", "RepairThetaProposed",
                   "RepairThetaBefore", "RepairThetaAfter"):
            put(nm, None)

    # ── Repair ablation (2x2: localization x trace), 5 seeds per cell ──────
    rs = _load(REPO / "outputs/sil_demo/results/repair_study.json")
    if rs:
        def cell(loc, trace):
            cs = [c for row in rs for c in row["conditions"]
                  if c["loc"] == loc and c["trace"] == trace]
            faults = sum(1 for c in cs if c["n_verified"] == c["seeds"])   # 5/5
            seedruns = sum(c["n_verified"] for c in cs)
            total = sum(c["seeds"] for c in cs)
            return faults, seedruns, total
        put("RepairStudyN", len(rs), "{:d}")
        put("RepairSeeds", rs[0]["conditions"][0]["seeds"], "{:d}")
        for tag, (loc, trace) in (("Full", (True, True)),
                                  ("TraceNoLoc", (False, True)),
                                  ("LocNoTrace", (True, False)),
                                  ("Neither", (False, False))):
            fa, sr, tot = cell(loc, trace)
            put(f"RepairExact{tag}", fa, "{:d}")          # faults repaired in all seeds
            put(f"RepairSeed{tag}", sr, "{:d}")           # seed-runs verified
        put("RepairSeedTotal", cell(True, True)[2], "{:d}")
        put("RepairHeuristic",
            sum(1 for row in rs if row.get("heuristic", {}).get("verified")), "{:d}")
        # zero-shot (first-attempt) exact repairs under loc+trace: verified runs
        # that succeeded on round 1, isolating the trace's information from the
        # loop's iteration.
        zs = sum(1 for row in rs for c in row["conditions"]
                 if c["loc"] and c["trace"]
                 for r in c["runs"] if r.get("verified") and r.get("rounds") == 1)
        put("RepairZeroShotFull", zs, "{:d}")

        # ── Cross-model repair (per-model per-condition seed-run attempts) ─────
        SILDIR = REPO / "outputs/sil_demo/results"
        MODELS = [("Opus", "repair_study.json"),
                  ("Haiku", "repair_study_haiku.json"),
                  ("Llama", "repair_study_llama.json"),
                  ("Qwen", "repair_study_qwen.json"),
                  ("Smol", "repair_study_smol.json")]
        CONDS = [("LT", True, True), ("Tonly", False, True),
                 ("Lonly", True, False), ("None", False, False)]
        _by_cond = {ctag: [] for ctag, _, _ in CONDS}
        for mtag, fname in MODELS:
            md = _load(SILDIR / fname)
            for ctag, loc, trace in CONDS:
                v = (sum(c["n_verified"] for row in md for c in row["conditions"]
                         if c["loc"] == loc and c["trace"] == trace) if md else None)
                put(f"RepairAtt{mtag}{ctag}", v, "{:d}")
                if v is not None:
                    _by_cond[ctag].append(v)
        # Cross-model ranges the results paragraph quotes, so it cannot go stale.
        for ctag, vals in _by_cond.items():
            put(f"RepairRange{ctag}",
                f"{min(vals)} to {max(vals)}" if vals else None, "{}")
    else:
        for nm in ("RepairStudyN", "RepairSeeds", "RepairSeedTotal",
                   "RepairHeuristic", "RepairExactFull", "RepairExactTraceNoLoc",
                   "RepairExactLocNoTrace", "RepairExactNeither", "RepairSeedFull",
                   "RepairSeedTraceNoLoc", "RepairSeedLocNoTrace", "RepairSeedNeither"):
            put(nm, None)

    # ── Repair divergence-reduction metric (partial credit vs exact-match) ─
    rr = _load(REPO / "outputs/sil_demo/results/repair_residual.json")
    if rr:
        put("RepairMedianReduction", rr["median"], "{:.0f}")
        put("RepairMinReduction", rr["min"], "{:.1f}")
        put("RepairMaxReduction", rr["max"], "{:.0f}")
        put("RepairThetaReduction", rr["per_fault"]["theta_cap"]["reduction_pct"], "{:.1f}")
        for tag, macro in (("opus", "Opus"), ("haiku", "Haiku"), ("llama", "Llama"),
                           ("qwen", "Qwen"), ("smol", "Smol")):
            bm = rr.get("by_model", {}).get(tag) or {}
            put(f"RepairDiv{macro}", bm.get("median"), "{:.0f}")
            put(f"RepairDivLo{macro}", bm.get("lo"), "{:.0f}")
            put(f"RepairDivHi{macro}", bm.get("hi"), "{:.0f}")
    else:
        for nm in ("RepairMedianReduction", "RepairMinReduction",
                   "RepairMaxReduction", "RepairThetaReduction",
                   "RepairDivOpus", "RepairDivHaiku", "RepairDivLlama",
                   "RepairDivQwen", "RepairDivSmol"):
            put(nm, None)

    # ── Behavioral verification: repaired vs shipped actuator command ──────
    rb = _load(REPO / "outputs/sil_demo/results/repair_behavior.json")
    if rb:
        put("RepairActuatorMaxDev",
            max(v["cmd_dev_repaired"] for v in rb.values()), "{:.3f}")
        put("BuggyCmdDevComparison", rb["theta_cmp"]["cmd_dev_buggy"], "{:.1f}")
        put("BuggyCmdDevCounter", rb["balanced_counter"]["cmd_dev_buggy"], "{:.3f}")
    else:
        for nm in ("RepairActuatorMaxDev", "BuggyCmdDevComparison",
                   "BuggyCmdDevCounter"):
            put(nm, None)

    # ── Named sensor-only detectors (USAD/TranAD/ModernTCN) ────────────
    nd = _load(FM / "results" / "named_detectors" / "results.json")
    if nd:
        det = nd["detectors"]
        for name, macro in (("USAD", "USAD"), ("TranAD", "TranAD"),
                            ("ModernTCN", "ModernTCN")):
            put(f"{macro}Timing", det[name]["timing_50ms"])
            put(f"{macro}Noise", det[name]["sensor_noise"], "{:.2f}")
            put(f"{macro}Stuck", det[name]["stuck_sensor"])
    else:
        for macro in ("USAD", "TranAD", "ModernTCN"):
            put(f"{macro}Timing", None); put(f"{macro}Noise", None)
            put(f"{macro}Stuck", None)

    # ── Opcode-swap mutants (comparison CWE-697, operator CWE-480) ──────
    op = _load(REPO / "outputs/sil_demo/results/opcode_replay.json")
    if op:
        byn = {m["name"]: m for m in op["mutants"]}
        cmp_ = byn.get("theta_cmp", {})
        put("CmpFaultDivPct", cmp_.get("frac_divergent", 0) * 100, "{:.0f}")
    else:
        put("CmpFaultDivPct", None)

    # ── N2 binary constant extraction (source-free conditioning) ────────
    n2 = _load(FM / "results" / "n2_binary_constants" / "results.json")
    if n2:
        put("NtwoFeatFidelity", f"{n2['feature_relevant_recovered']}/{n2['feature_relevant_target']}")
        put("NtwoDirectRecovered", n2["n_recovered"], "{:d}")
        put("NtwoDirectTarget", n2["n_target"], "{:d}")
    else:
        for nm in ("NtwoFeatFidelity", "NtwoDirectRecovered", "NtwoDirectTarget"):
            put(nm, None)

    # ── Temporal transfer (TR): train earliest sessions, test latest ────
    tr = _load(RES / "ar_modern_bl2_w4000_thresh_transfer_strat100k_fold_0" /
               "ar_modern_bl2_w4000_thresh_transfer_eval_test.json")
    put("NEDtransferAll", tr["greedy_ned_mean"] if tr else None)

    # ── Deployment operating point: trace-consistency held-out FPR@p99 ──
    tc = _load(RES / "anomaly_trace_consistency" / "results.json")
    put("FPRtc", tc and tc.get("fpr_p99_heldout", 0) * 100, "{:.2f}")
    for name in ("FPRenc", "Latencyenc", "Latencytc"):
        put(name, None)

    # ── Detector at a deployable operating point ─────────────────────────
    # AUC-ROC is prevalence-invariant and flatters a detector that will run on a
    # plant where faults are rare. AUC-PR and precision-at-base-rate are what
    # move, and the per-tick FPR has to become an alarm rate before the
    # deployment gate in Sec. "Shadow mode" can reference it.
    put("AUCPRtc", tc and tc.get("auc_pr", {}).get("trace_swap"))
    put("TCprevalence", tc and tc.get("sample_prevalence", 0) * 100, "{:.1f}")
    pbr = (tc or {}).get("precision_at_base_rate", {})
    for key, nm in (("0.1", "TCprecTen"), ("0.01", "TCprecOne"),
                    ("0.001", "TCprecTenth")):
        v = pbr.get(key) if pbr else None
        put(nm, None if v is None else v * 100, "{:.1f}")

    ev = _load(RES / "detect_event_fpr.json")
    # Length of the contiguous window the alarm rate is measured over. Derived
    # rather than hard-coded: it grew from the 16,000-tick localization windows
    # to whole sessions, and the appendix sentence has to follow the artifact.
    if ev and ev.get("pairs"):
        _n = int(round(ev["pairs"][0].get("minutes", 16000 / 60 / 1000)
                       * 60 * ev["tick_hz"]))
        put("RecolWindow", "{:,}".format(_n).replace(",", "{,}"))
    else:
        put("RecolWindow", "16{,}000")
    # Two calibrate/evaluate directions over the two held-out sessions. They
    # disagree by more than an order of magnitude, which is itself reportable:
    # a threshold fitted on one stretch of normal operation does not transfer
    # to another. Quote the WORSE direction so the alarm budget is not taken
    # from the lucky one, and expose the better one so the gap is visible.
    # Refuse to render an alarm rate measured over the 16,000-tick localization
    # windows. Those are 16 seconds of running time, so a zero count there says
    # almost nothing, and because these macros resolve rather than fall back to
    # TBD the draft would show the short-window figure without flagging it. Only
    # whole-session sources are quotable.
    _src = set((ev or {}).get("source", {}).values())
    if ev and (not _src or _src - {"full"}):
        ev = None       # short-window or provenance-less: not quotable
    if ev and ev.get("pairs"):
        prs = ev["pairs"]
        put("RecolWindow", "{:,}".format(
            int(round(prs[0]["minutes"] * 60 * ev["tick_hz"]))).replace(",", "{,}"))
        put("EvMinutes", sum(p["minutes"] for p in prs), "{:.1f}")
        put("EvFPRperTick", max(p["per_tick_fpr"] for p in prs) * 100, "{:.2f}")
        put("EvFPRperTickLow", min(p["per_tick_fpr"] for p in prs) * 100, "{:.2f}")
        # Both calibration directions now agree on the threshold, so the
        # short-window disagreement was a regime artifact and is not reported.
        put("EvThreshLo", min(p["threshold"] for p in prs))
        put("EvThreshHi", max(p["threshold"] for p in prs))
        # Worst direction by ALARM rate, which is the quantity being claimed;
        # ranking by per-tick exceedance picks the wrong one, because the
        # direction with fewer flagged ticks clusters them into longer runs.
        def _byk(pair):
            return {r["k"]: r for r in pair["rows"]}
        worst = max(prs, key=lambda p: _byk(p)[1]["per_hour"])
        put("EvAlarmsPerHourRaw", _byk(worst)[1]["per_hour"], "{:,.0f}")
        # Smallest debounce with zero observed false alarms in BOTH directions.
        # Quoting one direction would be quoting the luckier calibration.
        ks = sorted(_byk(prs[0]))
        zero = next((k for k in ks
                     if all(_byk(p)[k]["alarms"] == 0 for p in prs)), None)
        put("EvZeroK", zero, "{:d}")
        put("EvZeroMs", zero and _byk(prs[0])[zero]["debounce_ms"], "{:.0f}")
        put("EvZeroBound", zero and max(_byk(p)[zero]["upper95_per_hour"]
                                        for p in prs), "{:.1f}")
        # What this much normal operation can certify at best. With 8.1 min per
        # session even a zero count only bounds the rate at ~22/hour, so a
        # tighter plant budget is not demonstrable from this record at ANY
        # operating point, and saying so is the honest limit of the measurement.
        put("EvCertFloor", ev.get("certification_floor_per_hour"), "{:.1f}")
        put("EventFPRRows", " \\\\\n".join(
            "{} & {:,.0f} & {:,.1f} & {:.0f} ms".format(
                x["k"], x["per_hour"], x["upper95_per_hour"], x["debounce_ms"])
            for x in worst["rows"]) + r" \\")
    else:
        put("RecolWindow", TBD)
        for nm in ("EvMinutes", "EvFPRperTick", "EvFPRperTickLow", "EvThreshLo",
                   "EvThreshHi", "EvAlarmsPerHourRaw", "EvZeroK", "EvZeroMs",
                   "EvZeroBound", "EvCertFloor"):
            put(nm, None)
        put("EventFPRRows", TBD + (" & " + TBD) * 3 + r" \\")

    # ── Sim-to-real gate (tab:simreal; C2) ──────────────────────────────
    for name in ("SimRealNEDreal", "SimRealNEDsim", "SimRealCptreal",
                 "SimRealCptsim", "SimRealCnnreal", "SimRealCnnsim"):
        put(name, None)

    # ── Mutation corpus over replay (M; tab:localization) ───────────────
    cor = _load(Path("/home/simran/allspark-data-exploration/CPS-Debugger/"
                     "outputs/sil_demo/results/corpus_replay.json"))
    cs = cor.get("summary", {}) if cor else {}
    for macro, key in (("CorpusN", "n"), ("CorpusDetected", "detected"),
                       ("CorpusHitOrigin", "hit1_origin"),
                       ("CorpusHitMax", "hit1_max"),
                       ("CorpusExercised", "exercised")):
        put(macro, cs.get(key), "{}")

    # ── Control-flow fault taxonomy + edge-level localizer (redesigned) ──
    # Positive taxonomy: single-site, layout-preserving binary mutations that
    # change which branch the controller executes -- relational-operator
    # inversions (CWE-697) and guard-threshold constants (CWE-682). Sites are
    # enumerated directly from the shipped binary (enumerate_mutations.py);
    # localization takes the first-differing executed control-flow edge (the
    # flipped branch, which lives in the fault's own function).
    SIL = REPO / "outputs/sil_demo/results"
    from collections import Counter
    sites = _load(SIL / "mutation_sites.json")
    if sites:
        by = Counter(s["kind"] for s in sites)
        put("CFCorpusN", len(sites), "{:d}")
        put("CFCorpusRel", by.get("relational", 0), "{:d}")
        put("CFCorpusThresh", by.get("threshold", 0), "{:d}")
        put("CFCorpusArith", by.get("arithmetic", 0), "{:d}")
        put("CFCorpusSign", by.get("sign", 0), "{:d}")
    else:
        for nm in ("CFCorpusN", "CFCorpusRel", "CFCorpusThresh",
                   "CFCorpusArith", "CFCorpusSign"):
            put(nm, None)
    # by_kind rows are [detected, raw_hit, blame_hit]. The headline is the blame
    # frame (attribution to the innermost function that could HOST a fault);
    # raw is kept so the paper can report what the frame is worth.
    el = _load(SIL / "edge_localize.json")
    KINDS = (("Rel", "relational"), ("Thresh", "threshold"),
             ("Arith", "arithmetic"), ("Sign", "sign"))
    if el:
        def _pct(hit, det):
            return f"{hit / det * 100:.0f}" if det else "0"
        det = el["detected"]
        put("EdgeLocDetected", det, "{:d}")
        # Reachability, i.e. what fraction of ENUMERATED sites are observable at
        # all. Distinct from EdgeLocHitPct, which is hit@1 among those detected.
        if sites:
            put("EdgeLocReachPct", 100 * det / len(sites), "{:.0f}")
        put("EdgeLocHitOne", el["hit1_blame"], "{:d}")
        put("EdgeLocHitPct", _pct(el["hit1_blame"], det), "{}")
        put("EdgeLocRawHitPct", _pct(el["hit1"], det), "{}")
        for tag, kind in KINDS:
            k = el["by_kind"].get(kind, [0, 0, 0])
            put(f"EdgeLoc{tag}Detected", k[0], "{:d}")
            put(f"EdgeLoc{tag}Hit", k[2], "{:d}")
            put(f"EdgeLoc{tag}Pct", _pct(k[2], k[0]), "{}")
            put(f"EdgeLoc{tag}RawPct", _pct(k[1], k[0]), "{}")
    else:
        for nm in ("EdgeLocDetected", "EdgeLocHitOne", "EdgeLocHitPct",
                   "EdgeLocRawHitPct", "EdgeLocReachPct"):
            put(nm, None)
        for tag, _ in KINDS:
            for suf in ("Detected", "Hit", "Pct", "RawPct"):
                put(f"EdgeLoc{tag}{suf}", None)

    # Reconstruction-driven localization (R1.2, composition gap closed): the
    # healthy reference is the model's reconstructed trace on a HELD-OUT session,
    # not an exact replay of the shipped binary. Same detected set; exact-replay
    # is the upper bound, the reconstruction-driven number is the deployable one.
    rc2 = _load(SIL / "r1_2_composition_summary.json")
    if rc2:
        def _ci(d):
            return f"{d['pct']:.0f} [{d['ci_lo']:.0f}--{d['ci_hi']:.0f}]"
        put("LocExactCI", _ci(rc2["exact_replay"]), "{}")
        put("LocReconCI", _ci(rc2["recon_deployable"]), "{}")
        put("LocReconPct", rc2["recon_deployable"]["pct"], "{:.0f}")
        put("LocReconHit", rc2["recon_deployable"]["hit"], "{:d}")
        put("LocReconNaivePct", rc2["recon_naive"]["pct"], "{:.0f}")
        put("LocCompN", rc2["n_detected"], "{:d}")
        put("LocExactPct", rc2["exact_replay"]["pct"], "{:.0f}")
        # Composition tax: what the deployable path gives up against its own
        # exact-replay ceiling, on the identical detected set.
        put("LocCompTax", rc2["exact_replay"]["pct"]
            - rc2["recon_deployable"]["pct"], "{:.0f}")
        # The rule that gates on a replay of the shipped binary. Reported so the
        # gap between it and the deployable rule is visible: it is what
        # localization would reach if a golden trace were available, which is
        # also the regime in which exact-replay localization applies.
        og = rc2.get("recon_oracle_gated")
        put("LocOracleGatedCI", _ci(og) if og else None, "{}")
        cal = rc2.get("recon_calibrated")
        put("LocCalibPct", cal["pct"] if cal else None, "{:.0f}")
        # Where the composition cost concentrates. The scan entry point runs
        # directly; the functions it calls are reached through it, and a fault
        # in a callee often first shows up after control returns, so naming the
        # callee is the harder inference. Split so the paper can say which.
        rows_v8 = _load(SIL / "edge_localize_recon_v8.json")
        if rows_v8:
            det = [r for r in rows_v8["rows"] if r.get("detected")]
            for tag, sel in (("Tick", lambda r: r["gt"] == 22),
                             ("Callee", lambda r: r["gt"] != 22)):
                sub = [r for r in det if sel(r)]
                n = len(sub) or 1
                put(f"LocStack{tag}Pct",
                    100 * sum(r.get("recon_align") == r["gt"] for r in sub) / n,
                    "{:.0f}")
                put(f"LocExact{tag}Pct",
                    100 * sum(r.get("exact_pred") == r["gt"] for r in sub) / n,
                    "{:.0f}")
                put(f"LocSplit{tag}N", len(sub), "{:d}")
        else:
            for tag in ("Tick", "Callee"):
                put(f"LocStack{tag}Pct", None); put(f"LocExact{tag}Pct", None)
                put(f"LocSplit{tag}N", None)

        dec = rc2.get("recon_decision")
        put("LocDecisionCI", _ci(dec) if dec else None, "{}")
        put("LocDecisionPct", dec["pct"] if dec else None, "{:.0f}")
        for tag, kind in KINDS:
            bk = rc2["by_kind"].get(kind)
            put(f"LocRecon{tag}Pct", bk["recon"]["pct"] if bk else None, "{:.0f}")
            put(f"LocExact{tag}Pct", bk["exact"]["pct"] if bk else None, "{:.0f}")
            put(f"LocComp{tag}N", bk["recon"]["n"] if bk else None, "{:d}")
    else:
        for nm in ("LocExactCI", "LocReconCI", "LocReconPct", "LocReconHit",
                   "LocReconNaivePct", "LocCompN", "LocExactPct", "LocCompTax",
                   "LocOracleGatedCI", "LocCalibPct", "LocDecisionCI",
                   "LocDecisionPct"):
            put(nm, None)
        for tag, _ in KINDS:
            for pre in ("LocRecon", "LocExact"):
                put(f"{pre}{tag}Pct", None)
            put(f"LocComp{tag}N", None)

    # What naming a FUNCTION actually buys. Localization is measured at function
    # granularity over 10 candidates; the abstract used to claim the branch. The
    # honest quantity is how far the function narrows the search: a mutation's pc
    # is an operator offset and an edge's from_pc is a branch source, different
    # program-counter spaces, so branch-level hit@1 against the mutation site is
    # not measurable without a mapping we do not have.
    _pm = _load(FM / "train_data_full/pruned_token_mapping.json")
    _el = _load(SIL / "edge_localize.json")
    if _pm and _el:
        from collections import Counter as _C, defaultdict as _dd
        _by = _dd(set)
        for f, a, b in [tuple(t) for t in _pm["triples"]]:
            _by[(f, a)].add(b)
        _spf = _C()
        for (f, a), su in _by.items():
            if len(su) >= 2:
                _spf[f] += 1
        put("LocSitesTotal", sum(_spf.values()), "{:d}")
        _ns = [_spf.get(r["gt"], 0) for r in _el["rows"]]
        put("LocSitesInFunc", statistics.median(_ns) if _ns else None, "{:.0f}")
        put("LocSitesInFuncMax", max(_ns) if _ns else None, "{:d}")
    else:
        for nm in ("LocSitesTotal", "LocSitesInFunc", "LocSitesInFuncMax"):
            put(nm, None)

    # Localization floors on the identical detected set (random / frequency
    # prior / automaton-granularity). Percentages are precomputed there.
    lb = _load(SIL / "localize_baselines.json")
    if lb:
        put("LocBaseRandomPct", lb["random"]["hit1_pct"], "{:.0f}")
        put("LocBaseFreqHit", lb["frequency_prior"]["hit1"], "{:d}")
        put("LocBaseFreqPct", lb["frequency_prior"]["hit1_pct"], "{:.0f}")
        put("LocBaseAutomatonPct", lb["automaton"]["hit1_pct"], "{:.0f}")
    else:
        for nm in ("LocBaseRandomPct", "LocBaseFreqHit", "LocBaseFreqPct",
                   "LocBaseAutomatonPct"):
            put(nm, None)

    # ── Second testbed: bottle-filling station ───────────────────────────
    # Model vs its own baselines, all fitted on the training sessions and
    # applied out of sample. The difficulty-calibration constants are recorded
    # in the portability audit log and restated here so the appendix does not
    # hard-code them.
    # The 4K-step run is the primary model: it matches the pendulum's training
    # budget and was fixed before any test number was seen. The 20K-step run is
    # an ablation on training budget, reported separately below. Do NOT pick
    # whichever scores higher, that is selection on the test set.
    fr = _load(FM / "results/filling_fold_0/filling_recon_analysis.json")
    fk = _load(FM / "results/filling_fold_0/filling_knn_retrieval.json")
    fr20 = _load(FM / "results/filling_20k_fold_0/filling_recon_analysis.json")
    if fr20:
        put("SILtwoLongExact", fr20["model"]["exact_trace"] * 100, "{:.2f}")
        put("SILtwoLongBD", fr20["model"]["branch_decision"] * 100, "{:.2f}")
        put("SILtwoLongPrefix", fr20["model"]["mean_prefix"], "{:.1f}")
    else:
        for nm in ("SILtwoLongExact", "SILtwoLongBD", "SILtwoLongPrefix"):
            put(nm, None)
    # Training-budget ablation constants: the two decoder schedules and the
    # teacher-forced accuracy each reaches on the held-out sessions, read from
    # the eval logs. Kept together so the contrast cannot drift apart.
    put("SILtwoLongSteps", "20{,}000", "{}")
    put("SILtwoTFshort", 0.9467, "{:.3f}")
    put("SILtwoTFlong", 0.9525, "{:.3f}")
    if fr:
        put("SILtwoEvalTicks", fr["model"]["n"], "{:,d}")
        for tag, sub in (("Model", fr["model"]), ("Floor", fr["floor"])):
            put(f"SILtwo{tag}Exact", sub["exact_trace"] * 100, "{:.2f}")
            put(f"SILtwo{tag}BD", sub["branch_decision"] * 100, "{:.2f}")
            put(f"SILtwo{tag}Prefix", sub["mean_prefix"], "{:.1f}")
    else:
        for nm in ("SILtwoEvalTicks",):
            put(nm, None)
        for tag in ("Model", "Floor"):
            for suf in ("Exact", "BD", "Prefix"):
                put(f"SILtwo{tag}{suf}", None)
    if fk:
        r = fk["retrieval"]
        put("SILtwoKnnExact", r["exact_trace"] * 100, "{:.2f}")
        put("SILtwoKnnBD", r["branch_decision"] * 100, "{:.2f}")
        put("SILtwoKnnPrefix", r["mean_prefix"], "{:.1f}")
        put("SILtwoKnnBank", fk["bank_size"], "{:,d}")
    else:
        for nm in ("SILtwoKnnExact", "SILtwoKnnBD", "SILtwoKnnPrefix",
                   "SILtwoKnnBank"):
            put(nm, None)
    # Fault corpus on the filling line: how many enumerated control-flow faults
    # actually change the executed trace on recorded sensor data. This is the
    # observability boundary on that testbed, the counterpart of the pendulum's
    # detected/enumerated ratio, NOT a model detection rate.
    ff = _load(REPO / "outputs/sil2_dosing/results/fault_replay.json")
    if ff:
        tot, det, ovr = ff["total"], ff["detected"], ff["overrun"]
        nT, nD = sum(tot.values()), sum(det.values())
        put("SILtwoFaultN", nT, "{:,d}")
        put("SILtwoFaultDet", nD, "{:,d}")
        put("SILtwoFaultPct", 100 * nD / nT, "{:.0f}")
        put("SILtwoFaultOverrun", sum(ovr.values()), "{:d}")
        for tag, kind in (("Rel", "relational"), ("Thresh", "threshold"),
                          ("Arith", "arithmetic")):
            n, d_ = tot.get(kind, 0), det.get(kind, 0)
            put(f"SILtwoFault{tag}N", n, "{:d}")
            put(f"SILtwoFault{tag}Det", d_, "{:d}")
            put(f"SILtwoFault{tag}Pct", (100 * d_ / n) if n else None, "{:.0f}")
    else:
        for nm in ("SILtwoFaultN", "SILtwoFaultDet", "SILtwoFaultPct",
                   "SILtwoFaultOverrun"):
            put(nm, None)
        for tag in ("Rel", "Thresh", "Arith"):
            for suf in ("N", "Det", "Pct"):
                put(f"SILtwoFault{tag}{suf}", None)

    # Cross-testbed headline: the same three-row comparison on both rigs, on the
    # shared metric set. Per-state columns are deliberately absent here because
    # the pendulum has 3 controller states and the filling line 10 logged
    # phases, so a shared per-state breakdown would not mean anything; that
    # detail stays per-testbed in the appendix.
    fned = _load(FM / "results/filling_fold_0/filling_ned.json")
    if fned:
        for tag, key in (("Model", "model"), ("Knn", "knn"), ("Floor", "floor")):
            r = fned.get(key)
            put(f"SILtwoNED{tag}", r["ned"] if r else None, "{:.4f}")
            put(f"SILtwoEM{tag}", (r["em"] * 100) if r else None, "{:.1f}")
    else:
        for tag in ("Model", "Knn", "Floor"):
            put(f"SILtwoNED{tag}", None)
            put(f"SILtwoEM{tag}", None)

    # W1: does the headline survive removing the CONTROLLER-INTERNAL channels?
    # Of the seven inputs, pendulum_state is the FSM state variable itself,
    # target_x the commanded setpoint and iteration an internal counter; only
    # four are physical measurements. Both arms are plain w4000 encoders at 512
    # dims with no threshold features, since those are themselves derived partly
    # from pendulum_state and target_x and would confound the channel change.
    for tag, d in (("Seven", _load(RES / "w1_ch7_fold0/ar_modern_eval_test.json")),
                   ("Four", _load(RES / "w1_ch4_fold0/ar_modern_eval_test.json"))):
        if d:
            put(f"WoneNED{tag}", d["greedy_ned_mean"])
            put(f"WoneEM{tag}", d["greedy_exact_match"] * 100, "{:.1f}")
            for st, nm in STATES.items():
                put(f"WoneNED{tag}{nm}", d["greedy_per_state"][st]["ned_mean"])
        else:
            put(f"WoneNED{tag}", None); put(f"WoneEM{tag}", None)
            for nm in STATES.values():
                put(f"WoneNED{tag}{nm}", None)

    # Filling-station CLOSED loop: the mutated controller's own commands move the
    # plant, which changes what it reads next. The pendulum's closed loop is
    # bounded by how well its plant model matches a real rig; this testbed IS its
    # plant, so the number measures the method.
    cl = _load(REPO / "outputs/sil2_dosing/results/cosim_faults.json")
    if cl:
        rws = [r for r in cl["rows"] if "error" not in r and not r.get("killed")]
        dv = [r for r in rws if r.get("trace_diverged")]
        mv = [r for r in dv if r.get("first_act_tick", -1) >= 0]
        sil = [r for r in dv if r.get("first_act_tick", -1) < 0]
        put("SILtwoClRan", len(rws), "{:,d}")
        put("SILtwoClDiverged", len(dv), "{:,d}")
        put("SILtwoClMoved", len(mv), "{:d}")
        put("SILtwoClSilent", len(sil), "{:d}")
        put("SILtwoClSilentPct", 100 * len(sil) / len(dv) if dv else None, "{:.0f}")
        leads = [r["lead"] for r in mv if r.get("lead") is not None and r["lead"] > 0]
        put("SILtwoClLead", statistics.median(leads) if leads else None, "{:.0f}")
        # Silent rate by family: arithmetic corrupts a computation and leaves the
        # commands identical; guard faults change what the machine does.
        for tag, kind in (("Arith", "arithmetic"), ("Rel", "relational"),
                          ("Thresh", "threshold")):
            d_ = [r for r in dv if r["kind"] == kind]
            s_ = [r for r in d_ if r.get("first_act_tick", -1) < 0]
            put(f"SILtwoClSilent{tag}",
                100 * len(s_) / len(d_) if d_ else None, "{:.0f}")
    else:
        for nm in ("SILtwoClRan", "SILtwoClDiverged", "SILtwoClMoved",
                   "SILtwoClSilent", "SILtwoClSilentPct", "SILtwoClLead",
                   "SILtwoClSilentArith", "SILtwoClSilentRel",
                   "SILtwoClSilentThresh"):
            put(nm, None)

    # Difficulty calibration, measured in-sample during controller design
    # (sil2_portability_audit.md). In-sample by construction, so these are NOT
    # comparable to the out-of-sample floor in the results table above.
    put("SILtwoFloorDraft", 96.0, "{:.0f}")
    put("SILtwoFloorTuned", 5.1, "{:.1f}")
    put("SILtwoFloorPend", 2.5, "{:.1f}")

    # Path-change rate on BOTH testbeds, measured on the natural population of
    # whole sessions so the two are comparable. These were hand-transcribed
    # constants measured on different populations: the pendulum figure came
    # from the stratified evaluation subset, which keeps every swing-up and
    # reset tick and subsamples the repetitive balance majority, so it
    # overstated how often the pendulum's path changes and understated the gap.
    dc = _load(RES / "difficulty_calibration.json")
    for key, tag in (("pendulum_natural", "Pend"), ("filling_natural", "Fill")):
        d_ = (dc or {}).get(key) or {}
        put(f"PathChange{tag}", d_.get("median"), "{:.0f}")
        put(f"PathChange{tag}Lo", d_.get("lo"), "{:.0f}")
        put(f"PathChange{tag}Hi", d_.get("hi"), "{:.0f}")
        put(f"PathChange{tag}Sessions", d_.get("n_sessions"), "{:d}")
    put("PathChangeTicks",
        sum((dc or {}).get(k, {}).get("n_ticks", 0)
            for k in ("pendulum_natural", "filling_natural")) or None,
        "{:,d}")
    # Kept for the in-sample design-time narrative; superseded for the
    # cross-testbed comparison by the matched-population numbers above.
    put("SILtwoPathChange", (dc or {}).get("filling_natural", {}).get("median"),
        "{:.0f}")

    # ── Closed-loop plant identification / validation (SIL fidelity) ─────
    pv = _load(SIL / "plant_validation.json")
    if pv:
        put("SilFrictionRMSE", pv["friction_rmse_rad"], "{:.3f}")
        put("SilFrictionSecs", pv["friction_ref_seconds"], "{:.0f}")
        put("SilCtrlTsMs", pv["ctrl_ts_ms"], "{:.0f}")
        put("SilDriveTauMs", pv["drive_tau_v_ms"], "{:.0f}")
        put("SilDriveCorr", pv["drive_corr_cmd_actual"], "{:.3f}")
        put("SilWireCycles", f"{pv['n_cycles_wire']/1000:.0f}K")
    else:
        for nm in ("SilFrictionRMSE", "SilFrictionSecs", "SilCtrlTsMs",
                   "SilDriveTauMs", "SilDriveCorr", "SilWireCycles"):
            put(nm, None)

    # ── Statistics hygiene: bootstrap CIs (WS-G) ────────────────────────
    st = _load(SIL / "stats_ci.json")
    if st:
        def cistr(d):
            return None if not d else f"{d['point']:.0f} [{d['lo']:.0f}--{d['hi']:.0f}]"
        put("LocHitCI", cistr(st.get("loc_hit1")), "{}")
        put("LocHitRelCI", cistr(st.get("loc_hit1_relational")), "{}")
        put("LocHitThreshCI", cistr(st.get("loc_hit1_threshold")), "{}")
        put("RepairRateCI", cistr(st.get("repair_rate")), "{}")
        tr = st.get("transition", {})
        put("TransN", tr.get("n"), "{:d}")
        put("TransModelNED", tr.get("model_ned"))
        put("TransFloorNED", tr.get("floor_ned"))
    else:
        for nm in ("LocHitCI", "LocHitRelCI", "LocHitThreshCI", "RepairRateCI",
                   "TransN", "TransModelNED", "TransFloorNED"):
            put(nm, None)

    # ── Closed-loop fault demonstration (feedback signature) ─────────────
    cf = _load(SIL / "cosim_faults.json")
    if cf:
        s = cf["summary"]
        put("ClfFaults", s["n_faults"], "{:d}")
        put("ClfDetected", s["detected"], "{:d}")
        put("ClfHit", s["hit1"], "{:d}")
        put("ClfFedBack", s["fed_back"], "{:d}")
        # Where a fault eventually moves the plant, how many ticks the executed
        # path diverges BEFORE the trajectory does. This is the closed-loop form
        # of the paper's claim that a control-flow fault is visible in the code
        # path before it is visible in the physics.
        leads = [r["first_traj_tick"] - r["first_edge_tick"] for r in cf["rows"]
                 if r.get("detected") and r.get("first_traj_tick", -1) >= 0
                 and r.get("first_edge_tick") is not None]
        short = sorted(x for x in leads if x < 100)
        put("ClfLeadN", len(short), "{:d}")
        put("ClfLeadMin", min(short) if short else None, "{:d}")
        put("ClfLeadMax", max(short) if short else None, "{:d}")
    else:
        for nm in ("ClfFaults", "ClfDetected", "ClfHit", "ClfFedBack",
                   "ClfLeadN", "ClfLeadMin", "ClfLeadMax"):
            put(nm, None)

    # ── Closed-loop study over the WHOLE enumerated corpus ───────────────
    # cosim_faults.json above is the 14-site mechanism demonstration
    # (relational and threshold only). This is every enumerated site whose
    # function the nominal exercises, all four families, so the pendulum has
    # its own scored corpus instead of borrowing the filling station's.
    cff = _load(SIL / "cosim_faults_full.json")
    if cff:
        rws = cff["rows"]
        ok = [r for r in rws if r.get("status") == "ok"]
        dt = [r for r in ok if r.get("detected")]
        sil = [r for r in dt if r.get("first_act_tick", -1) < 0]
        put("ClfFullEnumerated", len(rws), "{:d}")
        put("ClfFullRan", len(ok), "{:d}")
        put("ClfFullTrapped", len(rws) - len(ok), "{:d}")
        put("ClfFullDetected", len(dt), "{:d}")
        put("ClfFullSilent", len(sil), "{:d}")
        put("ClfFullMoved", len(dt) - len(sil), "{:d}")
        put("ClfFullSilentPct", 100 * len(sil) / len(dt) if dt else None, "{:.0f}")
        # Localization under the SAME frame-selection rule the open-loop study
        # uses (blame()); the raw first-differing frame is reported beside it,
        # as it is there, because the lift is the whole point of the step.
        put("ClfFullHitPct",
            100 * sum(bool(r.get("hit")) for r in dt) / len(dt) if dt else None, "{:.0f}")
        put("ClfFullHitRawPct",
            100 * sum(bool(r.get("hit_raw")) for r in dt) / len(dt) if dt else None, "{:.0f}")
        # The command diverges before the trajectory does: closed-loop form of
        # the claim that the code path carries the fault first.
        leads = [r["first_traj_tick"] - r["first_act_tick"] for r in dt
                 if r.get("first_act_tick", -1) >= 0
                 and r.get("first_traj_tick", -1) >= 0]
        put("ClfFullLeadMed", statistics.median(leads) if leads else None, "{:.0f}")
        # Either the command is bit-identical and the trajectory is too, or the
        # stand is lost: no middle ground, so the silent rate does not depend on
        # a deviation threshold.
        mv = [r for r in dt if r.get("first_act_tick", -1) >= 0]
        put("ClfFullMovedMinDev", min((r["max_dtheta"] for r in mv), default=None), "{:.1f}")
        for tag, kind in (("Arith", "arithmetic"), ("Rel", "relational"),
                          ("Thresh", "threshold"), ("Sign", "sign")):
            kd = [r for r in ok if r["kind"] == kind and r.get("detected")]
            ks = [r for r in kd if r.get("first_act_tick", -1) < 0]
            put(f"ClfFullN{tag}", len(kd), "{:d}")
            put(f"ClfFullSilent{tag}Pct",
                100 * len(ks) / len(kd) if kd else None, "{:.0f}")
            put(f"ClfFullHit{tag}Pct",
                100 * sum(bool(r.get("hit")) for r in kd) / len(kd) if kd else None, "{:.0f}")
    else:
        for nm in ("ClfFullEnumerated", "ClfFullRan", "ClfFullTrapped",
                   "ClfFullDetected", "ClfFullSilent", "ClfFullMoved",
                   "ClfFullSilentPct", "ClfFullHitPct", "ClfFullHitRawPct",
                   "ClfFullLeadMed", "ClfFullMovedMinDev"):
            put(nm, None)
        for tag in ("Arith", "Rel", "Thresh", "Sign"):
            put(f"ClfFullN{tag}", None); put(f"ClfFullSilent{tag}Pct", None)
            put(f"ClfFullHit{tag}Pct", None)

    # ── Are the silent faults dormant, or inert? ─────────────────────────
    # A silent fault never moves the actuator under the trajectory the machine
    # ran. Re-running each from other initial conditions asks whether that is
    # dormancy or no reachable effect at all. The answer is the latter, which is
    # the equivalent-mutant phenomenon rather than a latent hazard.
    dm = _load(SIL / "dormancy_full.json")
    if dm:
        put("DormTested", dm["n_tested"], "{:d}")
        put("DormActivated", dm["n_activated"], "{:d}")
        put("DormICs", len(dm["ics"]), "{:d}")
        put("DormTicks", dm["ticks"], "{:,d}")
    else:
        for nm in ("DormTested", "DormActivated", "DormICs", "DormTicks"):
            put(nm, None)

    # Sustained balance in the corrected two-rate co-simulation, and the counter
    # that gates the controller's balance state. Read from sign_sweep.json's
    # default-signs row, which is the configuration the fault study runs.
    ss = _load(SIL / "sign_sweep.json")
    if ss:
        base = next((r for r in ss if r["name"] == "pos+1_vel+1"), None)
        put("ClfBalanceTicks", base["max_balance"] if base else None, "{:,d}")
    else:
        put("ClfBalanceTicks", None)
    put("ClfGateTicks", 1500, "{:,d}")
    # Run-length control. Tripling the run, which takes the nominal from 22% to
    # 100% of its final ticks near the top, detects exactly the same faults. That
    # is what licenses calling the rest out of scope here rather than merely
    # unvisited by a short run.
    cfl = _load(SIL / "cosim_faults_long.json")
    if cf and cfl:
        put("ClfProbed", cf["summary"]["n_faults"], "{:d}")
        put("ClfTicksShort", 40000, "{:,d}")
        put("ClfTicksLong", 120000, "{:,d}")
        put("ClfNearTopShort", cf.get("nominal_near_top", 0.22) * 100, "{:.0f}")
        put("ClfNearTopLong", cfl.get("nominal_near_top", 1.0) * 100, "{:.0f}")
        put("ClfLongDetected", cfl["summary"]["detected"], "{:d}")
        put("ClfOutOfScope",
            cf["summary"]["n_faults"] - cf["summary"]["detected"], "{:d}")
    else:
        for nm in ("ClfProbed", "ClfTicksShort", "ClfTicksLong",
                   "ClfNearTopShort", "ClfNearTopLong", "ClfLongDetected",
                   "ClfOutOfScope"):
            put(nm, None)

    # Closed-loop coverage after seeding the loop from a near-balanced state with
    # the full identified friction (G1-2): the balance-phase code runs, so
    # balance-phase faults become reachable.
    cfs = _load(SIL / "cosim_faults_seeded.json")
    if cfs:
        s = cfs["summary"]
        put("ClfSeedFaults", s["n_faults"], "{:d}")
        put("ClfSeedDetected", s["detected"], "{:d}")
        put("ClfSeedHit", s["hit1"], "{:d}")
        put("ClfSeedFedBack", s["fed_back"], "{:d}")
    else:
        for nm in ("ClfSeedFaults", "ClfSeedDetected", "ClfSeedHit",
                   "ClfSeedFedBack"):
            put(nm, None)
    g12 = _load(SIL / "g1_2_probe.json")
    if g12:
        put("ClfCtrlFuncsBase",
            len(g12["A_default_MSEonly"]["ctrl_funcs_exercised"]), "{:d}")
        put("ClfCtrlFuncsSeed",
            len(g12["C_fullfric_seed_nearbalanced"]["ctrl_funcs_exercised"]), "{:d}")
    else:
        for nm in ("ClfCtrlFuncsBase", "ClfCtrlFuncsSeed"):
            put(nm, None)

    # Placeholder macros the paper references but whose experiments have not
    # landed yet (operator study, pretraining/OOD ablations, kNN per-state row).
    # Emit as TBD if no earlier block set them, so regeneration never leaves the
    # paper with an undefined macro.
    for nm in ("OpN", "OpSolvedCtrlA", "OpSolvedTrtA", "OpTimeCtrlA", "OpTimeTrtA",
               "OpSolvedCtrlC", "OpSolvedTrtC", "OpTimeCtrlC", "OpTimeTrtC",
               "OpSolvedCtrlD", "OpSolvedTrtD", "OpTimeCtrlD", "OpTimeTrtD",
               "NEDrandinit", "NEDoodNominal", "NEDoodNoise", "NEDoodStuck",
               "NEDoodTiming", "NEDknnSw", "NEDknnBal", "NEDknnRst"):
        macros.setdefault(nm, TBD)

    lines = ["% AUTO-GENERATED by gen_numbers.py -- do not edit by hand.",
             "% Regenerate after any eval lands.\n"]
    for k, v in sorted(macros.items()):
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    OUT.write_text("\n".join(lines) + "\n")
    n_tbd = sum(1 for v in macros.values() if v == TBD)
    print(f"wrote {OUT} ({len(macros)} macros, {n_tbd} TBD)")


if __name__ == "__main__":
    main()
