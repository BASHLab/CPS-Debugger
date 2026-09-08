"""Divergence-reduction metric for the repair study, so exact-match is not the
only lens. For each fault under the localization+trace condition, report how
much of the fault's trace divergence the repaired binary removes:

    reduction = (frac_before - residual) / frac_before

where frac_before is the buggy binary's divergence from the shipped binary and
residual is the repaired binary's (0 when exact). This gives partial credit:
the balance-entry threshold recovers to sensor resolution (residual 0.1%
against a 3.1% fault, ~97% removed) rather than reading as a plain failure.

No API calls; reuses cached replays. Writes results/repair_residual.json.
"""
import json
from pathlib import Path

import numpy as np

from replay import load_session, replay, compare, CODE, DATA
from repair_study import FAULTS, BASE, repaired_bytes

NOM = np.load("/tmp/repair_mat_nom.npy")
OUT = Path(__file__).parent / "results"


def buggy_proposal(f):
    """The proposal that reproduces the deployed (buggy) binary."""
    if f["kind"] == "opcode":
        return next(k for k, v in f["ops"].items() if v == f["seeded"])
    return f["seeded"]


MODELS = [("opus", "repair_study.json"), ("haiku", "repair_study_haiku.json"),
          ("llama", "repair_study_llama.json"), ("qwen", "repair_study_qwen.json"),
          ("smol", "repair_study_smol.json")]


def residual_of(run, fb):
    """The divergence a run's final binary still carries. Zero when the repair
    verified; the last proposal's divergence when it produced an applicable but
    non-exact fix; the full buggy divergence when it never produced an
    applicable fix (an unparseable or unpatchable proposal is a repair failure,
    not missing data, so it scores 0% removed rather than being excluded).
    call_llm retries with backoff, so a run that still has no applicable
    proposal failed on the model's output, not on the API."""
    if run.get("verified"):
        return 0.0
    if run.get("final_frac") is not None:
        return run["final_frac"]
    return fb


def model_div_removed(byname, frac_before):
    """Percent of trace divergence removed under localization+trace, over every
    fault-seed run. Returns the median and the [min, max] range across all
    fault-seed runs, so the 5-seed spread is visible."""
    reds = []
    for name, fb in frac_before.items():
        lt = [c for c in byname[name]["conditions"] if c["loc"] and c["trace"]][0]
        for r in lt["runs"]:
            resid = residual_of(r, fb)
            reds.append(max(0.0, 100.0 * (fb - resid) / fb) if fb else 0.0)
    return dict(median=float(np.median(reds)), lo=float(np.min(reds)),
                hi=float(np.max(reds)))


def main():
    angle, x, _ = load_session(sorted(Path(DATA).glob("*.parquet"))[0].as_posix(), 0, 15000)

    # buggy-binary divergence per fault (model-independent), computed once
    frac_before = {}
    for f in FAULTS:
        Path(f"/tmp/rr_{f['name']}.wasm").write_bytes(repaired_bytes(f, buggy_proposal(f)))
        frac_before[f["name"]] = compare(
            NOM, replay(f"/tmp/rr_{f['name']}.wasm", angle, x))["frac_divergent"]

    # Opus per-fault detail (for the headline median/range in the text)
    opus = {r["fault"]: r for r in json.loads((OUT / "repair_study.json").read_text())}
    per_fault = {}
    for f in FAULTS:
        lt = [c for c in opus[f["name"]]["conditions"] if c["loc"] and c["trace"]][0]
        fb = frac_before[f["name"]]
        res = [residual_of(r, fb) for r in lt["runs"]]
        red = 100.0 * (fb - float(np.median(res))) / fb
        per_fault[f["name"]] = dict(mechanism=f["ftype"], reduction_pct=red,
                                    exact=(lt["n_verified"] == lt["seeds"]))

    # per-model div-removed for the cross-model table
    by_model = {}
    for tag, fname in MODELS:
        p = OUT / fname
        if not p.exists():
            continue
        byname = {r["fault"]: r for r in json.loads(p.read_text())}
        by_model[tag] = model_div_removed(byname, frac_before)
        m = by_model[tag]
        print(f"{tag:6} div removed (loc+trace): median {m['median']:.1f}% "
              f"[{m['lo']:.1f}, {m['hi']:.1f}]")

    reds = [r["reduction_pct"] for r in per_fault.values()]
    summary = dict(per_fault=per_fault, by_model=by_model,
                   median=float(np.median(reds)),
                   min=float(np.min(reds)), max=float(np.max(reds)))
    OUT.mkdir(exist_ok=True)
    (OUT / "repair_residual.json").write_text(json.dumps(summary, indent=2))
    print(f"Opus divergence removed: median {summary['median']:.1f}%  "
          f"range [{summary['min']:.1f}, {summary['max']:.1f}]%")


if __name__ == "__main__":
    main()
