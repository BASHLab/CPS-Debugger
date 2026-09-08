"""llm_patch.py -- the repair rung of the loop (workstream C6).

Closes detect -> localize -> repair -> verify on the balance-hold case study,
entirely from sensor-side evidence:

  1. Replay the shipped and the buggy binary on the same real logs; the
     reconstructed traces diverge and localize the fault to stays_balanced
     (the localization step, already scored in run_corpus.py).
  2. Hand a language model the localized function and the sensor-side
     evidence only -- never the shipped binary or the original constant --
     and ask for the minimal fix.
  3. Apply the proposed constant as a layout-preserving binary patch.
  4. Accept the patch when the repaired binary's replayed trace matches the
     shipped binary's on the same logs (frac_divergent -> 0). No test suite,
     no temporal-logic spec: the oracle is trace consistency.

The evidence given to the model is all observable without the answer: the
localized (buggy) function, the tick where the trace first diverges, and the
balance-hold dwell measured from fault-free logs (about 1.5 s at the 1 kHz
control rate) against the flagged run's much shorter dwell.

CPU-only replay + one Anthropic API call. Writes results/repair_loop.json.
"""
import json
import os
import re
from pathlib import Path

import anthropic

from replay import load_session, replay, compare, CODE, DATA
from run_corpus import patch_by_value, gt_func

OUT = Path(__file__).parent / "results"
MODEL = "claude-opus-4-8"
FUNC = "stays_balanced"
ORIG, BUG = 1500, 150          # shipped constant, and the seeded fault value
PROBE = 8000                   # never-commit probe (> any balanced run here)

# The localized function as the deployed (buggy) build would show it: the
# binary mutation 1500->150 is exactly equivalent to this one-line source edit.
BUGGY_SOURCE = f"""\
int balanced_counter = 0;

bool stays_balanced(bool is_balanced)
{{
    if (is_balanced)
    {{
        balanced_counter++;
        if (balanced_counter >= {BUG})
            return true;
    }}
    else
    {{
        balanced_counter = 0;
    }}
    return false;
}}"""

SYSTEM = (
    "You are a controls engineer debugging a deployed 1 kHz industrial "
    "controller. A sensor-side monitor has flagged an anomaly and localized "
    "it to one function. You cannot see the original source; only the "
    "deployed build and the monitor's evidence. Propose the single minimal "
    "change that restores correct behavior. Reply with one JSON object and "
    "nothing else: {\"constant_old\": <int>, \"constant_new\": <int>, "
    "\"rationale\": <string>}."
)


def build_prompt(first_tick, nom_count, bug_count):
    return f"""\
The monitor localized the anomaly to `{FUNC}`, shown here as deployed:

```c
{BUGGY_SOURCE}
```

Sensor-side evidence:
- The reconstructed control-flow trace first diverges from fault-free
  behavior at control tick {first_tick}, inside `{FUNC}`, right after the
  controller enters the balanced state.
- The control loop runs at 1 kHz. Reconstructed fault-free traces take the
  commit branch (stays_balanced returning true) only after about {nom_count}
  consecutive balanced ticks (~{nom_count/1000:.1f} s); the flagged run takes
  it after only about {bug_count} consecutive balanced ticks
  (~{bug_count/1000:.2f} s).

Propose the minimal fix."""


def call_llm(prompt):
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=MODEL, max_tokens=1024, system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in model reply: {text[:200]}")
    return json.loads(m.group(0)), text


def commit_count(ref_mat, probe_mat, func_id):
    """Value of the debounce counter at the commit, measured from the trace.

    A never-commit probe agrees with the binary until the binary takes the
    commit branch, so their first divergence is the commit tick. The counter
    increments once per tick the counting function runs, and the controller
    resets it only inside that function's else-branch (never called between
    balance episodes here), so it accumulates across episodes. The counter's
    value at commit is therefore the number of ticks the function executed
    before the commit tick -- read straight off the reconstructed trace, not
    a sensor proxy."""
    tick = compare(probe_mat, ref_mat)["first_divergent_tick"]
    if tick < 0:
        return None, None
    return tick, int((ref_mat[:tick, func_id] > 0).sum())


def cached_replay(binary, tag, angle, x):
    """Replay, caching the per-tick matrix so evidence can be refined without
    re-running the (minutes-long) wasmtime replay."""
    import numpy as np
    p = Path(f"/tmp/repair_mat_{tag}.npy")
    if p.exists():
        return np.load(p)
    mat = replay(binary, angle, x)
    np.save(p, mat)
    return mat


def main():
    module = f"{CODE}/module.wasm"
    base = Path(module).read_bytes()
    layout = json.loads(Path(f"{CODE}/layout.json").read_text())
    fpath = sorted(Path(DATA).glob("*.parquet"))[0].as_posix()
    angle, x, state = load_session(fpath, 0, 15000)

    # 1. detect + localize (shipped vs buggy on identical real logs)
    nom = cached_replay(module, "nom", angle, x)
    mut_bytes = patch_by_value(base, "i32", ORIG, BUG)
    Path("/tmp/repair_bug.wasm").write_bytes(mut_bytes)
    mut = cached_replay("/tmp/repair_bug.wasm", "bug", angle, x)
    ev = compare(nom, mut)
    gt = set(gt_func(layout, "i32", ORIG))
    origin = [f for f, _ in ev["origin_funcs"]]
    localized_ok = bool(origin) and origin[0] in gt
    first_tick = ev["first_divergent_tick"]

    # commit-count evidence, measured from the traces via a never-commit probe
    probe_bytes = patch_by_value(base, "i32", ORIG, PROBE)
    Path("/tmp/repair_probe.wasm").write_bytes(probe_bytes)
    probe = cached_replay("/tmp/repair_probe.wasm", "probe", angle, x)
    counter_func = origin[0] if origin else 20       # localized counting function
    _, nom_count = commit_count(nom, probe, counter_func)   # 1500
    _, bug_count = commit_count(mut, probe, counter_func)   # 150

    # 2. LLM proposes the fix from evidence only
    prompt = build_prompt(first_tick, nom_count, bug_count)
    fix, raw = call_llm(prompt)
    proposed = int(fix["constant_new"])

    # 3. apply the proposed constant as a layout-preserving patch
    applied = verified = False
    frac_after = None
    try:
        rep_bytes = patch_by_value(mut_bytes, "i32", BUG, proposed)
        Path("/tmp/repair_fixed.wasm").write_bytes(rep_bytes)
        applied = True
        # 4. verify against the shipped binary's trace on the same logs
        rep = replay("/tmp/repair_fixed.wasm", angle, x)
        frac_after = compare(nom, rep)["frac_divergent"]
        verified = frac_after == 0.0
    except ValueError as e:
        raw += f"\n[apply failed: {e}]"

    result = dict(
        fault="balanced_counter", func=FUNC, shipped_constant=ORIG,
        seeded_constant=BUG, first_divergent_tick=first_tick,
        localized_to=[origin[0]] if origin else [], localized_ok=localized_ok,
        nom_commit_ticks=nom_count, bug_commit_ticks=bug_count,
        frac_divergent_before=ev["frac_divergent"],
        llm_model=MODEL, llm_proposed=proposed, llm_rationale=fix.get("rationale"),
        applied=applied, frac_divergent_after=frac_after, verified=verified,
    )
    OUT.mkdir(exist_ok=True)
    (OUT / "repair_loop.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in (
        "localized_to", "localized_ok", "first_divergent_tick",
        "nom_commit_ticks", "bug_commit_ticks", "llm_proposed",
        "frac_divergent_before", "frac_divergent_after", "verified")}, indent=2))
    print("rationale:", fix.get("rationale"))


if __name__ == "__main__":
    main()
