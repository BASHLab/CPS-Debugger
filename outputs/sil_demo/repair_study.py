"""repair_study.py -- strengthened repair loop + the ablation that measures
what the reconstructed trace and branch localization are worth.

For each detected fault and each of four information conditions, a language
model proposes a fix; we score every proposal against ground truth (does the
repaired binary's replayed trace match the shipped binary's on the same logs).

Conditions (2x2):
  loc x trace, where
    localization = the model is told the faulted function (vs given the whole
                   control module and left to find it), and
    trace        = the model is given the reconstructed-trace evidence (the
                   debounce increment count, the guard crossing value, or the
                   inverted branch) AND may iterate against the
                   trace-consistency oracle (vs one shot from the symptom
                   alone, the situation on a machine with no recovered trace).

This isolates the value of localization from the value of the trace, and is
the machine half of the operator study (same task, same conditions).

CPU replay (cached by proposed value) + Anthropic calls. Writes
results/repair_study.json.
"""
import json
import os
import re
import struct
from pathlib import Path

import anthropic
import numpy as np

from replay import load_session, replay, compare, CODE, DATA
from run_corpus import patch_by_value

import argparse
MODEL = "claude-opus-4-8"
BACKEND = "anthropic"          # "anthropic" | "hf" (local open-source model)
MAX_ROUNDS = 4
PROBE = 8000                   # never-commit probe for counter measurement
OUT = Path(__file__).parent / "results"

# ---- fault registry ---------------------------------------------------------
# Source snippets are the deployed (buggy) form; the model never sees the
# shipped constant. Each fault carries a qualitative symptom (what an operator
# would observe) and the trace-derived evidence (the treatment).
FAULTS = [
    dict(name="balanced_counter", kind="i32", shipped=1500, seeded=150,
         func="stays_balanced", func_id=20, ftype="counter",
         symptom="After the pendulum balances, the controller advances to the "
                 "next position setpoint too soon, before the pendulum has "
                 "held steady long enough.",
         src="""bool stays_balanced(bool is_balanced) {
    if (is_balanced) {
        balanced_counter++;
        if (balanced_counter >= 150) return true;   // deployed constant
    } else balanced_counter = 0;
    return false;
}""",
         trace="The reconstructed trace shows the debounce counter reaching its "
               "bound after 1500 increments in fault-free logs, but after only "
               "150 in the flagged run."),
    dict(name="counter_goodrange", kind="i32", shipped=500, seeded=100,
         func="pou_lqr_sim", func_id=40, ftype="counter",
         symptom="The controller switches to its softer balancing mode earlier "
                 "than it should after the pendulum enters the good range.",
         src="""// lqrsim: dwell in the good range, then soften the LQR
if ((counter_goodrange < 100) && GOODRANGEMODE)   // deployed constant
    counter_goodrange++;
if (counter_goodrange >= 100) { /* switch to softer LQR */ }""",
         trace="The reconstructed trace shows the good-range dwell counter "
               "reaching its bound after 500 increments in fault-free logs, "
               "but after only 100 in the flagged run."),
    dict(name="theta_cap", kind="f64", shipped=0.07, seeded=0.20,
         func="pou_standup_rel", ftype="threshold",
         symptom="The controller declares the pendulum caught and starts "
                 "balancing while it is still well away from vertical.",
         src="""// standup_reliable: declare the pendulum caught near vertical
if ((fabs(theta_ist) < 0.20) && (fabs(theta_d_ist) < 0.05))  // deployed
    su_pndlum_ok = true;""",
         trace_tmpl="The reconstructed trace shows the balance-entry guard "
               "flipping when |theta| is about {c0} rad in fault-free logs, "
               "but about {c1} rad in the flagged run."),
    dict(name="theta_cmp", kind="opcode", offset=26760, shipped=0x63, seeded=0x64,
         func="pou_standup_rel", ftype="comparison", ops={"<": 0x63, ">": 0x64},
         symptom="The balance-entry test behaves backwards: the controller "
                 "treats large pendulum angles as caught and small angles as "
                 "not caught.",
         src="""// standup_reliable: balance-entry angle test (deployed)
if ((fabs(theta_ist) > 0.07) && (fabs(theta_d_ist) < 0.05))
    su_pndlum_ok = true;""",
         trace="The reconstructed trace shows pou_standup_rel taking the "
               "opposite branch of its angle comparison relative to fault-free "
               "logs, over most of the swing-up."),
]

# the three control functions in their CORRECT form; the no-localization
# search module shows all three with only the fault under test corrupted (its
# `src`), so the search space is coherent (exactly one bug).
FUNC_CLEAN = {
    "stays_balanced":
        "bool stays_balanced(bool is_balanced) {\n"
        "    if (is_balanced) {\n"
        "        balanced_counter++;\n"
        "        if (balanced_counter >= 1500) return true;\n"
        "    } else balanced_counter = 0;\n"
        "    return false;\n}",
    "pou_lqr_sim":
        "// lqrsim: dwell in the good range, then soften the LQR\n"
        "if ((counter_goodrange < 500) && GOODRANGEMODE)\n"
        "    counter_goodrange++;\n"
        "if (counter_goodrange >= 500) { /* switch to softer LQR */ }",
    "pou_standup_rel":
        "// standup_reliable: declare the pendulum caught near vertical\n"
        "if ((fabs(theta_ist) < 0.07) && (fabs(theta_d_ist) < 0.05))\n"
        "    su_pndlum_ok = true;",
}


def module_for(f):
    """The control module with every function correct except the one under
    test, which carries this fault's mutation."""
    return "\n\n".join(f["src"] if k == f["func"] else clean
                       for k, clean in FUNC_CLEAN.items())

BASE = Path(f"{CODE}/module.wasm").read_bytes()


def repaired_bytes(f, proposal):
    if f["kind"] == "opcode":
        b = bytearray(BASE)
        if b[f["offset"]] != f["shipped"]:
            raise ValueError("anchor byte moved")
        b[f["offset"]] = f["ops"][proposal]
        return bytes(b)
    val = int(proposal) if f["kind"] == "i32" else float(proposal)
    return patch_by_value(BASE, f["kind"], f["shipped"], val)


def verify(f, proposal, angle, x, cache):
    """frac_divergent of the repaired binary vs the shipped binary. Cached by
    (fault, proposal) so repeated proposals cost nothing."""
    key = (f["name"], str(proposal))
    if key in cache:
        return cache[key]
    mat = replay_cached(repaired_bytes(f, proposal), key, angle, x)
    frac = compare(NOM, mat)["frac_divergent"]
    cache[key] = frac
    return frac


# Shared (NFS) replay cache, so a compute-node sbatch job reuses matrices
# instead of recomputing every wasmtime replay on node-local /tmp.
CACHE = Path(__file__).parent / ".replay_cache"


def replay_cached(binary, key, angle, x):
    CACHE.mkdir(exist_ok=True)
    tag = f"{key[0]}_{re.sub('[^A-Za-z0-9]', '', key[1])}"
    p = CACHE / f"rs_{tag}.npy"
    if p.exists():
        return np.load(p)
    wasm = CACHE / f"rs_{tag}.wasm"
    wasm.write_bytes(binary)
    mat = replay(str(wasm), angle, x)
    np.save(p, mat)
    return mat


def ask_spec(f):
    if f["kind"] == "opcode":
        return ('{"operator": "<" or ">", "rationale": <string>}', "operator")
    return ('{"constant_new": <number>, "rationale": <string>}', "constant_new")


def build_prompt(f, loc, trace, history):
    fmt, _ = ask_spec(f)
    parts = [f"A deployed 1 kHz controller shows this symptom: {f['symptom']}"]
    if loc:
        parts.append(f"The fault is in `{f['func']}`, shown as deployed:\n"
                     f"```c\n{f['src']}\n```")
    else:
        parts.append("Here is the controller's control logic; find the "
                     f"responsible line:\n```c\n{module_for(f)}\n```")
    if trace:
        parts.append(f"Reconstructed-trace evidence: {f['trace']}")
    for h in history:
        if h.get("frac") is None:
            parts.append(f"Your previous fix ({h['proposal']}) could not be "
                         f"applied as a minimal in-place edit; propose a value of "
                         f"the same form and magnitude as the original constant.")
        else:
            parts.append(f"Your previous fix ({h['proposal']}) still left the "
                         f"reconstructed trace diverging from fault-free behavior on "
                         f"{h['frac']*100:.1f}% of ticks. Revise it.")
    parts.append(f"Propose the single minimal fix. Reply with one JSON object "
                 f"and nothing else: {fmt}.")
    return "\n\n".join(parts)


_HF = {}


def _hf_call(prompt):
    """Local open-source model via transformers (no cloud API), for the
    on-prem deployment arm."""
    import torch
    if "model" not in _HF:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _HF["tok"] = AutoTokenizer.from_pretrained(MODEL)
        _HF["model"] = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    tok, model = _HF["tok"], _HF["model"]
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt").to(model.device)
    out = model.generate(ids, max_new_tokens=512, do_sample=True,
                         temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def _one_call(prompt):
    if BACKEND == "hf":
        text = _hf_call(prompt)
    else:
        c = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        m = c.messages.create(model=MODEL, max_tokens=1024,
                              messages=[{"role": "user", "content": prompt}])
        text = "".join(b.text for b in m.content if b.type == "text")
    s = text.find("{"); depth = 0
    for i in range(s, len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            import ast
            blob = text[s:i + 1]
            try:
                return json.loads(blob)
            except json.JSONDecodeError:
                return ast.literal_eval(blob)
    raise ValueError(f"no json: {text[:150]}")


def call_llm(prompt, tries=5):
    """Retry with backoff so a transient API failure (rate limit, credit
    top-up gap) or a single unparseable local-model sample does not record a
    run as errored. A sampled model gets fresh draws on retry; only a
    persistent failure across all tries raises."""
    import time
    last = None
    for k in range(tries):
        try:
            return _one_call(prompt)
        except Exception as e:
            last = e
            time.sleep(2 ** k)
    raise last


def one_run(f, loc, trace, angle, x, cache):
    """One repair attempt: one shot when the trace is withheld (no oracle to
    iterate against), iterative up to MAX_ROUNDS when it is available.

    Two failure modes are kept distinct. A generation that fails after
    call_llm's retries is an infrastructure failure and ends the run. A
    well-formed proposal that cannot be applied as a minimal in-place edit
    (wrong type, or a value whose encoding is not layout-preserving) is a
    failed round: it is fed back so the oracle-guided loop can correct it, and
    it costs a round rather than the whole run. `final_frac` tracks the most
    recent applicable proposal, so a run that never produced one leaves it None
    and scores as zero divergence removed."""
    _, field = ask_spec(f)
    history, err = [], None
    last_proposal, last_frac = None, None
    for r in range(MAX_ROUNDS if trace else 1):
        try:
            fix = call_llm(build_prompt(f, loc, trace, history))
        except Exception as e:
            err = repr(e); break
        proposal = fix.get(field)
        try:
            frac = verify(f, proposal, angle, x, cache)
        except Exception as e:
            err = repr(e)
            history.append({"proposal": proposal, "frac": None})
            continue
        last_proposal, last_frac, err = proposal, frac, None
        history.append({"proposal": proposal, "frac": frac})
        if frac == 0.0:
            return dict(verified=True, rounds=r + 1, proposal=proposal)
    return dict(verified=False, rounds=len(history),
                error=err, proposal=last_proposal, final_frac=last_frac)


def run_condition(f, loc, trace, angle, x, cache, seeds):
    """Repeat the attempt over `seeds` LLM samples (temperature 1.0) and report
    the verified fraction, mirroring the 5-fold rigor used elsewhere."""
    runs = [one_run(f, loc, trace, angle, x, cache) for _ in range(seeds)]
    nver = sum(r["verified"] for r in runs)
    return dict(loc=loc, trace=trace, seeds=seeds, n_verified=nver,
                verified=(nver == seeds), majority=(nver * 2 > seeds), runs=runs)


def measured_value(f, angle, x, cache):
    """The value a no-LLM heuristic reads straight off the trace: a counter's
    fault-free commit count (increment edges before the commit branch, via a
    never-commit probe) or a threshold's measured crossing. A structural fault
    (wrong comparison) has no numeric value to read, so the heuristic has
    nothing to propose."""
    if f["ftype"] == "counter":
        probe = replay_cached(patch_by_value(BASE, f["kind"], f["shipped"], PROBE),
                              (f["name"], "probe"), angle, x)
        tick = compare(probe, NOM)["first_divergent_tick"]
        return None if tick < 0 else int((NOM[:tick, f["func_id"]] > 0).sum())
    if f["ftype"] == "threshold":
        return f.get("measured_c0")
    return None


def main():
    global NOM, MODEL, BACKEND
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--backend", default="anthropic", choices=["anthropic", "hf"])
    ap.add_argument("--tag", default="opus")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    MODEL = args.model
    BACKEND = args.backend
    outfile = OUT / (f"repair_study_{args.tag}.json" if args.tag != "opus"
                     else "repair_study.json")
    print(f"model={MODEL} -> {outfile.name}")
    angle, x, state = load_session(sorted(Path(DATA).glob("*.parquet"))[0].as_posix(), 0, 15000)
    CACHE.mkdir(exist_ok=True)
    nom_cache = CACHE / "nom.npy"
    if nom_cache.exists():
        NOM = np.load(nom_cache)
    elif Path("/tmp/repair_mat_nom.npy").exists():
        NOM = np.load("/tmp/repair_mat_nom.npy"); np.save(nom_cache, NOM)
    else:
        NOM = replay(f"{CODE}/module.wasm", angle, x); np.save(nom_cache, NOM)
    cache = {}

    # Fill threshold-fault trace evidence with the MEASURED guard-crossing
    # value (the sampled sensor reading where the balance controller first
    # engages), not the ground-truth constant, so the evidence never leaks the
    # answer. LQR onset is func 40.
    asig = np.abs(angle)
    for f in FAULTS:
        if "trace_tmpl" in f:
            mut = replay_cached(patch_by_value(BASE, f["kind"], f["shipped"], f["seeded"]),
                                (f["name"], "seeded"), angle, x)
            c0 = float(asig[int(np.argmax(NOM[:, 40] > 0))])
            c1 = float(asig[int(np.argmax(mut[:, 40] > 0))])
            f["measured_c0"] = round(c0, 3)
            f["trace"] = f["trace_tmpl"].format(c0=f"{c0:.3f}", c1=f"{c1:.3f}")
            print(f"[{f['name']}] measured crossing c0={c0:.3f} c1={c1:.3f}")

    results = []
    for f in FAULTS:
        row = {"fault": f["name"], "mechanism": f["ftype"], "conditions": []}
        # no-LLM heuristic: propose the value read straight off the trace
        mv = measured_value(f, angle, x, cache)
        if mv is None:
            row["heuristic"] = dict(verified=False, proposal=None,
                                    reason="no numeric value to read (structural fault)")
        else:
            frac = verify(f, mv, angle, x, cache)
            row["heuristic"] = dict(verified=(frac == 0.0), proposal=mv, frac=frac)
        print(f"{f['name']:18} HEURISTIC verified={row['heuristic']['verified']} "
              f"proposal={row['heuristic']['proposal']}")
        # LLM 2x2, `seeds` samples per cell
        for loc in (True, False):
            for trace in (True, False):
                res = run_condition(f, loc, trace, angle, x, cache, args.seeds)
                row["conditions"].append(res)
                print(f"{f['name']:18} loc={loc!s:5} trace={trace!s:5} "
                      f"verified={res['n_verified']}/{res['seeds']}")
        results.append(row)
        OUT.mkdir(exist_ok=True)
        outfile.write_text(json.dumps(results, indent=2))
    # summary: verified fraction per condition (sum over faults) + heuristic
    conds = {}
    for row in results:
        for c in row["conditions"]:
            k = f"loc={c['loc']},trace={c['trace']}"
            conds.setdefault(k, [0, 0])
            conds[k][0] += c["n_verified"]; conds[k][1] += c["seeds"]
    heur = sum(r["heuristic"]["verified"] for r in results)
    print("\nLLM verified (seed-runs) by condition:",
          {k: f"{v[0]}/{v[1]}" for k, v in conds.items()})
    print(f"heuristic (no LLM) exact repairs: {heur}/{len(results)}")


if __name__ == "__main__":
    main()
