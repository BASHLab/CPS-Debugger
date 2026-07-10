"""
task_3_1_build_anomaly_eval.py — Build 250-example anomaly explanation eval dataset.

Constructs 5 anomaly types × 50 examples from the existing 6-run aligned dataset.
Each example presents physical sensor readings + execution trace + controller code
and asks an LLM to identify and explain any inconsistencies.

Anomaly types:
  NORMAL       (50): Matched physical + trace — no anomaly
  TRACE_SWAP   (50): Physical from one state, trace from another state
  THRESHOLD_BUG(50): Cart near limit (0.15–0.17m) + RESET trace → premature reset
  TIMING_DELAY (50): Physical + trace misaligned by 50ms
  STUCK_SENSOR (50): Cart position frozen at constant while trace shows dynamic control

Output:
  outputs/phase3/anomaly_eval_dataset.jsonl
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT    = Path("/home/simran/allspark-data-exploration/CPS-Debugger")
P3_OUT  = ROOT / "outputs/phase3"
P3_OUT.mkdir(parents=True, exist_ok=True)

PHYSICAL_COLS = [
    "dl_current_x_mean", "dl_current_x_std", "dl_current_x_delta",
    "dl_angle_mean", "dl_angle_std", "dl_angle_delta",
    "dl_velocity_mean", "dl_ang_vel_mean", "dl_ang_vel_std",
]
STATE_NAMES = {0: "SWINGUP", 1: "BALANCE", 2: "RESET"}
N_PER_TYPE  = 50
SEED        = 42


def fmt_phys(row) -> dict:
    return {
        "cart_position_m":     round(float(row.get("dl_current_x_mean", 0)), 4),
        "cart_velocity_ms":    round(float(row.get("dl_velocity_mean", 0)), 5),
        "pendulum_angle_rad":  round(float(row.get("dl_angle_mean", 0)), 5),
        "angular_velocity_rads": round(float(row.get("dl_ang_vel_mean", 0)), 5),
        "cart_pos_std":        round(float(row.get("dl_current_x_std", 0)), 5),
        "angle_std":           round(float(row.get("dl_angle_std", 0)), 5),
        "cart_pos_delta_10ms": round(float(row.get("dl_current_x_delta", 0)), 5),
    }


def fmt_trace(row, vocab, fn_labels) -> dict:
    """Summarize the trace for a window: top-active branches by count."""
    active = {}
    for b in vocab:
        val = float(row.get(b, 0))
        if val > 0:
            label = fn_labels.get(b, b)
            active[label] = round(val, 1)
    # Sort by count descending, take top 20
    top = dict(sorted(active.items(), key=lambda x: -x[1])[:20])
    return top


def load_code_context() -> str:
    ctx_path = ROOT / "outputs/phase2/code_context.json"
    if ctx_path.exists():
        data = json.loads(ctx_path.read_text())
        # Return abbreviated version
        return data.get("abbreviated_code", data.get("code", ""))[:3000]
    return "(controller source code not available)"


def build_prompt(phys: dict, trace: dict, code_ctx: str) -> str:
    phys_str  = "\n".join(f"  {k}: {v}" for k, v in phys.items())
    trace_str = "\n".join(f"  {k}: {v} executions" for k, v in list(trace.items())[:15])
    return f"""You are analyzing a Bosch Rexroth ctrlX inverted-pendulum controller.
The controller runs at 1 kHz. Below are observations from one 10ms control window.

## Sensor readings (10ms average)
{phys_str}

## Execution trace (branch counts in this 10ms window)
{trace_str}

## Controller source code (abbreviated)
{code_ctx}

## Task
The cross-modal consistency checker has flagged this window. Analyze whether the
physical sensor readings and the execution trace are consistent with each other
and with the controller code.

Respond ONLY with valid JSON in this exact format:
{{
  "is_anomalous": true or false,
  "anomaly_type": "NORMAL" | "TRACE_SWAP" | "THRESHOLD_BUG" | "TIMING_DELAY" | "STUCK_SENSOR" | "OTHER",
  "physical_suggests": "one sentence: what state/behavior the sensors indicate",
  "trace_shows": "one sentence: what the execution trace indicates",
  "root_cause": "one sentence: the most likely cause of any inconsistency, or null if consistent",
  "implicated_code": "function or code component most likely responsible, or null",
  "severity": "normal" | "minor" | "moderate" | "critical",
  "reasoning": "step-by-step analysis (2-4 sentences)"
}}"""


def main():
    rng = random.Random(SEED)
    np.random.seed(SEED)

    # Load data
    data_path = ROOT / "outputs/experiments/aligned_dataset.parquet"
    if not data_path.exists():
        print("ERROR: aligned_dataset.parquet not found.")
        return
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} windows")

    vocab_info = json.loads((ROOT / "outputs/experiments/vocab_info.json").read_text())
    vocab      = vocab_info["vocab"]
    fn_labels  = vocab_info.get("labels", {})

    code_ctx = load_code_context()

    # State masks
    balance_mask = df["dl_state"] == 1
    swingup_mask = df["dl_state"] == 0
    reset_mask   = df["dl_state"] == 2

    print(f"BALANCE: {balance_mask.sum():,}  SWINGUP: {swingup_mask.sum():,}  "
          f"RESET: {reset_mask.sum():,}")

    examples = []

    # ── Type 1: NORMAL ────────────────────────────────────────────────────────
    print("\nBuilding NORMAL examples...")
    # Mix of states proportional to distribution (mostly BALANCE, some SWINGUP, few RESET)
    state_samples = {1: 35, 0: 10, 2: 5}
    for state, n in state_samples.items():
        sub = df[df["dl_state"] == state].sample(n=min(n, (df["dl_state"] == state).sum()),
                                                   random_state=SEED)
        for _, row in sub.iterrows():
            phys  = fmt_phys(row)
            trace = fmt_trace(row, vocab, fn_labels)
            examples.append({
                "anomaly_type": "NORMAL",
                "ground_truth": {
                    "is_anomalous": False,
                    "anomaly_type": "NORMAL",
                    "true_state": STATE_NAMES[state],
                    "implicated_code": None,
                    "severity": "normal",
                },
                "prompt": build_prompt(phys, trace, code_ctx),
                "_debug": {"run": row.get("run", ""), "win": int(row.get("win", 0)),
                           "state": state},
            })
    print(f"  Built {sum(1 for e in examples if e['anomaly_type']=='NORMAL')} NORMAL examples")

    # ── Type 2: TRACE_SWAP ────────────────────────────────────────────────────
    print("Building TRACE_SWAP examples...")
    # Primary: BALANCE physical + SWINGUP trace (cross-state swap)
    n_cross = min(N_PER_TYPE, balance_mask.sum())
    bal_rows     = df[balance_mask].sample(n=n_cross, random_state=SEED+1)
    swing_donors = df[swingup_mask].sample(n=n_cross, random_state=SEED+2, replace=True)

    for i, (_, phys_row) in enumerate(bal_rows.iterrows()):
        trace_row = swing_donors.iloc[i % len(swing_donors)]
        phys  = fmt_phys(phys_row)
        trace = fmt_trace(trace_row, vocab, fn_labels)
        examples.append({
            "anomaly_type": "TRACE_SWAP",
            "ground_truth": {
                "is_anomalous": True,
                "anomaly_type": "TRACE_SWAP",
                "true_physical_state": "BALANCE",
                "trace_from_state": "SWINGUP",
                "implicated_code": "tick() dispatch logic or pendulum_state variable",
                "severity": "critical",
            },
            "prompt": build_prompt(phys, trace, code_ctx),
            "_debug": {"phys_run": phys_row.get("run", ""), "trace_run": trace_row.get("run", "")},
        })

    # Supplement: cross-setpoint BALANCE swaps (cart at +0.09 paired with trace from -0.09)
    # These are subtler inconsistencies within BALANCE state
    n_current = sum(1 for e in examples if e["anomaly_type"] == "TRACE_SWAP")
    if n_current < N_PER_TYPE:
        pos_bal  = df[balance_mask & (df["dl_current_x_mean"] > 0.06)]
        neg_bal  = df[balance_mask & (df["dl_current_x_mean"] < -0.06)]
        n_extra  = N_PER_TYPE - n_current
        if len(pos_bal) >= n_extra and len(neg_bal) >= n_extra:
            phys_extra  = pos_bal.sample(n=n_extra, random_state=SEED+3)
            trace_extra = neg_bal.sample(n=n_extra, random_state=SEED+4)
            for (_, pr), (_, tr) in zip(phys_extra.iterrows(), trace_extra.iterrows()):
                examples.append({
                    "anomaly_type": "TRACE_SWAP",
                    "ground_truth": {
                        "is_anomalous": True,
                        "anomaly_type": "TRACE_SWAP",
                        "true_physical_state": "BALANCE",
                        "trace_from_state": "BALANCE_OPPOSITE_SETPOINT",
                        "implicated_code": "setpoint tracking or target_x register",
                        "severity": "moderate",
                    },
                    "prompt": build_prompt(fmt_phys(pr), fmt_trace(tr, vocab, fn_labels), code_ctx),
                    "_debug": {"phys_run": pr.get("run", ""), "trace_run": tr.get("run", "")},
                })
    print(f"  Built {sum(1 for e in examples if e['anomaly_type']=='TRACE_SWAP')} TRACE_SWAP examples")

    # ── Type 3: THRESHOLD_BUG ─────────────────────────────────────────────────
    print("Building THRESHOLD_BUG examples...")
    # BALANCE windows where cart is 0.15–0.17m (near real reset threshold)
    near_limit = df[balance_mask & (df["dl_current_x_mean"].abs() > 0.13)
                    & (df["dl_current_x_mean"].abs() < 0.17)]
    if len(near_limit) < N_PER_TYPE:
        # Widen range
        near_limit = df[balance_mask & (df["dl_current_x_mean"].abs() > 0.10)]
    if len(near_limit) < 5:
        # Synthesize by modifying existing rows
        base = df[balance_mask].sample(N_PER_TYPE, random_state=SEED+3)
        near_limit = base.copy()
        near_limit["dl_current_x_mean"] = np.random.uniform(0.15, 0.17, N_PER_TYPE)

    near_rows = near_limit.sample(n=min(N_PER_TYPE, len(near_limit)), random_state=SEED+3)
    reset_trace_donors = df[reset_mask].sample(n=len(near_rows), random_state=SEED+4,
                                                replace=(reset_mask.sum() < len(near_rows)))

    for i, ((_, phys_row), (_, trace_row)) in enumerate(
            zip(near_rows.iterrows(), reset_trace_donors.iterrows())):
        phys  = fmt_phys(phys_row)
        trace = fmt_trace(trace_row, vocab, fn_labels)
        examples.append({
            "anomaly_type": "THRESHOLD_BUG",
            "ground_truth": {
                "is_anomalous": True,
                "anomaly_type": "THRESHOLD_BUG",
                "true_physical_state": "BALANCE",
                "cart_position": float(phys_row.get("dl_current_x_mean", 0)),
                "trace_shows_reset": True,
                "implicated_code": "tick() reset threshold (MAX_X_SAFE constant)",
                "severity": "critical",
                "explanation": "Cart near track limit (0.15-0.17m) triggers RESET "
                               "trace — indicates reset threshold may have been lowered",
            },
            "prompt": build_prompt(phys, trace, code_ctx),
            "_debug": {},
        })
    print(f"  Built {sum(1 for e in examples if e['anomaly_type']=='THRESHOLD_BUG')} THRESHOLD_BUG examples")

    # ── Type 4: TIMING_DELAY ─────────────────────────────────────────────────
    print("Building TIMING_DELAY examples...")
    # Use BALANCE windows; pair physical row with trace from N_SHIFT windows later.
    # Collect all valid (i, i+shift) pairs across runs, then sample randomly.
    N_SHIFT = 5   # 5 windows = 50ms at 1kHz
    bal_sorted = df[balance_mask].sort_values(["run", "win"]).reset_index(drop=True)
    candidate_pairs = []  # list of (phys_idx, trace_idx) into bal_sorted
    for run_id, run_df in bal_sorted.groupby("run"):
        run_df = run_df.reset_index(drop=True)
        n = len(run_df)
        wins = run_df["win"].values
        # Only use pairs where win values are exactly N_SHIFT apart (truly consecutive)
        forward = wins[N_SHIFT:] - wins[:-N_SHIFT]
        valid = np.where(forward == N_SHIFT)[0]
        for vi in valid:
            candidate_pairs.append((run_df.index[vi], run_df.index[vi + N_SHIFT],
                                    run_id, int(wins[vi]), int(wins[vi + N_SHIFT])))

    # If fewer valid consecutive pairs than needed, relax to any N_SHIFT-apart windows
    if len(candidate_pairs) < N_PER_TYPE:
        candidate_pairs = []
        for run_id, run_df in bal_sorted.groupby("run"):
            run_df = run_df.reset_index(drop=True)
            wins = run_df["win"].values
            for vi in range(0, len(run_df) - N_SHIFT, max(1, (len(run_df) - N_SHIFT) // 200)):
                candidate_pairs.append((run_df.index[vi], run_df.index[vi + N_SHIFT],
                                        run_id, int(wins[vi]), int(wins[vi + N_SHIFT])))

    rng2 = np.random.RandomState(SEED + 10)
    chosen = rng2.choice(len(candidate_pairs),
                         size=min(N_PER_TYPE, len(candidate_pairs)),
                         replace=False)
    delay_examples = []
    for ci in chosen:
        pi, ti, run_id, win_p, win_t = candidate_pairs[ci]
        phys_row  = bal_sorted.iloc[pi]
        trace_row = bal_sorted.iloc[ti]
        delay_examples.append({
            "anomaly_type": "TIMING_DELAY",
            "ground_truth": {
                "is_anomalous": True,
                "anomaly_type": "TIMING_DELAY",
                "delay_ms": N_SHIFT * 10,
                "implicated_code": "data acquisition/synchronization pipeline",
                "severity": "moderate",
            },
            "prompt": build_prompt(fmt_phys(phys_row), fmt_trace(trace_row, vocab, fn_labels), code_ctx),
            "_debug": {"run": run_id, "win_phys": win_p, "win_trace": win_t},
        })
    examples.extend(delay_examples)
    print(f"  Built {len(delay_examples)} TIMING_DELAY examples")

    # ── Type 5: STUCK_SENSOR ─────────────────────────────────────────────────
    print("Building STUCK_SENSOR examples...")
    base = df[balance_mask].sample(n=N_PER_TYPE, random_state=SEED+5)
    for _, row in base.iterrows():
        phys  = fmt_phys(row)
        # Freeze cart position at a constant (0.0 or the mean for that run)
        frozen_val = round(rng.choice([-0.05, 0.0, 0.05, 0.1]), 2)
        phys_stuck = dict(phys)
        phys_stuck["cart_position_m"]     = frozen_val
        phys_stuck["cart_velocity_ms"]    = 0.0
        phys_stuck["cart_pos_std"]        = 0.0
        phys_stuck["cart_pos_delta_10ms"] = 0.0
        trace = fmt_trace(row, vocab, fn_labels)  # real, dynamic trace
        examples.append({
            "anomaly_type": "STUCK_SENSOR",
            "ground_truth": {
                "is_anomalous": True,
                "anomaly_type": "STUCK_SENSOR",
                "stuck_value": frozen_val,
                "implicated_code": "datalayer cart position sensor / ethercat read",
                "severity": "critical",
                "explanation": f"Cart position frozen at {frozen_val}m (zero std, zero velocity, "
                               "zero delta) while trace shows dynamic LQR control branches active",
            },
            "prompt": build_prompt(phys_stuck, trace, code_ctx),
            "_debug": {"run": row.get("run", ""), "win": int(row.get("win", 0))},
        })
    print(f"  Built {sum(1 for e in examples if e['anomaly_type']=='STUCK_SENSOR')} STUCK_SENSOR examples")

    # ── Shuffle and truncate to N_PER_TYPE per type ───────────────────────────
    from collections import defaultdict
    by_type = defaultdict(list)
    for e in examples:
        by_type[e["anomaly_type"]].append(e)

    final = []
    for t, items in by_type.items():
        rng.shuffle(items)
        final.extend(items[:N_PER_TYPE])
        print(f"  {t}: {len(items[:N_PER_TYPE])} examples")

    rng.shuffle(final)
    print(f"\nTotal eval examples: {len(final)}")

    # Save JSONL
    out_path = P3_OUT / "anomaly_eval_dataset.jsonl"
    with open(out_path, "w") as f:
        for ex in final:
            # Remove _debug for clean output
            clean = {k: v for k, v in ex.items() if k != "_debug"}
            f.write(json.dumps(clean) + "\n")

    print(f"Saved anomaly_eval_dataset.jsonl ({len(final)} examples)")
    print("Done.")


if __name__ == "__main__":
    main()
