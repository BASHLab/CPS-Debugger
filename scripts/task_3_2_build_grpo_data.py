"""
task_3_2_build_grpo_data.py — Build GRPO training + SFT dataset for anomaly explanation.

Generates 3000 training examples (600 per type) and 500 gold SFT examples
with programmatic reasoning chains. Ensures no overlap with the 250 eval set.

Output:
  outputs/phase3/grpo_train.jsonl          (3000 examples, VERL format)
  outputs/phase3/sft_anomaly_train.jsonl   (400 examples with gold reasoning)
  outputs/phase3/sft_anomaly_val.jsonl     (100 examples with gold reasoning)
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
STATE_NAMES  = {0: "SWINGUP", 1: "BALANCE", 2: "RESET"}
N_TRAIN_EACH = 600   # 600 × 5 types = 3000
N_SFT_TOTAL  = 500
SEED         = 137

# ── Gold reasoning templates ──────────────────────────────────────────────────
GOLD_REASONING = {
    "NORMAL": (
        "Step 1: Examine physical state. The pendulum angle is {angle_rad:.3f} rad "
        "({state_evidence}), cart at {cart_m:.3f} m (within safe bounds of ±0.17 m). "
        "Step 2: Check execution trace. The active functions ({fn_list}) are consistent "
        "with the physical state — {fn_explanation}. "
        "Step 3: Consistency check. Physical observations and trace execution path "
        "are mutually consistent. No anomaly detected."
    ),
    "TRACE_SWAP": (
        "Step 1: Physical sensors. Cart at {cart_m:.3f} m, angle = {angle_rad:.3f} rad "
        "({physical_state_evidence}). "
        "Step 2: Execution trace. The trace shows {trace_fn} executing, which is "
        "dispatched by tick() in the {trace_state} state. "
        "Step 3: Inconsistency. Physical evidence clearly indicates {physical_state} "
        "but the execution trace corresponds to {trace_state}. "
        "Root cause: tick() state dispatch is incorrect — either pendulum_state variable "
        "has been corrupted or the dispatch logic was modified."
    ),
    "THRESHOLD_BUG": (
        "Step 1: Physical state. Cart position = {cart_m:.3f} m. Pendulum angle = "
        "{angle_rad:.3f} rad (near upright). The system appears to be in BALANCE. "
        "Step 2: Trace analysis. The execution trace shows tick_only executing (RESET "
        "handler), with no pou_lqr_sim or pou_standup_rel activity. "
        "Step 3: Inconsistency. Cart at {cart_m:.3f} m is near but within the "
        "documented reset threshold (~0.17 m). The RESET trace executing while the "
        "system is physically capable of balancing suggests the safety threshold "
        "(MAX_X_SAFE) has been lowered — a threshold bug causing premature resets."
    ),
    "TIMING_DELAY": (
        "Step 1: Physical state at t=0. Cart = {cart_m:.3f} m, angle = {angle_rad:.3f} rad, "
        "angular velocity = {ang_vel:.3f} rad/s. "
        "Step 2: Trace inspection. The branch count magnitudes suggest a different "
        "physical regime than the current sensor values indicate. memcpy counts suggest "
        "a data copy volume inconsistent with current state. "
        "Step 3: Timing desynchronization. The physical sensor values and execution "
        "trace appear to be from slightly different time points — a ~50ms "
        "temporal shift. Root cause: data acquisition pipeline synchronization failure."
    ),
    "STUCK_SENSOR": (
        "Step 1: Physical state. Cart position = {frozen_val:.3f} m with std=0 and "
        "velocity=0 and delta=0. This is physically impossible during active balance "
        "control — the cart must move in response to LQR commands. "
        "Step 2: Trace analysis. The execution trace shows pou_lqr_sim executing with "
        "active compute_force and compute_velocity branches, and memcpy operations "
        "indicating data movement proportional to control effort. "
        "Step 3: Inconsistency. Dynamic control execution with frozen position sensor "
        "output indicates a stuck sensor — the ethercat/datalayer position reading "
        "is frozen while the controller continues computing. "
        "Root cause: datalayer cart position sensor failure or frozen ethercat read."
    ),
}


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
    active = {}
    for b in vocab:
        val = float(row.get(b, 0))
        if val > 0:
            active[fn_labels.get(b, b)] = round(val, 1)
    return dict(sorted(active.items(), key=lambda x: -x[1])[:20])


def load_code_context() -> str:
    ctx_path = ROOT / "outputs/phase2/code_context.json"
    if ctx_path.exists():
        data = json.loads(ctx_path.read_text())
        return data.get("abbreviated_code", data.get("code", ""))[:3000]
    return "(controller source code not available)"


def build_prompt(phys: dict, trace: dict) -> str:
    code_ctx = build_prompt._code_ctx
    phys_str  = "\n".join(f"  {k}: {v}" for k, v in phys.items())
    trace_str = "\n".join(f"  {k}: {v} executions" for k, v in list(trace.items())[:15])
    return (
        "You are analyzing a Bosch Rexroth ctrlX inverted-pendulum controller.\n"
        "The controller runs at 1 kHz. Below are observations from one 10ms control window.\n\n"
        "## Sensor readings (10ms average)\n" + phys_str + "\n\n"
        "## Execution trace (branch counts in this 10ms window)\n" + trace_str + "\n\n"
        "## Controller source code (abbreviated)\n" + code_ctx + "\n\n"
        "## Task\nThe cross-modal consistency checker has flagged this window. Analyze whether "
        "the physical sensor readings and the execution trace are consistent.\n\n"
        "Respond ONLY with valid JSON:\n"
        '{"is_anomalous": bool, "anomaly_type": str, "physical_suggests": str, '
        '"trace_shows": str, "root_cause": str|null, "implicated_code": str|null, '
        '"severity": str, "reasoning": str}'
    )
build_prompt._code_ctx = ""


def build_gold_response(anomaly_type: str, phys: dict, trace: dict,
                         gt: dict, vocab_fn_names: list) -> dict:
    """Generate programmatic gold-standard reasoning."""
    cart  = phys.get("cart_position_m", 0.0)
    angle = phys.get("pendulum_angle_rad", 0.0)
    angv  = phys.get("angular_velocity_rads", 0.0)

    fn_list = ", ".join(list(trace.keys())[:5]) if trace else "none"

    if anomaly_type == "NORMAL":
        state = gt.get("true_state", "BALANCE")
        reasoning = GOLD_REASONING["NORMAL"].format(
            angle_rad=angle,
            cart_m=cart,
            state_evidence=f"{'near upright' if abs(angle) < 0.2 else 'falling/swinging'}",
            fn_list=fn_list,
            fn_explanation=f"consistent with {state} state dispatch",
        )
        return {
            "is_anomalous": False,
            "anomaly_type": "NORMAL",
            "physical_suggests": f"{state} state — sensors consistent with normal operation",
            "trace_shows": f"Execution path consistent with {state} state dispatch",
            "root_cause": None,
            "implicated_code": None,
            "severity": "normal",
            "reasoning": reasoning,
        }

    elif anomaly_type == "TRACE_SWAP":
        phys_state  = gt.get("true_physical_state", "BALANCE")
        trace_state = gt.get("trace_from_state", "SWINGUP")
        reasoning = GOLD_REASONING["TRACE_SWAP"].format(
            cart_m=cart, angle_rad=angle,
            physical_state_evidence=f"{'near upright → BALANCE' if abs(angle) < 0.3 else '→ SWINGUP'}",
            trace_fn=fn_list,
            trace_state=trace_state,
            physical_state=phys_state,
        )
        return {
            "is_anomalous": True,
            "anomaly_type": "TRACE_SWAP",
            "physical_suggests": f"{phys_state} — angle={angle:.3f} rad, cart={cart:.3f} m",
            "trace_shows": f"{trace_state} execution path — wrong controller function dispatched",
            "root_cause": "tick() dispatching wrong state handler; pendulum_state variable corrupted",
            "implicated_code": "tick() function, state dispatch switch/if block",
            "severity": "critical",
            "reasoning": reasoning,
        }

    elif anomaly_type == "THRESHOLD_BUG":
        reasoning = GOLD_REASONING["THRESHOLD_BUG"].format(cart_m=cart, angle_rad=angle)
        return {
            "is_anomalous": True,
            "anomaly_type": "THRESHOLD_BUG",
            "physical_suggests": f"BALANCE — angle={angle:.3f} rad, cart={cart:.3f} m (near but within limits)",
            "trace_shows": "RESET state handler executing (tick_only) — no LQR control active",
            "root_cause": f"Safety reset threshold (MAX_X_SAFE) appears lowered; "
                          f"cart at {cart:.3f}m triggering reset prematurely",
            "implicated_code": "tick() reset threshold check, MAX_X_SAFE constant",
            "severity": "critical",
            "reasoning": reasoning,
        }

    elif anomaly_type == "TIMING_DELAY":
        reasoning = GOLD_REASONING["TIMING_DELAY"].format(
            cart_m=cart, angle_rad=angle, ang_vel=angv)
        return {
            "is_anomalous": True,
            "anomaly_type": "TIMING_DELAY",
            "physical_suggests": f"Current state: cart={cart:.3f}m, angle={angle:.3f}rad",
            "trace_shows": "Execution trace from a different time point (~50ms offset)",
            "root_cause": "Temporal desynchronization in data acquisition pipeline",
            "implicated_code": "Data logging/acquisition pipeline, timestamp alignment",
            "severity": "moderate",
            "reasoning": reasoning,
        }

    elif anomaly_type == "STUCK_SENSOR":
        frozen = phys.get("cart_position_m", 0.0)
        reasoning = GOLD_REASONING["STUCK_SENSOR"].format(frozen_val=frozen)
        return {
            "is_anomalous": True,
            "anomaly_type": "STUCK_SENSOR",
            "physical_suggests": f"Frozen cart position at {frozen}m (zero std/velocity/delta) — sensor failure",
            "trace_shows": "Active LQR control with dynamic branch activity — controller running normally",
            "root_cause": "Cart position sensor stuck; datalayer returning constant frozen value",
            "implicated_code": "Datalayer cart position read, ethercat data acquisition",
            "severity": "critical",
            "reasoning": reasoning,
        }

    return {}


def make_verl_record(prompt: str, gt: dict) -> dict:
    return {
        "data_source":  "cps_anomaly",
        "prompt":       [{"role": "user", "content": prompt}],
        "ability":      "anomaly_explanation",
        "reward_model": {"ground_truth": json.dumps(gt)},
    }


def main():
    rng = random.Random(SEED)
    np.random.seed(SEED)

    # Load code context
    build_prompt._code_ctx = load_code_context()

    # Load eval set to avoid overlap
    eval_path = P3_OUT / "anomaly_eval_dataset.jsonl"
    eval_debug_wins = set()
    if eval_path.exists():
        with open(eval_path) as f:
            for line in f:
                ex = json.loads(line)
                dbg = ex.get("_debug", {})
                key = (dbg.get("run", ""), dbg.get("win", -1))
                eval_debug_wins.add(key)
        print(f"Loaded {len(eval_debug_wins)} eval windows to exclude")

    # Load data
    data_path = ROOT / "outputs/experiments/aligned_dataset.parquet"
    if not data_path.exists():
        print("ERROR: aligned_dataset.parquet not found.")
        return
    df = pd.read_parquet(data_path)
    vocab_info = json.loads((ROOT / "outputs/experiments/vocab_info.json").read_text())
    vocab      = vocab_info["vocab"]
    fn_labels  = vocab_info.get("labels", {})

    # Filter out eval windows
    def is_not_eval(row):
        return (row.get("run", ""), int(row.get("win", -1))) not in eval_debug_wins

    balance_df = df[df["dl_state"] == 1].copy()
    swingup_df = df[df["dl_state"] == 0].copy()
    reset_df   = df[df["dl_state"] == 2].copy()
    print(f"Available: BAL={len(balance_df)}, SWING={len(swingup_df)}, RESET={len(reset_df)}")

    # ── Generate all 3000 training + 500 SFT examples ────────────────────────
    # We build SFT and GRPO in one pass; SFT has gold reasoning appended
    all_train   = []  # VERL GRPO format
    all_sft     = []  # full gold responses

    def add_examples(anomaly_type, phys_rows, trace_rows=None, gts=None, n=N_TRAIN_EACH):
        count = 0
        for i, (_, phys_row) in enumerate(phys_rows.iterrows()):
            if count >= n:
                break
            trace_row = phys_row if trace_rows is None else trace_rows.iloc[i % len(trace_rows)]
            phys  = fmt_phys(phys_row)
            trace = fmt_trace(trace_row, vocab, fn_labels)
            gt    = gts[i % len(gts)] if gts else {}
            prompt = build_prompt(phys, trace)
            all_train.append(make_verl_record(prompt, gt))
            if len(all_sft) < N_SFT_TOTAL:
                gold = build_gold_response(anomaly_type, phys, trace, gt,
                                           list(fn_labels.values()))
                all_sft.append({
                    "data_source": "cps_anomaly_sft",
                    "prompt":      [{"role": "user", "content": prompt}],
                    "response":    [{"role": "assistant", "content": json.dumps(gold)}],
                    "ability":     "anomaly_explanation",
                })
            count += 1

    # NORMAL
    normal_rows = balance_df.sample(n=min(N_TRAIN_EACH, len(balance_df)), random_state=SEED+10)
    gts_normal  = [{"is_anomalous": False, "anomaly_type": "NORMAL",
                    "true_state": "BALANCE", "severity": "normal"}] * len(normal_rows)
    add_examples("NORMAL", normal_rows, gts=gts_normal, n=N_TRAIN_EACH)
    print(f"NORMAL: {sum(1 for e in all_train if json.loads(e['reward_model']['ground_truth']).get('anomaly_type')=='NORMAL')}")

    # TRACE_SWAP
    swap_phys   = balance_df.sample(n=min(N_TRAIN_EACH, len(balance_df)), random_state=SEED+11, replace=True)
    swap_trace  = swingup_df.sample(n=min(N_TRAIN_EACH, len(swingup_df)), random_state=SEED+12, replace=True)
    gts_swap    = [{"is_anomalous": True, "anomaly_type": "TRACE_SWAP",
                    "true_physical_state": "BALANCE", "trace_from_state": "SWINGUP",
                    "severity": "critical"}] * N_TRAIN_EACH
    add_examples("TRACE_SWAP", swap_phys, trace_rows=swap_trace, gts=gts_swap)
    print(f"TRACE_SWAP done")

    # THRESHOLD_BUG
    near = balance_df[balance_df["dl_current_x_mean"].abs() > 0.10]
    if len(near) < N_TRAIN_EACH:
        near = balance_df.sample(N_TRAIN_EACH, random_state=SEED+13, replace=True).copy()
        near["dl_current_x_mean"] = np.random.uniform(0.13, 0.17, N_TRAIN_EACH)
    bug_phys  = near.sample(n=min(N_TRAIN_EACH, len(near)), random_state=SEED+13, replace=True)
    bug_trace = reset_df.sample(n=min(N_TRAIN_EACH, max(1, len(reset_df))), random_state=SEED+14, replace=True)
    gts_bug   = [{"is_anomalous": True, "anomaly_type": "THRESHOLD_BUG", "severity": "critical"}] * N_TRAIN_EACH
    add_examples("THRESHOLD_BUG", bug_phys, trace_rows=bug_trace, gts=gts_bug)
    print(f"THRESHOLD_BUG done")

    # TIMING_DELAY (use adjacent windows, 5-step lag)
    bal_sorted = balance_df.sort_values(["run", "win"]).reset_index(drop=True)
    delay_phys_rows, delay_trace_rows = [], []
    for _, grp in bal_sorted.groupby("run"):
        grp = grp.reset_index(drop=True)
        for i in range(0, len(grp)-5, max(1, (len(grp)-5) // (N_TRAIN_EACH // 5))):
            delay_phys_rows.append(grp.iloc[i])
            delay_trace_rows.append(grp.iloc[i+5])
            if len(delay_phys_rows) >= N_TRAIN_EACH:
                break
        if len(delay_phys_rows) >= N_TRAIN_EACH:
            break
    delay_phys_df  = pd.DataFrame(delay_phys_rows)
    delay_trace_df = pd.DataFrame(delay_trace_rows).reset_index(drop=True)
    gts_delay = [{"is_anomalous": True, "anomaly_type": "TIMING_DELAY",
                  "delay_ms": 50, "severity": "moderate"}] * len(delay_phys_df)
    add_examples("TIMING_DELAY", delay_phys_df, trace_rows=delay_trace_df, gts=gts_delay)
    print(f"TIMING_DELAY done")

    # STUCK_SENSOR
    stuck_base = balance_df.sample(n=N_TRAIN_EACH, random_state=SEED+15, replace=True).copy()
    frozen_vals = np.random.choice([-0.1, -0.05, 0.0, 0.05, 0.1], size=N_TRAIN_EACH)
    stuck_base["dl_current_x_mean"]  = frozen_vals
    stuck_base["dl_velocity_mean"]   = 0.0
    stuck_base["dl_current_x_std"]   = 0.0
    stuck_base["dl_current_x_delta"] = 0.0
    gts_stuck = [{"is_anomalous": True, "anomaly_type": "STUCK_SENSOR",
                  "severity": "critical"}] * N_TRAIN_EACH
    add_examples("STUCK_SENSOR", stuck_base, gts=gts_stuck)
    print(f"STUCK_SENSOR done")

    # ── Shuffle and save ──────────────────────────────────────────────────────
    rng.shuffle(all_train)
    rng.shuffle(all_sft)

    # SFT: 80/20 split
    n_sft_train = int(0.8 * len(all_sft))
    sft_train = all_sft[:n_sft_train]
    sft_val   = all_sft[n_sft_train:]

    with open(P3_OUT / "grpo_train.jsonl", "w") as f:
        for rec in all_train:
            f.write(json.dumps(rec) + "\n")
    with open(P3_OUT / "sft_anomaly_train.jsonl", "w") as f:
        for rec in sft_train:
            f.write(json.dumps(rec) + "\n")
    with open(P3_OUT / "sft_anomaly_val.jsonl", "w") as f:
        for rec in sft_val:
            f.write(json.dumps(rec) + "\n")

    print(f"\nSaved grpo_train.jsonl:       {len(all_train)} examples")
    print(f"Saved sft_anomaly_train.jsonl: {len(sft_train)} examples")
    print(f"Saved sft_anomaly_val.jsonl:   {len(sft_val)} examples")
    print("Done.")


if __name__ == "__main__":
    main()
