"""
phase2_B2_build_eval_dataset.py — Build LLM evaluation dataset.

Constructs 500 physics-to-code reasoning problems from aligned_dataset.parquet.
Each problem gives physical sensor readings; ground truth is state, function,
target_x, and branch vector.

Output: outputs/phase2/llm_eval_dataset.jsonl
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/simran/allspark-data-exploration/CPS-Debugger")
OUT  = ROOT / "outputs/phase2"
EXP  = ROOT / "outputs/experiments"
OUT.mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)

STATE_NAMES    = {0: "SWINGUP", 1: "BALANCE", 2: "RESET"}
STATE_FUNCS    = {0: "pou_standup_rel", 1: "pou_lqr_sim", 2: "tick_only"}
PHYSICAL_COLS  = [
    "dl_current_x_mean", "dl_current_x_std", "dl_current_x_delta",
    "dl_angle_mean", "dl_angle_std", "dl_angle_delta",
    "dl_velocity_mean", "dl_ang_vel_mean", "dl_ang_vel_std",
]


def build_prompt(row, code_context, vocab, labels):
    """Build LLM prompt from one window row."""
    # Pick the most relevant source snippet: pend_monolithic.c (tick function)
    tick_src = code_context["source_files"].get("pend_monolithic.c", "")
    # Keep first 150 lines (main tick logic)
    tick_lines = tick_src.splitlines()[:150]
    tick_snippet = "\n".join(tick_lines)

    lqr_snippet = code_context["source_files"].get("lqrsim.c", "")
    standup_snippet = code_context["source_files"].get("standup_reliable.c", "")

    prompt = f"""\
You are analyzing an industrial inverted pendulum controlled by a Bosch Rexroth ctrlX system.
The controller is C code compiled to WebAssembly, running at 1 kHz (1 ms per tick() call).

SYSTEM DESCRIPTION:
{code_context["system_description"]}

KEY SOURCE CODE — tick() (main control loop):
```c
{tick_snippet}
```

KEY SOURCE CODE — pou_lqr_sim() (LQR balance controller):
```c
{lqr_snippet}
```

KEY SOURCE CODE — pou_standup_rel() (swing-up controller):
```c
{standup_snippet}
```

CURRENT PHYSICAL SENSOR READINGS (10 ms window average):
- Cart position (current_x):  {row['dl_current_x_mean']:.5f} m  (std: {row['dl_current_x_std']:.5f})
- Cart velocity:               {row['dl_velocity_mean']:.5f} m/s
- Pendulum angle:              {row['dl_angle_mean']:.5f} rad  (0=upright, ±π=down, std: {row['dl_angle_std']:.5f})
- Angular velocity:            {row['dl_ang_vel_mean']:.5f} rad/s  (std: {row['dl_ang_vel_std']:.5f})
- Cart position change (Δ):    {row['dl_current_x_delta']:.5f} m
- Angle change (Δ):            {row['dl_angle_delta']:.5f} rad

Based on the physical sensor readings and the source code, reason step by step:
1. What controller state is the system in? (SWINGUP=0, BALANCE=1, RESET=2)
2. Which function is being called each tick? (pou_standup_rel, pou_lqr_sim, or tick_only)
3. If in BALANCE state, what target cart position is the controller seeking?
   (Options: 0.0 m, +0.05 m, -0.09 m, +0.09 m — or null if not in BALANCE)
4. What is the approximate execution intensity? (Will memcpy run many iterations?
   Will the LQR solver take the position-correction or angle-correction branch?)

Respond ONLY with valid JSON in exactly this format:
{{
  "reasoning": "<your step-by-step reasoning referencing the code>",
  "predicted_state": <0, 1, or 2>,
  "predicted_function": "<pou_standup_rel|pou_lqr_sim|tick_only>",
  "predicted_target_x": <null or one of 0.0, 0.05, -0.09, 0.09>,
  "confidence": <0.0 to 1.0>
}}"""
    return prompt


def main():
    print("Loading data...")
    df = pd.read_parquet(EXP / "aligned_dataset.parquet")
    with open(EXP / "vocab_info.json") as f:
        vocab_info = json.load(f)
    with open(OUT / "code_context.json") as f:
        code_context = json.load(f)

    vocab  = vocab_info["vocab"]
    labels = vocab_info.get("labels", {})

    df = df[df[vocab].sum(axis=1) > 0].reset_index(drop=True)

    # Hold out the last run (2025-03-17_10-21-19) for held-out testing
    HOLDOUT_RUN = "2025-03-17_10-21-19"
    df_dev  = df[df["run"] != HOLDOUT_RUN].reset_index(drop=True)
    df_held = df[df["run"] == HOLDOUT_RUN].reset_index(drop=True)
    print(f"Dev runs: {len(df_dev):,} windows | Held-out run: {len(df_held):,} windows")

    # Detect transition windows (state changes within ±5 windows)
    states = df_dev["dl_state"].values
    is_transition = np.zeros(len(df_dev), dtype=bool)
    for i in range(5, len(df_dev) - 5):
        if states[i] != states[i-5] or states[i] != states[i+5]:
            is_transition[i] = True

    # Sample 500 windows stratified:
    #   50 SWINGUP, 350 BALANCE (50/target × 4 + 150 random), 50 RESET, 50 transitions
    samples = []

    def safe_sample(sub_df, n):
        n = min(n, len(sub_df))
        return sub_df.sample(n=n, random_state=42) if n > 0 else sub_df.iloc[:0]

    # SWINGUP
    sw  = df_dev[df_dev["dl_state"] == 0]
    samples.append(safe_sample(sw, 50))

    # BALANCE — stratified by target_x
    bal = df_dev[df_dev["dl_state"] == 1]
    for tx in [0.0, 0.05, -0.09, 0.09]:
        sub = bal[np.abs(bal["dl_target_x"] - tx) < 0.005]
        samples.append(safe_sample(sub, 50))
    samples.append(safe_sample(bal, 150))  # general BALANCE

    # RESET
    rs  = df_dev[df_dev["dl_state"] == 2]
    samples.append(safe_sample(rs, 50))

    # Transitions
    tr_df = df_dev[is_transition]
    samples.append(safe_sample(tr_df, 50))

    all_samples = pd.concat(samples).drop_duplicates().sample(
        frac=1, random_state=42).head(500).reset_index(drop=True)
    print(f"Sampled {len(all_samples)} evaluation windows")
    print(f"  State distribution: { {STATE_NAMES[s]: int((all_samples['dl_state']==s).sum()) for s in [0,1,2]} }")

    # Build JSONL
    out_path = OUT / "llm_eval_dataset.jsonl"
    n_written = 0
    with open(out_path, "w") as f:
        for _, row in all_samples.iterrows():
            state    = int(row["dl_state"])
            target_x = float(row["dl_target_x"])
            bvec     = [float(row[b]) for b in vocab]

            prompt = build_prompt(row, code_context, vocab, labels)
            gt = {
                "state":         state,
                "state_name":    STATE_NAMES[state],
                "function":      STATE_FUNCS[state],
                "target_x":      target_x if state == 1 else None,
                "branch_vector": bvec,
            }
            meta = {
                "run":         row["run"],
                "win":         int(row["win"]),
                "is_transition": bool(is_transition[row.name] if row.name < len(is_transition) else False),
            }
            record = {
                "data_source":  "cps_debug",
                "prompt":       [{"role": "user", "content": prompt}],
                "ground_truth": gt,
                "metadata":     meta,
            }
            f.write(json.dumps(record) + "\n")
            n_written += 1

    # Also build held-out test set (200 windows)
    held_samples = df_held.copy()
    held_samples = held_samples[held_samples[vocab].sum(axis=1) > 0]
    held_samples = held_samples.sample(
        n=min(200, len(held_samples)), random_state=42).reset_index(drop=True)

    out_held = OUT / "llm_eval_dataset_holdout.jsonl"
    with open(out_held, "w") as f:
        for _, row in held_samples.iterrows():
            state    = int(row["dl_state"])
            target_x = float(row["dl_target_x"])
            bvec     = [float(row[b]) for b in vocab]
            prompt   = build_prompt(row, code_context, vocab, labels)
            gt       = {
                "state": state, "state_name": STATE_NAMES[state],
                "function": STATE_FUNCS[state],
                "target_x": target_x if state == 1 else None,
                "branch_vector": bvec,
            }
            record = {
                "data_source":  "cps_debug",
                "prompt":       [{"role": "user", "content": prompt}],
                "ground_truth": gt,
                "metadata":     {"run": row["run"], "win": int(row["win"])},
            }
            f.write(json.dumps(record) + "\n")

    print(f"\nSaved {n_written} eval samples → {out_path}")
    print(f"Saved {len(held_samples)} held-out samples → {out_held}")


if __name__ == "__main__":
    main()
