"""
phase2_C1_build_sft_dataset.py — Build SFT warmup dataset for GRPO training.

Generates 2000 programmatic instruction-response pairs from aligned_dataset.parquet.
Gold responses are generated from actual labels (not hand-written).

Output: outputs/phase2/sft_dataset.jsonl  (1600 train + 400 val)
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

random.seed(0)
np.random.seed(0)

STATE_NAMES = {0: "SWINGUP", 1: "BALANCE", 2: "RESET"}
STATE_FUNCS = {0: "pou_standup_rel", 1: "pou_lqr_sim", 2: "tick_only"}

TARGET_X_DESCRIPTIONS = {
    0.0:   "center (0.00 m)",
    0.05:  "right-center (+0.05 m)",
    -0.09: "left edge (−0.09 m)",
    0.09:  "right edge (+0.09 m)",
}


def nearest_target(tx):
    candidates = [0.0, 0.05, -0.09, 0.09]
    return min(candidates, key=lambda t: abs(t - tx))


def generate_reasoning(row, state, func, target_x, vocab, labels):
    """Generate a gold reasoning chain from actual sensor values and labels."""
    x    = row["dl_current_x_mean"]
    vx   = row["dl_velocity_mean"]
    ang  = row["dl_angle_mean"]
    w    = row["dl_ang_vel_mean"]
    dang = row["dl_angle_delta"]

    near_upright = abs(ang) < 0.5
    at_limit     = abs(x) > 0.14
    fast_rotation = abs(w) > 2.0

    if state == 0:
        reasoning = (
            f"The pendulum angle is {ang:.4f} rad. "
            f"{'This is close to ±π (pendulum hanging down)' if abs(ang) > 2.5 else 'The pendulum is mid-swing and not yet upright'}. "
            f"The angular velocity is {w:.4f} rad/s"
            f"{', suggesting rapid rotation toward upright' if fast_rotation else ', indicating slow rotation'}. "
            f"The cart position is {x:.4f} m and velocity is {vx:.4f} m/s. "
            f"In SWINGUP state, tick() calls pou_standup_rel() each cycle, which "
            f"uses a velocity-based energy pumping strategy to swing the pendulum upright. "
            f"No target position applies in this state."
        )
    elif state == 1:
        tx_desc = TARGET_X_DESCRIPTIONS.get(nearest_target(target_x), f"{target_x:.2f} m")
        pos_err  = x - target_x
        ang_err  = ang
        reasoning = (
            f"The pendulum angle is {ang:.4f} rad, which is near zero (upright), "
            f"confirming BALANCE state. "
            f"The cart is at {x:.4f} m with target position {tx_desc}. "
            f"Position error = {pos_err:.4f} m; angle error = {ang_err:.4f} rad. "
            f"In BALANCE, tick() calls pou_lqr_sim(), which computes LQR control gains "
            f"u = K * [pos_err, vel, angle, ang_vel]. "
            f"{'The angle error dominates — the controller is primarily correcting the pendulum tilt.' if abs(ang_err) > abs(pos_err)*0.5 else 'The position error is significant — the controller is moving the cart toward the target.'} "
            f"memcpy will execute to copy the state vector for the LQR computation; "
            f"iteration count depends on the size of the data transfer, which is constant per state but "
            f"the branch taken within memcpy depends on the alignment of the data buffer."
        )
    else:  # RESET
        reasoning = (
            f"The cart position is {x:.4f} m, which is {'near the track limit' if at_limit else 'returning from track limit'}. "
            f"When the cart reaches the track limits (±0.15–0.18 m), the controller "
            f"enters RESET state to prevent mechanical damage. "
            f"In RESET, tick() handles the return motion directly without calling "
            f"pou_standup_rel or pou_lqr_sim — it drives the cart back to center at low velocity. "
            f"The pendulum angle {ang:.4f} rad may be anywhere during this maneuver."
        )

    return reasoning


def build_prompt(row, code_context):
    """Build the user prompt (same format as B2, but shorter code snippet)."""
    tick_lines = code_context["source_files"].get("pend_monolithic.c", "").splitlines()[:100]
    tick_snippet = "\n".join(tick_lines)

    return (
        f"You are analyzing an industrial inverted pendulum controller.\n\n"
        f"KEY SOURCE CODE (tick function excerpt):\n```c\n{tick_snippet}\n```\n\n"
        f"SENSOR READINGS (10 ms window):\n"
        f"- Cart position: {row['dl_current_x_mean']:.5f} m\n"
        f"- Cart velocity: {row['dl_velocity_mean']:.5f} m/s\n"
        f"- Pendulum angle: {row['dl_angle_mean']:.5f} rad\n"
        f"- Angular velocity: {row['dl_ang_vel_mean']:.5f} rad/s\n\n"
        f"Reason step by step and output JSON:\n"
        f'{{"reasoning": "...", "predicted_state": 0|1|2, '
        f'"predicted_function": "...", "predicted_target_x": null|0.0|0.05|-0.09|0.09, '
        f'"confidence": 0.0-1.0}}'
    )


def main():
    print("Loading data...")
    df = pd.read_parquet(EXP / "aligned_dataset.parquet")
    with open(EXP / "vocab_info.json") as f:
        vocab_info = json.load(f)
    with open(OUT / "code_context.json") as f:
        code_context = json.load(f)

    vocab  = vocab_info["vocab"]
    labels = vocab_info.get("labels", {})

    # Exclude held-out run
    HOLDOUT_RUN = "2025-03-17_10-21-19"
    df = df[df["run"] != HOLDOUT_RUN].reset_index(drop=True)
    df = df[df[vocab].sum(axis=1) > 0].reset_index(drop=True)

    # Sample 2000 windows, stratified
    samples = []
    for state in [0, 1, 2]:
        sub = df[df["dl_state"] == state]
        n   = {0: 200, 1: 1600, 2: 200}[state]
        n   = min(n, len(sub))
        if state == 1:
            # Stratify by target_x within BALANCE
            for tx in [0.0, 0.05, -0.09, 0.09]:
                tx_sub = sub[np.abs(sub["dl_target_x"] - tx) < 0.005]
                samples.append(tx_sub.sample(n=min(400, len(tx_sub)), random_state=42))
        else:
            samples.append(sub.sample(n=n, random_state=42))

    df_sft = pd.concat(samples).drop_duplicates().sample(
        n=min(2000, sum(len(s) for s in samples)),
        random_state=42).reset_index(drop=True)
    print(f"SFT dataset: {len(df_sft)} windows")
    print(f"  State distribution: { {STATE_NAMES[s]: int((df_sft['dl_state']==s).sum()) for s in [0,1,2]} }")

    # Split 80/20
    n_train = int(len(df_sft) * 0.8)
    df_train = df_sft.iloc[:n_train]
    df_val   = df_sft.iloc[n_train:]

    def write_split(df_split, path):
        n = 0
        with open(path, "w") as f:
            for _, row in df_split.iterrows():
                state    = int(row["dl_state"])
                target_x = float(row["dl_target_x"])
                func     = STATE_FUNCS[state]
                bvec     = [float(row[b]) for b in vocab]

                # Programmatic gold response
                tx_nearest = nearest_target(target_x) if state == 1 else None
                reasoning  = generate_reasoning(row, state, func, target_x, vocab, labels)
                gold_response = json.dumps({
                    "reasoning":        reasoning,
                    "predicted_state":  state,
                    "predicted_function": func,
                    "predicted_target_x": tx_nearest,
                    "confidence":       0.95,
                })

                gt = {
                    "state":         state,
                    "function":      func,
                    "target_x":      tx_nearest,
                    "branch_vector": bvec,
                }
                record = {
                    "data_source": "cps_debug",
                    "prompt":      [{"role": "user",      "content": build_prompt(row, code_context)},
                                    {"role": "assistant", "content": gold_response}],
                    "ability":     "cps_reasoning",
                    "reward_model": {"ground_truth": gt},
                }
                f.write(json.dumps(record) + "\n")
                n += 1
        return n

    n_tr = write_split(df_train, OUT / "sft_train.jsonl")
    n_va = write_split(df_val,   OUT / "sft_val.jsonl")

    # Combined file
    import subprocess
    subprocess.run(
        f"cat {OUT}/sft_train.jsonl {OUT}/sft_val.jsonl > {OUT}/sft_dataset.jsonl",
        shell=True
    )
    print(f"\nSaved {n_tr} train + {n_va} val = {n_tr+n_va} total SFT examples")
    print(f"  {OUT}/sft_train.jsonl  (for VERL training)")
    print(f"  {OUT}/sft_val.jsonl    (for VERL validation)")
    print(f"  {OUT}/sft_dataset.jsonl (combined)")


if __name__ == "__main__":
    main()
