"""
phase2_C2_reward_function.py — Verifiable reward for CPS physics-to-code reasoning.

VERL-compatible reward function. Receives LLM response text and ground truth dict.
Returns scalar reward in range [-0.5, 1.0].

Reward components:
  +0.30  State prediction correct (coarse — verifiable from physics)
  +0.20  Function prediction correct (deterministic from state)
  +0.30  Target_x correct within BALANCE (±0.02 m tolerance); partial credit ±0.05
          OR correctly predicts null target when not in BALANCE
  +0.10  Reasoning references source code entities (functions, variables)
  +0.10  Reasoning references physical quantities (angle, velocity, etc.)
  −0.50  Parse failure (response not valid JSON)

This is the file VERL workers will import. No project-specific dependencies.
"""

import json
import re


# ── Constants ─────────────────────────────────────────────────────────────────
VALID_STATES      = {0, 1, 2}
VALID_FUNCTIONS   = {"pou_standup_rel", "pou_lqr_sim", "tick_only"}
TARGET_POSITIONS  = {0.0, 0.05, -0.09, 0.09}

CODE_KEYWORDS = [
    "tick", "pou_lqr_sim", "pou_standup_rel", "lqr", "standup",
    "track limit", "balance", "swingup", "reset", "ethercat",
    "memcpy", "log_data", "0.15", "0.18",
]
PHYSICS_KEYWORDS = [
    "upright", "angle", "velocity", "position", "cart", "pendulum",
    "angular", "energy", "swing", "oscillat", "tilt",
]


# ── JSON parsing ──────────────────────────────────────────────────────────────
def parse_json_from_response(text: str) -> dict:
    """Extract JSON dict from LLM response, tolerating markdown fences."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start < 0 or end <= 0:
        raise ValueError("No JSON object found")
    return json.loads(text[start:end])


# ── Main reward function (VERL signature) ─────────────────────────────────────
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth,
    extra_info=None,
) -> float:
    """
    Verifiable reward for CPS physics-to-code reasoning.

    Args:
        data_source:  Dataset name (e.g. "cps_debug"). Not used in computation.
        solution_str: Raw LLM response text.
        ground_truth: Dict or JSON string with keys:
                        state (int), function (str), target_x (float|None),
                        branch_vector (list[float], optional).
        extra_info:   Unused.

    Returns:
        float in [-0.5, 1.0]
    """
    # Parse ground truth
    if isinstance(ground_truth, str):
        gt = json.loads(ground_truth)
    else:
        gt = dict(ground_truth)

    # Parse LLM response
    try:
        pred = parse_json_from_response(solution_str)
    except Exception:
        return -0.5  # Format penalty: no parseable JSON

    reward = 0.0

    # ── Component 1: State prediction (0.30) ──────────────────────────────
    if pred.get("predicted_state") in VALID_STATES:
        if int(pred["predicted_state"]) == int(gt["state"]):
            reward += 0.30

    # ── Component 2: Function prediction (0.20) ───────────────────────────
    if pred.get("predicted_function") in VALID_FUNCTIONS:
        if pred["predicted_function"] == gt["function"]:
            reward += 0.20

    # ── Component 3: Target position (0.30) ──────────────────────────────
    if int(gt["state"]) == 1:  # BALANCE: target_x matters
        pred_tx = pred.get("predicted_target_x")
        if pred_tx is not None:
            try:
                pred_tx = float(pred_tx)
                true_tx = float(gt["target_x"])
                err = abs(pred_tx - true_tx)
                if err < 0.02:
                    reward += 0.30   # correct
                elif err < 0.05:
                    reward += 0.15   # adjacent target
            except (TypeError, ValueError):
                pass  # invalid format → 0 credit
    else:
        # Not BALANCE: should predict null target
        if pred.get("predicted_target_x") is None:
            reward += 0.30

    # ── Component 4: Reasoning quality (0.10 + 0.10) ────────────────────
    reasoning = str(pred.get("reasoning", "")).lower()
    if reasoning:
        code_hits    = sum(1 for kw in CODE_KEYWORDS if kw.lower() in reasoning)
        physics_hits = sum(1 for kw in PHYSICS_KEYWORDS if kw.lower() in reasoning)
        code_score    = min(code_hits    / 3.0, 1.0)
        physics_score = min(physics_hits / 3.0, 1.0)
        reward += 0.10 * code_score
        reward += 0.10 * physics_score

    return float(reward)


# ── Convenience: batch evaluation ────────────────────────────────────────────
def evaluate_dataset(jsonl_path: str, results_json_path: str | None = None):
    """
    Evaluate reward function on a JSONL dataset of LLM results.
    Expects each line to have 'ground_truth' and 'response' keys.
    """
    import pathlib
    rewards = []
    records = []
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            gt  = rec.get("ground_truth", rec.get("reward_model", {}).get("ground_truth", {}))
            sol = rec.get("response", rec.get("solution", ""))
            r   = compute_score("cps_debug", sol, gt)
            rewards.append(r)
            records.append({**rec, "reward": r})

    import statistics
    print(f"Reward stats over {len(rewards)} samples:")
    print(f"  Mean:   {statistics.mean(rewards):.3f}")
    print(f"  Median: {statistics.median(rewards):.3f}")
    print(f"  Stdev:  {statistics.stdev(rewards):.3f}")
    parse_fails = sum(1 for r in rewards if r == -0.5)
    print(f"  Parse failures: {parse_fails} ({parse_fails/len(rewards):.0%})")

    if results_json_path:
        with open(results_json_path, "w") as f:
            json.dump({"rewards": rewards, "records": records}, f, indent=2)
        print(f"Saved → {results_json_path}")

    return rewards


# ── CLI: validate reward on zero-shot results ────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path

    OUT = Path("/home/simran/allspark-data-exploration/CPS-Debugger/outputs/phase2")
    zs_path = OUT / "llm_zero_shot_results.json"

    if zs_path.exists():
        print("Validating reward function on zero-shot LLM results...")
        with open(zs_path) as f:
            zs = json.load(f)

        rewards = []
        for rec in zs.get("all_results", []):
            gt  = rec["ground_truth"]
            sol = json.dumps(rec.get("prediction", {}))
            r   = compute_score("cps_debug", sol, gt)
            rewards.append(r)

        if rewards:
            import statistics
            print(f"  N={len(rewards)}, Mean={statistics.mean(rewards):.3f}, "
                  f"Stdev={statistics.stdev(rewards):.3f}")
            print(f"  Parse failures: {sum(1 for r in rewards if r == -0.5)}")

            out_path = OUT / "reward_function_validation.json"
            with open(out_path, "w") as f:
                json.dump({"rewards": rewards, "mean": statistics.mean(rewards),
                           "std": statistics.stdev(rewards)}, f, indent=2)
            print(f"  Saved → {out_path}")
        else:
            print("  No results found in zero-shot results JSON")
    else:
        # Self-test with dummy examples
        print("Running self-test (no zero-shot results found)...")
        tests = [
            # Perfect prediction in BALANCE
            ('{"reasoning":"angle near 0 means balanced, cart at 0.05 m target, tick calls pou_lqr_sim, position error drives LQR","predicted_state":1,"predicted_function":"pou_lqr_sim","predicted_target_x":0.05,"confidence":0.9}',
             {"state": 1, "function": "pou_lqr_sim", "target_x": 0.05}),
            # Correct state, wrong target
            ('{"reasoning":"near upright balance mode","predicted_state":1,"predicted_function":"pou_lqr_sim","predicted_target_x":0.0,"confidence":0.7}',
             {"state": 1, "function": "pou_lqr_sim", "target_x": 0.05}),
            # SWINGUP correct
            ('{"reasoning":"angle large, pendulum swinging, standup called","predicted_state":0,"predicted_function":"pou_standup_rel","predicted_target_x":null,"confidence":0.85}',
             {"state": 0, "function": "pou_standup_rel", "target_x": None}),
            # Parse failure
            ("I think it is balanced", {"state": 1, "function": "pou_lqr_sim", "target_x": 0.05}),
        ]
        for sol, gt in tests:
            r = compute_score("cps_debug", sol, gt)
            print(f"  reward={r:.2f} | gt_state={gt['state']} | sol_snippet={sol[:60]}...")
