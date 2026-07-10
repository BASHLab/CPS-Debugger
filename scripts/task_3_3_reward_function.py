"""
task_3_3_reward_function.py — Write and validate the GRPO reward function.

Generates scripts/phase3_reward_anomaly.py (VERL-importable reward function)
and validates it against the zero-shot results to ensure the score distribution
is appropriate for RL training (not trivially too high or too low).

Output:
  scripts/phase3_reward_anomaly.py
  outputs/phase3/reward_validation.json
"""

import json
from pathlib import Path

ROOT   = Path("/home/simran/allspark-data-exploration/CPS-Debugger")
P3_OUT = ROOT / "outputs/phase3"
P3_OUT.mkdir(parents=True, exist_ok=True)

REWARD_FN_CODE = '''"""
phase3_reward_anomaly.py — VERL reward function for CPS anomaly explanation.

Verifiable reward with 4 components (0.25 each = max 1.0):
  1. Anomaly detection:    is_anomalous field correctly set
  2. Anomaly typing:       anomaly_type field matches ground truth
  3. Code localization:    response mentions correct function/component
  4. Reasoning quality:    response references physics, trace, and code

Usage (VERL custom reward):
  reward_model.reward_fn_path = "scripts/phase3_reward_anomaly.py"
  reward_model.reward_fn = "compute_score"
"""

import json
import re


ANOMALY_TYPES = {
    "NORMAL", "TRACE_SWAP", "THRESHOLD_BUG", "TIMING_DELAY", "STUCK_SENSOR", "OTHER"
}

# Keywords that indicate mention of each component for localization scoring
LOCALIZATION_KEYWORDS = {
    "tick":           ["tick", "dispatch", "state machine", "pendulum_state"],
    "pou_lqr_sim":    ["lqr", "pou_lqr", "compute_force", "compute_velocity"],
    "memcpy":         ["memcpy", "data copy", "state vector"],
    "threshold":      ["threshold", "max_x", "reset threshold", "safe"],
    "sensor":         ["sensor", "datalayer", "ethercat", "cart position"],
    "synchronization": ["sync", "timestamp", "timing", "pipeline", "delay"],
}


def parse_json_from_response(text: str) -> dict:
    """Extract JSON from a response string."""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\\{.*\\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {}


def compute_score(data_source: str, solution_str: str, ground_truth: str,
                  extra_info: dict = None) -> float:
    """
    VERL-compatible reward function for CPS anomaly explanation.

    Args:
        data_source:  "cps_anomaly"
        solution_str: Model-generated response text
        ground_truth: JSON string with ground truth fields
        extra_info:   Optional dict with additional context

    Returns:
        Reward in [0.0, 1.0]
    """
    try:
        gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    except Exception:
        return 0.0

    pred = parse_json_from_response(solution_str)
    if not pred:
        return 0.0  # Parse failure: zero reward

    score = 0.0

    # ── Component 1: Anomaly detection (0.25) ─────────────────────────────────
    gt_is_anon   = bool(gt.get("is_anomalous", False))
    pred_is_anon = pred.get("is_anomalous", False)
    if isinstance(pred_is_anon, str):
        pred_is_anon = pred_is_anon.lower() in ("true", "yes", "1")
    if bool(pred_is_anon) == gt_is_anon:
        score += 0.25

    # ── Component 2: Anomaly type (0.25) ─────────────────────────────────────
    gt_type   = gt.get("anomaly_type", "NORMAL").upper().strip()
    pred_type = pred.get("anomaly_type", "").upper().strip()
    # Normalize aliases
    type_aliases = {
        "NONE": "NORMAL", "NO_ANOMALY": "NORMAL", "CONSISTENT": "NORMAL",
        "SWAP": "TRACE_SWAP", "CODE_SWAP": "TRACE_SWAP",
        "BUG": "THRESHOLD_BUG", "PREMATURE_RESET": "THRESHOLD_BUG",
        "DELAY": "TIMING_DELAY", "DESYNC": "TIMING_DELAY",
        "FROZEN": "STUCK_SENSOR", "STUCK": "STUCK_SENSOR",
    }
    pred_type = type_aliases.get(pred_type, pred_type)
    if pred_type == gt_type:
        score += 0.25
    elif gt_type == "NORMAL" and not bool(pred_is_anon):
        score += 0.25  # Correctly identified as normal even if type label differs

    # ── Component 3: Code localization (0.25) ────────────────────────────────
    gt_code      = (gt.get("implicated_code") or "").lower()
    pred_text    = (solution_str or "").lower()
    pred_code    = (pred.get("implicated_code") or "").lower()
    pred_reason  = (pred.get("reasoning") or "").lower()
    pred_full    = pred_text + " " + pred_code + " " + pred_reason

    if not gt_code or gt_type == "NORMAL":
        score += 0.25  # No specific code to localize
    else:
        # Check if any keyword for the relevant component appears
        matched = False
        for component, kws in LOCALIZATION_KEYWORDS.items():
            if any(kw in gt_code for kw in kws):
                if any(kw in pred_full for kw in kws):
                    matched = True
                    break
        # Also check direct keyword match
        gt_words = [w for w in gt_code.replace(",", " ").split() if len(w) > 3]
        if not matched and any(w in pred_full for w in gt_words):
            matched = True
        if matched:
            score += 0.25

    # ── Component 4: Reasoning quality (0.25) ────────────────────────────────
    reasoning = pred.get("reasoning", "") or ""
    reasoning = reasoning.lower()

    has_phys  = any(kw in reasoning for kw in
                    ["angle", "cart", "sensor", "position", "velocity", "upright"])
    has_trace = any(kw in reasoning for kw in
                    ["trace", "branch", "function", "execut", "dispatch", "memcpy", "tick"])
    has_code  = any(kw in reasoning for kw in
                    ["tick", "lqr", "pou", "threshold", "memcpy", "code", "max_x",
                     "standup", "reset"])

    quality = (int(has_phys) + int(has_trace) + int(has_code)) / 3.0
    score += 0.25 * quality

    return round(score, 4)
'''

# ── Write the reward function file ────────────────────────────────────────────
reward_fn_path = ROOT / "scripts/phase3_reward_anomaly.py"
reward_fn_path.write_text(REWARD_FN_CODE)
print(f"Wrote {reward_fn_path}")


def main():
    # ── Validate against zero-shot results ───────────────────────────────────
    results_path = P3_OUT / "anomaly_zero_shot_results.json"
    if not results_path.exists():
        print("Zero-shot results not yet available — skipping validation.")
        print("(Run task_3_1b_llm_zero_shot.py first)")
        validation = {
            "status": "SKIPPED_NO_RESULTS",
            "message": "Run task_3_1b_llm_zero_shot.py before validating reward fn",
        }
        (P3_OUT / "reward_validation.json").write_text(json.dumps(validation, indent=2))
        return

    data = json.loads(results_path.read_text())
    all_results = data.get("all_results", [])

    # Import and test the reward function
    import importlib.util
    spec = importlib.util.spec_from_file_location("phase3_reward_anomaly", str(reward_fn_path))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    compute_score = mod.compute_score

    scores = []
    for rec in all_results:
        if not rec.get("parse_ok"):
            continue
        pred_str = json.dumps(rec.get("prediction", {}))
        gt_str   = json.dumps(rec.get("ground_truth", {}))
        s = compute_score("cps_anomaly", pred_str, gt_str)
        scores.append(s)

    if not scores:
        print("No parseable results to validate against.")
        return

    import numpy as np
    mean_s = float(np.mean(scores))
    std_s  = float(np.std(scores))
    print(f"\nReward distribution on zero-shot results:")
    print(f"  n={len(scores)}  mean={mean_s:.3f}  std={std_s:.3f}")
    print(f"  min={min(scores):.3f}  max={max(scores):.3f}")
    print(f"  frac > 0.5: {np.mean([s > 0.5 for s in scores]):.2f}")
    print(f"  frac > 0.75: {np.mean([s > 0.75 for s in scores]):.2f}")
    print(f"  frac == 0.0: {np.mean([s == 0 for s in scores]):.2f}")

    # Assess suitability
    if mean_s > 0.85:
        assessment = "TOO_EASY: Reward too high for zero-shot baseline — consider stricter scoring"
    elif mean_s < 0.15:
        assessment = "TOO_HARD: Reward too low — model won't get signal to improve"
    else:
        assessment = "SUITABLE: Reward distribution appropriate for RL training"
    print(f"\n  Assessment: {assessment}")

    validation = {
        "n_examples_scored": len(scores),
        "mean_reward":  mean_s,
        "std_reward":   std_s,
        "min_reward":   float(min(scores)),
        "max_reward":   float(max(scores)),
        "frac_gt_05":   float(np.mean([s > 0.5 for s in scores])),
        "frac_gt_075":  float(np.mean([s > 0.75 for s in scores])),
        "frac_zero":    float(np.mean([s == 0 for s in scores])),
        "assessment":   assessment,
        "reward_fn_path": str(reward_fn_path),
    }
    (P3_OUT / "reward_validation.json").write_text(json.dumps(validation, indent=2))
    print("\nSaved reward_validation.json")
    print("Done.")


if __name__ == "__main__":
    main()
