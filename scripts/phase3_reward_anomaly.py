"""
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
    m = re.search(r"\{.*\}", text, re.DOTALL)
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
