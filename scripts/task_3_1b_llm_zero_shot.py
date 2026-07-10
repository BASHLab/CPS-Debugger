"""
task_3_1b_llm_zero_shot.py — Zero-shot LLM evaluation on anomaly explanation.

Evaluates Claude Haiku on the 250-example anomaly explanation dataset.
Scores: detection (normal/anomalous), typing, code localization, reasoning quality.

Usage:
  python scripts/task_3_1b_llm_zero_shot.py [--model MODEL] [--n N]

Output:
  outputs/phase3/anomaly_zero_shot_results.json
  outputs/phase3/fig_anomaly_results.png
"""

import argparse
import json
import os
import time
from pathlib import Path

ROOT   = Path("/home/simran/allspark-data-exploration/CPS-Debugger")
P3_OUT = ROOT / "outputs/phase3"
P3_OUT.mkdir(parents=True, exist_ok=True)

ANOMALY_TYPES = ["NORMAL", "TRACE_SWAP", "THRESHOLD_BUG", "TIMING_DELAY", "STUCK_SENSOR"]


def parse_json_response(text: str) -> dict:
    """Extract JSON from model response."""
    import re
    # Try direct parse
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try extracting JSON block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {}


def score_response(pred: dict, gt: dict) -> dict:
    """Score a single prediction against ground truth."""
    scores = {"detection": 0.0, "typing": 0.0, "localization": 0.0,
              "reasoning": 0.0, "parse_ok": False, "total": 0.0}

    if not pred:
        return scores
    scores["parse_ok"] = True

    # 1. Anomaly detection (0.25)
    gt_anon  = gt.get("is_anomalous", False)
    pred_anon = pred.get("is_anomalous", False)
    if isinstance(pred_anon, str):
        pred_anon = pred_anon.lower() == "true"
    if pred_anon == gt_anon:
        scores["detection"] = 1.0

    # 2. Anomaly type (0.25)
    gt_type   = gt.get("anomaly_type", "NORMAL").upper()
    pred_type = pred.get("anomaly_type", "").upper()
    if pred_type == gt_type:
        scores["typing"] = 1.0
    elif not gt_anon and not pred_anon:
        scores["typing"] = 1.0  # both say NORMAL

    # 3. Code localization (0.25) — keyword match
    gt_code   = (gt.get("implicated_code") or "").lower()
    pred_code = (pred.get("implicated_code") or "").lower()
    reasoning = (pred.get("reasoning") or "").lower()
    if gt_code:
        keywords = [kw.strip() for kw in gt_code.replace(",", " ").split() if len(kw) > 3]
        if keywords and any(kw in pred_code or kw in reasoning for kw in keywords):
            scores["localization"] = 1.0
    else:
        scores["localization"] = 1.0  # No specific code to localize

    # 4. Reasoning quality (0.25) — references physics, trace, code
    has_phys  = any(kw in reasoning for kw in ["angle", "cart", "sensor", "position", "velocity"])
    has_trace = any(kw in reasoning for kw in ["trace", "branch", "function", "execut", "dispatch"])
    has_code  = any(kw in reasoning for kw in ["tick", "lqr", "pou", "memcpy", "code", "threshold"])
    scores["reasoning"] = (has_phys + has_trace + has_code) / 3.0

    scores["total"] = (scores["detection"] + scores["typing"] +
                       scores["localization"] + scores["reasoning"]) / 4.0
    return scores


def call_api_with_retry(client, model: str, prompt: str,
                        max_tokens: int = 2048, max_retries: int = 5) -> str:
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except Exception as e:
            err = str(e)
            if "529" in err or "overload" in err.lower():
                wait = (2 ** attempt) * 10
                print(f"  [API overloaded, attempt {attempt+1}/{max_retries}, "
                      f"waiting {wait}s]", flush=True)
                time.sleep(wait)
            else:
                print(f"  [API error: {err[:80]}]")
                return ""
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--n",     type=int, default=250, help="Max examples to eval")
    args = parser.parse_args()

    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        return
    client = anthropic.Anthropic(api_key=api_key)

    # Load eval dataset
    eval_path = P3_OUT / "anomaly_eval_dataset.jsonl"
    if not eval_path.exists():
        print("ERROR: anomaly_eval_dataset.jsonl not found. "
              "Run task_3_1_build_anomaly_eval.py first.")
        return

    records = []
    with open(eval_path) as f:
        for line in f:
            records.append(json.loads(line))
    records = records[:args.n]
    print(f"Evaluating {len(records)} examples with {args.model}")

    # ── Run evaluation ────────────────────────────────────────────────────────
    results = []
    parse_errors = 0
    for i, rec in enumerate(records):
        gt     = rec.get("ground_truth", {})
        prompt = rec.get("prompt", "")
        atype  = rec.get("anomaly_type", "UNKNOWN")

        print(f"[{i+1}/{len(records)}] type={atype:15s}", end=" ", flush=True)
        response_text = call_api_with_retry(client, args.model, prompt)

        if not response_text:
            parse_errors += 1
            print("ERROR")
            results.append({
                "idx": i, "anomaly_type": atype, "parse_ok": False,
                "scores": {}, "response": "", "ground_truth": gt,
            })
            continue

        pred   = parse_json_response(response_text)
        scores = score_response(pred, gt)

        if not scores["parse_ok"]:
            parse_errors += 1

        print(f"det={scores['detection']:.0f} typ={scores['typing']:.0f} "
              f"loc={scores['localization']:.2f} rsn={scores['reasoning']:.2f} "
              f"tot={scores['total']:.2f}")

        results.append({
            "idx":           i,
            "anomaly_type":  atype,
            "parse_ok":      scores["parse_ok"],
            "scores":        scores,
            "prediction":    pred,
            "ground_truth":  gt,
        })

    # ── Aggregate ─────────────────────────────────────────────────────────────
    import numpy as np
    from collections import defaultdict

    overall = {
        "detection_acc":    np.mean([r["scores"].get("detection", 0) for r in results]),
        "typing_acc":       np.mean([r["scores"].get("typing", 0) for r in results]),
        "localization_acc": np.mean([r["scores"].get("localization", 0) for r in results]),
        "reasoning_score":  np.mean([r["scores"].get("reasoning", 0) for r in results]),
        "total_score":      np.mean([r["scores"].get("total", 0) for r in results]),
        "parse_error_rate": parse_errors / len(results),
    }

    by_type = defaultdict(list)
    for r in results:
        by_type[r["anomaly_type"]].append(r["scores"].get("total", 0))
    per_type = {t: {"mean_score": float(np.mean(v)), "n": len(v)}
                for t, v in by_type.items()}

    print("\n── Overall Results ──")
    for k, v in overall.items():
        print(f"  {k:25s}: {v:.3f}")
    print("\n── Per-Type Scores ──")
    for t, info in sorted(per_type.items()):
        print(f"  {t:20s}: {info['mean_score']:.3f}  (n={info['n']})")

    out = {
        "model":      args.model,
        "n_examples": len(results),
        "overall":    {k: float(v) for k, v in overall.items()},
        "per_type":   per_type,
        "all_results": results,
    }
    model_slug = args.model.replace("/", "_").replace(":", "_")
    out_fname = f"anomaly_zero_shot_{model_slug}_results.json"
    (P3_OUT / out_fname).write_text(json.dumps(out, indent=2))
    # Also write to canonical path for backwards compatibility
    (P3_OUT / "anomaly_zero_shot_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_fname} and anomaly_zero_shot_results.json")

    # ── Figure ────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart: per-type total score
    ax = axes[0]
    types  = list(per_type.keys())
    scores = [per_type[t]["mean_score"] for t in types]
    colors = ["#4CAF50" if s >= 0.6 else "#FF9800" if s >= 0.4 else "#F44336"
              for s in scores]
    ax.bar(range(len(types)), scores, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(types, rotation=20, ha="right")
    ax.set_ylabel("Mean total score (0–1)")
    ax.set_title(f"Zero-shot anomaly explanation ({args.model})")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)

    # Score components overall
    ax2 = axes[1]
    components = ["detection_acc", "typing_acc", "localization_acc", "reasoning_score"]
    labels     = ["Detection", "Typing", "Localization", "Reasoning"]
    vals = [float(overall[c]) for c in components]
    ax2.bar(range(len(vals)), vals, color="#2196F3", edgecolor="black", linewidth=0.5)
    ax2.set_xticks(range(len(vals)))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Score")
    ax2.set_title("Score breakdown (all examples)")
    ax2.set_ylim(0, 1)
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)

    plt.tight_layout()
    fig_fname = f"fig_anomaly_results_{model_slug}.png"
    fig.savefig(P3_OUT / fig_fname)
    fig.savefig(P3_OUT / "fig_anomaly_results.png")  # canonical copy
    plt.close(fig)
    print(f"Saved {fig_fname}")
    print("Done.")


if __name__ == "__main__":
    main()
