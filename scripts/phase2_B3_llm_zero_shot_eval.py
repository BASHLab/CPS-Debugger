"""
phase2_B3_llm_zero_shot_eval.py — Zero-shot LLM evaluation on physics-to-code reasoning.

Uses Claude API (claude-sonnet-4-6) to evaluate the 500 eval problems.
Measures state accuracy, function accuracy, target_x accuracy (within BALANCE),
and saves full results + curated examples.

Usage:
    python3 scripts/phase2_B3_llm_zero_shot_eval.py [--n 50] [--model claude-haiku-4-5-20251001]

Arguments:
    --n:     Number of samples to evaluate (default: all 500; use 50 for quick test)
    --model: Claude model ID (default: claude-sonnet-4-6)
    --holdout: Evaluate held-out dataset instead of dev

Output: outputs/phase2/llm_zero_shot_results.json
        outputs/phase2/fig_B1_llm_zero_shot_results.png
        outputs/phase2/fig_B2_llm_reasoning_examples.md
"""

import argparse
import json
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/simran/allspark-data-exploration/CPS-Debugger")
OUT  = ROOT / "outputs/phase2"

STATE_NAMES = {0: "SWINGUP", 1: "BALANCE", 2: "RESET"}
STATE_FUNCS = {0: "pou_standup_rel", 1: "pou_lqr_sim", 2: "tick_only"}


def parse_json_from_response(text):
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Strip markdown
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    # Find first { ... }
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start < 0 or end <= 0:
        raise ValueError("No JSON object found in response")
    return json.loads(text[start:end])


def evaluate_prediction(pred, gt):
    """Compute per-component accuracy for one prediction."""
    results = {"state_correct": False, "function_correct": False,
               "target_x_correct": None, "parse_ok": True}
    state_correct   = pred.get("predicted_state") == gt["state"]
    func_correct    = pred.get("predicted_function") == gt["function"]
    results["state_correct"]    = state_correct
    results["function_correct"] = func_correct

    if gt["state"] == 1:  # BALANCE
        pred_tx = pred.get("predicted_target_x")
        if pred_tx is not None:
            results["target_x_correct"] = abs(pred_tx - gt["target_x"]) < 0.02
        else:
            results["target_x_correct"] = False
    else:
        results["target_x_correct"] = (pred.get("predicted_target_x") is None)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",       type=int,  default=500)
    parser.add_argument("--model",   type=str,  default="claude-sonnet-4-6")
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--delay",   type=float, default=0.5,
                        help="Seconds between API calls")
    args = parser.parse_args()

    # Load dataset
    dataset_path = (OUT / "llm_eval_dataset_holdout.jsonl" if args.holdout
                    else OUT / "llm_eval_dataset.jsonl")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Run phase2_B2 first: {dataset_path}")

    records = []
    with open(dataset_path) as f:
        for line in f:
            records.append(json.loads(line))
    records = records[:args.n]
    print(f"Evaluating {len(records)} samples with {args.model}...")

    # Try to import Anthropic
    try:
        import anthropic
        client = anthropic.Anthropic()
    except ImportError:
        print("ERROR: anthropic package not installed. Run: pip install anthropic")
        return
    except Exception as e:
        print(f"ERROR initialising Anthropic client: {e}")
        return

    # Run evaluations
    all_results = []
    state_correct_list    = []
    func_correct_list     = []
    target_x_correct_list = []
    parse_errors          = 0

    for i, rec in enumerate(records):
        gt     = rec["ground_truth"]
        prompt = rec["prompt"][0]["content"]

        response_text = ""
        pred          = {}
        eval_res      = {"state_correct": False, "function_correct": False,
                         "target_x_correct": None, "parse_ok": False}

        # Retry with exponential backoff for 529 overload errors
        for attempt in range(5):
            try:
                resp = client.messages.create(
                    model=args.model,
                    max_tokens=2048,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = resp.content[0].text
                pred = parse_json_from_response(response_text)
                eval_res = evaluate_prediction(pred, gt)
                break  # success
            except Exception as e:
                err_str = str(e)
                if "529" in err_str or "overloaded" in err_str.lower():
                    wait = (2 ** attempt) * 10  # 10, 20, 40, 80, 160s
                    print(f"  [API overloaded, attempt {attempt+1}/5, waiting {wait}s]")
                    time.sleep(wait)
                    continue
                # Non-overload error: log and break
                print(f"  [Parse/API error at sample {i}: {err_str[:80]}]")
                parse_errors += 1
                break
        else:
            # All retries exhausted (overload)
            print(f"  [All retries exhausted at sample {i}]")
            parse_errors += 1

        state_correct_list.append(eval_res["state_correct"])
        func_correct_list.append(eval_res["function_correct"])
        if eval_res["target_x_correct"] is not None:
            target_x_correct_list.append(eval_res["target_x_correct"])

        all_results.append({
            "idx":          i,
            "run":          rec["metadata"]["run"],
            "ground_truth": gt,
            "prediction":   pred,
            "eval":         eval_res,
            "response":     response_text[:2000],  # truncate for storage
        })

        if (i + 1) % 10 == 0:
            sa  = np.mean(state_correct_list)
            fa  = np.mean(func_correct_list)
            txa = np.mean(target_x_correct_list) if target_x_correct_list else float("nan")
            print(f"  [{i+1}/{len(records)}] state={sa:.3f} func={fa:.3f} "
                  f"target_x={txa:.3f} parse_errors={parse_errors}")

        time.sleep(args.delay)

    # Aggregate metrics
    state_acc   = float(np.mean(state_correct_list))
    func_acc    = float(np.mean(func_correct_list))
    target_x_acc = float(np.mean(target_x_correct_list)) if target_x_correct_list else None
    parse_error_rate = parse_errors / len(records)

    # Per-state breakdown
    per_state = {}
    for s, sname in STATE_NAMES.items():
        idxs = [j for j, r in enumerate(all_results) if r["ground_truth"]["state"] == s]
        if idxs:
            per_state[sname] = {
                "n":            len(idxs),
                "state_acc":    float(np.mean([state_correct_list[j] for j in idxs])),
                "func_acc":     float(np.mean([func_correct_list[j] for j in idxs])),
            }

    summary = {
        "model":           args.model,
        "n_samples":       len(records),
        "state_accuracy":  state_acc,
        "function_accuracy": func_acc,
        "target_x_accuracy": target_x_acc,
        "parse_error_rate":  parse_error_rate,
        "per_state":         per_state,
        "all_results":       all_results,
    }

    out_path = OUT / "llm_zero_shot_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved results → {out_path}")

    # Print final summary
    print("\n" + "="*50)
    print("ZERO-SHOT LLM RESULTS")
    print("="*50)
    print(f"  State prediction accuracy:    {state_acc:.3f}")
    print(f"  Function prediction accuracy: {func_acc:.3f}")
    print(f"  Target_x accuracy (BALANCE):  "
          f"{target_x_acc:.3f}" if target_x_acc else "  Target_x accuracy:  N/A")
    print(f"  Parse error rate:             {parse_error_rate:.3f}")
    for sname, vals in per_state.items():
        print(f"  {sname:10s} (n={vals['n']:3d}): "
              f"state={vals['state_acc']:.3f}, func={vals['func_acc']:.3f}")

    # fig_B1: bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    metrics = ["State\nAccuracy", "Function\nAccuracy"]
    values  = [state_acc, func_acc]
    colors  = ["#2e6da4", "#e07b54"]
    if target_x_acc is not None:
        metrics.append("Target_x\n(within BALANCE)")
        values.append(target_x_acc)
        colors.append("#6abf69")
    x = np.arange(len(metrics))
    ax.bar(x, values, color=colors, width=0.55)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.axhline(0.94, color="k", ls="--", lw=1.0, label="BALANCE class prior (94%)")
    ax.set_title(f"Zero-Shot LLM ({args.model})\nn={len(records)} windows")
    ax.legend(fontsize=9)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    fig.savefig(OUT / "fig_B1_llm_zero_shot_results.png")
    plt.close(fig)
    print("Saved fig_B1")

    # fig_B2: curated reasoning examples (markdown)
    # Select 2 good + 2 wrong state + 2 wrong target_x
    good   = [r for r in all_results if r["eval"]["state_correct"]
              and r["eval"]["target_x_correct"]][:2]
    bad_s  = [r for r in all_results if not r["eval"]["state_correct"]][:2]
    bad_tx = [r for r in all_results
              if r["ground_truth"]["state"] == 1
              and r["eval"]["target_x_correct"] is False][:2]
    examples = good + bad_s + bad_tx

    with open(OUT / "fig_B2_llm_reasoning_examples.md", "w") as f:
        f.write("# Zero-Shot LLM Reasoning Examples\n\n")
        for tag, ex_list in [("Correct", good), ("Wrong state", bad_s),
                              ("Wrong target_x", bad_tx)]:
            f.write(f"## {tag}\n\n")
            for ex in ex_list:
                gt   = ex["ground_truth"]
                pred = ex["prediction"]
                f.write(f"**Run**: `{ex['run']}`  \n")
                f.write(f"**Ground truth**: state={gt['state_name']}, "
                        f"function={gt['function']}, target_x={gt['target_x']}  \n")
                f.write(f"**Prediction**: state={pred.get('predicted_state')}, "
                        f"function={pred.get('predicted_function')}, "
                        f"target_x={pred.get('predicted_target_x')}, "
                        f"confidence={pred.get('confidence')}  \n\n")
                f.write(f"**Reasoning**:\n{pred.get('reasoning','(none)')}\n\n---\n\n")
    print("Saved fig_B2_llm_reasoning_examples.md")


if __name__ == "__main__":
    main()
