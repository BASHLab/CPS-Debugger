"""
phase2_C5_evaluate_grpo.py — Evaluate GRPO-trained model on held-out test set.

Compares zero-shot vs. SFT-only vs. GRPO-trained on the same eval metrics.
Requires a vLLM-compatible model checkpoint.

Usage:
    python3 scripts/phase2_C5_evaluate_grpo.py \
        --model_path outputs/phase2/checkpoints/grpo_v1/global_step_latest \
        --dataset outputs/phase2/llm_eval_dataset_holdout.jsonl \
        --output outputs/phase2/grpo_eval_results.json
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/simran/allspark-data-exploration/CPS-Debugger")
OUT  = ROOT / "outputs/phase2"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--dataset",    default=str(OUT / "llm_eval_dataset_holdout.jsonl"))
    p.add_argument("--output",     default=str(OUT / "grpo_eval_results.json"))
    p.add_argument("--n",          type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.0)  # greedy for eval
    return p.parse_args()


def evaluate_with_vllm(model_path, records, temperature=0.0):
    """Generate responses with vLLM and compute metrics."""
    from vllm import LLM, SamplingParams
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from phase2_C2_reward_function import compute_score, parse_json_from_response

    llm    = LLM(model=model_path, gpu_memory_utilization=0.85)
    params = SamplingParams(temperature=temperature, max_tokens=512)

    prompts = [rec["prompt"][0]["content"] for rec in records]
    outputs = llm.generate(prompts, params)

    results = []
    rewards = []
    for rec, out in zip(records, outputs):
        response = out.outputs[0].text
        gt       = rec["ground_truth"]
        reward   = compute_score("cps_debug", response, gt)
        rewards.append(reward)

        try:
            pred = parse_json_from_response(response)
        except Exception:
            pred = {}

        state_correct   = pred.get("predicted_state") == gt["state"]
        func_correct    = pred.get("predicted_function") == gt["function"]
        tx_correct      = None
        if gt["state"] == 1:
            pred_tx = pred.get("predicted_target_x")
            tx_correct = (pred_tx is not None and abs(pred_tx - gt["target_x"]) < 0.02)

        results.append({
            "ground_truth":   gt,
            "prediction":     pred,
            "reward":         reward,
            "state_correct":  state_correct,
            "func_correct":   func_correct,
            "tx_correct":     tx_correct,
        })

    return results, rewards


def main():
    args = parse_args()

    print(f"Loading dataset: {args.dataset}")
    records = []
    with open(args.dataset) as f:
        for line in f:
            records.append(json.loads(line))
    records = records[:args.n]
    print(f"  {len(records)} evaluation samples")

    print(f"\nEvaluating: {args.model_path}")
    results, rewards = evaluate_with_vllm(args.model_path, records, args.temperature)

    # Compute metrics
    state_accs  = [r["state_correct"] for r in results]
    func_accs   = [r["func_correct"]  for r in results]
    tx_accs     = [r["tx_correct"]    for r in results if r["tx_correct"] is not None]

    metrics = {
        "model_path":       args.model_path,
        "n_samples":        len(results),
        "state_accuracy":   float(np.mean(state_accs)),
        "function_accuracy": float(np.mean(func_accs)),
        "target_x_accuracy": float(np.mean(tx_accs)) if tx_accs else None,
        "mean_reward":      float(np.mean(rewards)),
        "std_reward":       float(np.std(rewards)),
    }

    print("\n── GRPO Evaluation Results ──")
    print(f"  State accuracy:    {metrics['state_accuracy']:.3f}")
    print(f"  Function accuracy: {metrics['function_accuracy']:.3f}")
    if metrics["target_x_accuracy"] is not None:
        print(f"  Target_x accuracy: {metrics['target_x_accuracy']:.3f}")
    print(f"  Mean reward:       {metrics['mean_reward']:.3f} ± {metrics['std_reward']:.3f}")

    # Load zero-shot baseline for comparison (if available)
    zs_path = OUT / "llm_zero_shot_results.json"
    comparison = {"grpo": metrics}
    if zs_path.exists():
        with open(zs_path) as f:
            zs = json.load(f)
        comparison["zero_shot"] = {
            "state_accuracy":   zs["state_accuracy"],
            "function_accuracy": zs["function_accuracy"],
            "target_x_accuracy": zs["target_x_accuracy"],
        }
        print("\n── Comparison (GRPO vs Zero-Shot) ──")
        for k in ["state_accuracy", "function_accuracy", "target_x_accuracy"]:
            g  = metrics.get(k)
            zv = zs.get(k)
            if g is not None and zv is not None:
                print(f"  {k:25s}: GRPO={g:.3f}  ZeroShot={zv:.3f}  Δ={g-zv:+.3f}")

    # Save results
    out_data = {"metrics": metrics, "comparison": comparison, "results": results}
    with open(args.output, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved → {args.output}")

    # Generate comparison figure
    if "zero_shot" in comparison:
        fig, ax = plt.subplots(figsize=(7, 4))
        metric_names = ["State\nAccuracy", "Function\nAccuracy", "Target_x\n(BALANCE)"]
        grpo_vals = [metrics.get("state_accuracy", 0),
                     metrics.get("function_accuracy", 0),
                     metrics.get("target_x_accuracy") or 0]
        zs_vals   = [comparison["zero_shot"].get("state_accuracy", 0),
                     comparison["zero_shot"].get("function_accuracy", 0),
                     comparison["zero_shot"].get("target_x_accuracy") or 0]
        x = np.arange(3)
        ax.bar(x - 0.2, zs_vals,   width=0.35, label="Zero-Shot", color="#aec6cf")
        ax.bar(x + 0.2, grpo_vals, width=0.35, label="GRPO", color="#2e6da4")
        ax.set_xticks(x); ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 1.1); ax.set_ylabel("Accuracy")
        ax.set_title("GRPO vs Zero-Shot: Physics-to-Code Reasoning")
        ax.legend()
        plt.tight_layout()
        fig.savefig(OUT / "fig_C5_grpo_vs_zeroshot.png")
        plt.close(fig)
        print(f"Saved fig_C5_grpo_vs_zeroshot.png")


if __name__ == "__main__":
    main()
