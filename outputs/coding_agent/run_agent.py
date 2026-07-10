#!/usr/bin/env python3
"""Run the CPS code debugging agent on injected bug scenarios.

Usage:
    python run_agent.py                      # run all bugs
    python run_agent.py --bugs bug_01 bug_03 # run specific bugs
    python run_agent.py --model claude-haiku-4-5-20250414
    python run_agent.py --dry-run            # print system prompt, skip API calls
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from bug_specs import BUGS
from cps_aci import CPS_ACI, FUNC_NAMES, TOOL_SCHEMAS

ROOT = Path(__file__).resolve().parent.parent.parent  # CPS-Debugger root
ORIGINAL_SRC = ROOT / "outputs" / "phase2" / "src" / "src"
BUG_DIR = ROOT / "outputs" / "coding_agent" / "bugs"
OUT_DIR = ROOT / "outputs" / "coding_agent"
ALIGNED_PARQUET = ROOT / "outputs" / "experiments" / "aligned_dataset.parquet"


# ── System prompt ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a CPS code debugging agent. You are investigating an anomaly detected
in an inverted pendulum controller running on a Bosch Rexroth ctrlX platform.

SYSTEM OVERVIEW:
- Controller is written in C, compiled to WebAssembly via WASI SDK
- Runs a 1kHz control loop (1ms period) on the ctrlX embedded PC
- Implements three FSM states: SWINGUP (energy-based), BALANCE (LQR), RESET
- Pittsburgh deployment: 13-bit absolute encoder, reversed angle convention

KEY FILES:
- lqrsim.c: LQR balance controller (compute_force, compute_velocity, pou_lqr_sim)
- main.c: FSM dispatch (pou_main)
- standup_reliable.c: Energy-based swing-up (pou_standup_rel)
- general.c: Sensor processing (encoder angle, velocity calculation)
- sys_params.c: All tuning parameters (LQR gains, limits, etc.)
- pend_monolithic.c: Monolithic build with tick() entry point and data logging

EXECUTION TRACES:
- The Wasm binary is instrumented to record control-flow edges
- Each edge is named func_id:from_pc:to_pc (e.g. 40:44:85)
- Edge counts per 10ms window are the trace columns
- func 22=tick, 33=pou_main, 40=pou_lqr_sim, 39=pou_standup_rel

YOUR TASK:
An anomaly has been detected by the monitoring system. You must:
1. Read the anomaly report to understand the symptoms
2. Use tools to investigate the source code and data
3. Identify the root cause (a single bug in the source code)
4. Submit your diagnosis and code fix using submit_diagnosis

APPROACH:
- Start with the anomaly report to identify which function/edges are affected
- Use read_reference to view the known-good repo version of the affected code
- Use read_source to view the deployed (possibly buggy) version
- COMPARE the two: the bug is a single modification to one line or expression
- The deployed code may differ from the reference -- find the difference that
  explains the anomaly symptoms
- Submit your fix when confident (you get one attempt)
"""


# ── Anomaly observation simulation ──────────────────────────────────────

def simulate_anomaly_observation(
    bug_spec: Dict,
    data: pd.DataFrame,
    physical_cols: List[str],
    branch_cols: List[str],
) -> Dict:
    """Create a simulated anomaly observation for a given bug.

    Since we cannot actually run the buggy Wasm, we fabricate the expected
    anomaly signature based on which edges the bug affects.
    """
    affected_edges = bug_spec.get("affected_edges", [])
    edge_cols = [c for c in branch_cols if any(e in c for e in affected_edges)]

    # Compute normal statistics for affected edges
    edge_summary = {}
    for col in edge_cols:
        normal_mean = float(data[col].mean())
        normal_std = float(data[col].std()) + 1e-8
        # Simulate deviation based on bug type
        if bug_spec["category"] == "parameter_fault":
            observed = normal_mean * np.random.uniform(0.3, 0.6)
        else:  # logic_fault
            observed = normal_mean * np.random.uniform(0.1, 0.3)
        deviation = abs(observed - normal_mean) / normal_std
        edge_summary[col] = {
            "normal_mean": round(normal_mean, 2),
            "observed_mean": round(float(observed), 2),
            "deviation_sigma": round(float(deviation), 1),
        }

    # For bugs with no direct edge effects (e.g. angle flip), report dynamics
    dynamics_residual = "HIGH" if not edge_cols else "ELEVATED"
    trace_residual = "HIGH" if edge_cols else "NORMAL"

    alert = {
        "anomaly_detected": True,
        "trace_residual": trace_residual,
        "sensor_residual": "NORMAL",  # sensor hardware is fine for code bugs
        "dynamics_residual": dynamics_residual,
        "affected_function": bug_spec["affected_function"],
        "affected_edges": affected_edges,
        "timestamp_range": "windows 5000-5100 (during BALANCE state)",
    }

    return {
        "alert": alert,
        "edge_summary": edge_summary,
        "bug_metadata": bug_spec,  # ground truth -- NOT shown to agent
    }


# ── Evaluation scoring ──────────────────────────────────────────────────

def check_diagnosis(diagnosis: Dict, ground_truth: Dict) -> Dict[str, bool]:
    """Score the agent's diagnosis against ground truth."""
    gt_category = ground_truth.get("category", "")
    gt_function = ground_truth.get("affected_function", "")

    diag_category = diagnosis.get("fault_type", "")
    diag_function = diagnosis.get("affected_function", "")

    # Category match
    cat_match = diag_category == gt_category

    # Function match: allow inner function names (compute_force is inside pou_lqr_sim)
    # Also allow naming the file's main function when the bug is in a helper
    FUNC_ALIASES = {
        "pou_lqr_sim": {"pou_lqr_sim", "compute_force", "compute_velocity", "reference_pt"},
        "tick": {"tick", "stays_balanced"},
        "pou_general_drive": {"pou_general_drive", "calculate_theta_ist_axis2",
                              "calculate_v", "calculate_x"},
    }
    aliases = FUNC_ALIASES.get(gt_function, {gt_function})
    func_match = any(alias in diag_function or diag_function in alias
                     for alias in aliases)

    # File match
    gt_file = ground_truth.get("file", "")
    diag_file = diagnosis.get("fix_file", "")
    file_match = (
        Path(gt_file).name in diag_file
        or diag_file in gt_file
    )

    return {
        "category_correct": cat_match,
        "function_correct": func_match,
        "file_correct": file_match,
        "diagnosis_correct": cat_match and func_match,
    }


def check_fix(diagnosis: Dict, ground_truth: Dict) -> Dict[str, bool]:
    """Check if the proposed fix reverses the injected bug."""
    fix_old = diagnosis.get("fix_old", "").strip()
    fix_new = diagnosis.get("fix_new", "").strip()
    gt_buggy = ground_truth.get("buggy", "").strip()
    gt_original = ground_truth.get("original", "").strip()

    # Exact reversal: agent's old==buggy, agent's new==original
    exact_fix = (fix_old == gt_buggy and fix_new == gt_original)

    # Semantic fix: the buggy text appears in old AND the original text appears in new
    # (agent may include surrounding context in the fix)
    semantic_fix = (gt_buggy in fix_old and gt_original in fix_new)

    # Partial: at least identifies the buggy line
    partial_fix = gt_buggy in fix_old or fix_old in gt_buggy

    return {
        "exact_fix": exact_fix,
        "semantic_fix": semantic_fix,
        "partial_fix": partial_fix,
    }


# ── Agent loop ──────────────────────────────────────────────────────────

def run_agent_episode(
    bug_spec: Dict,
    aci: CPS_ACI,
    observation: Dict,
    model: str = "claude-haiku-4-5-20250414",
    max_turns: int = 15,
    dry_run: bool = False,
) -> Dict:
    """Run one debugging episode using Anthropic's tool_use API."""
    import anthropic

    client = anthropic.Anthropic()

    # Filter tool schemas: remove submit_diagnosis from the list shown initially
    tools = TOOL_SCHEMAS

    # Start with the anomaly report as context
    anomaly_text = aci.get_anomaly_report()

    messages = [
        {
            "role": "user",
            "content": (
                "An anomaly has been detected in the pendulum controller. "
                "Here is the monitoring system's report:\n\n"
                f"{anomaly_text}\n\n"
                "Investigate this anomaly using the available tools. "
                "Read the source code, check sensor data and traces, "
                "then submit your diagnosis and fix using submit_diagnosis."
            ),
        }
    ]

    if dry_run:
        print("=== SYSTEM PROMPT ===")
        print(SYSTEM_PROMPT)
        print("\n=== USER MESSAGE ===")
        print(messages[0]["content"])
        print("\n=== TOOLS ===")
        for t in tools:
            print(f"  {t['name']}: {t['description'][:80]}...")
        return {"dry_run": True}

    transcript = []
    diagnosis = None
    total_input_tokens = 0
    total_output_tokens = 0

    for turn in range(max_turns):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=tools,
        )

        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        # Process response content blocks
        assistant_content = response.content
        transcript.append({
            "turn": turn + 1,
            "role": "assistant",
            "content": [_serialize_block(b) for b in assistant_content],
            "stop_reason": response.stop_reason,
        })

        # Check if agent wants to use tools
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input

                    # Check for submit_diagnosis (terminal action)
                    if tool_name == "submit_diagnosis":
                        diagnosis = tool_input
                        result_text = "Diagnosis submitted. Thank you."
                    else:
                        result_text = aci.dispatch(tool_name, tool_input)

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })

                    transcript.append({
                        "turn": turn + 1,
                        "role": "tool",
                        "tool_name": tool_name,
                        "tool_input": tool_input,
                        "result": result_text[:500] + ("..." if len(result_text) > 500 else ""),
                    })

            # If diagnosis was submitted, we're done
            if diagnosis is not None:
                break

            # Otherwise continue the conversation with tool results
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})

        elif response.stop_reason == "end_turn":
            # Agent finished without calling a tool -- nudge it
            text_blocks = [b.text for b in assistant_content if hasattr(b, "text")]
            full_text = "\n".join(text_blocks)

            # Check if agent embedded diagnosis in text (shouldn't with proper tools)
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({
                "role": "user",
                "content": (
                    "Please use the submit_diagnosis tool to provide your "
                    "final answer with the code fix."
                ),
            })
        else:
            break

    # Score
    gt = bug_spec
    diag_scores = check_diagnosis(diagnosis, gt) if diagnosis else {
        "category_correct": False, "function_correct": False,
        "file_correct": False, "diagnosis_correct": False,
    }
    fix_scores = check_fix(diagnosis, gt) if diagnosis else {
        "exact_fix": False, "partial_fix": False,
    }

    return {
        "bug_id": bug_spec["id"],
        "difficulty": bug_spec["difficulty"],
        "model": model,
        "turns": turn + 1,
        "diagnosis": diagnosis,
        "scores": {**diag_scores, **fix_scores},
        "tokens": {
            "input": total_input_tokens,
            "output": total_output_tokens,
            "total": total_input_tokens + total_output_tokens,
        },
        "transcript": transcript,
    }


def _serialize_block(block) -> Dict:
    """Convert an Anthropic content block to a JSON-serializable dict."""
    if hasattr(block, "text"):
        return {"type": "text", "text": block.text}
    elif hasattr(block, "name"):
        return {"type": "tool_use", "name": block.name, "input": block.input}
    return {"type": str(type(block))}


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CPS Code Debugging Agent PoC")
    parser.add_argument("--bugs", nargs="*", help="Bug IDs to run (default: all)")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001",
                        help="Anthropic model ID")
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts without calling API")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load aligned dataset
    print("Loading aligned dataset...")
    data = pd.read_parquet(ALIGNED_PARQUET)
    all_cols = list(data.columns)
    physical_cols = [c for c in all_cols if c.startswith("dl_") or c.startswith("sl_")]
    branch_cols = [c for c in all_cols if ":" in c]
    print(f"  {len(data)} windows, {len(physical_cols)} sensor cols, {len(branch_cols)} trace edges")

    # Select bugs to run
    if args.bugs:
        selected = [b for b in BUGS if b["id"] in args.bugs]
    else:
        selected = BUGS

    results = []
    for bug in selected:
        print(f"\n{'='*60}")
        print(f"Bug: {bug['id']} ({bug['difficulty']})")
        print(f"  {bug['description'][:80]}...")
        print(f"{'='*60}")

        bug_dir = BUG_DIR / bug["id"]
        if not bug_dir.exists():
            print(f"  SKIP: buggy variant not found at {bug_dir}")
            continue

        # Build ACI with buggy source + original reference
        observation = simulate_anomaly_observation(bug, data, physical_cols, branch_cols)
        aci = CPS_ACI(
            source_dir=bug_dir,
            aligned_data=data,
            physical_cols=physical_cols,
            branch_cols=branch_cols,
            anomaly_observation=observation,
            reference_dir=ORIGINAL_SRC,
        )

        t0 = time.time()
        result = run_agent_episode(
            bug_spec=bug,
            aci=aci,
            observation=observation,
            model=args.model,
            max_turns=args.max_turns,
            dry_run=args.dry_run,
        )
        elapsed = time.time() - t0

        if args.dry_run:
            break

        result["elapsed_sec"] = round(elapsed, 1)
        results.append(result)

        scores = result["scores"]
        print(f"\n  Results:")
        print(f"    Category correct: {scores['category_correct']}")
        print(f"    Function correct: {scores['function_correct']}")
        print(f"    Diagnosis correct: {scores['diagnosis_correct']}")
        print(f"    Exact fix: {scores['exact_fix']}")
        print(f"    Partial fix: {scores['partial_fix']}")
        print(f"    Turns: {result['turns']}, Time: {elapsed:.1f}s")
        print(f"    Tokens: {result['tokens']}")

        # Save transcript
        transcript_path = OUT_DIR / "transcripts" / f"{bug['id']}.json"
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        with open(transcript_path, "w") as f:
            json.dump(result["transcript"], f, indent=2)

    if results:
        # Print summary
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")

        n = len(results)
        n_diag = sum(r["scores"]["diagnosis_correct"] for r in results)
        n_exact = sum(r["scores"]["exact_fix"] for r in results)
        n_partial = sum(r["scores"]["partial_fix"] for r in results)
        avg_turns = np.mean([r["turns"] for r in results])
        total_tokens = sum(r["tokens"]["total"] for r in results)

        print(f"Model: {args.model}")
        print(f"Scenarios: {n}")
        print(f"Correct diagnosis: {n_diag}/{n} ({100*n_diag/n:.0f}%)")
        print(f"Exact fix: {n_exact}/{n} ({100*n_exact/n:.0f}%)")
        print(f"Partial fix: {n_partial}/{n} ({100*n_partial/n:.0f}%)")
        print(f"Avg turns: {avg_turns:.1f}")
        print(f"Total tokens: {total_tokens:,}")

        # Per-difficulty breakdown
        for diff in ["easy", "medium", "hard"]:
            subset = [r for r in results if r["difficulty"] == diff]
            if subset:
                d = sum(r["scores"]["diagnosis_correct"] for r in subset)
                f = sum(r["scores"]["exact_fix"] for r in subset)
                p = sum(r["scores"]["partial_fix"] for r in subset)
                print(f"  {diff}: diagnosis {d}/{len(subset)}, "
                      f"exact_fix {f}/{len(subset)}, "
                      f"partial_fix {p}/{len(subset)}")

        # Save results (without large transcripts)
        summary = []
        for r in results:
            s = {k: v for k, v in r.items() if k != "transcript"}
            summary.append(s)
        summary_path = OUT_DIR / "evaluation_results.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
