"""
Task 4: Visual Evidence in LLM Reasoning.

Constructs 30 test problems (10 normal + 5 each of 4 fault types).
For each problem, creates:
  - numbers-only version (text prompt)
  - numbers+trajectory version (text + CoTracker trajectory image)

Runs both through Claude Haiku. Compares detection accuracy.

Outputs:
  outputs/video_assessment/trajectory_plots/    PNG trajectory images
  outputs/video_assessment/task4_problems.json  all 30 problems + prompts
  outputs/video_assessment/task4_llm_results.json
  outputs/video_assessment/fig_task4_vlm_comparison.png
  outputs/video_assessment/fig_task4_per_type_improvement.png
"""

import base64
import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT      = Path(__file__).resolve().parents[2]
OUT_DIR   = ROOT / "outputs/video_assessment"
FEAT_DIR  = OUT_DIR / "features"
TRAJ_DIR  = OUT_DIR / "trajectory_plots"
TRAJ_DIR.mkdir(parents=True, exist_ok=True)
ALIGNED_PARQUET = ROOT / "outputs/phase3/aligned_dataset_expanded.parquet"

BALANCE_STATE = 1
WINDOW_FRAMES = 125   # ~1 second at 125 Hz video ≈ 100 datalayer windows

# Controller source code excerpt (LQR + tick dispatch)
SOURCE_CODE_EXCERPT = """
/* pou_lqr_sim: LQR balancing controller (called during BALANCE state) */
static void compute_force(lqr_sim_params_t *params,
                          const general_drive_out_t *gen_drive_out,
                          const double xsoll,
                          lqr_sim_outputs_t *lqr_sim_out) {
    double A1 = 12.0 * (xsoll - gen_drive_out->x_ist);
    double A2 = (fabs(gen_drive_out->v_ist) > 0.0001) ? 54.0 * gen_drive_out->v_ist : 0.0;
    double A3 = (fabs(gen_drive_out->theta_ist) > 0.005) ? 274.0 * gen_drive_out->theta_ist : 0.0;
    double A4 = 36.0 * gen_drive_out->theta_d_ist;
    // LQR: F = A1 + A2 + A3 + A4 (angle signs reversed for Pittsburgh pendulum)
    F_lqr = A1 + A2 + A3 + A4;
    F_lqr = fmax(-params->F_max, fmin(params->F_max, F_lqr));
    lqr_sim_out->LQ_F_soll = F_lqr;
}

/* State dispatch (simplified from pend_monolithic.c) */
if (flags.su_enable && !flags.lqr_enable) {
    pendulum_state = LOG_SWING_UP;   // state = 0
    pou_standup_rel(&gen_drive_out, &su_ctx, &su_sim_params,
                    &standup_out, &p1_motion_status);
}
if ((standup_out.su_pndlum_ok == true) || flags.lqr_enable) {
    pendulum_state = LOG_BALANCE;    // state = 1
    pou_lqr_sim(&lqr_ctx, &p1_lqr_sim_params, &gen_drive_out,
                main_out.x_soll, &lqr_sim_out);
}
/* Condition to switch from swing-up to balance: su_pndlum_ok set when
   |theta_ist| < BALANCE_THRESHOLD (approximately 0.05 rad = ~3 degrees) */
"""


def get_video_sessions():
    p = OUT_DIR / "dataset_params.json"
    d = json.loads(p.read_text())
    return d.get("full_video_sessions", d.get("video_sessions", []))


def load_session_data(run_id: str, df_all: pd.DataFrame):
    al_path = OUT_DIR / f"aligned_tier1_{run_id}.parquet"
    if not al_path.exists():
        return None, None
    ct = pd.read_parquet(al_path).set_index("win")
    ct.columns = [c.replace("t1_", "") for c in ct.columns]
    dl = df_all[df_all["run"] == run_id].sort_values("win").set_index("win")
    common = dl.index.intersection(ct.index)
    return ct.loc[common], dl.loc[common]


def render_trajectory(ct_window: pd.DataFrame, label: str, fault_note: str = "") -> str:
    """Render CoTracker trajectory, return base64 PNG."""
    fig, ax = plt.subplots(figsize=(4, 4))

    x = ct_window["ct_mean_x"].values
    y = ct_window["ct_mean_y"].values
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    if len(x) > 1:
        ax.plot(x, y, "b-", lw=1.5, alpha=0.7)
        ax.scatter(x[0],  y[0],  c="green", s=60, zorder=5, label="Start")
        ax.scatter(x[-1], y[-1], c="red",   s=60, zorder=5, label="End")

        # Direction arrows every 20 frames
        for i in range(0, len(x) - 1, max(1, len(x)//8)):
            dx, dy = x[i+1] - x[i], y[i+1] - y[i]
            ax.annotate("", xy=(x[i+1], y[i+1]), xytext=(x[i], y[i]),
                        arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))

    spread = ct_window["ct_spread_y"].mean() if "ct_spread_y" in ct_window.columns else np.nan
    title  = f"{label}\nVertical spread: {spread:.1f}px"
    if fault_note:
        title += f"\n[{fault_note}]"
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.legend(fontsize=7)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_sensor_summary(dl_window: pd.DataFrame, branch_cols: list) -> str:
    phys_cols = [c for c in dl_window.columns if c.startswith("dl_")
                 and c not in {"dl_state", "dl_target_x", "dl_iteration"}]
    lines = []
    for c in phys_cols:
        v = dl_window[c].mean()
        lines.append(f"  {c}: {v:.5f}")
    state_map = {0: "SWINGUP", 1: "BALANCE", 2: "RESET"}
    state_val = int(dl_window["dl_state"].mode()[0]) if "dl_state" in dl_window.columns else -1
    lines.append(f"  pendulum_state: {state_map.get(state_val, state_val)}")
    return "\n".join(lines)


def build_trace_summary(dl_window: pd.DataFrame, branch_cols: list) -> str:
    if not branch_cols:
        return "  (no branch data)"
    avail = [c for c in branch_cols if c in dl_window.columns][:10]
    vals  = dl_window[avail].mean()
    lines = [f"  {c}: {v:.1f}" for c, v in vals.items()]
    total_mean = dl_window[avail].values.mean()
    lines.append(f"  (mean across {len(avail)} branch counters: {total_mean:.1f})")
    return "\n".join(lines)


def build_prompt(sensor_text: str, trace_text: str, trajectory_b64: str | None = None) -> str:
    vis_section = ""
    if trajectory_b64:
        vis_section = """
## Visual evidence
The image shows the pendulum's tracked trajectory over ~1 second. Green=start, red=end.
Vertical spread indicates swing amplitude.
"""
    inst = ("Based on ALL available evidence — sensors, traces, source code, AND the visual trajectory"
            if trajectory_b64 else
            "Based on the sensor readings, execution traces, and source code")

    return f"""You are diagnosing a potential anomaly in an industrial inverted pendulum controlled by a Bosch Rexroth ctrlX system running LQR balancing in WebAssembly.

## Sensor readings (10ms window average)
{sensor_text}

## Execution trace (branch counter means)
{trace_text}

## Controller source code (key excerpt)
```c
{SOURCE_CODE_EXCERPT}
```
{vis_section}
{inst}, answer:
1. Is the physical-sensor-trace combination consistent? (yes/no)
2. If inconsistent, what type of fault is most likely?
   (normal / sensor_bias / sensor_freeze / sensor_drift / mode_confusion / timing_delay)
3. Which specific component is implicated?
4. Brief reasoning citing specific evidence.

Respond in JSON:
{{"consistent": true/false, "fault_type": "...", "component": "...", "reasoning": "..."}}"""


def construct_problems(sessions, df_all):
    """Build 30 test problems: 10 normal + 5×4 fault types."""
    problems = []
    BIAS_RAD = 5.0 * np.pi / 180
    DRIFT_RATE = 0.5 * np.pi / 180 / 100  # per window

    for run_id in sessions[:3]:  # use first 3 sessions
        ct, dl = load_session_data(run_id, df_all)
        if ct is None:
            continue

        branch_cols = [c for c in dl.columns if ":" in c]
        phys_cols   = [c for c in dl.columns if c.startswith("dl_")
                       and c not in {"dl_state", "dl_target_x", "dl_iteration"}]
        angle_col   = next((c for c in phys_cols if "angle" in c and "mean" in c), None)

        bal_idx = np.where(dl["dl_state"].values == BALANCE_STATE)[0]
        swi_idx = np.where(dl["dl_state"].values == 0)[0]
        if len(bal_idx) < WINDOW_FRAMES * 6:
            continue

        # Sample windows spaced apart
        sample_starts = bal_idx[::max(1, len(bal_idx)//(15))].tolist()

        for wi, start in enumerate(sample_starts[:15]):
            end = min(start + WINDOW_FRAMES, len(dl))
            ct_win = ct.iloc[start:end]
            dl_win = dl.iloc[start:end]

            sensor_text = build_sensor_summary(dl_win, branch_cols)
            trace_text  = build_trace_summary(dl_win, branch_cols)

            # Determine fault type for this problem slot
            slot = wi % 5  # 0=normal, 1=bias, 2=freeze, 3=mode_confusion, 4=drift

            fault_note = ""
            dl_win_fault = dl_win.copy()
            trace_text_fault = trace_text

            if slot == 0:  # normal
                gt_fault = "normal"
            elif slot == 1:  # sensor bias
                gt_fault = "sensor_bias"
                if angle_col:
                    dl_win_fault[angle_col] = dl_win[angle_col] + BIAS_RAD
                fault_note = f"angle sensor biased +{5}°"
                sensor_text = build_sensor_summary(dl_win_fault, branch_cols)
            elif slot == 2:  # sensor freeze
                gt_fault = "sensor_freeze"
                if angle_col:
                    dl_win_fault[angle_col] = dl_win[angle_col].iloc[0]
                fault_note = "angle sensor frozen at constant value"
                sensor_text = build_sensor_summary(dl_win_fault, branch_cols)
            elif slot == 3:  # mode confusion
                gt_fault = "mode_confusion"
                if len(swi_idx) >= WINDOW_FRAMES:
                    swi_start = swi_idx[len(swi_idx)//2]
                    swi_end   = min(swi_start + WINDOW_FRAMES, len(dl))
                    swi_win   = dl.iloc[swi_start:swi_end]
                    trace_text_fault = build_trace_summary(swi_win, branch_cols)
                fault_note = "traces from SWINGUP injected into BALANCE window"
                trace_text = trace_text_fault
            elif slot == 4:  # drift
                gt_fault = "sensor_drift"
                if angle_col:
                    drift_arr = np.arange(len(dl_win_fault)) * DRIFT_RATE
                    dl_win_fault[angle_col] = dl_win[angle_col].values + drift_arr
                fault_note = "angle sensor drifting at +0.5°/s"
                sensor_text = build_sensor_summary(dl_win_fault, branch_cols)

            traj_b64 = render_trajectory(ct_win, f"{run_id[-8:]} win{start}", fault_note)

            # Save trajectory PNG
            png_path = TRAJ_DIR / f"traj_{run_id}_{start}_{gt_fault}.png"
            png_path.write_bytes(base64.b64decode(traj_b64))

            problem = {
                "id":          len(problems),
                "run_id":      run_id,
                "window_start": start,
                "gt_fault":    gt_fault,
                "traj_path":   str(png_path),
                "traj_b64":    traj_b64,
                "prompt_text_only": build_prompt(sensor_text, trace_text, None),
                "prompt_with_vis":  build_prompt(sensor_text, trace_text, traj_b64),
            }
            problems.append(problem)
            if len(problems) >= 30:
                break
        if len(problems) >= 30:
            break

    print(f"  Constructed {len(problems)} problems")
    return problems


def call_claude(prompt: str, image_b64: str | None, model: str, retries: int = 3) -> dict:
    """Call Claude API. Returns parsed JSON response or error dict."""
    try:
        import anthropic
    except ImportError:
        return {"error": "anthropic not installed"}

    client = anthropic.Anthropic()

    messages_content = []
    if image_b64:
        messages_content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": image_b64},
        })
    messages_content.append({"type": "text", "text": prompt})

    for attempt in range(retries):
        try:
            rsp = client.messages.create(
                model=model,
                max_tokens=512,
                messages=[{"role": "user", "content": messages_content}],
            )
            text = rsp.content[0].text.strip()
            # Extract JSON
            if "{" in text:
                text = text[text.index("{"):text.rindex("}")+1]
            return json.loads(text)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return {"error": str(e)}


def main():
    print("=" * 60)
    print("Task 4: Visual Evidence in LLM Reasoning")
    print("=" * 60)

    df_all   = pd.read_parquet(ALIGNED_PARQUET)
    sessions = get_video_sessions()

    problems = construct_problems(sessions, df_all)
    (OUT_DIR / "task4_problems.json").write_text(
        json.dumps([{k: v for k, v in p.items() if k != "traj_b64"} for p in problems], indent=2))
    print(f"  Saved task4_problems.json ({len(problems)} problems)")

    # Determine if Claude API is available
    model = "claude-haiku-4-5-20251001"
    try:
        import anthropic
        api_available = True
        print(f"  Anthropic SDK available — running {len(problems)*2} API calls with {model}")
    except ImportError:
        api_available = False
        print("  anthropic SDK not installed — saving prompts only, skipping API calls")

    results = []
    if api_available:
        for pi, prob in enumerate(problems):
            print(f"  Problem {pi+1}/{len(problems)} ({prob['gt_fault']})...", end=" ")

            resp_text  = call_claude(prob["prompt_text_only"], None, model)
            resp_vis   = call_claude(prob["prompt_with_vis"], prob["traj_b64"], model)

            def correct(resp, gt):
                if "error" in resp:
                    return False
                pred = resp.get("fault_type", "").lower()
                return pred == gt or (gt == "normal" and resp.get("consistent", False) is True)

            results.append({
                "id":               prob["id"],
                "gt_fault":         prob["gt_fault"],
                "text_only_pred":   resp_text.get("fault_type", "error"),
                "text_only_correct":correct(resp_text, prob["gt_fault"]),
                "with_vis_pred":    resp_vis.get("fault_type", "error"),
                "with_vis_correct": correct(resp_vis, prob["gt_fault"]),
                "text_only_resp":   resp_text,
                "with_vis_resp":    resp_vis,
            })
            status = "✓" if results[-1]["with_vis_correct"] else "✗"
            print(f"text={results[-1]['text_only_correct']} vis={status}")

    (OUT_DIR / "task4_llm_results.json").write_text(json.dumps(results, indent=2))
    print("  Saved task4_llm_results.json")

    if not results:
        print("  No API results — skipping figure generation.")
        return

    # Figures
    fault_types = ["normal", "sensor_bias", "sensor_freeze", "mode_confusion", "sensor_drift"]
    text_acc = {ft: [] for ft in fault_types}
    vis_acc  = {ft: [] for ft in fault_types}
    for r in results:
        ft = r["gt_fault"]
        if ft in text_acc:
            text_acc[ft].append(int(r["text_only_correct"]))
            vis_acc[ft].append(int(r["with_vis_correct"]))

    overall_text = np.mean([r["text_only_correct"] for r in results])
    overall_vis  = np.mean([r["with_vis_correct"]   for r in results])

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(["Text only", "Text + Trajectory"], [overall_text, overall_vis],
            color=["#90CAF9", "#1E88E5"])
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"Overall accuracy\n(n={len(results)} problems)")
    for i, v in enumerate([overall_text, overall_vis]):
        ax1.text(i, v + 0.02, f"{v:.0%}", ha="center")

    labels = [ft.replace("_", "\n") for ft in fault_types]
    t_vals = [np.mean(text_acc[ft]) if text_acc[ft] else 0 for ft in fault_types]
    v_vals = [np.mean(vis_acc[ft])  if vis_acc[ft]  else 0 for ft in fault_types]
    x = np.arange(len(fault_types))
    ax2.bar(x - 0.2, t_vals, 0.35, label="Text only", color="#90CAF9")
    ax2.bar(x + 0.2, v_vals, 0.35, label="Text + Trajectory", color="#1E88E5")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Per-fault-type accuracy")
    ax2.legend()
    plt.tight_layout()
    fig1.savefig(OUT_DIR / "fig_task4_vlm_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # Per-type delta
    fig2, ax = plt.subplots(figsize=(8, 4))
    deltas = [np.mean(vis_acc[ft]) - np.mean(text_acc[ft]) if text_acc[ft] else 0
              for ft in fault_types]
    colors = ["#43A047" if d > 0 else "#E53935" for d in deltas]
    ax.bar(labels, deltas, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Accuracy delta (vis − text)")
    ax.set_title("Improvement from adding trajectory visualization")
    plt.tight_layout()
    fig2.savefig(OUT_DIR / "fig_task4_per_type_improvement.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("  Saved fig_task4_vlm_comparison.png, fig_task4_per_type_improvement.png")


if __name__ == "__main__":
    main()
