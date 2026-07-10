"""
Task 5: Code-Behavior Consistency Verification (10 proof-of-concept problems).

A VLM examines (trajectory image + sensor readings + trace summary + source code)
and judges whether observed physical behavior is consistent with the code's expected
output. Tests three scenario types:
  - Normal (4): everything consistent
  - Sensor-code inconsistency (3): biased angle sensor contradicts calm visual trajectory
  - Trace-code inconsistency (3): SWINGUP traces but BALANCE video

Outputs:
  outputs/video_assessment/task5_verification_results.json
  outputs/video_assessment/fig_task5_verification_examples.png
"""

import base64
import json
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
WINDOW_FRAMES = 125

FULL_SOURCE_EXCERPT = """
/* ── State dispatch (pend_monolithic.c) ─────────────────────────── */
if (flags.su_enable && !flags.lqr_enable) {
    pendulum_state = LOG_SWING_UP;   // FSM state = 0 = SWINGUP
    pou_standup_rel(&gen_drive_out, &su_ctx, &su_sim_params,
                    &standup_out, &p1_motion_status);
    // pou_standup_rel: energise pendulum with oscillating cart force
    // until |theta_ist| < ~0.05 rad → sets su_pndlum_ok = true
}
if ((standup_out.su_pndlum_ok == true) || flags.lqr_enable) {
    pendulum_state = LOG_BALANCE;    // FSM state = 1 = BALANCE
    pou_lqr_sim(&lqr_ctx, &p1_lqr_sim_params, &gen_drive_out,
                main_out.x_soll, &lqr_sim_out);
}

/* ── LQR force computation (lqrsim.c) ───────────────────────────── */
// LQR gains (Pittsburgh pendulum):
//   K1=12 (position), K2=54 (velocity), K3=274 (angle), K4=36 (angular velocity)
// F_lqr = K1*(x_ref - x_ist) + K2*v_ist + K3*theta_ist + K4*theta_d_ist
// For theta_ist = 0.26 rad (~15°): F_contribution = 274 * 0.26 = 71.2 N
// → large corrective force → large cart velocity → large visual motion
// For theta_ist = 0.03 rad (~2°): F_contribution = 274 * 0.03 = 8.2 N
// → small corrective force → calm small-amplitude motion
double A3 = (fabs(gen_drive_out->theta_ist) > 0.005) ? 274.0 * gen_drive_out->theta_ist : 0.0;
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


def render_trajectory(ct_window: pd.DataFrame, title: str) -> str:
    fig, ax = plt.subplots(figsize=(4, 4))
    x = ct_window["ct_mean_x"].values
    y = ct_window["ct_mean_y"].values
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) > 1:
        ax.plot(x, y, "b-", lw=1.5, alpha=0.7)
        ax.scatter(x[0], y[0], c="green", s=60, zorder=5, label="Start")
        ax.scatter(x[-1], y[-1], c="red", s=60, zorder=5, label="End")
    spread = ct_window["ct_spread_y"].mean() if "ct_spread_y" in ct_window.columns else np.nan
    ax.set_title(f"{title}\nVertical spread: {spread:.1f}px", fontsize=8)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.legend(fontsize=7)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def format_sensors(dl_window, phys_cols, angle_col, override_angle_deg=None):
    lines = []
    for c in phys_cols:
        v = dl_window[c].mean()
        if c == angle_col and override_angle_deg is not None:
            v = override_angle_deg * np.pi / 180
        unit = "rad" if "angle" in c else ("m" if "x" in c else "")
        lines.append(f"  {c}: {v:.5f} {unit}")
    state_val = int(dl_window["dl_state"].mode()[0]) if "dl_state" in dl_window.columns else 1
    lines.append(f"  pendulum_state: {'BALANCE' if state_val == 1 else 'SWINGUP'}")
    return "\n".join(lines)


def format_traces(dl_window, branch_cols, override_with_swingup_dl=None):
    src = override_with_swingup_dl if override_with_swingup_dl is not None else dl_window
    avail = [c for c in branch_cols if c in src.columns][:8]
    if not avail:
        return "  (no branch data)"
    vals = src[avail].mean()
    lines = [f"  {c}: {v:.1f}" for c, v in vals.items()]
    total = src[avail].values.mean()
    lines.append(f"  mean branch count: {total:.1f}")
    return "\n".join(lines)


def build_verification_prompt(traj_b64, sensor_text, trace_text, scenario_type):
    return f"""You are a cyber-physical systems debugging agent. Verify whether the observed
physical behavior is consistent with what the controller code should produce.

## Visual observation
The image shows the pendulum's tracked trajectory over ~1 second (125 frames).
Green dot = start, red dot = end. Vertical spread indicates swing amplitude.

## Reported sensor readings
{sensor_text}

## Reported execution trace (branch counter means)
{trace_text}

## Controller source code
```c
{FULL_SOURCE_EXCERPT}
```

## Your task
Think step by step. For each piece of evidence, state what it tells you and whether
it agrees with the other pieces. Consider:
- At theta_ist ≈ 0.03 rad (2°) in BALANCE: the LQR produces small forces → calm trajectory
- At theta_ist ≈ 0.26 rad (15°) in BALANCE: the LQR produces large forces → large trajectory
- SWINGUP trace counts look very different from BALANCE trace counts

Respond in JSON:
{{
  "visual_sensor_consistent": true/false,
  "visual_code_consistent": true/false,
  "trace_code_consistent": true/false,
  "overall_consistent": true/false,
  "fault_type": "normal" | "sensor_bias" | "mode_confusion",
  "reasoning": "..."
}}"""


def call_claude(prompt: str, image_b64: str, model: str, retries: int = 3) -> dict:
    try:
        import anthropic
    except ImportError:
        return {"error": "anthropic not installed"}
    client = anthropic.Anthropic()
    for attempt in range(retries):
        try:
            rsp = client.messages.create(
                model=model,
                max_tokens=800,
                messages=[{"role": "user", "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
                    {"type": "text", "text": prompt},
                ]}],
            )
            text = rsp.content[0].text.strip()
            if "{" in text:
                text = text[text.index("{"):text.rindex("}")+1]
            return json.loads(text)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return {"error": str(e)}


def construct_problems(sessions, df_all):
    problems = []
    BIAS_DEG = 15.0  # large bias to make visual inconsistency obvious

    for run_id in sessions[:2]:
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

        sample_starts = bal_idx[::max(1, len(bal_idx)//8)].tolist()

        for wi, start in enumerate(sample_starts[:6]):
            end = min(start + WINDOW_FRAMES, len(dl))
            ct_win = ct.iloc[start:end]
            dl_win = dl.iloc[start:end]

            slot = wi % 3  # 0=normal, 1=sensor inconsistency, 2=trace inconsistency

            if slot == 0:
                scenario = "normal"
                gt_overall = True
                sensor_text = format_sensors(dl_win, phys_cols, angle_col)
                trace_text  = format_traces(dl_win, branch_cols)
                traj_title  = "Normal BALANCE"

            elif slot == 1:
                scenario = "sensor_bias"
                gt_overall = False
                # Video shows calm BALANCE (~2°), but sensor reports 15°
                sensor_text = format_sensors(dl_win, phys_cols, angle_col,
                                             override_angle_deg=BIAS_DEG)
                trace_text  = format_traces(dl_win, branch_cols)
                traj_title  = "Calm BALANCE (true ~2°)"

            else:
                scenario = "mode_confusion"
                gt_overall = False
                sensor_text = format_sensors(dl_win, phys_cols, angle_col)
                # Inject SWINGUP traces
                if len(swi_idx) >= WINDOW_FRAMES:
                    swi_win = dl.iloc[swi_idx[len(swi_idx)//2]:swi_idx[len(swi_idx)//2]+WINDOW_FRAMES]
                    trace_text = format_traces(dl_win, branch_cols, override_with_swingup_dl=swi_win)
                else:
                    trace_text = format_traces(dl_win, branch_cols)
                traj_title = "BALANCE video (pendulum upright)"

            traj_b64 = render_trajectory(ct_win, traj_title)

            # Save PNG
            png_path = TRAJ_DIR / f"task5_{run_id}_{start}_{scenario}.png"
            png_path.write_bytes(base64.b64decode(traj_b64))

            prompt = build_verification_prompt(traj_b64, sensor_text, trace_text, scenario)

            problems.append({
                "id":           len(problems),
                "run_id":       run_id,
                "scenario":     scenario,
                "gt_overall_consistent": gt_overall,
                "traj_path":    str(png_path),
                "traj_b64":     traj_b64,
                "prompt":       prompt,
            })
            if len(problems) >= 10:
                break
        if len(problems) >= 10:
            break

    return problems


def main():
    print("=" * 60)
    print("Task 5: Code-Behavior Consistency Verification")
    print("=" * 60)

    df_all   = pd.read_parquet(ALIGNED_PARQUET)
    sessions = get_video_sessions()

    problems = construct_problems(sessions, df_all)
    print(f"  Constructed {len(problems)} problems")

    model = "claude-sonnet-4-6"   # Task 5 uses Sonnet for richer reasoning
    try:
        import anthropic
        api_available = True
        print(f"  Using {model} for {len(problems)} problems")
    except ImportError:
        api_available = False
        print("  anthropic SDK not available — saving prompts only")

    results = []
    if api_available:
        for pi, prob in enumerate(problems):
            print(f"  Problem {pi+1}/{len(problems)} ({prob['scenario']})...", end=" ")
            resp = call_claude(prob["prompt"], prob["traj_b64"], model)
            correct = (resp.get("overall_consistent") == prob["gt_overall_consistent"]
                       if "error" not in resp else False)
            results.append({
                "id":         prob["id"],
                "scenario":   prob["scenario"],
                "gt_overall": prob["gt_overall_consistent"],
                "response":   resp,
                "correct":    correct,
            })
            print("✓" if correct else "✗", f"  fault={resp.get('fault_type','?')}")

    # Save results
    out_dict = {
        "problems": [{k: v for k, v in p.items() if k not in ("traj_b64", "prompt")}
                     for p in problems],
        "responses": results,
    }
    (OUT_DIR / "task5_verification_results.json").write_text(json.dumps(out_dict, indent=2))
    print("  Saved task5_verification_results.json")

    # Figure: side-by-side examples (up to 3)
    n_examples = min(3, len(problems))
    fig, axes = plt.subplots(1, n_examples, figsize=(5 * n_examples, 5))
    if n_examples == 1:
        axes = [axes]

    for ei in range(n_examples):
        prob = problems[ei]
        img  = plt.imread(prob["traj_path"])
        axes[ei].imshow(img)
        axes[ei].axis("off")
        resp = results[ei] if results else {}
        pred = resp.get("response", {}).get("fault_type", "N/A")
        color = "green" if resp.get("correct", False) else "red"
        axes[ei].set_title(f"Scenario: {prob['scenario']}\nPredicted: {pred}",
                           fontsize=9, color=color)

    plt.suptitle("Task 5: Code-Behavior Consistency — Example Trajectories", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig_task5_verification_examples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_task5_verification_examples.png")

    if results:
        acc = np.mean([r["correct"] for r in results])
        print(f"\n  Overall accuracy: {acc:.0%} ({sum(r['correct'] for r in results)}/{len(results)})")
        for sc in ["normal", "sensor_bias", "mode_confusion"]:
            sc_res = [r for r in results if r["scenario"] == sc]
            if sc_res:
                sc_acc = np.mean([r["correct"] for r in sc_res])
                print(f"  {sc}: {sc_acc:.0%} ({len(sc_res)} problems)")


if __name__ == "__main__":
    main()
