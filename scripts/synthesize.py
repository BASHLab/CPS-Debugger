"""
synthesize.py — Wave 4: collect all Phase 3 results and write PHASE3_SUMMARY.md.
"""

import json
from pathlib import Path

ROOT   = Path("/home/simran/allspark-data-exploration/CPS-Debugger")
P3_OUT = ROOT / "outputs/phase3"
OUT_MD = P3_OUT / "PHASE3_SUMMARY.md"


def load_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default or {}


def section(title: str) -> str:
    return f"\n## {title}\n"


def main():
    lines = ["# CPS-Debugger Phase 3: Results Summary\n",
             f"Generated automatically by synthesize.py\n"]

    # ── 1. Per-function R² ────────────────────────────────────────────────────
    lines.append(section("1. Per-Function R² Within BALANCE State"))
    fn_r2 = load_json(P3_OUT / "function_r2_summary.json")
    if fn_r2:
        lines.append(f"**Overall mean within-BALANCE R²**: "
                     f"{fn_r2.get('overall_mean_within_balance_r2', 'N/A'):.3f}\n")
        lines.append("| Function Group | n branches | Mean R² | Frac R²>0.3 |")
        lines.append("|---|---|---|---|")
        for g in fn_r2.get("groups", []):
            lines.append(
                f"| {g['group']:20s} | {g['n_branches']:3d} | "
                f"{g['mean_r2']:.3f} | {g['frac_r2_gt_03']:.2f} |"
            )
        # Decision
        lqr = next((g for g in fn_r2.get("groups", []) if g["group"] == "pou_lqr_sim"), None)
        if lqr:
            r2 = lqr["mean_r2"]
            if r2 > 0.2:
                conclusion = f"**PROCEED**: pou_lqr_sim R²={r2:.3f} > 0.2 — fine-grained claim holds"
            else:
                conclusion = f"**REFRAME**: pou_lqr_sim R²={r2:.3f} — story is data-flow, not control-logic"
            lines.append(f"\n**Decision**: {conclusion}\n")
    else:
        lines.append("*Results not yet available*\n")

    # ── 2. Setpoint verification ──────────────────────────────────────────────
    lines.append(section("2. Setpoint Label Verification"))
    sv = load_json(P3_OUT / "setpoint_verification.json")
    if sv:
        lines.append("| Target label | n windows | Mean achieved pos | Tracking error |")
        lines.append("|---|---|---|---|")
        for s in sv.get("target_x_stats", []):
            lines.append(
                f"| {s['target_x_label']:+.3f} m | {s['n_windows']:,} | "
                f"{s['mean_achieved_x']:+.4f} m | {s['mean_tracking_error']:+.4f} m |"
            )
    else:
        lines.append("*Results not yet available*\n")

    # ── 3. Data inventory ─────────────────────────────────────────────────────
    lines.append(section("3. Dataset Inventory"))
    inv = load_json(P3_OUT / "data_inventory_summary.json")
    if inv:
        lines.append(f"- **Complete runs**: {inv.get('n_complete_runs', '?')}")
        lines.append(f"- **Approx 10ms windows**: {inv.get('approx_10ms_windows_complete', '?'):,}")
        lines.append(f"- **With audio**: {inv.get('complete_with_audio', '?')}")
        lines.append(f"- **With video**: {inv.get('complete_with_video', '?')}")
        lines.append(f"- **Unextracted tarballs**: {inv.get('n_unextracted_tarballs', '?')}")
        lines.append(f"\nComplete run IDs: {inv.get('complete_run_ids', [])}\n")
    else:
        lines.append("*Results not yet available*\n")

    # ── 4. Video / DINOv2 ─────────────────────────────────────────────────────
    lines.append(section("4. Video Modality (DINOv2)"))
    dino = load_json(P3_OUT / "dino_branch_r2_summary.json")
    vid  = load_json(P3_OUT / "video_inventory.json")
    if dino:
        lines.append(f"- Run analyzed: `{dino.get('run_id', '?')}`")
        lines.append(f"- Frames extracted: {dino.get('n_frames_extracted', '?')}")
        lines.append(f"- DINOv2→trace R² (within BALANCE): **{dino.get('dino_r2_mean_within_balance', 0):.3f}**")
        lines.append(f"- Physical→trace R² (same windows): {dino.get('physical_r2_mean_same_windows', 0):.3f}")
        additive = dino.get("is_video_additive", False)
        lines.append(f"- **Video is {'additive' if additive else 'NOT additive'} beyond physical sensors**\n")
    elif vid:
        lines.append(f"- {vid.get('n_runs_with_video', 0)} runs have video")
        best = vid.get("best_dino_candidate")
        if best:
            lines.append(f"- Best candidate for DINOv2: `{best}`")
            bi = vid.get("best_candidate_info", {})
            lines.append(f"  - Has datalayer: {bi.get('has_datalayer', '?')}")
            if not bi.get("has_datalayer"):
                lines.append("  - **WARNING**: Best video run has no datalayer — cannot correlate with trace")
        lines.append("*DINOv2 extraction pending (SLURM job)*\n")
    else:
        lines.append("*Video inventory not yet available*\n")

    # ── 5. Audio ──────────────────────────────────────────────────────────────
    lines.append(section("5. Audio Modality"))
    audio = load_json(P3_OUT / "audio_summary.json")
    if audio:
        status = audio.get("status", "")
        if status in ("NO_COMPLETE_AUDIO_RUNS", "NO_DATA"):
            lines.append(f"- **{status}**: {audio.get('message', '')}")
            lines.append("- Audio analysis deferred — no runs have both audio and physical sensors\n")
        else:
            lines.append(f"- Runs analyzed: {audio.get('n_audio_runs_analyzed', 0)}")
            lines.append(f"- Overall audio R²: **{audio.get('overall_audio_r2_mean', 0):.3f}**")
            lines.append(f"- Overall physical R²: {audio.get('overall_physical_r2_mean', 0):.3f}\n")
    else:
        lines.append("*Results not yet available*\n")

    # ── 6. LLM zero-shot anomaly explanation ──────────────────────────────────
    lines.append(section("6. Zero-Shot LLM Anomaly Explanation Baseline"))
    llm = load_json(P3_OUT / "anomaly_zero_shot_results.json")
    if llm:
        ov = llm.get("overall", {})
        lines.append(f"- Model: `{llm.get('model', '?')}`")
        lines.append(f"- n_examples: {llm.get('n_examples', '?')}")
        lines.append(f"- Detection accuracy:   **{ov.get('detection_acc', 0):.3f}**")
        lines.append(f"- Typing accuracy:      **{ov.get('typing_acc', 0):.3f}**")
        lines.append(f"- Localization accuracy: {ov.get('localization_acc', 0):.3f}")
        lines.append(f"- Reasoning quality:    {ov.get('reasoning_score', 0):.3f}")
        lines.append(f"- Total score:          **{ov.get('total_score', 0):.3f}**")
        lines.append(f"- Parse error rate:     {ov.get('parse_error_rate', 0):.3f}")
        lines.append("\nPer-type scores:")
        lines.append("| Type | Mean score | n |")
        lines.append("|---|---|---|")
        for t, info in sorted(llm.get("per_type", {}).items()):
            lines.append(f"| {t:20s} | {info['mean_score']:.3f} | {info['n']} |")
        lines.append("")
    else:
        lines.append("*LLM evaluation pending (requires API key + task_3_1b)*\n")

    # ── 7. GRPO pipeline status ───────────────────────────────────────────────
    lines.append(section("7. GRPO Training Pipeline"))
    rw = load_json(P3_OUT / "reward_validation.json")
    vc = load_json(P3_OUT / "verl_config_summary.json")
    if rw:
        lines.append(f"- Reward function: `scripts/phase3_reward_anomaly.py`")
        lines.append(f"- Mean reward (zero-shot baseline): "
                     f"{rw.get('mean_reward', 'N/A') if isinstance(rw.get('mean_reward'), float) else 'N/A'}")
        lines.append(f"- Assessment: {rw.get('assessment', rw.get('status', '?'))}\n")
    if vc:
        cfg = vc.get("gpu_config", {})
        lines.append(f"- Config: {cfg.get('n_gpus', '?')}× {cfg.get('gpu_type', '?')} GPU, "
                     f"model={cfg.get('model', '?')}, LoRA={cfg.get('use_lora', '?')}")
        lines.append(f"- Submit: `{vc.get('submit_command', 'sbatch slurm/submit_anomaly_grpo.sh')}`\n")
    else:
        lines.append("*GRPO config not yet generated*\n")

    # ── 8. Expanded dataset ───────────────────────────────────────────────────
    lines.append(section("8. Expanded Dataset Results"))
    exp = load_json(P3_OUT / "expanded_results_summary.json")
    if exp:
        wb = exp.get("within_balance_r2", {})
        cr = exp.get("cross_run_r2_ridge", {})
        lines.append(f"- Within-BALANCE R²: 6-run={wb.get('6_run', 'N/A'):.3f}, "
                     f"expanded={wb.get('expanded', 'N/A'):.3f} "
                     f"(Δ={wb.get('delta', 0):+.3f})")
        lines.append(f"- Cross-run R²:      6-run={cr.get('6_run', 'N/A'):.3f}, "
                     f"expanded={cr.get('expanded', 'N/A'):.3f} "
                     f"(Δ={cr.get('delta', 0):+.3f})")
        lines.append(f"- Runs: {exp.get('n_runs', {}).get('6_run', '?')} → "
                     f"{exp.get('n_runs', {}).get('expanded', '?')}\n")
    else:
        lines.append("*Expanded dataset results not yet available (needs task_1_3b)*\n")

    # ── 9. Training datasets ──────────────────────────────────────────────────
    lines.append(section("9. Training Dataset Status"))
    for fname, desc in [
        ("anomaly_eval_dataset.jsonl", "LLM eval dataset (250 examples)"),
        ("grpo_train.jsonl",           "GRPO training data (3000 examples)"),
        ("sft_anomaly_train.jsonl",    "SFT train data (gold reasoning)"),
        ("sft_anomaly_val.jsonl",      "SFT val data"),
    ]:
        path = P3_OUT / fname
        if path.exists():
            n = sum(1 for _ in open(path))
            lines.append(f"- **{fname}**: {n} examples ✓")
        else:
            lines.append(f"- **{fname}**: NOT YET BUILT")
    lines.append("")

    # Write file
    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD}")

    # Also print to stdout
    print("\n" + "="*60)
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
