"""
Phase 6: Generate VIDEO_VALUE_SUMMARY.md and fig_08_cotracker_visualization.png.

Reads all previous results and writes a comprehensive markdown report.

Outputs:
  outputs/video_assessment/VIDEO_VALUE_SUMMARY.md
  outputs/video_assessment/fig_08_cotracker_visualization.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs/video_assessment"

DATA_ROOT = Path("/home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs/extracted")


def load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def fmt(v, fmt_str=".4f"):
    if v is None:
        return "N/A"
    try:
        return format(float(v), fmt_str)
    except Exception:
        return str(v)


def make_cotracker_visualization():
    """Overlay tracked point visualization on a sample frame."""
    feat_dir = OUT_DIR / "features"

    # Find any CoTracker result
    tier1_files = sorted(feat_dir.glob("tier1_*.parquet"))
    if not tier1_files:
        print("  No CoTracker results found for visualization.")
        return

    run_id = tier1_files[0].stem.replace("tier1_", "")
    df     = pd.read_parquet(tier1_files[0])
    valid  = df[df["timestamp_ms"] > 0].head(1)
    if valid.empty:
        return

    # Load corresponding video frame
    run_dir  = DATA_ROOT / run_id
    mp4s     = sorted((run_dir / "camera-video").glob("output_*.mp4"))
    if not mp4s:
        return

    import cv2
    cap       = cv2.VideoCapture(str(mp4s[0]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(valid["frame_idx"].iloc[0]))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: raw frame
    axes[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Sample frame: {run_id[-8:]}")
    axes[0].axis("off")

    # Right: tracked motion heatmap (ct_mean_x, ct_mean_y over time)
    n_show = min(500, len(df[df["timestamp_ms"] > 0]))
    df_show = df[df["timestamp_ms"] > 0].head(n_show)
    if "ct_mean_x" in df_show.columns and "ct_mean_y" in df_show.columns:
        axes[1].scatter(df_show["ct_mean_x"], 1 - df_show["ct_mean_y"],
                        c=range(len(df_show)), cmap="viridis", s=5, alpha=0.6)
        axes[1].set_xlabel("Normalized X (cart position)")
        axes[1].set_ylabel("Normalized Y (height)")
        axes[1].set_title("Tracked centroid trajectory (first 500 frames)")
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_08_cotracker_visualization.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_08_cotracker_visualization.png")


def main():
    print("=" * 60)
    print("Phase 6: Summary Report")
    print("=" * 60)

    # Load all results
    params    = load_json(OUT_DIR / "dataset_params.json")
    manifest  = pd.read_csv(OUT_DIR / "session_manifest.csv") if (OUT_DIR / "session_manifest.csv").exists() else pd.DataFrame()
    p3_res    = load_json(OUT_DIR / "p3_prediction_results.json")
    p4_res    = load_json(OUT_DIR / "p4_anomaly_results.json")
    p5_res    = load_json(OUT_DIR / "p5_loso_results.json")
    align_rep = load_json(OUT_DIR / "alignment_report.json")

    # Generate CoTracker visualization
    print("\nGenerating CoTracker visualization...")
    make_cotracker_visualization()

    # Build markdown
    lines = []
    lines.append("# Video Value Assessment — CPS-Debugger")
    lines.append("")
    lines.append(f"**Generated:** 2026-03-24")
    lines.append("")

    # ── Section 1: Data parameters ─────────────────────────────────────────
    lines.append("## 1. Detected Data Parameters")
    lines.append("")
    if params:
        lines.append(f"| Parameter | Value |")
        lines.append(f"|-----------|-------|")
        lines.append(f"| Total sessions | {params.get('n_sessions', 'N/A')} |")
        lines.append(f"| Sessions with video (all three modalities) | {len(params.get('full_video_sessions', []))} |")
        lines.append(f"| Physical sensor features | {params.get('n_physical_cols', 'N/A')}: {', '.join(params.get('physical_cols', []))} |")
        lines.append(f"| Branch targets | {params.get('n_branch_cols', 'N/A')} |")
        lines.append(f"| Datalayer rate | {params.get('datalayer_rate_hz', 'N/A'):.1f} Hz ({params.get('win_ms', 10)}ms windows) |")

        if not manifest.empty:
            vid_manifest = manifest[manifest["has_video"] == True]
            if not vid_manifest.empty:
                fps_vals = vid_manifest["fps"].dropna()
                lines.append(f"| Video FPS | {fps_vals.mean():.1f} Hz (median frame interval {1000/fps_vals.mean():.1f}ms) |")
                res_w = int(vid_manifest["width"].dropna().iloc[0]) if "width" in vid_manifest.columns else "?"
                res_h = int(vid_manifest["height"].dropna().iloc[0]) if "height" in vid_manifest.columns else "?"
                lines.append(f"| Video resolution | {res_w}×{res_h} |")
                lines.append(f"| Avg overlap with datalayer | {vid_manifest['overlap_s'].dropna().mean():.0f}s |")
    lines.append("")

    # ── Section 2: Prediction ladder ───────────────────────────────────────
    lines.append("## 2. Prediction Ladder (A–L)")
    lines.append("")
    lines.append("RF with n=200 trees, 5-fold contiguous temporal block CV, within BALANCE state.")
    lines.append("")

    CONFIG_NAMES = {
        "A": "physical_only",
        "B": "tier0_flow",
        "C": "tier1_cotracker",
        "D": "tier2_dinov2_pca",
        "E": "tier3_vjepa2_pca",
        "F": "physical+tier0",
        "G": "physical+tier1",
        "H": "physical+tier2",
        "I": "physical+tier3",
        "J": "physical+tier1+tier3",
        "K": "all_video_no_physical",
        "L": "everything",
    }

    if p3_res and "configs" in p3_res:
        lines.append("| Config | Description | n_features | R² mean | R² std |")
        lines.append("|--------|-------------|------------|---------|--------|")
        baseline_r2 = None
        for cfg, name in CONFIG_NAMES.items():
            v = p3_res["configs"].get(cfg, {})
            r2m = v.get("r2_mean")
            r2s = v.get("r2_std")
            nf  = v.get("n_features", 0)
            note = " ← BASELINE" if cfg == "A" else ""
            if cfg == "A" and r2m is not None:
                baseline_r2 = r2m
            lines.append(f"| {cfg} | {name}{note} | {nf} | {fmt(r2m)} | {fmt(r2s)} |")
        lines.append("")
    else:
        lines.append("*Prediction results not yet available. Run p3_prediction.py.*")
        lines.append("")

    # ── Section 3: Residual analysis ───────────────────────────────────────
    lines.append("## 3. Residual Analysis — Unique Video Contribution")
    lines.append("")
    lines.append("Residual R² = how much of the physical-RF residuals each video tier explains.")
    lines.append("")

    if p3_res and "residual_analysis" in p3_res:
        res_a = p3_res["residual_analysis"]
        lines.append("| Tier | Residual R² mean | Residual R² std |")
        lines.append("|------|-----------------|-----------------|")
        for tier_key, v in res_a.items():
            lines.append(f"| {tier_key} | {fmt(v.get('residual_r2_mean'))} | {fmt(v.get('residual_r2_std'))} |")
        lines.append("")
    else:
        lines.append("*Not yet available.*")
        lines.append("")

    # ── Section 4: Cross-monitoring matrix ─────────────────────────────────
    lines.append("## 4. Cross-Monitoring Matrix")
    lines.append("")
    lines.append("AUC-ROC for 7 anomaly types × 5 monitors.")
    lines.append("")

    monitor_names = ["datalayer_only", "video_only", "combined", "video_to_dl", "dl_to_video"]
    anom_names    = ["A1_sensor_bias", "A2_sensor_freeze", "A3_sensor_drift",
                     "A4_added_mass", "A5_actuator_degrade", "A6_mode_confusion",
                     "A7_timing_delay"]

    if p4_res and "auc_results" in p4_res:
        header = "| Anomaly | " + " | ".join(monitor_names) + " |"
        sep    = "|---------|" + "|".join(["------" for _ in monitor_names]) + "|"
        lines.append(header)
        lines.append(sep)
        for anom in anom_names:
            row_vals = p4_res["auc_results"].get(anom, {})
            cells    = [fmt(row_vals.get(m), ".3f") for m in monitor_names]
            lines.append(f"| {anom} | " + " | ".join(cells) + " |")
        lines.append("")
    else:
        lines.append("*Not yet available. Run p4_anomaly.py.*")
        lines.append("")

    # ── Section 5: Detection latency ───────────────────────────────────────
    lines.append("## 5. Detection Latency")
    lines.append("")
    lines.append("Windows to detection (3 consecutive windows above p99 threshold) after injection.")
    lines.append("Each window = 10ms.")
    lines.append("")

    if p4_res and "latency_results" in p4_res:
        lat_res = p4_res["latency_results"]
        lines.append("| Anomaly | " + " | ".join(monitor_names) + " |")
        lines.append("|---------|" + "|".join(["------" for _ in monitor_names]) + "|")
        for anom in ["A1_sensor_bias", "A3_sensor_drift"]:
            if anom in lat_res:
                latencies = lat_res[anom]
                cells = [str(latencies.get(m, "N/A")) for m in monitor_names]
                lines.append(f"| {anom} | " + " | ".join(cells) + " |")
        lines.append("")
    else:
        lines.append("*Not yet available.*")
        lines.append("")

    # ── Section 6: LOSO Generalization ─────────────────────────────────────
    lines.append("## 6. LOSO Generalization")
    lines.append("")

    if p5_res:
        summary = p5_res.get("_summary", {})
        if summary:
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Mean physical R² (LOSO) | {fmt(summary.get('mean_physical'))} |")
            lines.append(f"| Mean combined R² (LOSO) | {fmt(summary.get('mean_combined'))} |")
            lines.append(f"| Δ R² | {fmt(summary.get('delta'))} |")
            lines.append("")

        lines.append("| Session | Physical R² | Combined R² | Δ R² |")
        lines.append("|---------|------------|------------|-------|")
        for run_id, v in p5_res.items():
            if run_id == "_summary":
                continue
            lines.append(
                f"| {run_id[-14:]} | {fmt(v.get('r2_physical'))} | "
                f"{fmt(v.get('r2_combined'))} | {fmt(v.get('delta'))} |"
            )
        lines.append("")
    else:
        lines.append("*Not yet available. Run p5_loso.py.*")
        lines.append("")

    # ── Section 7: Conclusion ───────────────────────────────────────────────
    lines.append("## 7. Conclusion")
    lines.append("")

    framing = p5_res.get("_summary", {}).get("framing", None) if p5_res else None
    if framing:
        lines.append(f"**Paper framing:** {framing}")
        lines.append("")

    # Determine which conclusion applies
    loso_delta = p5_res.get("_summary", {}).get("delta") if p5_res else None
    best_res_r2 = None
    if p3_res and "residual_analysis" in p3_res:
        res_vals = [v.get("residual_r2_mean", 0) for v in p3_res["residual_analysis"].values()
                    if v.get("residual_r2_mean") is not None]
        if res_vals:
            best_res_r2 = max(res_vals)

    # Best anomaly detection AUC for cross-modal monitors
    best_cross_modal_auc = None
    if p4_res and "auc_results" in p4_res:
        sensor_fault_aucs = []
        for anom in ["A1_sensor_bias", "A2_sensor_freeze", "A3_sensor_drift"]:
            if anom in p4_res["auc_results"]:
                v = p4_res["auc_results"][anom].get("video_to_dl")
                if v is not None:
                    sensor_fault_aucs.append(v)
        if sensor_fault_aucs:
            best_cross_modal_auc = max(sensor_fault_aucs)

    if loso_delta is not None and loso_delta > 0.03:
        conclusion = "(A) Video adds prediction signal"
        detail = (f"Combined LOSO R² exceeds physical-only by Δ={loso_delta:.4f} "
                  f"(> 0.03 threshold), generalizing across sessions.")
    elif best_cross_modal_auc is not None and best_cross_modal_auc > 0.8:
        conclusion = "(B) Video adds cross-monitoring value"
        detail = (f"Video→datalayer cross-prediction detects sensor faults with "
                  f"AUC={best_cross_modal_auc:.3f}, providing analytical redundancy "
                  f"independent of the datalayer (Costanzino et al., CVPR 2024).")
    else:
        conclusion = "(C) Video is redundant"
        detail = ("Video features do not significantly improve prediction R² or "
                  "anomaly detection beyond what the physical datalayer captures.")

    lines.append(f"**Conclusion: {conclusion}**")
    lines.append("")
    lines.append(detail)
    lines.append("")
    lines.append("### Supporting Evidence")
    lines.append("")

    if p3_res and "configs" in p3_res:
        a_r2 = p3_res["configs"].get("A", {}).get("r2_mean")
        best_vid_r2 = max(
            (p3_res["configs"].get(c, {}).get("r2_mean") or -np.inf
             for c in ["B","C","D","E"]),
            default=None
        )
        best_comb_r2 = max(
            (p3_res["configs"].get(c, {}).get("r2_mean") or -np.inf
             for c in ["F","G","H","I","J"]),
            default=None
        )
        if a_r2 is not None:
            lines.append(f"- Physical-only R² (config A): **{a_r2:.4f}**")
        if best_vid_r2 is not None and best_vid_r2 > -np.inf:
            lines.append(f"- Best video-only R² (configs B–E): **{best_vid_r2:.4f}**")
        if best_comb_r2 is not None and best_comb_r2 > -np.inf:
            lines.append(f"- Best combined R² (configs F–J): **{best_comb_r2:.4f}**")

    if best_res_r2 is not None:
        lines.append(f"- Best residual R² (unique video contribution): **{best_res_r2:.4f}**")
    if best_cross_modal_auc is not None:
        lines.append(f"- Best sensor-fault AUC (video→datalayer monitor): **{best_cross_modal_auc:.4f}**")
    if loso_delta is not None:
        lines.append(f"- LOSO Δ R² (combined vs physical): **{loso_delta:+.4f}**")

    lines.append("")
    lines.append("---")
    lines.append("*Generated by p6_report.py — CPS-Debugger Video Value Assessment Sprint*")

    # Write report
    report_text = "\n".join(lines)
    report_path = OUT_DIR / "VIDEO_VALUE_SUMMARY.md"
    report_path.write_text(report_text)
    print(f"\nSaved {report_path.name}")
    print(f"\n{'='*60}")
    print("Phase 6 complete. All outputs in outputs/video_assessment/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
