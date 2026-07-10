"""
build_assessment_document.py — Generate comprehensive experiment assessment document.

Reads all JSON results, figures, and markdown files from the CPS-Debugger project
and produces:
  outputs/FULL_ASSESSMENT_DOCUMENT.md   — full markdown with embedded figure paths
  outputs/FULL_ASSESSMENT_DOCUMENT.pdf  — self-contained PDF with all figures inline
  outputs/assessment_metrics.json       — machine-readable key metrics

Usage: python scripts/build_assessment_document.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path("/home/simran/allspark-data-exploration/CPS-Debugger")
EXP  = ROOT / "outputs/experiments"
PAR  = ROOT / "outputs/parallel"
PH3  = ROOT / "outputs/phase3"
OUT  = ROOT / "outputs"

# ─── Load all JSON data ────────────────────────────────────────────────────────

def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception as e:
        print(f"  WARN: could not load {path}: {e}")
        return {}

branch   = load_json(EXP / "branch_regression_results.json")
anomaly  = load_json(EXP / "anomaly_detection_results.json")
cross    = load_json(EXP / "cross_run_results.json")
setpt    = load_json(EXP / "setpoint_results.json")
vocab    = load_json(EXP / "vocab_info.json")

baseline  = load_json(PAR / "baseline_results.json")
latency   = load_json(PAR / "detection_latency.json")
fpr_res   = load_json(PAR / "false_positive_results.json")
infer     = load_json(PAR / "inference_cost.json")
limits    = load_json(PAR / "limitations.json")
temporal  = load_json(PAR / "temporal_stability.json")

zeroshot  = load_json(PH3 / "anomaly_zero_shot_results.json")
reward    = load_json(PH3 / "reward_validation.json")
inv       = load_json(PH3 / "data_inventory_summary.json")
expand    = load_json(PH3 / "expand_preprocess_summary.json")

# ─── Figure inventory ─────────────────────────────────────────────────────────

EXP_FIGS = sorted(EXP.glob("fig_*.png"))
PAR_FIGS = sorted(PAR.glob("fig_*.png"))
PH3_FIGS = sorted(PH3.glob("fig_*.png"))
ALL_FIGS = EXP_FIGS + PAR_FIGS + PH3_FIGS

# ─── Extract key metrics ──────────────────────────────────────────────────────

metrics = {
    # Dataset
    "dataset": {
        "n_complete_runs":    inv.get("n_complete_runs", 12),
        "n_windows_expanded": expand.get("n_windows", 431297),
        "n_windows_6run":     branch.get("n_windows", 137904),
        "window_ms":          10,
        "n_branches_vocab":   vocab.get("n_vocab", 100),
        "state_dist":         expand.get("state_distribution", {
            "BALANCE": 407931, "SWINGUP": 15340, "RESET": 8026
        }),
        "sessions_dates":     "March 2025 (12 sessions)",
    },
    # Cross-modal prediction (exp01)
    "cross_modal_prediction": {
        "mean_r2_all_features":  branch.get("all_features", {}).get("mean_r2", 0.736),
        "mean_r2_physical_only": branch.get("physical_only", {}).get("mean_r2", 0.748),
        "mean_r2_syslog_only":   branch.get("syslog_only", {}).get("mean_r2", -0.010),
        "frac_r2_above_03":      branch.get("all_features", {}).get("frac_r2_above_0.3", 0.83),
        "mean_cosine_sim":       branch.get("all_features", {}).get("mean_cosine_sim", 0.9999),
        "n_physical_unique":     branch.get("information_decomposition", {}).get("n_physical_unique", 75),
        "cv_method":             "5-fold temporal block cross-validation",
    },
    # Within-state prediction
    "within_state_prediction": {
        "within_balance_r2_6run":  0.5604,
        "setpoint_accuracy_physical": setpt.get("results_by_feature_set", {}).get("Physical only", {}).get("accuracy", 0.973),
    },
    # Cross-run generalization
    "cross_run_generalization": {
        "mean_micro_f1":          cross.get("mean_micro_f1", 0.925),
        "std_micro_f1":           cross.get("std_micro_f1", 0.008),
        "mean_r2_physical_loro":  cross.get("mean_r2_physical", 0.653),
        "temporal_r2":            temporal.get("r2_temporal_early_late", 0.651),
        "loro_ridge_r2":          temporal.get("r2_loro_mean", -0.189),
        "drift_per_run":          temporal.get("drift_per_run", 1.128),
        "stability_verdict":      temporal.get("interpretation", {}).get("verdict", "unstable"),
    },
    # Anomaly detection (exp03)
    "anomaly_detection": {
        "threshold_p95":     anomaly.get("normal_consistency", {}).get("p95", 17.61),
        "threshold_p99":     anomaly.get("normal_consistency", {}).get("p99", 22.62),
        "auc_trace_swap":    anomaly.get("anomaly_results", {}).get("Trace Swap (wrong code path)", {}).get("auc", 0.9999),
        "auc_sensor_noise":  anomaly.get("anomaly_results", {}).get("Physical Noise (3σ sensor noise)", {}).get("auc", 0.9989),
        "auc_timing_10ms":   anomaly.get("anomaly_results", {}).get("Temporal Shift 10ms (timing error)", {}).get("auc", 0.7007),
        "auc_timing_50ms":   anomaly.get("anomaly_results", {}).get("Temporal Shift 50ms (large timing error)", {}).get("auc", 0.9547),
    },
    # Baselines
    "baselines": {
        "state_mean_auc":    baseline.get("methods", {}).get("state_mean", {}).get("auc", 0.627),
        "ridge_auc":         baseline.get("methods", {}).get("ridge_regression", {}).get("auc", 0.565),
        "lag1_ar_r2":        baseline.get("methods", {}).get("lag1_autoregression", {}).get("r2", 0.652),
        "lag1_ar_auc":       baseline.get("methods", {}).get("lag1_autoregression", {}).get("auc", 0.988),
        "rf_auc":            baseline.get("methods", {}).get("random_forest", {}).get("auc", 0.9999),
        "rf_r2_global":      baseline.get("methods", {}).get("random_forest", {}).get("r2", -12.878),
        "note":              "RF R² negative in global 80/20 split due to distribution shift (temporal autocorrelation). Correct eval: temporal block CV → R²=0.736.",
    },
    # False positive rate
    "false_positive_rate": {
        "fpr_p95_pct":         fpr_res.get("fpr_pct", {}).get("p95_immediate", 5.15),
        "fpr_p99_pct":         fpr_res.get("fpr_pct", {}).get("p99_immediate", 1.21),
        "fpr_p95_3win_pct":    fpr_res.get("fpr_pct", {}).get("p95_3win", 1.05),
        "fpr_p99_3win_pct":    fpr_res.get("fpr_pct", {}).get("p99_3win", 0.12),
        "pct_near_transitions": fpr_res.get("fp_near_transitions", {}).get("pct_near_p95", 0.0),
    },
    # Detection latency
    "detection_latency": {
        "trace_swap_p95_same_window_pct":  latency.get("per_type", {}).get("Trace Swap", {}).get("p95_same_window_pct", 100.0),
        "trace_swap_latency_ms":           latency.get("per_type", {}).get("Trace Swap", {}).get("p95_latency_ms", 0.0),
        "sensor_corrupt_p99_same_pct":     latency.get("per_type", {}).get("Sensor Corruption (3σ)", {}).get("p99_same_window_pct", 89.2),
        "timing_50ms_p99_same_pct":        latency.get("per_type", {}).get("Timing Shift (50ms)", {}).get("p99_same_window_pct", 63.2),
    },
    # Inference cost
    "inference_cost": {
        "full_pipeline_ms":    infer.get("full_pipeline_ms", 3.40),
        "rf_single_window_ms": infer.get("rf_single_window_ms", 3.31),
        "feasible":            infer.get("feasibility", "real_time"),
        "margin_x":            round(10.0 / infer.get("full_pipeline_ms", 3.40), 1),
    },
    # Limitations
    "limitations": {
        "slow_drift_detectable_at_cm":    limits.get("summary", {}).get("slow_drift_detected_at_cm", 1.0),
        "near_setpoint_swap_auc":         limits.get("summary", {}).get("near_setpoint_swap_auc", 0.953),
        "logic_bug_no_physical_sig_auc":  limits.get("summary", {}).get("logic_bug_auc", 0.530),
    },
    # LLM zero-shot (Phase 3)
    "llm_zero_shot": {
        "model":           zeroshot.get("model", "claude-haiku-4-5-20251001"),
        "n_examples":      zeroshot.get("n_examples", 250),
        "detection_acc":   zeroshot.get("overall", {}).get("detection_acc", 0.716),
        "typing_acc":      zeroshot.get("overall", {}).get("typing_acc", 0.292),
        "localization_acc":zeroshot.get("overall", {}).get("localization_acc", 0.532),
        "reasoning_score": zeroshot.get("overall", {}).get("reasoning_score", 0.940),
        "total_score":     zeroshot.get("overall", {}).get("total_score", 0.620),
        "per_type":        zeroshot.get("per_type", {}),
    },
    # GRPO training setup
    "grpo_training": {
        "job_id":          1873620,
        "status":          "PENDING — Nodes DOWN/reserved",
        "scheduled":       "2026-03-17",
        "base_model":      "Qwen2.5-Coder-7B-Instruct",
        "algorithm":       "GRPO (group_size=8, kl_coef=0.001)",
        "infrastructure":  "4× A100, LoRA rank=16, VERL framework",
        "n_train":         3000,
        "n_eval":          250,
        "reward_mean":     reward.get("mean_reward", 0.614),
        "reward_std":      reward.get("std_reward", 0.223),
        "reward_assessment": reward.get("assessment", "SUITABLE"),
    },
    # Reward function
    "reward_function": {
        "components":       ["detection (0.25)", "typing (0.25)", "localization (0.25)", "reasoning (0.25)"],
        "n_scored":         reward.get("n_examples_scored", 194),
        "mean_reward":      reward.get("mean_reward", 0.614),
        "std_reward":       reward.get("std_reward", 0.223),
        "frac_gt_0.5":      reward.get("frac_gt_05", 0.376),
    },
    "generated_at": datetime.now().isoformat(),
}

# ─── Build Markdown document ──────────────────────────────────────────────────

m = metrics
zs = m["llm_zero_shot"]
ad = m["anomaly_detection"]
cm = m["cross_modal_prediction"]
ds = m["dataset"]
bl = m["baselines"]
fp = m["false_positive_rate"]
ic = m["inference_cost"]
cr = m["cross_run_generalization"]

md = f"""# CPS Debugger — Full Experiment Assessment Document
*Generated: {m['generated_at']}*
*GRPO Job {m['grpo_training']['job_id']}: {m['grpo_training']['status']}*

---

## Executive Summary

This document presents the complete experimental results for the CPS Debugger project: a
cross-modal anomaly detection and physics-to-code abductive reasoning system for the
Bosch Rexroth ctrlX industrial inverted-pendulum controller.

**Three core contributions:**
1. **Cross-modal predictive relationship**: Physical sensor observations predict WebAssembly
   execution branch counts (R²={cm['mean_r2_physical_only']:.3f} physical-only, mean across 100 branches).
2. **Physics-to-code abductive reasoning task**: LLM-based fault explanation with verifiable
   rewards from execution traces; RLVR training via GRPO.
3. **First multimodal CPS debugging benchmark**: {ds['n_windows_expanded']:,} windows across
   {ds['n_complete_runs']} sessions, synchronized 100Hz Wasm traces + physical sensors + C++ source.

**Key numbers:**

| Metric | Value |
|--------|-------|
| Dataset windows (expanded) | {ds['n_windows_expanded']:,} |
| Dataset sessions | {ds['n_complete_runs']} |
| Mean cross-modal R² (physical only) | {cm['mean_r2_physical_only']:.3f} |
| Mean cross-modal R² (all features) | {cm['mean_r2_all_features']:.3f} |
| Branches with R²≥0.3 | {cm['frac_r2_above_03']*100:.0f}% |
| Within-BALANCE R² | 0.560 |
| Trace-swap detection AUC | {ad['auc_trace_swap']:.4f} |
| Sensor noise detection AUC | {ad['auc_sensor_noise']:.4f} |
| Timing shift 50ms AUC | {ad['auc_timing_50ms']:.4f} |
| Timing shift 10ms AUC | {ad['auc_timing_10ms']:.4f} |
| Zero-shot total score (Haiku) | {zs['total_score']:.2f} |
| Zero-shot detection accuracy | {zs['detection_acc']*100:.1f}% |
| Zero-shot type accuracy | {zs['typing_acc']*100:.1f}% |
| Full pipeline latency | {ic['full_pipeline_ms']:.2f}ms (<10ms window) |
| GRPO results | *PENDING — job {m['grpo_training']['job_id']}* |

---

## 1. Dataset

### 1.1 Experimental Platform
- **Controller**: Bosch Rexroth ctrlX CORE real-time controller
- **Plant**: Inverted pendulum (~1m aluminum rod, servo-driven cart, 2.4m rail)
- **Software**: C++ control algorithm compiled to WebAssembly (Wasm)
- **Control frequency**: ~1kHz; analysis windows: 10ms (100 per second)

### 1.2 Data Statistics

| Property | Value |
|----------|-------|
| Complete experimental runs | {ds['n_complete_runs']} |
| Collection dates | {ds['sessions_dates']} |
| Windows (expanded 12-run dataset) | {ds['n_windows_expanded']:,} |
| Windows (original 6-run dataset) | {ds['n_windows_6run']:,} |
| Wasm branch vocabulary size | {ds['n_branches_vocab']} |
| Analysis window duration | {ds['window_ms']}ms |

**Controller state distribution (expanded dataset):**

| State | Windows | Fraction |
|-------|---------|----------|
| BALANCE | {ds['state_dist'].get('BALANCE', 407931):,} | {ds['state_dist'].get('BALANCE', 407931)/ds['n_windows_expanded']*100:.1f}% |
| SWINGUP | {ds['state_dist'].get('SWINGUP', 15340):,} | {ds['state_dist'].get('SWINGUP', 15340)/ds['n_windows_expanded']*100:.1f}% |
| RESET | {ds['state_dist'].get('RESET', 8026):,} | {ds['state_dist'].get('RESET', 8026)/ds['n_windows_expanded']*100:.1f}% |

---

## 2. Cross-Modal Trace Prediction (Exp01)

Physical sensor observations (9 features: position, angle, velocity, angular velocity)
are used to predict WebAssembly branch execution counts (top-100 branches by coefficient
of variation) using Random Forest regression with 5-fold temporal block cross-validation.

### 2.1 Prediction Performance

| Feature Set | Mean R² | Branches R²≥0.3 |
|-------------|---------|-----------------|
| Physical only | **{cm['mean_r2_physical_only']:.3f}** | {cm['frac_r2_above_03']*100:.0f}% |
| All features | {cm['mean_r2_all_features']:.3f} | {cm['frac_r2_above_03']*100:.0f}% |
| Syslog only | {cm['mean_r2_syslog_only']:.3f} | 7% |

**Mean cosine similarity**: {cm['mean_cosine_sim']:.4f}

**Key finding**: Physical features alone explain {cm['frac_r2_above_03']*100:.0f}% of branches (R²≥0.3).
Syslog adds almost nothing (R²≈0). This confirms the cross-modal relationship is driven
by physical dynamics, not software logging.

**Information decomposition**:
- Branches explained by physical only: {cm['n_physical_unique']}
- Branches explained by both: 7
- Branches explained by neither: 16

### 2.2 Within-State Prediction

Within BALANCE state (controlling for FSM state), physical features achieve:
- **6-run dataset**: R²=0.560
- **Expanded 12-run dataset**: pending (task_1_3c)

**Setpoint classification** (within BALANCE, 4 setpoints: 0m, ±0.05m, ±0.09m):
- Physical features only: accuracy={setpt.get('results_by_feature_set', {}).get('Physical only', {}).get('accuracy', 0.973)*100:.1f}%
- Branch traces: accuracy={setpt.get('results_by_feature_set', {}).get('Trace (branch counts)', {}).get('accuracy', 0.370)*100:.1f}%
- Physical clearly best: confirms sensors carry more setpoint info than execution traces

### 2.3 Figures

![R² distribution by branch (all features)](experiments/fig_01a_branch_r2_all_features.png)
![R² distribution by branch (physical only)](experiments/fig_01a_branch_r2_physical_only.png)
![Branch prediction scatter plots](experiments/fig_01b_branch_prediction_scatter.png)
![Feature importance](experiments/fig_01c_feature_importance.png)
![R² by FSM state](experiments/fig_01d_r2_by_state.png)
![Physical vs syslog R² comparison](experiments/fig_01e_physical_vs_syslog_r2.png)

---

## 3. Cross-Run Generalization (Exp06)

Leave-one-run-out (LORO) evaluation: train on 5 runs, test on held-out 6th run.

### 3.1 LORO Results

| Metric | Mean ± Std |
|--------|-----------|
| State classification micro-F1 | {cr['mean_micro_f1']:.3f} ± {cr['std_micro_f1']:.3f} |
| Regression R² (physical, LORO) | {cr['mean_r2_physical_loro']:.3f} |

**Per-run R² (physical only)**:
"""

for run_data in cross.get("per_run", []):
    run = run_data.get("test_run", "")
    r2 = run_data.get("regression_physical_only", {}).get("mean_r2", 0)
    f1 = run_data.get("classification", {}).get("micro_f1", 0)
    md += f"- {run}: R²={r2:.3f}, micro-F1={f1:.3f}\n"

md += f"""
### 3.2 Temporal Stability (B4)

Train on first {len(temporal.get('train_runs_early', []))} runs chronologically, test on last {len(temporal.get('test_runs_late', []))}.

| Metric | Value |
|--------|-------|
| Temporal R² (early→late) | {cr['temporal_r2']:.3f} |
| LORO Ridge R² | {cr['loro_ridge_r2']:.3f} |
| Consistency score drift | +{cr['drift_per_run']:.3f}/run |
| Stability verdict | **{cr['stability_verdict'].upper()}** |

**Interpretation**: The temporal R²=0.651 shows the RF trained on early runs predicts
late-run branches reasonably well. However, LORO Ridge R²=-0.189 indicates ridge regression
does not generalize across runs in leave-one-out setting. The +{cr['drift_per_run']:.2f}/run
upward drift in consistency score suggests gradual concept drift (the physical-trace
relationship shifts over time, likely due to mechanical wear or recalibration).

![Temporal stability analysis](parallel/fig_temporal_stability.png)

### 3.3 Figures

![Cross-run generalization](experiments/fig_06_cross_run_generalization.png)

---

## 4. Anomaly Detection (Exp03)

Consistency-based anomaly detection: flag windows where the L2 norm of the RF prediction
residual exceeds empirical thresholds (p95 or p99 of normal training data).

### 4.1 Thresholds

| Threshold | Value |
|-----------|-------|
| p95 | {ad['threshold_p95']:.2f} |
| p99 | {ad['threshold_p99']:.2f} |

### 4.2 Detection Performance (ROC AUC)

| Anomaly Type | AUC |
|-------------|-----|
| Trace Swap (wrong code path) | **{ad['auc_trace_swap']:.4f}** |
| Physical Noise (3σ sensor noise) | **{ad['auc_sensor_noise']:.4f}** |
| Temporal Shift 50ms (large timing error) | {ad['auc_timing_50ms']:.4f} |
| Temporal Shift 10ms (small timing error) | {ad['auc_timing_10ms']:.4f} |

**Key finding**: Structural faults (wrong code path, sensor corruption) are detected
near-perfectly. Timing misalignment is harder: 10ms shift (one analysis window) achieves
only AUC=0.701; 50ms shift (5 windows) achieves AUC=0.955.

### 4.3 Detection Latency (B2)

| Anomaly Type | Same-Window Detection (p95) | Mean Latency |
|-------------|---------------------------|--------------|
| Trace Swap | {latency.get('per_type', {}).get('Trace Swap', {}).get('p95_same_window_pct', 100.0):.0f}% | {latency.get('per_type', {}).get('Trace Swap', {}).get('p95_latency_ms', 0.0):.1f}ms |
| Sensor Corruption (3σ) | {latency.get('per_type', {}).get('Sensor Corruption (3σ)', {}).get('p95_same_window_pct', 95.8):.0f}% | {latency.get('per_type', {}).get('Sensor Corruption (3σ)', {}).get('p95_latency_ms', 0.4):.1f}ms |
| Timing Shift 50ms | {latency.get('per_type', {}).get('Timing Shift (50ms)', {}).get('p95_same_window_pct', 80.4):.0f}% | {latency.get('per_type', {}).get('Timing Shift (50ms)', {}).get('p95_latency_ms', 2.0):.1f}ms |

### 4.4 False Positive Rate (B3)

| Configuration | FPR |
|--------------|-----|
| p95 immediate | {fp['fpr_p95_pct']:.2f}% (as designed) |
| p99 immediate | {fp['fpr_p99_pct']:.2f}% |
| p95 + 3-window confirmation | {fp['fpr_p95_3win_pct']:.2f}% |
| p99 + 3-window confirmation | **{fp['fpr_p99_3win_pct']:.2f}%** |

FPs near state transitions (within 50ms): **{fp['pct_near_transitions']:.0f}%**
(Surprisingly, FPs do NOT cluster at transitions — they are distributed uniformly.)

### 4.5 Figures

![Consistency score timeline](experiments/fig_03a_consistency_timeline.png)
![Consistency score distribution](experiments/fig_03b_consistency_distribution.png)
![Anomaly detection ROC curves](experiments/fig_03c_anomaly_roc_curves.png)
![Anomaly score histograms](experiments/fig_03d_anomaly_score_histograms.png)
![Operator dashboard demo](parallel/fig_operator_dashboard.png)
![Detection example (±500ms zoom)](parallel/fig_detection_example.png)
![Detection latency analysis](parallel/fig_detection_latency.png)
![False positive rate analysis](parallel/fig_false_positive_analysis.png)

---

## 5. Baseline Comparison (B1)

Evaluated on last 20% of data (global split, same for all methods) to allow fair comparison.
Note: RF R² is negative due to distribution shift in global split — temporal block CV is
the correct evaluation methodology for the RF predictor.

| Method | R² | Cosine Sim | AUC |
|--------|-----|-----------|-----|
| State mean (no ML) | {bl['state_mean_auc']:.4f} (AUC only) | 0.998 | {bl['state_mean_auc']:.4f} |
| Ridge regression | negative | 0.998 | {bl['ridge_auc']:.4f} |
| **Lag-1 autoregression** | **{bl['lag1_ar_r2']:.3f}** | 0.9998 | **{bl['lag1_ar_auc']:.4f}** |
| Physics rule-based | N/A | N/A | 0.500 |
| **Random Forest (ours)** | {bl['rf_r2_global']:.3f}* | 0.9999 | **{bl['rf_auc']:.4f}** |

*RF R²={bl['rf_r2_global']:.3f} in global 80/20 split; R²={cm['mean_r2_all_features']:.3f} in temporal block CV (correct protocol)

**Critical finding**: Lag-1 AR achieves AUC=0.988 vs RF AUC=0.9999. The temporal
autocorrelation of branch counts is strong (branch counts in window t predict window t+1
well). The RF adds value at the margin — predicting from physical semantics rather than
recent history — which matters most when a fault disrupts the physics-to-code relationship.

![Baseline comparison](parallel/fig_baseline_comparison.png)

---

## 6. Inference Cost (B2)

| Component | Latency |
|-----------|---------|
| RF single-window prediction | {ic['rf_single_window_ms']:.2f}ms |
| Full pipeline (RF + consistency score) | **{ic['full_pipeline_ms']:.2f}ms** |
| Analysis window | 10ms |
| Margin | {ic['margin_x']}× |

**Verdict**: Real-time feasible. The full pipeline runs in {ic['full_pipeline_ms']:.2f}ms,
well within the 10ms analysis window. The system can monitor every window on a single CPU core.

LLM anomaly explanation (Claude Haiku): ~5s per call, invoked only on anomaly events.
At p99 FPR <1%, this means ~few calls per hour in steady-state operation.

---

## 7. Limitations (A3)

Three categories of hard-to-detect faults:

| Scenario | Result | Detectable? |
|----------|--------|-------------|
| Slow sensor drift | Detectable after {m['limitations']['slow_drift_detectable_at_cm']:.1f}cm cumulative displacement | Marginally |
| Near-setpoint BALANCE swap | AUC={m['limitations']['near_setpoint_swap_auc']:.3f} | Yes (setpoint gap large enough) |
| Logic bug ±1 branch count | AUC={m['limitations']['logic_bug_no_physical_sig_auc']:.3f} | **No** |

**Logic bugs with no physical signature** are undetectable: perturbing 5 of 100 branch
counts by ±1 achieves AUC=0.530 (near random). The ±1 perturbation falls within natural
variance. The L2 residual is dominated by larger-variance high-count branches.

![Limitation analysis](parallel/fig_limitation_analysis.png)

---

## 8. Phase 3: LLM-Based Anomaly Explanation

### 8.1 Task Definition

**Physics-to-code abductive reasoning**: given a window with high consistency score,
physical sensor readings, relevant source code snippets, and branch vocabulary, produce:
```json
{{
  "is_anomalous": bool,
  "anomaly_type": "THRESHOLD_BUG | TRACE_SWAP | TIMING_DELAY | STUCK_SENSOR | NORMAL",
  "implicated_code": "source location string",
  "reasoning": "natural language explanation"
}}
```

**Reward**: R = 0.25·R_detect + 0.25·R_type + 0.25·R_localize + 0.25·R_reason

### 8.2 Dataset

| Split | Size |
|-------|------|
| Evaluation (zero-shot benchmark) | 250 examples (50 per type) |
| GRPO training | 3,000 examples |
| SFT training | 400 examples |
| SFT validation | 100 examples |

### 8.3 Zero-Shot Baseline (Claude Haiku)

| Metric | Score |
|--------|-------|
| Detection accuracy | {zs['detection_acc']*100:.1f}% |
| Type accuracy | {zs['typing_acc']*100:.1f}% |
| Localization score | {zs['localization_acc']*100:.1f}% |
| Reasoning quality | {zs['reasoning_score']*100:.1f}% |
| **Total score** | **{zs['total_score']:.2f}** |

**Per-type zero-shot scores:**

| Anomaly Type | Score |
|-------------|-------|
"""

for t, v in sorted(zs.get("per_type", {}).items(), key=lambda x: x[1].get("mean_score", 0), reverse=True):
    score = v.get("mean_score", 0) if isinstance(v, dict) else v
    md += f"| {t} | {score:.3f} |\n"

md += f"""
**Key finding**: STUCK_SENSOR and TIMING_DELAY are easiest (physical signal obvious);
THRESHOLD_BUG and TRACE_SWAP are hardest (require understanding code semantics).

### 8.4 Reward Function Validation

| Metric | Value |
|--------|-------|
| n examples scored | {m['reward_function']['n_scored']} |
| Mean reward | {m['reward_function']['mean_reward']:.3f} |
| Std reward | {m['reward_function']['std_reward']:.3f} |
| Fraction > 0.5 | {m['reward_function']['frac_gt_0.5']*100:.1f}% |
| Assessment | **{reward.get('assessment', 'SUITABLE')}** |

### 8.5 GRPO Training

| Parameter | Value |
|-----------|-------|
| Job ID | {m['grpo_training']['job_id']} |
| Status | {m['grpo_training']['status']} |
| Base model | {m['grpo_training']['base_model']} |
| Algorithm | {m['grpo_training']['algorithm']} |
| Infrastructure | {m['grpo_training']['infrastructure']} |
| Training examples | {m['grpo_training']['n_train']:,} |
| Evaluation examples | {m['grpo_training']['n_eval']} |

*GRPO results (SFT vs GRPO comparison) will be added once job {m['grpo_training']['job_id']} completes.*

---

## 9. Additional Analyses

### 9.1 Setpoint Verification

4-setpoint classification within BALANCE state (0m, +0.05m, −0.09m, +0.09m):
- Physical-only accuracy: {setpt.get('results_by_feature_set', {}).get('Physical only', {}).get('accuracy', 0.973)*100:.1f}%
- Branch trace accuracy: {setpt.get('results_by_feature_set', {}).get('Trace (branch counts)', {}).get('accuracy', 0.370)*100:.1f}%

Physical sensors carry far more setpoint information than execution traces — confirming
that sensors, not traces, encode the continuous control state.

![Setpoint accuracy comparison](experiments/fig_02a_setpoint_accuracy_comparison.png)
![Setpoint UMAP](experiments/fig_02e_setpoint_umap.png)

### 9.2 Audio/Microphone Modality

An additional exploratory analysis checked whether acoustic sensor data (microphone)
provides complementary cross-modal information:

![Audio branch R²](experiments/fig_05a_audio_branch_r2.png)
![Audio feature importance](experiments/fig_05b_audio_feature_importance.png)
![Audio trace timeline](experiments/fig_05c_audio_trace_timeline.png)

---

## 10. Open Issues and Validity Concerns

| Issue | Impact | Mitigation |
|-------|--------|-----------|
| Synthetic anomalies only | Generalization unknown | Acknowledge in limitations |
| Single physical system | Single-system | Method is general; system-specific model expected |
| RF R² negative in global split | Confusing for readers | Use temporal block CV; explain in paper |
| Lag-1 AR high AUC (0.988) | Baseline competitive | RF still +0.012 AUC; temporal autocorrelation explained |
| LORO Ridge R² negative | Instability signal | Physical R² (LORO) = 0.653; Ridge overfits |
| Consistency score drift (+1.13/run) | Threshold recalibration needed | Adaptive thresholding as future work |
| GRPO results pending | Paper incomplete | Job {m['grpo_training']['job_id']} scheduled {m['grpo_training']['scheduled']} |

---

## 11. Figure Index

All {len(ALL_FIGS)} figures generated by this project:

**Experiment Figures (outputs/experiments/):**
"""

for f in EXP_FIGS:
    size_kb = f.stat().st_size // 1024
    md += f"- [{f.name}](experiments/{f.name}) ({size_kb}KB)\n"

md += "\n**Parallel Analysis Figures (outputs/parallel/):**\n"
for f in PAR_FIGS:
    size_kb = f.stat().st_size // 1024
    md += f"- [{f.name}](parallel/{f.name}) ({size_kb}KB)\n"

if PH3_FIGS:
    md += "\n**Phase 3 Figures (outputs/phase3/):**\n"
    for f in PH3_FIGS:
        size_kb = f.stat().st_size // 1024
        md += f"- [{f.name}](phase3/{f.name}) ({size_kb}KB)\n"

md += f"""
---

## 12. File Index

**Raw Data:**
- `outputs/experiments/aligned_dataset.parquet` — 6-run aligned dataset (137,904 × 119 cols)
- `outputs/phase3/aligned_dataset_expanded.parquet` — 12-run expanded dataset (431,297 × 119 cols)

**Experiment Results:**
- `outputs/experiments/branch_regression_results.json` — R² by branch and feature set
- `outputs/experiments/anomaly_detection_results.json` — AUC by anomaly type
- `outputs/experiments/cross_run_results.json` — LORO generalization results
- `outputs/experiments/setpoint_results.json` — 4-setpoint classification results

**Parallel Analyses:**
- `outputs/parallel/baseline_results.json` — 5-method baseline comparison
- `outputs/parallel/detection_latency.json` — per-type detection latency
- `outputs/parallel/false_positive_results.json` — FPR at p95/p99
- `outputs/parallel/inference_cost.json` — pipeline latency breakdown
- `outputs/parallel/limitations.json` — hard-to-detect fault analysis
- `outputs/parallel/temporal_stability.json` — cross-time generalization

**Phase 3 (LLM):**
- `outputs/phase3/anomaly_eval_dataset.jsonl` — 250 evaluation examples
- `outputs/phase3/grpo_train.jsonl` — 3,000 GRPO training examples
- `outputs/phase3/anomaly_zero_shot_results.json` — Haiku zero-shot results
- `outputs/phase3/reward_validation.json` — reward function validation

**Documents:**
- `outputs/parallel/related_work_notes.md` — 6-section related work survey
- `outputs/parallel/paper_outline.md` — NeurIPS 9-page structure
- `outputs/parallel/contribution_statement.md` — 3 contributions + objections
- `outputs/PHASE3_SUMMARY.md` — phase 3 status summary
- `outputs/assessment_metrics.json` — machine-readable key metrics (this run)

---
*Document generated by `scripts/build_assessment_document.py`*
"""

# ─── Write Markdown ───────────────────────────────────────────────────────────
md_path = OUT / "FULL_ASSESSMENT_DOCUMENT.md"
md_path.write_text(md)
print(f"Wrote {md_path} ({md_path.stat().st_size // 1024}KB)")

# ─── Write metrics JSON ───────────────────────────────────────────────────────
json_path = OUT / "assessment_metrics.json"
json_path.write_text(json.dumps(metrics, indent=2))
print(f"Wrote {json_path} ({json_path.stat().st_size // 1024}KB)")

# ─── Generate PDF ─────────────────────────────────────────────────────────────
print("\nGenerating PDF...")

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor, black, white, grey
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image, PageBreak, HRFlowable, KeepTogether
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    from PIL import Image as PILImage
    HAS_REPORTLAB = True
except ImportError as e:
    print(f"  reportlab/PIL not available: {e}")
    HAS_REPORTLAB = False

if HAS_REPORTLAB:
    pdf_path = OUT / "FULL_ASSESSMENT_DOCUMENT.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch,
    )

    styles = getSampleStyleSheet()
    W = A4[0] - 1.5*inch  # usable width

    # Custom styles
    title_style = ParagraphStyle("title", parent=styles["Title"],
        fontSize=18, spaceAfter=6, textColor=HexColor("#1A237E"))
    h1_style = ParagraphStyle("h1", parent=styles["Heading1"],
        fontSize=14, spaceBefore=14, spaceAfter=4, textColor=HexColor("#1565C0"))
    h2_style = ParagraphStyle("h2", parent=styles["Heading2"],
        fontSize=11, spaceBefore=10, spaceAfter=3, textColor=HexColor("#1976D2"))
    h3_style = ParagraphStyle("h3", parent=styles["Heading3"],
        fontSize=10, spaceBefore=8, spaceAfter=2, textColor=HexColor("#424242"))
    body_style = ParagraphStyle("body", parent=styles["Normal"],
        fontSize=9, spaceAfter=4, leading=13)
    code_style = ParagraphStyle("code", parent=styles["Code"],
        fontSize=7.5, spaceAfter=4, leading=11, fontName="Courier",
        backColor=HexColor("#F5F5F5"))
    caption_style = ParagraphStyle("caption", parent=styles["Normal"],
        fontSize=8, textColor=HexColor("#555555"), alignment=TA_CENTER,
        spaceAfter=10, spaceBefore=2)
    warn_style = ParagraphStyle("warn", parent=styles["Normal"],
        fontSize=9, textColor=HexColor("#B71C1C"), spaceAfter=4)

    def make_table(headers, rows, col_widths=None, highlight_col=None):
        data = [headers] + rows
        if col_widths is None:
            col_widths = [W / len(headers)] * len(headers)
        ts = TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1565C0")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#F8F9FA"), HexColor("#FFFFFF")]),
            ("GRID",       (0, 0), (-1, -1), 0.3, HexColor("#CCCCCC")),
            ("PADDING",    (0, 0), (-1, -1), 4),
            ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ])
        t = Table(data, colWidths=col_widths)
        t.setStyle(ts)
        return t

    def embed_figure(path, max_width=None, max_height=None, caption=None):
        """Return [Image, caption_paragraph] or [] if file missing."""
        if not Path(path).exists():
            return [Paragraph(f"[Figure not found: {Path(path).name}]", warn_style)]
        if max_width is None:
            max_width = W
        if max_height is None:
            max_height = 4 * inch
        try:
            img = PILImage.open(path)
            iw, ih = img.size
            ratio = min(max_width / iw, max_height / ih)
            rw, rh = iw * ratio, ih * ratio
            el = Image(str(path), width=rw, height=rh)
            result = [el]
            if caption:
                result.append(Paragraph(caption, caption_style))
            return result
        except Exception as e:
            return [Paragraph(f"[Error loading {Path(path).name}: {e}]", warn_style)]

    story = []

    # ── Title page ─────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("CPS Debugger — Full Experiment Assessment", title_style))
    story.append(Paragraph(
        f"Cross-Modal Anomaly Detection and Physics-to-Code Abductive Reasoning",
        ParagraphStyle("subtitle", parent=styles["Normal"],
            fontSize=12, textColor=HexColor("#455A64"), spaceAfter=4)
    ))
    story.append(Paragraph(f"Generated: {m['generated_at'][:19]}", body_style))
    story.append(Paragraph(
        f"GRPO Job {m['grpo_training']['job_id']}: {m['grpo_training']['status']}",
        warn_style
    ))
    story.append(HRFlowable(width=W, thickness=1.5, color=HexColor("#1565C0")))
    story.append(Spacer(1, 0.1*inch))

    # Summary table
    story.append(Paragraph("Key Metrics at a Glance", h2_style))
    summary_rows = [
        ["Dataset windows (expanded)", f"{ds['n_windows_expanded']:,}"],
        ["Dataset sessions", str(ds['n_complete_runs'])],
        ["Mean R² (physical→branches)", f"{cm['mean_r2_physical_only']:.3f}"],
        ["Within-BALANCE R²", "0.560"],
        ["Trace-swap AUC", f"{ad['auc_trace_swap']:.4f}"],
        ["Sensor noise AUC", f"{ad['auc_sensor_noise']:.4f}"],
        ["Timing 50ms AUC", f"{ad['auc_timing_50ms']:.4f}"],
        ["Timing 10ms AUC", f"{ad['auc_timing_10ms']:.4f}"],
        ["FPR at p99 (3-win)", f"{fp['fpr_p99_3win_pct']:.2f}%"],
        ["Full pipeline latency", f"{ic['full_pipeline_ms']:.2f}ms"],
        ["Zero-shot total score (Haiku)", f"{zs['total_score']:.2f}"],
        ["Zero-shot detection accuracy", f"{zs['detection_acc']*100:.1f}%"],
        ["GRPO results", "PENDING"],
    ]
    story.append(make_table(
        ["Metric", "Value"],
        summary_rows,
        col_widths=[W*0.65, W*0.35]
    ))
    story.append(PageBreak())

    # ── Section 1: Dataset ──────────────────────────────────────────────────────
    story.append(Paragraph("1. Dataset", h1_style))
    story.append(Paragraph(
        f"Bosch Rexroth ctrlX CORE real-time controller running an inverted pendulum (C++→Wasm). "
        f"{ds['n_complete_runs']} complete experimental sessions collected in March 2025. "
        f"Analysis windows: {ds['window_ms']}ms each at 100Hz.",
        body_style
    ))
    story.append(make_table(
        ["Property", "Value"],
        [
            ["Complete runs", str(ds['n_complete_runs'])],
            ["Windows (expanded)", f"{ds['n_windows_expanded']:,}"],
            ["Windows (6-run)", f"{ds['n_windows_6run']:,}"],
            ["Branch vocabulary", str(ds['n_branches_vocab'])],
            ["Collection dates", ds['sessions_dates']],
            ["BALANCE windows", f"{ds['state_dist'].get('BALANCE',407931):,} ({ds['state_dist'].get('BALANCE',407931)/ds['n_windows_expanded']*100:.1f}%)"],
            ["SWINGUP windows", f"{ds['state_dist'].get('SWINGUP',15340):,} ({ds['state_dist'].get('SWINGUP',15340)/ds['n_windows_expanded']*100:.1f}%)"],
            ["RESET windows", f"{ds['state_dist'].get('RESET',8026):,} ({ds['state_dist'].get('RESET',8026)/ds['n_windows_expanded']*100:.1f}%)"],
        ],
        col_widths=[W*0.5, W*0.5]
    ))

    # ── Section 2: Cross-modal prediction ──────────────────────────────────────
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("2. Cross-Modal Trace Prediction", h1_style))
    story.append(Paragraph(
        "Random Forest regression: 9 physical features → 100 Wasm branch counts. "
        "5-fold temporal block cross-validation (prevents data leakage).",
        body_style
    ))
    story.append(make_table(
        ["Feature Set", "Mean R²", "Branches R²≥0.3"],
        [
            ["Physical only", f"{cm['mean_r2_physical_only']:.3f}", f"{cm['frac_r2_above_03']*100:.0f}%"],
            ["All features", f"{cm['mean_r2_all_features']:.3f}", f"{cm['frac_r2_above_03']*100:.0f}%"],
            ["Syslog only", f"{cm['mean_r2_syslog_only']:.3f}", "7%"],
        ],
        col_widths=[W*0.45, W*0.28, W*0.27]
    ))
    story.append(Spacer(1, 0.05*inch))

    for fig_path, cap in [
        (EXP / "fig_01a_branch_r2_physical_only.png", "Fig 1a: R² distribution across 100 branches (physical features only)"),
        (EXP / "fig_01b_branch_prediction_scatter.png", "Fig 1b: Top-branch prediction scatter plots"),
        (EXP / "fig_01d_r2_by_state.png", "Fig 1d: R² decomposition by FSM state"),
        (EXP / "fig_01e_physical_vs_syslog_r2.png", "Fig 1e: Physical vs syslog R² comparison"),
    ]:
        story.extend(embed_figure(fig_path, max_height=2.8*inch, caption=cap))

    # ── Section 3: Anomaly detection ───────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("3. Anomaly Detection", h1_style))
    story.append(make_table(
        ["Anomaly Type", "AUC"],
        [
            ["Trace Swap", f"{ad['auc_trace_swap']:.4f}"],
            ["Physical Noise (3σ)", f"{ad['auc_sensor_noise']:.4f}"],
            ["Timing Shift 50ms", f"{ad['auc_timing_50ms']:.4f}"],
            ["Timing Shift 10ms", f"{ad['auc_timing_10ms']:.4f}"],
        ],
        col_widths=[W*0.65, W*0.35]
    ))

    for fig_path, cap in [
        (EXP / "fig_03c_anomaly_roc_curves.png", "Fig 3c: ROC curves for 4 anomaly types"),
        (PAR / "fig_operator_dashboard.png", "Fig A1: Operator dashboard — 4-panel view with consistency score"),
        (PAR / "fig_detection_example.png", "Fig A1b: Detection example ±500ms zoom around injected anomaly"),
        (PAR / "fig_detection_latency.png", "Fig A2: Detection latency by anomaly type"),
        (PAR / "fig_false_positive_analysis.png", "Fig B3: False positive rate analysis"),
    ]:
        story.extend(embed_figure(fig_path, max_height=3.0*inch, caption=cap))

    # ── Section 4: Baselines ───────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("4. Baseline Comparison", h1_style))
    story.append(Paragraph(
        "All methods evaluated on last 20% of data. RF R² negative due to "
        "global split distribution shift — temporal block CV (R²=0.736) is correct.",
        body_style
    ))
    story.append(make_table(
        ["Method", "R²", "AUC"],
        [
            ["State mean", "N/A", f"{bl['state_mean_auc']:.4f}"],
            ["Ridge regression", "negative", f"{bl['ridge_auc']:.4f}"],
            ["Lag-1 autoregression", f"{bl['lag1_ar_r2']:.3f}", f"{bl['lag1_ar_auc']:.4f}"],
            ["Physics rule-based", "N/A", "0.500"],
            ["Random Forest (ours)", f"{bl['rf_r2_global']:.3f}*", f"{bl['rf_auc']:.4f}"],
        ],
        col_widths=[W*0.5, W*0.25, W*0.25]
    ))
    story.append(Paragraph("*RF R² in global split; R²=0.736 in temporal block CV", caption_style))
    story.extend(embed_figure(PAR / "fig_baseline_comparison.png",
        max_height=3.0*inch, caption="Fig B1: Baseline comparison"))

    # ── Section 5: Temporal stability ─────────────────────────────────────────
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("5. Temporal Stability", h1_style))
    story.append(make_table(
        ["Metric", "Value"],
        [
            ["Temporal R² (early→late)", f"{cr['temporal_r2']:.3f}"],
            ["LORO Ridge R²", f"{cr['loro_ridge_r2']:.3f}"],
            ["Consistency score drift", f"+{cr['drift_per_run']:.3f}/run"],
            ["Stability verdict", cr['stability_verdict'].upper()],
        ],
        col_widths=[W*0.6, W*0.4]
    ))
    story.extend(embed_figure(PAR / "fig_temporal_stability.png",
        max_height=3.2*inch, caption="Fig B4: Temporal stability — per-run R² and consistency drift"))

    # ── Section 6: Limitations ─────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("6. Limitations", h1_style))
    story.append(make_table(
        ["Scenario", "AUC / Result", "Detectable?"],
        [
            ["Slow sensor drift", f"After {m['limitations']['slow_drift_detectable_at_cm']:.1f}cm drift", "Marginally"],
            ["Near-setpoint swap", f"AUC={m['limitations']['near_setpoint_swap_auc']:.3f}", "Yes"],
            ["Logic bug (±1 count)", f"AUC={m['limitations']['logic_bug_no_physical_sig_auc']:.3f}", "No"],
        ],
        col_widths=[W*0.4, W*0.35, W*0.25]
    ))
    story.extend(embed_figure(PAR / "fig_limitation_analysis.png",
        max_height=3.0*inch, caption="Fig A3: Limitation analysis — 3 hard scenarios"))

    # ── Section 7: LLM reasoning ───────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("7. LLM-Based Anomaly Explanation (Phase 3)", h1_style))

    story.append(Paragraph("Zero-Shot Baseline (Claude Haiku)", h2_style))
    story.append(make_table(
        ["Metric", "Score"],
        [
            ["Detection accuracy", f"{zs['detection_acc']*100:.1f}%"],
            ["Type accuracy", f"{zs['typing_acc']*100:.1f}%"],
            ["Localization score", f"{zs['localization_acc']*100:.1f}%"],
            ["Reasoning quality", f"{zs['reasoning_score']*100:.1f}%"],
            ["Total score", f"{zs['total_score']:.2f}"],
        ],
        col_widths=[W*0.6, W*0.4]
    ))

    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("Per-Type Zero-Shot Scores:", h3_style))
    per_type_rows = sorted(
        [(t, v.get("mean_score", 0) if isinstance(v, dict) else v)
         for t, v in zs.get("per_type", {}).items()],
        key=lambda x: x[1], reverse=True
    )
    story.append(make_table(
        ["Anomaly Type", "Mean Score"],
        [[t, f"{s:.3f}"] for t, s in per_type_rows],
        col_widths=[W*0.6, W*0.4]
    ))

    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("GRPO Training Configuration", h2_style))
    story.append(make_table(
        ["Parameter", "Value"],
        [
            ["Job ID", str(m['grpo_training']['job_id'])],
            ["Status", m['grpo_training']['status']],
            ["Base model", m['grpo_training']['base_model']],
            ["Algorithm", m['grpo_training']['algorithm']],
            ["Infrastructure", m['grpo_training']['infrastructure']],
            ["Training examples", f"{m['grpo_training']['n_train']:,}"],
            ["Reward mean (validation)", f"{m['grpo_training']['reward_mean']:.3f}"],
            ["Reward std", f"{m['grpo_training']['reward_std']:.3f}"],
        ],
        col_widths=[W*0.45, W*0.55]
    ))

    # ── Section 8: Complete figure gallery ────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("8. Complete Figure Gallery", h1_style))
    story.append(Paragraph(
        f"All {len(ALL_FIGS)} figures generated by this project, in order.",
        body_style
    ))

    for fig_path in ALL_FIGS:
        section = "experiments" if fig_path.parent == EXP else \
                  "parallel" if fig_path.parent == PAR else "phase3"
        cap = f"{fig_path.name} ({section}/)"
        elems = embed_figure(fig_path, max_height=3.5*inch, caption=cap)
        story.extend(elems)
        story.append(Spacer(1, 0.1*inch))

    # ── Build PDF ──────────────────────────────────────────────────────────────
    doc.build(story)
    print(f"Wrote {pdf_path} ({pdf_path.stat().st_size // 1024}KB)")

else:
    print("Skipping PDF generation (reportlab/PIL not available).")
    print("Install with: pip install reportlab Pillow")

print("\nDone.")
print(f"  MD:   {OUT / 'FULL_ASSESSMENT_DOCUMENT.md'}")
print(f"  JSON: {OUT / 'assessment_metrics.json'}")
if HAS_REPORTLAB:
    print(f"  PDF:  {OUT / 'FULL_ASSESSMENT_DOCUMENT.pdf'}")
