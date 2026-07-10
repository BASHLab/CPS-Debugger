# CPS Debugger — Full Experiment Assessment Document
*Generated: 2026-03-11T22:39:48.623115*
*GRPO Job 1873620: PENDING — Nodes DOWN/reserved*

---

## Executive Summary

This document presents the complete experimental results for the CPS Debugger project: a
cross-modal anomaly detection and physics-to-code abductive reasoning system for the
Bosch Rexroth ctrlX industrial inverted-pendulum controller.

**Three core contributions:**
1. **Cross-modal predictive relationship**: Physical sensor observations predict WebAssembly
   execution branch counts (R²=0.748 physical-only, mean across 100 branches).
2. **Physics-to-code abductive reasoning task**: LLM-based fault explanation with verifiable
   rewards from execution traces; RLVR training via GRPO.
3. **First multimodal CPS debugging benchmark**: 431,297 windows across
   12 sessions, synchronized 100Hz Wasm traces + physical sensors + C++ source.

**Key numbers:**

| Metric | Value |
|--------|-------|
| Dataset windows (expanded) | 431,297 |
| Dataset sessions | 12 |
| Mean cross-modal R² (physical only) | 0.748 |
| Mean cross-modal R² (all features) | 0.736 |
| Branches with R²≥0.3 | 83% |
| Within-BALANCE R² | 0.560 |
| Trace-swap detection AUC | 1.0000 |
| Sensor noise detection AUC | 0.9989 |
| Timing shift 50ms AUC | 0.9547 |
| Timing shift 10ms AUC | 0.7008 |
| Zero-shot total score (Haiku) | 0.62 |
| Zero-shot detection accuracy | 71.6% |
| Zero-shot type accuracy | 29.2% |
| Full pipeline latency | 3.40ms (<10ms window) |
| GRPO results | *PENDING — job 1873620* |

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
| Complete experimental runs | 12 |
| Collection dates | March 2025 (12 sessions) |
| Windows (expanded 12-run dataset) | 431,297 |
| Windows (original 6-run dataset) | 137,904 |
| Wasm branch vocabulary size | 100 |
| Analysis window duration | 10ms |

**Controller state distribution (expanded dataset):**

| State | Windows | Fraction |
|-------|---------|----------|
| BALANCE | 407,931 | 94.6% |
| SWINGUP | 15,340 | 3.6% |
| RESET | 8,026 | 1.9% |

---

## 2. Cross-Modal Trace Prediction (Exp01)

Physical sensor observations (9 features: position, angle, velocity, angular velocity)
are used to predict WebAssembly branch execution counts (top-100 branches by coefficient
of variation) using Random Forest regression with 5-fold temporal block cross-validation.

### 2.1 Prediction Performance

| Feature Set | Mean R² | Branches R²≥0.3 |
|-------------|---------|-----------------|
| Physical only | **0.748** | 83% |
| All features | 0.736 | 83% |
| Syslog only | -0.010 | 7% |

**Mean cosine similarity**: 0.9999

**Key finding**: Physical features alone explain 83% of branches (R²≥0.3).
Syslog adds almost nothing (R²≈0). This confirms the cross-modal relationship is driven
by physical dynamics, not software logging.

**Information decomposition**:
- Branches explained by physical only: 75
- Branches explained by both: 7
- Branches explained by neither: 16

### 2.2 Within-State Prediction

Within BALANCE state (controlling for FSM state), physical features achieve:
- **6-run dataset**: R²=0.560
- **Expanded 12-run dataset**: pending (task_1_3c)

**Setpoint classification** (within BALANCE, 4 setpoints: 0m, ±0.05m, ±0.09m):
- Physical features only: accuracy=97.3%
- Branch traces: accuracy=37.0%
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
| State classification micro-F1 | 0.925 ± 0.008 |
| Regression R² (physical, LORO) | 0.653 |

**Per-run R² (physical only)**:
- 2025-03-13_09-23-44: R²=0.637, micro-F1=0.910
- 2025-03-13_11-12-19: R²=0.712, micro-F1=0.921
- 2025-03-13_13-31-50: R²=0.623, micro-F1=0.923
- 2025-03-13_14-32-27: R²=0.526, micro-F1=0.926
- 2025-03-17_10-06-14: R²=0.676, micro-F1=0.934
- 2025-03-17_10-21-19: R²=0.742, micro-F1=0.934

### 3.2 Temporal Stability (B4)

Train on first 3 runs chronologically, test on last 3.

| Metric | Value |
|--------|-------|
| Temporal R² (early→late) | 0.651 |
| LORO Ridge R² | -0.189 |
| Consistency score drift | +1.128/run |
| Stability verdict | **UNSTABLE** |

**Interpretation**: The temporal R²=0.651 shows the RF trained on early runs predicts
late-run branches reasonably well. However, LORO Ridge R²=-0.189 indicates ridge regression
does not generalize across runs in leave-one-out setting. The +1.13/run
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
| p95 | 17.61 |
| p99 | 22.62 |

### 4.2 Detection Performance (ROC AUC)

| Anomaly Type | AUC |
|-------------|-----|
| Trace Swap (wrong code path) | **1.0000** |
| Physical Noise (3σ sensor noise) | **0.9989** |
| Temporal Shift 50ms (large timing error) | 0.9547 |
| Temporal Shift 10ms (small timing error) | 0.7008 |

**Key finding**: Structural faults (wrong code path, sensor corruption) are detected
near-perfectly. Timing misalignment is harder: 10ms shift (one analysis window) achieves
only AUC=0.701; 50ms shift (5 windows) achieves AUC=0.955.

### 4.3 Detection Latency (B2)

| Anomaly Type | Same-Window Detection (p95) | Mean Latency |
|-------------|---------------------------|--------------|
| Trace Swap | 100% | 0.0ms |
| Sensor Corruption (3σ) | 96% | 0.4ms |
| Timing Shift 50ms | 80% | 2.0ms |

### 4.4 False Positive Rate (B3)

| Configuration | FPR |
|--------------|-----|
| p95 immediate | 5.15% (as designed) |
| p99 immediate | 1.21% |
| p95 + 3-window confirmation | 1.05% |
| p99 + 3-window confirmation | **0.12%** |

FPs near state transitions (within 50ms): **0%**
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
| State mean (no ML) | 0.6268 (AUC only) | 0.998 | 0.6268 |
| Ridge regression | negative | 0.998 | 0.5650 |
| **Lag-1 autoregression** | **0.652** | 0.9998 | **0.9884** |
| Physics rule-based | N/A | N/A | 0.500 |
| **Random Forest (ours)** | -12.878* | 0.9999 | **0.9999** |

*RF R²=-12.878 in global 80/20 split; R²=0.736 in temporal block CV (correct protocol)

**Critical finding**: Lag-1 AR achieves AUC=0.988 vs RF AUC=0.9999. The temporal
autocorrelation of branch counts is strong (branch counts in window t predict window t+1
well). The RF adds value at the margin — predicting from physical semantics rather than
recent history — which matters most when a fault disrupts the physics-to-code relationship.

![Baseline comparison](parallel/fig_baseline_comparison.png)

---

## 6. Inference Cost (B2)

| Component | Latency |
|-----------|---------|
| RF single-window prediction | 3.31ms |
| Full pipeline (RF + consistency score) | **3.40ms** |
| Analysis window | 10ms |
| Margin | 2.9× |

**Verdict**: Real-time feasible. The full pipeline runs in 3.40ms,
well within the 10ms analysis window. The system can monitor every window on a single CPU core.

LLM anomaly explanation (Claude Haiku): ~5s per call, invoked only on anomaly events.
At p99 FPR <1%, this means ~few calls per hour in steady-state operation.

---

## 7. Limitations (A3)

Three categories of hard-to-detect faults:

| Scenario | Result | Detectable? |
|----------|--------|-------------|
| Slow sensor drift | Detectable after 1.0cm cumulative displacement | Marginally |
| Near-setpoint BALANCE swap | AUC=0.953 | Yes (setpoint gap large enough) |
| Logic bug ±1 branch count | AUC=0.530 | **No** |

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
{
  "is_anomalous": bool,
  "anomaly_type": "THRESHOLD_BUG | TRACE_SWAP | TIMING_DELAY | STUCK_SENSOR | NORMAL",
  "implicated_code": "source location string",
  "reasoning": "natural language explanation"
}
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
| Detection accuracy | 71.6% |
| Type accuracy | 29.2% |
| Localization score | 53.2% |
| Reasoning quality | 94.0% |
| **Total score** | **0.62** |

**Per-type zero-shot scores:**

| Anomaly Type | Score |
|-------------|-------|
| STUCK_SENSOR | 0.832 |
| TIMING_DELAY | 0.820 |
| NORMAL | 0.532 |
| TRACE_SWAP | 0.477 |
| THRESHOLD_BUG | 0.440 |

**Key finding**: STUCK_SENSOR and TIMING_DELAY are easiest (physical signal obvious);
THRESHOLD_BUG and TRACE_SWAP are hardest (require understanding code semantics).

### 8.4 Reward Function Validation

| Metric | Value |
|--------|-------|
| n examples scored | 194 |
| Mean reward | 0.614 |
| Std reward | 0.223 |
| Fraction > 0.5 | 37.6% |
| Assessment | **SUITABLE: Reward distribution appropriate for RL training** |

### 8.5 GRPO Training

| Parameter | Value |
|-----------|-------|
| Job ID | 1873620 |
| Status | PENDING — Nodes DOWN/reserved |
| Base model | Qwen2.5-Coder-7B-Instruct |
| Algorithm | GRPO (group_size=8, kl_coef=0.001) |
| Infrastructure | 4× A100, LoRA rank=16, VERL framework |
| Training examples | 3,000 |
| Evaluation examples | 250 |

*GRPO results (SFT vs GRPO comparison) will be added once job 1873620 completes.*

---

## 9. Additional Analyses

### 9.1 Setpoint Verification

4-setpoint classification within BALANCE state (0m, +0.05m, −0.09m, +0.09m):
- Physical-only accuracy: 97.3%
- Branch trace accuracy: 37.0%

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
| GRPO results pending | Paper incomplete | Job 1873620 scheduled 2026-03-17 |

---

## 11. Figure Index

All 30 figures generated by this project:

**Experiment Figures (outputs/experiments/):**
- [fig_01a_branch_r2_all_features.png](experiments/fig_01a_branch_r2_all_features.png) (229KB)
- [fig_01a_branch_r2_physical_only.png](experiments/fig_01a_branch_r2_physical_only.png) (244KB)
- [fig_01b_branch_prediction_scatter.png](experiments/fig_01b_branch_prediction_scatter.png) (720KB)
- [fig_01c_feature_importance.png](experiments/fig_01c_feature_importance.png) (189KB)
- [fig_01d_r2_by_state.png](experiments/fig_01d_r2_by_state.png) (149KB)
- [fig_01e_physical_vs_syslog_r2.png](experiments/fig_01e_physical_vs_syslog_r2.png) (203KB)
- [fig_02a_setpoint_accuracy_comparison.png](experiments/fig_02a_setpoint_accuracy_comparison.png) (230KB)
- [fig_02b_setpoint_confusion_best.png](experiments/fig_02b_setpoint_confusion_best.png) (172KB)
- [fig_02c_setpoint_feature_importance.png](experiments/fig_02c_setpoint_feature_importance.png) (199KB)
- [fig_02d_setpoint_prediction_timeline.png](experiments/fig_02d_setpoint_prediction_timeline.png) (291KB)
- [fig_02e_setpoint_umap.png](experiments/fig_02e_setpoint_umap.png) (1244KB)
- [fig_03a_consistency_timeline.png](experiments/fig_03a_consistency_timeline.png) (718KB)
- [fig_03b_consistency_distribution.png](experiments/fig_03b_consistency_distribution.png) (174KB)
- [fig_03c_anomaly_roc_curves.png](experiments/fig_03c_anomaly_roc_curves.png) (232KB)
- [fig_03d_anomaly_score_histograms.png](experiments/fig_03d_anomaly_score_histograms.png) (342KB)
- [fig_05a_audio_branch_r2.png](experiments/fig_05a_audio_branch_r2.png) (175KB)
- [fig_05b_audio_feature_importance.png](experiments/fig_05b_audio_feature_importance.png) (185KB)
- [fig_05c_audio_trace_timeline.png](experiments/fig_05c_audio_trace_timeline.png) (312KB)
- [fig_06_cross_run_generalization.png](experiments/fig_06_cross_run_generalization.png) (225KB)

**Parallel Analysis Figures (outputs/parallel/):**
- [fig_baseline_comparison.png](parallel/fig_baseline_comparison.png) (176KB)
- [fig_detection_example.png](parallel/fig_detection_example.png) (372KB)
- [fig_detection_latency.png](parallel/fig_detection_latency.png) (218KB)
- [fig_false_positive_analysis.png](parallel/fig_false_positive_analysis.png) (593KB)
- [fig_limitation_analysis.png](parallel/fig_limitation_analysis.png) (303KB)
- [fig_operator_dashboard.png](parallel/fig_operator_dashboard.png) (759KB)
- [fig_temporal_stability.png](parallel/fig_temporal_stability.png) (415KB)

**Phase 3 Figures (outputs/phase3/):**
- [fig_anomaly_results.png](phase3/fig_anomaly_results.png) (91KB)
- [fig_function_r2_branches.png](phase3/fig_function_r2_branches.png) (137KB)
- [fig_function_r2_groups.png](phase3/fig_function_r2_groups.png) (160KB)
- [fig_setpoint_verification.png](phase3/fig_setpoint_verification.png) (338KB)

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
