# Paper Outline — NeurIPS Format (9 pages + references)

**Tentative Title:** "Cross-Modal Consistency Monitoring and Physics-to-Code Abductive Reasoning for CPS Debugging"

**Alternate shorter titles:**
- "Catching Bugs in the Physical World: Multimodal Anomaly Detection for CPS via RLVR"
- "When the Robot Lies: Execution Trace Anomaly Explanation from Physical Sensor Observations"

---

## 1. Introduction (1 page)

**Paragraph 1 — The problem:**
Cyber-Physical Systems (CPS) are increasingly deployed in safety-critical settings, yet debugging them remains uniquely difficult. Unlike pure software, faults in CPS manifest through the interaction of code execution and physical dynamics: a wrong threshold constant keeps the pendulum balanced for seconds before it falls; a timing misalignment produces no visible symptom at the sensor level until a state transition amplifies it. Traditional software debugging tools (debuggers, test harnesses, static analyzers) are blind to this physical context.

**Paragraph 2 — The opportunity:**
We observe that industrial CPS controllers generate two simultaneous streams of information: physical sensor readings (angle, position, velocity) and software execution traces (which branches of compiled code are being taken). These streams are *causally linked*: the physical state determines which code paths execute. This cross-modal relationship is a rich, continuously available supervision signal — no labels, no test harnesses, no offline analysis required.

**Paragraph 3 — The contributions:**
We make three contributions: (1) we characterize the physical-to-execution-trace predictive relationship on a Bosch Rexroth ctrlX industrial pendulum controller, establishing the empirical basis for cross-modal anomaly detection; (2) we define *physics-to-code abductive reasoning* — a new task where an LLM, given sensor readings and source code, explains inconsistencies in terms of executable hypotheses — and train it via RLVR with execution trace rewards; (3) we release the first multimodal CPS debugging dataset with synchronized 100Hz Wasm execution traces, physical sensors, and C++ source code.

**Paragraph 4 — Summary of results:**
Brief forward-reference to key numbers: R²=0.74 mean cross-modal prediction, AUC=0.9999 for trace-swap detection, GRPO vs zero-shot LLM improvement on explanation task.

---

## 2. Related Work (1.5 pages)

*(Structure matches `related_work_notes.md`)*

**2.1 Anomaly Detection in CPS** — SWaT/WADI, LSTM/graph methods, sensor-only gap.

**2.2 Execution Trace Analysis** — SBFL, LDB, NExT. Gap: require test oracles; not for deployed real-time systems.

**2.3 Side-Channel Analysis** — EM/timing side channels. Note: opposite direction (physical→code vs. our code→physical consistency check).

**2.4 LLMs for Code Debugging** — ChatDBG, RepairAgent. Gap: no physical context; no CPS.

**2.5 LLMs for Physical Reasoning** — Penetrative AI, IoT-LM, SensorBench. Gap: no connection to execution traces or source code.

**2.6 RLVR** — DeepSeek-R1, CodeRL. Note: CPS is first application of RLVR outside math/code domains (to our knowledge).

---

## 3. Problem Formulation (1 page)

**3.1 System Model:**
- Controller C compiles to Wasm; executes at ~1kHz control frequency
- Physical state s_t ∈ ℝ^9: position, angle, velocity, angular velocity (aggregated over 10ms windows)
- Execution trace e_t ∈ ℕ^100: branch execution counts per 10ms window (top-100 by coefficient of variation)
- Controller state z_t ∈ {SWINGUP, BALANCE, RESET}: discrete FSM state

**3.2 Cross-Modal Consistency:**
Define the consistency score: C(s_t, e_t) = ‖e_t - f̂(s_t)‖₂ where f̂ is a trained predictor (Random Forest). A window is anomalous if C(s_t, e_t) > τ for some threshold τ.

**3.3 The Anomaly Explanation Task:**
Given: a window with high consistency score C(s_t, e_t) > τ_p99, physical readings s_t, relevant source code snippets, and the branch vocabulary.
Required output: structured JSON with fields:
  - `is_anomalous`: bool
  - `anomaly_type`: THRESHOLD_BUG | TRACE_SWAP | TIMING_DELAY | STUCK_SENSOR | NORMAL
  - `implicated_code`: source location string
  - `reasoning`: natural language explanation

**3.4 Verifiable Reward Definition:**
R = 0.25·R_detect + 0.25·R_type + 0.25·R_localize + 0.25·R_reason
- R_detect: binary match (detected anomaly or correctly classified NORMAL)
- R_type: exact match on anomaly_type (F1)
- R_localize: token-overlap of implicated_code vs. ground truth
- R_reason: LLM-graded quality score (0–1 rubric)

All components verifiable from the execution trace ground truth or automated grading.

---

## 4. Method (2 pages)

**4.1 Cross-Modal Trace Prediction (0.75 page)**
- Feature extraction: 10ms aggregation (mean, std, delta) over raw datalayer rows
- Vocabulary selection: top-100 branches by coefficient of variation (>5% presence filter)
- Model: Random Forest Regressor (n_estimators=100, max_depth=12)
- Evaluation: 5-fold temporal block cross-validation (prevents data leakage)
- Key result: mean R²=0.74 across 100 branches (physical features only: 0.75, syslog only: -0.01)
- Figure: R² distribution histogram, top-10 branch scatter plots

**4.2 Consistency-Based Anomaly Detection (0.5 page)**
- Consistency score: L2 norm of prediction residual
- Threshold calibration: empirical percentiles (p95, p99) on normal training windows
- One-class problem: no anomaly labels needed for training
- ROC analysis: AUC=0.9999 for trace-swap, 0.9989 for sensor noise, 0.70 for 10ms timing shift, 0.95 for 50ms timing shift

**4.3 Physics-to-Code Abductive Reasoning Task (0.5 page)**
- Dataset construction: 250 evaluation examples, 3000 GRPO training examples
- 5 anomaly types: THRESHOLD_BUG, TRACE_SWAP, TIMING_DELAY, STUCK_SENSOR, NORMAL (50 each)
- Prompt structure: physical readings + branch counts + source code + task description
- Zero-shot baseline: Claude Haiku, 71.6% detection, 29.2% type accuracy, total score 0.62

**4.4 RLVR Training with Execution Trace Rewards (0.25 page)**
- Base model: Qwen2.5-Coder-7B-Instruct (SFT warmup 1 epoch → GRPO 3 epochs)
- Algorithm: GRPO (group_size=8, kl_coef=0.001)
- Infrastructure: VERL framework, 4× A100, LoRA rank=16
- Expected results section: [to be filled after GRPO job 1873620 completes]

---

## 5. Experimental Setup (0.5 page)

**5.1 Hardware Platform:**
- Bosch Rexroth ctrlX CORE real-time controller
- Inverted pendulum mechanical plant: ~1m aluminum rod, servo-driven cart, 2.4m rail
- Sensors: encoder (position/angle at 1kHz), CNAP blood pressure sensor (for reference), USB camera, microphone
- Software: C++ control algorithm compiled to Wasm, ctrlX Data Layer telemetry

**5.2 Data Collection:**
- 12 complete experimental sessions (March 2025), 431,297 analysis windows (10ms each = ~71.9 minutes total)
- 6 runs used for model development, 6 additional runs for expanded validation
- Vocab: top-100 branches by CV, frozen from initial 6-run analysis

**5.3 Evaluation Protocol:**
- Temporal block CV for in-distribution evaluation
- Leave-one-run-out for cross-run generalization
- Temporal train (early) → test (late) for distribution shift evaluation
- Synthetic anomaly injection following exp03 protocol

---

## 6. Results (2 pages)

**6.1 Within-State Trace Prediction (R² Decomposition) (0.4 page)**
- Table: mean R² × feature set (physical / syslog / all) × state
- Key finding: physical features explain 84% of branches (R²≥0.3); syslog adds almost nothing
- Within-BALANCE R²=0.560 (harder than cross-state: less variation to predict)
- Bar chart: R² distribution across 100 branches

**6.2 Cross-Run Generalization (0.3 page)**
- LORO R² (Ridge): [from task_1_3c results when available]
- Temporal stability: early-run trained model vs. late-run test set
- Key finding: relationship is stable / shows drift [fill from temporal_stability.json]

**6.3 Anomaly Detection Performance (0.4 page)**
- Table: AUC by anomaly type (4 types × 2 thresholds p95/p99)
- Key finding: near-perfect detection of structural faults (trace swap: AUC=0.9999); timing misalignment harder (10ms: 0.70)
- FPR analysis: p95 FPR=[from B3], p99 FPR=[from B3], 3-window confirmation reduces to [from B3]
- Detection latency: trace-swap detected in [from A2]ms at p99 threshold

**6.4 LLM Reasoning: Zero-Shot vs SFT vs GRPO (0.5 page)**
- Table: detection_acc / typing_acc / localization_acc / reasoning_score / total for all three
- Zero-shot (Haiku): 71.6% / 29.2% / 53.2% / 94.0% / 0.62
- SFT warmup: [to be filled]
- GRPO: [to be filled]
- Per-type breakdown: STUCK_SENSOR and TIMING_DELAY easiest, THRESHOLD_BUG hardest for type classification

**6.5 Ablations and Baselines (0.4 page)**
- Baseline comparison table: state-mean / Ridge / lag-1 AR / physics-rule / RF (ours)
- Key finding: lag-1 AR R²=[from B1] vs RF R²=0.74 (gap justifies physics features)
- Inference cost: [from B2] ms/window, real-time feasibility conclusion

---

## 7. Discussion and Limitations (1 page)

**7.1 What the System Can Detect:**
- Structural faults: wrong code paths (trace swap), threshold bugs, sensor faults
- Timing-related faults with sufficient temporal shift (≥50ms reliably)
- Any fault that produces a discrepancy between physical state and execution path

**7.2 What the System Cannot Detect:**
- Slow proportional sensor drift (detectable only after [X]cm cumulative displacement)
- Near-setpoint same-state execution anomalies (AUC=[from A3], near random)
- Pure logic bugs with no physical signature (AUC=[from A3])
- Faults that change the *output* of computation but not the *path* taken

**7.3 Single-System Limitation:**
- Results on one physical system (Bosch Rexroth ctrlX pendulum)
- Generalization to other CPS platforms not yet validated
- Vocabulary selection (CV-based) may need retuning for different controllers
- The cross-modal relationship is system-specific; transfer learning not explored

**7.4 Synthetic vs. Real Faults:**
- All 5 anomaly types are synthetically injected (no real bugs captured in the wild)
- Real faults may produce more subtle or compound symptoms
- Dataset expansion to include real field bugs is a priority for future work

---

## 8. Conclusion

Three sentences: (1) we introduced cross-modal CPS debugging using physical sensors and execution traces; (2) we demonstrated that this relationship is learnable (R²=0.74), produces near-perfect anomaly detection (AUC=0.9999 for structural faults), and enables LLM-based explanation via RLVR; (3) we release dataset, code, and trained models to support future research.

---

## Figures List (planned)

1. **fig_system_overview** — System diagram: pendulum → sensors → Wasm traces → consistency score → LLM explanation
2. **fig_operator_dashboard** — 4-panel operator view (from A1)
3. **fig_detection_example** — ±500ms zoom around injected anomaly (from A1)
4. **fig_r2_distribution** — R² histogram and top-branch scatter (from exp01)
5. **fig_roc_curves** — ROC curves for 4 anomaly types (from exp03)
6. **fig_baseline_comparison** — Grouped bar: R² and AUC across all methods (from B1)
7. **fig_false_positive_analysis** — FPR timeline and transition-proximity histogram (from B3)
8. **fig_grpo_training_curve** — GRPO reward vs. training steps (from GRPO job)
9. **fig_limitation_analysis** — 3-panel limitations (from A3)

*Estimated total figures: 9, target 6-7 in main paper, rest in appendix*

---

## Tables List (planned)

1. **tab_dataset** — Data statistics (runs, windows, states, vocab size)
2. **tab_r2_by_state** — R² decomposition by feature set and controller state
3. **tab_anomaly_detection** — AUC by type at p95/p99 with FPR
4. **tab_llm_results** — Zero-shot vs SFT vs GRPO on all metrics
5. **tab_baselines** — Baseline comparison (R², cosine sim, AUC)
6. **tab_inference** — Inference cost comparison

---

## Page Budget Estimate

| Section       | Target pages |
|---------------|-------------|
| Introduction  | 1.0 |
| Related Work  | 1.5 |
| Formulation   | 1.0 |
| Method        | 2.0 |
| Setup         | 0.5 |
| Results       | 2.0 |
| Discussion    | 1.0 |
| Conclusion    | 0.25 |
| References    | 0.75+ |
| **Total**     | **10+** |

*Will need to trim to 9 pages. Candidates: collapse setup into method, compress related work to 1 page, move some results to appendix.*
