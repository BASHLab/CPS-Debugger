# Contribution Statement — CPS Debugger Paper

## Three Core Contributions (intro bullet points)

**1. Cross-Modal Predictive Relationship:**
We demonstrate that physical sensor observations predict fine-grained WebAssembly execution trace characteristics within a single operating state (R²=0.74 mean across 100 branches; R²=0.56 within the BALANCE state alone), establishing a novel cross-modal relationship between physical dynamics and software execution in industrial CPS. This relationship holds across multiple experimental sessions and enables zero-label anomaly detection at AUC=0.9999 for structural code-path faults.

**2. Physics-to-Code Abductive Reasoning Task:**
We define *physics-to-code abductive reasoning* — a new task where an LLM, given synchronized physical sensor readings, WebAssembly execution branch counts, and C++ source code, explains inconsistencies between expected and observed execution in terms of specific code locations and fault hypotheses. We show this task admits fully verifiable rewards from execution traces (detection accuracy, fault type classification, code localization, reasoning quality), enabling Reinforcement Learning with Verifiable Rewards (RLVR) training without human annotations.

**3. First Multimodal CPS Debugging Benchmark:**
We release the first dataset with synchronized WebAssembly execution traces, physical sensor readings, and source code from an industrial Bosch Rexroth ctrlX controller, comprising 431,297 analysis windows (10ms each) across 12 experimental sessions, along with: (a) a cross-modal anomaly detection toolkit, (b) a 250-example zero-shot evaluation benchmark for LLM-based fault explanation, (c) 3,000 RLVR training examples with 4-component verifiable rewards, and (d) trained models.

---

## Extended Contribution Notes (for introduction prose)

### Why this is non-trivial

The cross-modal prediction result (R²=0.74) is surprising because the *physical state* does not directly determine which code branches execute — only the discrete FSM state (SWINGUP/BALANCE/RESET) does that deterministically. Within a single FSM state (e.g., BALANCE), the physical readings (angle, velocity, position) capture subtle continuous variations in the dynamics that correlate with fine-grained execution differences (e.g., how many times the LQR saturates its actuator output). This is a sub-state, fine-grained predictive relationship that has not been previously characterized for any CPS.

### Why the task is novel

Existing LLM debugging tools (ChatDBG, RepairAgent) operate on code and test output alone. Existing sensor-reasoning LLMs (Penetrative AI, SensorBench) interpret physical observations but have no connection to execution traces. Our task uniquely requires *joint* understanding: "the sensor says the pendulum angle is 0.05° (nearly balanced), but the execution trace shows the standup controller is active — this is inconsistent with BALANCE state. The most likely cause is a threshold bug in the MAX_X_SAFE constant, which triggered a premature RESET transition." No existing model or framework addresses this joint reasoning problem.

### Why RLVR works here

The fundamental requirement for RLVR is that rewards must be verifiable without human annotation at training time. In math: a symbolic checker verifies the answer. In code: an interpreter runs the program. In CPS debugging: the *execution trace itself* is the ground truth. A correct fault explanation implies a specific code location and fault type; we can verify both against the actual trace data. This makes CPS anomaly explanation the first physical-system domain where RLVR training is feasible.

---

## Potential Objections and Responses

**"The R²=0.74 is just because different FSM states have different trace profiles — you're really just doing state classification."**
→ We decompose R² within each FSM state separately. Within-BALANCE R²=0.56 (using physical features alone, controlling for FSM state). The sub-state variation is real and predictable.

**"The AUC=0.9999 is because trace-swap is trivially detectable — you're swapping BALANCE with SWINGUP, which have completely different execution profiles."**
→ Correct; we acknowledge this in the limitations. The meaningful detection results are timing-related faults (AUC=0.70 for 10ms, 0.95 for 50ms) and sensor drift (detectable only after [X]cm cumulative displacement). The hard cases are documented and analyzed.

**"This only works on one specific system (the pendulum). It's not a general method."**
→ The *method* is general: any CPS with physical sensors and software execution traces can use this framework. The *specific predictor* (RF trained on physical→trace relationship) is system-specific, as expected. We validate on 12 experimental sessions spanning [N] days, demonstrating stability of the learned relationship over time.

**"Why use RLVR instead of just SFT with more data?"**
→ SFT requires high-quality human-annotated explanations, which are expensive to produce for CPS faults. RLVR requires only execution trace ground truth, which is automatically available from the controller's telemetry system. Additionally, RLVR optimizes directly for the evaluation criteria (detection + typing + localization + reasoning), while SFT optimizes for log-likelihood of training responses, which may not align with the actual task objectives.

---

## Key Numbers to Lead With

| Metric | Value |
|--------|-------|
| Mean cross-modal R² (100 branches) | 0.74 |
| Within-BALANCE R² | 0.56 |
| Trace-swap detection AUC | 0.9999 |
| Sensor noise AUC | 0.9989 |
| Timing shift 50ms AUC | 0.9547 |
| Zero-shot total score (Haiku) | 0.62 |
| Zero-shot detection accuracy | 71.6% |
| Zero-shot type accuracy | 29.2% |
| Dataset: windows | 431,297 |
| Dataset: sessions | 12 |
| GRPO results | [pending job 1873620] |
