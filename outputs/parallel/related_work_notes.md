# Related Work Notes — CPS Debugger Paper

## 1. Multimodal Anomaly Detection in CPS

**What exists:**
- SWaT (Secure Water Treatment) and WADI datasets: physical sensor-only anomaly detection benchmarks for industrial control systems [Goh et al., S&P 2016; Ahmed et al., CPS-SPC 2017]
- LSTM-based approaches: DeepLog [Du et al., CCS 2017], DAGMM [Zong et al., ICLR 2018], OmniAnomaly [Su et al., KDD 2019] — all single-modality (sensor time-series only)
- Graph-based methods: GDN [Deng & Hooi, AAAI 2021], MTAD-GAT [Zhao et al., ICDM 2020] — model sensor correlations but no software execution
- Multi-sensor fusion for fault detection: [Nanduri & Sherry, AAAI 2016], mostly aerospace/automotive, no code-level signal

**Gap our work fills:**
None of these methods observe the software's execution path. They treat the controller as a black box. When a software fault produces physically plausible-looking sensor readings (e.g., a logic bug that still keeps the pendulum balanced), these methods are blind. Our work introduces *execution trace* as a second modality alongside physical sensors — enabling detection of faults that leave no physical signature until they escalate.

---

## 2. Execution Trace Analysis for Debugging

**What exists:**
- Spectrum-based fault localization (SBFL): Tarantula [Jones et al., ICSE 2002], Ochiai [Abreu et al., 2006] — ranks program statements by correlation with test failures. Requires test oracles.
- Dynamic slicing and trace differencing: [Agrawal et al., IEEE TSE 1990; Renieris & Reiss, ICSE 2003]
- LDB [Zeller, FSE 1999]: isolate failure-inducing program states via delta debugging. Requires failing/passing runs explicitly labeled.
- NExT [Sridharan et al., OOPSLA 2007]: fault localization from execution traces without test suites.
- TRACED [Chilimbi et al., ASPLOS 2004]: production execution logging for debugging. Offline analysis only.
- For Wasm specifically: [McFadden et al., NDSS 2017; Cabrera Arteaga et al., 2021] — side-channel and security analysis of Wasm traces, not debugging-focused.

**Gap our work fills:**
All existing trace analysis methods require either (a) labeled failing vs. passing runs, or (b) an explicit oracle. In CPS, faults manifest as gradual degradation; there is no binary pass/fail oracle. We use the *physical sensor state* as an oracle-free reference: if the physical dynamics and the execution path are inconsistent, something is wrong. No labels, no test harness required.

---

## 3. Side-Channel Analysis of Software Execution

**What exists:**
- EM-based instruction recovery from power/EM side channels: [Longo et al., TCHES 2015; Eisenbarth et al., CHES 2010] — infer executed instructions from physical EM emissions during chip execution
- CacheBleed, Spectre [Kocher et al., S&P 2019]: CPU cache timing to infer program execution
- Zeus [Kurmus et al., 2011]: EM-based malware detection by comparing expected vs. observed execution profiles
- PLC fingerprinting from network traffic: [McLaughlin et al., USENIX Security 2014]

**Gap our work fills:**
Side-channel work goes in the *opposite direction*: from physical phenomena to inferring code execution. Our work goes from *physical sensor semantics* (what the physical system is doing) to *predicting* code execution. The physical signal is not a side channel of computation — it is the semantic context driving the computation. We also operate at a higher level of abstraction (Wasm branch execution counts, not instruction-level EM emissions).

---

## 4. LLMs for Code Debugging

**What exists:**
- ChatDBG [Levin et al., 2024]: LLM-powered interactive debugging with GDB integration. Produces natural language explanations of crashes. No physical context.
- RepairAgent [Bouzenia et al., 2024]: autonomous LLM-based program repair. Code-only, no CPS/sensor context.
- LLMAO [Feng et al., 2024]: LLM for Android malware analysis. Single-modality (code).
- Copilot for Bugs: code completion for fixing common bugs. Supervised, no RL.
- SelfEval / SelfDebug [Chen et al., 2023]: LLM uses execution feedback to iteratively fix code. Requires runnable code, not deployed CPS.
- VulRepair, CodeBERT-based fault localization: fine-tuned on code corpora, no physical signal.

**Gap our work fills:**
Existing LLM debugging tools operate on code and test output alone. None receive *physical sensor readings* as part of the debugging context. In CPS, the question "why is the cart drifting?" requires connecting physical observations to code behavior — a form of abductive reasoning across modalities. We introduce this as a structured task with verifiable rewards from execution traces.

---

## 5. LLMs for Sensor/Physical Reasoning

**What exists:**
- Penetrative AI [Xu et al., HotMobile 2023]: LLM reasons about physical world from sensor data (temperature, sound, etc.)
- IoT-LM [Qin et al., 2024]: language model pre-training on IoT sensor streams for general IoT tasks
- LLaSA [Liu et al., 2024]: multimodal LLM for sensor-augmented question answering
- SensorBench [Yuan et al., 2024]: benchmark for evaluating LLM reasoning over sensor data
- Time series LLMs: TEMPO, Time-LLM, GPT4TS — adapt LLMs for time-series prediction. Sensor-only, no code.

**Gap our work fills:**
Existing sensor-reasoning LLMs interpret sensor data to answer questions about the physical environment. None connect sensor readings to software execution. Our task requires the LLM to explain an anomaly in terms of *source code*: "the sensor reading X is inconsistent with the execution trace because the code path Y was taken, which should only happen when Z." This requires joint understanding of physical dynamics and software semantics.

---

## 6. RLVR (Reinforcement Learning with Verifiable Rewards) Beyond Math/Code

**What exists:**
- DeepSeek-R1 [DeepSeek, 2025]: GRPO with rule-based rewards for math/code reasoning. Seminal RLVR work.
- OpenR1 [Hugging Face, 2025]: open reproduction of RLVR for mathematical reasoning
- PRIME [Cui et al., 2025]: RLVR for general instruction following via implicit rewards
- rStar-Math [Li et al., 2025]: MCTS + RLVR for mathematical reasoning
- WebRL [Qi et al., 2024]: RL with online feedback for web agents
- CodeRL [Le et al., NeurIPS 2022]: RL from execution feedback for code generation
- Science proofs [Welleck et al., 2022]: verified rewards for formal mathematics

**Applications to CPS/physical systems:** None found in surveyed literature.

**Gap our work fills:**
RLVR has proven transformative for math and code — domains where ground truth is verifiable by a checker. Physical system debugging has the same property: the *execution trace* is ground truth. If the LLM correctly explains that a THRESHOLD_BUG caused a BALANCE→RESET misclassification, the explanation is verifiable by checking the actual trace. We demonstrate that CPS anomaly explanation is a viable RLVR domain, with 4-component rewards: anomaly detection (binary), type classification (categorical), code localization (string match), and reasoning quality (LLM-graded).

---

## Notes on Positioning

- Primary comparison: vs. sensor-only anomaly detection (SWaT/WADI baselines) — we add execution trace modality
- Secondary comparison: vs. static analysis / SBFL tools — we work on deployed real-time systems without test harnesses
- Tertiary: vs. LLM debugging tools (ChatDBG) — we add physical context and RLVR training
- Key differentiator: *synchronized* multimodal data (sensor + trace at 100Hz) on a deployed industrial controller (not a simulator)
