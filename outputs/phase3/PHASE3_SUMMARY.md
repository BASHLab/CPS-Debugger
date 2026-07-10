# CPS-Debugger Phase 3: Results Summary

Generated automatically by synthesize.py


## 1. Per-Function R² Within BALANCE State

**Overall mean within-BALANCE R²**: 0.614

| Function Group | n branches | Mean R² | Frac R²>0.3 |
|---|---|---|---|
| pou_lqr_sim          |   2 | 0.202 | 0.00 |
| tick                 |  38 | 0.720 | 0.84 |
| memcpy               |  33 | 0.849 | 0.88 |
| memset               |   3 | 0.524 | 1.00 |
| math_lib             |   4 | -0.007 | 0.00 |
| logging              |  20 | 0.205 | 0.35 |

**Decision**: **PROCEED**: pou_lqr_sim R²=0.202 > 0.2 — fine-grained claim holds


## 2. Setpoint Label Verification

| Target label | n windows | Mean achieved pos | Tracking error |
|---|---|---|---|
| -0.090 m | 37,690 | -0.0716 m | +0.0184 m |
| +0.000 m | 32,634 | +0.0117 m | +0.0117 m |
| +0.050 m | 27,983 | +0.0392 m | -0.0108 m |
| +0.090 m | 31,718 | +0.0603 m | -0.0297 m |

## 3. Dataset Inventory

- **Complete runs**: 12
- **Approx 10ms windows**: 438,024
- **With audio**: 6
- **With video**: 6
- **Unextracted tarballs**: 1

Complete run IDs: ['2025-03-13_09-23-44', '2025-03-13_11-12-19', '2025-03-13_13-31-50', '2025-03-13_14-32-27', '2025-03-17_10-06-14', '2025-03-17_10-21-19', '2025-03-17_10-51-58', '2025-03-17_11-06-39', '2025-03-18_12-39-10', '2025-03-19_10-05-47', '2025-03-19_10-20-35', '2025-03-19_10-37-56']


## 4. Video Modality (DINOv2)

- 3 runs have video
- Best candidate for DINOv2: `2025-01-24_16-42-06`
  - Has datalayer: False
  - **WARNING**: Best video run has no datalayer — cannot correlate with trace
*DINOv2 extraction pending (SLURM job)*


## 5. Audio Modality

- **NO_COMPLETE_AUDIO_RUNS**: No runs found with audio + datalayer + trace. Feb04 run has audio but no physical sensor data.
- Audio analysis deferred — no runs have both audio and physical sensors


## 6. Zero-Shot LLM Anomaly Explanation Baseline

- Model: `claude-haiku-4-5-20251001`
- n_examples: 250
- Detection accuracy:   **0.716**
- Typing accuracy:      **0.292**
- Localization accuracy: 0.532
- Reasoning quality:    0.940
- Total score:          **0.620**
- Parse error rate:     0.000

Per-type scores:
| Type | Mean score | n |
|---|---|---|
| NORMAL               | 0.532 | 50 |
| STUCK_SENSOR         | 0.832 | 50 |
| THRESHOLD_BUG        | 0.440 | 50 |
| TIMING_DELAY         | 0.820 | 50 |
| TRACE_SWAP           | 0.477 | 50 |


## 7. GRPO Training Pipeline

- Reward function: `scripts/phase3_reward_anomaly.py`
- Mean reward (zero-shot baseline): 0.6142675257731959
- Assessment: SUITABLE: Reward distribution appropriate for RL training

- Config: 4× A100 GPU, model=Qwen/Qwen2.5-Coder-7B-Instruct, LoRA=True
- Submit: `sbatch /home/simran/allspark-data-exploration/CPS-Debugger/slurm/submit_anomaly_grpo.sh`


## 8. Expanded Dataset Results

*Expanded dataset results not yet available (needs task_1_3b)*


## 9. Training Dataset Status

- **anomaly_eval_dataset.jsonl**: 250 examples ✓
- **grpo_train.jsonl**: 3000 examples ✓
- **sft_anomaly_train.jsonl**: 400 examples ✓
- **sft_anomaly_val.jsonl**: 100 examples ✓
