#!/bin/bash
# orchestrate.sh — Phase 3 wave orchestrator
#
# Runs all analysis tasks in the correct dependency order,
# parallelizing independent tasks within each wave.
#
# Usage:
#   cd <this checkout>
#   bash orchestrate.sh
#
# To skip a wave (e.g. if Wave 1 already ran):
#   SKIP_WAVE1=1 bash orchestrate.sh

set -euo pipefail

REPO="${CPSD_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${CPSD_PYTHON:-python3}"
LOGS=$REPO/logs
mkdir -p "$LOGS" "$REPO/outputs/phase3" "$REPO/outputs/slurm"


echo "=================================================="
echo "  CPS-Debugger Phase 3 Orchestrator"
echo "  $(date)"
echo "=================================================="

# ── Helper ────────────────────────────────────────────────────────────────────
run_bg() {
  local name=$1; shift
  echo "  [LAUNCH] $name"
  $PYTHON -u "$@" > "$LOGS/${name}.log" 2>&1 &
  echo $! > "$LOGS/${name}.pid"
}

wait_all() {
  echo "  Waiting for all background tasks..."
  wait
  echo "  All tasks complete."
}

check_output() {
  local file=$1
  if [[ -f "$file" ]]; then
    echo "  [OK] $file"
    return 0
  else
    echo "  [MISSING] $file"
    return 1
  fi
}

cd "$REPO"

# ══════════════════════════════════════════════════════════════════════════════
if [[ "${SKIP_WAVE1:-0}" != "1" ]]; then
  echo ""
  echo "── WAVE 1 (all independent) ──────────────────────────────────────────"
  echo "  Starting $(date)"

  run_bg task_1_1 scripts/task_1_1_function_r2.py
  run_bg task_1_2 scripts/task_1_2_setpoint_verify.py
  run_bg task_1_3 scripts/task_1_3_data_inventory.py
  run_bg task_2_1a scripts/task_2_1a_video_inventory.py

  wait_all
  echo "  Wave 1 finished $(date)"

  # Check outputs
  echo "  Checking Wave 1 outputs:"
  check_output "$REPO/outputs/phase3/function_r2_summary.json"    || true
  check_output "$REPO/outputs/phase3/setpoint_verification.json"  || true
  check_output "$REPO/outputs/phase3/data_inventory.csv"          || true
  check_output "$REPO/outputs/phase3/video_inventory.json"        || true
fi

# ── WAVE 1 DECISION GATE ─────────────────────────────────────────────────────
echo ""
echo "── DECISION GATE ─────────────────────────────────────────────────────"
if [[ -f "$REPO/outputs/phase3/function_r2_summary.json" ]]; then
  LQR_R2=$(python3 -c "
import json
d = json.load(open('$REPO/outputs/phase3/function_r2_summary.json'))
groups = {g['group']: g['mean_r2'] for g in d.get('groups', [])}
print(f\"pou_lqr_sim R2 = {groups.get('pou_lqr_sim', 'N/A')}\")
print(f\"memcpy R2      = {groups.get('memcpy', 'N/A')}\")
print(f\"tick R2        = {groups.get('tick', 'N/A')}\")
" 2>/dev/null)
  echo "  $LQR_R2"
fi
if [[ -f "$REPO/outputs/phase3/data_inventory_summary.json" ]]; then
  N_RUNS=$(python3 -c "
import json
d = json.load(open('$REPO/outputs/phase3/data_inventory_summary.json'))
print(f\"Complete runs: {d.get('n_complete_runs', '?')}\")
" 2>/dev/null)
  echo "  $N_RUNS"
fi

# ══════════════════════════════════════════════════════════════════════════════
if [[ "${SKIP_WAVE2:-0}" != "1" ]]; then
  echo ""
  echo "── WAVE 2 (after Wave 1) ─────────────────────────────────────────────"
  echo "  Starting $(date)"

  # task_3_1 and task_3_2 don't need expanded dataset — launch immediately
  run_bg task_3_1 scripts/task_3_1_build_anomaly_eval.py
  run_bg task_3_2 scripts/task_3_2_build_grpo_data.py
  run_bg task_2_2 scripts/task_2_2_audio_features.py
  run_bg task_2_1a_video scripts/task_2_1a_video_inventory.py  # lightweight, re-run to ensure

  # task_1_3b is heavy (~2 hours) — only run if more than 6 runs available
  N_COMPLETE=$(python3 -c "
import json; d = json.load(open('$REPO/outputs/phase3/data_inventory_summary.json'))
print(d.get('n_complete_runs', 6))" 2>/dev/null || echo "6")
  if [[ "$N_COMPLETE" -gt 6 ]]; then
    echo "  Found $N_COMPLETE complete runs — launching expand_preprocess (heavy)"
    run_bg task_1_3b scripts/task_1_3b_expand_preprocess.py
  else
    echo "  Only $N_COMPLETE complete runs (same as baseline) — skipping expand_preprocess"
  fi

  # DINOv2: submit as SLURM job if GPU available, else run in background
  if [[ -f "$REPO/outputs/phase3/video_inventory.json" ]]; then
    BEST_RUN=$(python3 -c "
import json
d = json.load(open('$REPO/outputs/phase3/video_inventory.json'))
print(d.get('best_dino_candidate', ''))" 2>/dev/null || echo "")
    if [[ -n "$BEST_RUN" ]]; then
      if command -v sbatch &>/dev/null; then
        echo "  Submitting DINOv2 as SLURM job for run: $BEST_RUN"
        cat > /tmp/dino_slurm.sh << SLURMEOF
#!/bin/bash
#SBATCH --job-name=dino_extract
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=$LOGS/dino_extract_%j.log
cd $REPO
$PYTHON -u scripts/task_2_1b_dino_extract.py --run_id $BEST_RUN
SLURMEOF
        sbatch /tmp/dino_slurm.sh || echo "  SLURM submission failed, skipping DINOv2"
      else
        run_bg task_2_1b scripts/task_2_1b_dino_extract.py --run_id "$BEST_RUN"
      fi
    fi
  fi

  wait_all
  echo "  Wave 2 finished $(date)"

  echo "  Checking Wave 2 outputs:"
  check_output "$REPO/outputs/phase3/anomaly_eval_dataset.jsonl" || true
  check_output "$REPO/outputs/phase3/grpo_train.jsonl"           || true
  check_output "$REPO/outputs/phase3/sft_anomaly_train.jsonl"    || true
fi

# ══════════════════════════════════════════════════════════════════════════════
if [[ "${SKIP_WAVE3:-0}" != "1" ]]; then
  echo ""
  echo "── WAVE 3 (after Wave 2) ─────────────────────────────────────────────"
  echo "  Starting $(date)"

  # Check API key
  if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "  WARNING: ANTHROPIC_API_KEY not set — skipping LLM eval (task_3_1b)"
    SKIP_LLM=1
  else
    SKIP_LLM=0
  fi

  if [[ "$SKIP_LLM" == "0" ]]; then
    run_bg task_3_1b scripts/task_3_1b_llm_zero_shot.py --n 250
  fi

  # These don't need API key
  run_bg task_3_3 scripts/task_3_3_reward_function.py
  run_bg task_3_4 scripts/task_3_4_verl_config.py

  # task_1_3c needs expanded dataset
  if [[ -f "$REPO/outputs/phase3/aligned_dataset_expanded.parquet" ]]; then
    run_bg task_1_3c scripts/task_1_3c_rerun_analyses.py
  else
    echo "  No expanded dataset — skipping task_1_3c"
  fi

  wait_all
  echo "  Wave 3 finished $(date)"

  echo "  Checking Wave 3 outputs:"
  check_output "$REPO/outputs/phase3/anomaly_zero_shot_results.json" || true
  check_output "$REPO/scripts/phase3_reward_anomaly.py"              || true
  check_output "$REPO/slurm/submit_anomaly_grpo.sh"                  || true
fi

# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "── WAVE 4: Synthesis ─────────────────────────────────────────────────"
$PYTHON -u scripts/synthesize.py 2>&1 | tee "$LOGS/synthesize.log"

echo ""
echo "── Submit GRPO job ───────────────────────────────────────────────────"
if [[ -f "$REPO/slurm/submit_anomaly_grpo.sh" ]]; then
  echo "  To submit GRPO training: sbatch $REPO/slurm/submit_anomaly_grpo.sh"
  read -p "  Submit now? [y/N] " CONFIRM
  if [[ "${CONFIRM:-n}" == "y" ]]; then
    JOB_ID=$(sbatch "$REPO/slurm/submit_anomaly_grpo.sh" | awk '{print $NF}')
    echo "  Submitted GRPO job: $JOB_ID"
  fi
fi

echo ""
echo "=================================================="
echo "  Phase 3 complete: $(date)"
echo "  Results in: $REPO/outputs/phase3/"
echo "  Summary:    $REPO/outputs/phase3/PHASE3_SUMMARY.md"
echo "=================================================="
