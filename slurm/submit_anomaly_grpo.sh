#!/bin/bash
#SBATCH --job-name=cps_anomaly_grpo
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:4
#SBATCH --mem=200G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/slurm/%x_%j.out
#SBATCH --error=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/slurm/%x_%j.err

set -euo pipefail

REPO=/home/simran/allspark-data-exploration/CPS-Debugger
PYTHON=/home/simran/.conda/envs/grpo/bin/python3
export PATH="/home/simran/.conda/envs/grpo/bin:$PATH"
OUT="$REPO/outputs/phase3"

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "GPUs: ${SLURM_GPUS_ON_NODE:-4}"
nvidia-smi --list-gpus 2>/dev/null | head -5 || true
nvidia-smi 2>/dev/null | grep "MiB" | head -4 || true

module load cuda12.6/toolkit/12.6.2 2>/dev/null || true

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((4-1)))
fi

cd "$REPO"
mkdir -p "$OUT/slurm" "$OUT/checkpoints"

# ── Step 0: Verify prerequisites ──────────────────────────────────────────────
echo "[Step 0] Checking prerequisites..."
for f in \
  "$OUT/anomaly_eval_dataset.jsonl" \
  "$OUT/grpo_train.jsonl" \
  "$OUT/sft_anomaly_train.jsonl" \
  "$REPO/scripts/phase3_reward_anomaly.py" \
  "$REPO/configs/cps_anomaly_grpo.yaml"; do
  if [[ ! -f "$f" ]]; then
    echo "  ERROR: Missing $f"; exit 1
  fi
done
# Verify counts
EVAL_N=$(wc -l < "$OUT/anomaly_eval_dataset.jsonl")
TRAIN_N=$(wc -l < "$OUT/grpo_train.jsonl")
echo "  eval dataset: $EVAL_N examples (need ≥250)"
echo "  train dataset: $TRAIN_N examples (need ≥2500)"
[[ "$EVAL_N" -lt 250 ]] && { echo "  ERROR: eval too small"; exit 1; }
[[ "$TRAIN_N" -lt 2500 ]] && { echo "  ERROR: train too small"; exit 1; }
echo "  All prerequisites OK."

# ── Step 1: Verify VERL env ───────────────────────────────────────────────────
echo ""
echo "[Step 1] Checking VERL + torch versions..."
if ! $PYTHON -c "import verl, torch, tensordict" 2>/dev/null; then
  echo "  ERROR: grpo conda env missing verl/torch/tensordict."
  echo "  Run setup_grpo_env.sh first, then resubmit."
  exit 1
fi
VERL_VER=$($PYTHON -c "import verl; print(getattr(verl, '__version__', 'ok'))")
TORCH_VER=$($PYTHON -c "import torch; print(torch.__version__)")
echo "  verl: $VERL_VER  |  torch: $TORCH_VER"

# ── Step 2: SFT warmup ────────────────────────────────────────────────────────
echo ""
echo "[Step 2] SFT warmup (1 epoch on $(wc -l < $OUT/sft_anomaly_train.jsonl) examples)..."
SFT_CKPT="$OUT/checkpoints/sft_anomaly_warmup"
mkdir -p "$SFT_CKPT"

torchrun --nproc_per_node=4 \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files="$OUT/sft_anomaly_train.jsonl" \
  data.val_files="$OUT/sft_anomaly_val.jsonl" \
  data.prompt_key=prompt \
  data.response_key=response \
  data.max_length=4096 \
  model.partial_pretrain=Qwen/Qwen2.5-Coder-7B-Instruct \
  trainer.project_name=cps_debugger \
  trainer.experiment_name=sft_anomaly_warmup_v1 \
  trainer.total_epochs=1 \
  trainer.save_freq=999999 \
  trainer.logger='["console"]' \
  trainer.default_local_dir="$SFT_CKPT" \
  2>&1 | tee "$OUT/sft_anomaly_warmup.log"

SFT_LATEST=$(ls -td "$SFT_CKPT"/global_step_* 2>/dev/null | head -1 || echo "Qwen/Qwen2.5-Coder-7B-Instruct")
echo "  SFT warmup complete. Checkpoint: $SFT_LATEST"

# ── Step 3: GRPO training (primary config, LoRA rank=16) ──────────────────────
echo ""
echo "[Step 3] GRPO training (3 epochs, LoRA rank=16)..."
GRPO_CKPT="$OUT/checkpoints/anomaly_grpo_v1"
mkdir -p "$GRPO_CKPT"

$PYTHON -m verl.trainer.main_ppo \
  --config-name cps_anomaly_grpo \
  --config-path "$REPO/configs" \
  actor_rollout_ref.model.path="$SFT_LATEST" \
  trainer.n_gpus_per_node=4 \
  trainer.default_local_dir="$GRPO_CKPT" \
  trainer.logger='["console"]' \
  2>&1 | tee "$OUT/anomaly_grpo_training.log" || {
    echo ""
    echo "  Primary config failed (likely OOM). Trying fallback config..."
    $PYTHON -m verl.trainer.main_ppo \
      --config-name cps_anomaly_grpo_fallback \
      --config-path "$REPO/configs" \
      actor_rollout_ref.model.path="$SFT_LATEST" \
      trainer.n_gpus_per_node=4 \
      trainer.default_local_dir="${GRPO_CKPT}_fallback" \
      trainer.logger='["console"]' \
      2>&1 | tee "$OUT/anomaly_grpo_training_fallback.log"
    GRPO_CKPT="${GRPO_CKPT}_fallback"
  }

echo "  GRPO training complete."

# ── Step 4: Evaluate ──────────────────────────────────────────────────────────
echo ""
echo "[Step 4] Evaluating GRPO model..."
GRPO_LATEST=$(ls -td "$GRPO_CKPT"/global_step_* 2>/dev/null | head -1)

if [[ -n "${GRPO_LATEST:-}" ]]; then
  ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
  $PYTHON -u scripts/task_3_1b_llm_zero_shot.py \
    --model "$GRPO_LATEST" \
    --n 250 \
    2>&1 | tee "$OUT/anomaly_grpo_eval.log"
else
  echo "  No GRPO checkpoint found, skipping eval."
fi

echo ""
echo "Done: $(date)"
echo "All outputs in: $OUT"
