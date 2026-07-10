#!/bin/bash
#SBATCH --job-name=cps_grpo_phase2
#SBATCH --partition=long               # Long-running GPU partition (7d limit)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4                   # 4× A100 for 7B model; use gpu:1 for 3B smoke test
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00             # 48h — enough for 3 GRPO epochs on 2000 examples
#SBATCH --output=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/slurm/%x_%j.out
#SBATCH --error=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/slurm/%x_%j.err

# CPS GRPO Phase 2 Training Pipeline
#
# Pipeline:
#   1. Build SFT dataset (if not already built)
#   2. Build eval dataset (if not already built)
#   3. SFT warmup: 1 epoch on 1600 examples
#   4. GRPO training: 3 epochs
#   5. Evaluate trained model on held-out test set
#
# Usage:
#   sbatch slurm/submit_verl_grpo.sh
#   Or with quick 3B smoke test:
#   sbatch --gres=gpu:1 --export=MODEL=Qwen/Qwen2.5-Coder-3B-Instruct slurm/submit_verl_grpo.sh

set -euo pipefail

REPO=/home/simran/allspark-data-exploration/CPS-Debugger
PYTHON=/home/simran/.conda/envs/slimllm/bin/python3
OUT="$REPO/outputs/phase2"
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
N_GPUS="${SLURM_GPUS_ON_NODE:-4}"

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Model: $MODEL"
echo "GPUs requested: $N_GPUS"
nvidia-smi --list-gpus 2>/dev/null | head -5 || echo "(nvidia-smi unavailable)"
echo ""

module load cuda12.6/toolkit/12.6.2 2>/dev/null || true

# Build CUDA_VISIBLE_DEVICES from SLURM allocation if not set
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS-1)))
fi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Anthropic API key (needed if eval steps use Claude)
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"

cd "$REPO"

# ── Step 0: Verify Phase 2 prerequisites ────────────────────────────────────
echo "[Step 0] Checking Phase 2 outputs..."
if [[ ! -f "$OUT/code_context.json" ]]; then
  echo "  Building code context (B1)..."
  $PYTHON -u scripts/phase2_B1_prepare_code_context.py
fi

if [[ ! -f "$OUT/llm_eval_dataset.jsonl" ]]; then
  echo "  Building eval dataset (B2)..."
  $PYTHON -u scripts/phase2_B2_build_eval_dataset.py
fi

if [[ ! -f "$OUT/sft_train.jsonl" ]]; then
  echo "  Building SFT dataset (C1)..."
  $PYTHON -u scripts/phase2_C1_build_sft_dataset.py
fi
echo "  Prerequisites OK."

# ── Step 1: Install VERL ─────────────────────────────────────────────────────
echo ""
echo "[Step 1] Checking VERL installation..."
if ! $PYTHON -c "import verl" 2>/dev/null; then
  echo "  Installing verl..."
  $PYTHON -m pip install verl --quiet
fi
echo "  VERL ready."

# ── Step 2: SFT warmup (verl 0.7.0 API) ─────────────────────────────────────
echo ""
echo "[Step 2] SFT warmup (1 epoch on $(wc -l < $OUT/sft_train.jsonl) examples)..."
SFT_CKPT="$OUT/checkpoints/sft_warmup"
mkdir -p "$SFT_CKPT"

# verl 0.7.0: fsdp_sft_trainer is invoked as a module with hydra overrides
torchrun --nproc_per_node="$N_GPUS" \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files="$OUT/sft_train.jsonl" \
  data.val_files="$OUT/sft_val.jsonl" \
  data.prompt_key=prompt \
  data.response_key=prompt \
  data.max_length=4096 \
  model.partial_pretrain="$MODEL" \
  trainer.project_name=cps_debugger \
  trainer.experiment_name=sft_warmup_v1 \
  trainer.total_epochs=1 \
  trainer.save_freq=999999 \
  trainer.logger='["console"]' \
  trainer.default_local_dir="$SFT_CKPT" \
  2>&1 | tee "$OUT/sft_warmup.log"

# Find the last checkpoint
SFT_LATEST=$(ls -td "$SFT_CKPT"/global_step_* 2>/dev/null | head -1 || echo "$MODEL")
echo "  SFT warmup complete. Checkpoint: $SFT_LATEST"

# ── Step 3: GRPO training (verl 0.7.0 main_ppo) ──────────────────────────────
echo ""
echo "[Step 3] GRPO training (3 epochs)..."
GRPO_CKPT="$OUT/checkpoints/grpo_v1"
mkdir -p "$GRPO_CKPT"

$PYTHON -m verl.trainer.main_ppo \
  --config-name cps_grpo \
  --config-path "$REPO/configs" \
  actor_rollout_ref.model.path="$SFT_LATEST" \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.default_local_dir="$GRPO_CKPT" \
  trainer.logger='["console"]' \
  2>&1 | tee "$OUT/grpo_training.log"

echo "  GRPO training complete."

# ── Step 4: Evaluation ───────────────────────────────────────────────────────
echo ""
echo "[Step 4] Evaluating GRPO model on held-out test set..."
$PYTHON -u scripts/phase2_C5_evaluate_grpo.py \
  --model_path "$GRPO_CKPT/global_step_latest" \
  --dataset "$OUT/llm_eval_dataset_holdout.jsonl" \
  --output "$OUT/grpo_eval_results.json" \
  2>&1 | tee "$OUT/grpo_eval.log"

echo ""
echo "Done: $(date)"
echo "All outputs in: $OUT"
