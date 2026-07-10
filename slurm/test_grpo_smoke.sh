#!/bin/bash
#SBATCH --job-name=grpo_smoke
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=80G
#SBATCH --time=0-00:30:00
#SBATCH --output=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/slurm/%x_%j.out
#SBATCH --error=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/slurm/%x_%j.err

# Smoke test: validates the full grpo training pipeline (SFT + GRPO) on 4xA30.
# Runs just 3 SFT steps + 2 GRPO steps — enough to confirm the code path works
# before trusting the full 7-day L40S job.

set -euo pipefail

REPO=/home/simran/allspark-data-exploration/CPS-Debugger
PYTHON=/home/simran/.conda/envs/grpo/bin/python3
export PATH="/home/simran/.conda/envs/grpo/bin:$PATH"
OUT="$REPO/outputs/phase3"
SMOKE_OUT="$REPO/outputs/smoke_test"
mkdir -p "$SMOKE_OUT"

echo "Host: $(hostname)"
echo "Start: $(date)"
nvidia-smi --list-gpus 2>/dev/null | head -5 || true
nvidia-smi 2>/dev/null | grep "MiB" | head -5 || true

module load cuda12.6/toolkit/12.6.2 2>/dev/null || true
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Step 1: Verify env on this compute node ───────────────────────────────────
echo ""
echo "[Step 1] Verifying grpo env on $(hostname)..."
$PYTHON -c "
import torch, tensordict, verl, transformers
print(f'  torch:       {torch.__version__}')
print(f'  tensordict:  {tensordict.__version__}')
print(f'  transformers:{transformers.__version__}')
print(f'  CUDA:        {torch.cuda.is_available()} ({torch.cuda.device_count()} GPUs)')
print(f'  _list_flatten_with_keys: {hasattr(torch.utils._pytree, \"_list_flatten_with_keys\")}')
"
echo "  env OK."

# ── Step 2: SFT smoke (3 steps, single micro-batch) ──────────────────────────
echo ""
echo "[Step 2] SFT smoke test (3 gradient steps, 4xA30)..."
SFT_CKPT="$SMOKE_OUT/sft_smoke"
mkdir -p "$SFT_CKPT"

HYDRA_FULL_ERROR=1 torchrun --nproc_per_node=2 \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files="$OUT/sft_anomaly_train.parquet" \
  data.val_files="$OUT/sft_anomaly_val.parquet" \
  data.prompt_key=prompt \
  data.response_key=response \
  data.max_length=1024 \
  data.train_batch_size=2 \
  data.micro_batch_size_per_gpu=1 \
  model.partial_pretrain=Qwen/Qwen2.5-Coder-7B-Instruct \
  model.enable_gradient_checkpointing=true \
  trainer.project_name=cps_smoke \
  trainer.experiment_name=sft_smoke \
  trainer.total_epochs=1 \
  trainer.total_training_steps=3 \
  trainer.save_freq=-1 \
  trainer.logger='["console"]' \
  trainer.n_gpus_per_node=2 \
  trainer.default_local_dir="$SFT_CKPT" \
  2>&1 | tee "$SMOKE_OUT/sft_smoke.log"

echo "  SFT smoke: PASSED"

# ── Step 3: GRPO smoke (2 steps) ─────────────────────────────────────────────
echo ""
echo "[Step 3] GRPO smoke test (2 rollout steps, 4xA30)..."
GRPO_CKPT="$SMOKE_OUT/grpo_smoke"
mkdir -p "$GRPO_CKPT"

$PYTHON -m verl.trainer.main_ppo \
  --config-name cps_anomaly_grpo \
  --config-path "$REPO/configs" \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-Coder-7B-Instruct \
  trainer.n_gpus_per_node=2 \
  trainer.default_local_dir="$GRPO_CKPT" \
  trainer.logger='["console"]' \
  trainer.total_training_steps=2 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size=1 \
  critic.ppo_micro_batch_size=1 \
  2>&1 | tee "$SMOKE_OUT/grpo_smoke.log"

echo "  GRPO smoke: PASSED"
echo ""
echo "══════════════════════════════════════"
echo "  SMOKE TEST PASSED — $(date)"
echo "  Full L40S job (1874938) is safe to run."
echo "══════════════════════════════════════"
