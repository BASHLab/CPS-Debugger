#!/bin/bash
#SBATCH --job-name=cps_trace_mlp_holdout
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/slurm/%x_%j.out
#SBATCH --error=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/slurm/%x_%j.err

set -euo pipefail

if [[ -z "${TEST_RUN:-}" ]]; then
  echo "TEST_RUN is required (e.g., 2025-03-13_09-23-44)"
  exit 2
fi

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "TEST_RUN=${TEST_RUN}"

module load python39 >/dev/null 2>&1 || true
module load cuda12.6/toolkit/12.6.2 >/dev/null 2>&1 || true
if [[ -f /scratch/home/ims1pi/env_gpu/bin/activate ]]; then
  source /scratch/home/ims1pi/env_gpu/bin/activate
fi

REPO=/home/simran/allspark-data-exploration/CPS-Debugger
DATA=/home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs/extracted
OUT=$REPO/outputs
mkdir -p "$OUT/slurm"

RUNS=(
  "$DATA/2025-03-13_09-23-44"
  "$DATA/2025-03-13_11-12-19"
  "$DATA/2025-03-13_13-31-50"
  "$DATA/2025-03-13_14-32-27"
  "$DATA/2025-03-17_10-06-14"
  "$DATA/2025-03-17_10-21-19"
)

cd "$REPO"
python3 train.py \
  --runs "${RUNS[@]}" \
  --split_mode run_holdout \
  --test_runs "$TEST_RUN" \
  --target edges \
  --topk_edges 100 \
  --features fused \
  --model mlp \
  --missing_mode train_dropout \
  --random_state 0 \
  --mlp_hidden 256 \
  --mlp_dropout 0.3 \
  --mlp_lr 8e-4 \
  --mlp_weight_decay 1e-4 \
  --mlp_batch_size 512 \
  --mlp_epochs 35 \
  --train_dropout_prob 0.25 \
  --mlp_max_pos_weight 25.0 \
  --mlp_grad_clip_norm 1.0 \
  --output_json "$OUT/trace_pred_holdout_${TEST_RUN}_mlp_train_dropout.json"
