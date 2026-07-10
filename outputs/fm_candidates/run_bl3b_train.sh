#!/bin/bash
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --exclude=gpu-6-[01-20]
#SBATCH --output=models/slurm_bl3b_train_%j.log

# BL-3b training (LoRA-adapted V-JEPA + Chronos-2 sensor + AR-modern).
# V-JEPA is forwarded every step → ~9 hr / fold expected at 4000 steps.
# 24h time limit gives headroom for slow GPUs / queueing.

SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"
export PATH="/home/simran/.conda/envs/slimllm/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p models

echo "BL-3b training: $@"
nvidia-smi || true

python3 -u bl3b_train.py --device cuda --num-workers 2 "$@"
echo "Exit code: $?"
