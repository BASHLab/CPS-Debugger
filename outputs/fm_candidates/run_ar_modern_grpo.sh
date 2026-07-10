#!/bin/bash
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --exclude=gpu-6-[01-20]
#SBATCH --output=models/slurm_ar_modern_grpo_%j.log

# GRPO post-step on the AR-modern decoder (Phase B).
# Runs under the `grpo` conda env (transformers 4.57 + peft 0.18).
# Usage:
#   sbatch run_ar_modern_grpo.sh --policy-ckpt PATH --emb-dir DIR \
#          --train-sessions s1 s2 ... --output-dir DIR

SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"
export PATH="/home/simran/.conda/envs/grpo/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p models

echo "AR-modern GRPO"
nvidia-smi || true

python3 -u ar_modern_grpo.py --device cuda "$@"
RC=$?

echo "Exit code: $RC"
exit $RC
