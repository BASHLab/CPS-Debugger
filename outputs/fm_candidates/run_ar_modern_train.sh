#!/bin/bash
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=gpu-6-[01-20]
#SBATCH --output=models/slurm_ar_modern_train_%j.log

# AR-modern training (RMSNorm + SwiGLU + QK-Norm + Muon optimizer)
# Usage:
#   sbatch run_ar_modern_train.sh                      # full training (1 GPU)
#   sbatch run_ar_modern_train.sh --sanity-check       # sanity check

SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"
export PATH="/home/simran/.conda/envs/slimllm/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p models

echo "AR-modern training"
nvidia-smi || true

python3 -u ar_modern_train.py \
    --device cuda \
    --num-workers 4 \
    "$@"
RC=$?

echo "Exit code: $RC"
exit $RC
