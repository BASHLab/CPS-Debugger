#!/bin/bash
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=gpu-6-[01-20]
#SBATCH --output=models/slurm_bl_video_train_%j.log

# Video-conditioned AR-modern training (BL-3 / BL-4 / BL-5).
# Usage: sbatch run_bl_video_train.sh --variant {bl3|bl4|bl5} ...

SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"
export PATH="/home/simran/.conda/envs/slimllm/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p models

echo "BL-video training: $@"
nvidia-smi || true

python3 -u bl_video_train.py \
    --device cuda \
    --num-workers 4 \
    "$@"

echo "Exit code: $?"
