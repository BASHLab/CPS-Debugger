#!/bin/bash
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --exclude=gpu-6-[01-20]
#SBATCH --output=models/slurm_finetune_chronos_%j.log

# Stage 1: Fine-tune Chronos-2 on domain sensor data

SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"
export PATH="/home/simran/.conda/envs/slimllm/bin:$PATH"

mkdir -p models

echo "BL-2 Stage 1: Fine-tune Chronos-2"
echo "Device: $(python3 -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
nvidia-smi || true

python3 finetune_chronos.py \
    --device cuda \
    "$@"

echo "Exit code: $?"
