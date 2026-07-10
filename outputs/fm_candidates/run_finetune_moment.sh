#!/bin/bash
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --exclude=gpu-6-[01-20]
#SBATCH --job-name=finetune_moment
#SBATCH --output=models/slurm_finetune_moment_%j.log

SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"
export PATH="/home/simran/.conda/envs/slimllm/bin:$PATH"

mkdir -p models

echo "MOMENT-1-large fine-tune (BL-6 stage 1)"
nvidia-smi || true

python3 -u finetune_moment.py --device cuda "$@"
echo "Exit code: $?"
