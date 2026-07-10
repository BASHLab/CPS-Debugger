#!/bin/bash
#SBATCH --job-name=vid_tier2
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:L40S:1
#SBATCH --constraint=L40S
#SBATCH --time=02:00:00
#SBATCH --output=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/slurm/%x_%j.out
#SBATCH --error=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/slurm/%x_%j.err

# Tier 2: DINOv2 ViT-L/14 per-frame feature extraction.
# Submit one job per session:
#   sbatch --export=RUN_ID=2025-03-17_10-51-58 slurm/video_tier2_session.sh
# GPU memory: ~3GB. Works on any available GPU.

set -euo pipefail
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "RUN_ID: ${RUN_ID:?'RUN_ID must be set'}"

export PATH="/home/simran/.conda/envs/slimllm/bin:$PATH"
PYTHON=/home/simran/.conda/envs/slimllm/bin/python3
REPO=/home/simran/allspark-data-exploration/CPS-Debugger
cd "$REPO"

$PYTHON -u scripts/video_assessment/tier2_dinov2.py --run "$RUN_ID"

echo "Done: $(date)"
