#!/bin/bash
#SBATCH --job-name=vid_tier1
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:L40S:1
#SBATCH --constraint=L40S
#SBATCH --time=03:00:00
#SBATCH --output=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/slurm/%x_%j.out
#SBATCH --error=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/slurm/%x_%j.err

# Tier 1: CoTracker3 point trajectory extraction.
# Submit one job per session:
#   sbatch --export=RUN_ID=2025-03-17_10-51-58 slurm/video_tier1_session.sh
# GPU memory: ~4GB. Compatible with V100, L40S, RTX 6000 Ada.

set -euo pipefail
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "RUN_ID: ${RUN_ID:?'RUN_ID must be set'}"

export PATH="/home/simran/.conda/envs/slimllm/bin:$PATH"
PYTHON=/home/simran/.conda/envs/slimllm/bin/python3
REPO=/home/simran/allspark-data-exploration/CPS-Debugger
cd "$REPO"

$PYTHON -u scripts/video_assessment/tier1_cotracker.py --run "$RUN_ID"

echo "Done: $(date)"
