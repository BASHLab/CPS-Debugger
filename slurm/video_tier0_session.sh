#!/bin/bash
#SBATCH --job-name=vid_t0
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/slurm/%x_%j.out
#SBATCH --error=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/slurm/%x_%j.err

# Tier 0: Optical flow for one session.
# sbatch --export=RUN_ID=2025-03-17_10-51-58 slurm/video_tier0_session.sh

set -euo pipefail
echo "Host: $(hostname) | Start: $(date)"
echo "RUN_ID: ${RUN_ID:?'RUN_ID must be set'}"

export PATH="/home/simran/.conda/envs/slimllm/bin:$PATH"
PYTHON=/home/simran/.conda/envs/slimllm/bin/python3
cd /home/simran/allspark-data-exploration/CPS-Debugger

$PYTHON -u scripts/video_assessment/tier0_optical_flow.py --run "$RUN_ID"

echo "Done: $(date)"
