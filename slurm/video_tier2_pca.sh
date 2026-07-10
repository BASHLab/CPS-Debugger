#!/bin/bash
#SBATCH --job-name=vid_t2_pca
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/slurm/%x_%j.out
#SBATCH --error=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/slurm/%x_%j.err

# Tier 2: Fit PCA on training sessions, project all sessions to 32 dims.
# Run after ALL tier2_session jobs complete.
# No GPU needed.

set -euo pipefail
echo "Host: $(hostname)"
echo "Start: $(date)"

export PATH="/home/simran/.conda/envs/slimllm/bin:$PATH"
PYTHON=/home/simran/.conda/envs/slimllm/bin/python3
REPO=/home/simran/allspark-data-exploration/CPS-Debugger
cd "$REPO"

$PYTHON -u scripts/video_assessment/tier2_dinov2.py --pca

echo "Done: $(date)"
