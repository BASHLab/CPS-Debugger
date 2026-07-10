#!/bin/bash
#SBATCH --job-name=vid_tier0
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/slurm/%x_%j.out
#SBATCH --error=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/slurm/%x_%j.err

# Tier 0: Optical flow extraction (CPU-only, Farneback).
# Processes all 6 video sessions. ~2 hours total.

set -euo pipefail
echo "Host: $(hostname)"
echo "Start: $(date)"

export PATH="/home/simran/.conda/envs/slimllm/bin:$PATH"
PYTHON=/home/simran/.conda/envs/slimllm/bin/python3
REPO=/home/simran/allspark-data-exploration/CPS-Debugger
cd "$REPO"

# Run Phase 0 first if manifest doesn't exist
if [ ! -f outputs/video_assessment/dataset_params.json ]; then
    echo "Running Phase 0 recon..."
    $PYTHON -u scripts/video_assessment/p0_recon.py
fi

echo "Running Tier 0 optical flow..."
$PYTHON -u scripts/video_assessment/tier0_optical_flow.py --all

echo "Done: $(date)"
